from __future__ import annotations

import logging
import re
import time
from collections import Counter
from functools import lru_cache

import numpy as np

from app.rag.embedder import get_embedder
from app.rag.metrics import citation_coverage
from app.rag.reranker import get_reranker
from evaluation.config import get_eval_settings
from evaluation.schemas import (
    EvalResult,
    EvalSample,
    GenerationResult,
    RetrievalResult,
)

logger = logging.getLogger(__name__)
RAGAS_BATCH_SIZE = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _lcs_length(x: list[str], y: list[str]) -> int:
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def _rouge_l_f1(reference: str, hypothesis: str) -> float:
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs_len = _lcs_length(ref_tokens, hyp_tokens)
    precision = lcs_len / len(hyp_tokens)
    recall = lcs_len / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _token_f1(reference: str, hypothesis: str) -> float:
    ref_tokens = Counter(_tokenize(reference))
    hyp_tokens = Counter(_tokenize(hypothesis))
    if not ref_tokens or not hyp_tokens:
        return 0.0
    common = sum((ref_tokens & hyp_tokens).values())
    if common == 0:
        return 0.0
    precision = common / sum(hyp_tokens.values())
    recall = common / sum(ref_tokens.values())
    return 2 * precision * recall / (precision + recall)


def _normalize_for_match(text: str) -> str:
    """Collapse whitespace runs to a single space, matching chunker.py's
    `" ".join(text.split())` — ground_truth_contexts from dataset loaders keep
    raw formatting (e.g. HotpotQA's double-spaced sentences) while
    retrieved_contexts always pass through the chunker, so comparing them
    without this normalization causes false-negative substring matches."""
    return " ".join(text.lower().split())


def _context_overlap(retrieved: list[str], ground_truth: list[str]) -> tuple[float, float]:
    if not ground_truth:
        return 0.0, 0.0
    retrieved_norm = [_normalize_for_match(r) for r in retrieved]
    hits = []
    for gt in ground_truth:
        gt_norm = _normalize_for_match(gt)
        found = any(gt_norm in r or r in gt_norm for r in retrieved_norm)
        hits.append(found)
    recall = sum(hits) / len(ground_truth)

    gt_norms = [_normalize_for_match(g) for g in ground_truth]
    rr = 0.0
    for rank, ret_norm in enumerate(retrieved_norm, start=1):
        matched = any(g in ret_norm or ret_norm in g for g in gt_norms)
        if matched:
            rr = 1.0 / rank
            break
    return recall, rr


def _compute_relevance_vector(retrieved: list[str], ground_truth: list[str]) -> list[bool]:
    """Binary relevance for each retrieved doc. Single O(n*m) pass."""
    gt_norms = [_normalize_for_match(g) for g in ground_truth]
    relevances = []
    for ret_ctx in retrieved:
        ret_norm = _normalize_for_match(ret_ctx)
        matched = any(g in ret_norm or ret_norm in g for g in gt_norms)
        relevances.append(matched)
    return relevances


def _ndcg_from_relevances(relevances: list[bool]) -> float:
    if not relevances or not any(relevances):
        return 0.0
    dcg = sum((1.0 if rel else 0.0) / np.log2(rank + 2) for rank, rel in enumerate(relevances))
    ideal_rels = sorted(relevances, reverse=True)
    idcg = sum((1.0 if rel else 0.0) / np.log2(rank + 2) for rank, rel in enumerate(ideal_rels))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


@lru_cache
def _get_nli_model() -> tuple:
    from sentence_transformers import CrossEncoder

    settings = get_eval_settings()
    model = CrossEncoder(settings.nli_model_name)
    id2label = getattr(model.model.config, "id2label", None)
    if id2label:
        entailment_idx = next(
            (int(i) for i, label in id2label.items() if label.lower() == "entailment"),
            1,
        )
    else:
        entailment_idx = 1
    return model, entailment_idx


def _is_valid_answer(answer: str) -> bool:
    return bool(answer) and not answer.startswith(("[ERROR", "[SKIPPED"))


def _reference_answers(sample: EvalSample) -> list[str]:
    """All acceptable reference answers for a sample, deduped, non-empty.

    Datasets with multiple valid phrasings (e.g. natural_questions, asqa) store
    them in metadata["all_short_answers"]; others fall back to the single
    ground_truth_answer, matching the previous single-reference behavior.
    """
    refs = sample.metadata.get("all_short_answers") or [sample.ground_truth_answer]
    return list(dict.fromkeys(r for r in refs if r))


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------


def compute_retrieval_metrics(
    sample: EvalSample,
    retrieval: RetrievalResult,
    *,
    compute_context_relevance: bool = True,
) -> dict[str, float]:
    """Compute retrieval-only metrics: recall, rr, precision_at_k, hit_rate_at_k, ndcg_at_k, context_relevance."""
    scores: dict[str, float] = {}
    retrieved = retrieval.retrieved_contexts

    recall, rr = _context_overlap(retrieved, sample.ground_truth_contexts)
    scores["recall"] = recall
    scores["rr"] = rr

    relevances = _compute_relevance_vector(retrieved, sample.ground_truth_contexts)
    scores["precision_at_k"] = sum(relevances) / len(relevances) if relevances else 0.0
    scores["hit_rate_at_k"] = 1.0 if any(relevances) else 0.0
    scores["ndcg_at_k"] = _ndcg_from_relevances(relevances)

    if compute_context_relevance and retrieved:
        reranker = get_reranker()
        ranking = reranker.rerank(sample.question, retrieved, len(retrieved))
        ctx_scores = [s for _, s in ranking]
        scores["context_relevance"] = float(np.mean(ctx_scores)) if ctx_scores else 0.0
    else:
        scores["context_relevance"] = 0.0

    return scores


def compute_generation_metrics(
    sample: EvalSample,
    retrieval: RetrievalResult,
    generation: GenerationResult,
) -> dict[str, float]:
    """Compute generation metrics: similarity, rouge_l, token_f1, citation, faithfulness."""
    scores: dict[str, float] = {}
    answer = generation.generated_answer
    retrieved = retrieval.retrieved_contexts

    if not _is_valid_answer(answer):
        for k in (
            "answer_similarity",
            "rouge_l",
            "token_f1",
            "citation_coverage",
            "faithfulness_nli",
        ):
            scores[k] = 0.0
        return scores

    embedder = get_embedder()
    references = _reference_answers(sample)
    if not references:
        scores["answer_similarity"] = float("nan")
    else:
        a_emb = np.array(embedder.embed_query(answer))
        sims = [
            max(0.0, min(1.0, float(np.array(embedder.embed_query(ref)) @ a_emb)))
            for ref in references
        ]
        scores["answer_similarity"] = max(sims)
    scores["rouge_l"] = max(_rouge_l_f1(ref, answer) for ref in references) if references else 0.0
    scores["token_f1"] = max(_token_f1(ref, answer) for ref in references) if references else 0.0

    available = set(range(1, len(retrieved) + 1))
    scores["citation_coverage"] = citation_coverage(answer, available)

    if retrieved:
        nli_model, entailment_idx = _get_nli_model()
        sentences = _split_sentences(answer)
        if sentences:
            # Score each sentence against each chunk individually (not the
            # chunks joined into one string) — the NLI model's 512-token
            # limit would otherwise silently truncate away whichever chunks
            # don't fit, discarding evidence that supports the answer.
            n_chunks = len(retrieved)
            pairs = [(chunk, s) for s in sentences for chunk in retrieved]
            preds = nli_model.predict(pairs)
            if hasattr(preds[0], "__len__"):
                ent = [float(p[entailment_idx]) for p in preds]
            else:
                ent = [float(p) for p in preds]
            per_sentence_max = [
                max(ent[i * n_chunks : (i + 1) * n_chunks]) for i in range(len(sentences))
            ]
            scores["faithfulness_nli"] = sum(1 for s in per_sentence_max if s > 0.5) / len(
                sentences
            )
        else:
            scores["faithfulness_nli"] = 0.0
    else:
        scores["faithfulness_nli"] = 0.0

    return scores


def compute_sample_metrics(
    result: EvalResult, *, compute_context_relevance: bool = True
) -> dict[str, float]:
    scores = compute_retrieval_metrics(
        result.sample, result.retrieval, compute_context_relevance=compute_context_relevance
    )
    scores.update(compute_generation_metrics(result.sample, result.retrieval, result.generation))
    return scores


# ---------------------------------------------------------------------------
# Latency percentiles
# ---------------------------------------------------------------------------


def _compute_latency_percentiles(results: list[EvalResult]) -> dict[str, float]:
    """Compute latency percentiles from EvalResults."""
    if not results:
        return {}
    metrics: dict[str, float] = {}
    ret_latencies = [r.retrieval.latency_ms for r in results]
    if ret_latencies:
        metrics["retrieval_latency_p50"] = float(np.percentile(ret_latencies, 50))
        metrics["retrieval_latency_p95"] = float(np.percentile(ret_latencies, 95))
        metrics["retrieval_latency_mean"] = float(np.mean(ret_latencies))
    gen_latencies = [r.generation.latency_ms for r in results]
    if any(lat > 0 for lat in gen_latencies):
        metrics["generation_latency_p50"] = float(np.percentile(gen_latencies, 50))
        metrics["generation_latency_p95"] = float(np.percentile(gen_latencies, 95))
        metrics["generation_latency_mean"] = float(np.mean(gen_latencies))
    return metrics


def compute_retrieval_latency_percentiles(
    retrieval_results: list[RetrievalResult],
) -> dict[str, float]:
    """Compute latency percentiles for retrieval-only pipeline."""
    if not retrieval_results:
        return {}
    latencies = [r.latency_ms for r in retrieval_results]
    return {
        "retrieval_latency_p50": float(np.percentile(latencies, 50)),
        "retrieval_latency_p95": float(np.percentile(latencies, 95)),
        "retrieval_latency_mean": float(np.mean(latencies)),
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def compute_all_metrics(results: list[EvalResult], *, use_ragas: bool = False) -> dict[str, float]:
    if use_ragas:
        return _compute_ragas_aggregate(results)

    all_keys = {k for r in results for k in r.metric_scores}
    if not all_keys:
        return {}
    agg = {
        k: float(np.nanmean([r.metric_scores.get(k, 0.0) for r in results]))
        for k in sorted(all_keys)
    }
    agg.update(_compute_latency_percentiles(results))
    return agg


# ---------------------------------------------------------------------------
# RAGAS (opt-in, requires API)
# ---------------------------------------------------------------------------


def _compute_ragas_aggregate(results: list[EvalResult]) -> dict[str, float]:
    metrics: dict[str, float] = {}

    if not results:
        return metrics

    recall_scores: list[float] = []
    rr_scores: list[float] = []
    cov_scores: list[float] = []

    for r in results:
        rc, rr = _context_overlap(r.retrieval.retrieved_contexts, r.sample.ground_truth_contexts)
        recall_scores.append(rc)
        rr_scores.append(rr)

        available = set(range(1, len(r.retrieval.retrieved_contexts) + 1))
        cov_scores.append(citation_coverage(r.generation.generated_answer, available))

    metrics["recall_at_5"] = float(np.mean(recall_scores))
    metrics["mrr"] = float(np.mean(rr_scores))
    metrics["citation_coverage_avg"] = float(np.mean(cov_scores))
    metrics.update(_compute_latency_percentiles(results))

    has_generation = any(_is_valid_answer(r.generation.generated_answer) for r in results)
    if has_generation:
        try:
            metrics.update(compute_ragas_metrics(results))
        except (ImportError, Exception):
            pass

    return metrics


def compute_ragas_metrics(results: list[EvalResult]) -> dict[str, float]:
    if not results:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
        }

    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness
    except ImportError as exc:
        raise ImportError(
            "ragas is required for RAGAS metrics. Install: pip install ragas langchain-google-genai"
        ) from exc

    samples = [
        SingleTurnSample(
            user_input=r.sample.question,
            response=r.generation.generated_answer,
            retrieved_contexts=r.retrieval.retrieved_contexts,
            reference=r.sample.ground_truth_answer,
        )
        for r in results
    ]

    ragas_metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision()]
    settings = get_eval_settings()

    all_scores: list[dict[str, float]] = []
    for batch_start in range(0, len(samples), RAGAS_BATCH_SIZE):
        batch = samples[batch_start : batch_start + RAGAS_BATCH_SIZE]
        dataset = EvaluationDataset(samples=batch)

        try:
            report = evaluate(dataset=dataset, metrics=ragas_metrics)
            all_scores.append(report.to_pandas().mean(numeric_only=True).to_dict())
        except Exception:
            logger.warning("RAGAS batch %d failed, skipping", batch_start)
            continue

        if batch_start + RAGAS_BATCH_SIZE < len(samples):
            pause = settings.api_delay_seconds * len(batch)
            logger.info("RAGAS batch done, pausing %.1fs", pause)
            time.sleep(pause)

    if not all_scores:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
        }

    return {
        k: float(np.mean([s.get(k, 0.0) for s in all_scores]))
        for k in ("faithfulness", "answer_relevancy", "context_precision")
    }
