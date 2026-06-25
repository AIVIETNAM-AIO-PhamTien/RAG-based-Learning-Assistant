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
from evaluation.schemas import EvalResult

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


def _context_overlap(
    retrieved: list[str], ground_truth: list[str]
) -> tuple[float, float]:
    if not ground_truth:
        return 0.0, 0.0
    hits = []
    for gt in ground_truth:
        gt_lower = gt.lower().strip()
        found = any(gt_lower in r.lower() or r.lower() in gt_lower for r in retrieved)
        hits.append(found)
    recall = sum(hits) / len(ground_truth)

    rr = 0.0
    for rank, ret_ctx in enumerate(retrieved, start=1):
        ret_lower = ret_ctx.lower()
        matched = any(
            g.lower().strip() in ret_lower or ret_lower in g.lower().strip()
            for g in ground_truth
        )
        if matched:
            rr = 1.0 / rank
            break
    return recall, rr


@lru_cache
def _get_nli_model():
    from sentence_transformers import CrossEncoder

    settings = get_eval_settings()
    return CrossEncoder(settings.nli_model_name)


def _is_valid_answer(answer: str) -> bool:
    return bool(answer) and not answer.startswith(("[ERROR", "[SKIPPED"))


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------


def compute_sample_metrics(result: EvalResult) -> dict[str, float]:
    scores: dict[str, float] = {}
    retrieved = result.retrieval.retrieved_contexts

    recall, rr = _context_overlap(retrieved, result.sample.ground_truth_contexts)
    scores["recall"] = recall
    scores["rr"] = rr

    answer = result.generation.generated_answer

    if _is_valid_answer(answer):
        embedder = get_embedder()
        q_emb = np.array(embedder.embed_query(result.sample.question))
        a_emb = np.array(embedder.embed_query(answer))
        scores["answer_similarity"] = max(0.0, min(1.0, float(q_emb @ a_emb)))
        scores["rouge_l"] = _rouge_l_f1(result.sample.ground_truth_answer, answer)
        scores["token_f1"] = _token_f1(result.sample.ground_truth_answer, answer)

        available = set(range(1, len(retrieved) + 1))
        scores["citation_coverage"] = citation_coverage(answer, available)

        if retrieved:
            nli_model = _get_nli_model()
            context_text = " ".join(retrieved)
            sentences = _split_sentences(answer)
            if sentences:
                pairs = [(context_text, s) for s in sentences]
                preds = nli_model.predict(pairs)
                if hasattr(preds[0], "__len__"):
                    ent = [float(p[0]) for p in preds]
                else:
                    ent = [float(p) for p in preds]
                scores["faithfulness_nli"] = (
                    sum(1 for s in ent if s > 0.5) / len(sentences)
                )
            else:
                scores["faithfulness_nli"] = 0.0
        else:
            scores["faithfulness_nli"] = 0.0
    else:
        for k in (
            "answer_similarity", "rouge_l", "token_f1",
            "citation_coverage", "faithfulness_nli",
        ):
            scores[k] = 0.0

    if retrieved:
        reranker = get_reranker()
        ranking = reranker.rerank(result.sample.question, retrieved, len(retrieved))
        ctx_scores = [s for _, s in ranking]
        scores["context_relevance"] = (
            float(np.mean(ctx_scores)) if ctx_scores else 0.0
        )
    else:
        scores["context_relevance"] = 0.0

    return scores


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def compute_all_metrics(
    results: list[EvalResult], *, use_ragas: bool = False
) -> dict[str, float]:
    if use_ragas:
        return _compute_ragas_aggregate(results)

    all_keys = {k for r in results for k in r.metric_scores}
    if not all_keys:
        return {}
    return {
        k: float(np.mean([r.metric_scores.get(k, 0.0) for r in results]))
        for k in sorted(all_keys)
    }


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
    ret_latencies: list[float] = []
    gen_latencies: list[float] = []

    for r in results:
        rc, rr = _context_overlap(
            r.retrieval.retrieved_contexts, r.sample.ground_truth_contexts
        )
        recall_scores.append(rc)
        rr_scores.append(rr)

        available = set(range(1, len(r.retrieval.retrieved_contexts) + 1))
        cov_scores.append(citation_coverage(r.generation.generated_answer, available))
        ret_latencies.append(r.retrieval.latency_ms)
        gen_latencies.append(r.generation.latency_ms)

    metrics["recall_at_5"] = float(np.mean(recall_scores))
    metrics["mrr"] = float(np.mean(rr_scores))
    metrics["citation_coverage_avg"] = float(np.mean(cov_scores))
    metrics["retrieval_latency_p50"] = float(np.percentile(ret_latencies, 50))
    metrics["retrieval_latency_p95"] = float(np.percentile(ret_latencies, 95))
    metrics["generation_latency_p50"] = float(np.percentile(gen_latencies, 50))
    metrics["generation_latency_p95"] = float(np.percentile(gen_latencies, 95))

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
