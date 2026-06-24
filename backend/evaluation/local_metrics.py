from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

import numpy as np

from app.rag.embedder import get_embedder
from app.rag.reranker import get_reranker
from evaluation.config import get_eval_settings
from evaluation.schemas import EvalResult


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@lru_cache
def _get_nli_model():
    from sentence_transformers import CrossEncoder

    settings = get_eval_settings()
    return CrossEncoder(settings.nli_model_name)


def compute_faithfulness_nli(results: list[EvalResult]) -> dict[str, float]:
    if not results:
        return {"faithfulness_nli": 0.0}

    nli_model = _get_nli_model()
    scores: list[float] = []

    for r in results:
        answer = r.generation.generated_answer
        if not answer or answer.startswith("[ERROR") or answer.startswith("[SKIPPED"):
            continue

        contexts = r.retrieval.retrieved_contexts
        if not contexts:
            scores.append(0.0)
            continue

        context_text = " ".join(contexts)
        sentences = _split_sentences(answer)
        if not sentences:
            scores.append(0.0)
            continue

        pairs = [(context_text, sent) for sent in sentences]
        predictions = nli_model.predict(pairs)

        if hasattr(predictions[0], "__len__"):
            entailment_scores = [float(pred[0]) for pred in predictions]
        else:
            entailment_scores = [float(pred) for pred in predictions]

        threshold = 0.5
        entailed = sum(1 for s in entailment_scores if s > threshold)
        scores.append(entailed / len(sentences))

    return {
        "faithfulness_nli": float(np.mean(scores)) if scores else 0.0,
    }


def compute_answer_relevancy(results: list[EvalResult]) -> dict[str, float]:
    if not results:
        return {"answer_similarity": 0.0, "rouge_l": 0.0, "token_f1": 0.0}

    embedder = get_embedder()
    similarities: list[float] = []
    rouge_scores: list[float] = []
    f1_scores: list[float] = []

    for r in results:
        answer = r.generation.generated_answer
        if not answer or answer.startswith("[ERROR") or answer.startswith("[SKIPPED"):
            continue

        q_emb = np.array(embedder.embed_query(r.sample.question))
        a_emb = np.array(embedder.embed_query(answer))
        sim = float(q_emb @ a_emb)
        similarities.append(max(0.0, min(1.0, sim)))

        rouge_scores.append(_rouge_l_f1(r.sample.ground_truth_answer, answer))
        f1_scores.append(_token_f1(r.sample.ground_truth_answer, answer))

    return {
        "answer_similarity": float(np.mean(similarities)) if similarities else 0.0,
        "rouge_l": float(np.mean(rouge_scores)) if rouge_scores else 0.0,
        "token_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
    }


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


def compute_context_relevance(results: list[EvalResult]) -> dict[str, float]:
    if not results:
        return {"context_relevance_avg": 0.0}

    reranker = get_reranker()
    scores: list[float] = []

    for r in results:
        contexts = r.retrieval.retrieved_contexts
        if not contexts:
            scores.append(0.0)
            continue

        query = r.sample.question
        ranking = reranker.rerank(query, contexts, len(contexts))
        ctx_scores = [score for _, score in ranking]
        scores.append(float(np.mean(ctx_scores)) if ctx_scores else 0.0)

    return {
        "context_relevance_avg": float(np.mean(scores)) if scores else 0.0,
    }


def compute_local_metrics(results: list[EvalResult]) -> dict[str, float]:
    from evaluation.metrics import compute_generation_metrics, compute_retrieval_metrics

    metrics: dict[str, float] = {}
    metrics.update(compute_retrieval_metrics(results))
    metrics.update(compute_generation_metrics(results))

    has_generation = any(
        r.generation.generated_answer
        and not r.generation.generated_answer.startswith("[")
        for r in results
    )
    if has_generation:
        metrics.update(compute_faithfulness_nli(results))
        metrics.update(compute_answer_relevancy(results))
    metrics.update(compute_context_relevance(results))

    return metrics
