from __future__ import annotations

import logging
import time

import numpy as np

from app.rag.metrics import citation_coverage
from evaluation.config import get_eval_settings
from evaluation.schemas import EvalResult

logger = logging.getLogger(__name__)
RAGAS_BATCH_SIZE = 20


def _context_overlap(retrieved: list[str], ground_truth: list[str]) -> list[bool]:
    hits = []
    for gt in ground_truth:
        gt_lower = gt.lower().strip()
        found = any(gt_lower in r.lower() or r.lower() in gt_lower for r in retrieved)
        hits.append(found)
    return hits


def compute_retrieval_metrics(results: list[EvalResult]) -> dict[str, float]:
    if not results:
        return {"recall_at_5": 0.0, "mrr": 0.0}

    recall_scores: list[float] = []
    rr_scores: list[float] = []

    for result in results:
        gt_contexts = result.sample.ground_truth_contexts
        retrieved = result.retrieval.retrieved_contexts

        if not gt_contexts:
            continue

        hits = _context_overlap(retrieved, gt_contexts)
        recall = sum(hits) / len(gt_contexts)
        recall_scores.append(recall)

        rr = 0.0
        for rank, ret_ctx in enumerate(retrieved, start=1):
            ret_lower = ret_ctx.lower()
            if any(
                gt.lower().strip() in ret_lower or ret_lower in gt.lower().strip()
                for gt in gt_contexts
            ):
                rr = 1.0 / rank
                break
        rr_scores.append(rr)

    return {
        "recall_at_5": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "mrr": float(np.mean(rr_scores)) if rr_scores else 0.0,
    }


def compute_generation_metrics(results: list[EvalResult]) -> dict[str, float]:
    if not results:
        return {
            "citation_coverage_avg": 0.0,
            "retrieval_latency_p50": 0.0,
            "retrieval_latency_p95": 0.0,
            "generation_latency_p50": 0.0,
            "generation_latency_p95": 0.0,
        }

    cov_scores: list[float] = []
    ret_latencies: list[float] = []
    gen_latencies: list[float] = []

    for result in results:
        num_contexts = len(result.retrieval.retrieved_contexts)
        available = set(range(1, num_contexts + 1))
        cov = citation_coverage(result.generation.generated_answer, available)
        cov_scores.append(cov)
        ret_latencies.append(result.retrieval.latency_ms)
        gen_latencies.append(result.generation.latency_ms)

    return {
        "citation_coverage_avg": float(np.mean(cov_scores)),
        "retrieval_latency_p50": float(np.percentile(ret_latencies, 50)),
        "retrieval_latency_p95": float(np.percentile(ret_latencies, 95)),
        "generation_latency_p50": float(np.percentile(gen_latencies, 50)),
        "generation_latency_p95": float(np.percentile(gen_latencies, 95)),
    }


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
            "ragas is required for RAGAS metrics. Install with: uv sync --group eval"
        ) from exc

    samples = []
    for r in results:
        samples.append(
            SingleTurnSample(
                user_input=r.sample.question,
                response=r.generation.generated_answer,
                retrieved_contexts=r.retrieval.retrieved_contexts,
                reference=r.sample.ground_truth_answer,
            )
        )

    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision()]
    settings = get_eval_settings()

    all_scores: list[dict[str, float]] = []
    for batch_start in range(0, len(samples), RAGAS_BATCH_SIZE):
        batch = samples[batch_start : batch_start + RAGAS_BATCH_SIZE]
        dataset = EvaluationDataset(samples=batch)

        try:
            report = evaluate(dataset=dataset, metrics=metrics)
            batch_df = report.to_pandas()
            all_scores.append(batch_df.mean(numeric_only=True).to_dict())
        except Exception:
            logger.warning("RAGAS batch %d failed, skipping", batch_start)
            continue

        if batch_start + RAGAS_BATCH_SIZE < len(samples):
            pause = settings.api_delay_seconds * len(batch)
            logger.info("RAGAS batch done, pausing %.1fs before next batch", pause)
            time.sleep(pause)

    if not all_scores:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
        }

    combined = {
        k: float(np.mean([s.get(k, 0.0) for s in all_scores]))
        for k in ("faithfulness", "answer_relevancy", "context_precision")
    }
    return combined


def compute_all_metrics(
    results: list[EvalResult], *, use_ragas: bool = False
) -> dict[str, float]:
    if use_ragas:
        metrics: dict[str, float] = {}
        metrics.update(compute_retrieval_metrics(results))
        metrics.update(compute_generation_metrics(results))
        has_generation = any(r.generation.generated_answer for r in results)
        if has_generation:
            try:
                metrics.update(compute_ragas_metrics(results))
            except (ImportError, Exception):
                pass
        return metrics

    from evaluation.local_metrics import compute_local_metrics

    return compute_local_metrics(results)
