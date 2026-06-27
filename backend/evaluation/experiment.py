from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

import numpy as np

from evaluation.datasets.base import get_dataset_loader
from evaluation.metrics import (
    compute_all_metrics,
    compute_retrieval_latency_percentiles,
    compute_retrieval_metrics,
    compute_sample_metrics,
)
from evaluation.pipeline import EvalPipeline
from evaluation.schemas import (
    EvalResult,
    ExperimentConfig,
    ExperimentReport,
    RetrievalArtifact,
)

logger = logging.getLogger(__name__)

# Sync with keys from compute_retrieval_metrics() and _compute_ragas_aggregate() in metrics.py.
# "mrr" only appears in the RAGAS aggregate path.
_RET_PREFIXES = (
    "recall", "rr", "mrr", "precision", "hit_rate", "ndcg", "context_relevance",
)
# Sync with keys from compute_generation_metrics() in metrics.py.
_GEN_PREFIXES = (
    "answer_similarity", "rouge_l", "token_f1", "citation_coverage", "faithfulness",
)
_LAT_PREFIXES = ("retrieval_latency", "generation_latency")


def _group_metrics(
    flat: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    ret_m: dict[str, float] = {}
    gen_m: dict[str, float] = {}
    lat_m: dict[str, float] = {}
    for k, v in flat.items():
        if any(k.startswith(p) for p in _LAT_PREFIXES):
            lat_m[k] = v
        elif any(k.startswith(p) for p in _RET_PREFIXES):
            ret_m[k] = v
        elif any(k.startswith(p) for p in _GEN_PREFIXES):
            gen_m[k] = v
        else:
            gen_m[k] = v
    return ret_m, gen_m, lat_m


def _load_samples(config: ExperimentConfig):
    loader_kwargs = {}
    if config.dataset_path:
        loader_kwargs["path"] = config.dataset_path
    loader = get_dataset_loader(config.dataset_name, **loader_kwargs)
    samples = loader.load(num_samples=config.num_samples)
    logger.info("Loaded %d samples from %s", len(samples), config.dataset_name)
    return samples


class ExperimentRunner:
    def __init__(
        self,
        configs: list[ExperimentConfig],
        *,
        use_ragas: bool = False,
        warmup: bool = True,
    ) -> None:
        self.configs = configs
        self._use_ragas = use_ragas
        self._warmup = warmup

    def run_retrieval(self, config: ExperimentConfig) -> RetrievalArtifact:
        """Load dataset, run retrieval only, compute retrieval metrics."""
        logger.info("Starting retrieval: %s", config.name)
        start = time.perf_counter()

        samples = _load_samples(config)
        pipeline = EvalPipeline(config)
        if self._warmup:
            pipeline.warmup()
        retrieval_results = pipeline.retrieve_batch(samples)

        ret_metrics_list = [
            compute_retrieval_metrics(
                s, r, compute_context_relevance=config.compute_context_relevance
            )
            for s, r in zip(samples, retrieval_results, strict=True)
        ]
        all_keys = {k for m in ret_metrics_list for k in m}
        aggregate = {
            k: float(np.nanmean([m.get(k, 0.0) for m in ret_metrics_list]))
            for k in sorted(all_keys)
        }

        latency_stats = compute_retrieval_latency_percentiles(retrieval_results)

        duration = time.perf_counter() - start
        artifact = RetrievalArtifact(
            config=config,
            samples=samples,
            retrieval_results=retrieval_results,
            retrieval_metrics=aggregate,
            latency_metrics=latency_stats,
            timestamp=datetime.now(UTC).isoformat(),
            duration_seconds=round(duration, 2),
        )
        logger.info(
            "Retrieval %s completed in %.1fs. Metrics: %s",
            config.name,
            duration,
            {k: f"{v:.4f}" for k, v in aggregate.items()},
        )
        return artifact

    def run_generation(
        self, config: ExperimentConfig, retrieval_artifact: RetrievalArtifact
    ) -> ExperimentReport:
        """Run generation on existing retrieval results, compute all metrics."""
        logger.info("Starting generation: %s", config.name)
        start = time.perf_counter()

        pipeline = EvalPipeline(config)
        if self._warmup:
            pipeline.warmup()
        generation_results = pipeline.generate_batch(
            retrieval_artifact.samples, retrieval_artifact.retrieval_results
        )

        results: list[EvalResult] = []
        for sample, ret, gen in zip(
            retrieval_artifact.samples,
            retrieval_artifact.retrieval_results,
            generation_results,
            strict=True,
        ):
            ev = EvalResult(sample=sample, retrieval=ret, generation=gen)
            if not self._use_ragas:
                ev.metric_scores = compute_sample_metrics(
                    ev, compute_context_relevance=config.compute_context_relevance
                )
            results.append(ev)

        if not self._use_ragas:
            all_keys = {k for r in results for k in r.metric_scores}
            aggregate = {
                k: float(np.nanmean([r.metric_scores.get(k, 0.0) for r in results]))
                for k in sorted(all_keys)
            }
        else:
            aggregate = compute_all_metrics(results, use_ragas=True)

        ret_m, gen_m, lat_m = _group_metrics(aggregate)

        duration = time.perf_counter() - start
        report = ExperimentReport(
            config=config,
            results=results,
            aggregate_metrics=aggregate,
            retrieval_metrics=ret_m,
            generation_metrics=gen_m,
            latency_metrics=lat_m,
            timestamp=datetime.now(UTC).isoformat(),
            duration_seconds=round(
                retrieval_artifact.duration_seconds + duration, 2
            ),
            stages_completed=["retrieval", "generation"],
        )
        logger.info(
            "Generation %s completed in %.1fs. Metrics: %s",
            config.name,
            duration,
            {k: f"{v:.4f}" for k, v in aggregate.items()},
        )
        return report

    def run_single(self, config: ExperimentConfig) -> ExperimentReport:
        logger.info("Starting experiment: %s", config.name)
        start = time.perf_counter()

        samples = _load_samples(config)
        pipeline = EvalPipeline(config)
        if self._warmup:
            pipeline.warmup()
        results = pipeline.evaluate_batch(samples)

        if not self._use_ragas:
            for result in results:
                result.metric_scores = compute_sample_metrics(
                    result, compute_context_relevance=config.compute_context_relevance
                )
            all_keys = {k for r in results for k in r.metric_scores}
            aggregate = {
                k: float(np.nanmean([r.metric_scores.get(k, 0.0) for r in results]))
                for k in sorted(all_keys)
            }
        else:
            aggregate = compute_all_metrics(results, use_ragas=True)

        ret_m, gen_m, lat_m = _group_metrics(aggregate)
        duration = time.perf_counter() - start

        report = ExperimentReport(
            config=config,
            results=results,
            aggregate_metrics=aggregate,
            retrieval_metrics=ret_m,
            generation_metrics=gen_m,
            latency_metrics=lat_m,
            timestamp=datetime.now(UTC).isoformat(),
            duration_seconds=round(duration, 2),
            stages_completed=["retrieval", "generation"],
        )

        logger.info(
            "Experiment %s completed in %.1fs. Metrics: %s",
            config.name,
            duration,
            {k: f"{v:.4f}" for k, v in aggregate.items()},
        )
        return report

    def run_comparison(self) -> list[ExperimentReport]:
        reports: list[ExperimentReport] = []
        for i, config in enumerate(self.configs, start=1):
            logger.info("Running experiment %d/%d: %s", i, len(self.configs), config.name)
            report = self.run_single(config)
            reports.append(report)
        return reports
