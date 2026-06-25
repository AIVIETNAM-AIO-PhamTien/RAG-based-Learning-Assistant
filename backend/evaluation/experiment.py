from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

import numpy as np

from evaluation.datasets.base import get_dataset_loader
from evaluation.metrics import compute_all_metrics, compute_sample_metrics
from evaluation.pipeline import EvalPipeline
from evaluation.schemas import ExperimentConfig, ExperimentReport

logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self, configs: list[ExperimentConfig], *, use_ragas: bool = False) -> None:
        self.configs = configs
        self._use_ragas = use_ragas

    def run_single(self, config: ExperimentConfig) -> ExperimentReport:
        logger.info("Starting experiment: %s", config.name)
        start = time.perf_counter()

        loader_kwargs = {}
        if config.dataset_path:
            loader_kwargs["path"] = config.dataset_path

        loader = get_dataset_loader(config.dataset_name, **loader_kwargs)
        samples = loader.load(num_samples=config.num_samples)
        logger.info("Loaded %d samples from %s", len(samples), config.dataset_name)

        pipeline = EvalPipeline(config)
        results = pipeline.evaluate_batch(samples)

        if not self._use_ragas:
            for result in results:
                result.metric_scores = compute_sample_metrics(result)
            all_keys = {k for r in results for k in r.metric_scores}
            aggregate = {
                k: float(np.mean([r.metric_scores.get(k, 0.0) for r in results]))
                for k in sorted(all_keys)
            }
        else:
            aggregate = compute_all_metrics(results, use_ragas=True)
        duration = time.perf_counter() - start

        report = ExperimentReport(
            config=config,
            results=results,
            aggregate_metrics=aggregate,
            timestamp=datetime.now(UTC).isoformat(),
            duration_seconds=round(duration, 2),
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
