import csv
import json
from pathlib import Path

from evaluation.export import export_csv, export_json
from evaluation.schemas import (
    EvalResult,
    EvalSample,
    ExperimentConfig,
    ExperimentReport,
    GenerationResult,
    RetrievalResult,
)


def _make_report(with_metrics: bool = False) -> ExperimentReport:
    result = EvalResult(
        sample=EvalSample(question="Q?", ground_truth_answer="A"),
        retrieval=RetrievalResult(
            retrieved_contexts=["ctx1"], retrieved_scores=[0.9], latency_ms=10.0
        ),
        generation=GenerationResult(
            generated_answer="Answer [1]", latency_ms=100.0, citations_used=[1]
        ),
    )
    if with_metrics:
        result.metric_scores = {"recall": 0.85, "token_f1": 0.72, "rouge_l": 0.65}
    return ExperimentReport(
        config=ExperimentConfig(name="test"),
        results=[result],
        aggregate_metrics={"recall_at_5": 1.0, "mrr": 1.0},
        timestamp="2026-06-24T00:00:00Z",
        duration_seconds=5.0,
    )


def test_export_csv_without_metrics(tmp_path: Path):
    report = _make_report(with_metrics=False)
    csv_path = tmp_path / "test.csv"
    export_csv(report, csv_path)

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    assert "question" in headers
    assert "recall" not in headers
    assert len(rows) == 1


def test_export_csv_with_per_sample_metrics(tmp_path: Path):
    report = _make_report(with_metrics=True)
    csv_path = tmp_path / "test.csv"
    export_csv(report, csv_path)

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    assert "recall" in headers
    assert "token_f1" in headers
    assert "rouge_l" in headers
    assert len(rows) == 1
    assert rows[0]["recall"] == "0.8500"
    assert rows[0]["token_f1"] == "0.7200"


def test_export_json_roundtrip(tmp_path: Path):
    report = _make_report()
    json_path = tmp_path / "test_report.json"
    export_json(report, json_path)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    restored = ExperimentReport(**data)
    assert restored.config.name == "test"
    assert restored.aggregate_metrics["recall_at_5"] == 1.0
    assert len(restored.results) == 1
    assert restored.results[0].sample.question == "Q?"
