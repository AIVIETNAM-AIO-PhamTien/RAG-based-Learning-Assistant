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


def _make_report() -> ExperimentReport:
    result = EvalResult(
        sample=EvalSample(question="Q?", ground_truth_answer="A"),
        retrieval=RetrievalResult(
            retrieved_contexts=["ctx1"], retrieved_scores=[0.9], latency_ms=10.0
        ),
        generation=GenerationResult(
            generated_answer="Answer [1]", latency_ms=100.0, citations_used=[1]
        ),
    )
    return ExperimentReport(
        config=ExperimentConfig(name="test"),
        results=[result],
        aggregate_metrics={"recall_at_5": 1.0, "mrr": 1.0},
        timestamp="2026-06-24T00:00:00Z",
        duration_seconds=5.0,
    )


def test_export_csv_no_metric_columns(tmp_path: Path):
    report = _make_report()
    csv_path = tmp_path / "test.csv"
    export_csv(report, csv_path)

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    assert "question" in headers
    assert "generated_answer" in headers
    assert len(rows) == 1
    assert rows[0]["question"] == "Q?"
    for h in headers:
        assert rows[0][h] != "0.0000"


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
