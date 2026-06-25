from __future__ import annotations

import csv
import json
from pathlib import Path

from evaluation.schemas import ExperimentReport


def export_json(report: ExperimentReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, ensure_ascii=False, indent=2, default=str)


def export_csv(report: ExperimentReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not report.results:
        return

    base_fields = [
        "question",
        "ground_truth_answer",
        "generated_answer",
        "retrieval_latency_ms",
        "generation_latency_ms",
        "num_retrieved",
        "citations_used",
    ]

    metric_keys: set[str] = set()
    for r in report.results:
        metric_keys.update(r.metric_scores.keys())
    metric_fields = sorted(metric_keys)
    fieldnames = base_fields + metric_fields

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in report.results:
            row = {
                "question": r.sample.question,
                "ground_truth_answer": r.sample.ground_truth_answer,
                "generated_answer": r.generation.generated_answer,
                "retrieval_latency_ms": f"{r.retrieval.latency_ms:.1f}",
                "generation_latency_ms": f"{r.generation.latency_ms:.1f}",
                "num_retrieved": len(r.retrieval.retrieved_contexts),
                "citations_used": str(r.generation.citations_used),
            }
            for mk in metric_fields:
                row[mk] = f"{r.metric_scores.get(mk, 0.0):.4f}"
            writer.writerow(row)


def export_comparison_csv(reports: list[ExperimentReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not reports:
        return

    all_metric_keys: set[str] = set()
    for report in reports:
        all_metric_keys.update(report.aggregate_metrics.keys())
    metric_fields = sorted(all_metric_keys)

    config_fields = [
        "name",
        "dataset_name",
        "chunk_size",
        "chunk_overlap",
        "top_k",
        "rerank_enabled",
        "num_samples",
        "duration_seconds",
    ]
    fieldnames = config_fields + metric_fields

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for report in reports:
            row = {
                "name": report.config.name,
                "dataset_name": report.config.dataset_name,
                "chunk_size": report.config.chunk_size,
                "chunk_overlap": report.config.chunk_overlap,
                "top_k": report.config.top_k,
                "rerank_enabled": report.config.rerank_enabled,
                "num_samples": len(report.results),
                "duration_seconds": f"{report.duration_seconds:.1f}",
            }
            for mk in metric_fields:
                row[mk] = f"{report.aggregate_metrics.get(mk, 0.0):.4f}"
            writer.writerow(row)


def export_latex_table(reports: list[ExperimentReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not reports:
        return

    all_metric_keys: set[str] = set()
    for report in reports:
        all_metric_keys.update(report.aggregate_metrics.keys())
    metric_fields = sorted(all_metric_keys)

    header_cols = ["Config"] + [m.replace("_", " ").title() for m in metric_fields]
    col_spec = "l" + "c" * len(metric_fields)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(header_cols) + " \\\\",
        "\\midrule",
    ]

    for report in reports:
        values = [report.config.name]
        for mk in metric_fields:
            v = report.aggregate_metrics.get(mk, 0.0)
            values.append(f"{v:.4f}")
        lines.append(" & ".join(values) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{RAG Evaluation Results}",
        "\\label{tab:rag-eval}",
        "\\end{table}",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def export_summary(reports: list[ExperimentReport]) -> str:
    try:
        from tabulate import tabulate
    except ImportError:
        return _fallback_summary(reports)

    if not reports:
        return "No results to display."

    all_metric_keys: set[str] = set()
    for report in reports:
        all_metric_keys.update(report.aggregate_metrics.keys())
    metric_fields = sorted(all_metric_keys)

    headers = ["Config"] + metric_fields + ["Duration(s)"]
    rows = []
    for report in reports:
        row = [report.config.name]
        for mk in metric_fields:
            row.append(f"{report.aggregate_metrics.get(mk, 0.0):.4f}")
        row.append(f"{report.duration_seconds:.1f}")
        rows.append(row)

    return tabulate(rows, headers=headers, tablefmt="grid")


def _fallback_summary(reports: list[ExperimentReport]) -> str:
    lines = []
    for report in reports:
        lines.append(f"\n=== {report.config.name} ===")
        for k, v in sorted(report.aggregate_metrics.items()):
            lines.append(f"  {k}: {v:.4f}")
        lines.append(f"  duration: {report.duration_seconds:.1f}s")
    return "\n".join(lines)
