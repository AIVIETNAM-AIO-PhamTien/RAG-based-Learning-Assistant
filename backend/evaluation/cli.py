from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from evaluation.schemas import ExperimentConfig


def _build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        name=args.name or f"{args.dataset}_{args.top_k}k_rerank{args.rerank}",
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        num_samples=args.num_samples,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        rerank_enabled=args.rerank,
        rerank_candidate_k=args.rerank_candidate_k,
    )


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    """Add shared arguments for retrieval/full commands."""
    parser.add_argument("--name", help="Experiment name")
    parser.add_argument("--dataset", default="pdf_qa", help="Dataset name")
    parser.add_argument("--dataset-path", help="Path to dataset file/dir")
    parser.add_argument("--num-samples", type=int, help="Number of samples")
    parser.add_argument("--chunk-size", type=int, default=1600)
    parser.add_argument("--chunk-overlap", type=int, default=250)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerank-candidate-k", type=int, default=20)
    parser.add_argument("--output-dir", default="evaluation/results")
    parser.add_argument(
        "--no-warmup", action="store_true", default=False,
        help="Skip model warmup (first sample includes model loading time)",
    )


# ── Commands ──────────────────────────────────────────────────────────


def cmd_generate_qa(args: argparse.Namespace) -> None:
    from evaluation.datasets.qa_generator import QAGenerator

    gen = QAGenerator()
    samples = gen.generate_from_pdf(
        pdf_path=args.pdf_path,
        num_questions_per_page=args.num_questions,
    )
    gen.save(samples, args.output)
    print(f"Generated {len(samples)} Q/A pairs → {args.output}")


def cmd_retrieval(args: argparse.Namespace) -> None:
    from evaluation.artifacts import get_experiment_dir, save_artifact
    from evaluation.experiment import ExperimentRunner
    from evaluation.export import export_summary
    from evaluation.schemas import ExperimentReport

    config = _build_config_from_args(args)
    runner = ExperimentRunner([config], warmup=not args.no_warmup)
    artifact = runner.run_retrieval(config)

    exp_dir = get_experiment_dir(Path(args.output_dir), config.name)
    save_artifact(artifact, exp_dir / "retrieval.json")

    summary_report = ExperimentReport(
        config=config,
        aggregate_metrics={**artifact.retrieval_metrics, **artifact.latency_metrics},
        retrieval_metrics=artifact.retrieval_metrics,
        latency_metrics=artifact.latency_metrics,
        timestamp=artifact.timestamp,
        duration_seconds=artifact.duration_seconds,
        stages_completed=["retrieval"],
    )
    print(export_summary([summary_report]))
    print(f"\nRetrieval artifact saved → {exp_dir / 'retrieval.json'}")


def cmd_generation(args: argparse.Namespace) -> None:
    from evaluation.artifacts import (
        load_retrieval_artifact,
        resolve_retrieval_artifact,
        validate_config_compatibility,
    )
    from evaluation.experiment import ExperimentRunner
    from evaluation.export import export_csv, export_json, export_summary

    retrieval_path = resolve_retrieval_artifact(args.from_artifact)
    retrieval_artifact = load_retrieval_artifact(retrieval_path)

    config = retrieval_artifact.config.model_copy()
    if args.gemini_model:
        config = config.model_copy(update={"gemini_model": args.gemini_model})
    if args.name:
        config = config.model_copy(update={"name": args.name})

    warnings = validate_config_compatibility(retrieval_artifact.config, config)
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)

    runner = ExperimentRunner(
        [config], use_ragas=args.ragas, warmup=not getattr(args, "no_warmup", False)
    )
    report = runner.run_generation(config, retrieval_artifact)

    exp_dir = retrieval_path.parent
    export_json(report, exp_dir / "report.json")
    export_csv(report, exp_dir / "samples.csv")
    print(export_summary([report]))
    print(f"\nReport saved → {exp_dir / 'report.json'}")


def cmd_full(args: argparse.Namespace) -> None:
    from evaluation.artifacts import get_experiment_dir, save_artifact
    from evaluation.experiment import ExperimentRunner
    from evaluation.export import export_csv, export_json, export_summary

    config = _build_config_from_args(args)
    runner = ExperimentRunner([config], use_ragas=args.ragas, warmup=not args.no_warmup)

    retrieval_artifact = runner.run_retrieval(config)
    report = runner.run_generation(config, retrieval_artifact)

    exp_dir = get_experiment_dir(Path(args.output_dir), config.name)
    save_artifact(retrieval_artifact, exp_dir / "retrieval.json")
    export_json(report, exp_dir / "report.json")
    export_csv(report, exp_dir / "samples.csv")
    print(export_summary([report]))
    print(f"\nResults saved → {exp_dir}")


def cmd_run(args: argparse.Namespace) -> None:
    """Backward-compatible alias for 'full'."""
    cmd_full(args)


def cmd_compare(args: argparse.Namespace) -> None:
    import yaml

    from evaluation.experiment import ExperimentRunner
    from evaluation.export import (
        export_comparison_csv,
        export_json,
        export_latex_table,
        export_summary,
    )

    with open(args.config_file, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    configs = []
    for exp in raw["experiments"]:
        cfg = ExperimentConfig(**exp)
        if args.dataset_path:
            cfg = cfg.model_copy(update={"dataset_path": args.dataset_path})
        if args.num_samples:
            cfg = cfg.model_copy(update={"num_samples": args.num_samples})
        configs.append(cfg)

    runner = ExperimentRunner(
        configs, use_ragas=args.ragas, warmup=not getattr(args, "no_warmup", False)
    )
    reports = runner.run_comparison()

    output_dir = Path(args.output_dir)
    export_comparison_csv(reports, output_dir / "comparison.csv")
    export_latex_table(reports, output_dir / "comparison.tex")
    for report in reports:
        export_json(report, output_dir / f"{report.config.name}_report.json")

    print(export_summary(reports))


def cmd_report(args: argparse.Namespace) -> None:
    import json

    from evaluation.export import (
        export_comparison_csv,
        export_latex_table,
        export_summary,
    )
    from evaluation.schemas import ExperimentReport

    results_dir = Path(args.results_dir)
    reports = []

    # Scan flat files (legacy)
    for json_file in sorted(results_dir.glob("*_report.json")):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        reports.append(ExperimentReport(**data))

    # Scan per-experiment directories
    for json_file in sorted(results_dir.glob("*/report.json")):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        reports.append(ExperimentReport(**data))

    if not reports:
        print(f"No report files found in {results_dir}")
        return

    if args.format == "latex":
        out = results_dir / "combined.tex"
        export_latex_table(reports, out)
        print(f"LaTeX table → {out}")
    elif args.format == "csv":
        out = results_dir / "combined_comparison.csv"
        export_comparison_csv(reports, out)
        print(f"Comparison CSV → {out}")
    else:
        print(export_summary(reports))


# ── Parser ────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="evaluation", description="RAG Evaluation Pipeline"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # generate-qa
    gen = sub.add_parser("generate-qa", help="Generate Q/A pairs from a PDF")
    gen.add_argument("--pdf-path", required=True, help="Path to PDF file")
    gen.add_argument("--num-questions", type=int, default=3, help="Questions per chunk")
    gen.add_argument("--output", required=True, help="Output JSON path")

    # retrieval
    ret = sub.add_parser("retrieval", help="Run retrieval only, save artifact")
    _add_common_run_args(ret)

    # generation
    gencmd = sub.add_parser(
        "generation", help="Run generation from retrieval artifact"
    )
    gencmd.add_argument(
        "--from", dest="from_artifact", required=True,
        help="Path to retrieval artifact or experiment directory",
    )
    gencmd.add_argument("--name", help="Override experiment name")
    gencmd.add_argument("--gemini-model", help="Override Gemini model")
    gencmd.add_argument(
        "--ragas", action="store_true", help="Use RAGAS metrics (requires API)"
    )
    gencmd.add_argument(
        "--no-warmup", action="store_true", default=False,
        help="Skip model warmup",
    )

    # full
    full = sub.add_parser("full", help="Run full pipeline (retrieval + generation)")
    _add_common_run_args(full)
    full.add_argument(
        "--ragas", action="store_true", help="Use RAGAS metrics (requires API)"
    )

    # run (alias for full, backward compat)
    run = sub.add_parser("run", help="Run a single evaluation experiment (alias for full)")
    _add_common_run_args(run)
    run.add_argument(
        "--ragas", action="store_true", help="Use RAGAS metrics (requires API)"
    )

    # compare
    cmp = sub.add_parser("compare", help="Run comparison across multiple configs")
    cmp.add_argument("--config-file", required=True, help="YAML config file")
    cmp.add_argument("--dataset-path", help="Override dataset path for all experiments")
    cmp.add_argument("--num-samples", type=int, help="Override num samples")
    cmp.add_argument("--output-dir", default="evaluation/results")
    cmp.add_argument("--ragas", action="store_true", help="Use RAGAS metrics (requires API)")
    cmp.add_argument("--no-warmup", action="store_true", default=False, help="Skip model warmup")

    # report
    rpt = sub.add_parser("report", help="Generate report from saved results")
    rpt.add_argument("--results-dir", default="evaluation/results")
    rpt.add_argument("--format", choices=["table", "csv", "latex"], default="table")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    commands = {
        "generate-qa": cmd_generate_qa,
        "retrieval": cmd_retrieval,
        "generation": cmd_generation,
        "full": cmd_full,
        "run": cmd_run,
        "compare": cmd_compare,
        "report": cmd_report,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
