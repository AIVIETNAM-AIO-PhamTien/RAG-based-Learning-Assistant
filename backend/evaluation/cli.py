from __future__ import annotations

import argparse
import logging
from pathlib import Path

from evaluation.schemas import ExperimentConfig


def cmd_generate_qa(args: argparse.Namespace) -> None:
    from evaluation.datasets.qa_generator import QAGenerator

    gen = QAGenerator()
    samples = gen.generate_from_pdf(
        pdf_path=args.pdf_path,
        num_questions_per_page=args.num_questions,
    )
    gen.save(samples, args.output)
    print(f"Generated {len(samples)} Q/A pairs → {args.output}")


def cmd_run(args: argparse.Namespace) -> None:
    from evaluation.experiment import ExperimentRunner
    from evaluation.export import export_csv, export_json, export_summary

    config = ExperimentConfig(
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

    runner = ExperimentRunner([config], use_ragas=args.ragas)
    report = runner.run_single(config)

    output_dir = Path(args.output_dir)
    export_json(report, output_dir / f"{config.name}_report.json")
    export_csv(report, output_dir / f"{config.name}_samples.csv")
    print(export_summary([report]))


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

    runner = ExperimentRunner(configs, use_ragas=args.ragas)
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
    for json_file in sorted(results_dir.glob("*_report.json")):
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

    # run
    run = sub.add_parser("run", help="Run a single evaluation experiment")
    run.add_argument("--name", help="Experiment name")
    run.add_argument("--dataset", default="pdf_qa", help="Dataset name")
    run.add_argument("--dataset-path", help="Path to dataset file/dir")
    run.add_argument("--num-samples", type=int, help="Number of samples")
    run.add_argument("--chunk-size", type=int, default=1600)
    run.add_argument("--chunk-overlap", type=int, default=250)
    run.add_argument("--top-k", type=int, default=5)
    run.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--rerank-candidate-k", type=int, default=20)
    run.add_argument("--output-dir", default="evaluation/results")
    run.add_argument("--ragas", action="store_true", help="Use RAGAS metrics (requires API)")

    # compare
    cmp = sub.add_parser("compare", help="Run comparison across multiple configs")
    cmp.add_argument("--config-file", required=True, help="YAML config file")
    cmp.add_argument("--dataset-path", help="Override dataset path for all experiments")
    cmp.add_argument("--num-samples", type=int, help="Override num samples")
    cmp.add_argument("--output-dir", default="evaluation/results")
    cmp.add_argument("--ragas", action="store_true", help="Use RAGAS metrics (requires API)")

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
        "run": cmd_run,
        "compare": cmd_compare,
        "report": cmd_report,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
