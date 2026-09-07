"""Command-line interface for reproducible LitDataMatcher runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import cached_client, search_dataset_sources, search_literature_sources
from .annotation_splits import DEFAULT_SPLIT_FRACTIONS, DEFAULT_SPLIT_SEED
from .annotations import export_annotation_corpus
from .artifact_validation import validate_run_artifacts
from .calibration import calibrate_ranking_threshold
from .capability_registry import capability_summary, infer_dataset_capabilities
from .evaluation import (
    evaluate_question_extraction,
    evaluate_ranking,
    load_gold_rows,
    write_evaluation_report,
)
from .grobid import DEFAULT_GROBID_URL, process_pdf_to_tei
from .ingestion import ingest_literature_sources
from .manual_smoke import run_manual_smoke_test
from .pipeline import run_demo, run_pipeline
from .reporting import write_methods_report
from .review import export_review_csv, load_review_labels, summarize_review_labels
from .review_queue import write_daily_review_queue
from .schemas import DatasetRecord, MatchCandidate, QuestionCandidate
from .storage import read_jsonl, write_jsonl
from .stress import run_synthetic_stress_test


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        prog="litdatamatcher",
        description="Extract open research questions and rank matching public datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser(
        "ingest", help="Convert local text, Markdown, PDF, or JSONL sources into literature JSONL."
    )
    ingest.add_argument("--input", nargs="+", required=True, help="Input files or directories.")
    ingest.add_argument("--out", required=True, help="Output literature JSONL path.")
    ingest.add_argument(
        "--recursive", action="store_true", help="Recursively scan input directories."
    )
    ingest.add_argument(
        "--on-error",
        choices=["raise", "skip"],
        default="raise",
        help="Raise on unreadable files or skip them with manifest diagnostics.",
    )

    grobid = subparsers.add_parser(
        "grobid-tei", help="Convert a PDF to GROBID TEI XML using a running GROBID service."
    )
    grobid.add_argument("--input", required=True, help="Input PDF path.")
    grobid.add_argument("--out", required=True, help="Output TEI XML path.")
    grobid.add_argument(
        "--server-url", default=DEFAULT_GROBID_URL, help="Base URL for the GROBID service."
    )
    grobid.add_argument(
        "--consolidate-header",
        action="store_true",
        help="Ask GROBID to consolidate header bibliographic metadata.",
    )
    grobid.add_argument(
        "--consolidate-citations",
        action="store_true",
        help="Ask GROBID to consolidate bibliographic references.",
    )
    grobid.add_argument(
        "--include-raw-affiliations",
        action="store_true",
        help="Ask GROBID to include raw affiliation strings in the TEI response.",
    )

    literature_search = subparsers.add_parser(
        "literature-search", help="Search optional live literature metadata sources."
    )
    literature_search.add_argument("--query", required=True, help="Literature search query.")
    literature_search.add_argument("--out", required=True, help="Output JSONL path.")
    literature_search.add_argument(
        "--source",
        nargs="+",
        default=["pubmed"],
        choices=["crossref", "europepmc", "openalex", "pubmed"],
        help="Live literature source adapter(s) to query.",
    )
    literature_search.add_argument("--limit", type=int, default=25, help="Maximum rows to write.")
    literature_search.add_argument(
        "--cache-dir", default="local/http_cache", help="HTTP cache directory."
    )
    literature_search.add_argument(
        "--offline",
        action="store_true",
        help="Replay cached responses only; fail before a network request on a cache miss.",
    )

    dataset_search = subparsers.add_parser(
        "dataset-search", help="Search optional live dataset/source adapters."
    )
    dataset_search.add_argument("--query", required=True, help="Dataset search query.")
    dataset_search.add_argument("--out", required=True, help="Output dataset JSONL path.")
    dataset_search.add_argument(
        "--source",
        nargs="+",
        default=["clinicaltrials"],
        choices=["clinicaltrials", "geo", "mgnify"],
        help="Live dataset source adapter(s) to query.",
    )
    dataset_search.add_argument("--limit", type=int, default=25, help="Maximum rows to write.")
    dataset_search.add_argument(
        "--cache-dir", default="local/http_cache", help="HTTP cache directory."
    )

    # Pipeline commands create the canonical run artifacts.
    run = subparsers.add_parser("run", help="Run the full pipeline on a JSONL literature file.")
    run.add_argument("--input", required=True, help="JSONL file with title, abstract, text, doi fields.")
    run.add_argument("--out", default="run", help="Output directory for JSONL, SQLite, and summary files.")
    run.add_argument("--catalog", default=None, help="Optional JSONL dataset catalog.")
    run.add_argument("--top-n", type=int, default=100, help="Maximum ranked matches to write.")

    demo = subparsers.add_parser("demo", help="Run the built-in reproducible demo corpus.")
    demo.add_argument("--out", default="run/demo", help="Output directory for demo artifacts.")

    # Evaluation and review commands operate on completed run directories.
    evaluate = subparsers.add_parser("evaluate", help="Evaluate a completed run against gold labels.")
    evaluate.add_argument("--run-dir", required=True, help="Directory containing questions/matches JSONL.")
    evaluate.add_argument("--gold-questions", default=None, help="Gold question JSONL.")
    evaluate.add_argument("--gold-ranking", default=None, help="Gold match relevance JSONL.")
    evaluate.add_argument("--out", default=None, help="Evaluation JSONL path.")
    evaluate.add_argument("--k", type=int, default=10, help="Ranking cutoff.")

    review = subparsers.add_parser("review-export", help="Export a run's matches for expert review.")
    review.add_argument("--run-dir", required=True, help="Directory containing matches.jsonl.")
    review.add_argument("--out", required=True, help="CSV output path.")

    review_summary = subparsers.add_parser("review-summary", help="Summarize completed review labels.")
    review_summary.add_argument("--labels", required=True, help="CSV or JSONL review labels.")

    review_queue = subparsers.add_parser(
        "review-queue", help="Write a small daily scoring queue from ranked matches."
    )
    review_queue.add_argument("--run-dir", required=True, help="Directory containing matches.jsonl.")
    review_queue.add_argument("--out", required=True, help="CSV or JSONL queue output path.")
    review_queue.add_argument("--limit", type=int, default=5, help="Number of matches to queue.")
    review_queue.add_argument("--reviewer-id", default="", help="Reviewer ID to prefill.")
    review_queue.add_argument("--review-date", default=None, help="Optional YYYY-MM-DD review date.")

    annotations = subparsers.add_parser(
        "annotation-export", help="Export completed review labels as training JSONL."
    )
    annotations.add_argument(
        "--labels", nargs="+", required=True, help="Completed review CSV/JSONL files."
    )
    annotations.add_argument("--out", required=True, help="Output directory for label JSONL files.")
    annotations.add_argument("--annotator-id", default="", help="Annotator identifier to store.")
    annotations.add_argument(
        "--include-unlabeled", action="store_true", help="Include rows without expert labels."
    )
    annotations.add_argument(
        "--split-strategy",
        default="none",
        choices=["none", "by_question_id", "by_document_id", "by_source_id", "random"],
        help="Optional grouped split strategy for train/validation/test label JSONL files.",
    )
    annotations.add_argument(
        "--split-fractions",
        nargs=3,
        type=float,
        default=DEFAULT_SPLIT_FRACTIONS,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        help="Fractions for optional train/validation/test splits.",
    )
    annotations.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Deterministic seed for optional split generation.",
    )

    capability_export = subparsers.add_parser(
        "capability-export", help="Infer observed and derived dataset capabilities."
    )
    capability_export.add_argument("--datasets", required=True, help="Dataset JSONL input path.")
    capability_export.add_argument("--out", required=True, help="Capability JSONL output path.")

    calibrate = subparsers.add_parser(
        "calibrate-ranking", help="Calibrate ranking score thresholds from expert labels."
    )
    calibrate.add_argument("--matches", required=True, help="matches.jsonl from a completed run.")
    calibrate.add_argument(
        "--labels", required=True, help="question_data_match_labels.jsonl from annotation export."
    )
    calibrate.add_argument("--out", required=True, help="Calibration report JSON path.")

    stress = subparsers.add_parser(
        "stress-demo", help="Run a deterministic synthetic corpus stress test."
    )
    stress.add_argument("--out", required=True, help="Output directory.")
    stress.add_argument("--documents", type=int, default=25, help="Synthetic documents to generate.")
    stress.add_argument("--top-n", type=int, default=100, help="Maximum ranked matches to write.")

    manual_smoke = subparsers.add_parser(
        "manual-smoke", help="Prepare or run a 3-5 real-file smoke test."
    )
    manual_smoke.add_argument(
        "--input-dir",
        default="local/manual_smoke_inputs/microbiology_3_5",
        help="Folder containing 3-5 real literature files.",
    )
    manual_smoke.add_argument(
        "--out",
        default="run/manual_smoke_microbiology_3_5",
        help="Output folder for corpus, pipeline, and notes artifacts.",
    )
    manual_smoke.add_argument("--top-n", type=int, default=50, help="Maximum matches to write.")
    manual_smoke.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        default=True,
        help="Only scan the input folder itself.",
    )
    manual_smoke.add_argument(
        "--on-error",
        choices=["raise", "skip"],
        default="skip",
        help="Raise on unreadable files or skip them with manifest diagnostics.",
    )
    manual_smoke.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create folders and manual notes template without running ingest/pipeline.",
    )

    report = subparsers.add_parser("report", help="Write a publication-style report for a run.")
    report.add_argument("--run-dir", required=True, help="Directory containing run artifacts.")
    report.add_argument("--out", default=None, help="Markdown output path.")

    validate = subparsers.add_parser(
        "validate-artifacts", help="Inspect run artifacts for readability and provenance readiness."
    )
    validate.add_argument("--run-dir", required=True, help="Directory containing run artifacts.")
    validate.add_argument(
        "--out",
        default=None,
        help="Output directory for artifact_validation_report.md and artifact_validation_summary.json.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ingest":
        metrics = ingest_literature_sources(
            args.input,
            args.out,
            recursive=args.recursive,
            on_error=args.on_error,
        )
    elif args.command == "grobid-tei":
        metrics = process_pdf_to_tei(
            args.input,
            args.out,
            server_url=args.server_url,
            consolidate_header=args.consolidate_header,
            consolidate_citations=args.consolidate_citations,
            include_raw_affiliations=args.include_raw_affiliations,
        )
    elif args.command == "literature-search":
        client = cached_client(args.cache_dir, offline=args.offline)
        rows = search_literature_sources(args.query, args.source, client=client, limit=args.limit)
        write_jsonl(args.out, rows)
        metrics = {"rows": len(rows), "out": args.out, "sources": args.source}
    elif args.command == "dataset-search":
        client = cached_client(args.cache_dir)
        records = search_dataset_sources(args.query, args.source, client=client, limit=args.limit)
        write_jsonl(args.out, [record.to_dict() for record in records])
        metrics = {"records": len(records), "out": args.out, "sources": args.source}
    elif args.command == "run":
        # Full runs are the only path that requires an external literature input file.
        metrics = run_pipeline(args.input, args.out, catalog_path=args.catalog, top_n=args.top_n)
    elif args.command == "demo":
        metrics = run_demo(args.out)
    elif args.command == "evaluate":
        metrics = _run_evaluation(args)
    elif args.command == "review-export":
        matches = _load_matches(args.run_dir)
        export_review_csv(matches, args.out)
        metrics = {"matches": len(matches), "out": args.out}
    elif args.command == "review-summary":
        metrics = summarize_review_labels(load_review_labels(args.labels))
    elif args.command == "review-queue":
        metrics = write_daily_review_queue(
            args.run_dir,
            args.out,
            limit=args.limit,
            reviewer_id=args.reviewer_id,
            review_date=args.review_date,
        )
    elif args.command == "annotation-export":
        metrics = export_annotation_corpus(
            args.labels,
            args.out,
            annotator_id=args.annotator_id,
            include_unlabeled=args.include_unlabeled,
            split_strategy=args.split_strategy,
            split_fractions=args.split_fractions,
            split_seed=args.split_seed,
        )
    elif args.command == "capability-export":
        datasets = [DatasetRecord.from_dict(row) for row in read_jsonl(args.datasets)]
        capabilities = [
            capability
            for dataset in datasets
            for capability in infer_dataset_capabilities(dataset)
        ]
        write_jsonl(args.out, [capability.to_dict() for capability in capabilities])
        metrics = {"out": args.out, **capability_summary(capabilities)}
    elif args.command == "calibrate-ranking":
        report_payload = calibrate_ranking_threshold(args.matches, args.labels)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metrics = {
            "out": args.out,
            "labeled_matches": report_payload["labeled_matches"],
            "positive_labels": report_payload["positive_labels"],
            "negative_labels": report_payload["negative_labels"],
            "best_threshold": report_payload["best_threshold"],
            "best_f1": report_payload["best_threshold_metrics"]["f1"],
        }
    elif args.command == "stress-demo":
        metrics = run_synthetic_stress_test(args.out, documents=args.documents, top_n=args.top_n)
    elif args.command == "manual-smoke":
        metrics = run_manual_smoke_test(
            args.input_dir,
            args.out,
            top_n=args.top_n,
            recursive=args.recursive,
            on_error=args.on_error,
            prepare_only=args.prepare_only,
        )
    elif args.command == "report":
        out = write_methods_report(args.run_dir, args.out)
        metrics = {"report": str(out)}
    elif args.command == "validate-artifacts":
        out_dir = args.out or str(Path(args.run_dir) / "artifact_validation")
        summary = validate_run_artifacts(args.run_dir, out_dir=out_dir)
        metrics = {
            "status": summary["status"],
            "run_dir": summary["run_dir"],
            "out": out_dir,
            "issues": len(summary["issues"]),
            "required_missing": summary["artifact_counts"]["required_missing"],
        }
    else:
        parser.error(f"Unknown command: {args.command}")
    # All commands return JSON metrics so shell scripts can capture them reproducibly.
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def _load_questions(run_dir: str) -> list[QuestionCandidate]:
    """Load question candidates from a completed run."""

    return [QuestionCandidate.from_dict(row) for row in read_jsonl(f"{run_dir}/questions.jsonl")]


def _load_matches(run_dir: str) -> list[MatchCandidate]:
    """Load ranked matches from a completed run."""

    return [MatchCandidate.from_dict(row) for row in read_jsonl(f"{run_dir}/matches.jsonl")]


def _run_evaluation(args) -> dict:
    """Execute extraction and/or ranking evaluation from CLI args."""

    metrics: dict[str, dict] = {}
    # Each gold-label file enables one independent evaluation task.
    if args.gold_questions:
        metrics["question_extraction"] = evaluate_question_extraction(
            _load_questions(args.run_dir), load_gold_rows(args.gold_questions)
        ).to_dict()
    if args.gold_ranking:
        metrics["ranking"] = evaluate_ranking(
            _load_matches(args.run_dir), load_gold_rows(args.gold_ranking), k=args.k
        ).to_dict()
    out_path = args.out or f"{args.run_dir}/evaluation.jsonl"
    if metrics:
        write_evaluation_report(out_path, metrics)
    return {"tasks": sorted(metrics), "out": out_path}


if __name__ == "__main__":
    raise SystemExit(main())
