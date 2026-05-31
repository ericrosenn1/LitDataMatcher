"""Command-line interface for reproducible LitDataMatcher runs."""

from __future__ import annotations

import argparse
import json

from .evaluation import (
    evaluate_question_extraction,
    evaluate_ranking,
    load_gold_rows,
    write_evaluation_report,
)
from .pipeline import run_demo, run_pipeline
from .reporting import write_methods_report
from .review import export_review_csv, load_review_labels, summarize_review_labels
from .schemas import MatchCandidate, QuestionCandidate
from .storage import read_jsonl


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        prog="litdatamatcher",
        description="Extract open research questions and rank matching public datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the full pipeline on a JSONL literature file.")
    run.add_argument("--input", required=True, help="JSONL file with title, abstract, text, doi fields.")
    run.add_argument("--out", default="run", help="Output directory for JSONL, SQLite, and summary files.")
    run.add_argument("--catalog", default=None, help="Optional JSONL dataset catalog.")
    run.add_argument("--top-n", type=int, default=100, help="Maximum ranked matches to write.")

    demo = subparsers.add_parser("demo", help="Run the built-in reproducible demo corpus.")
    demo.add_argument("--out", default="run/demo", help="Output directory for demo artifacts.")

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

    report = subparsers.add_parser("report", help="Write a publication-style report for a run.")
    report.add_argument("--run-dir", required=True, help="Directory containing run artifacts.")
    report.add_argument("--out", default=None, help="Markdown output path.")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
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
    elif args.command == "report":
        out = write_methods_report(args.run_dir, args.out)
        metrics = {"report": str(out)}
    else:
        parser.error(f"Unknown command: {args.command}")
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
