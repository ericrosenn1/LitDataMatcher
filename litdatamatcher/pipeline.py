"""End-to-end LitDataMatcher pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .datasets import default_adapters, discover_datasets_for_question
from .literature import analyze_literature_records
from .meta_analysis import run_meta_analysis_node, synthesis_index
from .ranking import rank_matches
from .schemas import DatasetRecord, MatchCandidate, QuestionCandidate
from .storage import PipelineStore, read_jsonl, write_jsonl
from .review import export_review_csv, export_review_jsonl, match_review_records
from .reporting import write_methods_report
from .provenance import check_provenance_transfer, module_boundary_map, summarize_source_provenance


def collect_candidate_datasets(
    questions: list[QuestionCandidate], catalog_path: str | Path | None = None
) -> list[DatasetRecord]:
    """Discover and de-duplicate datasets for extracted questions."""

    adapters = default_adapters(catalog_path)
    by_id: dict[str, DatasetRecord] = {}
    for question in questions:
        # Dataset adapters search from the question's text, variables, and population.
        for dataset in discover_datasets_for_question(question, adapters=adapters):
            by_id[dataset.dataset_id] = dataset
    return list(by_id.values())


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    catalog_path: str | Path | None = None,
    top_n: int = 100,
) -> dict[str, Any]:
    """Run the canonical literature-to-ranked-matches workflow."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Source provenance enters here on input records and is copied into questions downstream.
    records = read_jsonl(input_path)
    # Each assignment is a node handoff: records -> questions -> syntheses/datasets -> matches.
    questions = analyze_literature_records(records)
    syntheses = run_meta_analysis_node(questions)
    datasets = collect_candidate_datasets(questions, catalog_path=catalog_path)
    matches = rank_matches(questions, datasets, synthesis_index(syntheses), top_n=top_n)

    # Artifacts are written twice: portable JSONL/Markdown for review, SQLite for querying.
    write_pipeline_outputs(output_dir, questions, datasets, syntheses, matches, source_records=records)
    store = PipelineStore(output_dir / "litdatamatcher.sqlite")
    try:
        store.reset_run_tables()
        store.store_questions(questions)
        store.store_datasets(datasets)
        store.store_syntheses(syntheses)
        store.store_matches(matches)
    finally:
        store.close()
    write_methods_report(output_dir)

    # Keep run accounting minimal and machine-readable for later provenance work.
    metrics = {
        "documents": len(records),
        "questions": len(questions),
        "datasets": len(datasets),
        "syntheses": len(syntheses),
        "matches": len(matches),
        "output_dir": str(output_dir),
    }
    write_jsonl(output_dir / "metrics.jsonl", [metrics])
    return metrics


def write_pipeline_outputs(
    output_dir: Path,
    questions: list[QuestionCandidate],
    datasets: list[DatasetRecord],
    syntheses,
    matches: list[MatchCandidate],
    source_records: list[dict] | None = None,
) -> None:
    """Write canonical node outputs, summaries, and expert-review sheets."""

    # Full nested objects stay in JSONL; CSV keeps a flatter human-facing view.
    write_jsonl(output_dir / "questions.jsonl", [question.to_dict() for question in questions])
    write_jsonl(output_dir / "datasets.jsonl", [dataset.to_dict() for dataset in datasets])
    write_jsonl(output_dir / "syntheses.jsonl", [synthesis.to_dict() for synthesis in syntheses])
    write_jsonl(output_dir / "matches.jsonl", [match.to_dict() for match in matches])
    question_dicts = [question.to_dict() for question in questions]
    dataset_dicts = [dataset.to_dict() for dataset in datasets]
    review_records = match_review_records(matches)
    # This summary counts provenance entries, which may differ from unique documents/datasets.
    provenance_summary = summarize_source_provenance(question_dicts + dataset_dicts)
    (output_dir / "source_provenance_summary.json").write_text(
        json.dumps(provenance_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "module_boundary_map.json").write_text(
        json.dumps(module_boundary_map(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    transfer_check = check_provenance_transfer(
        source_records=source_records or [],
        questions=question_dicts,
        datasets=dataset_dicts,
        review_records=review_records,
        report_summary=provenance_summary,
    )
    (output_dir / "provenance_transfer_check.json").write_text(
        json.dumps(transfer_check, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown_summary(output_dir / "summary.md", matches)
    export_review_csv(matches, output_dir / "review_sheet.csv")
    export_review_jsonl(matches, output_dir / "review_sheet.jsonl")


def write_markdown_summary(path: str | Path, matches: list[MatchCandidate], limit: int = 25) -> None:
    """Write a compact Markdown table of top ranked matches."""

    path = Path(path)
    lines = [
        "# LitDataMatcher Run Summary",
        "",
        "| Rank | Score | Question | Dataset | Why |",
        "| ---: | ---: | --- | --- | --- |",
    ]
    for rank, match in enumerate(matches[:limit], 1):
        # Escape table separators so question/dataset text cannot break Markdown columns.
        why = "; ".join(match.rationale)
        lines.append(
            "| {rank} | {score:.3f} | {question} | {dataset} | {why} |".format(
                rank=rank,
                score=match.score.combined,
                question=match.question.question.replace("|", "/"),
                dataset=match.dataset.title.replace("|", "/"),
                why=why.replace("|", "/"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def demo_input() -> list[dict]:
    """Return a compact demo corpus for smoke tests and examples."""

    return [
        {
            "title": "Longitudinal microbiome recovery after antibiotics in IBD",
            "abstract": (
                "Antibiotic exposure may perturb gut microbiome composition in patients "
                "with inflammatory bowel disease. Future studies should examine whether "
                "longitudinal microbiome and clinical outcome data can predict recovery."
            ),
            "text": (
                "Limitations. The small sample size and incomplete diet metadata limit "
                "causal interpretation. Discussion. Further research should test whether "
                "metabolomics improves prediction of remission after antibiotic exposure."
            ),
            "doi": "10.0000/demo-ibd-microbiome",
        }
    ]


def run_demo(output_dir: str | Path) -> dict[str, Any]:
    """Run the full pipeline on the built-in demo corpus."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "demo_input.jsonl"
    write_jsonl(input_path, demo_input())
    return run_pipeline(input_path=input_path, output_dir=output_dir, top_n=25)
