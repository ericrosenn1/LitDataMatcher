"""End-to-end LitDataMatcher pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .datasets import default_adapters, discover_datasets_for_question
from .literature import analyze_literature_records
from .meta_analysis import run_meta_analysis_node, synthesis_index
from .ranking import rank_matches
from .schemas import DatasetRecord, MatchCandidate, QuestionCandidate
from .storage import PipelineStore, read_jsonl, write_jsonl
from .review import export_review_csv, export_review_jsonl
from .reporting import write_methods_report


def collect_candidate_datasets(
    questions: list[QuestionCandidate], catalog_path: str | Path | None = None
) -> list[DatasetRecord]:
    """Discover and de-duplicate datasets for all candidate questions."""

    adapters = default_adapters(catalog_path)
    by_id: dict[str, DatasetRecord] = {}
    for question in questions:
        for dataset in discover_datasets_for_question(question, adapters=adapters):
            by_id[dataset.dataset_id] = dataset
    return list(by_id.values())


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    catalog_path: str | Path | None = None,
    top_n: int = 100,
) -> dict[str, Any]:
    """Run literature extraction, synthesis, dataset discovery, and ranking."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(input_path)
    questions = analyze_literature_records(records)
    syntheses = run_meta_analysis_node(questions)
    datasets = collect_candidate_datasets(questions, catalog_path=catalog_path)
    matches = rank_matches(questions, datasets, synthesis_index(syntheses), top_n=top_n)

    write_pipeline_outputs(output_dir, questions, datasets, syntheses, matches)
    store = PipelineStore(output_dir / "litdatamatcher.sqlite")
    try:
        store.store_questions(questions)
        store.store_datasets(datasets)
        store.store_syntheses(syntheses)
        store.store_matches(matches)
    finally:
        store.close()
    write_methods_report(output_dir)

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
) -> None:
    """Write node outputs as JSONL artifacts for auditability."""

    write_jsonl(output_dir / "questions.jsonl", [question.to_dict() for question in questions])
    write_jsonl(output_dir / "datasets.jsonl", [dataset.to_dict() for dataset in datasets])
    write_jsonl(output_dir / "syntheses.jsonl", [synthesis.to_dict() for synthesis in syntheses])
    write_jsonl(output_dir / "matches.jsonl", [match.to_dict() for match in matches])
    write_markdown_summary(output_dir / "summary.md", matches)
    export_review_csv(matches, output_dir / "review_sheet.csv")
    export_review_jsonl(matches, output_dir / "review_sheet.jsonl")


def write_markdown_summary(path: str | Path, matches: list[MatchCandidate], limit: int = 25) -> None:
    """Write a human-readable ranking summary."""

    path = Path(path)
    lines = [
        "# LitDataMatcher Run Summary",
        "",
        "| Rank | Score | Question | Dataset | Why |",
        "| ---: | ---: | --- | --- | --- |",
    ]
    for rank, match in enumerate(matches[:limit], 1):
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
