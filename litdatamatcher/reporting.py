"""Publication-oriented reporting utilities."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sqlite3

from .ontology import concept_table
from .storage import read_jsonl


def write_methods_report(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    """Write a manuscript-style methods and results summary for a run."""

    run_dir = Path(run_dir)
    output_path = Path(output_path) if output_path else run_dir / "publication_report.md"
    questions = read_jsonl(run_dir / "questions.jsonl") if (run_dir / "questions.jsonl").exists() else []
    datasets = read_jsonl(run_dir / "datasets.jsonl") if (run_dir / "datasets.jsonl").exists() else []
    matches = read_jsonl(run_dir / "matches.jsonl") if (run_dir / "matches.jsonl").exists() else []
    syntheses = read_jsonl(run_dir / "syntheses.jsonl") if (run_dir / "syntheses.jsonl").exists() else []

    dataset_sources = Counter(row.get("source", "") for row in datasets)
    variables = Counter()
    for dataset in datasets:
        for variable in dataset.get("variables", []):
            variables[variable.get("name", "")] += 1

    lines = [
        "# LitDataMatcher Publication Report",
        "",
        "## Run Summary",
        "",
        f"- Questions extracted: {len(questions)}",
        f"- Evidence-synthesis clusters: {len(syntheses)}",
        f"- Candidate datasets: {len(datasets)}",
        f"- Ranked matches: {len(matches)}",
        "",
        "## Methods",
        "",
        "Literature records were processed with deterministic sentence splitting, sectioning, "
        "candidate open-question extraction, ontology-backed variable harmonization, "
        "meta-analysis style question clustering, dataset metadata normalization, and "
        "explainable question-to-dataset ranking. Each node wrote JSONL artifacts and "
        "a SQLite database to preserve provenance and enable independent audit.",
        "",
        "Ranking combined literature significance, evidence recurrence, variable overlap, "
        "semantic relevance, population fit, sample adequacy, dataset quality, governance "
        "reuse, design fit, and uncertainty penalties.",
        "",
        "## Dataset Sources",
        "",
        "| Source | Count |",
        "| --- | ---: |",
    ]
    for source, count in dataset_sources.most_common():
        lines.append(f"| {source or 'unknown'} | {count} |")

    lines.extend(["", "## Most Common Normalized Variables", "", "| Variable | Dataset Count |", "| --- | ---: |"])
    for variable, count in variables.most_common(15):
        lines.append(f"| {variable} | {count} |")

    lines.extend(["", "## Top Ranked Opportunities", "", "| Rank | Score | Question | Dataset |", "| ---: | ---: | --- | --- |"])
    for rank, match in enumerate(matches[:20], 1):
        lines.append(
            "| {rank} | {score:.3f} | {question} | {dataset} |".format(
                rank=rank,
                score=float(match.get("score", {}).get("combined", 0.0)),
                question=match.get("question", {}).get("question", "").replace("|", "/"),
                dataset=match.get("dataset", {}).get("title", "").replace("|", "/"),
            )
        )

    lines.extend(["", "## Ontology Concepts", "", "| Concept | Category | Synonyms |", "| --- | --- | --- |"])
    for concept in concept_table():
        lines.append(f"| {concept['label']} | {concept['category']} | {concept['synonyms']} |")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Current default extraction is deterministic and should be benchmarked against expert annotations.",
            "- Live repository adapters require network access, API stability checks, and source-specific validation.",
            "- Ranked opportunities are hypotheses for expert review, not claims that a dataset definitively answers a question.",
            "- Human-subject datasets require governance, consent, and license review before downstream analysis.",
        ]
    )

    if (run_dir / "litdatamatcher.sqlite").exists():
        lines.extend(["", "## SQLite Tables", "", *sqlite_table_summary(run_dir / "litdatamatcher.sqlite")])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def sqlite_table_summary(path: str | Path) -> list[str]:
    """Return a small table-count summary from a run database."""

    conn = sqlite3.connect(path)
    try:
        rows = []
        for table in ("questions", "datasets", "syntheses", "matches"):
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            rows.append(f"- `{table}`: {count}")
        return rows
    finally:
        conn.close()
