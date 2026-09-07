"""Publication-oriented reporting utilities.

Reports summarize run artifacts for inspection; they should not be read as
validation that extracted questions, live metadata, or ranked matches are true.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sqlite3

from .ontology import concept_table
from .provenance import summarize_source_provenance
from .storage import read_jsonl


def _md_cell(value: object) -> str:
    """Return text safe for a generated Markdown table cell."""

    return str(value or "").replace("|", "/")


def write_methods_report(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    """Write a manuscript-style methods and results summary for a run."""

    run_dir = Path(run_dir)
    output_path = Path(output_path) if output_path else run_dir / "publication_report.md"
    # Missing artifacts are treated as empty so partial runs can still be inspected.
    questions = read_jsonl(run_dir / "questions.jsonl") if (run_dir / "questions.jsonl").exists() else []
    datasets = read_jsonl(run_dir / "datasets.jsonl") if (run_dir / "datasets.jsonl").exists() else []
    matches = read_jsonl(run_dir / "matches.jsonl") if (run_dir / "matches.jsonl").exists() else []
    syntheses = read_jsonl(run_dir / "syntheses.jsonl") if (run_dir / "syntheses.jsonl").exists() else []
    # Provenance summaries describe input depth and caveats, not answerability.
    provenance_summary = summarize_source_provenance([*questions, *datasets])

    dataset_sources = Counter(row.get("source", "") for row in datasets)
    variables = Counter()
    for dataset in datasets:
        # Variable counts summarize available data breadth across normalized records.
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
        f"- Records with source provenance: {provenance_summary.get('records_with_provenance', 0)}",
        f"- Records without source provenance: {provenance_summary.get('records_without_provenance', 0)}",
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
        "Review artifacts include full nested match records in `matches.jsonl` and "
        "`review_sheet.jsonl`, plus a flattened `review_sheet.csv` for expert scoring.",
        "",
        "## Source Provenance",
        "",
        "These counts describe provenance entries and caveats, not unique papers, unique datasets, "
        "or proof that a dataset answers a question.",
        "",
        "| Field | Counts |",
        "| --- | --- |",
        f"| Source types | {_count_dict_cell(provenance_summary.get('source_types', {}))} |",
        f"| Content scopes | {_count_dict_cell(provenance_summary.get('content_scopes', {}))} |",
        f"| Acquisition methods | {_count_dict_cell(provenance_summary.get('acquisition_methods', {}))} |",
        f"| Statuses | {_count_dict_cell(provenance_summary.get('statuses', {}))} |",
        "",
    ]
    limitations = provenance_summary.get("limitations", {})
    warnings = provenance_summary.get("warnings", {})
    if limitations or warnings:
        lines.extend(
            [
                "### Provenance Caveats",
                "",
                f"- Limitations: {_count_dict_cell(limitations)}",
                f"- Warnings: {_count_dict_cell(warnings)}",
                "",
            ]
        )
    review_caveats = provenance_summary.get("review_caveats", {})
    if review_caveats:
        lines.extend(
            [
                "### Reviewer Interpretation",
                "",
                "| Caveat | Count |",
                "| --- | ---: |",
            ]
        )
        for caveat, count in sorted(review_caveats.items(), key=lambda item: (-int(item[1]), item[0])):
            lines.append(f"| {_md_cell(caveat)} | {count} |")
        lines.append("")

    lines.extend(
        [
        "## Dataset Sources",
        "",
        "| Source | Count |",
        "| --- | ---: |",
        ]
    )
    for source, count in dataset_sources.most_common():
        lines.append(f"| {source or 'unknown'} | {count} |")

    lines.extend(["", "## Most Common Normalized Variables", "", "| Variable | Dataset Count |", "| --- | ---: |"])
    for variable, count in variables.most_common(15):
        lines.append(f"| {variable} | {count} |")

    lines.extend(
        [
            "",
            "## Top Ranked Opportunities",
            "",
            "| Rank | Score | Question | Dataset | Missing Variables | Recommended Design | Governance |",
            "| ---: | ---: | --- | --- | --- | --- | ---: |",
        ]
    )
    for rank, match in enumerate(matches[:20], 1):
        feasibility = match.get("assessments", {}).get("feasibility", {})
        missing = "; ".join(match.get("missing_variables", []))
        lines.append(
            "| {rank} | {score:.3f} | {question} | {dataset} | {missing} | {design} | {governance:.3f} |".format(
                rank=rank,
                score=float(match.get("score", {}).get("combined", 0.0)),
                question=_md_cell(match.get("question", {}).get("question", "")),
                dataset=_md_cell(match.get("dataset", {}).get("title", "")),
                missing=_md_cell(missing or "none"),
                design=_md_cell(feasibility.get("recommended_design", "")),
                governance=float(match.get("score", {}).get("governance", 0.0)),
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
            # Keep limitations visible because the report may be shared outside the codebase.
            "- Current default extraction is deterministic and should be benchmarked against expert annotations.",
            "- Live repository adapters require network access, API stability checks, and source-specific validation.",
            "- Ranked opportunities are hypotheses for expert review, not claims that a dataset definitively answers a question.",
            "- Human-subject datasets require governance, consent, and license review before downstream analysis.",
        ]
    )

    if (run_dir / "litdatamatcher.sqlite").exists():
        # SQLite counts cross-check the JSONL artifacts without requiring manual queries.
        lines.extend(["", "## SQLite Tables", "", *sqlite_table_summary(run_dir / "litdatamatcher.sqlite")])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def sqlite_table_summary(path: str | Path) -> list[str]:
    """Return a small table-count summary from a run database."""

    conn = sqlite3.connect(path)
    try:
        rows = []
        # Table names are fixed by PipelineStore, keeping this query path simple.
        for table in ("questions", "datasets", "syntheses", "matches"):
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            rows.append(f"- `{table}`: {count}")
        return rows
    finally:
        conn.close()


def _count_dict_cell(values: dict) -> str:
    """Render a provenance count dictionary for Markdown reports."""

    if not values:
        return "none"
    return "; ".join(f"{_md_cell(key)}: {values[key]}" for key in sorted(values))
