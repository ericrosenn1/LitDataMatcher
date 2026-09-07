"""Controlled real-file smoke-test workflow for local corpora."""

from __future__ import annotations

import json
from pathlib import Path

from .ingestion import SUPPORTED_SUFFIXES, discover_literature_files, ingest_literature_sources
from .pipeline import run_pipeline
from .schemas import JsonDict


HELPER_FILE_PREFIXES = (
    "input_inventory",
    "smoke_test_instruction",
    "smoke-test-instruction",
    "manual_smoke_instruction",
    "manual-smoke-instruction",
)
HELPER_FILE_NAMES = {
    "readme.md",
    "readme.markdown",
    "smoke_test_instructions.md",
    "smoke-test-instructions.md",
    "manual_smoke_instructions.md",
    "manual-smoke-instructions.md",
}


def run_manual_smoke_test(
    input_dir: str | Path,
    out_dir: str | Path,
    top_n: int = 50,
    recursive: bool = True,
    on_error: str = "skip",
    prepare_only: bool = False,
) -> JsonDict:
    """Prepare or run a small real-file ingest-to-review smoke test."""

    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    corpus_dir = out_dir / "corpus"
    pipeline_dir = out_dir / "pipeline"
    input_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    discovered_files = discover_literature_files([input_dir], recursive=recursive)
    files, excluded_files = filter_manual_smoke_input_files(discovered_files)
    notes_path = write_manual_review_notes_template(
        out_dir / "manual_review_notes.md",
        out_dir=out_dir,
        input_files=files,
        overwrite=False,
    )
    summary: JsonDict = {
        "schema_version": "manual_smoke_test_v1",
        "status": "prepared" if prepare_only else "awaiting_inputs",
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "corpus_dir": str(corpus_dir),
        "pipeline_dir": str(pipeline_dir),
        "manual_review_notes": str(notes_path),
        "supported_suffixes": sorted(SUPPORTED_SUFFIXES),
        "input_files": [str(path) for path in files],
        "input_file_count": len(files),
        "excluded_helper_files": [str(path) for path in excluded_files],
        "excluded_helper_file_count": len(excluded_files),
        "top_n": int(top_n),
        "recursive": bool(recursive),
    }

    if prepare_only or not files:
        summary["next_action"] = (
            f"Place 3-5 supported files in {input_dir} and rerun without --prepare-only."
        )
        return _write_smoke_outputs(out_dir, corpus_dir, pipeline_dir, summary)

    literature_path = corpus_dir / "literature.jsonl"
    ingestion = ingest_literature_sources(
        files,
        literature_path,
        recursive=recursive,
        on_error=on_error,
    )
    pipeline = run_pipeline(literature_path, pipeline_dir, top_n=top_n)
    summary.update(
        {
            "status": "completed",
            "ingestion": ingestion,
            "pipeline": pipeline,
            "records_ingested": ingestion.get("records", 0),
            "skipped_files": ingestion.get("skipped", 0),
            "questions": pipeline.get("questions", 0),
            "matches": pipeline.get("matches", 0),
            "next_action": "Inspect the smoke-test artifacts and fill in manual_review_notes.md.",
        }
    )
    return _write_smoke_outputs(out_dir, corpus_dir, pipeline_dir, summary)


def filter_manual_smoke_input_files(files: list[Path]) -> tuple[list[Path], list[Path]]:
    """Drop generated local helper files without disabling normal text articles."""

    kept: list[Path] = []
    excluded: list[Path] = []
    for path in files:
        if _is_manual_smoke_helper_file(path):
            excluded.append(path)
        else:
            kept.append(path)
    return kept, excluded


def _is_manual_smoke_helper_file(path: Path) -> bool:
    """Return true for instruction or inventory files created around smoke inputs."""

    name = path.name.lower()
    stem = path.stem.lower()
    if name in HELPER_FILE_NAMES:
        return True
    return any(stem.startswith(prefix) for prefix in HELPER_FILE_PREFIXES)


def write_manual_review_notes_template(
    path: str | Path,
    out_dir: str | Path,
    input_files: list[Path],
    overwrite: bool = False,
) -> Path:
    """Write the human checklist used to judge a real-file smoke test."""

    path = Path(path)
    if path.exists() and not overwrite:
        return path
    out_dir = Path(out_dir)
    input_lines = "\n".join(f"- {item}" for item in input_files) or "- "
    text = f"""# Manual Smoke Test Notes

Run:
{out_dir}

Input files:
{input_lines}

## Ingestion

Were all files ingested?
Were source_id/document_id fields stable and readable?
Were titles/abstracts/text extracted acceptably?
Any skipped files or warnings?

## Question Extraction

Best extracted question:
Worst extracted question:
Most awkward wording:
Most obvious missed question:
False positives:
False negatives:

## Variable Inference

Correct inferred variables:
Incorrect inferred variables:
Missing variables:
Overbroad variables:

## Matching

Best match:
Worst match:
Cases where similarity did not imply answerability:
Cases where dataset looked useful but was ranked too low:

## Review Sheet

Are columns sufficient for manual review?
What fields are missing?
Would this support daily labeling?

## Decision

Next slice should be:
[ ] question wording cleanup
[ ] extraction rule cleanup
[ ] variable/ontology cleanup
[ ] review-sheet column expansion
[ ] JATS/XML parser
[ ] GROBID/TEI parser
[ ] parser-specific section provenance
[ ] other:
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_smoke_outputs(
    out_dir: Path,
    corpus_dir: Path,
    pipeline_dir: Path,
    summary: JsonDict,
) -> JsonDict:
    """Write machine and human-readable smoke-test summaries."""

    summary["artifacts"] = _artifact_status(out_dir, corpus_dir, pipeline_dir)
    summary_path = out_dir / "smoke_test_summary.json"
    report_path = out_dir / "smoke_test_summary.md"
    summary["summary_json"] = str(summary_path)
    summary["summary_report"] = str(report_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _artifact_status(out_dir: Path, corpus_dir: Path, pipeline_dir: Path) -> list[JsonDict]:
    """Return expected smoke-test artifacts and whether each exists."""

    paths = [
        corpus_dir / "literature.jsonl",
        corpus_dir / "literature.manifest.json",
        corpus_dir / "literature.ingestion_report.md",
        pipeline_dir / "questions.jsonl",
        pipeline_dir / "matches.jsonl",
        pipeline_dir / "review_sheet.csv",
        pipeline_dir / "summary.md",
        pipeline_dir / "publication_report.md",
        out_dir / "manual_review_notes.md",
    ]
    return [
        {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
        for path in paths
    ]


def _summary_markdown(summary: JsonDict) -> str:
    """Render a compact smoke-test report."""

    lines = [
        "# Manual Smoke Test Summary",
        "",
        f"- Status: {summary.get('status', '')}",
        f"- Input files: {summary.get('input_file_count', 0)}",
        f"- Records ingested: {summary.get('records_ingested', 0)}",
        f"- Skipped files: {summary.get('skipped_files', 0)}",
        f"- Questions: {summary.get('questions', 0)}",
        f"- Matches: {summary.get('matches', 0)}",
        f"- Next action: {summary.get('next_action', '')}",
        "",
        "## Artifacts",
        "",
        "| Artifact | Exists | Size bytes |",
        "| --- | --- | ---: |",
    ]
    for artifact in summary.get("artifacts", []):
        lines.append(
            f"| `{artifact['path']}` | {artifact['exists']} | {artifact['size_bytes']} |"
        )
    lines.extend(
        [
            "",
            "## Manual Review Focus",
            "",
            "- Are extracted questions readable and genuinely open?",
            "- Does each evidence sentence support the extracted question?",
            "- Are variables and populations inferred sensibly?",
            "- Do top-ranked datasets plausibly help answer the question?",
            "- Is the review sheet sufficient for daily labeling?",
        ]
    )
    return "\n".join(lines) + "\n"
