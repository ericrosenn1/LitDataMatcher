"""Human-readable QA reports for annotation-corpus exports."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .annotation_agreement import AGREEMENT_QA_LIMITATIONS


JsonDict = dict[str, Any]


def build_annotation_corpus_report(
    manifest: JsonDict,
    warnings: Iterable[JsonDict],
    skipped_rows: Iterable[JsonDict],
    duplicates: Iterable[JsonDict],
    conflicts: Iterable[JsonDict],
) -> str:
    """Return a Markdown QA report for a completed annotation export."""

    warnings = list(warnings)
    skipped_rows = list(skipped_rows)
    duplicates = list(duplicates)
    conflicts = list(conflicts)
    summary = dict(manifest.get("summary", {}))
    validation = dict(manifest.get("validation", {}))
    source_files = list(manifest.get("source_files", []))
    outputs = dict(manifest.get("outputs", {}))
    agreement = dict(manifest.get("agreement", summary.get("agreement", {})))
    readiness = dict(
        manifest.get("training_readiness")
        or assess_training_readiness(summary, warnings, skipped_rows, duplicates, conflicts)
    )
    splits = dict(manifest.get("splits", {}))

    lines = [
        "# Annotation Corpus Report",
        "",
        "## Export",
        "",
        f"- Created UTC: {manifest.get('created_at_utc', '')}",
        f"- Corpus version: {manifest.get('corpus_version', '')}",
        f"- Schema version: {manifest.get('schema_version', '')}",
        f"- Annotator fallback: {manifest.get('annotator_id', '') or 'blank'}",
        f"- Include unlabeled rows: {manifest.get('include_unlabeled', False)}",
        "",
        "## Corpus Summary",
        "",
        f"- Source rows: {summary.get('source_rows', 0)}",
        f"- Valid rows: {summary.get('valid_rows', validation.get('valid_rows', validation.get('usable_rows', 0)))}",
        f"- Exported label rows: {summary.get('exported_label_rows', 0)}",
        f"- Skipped rows: {summary.get('skipped_rows', validation.get('skipped_rows', 0))}",
        f"- Warnings: {summary.get('warning_count', validation.get('warnings', 0))}",
        f"- Duplicate rows: {summary.get('duplicate_count', validation.get('duplicates', 0))}",
        f"- Conflicts/disagreements: {summary.get('conflict_count', validation.get('conflicts', 0))}",
        "",
    ]

    lines.extend(_source_file_section(source_files))
    lines.extend(_label_summary_section(summary))
    lines.extend(_source_caveat_section(summary))
    lines.extend(
        [
            "## Reviewer Summary",
            "",
            f"- Reviewer count: {validation.get('reviewer_count', 0)}",
            f"- Reviewers: {_format_reviewers(validation.get('reviewers', []))}",
        ]
    )
    lines.extend(_dict_bullets("Rows per reviewer", validation.get("rows_per_reviewer", {})))
    lines.extend(_dict_bullets("Labels per reviewer", summary.get("labels_per_reviewer", {})))
    lines.append("")
    lines.extend(_agreement_summary_section(agreement, manifest))
    lines.extend(_outputs_section(outputs))
    lines.extend(_split_summary_section(splits))
    lines.extend(
        [
            "## QA Findings",
            "",
            f"- Warnings: {len(warnings)}",
            f"- Skipped rows: {len(skipped_rows)}",
            f"- Duplicate rows: {len(duplicates)}",
            f"- Conflicts/disagreements: {len(conflicts)}",
            "",
        ]
    )
    if not any((warnings, skipped_rows, duplicates, conflicts)):
        lines.extend(["No validation issues were recorded.", ""])

    lines.extend(_issue_section("Warnings", warnings))
    lines.extend(_issue_section("Skipped Rows", skipped_rows))
    lines.extend(_issue_section("Duplicate Rows", duplicates))
    lines.extend(_issue_section("Conflicts And Disagreements", conflicts))
    lines.extend(
        [
            "## Training Readiness",
            "",
            f"- Status: {readiness['status']}",
            f"- Blocking issues: {readiness['blocking_issues']}",
            f"- Recommended next action: {readiness['recommended_next_action']}",
            "",
        ]
    )
    lines.extend(_recommendations(readiness, warnings, skipped_rows, duplicates, conflicts))
    return "\n".join(lines).rstrip() + "\n"


def _label_summary_section(summary: JsonDict) -> list[str]:
    """Format exported label counts and simple distributions."""

    lines = [
        "## Label Summary",
        "",
        f"- Question-data match labels: {summary.get('question_data_match_labels', 0)}",
        f"- Question quality scores: {summary.get('question_quality_scores', 0)}",
        f"- Relevant match labels: {summary.get('relevant_match_labels', 0)}",
        f"- Not-relevant match labels: {summary.get('not_relevant_match_labels', 0)}",
        f"- Mean match relevance: {summary.get('mean_match_relevance', 0.0)}",
        f"- Mean question quality: {summary.get('mean_question_quality', 0.0)}",
        "",
    ]
    lines.extend(_dict_bullets("Relevance distribution", summary.get("relevance_distribution", {})))
    lines.extend(
        _dict_bullets(
            "Question-quality score distribution",
            summary.get("question_quality_distribution", {}),
        )
    )
    lines.append("")
    return lines


def _dict_bullets(title: str, values: object) -> list[str]:
    """Format a small dictionary as nested-free Markdown bullets."""

    if not isinstance(values, dict) or not values:
        return [f"- {title}: none recorded"]
    parts = [f"{key}={values[key]}" for key in sorted(values)]
    return [f"- {title}: " + ", ".join(parts)]


def _source_caveat_section(summary: JsonDict) -> list[str]:
    """Format provenance-derived caveats preserved during label export."""

    caveats = summary.get("source_caveat_counts", {})
    lines = ["## Source Caveats", ""]
    if not isinstance(caveats, dict) or not caveats:
        return [*lines, "- Source caveats: none recorded", ""]
    lines.extend(["| Caveat | Count |", "| --- | ---: |"])
    for caveat, count in sorted(caveats.items(), key=lambda item: (-int(item[1]), item[0])):
        lines.append(f"| {_md(str(caveat))} | {count} |")
    lines.append("")
    return lines


def assess_training_readiness(
    summary: JsonDict,
    warnings: list[JsonDict],
    skipped_rows: list[JsonDict],
    duplicates: list[JsonDict],
    conflicts: list[JsonDict],
) -> JsonDict:
    """Classify whether exported labels are ready for exploratory training."""

    exported_labels = int(summary.get("exported_label_rows", 0) or 0)
    blocking: list[str] = []
    if exported_labels == 0:
        blocking.append("no exported labels")
    if skipped_rows:
        blocking.append("skipped rows require review")
    if conflicts:
        blocking.append("conflicts/disagreements require review")
    if duplicates:
        blocking.append("duplicate rows require review")
    if int(summary.get("unresolved_adjudication_count", 0) or 0):
        blocking.append("unresolved adjudication records require review")

    if not blocking and not warnings:
        status = "ready for exploratory training"
        action = "Proceed with exploratory model development and retain this manifest."
    elif not blocking and warnings:
        status = "usable with caution"
        action = "Inspect warnings before treating the corpus as stable training data."
    else:
        status = "not ready for training"
        if exported_labels == 0:
            action = "Complete review labels before using this corpus for training/export use."
        else:
            action = "Resolve blocking QA findings before using labels for model training."

    return {
        "status": status,
        "blocking_issues": ", ".join(blocking) if blocking else "none",
        "recommended_next_action": action,
    }


def write_annotation_corpus_report(
    path: str | Path,
    manifest: JsonDict,
    warnings: Iterable[JsonDict],
    skipped_rows: Iterable[JsonDict],
    duplicates: Iterable[JsonDict],
    conflicts: Iterable[JsonDict],
) -> Path:
    """Write the Markdown QA report and return its path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = build_annotation_corpus_report(
        manifest,
        warnings=warnings,
        skipped_rows=skipped_rows,
        duplicates=duplicates,
        conflicts=conflicts,
    )
    path.write_text(report, encoding="utf-8")
    return path


def _source_file_section(source_files: list[JsonDict]) -> list[str]:
    """Format source file provenance as a compact table."""

    lines = ["## Source Files", ""]
    if not source_files:
        return [*lines, "No source files were recorded.", ""]
    lines.extend(["| File | Size bytes | SHA-256 |", "| --- | ---: | --- |"])
    for source in source_files:
        digest = str(source.get("sha256", ""))
        lines.append(
            "| "
            + " | ".join(
                (
                    _md(str(source.get("path", source.get("name", "")))),
                    str(source.get("size_bytes", "")),
                    _md(digest[:12] + ("..." if len(digest) > 12 else "")),
                )
            )
            + " |"
        )
    lines.append("")
    return lines


def _outputs_section(outputs: JsonDict) -> list[str]:
    """Format export artifact paths."""

    lines = ["## Output Artifacts", ""]
    if not outputs:
        return [*lines, "No output artifact paths were recorded.", ""]
    for key in sorted(outputs):
        lines.append(f"- `{key}`: `{outputs[key]}`")
    lines.append("")
    return lines


def _agreement_summary_section(agreement: JsonDict, manifest: JsonDict) -> list[str]:
    """Format lightweight reviewer agreement and adjudication QA."""

    lines = ["## Agreement And Adjudication QA", ""]
    if not agreement:
        return [*lines, "- Agreement summary: not available", ""]
    lines.extend(
        [
            f"- Reviewer count: {agreement.get('reviewer_count', 0)}",
            f"- Multi-reviewed targets: {agreement.get('multi_reviewed_target_count', 0)}",
            f"- Reviewer pairs with overlap: {agreement.get('reviewer_pair_count', 0)}",
            f"- Total pair overlaps: {agreement.get('total_pair_overlap_count', 0)}",
            f"- Total pair agreements: {agreement.get('total_pair_agreement_count', 0)}",
            f"- Total pair disagreements: {agreement.get('total_pair_disagreement_count', 0)}",
            f"- Observed agreement: {agreement.get('observed_agreement', 'not available')}",
            f"- Adjudication-needed records: {agreement.get('adjudication_needed_count', 0)}",
            f"- Agreement summary: `{manifest.get('agreement_summary_path', '')}`",
            f"- Adjudication records: `{manifest.get('adjudication_needed_path', '')}`",
            "",
            AGREEMENT_QA_LIMITATIONS,
            "",
        ]
    )
    pairs = agreement.get("reviewer_pairs", [])
    if pairs:
        lines.extend(
            [
                "| Reviewer A | Reviewer B | Overlap | Agree | Disagree | Observed agreement | Kappa |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for pair in pairs:
            lines.append(
                "| "
                + " | ".join(
                    (
                        _md(str(pair.get("reviewer_a", ""))),
                        _md(str(pair.get("reviewer_b", ""))),
                        str(pair.get("overlap_count", 0)),
                        str(pair.get("agreement_count", 0)),
                        str(pair.get("disagreement_count", 0)),
                        str(pair.get("observed_agreement", "")),
                        str(pair.get("cohen_kappa", "")),
                    )
                )
                + " |"
            )
        lines.append("")
    return lines


def _split_summary_section(splits: JsonDict) -> list[str]:
    """Format optional train/validation/test split metadata."""

    lines = ["## Split Summary", ""]
    if not splits or not splits.get("enabled", False):
        return [*lines, "- Split generation: not requested", ""]
    lines.extend(
        [
            f"- Split generation: enabled",
            f"- Strategy: {splits.get('split_strategy', '')}",
            f"- Seed: {splits.get('split_seed', '')}",
        ]
    )
    lines.extend(_dict_bullets("Fractions", splits.get("split_fractions", {})))
    lines.extend(_dict_bullets("Row counts", splits.get("split_row_counts", {})))
    lines.extend(_dict_bullets("Group counts", splits.get("split_group_counts", {})))
    warnings = splits.get("warnings", [])
    if warnings:
        lines.extend(_dict_bullets("Split warnings", {str(i + 1): item for i, item in enumerate(warnings)}))
    lines.append("")
    return lines


def _issue_section(title: str, issues: list[JsonDict]) -> list[str]:
    """Format a validation issue section with counts and examples."""

    if not issues:
        return []
    lines = [f"## {title}", "", f"Total: {len(issues)}", ""]
    code_counts = Counter(str(issue.get("code", "unknown")) for issue in issues)
    lines.extend(["| Code | Count |", "| --- | ---: |"])
    for code, count in sorted(code_counts.items()):
        lines.append(f"| `{_md(code)}` | {count} |")
    lines.extend(["", "Examples:", ""])
    lines.extend(["| Source row | Code | Field | Target | Message |", "| ---: | --- | --- | --- | --- |"])
    for issue in issues[:5]:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(issue.get("row_number", "")),
                    f"`{_md(str(issue.get('code', '')))}`",
                    _md(str(issue.get("field_name", ""))),
                    _md(str(issue.get("target_key", ""))),
                    _md(str(issue.get("message", ""))),
                )
            )
            + " |"
        )
    lines.append("")
    return lines


def _recommendations(
    readiness: JsonDict,
    warnings: list[JsonDict],
    skipped_rows: list[JsonDict],
    duplicates: list[JsonDict],
    conflicts: list[JsonDict],
) -> list[str]:
    """Return review-oriented next actions based on QA issues."""

    lines = ["## Recommended Review Actions", ""]
    actions: list[str] = []
    if skipped_rows:
        actions.append("Inspect `skipped_rows.jsonl` before using labels for training.")
    if duplicates:
        actions.append("Review `duplicates.jsonl` to confirm repeated rows were accidental.")
    if conflicts:
        actions.append("Review `conflicts.jsonl` before treating labels as adjudicated gold data.")
    if warnings:
        actions.append("Inspect `warnings.jsonl` for unknown fields or missing reviewer metadata.")
    if readiness.get("status") == "not ready for training" and not any(
        (warnings, skipped_rows, duplicates, conflicts)
    ):
        actions.append(str(readiness.get("recommended_next_action", "")).strip())
    if not actions:
        actions.append("No immediate QA action is required before exploratory use.")
    lines.extend(f"- {action}" for action in actions)
    lines.append("")
    return lines


def _format_reviewers(reviewers: object) -> str:
    """Format reviewer IDs without implying adjudication."""

    if not reviewers:
        return "none recorded"
    if isinstance(reviewers, list):
        return ", ".join(str(item) for item in reviewers) or "none recorded"
    return str(reviewers)


def _md(value: str) -> str:
    """Escape table-breaking Markdown characters in generated reports."""

    return value.replace("|", "\\|").replace("\n", " ").strip()
