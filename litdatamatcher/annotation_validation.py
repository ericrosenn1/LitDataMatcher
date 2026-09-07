"""Validation helpers for review-derived annotation corpora."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


JsonDict = dict[str, Any]

RELEVANCE_FIELDS = ("match_relevance", "expert_match_relevance", "expert_relevance")
QUALITY_FIELDS = (
    "expert_question_quality",
    "expert_data_match_quality",
    "question_quality_score",
    "data_match_quality_score",
)
NOTE_FIELDS = ("expert_notes", "review_notes", "notes")
ANNOTATOR_FIELDS = ("annotator_id", "reviewer_id", "expert_id")

KNOWN_REVIEW_FIELDS = {
    "_source_review_file",
    "_source_row_number",
    "_source_format",
    "rank",
    "match_id",
    "question_id",
    "dataset_id",
    "source_id",
    "primary_source_id",
    "document_id",
    "source_ids",
    "document_ids",
    "evidence_dois",
    "evidence_titles",
    "evidence_sections",
    "evidence_sentence_indices",
    "score",
    "question",
    "dataset_title",
    "dataset_source",
    "required_variables",
    "missing_variables",
    "rationale",
    "recommended_design",
    "score_components",
    "score_components_json",
    "assessments",
    "assessments_json",
    "match",
    "match_relevance",
    "expert_match_relevance",
    "expert_relevance",
    "expert_question_quality",
    "expert_data_match_quality",
    "question_quality_score",
    "data_match_quality_score",
    "expert_notes",
    "review_notes",
    "notes",
    "annotator_id",
    "reviewer_id",
    "expert_id",
}


@dataclass(slots=True)
class ReviewIssue:
    """A row-level warning, skip reason, duplicate, or conflict."""

    code: str
    message: str
    severity: str = "warning"
    row_number: int = 0
    source_review_file: str = ""
    field_name: str = ""
    target_key: str = ""
    annotator_id: str = ""
    value: str = ""
    row: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        """Serialize the issue for JSONL QA artifacts."""

        data = asdict(self)
        return {key: value for key, value in data.items() if value not in ("", 0, {}, [])}


@dataclass(slots=True)
class ReviewValidationResult:
    """Validated rows plus QA issues discovered before label export."""

    source_rows: list[JsonDict]
    usable_rows: list[JsonDict]
    warnings: list[ReviewIssue] = field(default_factory=list)
    skipped_rows: list[ReviewIssue] = field(default_factory=list)
    duplicates: list[ReviewIssue] = field(default_factory=list)
    conflicts: list[ReviewIssue] = field(default_factory=list)

    def summary(self) -> JsonDict:
        """Return corpus-level validation counts for manifests."""

        reviewer_rows: dict[str, int] = {}
        for row in self.usable_rows:
            reviewer = str(row.get("_effective_annotator_id", "")).strip()
            if reviewer:
                reviewer_rows[reviewer] = reviewer_rows.get(reviewer, 0) + 1
        reviewers = sorted(reviewer_rows)
        return {
            "source_rows": len(self.source_rows),
            "valid_rows": len(self.usable_rows),
            "usable_rows": len(self.usable_rows),
            "skipped_rows": len(self.skipped_rows),
            "warnings": len(self.warnings),
            "duplicates": len(self.duplicates),
            "conflicts": len(self.conflicts),
            "reviewers": reviewers,
            "reviewer_count": len(reviewers),
            "rows_per_reviewer": reviewer_rows,
        }


def validate_review_rows(
    rows: Iterable[dict],
    annotator_id: str = "",
    include_unlabeled: bool = False,
) -> ReviewValidationResult:
    """Validate review rows and return rows safe to convert into labels."""

    source_rows = [dict(row) for row in rows]
    usable_rows: list[JsonDict] = []
    warnings: list[ReviewIssue] = []
    skipped: list[ReviewIssue] = []
    duplicates: list[ReviewIssue] = []
    conflicts: list[ReviewIssue] = []
    seen_exact: dict[tuple[str, str, str], JsonDict] = {}
    seen_by_annotator: dict[tuple[str, str], JsonDict] = {}
    seen_by_target: dict[str, list[JsonDict]] = {}

    for fallback_index, row in enumerate(source_rows, 1):
        _ensure_source_row(row, fallback_index)
        row["_effective_annotator_id"] = _effective_annotator(row, annotator_id)
        row_warnings, row_skips = validate_review_row(row, include_unlabeled)
        warnings.extend(row_warnings)
        if row_skips:
            skipped.extend(row_skips)
            continue

        target_key = _target_key(row)
        label_signature = _label_signature(row)
        annotator = str(row.get("_effective_annotator_id", ""))
        exact_key = (target_key, annotator, label_signature)
        annotator_key = (target_key, annotator)

        if exact_key in seen_exact:
            issue = _issue(
                row,
                "duplicate_row",
                "Duplicate annotation row skipped to avoid double-counting.",
                severity="skip",
                target_key=target_key,
            )
            duplicates.append(issue)
            skipped.append(issue)
            continue

        prior_same_annotator = seen_by_annotator.get(annotator_key)
        if prior_same_annotator and _label_signature(prior_same_annotator) != label_signature:
            issue = _issue(
                row,
                "conflicting_label",
                "Same annotator provided conflicting labels for the same target.",
                severity="skip",
                target_key=target_key,
            )
            conflicts.append(issue)
            skipped.append(issue)
            continue

        for prior in seen_by_target.get(target_key, []):
            if (
                str(prior.get("_effective_annotator_id", "")) != annotator
                and _label_signature(prior) != label_signature
            ):
                conflicts.append(
                    _issue(
                        row,
                        "cross_reviewer_disagreement",
                        "Different reviewers disagree on the same annotation target.",
                        target_key=target_key,
                    )
                )
                break

        seen_exact[exact_key] = row
        seen_by_annotator[annotator_key] = row
        seen_by_target.setdefault(target_key, []).append(row)
        usable_rows.append(row)

    return ReviewValidationResult(
        source_rows=source_rows,
        usable_rows=usable_rows,
        warnings=warnings,
        skipped_rows=skipped,
        duplicates=duplicates,
        conflicts=conflicts,
    )


def validate_review_row(row: dict, include_unlabeled: bool = False) -> tuple[list[ReviewIssue], list[ReviewIssue]]:
    """Validate one review row without checking cross-row duplicates."""

    warnings: list[ReviewIssue] = []
    skipped: list[ReviewIssue] = []
    has_label = _has_training_label(row)
    needs_ids = has_label or include_unlabeled

    if not str(row.get("_effective_annotator_id", "")).strip():
        warnings.append(
            _issue(row, "blank_annotator_id", "No annotator/reviewer ID was provided.")
        )

    if needs_ids:
        for field_name in ("match_id", "question_id", "dataset_id"):
            if not str(_review_value(row, field_name)).strip():
                skipped.append(
                    _issue(
                        row,
                        f"missing_{field_name}",
                        f"Required field {field_name} is blank for a labeled row.",
                        severity="skip",
                        field_name=field_name,
                    )
                )

    for field_name in RELEVANCE_FIELDS:
        skipped.extend(_validate_numeric_field(row, field_name, maximum=5.0))
    for field_name in QUALITY_FIELDS:
        skipped.extend(_validate_numeric_field(row, field_name, maximum=5.0))

    warnings.extend(_unknown_field_warnings(row))
    return warnings, skipped


def issue_dicts(issues: Iterable[ReviewIssue]) -> list[JsonDict]:
    """Convert validation issues to dictionaries for JSONL writing."""

    return [issue.to_dict() for issue in issues]


def _ensure_source_row(row: JsonDict, fallback_index: int) -> None:
    """Attach source-row metadata when loading code did not provide it."""

    row.setdefault("_source_row_number", fallback_index)
    row.setdefault("_source_review_file", "")
    row.setdefault("_source_format", "")


def _effective_annotator(row: dict, default: str = "") -> str:
    """Return row-level annotator metadata with CLI/default fallback."""

    for field_name in ANNOTATOR_FIELDS:
        value = str(row.get(field_name, "") or "").strip()
        if value:
            return value
    return str(default or "").strip()


def _has_training_label(row: dict) -> bool:
    """Return true when a row contains any reviewer-supplied label content."""

    return any(str(row.get(field_name, "") or "").strip() for field_name in (*RELEVANCE_FIELDS, *QUALITY_FIELDS, *NOTE_FIELDS))


def _validate_numeric_field(row: dict, field_name: str, maximum: float) -> list[ReviewIssue]:
    """Validate an optional numeric score field."""

    raw = str(row.get(field_name, "") or "").strip()
    if not raw:
        return []
    try:
        value = float(raw)
    except ValueError:
        return [
            _issue(
                row,
                "malformed_numeric_score",
                f"Field {field_name} must be numeric when supplied.",
                severity="skip",
                field_name=field_name,
                value=raw,
            )
        ]
    if value < 0 or value > maximum:
        return [
            _issue(
                row,
                "score_out_of_range",
                f"Field {field_name} must be between 0 and {maximum:g}.",
                severity="skip",
                field_name=field_name,
                value=raw,
            )
        ]
    return []


def _unknown_field_warnings(row: dict) -> list[ReviewIssue]:
    """Warn about unexpected user-facing columns while ignoring internal metadata."""

    warnings: list[ReviewIssue] = []
    for field_name in row:
        if field_name.startswith("_") or field_name in KNOWN_REVIEW_FIELDS:
            continue
        if field_name.startswith("score_") or field_name.startswith("feasibility_"):
            continue
        if field_name.startswith("governance_"):
            continue
        warnings.append(
            _issue(
                row,
                "unknown_review_field",
                f"Unrecognized review field {field_name!r} was preserved but not interpreted.",
                field_name=field_name,
            )
        )
    return warnings


def _review_value(row: dict, key: str, default: object = "") -> object:
    """Read a top-level review value with nested-match fallback."""

    if str(row.get(key, "") or "").strip():
        return row.get(key, default)
    match = row.get("match", {})
    if not isinstance(match, dict):
        return default
    if key == "match_id":
        return match.get("match_id", default)
    if key == "question_id":
        question = match.get("question", {})
        return question.get("question_id", default) if isinstance(question, dict) else default
    if key == "dataset_id":
        dataset = match.get("dataset", {})
        return dataset.get("dataset_id", default) if isinstance(dataset, dict) else default
    return default


def _first_review_field(row: dict, *keys: str) -> str:
    """Return the first nonblank field value as text."""

    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _target_key(row: dict) -> str:
    """Return a stable target key for duplicate/conflict detection."""

    return "|".join(
        str(_review_value(row, field_name, "") or "").strip()
        for field_name in ("match_id", "question_id", "dataset_id")
    )


def _label_signature(row: dict) -> str:
    """Return normalized reviewer label content for duplicate checks."""

    relevance = _normalized_relevance_text(row)
    question_quality = _normalized_score_text(
        _first_review_field(row, "expert_question_quality", "question_quality_score")
    )
    data_quality = _normalized_score_text(
        _first_review_field(row, "expert_data_match_quality", "data_match_quality_score")
    )
    notes = _first_review_field(row, *NOTE_FIELDS).strip().lower()
    return "|".join((relevance, question_quality, data_quality, notes))


def _normalized_relevance_text(row: dict) -> str:
    """Normalize accepted relevance ranges for comparison."""

    raw = _first_review_field(row, *RELEVANCE_FIELDS)
    if not raw:
        return ""
    try:
        value = float(raw)
    except ValueError:
        return raw.strip().lower()
    if value > 1:
        value = min(5.0, value) / 5.0
    return f"{max(0.0, min(1.0, value)):.3f}"


def _normalized_score_text(raw: str) -> str:
    """Normalize optional 0..5 scores for comparison."""

    if not raw:
        return ""
    try:
        value = float(raw)
    except ValueError:
        return raw.strip().lower()
    return f"{max(0.0, min(5.0, value)):.3f}"


def _issue(
    row: dict,
    code: str,
    message: str,
    severity: str = "warning",
    field_name: str = "",
    target_key: str = "",
    value: str = "",
) -> ReviewIssue:
    """Build a validation issue with stable row metadata."""

    return ReviewIssue(
        code=code,
        message=message,
        severity=severity,
        row_number=int(row.get("_source_row_number") or 0),
        source_review_file=str(row.get("_source_review_file", "") or ""),
        field_name=field_name,
        target_key=target_key or _target_key(row),
        annotator_id=str(row.get("_effective_annotator_id", "") or ""),
        value=value,
        row={key: value for key, value in row.items() if not key.startswith("_")},
    )
