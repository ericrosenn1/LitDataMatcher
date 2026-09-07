"""Lightweight reviewer-agreement QA for exported annotation labels."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

from .schemas import JsonDict, QuestionDataMatchLabel


AGREEMENT_SCHEMA_VERSION = "annotation_agreement_v1"
AGREEMENT_QA_LIMITATIONS = (
    "Agreement metrics are lightweight binary QA summaries over exported "
    "question-data match labels; they are not final validation statistics."
)


def build_agreement_artifacts(
    match_labels: Iterable[QuestionDataMatchLabel],
    conflict_rows: Iterable[JsonDict] | None = None,
) -> tuple[JsonDict, list[JsonDict]]:
    """Return agreement summary and adjudication records for match labels."""

    labels = list(match_labels)
    conflicts = list(conflict_rows or [])
    labels_by_target = _labels_by_target(labels)
    reviewer_ids = sorted(
        {label.annotator_id for label in labels if str(label.annotator_id or "").strip()}
    )
    pair_summaries = _pair_summaries(labels_by_target)
    disagreement_records = _cross_reviewer_disagreements(labels_by_target)
    validation_records = _validation_conflict_records(conflicts)
    adjudication_records = [*validation_records, *disagreement_records]
    target_count = len(labels_by_target)
    labeled_targets = [
        target_labels
        for target_labels in labels_by_target.values()
        if any(_binary_relevance(label) is not None for label in target_labels)
    ]
    multi_reviewed_targets = [
        target_labels
        for target_labels in labels_by_target.values()
        if len({label.annotator_id for label in target_labels if label.annotator_id}) >= 2
    ]
    agreement = sum(int(pair["agreement_count"]) for pair in pair_summaries)
    disagreement = sum(int(pair["disagreement_count"]) for pair in pair_summaries)
    overlap = sum(int(pair["overlap_count"]) for pair in pair_summaries)
    summary = {
        "schema_version": AGREEMENT_SCHEMA_VERSION,
        "reviewer_count": len(reviewer_ids),
        "reviewers": reviewer_ids,
        "target_count": target_count,
        "labeled_target_count": len(labeled_targets),
        "multi_reviewed_target_count": len(multi_reviewed_targets),
        "reviewer_pairs": pair_summaries,
        "reviewer_pair_count": len(pair_summaries),
        "total_pair_overlap_count": overlap,
        "total_pair_agreement_count": agreement,
        "total_pair_disagreement_count": disagreement,
        "observed_agreement": round(agreement / overlap, 3) if overlap else None,
        "adjudication_needed_count": len(adjudication_records),
        "metric_limitations": AGREEMENT_QA_LIMITATIONS,
    }
    return summary, adjudication_records


def _labels_by_target(labels: list[QuestionDataMatchLabel]) -> dict[str, list[QuestionDataMatchLabel]]:
    """Group match labels by the ranked match they judge."""

    by_target: dict[str, list[QuestionDataMatchLabel]] = {}
    for label in labels:
        target_id = str(label.match_id or "").strip()
        if not target_id:
            continue
        by_target.setdefault(target_id, []).append(label)
    return by_target


def _pair_summaries(
    labels_by_target: dict[str, list[QuestionDataMatchLabel]]
) -> list[JsonDict]:
    """Compute pairwise binary overlap and agreement counts."""

    pair_values: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for labels in labels_by_target.values():
        by_reviewer: dict[str, int] = {}
        for label in labels:
            reviewer = str(label.annotator_id or "").strip()
            binary = _binary_relevance(label)
            if not reviewer or binary is None:
                continue
            by_reviewer.setdefault(reviewer, binary)
        for reviewer_a, reviewer_b in combinations(sorted(by_reviewer), 2):
            pair_values.setdefault((reviewer_a, reviewer_b), []).append(
                (by_reviewer[reviewer_a], by_reviewer[reviewer_b])
            )

    summaries: list[JsonDict] = []
    for (reviewer_a, reviewer_b), values in sorted(pair_values.items()):
        overlap = len(values)
        agreement = sum(1 for a_value, b_value in values if a_value == b_value)
        positive_agreement = sum(1 for a_value, b_value in values if a_value == b_value == 1)
        negative_agreement = sum(1 for a_value, b_value in values if a_value == b_value == 0)
        disagreement = overlap - agreement
        summaries.append(
            {
                "reviewer_a": reviewer_a,
                "reviewer_b": reviewer_b,
                "overlap_count": overlap,
                "agreement_count": agreement,
                "positive_agreement_count": positive_agreement,
                "negative_agreement_count": negative_agreement,
                "disagreement_count": disagreement,
                "observed_agreement": round(agreement / overlap, 3) if overlap else None,
                "cohen_kappa": _cohen_kappa(values),
            }
        )
    return summaries


def _cross_reviewer_disagreements(
    labels_by_target: dict[str, list[QuestionDataMatchLabel]]
) -> list[JsonDict]:
    """Return adjudication records where reviewers disagree on a target."""

    records: list[JsonDict] = []
    for target_id, labels in sorted(labels_by_target.items()):
        by_reviewer: dict[str, list[QuestionDataMatchLabel]] = {}
        binary_values: dict[str, int] = {}
        for label in labels:
            reviewer = str(label.annotator_id or "").strip() or "blank"
            by_reviewer.setdefault(reviewer, []).append(label)
            binary = _binary_relevance(label)
            if binary is not None:
                binary_values.setdefault(reviewer, binary)
        if len(set(binary_values.values())) <= 1:
            continue
        representative = labels[0]
        records.append(
            {
                "target_type": "question_data_match",
                "target_id": target_id,
                "match_id": representative.match_id,
                "question_id": representative.question_id,
                "dataset_id": representative.dataset_id,
                "primary_source_id": _metadata_value(representative, "primary_source_id", "source_id"),
                "source_id": _metadata_value(representative, "source_id", "primary_source_id"),
                "reviewers": sorted(by_reviewer),
                "labels_by_reviewer": {
                    reviewer: [_label_snapshot(label) for label in reviewer_labels]
                    for reviewer, reviewer_labels in sorted(by_reviewer.items())
                },
                "disagreement_type": "cross_reviewer_disagreement",
                "source_files": _source_files(labels),
                "notes": _notes(labels),
            }
        )
    return records


def _validation_conflict_records(conflicts: list[JsonDict]) -> list[JsonDict]:
    """Convert validation conflicts into adjudication records."""

    records: list[JsonDict] = []
    for issue in conflicts:
        code = str(issue.get("code", "") or "")
        if code == "cross_reviewer_disagreement":
            continue
        row = issue.get("row", {}) if isinstance(issue.get("row", {}), dict) else {}
        records.append(
            {
                "target_type": "question_data_match",
                "target_id": str(issue.get("target_key", "") or row.get("match_id", "")),
                "match_id": str(row.get("match_id", "")),
                "question_id": str(row.get("question_id", "")),
                "dataset_id": str(row.get("dataset_id", "")),
                "primary_source_id": str(
                    row.get("primary_source_id", "") or row.get("source_id", "")
                ),
                "source_id": str(row.get("source_id", "") or row.get("primary_source_id", "")),
                "reviewers": [str(issue.get("annotator_id", "") or "")],
                "labels_by_reviewer": {
                    str(issue.get("annotator_id", "") or "blank"): {
                        "raw_match_relevance": str(row.get("match_relevance", "") or ""),
                        "expert_question_quality": str(row.get("expert_question_quality", "") or ""),
                        "expert_data_match_quality": str(
                            row.get("expert_data_match_quality", "") or ""
                        ),
                    }
                },
                "disagreement_type": code or "validation_conflict",
                "source_files": [str(issue.get("source_review_file", "") or "")]
                if issue.get("source_review_file", "")
                else [],
                "notes": str(row.get("expert_notes", "") or row.get("notes", "") or ""),
            }
        )
    return records


def _binary_relevance(label: QuestionDataMatchLabel) -> int | None:
    """Map normalized match relevance labels to binary QA values."""

    if label.relevance_score is not None:
        return 1 if float(label.relevance_score) > 0 else 0
    if label.label == "relevant":
        return 1
    if label.label == "not_relevant":
        return 0
    return None


def _cohen_kappa(values: list[tuple[int, int]]) -> float | None:
    """Compute binary Cohen's kappa for one reviewer pair."""

    if not values:
        return None
    n = len(values)
    observed = sum(1 for a_value, b_value in values if a_value == b_value) / n
    a_pos = sum(a_value for a_value, _ in values) / n
    b_pos = sum(b_value for _, b_value in values) / n
    expected = (a_pos * b_pos) + ((1 - a_pos) * (1 - b_pos))
    if expected == 1:
        return None
    return round((observed - expected) / (1 - expected), 3)


def _label_snapshot(label: QuestionDataMatchLabel) -> JsonDict:
    """Return compact label context for adjudication records."""

    return {
        "label_id": label.label_id,
        "label": label.label,
        "relevance_score": label.relevance_score,
        "binary_relevance": _binary_relevance(label),
        "question_quality_score": label.question_quality_score,
        "data_match_quality_score": label.data_match_quality_score,
        "notes": label.notes,
        "source_review_file": label.metadata.get("source_review_file", ""),
    }


def _metadata_value(label: QuestionDataMatchLabel, *keys: str) -> str:
    """Return the first nonblank metadata value."""

    for key in keys:
        value = label.metadata.get(key, "")
        if isinstance(value, list):
            value = value[0] if value else ""
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _source_files(labels: list[QuestionDataMatchLabel]) -> list[str]:
    """Collect review source files represented in a disagreement."""

    files: list[str] = []
    for label in labels:
        source = str(label.metadata.get("source_review_file", "") or "").strip()
        if source and source not in files:
            files.append(source)
    return files


def _notes(labels: list[QuestionDataMatchLabel]) -> str:
    """Join short reviewer notes for adjudication context."""

    notes = [label.notes for label in labels if str(label.notes or "").strip()]
    return " | ".join(notes)
