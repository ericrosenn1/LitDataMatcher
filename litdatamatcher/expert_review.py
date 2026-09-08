"""Versioned, blinded expert-review packets and conservative label QA."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

from .data_plane import digest
from .schemas import stable_id


PACKET_SCHEMA_VERSION = "expert_review_packet_v1"
LABEL_SCHEMA_VERSION = "expert_review_label_v1"
REVIEW_STATUS = "PENDING_EXPERT_REVIEW"
LABEL_VALUES = {
    "relevance": {"relevant", "not_relevant", "uncertain"},
    "question_validity": {"valid", "invalid", "uncertain"},
    "dataset_compatibility": {"exact_fit", "directly_answerable", "partial_fit", "indirect_support", "requires_additional_data", "incompatible", "unknown"},
    "answerability": {"answerable", "partially_answerable", "requires_additional_data", "unknown"},
    "novelty": {"unresolved_in_coverage", "answered_in_coverage", "insufficient_coverage", "uncertain"},
    "evidence_classification": {"same_underlying_evidence", "derivative_evidence", "duplicated_cohort", "replicated_evidence", "orthogonal_evidence", "direct_perturbational_evidence", "associative_evidence", "mechanistic_evidence", "indirect_evidence", "contradictory_evidence", "incompatible_evidence", "unknown_dependence"},
}
MASKED_KEYS = {"score", "scores", "rank", "ranking", "prediction", "predicted_label", "model", "heuristic", "calibration", "gold_label", "reviewer_id", "annotator_id"}


def _value(record: dict, key: str) -> Any:
    if key in record:
        return record[key]
    match = record.get("match", {})
    return match.get(key) if isinstance(match, dict) else None


def _question(record: dict) -> dict:
    value = _value(record, "question")
    return value if isinstance(value, dict) else {}


def _dataset(record: dict) -> dict:
    value = _value(record, "dataset")
    return value if isinstance(value, dict) else {}


def _has_masked_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(str(key).casefold() in MASKED_KEYS or _has_masked_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_has_masked_key(item) for item in value)
    return False


def build_blinded_review_packet(records: list[dict], reviewer_ids: list[str]) -> dict:
    """Create source-preserving items without scores, predictions, or reviewer IDs."""
    reviewers = [str(value).strip() for value in reviewer_ids]
    if not reviewers or any(not value for value in reviewers) or len(set(reviewers)) != len(reviewers):
        raise ValueError("Review packet requires unique non-empty reviewer identities")
    items = []
    linkage = []
    for index, record in enumerate(records):
        question = _question(record)
        dataset = _dataset(record)
        question_text = str(question.get("question", record.get("question_text", ""))).strip()
        dataset_id = str(dataset.get("dataset_id", record.get("dataset_id", ""))).strip()
        if not question_text or not dataset_id:
            raise ValueError("Review packet record requires question text and dataset ID")
        match_id = str(_value(record, "match_id") or stable_id("packet_input", question_text, dataset_id)).strip()
        item_id = stable_id("blind_review", match_id, index)
        source_spans = question.get("evidence", question.get("evidence_spans", []))
        item = {
            "review_item_id": item_id,
            "question": question_text,
            "question_source_ids": list(question.get("source_ids", [])),
            "question_source_spans": source_spans if isinstance(source_spans, list) else [],
            "dataset": {
                "dataset_id": dataset_id,
                "title": str(dataset.get("title", record.get("dataset_title", ""))),
                "source": str(dataset.get("source", record.get("dataset_source", ""))),
                "organisms": list(dataset.get("organisms", [])),
                "assay_types": list(dataset.get("assay_types", [])),
                "access_type": str(dataset.get("access_type", "unknown")),
                "source_provenance": dataset.get("source_provenance", record.get("dataset_source_provenance", [])),
            },
            "evidence": record.get("evidence_items", record.get("evidence", [])),
            "label_origin": str(record.get("label_origin", "unreviewed")),
            "review_status": REVIEW_STATUS,
        }
        if _has_masked_key(item):
            raise ValueError("Blinded packet attempted to retain a masked field")
        items.append(item)
        linkage.append({"review_item_id": item_id, "match_id": match_id})
    packet = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "packet_id": stable_id("expert_review_packet", digest(items)),
        "review_status": REVIEW_STATUS,
        "assignment_count": len(reviewers),
        "items": items,
        "masking": {"excluded_fields": sorted(MASKED_KEYS), "source_spans_and_provenance_retained": True},
        "limitations": "Packet infrastructure only; no expert labels, adjudication, calibration, or scientific claim is supplied.",
    }
    return {"packet": packet, "linkage": linkage}


def validate_review_labels(packet: dict, labels: list[dict]) -> dict:
    """Validate human-entered categorical labels without assigning any labels."""
    item_ids = {item["review_item_id"] for item in packet.get("items", [])}
    valid, invalid, seen = [], [], set()
    for row in labels:
        reviewer = str(row.get("reviewer_id", "")).strip()
        item_id = str(row.get("review_item_id", "")).strip()
        values = row.get("labels")
        if not reviewer or item_id not in item_ids or not isinstance(values, dict) or not values:
            invalid.append({"row": row, "reason": "missing_or_unknown_identity_or_labels"})
            continue
        normalized = {}
        reason = ""
        for dimension, value in values.items():
            value = str(value).strip()
            key = (reviewer, item_id, str(dimension))
            if dimension not in LABEL_VALUES or value not in LABEL_VALUES[dimension]:
                reason = "unsupported_label_value"
                break
            if key in seen:
                reason = "duplicate_reviewer_identity_label"
                break
            normalized[str(dimension)] = value
        if reason:
            invalid.append({"row": row, "reason": reason})
            continue
        for dimension in normalized:
            seen.add((reviewer, item_id, dimension))
        valid.append({"schema_version": LABEL_SCHEMA_VERSION, "reviewer_id": reviewer, "review_item_id": item_id, "labels": normalized, "notes": str(row.get("notes", ""))})
    return {"status": REVIEW_STATUS, "valid_labels": valid, "invalid_labels": invalid}


def agreement_and_adjudication(labels: list[dict]) -> dict:
    """Report exact categorical agreement and preserve disagreements for adjudication."""
    by_target: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row in labels:
        for dimension, value in row["labels"].items():
            by_target[(row["review_item_id"], dimension)][row["reviewer_id"]] = value
    pairs, adjudication = [], []
    for (item_id, dimension), reviewer_values in sorted(by_target.items()):
        for reviewer_a, reviewer_b in combinations(sorted(reviewer_values), 2):
            same = reviewer_values[reviewer_a] == reviewer_values[reviewer_b]
            pairs.append({"review_item_id": item_id, "dimension": dimension, "reviewer_a": reviewer_a, "reviewer_b": reviewer_b, "agreement": same})
            if not same:
                adjudication.append({"review_item_id": item_id, "dimension": dimension, "labels_by_reviewer": {reviewer_a: reviewer_values[reviewer_a], reviewer_b: reviewer_values[reviewer_b]}, "status": "PENDING_ADJUDICATION"})
    return {"status": REVIEW_STATUS, "pairwise_comparisons": pairs, "observed_agreement": sum(pair["agreement"] for pair in pairs) / len(pairs) if pairs else None, "adjudication_records": adjudication, "limitations": "Descriptive categorical agreement only; no expert consensus, kappa interpretation, calibration, or gold label is claimed."}
