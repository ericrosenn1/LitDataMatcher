"""Versioned, blinded expert-review packets and conservative label QA."""

from __future__ import annotations

from collections import Counter, defaultdict
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
    "dataset_compatibility": {
        "exact_fit",
        "directly_answerable",
        "partial_fit",
        "indirect_support",
        "requires_additional_data",
        "incompatible",
        "unknown",
    },
    "answerability": {"answerable", "partially_answerable", "requires_additional_data", "unknown"},
    "novelty": {
        "unresolved_in_coverage",
        "answered_in_coverage",
        "insufficient_coverage",
        "uncertain",
    },
    "evidence_classification": {
        "same_underlying_evidence",
        "derivative_evidence",
        "duplicated_cohort",
        "replicated_evidence",
        "orthogonal_evidence",
        "direct_perturbational_evidence",
        "associative_evidence",
        "mechanistic_evidence",
        "indirect_evidence",
        "contradictory_evidence",
        "incompatible_evidence",
        "unknown_dependence",
    },
}
MASKED_KEYS = {
    "score",
    "scores",
    "rank",
    "ranking",
    "prediction",
    "predicted_label",
    "model",
    "heuristic",
    "calibration",
    "gold_label",
    "reviewer_id",
    "annotator_id",
}


def _identity(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _packet_items(packet: dict) -> dict[str, dict]:
    if not isinstance(packet, dict) or not isinstance(packet.get("items"), list):
        raise ValueError("Review packet requires an item list")
    if packet.get("schema_version") != PACKET_SCHEMA_VERSION or packet.get(
        "packet_id"
    ) != stable_id("expert_review_packet", digest(packet["items"])):
        raise ValueError("Review packet version or content identity is invalid")
    result = {}
    for item in packet["items"]:
        item_id = _identity(item.get("review_item_id")) if isinstance(item, dict) else ""
        if not item_id or item_id in result:
            raise ValueError("Review packet requires unique non-empty item identities")
        result[item_id] = item
    return result


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
        return any(
            str(key).casefold() in MASKED_KEYS or _has_masked_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_has_masked_key(item) for item in value)
    return False


def build_blinded_review_packet(records: list[dict], reviewer_ids: list[str]) -> dict:
    """Create source-preserving items without scores, predictions, or reviewer IDs."""
    reviewers = [_identity(value) for value in reviewer_ids]
    if (
        not reviewers
        or any(not value for value in reviewers)
        or len(set(reviewers)) != len(reviewers)
    ):
        raise ValueError("Review packet requires unique non-empty reviewer identities")
    items = []
    linkage = []
    for index, record in enumerate(records):
        question = _question(record)
        dataset = _dataset(record)
        question_text = _identity(question.get("question", record.get("question_text", "")))
        dataset_id = _identity(dataset.get("dataset_id", record.get("dataset_id", "")))
        if not question_text or not dataset_id:
            raise ValueError("Review packet record requires question text and dataset ID")
        match_id = str(
            _value(record, "match_id") or stable_id("packet_input", question_text, dataset_id)
        ).strip()
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
                "source_provenance": dataset.get(
                    "source_provenance", record.get("dataset_source_provenance", [])
                ),
            },
            "evidence": record.get("evidence_items", record.get("evidence", [])),
            "label_origin": str(record.get("label_origin", "unreviewed")),
            "review_status": REVIEW_STATUS,
        }
        if _has_masked_key(item):
            raise ValueError("Blinded packet attempted to retain a masked field")
        items.append(item)
        linkage.append({"review_item_id": item_id, "match_id": match_id})
    # Do not carry the input ranking order into the blinded review material.
    items.sort(key=lambda item: item["review_item_id"])
    packet = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "packet_id": stable_id("expert_review_packet", digest(items)),
        "review_status": REVIEW_STATUS,
        "assignment_count": len(reviewers),
        "items": items,
        "masking": {
            "excluded_fields": sorted(MASKED_KEYS),
            "source_spans_and_provenance_retained": True,
        },
        "limitations": "Packet infrastructure only; no expert labels, adjudication, calibration, or scientific claim is supplied.",
    }
    return {"packet": packet, "linkage": linkage}


def validate_review_labels(packet: dict, labels: list[dict]) -> dict:
    """Validate human-entered categorical labels without assigning any labels."""
    item_ids = set(_packet_items(packet))
    valid, invalid, seen = [], [], set()
    for row in labels:
        if not isinstance(row, dict):
            invalid.append({"row": row, "reason": "label_row_not_object"})
            continue
        reviewer = _identity(row.get("reviewer_id"))
        item_id = _identity(row.get("review_item_id"))
        values = row.get("labels")
        if not reviewer or item_id not in item_ids or not isinstance(values, dict) or not values:
            invalid.append({"row": row, "reason": "missing_or_unknown_identity_or_labels"})
            continue
        normalized = {}
        reason = ""
        for dimension, value in values.items():
            dimension = _identity(dimension).casefold()
            value = _identity(value).casefold()
            key = (reviewer, item_id, dimension)
            if dimension not in LABEL_VALUES or value not in LABEL_VALUES[dimension]:
                reason = "unsupported_label_value"
                break
            if key in seen or dimension in normalized:
                reason = "duplicate_reviewer_identity_label"
                break
            normalized[str(dimension)] = value
        if reason:
            invalid.append({"row": row, "reason": reason})
            continue
        for dimension in normalized:
            seen.add((reviewer, item_id, dimension))
        valid.append(
            {
                "schema_version": LABEL_SCHEMA_VERSION,
                "reviewer_id": reviewer,
                "review_item_id": item_id,
                "labels": normalized,
                "notes": str(row.get("notes", "")),
            }
        )
    return {"status": REVIEW_STATUS, "valid_labels": valid, "invalid_labels": invalid}


def agreement_and_adjudication(labels: list[dict]) -> dict:
    """Report exact categorical agreement and preserve disagreements for adjudication."""
    by_target: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row in labels:
        if (
            not isinstance(row, dict)
            or not _identity(row.get("reviewer_id"))
            or not _identity(row.get("review_item_id"))
            or not isinstance(row.get("labels"), dict)
            or not row["labels"]
        ):
            raise ValueError("Agreement requires validated categorical labels")
        reviewer = _identity(row["reviewer_id"])
        item_id = _identity(row["review_item_id"])
        for dimension, value in row["labels"].items():
            dimension = _identity(dimension).casefold()
            value = _identity(value).casefold()
            if (
                dimension not in LABEL_VALUES
                or not isinstance(value, str)
                or value not in LABEL_VALUES[dimension]
            ):
                raise ValueError("Agreement requires supported categorical labels")
            if reviewer in by_target[(item_id, dimension)]:
                raise ValueError("Duplicate reviewer/item/dimension in agreement input")
            by_target[(item_id, dimension)][reviewer] = value
    pairs, adjudication = [], []
    for (item_id, dimension), reviewer_values in sorted(by_target.items()):
        for reviewer_a, reviewer_b in combinations(sorted(reviewer_values), 2):
            same = reviewer_values[reviewer_a] == reviewer_values[reviewer_b]
            pairs.append(
                {
                    "review_item_id": item_id,
                    "dimension": dimension,
                    "reviewer_a": reviewer_a,
                    "reviewer_b": reviewer_b,
                    "agreement": same,
                }
            )
            if not same:
                adjudication.append(
                    {
                        "review_item_id": item_id,
                        "dimension": dimension,
                        "labels_by_reviewer": {
                            reviewer_a: reviewer_values[reviewer_a],
                            reviewer_b: reviewer_values[reviewer_b],
                        },
                        "status": "PENDING_ADJUDICATION",
                    }
                )
    return {
        "status": REVIEW_STATUS,
        "pairwise_comparisons": pairs,
        "observed_agreement": sum(pair["agreement"] for pair in pairs) / len(pairs)
        if pairs
        else None,
        "adjudication_records": adjudication,
        "limitations": "Descriptive categorical agreement only; no expert consensus, kappa interpretation, calibration, or gold label is claimed.",
    }


def finalize_adjudication(
    packet: dict, labels: list[dict], decisions: list[dict], *, policy_id: str
) -> dict:
    """Resolve only valid, unique policy decisions; this never establishes calibration."""
    items = _packet_items(packet)
    checked = validate_review_labels(packet, labels)
    agreement = agreement_and_adjudication(checked["valid_labels"])
    pending = {
        (row["review_item_id"], row["dimension"]) for row in agreement["adjudication_records"]
    }
    policy = _identity(policy_id)
    keys = Counter(
        (_identity(row.get("review_item_id")), _identity(row.get("dimension")).casefold())
        for row in decisions
        if isinstance(row, dict)
    )
    accepted, invalid = [], []
    for decision in decisions:
        if not isinstance(decision, dict):
            invalid.append({"decision": decision, "reason": "decision_not_object"})
            continue
        key = (
            _identity(decision.get("review_item_id")),
            _identity(decision.get("dimension")).casefold(),
        )
        value = _identity(decision.get("decision")).casefold()
        reason = ""
        if not policy:
            reason = "missing_policy_identity"
        elif checked["invalid_labels"]:
            reason = "invalid_input_labels"
        elif (
            key not in pending
            or not _identity(decision.get("adjudicator_id"))
            or not _identity(decision.get("rationale"))
        ):
            reason = "missing_or_nonpending_adjudication"
        elif keys[key] != 1:
            reason = "duplicate_adjudication_target"
        elif value not in LABEL_VALUES.get(key[1], set()):
            reason = "unsupported_label_value"
        elif _has_masked_key(decision) or "reviewer_identity" in decision:
            reason = "reviewer_or_model_leakage"
        if reason:
            invalid.append({"decision": decision, "reason": reason})
            continue
        item = items[key[0]]
        accepted.append(
            {
                "review_item_id": key[0],
                "dimension": key[1],
                "decision": value,
                "rationale": _identity(decision["rationale"]),
                "policy_id": policy,
                "adjudicator_id": _identity(decision["adjudicator_id"]),
                "provenance": {
                    "packet_id": packet.get("packet_id"),
                    "question_source_spans": item.get("question_source_spans", []),
                    "dataset_provenance": item.get("dataset", {}).get("source_provenance", []),
                },
            }
        )
    unresolved = [
        record
        for record in agreement["adjudication_records"]
        if (record["review_item_id"], record["dimension"])
        not in {(row["review_item_id"], row["dimension"]) for row in accepted}
    ]
    complete = bool(accepted and not unresolved and not invalid and not checked["invalid_labels"])
    return {
        "schema_version": "expert_adjudication_v2",
        "status": "ADJUDICATED" if complete else REVIEW_STATUS,
        "agreement_status": "NOT_COMPUTABLE"
        if agreement["observed_agreement"] is None
        else (
            "LOW_AGREEMENT" if agreement["observed_agreement"] < 0.8 else "DESCRIPTIVE_AGREEMENT"
        ),
        "reviewer_blind_agreement": [
            {k: v for k, v in row.items() if k not in {"reviewer_a", "reviewer_b"}}
            for row in agreement["pairwise_comparisons"]
        ],
        "accepted_decisions": accepted,
        "unresolved": unresolved,
        "invalid_decisions": invalid,
        "invalid_labels": checked["invalid_labels"],
        "calibration_eligibility": "ADJUDICATED_POLICY_LABELS_ONLY" if complete else REVIEW_STATUS,
        "limitations": "Validated policy decisions only; no calibration fit, expert credentials, or independent scientific validation is inferred.",
    }
