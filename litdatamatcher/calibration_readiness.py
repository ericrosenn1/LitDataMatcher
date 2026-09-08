"""Fail-closed V2.4 calibration-readiness scorecards."""
from __future__ import annotations

import math
from collections import Counter


FORBIDDEN_CALIBRATION_DIMENSIONS = {"novelty", "unresolvedness", "scientific_significance"}


def build_calibration_scorecard(records: list[dict], *, split_family: str) -> dict:
    """Report metrics only for retained, source-determined labels with valid denominators."""
    retained, pending, excluded, ambiguous = [], [], [], []
    for row in records:
        status = str(row.get("record_status", "")).upper()
        origin = str(row.get("label_origin", "")).casefold()
        if status == "EXCLUDED":
            excluded.append(row); continue
        if status == "AMBIGUOUS":
            ambiguous.append(row); continue
        if origin == "pending_expert":
            pending.append(row); continue
        if origin != "source_determined" or not row.get("label_provenance"):
            excluded.append(row); continue
        retained.append(row)
    invalid_dimension = [row for row in retained if str(row.get("dimension", "")).casefold() in FORBIDDEN_CALIBRATION_DIMENSIONS]
    valid = [row for row in retained if row not in invalid_dimension and _valid_label_row(row, split_family)]
    reasons = []
    if pending: reasons.append("pending_expert_labels")
    if ambiguous: reasons.append("ambiguous_records")
    if invalid_dimension: reasons.append("noncalibratable_scientific_dimension")
    if len(valid) != len(retained) - len(invalid_dimension): reasons.append("invalid_label_provenance_or_denominator")
    labels = [int(row["label"]) for row in valid]
    if not valid:
        reasons.append("no_valid_source_determined_labels")
    elif not (0 in labels and 1 in labels):
        reasons.append("single_class_denominator")
    status = "CALIBRATED" if not reasons else ("PENDING_EXPERT_REVIEW" if pending else "NOT_CALIBRATED")
    metrics = _metrics(valid) if status == "CALIBRATED" else None
    return {"schema_version": "v2_4_calibration_scorecard_v1", "split_family": split_family, "calibration_status": status, "counts": {"retained_source_determined": len(retained), "valid_calibration_rows": len(valid), "pending_expert": len(pending), "excluded": len(excluded), "ambiguous": len(ambiguous)}, "reason_codes": sorted(set(reasons)), "retained_label_provenance": [row["label_provenance"] for row in valid], "metrics": metrics, "ablation_reporting": _ablation(valid) if metrics else None, "limitations": "Novelty, unresolvedness, and scientific significance are never calibrated here; expert labels remain pending."}


def _valid_label_row(row: dict, split_family: str) -> bool:
    return row.get("split_family") == split_family and type(row.get("label")) is int and row["label"] in {0, 1} and type(row.get("score")) in {int, float} and math.isfinite(row["score"])


def _metrics(rows: list[dict]) -> dict:
    predictions = [int(row["score"] >= 0.5) for row in rows]
    labels = [row["label"] for row in rows]
    return {"denominator": len(rows), "accuracy_at_0_5": sum(p == y for p, y in zip(predictions, labels, strict=True)) / len(rows), "positive_labels": sum(labels), "negative_labels": len(rows) - sum(labels)}


def _ablation(rows: list[dict]) -> dict:
    groups = Counter(str(row.get("ablation", "full")) for row in rows)
    return {"status": "DESCRIPTIVE_ONLY", "denominators_by_ablation": dict(sorted(groups.items()))}
