"""Descriptive label QA and explicitly scoped, versioned isotonic calibration.

The scorecard does not fit anything. Calibration is a separate opt-in operation
with a supplied protocol and disjoint fit/validation groups. Test fixtures can
exercise that operation but cannot establish scientific calibration.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import Counter, defaultdict
from copy import deepcopy

from .data_plane import digest

FORBIDDEN_CALIBRATION_DIMENSIONS = {"novelty", "unresolvedness", "scientific_significance"}
CALIBRATION_DIMENSIONS = {
    "relevance",
    "question_validity",
    "dataset_compatibility",
    "answerability",
}
CALIBRATION_SCHEMA = "grouped_isotonic_calibration_v1"


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _finite(value: object, low: float, high: float) -> bool:
    try:
        return type(value) in {int, float} and math.isfinite(value) and low <= value <= high
    except OverflowError:
        return False


def _valid_provenance(value: object, *, expert: bool = False) -> bool:
    if not isinstance(value, dict) or not _text(value.get("source_locator")):
        return False
    return not expert or bool(
        _text(value.get("reviewer_id")) and value.get("method") == "human_review"
    )


def _checked_rows(records: list[dict], split_family: str) -> dict:
    """Quarantine malformed/duplicate identities before constructing a denominator."""
    valid, invalid, pending, excluded, ambiguous = [], [], [], [], []
    reasons = set()
    if not _text(split_family):
        reasons.add("invalid_split_family")
    candidates = []
    for source in records:
        if not isinstance(source, dict):
            invalid.append(source)
            reasons.add("invalid_label_provenance_or_denominator")
            continue
        row = deepcopy(source)
        status = _text(row.get("record_status")).upper()
        origin = _text(row.get("label_origin")).casefold()
        if status == "EXCLUDED":
            excluded.append(row)
            continue
        if status == "AMBIGUOUS":
            ambiguous.append(row)
            reasons.add("ambiguous_records")
            continue
        if origin == "pending_expert":
            pending.append(row)
            reasons.add("pending_expert_labels")
            continue
        dimension = _text(row.get("dimension")).casefold()
        if dimension in FORBIDDEN_CALIBRATION_DIMENSIONS:
            reasons.add("noncalibratable_scientific_dimension")
        if (
            status != "RETAINED"
            or origin not in {"source_determined", "expert"}
            or not _text(row.get("record_id"))
            or dimension not in CALIBRATION_DIMENSIONS
            or not _valid_provenance(row.get("label_provenance"), expert=origin == "expert")
            or not _text(row.get("split_family"))
            or row["split_family"] != split_family
            or type(row.get("label")) is not int
            or row["label"] not in {0, 1}
            or not _finite(row.get("score"), 0, 1)
        ):
            invalid.append(row)
            reasons.add("invalid_label_provenance_or_denominator")
            continue
        row.update(
            record_id=_text(row["record_id"]),
            dimension=dimension,
            label_origin=origin,
            record_status=status,
            ablation=_text(row.get("ablation", "full")),
        )
        if not row["ablation"]:
            invalid.append(row)
            reasons.add("invalid_label_provenance_or_denominator")
            continue
        candidates.append(row)
    # One observation may occur in different ablations, but must retain its label
    # and group. Within an ablation all copies of a repeated identity are invalid.
    keys = Counter((r["record_id"], r["dimension"], r["ablation"]) for r in candidates)
    identities = defaultdict(set)
    for row in candidates:
        identities[(row["record_id"], row["dimension"])].add(
            (row["label"], _text(row.get("group_id")))
        )
    for row in candidates:
        if (
            keys[(row["record_id"], row["dimension"], row["ablation"])] > 1
            or len(identities[(row["record_id"], row["dimension"])]) > 1
        ):
            invalid.append(row)
            reasons.add("duplicate_or_conflicting_observation")
        else:
            valid.append(row)
    return {
        "valid": valid,
        "invalid": invalid,
        "pending": pending,
        "excluded": excluded,
        "ambiguous": ambiguous,
        "reasons": reasons,
    }


def build_calibration_scorecard(records: list[dict], *, split_family: str) -> dict:
    """Report valid label accuracy as descriptive QA, never as fitted calibration."""
    checked = _checked_rows(records, split_family)
    rows = checked["valid"]
    reasons = set(checked["reasons"])
    if not rows:
        reasons.add("no_valid_source_determined_labels")
    elif len({r["dimension"] for r in rows}) != 1:
        reasons.add("mixed_label_dimensions")
    elif {r["label"] for r in rows} != {0, 1}:
        reasons.add("single_class_denominator")
    metrics = _descriptive_metrics(rows) if not reasons else None
    return {
        "schema_version": "v2_4_calibration_scorecard_v2",
        "split_family": split_family,
        "calibration_status": "PENDING_EXPERT_REVIEW" if checked["pending"] else "NOT_CALIBRATED",
        "readiness_status": "DESCRIPTIVE_LABEL_QA_READY" if metrics else "LABEL_QA_INCOMPLETE",
        "counts": {
            "retained_source_determined": sum(
                r["label_origin"] == "source_determined" for r in rows
            ),
            "valid_calibration_rows": len(rows),
            "pending_expert": len(checked["pending"]),
            "excluded": len(checked["excluded"]),
            "ambiguous": len(checked["ambiguous"]),
            "invalid": len(checked["invalid"]),
        },
        "reason_codes": sorted(reasons),
        "retained_label_provenance": [r["label_provenance"] for r in rows],
        "metrics": metrics,
        "ablation_reporting": {
            "status": "DESCRIPTIVE_ONLY",
            "denominators_by_ablation": dict(sorted(Counter(r["ablation"] for r in rows).items())),
        }
        if metrics
        else None,
        "limitations": "Descriptive fixed-threshold label QA only; no calibration is fitted. Novelty, unresolvedness and scientific significance are not calibratable dimensions here.",
    }


def _descriptive_metrics(rows: list[dict]) -> dict:
    return {
        "status": "DESCRIPTIVE_ONLY",
        "denominator": len(rows),
        "accuracy_at_0_5": sum(int(r["score"] >= 0.5) == r["label"] for r in rows) / len(rows),
        "positive_labels": sum(r["label"] for r in rows),
        "negative_labels": sum(r["label"] == 0 for r in rows),
    }


def _isotonic_model(rows: list[dict]) -> dict:
    """Pool adjacent violators, pooling tied scores before fitting."""
    by_score = defaultdict(list)
    group_counts = Counter(row["group_id"] for row in rows)
    for row in rows:
        by_score[float(row["score"])].append((row["label"], 1 / group_counts[row["group_id"]]))
    blocks = []
    for score, observations in sorted(by_score.items()):
        blocks.append(
            [
                score,
                score,
                sum(label * weight for label, weight in observations),
                sum(weight for _, weight in observations),
            ]
        )
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    return {
        "method": "isotonic_pava_step_v1",
        "x_min": min(by_score),
        "x_max": max(by_score),
        "blocks": [
            {"lower": b[0], "upper": b[1], "probability": b[2] / b[3], "group_weight": b[3]}
            for b in blocks
        ],
        "weighting": "equal_group_weight",
        "interpolation": "left_step_between_blocks; boundary blocks outside fitted range",
    }


def _predict(model: dict, score: float) -> float:
    block = max(0, bisect_right([b["lower"] for b in model["blocks"]], score) - 1)
    return model["blocks"][block]["probability"]


def fit_validate_calibration(records: list[dict], *, protocol: dict) -> dict:
    """Fit and assess one binary outcome under a caller-supplied frozen protocol.

    No default thresholds or implicit scientific endpoint conversions are used.
    The protocol is retained verbatim. Synthetic inputs always have a separate
    result, including when their numerical fit and validation checks pass.
    """
    if not isinstance(protocol, dict):
        raise ValueError("Calibration requires a versioned, predeclared protocol")
    required_text = (
        "protocol_id",
        "source_locator",
        "split_family",
        "dimension",
        "label_definition",
        "score_definition",
        "partition_digest",
    )
    if (
        any(not _text(protocol.get(key)) for key in required_text)
        or protocol.get("method") != "isotonic_pava_step_v1"
        or protocol.get("predeclared") is not True
    ):
        raise ValueError(
            "Calibration requires an explicit supported protocol and source provenance"
        )
    for key in ("minimum_records_per_split", "minimum_groups_per_split"):
        if type(protocol.get(key)) is not int or protocol[key] < 2:
            raise ValueError(
                "Calibration protocol must require at least two records and groups per split"
            )
    if not _finite(protocol.get("maximum_brier_score"), 0, 1) or not _finite(
        protocol.get("minimum_brier_improvement"), 0, 1
    ):
        raise ValueError(
            "Calibration protocol requires finite, predeclared proper-score thresholds"
        )
    if not isinstance(protocol.get("data_origin"), str) or protocol["data_origin"] not in {
        "real",
        "synthetic_fixture",
    }:
        raise ValueError("Calibration requires explicit real versus synthetic-fixture provenance")
    checked = _checked_rows(records, protocol["split_family"])
    rows = sorted(checked["valid"], key=lambda r: r["record_id"])
    if checked["reasons"] or checked["excluded"] or not rows:
        raise ValueError("Calibration cannot fit invalid, pending, excluded or ambiguous labels")
    if {r["dimension"] for r in rows} != {protocol["dimension"]} or len(
        {r["ablation"] for r in rows}
    ) != 1:
        raise ValueError("Calibration fits one declared outcome and ablation at a time")
    if len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("Calibration observation identities must be unique")
    for row in rows:
        if (
            not _text(row.get("group_id"))
            or not isinstance(row.get("split_role"), str)
            or row["split_role"] not in {"FIT", "VALIDATION"}
            or row.get("data_origin") != protocol["data_origin"]
            or row["label_provenance"].get("label_definition") != protocol["label_definition"]
            or row.get("score_definition") != protocol["score_definition"]
            or not _valid_provenance(row.get("split_provenance"))
        ):
            raise ValueError(
                "Calibration requires group/split, label-definition and score provenance"
            )
        row["group_id"] = _text(row["group_id"])
    partition = [{k: r[k] for k in ("record_id", "group_id", "split_role")} for r in rows]
    if digest(partition) != protocol["partition_digest"]:
        raise ValueError("Calibration records differ from the declared partition")
    fit = [r for r in rows if r["split_role"] == "FIT"]
    validation = [r for r in rows if r["split_role"] == "VALIDATION"]
    groups = [{r["group_id"] for r in part} for part in (fit, validation)]
    if groups[0] & groups[1]:
        raise ValueError("Calibration fit and validation groups overlap")
    for part, group_ids in zip((fit, validation), groups, strict=True):
        if (
            len(part) < protocol["minimum_records_per_split"]
            or len(group_ids) < protocol["minimum_groups_per_split"]
            or {r["label"] for r in part} != {0, 1}
        ):
            raise ValueError(
                "Calibration split lacks the declared denominator, groups or both classes"
            )
    if len({r["score"] for r in fit}) < 2:
        raise ValueError("Calibration fitting requires varying scores")
    model = _isotonic_model(fit)
    predictions = [
        {
            "record_id": r["record_id"],
            "group_id": r["group_id"],
            "label": r["label"],
            "raw_score": r["score"],
            "calibrated_score": _predict(model, r["score"]),
        }
        for r in validation
    ]
    # Give each independently assigned group equal weight in the proper score.
    by_group = defaultdict(list)
    for row in predictions:
        by_group[row["group_id"]].append(row)

    def brier(field):
        return sum(
            sum((r[field] - r["label"]) ** 2 for r in group) / len(group)
            for group in by_group.values()
        ) / len(by_group)

    before, after = brier("raw_score"), brier("calibrated_score")
    passed = (
        after <= protocol["maximum_brier_score"]
        and before - after >= protocol["minimum_brier_improvement"]
    )
    origins = sorted({r["label_origin"] for r in rows})
    status = "FITTED_NOT_VALIDATED"
    if passed:
        status = (
            "SYNTHETIC_VALIDATION_ONLY"
            if protocol["data_origin"] == "synthetic_fixture"
            else (
                "EXPERT_CALIBRATED_IN_SCOPE"
                if origins == ["expert"]
                else "SOURCE_CALIBRATION_VALIDATED_IN_SCOPE"
            )
        )
    result = {
        "schema_version": CALIBRATION_SCHEMA,
        "calibration_status": status,
        "protocol": deepcopy(protocol),
        "protocol_digest": digest(protocol),
        "input_records": rows,
        "input_digest": digest(rows),
        "model": model,
        "model_digest": digest(model),
        "label_origins": origins,
        "validation": {
            "status": "PASS" if passed else "FAIL",
            "metric": "equal_group_weighted_brier",
            "fit_records": len(fit),
            "validation_records": len(validation),
            "fit_groups": len(groups[0]),
            "validation_groups": len(groups[1]),
            "raw_brier": before,
            "calibrated_brier": after,
            "improvement": before - after,
            "predictions": predictions,
        },
        "limitations": "Calibration applies only to the declared outcome, score definition, groups and protocol. Source labels are not expert labels. Synthetic validation is machinery evidence only; no novelty or truth probability is established.",
    }
    result["result_digest"] = digest(result)
    return result


def verify_calibration_result(result: object, *, require_real_expert: bool = False) -> bool:
    """Recompute fit and validation; reject forged statuses and stale hashes."""
    try:
        if not isinstance(result, dict) or result.get("schema_version") != CALIBRATION_SCHEMA:
            return False
        recomputed = fit_validate_calibration(result["input_records"], protocol=result["protocol"])
        if digest(recomputed) != digest(result):
            return False
        if require_real_expert:
            return (
                result["calibration_status"] == "EXPERT_CALIBRATED_IN_SCOPE"
                and result["protocol"]["data_origin"] == "real"
            )
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
