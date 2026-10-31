"""Regression cases for real-label boundaries; every label here is a test fixture."""

import hashlib
import json
from copy import deepcopy

import pytest

from litdatamatcher.acceptance import _calibration
from litdatamatcher.calibration_readiness import (
    build_calibration_scorecard,
    fit_validate_calibration,
    verify_calibration_result,
)
from litdatamatcher.data_plane import digest
from litdatamatcher.expert_review import (
    agreement_and_adjudication,
    build_blinded_review_packet,
    finalize_adjudication,
    validate_review_labels,
)


def label_rows():
    base = {
        "record_status": "RETAINED",
        "label_origin": "source_determined",
        "label_provenance": {"source_locator": "test-fixture:label"},
        "split_family": "fixture-family",
        "dimension": "dataset_compatibility",
        "ablation": "full",
    }
    return [
        dict(base, record_id="r1", label=1, score=0.9),
        dict(base, record_id="r2", label=0, score=0.1),
    ]


def disagreement():
    packet = build_blinded_review_packet(
        [
            {
                "question": {
                    "question": "Fixture question",
                    "evidence": [{"source_locator": "test-fixture:span"}],
                },
                "dataset_id": "D1",
            }
        ],
        ["fixture-reviewer-a", "fixture-reviewer-b"],
    )["packet"]
    item_id = packet["items"][0]["review_item_id"]
    labels = validate_review_labels(
        packet,
        [
            {
                "reviewer_id": "fixture-reviewer-a",
                "review_item_id": item_id,
                "labels": {"relevance": "relevant"},
            },
            {
                "reviewer_id": "fixture-reviewer-b",
                "review_item_id": item_id,
                "labels": {"relevance": "not_relevant"},
            },
        ],
    )["valid_labels"]
    decision = {
        "review_item_id": item_id,
        "dimension": "relevance",
        "adjudicator_id": "fixture-chair",
        "decision": "relevant",
        "rationale": "Fixture source evidence",
    }
    return packet, labels, decision


@pytest.mark.parametrize(
    "change",
    [
        {"label_provenance": "garbage"},
        {"split_family": ""},
        {"record_id": "duplicate"},
        {"dimension": "novelty "},
        {"record_status": "FAILED"},
        {"score": 10.0},
    ],
)
def test_invalid_scorecard_rows_never_produce_metrics(change):
    rows = [{**row, **change} for row in label_rows()]
    report = build_calibration_scorecard(
        rows, split_family=change.get("split_family", "fixture-family")
    )
    assert report["calibration_status"] == "NOT_CALIBRATED"
    assert report["metrics"] is None


def test_source_label_accuracy_is_not_fitted_calibration():
    report = build_calibration_scorecard(label_rows(), split_family="fixture-family")
    assert report["calibration_status"] != "CALIBRATED"


@pytest.mark.parametrize(
    "changes,policy_id",
    [
        ({"decision": "invented"}, "fixture-policy"),
        ({}, ""),
    ],
)
def test_invalid_adjudication_never_resolves_a_target(changes, policy_id):
    packet, labels, decision = disagreement()
    result = finalize_adjudication(packet, labels, [{**decision, **changes}], policy_id=policy_id)
    assert result["calibration_eligibility"] == "PENDING_EXPERT_REVIEW"
    assert result["unresolved"]


def test_conflicting_duplicate_adjudication_never_resolves_a_target():
    packet, labels, decision = disagreement()
    result = finalize_adjudication(
        packet,
        labels,
        [decision, {**decision, "decision": "not_relevant"}],
        policy_id="fixture-policy",
    )
    assert result["calibration_eligibility"] == "PENDING_EXPERT_REVIEW"
    assert result["accepted_decisions"] == []
    assert result["unresolved"]


def test_expert_label_origin_alone_is_not_expert_calibration():
    status, evidence = _calibration(
        [
            {
                "manifest": {"evaluation": {"label_origins": ["expert"]}},
                "manifest_path": "fixture/RUN_MANIFEST.json",
            }
        ]
    )
    assert status != "EXPERT_CALIBRATED"


def calibration_fixture():
    """Small declared synthetic fixture, never real or expert validation evidence."""
    rows = []
    for role in ("FIT", "VALIDATION"):
        for group in range(2):
            for label in (0, 1):
                rows.append(
                    {
                        "record_id": f"{role}-{group}-{label}",
                        "record_status": "RETAINED",
                        "label_origin": "source_determined",
                        "data_origin": "synthetic_fixture",
                        "label_provenance": {
                            "source_locator": "test-fixture:label",
                            "label_definition": "fixture binary outcome",
                        },
                        "split_provenance": {"source_locator": "test-fixture:partition"},
                        "split_role": role,
                        "split_family": "fixture-family",
                        "group_id": f"{role}-group-{group}",
                        "dimension": "relevance",
                        "label": label,
                        "score": 0.6 if label else 0.4,
                        "score_definition": "fixture precomputed score",
                        "ablation": "full",
                    }
                )
    protocol = {
        "protocol_id": "fixture-protocol-v1",
        "source_locator": "test-fixture:protocol",
        "split_family": "fixture-family",
        "dimension": "relevance",
        "label_definition": "fixture binary outcome",
        "score_definition": "fixture precomputed score",
        "method": "isotonic_pava_step_v1",
        "predeclared": True,
        "data_origin": "synthetic_fixture",
        "minimum_records_per_split": 4,
        "minimum_groups_per_split": 2,
        "maximum_brier_score": 0.1,
        "minimum_brier_improvement": 0.1,
    }
    protocol["partition_digest"] = partition_digest(rows)
    return rows, protocol


def partition_digest(rows):
    return digest(
        [
            {key: row[key] for key in ("record_id", "group_id", "split_role")}
            for row in sorted(rows, key=lambda row: row["record_id"])
        ]
    )


def test_actual_fit_and_disjoint_validation_remain_fixture_only_and_replay_exactly():
    rows, protocol = calibration_fixture()
    result = fit_validate_calibration(rows, protocol=protocol)
    assert result["calibration_status"] == "SYNTHETIC_VALIDATION_ONLY"
    assert result["validation"]["raw_brier"] == pytest.approx(0.16)
    assert result["validation"]["calibrated_brier"] == 0
    assert [block["probability"] for block in result["model"]["blocks"]] == [0, 1]
    assert fit_validate_calibration(list(reversed(rows)), protocol=protocol) == result
    assert verify_calibration_result(result)
    assert not verify_calibration_result(result, require_real_expert=True)
    assert rows[0]["score"] == 0.4  # Inputs are preserved.


def test_validation_labels_are_never_used_to_fit_and_failed_validation_is_visible():
    rows, protocol = calibration_fixture()
    incumbent = fit_validate_calibration(rows, protocol=protocol)
    for row in rows:
        if row["split_role"] == "VALIDATION":
            row["label"] = 1 - row["label"]
    failed = fit_validate_calibration(rows, protocol=protocol)
    assert failed["model"] == incumbent["model"]
    assert failed["calibration_status"] == "FITTED_NOT_VALIDATED"
    assert failed["validation"]["status"] == "FAIL"


def test_isotonic_fit_pools_the_nonmonotonic_empirical_rates():
    rows, protocol = calibration_fixture()
    for row, score in zip(
        [r for r in rows if r["split_role"] == "FIT"], [0.1, 0.2, 0.3, 0.4], strict=True
    ):
        row["score"] = score
    result = fit_validate_calibration(rows, protocol=protocol)
    # The middle empirical rates are 1 then 0 with equal group weights, so
    # the monotonic least-squares solution has one pooled rate of one half.
    assert [block["probability"] for block in result["model"]["blocks"]] == [0, 0.5, 1]


def test_group_identity_whitespace_cannot_hide_fit_validation_overlap():
    rows, protocol = calibration_fixture()
    rows[-1]["group_id"] = " " + rows[0]["group_id"] + " "
    protocol["partition_digest"] = partition_digest(rows)
    with pytest.raises(ValueError):
        fit_validate_calibration(rows, protocol=protocol)


@pytest.mark.parametrize(
    "mutation",
    [
        "overlap",
        "missing_provenance",
        "changed_partition",
        "one_class",
        "few_groups",
        "unreviewed",
        "mixed_origin",
        "wrong_definition",
        "nan",
        "bool",
    ],
)
def test_fit_rejects_unsupported_provenance_splits_and_denominators(mutation):
    rows, protocol = calibration_fixture()
    if mutation == "overlap":
        rows[-1]["group_id"] = rows[0]["group_id"]
        protocol["partition_digest"] = partition_digest(rows)
    elif mutation == "missing_provenance":
        rows[0]["split_provenance"] = {}
    elif mutation == "changed_partition":
        rows[0]["group_id"] = "other"
    elif mutation == "one_class":
        for row in rows:
            if row["split_role"] == "FIT":
                row["label"] = 1
    elif mutation == "few_groups":
        protocol["minimum_groups_per_split"] = 3
    elif mutation == "unreviewed":
        rows[0]["label_origin"] = "unreviewed"
    elif mutation == "mixed_origin":
        rows[0]["data_origin"] = "real"
    elif mutation == "wrong_definition":
        rows[0]["label_provenance"]["label_definition"] = "other endpoint"
    elif mutation == "nan":
        rows[0]["score"] = float("nan")
    else:
        rows[0]["label"] = True
    with pytest.raises(ValueError):
        fit_validate_calibration(rows, protocol=protocol)


@pytest.mark.parametrize(
    "mutation", ["threshold", "method", "predeclared", "unscoped", "denominator"]
)
def test_no_implicit_calibration_protocol_or_threshold(mutation):
    rows, protocol = calibration_fixture()
    if mutation == "threshold":
        protocol["maximum_brier_score"] = float("inf")
    elif mutation == "method":
        protocol["method"] = "automatic"
    elif mutation == "predeclared":
        protocol["predeclared"] = False
    elif mutation == "unscoped":
        protocol["source_locator"] = ""
    else:
        protocol["minimum_records_per_split"] = True
    with pytest.raises(ValueError):
        fit_validate_calibration(rows, protocol=protocol)


def test_fit_result_status_model_and_label_tampering_are_rejected():
    rows, protocol = calibration_fixture()
    result = fit_validate_calibration(rows, protocol=protocol)
    for part in ("status", "model", "input"):
        forged = deepcopy(result)
        if part == "status":
            forged["calibration_status"] = "EXPERT_CALIBRATED_IN_SCOPE"
        elif part == "model":
            forged["model"]["blocks"][0]["probability"] = 0.8
        else:
            forged["input_records"][0]["score"] = 0.3
        assert not verify_calibration_result(forged)


def test_finalization_revalidates_input_labels_and_retains_adjudicator_provenance():
    packet, labels, decision = disagreement()
    complete = finalize_adjudication(packet, labels, [decision], policy_id="fixture-policy")
    assert complete["accepted_decisions"][0]["adjudicator_id"] == "fixture-chair"
    assert complete["accepted_decisions"][0]["provenance"]["packet_id"] == packet["packet_id"]
    invalid = finalize_adjudication(
        packet, labels + [labels[0]], [decision], policy_id="fixture-policy"
    )
    assert invalid["calibration_eligibility"] == "PENDING_EXPERT_REVIEW"
    assert invalid["invalid_labels"]


def test_review_import_rejects_null_identity_and_packet_tampering():
    packet, labels, decision = disagreement()
    invalid = validate_review_labels(packet, [{**labels[0], "reviewer_id": None}])
    assert invalid["invalid_labels"] and not invalid["valid_labels"]
    packet["items"][0]["question"] = "Altered fixture evidence"
    with pytest.raises(ValueError, match="content identity"):
        finalize_adjudication(packet, labels, [decision], policy_id="fixture-policy")


def test_agreement_rejects_aliases_of_the_same_reviewer_identity():
    _, labels, _ = disagreement()
    with pytest.raises(ValueError, match="Duplicate reviewer"):
        agreement_and_adjudication(
            [labels[0], {**labels[0], "reviewer_id": " " + labels[0]["reviewer_id"] + " "}]
        )


def calibration_run(tmp_path, result):
    """Acceptance wiring fixture; no real expert-validation artifact is produced."""
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return {
        "manifest": {
            "evaluation": {
                "label_origins": ["expert"],
                "split_role": "VALIDATION",
                "holdout_exposed_to_tuning": False,
                "protocol_version": "fixture-protocol-v1",
            }
        },
        "manifest_path": tmp_path / "RUN_MANIFEST.json",
        "checks": [{"target": "G10", "kind": "label_provenance", "artifacts": [path.name]}],
        "validated_artifacts": {
            path.name: {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        },
    }


def test_hashed_synthetic_result_cannot_claim_expert_acceptance(tmp_path):
    rows, protocol = calibration_fixture()
    result = fit_validate_calibration(rows, protocol=protocol)
    run = calibration_run(tmp_path, result)
    assert _calibration([run])[0] == "PENDING_EXPERT_LABELS"
    result["calibration_status"] = "EXPERT_CALIBRATED_IN_SCOPE"
    assert _calibration([calibration_run(tmp_path, result)])[0] == "PENDING_EXPERT_LABELS"


def test_acceptance_requires_hash_protocol_role_and_executed_reference(tmp_path, monkeypatch):
    # Replace only the separately tested real-result verifier to test the
    # acceptance envelope without fabricating real expert labels.
    import litdatamatcher.calibration_readiness as calibration

    monkeypatch.setattr(calibration, "verify_calibration_result", lambda result, **kwargs: True)
    payload = {"protocol": {"protocol_id": "fixture-protocol-v1"}}
    run = calibration_run(tmp_path, payload)
    assert _calibration([run])[0] == "EXPERT_CALIBRATED"
    for mutation in ("hash", "protocol", "role", "reference", "tuning"):
        invalid = deepcopy(run)
        if mutation == "hash":
            invalid["validated_artifacts"]["calibration.json"]["sha256"] = "0" * 64
        elif mutation == "protocol":
            invalid["manifest"]["evaluation"]["protocol_version"] = "another"
        elif mutation == "role":
            invalid["manifest"]["evaluation"]["split_role"] = "DEVELOPMENT"
        elif mutation == "reference":
            invalid["checks"] = []
        else:
            invalid["manifest"]["evaluation"]["holdout_exposed_to_tuning"] = True
        assert _calibration([invalid])[0] == "PENDING_EXPERT_LABELS"
