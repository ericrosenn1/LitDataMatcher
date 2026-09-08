from litdatamatcher.calibration_readiness import build_calibration_scorecard


def row(**overrides):
    base = {"record_id": "r1", "record_status": "RETAINED", "label_origin": "source_determined", "label_provenance": {"source_locator": "fixture:1"}, "split_family": "source-family-a", "dimension": "dataset_compatibility", "label": 1, "score": 0.9, "ablation": "full"}
    return {**base, **overrides}


def test_scorecard_calibrates_only_valid_source_determined_binary_denominator():
    report = build_calibration_scorecard([row(), row(record_id="r2", label=0, score=0.1, ablation="no_provenance")], split_family="source-family-a")
    assert report["calibration_status"] == "CALIBRATED"
    assert report["metrics"]["denominator"] == 2
    assert report["ablation_reporting"]["denominators_by_ablation"] == {"full": 1, "no_provenance": 1}


def test_pending_expert_ambiguous_and_novelty_never_become_calibrated():
    report = build_calibration_scorecard([row(label_origin="pending_expert"), row(record_status="AMBIGUOUS"), row(dimension="novelty")], split_family="source-family-a")
    assert report["calibration_status"] == "PENDING_EXPERT_REVIEW"
    assert report["metrics"] is None
    assert {"pending_expert_labels", "ambiguous_records", "noncalibratable_scientific_dimension"} <= set(report["reason_codes"])


def test_invalid_provenance_split_or_single_class_is_not_calibrated():
    report = build_calibration_scorecard([row(label_provenance={}), row(record_id="r2", split_family="other")], split_family="source-family-a")
    assert report["calibration_status"] == "NOT_CALIBRATED"
    assert report["metrics"] is None
