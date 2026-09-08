from litdatamatcher.cross_source_adversarial import build_cross_source_receipt


def test_integrated_cross_source_adversarial_receipt_is_deterministic_and_passing():
    first = build_cross_source_receipt()
    second = build_cross_source_receipt()
    assert first["validation_status"] == "PASS"
    assert first["input_digest"] == second["input_digest"]
    assert first["outcomes"]["independent_support"] is None
