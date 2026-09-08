import pytest

from litdatamatcher.expert_review import (
    REVIEW_STATUS,
    agreement_and_adjudication,
    build_blinded_review_packet,
    validate_review_labels,
    finalize_adjudication,
)


def source_determined_record():
    return {
        "match_id": "match-source-determined",
        "label_origin": "source_determined",
        "question": {"question": "Does source-described exposure affect outcome?", "source_ids": ["PMID:1"], "evidence": [{"source_locator": "PMC:span:1", "text": "source span"}]},
        "dataset": {"dataset_id": "GSE1", "title": "Source-described dataset", "source": "GEO", "organisms": ["Homo sapiens"], "assay_types": ["RNA-seq"]},
        "score": 0.99,
        "rank": 1,
    }


def test_blinded_packet_masks_scores_and_preserves_source_span():
    result = build_blinded_review_packet([source_determined_record()], ["expert-a", "expert-b"])
    item = result["packet"]["items"][0]
    assert result["packet"]["review_status"] == REVIEW_STATUS
    assert "score" not in item and "rank" not in item
    assert item["question_source_spans"][0]["source_locator"] == "PMC:span:1"
    assert result["linkage"][0]["match_id"] == "match-source-determined"


def test_packet_rejects_duplicate_reviewer_identity_and_packet_leakage():
    with pytest.raises(ValueError, match="unique non-empty reviewer"):
        build_blinded_review_packet([source_determined_record()], ["expert-a", "expert-a"])
    leaky = source_determined_record()
    leaky["question"]["evidence"] = [{"score": 1, "source_locator": "span"}]
    with pytest.raises(ValueError, match="masked field"):
        build_blinded_review_packet([leaky], ["expert-a"])


def test_label_validation_and_adjudication_keep_expert_review_pending():
    packet = build_blinded_review_packet([source_determined_record()], ["expert-a", "expert-b"])["packet"]
    item_id = packet["items"][0]["review_item_id"]
    checked = validate_review_labels(packet, [
        {"reviewer_id": "expert-a", "review_item_id": item_id, "labels": {"relevance": "relevant", "question_validity": "valid"}},
        {"reviewer_id": "expert-b", "review_item_id": item_id, "labels": {"relevance": "not_relevant", "question_validity": "valid"}},
        {"reviewer_id": "expert-a", "review_item_id": item_id, "labels": {"relevance": "relevant"}},
        {"reviewer_id": "expert-c", "review_item_id": item_id, "labels": {"novelty": "invented"}},
    ])
    assert checked["status"] == REVIEW_STATUS
    assert len(checked["valid_labels"]) == 2
    assert {row["reason"] for row in checked["invalid_labels"]} == {"duplicate_reviewer_identity_label", "unsupported_label_value"}
    agreement = agreement_and_adjudication(checked["valid_labels"])
    assert agreement["status"] == REVIEW_STATUS
    assert agreement["observed_agreement"] == 0.5
    assert agreement["adjudication_records"][0]["dimension"] == "relevance"


def test_adjudication_is_blind_explicit_and_calibration_gated():
    packet = build_blinded_review_packet([source_determined_record()], ["a", "b"])["packet"]
    item = packet["items"][0]["review_item_id"]
    labels = validate_review_labels(packet, [{"reviewer_id":"a","review_item_id":item,"labels":{"relevance":"relevant"}}, {"reviewer_id":"b","review_item_id":item,"labels":{"relevance":"not_relevant"}}])["valid_labels"]
    pending = finalize_adjudication(packet, labels, [], policy_id="p1")
    assert pending["agreement_status"] == "LOW_AGREEMENT"
    assert pending["calibration_eligibility"] == "PENDING_EXPERT_REVIEW"
    done = finalize_adjudication(packet, labels, [{"review_item_id":item,"dimension":"relevance","adjudicator_id":"chair","decision":"relevant","rationale":"source evidence"}], policy_id="p1")
    assert done["status"] == "ADJUDICATED"
    assert not done["reviewer_blind_agreement"][0].keys() & {"reviewer_a","reviewer_b"}
    leaked = finalize_adjudication(packet, labels, [{"review_item_id":item,"dimension":"relevance","adjudicator_id":"chair","decision":"relevant","rationale":"x","reviewer_id":"a"}], policy_id="p1")
    assert leaked["invalid_decisions"][0]["reason"] == "reviewer_or_model_leakage"
