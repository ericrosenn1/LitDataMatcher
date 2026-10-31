"""Review packets must leave scientific labels and reviewer identities empty."""

import pytest

from scripts.campaign_20261031.prepare_scientific_review import (
    CANDIDATE_FIELDS,
    QUESTION_FIELDS,
    _blank,
    _dataset_evidence,
    validate_reviews,
)


def test_blank_review_fields_pass_and_filled_judgment_is_rejected():
    question = {
        "review_item_id": "q",
        "question": "Exact original question?",
        **_blank(QUESTION_FIELDS),
    }
    candidate = {
        "review_item_id": "c",
        "question": "Exact original question?",
        **_blank(CANDIDATE_FIELDS),
    }
    validate_reviews([question], [candidate])
    candidate["full_feasibility"] = "exact_fit"
    with pytest.raises(ValueError, match="human fields"):
        validate_reviews([question], [candidate])


def test_reserved_evaluation_cannot_enter_dataset_evidence():
    with pytest.raises(ValueError, match="Reserved"):
        _dataset_evidence({"dataset_id": "GSE279879"})


def test_dataset_evidence_excludes_ranking_but_preserves_unknowns():
    result = _dataset_evidence(
        {
            "dataset_id": "fixture",
            "score": 1,
            "capabilities": {"outcome": {"status": "unknown", "value": None}},
            "metadata": {"source": "fixture"},
        }
    )
    assert "score" not in result
    assert result["capabilities"]["outcome"]["value"] is None
