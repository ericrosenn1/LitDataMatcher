"""Known-target metrics must not mislabel unjudged datasets as negatives."""

import pytest

from scripts.campaign_20261031.benchmark_dataset_ranking import candidate_text, metrics


def test_known_target_recovery_counts_missing_and_below_cutoff():
    cases = [
        {"case_id": "one", "selected_dataset": "A"},
        {"case_id": "two", "selected_dataset": "B"},
    ]
    result = metrics(cases, [[{"id": "unjudged"}, {"id": "A"}], []], 5)
    assert result["mrr_at_k"] == 0.25
    assert result["known_target_recall_at_k"] == 0.5
    assert result["known_target_hit_at_1"] == 0
    assert metrics(cases, [[{"id": "unjudged"}, {"id": "A"}], []], 1)["mrr_at_k"] == 0


def test_duplicate_rank_identity_is_rejected():
    with pytest.raises(ValueError):
        metrics([{"case_id": "one", "selected_dataset": "A"}], [[{"id": "A"}, {"id": "A"}]], 5)


def test_retrieval_text_does_not_include_reference_labels_or_inferred_variables():
    record = {
        "dataset_id": "A",
        "title": "Real source title",
        "question": "Do not leak target question",
        "variables": [{"name": "text_inferred_feature"}],
        "inspection": {
            "variables": {
                "age": {"source_definition": {"description": "age at visit", "units": "years"}}
            }
        },
    }
    text = candidate_text(record)
    assert "age at visit" in text and "years" in text
    assert "Do not leak" not in text and "text_inferred_feature" not in text
