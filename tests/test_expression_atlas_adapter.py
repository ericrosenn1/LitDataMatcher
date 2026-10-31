"""Constructed catalogues test source schema, bounded search and missingness."""

import pytest

from litdatamatcher.expression_atlas_adapter import ExpressionAtlasDatasetAdapter


class Client:
    last_response_metadata = {}

    def __init__(self, payload):
        self.payload = payload

    def get_json(self, url):
        return self.payload


def test_catalogue_search_keeps_assays_distinct_from_subjects():
    fixture = {
        "experimentAccession": "E-MTAB-1",
        "experimentDescription": "Human diabetes study",
        "species": "Homo sapiens",
        "numberOfAssays": 100,
        "experimentalFactors": ["disease"],
    }
    adapter = ExpressionAtlasDatasetAdapter(Client({"experiments": [fixture]}))
    record = adapter.search("diabetes")[0]
    assert record.sample_size == 0
    assert record.metadata["reported_assay_count"] == 100
    assert record.metadata["field_evidence"]["organisms"]["value"] == ["Homo sapiens"]
    assert adapter.search("unrelated") == []
    assert adapter.last_search_status["matching_catalogue_records"] == 0


def test_retained_records_are_bounded_with_true_local_match_count():
    fixtures = [
        {"experimentAccession": f"E-MTAB-{i}", "experimentDescription": "diabetes"}
        for i in range(40)
    ]
    adapter = ExpressionAtlasDatasetAdapter(Client({"experiments": fixtures}))
    assert len(adapter.search("diabetes")) == 25
    assert adapter.last_search_status["matching_catalogue_records"] == 40


@pytest.mark.parametrize(
    "payload", [{}, {"experiments": [None]}, {"experiments": [{"experimentAccession": "bad"}]}]
)
def test_catalogue_drift_fails_closed(payload):
    with pytest.raises(ValueError):
        ExpressionAtlasDatasetAdapter(Client(payload)).search("diabetes")
