"""DataCite contracts use fixture metadata and preserve partial retrievals."""

import pytest

from litdatamatcher.adapters import DataCiteDatasetAdapter, search_dataset_sources


class FakeClient:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.calls = []
        self.last_response_metadata = {}

    def get_json(self, url, params=None):
        self.calls.append((url, params))
        self.last_response_metadata = {"retrieval_time_utc": f"2020-01-0{len(self.calls)}T00:00:00Z", "cache_content_sha256": str(len(self.calls)) * 64, "cache_status": "cache_hit"}
        return next(self.pages)


def record(doi="10.1234/dataset", **overrides):
    return {"id": doi, "type": "dois", "attributes": {
        "doi": doi, "titles": [{"title": "Human transcriptomics dataset"}],
        "types": {"resourceTypeGeneral": "Dataset"},
        "rightsList": [{"rights": "CC BY 4.0", "rightsUri": "https://creativecommons.org/licenses/by/4.0/"}],
        "relatedIdentifiers": [{"relatedIdentifier": "10.1234/paper", "relationType": "IsSupplementTo"}],
        **overrides,
    }}


def test_dataset_filter_pagination_provenance_and_unknown_measurements():
    client = FakeClient([
        {"data": [record()], "links": {"next": "https://api.datacite.org/dois?page%5Bcursor%5D=second"}},
        {"data": [record("10.1234/second"), record()], "links": {"next": None}},
    ])
    adapter = DataCiteDatasetAdapter(client)
    records = adapter.search("brain")
    assert len(records) == 2
    assert client.calls[0][1]["resource-type-id"] == "dataset"
    assert client.calls[1][1]["page[cursor]"] == "second"
    assert records[1].metadata["source_provenance"]["retrieval_time_utc"] == "2020-01-02T00:00:00Z"
    assert records[0].sample_size == 0 and records[0].metadata["sample_count"] is None
    assert all(variable.category == "inferred" for variable in records[0].variables)
    assert records[0].license == "CC BY 4.0"
    assert records[0].metadata["related_identifiers"][0]["relationType"] == "IsSupplementTo"
    assert records[0].metadata["source_provenance"]["metadata"]["source_profile"]["category"] == "dataset_metadata"
    assert adapter.last_search_status["status"] == "COMPLETE"


@pytest.mark.parametrize("bad", [record(types={"resourceTypeGeneral": "Text"}), record("not-a-doi"), record(titles=[]), {"id": "x", "attributes": None}])
def test_invalid_or_non_dataset_records_are_not_synthetic_successes(bad):
    results = search_dataset_sources("brain", ["datacite"], FakeClient([{"data": [bad], "links": {}}]))
    assert not results
    assert results.source_statuses[0]["status"] == "UNKNOWN_RETRIEVAL_OR_SCHEMA_FAILURE"


def test_untrusted_next_url_is_not_followed_and_current_valid_page_survives():
    client = FakeClient([{"data": [record()], "links": {"next": "http://127.0.0.1/private"}}])
    adapter = DataCiteDatasetAdapter(client)
    assert len(adapter.search("brain")) == 1
    assert len(client.calls) == 1
    assert adapter.last_search_status["status"] == "SCHEMA_DRIFT"


def test_repeated_cursor_is_partial_and_stops():
    client = FakeClient([{"data": [record()], "links": {"next": "https://api.datacite.org/dois?page%5Bcursor%5D=1"}}])
    adapter = DataCiteDatasetAdapter(client)
    assert len(adapter.search("brain")) == 1
    assert adapter.last_search_status["status"] == "REPEATED_CURSOR"


def test_legitimate_empty_response_is_distinct_from_provider_error():
    empty = search_dataset_sources("nothing", ["datacite"], FakeClient([{"data": [], "links": {}}]))
    failed = search_dataset_sources("nothing", ["datacite"], FakeClient([{"errors": [{"title": "Unavailable"}]}]))
    assert not empty and empty.source_statuses[0]["status"] == "OBSERVED"
    assert not failed and failed.source_statuses[0]["status"] == "UNKNOWN_RETRIEVAL_OR_SCHEMA_FAILURE"
