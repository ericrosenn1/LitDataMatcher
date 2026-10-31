"""Constructed API fixtures verify linked metadata and scientific boundaries."""

import pytest

from litdatamatcher.ncbi_metadata_adapters import BioProjectDatasetAdapter, BioSampleDatasetAdapter
from litdatamatcher.scientific_v2 import assess_requirements
from litdatamatcher.v2 import normalize_dataset


class Client:
    last_response_metadata = {}

    def __init__(self, summary, search=None):
        self.summary = summary
        self.search = search or {"esearchresult": {"idlist": ["1"], "count": "1"}}
        self.calls = []

    def get_json(self, url, params=None):
        self.calls.append((url, params))
        return self.search if url.endswith("esearch.fcgi") else self.summary


def test_project_observed_organism_without_inferred_measurements():
    client = Client(
        {
            "result": {
                "1": {
                    "uid": "1",
                    "project_acc": "PRJNA1",
                    "project_title": "Fixture",
                    "organism_name": "Homo sapiens",
                }
            }
        }
    )
    raw = BioProjectDatasetAdapter(client).search("fixture")[0].to_dict()
    record = normalize_dataset(raw)
    assert raw["metadata"]["entity_type"] == "project_metadata"
    assert (
        assess_requirements([{"field": "species", "expected": "human"}], record)["eligibility"]
        == "DIRECT_FIT"
    )
    assert (
        assess_requirements([{"field": "outcome", "expected": "survival"}], record)["eligibility"]
        == "REQUIRES_INSPECTION"
    )
    assert raw["sample_size"] == 0


def test_sample_retains_parent_link_and_exact_attributes():
    xml = '<BioSample accession="SAMN1"><Attributes><Attribute attribute_name="tissue">pancreas</Attribute></Attributes><Links><Link target="bioproject" label="PRJNA1">1</Link></Links></BioSample>'
    client = Client(
        {
            "result": {
                "1": {
                    "uid": "1",
                    "accession": "SAMN1",
                    "title": "Fixture",
                    "organism": "Homo sapiens",
                    "sampledata": xml,
                }
            }
        }
    )
    record = BioSampleDatasetAdapter(client).search("fixture")[0]
    assert record.metadata["entity_type"] == "sample_metadata"
    assert record.metadata["linked_bioprojects"] == ["PRJNA1"]
    assert record.metadata["reported_attributes"] == {"tissue": ["pancreas"]}
    assert client.calls[0][1]["db"] == "biosample"


def test_missing_summary_is_partial_source_failure():
    adapter = BioProjectDatasetAdapter(Client({"result": {"uids": ["1"]}}))
    assert adapter.search("fixture") == []
    assert adapter.last_search_status["status"] == "SCHEMA_DRIFT"
    assert adapter.last_search_status["missing_ids"] == ["1"]


@pytest.mark.parametrize(
    "search",
    [
        {"error": "limit"},
        {"esearchresult": {"count": 1, "idlist": ["1", "1"]}},
        {"esearchresult": {"count": True, "idlist": []}},
    ],
)
def test_search_errors_and_identity_drift_are_not_clean_empty_results(search):
    with pytest.raises(ValueError):
        BioProjectDatasetAdapter(Client({}, search)).search("fixture")


@pytest.mark.parametrize(
    "xml", ['<!DOCTYPE x><BioSample accession="SAMN1"/>', '<BioSample accession="SAMN2"/>']
)
def test_sample_xml_schema_and_identity_guard(xml):
    payload = {"result": {"1": {"uid": "1", "accession": "SAMN1", "sampledata": xml}}}
    with pytest.raises(ValueError):
        BioSampleDatasetAdapter(Client(payload)).search("fixture")
