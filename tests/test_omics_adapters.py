"""Bounded metadata contracts use constructed fixtures, never synthetic biological evidence."""

import pytest

from litdatamatcher.omics_adapters import MetabolomicsWorkbenchDatasetAdapter, PRIDEDatasetAdapter
from litdatamatcher.scientific_v2 import assess_requirements
from litdatamatcher.v2 import normalize_dataset


class Client:
    last_response_metadata = {"retrieval_time_utc": "2026-09-13T00:00:00Z", "cache_content_sha256": "a" * 64}

    def __init__(self, data):
        self.data = data
        self.calls = []

    def get_json(self, url, params=None):
        self.calls.append((url, params))
        return self.data


def test_pride_preserves_protocols_and_never_infers_units_from_files():
    client = Client([{"accession": "PXD123456", "title": "Fixture proteomics", "organisms": ["Homo sapiens (human)"], "projectFileNames": ["measurements.raw", "proteins.txt"], "sampleProcessingProtocol": "Fixture source method", "updatedDate": "2026-09-01"}])
    record = PRIDEDatasetAdapter(client).search("fixture")[0]
    assert len(client.calls) == 1 and record.sample_size == 0
    normalized = normalize_dataset(record.to_dict())
    assessment = assess_requirements([{"field": "species", "expected": "Homo sapiens"}, {"field": "modality", "expected": "proteomics"}], normalized)
    assert assessment["eligibility"] == "DIRECT_FIT" and assessment["independent_units"] is None
    assert assess_requirements([{"field": "feature_unit", "expected": "intensity"}], normalized)["eligibility"] == "REQUIRES_INSPECTION"
    assert record.metadata["source_protocols"]["sampleProcessingProtocol"] == "Fixture source method"


def test_metabolomics_preserves_declared_count_without_turning_it_into_donors():
    client = Client({"study_id": "ST123456", "study_title": "Fixture metabolomics", "species": "Homo sapiens", "number_of_samples": "24", "analysis_type": "LC-MS", "license": "CC BY 4.0"})
    record = MetabolomicsWorkbenchDatasetAdapter(client).search("ST123456")[0]
    assert record.metadata["reported_sample_count"] == "24" and record.sample_size == 0
    assert "/study_id/ST123456/summary" in client.calls[0][0]
    assessment = assess_requirements([{"field": "donor_count", "expected": 24}], normalize_dataset(record.to_dict()))
    assert assessment["eligibility"] == "REQUIRES_INSPECTION"


@pytest.mark.parametrize("cls,payload", [(PRIDEDatasetAdapter, {}), (PRIDEDatasetAdapter, [{"accession": "bad", "title": "invalid"}]), (MetabolomicsWorkbenchDatasetAdapter, {"error": "not found"}), (MetabolomicsWorkbenchDatasetAdapter, {"study_id": "bad", "study_title": "invalid"})])
def test_schema_drift_does_not_become_a_clean_empty_catalog(cls, payload):
    with pytest.raises(ValueError):
        cls(Client(payload)).search("fixture")


def test_title_query_is_encoded_as_one_path_value():
    client = Client({})
    assert MetabolomicsWorkbenchDatasetAdapter(client).search("diabetes / cancer?") == []
    assert "/study_title/diabetes%20%2F%20cancer%3F/summary" in client.calls[0][0]

