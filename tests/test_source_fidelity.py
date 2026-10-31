"""Source evidence must remain distinct from absence, inference, and replay time."""

import json
import os
from pathlib import Path

import pytest

from litdatamatcher.adapters import (
    ClinicalTrialsDatasetAdapter,
    OpenAlexLiteratureAdapter,
    search_dataset_sources,
)
from litdatamatcher.capability_registry import infer_dataset_capabilities
from litdatamatcher.datasets import classify_dataset_record
from litdatamatcher.governance import assess_governance
from litdatamatcher.http_cache import CachedHttpClient
from litdatamatcher.schemas import DatasetRecord


class FakeClient:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.last_response_metadata = {
            "retrieval_time_utc": "2001-01-01T00:00:00Z",
            "cache_status": "cache_hit",
            "cache_content_sha256": "a" * 64,
        }

    def get_json(self, *args, **kwargs):
        return next(self.payloads)


def test_clinical_invalid_identifiers_cannot_be_reported_as_empty_success():
    rows = search_dataset_sources(
        "brain",
        ["clinicaltrials"],
        FakeClient(
            [{"studies": [{"protocolSection": {"identificationModule": {"nctId": "invalid"}}}]}]
        ),
    )
    assert rows == []
    status = rows.source_statuses[0]
    assert status["status"] == "UNKNOWN_RETRIEVAL_OR_SCHEMA_FAILURE"
    assert status["pagination"]["rejected_record_indices"] == [0]
    assert not status["pagination"]["completed"]


def test_clinical_rejects_bad_identity_but_preserves_valid_records():
    adapter = ClinicalTrialsDatasetAdapter(
        FakeClient(
            [
                {
                    "studies": [
                        {"protocolSection": {"identificationModule": {"nctId": "bad"}}},
                        {
                            "protocolSection": {
                                "identificationModule": {
                                    "nctId": "NCT12345678",
                                    "briefTitle": "Valid",
                                }
                            }
                        },
                    ]
                }
            ]
        )
    )
    rows = adapter.search("brain")
    assert len(rows) == 1
    assert adapter.last_search_status["status"] == "SCHEMA_DRIFT"
    assert rows[0].metadata["pagination"]["rejected_record_indices"] == [0]


@pytest.mark.parametrize(
    "summary", [{"error": "failure"}, {"result": {}}, {"result": {"123": "bad"}}]
)
def test_geo_invalid_summary_never_creates_fallback_dataset(summary):
    result = search_dataset_sources(
        "brain", ["geo"], FakeClient([{"esearchresult": {"idlist": ["123"]}}, summary])
    )
    assert not result
    assert result.source_statuses[0]["status"] == "UNKNOWN_RETRIEVAL_OR_SCHEMA_FAILURE"


def test_openalex_replay_uses_acquisition_time_and_hash():
    record = OpenAlexLiteratureAdapter(
        FakeClient(
            [
                {
                    "results": [
                        {
                            "id": "https://openalex.org/W123",
                            "title": "Example",
                            "abstract_inverted_index": {},
                        }
                    ]
                }
            ]
        )
    ).search_literature("brain")[0]
    provenance = record["source_provenance"]
    assert provenance["retrieval_time_utc"] == "2001-01-01T00:00:00Z"
    assert provenance["metadata"]["cache_snapshot"]["cache_content_sha256"] == "a" * 64


def test_clinical_missing_fields_are_not_observed_or_derivation_inputs():
    record = ClinicalTrialsDatasetAdapter(
        FakeClient(
            [
                {
                    "studies": [
                        {
                            "protocolSection": {
                                "identificationModule": {
                                    "nctId": "NCT12345678",
                                    "briefTitle": "Observational cohort",
                                },
                                "designModule": {"studyType": "OBSERVATIONAL"},
                            }
                        }
                    ]
                }
            ]
        )
    ).search("cohort")[0]
    capabilities = infer_dataset_capabilities(record)
    observed = {cap.variable_name for cap in capabilities if cap.capability_type == "observed"}
    assert not {"treatment", "outcome", "condition", "eligibility_criteria"} & observed
    assert not any(cap.variable_name == "treatment_response" for cap in capabilities)
    assert record.sample_size == 0


def test_inferred_variables_stay_unknown_capabilities():
    record = classify_dataset_record(
        {"title": "Human longitudinal transcriptomics age study", "source": "test"}
    )
    capabilities = infer_dataset_capabilities(record)
    assert capabilities
    assert all(cap.capability_type == "unknown" for cap in capabilities)
    assert all(cap.metadata["evidence_status"] == "INFERRED" for cap in capabilities)


def test_explicit_variable_still_supports_legacy_derivation():
    record = classify_dataset_record(
        {
            "title": "Explicit source",
            "source": "test",
            "variables": [
                {"name": "treatment", "completeness": 1},
                {"name": "outcome", "completeness": 1},
            ],
        }
    )
    assert any(
        cap.variable_name == "treatment_response" and cap.capability_type == "derived"
        for cap in infer_dataset_capabilities(record)
    )


def test_restriction_overrides_public_metadata_under_versioned_policy():
    record = DatasetRecord(
        "x", "Human cohort", "test", access_type="public metadata; controlled individual data"
    )
    current = assess_governance(record)
    legacy = assess_governance(record, policy="legacy-v1")
    assert current.access_score == 0.35
    assert "controlled_or_restricted_access" in current.risk_flags
    assert current.policy_version == "metadata-aware-v2"
    assert legacy.access_score == 0.9


def test_cache_acquisition_survives_modified_payload_mtime(tmp_path, monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": "source content"}

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: Response())
    client = CachedHttpClient(cache_dir=tmp_path)
    url = "https://source.invalid/record"
    client.get_json(url)
    recorded = dict(client.last_response_metadata)
    path = client._cache_path(url, None)
    os.utime(path, (1, 1))
    offline = CachedHttpClient(cache_dir=tmp_path, offline=True)
    assert offline.get_json(url) == {"message": "source content"}
    assert offline.last_response_metadata["retrieval_time_utc"] == recorded["retrieval_time_utc"]
    assert offline.last_response_metadata["retrieval_time_basis"] == "recorded_content_acquisition"


def test_legacy_cache_timestamp_is_explicitly_unverified(tmp_path):
    client = CachedHttpClient(cache_dir=tmp_path, offline=True)
    path = client._cache_path("https://source.invalid/record", None)
    path.write_text(json.dumps({"legacy": True}), encoding="utf-8")
    client.get_json("https://source.invalid/record")
    assert client.last_response_metadata["retrieval_time_basis"] == "legacy_cache_mtime_unverified"


def test_geo_partial_summary_retains_valid_records_and_reports_missing_ids():
    result = search_dataset_sources(
        "brain",
        ["geo"],
        FakeClient(
            [
                {"esearchresult": {"idlist": ["123", "456"]}},
                {"result": {"123": {"accession": "GSE123", "title": "Valid source"}}},
            ]
        ),
    )
    assert [record.dataset_id for record in result] == ["GSE123"]
    assert result.source_statuses[0]["status"] == "UNKNOWN_RETRIEVAL_OR_SCHEMA_FAILURE"
    assert result.source_statuses[0]["pagination"]["missing_ids"] == ["456"]


def test_cache_envelope_mismatch_fails_closed(tmp_path, monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"source": "record"}

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: Response())
    client = CachedHttpClient(cache_dir=tmp_path)
    client.get_json("https://source.invalid/record")
    receipt = Path(client.last_response_metadata["acquisition_envelope_path"])
    data = json.loads(receipt.read_text())
    data["cache_content_sha256"] = "wrong"
    receipt.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        CachedHttpClient(cache_dir=tmp_path, offline=True).get_json("https://source.invalid/record")


def test_identical_refresh_keeps_first_acquisition_and_reports_refresh(tmp_path, monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"source": "record"}

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: Response())
    client = CachedHttpClient(cache_dir=tmp_path)
    client.get_json("https://source.invalid/record")
    timestamp = client.last_response_metadata["retrieval_time_utc"]
    client.get_json("https://source.invalid/record", refresh=True)
    assert client.last_response_metadata["retrieval_time_utc"] == timestamp
    assert client.last_response_metadata["cache_status"] == "live_refreshed"


def test_cache_receipt_redacts_credential_query_parameters(tmp_path, monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"result": "public metadata"}

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: Response())
    client = CachedHttpClient(cache_dir=tmp_path)
    client.get_json(
        "https://source.invalid/record?api_key=secret", {"token": "secret", "query": "brain"}
    )
    scope = client.last_response_metadata["request_scope"]
    assert "secret" not in scope
    assert "brain" in scope


def test_proxy_capability_is_not_reported_as_observed():
    record = classify_dataset_record(
        {
            "title": "Proxy",
            "source": "test",
            "variables": [
                {
                    "name": "predictor",
                    "category": "derived_or_proxy_capability",
                    "completeness": 0.8,
                },
            ],
        }
    )
    capability = infer_dataset_capabilities(record)[0]
    assert capability.capability_type == "unknown"
    assert capability.metadata["evidence_status"] == "INFERRED"
