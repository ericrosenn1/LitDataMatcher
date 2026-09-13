"""Acquisition errors remain visible even when no record can carry provenance."""

import json
from copy import deepcopy

import pytest

from litdatamatcher.adapters import EuropePMCLiteratureAdapter
from litdatamatcher.cli import main
from litdatamatcher.literature_integrity import (
    consolidate_literature_rows,
    invalidate_affected_derivations,
)


@pytest.mark.parametrize("command,source", [("literature-search", "europepmc"), ("literature-search", "crossref"), ("dataset-search", "clinicaltrials")])
def test_empty_offline_cache_is_nonzero_with_persisted_failure(tmp_path, monkeypatch, command, source):
    def forbidden(*args, **kwargs):
        pytest.fail("offline search attempted networking")
    monkeypatch.setattr("requests.get", forbidden)
    output = tmp_path / "rows.jsonl"
    status = main([command, "--source", source, "--query", "fixture query", "--limit", "1", "--cache-dir", str(tmp_path / "cache"), "--offline", "--out", str(output)])
    assert status != 0
    receipt = json.loads(output.with_suffix(".jsonl.status.json").read_text())
    assert receipt["status"] == "FAIL" and receipt["source_statuses"]


def test_successful_empty_search_is_distinct_from_failed_retrieval(tmp_path, monkeypatch):
    monkeypatch.setattr("litdatamatcher.http_cache.CachedHttpClient.get_json", lambda *a, **k: {"message": {"items": []}})
    output = tmp_path / "empty.jsonl"
    assert main(["literature-search", "--source", "crossref", "--query", "fixture", "--out", str(output), "--cache-dir", str(tmp_path / "cache")]) == 0
    assert json.loads(output.with_suffix(".jsonl.status.json").read_text())["status"] == "PASS"


def test_one_failed_source_preserves_useful_rows_and_reports_partial(tmp_path, monkeypatch):
    def response(self, url, **kwargs):
        if "crossref" in url:
            return {"message": {"items": [{"DOI": "10.1234/fixture", "title": ["Fixture source record"]}]}}
        raise FileNotFoundError("fixture missing cached source")
    monkeypatch.setattr("litdatamatcher.http_cache.CachedHttpClient.get_json", response)
    output = tmp_path / "partial.jsonl"
    assert main(["literature-search", "--source", "europepmc", "crossref", "--query", "fixture", "--out", str(output), "--cache-dir", str(tmp_path / "cache"), "--offline"]) != 0
    assert json.loads(output.with_suffix(".jsonl.status.json").read_text())["status"] == "PARTIAL"
    assert json.loads(output.read_text())["doi"] == "10.1234/fixture"


@pytest.mark.parametrize("change", ["abstract", "snapshot", "adapter_snapshot", "alternate_abstract"])
def test_changed_literature_content_invalidates_derivations(change):
    raw = {"source_id": "pubmed:1", "abstract": "original", "source_provenance": {"metadata": {"cache_snapshot": {"sha256": "a" * 64}}}, "metadata": {"merged_source_records": [{"source_id": "europepmc:MED:1", "abstract": "original alternate"}]}}
    updated = deepcopy(raw)
    if change == "abstract":
        updated["abstract"] = "changed result"
    elif change == "snapshot":
        updated["source_provenance"]["metadata"]["cache_snapshot"]["sha256"] = "b" * 64
    elif change == "adapter_snapshot":
        for row, value in ((raw, "a"), (updated, "b")):
            row["source_provenance"]["metadata"]["cache_snapshot"] = {"cache_content_sha256": value * 64}
    else:
        updated["metadata"]["merged_source_records"][0]["abstract"] = "alternate correction"
    before = consolidate_literature_rows([raw])[0]
    after = consolidate_literature_rows([updated])[0]
    assert invalidate_affected_derivations(before, after, ["claim-1"])["status"] == "INVALIDATED"


def test_missing_invalidation_keys_cannot_certify_unchanged_derivations():
    assert invalidate_affected_derivations({}, {}, ["claim-1"])["status"] == "INVALIDATED"


def test_retrieval_timestamp_change_alone_is_not_scientific_invalidation():
    raw = {"source_id": "pubmed:1", "abstract": "same", "source_provenance": {"retrieval_time_utc": "2026-09-01T00:00:00Z"}}
    after = deepcopy(raw)
    after["source_provenance"]["retrieval_time_utc"] = "2026-09-13T00:00:00Z"
    assert invalidate_affected_derivations(consolidate_literature_rows([raw])[0], consolidate_literature_rows([after])[0], ["claim-1"])["status"] == "UNCHANGED"


@pytest.mark.parametrize("kind,expected", [("Comment in", "ACTIVE_METADATA_ONLY"), ("Erratum in", "CORRECTED_REQUIRES_VERSION_REVIEW"), ("Retraction in", "RETRACTED"), ("Expression of concern in", "FLAGGED_REQUIRES_SOURCE_REVIEW"), ("unrecognized notice", "FLAGGED_REQUIRES_SOURCE_REVIEW")])
def test_europepmc_notice_type_drives_lifecycle_instead_of_container_name(kind, expected):
    row = {"source_id": "europepmc:MED:1", "version_relationships": {"commentCorrection": [{"type": kind, "id": "2"}]}}
    assert consolidate_literature_rows([row])[0]["metadata"]["literature_integrity"]["lifecycle_status"] == expected


def test_each_literature_record_points_to_its_own_response_page():
    class Pages:
        last_response_metadata = {}
        def get_json(self, url, params):
            second = params["cursorMark"] != "*"
            self.last_response_metadata = {"cache_content_sha256": ("b" if second else "a") * 64, "retrieval_time_utc": "2026-09-13T00:00:00Z"}
            return {"resultList": {"result": [{"id": "2" if second else "1", "source": "MED", "title": "fixture"}]}, **({} if second else {"nextCursorMark": "page2"})}
    rows = EuropePMCLiteratureAdapter(Pages()).search_literature("fixture", 101)
    assert [row["source_provenance"]["metadata"]["cache_snapshot"]["cache_content_sha256"] for row in rows] == ["a" * 64, "b" * 64]
