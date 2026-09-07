"""Replay and failure injection for bounded contextual reference import."""
import hashlib
import json
import urllib.error

import pytest

from litdatamatcher import external_evidence as ee


def payload():
    return {"primaryAccession":"P01375", "entryType":"UniProtKB reviewed (Swiss-Prot)",
            "entryAudit":{"entryVersion":1,"lastAnnotationUpdateDate":"2026-01-01"},
            "organism":{"taxonId":9606,"scientificName":"Homo sapiens"},
            "genes":[{"geneName":{"value":"TNF"},"synonyms":[{"value":"TNFA"}]}],
            "comments":[{"commentType":"FUNCTION","texts":[{"value":"Constructed reference fixture.",
                "evidences":[{"source":"PubMed","id":"123","evidenceCode":"ECO:0000269"}]}]}]}


def install_fetch(monkeypatch, entry=None):
    def fetch(url):
        row={"content":"https://creativecommons.org/licenses/by/4.0/"} if url==ee.LICENSE_URL else (payload() if entry is None else entry)
        return json.dumps(row).encode(), {"release":"fixture-1","release_date":"2026-01-01"}
    monkeypatch.setattr(ee,"_fetch",fetch)


def test_offline_replay_hash_verified_no_network(tmp_path,monkeypatch):
    install_fetch(monkeypatch)
    first=ee.import_resource(tmp_path,accessions=["P01375"])
    monkeypatch.setattr(ee,"_fetch",lambda *a: pytest.fail("offline network access"))
    assert first==ee.import_resource(tmp_path,True,accessions=["P01375"])


def test_context_only_preserves_primary_citations(tmp_path,monkeypatch):
    install_fetch(monkeypatch)
    records=ee.import_resource(tmp_path,accessions=["P01375"])
    result=ee.query_resource(records,["TNFA"],proposition_id="question-proposition")
    assert len(result)==1
    row=result[0]
    assert row["proposition_id"] is None and row["related_proposition_id"]=="question-proposition"
    assert row["measurement_type"]=="curation" and not row["answers_question"]
    assert row["publication_id"]=="PMID:123"
    assert row["experimental_lineage"]=="UNKNOWN"


def test_query_does_not_match_substrings(tmp_path,monkeypatch):
    install_fetch(monkeypatch)
    records=ee.import_resource(tmp_path,accessions=["P01375"])
    assert ee.query_resource(records,["TN"])==[]
    with pytest.raises(ValueError): ee.query_resource(records,"TNF")


def test_corrupted_snapshot_rejected(tmp_path,monkeypatch):
    install_fetch(monkeypatch)
    records=ee.import_resource(tmp_path,accessions=["P01375"])
    sha=records[0]["source_manifest"]["sha256"]
    (tmp_path/"external_evidence/uniprot/snapshots"/(sha+".json")).write_bytes(b"corrupt")
    with pytest.raises(ee.ResourceUnavailable,match="Corrupt reference cache"):
        ee.import_resource(tmp_path,True,accessions=["P01375"])


def test_missing_offline_cache_not_empty_success(tmp_path):
    with pytest.raises(ee.ResourceUnavailable): ee.import_resource(tmp_path,True)


def test_schema_failure_preserves_valid_pointer(tmp_path,monkeypatch):
    install_fetch(monkeypatch)
    first=ee.import_resource(tmp_path,accessions=["P01375"])
    install_fetch(monkeypatch,{"primaryAccession":"P01375"})
    with pytest.raises(ee.ResourceUnavailable): ee.import_resource(tmp_path,accessions=["P01375"],refresh=True)
    replay=ee.import_resource(tmp_path,True,accessions=["P01375"])
    assert first[0]["source_manifest"]==replay[0]["source_manifest"]
    assert first[0]["annotations"]==replay[0]["annotations"]


def test_http404_not_retried_or_negative_evidence(monkeypatch):
    calls=[]
    def missing(*args,**kwargs):
        calls.append(1)
        raise urllib.error.HTTPError("https://rest.uniprot.org/x",404,"missing",{},None)
    monkeypatch.setattr(ee.urllib.request,"urlopen",missing)
    with pytest.raises(ee.ResourceUnavailable,match="HTTP 404"): ee._fetch("https://rest.uniprot.org/x")
    assert len(calls)==1


def test_http429_long_retry_after_defers_without_bypassing(monkeypatch):
    def limited(*args,**kwargs):
        raise urllib.error.HTTPError("https://rest.uniprot.org/x",429,"limit",{"Retry-After":"60"},None)
    monkeypatch.setattr(ee.urllib.request,"urlopen",limited)
    monkeypatch.setattr(ee.time,"sleep",lambda _: pytest.fail("must defer long retry"))
    with pytest.raises(ee.ResourceUnavailable,match="defer"): ee._fetch("https://rest.uniprot.org/x")


def test_invalid_species_not_normalized_to_human(tmp_path,monkeypatch):
    entry=payload();entry["organism"]["taxonId"]=10090
    install_fetch(monkeypatch,entry)
    with pytest.raises(ee.ResourceUnavailable,match="human entries"):
        ee.import_resource(tmp_path,accessions=["P01375"])


def test_safe_identifiers_and_offline_refresh_guard(tmp_path):
    with pytest.raises(ValueError): ee.import_resource(tmp_path,accessions=["../../arbitrary"])
    with pytest.raises(ValueError): ee.import_resource(tmp_path,True,refresh=True)
