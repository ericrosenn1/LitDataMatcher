"""Exact publication identifiers must not manufacture independent evidence."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from litdatamatcher import adapters
from litdatamatcher.literature_integrity import invalidate_affected_derivations


def record(source, native_id, **identifiers):
    return {"source": source, "source_id": native_id, "title": "Identical title",
            "source_provenance": {"source_type": source, "raw_record_id": native_id,
                                  "retrieval_time_utc": "2026-09-13T00:00:00Z"},
            "metadata": {"source_specific": native_id}, **identifiers}


def search(monkeypatch, rows):
    sources = [SimpleNamespace(name=row["source"], search_literature=lambda *a, row=row, **kw: [row]) for row in rows]
    monkeypatch.setattr(adapters, "build_literature_adapters", lambda *a, **kw: sources)
    return adapters.search_literature_sources("fixture", [], limit=20)


@pytest.mark.parametrize("ids", [
    ({"pmid": "12345"}, {"pmid": " 0012345 "}),
    ({"pmid": 12345}, {"pmid": "12345"}),
    ({"pmcid": "PMC12345"}, {"pmcid": " pmc0012345 "}),
    ({"pmid": "1", "pmcid": "PMC12345"}, {"pmcid": "PMC12345"}),
])
def test_shared_exact_identifier_merges_preserving_source_rows(monkeypatch, ids):
    original = [record("pubmed", "pubmed:12345", **ids[0]), record("europepmc", "europepmc:MED:12345", **ids[1])]
    before = deepcopy(original)
    result = search(monkeypatch, original)
    assert len(result) == 1
    assert original == before
    merged = result[0]
    assert merged["metadata"]["alternate_source_ids"] == [original[1]["source_id"]]
    assert merged["metadata"]["merged_source_records"] == [original[1]]
    integrity = merged["metadata"]["literature_integrity"]
    assert [s["source_type"] for s in integrity["source_snapshots"]] == ["pubmed", "europepmc"]
    assert integrity["evidence_eligibility"] == "INELIGIBLE_REQUIRES_VERSION_REVIEW"


@pytest.mark.parametrize("value", [None, "", " ", True, False, 0, -1, 1.5, "123;456", ["123"], {"id": "123"}, "PMID:123", "１２３", "0"])
def test_invalid_pmids_never_form_a_shared_key(monkeypatch, value):
    assert len(search(monkeypatch, [record("pubmed", "p:a", pmid=value), record("europepmc", "e:b", pmid=value)])) == 2


@pytest.mark.parametrize("value", ["123", "PMC0", "PMC1.2", "PMC123;PMC456", ["PMC123"], True])
def test_invalid_pmcids_remain_separate(monkeypatch, value):
    assert len(search(monkeypatch, [record("pubmed", "p:a", pmcid=value), record("europepmc", "e:b", pmcid=value)])) == 2


def test_missing_identifiers_do_not_merge_on_titles_or_unscoped_ids(monkeypatch):
    assert len(search(monkeypatch, [record("pubmed", ""), record("europepmc", "")])) == 2
    assert len(search(monkeypatch, [record("pubmed", "123"), record("europepmc", "123")])) == 2
    assert len(search(monkeypatch, [record("pubmed", "p:a", pmid="1"), record("europepmc", "e:b", pmid="2")])) == 2


@pytest.mark.parametrize("ids", [
    ({"pmid": "1", "pmcid": "PMC2"}, {"pmid": "1", "pmcid": "PMC3"}),
    ({"pmid": "1", "pmcid": "PMC2"}, {"pmid": "3", "pmcid": "PMC2"}),
])
def test_conflicting_identifiers_remain_separate_and_explicit(monkeypatch, ids):
    rows = search(monkeypatch, [record("pubmed", "p:a", **ids[0]), record("europepmc", "e:b", **ids[1])])
    assert len(rows) == 2
    assert all(row["metadata"]["identifier_conflicts"] for row in rows)


def test_ambiguous_bridge_does_not_choose_one_group(monkeypatch):
    rows = search(monkeypatch, [record("pubmed", "p:a", pmid="1", pmcid="PMC2"),
                               record("europepmc", "e:b", pmid="1", pmcid="PMC3"),
                               record("pubmed", "p:c", pmid="1")])
    assert len(rows) == 3
    assert rows[-1]["metadata"]["identifier_conflicts"]


def test_group_remembers_identifier_declared_only_by_a_duplicate(monkeypatch):
    rows = search(monkeypatch, [record("pubmed", "p:a", pmid="1"),
                               record("europepmc", "e:b", pmid="1", pmcid="PMC2"),
                               record("pubmed", "p:c", pmid="1", pmcid="PMC3")])
    assert len(rows) == 2


def test_doi_precedence_preserved_and_conflicts_disclosed(monkeypatch):
    rows = search(monkeypatch, [record("pubmed", "p:a", doi="https://doi.org/10.1/SAME", pmid="1"),
                               record("europepmc", "e:b", doi="10.1/same", pmid="2")])
    assert len(rows) == 1
    assert rows[0]["metadata"]["identifier_conflicts"]
    assert len(search(monkeypatch, [record("pubmed", "p:a", doi="10.1/a", pmid="1"),
                                   record("europepmc", "e:b", doi="10.1/b", pmid="1")])) == 2


@pytest.mark.parametrize("relation,status", [("is-correction-of", "CORRECTED_REQUIRES_VERSION_REVIEW"),
                                            ("is-retracted-by", "RETRACTED"),
                                            ("is-version-of", "VERSIONED_REQUIRES_VERSION_REVIEW")])
def test_identifier_merge_preserves_alternate_lifecycle_even_with_primary_relations(monkeypatch, relation, status):
    first = record("pubmed", "p:a", pmid="1", version_relationships={"references": [{"id": "other"}]})
    second = record("europepmc", "e:b", pmid="1", version_relationships={relation: [{"id": "notice"}]})
    before = search(monkeypatch, [first])[0]
    merged = search(monkeypatch, [first, second])[0]
    assert merged["metadata"]["literature_integrity"]["lifecycle_status"] == status
    assert invalidate_affected_derivations(before, merged, ["claim:1"])["status"] == "INVALIDATED"


def test_pubmed_pmcid_is_exposed_from_declared_summary_metadata():
    class Client:
        def get_json(self, url, params=None):
            if "esearch" in url: return {"esearchresult": {"idlist": ["123"]}}
            return {"result": {"123": {"title": "Study", "articleids": [{"idtype": "pmc", "value": "PMC456"}]}}}
        def get_text(self, *a, **kw): return "<PubmedArticleSet/>"
    assert adapters.PubMedLiteratureAdapter(Client()).search_literature("fixture")[0]["pmcid"] == "PMC456"


def test_duplicate_without_provenance_does_not_shift_later_source_snapshot(monkeypatch):
    middle = record("europepmc", "e:b", pmid="1")
    middle.pop("source_provenance")
    rows = search(monkeypatch, [record("pubmed", "p:a", pmid="1"), middle, record("crossref", "c:c", pmid="1")])
    snapshots = rows[0]["metadata"]["literature_integrity"]["source_snapshots"]
    assert [(s["source_id"], s["source_type"], s["status"]) for s in snapshots] == [
        ("p:a", "pubmed", "OBSERVED"), ("e:b", "unknown", "UNKNOWN"), ("c:c", "crossref", "OBSERVED")]


@pytest.mark.parametrize("limit", [1, 5])
def test_real_cli_offline_cache_reconciliation(tmp_path, monkeypatch, limit):
    import json
    from litdatamatcher.cli import main
    from litdatamatcher.http_cache import CachedHttpClient
    cache = CachedHttpClient(cache_dir=tmp_path / "cache", offline=True)
    def cached(url, params, payload):
        cache._cache_path(url, params).write_text(json.dumps(payload), encoding="utf-8")
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    cached(base + "esearch.fcgi", {"db": "pubmed", "term": "fixture", "retmode": "json", "retmax": limit},
           {"esearchresult": {"idlist": ["12345"]}})
    cached(base + "esummary.fcgi", {"db": "pubmed", "id": "12345", "retmode": "json"},
           {"result": {"12345": {"title": "Study", "articleids": [{"idtype": "pmc", "value": "PMC456"}]}}})
    cached("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
           {"query": "fixture", "format": "json", "resultType": "core", "pageSize": limit, "cursorMark": "*"},
           {"resultList": {"result": [{"source": "MED", "id": "12345", "title": "Study", "pmcid": "PMC456",
                                      "commentCorrectionList": {"is-retracted-by": [{"id": "notice"}]}}]}})
    monkeypatch.setattr("requests.get", lambda *a, **k: pytest.fail("network attempted"))
    monkeypatch.setattr("socket.create_connection", lambda *a, **k: pytest.fail("network attempted"))
    out = tmp_path / "merged.jsonl"
    args = ["literature-search", "--query", "fixture", "--source", "pubmed", "europepmc", "--offline",
            "--cache-dir", str(tmp_path / "cache"), "--limit", str(limit), "--out", str(out)]
    assert main(args) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["pmcid"] == "PMC456"
    assert rows[0]["metadata"]["literature_integrity"]["lifecycle_status"] == "RETRACTED"
    assert len(rows[0]["metadata"]["literature_integrity"]["source_snapshots"]) == 2
    first = out.read_bytes()
    assert main(args) == 0
    assert out.read_bytes() == first
