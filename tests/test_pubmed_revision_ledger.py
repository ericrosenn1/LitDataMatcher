"""Synthetic revision, deletion, full-sequence, and recovery contracts."""

import gzip
import hashlib
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from litdatamatcher.pubmed_revision_ledger import (
    build_revision_ledger,
    build_shard,
    iter_xml_events,
    resolve_pmid_events,
)


def article(pmid, version="1", revised=""):
    attr = f' Version="{version}"' if version is not None else ""
    date = f"<DateRevised>{revised}</DateRevised>" if revised else ""
    return f"<PubmedArticle><MedlineCitation><PMID{attr}>{pmid}</PMID>{date}<Article><ArticleTitle>Test</ArticleTitle></Article></MedlineCitation></PubmedArticle>"


def events(xml):
    return list(iter_xml_events(io.BytesIO(("<PubmedArticleSet>" + xml + "</PubmedArticleSet>").encode()), source_shard="fixture.xml.gz", source="updates", shard_order=1))


def create_shard(root, name="pubmed26n1339.xml.gz", xml=None, pmids=None):
    raw = root / "raw" / name
    parquet = root / "normalized" / name.replace(".xml.gz", ".parquet")
    raw.parent.mkdir(parents=True, exist_ok=True)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    text = xml if xml is not None else article("123", "2") + article("123", "1")
    raw.write_bytes(gzip.compress(("<PubmedArticleSet>" + text + "</PubmedArticleSet>").encode(), mtime=0))
    pmids = pmids if pmids is not None else [event["pmid"] for event in events(text) if event["event_type"] == "upsert"]
    pq.write_table(pa.table({"pmid": pa.array(pmids, type=pa.string())}), parquet)
    return {
        "source": "updates", "source_shard": name,
        "raw_relative_path": raw.relative_to(root).as_posix(),
        "relative_path": parquet.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "acquisition_md5": hashlib.md5(raw.read_bytes()).hexdigest(), "rows": len(pmids),
    }


def test_version_two_then_one_selects_two_and_preserves_order():
    rows = events(article("123", "2") + article("123", "1"))
    result = resolve_pmid_events(rows)
    assert [row["version"] for row in result["history"]] == [2, 1]
    assert result["selected"]["version"] == 2
    assert [row["legacy_row_index"] for row in rows] == [0, 1]


def test_same_version_uses_event_order_not_date_guess():
    rows = events(article("123", "2", "<Year>2026</Year><Month>10</Month><Day>12</Day>") + article("123", "2", "<Year>2026</Year><Month>9</Month><Day>1</Day>"))
    result = resolve_pmid_events(rows)
    assert result["selected"]["event_index"] == 1
    assert result["selected"]["date_revised"] == "2026-09-01"
    assert len(result["history"]) == 2


def test_missing_version_and_partial_date_are_not_invented():
    row = events(article("123", None, "<Year>2026</Year>"))[0]
    assert row["version"] is None
    assert row["version_raw"] is None
    assert row["date_revised"] is None
    assert json.loads(row["date_revised_json"]) == {"Year": "2026"}
    assert resolve_pmid_events([row])["status"] == "UNRESOLVED_VERSION"


def test_delete_events_have_no_legacy_row_and_obey_order():
    rows = events(article("123", "2") + '<DeleteCitation><PMID>123</PMID><PMID>999</PMID></DeleteCitation>' + article("123", "1"))
    assert [row["legacy_row_index"] for row in rows] == [0, None, None, 1]
    result = resolve_pmid_events([row for row in rows if row["pmid"] == "123"])
    assert result["selected"]["version"] == 1
    assert result["active_versions"] == [1]
    assert resolve_pmid_events([row for row in rows[:3] if row["pmid"] == "123"])["status"] == "DELETED"


def test_version_specific_delete_preserves_other_versions():
    rows = events(article("123", "2") + article("123", "1") + '<DeleteCitation><PMID Version="2">123</PMID></DeleteCitation>')
    result = resolve_pmid_events(rows)
    assert result["selected"]["version"] == 1
    assert len(result["history"]) == 3


def test_invalid_delete_version_is_unresolved_not_wide_tombstone():
    rows = events(article("123") + '<DeleteCitation><PMID Version="unknown">123</PMID></DeleteCitation>')
    result = resolve_pmid_events(rows)
    assert result["status"] == "UNRESOLVED_VERSION"
    assert result["active_versions"] == [1]
    assert result["selected"] is None


def test_resolver_rejects_mixed_identity_and_ambiguous_event_positions():
    rows = events(article("123") + article("456"))
    with pytest.raises(ValueError, match="one PMID"):
        resolve_pmid_events(rows)
    with pytest.raises(ValueError, match="duplicate event"):
        resolve_pmid_events([rows[0], rows[0]])


def test_full_sequence_sidecar_resume_and_corruption_repair(tmp_path, monkeypatch):
    corpus, output = tmp_path / "corpus", tmp_path / "ledger"
    shard = create_shard(corpus)
    source_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in corpus.rglob("*") if path.is_file()}
    first = build_shard(corpus, output, shard, 1338)
    assert first["upserts"] == 2 and first["events"] == 2
    assert first["sequence_verified"] is True
    sidecar = output / "partitions/updates/pubmed26n1339/events.parquet"
    rows = pq.read_table(sidecar).to_pylist()
    assert [row["version"] for row in rows] == [2, 1]
    assert resolve_pmid_events(rows)["selected"]["legacy_row_index"] == 0
    import litdatamatcher.pubmed_revision_ledger as module

    original = module.iter_xml_events
    monkeypatch.setattr(module, "iter_xml_events", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must reuse verified shard")))
    assert build_shard(corpus, output, shard, 1338)["reused"] is True
    monkeypatch.setattr(module, "iter_xml_events", original)
    sidecar.write_bytes(b"corrupt derivative")
    repaired = build_shard(corpus, output, shard, 1338)
    assert repaired["reused"] is False
    assert any((output / "quarantine").iterdir())
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == checksum for path, checksum in source_hashes.items())


@pytest.mark.parametrize("pmids", [["123", "456"], ["123"], ["123", "123", "123"]])
def test_mismatch_never_publishes_partial_shard(tmp_path, pmids):
    corpus, output = tmp_path / "corpus", tmp_path / "ledger"
    shard = create_shard(corpus, pmids=pmids)
    with pytest.raises(ValueError, match="mismatch|additional"):
        build_shard(corpus, output, shard, 0)
    assert not (output / "partitions/updates/pubmed26n1339").exists()
    assert any((output / "failed").iterdir())


def test_checksum_and_original_output_protection(tmp_path):
    corpus = tmp_path / "corpus"
    shard = create_shard(corpus)
    shard["acquisition_md5"] = "0" * 32
    with pytest.raises(ValueError, match="MD5"):
        build_shard(corpus, tmp_path / "ledger", shard, 0)
    with pytest.raises(ValueError, match="outside"):
        build_shard(corpus, corpus / "new-outputs", shard, 0)


def test_bounded_campaign_is_partial_and_resume_skips_verified(tmp_path):
    corpus, output = tmp_path / "corpus", tmp_path / "ledger"
    shards = [create_shard(corpus, name) for name in ["pubmed26n1339.xml.gz", "pubmed26n1340.xml.gz"]]
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"status": "FAIL", "shards": shards}))
    checksum = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    partial = build_revision_ledger(corpus, snapshot, output, max_shards=1)
    assert partial["status"] == "PARTIAL" and partial["verified_shards"] == 1
    assert partial["prior_snapshot_status"] == "FAIL"
    full = build_revision_ledger(corpus, snapshot, output)
    assert full["status"] == "PASS" and full["reused_shards"] == 1
    assert full["upserts"] == 4
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == checksum
    assert len(list(output.glob("RUN_RECEIPT_*.json"))) == 2


def test_worker_bounds_are_explicit(tmp_path):
    for workers in [0, 5, True]:
        with pytest.raises(ValueError, match="workers"):
            build_revision_ledger(tmp_path / "corpus", tmp_path / "snapshot", tmp_path / "output", workers=workers)


def test_changed_legacy_input_is_not_silently_accepted_on_resume(tmp_path):
    corpus, output = tmp_path / "corpus", tmp_path / "ledger"
    shard = create_shard(corpus)
    build_shard(corpus, output, shard, 0)
    pq.write_table(pa.table({"pmid": ["999", "999"]}), corpus / shard["relative_path"])
    with pytest.raises(ValueError, match="changed since"):
        build_shard(corpus, output, shard, 0)


def test_interrupted_driver_releases_lease_and_can_resume(tmp_path, monkeypatch):
    import litdatamatcher.pubmed_revision_ledger as module

    corpus, output = tmp_path / "corpus", tmp_path / "ledger"
    shard = create_shard(corpus)
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"status": "FAIL", "shards": [shard]}))
    original = module.build_shard

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(module, "build_shard", interrupted)
    with pytest.raises(KeyboardInterrupt):
        build_revision_ledger(corpus, snapshot, output)
    monkeypatch.setattr(module, "build_shard", original)
    assert build_revision_ledger(corpus, snapshot, output)["status"] == "PASS"


def test_book_citation_uses_direct_book_pmid():
    xml = '<PubmedBookArticle><BookDocument><PMID Version="3">123</PMID><Book><PMID>999</PMID></Book></BookDocument></PubmedBookArticle>'
    rows = events(xml)
    assert len(rows) == 1
    assert rows[0]["pmid"] == "123" and rows[0]["version"] == 3


def test_spawned_process_matches_thread_sidecars_and_resumes(tmp_path):
    corpus = tmp_path / "corpus"
    shards = [create_shard(corpus, name) for name in ["pubmed26n1339.xml.gz", "pubmed26n1340.xml.gz"]]
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"status": "FAIL", "shards": shards}))
    threaded = build_revision_ledger(corpus, snapshot, tmp_path / "thread", executor="thread")
    processed = build_revision_ledger(corpus, snapshot, tmp_path / "process", executor="process")
    assert threaded["status"] == processed["status"] == "PASS"
    assert processed["process_start_method"] == "spawn"
    assert processed["pool_startup_included"] is True
    for thread_shard, process_shard in zip(threaded["shards"], processed["shards"], strict=True):
        assert thread_shard["sidecar_sha256"] == process_shard["sidecar_sha256"]
        assert thread_shard["pmid_sequence_sha256"] == process_shard["pmid_sequence_sha256"]
    resumed = build_revision_ledger(corpus, snapshot, tmp_path / "process", executor="process")
    assert resumed["reused_shards"] == 2


def test_process_worker_failure_is_reported_without_false_success(tmp_path):
    corpus = tmp_path / "corpus"
    shard = create_shard(corpus, pmids=["wrong", "123"])
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"status": "FAIL", "shards": [shard]}))
    report = build_revision_ledger(corpus, snapshot, tmp_path / "ledger", executor="process")
    assert report["status"] == "FAIL"
    assert report["verified_shards"] == 0
    assert report["failures"][0]["error_type"] == "ValueError"


def test_invalid_executor_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="executor"):
        build_revision_ledger(tmp_path / "corpus", tmp_path / "snapshot", tmp_path / "output", executor="unbounded")
