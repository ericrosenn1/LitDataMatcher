"""Current-view correctness, interruption recovery, and source-grounded search."""

import hashlib
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from litdatamatcher.corpus_snapshot import PUBMED_FIELDS, file_digest
from litdatamatcher.data_plane import atomic_json
from litdatamatcher.literature_index import LiteratureIndex, materialize_pubmed
from litdatamatcher.pubmed_revision_ledger import LEDGER_VERSION, _schema, resolve_pmid_events


def event(pmid, version=1, kind="upsert", raw=None):
    return {
        "pmid": str(pmid),
        "version": version,
        "event_type": kind,
        "version_raw": str(version) if version is not None else raw,
    }


def fixture(tmp_path, groups):
    corpus, ledger = tmp_path / "corpus", tmp_path / "ledger"
    corpus.mkdir(exist_ok=True)
    ledger.mkdir(exist_ok=True)
    atomic_json(corpus / "CORPUS_ROOT.json", {"store_uuid": "test-corpus"})
    shards, all_events = [], []
    for order, group in enumerate(groups):
        source = "baseline" if order == 0 else "updates"
        name = f"pubmed26n{order + 1:04d}"
        relative = f"normalized/{name}.parquet"
        normal = corpus / relative
        normal.parent.mkdir(exist_ok=True)
        events, records, sequence = [], [], hashlib.sha256()
        for i, value in enumerate(group):
            row = {
                **value,
                "date_revised": None,
                "date_revised_json": None,
                "source": source,
                "source_shard": name + ".xml.gz",
                "shard_order": order,
                "event_index": i,
                "legacy_row_index": None,
            }
            if row["event_type"] == "upsert":
                row["legacy_row_index"] = len(records)
                sequence.update((row["pmid"] + "\n").encode())
                records.append(
                    {
                        **dict.fromkeys(PUBMED_FIELDS, ""),
                        "pmid": row["pmid"],
                        "source_shard": row["source_shard"],
                        "title": f"Cancer study {row['pmid']} version {row['version']}",
                        "abstract": f"Biomarkers were measured. Evidence from shard {order} event {i}.",
                    }
                )
            events.append(row)
        pq.write_table(
            pa.Table.from_pylist(
                records, schema=pa.schema([(f, pa.string()) for f in sorted(PUBMED_FIELDS)])
            ),
            normal,
        )
        part = ledger / "partitions" / source / name
        part.mkdir(parents=True, exist_ok=True)
        sidecar = part / "events.parquet"
        pq.write_table(pa.Table.from_pylist(events, schema=_schema()), sidecar)
        shards.append(
            {
                "source": source,
                "source_shard": name + ".xml.gz",
                "shard_order": order,
                "status": "PASS",
                "sequence_verified": True,
                "events": len(events),
                "upserts": len(records),
                "deletions": len(events) - len(records),
                "legacy_relative_path": relative,
                "sidecar_sha256": file_digest(sidecar),
                "legacy": {"sha256": file_digest(normal)},
                "pmid_sequence_sha256": sequence.hexdigest(),
            }
        )
        all_events.extend(events)
    manifest = ledger / "LEDGER_MANIFEST.json"
    atomic_json(
        manifest,
        {
            "ledger_version": LEDGER_VERSION,
            "status": "PASS",
            "available_shards": len(shards),
            "shards": shards,
        },
    )
    return corpus, manifest, tmp_path / "index.sqlite3", all_events


def test_current_replacement_version_deletion_and_source_passages(tmp_path):
    corpus, manifest, db, _ = fixture(
        tmp_path,
        [
            [event(1, 2), event(1, 1), event(2)],
            [event(1, 2, "delete"), event(2, 1), event(3)],
            [event(2, None, "delete"), event(3, None), event(4)],
            [event(3, None, "delete"), event(3, 1)],
        ],
    )
    first = materialize_pubmed(corpus, manifest, db, max_shards=1, batch_size=1)
    assert first["status"] == "PARTIAL" and first["current_records"] == 2
    with LiteratureIndex(db) as index:
        assert index.get(1)["version"] == 2
        assert index.get(1)["source_provenance"]["event"]["event_index"] == 0
    second = materialize_pubmed(corpus, manifest, db, max_shards=2, batch_size=2)
    assert second["reused_shards"] == 1 and second["unresolved_pmids"] == 1
    with LiteratureIndex(db) as index:
        assert index.get(1)["version"] == 1
        assert index.get(2) is None and index.get(3) is None
        assert {r["pmid"] for r in index.search("cancer")} == {"1", "4"}
        assert index.validate() == {"sqlite": "PASS", "fts_content": "PASS"}
    final = materialize_pubmed(corpus, manifest, db, max_shards=None)
    assert final["status"] == "COMPLETE" and final["current_records"] == 3
    with LiteratureIndex(db, read_only=True) as index:
        assert index.get("PMID:3")["version"] == 1
        record = index.get(3)
        passages = index.passages(3)
        assert len(passages) == 3
        for passage in passages:
            assert record[passage["field"]][passage["start"] : passage["end"]] == passage["text"]
        assert len({p["passage_id"] for p in passages}) == 3
        assert list(index.iter_records(batch_size=1)) == [index.get(pmid) for pmid in (1, 3, 4)]
    replay = materialize_pubmed(corpus, manifest, db)
    assert replay["newly_applied_shards"] == 0 and replay["reused_shards"] == 4


def test_equivalence_to_authoritative_resolver_over_mixed_events(tmp_path):
    rng = random.Random(101)
    groups = []
    for _ in range(8):
        groups.append(
            [
                event(
                    rng.randrange(1, 15),
                    rng.choice([None, 1, 2, 3]),
                    rng.choice(["upsert", "upsert", "delete"]),
                    rng.choice([None, "invalid"]),
                )
                for _ in range(40)
            ]
        )
    corpus, manifest, db, all_events = fixture(tmp_path, groups)
    for order in range(len(groups)):
        materialize_pubmed(corpus, manifest, db, max_shards=1, batch_size=7)
        with LiteratureIndex(db) as index:
            for pmid in map(str, range(1, 15)):
                history = [
                    event
                    for event in all_events
                    if event["pmid"] == pmid and event["shard_order"] <= order
                ]
                expected = resolve_pmid_events(history)["selected"] if history else None
                actual = index.get(pmid)
                assert (actual is None) == (expected is None)
                if expected:
                    assert actual["source_provenance"]["event"] == expected
            index.validate()


def test_failed_shard_rolls_back_search_and_checkpoint(tmp_path, monkeypatch):
    import litdatamatcher.literature_index as module

    corpus, manifest, db, _ = fixture(tmp_path, [[event(1)], [event(1, None, "delete"), event(2)]])
    materialize_pubmed(corpus, manifest, db, max_shards=1)
    original = module._apply_batch

    def interrupted(*args):
        original(*args)
        raise RuntimeError("simulated worker failure")

    monkeypatch.setattr(module, "_apply_batch", interrupted)
    with pytest.raises(RuntimeError, match="worker failure"):
        materialize_pubmed(corpus, manifest, db)
    with LiteratureIndex(db) as index:
        assert index.status()["applied_shards"] == 1
        assert [r["pmid"] for r in index.search("cancer")] == ["1"]
        assert index.get(2) is None
        index.validate()
    monkeypatch.setattr(module, "_apply_batch", original)
    assert materialize_pubmed(corpus, manifest, db)["status"] == "COMPLETE"


def test_append_only_manifest_and_changed_input_detection(tmp_path):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1)]])
    materialize_pubmed(corpus, manifest, db)
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1)], [event(2)]])
    assert materialize_pubmed(corpus, manifest, db)["current_records"] == 2
    with pytest.raises(FileExistsError):
        materialize_pubmed(corpus, manifest, db, resume=False)
    payload = json.loads(manifest.read_text())
    payload["shards"][0]["legacy"]["sha256"] = "0" * 64
    atomic_json(manifest, payload)
    with pytest.raises(ValueError, match="prefix changed"):
        materialize_pubmed(corpus, manifest, db)


def test_checksum_and_mapping_failure_cannot_commit_partial_shard(tmp_path):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1), event(2)]])
    payload = json.loads(manifest.read_text())
    normal = corpus / payload["shards"][0]["legacy_relative_path"]
    records = pq.read_table(normal).to_pylist()
    records[1]["pmid"] = "123"
    pq.write_table(pa.Table.from_pylist(records), normal)
    with pytest.raises(ValueError, match="checksum changed"):
        materialize_pubmed(corpus, manifest, db)
    payload["shards"][0]["legacy"]["sha256"] = file_digest(normal)
    atomic_json(manifest, payload)
    with pytest.raises(ValueError, match="mapping changed"):
        materialize_pubmed(corpus, manifest, db, batch_size=1)
    with LiteratureIndex(db) as index:
        assert index.status()["current_records"] == 0
        assert index.status()["applied_shards"] == 0
        assert index.search("cancer") == []


def test_input_change_on_resume_and_restricted_derivative_path(tmp_path):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1)]])
    for forbidden in (corpus / "derived.sqlite3", manifest.parent / "derived.sqlite3"):
        with pytest.raises(ValueError, match="protected"):
            materialize_pubmed(corpus, manifest, forbidden)
        assert not forbidden.exists()
    materialize_pubmed(corpus, manifest, db)
    payload = json.loads(manifest.read_text())
    normal = corpus / payload["shards"][0]["legacy_relative_path"]
    normal.write_bytes(normal.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="source bytes changed"):
        materialize_pubmed(corpus, manifest, db)


def test_queries_are_literal_bounded_and_missing_ids_are_safe(tmp_path):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1)]])
    materialize_pubmed(corpus, manifest, db)
    with LiteratureIndex(db) as index:
        assert index.search('"cancer" OR NOT (title:*)')[0]["pmid"] == "1"
        assert index.search('( " : ***') == []
        assert index.passages("absent") == []
        assert index.get("1' OR 1=1 --") is None
        for limit in (0, -1, 1001, True):
            with pytest.raises(ValueError):
                index.search("cancer", limit)
        with pytest.raises(ValueError):
            index.search("x" * 4001)


def test_semantic_rerank_preserves_explicit_shortlist_scope(tmp_path, monkeypatch):
    import litdatamatcher.retrieval as retrieval

    corpus, manifest, db, _ = fixture(tmp_path, [[event(1), event(2)]])
    materialize_pubmed(corpus, manifest, db)

    class FakeSemantic:
        receipt = {"model": "test-double"}

        def __init__(self, *_args, **_kwargs):
            pass

        def fit(self, records):
            self.records = records

        def search(self, _query, k):
            return [{"id": row["id"], "score": 0.5} for row in self.records[::-1][:k]]

    monkeypatch.setattr(retrieval, "CachedSemanticIndex", FakeSemantic)
    with LiteratureIndex(db) as index:
        result = index.semantic_search("cancer", Path("unused"), tmp_path / "cache", limit=1)
        assert result[0]["pmid"] == "2"
        assert result[0]["semantic_retrieval_scope"] == "reranked_lexical_shortlist"
        assert result[0]["source_provenance"]["event"]["pmid"] == "2"


def test_snapshot_identity_changes_only_when_committed_prefix_changes(tmp_path):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1)], [event(2)]])
    materialize_pubmed(corpus, manifest, db, max_shards=1)
    with LiteratureIndex(db, read_only=True) as reader:
        first = reader.snapshot_identity()
        assert first["applied_shards"] == 1
        queries = []
        reader.conn.set_trace_callback(queries.append)
        assert reader.status(include_counts=False)["current_records"] == 1
        assert not any(
            "FROM versions" in query or "FROM current_records" in query for query in queries
        )
        materialize_pubmed(corpus, manifest, db, max_shards=1)
        second = reader.snapshot_identity()
        assert second["applied_shards"] == 2
        assert second["snapshot_sha256"] != first["snapshot_sha256"]
        assert reader.status(include_counts=False)["current_records"] == 2
    materialize_pubmed(corpus, manifest, db)
    with LiteratureIndex(db, read_only=True) as reader:
        assert reader.snapshot_identity() == second


def test_fast_status_handles_older_builders_without_stale_counts(tmp_path):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1)]])
    materialize_pubmed(corpus, manifest, db)
    with LiteratureIndex(db) as index:
        index.conn.execute("DELETE FROM metadata WHERE key='current_records_count'")
        index.conn.commit()
        assert index.status(include_counts=False)["current_records"] is None
        assert index.status()["current_records"] == 1


@pytest.mark.parametrize(
    "publication_type, expected",
    [
        ("Retracted Publication", "RETRACTED"),
        ("Published Erratum", "CORRECTED_REQUIRES_VERSION_REVIEW"),
        ("Expression of Concern", "FLAGGED_REQUIRES_SOURCE_REVIEW"),
        ("Journal Article", "ACTIVE_METADATA_ONLY"),
    ],
)
def test_current_records_preserve_explicit_lifecycle_notices(tmp_path, publication_type, expected):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1)]])
    payload = json.loads(manifest.read_text())
    normal = corpus / payload["shards"][0]["legacy_relative_path"]
    records = pq.read_table(normal).to_pylist()
    records[0]["publication_types_json"] = json.dumps([publication_type])
    pq.write_table(pa.Table.from_pylist(records), normal)
    payload["shards"][0]["legacy"]["sha256"] = file_digest(normal)
    atomic_json(manifest, payload)
    materialize_pubmed(corpus, manifest, db)
    with LiteratureIndex(db, read_only=True) as index:
        record = index.get(1)
        assert record["metadata"]["literature_integrity"]["lifecycle_status"] == expected
        assert record["publication_types"] == [publication_type]
        if expected != "ACTIVE_METADATA_ONLY":
            assert (
                record["metadata"]["literature_integrity"]["evidence_eligibility"]
                == "INELIGIBLE_REQUIRES_VERSION_REVIEW"
            )


def test_one_record_update_does_not_scan_the_existing_corpus(tmp_path):
    from litdatamatcher.literature_index import _apply_batch

    corpus, manifest, db, events = fixture(
        tmp_path, [[event(i) for i in range(1, 5001)], [event(9999)]]
    )
    materialize_pubmed(corpus, manifest, db, max_shards=1)
    shard = json.loads(manifest.read_text())["shards"][1]
    record = pq.read_table(corpus / shard["legacy_relative_path"]).to_pylist()[0]
    steps = 0

    def bounded_sql_work():
        nonlocal steps
        steps += 1000
        return int(steps > 30000)

    with LiteratureIndex(db) as index:
        index.conn.set_progress_handler(bounded_sql_work, 1000)
        try:
            with index.conn:
                _apply_batch(index, [(events[-1], record)], shard)
        finally:
            index.conn.set_progress_handler(None, 0)
        assert index.get(9999)["pmid"] == "9999"
        assert steps <= 30000
