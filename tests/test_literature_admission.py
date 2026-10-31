"""Cheap continuation admission preserves checkpoint and schema invariants."""

import sqlite3

import pytest
from test_literature_index import event, fixture

import litdatamatcher.literature_index as module
from litdatamatcher.literature_index import LiteratureIndex, materialize_pubmed


def test_incremental_admission_defers_full_integrity_scan(tmp_path, monkeypatch):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1)], [event(2)]])
    materialize_pubmed(corpus, manifest, db, max_shards=1)

    class NoFullScan(LiteratureIndex):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.conn.set_authorizer(
                lambda action, name, *_: (
                    sqlite3.SQLITE_DENY
                    if action == sqlite3.SQLITE_PRAGMA
                    and name in {"quick_check", "integrity_check"}
                    else sqlite3.SQLITE_OK
                )
            )

    monkeypatch.setattr(module, "LiteratureIndex", NoFullScan)
    result = materialize_pubmed(corpus, manifest, db, max_shards=1)
    assert result["status"] == "COMPLETE" and result["reused_shards"] == 1
    assert result["full_integrity_validation"] == "REQUIRED_SEPARATELY"
    with LiteratureIndex(db) as index:
        assert index.validate() == {"sqlite": "PASS", "fts_content": "PASS"}


def test_changed_trigger_is_rejected_before_applying_more_shards(tmp_path):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1)], [event(2)]])
    materialize_pubmed(corpus, manifest, db, max_shards=1)
    with LiteratureIndex(db) as index:
        index.conn.executescript(
            "DROP TRIGGER current_insert; CREATE TRIGGER current_insert AFTER INSERT ON current_records BEGIN SELECT 1; END;"
        )
    with pytest.raises(ValueError, match="schema"):
        materialize_pubmed(corpus, manifest, db)
    with LiteratureIndex(db, read_only=True) as index:
        assert index.status()["applied_shards"] == 1


def test_checkpoint_event_counter_must_match_committed_receipts(tmp_path):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1)], [event(2)]])
    materialize_pubmed(corpus, manifest, db, max_shards=1)
    with LiteratureIndex(db) as index, index.conn:
        index._set_metadata("events_applied", 999)
    with pytest.raises(ValueError, match="event counter"):
        materialize_pubmed(corpus, manifest, db)


def test_version_deletion_does_not_scan_all_current_foreign_key_references(tmp_path):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(i) for i in range(1, 20001)]])
    materialize_pubmed(corpus, manifest, db)
    with LiteratureIndex(db) as index:
        callbacks = 0

        def bounded_work():
            nonlocal callbacks
            callbacks += 1
            return callbacks > 100

        index.conn.set_progress_handler(bounded_work, 100)
        try:
            module._apply_batch(index, [(event(1, 1, "delete"), None)], {})
        finally:
            index.conn.set_progress_handler(None, 0)
        index.conn.commit()
        assert index.get(1) is None
        assert index.status()["current_records"] == 19999
        assert index.validate()["fts_content"] == "PASS"


def test_reference_index_migration_preserves_current_projection_and_source_snapshot(tmp_path):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1)], [event(1, 2), event(2)]])
    materialize_pubmed(corpus, manifest, db, max_shards=None)
    with LiteratureIndex(db) as index:
        records = list(index.iter_records())
        snapshot = index.snapshot_identity()
        index.conn.execute("DROP INDEX current_version_reference")
        index.conn.commit()
        old_plan = [
            tuple(row)
            for row in index.conn.execute(
                "EXPLAIN QUERY PLAN DELETE FROM versions WHERE pmid=? AND version=?", ("1", 1)
            )
        ]
        assert any("SCAN current_records" in row[-1] for row in old_plan)
    with LiteratureIndex(db) as index:
        assert list(index.iter_records()) == records
        assert index.snapshot_identity() == snapshot
        new_plan = [
            tuple(row)
            for row in index.conn.execute(
                "EXPLAIN QUERY PLAN DELETE FROM versions WHERE pmid=? AND version=?", ("1", 1)
            )
        ]
        assert not any("SCAN current_records" in row[-1] for row in new_plan)
        assert any("current_version_reference" in row[-1] for row in new_plan)
        assert index.validate_admission()["schema"] == "PASS"
