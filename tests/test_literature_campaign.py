"""Actual native controller chunks, pinned admission and resumable receipts."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_literature_index import event, fixture

from litdatamatcher.controller import Controller
from litdatamatcher.corpus_snapshot import file_digest
from litdatamatcher.data_plane import atomic_json
from litdatamatcher.literature_campaign import continue_materialization
from litdatamatcher.literature_index import LiteratureIndex, main, materialize_pubmed


def run(corpus, manifest, db, root, **extra):
    return continue_materialization(
        corpus,
        manifest,
        db,
        root,
        chunk_seconds=2,
        max_total_seconds=300,
        shards_per_chunk=1,
        **extra,
    )


def test_actual_controller_chunks_resume_without_repeating_completed_work(tmp_path):
    corpus, manifest, db, _ = fixture(
        tmp_path, [[event(1)], [event(2)], [event(1, None, "delete"), event(3)]]
    )
    root = tmp_path / "campaign"
    first = run(corpus, manifest, db, root, max_chunks=1)
    assert first["status"] == "CHUNK_LIMIT"
    first_receipt = root / "chunks/corpus-000001.json"
    original = first_receipt.read_bytes()
    result = run(corpus, manifest, db, root, resume=True)
    assert result["status"] == "COMPLETE"
    assert result["coordinator_active"] is False
    assert len(result["completed_chunks"]) == 3
    assert first_receipt.read_bytes() == original
    assert 0 < result["budget_charged_seconds"] < 300
    with LiteratureIndex(db, read_only=True) as index:
        assert index.get(1) is None
        assert index.status()["current_records"] == 2
    controller = Controller(root / "controller")
    before = [(row["id"], row["attempts"]) for row in controller.jobs()]
    controller.close()
    assert run(corpus, manifest, db, root, resume=True)["status"] == "COMPLETE"
    controller = Controller(root / "controller")
    assert [(row["id"], row["attempts"]) for row in controller.jobs()] == before
    controller.close()


def test_missing_controller_receipt_replays_chunk_without_advancing_index(tmp_path):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1)], [event(2)]])
    root = tmp_path / "campaign"
    run(corpus, manifest, db, root, max_chunks=1)
    controller = Controller(root / "controller")
    controller._receipt("corpus-000001").unlink()
    controller.close()
    result = run(corpus, manifest, db, root, resume=True, max_chunks=1)
    assert result["status"] == "CHUNK_LIMIT"
    with LiteratureIndex(db, read_only=True) as index:
        assert index.status()["applied_shards"] == 1
    controller = Controller(root / "controller")
    assert controller.jobs()[0]["status"] == "SUCCEEDED"
    assert controller.jobs()[0]["attempts"] == 2
    controller.close()


def test_cli_receipt_replay_validates_historical_prefix_after_later_chunks(tmp_path):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1)], [event(2)]])
    receipt = tmp_path / "chunk1.json"
    args = [
        "materialize",
        "--corpus-root",
        str(corpus),
        "--ledger-manifest",
        str(manifest),
        "--database",
        str(db),
        "--max-shards",
        "1",
        "--receipt-out",
        str(receipt),
    ]
    assert main(args) == 0
    original = receipt.read_bytes()
    materialize_pubmed(corpus, manifest, db, max_shards=1)
    assert main(args) == 0
    assert receipt.read_bytes() == original
    with LiteratureIndex(db, read_only=True) as index:
        assert index.status()["applied_shards"] == 2
    payload = json.loads(receipt.read_text())
    payload["current_records"] = 999
    atomic_json(receipt, payload)
    with pytest.raises(ValueError, match="receipt identity changed"):
        main(args)


def test_campaign_identity_and_cumulative_budget_fail_closed(tmp_path):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1)], [event(2)]])
    root = tmp_path / "campaign"
    run(corpus, manifest, db, root, max_chunks=1)
    content = manifest.read_bytes()
    manifest.write_bytes(content + b" ")
    with pytest.raises(ValueError, match="identity changed"):
        run(corpus, manifest, db, root, resume=True)
    manifest.write_bytes(content)
    state_path = root / "CORPUS_CAMPAIGN.json"
    state = json.loads(state_path.read_text())
    state["attempts"]["lost-coordinator-reservation"] = {
        "charged_seconds": 300,
        "status": "RESERVED",
    }
    atomic_json(state_path, state)
    assert run(corpus, manifest, db, root, resume=True)["status"] == "BUDGET_LIMIT"
    with LiteratureIndex(db, read_only=True) as index:
        assert index.status()["applied_shards"] == 1


def test_source_schema_failure_stops_controller_without_retrying_bad_evidence(tmp_path):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1)]])
    payload = json.loads(manifest.read_text())
    normal = corpus / payload["shards"][0]["legacy_relative_path"]
    rows = pq.read_table(normal).to_pylist()
    rows[0]["title"] = 123
    pq.write_table(pa.Table.from_pylist(rows), normal)
    payload["shards"][0]["legacy"]["sha256"] = file_digest(normal)
    atomic_json(manifest, payload)
    root = tmp_path / "campaign"
    result = run(corpus, manifest, db, root)
    assert result["status"] == "FAILED"
    controller = Controller(root / "controller")
    assert controller.jobs()[0]["attempts"] == 1
    controller.close()
    with LiteratureIndex(db, read_only=True) as index:
        assert index.status()["applied_shards"] == 0


def test_changed_implementation_is_rejected_before_database_creation(tmp_path):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1)]])
    with pytest.raises(ValueError, match="implementation changed"):
        materialize_pubmed(corpus, manifest, db, expected_implementation_sha256="0" * 64)
    assert not db.exists()
