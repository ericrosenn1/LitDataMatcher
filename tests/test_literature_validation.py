"""Independent full current projection, including deletions and wrong winners."""

import json
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_literature_index import event, fixture

from litdatamatcher.corpus_snapshot import file_digest
from litdatamatcher.data_plane import atomic_json
from litdatamatcher.literature_index import LiteratureIndex, materialize_pubmed
from litdatamatcher.literature_validation import (
    EVENT_DTYPE,
    _lexical_order,
    _select_current,
    validate_current_snapshot,
)


def test_independent_full_projection_and_source_events(tmp_path):
    corpus, manifest, db, _ = fixture(
        tmp_path,
        [
            [event(1), event(1, 2), event(10), event(100), event(2), event(3)],
            [event(1, 2, "delete"), event(2), event(3, 1, "delete")],
            [event(2, 3), event(2, 2), event(10, 1, "delete"), event(10)],
        ],
    )
    materialize_pubmed(corpus, manifest, db, max_shards=None)
    out = tmp_path / "validation"
    result = validate_current_snapshot(corpus, manifest, db, out)
    assert result["status"] == "PASS"
    assert result["events_compared"] == 13 and result["current_records"] == 4
    assert result["actual_projection_sha256"] == result["expected_projection_sha256"]
    samples = json.loads((out / "WINNING_EVENT_SAMPLES.json").read_text())
    assert [(r["document_id"], r["version"]) for r in samples] == [
        ("PMID:1", 1),
        ("PMID:10", 1),
        ("PMID:100", 1),
        ("PMID:2", 3),
    ]
    assert result["integrity"] == {"sqlite": "PASS", "fts_content": "PASS"}


def test_wrong_current_version_is_detected_even_when_fts_matches(tmp_path):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1), event(1, 2)]])
    materialize_pubmed(corpus, manifest, db)
    with LiteratureIndex(db) as index:
        with index.conn:
            index.conn.execute("DELETE FROM current_records WHERE pmid='1'")
            index.conn.execute(
                "INSERT INTO current_records SELECT pmid,id FROM versions WHERE version=1"
            )
        assert index.validate()["fts_content"] == "PASS"
    with pytest.raises(ValueError, match="Independent current selection mismatch"):
        validate_current_snapshot(corpus, manifest, db, tmp_path / "validation")


def test_rejects_partial_scope_and_unsupported_oracle_semantics(tmp_path):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1)], [event(2, None)]])
    materialize_pubmed(corpus, manifest, db, max_shards=1)
    with pytest.raises(ValueError, match="complete materialized"):
        validate_current_snapshot(corpus, manifest, db, tmp_path / "partial")
    materialize_pubmed(corpus, manifest, db)
    with pytest.raises(ValueError, match="known positive versions"):
        validate_current_snapshot(corpus, manifest, db, tmp_path / "unknown")
    with pytest.raises(ValueError, match="resource bound"):
        validate_current_snapshot(corpus, manifest, db, tmp_path / "budget", max_events=1)


def test_numeric_order_matches_sqlite_decimal_string_order():
    values = [1, 100, 10, 101, 2, 20, 11, 1000, 1001, 9, 999999999999999999]
    events = np.array([(v, 1, 0, i, True) for i, v in enumerate(values)], dtype=EVENT_DTYPE)
    assert list(_lexical_order(events)["pmid"]) == sorted(values, key=str)


def test_independent_array_projection_matches_simple_dictionary_history():
    rng = random.Random(90210)
    rows, active = [], {}
    for order in range(10000):
        pmid, version, upsert = rng.randrange(1, 101), rng.randrange(1, 6), rng.random() > 0.25
        row = (pmid, version, order // 100, order % 100, upsert)
        rows.append(row)
        if upsert:
            active[(pmid, version)] = row
        else:
            active.pop((pmid, version), None)
    expected = {}
    for (pmid, version), row in active.items():
        if pmid not in expected or version > expected[pmid][1]:
            expected[pmid] = row
    actual = _select_current(np.array(rows, dtype=EVENT_DTYPE))
    assert actual.tolist() == [expected[pmid] for pmid in sorted(expected)]


def test_validator_rejects_checkpoint_not_bound_to_manifest(tmp_path):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1)]])
    materialize_pubmed(corpus, manifest, db)
    with LiteratureIndex(db) as index, index.conn:
        index.conn.execute("UPDATE shards SET identity=?", ("0" * 64,))
    with pytest.raises(ValueError, match="checkpoint identity"):
        validate_current_snapshot(corpus, manifest, db, tmp_path / "validation")


@pytest.mark.parametrize(
    "field", ["pmid", "version", "source_shard", "legacy_row_index", "legacy_sha256"]
)
def test_validator_checks_complete_sampled_event_and_hash_provenance(tmp_path, field):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1)]])
    materialize_pubmed(corpus, manifest, db)
    with LiteratureIndex(db) as index:
        value = json.loads(index.conn.execute("SELECT provenance_json FROM versions").fetchone()[0])
        if field == "legacy_sha256":
            value[field] = "0" * 64
        else:
            value["event"][field] = "wrong"
        with index.conn:
            index.conn.execute("UPDATE versions SET provenance_json=?", (json.dumps(value),))
    with pytest.raises(ValueError, match="Winning source"):
        validate_current_snapshot(corpus, manifest, db, tmp_path / "validation")


def test_validator_detects_text_corruption_with_consistent_fts_and_provenance(tmp_path):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1)]])
    materialize_pubmed(corpus, manifest, db)
    with LiteratureIndex(db) as index:
        with index.conn:
            index.conn.execute("DELETE FROM current_records")
            index.conn.execute("UPDATE versions SET abstract='Fabricated source text'")
            index.conn.execute("INSERT INTO current_records SELECT pmid,id FROM versions")
        assert index.validate()["fts_content"] == "PASS"
    with pytest.raises(ValueError, match="text differs from normalized source"):
        validate_current_snapshot(corpus, manifest, db, tmp_path / "validation")


def test_native_null_abstract_remains_missing_evidence_in_validated_projection(tmp_path):
    corpus, manifest, db, _ = fixture(tmp_path, [[event(1)]])
    payload = json.loads(manifest.read_text())
    path = corpus / payload["shards"][0]["legacy_relative_path"]
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["abstract"] = None
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    payload["shards"][0]["legacy"]["sha256"] = file_digest(path)
    atomic_json(manifest, payload)
    materialize_pubmed(corpus, manifest, db)
    result = validate_current_snapshot(corpus, manifest, db, tmp_path / "validation")
    assert result["status"] == "PASS" and result["normalized_text_samples"] == 1
    with LiteratureIndex(db, read_only=True) as index:
        assert index.get(1)["abstract"] == index.get(1)["text"] == ""
