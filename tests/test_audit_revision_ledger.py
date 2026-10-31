"""Independent auditor fixtures; no builder or XML parser is invoked."""

import copy
import hashlib
import json
import os
import subprocess

import pytest

from scripts.campaign_20261031 import audit_revision_ledger as audit

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def schema():
    return pa.schema(
        [
            ("event_type", pa.string()),
            ("pmid", pa.string()),
            ("version", pa.int32()),
            ("version_raw", pa.string()),
            ("date_revised", pa.string()),
            ("date_revised_json", pa.string()),
            ("event_index", pa.int64()),
            ("legacy_row_index", pa.int64()),
            ("source_shard", pa.string()),
            ("source", pa.string()),
            ("shard_order", pa.int32()),
        ],
        metadata={b"ledger_version": b"pubmed-revision-ledger-1.0"},
    )


def fixture(tmp_path, mutate=None, partial=False):
    corpus, ledger = tmp_path / "corpus", tmp_path / "ledger"
    corpus.mkdir()
    ledger.mkdir()
    write(corpus / "CORPUS_ROOT.json", {"store_uuid": "fixture-store"})
    source_manifest = corpus / "acquisition.json"
    write(
        source_manifest,
        {"provenance": "synthetic fixture; opaque raw bytes are not XML validation"},
    )
    first = [
        ("upsert", "100", "2", {"Year": "2026", "Month": "10", "Day": "31"}, "2026-10-31"),
        ("upsert", "100", "1", {"Year": "2026"}, None),
        ("delete", "100", None, None, None),
        ("upsert", "200", None, None, None),
        ("delete", "300", "invalid", None, None),
        ("delete", "400", "3", None, None),
    ]
    specs = [
        first,
        [("upsert", "500", "1", {"Year": "2026", "Month": "2", "Day": "28"}, "2026-02-28")],
    ]
    shards, receipts = [], []
    for order, spec in enumerate(specs):
        source = "baseline" if order == 0 else "updates"
        name = f"pubmed26n{order + 1:04}.xml.gz"
        raw_path, legacy_path = (
            corpus / "raw" / name,
            corpus / "normalized" / name.replace(".xml.gz", ".parquet"),
        )
        raw_path.parent.mkdir(exist_ok=True)
        legacy_path.parent.mkdir(exist_ok=True)
        raw_path.write_bytes(b"opaque-raw-fixture-" + bytes([order]))
        legacy = [pmid for kind, pmid, *_ in spec if kind == "upsert"]
        pq.write_table(pa.table({"pmid": pa.array(legacy, type=pa.string())}), legacy_path)
        rows, index = [], 0
        for event, (kind, pmid, version_raw, revised_raw, revised) in enumerate(spec):
            version = (
                int(version_raw)
                if version_raw
                and version_raw.isascii()
                and version_raw.isdigit()
                and int(version_raw) > 0
                else None
            )
            rows.append(
                {
                    "event_type": kind,
                    "pmid": pmid,
                    "version": version,
                    "version_raw": version_raw,
                    "date_revised": revised,
                    "date_revised_json": json.dumps(revised_raw, sort_keys=True)
                    if revised_raw is not None
                    else None,
                    "event_index": event,
                    "legacy_row_index": index if kind == "upsert" else None,
                    "source_shard": name,
                    "source": source,
                    "shard_order": order,
                }
            )
            index += kind == "upsert"
        original = copy.deepcopy(rows)
        if mutate and order == 0:
            mutate(rows)
        partition = ledger / "partitions" / source / name.removesuffix(".xml.gz")
        partition.mkdir(parents=True)
        sidecar = partition / "events.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=schema()), sidecar)
        raw_md5 = hashlib.md5(raw_path.read_bytes()).hexdigest()
        shard = {
            "source": source,
            "source_shard": name,
            "raw_relative_path": raw_path.relative_to(corpus).as_posix(),
            "relative_path": legacy_path.relative_to(corpus).as_posix(),
            "sha256": sha(legacy_path),
            "raw_md5": raw_md5,
            "acquisition_md5": raw_md5,
            "rows": len(legacy),
            "bytes": legacy_path.stat().st_size,
            "raw_bytes": raw_path.stat().st_size,
            "mtime_ns": legacy_path.stat().st_mtime_ns,
            "raw_mtime_ns": raw_path.stat().st_mtime_ns,
            "status": "PASS",
        }
        receipt = {
            "ledger_version": audit.LEDGER_VERSION,
            "implementation_sha256": "a" * 64,
            "source": source,
            "source_shard": name,
            "shard_order": order,
            "status": "PASS",
            "raw_relative_path": shard["raw_relative_path"],
            "legacy_relative_path": shard["relative_path"],
            "raw": {"sha256": sha(raw_path), "md5": raw_md5, "size_bytes": raw_path.stat().st_size},
            "legacy": {
                "sha256": sha(legacy_path),
                "md5": hashlib.md5(legacy_path.read_bytes()).hexdigest(),
                "size_bytes": legacy_path.stat().st_size,
            },
            "events": len(original),
            "upserts": len(legacy),
            "deletions": len(original) - len(legacy),
            "unknown_version": sum(r["version"] is None for r in original),
            "unresolved_date_revised": sum(
                r["event_type"] == "upsert" and r["date_revised"] is None for r in original
            ),
            "sequence_verified": True,
            "sidecar_sha256": sha(sidecar),
            "pmid_sequence_sha256": hashlib.sha256(
                "".join(x + "\n" for x in legacy).encode()
            ).hexdigest(),
            "reused": False,
        }
        write(partition / "SHARD_RECEIPT.json", receipt)
        shards.append(shard)
        receipts.append(receipt)
    snapshot = {
        "schema_version": "corpus-snapshot-1.0",
        "store_uuid": "fixture-store",
        "status": "FAIL",
        "input_manifests": [{"relative_path": "acquisition.json", "sha256": sha(source_manifest)}],
        "shards": shards,
        "rows": 4,
        "shards_available": 2,
        "shards_audited": 2,
    }
    snapshot["snapshot_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "store_uuid": snapshot["store_uuid"],
                "inputs": snapshot["input_manifests"],
                "shards": shards,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    snapshot_path = tmp_path / "SNAPSHOT.json"
    write(snapshot_path, snapshot)
    selected = receipts[:1] if partial else receipts
    manifest = {
        "ledger_version": audit.LEDGER_VERSION,
        "status": "PARTIAL" if partial else "PASS",
        "scope": "bounded_shard_subset" if partial else "baseline_and_updates",
        "snapshot_sha256": sha(snapshot_path),
        "prior_snapshot_status": "FAIL",
        "prior_snapshot_modified": False,
        "available_shards": 2,
        "selected_shards": len(selected),
        "verified_shards": len(selected),
        "shards": selected,
        "upserts": sum(x["upserts"] for x in selected),
        "deletions": sum(x["deletions"] for x in selected),
        "failures": [],
    }
    write(ledger / "LEDGER_MANIFEST.json", manifest)
    write(ledger / "RUN_RECEIPT_1.json", manifest)
    write(ledger / "owner.json", {"owner": None})
    write(
        ledger / "BUILD_STATE.json",
        {"completed": len(selected), "selected": len(selected), "failed": []},
    )
    return corpus, ledger, snapshot_path


def reseal_manifest(ledger, change):
    value = json.loads((ledger / "LEDGER_MANIFEST.json").read_bytes())
    change(value)
    write(ledger / "LEDGER_MANIFEST.json", value)
    write(ledger / "RUN_RECEIPT_1.json", value)


def run(paths, **kwargs):
    return audit.audit_revision_ledger(*paths, batch_size=2, **kwargs)


def test_complete_vectorized_audit_preserves_unknowns_and_all_history(tmp_path):
    paths = fixture(tmp_path)
    result = run(paths)
    assert result["status"] == "PASS", result["errors"]
    assert result["totals"] == {
        "events": 7,
        "upserts": 4,
        "deletions": 3,
        "unknown_version": 3,
        "unresolved_date_revised": 2,
        "pmid_wide_deletions": 1,
        "version_specific_deletions": 1,
        "unresolved_deletions": 1,
    }
    assert result["prior_snapshot_status"] == "FAIL"
    assert "selected" not in result and "current_view" not in result


def test_partial_is_explicit_and_default_does_not_accept_it(tmp_path):
    paths = fixture(tmp_path, partial=True)
    assert run(paths)["status"] == "FAIL"
    result = run(paths, allow_partial=True)
    assert result["status"] == "PARTIAL"
    assert result["coverage"] == {"available": 2, "selected": 1, "verified": 1}


def test_full_ledger_rejects_resealed_mixed_implementations(tmp_path):
    paths = fixture(tmp_path)
    receipt = paths[1] / "partitions/updates/pubmed26n0002/SHARD_RECEIPT.json"
    value = json.loads(receipt.read_bytes())
    value["implementation_sha256"] = "b" * 64
    write(receipt, value)
    reseal_manifest(paths[1], lambda m: m["shards"][1].update(implementation_sha256="b" * 64))
    result = run(paths)
    assert result["status"] == "FAIL"
    assert "mixed implementation" in result["errors"][-1]["error"]
    assert result["implementation_variants"] == [
        {"sha256": "a" * 64, "partitions": 1},
        {"sha256": "b" * 64, "partitions": 1},
    ]


def test_partial_legacy_implementation_remains_explicitly_unknown(tmp_path):
    paths = fixture(tmp_path, partial=True)
    receipt = paths[1] / "partitions/baseline/pubmed26n0001/SHARD_RECEIPT.json"
    value = json.loads(receipt.read_bytes())
    del value["implementation_sha256"]
    write(receipt, value)
    reseal_manifest(paths[1], lambda m: m["shards"][0].pop("implementation_sha256"))
    result = run(paths, allow_partial=True)
    assert result["status"] == "PARTIAL"
    assert result["implementation_variants"] == [{"sha256": None, "partitions": 1}]


@pytest.mark.parametrize("raw", ["", "0", "-1", "+1", "1.0"])
def test_invalid_version_stays_unknown_and_cannot_become_wide_deletion(tmp_path, raw):
    def mutation(rows):
        rows[3]["version_raw"] = raw
        rows[4]["version_raw"] = raw if raw else "invalid"

    result = run(fixture(tmp_path, mutate=mutation))
    assert result["status"] == "PASS"
    assert result["totals"]["unknown_version"] == 3
    assert result["totals"]["unresolved_deletions"] == 1
    assert result["totals"]["pmid_wide_deletions"] == 1


def test_leading_zero_numeric_version_and_partial_date_are_not_rewritten(tmp_path):
    def mutation(rows):
        rows[0]["version_raw"] = "0002"
        rows[1]["date_revised_json"] = '{"Year":"2026","Month":"13"}'

    result = run(fixture(tmp_path, mutate=mutation))
    assert result["status"] == "PASS"
    assert result["totals"]["unresolved_date_revised"] == 2


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[1].update(event_index=0),
        lambda rows: rows[1].update(legacy_row_index=4),
        lambda rows: rows[2].update(legacy_row_index=2),
        lambda rows: rows[0].update(version=1),
        lambda rows: rows[3].update(version=1),
        lambda rows: rows[4].update(version=1),
        lambda rows: rows[0].update(date_revised="2026-10-30"),
        lambda rows: rows[2].update(date_revised="2026-10-31"),
        lambda rows: rows[1].update(date_revised_json='{"Year":"2026","Year":"2025"}'),
        lambda rows: rows[0].update(source="updates"),
        lambda rows: rows[0].update(source_shard="pubmed26n9999.xml.gz"),
        lambda rows: rows[0].update(shard_order=1),
        lambda rows: rows[0].update(pmid="999"),
        lambda rows: rows[0].update(pmid=""),
        lambda rows: rows[0].update(event_type="replace"),
    ],
)
def test_semantic_corruption_is_detected_even_with_resealed_sidecar_hash(tmp_path, mutation):
    result = run(fixture(tmp_path, mutate=mutation))
    assert result["status"] == "FAIL" and result["errors"]


def test_changed_sidecar_bytes_fail_checksum(tmp_path):
    paths = fixture(tmp_path)
    path = next(paths[1].glob("partitions/*/*/events.parquet"))
    path.write_bytes(b"corrupt")
    result = run(paths)
    assert result["status"] == "FAIL"
    assert "checksum" in result["errors"][0]["error"]


def test_changed_standalone_receipt_cannot_hide_behind_manifest(tmp_path):
    paths = fixture(tmp_path)
    receipt = next(paths[1].glob("partitions/*/*/SHARD_RECEIPT.json"))
    value = json.loads(receipt.read_bytes())
    value["upserts"] += 1
    write(receipt, value)
    assert "receipt mismatch" in run(paths)["errors"][0]["error"]


def test_changed_snapshot_and_store_uuid_are_rejected(tmp_path):
    paths = fixture(tmp_path)
    snapshot = json.loads(paths[2].read_bytes())
    snapshot["rows"] += 1
    write(paths[2], snapshot)
    assert "snapshot hash" in run(paths)["errors"][0]["error"]
    reseal_manifest(paths[1], lambda m: m.update(snapshot_sha256=sha(paths[2])))
    assert "row counts" in run(paths)["errors"][0]["error"]
    write(paths[0] / "CORPUS_ROOT.json", {"store_uuid": "another-store"})
    assert "UUID" in run(paths)["errors"][0]["error"]


def test_active_owner_and_nonfinal_build_state_rejected(tmp_path):
    paths = fixture(tmp_path)
    write(paths[1] / "owner.json", {"owner": "builder"})
    assert "owned" in run(paths)["errors"][0]["error"]
    write(paths[1] / "owner.json", {"owner": None})
    write(paths[1] / "BUILD_STATE.json", {"completed": 1, "selected": 2, "failed": []})
    assert "Build state" in run(paths)["errors"][0]["error"]


def test_missing_immutable_final_receipt_and_wrong_aggregate_are_rejected(tmp_path):
    paths = fixture(tmp_path)
    (paths[1] / "RUN_RECEIPT_1.json").unlink()
    assert "immutable" in run(paths)["errors"][0]["error"]
    reseal_manifest(paths[1], lambda m: m.update(upserts=100))
    assert "Aggregate" in run(paths)["errors"][-1]["error"]


@pytest.mark.parametrize(
    "payload", ["[]", '{"owner":null,"owner":null}', '{"owner":null,"elapsed":1e999}']
)
def test_malformed_control_receipts_produce_failed_report(tmp_path, payload):
    paths = fixture(tmp_path)
    (paths[1] / "owner.json").write_text(payload)
    assert run(paths)["status"] == "FAIL"


def test_unreferenced_active_partition_is_not_silently_ignored(tmp_path):
    paths = fixture(tmp_path)
    extra = paths[1] / "partitions/updates/pubmed26n0999"
    extra.mkdir()
    (extra / "events.parquet").write_bytes(b"unreferenced")
    result = run(paths)
    assert result["status"] == "FAIL" and "Unreferenced" in result["errors"][0]["error"]


def test_laundered_partial_manifest_cannot_claim_full_pass(tmp_path):
    paths = fixture(tmp_path, partial=True)
    reseal_manifest(paths[1], lambda m: m.update(status="PASS", scope="baseline_and_updates"))
    assert run(paths, allow_partial=True)["status"] == "FAIL"


@pytest.mark.parametrize(
    "relative", ["../outside", "/absolute", "C:\\outside", "x/../../outside", "\\\\?\\C:\\outside"]
)
def test_unsafe_receipt_paths_are_rejected(tmp_path, relative):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(audit.AuditError):
        audit._inside(root, relative)


def test_symlink_escape_is_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_text("x")
    try:
        os.symlink(outside, root / "link", target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(root / "link"), str(outside)], capture_output=True
        )
        assert completed.returncode == 0, completed.stderr
    try:
        with pytest.raises(audit.AuditError, match="escapes"):
            audit._inside(root, "link/file")
    finally:
        # Windows directory junctions use rmdir; POSIX symlinks use unlink.
        if os.name == "nt":
            os.rmdir(root / "link")
        else:
            (root / "link").unlink()


def test_report_publication_preserves_existing_and_protected_paths(tmp_path):
    paths = fixture(tmp_path)
    result = run(paths)
    output = tmp_path / "reports" / "audit.json"
    audit._write_new_report(output, result, paths[:2])
    original = output.read_bytes()
    with pytest.raises(audit.AuditError, match="already exists"):
        audit._write_new_report(output, {}, paths[:2])
    assert output.read_bytes() == original
    with pytest.raises(audit.AuditError, match="outside"):
        audit._write_new_report(paths[1] / "extra.json", result, paths[:2])


@pytest.mark.parametrize("kind", ["raw", "legacy", "sidecar", "receipt"])
def test_mutation_of_earlier_partition_is_rejected_at_completion(tmp_path, kind):
    paths = fixture(tmp_path)
    targets = {
        "raw": paths[0] / "raw/pubmed26n0001.xml.gz",
        "legacy": paths[0] / "normalized/pubmed26n0001.parquet",
        "sidecar": paths[1] / "partitions/baseline/pubmed26n0001/events.parquet",
        "receipt": paths[1] / "partitions/baseline/pubmed26n0001/SHARD_RECEIPT.json",
    }

    def mutate(done, total):
        if done == 1:
            target = targets[kind]
            before = target.stat()
            content = target.read_bytes()
            target.write_bytes(bytes([content[0] ^ 1]) + content[1:])
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000000))

    result = audit.audit_revision_ledger(*paths, batch_size=2, progress=mutate)
    assert result["status"] == "FAIL"
    assert "changed" in result["errors"][-1]["error"]


@pytest.mark.parametrize("field", ["mtime_ns", "raw_mtime_ns"])
def test_missing_snapshot_timestamps_cannot_disable_source_verification(tmp_path, field):
    paths = fixture(tmp_path)
    snapshot = json.loads(paths[2].read_bytes())
    del snapshot["shards"][0][field]
    snapshot["snapshot_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "store_uuid": snapshot["store_uuid"],
                "inputs": snapshot["input_manifests"],
                "shards": snapshot["shards"],
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    write(paths[2], snapshot)
    reseal_manifest(paths[1], lambda m: m.update(snapshot_sha256=sha(paths[2])))
    result = run(paths)
    assert result["status"] == "FAIL"
    assert "timestamp binding" in result["errors"][0]["error"]


def test_existing_report_is_rejected_before_any_large_scan(tmp_path, monkeypatch):
    paths = fixture(tmp_path)
    target = tmp_path / "existing.json"
    target.write_text("preserve")
    monkeypatch.setattr(
        audit,
        "audit_revision_ledger",
        lambda *a, **k: pytest.fail("Should not scan before output preflight"),
    )
    with pytest.raises(audit.AuditError, match="already exists"):
        audit.main(
            [
                "--corpus-root",
                str(paths[0]),
                "--ledger",
                str(paths[1]),
                "--snapshot",
                str(paths[2]),
                "--report",
                str(target),
            ]
        )
    assert target.read_text() == "preserve"
