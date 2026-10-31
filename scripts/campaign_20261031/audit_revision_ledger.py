"""Independently audit additive PubMed sidecars; never construct a current view."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from functools import lru_cache
from pathlib import Path

LEDGER_VERSION = "pubmed-revision-ledger-1.0"
AUDIT_VERSION = "independent-revision-ledger-audit-1.0"


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path):
    require(path.stat().st_size <= 128 * 1024 * 1024, "JSON receipt exceeds 128 MiB")

    def finite(value):
        parsed = float(value)
        require(math.isfinite(parsed), "Nonfinite JSON number")
        return parsed

    value = json.loads(
        path.read_bytes(),
        object_pairs_hook=_pairs,
        parse_float=finite,
        parse_constant=lambda value: (_ for _ in ()).throw(AuditError(f"Nonfinite JSON: {value}")),
    )
    require(isinstance(value, dict), "JSON receipt must be an object")
    return value


def _sha(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    require(
        (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        f"File changed during hashing: {path}",
    )
    return digest.hexdigest()


def _stat_identity(path):
    value = path.stat()
    return (value.st_size, value.st_mtime_ns, value.st_dev, value.st_ino)


def _key(path):
    value = str(path)
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\") and re.match(r"^[A-Za-z]:\\", value[4:]):
        value = value[4:]
    return os.path.normcase(os.path.normpath(value))


def _contains(root, path):
    try:
        return os.path.commonpath([_key(root), _key(path)]) == _key(root)
    except ValueError:
        return False


def _inside(root, relative):
    require(
        isinstance(relative, str) and relative and "\x00" not in relative,
        "Missing or malformed relative path",
    )
    parts = relative.replace("\\", "/").split("/")
    require(
        not any(p in {"", ".", ".."} or ":" in p for p in parts),
        f"Unsafe relative path: {relative}",
    )
    path = (root / Path(*parts)).resolve(strict=True)
    require(_contains(root, path), f"Path escapes declared root: {relative}")
    return path


def _integer(value, name):
    require(type(value) is int and value >= 0, f"Invalid nonnegative count: {name}")
    return value


def _hex(value, length, name):
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value),
        f"Invalid {name} digest",
    )


def _schema():
    import pyarrow as pa

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
        metadata={b"ledger_version": LEDGER_VERSION.encode()},
    )


def _all(mask, message):
    import pyarrow.compute as pc

    if len(mask) == 0:
        return
    require(bool(pc.all(pc.fill_null(mask, False)).as_py()), message)


def _same(left, right, message):
    import pyarrow.compute as pc

    _all(
        pc.or_(
            pc.and_(pc.is_null(left), pc.is_null(right)), pc.fill_null(pc.equal(left, right), False)
        ),
        message,
    )


@lru_cache(maxsize=8192)
def _revised_date(raw):
    if raw is None:
        return None
    value = json.loads(raw, object_pairs_hook=_pairs)
    require(
        isinstance(value, dict)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()),
        "Malformed DateRevised source fields",
    )
    try:
        return dt.date(int(value["Year"]), int(value["Month"]), int(value["Day"])).isoformat()
    except (KeyError, TypeError, ValueError):
        return None


class _LegacyCursor:
    def __init__(self, path, rows, batch_size):
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.parquet = pq.ParquetFile(path)
        require(
            self.parquet.metadata.num_rows == rows, "Legacy Parquet row count differs from snapshot"
        )
        require("pmid" in self.parquet.schema_arrow.names, "Legacy Parquet lacks PMID column")
        field = self.parquet.schema_arrow.field("pmid")
        require(
            pa.types.is_string(field.type) or pa.types.is_large_string(field.type),
            "Legacy PMID column is not text",
        )
        self.iterator = iter(
            self.parquet.iter_batches(batch_size=batch_size, columns=["pmid"], use_threads=False)
        )
        self.current, self.offset, self.consumed = None, 0, 0

    def compare(self, values):
        import pyarrow as pa
        import pyarrow.compute as pc

        start = 0
        while start < len(values):
            if self.current is None or self.offset == len(self.current):
                batch = next(self.iterator, None)
                require(batch is not None, "Sidecar has more upserts than legacy rows")
                self.current = pc.cast(batch.column(0), pa.string())
                self.offset = 0
                if not len(self.current):
                    continue
            size = min(len(values) - start, len(self.current) - self.offset)
            _same(
                values.slice(start, size),
                self.current.slice(self.offset, size),
                f"Legacy PMID sequence differs near row {self.consumed}",
            )
            self.offset += size
            self.consumed += size
            start += size

    def finish(self, count):
        require(self.consumed == count, "Incomplete legacy PMID comparison")
        require(self.current is None or self.offset == len(self.current), "Unconsumed legacy rows")
        require(all(batch.num_rows == 0 for batch in self.iterator), "Extra legacy PMID rows")


def _audit_partition(corpus, ledger, shard, record, order, batch_size):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    name, source = shard["source_shard"], shard["source"]
    require(source in {"baseline", "updates"}, "Unknown source partition")
    require(re.fullmatch(r"pubmed[0-9]{2}n[0-9]+\.xml\.gz", name), "Invalid source shard name")
    directory = f"partitions/{source}/{name.removesuffix('.xml.gz')}"
    receipt_path = _inside(ledger, directory + "/SHARD_RECEIPT.json")
    sidecar = _inside(ledger, directory + "/events.parquet")
    checked_files = {
        str(receipt_path): _stat_identity(receipt_path),
        str(sidecar): _stat_identity(sidecar),
    }
    receipt_sha = _sha(receipt_path)
    receipt = _load(receipt_path)
    require(
        {k: v for k, v in receipt.items() if k != "reused"}
        == {k: v for k, v in record.items() if k != "reused"},
        "Manifest/shard receipt mismatch",
    )
    require(
        receipt.get("ledger_version") == LEDGER_VERSION and receipt.get("status") == "PASS",
        "Partition is not a successful supported ledger",
    )
    require(
        receipt.get("source") == source
        and receipt.get("source_shard") == name
        and receipt.get("shard_order") == order,
        "Partition source/order binding changed",
    )
    _integer(receipt["shard_order"], "shard_order")
    if receipt.get("implementation_sha256") is not None:
        _hex(receipt["implementation_sha256"], 64, "implementation")
    require(receipt.get("sequence_verified") is True, "Missing builder sequence verification")
    for field in ("events", "upserts", "deletions", "unknown_version", "unresolved_date_revised"):
        _integer(receipt[field], field)
    require(receipt["upserts"] == shard["rows"], "Snapshot/upsert count differs")
    _integer(shard["rows"], "snapshot shard rows")
    require(
        receipt["events"] == receipt["upserts"] + receipt["deletions"],
        "Receipt event counts disagree",
    )
    for field, expected, snapshot_size in (
        ("raw", shard["raw_relative_path"], "raw_bytes"),
        ("legacy", shard["relative_path"], "bytes"),
    ):
        require(receipt[field + "_relative_path"] == expected, "Input path binding changed")
        identity = receipt[field]
        _hex(identity["sha256"], 64, field)
        _hex(identity["md5"], 32, field)
        require(identity["size_bytes"] == shard[snapshot_size], "Input size identity differs")
        path = _inside(corpus, expected)
        checked_files[str(path)] = _stat_identity(path)
        require(
            checked_files[str(path)][0] == identity["size_bytes"], "Current source size differs"
        )
        time_field = "raw_mtime_ns" if field == "raw" else "mtime_ns"
        require(time_field in shard, "Snapshot lacks required source timestamp binding")
        _integer(shard[time_field], time_field)
        require(
            checked_files[str(path)][1] == shard[time_field],
            "Current source timestamp differs; independent source hashing is required",
        )
    require(
        receipt["raw"]["md5"] == shard["acquisition_md5"] == shard["raw_md5"],
        "Raw acquisition checksum binding changed",
    )
    require(
        receipt["legacy"]["sha256"] == shard["sha256"], "Legacy snapshot checksum binding changed"
    )
    _hex(receipt["sidecar_sha256"], 64, "sidecar")
    _hex(receipt["pmid_sequence_sha256"], 64, "PMID sequence")
    require(_sha(sidecar) == receipt["sidecar_sha256"], "Sidecar checksum mismatch")
    source_path = _inside(corpus, shard["relative_path"])
    source_stat, sidecar_stat = source_path.stat(), sidecar.stat()
    cursor = _LegacyCursor(source_path, shard["rows"], batch_size)
    parquet = pq.ParquetFile(sidecar)
    require(
        parquet.schema_arrow.equals(_schema(), check_metadata=True), "Sidecar Arrow schema mismatch"
    )
    require(parquet.metadata.num_rows == receipt["events"], "Sidecar footer count differs")
    counts = collections.Counter()
    sequence = hashlib.sha256()
    for batch in parquet.iter_batches(batch_size=batch_size, use_threads=False):
        col = {name: batch.column(i) for i, name in enumerate(batch.schema.names)}
        for name in ("event_type", "pmid", "event_index", "source_shard", "source", "shard_order"):
            require(col[name].null_count == 0, f"Null required field: {name}")
        _all(
            pc.equal(
                col["event_index"],
                pa.array(
                    range(counts["events"], counts["events"] + batch.num_rows), type=pa.int64()
                ),
            ),
            "Event index is not contiguous",
        )
        _all(pc.equal(col["source_shard"], shard["source_shard"]), "Row shard identity differs")
        _all(pc.equal(col["source"], source), "Row source identity differs")
        _all(pc.equal(col["shard_order"], order), "Row shard order differs")
        _all(pc.match_substring_regex(col["pmid"], r"^[0-9]+$"), "Malformed PMID")
        upsert, delete = (
            pc.equal(col["event_type"], "upsert"),
            pc.equal(col["event_type"], "delete"),
        )
        _all(pc.or_(upsert, delete), "Unknown event type")
        up_indices = pc.filter(col["legacy_row_index"], upsert)
        _same(
            up_indices,
            pa.array(
                range(counts["upserts"], counts["upserts"] + len(up_indices)), type=pa.int64()
            ),
            "Legacy row index is not contiguous",
        )
        require(
            pc.filter(col["legacy_row_index"], delete).null_count
            == pc.sum(pc.cast(delete, pa.int64())).as_py(),
            "Deletion references a legacy row",
        )
        raw = col["version_raw"]
        numeric = pc.fill_null(pc.match_substring_regex(raw, r"^[0-9]+$"), False)
        stripped = pc.replace_substring_regex(raw, r"^0+", "")
        valid = pc.and_(numeric, pc.fill_null(pc.not_equal(stripped, ""), False))
        expected = pc.cast(pc.if_else(valid, stripped, pa.scalar(None, pa.string())), pa.int64())
        _all(
            pc.or_(pc.is_null(expected), pc.fill_null(pc.less_equal(expected, 2147483647), False)),
            "Version outside int32 range",
        )
        _same(
            pc.cast(col["version"], pa.int64()),
            expected,
            "Parsed version differs from raw version; no inference allowed",
        )
        for name in ("date_revised", "date_revised_json"):
            values = pc.filter(col[name], delete)
            require(values.null_count == len(values), "Deletion carries a revision date")
        encoded = pc.dictionary_encode(pc.filter(col["date_revised_json"], upsert))
        expected_dates = pa.DictionaryArray.from_arrays(
            encoded.indices,
            pa.array([_revised_date(v) for v in encoded.dictionary.to_pylist()], type=pa.string()),
        ).dictionary_decode()
        _same(
            pc.filter(col["date_revised"], upsert),
            expected_dates,
            "DateRevised normalization differs from retained source fields",
        )
        pmids = pc.filter(col["pmid"], upsert)
        cursor.compare(pmids)
        sequence.update(("\n".join(pmids.to_pylist()) + ("\n" if len(pmids) else "")).encode())
        counts["events"] += batch.num_rows
        counts["upserts"] += len(pmids)
        counts["deletions"] += batch.num_rows - len(pmids)
        counts["unknown_version"] += col["version"].null_count
        counts["unresolved_date_revised"] += pc.filter(col["date_revised"], upsert).null_count
        unversioned = pc.and_(delete, pc.is_null(col["version"]))
        wide = pc.and_(unversioned, pc.or_(pc.is_null(raw), pc.fill_null(pc.equal(raw, ""), False)))
        counts["pmid_wide_deletions"] += pc.sum(pc.cast(wide, pa.int64())).as_py()
        counts["unresolved_deletions"] += (
            pc.sum(pc.cast(unversioned, pa.int64())).as_py()
            - pc.sum(pc.cast(wide, pa.int64())).as_py()
        )
        counts["version_specific_deletions"] += pc.sum(
            pc.cast(pc.and_(delete, pc.invert(pc.is_null(col["version"]))), pa.int64())
        ).as_py()
    cursor.finish(receipt["upserts"])
    for key in ("events", "upserts", "deletions", "unknown_version", "unresolved_date_revised"):
        require(counts[key] == receipt[key], f"Scanned {key} count differs from receipt")
    require(
        sequence.hexdigest() == receipt["pmid_sequence_sha256"], "PMID sequence digest mismatch"
    )
    require(_sha(receipt_path) == receipt_sha, "Shard receipt changed during audit")
    for path, before in ((source_path, source_stat), (sidecar, sidecar_stat)):
        require(
            (path.stat().st_size, path.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns),
            "Input changed during scanning",
        )
    return {
        "source": source,
        "source_shard": shard["source_shard"],
        "shard_order": order,
        "receipt_sha256": receipt_sha,
        "sidecar_sha256": receipt["sidecar_sha256"],
        "implementation_sha256": receipt.get("implementation_sha256"),
        "_checked_files": checked_files,
        "_receipt_path": str(receipt_path),
        **dict(counts),
    }


def audit_revision_ledger(
    corpus_root, ledger_root, snapshot_path, *, allow_partial=False, batch_size=65536, progress=None
):
    started = time.perf_counter()
    report = {
        "audit_version": AUDIT_VERSION,
        "status": "FAIL",
        "partitions": [],
        "errors": [],
        "batch_size": batch_size,
        "python_version": sys.version.split()[0],
        "method": "standalone_vectorized_parquet_no_builder_validator",
        "limitations": [
            "Raw XML is neither reparsed nor rehashed; recorded acquisition digests and current size/timestamp are checked.",
            "Legacy Parquet PMID columns are compared in full; non-PMID values are not independently rehashed.",
            "This validates event-history integrity, not current-view resolution or biomedical validity.",
        ],
    }
    try:
        require(type(batch_size) is int and 1 <= batch_size <= 1048576, "Invalid batch size")
        corpus, ledger = (
            Path(corpus_root).resolve(strict=True),
            Path(ledger_root).resolve(strict=True),
        )
        require(
            not _contains(corpus, ledger) and not _contains(ledger, corpus),
            "Corpus and ledger roots must be separate",
        )
        controls = [
            _inside(ledger, "LEDGER_MANIFEST.json"),
            Path(snapshot_path).resolve(strict=True),
        ]
        identities = {str(path): _sha(path) for path in controls}
        manifest, snapshot = (_load(path) for path in controls)
        require(manifest.get("ledger_version") == LEDGER_VERSION, "Unsupported ledger version")
        require(
            manifest.get("status") in {"PASS", "PARTIAL"} and not manifest.get("failures"),
            "Ledger is not a completed successful build",
        )
        for key in (
            "available_shards",
            "selected_shards",
            "verified_shards",
            "upserts",
            "deletions",
        ):
            _integer(manifest[key], key)
        require(
            manifest["snapshot_sha256"] == identities[str(controls[1])],
            "Ledger/snapshot hash binding differs",
        )
        require(
            snapshot.get("schema_version") == "corpus-snapshot-1.0", "Unsupported corpus snapshot"
        )
        marker = _inside(corpus, "CORPUS_ROOT.json")
        require(
            isinstance(snapshot.get("store_uuid"), str) and snapshot["store_uuid"],
            "Snapshot lacks a store UUID",
        )
        require(
            _load(marker).get("store_uuid") == snapshot["store_uuid"], "Corpus store UUID mismatch"
        )
        identities[str(marker)] = _sha(marker)
        stable = [
            {k: v for k, v in row.items() if k != "elapsed_seconds"} for row in snapshot["shards"]
        ]
        digest = hashlib.sha256(
            json.dumps(
                {
                    "store_uuid": snapshot["store_uuid"],
                    "inputs": snapshot["input_manifests"],
                    "shards": stable,
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        require(digest == snapshot["snapshot_sha256"], "Snapshot internal digest mismatch")
        for item in snapshot["input_manifests"]:
            path = _inside(corpus, item["relative_path"])
            require(_sha(path) == item["sha256"], "Snapshot source manifest changed")
            identities[str(path)] = item["sha256"]
        shards = sorted(
            snapshot["shards"], key=lambda row: (row["source"] != "baseline", row["source_shard"])
        )
        mapping = {(row["source"], row["source_shard"]): (i, row) for i, row in enumerate(shards)}
        require(len(mapping) == len(shards) and shards, "Empty or duplicated snapshot shards")
        records = manifest["shards"]
        keys = [(r["source"], r["source_shard"]) for r in records]
        require(
            len(keys) == len(set(keys)) and all(k in mapping for k in keys),
            "Duplicate or unknown ledger shard",
        )
        full = manifest["status"] == "PASS" and manifest.get("scope") == "baseline_and_updates"
        full = (
            full
            and len(records)
            == len(shards)
            == snapshot["shards_available"]
            == snapshot["shards_audited"]
        )
        require(
            manifest["status"] != "PASS" or full,
            "PASS manifest does not cover the complete corpus snapshot",
        )
        require(
            manifest["status"] != "PARTIAL" or manifest.get("scope") == "bounded_shard_subset",
            "Partial manifest scope is inconsistent",
        )
        require(
            full or allow_partial,
            "Full PASS ledger required; use --allow-partial only for an explicit pilot",
        )
        if full:
            actual = {
                p.relative_to(ledger).as_posix()
                for p in ledger.glob("partitions/*/*/events.parquet")
            }
            expected = {
                f"partitions/{r['source']}/{r['source_shard'].removesuffix('.xml.gz')}/events.parquet"
                for r in records
            }
            require(actual == expected, "Unreferenced or missing active sidecar partition")
            receipt_files = {
                p.relative_to(ledger).as_posix()
                for p in ledger.glob("partitions/*/*/SHARD_RECEIPT.json")
            }
            require(
                receipt_files
                == {name.replace("events.parquet", "SHARD_RECEIPT.json") for name in expected},
                "Unreferenced or missing active shard receipt",
            )
        require(
            manifest["available_shards"] == len(shards)
            and manifest["selected_shards"] == manifest["verified_shards"] == len(records),
            "Manifest partition counts disagree",
        )
        require(
            manifest.get("prior_snapshot_modified") is False
            and manifest.get("prior_snapshot_status") == snapshot["status"],
            "Prior snapshot status binding changed",
        )
        require(
            sum(row["rows"] for row in snapshot["shards"]) == snapshot["rows"],
            "Snapshot row counts disagree",
        )
        _integer(snapshot["rows"], "snapshot rows")
        if "reused_shards" in manifest:
            _integer(manifest["reused_shards"], "reused_shards")
            require(
                manifest["reused_shards"] == sum(r.get("reused") is True for r in records),
                "Reused partition count differs",
            )
        for name in ("owner.json", "BUILD_STATE.json"):
            path = ledger / name
            if path.exists():
                value = _load(path)
                if name == "owner.json":
                    require(value.get("owner") is None, "Ledger build is still owned or unclosed")
                else:
                    require(
                        value["completed"] == len(records)
                        and value["selected"] == len(records)
                        and not value["failed"],
                        "Build state is not complete or differs from final manifest",
                    )
                identities[str(path)] = _sha(path)
        run_receipts = sorted(ledger.glob("RUN_RECEIPT_*.json"))
        require(run_receipts, "No immutable final run receipt")
        matching = [path for path in run_receipts if _load(path) == manifest]
        require(matching, "Final manifest lacks matching immutable run receipt")
        for path in run_receipts:
            identities[str(path)] = _sha(path)
        require(
            [r["shard_order"] for r in records] == sorted(r["shard_order"] for r in records),
            "Manifest shard order is not sorted",
        )
        checked_files = {}
        for record in records:
            key = (record["source"], record["source_shard"])
            order, shard = mapping[key]
            try:
                result = _audit_partition(corpus, ledger, shard, record, order, batch_size)
                require(
                    not full or result["implementation_sha256"],
                    "Full ledger partition lacks implementation fingerprint",
                )
                checked_files.update(result.pop("_checked_files"))
                identities[result.pop("_receipt_path")] = result["receipt_sha256"]
                report["partitions"].append(result)
            except (ValueError, TypeError, KeyError, OSError, OverflowError) as error:
                report["errors"].append(
                    {"source": key[0], "source_shard": key[1], "error": str(error)}
                )
            if progress:
                progress(len(report["partitions"]) + len(report["errors"]), len(records))
        totals = collections.Counter()
        for row in report["partitions"]:
            totals.update(
                {
                    k: row.get(k, 0)
                    for k in (
                        "events",
                        "upserts",
                        "deletions",
                        "unknown_version",
                        "unresolved_date_revised",
                        "pmid_wide_deletions",
                        "version_specific_deletions",
                        "unresolved_deletions",
                    )
                }
            )
        if not report["errors"]:
            require(
                totals["upserts"] == manifest["upserts"]
                and totals["deletions"] == manifest["deletions"],
                "Aggregate manifest counts differ",
            )
            require(len(report["partitions"]) == len(records), "Incomplete partition scan")
        require(
            all(_sha(Path(path)) == digest for path, digest in identities.items()),
            "Control inputs changed during audit",
        )
        require(
            all(_stat_identity(Path(path)) == expected for path, expected in checked_files.items()),
            "Audited source, sidecar or receipt changed before audit completion",
        )
        report.update(
            {
                "status": "FAIL" if report["errors"] else "PASS" if full else "PARTIAL",
                "coverage": {
                    "available": len(shards),
                    "selected": len(records),
                    "verified": len(report["partitions"]),
                },
                "totals": dict(totals),
                "snapshot_sha256": manifest["snapshot_sha256"],
                "prior_snapshot_status": snapshot["status"],
                "scope": "full_snapshot" if full else "bounded_manifest_subset",
                "control_sha256": identities,
                "input_files_stat_rechecked": len(checked_files),
                "legacy_unfingerprinted_partitions": sum(
                    not r["implementation_sha256"] for r in report["partitions"]
                ),
            }
        )
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as error:
        report["errors"].append({"stage": "manifest_or_snapshot", "error": str(error)})
    report["elapsed_seconds"] = time.perf_counter() - started
    report["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    report["auditor_sha256"] = _sha(Path(__file__))
    return report


def _write_new_report(path, report, protected):
    target = Path(path).resolve()
    require(
        not any(_contains(root, target) for root in protected),
        "Report must be outside corpus and ledger",
    )
    require(not target.exists(), "Report already exists; choose a new path")
    target.parent.mkdir(parents=True, exist_ok=True)
    target = target.parent.resolve() / target.name
    require(
        not any(_contains(root, target) for root in protected),
        "Report parent resolves into protected inputs",
    )
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.link(temporary, target)  # Atomic and refuses an existing destination.
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--batch-size", type=int, default=65536)
    args = parser.parse_args(argv)
    protected = [args.corpus_root.resolve(), args.ledger.resolve()]
    target = args.report.resolve()
    require(
        not any(_contains(root, target) for root in protected),
        "Report must be outside corpus and ledger",
    )
    require(not target.exists(), "Report already exists; choose a new path")

    def progress(done, total):
        if done % 100 == 0 or done == total:
            print(f"Audited {done}/{total} partitions", flush=True)

    report = audit_revision_ledger(
        args.corpus_root,
        args.ledger,
        args.snapshot,
        allow_partial=args.allow_partial,
        batch_size=args.batch_size,
        progress=progress,
    )
    _write_new_report(target, report, protected)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"partitions", "control_sha256"}}, indent=2
        )
    )
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
