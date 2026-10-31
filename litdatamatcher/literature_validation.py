"""Independent bounded validation of known-version current PubMed snapshots.

The audited 2026 ledger has positive versions, including deletions. This oracle
rejects other event semantics explicitly; the materializer supports more cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .controller import integration_lease
from .corpus_snapshot import file_digest
from .data_plane import atomic_json
from .literature_index import LiteratureIndex, _manifest, _paths, _shard_identity

EVENT_DTYPE = np.dtype(
    [("pmid", "<u8"), ("version", "<i4"), ("shard", "<i4"), ("event", "<i4"), ("upsert", "?")]
)
PAIR_DTYPE = np.dtype([("pmid", "<u8"), ("version", "<i4")])
VERSION = "independent-current-ledger-validation-1.0"
_IMPLEMENTATION_SHA256 = file_digest(Path(__file__))


def _sampled_events(corpus, manifest_path, manifest, selected):
    """Fetch complete sampled source events, reading each needed row group once."""
    by_shard = {}
    for value in selected:
        by_shard.setdefault(int(value["shard"]), set()).add(int(value["event"]))
    result = {}
    for order, positions in by_shard.items():
        shard = manifest["shards"][order]
        _, path = _paths(corpus, manifest_path.parent, shard)
        if file_digest(path) != shard["sidecar_sha256"]:
            raise ValueError("Sampled ledger bytes changed")
        parquet = pq.ParquetFile(path)
        offset = 0
        for group in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            needed = [position for position in positions if offset <= position < offset + count]
            if needed:
                table = parquet.read_row_group(group)
                for position in needed:
                    result[(order, position)] = table.slice(position - offset, 1).to_pylist()[0]
                del table
            offset += count
    return result


def _check_source_text_samples(index, corpus, manifest_path, manifest, samples):
    """Check up to 32 evenly spaced samples against hashed normalized source rows."""
    positions = np.linspace(0, len(samples) - 1, min(32, len(samples)), dtype=int)
    by_shard = {}
    for position in positions:
        record = index.get(samples[position]["document_id"])
        event = record["source_provenance"]["event"]
        by_shard.setdefault(event["shard_order"], {})[event["legacy_row_index"]] = record
    checked = []
    for order, wanted in by_shard.items():
        shard = manifest["shards"][order]
        path, _ = _paths(corpus, manifest_path.parent, shard)
        if file_digest(path) != shard["legacy"]["sha256"]:
            raise ValueError("Sampled normalized source checksum changed")
        parquet = pq.ParquetFile(path)
        offset = 0
        for group in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            needed = [position for position in wanted if offset <= position < offset + count]
            if needed:
                table = parquet.read_row_group(
                    group, columns=["pmid", "source_shard", "title", "abstract"]
                )
                for position in needed:
                    native = table.slice(position - offset, 1).to_pylist()[0]
                    record = wanted[position]
                    if (
                        record["pmid"] != native["pmid"]
                        or native["source_shard"] != shard["source_shard"]
                        or any(
                            record[field] != (native[field] or "")
                            for field in ("title", "abstract")
                        )
                        or record["text"] != (native["abstract"] or "")
                    ):
                        raise ValueError("Sampled materialized text differs from normalized source")
                    checked.append(
                        {
                            "document_id": record["document_id"],
                            "source": shard["source"],
                            "source_shard": shard["source_shard"],
                            "legacy_relative_path": shard["legacy_relative_path"],
                            "legacy_row_index": position,
                            "legacy_sha256": shard["legacy"]["sha256"],
                        }
                    )
                del table
            offset += count
    if len(checked) != len(positions):
        raise ValueError("Sampled normalized source row was not found")
    return checked


def _select_current(events):
    """Resolve the final event per version, then choose the highest live version."""
    if not len(events):
        return events.copy()
    ordered = events[
        np.lexsort((events["event"], events["shard"], events["version"], events["pmid"]))
    ]
    last = np.empty(len(ordered), dtype=bool)
    last[-1] = True
    last[:-1] = (ordered["pmid"][:-1] != ordered["pmid"][1:]) | (
        ordered["version"][:-1] != ordered["version"][1:]
    )
    active = ordered[last & ordered["upsert"]]
    del ordered, last
    if not len(active):
        return active
    highest = np.empty(len(active), dtype=bool)
    highest[-1] = True
    highest[:-1] = active["pmid"][:-1] != active["pmid"][1:]
    return active[highest]


def _lexical_order(events):
    """SQLite string order for canonical decimal IDs, without Python strings."""
    if not len(events):
        return events
    largest = int(events["pmid"].max())
    if largest >= 10**18 or int(events["pmid"].min()) < 1:
        raise ValueError("Oracle requires canonical positive PMIDs below 10^18")
    width = len(str(largest))
    powers = np.array([10**i for i in range(width + 1)], dtype=np.uint64)
    digits = np.searchsorted(powers[1:], events["pmid"], side="right") + 1
    padded = events["pmid"] * powers[width - digits]
    return events[np.lexsort((digits, padded))]


def _read_events(corpus, manifest_path, manifest, progress):
    total = sum(shard["events"] for shard in manifest["shards"])
    events = np.empty(total, dtype=EVENT_DTYPE)
    offset = 0
    for shard in manifest["shards"]:
        _, path = _paths(corpus, manifest_path.parent, shard)
        if file_digest(path) != shard["sidecar_sha256"]:
            raise ValueError(f"Ledger sidecar checksum changed: {path}")
        count, upserts = 0, 0
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=65536,
            columns=["pmid", "version", "shard_order", "event_index", "event_type"],
        ):
            cols = {name: batch.column(name) for name in batch.schema.names}
            if any(column.null_count for column in cols.values()):
                raise ValueError("Independent oracle requires known positive versions and IDs")
            numeric = pc.cast(cols["pmid"], pa.uint64(), safe=True)
            if not pc.all(pc.equal(pc.cast(numeric, pa.string()), cols["pmid"])).as_py():
                raise ValueError("PMIDs are not canonical decimal strings")
            kind = cols["event_type"]
            if not pc.all(pc.is_in(kind, value_set=pa.array(["upsert", "delete"]))).as_py():
                raise ValueError("Unknown event type")
            size = batch.num_rows
            values = events[offset : offset + size]
            values["pmid"] = numeric.to_numpy(zero_copy_only=False)
            values["version"] = cols["version"].to_numpy(zero_copy_only=False)
            values["shard"] = cols["shard_order"].to_numpy(zero_copy_only=False)
            values["event"] = cols["event_index"].to_numpy(zero_copy_only=False)
            values["upsert"] = pc.equal(kind, "upsert").to_numpy(zero_copy_only=False)
            if np.any(values["version"] < 1) or np.any(values["pmid"] < 1):
                raise ValueError("Independent oracle requires positive numeric IDs and versions")
            if np.any(values["shard"] != shard["shard_order"]) or not np.array_equal(
                values["event"], np.arange(count, count + size)
            ):
                raise ValueError("Ledger event positions disagree with manifest")
            upserts += int(values["upsert"].sum())
            count += size
            offset += size
        if (
            count != shard["events"]
            or upserts != shard["upserts"]
            or count - upserts != shard["deletions"]
        ):
            raise ValueError("Ledger shard counts disagree with manifest")
        if (shard["shard_order"] + 1) % 100 == 0:
            progress("READING_LEDGER", shards_read=shard["shard_order"] + 1, events_read=offset)
    if offset != total:
        raise ValueError("Ledger event count mismatch")
    return events


def validate_current_snapshot(
    corpus_root,
    ledger_manifest,
    database,
    out,
    *,
    max_events=50_000_000,
    memory_limit_gb=6,
    sample_size=1000,
):
    """Compare every current PMID/version, source samples, SQLite and FTS content."""
    corpus, manifest_path, database, out = [
        Path(p).resolve() for p in (corpus_root, ledger_manifest, database, out)
    ]
    if out.is_relative_to(corpus) or out.is_relative_to(manifest_path.parent):
        raise ValueError("Validation outputs must stay outside protected sources")
    if out.exists():
        raise FileExistsError("Use a fresh validation output directory")
    if type(max_events) is not int or not 1 <= max_events <= 100_000_000:
        raise ValueError("max_events must be in 1..100000000")
    if (
        not 1 <= memory_limit_gb <= 16
        or type(sample_size) is not int
        or not 1 <= sample_size <= 10000
    ):
        raise ValueError("Invalid bounded validation resources")
    manifest = _manifest(manifest_path)
    if manifest["status"] != "PASS":
        raise ValueError("Full validation requires a complete ledger")
    total = sum(shard["events"] for shard in manifest["shards"])
    estimated_bytes = total * 100 + 128 * 1024**2
    if total > max_events or estimated_bytes > memory_limit_gb * 1024**3:
        raise ValueError("Ledger exceeds the explicit validation resource bound")
    if psutil.virtual_memory().available < estimated_bytes * 1.2:
        raise ValueError("Insufficient currently available memory for bounded validation")
    ledger_hash = file_digest(manifest_path)
    started = time.perf_counter()
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    with (
        integration_lease(
            database.parent / (database.name + ".writer"), "pubmed-current-validation"
        ),
        LiteratureIndex(database) as index,
    ):
        snapshot = index.snapshot_identity()
        if snapshot["status"] != "COMPLETE" or snapshot["applied_shards"] != len(
            manifest["shards"]
        ):
            raise ValueError("Full validation requires a complete materialized snapshot")
        checkpoints = [
            tuple(row)
            for row in index.conn.execute(
                "SELECT shard_order,identity FROM shards ORDER BY shard_order"
            )
        ]
        if checkpoints != [
            (shard["shard_order"], _shard_identity(shard)) for shard in manifest["shards"]
        ]:
            raise ValueError("Committed checkpoint identity differs from the ledger manifest")
        marker = json.loads((corpus / "CORPUS_ROOT.json").read_text(encoding="utf-8"))
        if (
            index._metadata("ledger_manifest_sha256") != ledger_hash
            or snapshot["corpus_store_uuid"] != marker["store_uuid"]
        ):
            raise ValueError("Materialized snapshot source identity mismatch")
        out.mkdir(parents=True)

        def progress(stage, **values):
            atomic_json(
                out / "PROGRESS.json",
                {
                    "stage": stage,
                    "elapsed_seconds": time.perf_counter() - started,
                    **values,
                },
            )

        try:
            progress("READING_LEDGER")
            events = _read_events(corpus, manifest_path, manifest, progress)
            progress("PROJECTING_CURRENT", events=total)
            expected = _select_current(events)
            del events
            expected = _lexical_order(expected)
            progress("COMPARING_CURRENT", expected_current_records=len(expected))
            cursor = index.conn.execute(
                "SELECT v.pmid,v.version FROM versions v INDEXED BY version_selection "
                "JOIN current_records c ON c.pmid=v.pmid WHERE c.version_id=v.id ORDER BY v.pmid"
            )
            compared = 0
            actual_digest, expected_digest = hashlib.sha256(), hashlib.sha256()
            while rows := cursor.fetchmany(65536):
                actual = np.fromiter(
                    ((int(row[0]), row[1]) for row in rows),
                    dtype=PAIR_DTYPE,
                    count=len(rows),
                )
                wanted = expected[compared : compared + len(rows)]
                if (
                    len(wanted) != len(rows)
                    or not np.array_equal(actual["pmid"], wanted["pmid"])
                    or not np.array_equal(actual["version"], wanted["version"])
                ):
                    raise ValueError(f"Independent current selection mismatch near row {compared}")
                if any(row[0] != str(int(wanted["pmid"][i])) for i, row in enumerate(rows)):
                    raise ValueError("Current index PMID string identity differs from ledger")
                pairs = np.empty(len(wanted), dtype=PAIR_DTYPE)
                pairs["pmid"], pairs["version"] = wanted["pmid"], wanted["version"]
                actual_digest.update(actual.tobytes())
                expected_digest.update(pairs.tobytes())
                compared += len(rows)
                if compared % (65536 * 16) == 0:
                    progress("COMPARING_CURRENT", compared_records=compared)
            actual_count = index.conn.execute("SELECT count(*) FROM current_records").fetchone()[0]
            if (
                compared != len(expected)
                or actual_count != compared
                or index.status(include_counts=False)["current_records"] != compared
            ):
                raise ValueError("Current projection, table or persisted count mismatch")
            samples = []
            positions = np.linspace(
                0, len(expected) - 1, min(sample_size, len(expected)), dtype=int
            )
            progress("CHECKING_SOURCE_EVENT_SAMPLES", sample_size=len(positions))
            source_events = _sampled_events(corpus, manifest_path, manifest, expected[positions])
            for position in positions:
                value = expected[position]
                record = index.get(str(int(value["pmid"])))
                winning = record["source_provenance"]["event"]
                key = (int(value["shard"]), int(value["event"]))
                if winning != source_events[key]:
                    raise ValueError("Winning source event mismatch")
                shard = manifest["shards"][key[0]]
                required = {
                    "source": shard["source"],
                    "source_shard": shard["source_shard"],
                    "legacy_relative_path": shard["legacy_relative_path"],
                    "legacy_sha256": shard["legacy"]["sha256"],
                    "sidecar_sha256": shard["sidecar_sha256"],
                    "source_locator": f"https://pubmed.ncbi.nlm.nih.gov/{int(value['pmid'])}/",
                }
                if any(
                    record["source_provenance"].get(key) != wanted
                    for key, wanted in required.items()
                ):
                    raise ValueError("Winning source provenance mismatch")
                samples.append(
                    {
                        "document_id": record["document_id"],
                        "version": record["version"],
                        "source_provenance": record["source_provenance"],
                    }
                )
            del expected
            atomic_json(out / "WINNING_EVENT_SAMPLES.json", samples)
            progress("CHECKING_NORMALIZED_TEXT_SAMPLES")
            text_samples = _check_source_text_samples(
                index, corpus, manifest_path, manifest, samples
            )
            atomic_json(out / "NORMALIZED_TEXT_SAMPLES.json", text_samples)
            progress("SQLITE_AND_FTS_INTEGRITY", compared_records=compared)
            integrity = index.validate()
            if file_digest(Path(__file__)) != _IMPLEMENTATION_SHA256:
                raise ValueError("Validator implementation changed during execution")
            if file_digest(manifest_path) != ledger_hash or index.snapshot_identity() != snapshot:
                raise ValueError("Source or snapshot identity changed during validation")
            report = {
                "schema_version": VERSION,
                "status": "PASS",
                "snapshot": snapshot,
                "ledger_manifest_sha256": ledger_hash,
                "events_compared": total,
                "current_records": compared,
                "actual_projection_sha256": actual_digest.hexdigest(),
                "expected_projection_sha256": expected_digest.hexdigest(),
                "source_event_samples": len(samples),
                "normalized_text_samples": len(text_samples),
                "sampling_policy": {
                    "winning_events": "Evenly spaced across current PMID SQLite BINARY order",
                    "normalized_text": "Up to 32 evenly spaced members of winning-event sample; native null text normalized to empty",
                    "winning_event_sources": {
                        source: sum(row["source_provenance"]["source"] == source for row in samples)
                        for source in ("baseline", "updates")
                    },
                    "normalized_text_sources": {
                        source: sum(row["source"] == source for row in text_samples)
                        for source in ("baseline", "updates")
                    },
                },
                "integrity": integrity,
                "estimated_peak_memory_bytes": estimated_bytes,
                "elapsed_seconds": time.perf_counter() - started,
                "implementation_sha256": _IMPLEMENTATION_SHA256,
                "versions": {"numpy": np.__version__, "pyarrow": pa.__version__},
                "scope_note": "Computational current PMID/version and winning-event fidelity, plus SQLite/FTS agreement; not biomedical truth validation.",
            }
            atomic_json(out / "VALIDATION.json", report)
            progress("COMPLETE", current_records=compared)
            return report
        except BaseException as exc:
            progress("FAILED", error=f"{type(exc).__name__}: {exc}")
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus-root", "ledger-manifest", "database", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--max-events", type=int, default=50_000_000)
    parser.add_argument("--memory-limit-gb", type=float, default=6)
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            validate_current_snapshot(
                args.corpus_root,
                args.ledger_manifest,
                args.database,
                args.out,
                max_events=args.max_events,
                memory_limit_gb=args.memory_limit_gb,
                sample_size=args.sample_size,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
