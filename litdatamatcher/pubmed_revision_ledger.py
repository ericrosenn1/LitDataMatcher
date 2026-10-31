"""Additive PubMed revision/deletion sidecars; original corpus inputs are read-only."""

from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import datetime as dt
import gzip
import hashlib
import json
import multiprocessing
import os
import re
import time
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path, PureWindowsPath

from .controller import integration_lease
from .data_plane import atomic_json

LEDGER_VERSION = "pubmed-revision-ledger-1.0"


def _hashes(path: Path) -> dict:
    before = path.stat()
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
            md5.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Input changed while hashing: {path}")
    return {"sha256": sha.hexdigest(), "md5": md5.hexdigest(), "size_bytes": after.st_size}


def _comparison_path(path: Path):
    """Compare equivalent Windows DOS/UNC anchors without changing the I/O path.

    Windows realpath can retain an extended prefix while a missing ancestor is
    concurrently created. Device and volume namespaces are not DOS/UNC aliases.
    Callers still resolve symlinks before testing containment.
    """
    if os.name != "nt":
        return path
    value = str(path)
    if value[:8].casefold() == "\\\\?\\unc\\":
        value = "\\\\" + value[8:]
    elif re.match(r"^\\\\\?\\[A-Za-z]:\\", value):
        value = value[4:]
    elif value.startswith(("\\\\?\\", "\\\\.\\")):
        raise ValueError("Unsupported Windows device namespace")
    return PureWindowsPath(value)


def _resolved_path(path: Path) -> Path:
    _comparison_path(path)  # Reject device input before realpath can erase its prefix.
    return path.resolve()


def _is_inside(path: Path, root: Path) -> bool:
    return _comparison_path(path).is_relative_to(_comparison_path(root))


def _inside(root: Path, relative: str) -> Path:
    path = _resolved_path(root / relative)
    if not _is_inside(path, root):
        raise ValueError("Input or derivative path escapes its declared root")
    return path


def _version(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value) if re.fullmatch(r"[0-9]+", value) and int(value) > 0 else None


def _date_revised(element) -> tuple[str | None, str | None]:
    if element is None:
        return None, None
    parts = {child.tag: (child.text or "").strip() for child in element}
    normalized = None
    with contextlib.suppress(KeyError, TypeError, ValueError):
        normalized = dt.date(int(parts["Year"]), int(parts["Month"]), int(parts["Day"])).isoformat()
    return normalized, json.dumps(parts, sort_keys=True)


class _HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.sha = hashlib.sha256()
        self.md5 = hashlib.md5()

    def read(self, size=-1):
        block = self.stream.read(size)
        self.sha.update(block)
        self.md5.update(block)
        return block


def iter_xml_events(stream, *, source_shard: str, source: str, shard_order: int):
    """Stream complete records and clear root children to bound XML-tree memory."""
    context = ET.iterparse(stream, events=("start", "end"))
    _, root = next(context)
    row_index = event_index = 0
    for event, element in context:
        if event != "end":
            continue
        tag = element.tag.rsplit("}", 1)[-1]
        if tag in {"PubmedArticle", "PubmedBookArticle"}:
            container = element.find("MedlineCitation" if tag == "PubmedArticle" else "BookDocument")
            pmid_element = container.find("PMID") if container is not None else None
            if pmid_element is None or not (pmid_element.text or "").strip().isdigit():
                raise ValueError(f"Missing/malformed citation PMID at legacy row {row_index}")
            version_raw = pmid_element.get("Version")
            revised, revised_raw = _date_revised(container.find("DateRevised"))
            yield {
                "event_type": "upsert", "pmid": pmid_element.text.strip(),
                "version": _version(version_raw), "version_raw": version_raw,
                "date_revised": revised, "date_revised_json": revised_raw,
                "event_index": event_index, "legacy_row_index": row_index,
                "source_shard": source_shard, "source": source, "shard_order": shard_order,
            }
            row_index += 1
            event_index += 1
            root.clear()
        elif tag == "DeleteCitation":
            for pmid_element in element.findall("PMID"):
                if not (pmid_element.text or "").strip().isdigit():
                    raise ValueError("Missing/malformed deletion PMID")
                version_raw = pmid_element.get("Version")
                yield {
                    "event_type": "delete", "pmid": pmid_element.text.strip(),
                    "version": _version(version_raw), "version_raw": version_raw,
                    "date_revised": None, "date_revised_json": None,
                    "event_index": event_index, "legacy_row_index": None,
                    "source_shard": source_shard, "source": source, "shard_order": shard_order,
                }
                event_index += 1
            root.clear()


def _schema():
    import pyarrow as pa

    return pa.schema([
        ("event_type", pa.string()), ("pmid", pa.string()), ("version", pa.int32()),
        ("version_raw", pa.string()), ("date_revised", pa.string()),
        ("date_revised_json", pa.string()), ("event_index", pa.int64()),
        ("legacy_row_index", pa.int64()), ("source_shard", pa.string()),
        ("source", pa.string()), ("shard_order", pa.int32()),
    ], metadata={b"ledger_version": LEDGER_VERSION.encode()})


def _legacy_pmids(path: Path):
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=["pmid"]):
        yield from batch.column(0).to_pylist()


def _validate_sidecar(path: Path, expected: dict) -> None:
    import pyarrow.parquet as pq

    counts = collections.Counter()
    sequence = hashlib.sha256()
    event_index = legacy_index = 0
    parquet = pq.ParquetFile(path)
    if parquet.schema_arrow != _schema():
        raise ValueError("Revision sidecar schema changed")
    for batch in parquet.iter_batches(batch_size=4096):
        for row in batch.to_pylist():
            if row["event_index"] != event_index:
                raise ValueError("Revision sidecar event order corrupted")
            event_index += 1
            kind = row["event_type"]
            if kind not in {"upsert", "delete"}:
                raise ValueError("Unknown revision event")
            counts[kind] += 1
            if kind == "upsert":
                if row["legacy_row_index"] != legacy_index:
                    raise ValueError("Revision sidecar legacy row mapping corrupted")
                legacy_index += 1
                sequence.update((row["pmid"] + "\n").encode())
            elif row["legacy_row_index"] is not None:
                raise ValueError("Deletion cannot reference a legacy citation row")
    if event_index != expected["events"] or counts["upsert"] != expected["upserts"] or counts["delete"] != expected["deletions"]:
        raise ValueError("Revision sidecar counts changed")
    if sequence.hexdigest() != expected["pmid_sequence_sha256"]:
        raise ValueError("Revision sidecar PMID sequence changed")


def build_shard(corpus_root: Path, output: Path, shard: dict, shard_order: int) -> dict:
    """Verify and atomically publish one sidecar; reuse verified partitions."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    corpus_root, output = _resolved_path(Path(corpus_root)), _resolved_path(Path(output))
    if _is_inside(output, corpus_root) or _is_inside(corpus_root, output):
        raise ValueError("Ledger output must be outside and separate from the original corpus")
    raw = _inside(corpus_root, shard["raw_relative_path"])
    legacy = _inside(corpus_root, shard["relative_path"])
    if shard["source"] not in {"baseline", "updates"} or not re.fullmatch(r"pubmed[0-9]{2}n[0-9]+\.xml\.gz", shard["source_shard"]):
        raise ValueError("Invalid source/shard identity")
    implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    name = shard["source_shard"].removesuffix(".xml.gz")
    destination = _inside(output, f"partitions/{shard['source']}/{name}")
    receipt_path = destination / "SHARD_RECEIPT.json"
    legacy_identity = _hashes(legacy)
    if legacy_identity["sha256"] != shard["sha256"]:
        raise ValueError("Normalized Parquet changed since corpus snapshot")
    if destination.exists():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            current_raw = _hashes(raw)
            if (receipt["ledger_version"] != LEDGER_VERSION or receipt["shard_order"] != shard_order
                or receipt.get("implementation_sha256") != implementation_sha256
                or receipt["status"] != "PASS" or receipt["upserts"] != shard["rows"]
                or receipt["raw_relative_path"] != shard["raw_relative_path"]
                or receipt["legacy_relative_path"] != shard["relative_path"]
                or receipt["raw"] != current_raw or receipt["legacy"] != legacy_identity
                or current_raw["md5"] != shard["acquisition_md5"]):
                raise ValueError("Revision sidecar input identity changed")
            if _hashes(destination / "events.parquet")["sha256"] != receipt["sidecar_sha256"]:
                raise ValueError("Revision sidecar checksum changed")
            _validate_sidecar(destination / "events.parquet", receipt)
            return {**receipt, "reused": True}
        except (OSError, ValueError, KeyError, TypeError):
            quarantine = _inside(output, f"quarantine/{time.time_ns()}-{name}")
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            destination.replace(quarantine)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp-" + uuid.uuid4().hex)
    temporary.mkdir()
    raw_stat, legacy_stat = raw.stat(), legacy.stat()
    legacy_rows = iter(_legacy_pmids(legacy))
    counts = collections.Counter()
    sequence = hashlib.sha256()
    buffer = []
    sidecar = temporary / "events.parquet"
    started = time.perf_counter()
    try:
        with raw.open("rb") as stream, pq.ParquetWriter(sidecar, _schema(), compression="zstd") as writer:
            hashed = _HashingReader(stream)
            with gzip.GzipFile(fileobj=hashed, mode="rb") as decoded:
                for row in iter_xml_events(decoded, source_shard=shard["source_shard"], source=shard["source"], shard_order=shard_order):
                    counts["events"] += 1
                    counts[row["event_type"]] += 1
                    counts["unknown_version"] += row["version"] is None
                    if row["event_type"] == "upsert":
                        sentinel = object()
                        expected_pmid = next(legacy_rows, sentinel)
                        if expected_pmid is sentinel or expected_pmid != row["pmid"]:
                            raise ValueError(f"Raw/Parquet PMID sequence mismatch at row {row['legacy_row_index']}")
                        sequence.update((row["pmid"] + "\n").encode())
                        counts["unresolved_date_revised"] += row["date_revised"] is None
                    buffer.append(row)
                    if len(buffer) >= 4096:
                        writer.write_table(pa.Table.from_pylist(buffer, schema=_schema()))
                        buffer.clear()
            while hashed.read(1024 * 1024):
                pass
            raw_identity = {"sha256": hashed.sha.hexdigest(), "md5": hashed.md5.hexdigest(), "size_bytes": raw_stat.st_size}
            if buffer:
                writer.write_table(pa.Table.from_pylist(buffer, schema=_schema()))
        sentinel = object()
        if next(legacy_rows, sentinel) is not sentinel:
            raise ValueError("Normalized Parquet has additional rows absent from raw XML")
        if raw_identity["md5"] != shard["acquisition_md5"]:
            raise ValueError("Raw shard MD5 differs from acquired source checksum")
        if (raw.stat().st_size, raw.stat().st_mtime_ns) != (raw_stat.st_size, raw_stat.st_mtime_ns):
            raise ValueError("Raw XML changed during processing")
        if (legacy.stat().st_size, legacy.stat().st_mtime_ns) != (legacy_stat.st_size, legacy_stat.st_mtime_ns):
            raise ValueError("Normalized Parquet changed during processing")
        receipt = {
            "ledger_version": LEDGER_VERSION, "source": shard["source"],
            "implementation_sha256": implementation_sha256,
            "prior_snapshot_shard_status": shard.get("status"),
            "prior_snapshot_issues": shard.get("issues", []),
            "source_shard": shard["source_shard"], "shard_order": shard_order,
            "raw_relative_path": shard["raw_relative_path"], "legacy_relative_path": shard["relative_path"],
            "raw": raw_identity, "legacy": legacy_identity, "events": counts["events"],
            "upserts": counts["upsert"], "deletions": counts["delete"],
            "unknown_version": counts["unknown_version"],
            "unresolved_date_revised": counts["unresolved_date_revised"],
            "pmid_sequence_sha256": sequence.hexdigest(), "sequence_verified": True,
            "sidecar_sha256": _hashes(sidecar)["sha256"],
            "status": "PASS", "elapsed_seconds": time.perf_counter() - started,
        }
        if counts["upsert"] != shard["rows"]:
            raise ValueError("Full raw/Parquet row count differs from snapshot")
        _validate_sidecar(sidecar, receipt)
        atomic_json(temporary / "SHARD_RECEIPT.json", receipt)
        temporary.replace(destination)
        return {**receipt, "reused": False}
    except BaseException:
        legacy_rows.close()
        if temporary.exists():
            failed = _inside(output, f"failed/{time.time_ns()}-{name}")
            failed.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(failed)
        raise


def _process_initializer():
    """Bound Arrow pools in spawned workers instead of multiplying CPU pools."""
    import pyarrow as pa

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def _build_shard_task(corpus_root, output, shard, shard_order):
    """Top-level picklable target shared by thread and spawned-process pools."""
    return build_shard(Path(corpus_root), Path(output), shard, shard_order)


def build_revision_ledger(corpus_root: Path, snapshot_path: Path, output: Path, *, workers=2, max_shards=None, shard_names=None, executor="thread") -> dict:
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("workers must be between one and four")
    if max_shards is not None and (type(max_shards) is not int or max_shards < 1):
        raise ValueError("max_shards must be a positive integer")
    if executor not in {"thread", "process"}:
        raise ValueError("executor must be thread or process")
    corpus_root, output = _resolved_path(Path(corpus_root)), _resolved_path(Path(output))
    if _is_inside(output, corpus_root) or _is_inside(corpus_root, output):
        raise ValueError("Ledger output must be outside and separate from the original corpus")
    snapshot_bytes = Path(snapshot_path).read_bytes()
    snapshot = json.loads(snapshot_bytes)
    snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    shards = sorted(snapshot["shards"], key=lambda item: (item["source"] != "baseline", item["source_shard"]))
    if not shards:
        raise ValueError("Corpus snapshot contains no shards")
    if len({(item["source"], item["source_shard"]) for item in shards}) != len(shards):
        raise ValueError("Duplicate shard identity in snapshot")
    selected = [(order, item) for order, item in enumerate(shards) if not shard_names or item["source_shard"] in shard_names]
    if shard_names and set(shard_names) != {item["source_shard"] for _, item in selected}:
        raise ValueError("Requested shard is absent from the input snapshot")
    if max_shards is not None:
        selected = selected[:max_shards]
    started = time.perf_counter()
    completed, failed = [], []
    with integration_lease(output, "pubmed-revision-ledger"):
        pool = (
            concurrent.futures.ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=_process_initializer,
            )
            if executor == "process"
            else concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        )
        try:
            futures = {pool.submit(_build_shard_task, str(corpus_root), str(output), shard, order): shard for order, shard in selected}
            for future in concurrent.futures.as_completed(futures):
                shard = futures[future]
                try:
                    completed.append(future.result())
                except Exception as error:
                    failed.append({"source_shard": shard["source_shard"], "error_type": type(error).__name__, "error": str(error)})
                atomic_json(output / "BUILD_STATE.json", {"ledger_version": LEDGER_VERSION, "selected": len(selected), "completed": len(completed), "failed": failed, "updated_at": time.time()})
        finally:
            # An interrupt waits only for active partitions, not every queued
            # shard. Their atomic receipts remain reusable on the next run.
            pool.shutdown(wait=True, cancel_futures=True)
        if _hashes(Path(snapshot_path))["sha256"] != snapshot_sha256:
            failed.append({"source_shard": None, "error_type": "InputChanged", "error": "Corpus snapshot changed during processing"})
        full = not shard_names and max_shards is None
        report = {
            "ledger_version": LEDGER_VERSION, "status": "FAIL" if failed else "PASS" if full else "PARTIAL",
            "scope": "baseline_and_updates" if full else "bounded_shard_subset",
            "snapshot_sha256": snapshot_sha256,
            "prior_snapshot_status": snapshot.get("status"), "prior_snapshot_modified": False,
            "available_shards": len(shards), "selected_shards": len(selected),
            "verified_shards": len(completed), "reused_shards": sum(item["reused"] for item in completed),
            "upserts": sum(item["upserts"] for item in completed), "deletions": sum(item["deletions"] for item in completed),
            "workers": workers, "executor": executor,
            "process_start_method": "spawn" if executor == "process" else None,
            "pool_startup_included": True,
            "elapsed_seconds": time.perf_counter() - started,
            "shards": sorted(completed, key=lambda item: item["shard_order"]), "failures": failed,
            "limitations": ["This additive ledger verifies citation identity/order, not biomedical truth.", "No missing citation version is inferred.", "Current-view resolution is a separate operation; every historical event remains retained."],
        }
        atomic_json(output / "LEDGER_MANIFEST.json", report)
        atomic_json(output / f"RUN_RECEIPT_{time.time_ns()}.json", report)
    return report


def resolve_pmid_events(events: Iterable[dict]) -> dict:
    """Retain all history; select highest active numeric version, then last event.

    Explicit-version deletions remove that version. Unversioned deletions are
    PMID-wide tombstones, not an invented version. After such a tombstone, only
    newly observed upserts can restore the citation.
    """
    history = sorted((dict(event) for event in events), key=lambda event: (event["shard_order"], event["event_index"]))
    if not history:
        raise ValueError("At least one PMID event is required")
    pmids = {event["pmid"] for event in history}
    if len(pmids) != 1:
        raise ValueError("Resolve one PMID at a time")
    positions = [(event["shard_order"], event["event_index"]) for event in history]
    if len(set(positions)) != len(positions):
        raise ValueError("Ambiguous duplicate event position")
    active, unknown, unresolved_deletions = {}, [], []
    for event in history:
        if any(type(event[key]) is not int or event[key] < 0 for key in ("shard_order", "event_index")):
            raise ValueError("Event positions must be nonnegative integers")
        version = event.get("version")
        if version is not None and (type(version) is not int or version < 1):
            raise ValueError("Citation version must be a positive integer or unknown")
        if event["event_type"] == "upsert":
            if version is None:
                unknown.append(event)
            else:
                active[version] = event
        elif event["event_type"] == "delete":
            if version is None and event.get("version_raw") not in {None, ""}:
                unresolved_deletions.append(event)
                continue
            if version is None:
                active.clear()
                unknown.clear()
                unresolved_deletions.clear()
            else:
                active.pop(version, None)
        else:
            raise ValueError("Unknown revision event type")
    selected = active[max(active)] if active and not unknown and not unresolved_deletions else None
    return {
        "pmid": history[0]["pmid"], "history": history,
        "active_versions": sorted(active), "unresolved_version_events": unknown,
        "unresolved_deletion_events": unresolved_deletions,
        "selected": selected,
        "status": "UNRESOLVED_VERSION" if unknown or unresolved_deletions else "ACTIVE" if selected else "DELETED",
        "selection_rule": "highest_numeric_version_then_latest_event_within_version",
    }
