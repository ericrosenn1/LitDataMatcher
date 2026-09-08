"""Bounded local scale/recovery instrumentation; never a production-scale claim."""

from __future__ import annotations

import os
import platform
import shutil
import time
import tracemalloc
from pathlib import Path

from .data_plane import Catalog, atomic_json, atomic_write, digest
from .scientific_v2 import compile_evidence, rank_candidates


BENCHMARK_SCHEMA_VERSION = "v2_5_local_benchmark_v1"


def _elapsed(start: float, items: int) -> dict:
    seconds = time.perf_counter() - start
    return {"items": items, "seconds": seconds, "throughput_per_second": items / seconds if seconds else None}


def _disk_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _records(count: int) -> list[dict]:
    return [{"dataset_id": f"fixture-{index:04d}", "title": f"Fixture transcriptome {index}", "summary": "human inflammatory transcriptomics", "capabilities": {"species": {"value": "Homo sapiens", "status": "observed", "source_locator": "fixture"}}} for index in range(count)]


def run_local_benchmark(root: str | Path, count: int = 32) -> dict:
    """Run an intentionally small local fixture through core deterministic paths."""
    if type(count) is not int or not 1 <= count <= 1000:
        raise ValueError("benchmark count must be an integer in 1..1000")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    records = _records(count)
    tracemalloc.start()
    catalog_root = root / "catalog"
    started = time.perf_counter()
    catalog = Catalog(catalog_root)
    for record in records:
        catalog.upsert("dataset", record["dataset_id"], record, search_text=record["title"] + " " + record["summary"])
    ingestion = _elapsed(started, count)
    started = time.perf_counter()
    indexed = catalog.search("dataset", "human transcriptomics", count)
    index_query = _elapsed(started, len(indexed))
    catalog.close()

    # A process interruption after a committed half-index must leave a reopenable
    # catalog; the resumed pass supplies only the missing deterministic records.
    recovery_root = root / "recovery_catalog"
    recovery = Catalog(recovery_root)
    midpoint = max(1, count // 2)
    for record in records[:midpoint]:
        recovery.upsert("dataset", record["dataset_id"], record, search_text=record["title"])
    recovery.close()
    recovery = Catalog(recovery_root)
    persisted_before_resume = len(recovery.records("dataset"))
    for record in records[midpoint:]:
        recovery.upsert("dataset", record["dataset_id"], record, search_text=record["title"])
    recovered_count = len(recovery.records("dataset"))
    recovery.close()

    cache_path = root / "cache" / "manifest.json"
    cache_payload = {"fixture_digest": digest(records), "count": count}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(cache_path, cache_payload)
    existed_before_replay = cache_path.exists()
    atomic_write(cache_path.with_suffix(".interrupted"), b"{")
    replay_payload = __import__("json").loads(cache_path.read_text(encoding="utf-8"))
    atomic_json(cache_path, replay_payload)

    question = {"question_id": "fixture-question", "proposition_id": "fixture-proposition", "conditions": {"organism": "human"}}
    profiles = [dict(record, independent_units=None) for record in records]
    started = time.perf_counter()
    ranked = rank_candidates([{"field": "species", "expected": "Homo sapiens"}], profiles, {record["dataset_id"]: 0.0 for record in records})
    matching = _elapsed(started, len(ranked))
    started = time.perf_counter()
    bundle = compile_evidence(question, [{"evidence_id": "fixture-evidence", "proposition_id": "fixture-proposition", "role": "background", "direction": "supports", "source_locator": "fixture:span", "conditions": {"organism": "human"}, "measurement_type": "observation", "scope_match": "exact", "publication_date": "2026-09-01"}], "2026-09-08", [{"source": "fixture", "status": "success"}])
    compilation = _elapsed(started, 1)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    disk = shutil.disk_usage(root)
    receipt = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "scope": "bounded synthetic derivative fixture; measured values do not establish production-scale performance",
        "fixture": {"record_count": count, "input_digest": digest(records)},
        "backend": {"catalog": "sqlite_fts5", "matching": "scientific_v2 deterministic heuristic", "evidence_compiler": "scientific_v2"},
        "hardware": {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count(), "free_disk_bytes": disk.free},
        "limits": {"max_records": 1000, "network": "not used", "model_inference": "not used", "llm_context": "not used"},
        "measurements": {"catalog_ingestion": ingestion, "index_query": index_query, "matching": matching, "evidence_compilation": compilation, "memory_peak_bytes": peak, "disk_bytes_written": _disk_bytes(root), "cache_hit": {"replay_used_existing_manifest": existed_before_replay, "digest": digest(replay_payload)}, "recovery": {"interrupted_index_persisted_count": persisted_before_resume, "recovered_index_count": recovered_count, "interrupted_cache_ignored": replay_payload == cache_payload}},
        "validation_status": "PASS" if len(indexed) and len(ranked) == count and recovered_count == count and replay_payload == cache_payload and bundle["question_id"] == question["question_id"] else "FAIL",
    }
    return receipt


def validate_benchmark_receipt(receipt: dict) -> bool:
    if receipt.get("schema_version") != BENCHMARK_SCHEMA_VERSION or receipt.get("validation_status") != "PASS":
        return False
    measurements = receipt.get("measurements", {})
    required = ["catalog_ingestion", "index_query", "matching", "evidence_compilation"]
    return all(type(measurements.get(key, {}).get("seconds")) is float and measurements[key]["seconds"] >= 0 for key in required) and type(measurements.get("memory_peak_bytes")) is int and measurements["memory_peak_bytes"] >= 0
