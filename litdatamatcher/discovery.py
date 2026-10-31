"""Bounded query-plan acquisition with independent, replayable source nodes.

Each source job owns its HTTP client. Successful checkpoints are immutable and
hash-verified on resume; empty results and source failures remain distinct.
Only metadata is acquired here. Discovery never asserts scientific feasibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .adapters import DATASET_ADAPTERS, LITERATURE_ADAPTERS
from .controller import integration_lease
from .data_plane import atomic_json, atomic_write, digest
from .http_cache import CachedHttpClient
from .strict_artifacts import strict_json_loads

VERSION = "discovery-plan-1.0"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_plan(plan: dict) -> list[dict]:
    """Expand a small explicit plan; reject ambiguous IDs and unbounded requests."""
    if not isinstance(plan, dict) or plan.get("schema_version") != VERSION:
        raise ValueError(f"Expected {VERSION} plan")
    queries = plan.get("queries")
    if not isinstance(queries, list) or not 1 <= len(queries) <= 100:
        raise ValueError("A plan requires 1..100 queries")
    jobs, seen = [], set()
    for row in queries:
        if not isinstance(row, dict):
            raise ValueError("Each query must be a JSON object")
        key = row.get("id", "")
        if not isinstance(key, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", key) or key in seen:
            raise ValueError("Query IDs must be unique, safe path components")
        seen.add(key)
        if not isinstance(row.get("query"), str) or not row["query"].strip():
            raise ValueError("A nonempty query is required")
        limit = row.get("limit", 5)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Per-source retained result limit must be 1..100")
        before = len(jobs)
        for kind, registry in (("dataset", DATASET_ADAPTERS), ("literature", LITERATURE_ADAPTERS)):
            sources = row.get(kind + "_sources", [])
            if (
                not isinstance(sources, list)
                or not all(isinstance(s, str) for s in sources)
                or len(set(sources)) != len(sources)
            ):
                raise ValueError("Sources must be a list without duplicates")
            for source in sources:
                if source not in registry:
                    raise ValueError(f"Unknown {kind} source: {source}")
                jobs.append(
                    {
                        "job_id": f"{key}--{kind}--{source}",
                        "query_id": key,
                        "kind": kind,
                        "source": source,
                        "query": row["query"],
                        "limit": limit,
                    }
                )
        if before == len(jobs):
            raise ValueError("Each query requires at least one source")
    return jobs


def _fetch(job: dict, cache: Path, offline: bool) -> dict:
    started = time.perf_counter()
    client = CachedHttpClient(cache, offline=offline)
    registry = DATASET_ADAPTERS if job["kind"] == "dataset" else LITERATURE_ADAPTERS
    adapter = registry[job["source"]](client)
    try:
        values = (
            adapter.search(job["query"])
            if job["kind"] == "dataset"
            else adapter.search_literature(job["query"], limit=job["limit"])
        )
        values = [value.to_dict() if hasattr(value, "to_dict") else value for value in values]
        pagination = getattr(adapter, "last_search_status", {})
        failure = pagination.get("status") in {"ERROR", "SCHEMA_DRIFT", "REPEATED_CURSOR"}
        return {
            "status": "PARTIAL_SOURCE_FAILURE" if failure else ("SUCCESS" if values else "EMPTY"),
            "records": values[: job["limit"]],
            "source_status": pagination,
            "returned_before_limit": len(values),
            "retained_limit": job["limit"],
            "elapsed_seconds": time.perf_counter() - started,
        }
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        return {
            "status": "SOURCE_FAILURE",
            "records": [],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.perf_counter() - started,
        }


def _native_record(record: dict, kind: str, job: dict) -> dict:
    value = dict(record)
    value["split_context"] = "development"
    value["discovery_observations"] = [job]
    if kind == "dataset":
        provenance = value.get("metadata", {}).get("source_provenance", {})
        value.update(
            schema_version="discovery-metadata-1.0",
            summary=value.get("description", ""),
            source_locator=provenance.get("source_locator") or value.get("url"),
            source_provenance=provenance,
            capabilities={
                field: {
                    "value": None,
                    "status": "unknown",
                    "source_locator": None,
                    "reason": "Not established by structured source evidence",
                }
                for field in (
                    "species",
                    "organism",
                    "assay",
                    "modality",
                    "tissue",
                    "intervention",
                    "comparator",
                    "outcome",
                    "paired",
                    "timepoint",
                )
            },
            independent_units=None,
            independent_unit_status="unknown",
        )
    else:
        value["document_id"] = value.get("document_id") or value.get("source_id")
        if not value["document_id"]:
            raise ValueError("Literature record lacks source identity")
        value["text"] = value.get("text") or value.get("abstract", "")
        value["source_locator"] = (
            value.get("source_provenance", {}).get("source_locator")
            or value.get("url")
            or value["document_id"]
        )
    return value


def discover(
    plan: dict,
    out: Path,
    *,
    workers: int = 2,
    offline: bool = False,
    resume: bool = False,
    retry_failed: bool = False,
    _fetcher=None,
) -> dict:
    """Build a native workspace from an explicit plan, retaining node-level evidence."""
    jobs = validate_plan(plan)
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use 1..4 concurrent I/O workers")
    out = Path(out).resolve()
    implementation = {
        name: _hash(Path(__file__).with_name(name))
        for name in (
            "discovery.py",
            "adapters.py",
            "omics_adapters.py",
            "http_cache.py",
            "provenance.py",
            "datasets.py",
        )
    }
    identity = {"plan": plan, "implementation": implementation, "version": VERSION}
    fingerprint = digest(identity)
    with integration_lease(out, "query-discovery"):
        state_path = out / "DISCOVERY_STATE.json"
        if state_path.exists():
            state = strict_json_loads(state_path.read_text(encoding="utf-8"))
            if not resume or state.get("fingerprint") != fingerprint:
                raise ValueError("Resume requires the same query plan and implementation")
        else:
            if resume:
                raise ValueError("No discovery state to resume")
            if any(path.name not in {"integration.lock", "owner.json"} for path in out.iterdir()):
                raise ValueError("A new discovery requires an empty output directory")
            state = {"fingerprint": fingerprint, "identity": identity, "completed": {}}
            atomic_json(state_path, state)
        inspections = out / "catalog" / "processed_inspections.jsonl"
        if inspections.exists() and inspections.stat().st_size:
            raise ValueError(
                "Preserving nonempty processed-inspection catalog; use a new discovery output"
            )
        pending = []
        for job in jobs:
            checkpoint = out / "checkpoints" / (job["job_id"] + ".json")
            expected = state["completed"].get(job["job_id"])
            if expected:
                if not checkpoint.is_file() or _hash(checkpoint) != expected:
                    raise ValueError(f"Discovery checkpoint integrity failure: {job['job_id']}")
                cached = strict_json_loads(checkpoint.read_text(encoding="utf-8"))
                if not retry_failed or cached["result"]["status"] in {"SUCCESS", "EMPTY"}:
                    continue
            pending.append(job)
        fetcher = _fetcher or _fetch
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="discovery") as pool:
            futures = {
                pool.submit(fetcher, job, out / "http_cache" / job["job_id"], offline): job
                for job in pending
            }
            for future in as_completed(futures):
                job = futures[future]
                result = future.result()
                checkpoint = out / "checkpoints" / (job["job_id"] + ".json")
                # Preserve failed-attempt bytes before an explicitly requested retry.
                if checkpoint.exists():
                    atomic_write(
                        out / "attempt_history" / (_hash(checkpoint) + ".json"),
                        checkpoint.read_bytes(),
                    )
                atomic_json(checkpoint, {"job": job, "result": result})
                state["completed"][job["job_id"]] = _hash(checkpoint)
                atomic_json(state_path, state)
        catalogues: dict[str, dict] = {"dataset": {}, "literature": {}}
        statuses = []
        for job in jobs:
            checkpoint = out / "checkpoints" / (job["job_id"] + ".json")
            result = strict_json_loads(checkpoint.read_text(encoding="utf-8"))["result"]
            statuses.append(
                {
                    "job_id": job["job_id"],
                    "status": result["status"],
                    "records": len(result["records"]),
                    "checkpoint_sha256": _hash(checkpoint),
                }
            )
            for raw in result["records"]:
                observation = {
                    **job,
                    "record_sha256": digest(raw),
                    "checkpoint": checkpoint.relative_to(out).as_posix(),
                }
                row = _native_record(raw, job["kind"], observation)
                key = row["dataset_id" if job["kind"] == "dataset" else "document_id"]
                if key in catalogues[job["kind"]]:
                    catalogues[job["kind"]][key]["discovery_observations"].append(observation)
                else:
                    catalogues[job["kind"]][key] = row
        outputs = []
        for kind, filename in (("dataset", "studies.jsonl"), ("literature", "literature.jsonl")):
            rows = [catalogues[kind][key] for key in sorted(catalogues[kind])]
            path = out / "catalog" / filename
            atomic_write(
                path,
                b"".join(
                    json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
                    + b"\n"
                    for row in rows
                ),
            )
            outputs.append(
                {
                    "path": path.relative_to(out).as_posix(),
                    "sha256": _hash(path),
                    "records": len(rows),
                }
            )
        # Metadata discovery performs no processed-data inspection. Native
        # analysis still requires the explicit, valid zero-row catalogue.
        atomic_write(inspections, b"")
        outputs.append(
            {
                "path": inspections.relative_to(out).as_posix(),
                "sha256": _hash(inspections),
                "records": 0,
                "record_count": 0,
                "inspection_status": "NOT_PERFORMED",
            }
        )
        failures = sum(
            row["status"] in {"SOURCE_FAILURE", "PARTIAL_SOURCE_FAILURE"} for row in statuses
        )
        report = {
            "schema_version": VERSION,
            "status": "PASS_WITH_SOURCE_FAILURES" if failures else "PASS",
            "fingerprint": fingerprint,
            "jobs": statuses,
            "failed_source_jobs": failures,
            "offline": offline,
            "workers": workers,
            "outputs": outputs,
            "interpretation": "Bounded metadata discovery; empty retrieval is not evidence of absence; no dataset feasibility inferred.",
        }
        atomic_json(out / "DISCOVERY_MANIFEST.json", report)
        return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    report = discover(
        strict_json_loads(args.plan.read_text(encoding="utf-8-sig")),
        args.out,
        workers=args.workers,
        offline=args.offline,
        resume=args.resume,
        retry_failed=args.retry_failed,
    )
    print(json.dumps(report, indent=2))
    return int(bool(report["failed_source_jobs"]))


if __name__ == "__main__":
    raise SystemExit(main())
