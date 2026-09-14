#!/usr/bin/env python3
"""Non-destructive heartbeat recorder for the LitDataMatcher v0.4 campaign.

This utility observes declared PIDs and artifacts.  It never starts, stops, or
retries jobs and does not alter campaign conclusions or scientific results.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def json_load(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def process_status(pid: int) -> dict[str, Any]:
    result: dict[str, Any] = {"pid": pid, "alive": False}
    try:
        import psutil  # type: ignore

        proc = psutil.Process(pid)
        result.update(
            alive=proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE,
            cpu_percent=proc.cpu_percent(interval=None),
            rss_bytes=proc.memory_info().rss,
            create_time_utc=dt.datetime.fromtimestamp(proc.create_time(), tz=dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            command=proc.cmdline(),
        )
        return result
    except ImportError:
        pass
    except (OSError, ProcessLookupError):
        return result
    try:
        completed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        result["alive"] = str(pid) in completed.stdout and "No tasks" not in completed.stdout
    except (OSError, subprocess.SubprocessError):
        result["alive"] = None
    return result


def memory_status() -> dict[str, int] | None:
    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_phys", ctypes.c_ulonglong),
            ("avail_phys", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("avail_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("avail_virtual", ctypes.c_ulonglong),
            ("avail_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return {"total_bytes": status.total_phys, "available_bytes": status.avail_phys, "load_percent": status.memory_load}


def gpu_status() -> list[dict[str, str]] | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    fields = ("index", "name", "utilization_percent", "memory_used_mb", "memory_total_mb")
    return [dict(zip(fields, (part.strip() for part in line.split(",")), strict=True)) for line in completed.stdout.splitlines() if line.strip()]


def tree_metrics(root: Path) -> dict[str, Any]:
    total_bytes = 0
    file_count = 0
    latest: Path | None = None
    latest_mtime = -1.0
    try:
        for item in root.rglob("*"):
            if not item.is_file():
                continue
            stat = item.stat()
            total_bytes += stat.st_size
            file_count += 1
            if stat.st_mtime > latest_mtime:
                latest, latest_mtime = item, stat.st_mtime
    except OSError as exc:
        return {"error": str(exc)}
    return {
        "path": str(root),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "latest_artifact": str(latest) if latest else None,
        "latest_artifact_utc": dt.datetime.fromtimestamp(latest_mtime, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z") if latest else None,
    }


def pubmed_progress(corpus_root: Path) -> dict[str, Any] | None:
    state = json_load(corpus_root / "manifests" / "acquisition" / "pubmed_2026_baseline_state.json")
    if not state:
        return None
    files = state.get("files", {}).values()
    tally: dict[str, int] = {}
    verified_bytes = 0
    latest = None
    for entry in files:
        status = entry.get("status", "UNKNOWN")
        tally[status] = tally.get(status, 0) + 1
        if status == "CHECKSUM_PASS":
            verified_bytes += int(entry.get("bytes", 0))
        completed = entry.get("completed_utc")
        if completed and (latest is None or completed > latest):
            latest = completed
    return {"expected_shards": state.get("expected_shards"), "status_counts": tally, "verified_bytes": verified_bytes, "latest_success_utc": latest, "state_updated_utc": state.get("updated_utc")}


def pubmed_normalization_progress(corpus_root: Path) -> dict[str, Any] | None:
    state = json_load(corpus_root / "manifests" / "acquisition" / "pubmed_2026_normalization_state.json")
    if not state:
        return None
    shards = state.get("shards", {}).values()
    tally: dict[str, int] = {}
    records = 0
    latest = None
    for entry in shards:
        status = entry.get("status", "UNKNOWN")
        tally[status] = tally.get(status, 0) + 1
        if status == "NORMALIZED":
            records += int(entry.get("records", 0))
        completed = entry.get("completed_utc")
        if completed and (latest is None or completed > latest):
            latest = completed
    return {"status_counts": tally, "normalized_records": records, "latest_success_utc": latest, "state_created_utc": state.get("created_utc")}


def heartbeat(args: argparse.Namespace) -> dict[str, Any]:
    campaign_state = json_load(args.state) or {}
    corpus_root = args.corpus_root
    artifact_root = corpus_root / "raw" / "pubmed" / "2026_baseline"
    disk = shutil.disk_usage(corpus_root)
    return {
        "schema_version": "v0.4-watchdog-heartbeat-1.0",
        "timestamp_utc": utc_now(),
        "campaign_stage": campaign_state.get("current_stage"),
        "campaign_substage": campaign_state.get("current_substage"),
        "campaign_status_observed": campaign_state.get("overall_status"),
        "processes": [process_status(pid) for pid in args.pid],
        "system_memory": memory_status(),
        "gpu": gpu_status(),
        "disk": {"path": str(corpus_root), "free_bytes": disk.free, "total_bytes": disk.total},
        "output_metrics": tree_metrics(artifact_root),
        "pubmed_baseline_progress": pubmed_progress(corpus_root),
        "pubmed_normalization_progress": pubmed_normalization_progress(corpus_root),
        "last_successful_checkpoint": campaign_state.get("last_progress_timestamp"),
        "last_error": None,
        "watchdog_policy": "OBSERVE_ONLY__NO_JOB_CONTROL__NO_SCIENTIFIC_CONCLUSIONS"
    }


def write_record(record: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = output_dir / "heartbeat.json"
    history = output_dir / "heartbeat_history.jsonl"
    latest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(r"C:\\Codex\\LitDataMatcher-v2\\data\\v0.4_compiler_20260913\\controller"))
    parser.add_argument("--pid", type=int, action="append", default=[])
    parser.add_argument("--once", action="store_true", help="Write one heartbeat (the default).")
    parser.add_argument("--loop", action="store_true", help="Repeat at a conservative heartbeat interval.")
    parser.add_argument("--interval-minutes", type=int, default=15)
    args = parser.parse_args()
    if args.loop and not 10 <= args.interval_minutes <= 30:
        parser.error("--interval-minutes must be between 10 and 30 in --loop mode")
    if args.once and args.loop:
        parser.error("choose at most one of --once and --loop")
    while True:
        record = heartbeat(args)
        write_record(record, args.out)
        print(json.dumps({"heartbeat": str(args.out / "heartbeat.json"), "timestamp_utc": record["timestamp_utc"]}, sort_keys=True))
        if not args.loop:
            return 0
        time.sleep(args.interval_minutes * 60)


if __name__ == "__main__":
    raise SystemExit(main())
