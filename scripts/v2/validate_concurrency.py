"""Run two useful read-only contract suites concurrently and retain a receipt."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import time
from pathlib import Path


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.source.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    suites = [
        "tests/test_v2_scientific_challenges.py",
        "tests/test_v2_reliability_challenges.py",
    ]
    running = []
    for index, suite in enumerate(suites, 1):
        log = args.out / f"reader-{index}.log"
        handle = log.open("wb")
        started_at = _now()
        started = time.monotonic()
        process = subprocess.Popen(
            [str(args.python), "-m", "pytest", suite, "-q"],
            cwd=source,
            stdout=handle,
            stderr=subprocess.STDOUT,
            shell=False,
        )
        running.append((suite, process, handle, log, started, started_at))
    jobs = []
    for suite, process, handle, log, started, started_at in running:
        code = process.wait(timeout=120)
        finished = time.monotonic()
        finished_at = _now()
        handle.close()
        jobs.append(
            {
                "suite": suite,
                "pid": process.pid,
                "started_at": started_at,
                "finished_at": finished_at,
                "started_monotonic": started,
                "finished_monotonic": finished,
                "exit_code": code,
                "log": log.name,
                "log_bytes": log.stat().st_size,
                "log_sha256": _sha256(log),
            }
        )
    overlap = max(
        0.0,
        min(job["finished_monotonic"] for job in jobs)
        - max(job["started_monotonic"] for job in jobs),
    )
    receipt = {
        "schema_version": "concurrency-validation-v1",
        "status": "PASS"
        if overlap > 0 and all(job["exit_code"] == 0 for job in jobs)
        else "FAIL",
        "source_commit": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "jobs": jobs,
        "overlap_seconds": overlap,
        "writer_policy": "read jobs may overlap; controller integration lease permits one writer",
    }
    target = args.out / "concurrency_validation.json"
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return int(receipt["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
