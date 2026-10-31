"""Execute the twelve source-defined comparisons through the local application API."""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import requests

from litdatamatcher.data_plane import atomic_json


def demonstrate(source, out):
    base = "http://127.0.0.1:8765"
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    receipt = out / "API_DEMONSTRATION.json"
    state = (
        json.loads(receipt.read_text(encoding="utf-8"))
        if receipt.exists()
        else {"source_sha256": digest, "cases": {}}
    )
    if state["source_sha256"] != digest:
        raise ValueError("Changed requests require a new output directory")
    cases = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(cases) == len({case["case_id"] for case in cases}) == 12
    for case in cases:
        key = case["case_id"]
        target = re.search(r"\bUCI:\d+", case["question"]).group(0)
        if key not in state["cases"]:
            payload = {
                "question": case["question"],
                "query": " ".join(
                    str(r["expected"])
                    for r in case["user_constraints"]
                    if r["field"] in {"outcome_variable", "grouping_variable"}
                ),
                "requirements": case["user_constraints"],
                "literature_sources": [],
                "dataset_sources": [],
                "inspection_limit": 0,
                "limit": 3,
                "extraction_mode": "rules",
                "ranking": "bm25",
            }
            response = requests.post(base + "/api/jobs", json=payload, timeout=30)
            response.raise_for_status()
            state["cases"][key] = {
                "job_id": response.json()["id"],
                "target": target,
                "request": payload,
            }
            atomic_json(receipt, state)
        row = state["cases"][key]
        deadline = time.monotonic() + 180
        while True:
            response = requests.get(base + "/api/jobs/" + row["job_id"], timeout=30)
            response.raise_for_status()
            job = response.json()
            if job["status"] in {"SUCCEEDED", "FAILED", "RECOVERY_REQUIRED"}:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Preserved resumable job " + row["job_id"])
            time.sleep(0.5)
        atomic_json(out / (key + ".json"), job)
        assert job["status"] == "SUCCEEDED", job.get("error")
        result = job["result"]
        assert result["processing_status"] == "COMPLETE"
        matched = next(c for c in result["candidates"] if c["dataset_id"] == target)
        assert matched["assessment"]["eligibility"] == "DIRECT_FIT"
        methods = {r["field"]: r for r in matched["assessment"]["requirements"]}
        for field in ("joint_observation", "outcome_ascertainment"):
            assert (
                methods[field]["essential"]
                and methods[field]["expected"] is True
                and methods[field]["status"] == "MATCH"
            )
        original = next(d for d in result["datasets"] if d["dataset_id"] == target)
        assert "comparison_scope" not in original
        assert not any(
            f in original["capabilities"]
            for f in ("joint_observation", "outcome_ascertainment", "temporal_order")
        )
        row.update(
            status="PASS",
            result_id=result["result_id"],
            result_sha256=hashlib.sha256((out / (key + ".json")).read_bytes()).hexdigest(),
        )
        atomic_json(receipt, state)
        print(key, row["job_id"], "DIRECT_FIT; method evidence verified", flush=True)
    state.update(
        status="PASS_WITH_LIMITATIONS",
        completed=12,
        distinct_datasets=2,
        scientific_status="SOURCE_ASSISTED_PENDING_EXPERT_REVIEW",
        limitations=[
            "Twelve descriptive development contracts reuse two datasets.",
            "No novelty, independent units, statistical power or clinical utility is established.",
        ],
    )
    atomic_json(receipt, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    demonstrate(args.requests, args.out)
