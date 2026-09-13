"""Bounded metadata acquisition through existing adapters; no analysis or raw downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import psutil
import requests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from litdatamatcher.acquisition_v2 import (  # noqa: E402
    SnapshotClient,
    StageLease,
    write_json,
    write_jsonl,
)
from litdatamatcher.adapters import build_dataset_adapters, build_literature_adapters  # noqa: E402

ROOT = Path(r"C:\Codex\LitDataMatcher-v2\data\final_campaign_20260913\expanded")
DATA = ROOT.parents[1]
SEALED_MANIFEST = DATA / "evaluation/final_holdout_v4/run/RUN_MANIFEST.json"
SEALED_SHA256 = "79f92391aee7b8ba0afd13032417029796975dd7b8798c3dacb9f57343727c32"
DOMAIN_QUERIES = {
    "cancer": '(cancer AND (immunotherapy OR "immune checkpoint") AND (trial OR treatment))',
    "environmental": '((soil OR marine OR ocean OR wastewater) AND (microbiome OR metagenomics))',
    "metabolic": '(("type 2 diabetes" OR "nonalcoholic fatty liver") AND (trial OR treatment OR intervention))',
    "neurologic": '(("Alzheimer disease" OR Parkinson) AND (trial OR biomarker OR treatment))',
}
DATASET_QUERIES = [
    ("cancer", "clinicaltrials", "lung", '"lung cancer" AND immunotherapy'),
    ("cancer", "clinicaltrials", "breast", '"breast cancer" AND treatment'),
    ("environmental", "mgnify", "soil", "soil"),
    ("environmental", "mgnify", "marine", "marine"),
    ("environmental", "mgnify", "wastewater", "wastewater"),
    ("environmental", "ena", "soil", 'library_source="METAGENOMIC" AND study_title="*soil*"'),
    ("environmental", "ena", "marine", 'library_source="METAGENOMIC" AND study_title="*marine*"'),
    ("metabolic", "clinicaltrials", "diabetes", '"type 2 diabetes" AND intervention'),
    ("neurologic", "clinicaltrials", "parkinson", '"Parkinson disease"'),
]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def protected_inputs():
    paths = [REPO / "benchmarks/v2" / name for name in (
        "split_reservations.json", "final_holdout_reservation_v4.json",
        "final_holdout_reservation_v3.json", "replacement_final_holdout_reservation.json",
    )]
    hashes = {str(path): digest(path.read_bytes()) for path in [*paths, SEALED_MANIFEST]}
    if hashes[str(SEALED_MANIFEST)] != SEALED_SHA256:
        raise ValueError("Frozen holdout manifest hash changed; acquisition refused")
    ids = set()
    pmids = set()
    pmcids = set()
    for path in paths:
        data = read_json(path)
        if path.name == "split_reservations.json":
            for values in data.values():
                if isinstance(values, list):
                    ids.update(str(value).upper() for value in values if isinstance(value, str))
        lineage = data.get("identifier_lineage", data.get("identifier_only_lineage", {}))
        for kind, values in lineage.items():
            if not isinstance(values, list):
                continue
            if kind == "pubmed":
                pmids.update(str(value) for value in values)
            elif kind == "pmc":
                pmcids.update("PMC" + str(value).removeprefix("PMC") for value in values)
            else:
                ids.update(str(value).upper() for value in values)
    return {"file_sha256": hashes, "known_family_ids": sorted(ids),
            "pmids": sorted(pmids), "pmcids": sorted(pmcids),
            "scope": "Exact declared identifiers only; undisclosed family links remain unknown"}


def partition_plan(protected):
    excluded = " OR ".join([*("EXT_ID:" + value for value in protected["pmids"]),
                            *("PMCID:" + value for value in protected["pmcids"])])
    partitions = []
    for domain, query in DOMAIN_QUERIES.items():
        partitions.append({"id": domain + "_europepmc", "domain": domain,
                           "kind": "literature", "source": "europepmc", "limit": 400,
                           "max_requests": 4,
                           "query": f"{query} AND FIRST_PDATE:[2018-01-01 TO 2025-12-31] NOT ({excluded})"})
        for group, source, label, dataset_query in DATASET_QUERIES:
            if group == domain:
                partitions.append({"id": f"{domain}_{source}_{label}", "domain": domain,
                                   "kind": "datasets", "source": source,
                                   "limit": 100 if source in {"clinicaltrials", "ena"} else 25,
                                   "max_requests": 4 if source == "clinicaltrials" else 1,
                                   "query": dataset_query})
    return partitions


def prepare(root):
    protected = protected_inputs()
    partitions = partition_plan(protected)
    path = root / "PREDECLARED_PLAN.json"
    if path.exists():
        plan = read_json(path)
        if plan["protected"] != protected or plan["partitions"] != partitions:
            raise ValueError("Persisted plan or protected identifiers differ; do not rewrite a started campaign")
        return plan
    plan = {"schema_version": 1, "declared_utc": now(),
            "source_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
            "script_sha256": digest(Path(__file__).read_bytes()),
            "protected": protected, "partitions": partitions,
            "targets": {"unique_literature": 1000, "unique_study_ids": 300,
                        "dataset_repositories": 2, "new_domains": 3},
            "scope": "Metadata and abstracts only; no article bodies, raw files, model calls, labels, tuning, or holdout execution",
            "resources": {"logical_cpus": os.cpu_count(), "ram_bytes": psutil.virtual_memory().total,
                          "available_ram_bytes": psutil.virtual_memory().available,
                          "concurrent_requests": 1, "max_response_bytes": 8_000_000,
                          "max_request_attempts": 3, "max_retry_wait_seconds": 15},
            "replay_ignored_field": "cache_status only, recursively; all other row content compared",
            "universe": "Predeclared bounded query samples; never an exhaustive source universe"}
    root.mkdir(parents=True, exist_ok=True)
    write_json(path, plan)
    return plan


class AdapterSnapshotClient:
    """Adapt the existing raw-byte snapshot client to existing metadata adapters."""

    def __init__(self, root, partition, offline=False):
        self.snapshot = SnapshotClient(root, offline=offline, max_bytes=8_000_000, max_retry_wait=15)
        self.offline = offline
        self.partition = partition
        self.last_response_metadata = {}
        self.events = []

    def get_json(self, url, params=None, **kwargs):
        if len(self.events) >= self.partition["max_requests"]:
            raise RuntimeError("Predeclared request bound exceeded")
        self.last_response_metadata = {}
        event = {"url": url, "params": params or {}}
        self.events.append(event)
        try:
            if not self.offline:
                time.sleep(0.25)
            payload, meta = self.snapshot.json(url, params)
            cache_status = self.snapshot.events[-1].get("status", "live_cached")
            self.last_response_metadata = {
                "cache_path": meta["object_path"], "cache_content_sha256": meta["sha256"],
                "retrieval_time_utc": meta["retrieved_at"], "cache_status": str(cache_status),
            }
            event.update({"status": "RETRIEVED", "snapshot": meta})
            if isinstance(payload, dict):
                event["source_coverage"] = {key: payload[key] for key in
                    ("hitCount", "totalCount", "count", "nextCursorMark", "nextPageToken", "links", "pagination")
                    if key in payload}
            elif isinstance(payload, list):
                event["source_coverage"] = {"returned_rows": len(payload)}
            return payload
        except Exception as exc:
            event.update({"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)})
            raise


def acquire_rows(root, partition, protected, offline=False):
    client = AdapterSnapshotClient(root / "snapshots" / partition["id"], partition, offline)
    constructor = build_literature_adapters if partition["kind"] == "literature" else build_dataset_adapters
    adapter = constructor([partition["source"]], client=client)[0]
    error = None
    try:
        if partition["kind"] == "literature":
            rows = adapter.search_literature(partition["query"], limit=partition["limit"])
        else:
            rows = [record.to_dict() for record in adapter.search(partition["query"])]
    except Exception as exc:
        rows = []
        error = {"type": type(exc).__name__, "message": str(exc)}
    ids = [*protected["known_family_ids"], *protected["pmids"], *protected["pmcids"]]
    pattern = re.compile(r"(?<![A-Z0-9])(?:" + "|".join(re.escape(value) for value in ids) + r")(?![A-Z0-9])", re.I)
    accepted, excluded = [], []
    for row in rows:
        matches = sorted(set(pattern.findall(json.dumps(row, ensure_ascii=False))))
        if matches:
            excluded.append({"source_id": row.get("source_id", row.get("dataset_id")),
                             "reason": "KNOWN_PROTECTED_FAMILY_IDENTIFIER", "identifiers": matches})
        else:
            accepted.append(row)
    pagination = getattr(adapter, "last_search_status", {})
    failed = error is not None or any(event["status"] == "ERROR" for event in client.events)
    failed = failed or pagination.get("status") in {"ERROR", "SCHEMA_DRIFT", "REPEATED_CURSOR"}
    receipt = {"partition": partition, "status": "SOURCE_FAILURE" if failed else "ACQUIRED_BOUNDED",
               "error": error, "source_pagination": pagination, "request_events": client.events,
               "snapshot_events": client.snapshot.events, "received_rows": len(rows),
               "accepted_rows": len(accepted), "protected_exclusions": excluded,
               "coverage": "BOUNDED_SAMPLE_NOT_EXHAUSTIVE",
               "ena_grain": "Technical run rows grouped into studies; runs are not biological samples" if partition["source"] == "ena" else None}
    return accepted, receipt


def identity(row, kind):
    if kind == "datasets":
        return str(row["dataset_id"])
    doi = str(row.get("doi", "")).strip().lower()
    return "doi:" + doi if doi else "source:" + str(row["source_id"])


def summarize(root, plan, complete=False):
    selected = {"literature": {}, "datasets": {}}
    memberships = []
    receipts = []
    domain_rows = {domain: {"literature": {}, "datasets": {}} for domain in DOMAIN_QUERIES}
    for partition in plan["partitions"]:
        folder = root / "partitions" / partition["id"]
        if not (folder / "receipt.json").exists():
            continue
        receipt = read_json(folder / "receipt.json")
        receipts.append(receipt)
        if receipt["status"] != "ACQUIRED_BOUNDED":
            continue
        path = folder / "records.jsonl"
        if digest(path.read_bytes()) != receipt["records_sha256"]:
            raise ValueError("Partition records changed: " + partition["id"])
        for row in read_rows(path):
            key = identity(row, partition["kind"])
            memberships.append({"identity_key": key, "partition": partition["id"],
                                "source_id": row.get("source_id", row.get("dataset_id"))})
            selected[partition["kind"]].setdefault(key, row)
            domain_rows[partition["domain"]][partition["kind"]].setdefault(key, row)
    files = {}
    for kind, rows in selected.items():
        path = root / "corpus" / (kind + ".jsonl")
        write_jsonl(path, [rows[key] for key in sorted(rows)])
        files[str(path)] = {"sha256": digest(path.read_bytes()), "rows": len(rows)}
    for domain, groups in domain_rows.items():
        for kind, rows in groups.items():
            path = root / "domains" / domain / (kind + ".jsonl")
            write_jsonl(path, [rows[key] for key in sorted(rows)])
    write_jsonl(root / "corpus/membership.jsonl", memberships)
    sources = Counter(row["source"] for row in selected["datasets"].values())
    domains = {domain: {kind: len(rows) for kind, rows in groups.items()} for domain, groups in domain_rows.items()}
    checks = {"literature_minimum": len(selected["literature"]) >= 1000,
              "study_minimum": len(selected["datasets"]) >= 300,
              "dataset_repositories": len(sources) >= 2,
              "new_domains": sum(value["literature"] > 0 and value["datasets"] > 0 for value in domains.values()) >= 3}
    receipt = {"status": ("PASS_WITH_LIMITATIONS" if all(checks.values()) else "INCOMPLETE") if complete else "RUNNING",
               "updated_utc": now(), "completion_utc": now() if complete else None,
               "source_commit": plan["source_commit"], "plan_sha256": digest((root / "PREDECLARED_PLAN.json").read_bytes()),
               "counts": {kind: len(rows) for kind, rows in selected.items()}, "dataset_sources": dict(sources),
               "domains": domains, "target_checks": checks, "files": files,
               "partitions": [{"id": value["partition"]["id"], "status": value["status"],
                               "accepted_rows": value["accepted_rows"]} for value in receipts],
               "limitations": ["Bounded query samples, not exhaustive source coverage", "Metadata/abstract evidence only; no scientific validity or independent-cohort claim", "First occurrence per exact DOI or source ID retained; every partition variant and membership retained", "Known identifiers excluded; undisclosed cohort/source aliases remain unresolved"]}
    write_json(root / "ACQUISITION_RECEIPT.json", receipt)
    return receipt


def acquire(root, plan):
    resume_path = root / "resume_runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    resume = {"started_utc": now(), "plan_sha256": digest((root / "PREDECLARED_PLAN.json").read_bytes()), "partitions": []}
    for partition in plan["partitions"]:
        folder = root / "partitions" / partition["id"]
        if (folder / "receipt.json").exists():
            prior = read_json(folder / "receipt.json")
            if prior["status"] == "ACQUIRED_BOUNDED":
                if digest((folder / "records.jsonl").read_bytes()) != prior["records_sha256"]:
                    raise ValueError("Refusing reuse of changed partition")
                resume["partitions"].append({"id": partition["id"], "action": "REUSED", "records_sha256": prior["records_sha256"]})
                write_json(resume_path, resume)
                print(json.dumps({"partition": partition["id"], "action": "REUSED", "rows": prior["accepted_rows"]}), flush=True)
                continue
        print(json.dumps({"partition": partition["id"], "action": "ACQUIRING", "utc": now()}), flush=True)
        rows, receipt = acquire_rows(root, partition, plan["protected"])
        receipt.update({"completed_utc": now(), "source_commit": plan["source_commit"],
                        "plan_sha256": digest((root / "PREDECLARED_PLAN.json").read_bytes())})
        attempt = folder / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        write_jsonl(attempt / "records.jsonl", rows)
        receipt["records_sha256"] = digest((attempt / "records.jsonl").read_bytes())
        write_json(attempt / "receipt.json", receipt)
        if receipt["status"] == "ACQUIRED_BOUNDED":
            write_jsonl(folder / "records.jsonl", rows)
        write_json(folder / "receipt.json", receipt)
        resume["partitions"].append({"id": partition["id"], "action": receipt["status"], "records_sha256": receipt["records_sha256"]})
        write_json(resume_path, resume)
        current = summarize(root, plan)
        print(json.dumps({"partition": partition["id"], "status": receipt["status"], "rows": len(rows), "totals": current["counts"]}), flush=True)
    if protected_inputs() != plan["protected"]:
        raise ValueError("Protected inputs changed during acquisition")
    result = summarize(root, plan, complete=True)
    resume["completed_utc"] = now()
    write_json(resume_path, resume)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if all(result["target_checks"].values()) else 1


def strip_cache_status(value):
    if isinstance(value, dict):
        return {key: strip_cache_status(item) for key, item in value.items() if key != "cache_status"}
    if isinstance(value, list):
        return [strip_cache_status(item) for item in value]
    return value


def replay(root, plan):
    calls = []
    def denied(*args, **kwargs):
        calls.append("network boundary reached")
        raise AssertionError("Network denied during acquired-partition replay")
    requests.sessions.Session.request = denied
    socket.create_connection = denied
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    checks = []
    for partition in plan["partitions"]:
        folder = root / "partitions" / partition["id"]
        if not (folder / "receipt.json").exists() or read_json(folder / "receipt.json")["status"] != "ACQUIRED_BOUNDED":
            continue
        expected = read_rows(folder / "records.jsonl")
        actual, receipt = acquire_rows(root, partition, plan["protected"], offline=True)
        before = json_bytes(strip_cache_status(expected))
        after = json_bytes(strip_cache_status(actual))
        checks.append({"partition": partition["id"], "rows": len(actual), "equal": before == after,
                       "original_comparable_sha256": digest(before), "replay_comparable_sha256": digest(after),
                       "status": receipt["status"]})
    result = {"status": "PASS" if checks and all(item["equal"] and item["status"] == "ACQUIRED_BOUNDED" for item in checks) and not calls else "FAIL",
              "verified_utc": now(), "network_guard_calls": len(calls), "ignored_field": "cache_status only, recursively",
              "partitions": checks, "protected_inputs_unchanged": protected_inputs() == plan["protected"]}
    write_json(root / "OFFLINE_REPLAY.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "PASS" and result["protected_inputs_unchanged"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--mode", choices=("plan", "acquire", "replay"), default="plan")
    args = parser.parse_args()
    if args.root.resolve() != ROOT.resolve():
        parser.error("This bounded campaign writes only to its assigned expanded data root")
    with StageLease(args.root / "locks/acquisition.lock"):
        plan = prepare(args.root)
        if args.mode == "plan":
            print(json.dumps(plan, indent=2))
            return 0
        if args.mode == "acquire":
            return acquire(args.root, plan)
        return replay(args.root, plan)


if __name__ == "__main__":
    raise SystemExit(main())
