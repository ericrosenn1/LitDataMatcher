"""Qualify bounded real omics metadata and exact offline replay; never raw measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from litdatamatcher.acquisition_v2 import SnapshotClient
from litdatamatcher.data_plane import atomic_json, digest
from litdatamatcher.omics_adapters import MetabolomicsWorkbenchDatasetAdapter, PRIDEDatasetAdapter
from litdatamatcher.v2 import write_rows

QUERIES = (("pride", "Alzheimer", PRIDEDatasetAdapter), ("metabolomicsworkbench", "Diabetes", MetabolomicsWorkbenchDatasetAdapter), ("metabolomicsworkbench", "ST000001", MetabolomicsWorkbenchDatasetAdapter))


class SnapshotBridge:
    def __init__(self, root, offline=False):
        self.client = SnapshotClient(root, offline=offline)
        self.last_response_metadata = {}
        self.manifests = []

    def get_json(self, url, params=None):
        data, meta = self.client.json(url, params)
        self.manifests.append(meta)
        self.last_response_metadata = {"cache_path": meta["object_path"], "cache_content_sha256": meta["sha256"], "retrieval_time_utc": meta["retrieved_at"], "cache_status": "immutable_source_snapshot"}
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / "QUALIFICATION.json").exists():
        raise ValueError("Qualification receipt already exists; preserve it and choose a new derivative output root")
    root.mkdir(parents=True, exist_ok=True)
    protocol = {"schema_version": "omics_qualification_v1", "created_at": datetime.now(timezone.utc).isoformat(), "queries": [{"source": name, "query": query} for name, query, _ in QUERIES], "scope": "PRIDE first 25 projects; Metabolomics Workbench first 100 study summaries plus one preserved accession. No census, raw files, independent-donor counts, or biological validation claim."}
    atomic_json(root / "PROTOCOL.json", protocol)
    connected = SnapshotBridge(root)
    rows, partitions = [], []
    for name, query, cls in QUERIES:
        records = [item.to_dict() for item in cls(connected).search(query)]
        if not records:
            raise ValueError(f"No qualifying metadata for {name}/{query}")
        rows.extend(records)
        partitions.append({"source": name, "query": query, "record_count": len(records), "records_digest": digest(records)})
    by_id = {row["dataset_id"]: row for row in rows}
    normalized = [by_id[key] for key in sorted(by_id)]
    write_rows(root / "studies.jsonl", normalized)
    blocked = []

    def deny(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            blocked.append(event)
            raise PermissionError("Omics replay denies Python socket/DNS networking")

    sys.addaudithook(deny)
    try:
        with socket.socket() as probe:
            probe.connect(("127.0.0.1", 9))
    except PermissionError:
        pass
    else:
        raise RuntimeError("Offline denial probe failed")
    probe_count = len(blocked)
    offline = SnapshotBridge(root, offline=True)
    for (_, query, cls), partition in zip(QUERIES, partitions, strict=True):
        replay = [item.to_dict() for item in cls(offline).search(query)]
        if digest(replay) != partition["records_digest"]:
            raise ValueError("Offline normalized replay differs")
    output = root / "studies.jsonl"
    receipt = {"schema_version": "omics_qualification_v1", "status": "PASS", "timestamp": datetime.now(timezone.utc).isoformat(), "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "protocol_sha256": hashlib.sha256((root / "PROTOCOL.json").read_bytes()).hexdigest(), "partitions": partitions, "unique_study_ids": len(normalized), "independent_study_groups": None, "source_snapshots": connected.manifests, "offline_byte_identity": True, "network_control": {"kind": "Python audit hook", "blocked_probe_count": probe_count, "unexpected_requests": len(blocked) - probe_count}, "artifact": {"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "size_bytes": output.stat().st_size}, "scientific_status": "METADATA_COMPATIBILITY_ONLY; raw measurements, donor identity and independence are unvalidated"}
    if receipt["network_control"]["unexpected_requests"]:
        raise ValueError("Offline replay attempted networking")
    atomic_json(root / "QUALIFICATION.json", receipt)
    print(json.dumps({"status": "PASS", "unique_study_ids": len(normalized), "out": str(root)}, sort_keys=True))


if __name__ == "__main__":
    main()
