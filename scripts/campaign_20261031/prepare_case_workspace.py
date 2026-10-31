"""Copy verified discovery catalogs into a new native analysis workspace.

No source records are changed. An empty inspection table explicitly records
that no processed measurements were qualified for these catalog records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source: Path, out: Path):
    report_path = source / "DISCOVERY_MANIFEST.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") not in {"PASS", "PASS_WITH_SOURCE_FAILURES"}:
        raise ValueError("A completed, inspectable discovery snapshot is required")
    if out.exists():
        raise ValueError("Use a new output directory; existing workspaces are preserved")
    receipts = {item["path"]: item for item in report["outputs"]}
    content = {}
    for name in ("literature.jsonl", "studies.jsonl"):
        relative = "catalog/" + name
        path = source / relative
        if checksum(path) != receipts[relative]["sha256"]:
            raise ValueError(f"Discovery catalog integrity failure: {name}")
        content[relative] = path.read_bytes()
    content["catalog/processed_inspections.jsonl"] = b""
    (out / "catalog").mkdir(parents=True)
    for relative, data in content.items():
        (out / relative).write_bytes(data)
    receipt = {
        "schema_version": "case-workspace-1.0",
        "status": "PASS",
        "parent_discovery_manifest": str(report_path.resolve()),
        "parent_discovery_sha256": checksum(report_path),
        "preparer_sha256": checksum(__file__),
        "processed_measurements_inspected": 0,
        "source_failures_retained": report["failed_source_jobs"],
        "outputs": [
            {"path": name, "sha256": checksum(out / name), "bytes": len(data)}
            for name, data in content.items()
        ],
    }
    (out / "WORKSPACE_MANIFEST.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.discovery, args.out), indent=2))


if __name__ == "__main__":
    main()
