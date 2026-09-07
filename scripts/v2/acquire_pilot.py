"""Execute independent source syncs; all bulk outputs stay in --root."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from litdatamatcher.acquisition_v2 import (
    audit_offline_recovery,
    run_numeric_alignment,
    sync_datasets,
    sync_literature,
    sync_targeted_studies,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=["literature", "datasets", "targeted", "integrate", "audit-offline", "all"], default="all")
    parser.add_argument("--accession", action="append", default=[], help="GEO series; repeat for multiple explicit targets")
    parser.add_argument("--expanded", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.offline and args.refresh:
        parser.error("offline and refresh are mutually exclusive")
    statuses = []
    if args.stage == "targeted":
        if not args.accession:
            parser.error("targeted stage requires --accession")
        _, report = sync_targeted_studies(args.root, args.accession, args.offline, args.refresh)
        print(json.dumps({k: v for k, v in report.items() if k not in ("closure", "events")}, indent=2))
        statuses.append(report["status"])
    if args.stage == "audit-offline":
        report = audit_offline_recovery(args.root)
        print(json.dumps(report, indent=2))
        statuses.append(report["status"])
    if args.stage in ("literature", "all"):
        _, report = sync_literature(args.root, 200 if args.expanded else 50, 50 if args.expanded else 20, args.offline, args.refresh)
        print(json.dumps({k: v for k, v in report.items() if k not in ("searches", "events", "failures")}, indent=2))
        statuses.append(report["status"])
    if args.stage == "integrate":
        report = run_numeric_alignment(args.root)
        print(json.dumps(report, indent=2))
        statuses.append(report["status"])
    if args.stage in ("datasets", "all"):
        _, report = sync_datasets(args.root, 100 if args.expanded else 50, 30 if args.expanded else 20, args.offline, args.refresh)
        print(json.dumps({k: v for k, v in report.items() if k not in ("searches", "events", "failures")}, indent=2))
        statuses.append(report["status"])
    return 0 if all(x == "PASS" for x in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
