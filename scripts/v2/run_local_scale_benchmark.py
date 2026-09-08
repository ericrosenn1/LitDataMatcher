"""Run the bounded V2.5 local benchmark and write a receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from litdatamatcher.data_plane import atomic_json
from litdatamatcher.scale_benchmark import compare_benchmark_baseline, run_local_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--baseline", default=None)
    args = parser.parse_args()
    receipt = run_local_benchmark(args.root, args.count)
    if args.baseline:
        receipt["baseline_comparison"] = compare_benchmark_baseline(
            receipt, json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        )
    atomic_json(args.out, receipt)


if __name__ == "__main__":
    main()
