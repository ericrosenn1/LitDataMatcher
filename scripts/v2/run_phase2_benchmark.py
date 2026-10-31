"""Execute the predeclared real-metadata benchmark without live retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from litdatamatcher.phase2_benchmark import run_benchmark  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--literature", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--acquisition-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--protocol", default=str(ROOT / "docs/v2/final_audit/PHASE2_BENCHMARK_PROTOCOL.md")
    )
    parser.add_argument("--model-dir")
    parser.add_argument("--dataset-supplement", action="append", default=[])
    parser.add_argument("--supplement-receipt", action="append", default=[])
    args = parser.parse_args()
    if len(args.dataset_supplement) != len(args.supplement_receipt):
        parser.error("Each dataset supplement needs its own acquisition receipt")
    receipt = run_benchmark(
        args.literature,
        args.datasets,
        args.acquisition_receipt,
        args.output,
        args.protocol,
        args.model_dir,
        list(zip(args.dataset_supplement, args.supplement_receipt, strict=True)),
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "engineering": receipt["engineering_status"],
                "metadata_evaluation": receipt["metadata_evaluation_status"],
                "inputs": receipt["input_files"],
                "output": args.output,
            },
            indent=2,
        )
    )
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
