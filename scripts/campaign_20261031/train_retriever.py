"""Run the predeclared two-epoch local source-link retrieval experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from litdatamatcher.retrieval_training import run_training  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = run_training(
        args.benchmark,
        args.model,
        args.out,
        device=args.device,
        resume=args.resume,
        progress=lambda event: print(json.dumps(event), flush=True),
    )
    print(
        json.dumps(
            {"status": result["status"], "selection": result["selection"], "out": str(args.out)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
