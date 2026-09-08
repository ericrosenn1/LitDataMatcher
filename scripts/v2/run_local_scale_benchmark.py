"""Run the bounded V2.5 local benchmark and write a receipt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from litdatamatcher.data_plane import atomic_json
from litdatamatcher.scale_benchmark import run_local_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()
    atomic_json(args.out, run_local_benchmark(args.root, args.count))


if __name__ == "__main__":
    main()
