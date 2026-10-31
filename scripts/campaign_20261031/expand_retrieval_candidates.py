"""Build frozen larger pools without model inference or reserved-question reads."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from litdatamatcher.retrieval_expansion import build_expansion  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--edges", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, default=[50000, 100000])
    parser.add_argument(
        "--views", nargs="+", choices=["full", "abstract_only"], default=["full", "abstract_only"]
    )
    parser.add_argument("--seed", default="retrieval-scale-20261031")
    args = parser.parse_args()
    result = build_expansion(
        args.corpus_root,
        args.snapshot,
        args.edges,
        args.benchmark,
        args.out,
        sizes=args.sizes,
        views=args.views,
        seed=args.seed,
        progress=lambda event: print(json.dumps(event), flush=True),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
