"""Build additive PubMed revision sidecars; defaults to two bounded workers."""

import argparse
import json
from pathlib import Path

from litdatamatcher.pubmed_revision_ledger import build_revision_ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--executor", choices=("thread", "process"), default="thread")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument(
        "--shard", action="append", help="Exact shard filename; any explicit subset is PARTIAL"
    )
    args = parser.parse_args()
    result = build_revision_ledger(
        args.corpus_root,
        args.snapshot,
        args.out,
        workers=args.workers,
        max_shards=args.max_shards,
        shard_names=args.shard,
        executor=args.executor,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "shards"}, indent=2))
    return 1 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
