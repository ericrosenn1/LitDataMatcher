"""Create a read-only corpus integrity receipt before a research experiment."""

from __future__ import annotations

import argparse
import json

from litdatamatcher.corpus_snapshot import qualify_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--verify-raw", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--max-shards", type=int)
    args = parser.parse_args()
    receipt = qualify_corpus(
        args.root,
        args.out,
        workers=args.workers,
        verify_raw=args.verify_raw,
        include_updates=not args.baseline_only,
        max_shards=args.max_shards,
    )
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "status",
                    "shards_audited",
                    "rows",
                    "failed_shards",
                    "snapshot_sha256",
                    "elapsed_seconds",
                )
            },
            indent=2,
        )
    )
    return 1 if receipt["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
