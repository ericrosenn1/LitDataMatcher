"""Evaluate fixed 50k/100k retrieval pools using two local models; no training."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from litdatamatcher.retrieval_scale_evaluation import evaluate_scale  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = evaluate_scale(
        args.root,
        args.pretrained,
        args.trained,
        args.out,
        device=args.device,
        resume=args.resume,
        progress=lambda event: print(json.dumps(event), flush=True),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "fingerprint": result["fingerprint"],
                "conditions": list(result["conditions"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
