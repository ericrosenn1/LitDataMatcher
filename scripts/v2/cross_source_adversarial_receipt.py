from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Local imports follow the standalone script path bootstrap.
from litdatamatcher.cross_source_adversarial import build_cross_source_receipt  # noqa: E402
from litdatamatcher.data_plane import atomic_json  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
args = parser.parse_args()
atomic_json(args.out, build_cross_source_receipt())
