"""Checkout convenience wrapper; installed use: python -m litdatamatcher.final_campaign_acceptance."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from litdatamatcher.final_campaign_acceptance import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
