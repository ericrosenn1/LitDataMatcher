"""Small daily review-queue exports for recurring expert scoring."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from .review import match_review_records, match_review_rows
from .schemas import MatchCandidate
from .storage import read_jsonl, write_jsonl


def write_daily_review_queue(
    run_dir: str | Path,
    out_path: str | Path,
    limit: int = 5,
    reviewer_id: str = "",
    review_date: str | None = None,
) -> dict:
    """Write a compact top-N review queue from a completed run directory."""

    run_dir = Path(run_dir)
    out_path = Path(out_path)
    review_date = review_date or date.today().isoformat()
    matches = [
        MatchCandidate.from_dict(row)
        for row in read_jsonl(run_dir / "matches.jsonl")
    ][: max(0, int(limit))]
    rows = match_review_rows(matches)
    for row in rows:
        row["reviewer_id"] = reviewer_id
        row["review_date"] = review_date
        row["review_status"] = "pending"
    records = match_review_records(matches)
    for record in records:
        record["reviewer_id"] = reviewer_id
        record["review_date"] = review_date
        record["review_status"] = "pending"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".jsonl":
        write_jsonl(out_path, records)
    else:
        fieldnames = [*rows[0].keys()] if rows else ["reviewer_id", "review_date", "review_status"]
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return {
        "run_dir": str(run_dir),
        "out": str(out_path),
        "review_date": review_date,
        "reviewer_id": reviewer_id,
        "queued_matches": len(rows),
    }
