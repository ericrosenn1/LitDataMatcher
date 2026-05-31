"""Human review export and label-ingestion helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .schemas import MatchCandidate
from .storage import read_jsonl, write_jsonl


def match_review_rows(matches: Iterable[MatchCandidate]) -> list[dict]:
    """Flatten ranked matches into rows suitable for expert review."""

    rows: list[dict] = []
    for rank, match in enumerate(matches, 1):
        rows.append(
            {
                "rank": rank,
                "match_id": match.match_id,
                "question_id": match.question.question_id,
                "dataset_id": match.dataset.dataset_id,
                "score": match.score.combined,
                "question": match.question.question,
                "dataset_title": match.dataset.title,
                "dataset_source": match.dataset.source,
                "required_variables": "; ".join(match.question.required_variables),
                "missing_variables": "; ".join(match.missing_variables),
                "rationale": "; ".join(match.rationale),
                "recommended_design": match.assessments.get("feasibility", {}).get(
                    "recommended_design", ""
                ),
                "expert_relevance": "",
                "expert_notes": "",
            }
        )
    return rows


def export_review_csv(matches: Iterable[MatchCandidate], path: str | Path) -> None:
    """Write a CSV review sheet for domain experts."""

    rows = match_review_rows(matches)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "rank",
        "match_id",
        "question_id",
        "dataset_id",
        "score",
        "question",
        "dataset_title",
        "dataset_source",
        "required_variables",
        "missing_variables",
        "rationale",
        "recommended_design",
        "expert_relevance",
        "expert_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_review_jsonl(matches: Iterable[MatchCandidate], path: str | Path) -> None:
    """Write a JSONL review sheet for programmatic annotation tools."""

    write_jsonl(path, match_review_rows(matches))


def load_review_labels(path: str | Path) -> list[dict]:
    """Load labels from CSV or JSONL review files."""

    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_review_labels(rows: Iterable[dict]) -> dict:
    """Summarize expert labels for active-learning and reporting."""

    rows = list(rows)
    labeled = [row for row in rows if str(row.get("expert_relevance", "")).strip()]
    relevant = 0
    for row in labeled:
        try:
            relevant += 1 if float(row.get("expert_relevance", 0)) > 0 else 0
        except ValueError:
            relevant += 1 if str(row.get("expert_relevance", "")).lower() in {"yes", "true"} else 0
    return {
        "rows": len(rows),
        "labeled": len(labeled),
        "relevant": relevant,
        "label_coverage": round(len(labeled) / len(rows), 3) if rows else 0.0,
        "relevance_rate": round(relevant / len(labeled), 3) if labeled else 0.0,
    }
