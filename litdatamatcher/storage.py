"""SQLite persistence for reproducible LitDataMatcher runs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from .schemas import DatasetRecord, EvidenceSynthesis, MatchCandidate, QuestionCandidate


SCHEMA_VERSION = 1


class PipelineStore:
    """Small SQLite repository used by local and publication workflows."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.initialize()

    def initialize(self) -> None:
        """Create tables if they do not already exist."""

        # WAL improves local read/write behavior without changing the database schema.
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS questions (
                question_id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS syntheses (
                cluster_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS matches (
                match_id TEXT PRIMARY KEY,
                question_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                combined_score REAL NOT NULL,
                payload TEXT NOT NULL
            );
            """
        )
        # Store a schema version so future migrations can be explicit.
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        self.conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        self.conn.close()

    def reset_run_tables(self) -> None:
        """Clear per-run tables before writing a fresh pipeline run."""

        self.conn.executescript(
            """
            DELETE FROM matches;
            DELETE FROM syntheses;
            DELETE FROM datasets;
            DELETE FROM questions;
            """
        )
        self.conn.commit()

    def store_questions(self, questions: Iterable[QuestionCandidate]) -> None:
        """Upsert question candidates."""

        self.conn.executemany(
            """
            INSERT OR REPLACE INTO questions(question_id, question, payload)
            VALUES(?, ?, ?)
            """,
            [
                # Full JSON payloads preserve nested evidence while columns support lookup.
                (question.question_id, question.question, json.dumps(question.to_dict(), sort_keys=True))
                for question in questions
            ],
        )
        self.conn.commit()

    def store_datasets(self, datasets: Iterable[DatasetRecord]) -> None:
        """Upsert dataset records."""

        self.conn.executemany(
            """
            INSERT OR REPLACE INTO datasets(dataset_id, title, source, payload)
            VALUES(?, ?, ?, ?)
            """,
            [
                (
                    dataset.dataset_id,
                    dataset.title,
                    dataset.source,
                    json.dumps(dataset.to_dict(), sort_keys=True),
                )
                for dataset in datasets
            ],
        )
        self.conn.commit()

    def store_syntheses(self, syntheses: Iterable[EvidenceSynthesis]) -> None:
        """Upsert evidence-synthesis records."""

        self.conn.executemany(
            """
            INSERT OR REPLACE INTO syntheses(cluster_id, payload)
            VALUES(?, ?)
            """,
            [
                (synthesis.cluster_id, json.dumps(synthesis.to_dict(), sort_keys=True))
                for synthesis in syntheses
            ],
        )
        self.conn.commit()

    def store_matches(self, matches: Iterable[MatchCandidate]) -> None:
        """Upsert ranked match records."""

        self.conn.executemany(
            """
            INSERT OR REPLACE INTO matches(
                match_id, question_id, dataset_id, combined_score, payload
            )
            VALUES(?, ?, ?, ?, ?)
            """,
            [
                (
                    match.match_id,
                    match.question.question_id,
                    match.dataset.dataset_id,
                    match.score.combined,
                    json.dumps(match.to_dict(), sort_keys=True),
                )
                for match in matches
            ],
        )
        self.conn.commit()


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Write dictionaries to a UTF-8 JSONL file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            # Sorted keys keep JSONL artifacts stable across reproducible runs.
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    """Read dictionaries from a UTF-8 JSONL file."""

    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                # Include the row number so malformed catalogs/reports are easy to fix.
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows
