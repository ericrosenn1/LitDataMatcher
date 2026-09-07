"""Content-addressed artifacts and a versioned, dependency-aware local catalog.

Source snapshots are immutable. Scientific identities are independent of content
hashes. SQLite owns normalized versions and invalidation, not raw source bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def atomic_write(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path: str | Path, value: Any) -> None:
    atomic_write(path, canonical_json(value) + b"\n")


class Catalog:
    """One shared catalog; serialized transactions and immutable source versions."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.root / "catalog.sqlite3", timeout=20)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 2):
            raise ValueError(f"Unsupported catalog schema {version}; explicit migration required")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS versions (
          kind TEXT NOT NULL, id TEXT NOT NULL, digest TEXT NOT NULL, payload TEXT NOT NULL,
          PRIMARY KEY(kind,id,digest));
        CREATE TABLE IF NOT EXISTS current (
          kind TEXT NOT NULL, id TEXT NOT NULL, digest TEXT NOT NULL, valid INTEGER NOT NULL,
          PRIMARY KEY(kind,id));
        CREATE TABLE IF NOT EXISTS dependencies (
          parent_kind TEXT NOT NULL,parent_id TEXT NOT NULL,child_kind TEXT NOT NULL,
          child_id TEXT NOT NULL,PRIMARY KEY(parent_kind,parent_id,child_kind,child_id));
        CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(kind UNINDEXED,id UNINDEXED,text);
        PRAGMA user_version=2;
        """)

    def close(self):
        self.conn.close()

    def snapshot(self, content: bytes, metadata: dict) -> dict:
        sha = hashlib.sha256(content).hexdigest()
        path = self.root / "sources" / sha[:2] / sha
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError(f"Corrupt immutable artifact: {path}")
        else:
            atomic_write(path, content)
        result = dict(metadata, sha256=sha, size_bytes=len(content), path=str(path))
        atomic_json(self.root / "source_manifests" / (digest(result) + ".json"), result)
        return result

    def upsert(self, kind: str, identity: str, payload: dict,
               parents: list[tuple[str, str]] = (), search_text: str = "") -> bool:
        if not kind or not identity:
            raise ValueError("Record kind and stable identity are required")
        encoded = canonical_json(payload).decode()
        sha = digest(payload)
        with self.conn:
            self.conn.execute("BEGIN IMMEDIATE")
            old = self.conn.execute("SELECT digest,valid FROM current WHERE kind=? AND id=?",
                                    (kind, identity)).fetchone()
            old_parents = set(map(tuple,self.conn.execute(
                'SELECT parent_kind,parent_id FROM dependencies WHERE child_kind=? AND child_id=?',
                (kind,identity)).fetchall()))
            # Parent links are part of derivation even when output content is unchanged.
            for pk, pi in parents:
                row = self.conn.execute("SELECT valid FROM current WHERE kind=? AND id=?", (pk, pi)).fetchone()
                if row is None or not row[0]:
                    raise ValueError(f"Missing or invalid parent {pk}/{pi}")
                if (pk, pi) == (kind, identity):
                    raise ValueError("Self dependency")
                ancestors=self.conn.execute('''WITH RECURSIVE ancestors(k,i) AS (
                  SELECT parent_kind,parent_id FROM dependencies WHERE child_kind=? AND child_id=?
                  UNION SELECT d.parent_kind,d.parent_id FROM dependencies d JOIN ancestors a
                    ON d.child_kind=a.k AND d.child_id=a.i) SELECT 1 FROM ancestors WHERE k=? AND i=?''',
                  (pk,pi,kind,identity)).fetchone()
                if ancestors: raise ValueError('Dependency cycle')
            changed = old is None or old[0] != sha or old_parents != set(parents)
            if changed:
                descendants = self.conn.execute("""WITH RECURSIVE children(k,i) AS (
                 SELECT child_kind,child_id FROM dependencies WHERE parent_kind=? AND parent_id=?
                 UNION SELECT d.child_kind,d.child_id FROM dependencies d JOIN children c
                   ON d.parent_kind=c.k AND d.parent_id=c.i) SELECT k,i FROM children""",
                                                (kind, identity)).fetchall()
                self.conn.executemany("UPDATE current SET valid=0 WHERE kind=? AND id=?", descendants)
            self.conn.execute("INSERT OR IGNORE INTO versions VALUES(?,?,?,?)", (kind, identity, sha, encoded))
            self.conn.execute("INSERT OR REPLACE INTO current VALUES(?,?,?,1)", (kind, identity, sha))
            self.conn.execute("DELETE FROM dependencies WHERE child_kind=? AND child_id=?", (kind, identity))
            self.conn.executemany("INSERT INTO dependencies VALUES(?,?,?,?)",
                                  [(pk, pi, kind, identity) for pk, pi in parents])
            self.conn.execute("DELETE FROM search WHERE kind=? AND id=?", (kind, identity))
            self.conn.execute("INSERT INTO search VALUES(?,?,?)", (kind, identity, search_text))
        return changed

    def records(self, kind: str, include_invalid: bool = False) -> list[dict]:
        rows = self.conn.execute("""SELECT v.payload FROM versions v JOIN current c
          ON v.kind=c.kind AND v.id=c.id AND v.digest=c.digest
          WHERE c.kind=? AND (c.valid=1 OR ?) ORDER BY c.id""", (kind, include_invalid))
        return [json.loads(row[0]) for row in rows]

    def search(self, kind: str, query: str, limit: int = 50) -> list[str]:
        # Quote user tokens; FTS syntax is never interpreted as a user program.
        tokens = query.split()[:32]
        safe = " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)
        if not safe:
            return []
        rows = self.conn.execute("""SELECT search.id FROM search JOIN current c
          ON c.id=search.id AND c.kind=search.kind WHERE search MATCH ?
          AND search.kind=? AND c.valid=1 ORDER BY rank LIMIT ?""", (safe, kind, min(max(limit, 1), 1000)))
        return [row[0] for row in rows]
