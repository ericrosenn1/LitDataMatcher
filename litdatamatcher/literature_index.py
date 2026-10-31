"""Resume-safe current PubMed materialization and source-bound local retrieval.

The existing revision ledger remains the historical authority. This derivative
keeps active versions and an FTS5 current view in SQLite; a complete shard and
its checkpoint commit together. It never changes the acquired corpus or ledger.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

from .controller import OwnershipConflict, integration_lease
from .corpus_snapshot import PUBMED_FIELDS, file_digest
from .data_plane import atomic_json, digest
from .literature_integrity import consolidate_literature_rows
from .pubmed_revision_ledger import LEDGER_VERSION, _inside, _is_inside, _resolved_path

INDEX_VERSION = "pubmed-current-index-1.0"
SELECTION_RULE = "highest_numeric_version_then_latest_event_within_version"
_IMPLEMENTATION_SHA256 = file_digest(Path(__file__))

_DDL = """
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS shards(
  shard_order INTEGER PRIMARY KEY, identity TEXT NOT NULL, receipt TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS versions(
  id INTEGER PRIMARY KEY, pmid TEXT NOT NULL, version_key TEXT NOT NULL,
  version INTEGER, title TEXT NOT NULL, abstract TEXT NOT NULL,
  metadata_json TEXT NOT NULL, provenance_json TEXT NOT NULL,
  UNIQUE(pmid, version_key));
CREATE INDEX IF NOT EXISTS version_selection ON versions(pmid, version DESC);
CREATE INDEX IF NOT EXISTS unknown_versions ON versions(pmid) WHERE version IS NULL;
CREATE TABLE IF NOT EXISTS unresolved_deletions(
  pmid TEXT NOT NULL, position TEXT NOT NULL, PRIMARY KEY(pmid, position));
CREATE TABLE IF NOT EXISTS current_records(
  pmid TEXT UNIQUE NOT NULL, version_id INTEGER NOT NULL REFERENCES versions(id));
CREATE INDEX IF NOT EXISTS current_version_reference ON current_records(version_id);
CREATE VIEW IF NOT EXISTS search_content AS
  SELECT c.rowid AS rowid, v.title, v.abstract
  FROM current_records c JOIN versions v ON v.id=c.version_id;
CREATE VIRTUAL TABLE IF NOT EXISTS literature_fts USING fts5(
  title, abstract, content='search_content', content_rowid='rowid', tokenize='unicode61');
CREATE TRIGGER IF NOT EXISTS current_delete BEFORE DELETE ON current_records BEGIN
  INSERT INTO literature_fts(literature_fts, rowid, title, abstract)
  SELECT 'delete', old.rowid, v.title, v.abstract FROM versions v WHERE v.id=old.version_id;
END;
CREATE TRIGGER IF NOT EXISTS current_insert AFTER INSERT ON current_records BEGIN
  INSERT INTO literature_fts(rowid, title, abstract)
  SELECT new.rowid, v.title, v.abstract FROM versions v WHERE v.id=new.version_id;
END;
"""


def _json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _positive_integer(value, name, maximum=None):
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        raise ValueError(
            f"{name} must be a positive integer" + (f" <= {maximum}" if maximum else "")
        )


def _shard_identity(shard):
    return digest(
        {
            key: shard[key]
            for key in (
                "source",
                "source_shard",
                "shard_order",
                "events",
                "upserts",
                "deletions",
                "sidecar_sha256",
                "legacy",
                "legacy_relative_path",
                "pmid_sequence_sha256",
            )
        }
    )


def _manifest(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("ledger_version") != LEDGER_VERSION or value.get("status") not in {
        "PASS",
        "PARTIAL",
    }:
        raise ValueError("A successful versioned revision-ledger manifest is required")
    shards = value.get("shards", [])
    if not shards:
        raise ValueError("Ledger manifest has no verified shards")
    orders, identities = set(), set()
    for row in shards:
        order = row.get("shard_order")
        identity = (row.get("source"), row.get("source_shard"))
        if (
            type(order) is not int
            or order < 0
            or order in orders
            or identity in identities
            or row.get("status") != "PASS"
            or row.get("sequence_verified") is not True
        ):
            raise ValueError("Ledger shards must have unique ordered, verified identities")
        if row["source"] not in {"baseline", "updates"} or not re.fullmatch(
            r"pubmed[0-9]{2}n[0-9]+\.xml\.gz", row["source_shard"]
        ):
            raise ValueError("Invalid PubMed source shard")
        orders.add(order)
        identities.add(identity)
        _shard_identity(row)
    value["shards"] = sorted(shards, key=lambda row: row["shard_order"])
    if value["status"] == "PASS" and (
        len(shards) != value.get("available_shards") or sorted(orders) != list(range(len(shards)))
    ):
        raise ValueError("A complete ledger must cover every consecutive shard position")
    return value


class LiteratureIndex:
    """A local current-literature index. Open a separate connection per thread."""

    def __init__(self, path, *, read_only=False):
        self.path = Path(path).resolve()
        self.read_only = read_only
        if read_only:
            self.conn = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=30)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA cache_size=-65536")
        self.conn.execute("PRAGMA temp_store=FILE")
        self.conn.execute("PRAGMA foreign_keys=ON")
        if not read_only:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=FULL")
            self.conn.executescript(_DDL)
            version = self._metadata("version")
            if version is None:
                self._set_metadata("version", INDEX_VERSION)
                self.conn.commit()
        if self._metadata("version") != INDEX_VERSION:
            self.close()
            raise ValueError("Unsupported literature index version")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.conn.close()

    def _metadata(self, key, default=None):
        row = self.conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def _set_metadata(self, key, value):
        self.conn.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (key, _json(value)))

    def snapshot_identity(self, *, applied_shards=None):
        """Pin a committed corpus prefix without hashing the large database.

        Each stored shard identity binds normalized/source-event checksums and
        ordering. Reading the prefix in one transaction gives an immutable
        scientific-input identity even while another connection builds ahead.
        """
        if applied_shards is not None and (type(applied_shards) is not int or applied_shards < 0):
            raise ValueError("applied_shards must be a nonnegative integer")
        own_transaction = not self.conn.in_transaction
        if own_transaction:
            self.conn.execute("BEGIN")
        try:
            shards = [
                list(row)
                for row in self.conn.execute(
                    "SELECT shard_order,identity FROM shards ORDER BY shard_order LIMIT ?",
                    (-1 if applied_shards is None else applied_shards,),
                )
            ]
            if applied_shards is not None and len(shards) != applied_shards:
                raise ValueError("Requested snapshot prefix has not been materialized")
            identity = {
                "schema_version": INDEX_VERSION,
                "corpus_store_uuid": self._metadata("corpus_store_uuid"),
                "selection_rule": SELECTION_RULE,
                "applied_shards": shards,
            }
            return {
                "snapshot_sha256": digest(identity),
                "schema_version": INDEX_VERSION,
                "corpus_store_uuid": identity["corpus_store_uuid"],
                "applied_shards": len(shards),
                "last_shard_order": shards[-1][0] if shards else None,
                "status": self._metadata("status", "EMPTY"),
            }
        finally:
            if own_transaction:
                self.conn.rollback()

    def status(self, *, include_counts=True):
        """Return scope and optional exact counts; omit counts for fast polling.

        ``include_counts=False`` never scans record tables. A persisted current
        count is used when available; older in-progress builders report null
        rather than presenting an outdated count as current.
        """
        current_count = self._metadata("current_records_count")
        if current_count is None and include_counts:
            current_count = self.conn.execute("SELECT count(*) FROM current_records").fetchone()[0]
        return {
            "schema_version": INDEX_VERSION,
            "selection_rule": SELECTION_RULE,
            "status": self._metadata("status", "EMPTY"),
            "applied_shards": self.conn.execute("SELECT count(*) FROM shards").fetchone()[0],
            "available_shards": self._metadata("available_shards", 0),
            "current_records": current_count,
            "active_versions": self.conn.execute("SELECT count(*) FROM versions").fetchone()[0]
            if include_counts
            else None,
            "unresolved_pmids": self.conn.execute(
                "SELECT count(*) FROM (SELECT pmid FROM versions WHERE version IS NULL "
                "UNION SELECT pmid FROM unresolved_deletions)"
            ).fetchone()[0]
            if include_counts
            else None,
            "counts_included": include_counts,
            "historical_events_applied": self._metadata("events_applied", 0),
            "corpus_store_uuid": self._metadata("corpus_store_uuid"),
            "last_shard_order": self.conn.execute("SELECT max(shard_order) FROM shards").fetchone()[
                0
            ],
            "scope_note": "Current only within the applied ledger scope; unknown citation versions are withheld.",
        }

    @staticmethod
    def _record(row):
        record = json.loads(row["metadata_json"])
        record.update(
            document_id=f"PMID:{row['pmid']}",
            pmid=row["pmid"],
            title=row["title"],
            abstract=row["abstract"],
            text=row["abstract"],
            version=row["version"],
            source="pubmed",
            source_id=row["pmid"],
            source_locator=f"https://pubmed.ncbi.nlm.nih.gov/{row['pmid']}/",
            source_provenance=json.loads(row["provenance_json"]),
        )
        record["url"] = record["source_locator"]
        record["source_provenance"]["source_type"] = "pubmed"
        # Mirror the owning PubMed adapter's explicit publication-type policy.
        # A current citation is not necessarily unretracted scientific evidence.
        relations = {}
        try:
            publication_types = json.loads(record.get("publication_types_json") or "[]")
            if not isinstance(publication_types, list) or any(
                not isinstance(value, str) for value in publication_types
            ):
                raise ValueError("Invalid publication-type metadata")
        except (ValueError, TypeError):
            publication_types = []
            relations["lifecycle_review"] = {
                "reason": "Malformed retained publication-type metadata"
            }
        for value in publication_types:
            kind = re.sub(r"[^a-z]", "", value.casefold())
            if kind in {"retractedpublication", "retractionofpublication"}:
                relations.setdefault("retracted_publication_type", []).append(value)
            elif kind in {"publishederrata", "publishederratum", "correctedandrepublishedarticle"}:
                relations.setdefault("correction_publication_type", []).append(value)
            elif kind == "expressionofconcern":
                relations.setdefault("lifecycle_review", []).append(value)
        record["version_relationships"] = relations
        record["publication_types"] = publication_types
        record["lifecycle_coverage"] = (
            "Retained publication types only; this normalized snapshot does not include comment/correction links."
        )
        record = consolidate_literature_rows([record])[0]
        record["input_sha256"] = digest(
            {
                key: record[key]
                for key in ("pmid", "version", "title", "abstract", "source_provenance")
            }
        )
        return record

    def get(self, pmid):
        pmid = str(pmid).removeprefix("PMID:")
        row = self.conn.execute(
            "SELECT v.* FROM current_records c JOIN versions v ON v.id=c.version_id WHERE c.pmid=?",
            (pmid,),
        ).fetchone()
        return self._record(row) if row else None

    def search(self, query, limit=20):
        """Literal-token FTS search; user text is never interpreted as FTS syntax."""
        _positive_integer(limit, "limit", 1000)
        if not isinstance(query, str) or len(query) > 4000:
            raise ValueError("Query must be text of at most 4000 characters")
        terms = list(dict.fromkeys(re.findall(r"[^\W_]+", query, re.UNICODE)))[:64]
        if not terms:
            return []
        expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        own_transaction = not self.conn.in_transaction
        if own_transaction:
            self.conn.execute("BEGIN")
        try:
            # SQLite's rank stream avoids loading/sorting full source payloads
            # for every hit. Consume every boundary tie before PMID ordering,
            # preserving the exact previous BM25 and deterministic tie policy.
            # https://www.sqlite.org/fts5.html#sorting_by_auxiliary_function_results
            cursor = self.conn.execute(
                "SELECT c.pmid,c.version_id,literature_fts.rank AS score FROM literature_fts "
                "JOIN current_records c ON c.rowid=literature_fts.rowid "
                "WHERE literature_fts MATCH ? AND literature_fts.rank MATCH 'bm25(2.0,1.0)' "
                "ORDER BY literature_fts.rank",
                (expression,),
            )
            selected = []
            try:
                for row in cursor:
                    item = (float(row["score"]), row["pmid"], row["version_id"])
                    if len(selected) == limit and item[0] > selected[-1][0]:
                        break
                    if len(selected) < limit or item < selected[-1]:
                        bisect.insort(selected, item)
                        del selected[limit:]
            finally:
                cursor.close()
            if not selected:
                return []
            placeholders = ",".join("?" for _ in selected)
            payloads = {
                row["id"]: row
                for row in self.conn.execute(
                    f"SELECT * FROM versions WHERE id IN ({placeholders})",
                    [item[2] for item in selected],
                )
            }
            scope = self._metadata("status", "EMPTY")
            return [
                dict(
                    self._record(payloads[version_id]),
                    score=-score,
                    score_kind="fts5_bm25_not_probability",
                    retrieval_scope=scope,
                )
                for score, _pmid, version_id in selected
            ]
        finally:
            if own_transaction:
                self.conn.rollback()

    def passages(self, pmid):
        """Return exact source spans with stable content-addressed passage IDs."""
        record = self.get(pmid)
        if record is None:
            return []
        result = []
        for field in ("title", "abstract"):
            text = record[field]
            for match in re.finditer(r"\S(?:.*?)(?:[.!?](?=\s|$)|$)", text, re.DOTALL):
                start, end = match.span()
                result.append(
                    {
                        "passage_id": f"{record['document_id']}:{field}:{start}-{end}:{digest(text)[:16]}",
                        "document_id": record["document_id"],
                        "pmid": record["pmid"],
                        "version": record["version"],
                        "field": field,
                        "start": start,
                        "end": end,
                        "text": text[start:end],
                        "source_locator": record["source_locator"] + f"#{field}:{start}-{end}",
                        "source_provenance": record["source_provenance"],
                        "input_sha256": record["input_sha256"],
                    }
                )
        return result

    def semantic_search(
        self, query, model_dir, cache_root, *, limit=20, candidate_limit=200, device="cpu"
    ):
        """Rerank a bounded lexical shortlist with the existing fingerprinted model.

        This is not exhaustive dense corpus search; that retrieval scope is
        explicit in every returned row. No model is downloaded by this method.
        """
        from .retrieval import CachedSemanticIndex

        _positive_integer(candidate_limit, "candidate_limit", 1000)
        _positive_integer(limit, "limit", candidate_limit)
        candidates = self.search(query, candidate_limit)
        if not candidates:
            return []
        model = CachedSemanticIndex(model_dir, cache_root, device=device)
        model.fit(
            [
                {"id": row["document_id"], "text": row["title"] + "\n" + row["abstract"]}
                for row in candidates
            ]
        )
        by_id = {row["document_id"]: row for row in candidates}
        return [
            dict(
                by_id[hit["id"]],
                semantic_score=hit["score"],
                semantic_retrieval_scope="reranked_lexical_shortlist",
                semantic_receipt=model.receipt,
            )
            for hit in model.search(query, k=limit)
        ]

    def iter_records(self, batch_size=1000):
        """Stream current records for existing extraction/retrieval adapters."""
        _positive_integer(batch_size, "batch_size", 10000)
        cursor = self.conn.execute(
            "SELECT v.* FROM current_records c JOIN versions v ON v.id=c.version_id ORDER BY c.pmid"
        )
        while rows := cursor.fetchmany(batch_size):
            yield from (self._record(row) for row in rows)

    def validate(self):
        """Validate SQLite and exact external-content/FTS agreement."""
        if self.conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity check failed")
        if not self.read_only:
            self.conn.execute(
                "INSERT INTO literature_fts(literature_fts,rank) VALUES('integrity-check',1)"
            )
            self.conn.commit()
        return {"sqlite": "PASS", "fts_content": "NOT_RUN_READ_ONLY" if self.read_only else "PASS"}

    def validate_admission(self):
        """Check schema identity without rescanning all payload and FTS pages.

        Atomic shard commits, pinned source-prefix receipts and SQLite recovery
        protect continuation. Full integrity remains a separate acceptance step.
        """
        query = "SELECT name,type,sql FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'"

        def signature(conn):
            return sorted(
                (row[0], row[1], " ".join((row[2] or "").split())) for row in conn.execute(query)
            )

        expected = sqlite3.connect(":memory:")
        try:
            expected.executescript(_DDL)
            if signature(self.conn) != signature(expected):
                raise ValueError("Literature database schema differs from the admitted contract")
        finally:
            expected.close()
        return {"schema": "PASS", "full_integrity_validation": "REQUIRED_SEPARATELY"}


def _apply_batch(index, pairs, shard):
    conn = index.conn
    pmids = sorted({event["pmid"] for event, _ in pairs})
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS touched(pmid TEXT PRIMARY KEY)")
    conn.execute("DELETE FROM touched")
    conn.executemany("INSERT INTO touched VALUES(?)", ((pmid,) for pmid in pmids))
    # Remove old FTS text before any version payload is modified or deleted.
    removed_current = conn.execute(
        "DELETE FROM current_records WHERE pmid IN (SELECT pmid FROM touched)"
    ).rowcount
    for event, record in pairs:
        pmid, version = event["pmid"], event["version"]
        if event["event_type"] == "upsert":
            key = (
                str(version)
                if version is not None
                else f"unknown:{event['shard_order']}:{event['event_index']}"
            )
            provenance = {
                "adapter_name": "pubmed-current-index",
                "adapter_version": INDEX_VERSION,
                "source_locator": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "source_shard": shard["source_shard"],
                "source": shard["source"],
                "legacy_relative_path": shard["legacy_relative_path"],
                "legacy_sha256": shard["legacy"]["sha256"],
                "sidecar_sha256": shard["sidecar_sha256"],
                "event": event,
                "selection_rule": SELECTION_RULE,
            }
            metadata = {
                key: value
                for key, value in record.items()
                if key not in {"title", "abstract", "pmid"}
            }
            conn.execute(
                "INSERT INTO versions(pmid,version_key,version,title,abstract,metadata_json,provenance_json) "
                "VALUES(?,?,?,?,?,?,?) ON CONFLICT(pmid,version_key) DO UPDATE SET "
                "title=excluded.title,abstract=excluded.abstract,metadata_json=excluded.metadata_json,provenance_json=excluded.provenance_json",
                (
                    pmid,
                    key,
                    version,
                    record["title"] or "",
                    record["abstract"] or "",
                    _json(metadata),
                    _json(provenance),
                ),
            )
        elif version is None and event.get("version_raw") not in {None, ""}:
            conn.execute(
                "INSERT INTO unresolved_deletions VALUES(?,?)",
                (pmid, f"{event['shard_order']}:{event['event_index']}"),
            )
        elif version is None:
            conn.execute("DELETE FROM versions WHERE pmid=?", (pmid,))
            conn.execute("DELETE FROM unresolved_deletions WHERE pmid=?", (pmid,))
        else:
            conn.execute("DELETE FROM versions WHERE pmid=? AND version=?", (pmid, version))
    added_current = conn.execute(
        # CROSS JOIN pins the small touched batch as the outer loop. Without
        # this, SQLite can scan every historical active version for each batch,
        # making full-corpus materialization quadratic despite indexed lookups.
        "INSERT INTO current_records(pmid,version_id) SELECT v.pmid,v.id FROM touched t CROSS JOIN versions v ON t.pmid=v.pmid "
        "WHERE v.version=(SELECT max(w.version) FROM versions w WHERE w.pmid=v.pmid) "
        "AND NOT EXISTS(SELECT 1 FROM versions u WHERE u.pmid=v.pmid AND u.version IS NULL) "
        "AND NOT EXISTS(SELECT 1 FROM unresolved_deletions d WHERE d.pmid=v.pmid)"
    ).rowcount
    current_count = index._metadata("current_records_count")
    if current_count is not None:
        index._set_metadata(
            "current_records_count", current_count - removed_current + added_current
        )


def _paths(corpus_root, ledger_root, shard):
    legacy = _inside(corpus_root, shard["legacy_relative_path"])
    sidecar = _inside(
        ledger_root,
        f"partitions/{shard['source']}/{shard['source_shard'].removesuffix('.xml.gz')}/events.parquet",
    )
    return legacy, sidecar


def _file_identity(path):
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _apply_shard(index, corpus_root, ledger_root, shard, batch_size):
    import pyarrow.parquet as pq

    legacy, sidecar = _paths(corpus_root, ledger_root, shard)
    identities = [_file_identity(path) for path in (legacy, sidecar)]
    for path, expected in ((legacy, shard["legacy"]["sha256"]), (sidecar, shard["sidecar_sha256"])):
        if file_digest(path) != expected:
            raise ValueError(f"Input checksum changed: {path}")
    rows = pq.ParquetFile(legacy)
    if PUBMED_FIELDS - set(rows.schema_arrow.names):
        raise ValueError("Normalized corpus lacks required source fields")
    if any(str(rows.schema_arrow.field(field).type) != "string" for field in PUBMED_FIELDS):
        raise ValueError("Normalized corpus fields must retain the declared string schema")
    records = (
        record for batch in rows.iter_batches(batch_size=batch_size) for record in batch.to_pylist()
    )
    sentinel, sequence, pending = object(), hashlib.sha256(), []
    upserts = deletions = events = 0
    with index.conn:
        for batch in pq.ParquetFile(sidecar).iter_batches(batch_size=batch_size):
            for event in batch.to_pylist():
                if (
                    event.get("event_index") != events
                    or event.get("shard_order") != shard["shard_order"]
                    or event.get("source_shard") != shard["source_shard"]
                    or event.get("source") != shard["source"]
                    or not isinstance(event.get("pmid"), str)
                    or not event["pmid"].isdigit()
                ):
                    raise ValueError("Event source, order, or PMID is invalid")
                version = event.get("version")
                if version is not None and (type(version) is not int or version < 1):
                    raise ValueError("Invalid citation version")
                record = None
                if event["event_type"] == "upsert":
                    record = next(records, sentinel)
                    if (
                        record is sentinel
                        or record["pmid"] != event["pmid"]
                        or record["source_shard"] != shard["source_shard"]
                        or event["legacy_row_index"] != upserts
                    ):
                        raise ValueError("Ledger-to-corpus source row mapping changed")
                    sequence.update((event["pmid"] + "\n").encode())
                    upserts += 1
                elif event["event_type"] == "delete" and event["legacy_row_index"] is None:
                    deletions += 1
                else:
                    raise ValueError("Unknown or malformed citation event")
                events += 1
                pending.append((event, record))
                if len(pending) >= batch_size:
                    _apply_batch(index, pending, shard)
                    pending.clear()
        if pending:
            _apply_batch(index, pending, shard)
        if (
            next(records, sentinel) is not sentinel
            or events != shard["events"]
            or upserts != shard["upserts"]
            or deletions != shard["deletions"]
            or sequence.hexdigest() != shard["pmid_sequence_sha256"]
        ):
            raise ValueError("Shard counts or PMID sequence changed")
        if identities != [_file_identity(path) for path in (legacy, sidecar)]:
            raise ValueError("Input files changed during materialization")
        receipt = {
            "source": shard["source"],
            "source_shard": shard["source_shard"],
            "shard_order": shard["shard_order"],
            "events": events,
            "input_stat_identities": identities,
        }
        index.conn.execute(
            "INSERT INTO shards VALUES(?,?,?)",
            (shard["shard_order"], _shard_identity(shard), _json(receipt)),
        )
        index._set_metadata("events_applied", index._metadata("events_applied", 0) + events)
    return receipt


def materialize_pubmed(
    corpus_root,
    ledger_manifest,
    database,
    *,
    max_shards=2,
    resume=True,
    max_seconds=None,
    batch_size=4096,
    rehash_applied=False,
    expected_ledger_sha256=None,
    expected_implementation_sha256=None,
):
    """Apply at most ``max_shards`` NEW shards; ``None`` explicitly allows all.

    Completed shards are reused after receipt, input-stat and SQLite checks.
    Changed input stats trigger full checksum verification; ``rehash_applied``
    also rehashes unchanged inputs. Appended ledger manifests are supported;
    modifying, removing or inserting before any applied shard fails closed.
    The time bound is checked between atomic shards, never midway through one.
    """
    if (
        expected_implementation_sha256 is not None
        and expected_implementation_sha256 != _IMPLEMENTATION_SHA256
    ):
        raise ValueError("Materializer implementation changed after campaign admission")
    if max_shards is not None:
        _positive_integer(max_shards, "max_shards")
    _positive_integer(batch_size, "batch_size", 65536)
    if max_seconds is not None and (
        not isinstance(max_seconds, (int, float)) or not 0 < max_seconds < float("inf")
    ):
        raise ValueError("max_seconds must be positive and finite")
    corpus_root, ledger_manifest, database = map(
        lambda p: _resolved_path(Path(p)), (corpus_root, ledger_manifest, database)
    )
    ledger_root = ledger_manifest.parent
    if _is_inside(database, corpus_root) or _is_inside(database, ledger_root):
        raise ValueError("The derivative database must be outside the protected corpus and ledger")
    if database.exists() and not resume:
        raise FileExistsError("An existing literature index requires resume=True")
    marker = json.loads((corpus_root / "CORPUS_ROOT.json").read_text(encoding="utf-8"))
    if not marker.get("store_uuid"):
        raise ValueError("Missing corpus store identity")
    manifest_hash = file_digest(ledger_manifest)
    if expected_ledger_sha256 is not None and expected_ledger_sha256 != manifest_hash:
        raise ValueError("Ledger manifest changed after campaign admission")
    manifest = _manifest(ledger_manifest)
    started, new_shards = time.perf_counter(), []
    with (
        integration_lease(
            database.parent / (database.name + ".writer"), "pubmed-current-materialization"
        ),
        LiteratureIndex(database) as index,
    ):
        index.validate_admission()
        previous_store = index._metadata("corpus_store_uuid")
        if previous_store is not None and previous_store != marker["store_uuid"]:
            raise ValueError("Cannot mix distinct source corpus stores")
        applied = index.conn.execute("SELECT * FROM shards ORDER BY shard_order").fetchall()
        if len(applied) > len(manifest["shards"]):
            raise ValueError("New manifest removed already applied shards")
        for old, shard in zip(applied, manifest["shards"], strict=False):
            if old["shard_order"] != shard["shard_order"] or old["identity"] != _shard_identity(
                shard
            ):
                raise ValueError("Applied ledger prefix changed; use a separate derivative index")
            saved = json.loads(old["receipt"])["input_stat_identities"]
            for path, identity, expected in zip(
                _paths(corpus_root, ledger_root, shard),
                saved,
                (shard["legacy"]["sha256"], shard["sidecar_sha256"]),
                strict=True,
            ):
                if (rehash_applied or identity != _file_identity(path)) and file_digest(
                    path
                ) != expected:
                    raise ValueError("Previously applied source bytes changed")
        if index._metadata("events_applied", 0) != sum(
            shard["events"] for shard in manifest["shards"][: len(applied)]
        ):
            raise ValueError("Committed event counter disagrees with the admitted shard prefix")
        with index.conn:
            index._set_metadata("corpus_store_uuid", marker["store_uuid"])
            index._set_metadata("available_shards", len(manifest["shards"]))
            index._set_metadata("status", "PARTIAL")
            if index._metadata("current_records_count") is None:
                index._set_metadata(
                    "current_records_count",
                    index.conn.execute("SELECT count(*) FROM current_records").fetchone()[0],
                )
        pending = manifest["shards"][len(applied) :]
        for shard in pending[:max_shards] if max_shards is not None else pending:
            if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                break
            new_shards.append(_apply_shard(index, corpus_root, ledger_root, shard, batch_size))
        if file_digest(ledger_manifest) != manifest_hash:
            raise ValueError(
                "Ledger manifest changed during materialization; committed shards remain recoverable"
            )
        complete = (
            len(applied) + len(new_shards) == len(manifest["shards"])
            and manifest.get("status") == "PASS"
            and len(manifest["shards"]) == manifest.get("available_shards")
        )
        with index.conn:
            index._set_metadata("status", "COMPLETE" if complete else "PARTIAL")
            index._set_metadata("ledger_manifest_sha256", manifest_hash)
        report = dict(
            index.status(),
            newly_applied_shards=len(new_shards),
            reused_shards=len(applied),
            elapsed_seconds=time.perf_counter() - started,
            database=str(database),
            ledger_manifest_sha256=manifest_hash,
            implementation_sha256_at_import=_IMPLEMENTATION_SHA256,
            snapshot=index.snapshot_identity(),
            batch_size=batch_size,
            verification="New inputs SHA256 verified; applied inputs stat-checked and rehashed when changed.",
            full_integrity_validation="REQUIRED_SEPARATELY",
            historical_events_retain_source_ledger=True,
            semantic_index="optional bounded lexical-shortlist reranking",
        )
        index.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        report["immutable_receipt_file"] = database.name + f".receipt-{time.time_ns()}.json"
        atomic_json(database.parent / report["immutable_receipt_file"], report)
        atomic_json(database.with_suffix(database.suffix + ".receipt.json"), report)
        return report


def _replay_chunk_receipt(path, database, ledger_manifest, expected_implementation):
    """Republish a verified completed chunk without advancing the database."""
    report = json.loads(path.read_text(encoding="utf-8"))
    name = report.get("immutable_receipt_file", "")
    if not re.fullmatch(re.escape(database.name) + r"\.receipt-[0-9]+\.json", name):
        raise ValueError("Chunk receipt lacks its immutable materialization receipt")
    original = database.parent / name
    if (
        json.loads(original.read_text(encoding="utf-8")) != report
        or report.get("database") != str(database)
        or report.get("ledger_manifest_sha256") != file_digest(ledger_manifest)
        or report.get("implementation_sha256_at_import") != expected_implementation
    ):
        raise ValueError("Completed chunk receipt identity changed")
    with LiteratureIndex(database, read_only=True) as index:
        prefix = index.snapshot_identity(applied_shards=report["applied_shards"])
        if prefix["snapshot_sha256"] != report["snapshot"]["snapshot_sha256"]:
            raise ValueError("Completed chunk's source prefix changed")
    atomic_json(path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("materialize")
    build.add_argument("--corpus-root", required=True, type=Path)
    build.add_argument("--ledger-manifest", required=True, type=Path)
    build.add_argument("--database", required=True, type=Path)
    build.add_argument("--max-shards", type=int, default=2)
    build.add_argument("--all-shards", action="store_true")
    build.add_argument("--max-seconds", type=float)
    build.add_argument("--batch-size", type=int, default=4096)
    build.add_argument("--rehash-applied", action="store_true")
    build.add_argument("--receipt-out", type=Path)
    build.add_argument("--expected-ledger-sha256")
    build.add_argument("--expected-implementation-sha256")
    for command in ("search", "status", "validate"):
        sub = commands.add_parser(command)
        sub.add_argument("--database", required=True, type=Path)
        if command == "search":
            sub.add_argument("query")
            sub.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)
    if args.command == "materialize":
        if (
            args.expected_implementation_sha256
            and args.expected_implementation_sha256 != _IMPLEMENTATION_SHA256
        ):
            raise ValueError("Materializer implementation changed after campaign admission")
        database = _resolved_path(args.database)
        receipt_out = _resolved_path(args.receipt_out) if args.receipt_out else None
        if receipt_out is not None:
            if (
                _is_inside(receipt_out, _resolved_path(args.corpus_root))
                or _is_inside(receipt_out, _resolved_path(args.ledger_manifest.parent))
                or receipt_out
                in {database, Path(str(database) + "-wal"), Path(str(database) + "-shm")}
            ):
                raise ValueError("Chunk receipt must not replace source files or the database")
            if receipt_out.exists():
                if (
                    args.expected_ledger_sha256
                    and file_digest(args.ledger_manifest) != args.expected_ledger_sha256
                ):
                    raise ValueError("Ledger manifest changed after campaign admission")
                result = _replay_chunk_receipt(
                    receipt_out,
                    database,
                    args.ledger_manifest,
                    args.expected_implementation_sha256 or _IMPLEMENTATION_SHA256,
                )
                print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
                return 0
        try:
            result = materialize_pubmed(
                args.corpus_root,
                args.ledger_manifest,
                args.database,
                max_shards=None if args.all_shards else args.max_shards,
                max_seconds=args.max_seconds,
                batch_size=args.batch_size,
                rehash_applied=args.rehash_applied,
                expected_ledger_sha256=args.expected_ledger_sha256,
                expected_implementation_sha256=args.expected_implementation_sha256,
            )
        except OwnershipConflict:
            print(json.dumps({"status": "WRITER_BUSY", "retryable_exit_code": 75}))
            return 75
        if receipt_out is not None:
            atomic_json(receipt_out, result)
    else:
        with LiteratureIndex(args.database, read_only=args.command != "validate") as index:
            result = (
                index.search(args.query, args.limit)
                if args.command == "search"
                else index.validate()
                if args.command == "validate"
                else index.status()
            )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
