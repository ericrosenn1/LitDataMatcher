"""Conservative lifecycle and duplicate integrity for normalized literature rows."""

from __future__ import annotations

from .data_plane import digest
from .schemas import JsonDict, stable_id


def _relations(row: JsonDict) -> dict:
    value = row.get("version_relationships") or row.get("metadata", {}).get("version_relationships", {})
    return value if isinstance(value, dict) else {}


def _lifecycle(relations: dict) -> str:
    keys = " ".join(_lifecycle_relation_keys(relations))
    if "retract" in keys:
        return "RETRACTED"
    if "correction" in keys or "update" in keys:
        return "CORRECTED_REQUIRES_VERSION_REVIEW"
    if "version" in keys:
        return "VERSIONED_REQUIRES_VERSION_REVIEW"
    return "ACTIVE_METADATA_ONLY"


def _lifecycle_relation_keys(relations: dict) -> list[str]:
    """Return direct relation keys, including one source-scoped relation level.

    DOI merges retain alternate-source relations as ``{source: {relation: ...}}``.
    Only relation names at those two contract levels are lifecycle signals: arbitrary
    deeper metadata must not be interpreted as a correction or retraction.
    """

    keys: list[str] = []
    for name, value in relations.items():
        keys.append(str(name).casefold())
        if isinstance(value, dict):
            keys.extend(str(nested_name).casefold() for nested_name in value)
    return keys


def consolidate_literature_rows(rows: list[JsonDict], source_statuses: list[JsonDict] | None = None) -> list[JsonDict]:
    """Attach lifecycle/source snapshots without declaring unresolved records clean."""
    result = []
    for row in rows:
        record = dict(row)
        metadata = dict(record.get("metadata", {}) or {})
        alternates = list(metadata.get("alternate_source_ids", []) or [])
        source_ids = [str(record.get("source_id", "")), *map(str, alternates)]
        provenance = [record.get("source_provenance", {}), *list(metadata.get("alternate_source_provenance", []) or [])]
        snapshots = []
        for source_id, item in zip(source_ids, provenance, strict=False):
            item = item if isinstance(item, dict) else {}
            snapshots.append({"source_id": source_id, "source_type": str(item.get("source_type", record.get("source", "unknown"))), "retrieval_time_utc": str(item.get("retrieval_time_utc", "")), "status": "OBSERVED"})
        relations = _relations(record)
        lifecycle = _lifecycle(relations)
        fulltext = record.get("fulltext_status")
        integrity = {
            "schema_version": "literature_integrity_v1",
            "dedup_group_id": stable_id("literature_dedup", *sorted(value for value in source_ids if value)),
            "source_snapshots": snapshots,
            "source_statuses": source_statuses or [{"source": record.get("source", "unknown"), "status": "OBSERVED"}],
            "version_relationships": relations,
            "lifecycle_status": lifecycle,
            "fulltext_status": str(fulltext) if fulltext else "UNKNOWN",
            "evidence_eligibility": "INELIGIBLE_REQUIRES_VERSION_REVIEW" if lifecycle != "ACTIVE_METADATA_ONLY" or len(source_ids) > 1 else "REQUIRES_SOURCE_REVIEW",
            "derivation_invalidation_key": digest({"sources": source_ids, "relations": relations, "lifecycle": lifecycle}),
        }
        metadata["literature_integrity"] = integrity
        record["metadata"] = metadata
        result.append(record)
    return result


def invalidate_affected_derivations(previous: JsonDict, current: JsonDict, derivation_ids: list[str]) -> JsonDict:
    """Return deterministic invalidation rather than retaining stale derived evidence."""
    before = previous.get("metadata", {}).get("literature_integrity", {}).get("derivation_invalidation_key")
    after = current.get("metadata", {}).get("literature_integrity", {}).get("derivation_invalidation_key")
    return {"status": "INVALIDATED" if before != after else "UNCHANGED", "derivation_ids": sorted(set(derivation_ids)) if before != after else [], "previous_key": before, "current_key": after}


def evidence_eligible_literature(row: JsonDict) -> bool:
    """Lifecycle-affected metadata never silently becomes eligible evidence."""
    integrity = row.get("metadata", {}).get("literature_integrity", {})
    return isinstance(integrity, dict) and integrity.get("evidence_eligibility") == "ELIGIBLE"
