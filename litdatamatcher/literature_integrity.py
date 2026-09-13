"""Conservative lifecycle and duplicate integrity for normalized literature rows."""

from __future__ import annotations

from .data_plane import digest
from .schemas import JsonDict, stable_id


def _relations(row: JsonDict) -> dict:
    primary = row.get("version_relationships", {})
    merged = row.get("metadata", {}).get("version_relationships", {})
    return {**(primary if isinstance(primary, dict) else {}),
            **(merged if isinstance(merged, dict) else {})}


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


def _content_version(row: JsonDict) -> dict:
    """Scientific source content, excluding replay times and local cache paths."""
    version = {key: row[key] for key in (
        "source_id", "title", "abstract", "text", "sections", "doi", "pmid",
        "pmcid", "year", "fulltext_status",
    ) if key in row}
    provenance = row.get("source_provenance", {}) or {}
    metadata = provenance.get("metadata", {}) if isinstance(provenance, dict) else {}
    snapshot = metadata.get("cache_snapshot", {}) if isinstance(metadata, dict) else {}
    for key, value in (("source_snapshot", row.get("source_snapshot")),
                       ("fulltext_snapshot", row.get("fulltext_snapshot")),
                       ("cache_snapshot", snapshot)):
        if isinstance(value, dict) and value.get("sha256"):
            version[key] = value["sha256"]
    return version


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
        if "merged_source_records" in metadata:
            # Pair each source identity with its own provenance, including an
            # explicit unknown when a duplicate did not supply provenance.
            members = [record, *metadata["merged_source_records"]]
            source_ids = list(dict.fromkeys(str(member.get("source_id", "")) for member in members))
            pairs = [(str(member.get("source_id", "")), member.get("source_provenance", {})) for member in members]
        else:
            pairs = list(zip(source_ids, provenance, strict=False))
        for source_id, item in pairs:
            item = item if isinstance(item, dict) else {}
            snapshots.append({"source_id": source_id, "source_type": str(item.get("source_type", "unknown")), "retrieval_time_utc": str(item.get("retrieval_time_utc", "")), "status": "OBSERVED" if item else "UNKNOWN"})
        relations = _relations(record)
        lifecycle = _lifecycle(relations)
        content_versions = sorted(
            (_content_version(member) for member in [record, *metadata.get("merged_source_records", [])]),
            key=digest,
        )
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
            "derivation_invalidation_key": digest({"sources": sorted(source_ids), "relations": relations, "lifecycle": lifecycle, "content_versions": content_versions}),
        }
        metadata["literature_integrity"] = integrity
        record["metadata"] = metadata
        result.append(record)
    return result


def invalidate_affected_derivations(previous: JsonDict, current: JsonDict, derivation_ids: list[str]) -> JsonDict:
    """Return deterministic invalidation rather than retaining stale derived evidence."""
    before = previous.get("metadata", {}).get("literature_integrity", {}).get("derivation_invalidation_key")
    after = current.get("metadata", {}).get("literature_integrity", {}).get("derivation_invalidation_key")
    changed = not before or not after or before != after
    return {"status": "INVALIDATED" if changed else "UNCHANGED", "derivation_ids": sorted(set(derivation_ids)) if changed else [], "previous_key": before, "current_key": after}


def evidence_eligible_literature(row: JsonDict) -> bool:
    """Lifecycle-affected metadata never silently becomes eligible evidence."""
    integrity = row.get("metadata", {}).get("literature_integrity", {})
    return isinstance(integrity, dict) and integrity.get("evidence_eligibility") == "ELIGIBLE"
