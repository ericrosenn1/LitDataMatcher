"""Manifest helpers for reproducible annotation-corpus exports."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]


ANNOTATION_CORPUS_SCHEMA_VERSION = "annotation_corpus_v1"


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest for a source or output file."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_source_file(path: str | Path) -> JsonDict:
    """Return reproducibility metadata for one review-label source file."""

    source = Path(path)
    return {
        "path": str(source),
        "name": source.name,
        "suffix": source.suffix.lower(),
        "size_bytes": source.stat().st_size,
        "sha256": file_sha256(source),
    }


def build_annotation_manifest(
    review_paths: Iterable[str | Path],
    outputs: dict[str, Path],
    summary: JsonDict,
    validation_summary: JsonDict,
    annotator_id: str = "",
    include_unlabeled: bool = False,
    corpus_version: int = 1,
    split_metadata: JsonDict | None = None,
    training_readiness: JsonDict | None = None,
    agreement_summary: JsonDict | None = None,
    adjudication_needed_count: int = 0,
) -> JsonDict:
    """Build a reproducibility manifest for an annotation export."""

    review_paths = [Path(path) for path in review_paths]
    split_metadata = split_metadata or {}
    agreement_summary = agreement_summary or {}
    return {
        "corpus_version": corpus_version,
        "schema_version": ANNOTATION_CORPUS_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "annotator_id": annotator_id,
        "include_unlabeled": include_unlabeled,
        "source_review_files": [str(path) for path in review_paths],
        "source_files": [describe_source_file(path) for path in review_paths],
        "outputs": {key: str(path) for key, path in outputs.items()},
        "summary": summary,
        "validation": validation_summary,
        "training_readiness": training_readiness or summary.get("training_readiness", {}),
        "agreement": agreement_summary,
        "agreement_summary_path": str(outputs.get("agreement_summary", "")),
        "adjudication_needed_path": str(outputs.get("adjudication_needed", "")),
        "reviewer_overlap_counts": summary.get("reviewer_overlap_counts", {}),
        "unresolved_adjudication_count": int(adjudication_needed_count or 0),
        "splits": split_metadata,
        "split_strategy": split_metadata.get("split_strategy", "none"),
        "split_seed": split_metadata.get("split_seed"),
        "split_fractions": split_metadata.get("split_fractions", {}),
        "split_row_counts": split_metadata.get("split_row_counts", {}),
        "split_group_counts": split_metadata.get("split_group_counts", {}),
        "split_output_files": split_metadata.get("split_output_files", {}),
    }
