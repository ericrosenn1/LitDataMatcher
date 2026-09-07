"""Deterministic train/validation/test splits for annotation-label exports."""

from __future__ import annotations

from collections import Counter
import random
from pathlib import Path
from typing import Iterable, Sequence

from .schemas import JsonDict, QuestionDataMatchLabel, QuestionQualityScore
from .storage import write_jsonl


SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
DEFAULT_SPLIT_SEED = 1729


def write_annotation_splits(
    output_dir: str | Path,
    match_labels: Iterable[QuestionDataMatchLabel],
    quality_scores: Iterable[QuestionQualityScore],
    strategy: str | None = None,
    fractions: Sequence[float] = DEFAULT_SPLIT_FRACTIONS,
    seed: int = DEFAULT_SPLIT_SEED,
) -> JsonDict:
    """Optionally write grouped split JSONL files and return split metadata."""

    strategy = _normalize_strategy(strategy)
    if strategy == "none":
        return _disabled_split_metadata(seed=seed, fractions=fractions)

    rows = _combined_label_rows(match_labels, quality_scores)
    split_dir = Path(output_dir) / "splits"
    split_paths = {name: split_dir / f"{name}.jsonl" for name in SPLIT_NAMES}
    split_rows, split_warnings = split_annotation_rows(
        rows,
        strategy=strategy,
        fractions=fractions,
        seed=seed,
    )
    for name in SPLIT_NAMES:
        write_jsonl(split_paths[name], split_rows[name])

    metadata = build_split_metadata(
        split_rows,
        split_paths,
        strategy=strategy,
        fractions=fractions,
        seed=seed,
        warnings=split_warnings,
    )
    if not rows:
        metadata["warnings"].append("no labels were available for splitting")
    return metadata


def split_annotation_rows(
    rows: Iterable[JsonDict],
    strategy: str,
    fractions: Sequence[float] = DEFAULT_SPLIT_FRACTIONS,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[dict[str, list[JsonDict]], list[str]]:
    """Split label rows with deterministic grouping to reduce leakage risk."""

    rows = [dict(row) for row in rows]
    fractions_dict = normalize_split_fractions(fractions)
    grouped, warnings = _group_rows(rows, strategy)
    split_groups = _assign_groups(grouped, fractions_dict, seed)
    split_rows = {name: [] for name in SPLIT_NAMES}
    for split_name, group_keys in split_groups.items():
        for group_key in group_keys:
            split_rows[split_name].extend(grouped[group_key])
    return split_rows, warnings


def build_split_metadata(
    split_rows: dict[str, list[JsonDict]],
    split_paths: dict[str, Path],
    strategy: str,
    fractions: Sequence[float],
    seed: int,
    warnings: Iterable[str] = (),
) -> JsonDict:
    """Summarize generated split files for the export manifest."""

    fractions_dict = normalize_split_fractions(fractions)
    split_group_counts = {
        name: len({_split_group(row) for row in split_rows.get(name, [])})
        for name in SPLIT_NAMES
    }
    warnings = list(warnings)
    warnings.extend(_empty_split_warnings(split_group_counts, fractions_dict))
    return {
        "enabled": True,
        "split_strategy": strategy,
        "split_seed": int(seed),
        "split_fractions": fractions_dict,
        "split_row_counts": {name: len(split_rows.get(name, [])) for name in SPLIT_NAMES},
        "split_group_counts": split_group_counts,
        "split_grouping_field_counts": _split_grouping_field_counts(split_rows),
        "split_output_files": {name: str(split_paths[name]) for name in SPLIT_NAMES},
        "warnings": sorted(set(warnings)),
    }


def normalize_split_fractions(fractions: Sequence[float]) -> JsonDict:
    """Normalize split fractions into train/validation/test proportions."""

    if len(fractions) != 3:
        raise ValueError("split fractions must contain train, validation, and test values.")
    values = [float(value) for value in fractions]
    if any(value < 0 for value in values):
        raise ValueError("split fractions cannot be negative.")
    total = sum(values)
    if total <= 0:
        raise ValueError("at least one split fraction must be greater than zero.")
    normalized = [value / total for value in values]
    return {name: round(value, 6) for name, value in zip(SPLIT_NAMES, normalized)}


def _combined_label_rows(
    match_labels: Iterable[QuestionDataMatchLabel],
    quality_scores: Iterable[QuestionQualityScore],
) -> list[JsonDict]:
    """Merge typed label families into one training-split row stream."""

    rows: list[JsonDict] = []
    for label in match_labels:
        row = label.to_dict()
        row["label_type"] = "question_data_match"
        rows.append(row)
    for score in quality_scores:
        row = score.to_dict()
        row["label_type"] = "question_quality"
        rows.append(row)
    return rows


def _normalize_strategy(strategy: str | None) -> str:
    """Normalize CLI/API strategy aliases."""

    value = str(strategy or "none").strip().lower().replace("-", "_")
    aliases = {
        "": "none",
        "false": "none",
        "no": "none",
        "source_id": "by_source_id",
        "document_id": "by_document_id",
    }
    value = aliases.get(value, value)
    allowed = {"none", "by_question_id", "by_document_id", "by_source_id", "random"}
    if value not in allowed:
        raise ValueError(f"split strategy must be one of {sorted(allowed)}.")
    return value


def _disabled_split_metadata(seed: int, fractions: Sequence[float]) -> JsonDict:
    """Return manifest metadata when split generation was not requested."""

    return {
        "enabled": False,
        "split_strategy": "none",
        "split_seed": int(seed),
        "split_fractions": normalize_split_fractions(fractions),
        "split_row_counts": {name: 0 for name in SPLIT_NAMES},
        "split_group_counts": {name: 0 for name in SPLIT_NAMES},
        "split_grouping_field_counts": {name: {} for name in SPLIT_NAMES},
        "split_output_files": {},
        "warnings": [],
    }


def _group_rows(rows: list[JsonDict], strategy: str) -> tuple[dict[str, list[JsonDict]], list[str]]:
    """Build split groups while recording provenance fallbacks."""

    grouped: dict[str, list[JsonDict]] = {}
    warnings: list[str] = []
    for index, row in enumerate(rows):
        group_key, grouping_field, warning = _group_key(row, strategy, index)
        _attach_split_metadata(row, group_key, strategy, grouping_field)
        grouped.setdefault(group_key, []).append(row)
        if warning:
            warnings.append(warning)
    return grouped, sorted(set(warnings))


def _group_key(row: JsonDict, strategy: str, index: int) -> tuple[str, str, str]:
    """Return the leakage-control group key for one label row."""

    if strategy == "random":
        label_id = _first_nonblank(row, "label_id")
        return f"row:{label_id or index}", "label_id" if label_id else "row_index", ""
    if strategy == "by_question_id":
        question_id = _first_nonblank(row, "question_id")
        return (
            f"question:{question_id or index}",
            "question_id" if question_id else "row_index",
            "" if question_id else "missing question_id",
        )

    document_key, grouping_field = _document_or_source_key(row, strategy)
    if document_key:
        return f"source:{document_key}", grouping_field, ""
    question_id = _first_nonblank(row, "question_id")
    if question_id:
        return f"question:{question_id}", "question_id", (
            f"{strategy} fell back to question_id for rows missing source/document IDs"
        )
    return f"row:{index}", "row_index", f"{strategy} fell back to row index for missing IDs"


def _document_or_source_key(row: JsonDict, strategy: str) -> tuple[str, str]:
    """Find a document/source grouping value from top-level or metadata fields."""

    if strategy == "by_document_id":
        fields = ("document_id", "document_ids", "source_id", "primary_source_id", "source_ids")
    else:
        fields = ("source_id", "primary_source_id", "source_ids", "document_id", "document_ids")
    for field in fields:
        value = _field_value(row, field)
        if value:
            return value, field
    return "", ""


def _attach_split_metadata(row: JsonDict, group_key: str, strategy: str, grouping_field: str) -> None:
    """Store split bookkeeping inside metadata instead of polluting label roots."""

    metadata = row.get("metadata", {})
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    metadata["split_group"] = group_key
    metadata["split_strategy"] = strategy
    metadata["split_grouping_field"] = grouping_field
    row["metadata"] = metadata


def _field_value(row: JsonDict, field: str) -> str:
    """Return a top-level or metadata grouping value."""

    value = _joined_list_value(row.get(field))
    if value:
        return value
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        return ""
    return _joined_list_value(metadata.get(field))


def _first_nonblank(row: JsonDict, *fields: str) -> str:
    """Return the first non-empty string value from a mapping."""

    for field in fields:
        value = str(row.get(field, "") or "").strip()
        if value:
            return value
    return ""


def _joined_list_value(value: object) -> str:
    """Return a stable group value from a list-like source field."""

    if isinstance(value, (list, tuple, set)):
        parts = sorted(str(item).strip() for item in value if str(item).strip())
        return "|".join(parts)
    return str(value or "").strip()


def _assign_groups(
    grouped: dict[str, list[JsonDict]],
    fractions: JsonDict,
    seed: int,
) -> dict[str, list[str]]:
    """Assign whole groups to splits with deterministic shuffling."""

    group_keys = sorted(grouped)
    rng = random.Random(int(seed))
    rng.shuffle(group_keys)
    group_count = len(group_keys)
    counts = _target_split_counts(group_count, fractions)
    train_end = counts["train"]
    validation_end = train_end + counts["validation"]
    return {
        "train": group_keys[:train_end],
        "validation": group_keys[train_end:validation_end],
        "test": group_keys[validation_end:],
    }


def _target_split_counts(group_count: int, fractions: JsonDict) -> dict[str, int]:
    """Convert fractional split targets into whole group counts."""

    raw_counts = {
        name: group_count * float(fractions.get(name, 0.0))
        for name in SPLIT_NAMES
    }
    counts = {name: int(raw_counts[name]) for name in SPLIT_NAMES}
    remaining = group_count - sum(counts.values())
    remainders = sorted(
        SPLIT_NAMES,
        key=lambda name: (raw_counts[name] - counts[name], fractions.get(name, 0.0)),
        reverse=True,
    )
    for name in remainders[:remaining]:
        counts[name] += 1
    return counts


def _split_group(row: JsonDict) -> str:
    """Read the assigned split group from row metadata."""

    metadata = row.get("metadata", {})
    if isinstance(metadata, dict):
        return str(metadata.get("split_group", "") or "")
    return ""


def _split_grouping_field(row: JsonDict) -> str:
    """Read the field used to create the split group."""

    metadata = row.get("metadata", {})
    if isinstance(metadata, dict):
        return str(metadata.get("split_grouping_field", "") or "")
    return ""


def _split_grouping_field_counts(split_rows: dict[str, list[JsonDict]]) -> JsonDict:
    """Count grouping-field provenance per split for manifest QA."""

    counts: JsonDict = {}
    for split_name, rows in split_rows.items():
        counter = Counter(_split_grouping_field(row) for row in rows if _split_grouping_field(row))
        counts[split_name] = dict(sorted(counter.items()))
    return counts


def _empty_split_warnings(split_group_counts: JsonDict, fractions: JsonDict) -> list[str]:
    """Warn when nonzero requested fractions produce empty grouped splits."""

    total_groups = sum(int(value) for value in split_group_counts.values())
    if total_groups == 0:
        return []
    return [
        f"{name} split is empty because grouped corpus size is small relative to fractions"
        for name in SPLIT_NAMES
        if float(fractions.get(name, 0.0)) > 0 and int(split_group_counts.get(name, 0)) == 0
    ]
