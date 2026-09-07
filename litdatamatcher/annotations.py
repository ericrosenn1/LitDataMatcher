"""Annotation-corpus loading and export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .annotation_agreement import build_agreement_artifacts
from .annotation_manifest import build_annotation_manifest
from .annotation_reports import assess_training_readiness, write_annotation_corpus_report
from .annotation_splits import DEFAULT_SPLIT_FRACTIONS, DEFAULT_SPLIT_SEED, write_annotation_splits
from .annotation_validation import issue_dicts, validate_review_rows
from .review import (
    load_review_labels,
    review_rows_to_match_labels,
    review_rows_to_question_quality_scores,
)
from .schemas import JsonDict, QuestionDataMatchLabel, QuestionQualityScore
from .storage import write_jsonl


ANNOTATION_CORPUS_VERSION = 1


def load_review_corpus_rows(paths: Iterable[str | Path]) -> list[JsonDict]:
    """Load completed review CSV/JSONL files with source-file provenance."""

    rows: list[JsonDict] = []
    for path in paths:
        source = Path(path)
        source_format = source.suffix.lower().lstrip(".") or "unknown"
        for row_number, row in enumerate(load_review_labels(source), 1):
            payload = dict(row)
            payload["_source_review_file"] = str(source)
            payload["_source_row_number"] = row_number
            payload["_source_format"] = source_format
            rows.append(payload)
    return rows


def build_training_labels(
    review_rows: Iterable[dict],
    annotator_id: str = "",
    include_unlabeled: bool = False,
) -> dict[str, list]:
    """Convert loaded review rows into typed training-label objects."""

    rows = list(review_rows)
    return {
        "question_data_match_labels": review_rows_to_match_labels(
            rows, annotator_id=annotator_id, include_unlabeled=include_unlabeled
        ),
        "question_quality_scores": review_rows_to_question_quality_scores(
            rows, annotator_id=annotator_id, include_unlabeled=include_unlabeled
        ),
    }


def summarize_annotation_corpus(
    match_labels: Iterable[QuestionDataMatchLabel],
    quality_scores: Iterable[QuestionQualityScore],
    source_rows: int = 0,
    validation_summary: JsonDict | None = None,
) -> JsonDict:
    """Summarize normalized labels for corpus QA."""

    match_labels = list(match_labels)
    quality_scores = list(quality_scores)
    relevance_scores = [
        label.relevance_score for label in match_labels if label.relevance_score is not None
    ]
    quality_values = [
        score.overall_score for score in quality_scores if score.overall_score is not None
    ]
    relevant = sum(1 for label in match_labels if label.label == "relevant")
    not_relevant = sum(1 for label in match_labels if label.label == "not_relevant")
    label_rows = len(match_labels) + len(quality_scores)
    summary = {
        "corpus_version": ANNOTATION_CORPUS_VERSION,
        "source_rows": source_rows,
        "exported_label_rows": label_rows,
        "exported_match_labels": len(match_labels),
        "exported_question_quality_labels": len(quality_scores),
        "question_data_match_labels": len(match_labels),
        "question_quality_scores": len(quality_scores),
        "relevant_match_labels": relevant,
        "not_relevant_match_labels": not_relevant,
        "relevance_distribution": {
            "relevant": relevant,
            "not_relevant": not_relevant,
            "uncertain": sum(1 for label in match_labels if label.label == "uncertain"),
            "unlabeled": sum(1 for label in match_labels if label.label == "unlabeled"),
        },
        "question_quality_distribution": _score_distribution(quality_values),
        "labels_per_reviewer": _labels_per_reviewer(match_labels, quality_scores),
        "source_caveat_counts": _source_caveat_counts(match_labels, quality_scores),
        "mean_match_relevance": round(sum(relevance_scores) / len(relevance_scores), 3)
        if relevance_scores
        else 0.0,
        "mean_question_quality": round(sum(quality_values) / len(quality_values), 3)
        if quality_values
        else 0.0,
    }
    if validation_summary:
        summary.update(
            {
                "valid_rows": validation_summary.get(
                    "valid_rows", validation_summary.get("usable_rows", 0)
                ),
                "usable_rows": validation_summary.get("usable_rows", 0),
                "skipped_rows": validation_summary.get("skipped_rows", 0),
                "warning_count": validation_summary.get("warnings", 0),
                "duplicate_count": validation_summary.get("duplicates", 0),
                "conflict_count": validation_summary.get("conflicts", 0),
                "reviewer_count": validation_summary.get("reviewer_count", 0),
            }
        )
    return summary


def _score_distribution(values: Iterable[float | None]) -> JsonDict:
    """Bucket optional 0..5 reviewer scores for QA reporting."""

    buckets = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    for value in values:
        if value is None:
            continue
        score = max(0.0, min(5.0, float(value)))
        buckets[str(int(round(score)))] += 1
    return buckets


def _labels_per_reviewer(
    match_labels: Iterable[QuestionDataMatchLabel],
    quality_scores: Iterable[QuestionQualityScore],
) -> JsonDict:
    """Count exported labels per annotator for reviewer coverage checks."""

    counts: dict[str, int] = {}
    for label in [*match_labels, *quality_scores]:
        reviewer = str(label.annotator_id or "").strip() or "blank"
        counts[reviewer] = counts.get(reviewer, 0) + 1
    return counts


def _source_caveat_counts(
    match_labels: Iterable[QuestionDataMatchLabel],
    quality_scores: Iterable[QuestionQualityScore],
) -> JsonDict:
    """Count reviewer-facing source caveats preserved in exported label metadata."""

    counts: dict[str, int] = {}
    for label in [*match_labels, *quality_scores]:
        caveats = label.metadata.get("source_caveats", [])
        if isinstance(caveats, str):
            caveats = [part.strip() for part in caveats.split(";")]
        if not isinstance(caveats, list):
            continue
        for caveat in caveats:
            text = str(caveat or "").strip()
            if text:
                counts[text] = counts.get(text, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def export_annotation_corpus(
    review_paths: Iterable[str | Path],
    output_dir: str | Path,
    annotator_id: str = "",
    include_unlabeled: bool = False,
    split_strategy: str | None = None,
    split_fractions: Iterable[float] = DEFAULT_SPLIT_FRACTIONS,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> JsonDict:
    """Write normalized training-label JSONL files from completed review files."""

    review_paths = [Path(path) for path in review_paths]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_review_corpus_rows(review_paths)
    validation = validate_review_rows(
        rows, annotator_id=annotator_id, include_unlabeled=include_unlabeled
    )
    validation_summary = validation.summary()
    labels = build_training_labels(validation.usable_rows, annotator_id, include_unlabeled)
    match_labels: list[QuestionDataMatchLabel] = labels["question_data_match_labels"]
    quality_scores: list[QuestionQualityScore] = labels["question_quality_scores"]

    outputs = {
        "question_data_match_labels": output_dir / "question_data_match_labels.jsonl",
        "question_quality_scores": output_dir / "question_quality_scores.jsonl",
        "summary": output_dir / "annotation_corpus_summary.json",
        "manifest": output_dir / "manifest.json",
        "report": output_dir / "annotation_corpus_report.md",
        "agreement_summary": output_dir / "agreement_summary.json",
        "adjudication_needed": output_dir / "adjudication_needed.jsonl",
        "warnings": output_dir / "warnings.jsonl",
        "skipped_rows": output_dir / "skipped_rows.jsonl",
        "duplicates": output_dir / "duplicates.jsonl",
        "conflicts": output_dir / "conflicts.jsonl",
    }
    write_jsonl(outputs["question_data_match_labels"], [label.to_dict() for label in match_labels])
    write_jsonl(outputs["question_quality_scores"], [score.to_dict() for score in quality_scores])
    warning_rows = issue_dicts(validation.warnings)
    skipped_rows = issue_dicts(validation.skipped_rows)
    duplicate_rows = issue_dicts(validation.duplicates)
    conflict_rows = issue_dicts(validation.conflicts)
    write_jsonl(outputs["warnings"], warning_rows)
    write_jsonl(outputs["skipped_rows"], skipped_rows)
    write_jsonl(outputs["duplicates"], duplicate_rows)
    write_jsonl(outputs["conflicts"], conflict_rows)

    summary = summarize_annotation_corpus(
        match_labels,
        quality_scores,
        source_rows=len(rows),
        validation_summary=validation_summary,
    )
    agreement_summary, adjudication_needed = build_agreement_artifacts(
        match_labels, conflict_rows=conflict_rows
    )
    write_jsonl(outputs["adjudication_needed"], adjudication_needed)
    split_metadata = write_annotation_splits(
        output_dir,
        match_labels,
        quality_scores,
        strategy=split_strategy,
        fractions=tuple(split_fractions),
        seed=split_seed,
    )
    if split_metadata.get("enabled"):
        for split_name, split_path in split_metadata.get("split_output_files", {}).items():
            outputs[f"split_{split_name}"] = Path(split_path)
    summary["agreement"] = agreement_summary
    summary["reviewer_overlap_counts"] = {
        f"{pair['reviewer_a']}|{pair['reviewer_b']}": pair["overlap_count"]
        for pair in agreement_summary.get("reviewer_pairs", [])
    }
    summary["unresolved_adjudication_count"] = agreement_summary.get(
        "adjudication_needed_count", 0
    )
    readiness = assess_training_readiness(
        summary,
        warnings=warning_rows,
        skipped_rows=skipped_rows,
        duplicates=duplicate_rows,
        conflicts=conflict_rows,
    )
    summary["training_readiness"] = readiness
    summary["splits"] = split_metadata
    _write_json(outputs["agreement_summary"], agreement_summary)
    manifest = build_annotation_manifest(
        review_paths,
        outputs,
        summary,
        validation_summary,
        annotator_id=annotator_id,
        include_unlabeled=include_unlabeled,
        corpus_version=ANNOTATION_CORPUS_VERSION,
        split_metadata=split_metadata,
        training_readiness=readiness,
        agreement_summary=agreement_summary,
        adjudication_needed_count=len(adjudication_needed),
    )
    write_annotation_corpus_report(
        outputs["report"],
        manifest,
        warnings=warning_rows,
        skipped_rows=skipped_rows,
        duplicates=duplicate_rows,
        conflicts=conflict_rows,
    )
    _write_json(outputs["summary"], summary)
    _write_json(outputs["manifest"], manifest)
    return manifest


def _write_json(path: str | Path, payload: JsonDict) -> None:
    """Write stable pretty JSON for corpus manifests and summaries."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
