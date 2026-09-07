"""Simple label-driven calibration utilities for ranked matches."""

from __future__ import annotations

from pathlib import Path

from .schemas import JsonDict, MatchCandidate, QuestionDataMatchLabel
from .storage import read_jsonl


def calibrate_ranking_threshold(
    matches_path: str | Path,
    labels_path: str | Path,
) -> JsonDict:
    """Evaluate match-score thresholds against normalized expert labels."""

    matches = {
        match.match_id: match
        for match in (MatchCandidate.from_dict(row) for row in read_jsonl(matches_path))
    }
    labels = [
        QuestionDataMatchLabel.from_dict(row)
        for row in read_jsonl(labels_path)
    ]
    examples: list[tuple[float, int, str]] = []
    for label in labels:
        match = matches.get(label.match_id)
        if match is None or label.relevance_score is None:
            continue
        examples.append((match.score.combined, 1 if label.relevance_score > 0 else 0, label.match_id))

    thresholds = [round(index / 20, 2) for index in range(21)]
    threshold_metrics = [_threshold_metrics(examples, threshold) for threshold in thresholds]
    best = max(
        threshold_metrics,
        key=lambda row: (row["f1"], row["precision"], row["recall"]),
        default=_threshold_metrics([], 0.5),
    )
    positives = sum(label for _, label, _ in examples)
    return {
        "schema_version": "ranking_calibration_v1",
        "matches_path": str(matches_path),
        "labels_path": str(labels_path),
        "labeled_matches": len(examples),
        "positive_labels": positives,
        "negative_labels": len(examples) - positives,
        "best_threshold": best["threshold"],
        "best_threshold_metrics": best,
        "threshold_metrics": threshold_metrics,
        "calibration_bins": _calibration_bins(examples),
        "limitations": (
            "This is a deterministic threshold calibration report, not a trained "
            "predictive model. Use it for exploratory tuning and QA only."
        ),
    }


def _threshold_metrics(examples: list[tuple[float, int, str]], threshold: float) -> JsonDict:
    """Compute binary metrics for one score threshold."""

    tp = fp = tn = fn = 0
    for score, label, _ in examples:
        pred = 1 if score >= threshold else 0
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and not label:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(examples) if examples else 0.0
    return {
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round(accuracy, 3),
    }


def _calibration_bins(examples: list[tuple[float, int, str]]) -> list[JsonDict]:
    """Summarize observed relevance rates by score decile."""

    bins: list[JsonDict] = []
    for index in range(10):
        low = index / 10
        high = (index + 1) / 10
        members = [
            (score, label, match_id)
            for score, label, match_id in examples
            if low <= score < high or (index == 9 and score == 1.0)
        ]
        positives = sum(label for _, label, _ in members)
        bins.append(
            {
                "score_min": round(low, 1),
                "score_max": round(high, 1),
                "count": len(members),
                "positive_labels": positives,
                "observed_positive_rate": round(positives / len(members), 3)
                if members
                else None,
                "match_ids": [match_id for _, _, match_id in members[:10]],
            }
        )
    return bins
