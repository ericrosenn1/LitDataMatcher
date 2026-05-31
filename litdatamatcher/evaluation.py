"""Evaluation utilities for extraction and ranking benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from pathlib import Path
from typing import Iterable

from .schemas import MatchCandidate, QuestionCandidate
from .storage import read_jsonl, write_jsonl
from .text import lexical_similarity


@dataclass(slots=True)
class ClassificationMetrics:
    """Precision/recall/F1 metrics for binary extraction tasks."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict:
        """Serialize metrics."""

        return asdict(self)


@dataclass(slots=True)
class RankingMetrics:
    """Ranking-quality metrics for expert relevance labels."""

    precision_at_k: float
    mean_reciprocal_rank: float
    ndcg_at_k: float
    judged_matches: int
    relevant_matches: int

    def to_dict(self) -> dict:
        """Serialize metrics."""

        return asdict(self)


def _f1(tp: int, fp: int, fn: int) -> ClassificationMetrics:
    """Compute precision, recall, and F1 with zero-safe denominators."""

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ClassificationMetrics(tp, fp, fn, round(precision, 3), round(recall, 3), round(f1, 3))


def evaluate_question_extraction(
    predicted: Iterable[QuestionCandidate],
    gold_rows: Iterable[dict],
    similarity_threshold: float = 0.72,
) -> ClassificationMetrics:
    """Evaluate extracted questions against gold rows.

    Gold rows should contain a `question` field. Matching is approximate by
    lexical similarity so small wording differences do not dominate early
    annotation-set evaluation.
    """

    predictions = list(predicted)
    gold_questions = [str(row.get("question", "")).strip() for row in gold_rows if row.get("question")]
    matched_gold: set[int] = set()
    tp = 0
    fp = 0
    for prediction in predictions:
        best_idx = -1
        best_score = 0.0
        for idx, gold in enumerate(gold_questions):
            if idx in matched_gold:
                continue
            score = lexical_similarity(prediction.question, gold)
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx >= 0 and best_score >= similarity_threshold:
            tp += 1
            matched_gold.add(best_idx)
        else:
            fp += 1
    fn = len(gold_questions) - len(matched_gold)
    return _f1(tp, fp, fn)


def evaluate_ranking(
    matches: Iterable[MatchCandidate],
    relevance_rows: Iterable[dict],
    k: int = 10,
) -> RankingMetrics:
    """Evaluate ranked matches against expert relevance labels.

    Relevance rows may identify matches by `match_id` or by the tuple
    (`question_id`, `dataset_id`). `relevance` should be numeric, where values
    greater than zero are considered relevant.
    """

    ranked = list(matches)[:k]
    relevance: dict[str, float] = {}
    for row in relevance_rows:
        label = float(row.get("relevance", row.get("label", 0)) or 0)
        if row.get("match_id"):
            relevance[str(row["match_id"])] = label
        elif row.get("question_id") and row.get("dataset_id"):
            relevance[f"{row['question_id']}::{row['dataset_id']}"] = label

    judged = 0
    relevant_count = 0
    reciprocal_rank = 0.0
    gains: list[float] = []
    ideal_gains = sorted([value for value in relevance.values() if value > 0], reverse=True)[:k]
    for rank, match in enumerate(ranked, 1):
        keys = [
            match.match_id,
            f"{match.question.question_id}::{match.dataset.dataset_id}",
        ]
        label = next((relevance[key] for key in keys if key in relevance), None)
        if label is None:
            gains.append(0.0)
            continue
        judged += 1
        gains.append(max(0.0, label))
        if label > 0:
            relevant_count += 1
            if reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / rank

    precision_at_k = relevant_count / k if k else 0.0
    ndcg = _ndcg(gains, ideal_gains, k)
    return RankingMetrics(
        precision_at_k=round(precision_at_k, 3),
        mean_reciprocal_rank=round(reciprocal_rank, 3),
        ndcg_at_k=round(ndcg, 3),
        judged_matches=judged,
        relevant_matches=relevant_count,
    )


def _dcg(gains: list[float], k: int) -> float:
    """Discounted cumulative gain."""

    return sum(gain / math.log2(idx + 2) for idx, gain in enumerate(gains[:k]))


def _ndcg(gains: list[float], ideal_gains: list[float], k: int) -> float:
    """Normalized discounted cumulative gain."""

    ideal = _dcg(ideal_gains, k)
    return _dcg(gains, k) / ideal if ideal else 0.0


def write_evaluation_report(path: str | Path, metrics: dict[str, dict]) -> None:
    """Write evaluation metrics to JSONL for reproducible benchmarking."""

    write_jsonl(path, [{"task": key, **value} for key, value in metrics.items()])


def load_gold_rows(path: str | Path) -> list[dict]:
    """Load gold labels from JSONL."""

    return read_jsonl(path)
