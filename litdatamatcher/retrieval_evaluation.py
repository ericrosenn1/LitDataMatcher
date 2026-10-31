"""Source-linked retrieval metrics with explicit per-query denominators."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .data_plane import digest


def evaluate_retrieval(
    queries: list[dict], rankings: list[list[dict]], *, k: int = 10, seed: int = 1729
) -> dict:
    """Evaluate known source links, not suitability of unjudged alternatives."""
    if not queries or len(queries) != len(rankings):
        raise ValueError("Nonempty queries and equally sized rankings are required")
    if type(k) is not int or k < 1:
        raise ValueError("Positive integer k is required")
    if len({row["id"] for row in queries}) != len(queries):
        raise ValueError("Duplicate query identifiers")
    reciprocal, recall1, recallk, ndcg, details = [], [], [], [], []
    by_source = defaultdict(list)
    for query, ranking in zip(queries, rankings, strict=True):
        identifiers = [hit["id"] for hit in ranking[:k]]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Duplicate candidate in one ranking")
        target = query["positive_doc_id"]
        rank = identifiers.index(target) + 1 if target in identifiers else None
        rr, hit = (1 / rank, 1) if rank is not None else (0.0, 0)
        reciprocal.append(rr)
        recall1.append(int(rank == 1))
        recallk.append(hit)
        ndcg.append(1 / math.log2(rank + 1) if rank else 0.0)
        by_source[query.get("source_corpus", "unknown")].append(rr)
        details.append(
            {
                "query_id": query["id"],
                "positive_doc_id": target,
                "query_sha256": digest(query),
                "rank": rank,
                "reciprocal_rank": rr,
            }
        )
    values = np.asarray(reciprocal)
    samples = (
        np.random.default_rng(seed)
        .choice(values, size=(1000, len(values)), replace=True)
        .mean(axis=1)
    )
    return {
        "queries": len(queries),
        "cutoff": k,
        "known_link_recall_at_1": float(np.mean(recall1)),
        "known_link_recall_at_k": float(np.mean(recallk)),
        "mrr_at_k": float(np.mean(reciprocal)),
        "ndcg_at_k": float(np.mean(ndcg)),
        "mrr_query_bootstrap_95_interval": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        "by_source": {
            source: {"queries": len(rows), "mrr_at_k": float(np.mean(rows))}
            for source, rows in sorted(by_source.items())
        },
        "query_results": details,
        "interpretation": "Recovery of recorded source-paper links; unjudged alternatives are not proven scientific negatives.",
    }


def paired_comparison(incumbent: dict, candidate: dict, *, seed: int = 1729) -> dict:
    """Compare the same queries; reject changed denominators."""
    if incumbent.get("cutoff") != candidate.get("cutoff"):
        raise ValueError("Paired comparison requires identical cutoffs")
    for metrics in [incumbent, candidate]:
        rows = metrics["query_results"]
        if len({row["query_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate query in paired metrics")
        if any(
            not math.isfinite(row["reciprocal_rank"]) or not 0 <= row["reciprocal_rank"] <= 1
            for row in rows
        ):
            raise ValueError("Invalid reciprocal rank")
    left_identity = {
        row["query_id"]: (row["positive_doc_id"], row.get("query_sha256"))
        for row in incumbent["query_results"]
    }
    right_identity = {
        row["query_id"]: (row["positive_doc_id"], row.get("query_sha256"))
        for row in candidate["query_results"]
    }
    if left_identity != right_identity:
        raise ValueError("Paired comparison requires identical queries and targets")
    left = {row["query_id"]: row["reciprocal_rank"] for row in incumbent["query_results"]}
    right = {row["query_id"]: row["reciprocal_rank"] for row in candidate["query_results"]}
    if not left or set(left) != set(right):
        raise ValueError("Paired comparison requires identical nonempty query sets")
    differences = np.array([right[key] - left[key] for key in sorted(left)], dtype=float)
    samples = (
        np.random.default_rng(seed)
        .choice(differences, size=(1000, len(differences)), replace=True)
        .mean(axis=1)
    )
    return {
        "queries": len(left),
        "mean_reciprocal_rank_delta": float(differences.mean()),
        "query_bootstrap_95_interval": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        "improved_queries": int((differences > 0).sum()),
        "worsened_queries": int((differences < 0).sum()),
        "unchanged_queries": int((differences == 0).sum()),
        "interpretation": "Paired development comparison, not untouched-test or expert-validity evidence.",
    }
