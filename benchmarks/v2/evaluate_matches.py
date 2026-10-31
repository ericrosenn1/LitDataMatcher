"""Frozen, source-traceable v2 candidate-ranking evaluation.

This evaluator reads acquired catalogue snapshots without redownloading them.  It
benchmarks lexical, MiniLM hybrid, and the lead compatibility gate over an
identical, fully labelled candidate universe.  Its relevance labels are
source-determined metadata retrieval labels, not expert biological gold labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_CATALOG = Path(r"C:\Codex\LitDataMatcher-v2\data\catalog\studies.jsonl")
DEFAULT_LEAD = Path(r"C:\Codex\LitDataMatcher-v2\lead")
DEFAULT_MODEL = Path(
    r"C:\Codex\LitDataMatcher-v2\data\models\all-MiniLM-L6-v2\1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)
DEFAULT_CONTROLLER_JUNIT = Path(
    r"C:\Codex\LitDataMatcher-v2\data\evaluation\E03_controller_independent.xml"
)
TOKEN = re.compile(r"[A-Za-z0-9]+")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_catalog(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows or any(not row.get("dataset_id") for row in rows):
        raise ValueError("Catalog must contain nonempty dataset_id records")
    return rows


def selected_records(
    rows: list[dict[str, Any]], topic: str, count: int, excluded: set[str]
) -> list[dict[str, Any]]:
    """Select source-order records, excluding known linked lineages before scoring."""
    chosen: list[dict[str, Any]] = []
    lineage_tokens: set[str] = set()
    for row in rows:
        if row.get("topic") != topic or row["dataset_id"] in excluded:
            continue
        lineage = {str(x) for x in row.get("study_lineage", []) if x}
        if not lineage:
            lineage = {row["dataset_id"]}
        if lineage & lineage_tokens:
            continue
        chosen.append(row)
        lineage_tokens.update(lineage)
        if len(chosen) == count:
            return chosen
    raise ValueError(f"Only {len(chosen)} unlinked {topic} records available; need {count}")


def locator(row: dict[str, Any]) -> dict[str, Any]:
    snapshot = row.get("source_snapshot") or {}
    if not row.get("source_locator") or not snapshot.get("sha256"):
        raise ValueError(f"{row['dataset_id']} lacks source locator or snapshot hash")
    return {
        "url": row["source_locator"],
        "snapshot_sha256": snapshot["sha256"],
        "snapshot_url": snapshot.get("url"),
        "json_pointer": f"/result/{row['dataset_id']}",
    }


def source_profile(row: dict[str, Any]) -> dict[str, Any]:
    """Build only a provenance-bearing profile; never infer donors or comparators."""
    loc = locator(row)
    required = ("organism", "assay", "title")
    if any(not row.get(field) for field in required):
        raise ValueError(f"{row['dataset_id']} missing a required acquired metadata field")
    count = row.get("sample_count_reported")
    if type(count) is not int or count < 0:
        raise ValueError(f"{row['dataset_id']} lacks a valid source-reported sample count")

    def observed(value: Any) -> dict[str, Any]:
        return {
            "value": value,
            "status": "observed",
            "source_locator": loc["url"],
            "mapping_type": "exact",
        }

    return {
        "dataset_id": row["dataset_id"],
        "availability": "SOURCE_METADATA_CAPTURED",
        "independent_units": None,
        "capabilities": {
            "organism": observed(row["organism"]),
            "assay": observed(row["assay"]),
            "study_title": observed(row["title"]),
            "reported_sample_count": observed(count),
            "comparator": {
                "value": None,
                "status": "unknown",
                "reason": "not assessed from acquired summary metadata",
            },
        },
        "source_fact_locator": loc,
    }


def question_for(row: dict[str, Any], split: str) -> dict[str, Any]:
    """A study-description retrieval question; it contains no accession identifier."""
    return {
        "question_id": f"{split}-{row['dataset_id']}",
        "split": split,
        "question": (
            "Find the source-described study matching this experimental context: "
            + row["title"]
            + ". Required organism: "
            + row["organism"]
            + "; required assay: "
            + row["assay"]
            + "."
        ),
        "requirements": [
            {
                "field": "organism",
                "expected": row["organism"],
                "essential": True,
                "source_locator": "frozen source-determined query",
            },
            {
                "field": "assay",
                "expected": row["assay"],
                "essential": True,
                "source_locator": "frozen source-determined query",
            },
            {
                "field": "study_title",
                "expected": row["title"],
                "essential": True,
                "source_locator": "frozen source-determined query",
            },
        ],
        "source_fact_locator": locator(row),
    }


def tokenize(text: str) -> set[str]:
    return {x.casefold() for x in TOKEN.findall(text) if len(x) > 1}


def lexical_score(query: str, candidate_text: str) -> float:
    left, right = tokenize(query), tokenize(candidate_text)
    return len(left & right) / len(left | right) if left or right else 0.0


def candidate_text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get(key, "")) for key in ("title", "summary", "organism", "assay"))


def order_scores(scores: dict[str, float]) -> list[str]:
    return [key for key, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))]


def relevance(question: dict[str, Any], candidate: dict[str, Any]) -> int:
    """Full-universe source-determined label: exact source description is direct; others are negatives."""
    return 3 if candidate["dataset_id"] == question["question_id"].split("-", 1)[1] else 0


def ranking_metrics(order: list[str], labels: dict[str, int], k: int = 10) -> dict[str, Any]:
    gains = [labels[item] for item in order]
    relevant = [index for index, gain in enumerate(gains, 1) if gain > 0]
    hits = sum(gain > 0 for gain in gains[:k])
    return {
        "queries": 1,
        "candidate_relevance_labels": len(labels),
        "positive_queries": int(bool(relevant)),
        "recall_at_10_numerator": int(bool(relevant and relevant[0] <= k)),
        "recall_at_10_denominator": int(bool(relevant)),
        "precision_at_5_numerator": hits if k == 5 else sum(gain > 0 for gain in gains[:5]),
        "precision_at_5_denominator": 5,
        "ndcg_at_5": (
            sum(gain / math.log2(index + 1) for index, gain in enumerate(gains[:5], 1))
            / (
                sum(
                    gain / math.log2(index + 1)
                    for index, gain in enumerate(sorted(labels.values(), reverse=True)[:5], 1)
                )
                or 1.0
            )
        ),
        "first_relevant_rank": relevant[0] if relevant else None,
        "invalid_top_match": int(labels[order[0]] == 0),
    }


def aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    sums = Counter()
    for item in metrics:
        for key in (
            "queries",
            "candidate_relevance_labels",
            "positive_queries",
            "recall_at_10_numerator",
            "recall_at_10_denominator",
            "precision_at_5_numerator",
            "precision_at_5_denominator",
            "invalid_top_match",
        ):
            sums[key] += item[key]
    denominator = sums["recall_at_10_denominator"]
    return {
        **dict(sums),
        "recall_at_10": sums["recall_at_10_numerator"] / denominator if denominator else None,
        "precision_at_5": sums["precision_at_5_numerator"] / sums["precision_at_5_denominator"],
        "mean_ndcg_at_5": sum(x["ndcg_at_5"] for x in metrics) / len(metrics),
        "invalid_top_match_rate": sums["invalid_top_match"] / sums["queries"]
        if sums["queries"]
        else None,
    }


def capability_audit(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    fields = Counter()
    failures: list[str] = []
    for profile in profiles:
        fields["organism"] += int(profile["capabilities"]["organism"]["status"] == "observed")
        fields["assay"] += int(profile["capabilities"]["assay"]["status"] == "observed")
        fields["study_title"] += int(profile["capabilities"]["study_title"]["status"] == "observed")
        fields["reported_sample_count"] += int(
            profile["capabilities"]["reported_sample_count"]["status"] == "observed"
        )
        fields["comparator_unknown"] += int(
            profile["capabilities"]["comparator"]["status"] == "unknown"
        )
        if profile["independent_units"] is not None:
            failures.append(profile["dataset_id"] + ": inferred independent units")
    total = sum(fields.values())
    return {
        "field_family_counts": dict(fields),
        "correct_numerator": total - len(failures),
        "denominator": total,
        "correctness": (total - len(failures)) / total if total else None,
        "source_families": len(profiles),
        "failures": failures,
        "comparators_retained_unknown": fields["comparator_unknown"],
        "independent_donor_counts_inferred": 0,
    }


def semantic_scores(
    model_dir: Path, texts: list[str], queries: list[str]
) -> tuple[list[list[float]], dict[str, Any]]:
    from litdatamatcher.semantic_runtime import PretrainedSemanticIndex, verify_model

    manifest = verify_model(model_dir)
    index = PretrainedSemanticIndex(model_dir, device="cpu").fit(
        [{"id": str(i), "text": value} for i, value in enumerate(texts)]
    )
    matrix: list[list[float]] = []
    for query in queries:
        hits = index.search(query, k=len(texts))
        row = [0.0] * len(texts)
        for hit in hits:
            row[int(hit["id"])] = hit["score"]
        matrix.append(row)
    return matrix, {
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        "license": manifest["license"],
        "runtime": "transformers PretrainedSemanticIndex",
        "device": "cpu",
        "cache_origin": "fresh_local_inference_no_cache",
    }


def scheduler_prerequisite_verdict(junit: Path) -> dict[str, Any]:
    """Report only the independently tested controller prerequisite scope."""
    root = ET.parse(junit).getroot()
    suites = root.findall(".//testsuite")
    counts = {
        name: sum(int(suite.attrib.get(name, "0")) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    passed = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    return {
        "junit_path": str(junit),
        "tests": counts["tests"],
        "passed": passed,
        "failures": counts["failures"],
        "errors": counts["errors"],
        "skipped": counts["skipped"],
        "verdict": "TESTED_PREREQUISITE_READY_FOR_SCHEDULED_SUPERVISOR_REVIEW"
        if not any(counts[key] for key in ("failures", "errors"))
        else "PREREQUISITE_NOT_READY",
        "scope": "Controller recovery, lease, path, stale-artifact and pause/resume contracts only; not a product, calibration, or scheduled-operation approval.",
    }


def evaluate_split(rows: list[dict[str, Any]], split: str, model_dir: Path) -> dict[str, Any]:
    from litdatamatcher.scientific_v2 import rank_candidates

    profiles = [source_profile(row) for row in rows]
    questions = [question_for(row, split) for row in rows]
    texts = [candidate_text(row) for row in rows]
    semantic, model = semantic_scores(model_dir, texts, [q["question"] for q in questions])
    by_method: dict[str, list[dict[str, Any]]] = {
        "lexical": [],
        "minilm_hybrid": [],
        "compatibility_aware": [],
    }
    per_query: list[dict[str, Any]] = []
    for index, question in enumerate(questions):
        lexical = {
            row["dataset_id"]: lexical_score(question["question"], texts[pos])
            for pos, row in enumerate(rows)
        }
        minilm = {row["dataset_id"]: semantic[index][pos] for pos, row in enumerate(rows)}
        hybrid = {key: 0.5 * lexical[key] + 0.5 * ((minilm[key] + 1.0) / 2.0) for key in lexical}
        labels = {row["dataset_id"]: relevance(question, row) for row in rows}
        orders = {"lexical": order_scores(lexical), "minilm_hybrid": order_scores(hybrid)}
        compatibility = rank_candidates(
            question["requirements"],
            profiles,
            {key: 2.0 * value - 1.0 for key, value in hybrid.items()},
        )
        orders["compatibility_aware"] = [item["dataset_id"] for item in compatibility]
        metrics = {name: ranking_metrics(order, labels) for name, order in orders.items()}
        for name, value in metrics.items():
            by_method[name].append(value)
        per_query.append(
            {
                "question_id": question["question_id"],
                "source_fact_locator": question["source_fact_locator"],
                "candidate_universe": [row["dataset_id"] for row in rows],
                "negative_labels": len(rows) - 1,
                "top_candidates": {name: order[:5] for name, order in orders.items()},
                "metrics": metrics,
            }
        )
    return {
        "split": split,
        "queries": questions,
        "candidate_profiles": profiles,
        "per_query": per_query,
        "metrics": {name: aggregate(values) for name, values in by_method.items()},
        "capability_audit": capability_audit(profiles),
        "model": model,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(__file__).parent / "fixtures" / "evaluation_v2_labels.json",
    )
    parser.add_argument("--lead", type=Path, default=DEFAULT_LEAD)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--controller-junit", type=Path, default=DEFAULT_CONTROLLER_JUNIT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.lead))
    labels = read_json(args.labels)
    rows = read_catalog(args.catalog)
    excluded = set(labels["reserved_excluded"])
    primary = selected_records(rows, "primary", 20, excluded)
    transfer = selected_records(rows, "transfer", 10, excluded)
    if [x["dataset_id"] for x in primary] != labels["primary_ids"] or [
        x["dataset_id"] for x in transfer
    ] != labels["transfer_ids"]:
        raise ValueError(
            "Acquired catalog no longer matches the frozen source-order selection fixture"
        )
    started = time.perf_counter()
    result = {
        "protocol_id": labels["protocol_id"],
        "label_origin": labels["label_origin"],
        "expert_labels_available": False,
        "catalog": str(args.catalog),
        "catalog_sha256": hashlib.file_digest(args.catalog.open("rb"), "sha256").hexdigest(),
        "holdout_exposure": labels["holdout_exposure"],
        "reserved_excluded": sorted(excluded),
        "scheduler_prerequisite_verdict": scheduler_prerequisite_verdict(args.controller_junit),
        "primary": evaluate_split(primary, "development_primary", args.model),
        "transfer": evaluate_split(transfer, "development_transfer", args.model),
        "limitations": [
            "Metadata retrieval labels are source-determined, not expert biological fit labels.",
            "Comparator and independent-donor facts were not inferred from title/summary metadata.",
            "Reserved GSE112372 is exposed metadata and is not claimed untouched; GSE214695/GSE226875 were excluded.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    primary_metrics = result["primary"]["metrics"]
    result["gate_assessment"] = {
        "capability_floor": {
            "numerator": result["primary"]["capability_audit"]["correct_numerator"],
            "denominator": result["primary"]["capability_audit"]["denominator"],
            "threshold": 0.95,
            "pass": result["primary"]["capability_audit"]["correctness"] >= 0.95
            and result["primary"]["capability_audit"]["source_families"] >= 10,
        },
        "primary_recall_floor": {
            "numerator": primary_metrics["compatibility_aware"]["recall_at_10_numerator"],
            "denominator": primary_metrics["compatibility_aware"]["recall_at_10_denominator"],
            "threshold": 0.90,
            "pass": primary_metrics["compatibility_aware"]["recall_at_10"] >= 0.90,
        },
        "invalid_top": {
            "count": primary_metrics["compatibility_aware"]["invalid_top_match"],
            "threshold": 0,
            "pass": primary_metrics["compatibility_aware"]["invalid_top_match"] == 0,
        },
        "overall": "NOT_PRODUCT_APPROVAL; coverage is metadata-retrieval only and other protocol gates remain pending",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
