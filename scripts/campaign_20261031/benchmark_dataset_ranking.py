#!/usr/bin/env python3
"""Compare dataset retrieval against explicit processed-variable availability.

The reference labels are deterministic source-table availability cases, not
expert assessments, research novelty or scientific feasibility labels. Metadata
alternatives remain unjudged. No training or sealed evaluation is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import psutil

from litdatamatcher.data_plane import atomic_json, digest
from litdatamatcher.retrieval import BM25Index, CachedSemanticIndex, reciprocal_rank_fusion
from litdatamatcher.scientific_v2 import rank_candidates
from litdatamatcher.semantic_runtime import runtime_dependency_versions
from litdatamatcher.v2 import normalize_dataset
from litdatamatcher.workflow_checkpoint import CheckpointRun, file_identity

VERSION = "dataset-ranking-development-1.0"


def candidate_text(row):
    """Use observed metadata/variable definitions without inserting case labels."""
    parts = [str(row.get(key, "")) for key in ("title", "summary", "description")]
    for name, value in row.get("inspection", {}).get("variables", {}).items():
        definition = value.get("source_definition", {})
        parts.extend(
            [name, str(definition.get("description") or ""), str(definition.get("units") or "")]
        )
    for key, values in row.get("metadata", {}).get("reported_attributes", {}).items():
        parts.extend([key, " ".join(map(str, values))])
    return " ".join(part for part in parts if part).strip() or row["dataset_id"]


def metrics(cases, rankings, k):
    if type(k) is not int or k < 1:
        raise ValueError("Cutoff must be a positive integer")
    if len(cases) != len(rankings) or not cases:
        raise ValueError("Cases and rankings must be equally sized and nonempty")
    details = []
    for case, rows in zip(cases, rankings, strict=True):
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate ranking identity")
        target = case["selected_dataset"]
        rank = ids.index(target) + 1 if target in ids else None
        found = rank is not None and rank <= k
        details.append(
            {
                "case_id": case["case_id"],
                "target": target,
                "rank": rank,
                "reciprocal_rank_at_k": 1 / rank if found else 0.0,
                "hit_at_1": int(rank == 1),
                "recall_at_k": int(found),
                "ndcg_at_k": 1 / math.log2(rank + 1) if found else 0.0,
            }
        )
    return {
        "queries": len(cases),
        "cutoff": k,
        "known_target_hit_at_1": sum(row["hit_at_1"] for row in details) / len(details),
        "known_target_recall_at_k": sum(row["recall_at_k"] for row in details) / len(details),
        "mrr_at_k": sum(row["reciprocal_rank_at_k"] for row in details) / len(details),
        "ndcg_at_k": sum(row["ndcg_at_k"] for row in details) / len(details),
        "details": details,
    }


def load_inputs(root):
    report = json.loads((root / "QUALIFICATION_DEMONSTRATION.json").read_text(encoding="utf-8"))
    if (
        report.get("status") != "PASS_WITH_LIMITATIONS"
        or report.get("positive_availability_matches") != 12
    ):
        raise ValueError("Expected the preserved twelve-case availability demonstration")
    datasets = [
        json.loads(line)
        for line in (root / "inspected_datasets.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    input_files = [root / "QUALIFICATION_DEMONSTRATION.json", root / "inspected_datasets.jsonl"]
    for filename in (
        "biostudies-discovery.json",
        "arrayexpress-discovery.json",
        "bioproject-discovery.json",
        "biosample-discovery.json",
        "expressionatlas-discovery-v2.json",
    ):
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError("Missing frozen source-discovery input: " + str(path))
        input_files.append(path)
        datasets.extend(json.loads(path.read_text(encoding="utf-8"))["records"])
    unique = {}
    for row in datasets:
        key = row["dataset_id"]
        if key in unique:
            unique[key].setdefault("alternate_source_records", []).append(row)
        else:
            unique[key] = row
    return report, list(unique.values()), input_files


def run(root, model, out):
    import torch

    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 2:
        torch.set_num_interop_threads(2)
    reference, datasets, input_files = load_inputs(root)
    cases = reference["cases"]
    identity = {
        "schema_version": VERSION,
        "input_files": [file_identity(p) for p in input_files],
        "model": file_identity(model),
        "code": file_identity(Path(__file__)),
        "implementation": [
            file_identity(Path(__file__).parents[2] / "litdatamatcher" / name)
            for name in (
                "retrieval.py",
                "semantic_runtime.py",
                "scientific_v2.py",
                "v2.py",
                "ontology.py",
                "modality_contract.py",
            )
        ],
        "runtime_versions": runtime_dependency_versions(),
        "cpu_threads": 4,
        "device": "cpu",
    }
    with CheckpointRun(out, identity, resume=(out / ".workflow/state.json").exists()) as workflow:
        prior = workflow.completed()
        if prior is not None:
            return prior
        rss_start = psutil.Process().memory_info().rss
        records = [{"id": row["dataset_id"], "text": candidate_text(row)} for row in datasets]
        queries = [case["question"] for case in cases]
        started = time.perf_counter()
        bm25 = BM25Index().fit(records)
        lexical = [bm25.search(query, k=len(records)) for query in queries]
        lexical_seconds = time.perf_counter() - started
        started = time.perf_counter()
        dense_index = CachedSemanticIndex(model, out / "vectors", device="cpu").fit(records)
        dense = dense_index.search_many(queries, k=len(records))
        dense_seconds = time.perf_counter() - started
        started = time.perf_counter()
        fused = [
            reciprocal_rank_fusion([left, right], k=len(records))
            for left, right in zip(lexical, dense, strict=True)
        ]
        fusion_seconds = time.perf_counter() - started
        normalized = [normalize_dataset(row) for row in datasets]
        started = time.perf_counter()
        required = []
        for case, dense_rows in zip(cases, dense, strict=True):
            # These constraints were explicitly designated availability-only in
            # the frozen demonstration. Full scientific decisions stay separate.
            requirement_rows = case["availability_only_candidates"][0]["assessment"]["requirements"]
            constraints = [
                {key: row[key] for key in ("field", "expected", "essential", "source_locator")}
                for row in requirement_rows
            ]
            ranked = rank_candidates(
                constraints, normalized, {row["id"]: row["score"] for row in dense_rows}
            )
            required.append(
                [
                    {
                        "id": row["dataset_id"],
                        "score": row["score"],
                        "decision": row["assessment"]["eligibility"],
                        "is_qualified": row["is_qualified"],
                    }
                    for row in ranked
                ]
            )
        requirement_seconds = time.perf_counter() - started
        rankings = {
            "bm25": lexical,
            "dense": dense,
            "hybrid_rrf": fused,
            "requirement_aware_dense": required,
        }
        evaluations = {method: metrics(cases, rows, 5) for method, rows in rankings.items()}
        controls = []
        for control in reference["controls"]:
            selected = next(row for row in normalized if row["dataset_id"] == control["dataset_id"])
            result = rank_candidates([control["requirement"]], [selected])[0]
            controls.append(
                {
                    "dataset_id": control["dataset_id"],
                    "control_kind": control["control_kind"],
                    "expected": control["result"]["assessment"]["eligibility"],
                    "actual": result["assessment"]["eligibility"],
                    "pass": result["assessment"]["eligibility"]
                    == control["result"]["assessment"]["eligibility"],
                }
            )
        report = {
            "schema_version": VERSION,
            "status": "PASS_WITH_LIMITATIONS" if all(row["pass"] for row in controls) else "FAIL",
            "reference_label_origin": "deterministic_processed_source_variable_availability_not_expert_labels",
            "queries": len(cases),
            "distinct_positive_datasets": 2,
            "candidate_count": len(datasets),
            "candidate_universe_sha256": digest(records),
            "model": dense_index.receipt,
            "metrics": evaluations,
            "controls": controls,
            "timings_seconds": {
                "bm25_fit_and_queries": lexical_seconds,
                "dense_model_init_fit_and_batched_queries": dense_seconds,
                "hybrid_fusion_after_retrieval": fusion_seconds,
                "requirement_reranking_after_retrieval": requirement_seconds,
            },
            "memory": {
                "rss_before_bytes": rss_start,
                "rss_after_bytes": psutil.Process().memory_info().rss,
                "peak_rss": None,
                "scope": "single process snapshots, not sampled peak",
            },
            "comparison_to_bm25_mrr": {
                method: value["mrr_at_k"] - evaluations["bm25"]["mrr_at_k"]
                for method, value in evaluations.items()
            },
            "full_scientific_qualified_matches_in_reference": reference[
                "fully_qualified_scientific_matches"
            ],
            "limitations": [
                "Twelve availability questions reuse two processed datasets; they are not independent scientific evaluation cases.",
                "Labels come from source-variable availability and are not independent expert judgements or novel unanswered questions.",
                "Metadata-only alternatives are unjudged. Known-target recovery is not scientific precision or exhaustive candidate recall.",
                "Requirement-aware ranking checks the same declared availability contract used to construct labels; its success is engineering consistency, not external validation.",
                "Small fixed development pool; no significance claim, training, tuned thresholds or sealed holdout use.",
                "Full scientific contracts remain uncertain, including temporal order and outcome ascertainment.",
            ],
            "resources": {"device": "cpu", "torch_threads": 4, "python": platform.python_version()},
            "inputs": identity,
        }
        atomic_json(out / "candidate_texts.json", records)
        atomic_json(out / "rankings.json", rankings)
        atomic_json(out / "RANKING_COMPARISON.json", report)
        (out / "RUN_MANIFEST.json").unlink(missing_ok=True)
        workflow.finish(report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.inputs, args.model, args.out)
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidates": result["candidate_count"],
                "mrr_at_5": {key: val["mrr_at_k"] for key, val in result["metrics"].items()},
                "controls_passed": sum(row["pass"] for row in result["controls"]),
            }
        )
    )
    return int(result["status"] == "FAIL")


if __name__ == "__main__":
    raise SystemExit(main())
