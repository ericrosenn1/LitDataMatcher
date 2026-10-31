#!/usr/bin/env python3
"""Replay real processed UCI sources into explicit development feasibility cases.

These examples evaluate availability for defined analyses, not scientific
novelty, causal effects, classifier accuracy, or independent expert agreement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from litdatamatcher.data_plane import atomic_json, atomic_write
from litdatamatcher.dataset_inspection import inspect_tabular_dataset
from litdatamatcher.question_compiler import compile_question, matching_requirements
from litdatamatcher.scientific_v2 import rank_candidates

CASE_SPECIFICATIONS = [
    (17, "radius1", "Diagnosis", "mean radius", "tumor diagnosis"),
    (17, "texture1", "Diagnosis", "mean texture", "tumor diagnosis"),
    (17, "perimeter1", "Diagnosis", "mean perimeter", "tumor diagnosis"),
    (17, "area1", "Diagnosis", "mean area", "tumor diagnosis"),
    (17, "smoothness1", "Diagnosis", "mean smoothness", "tumor diagnosis"),
    (17, "compactness1", "Diagnosis", "mean compactness", "tumor diagnosis"),
    (45, "age", "num", "age", "angiographic diagnosis of heart disease"),
    (45, "trestbps", "num", "resting blood pressure", "angiographic diagnosis of heart disease"),
    (45, "chol", "num", "serum cholesterol", "angiographic diagnosis of heart disease"),
    (
        45,
        "thalach",
        "num",
        "maximum achieved heart rate",
        "angiographic diagnosis of heart disease",
    ),
    (
        45,
        "oldpeak",
        "num",
        "exercise-induced ST depression",
        "angiographic diagnosis of heart disease",
    ),
    (45, "exang", "num", "exercise-induced angina", "angiographic diagnosis of heart disease"),
]


def build(cached_root: Path, out: Path) -> dict:
    if (out / "QUALIFICATION_DEMONSTRATION.json").exists():
        raise ValueError("Preserve prior demonstration; choose a new output directory")
    manifest = json.loads((cached_root / "INSPECTION_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise ValueError("Only the successful original UCI source manifest may be replayed")
    retrievals = {row["url"]: row for row in manifest["retrievals"]}
    datasets, sources = [], []
    for uci_id in (17, 45):
        source_url = f"https://archive.ics.uci.edu/static/public/{uci_id}/data.csv"
        metadata_url = f"https://archive.ics.uci.edu/api/dataset?id={uci_id}"
        for name, url in (
            (f"uci_{uci_id}.csv", source_url),
            (f"uci_{uci_id}_metadata.json", metadata_url),
        ):
            actual = hashlib.sha256((cached_root / name).read_bytes()).hexdigest()
            if retrievals.get(url, {}).get("sha256") != actual:
                raise ValueError("Source asset differs from original verified acquisition: " + name)
            sources.append({"path": str((cached_root / name).resolve()), **retrievals[url]})
        metadata = json.loads(
            (cached_root / f"uci_{uci_id}_metadata.json").read_text(encoding="utf-8")
        )["data"]
        datasets.append(
            inspect_tabular_dataset(
                cached_root / f"uci_{uci_id}.csv",
                metadata,
                source_url=source_url,
                metadata_url=metadata_url,
            )
        )
    cases = []
    for index, (uci_id, feature, target, label, target_label) in enumerate(CASE_SPECIFICATIONS, 1):
        dataset = next(item for item in datasets if item["dataset_id"] == f"UCI:{uci_id}")
        identifier = f"processed-availability-{index:02}"
        question = f"Availability question: are {label} and {target_label} recorded together for an exploratory association analysis?"
        constraints = [
            {"field": "measured_variable:" + feature, "expected": True},
            {"field": "measured_variable:" + target, "expected": True},
            {"field": "processed_measurements", "expected": True},
        ]
        compilation = compile_question(
            question,
            source_locator="development-protocol:" + identifier,
            user_constraints=constraints,
        )
        requirements = matching_requirements(compilation)
        ranked = rank_candidates(requirements, datasets)
        selected = next(item for item in ranked if item["dataset_id"] == dataset["dataset_id"])
        availability_ranked = rank_candidates(constraints, datasets)
        availability_selected = next(
            item for item in availability_ranked if item["dataset_id"] == dataset["dataset_id"]
        )
        pair = "|".join(sorted((feature, target)))
        joint_rows = dataset["inspection"]["joint_observed_rows"][pair]
        cases.append(
            {
                "case_id": identifier,
                "question": question,
                "question_origin": "predeclared_development_availability_protocol_not_source_stated_gap",
                "compilation": compilation,
                "ranked_candidates": ranked,
                "selected_dataset": dataset["dataset_id"],
                "matched_source_rows": joint_rows,
                "availability_only_candidates": availability_ranked,
                "positive_availability_match": availability_selected["is_qualified"]
                and joint_rows > 0,
                "fully_qualified_scientific_match": selected["is_qualified"],
                "scientific_scope": "Feasible variable co-observation; estimand, independence, confounding, missingness policy and power need study-specific review",
            }
        )
    negatives = []
    for dataset in datasets:
        for field, expected, kind in [
            ("measured_variable", "IL6", "unavailable_measurement"),
            ("independent_unit_count", 100, "unknown_independence"),
        ]:
            ranked = rank_candidates([{"field": field, "expected": expected}], [dataset])
            negatives.append(
                {
                    "dataset_id": dataset["dataset_id"],
                    "control_kind": kind,
                    "requirement": {"field": field, "expected": expected},
                    "result": ranked[0],
                }
            )
    report = {
        "schema_version": "dataset-qualification-demonstration-1.0",
        "status": "PASS_WITH_LIMITATIONS",
        "positive_availability_matches": sum(row["positive_availability_match"] for row in cases),
        "fully_qualified_scientific_matches": sum(
            row["fully_qualified_scientific_match"] for row in cases
        ),
        "positive_case_count": len(cases),
        "distinct_processed_datasets": len(datasets),
        "negative_and_uncertain_controls": len(negatives),
        "sources": sources,
        "sources_reused_without_download": True,
        "cases": cases,
        "controls": negatives,
        "limits": [
            "Twelve question/dataset pairs use two real processed tables; they are not twelve independent datasets.",
            "Cases are conventional analysis-availability examples, not established novel research opportunities.",
            "No outcome tests, model training, untouched evaluation, or expert labeling were performed.",
            "Source-defined variables and missingness remain unchanged; statistical adequacy remains UNKNOWN.",
            "Full compiled scientific requirements retain unresolved temporal order and outcome ascertainment; availability alone is not scientific qualification.",
        ],
        "sealed_final_holdout_accessed": False,
    }
    if report["positive_availability_matches"] != 12:
        report["status"] = "FAIL"
    expected = {
        "unavailable_measurement": "NOT_QUALIFIED",
        "unknown_independence": "REQUIRES_INSPECTION",
    }
    if any(
        item["result"]["assessment"]["eligibility"] != expected[item["control_kind"]]
        for item in negatives
    ):
        report["status"] = "FAIL"
    out.mkdir(parents=True, exist_ok=True)
    atomic_write(
        out / "inspected_datasets.jsonl",
        b"".join(
            json.dumps(row, sort_keys=True, allow_nan=False).encode() + b"\n" for row in datasets
        ),
    )
    atomic_json(out / "QUALIFICATION_DEMONSTRATION.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-uci-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.cached_uci_root, args.out)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "positive_availability_matches",
                    "fully_qualified_scientific_matches",
                    "distinct_processed_datasets",
                    "negative_and_uncertain_controls",
                )
            }
        )
    )
    return int(report["status"] == "FAIL")


if __name__ == "__main__":
    raise SystemExit(main())
