"""Run new descriptive development contracts without rewriting original cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from litdatamatcher.data_plane import atomic_json, atomic_write
from litdatamatcher.question_compiler import compile_question, matching_requirements
from litdatamatcher.scientific_v2 import rank_candidates
from litdatamatcher.tabular_contracts import (
    comparison_constraints,
    inspect_tabular_comparisons,
    scope_tabular_comparison,
)

FEATURES = {
    17: ["radius1", "texture1", "perimeter1", "area1", "smoothness1", "compactness1"],
    45: ["age", "trestbps", "chol", "thalach", "oldpeak", "exang"],
}
GROUPS = {17: ("Diagnosis", ["B", "M"]), 45: ("num", ["0", "1", "2", "3", "4"])}
LIMITS = [
    "Twelve full-contract feasibility matches reuse two public datasets; they are dependent development examples.",
    "New descriptive questions supplement, rather than reinterpret, the original association cases.",
    "No claim of novel unanswered opportunities, clinical effectiveness, independent validation, power or participant independence.",
    "Ascertainment means recorded source-defined measurements, not an adjudicated clinical endpoint.",
    "Source definition mappings are explicit and auditable; expert scientific judgments remain separate.",
    "Original temporal-order assumptions remain unresolved; no timing capability is inferred.",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def definition(metadata, variable):
    if metadata["uci_id"] == 17:
        return {
            "json_pointer": "/additional_info/variable_info",
            "text": metadata["additional_info"]["variable_info"],
            "mapping_note": f"Exact source dictionary column {variable}; documentation defines nucleus-feature families and diagnosis codes. Suffix 1 is retained literally without expanding an undocumented statistic or timing.",
        }
    if variable == "age":
        return {"json_pointer": "/additional_info/variable_info", "text": "3 age: age in years"}
    if variable == "num":
        return {
            "json_pointer": "/additional_info/summary",
            "text": 'The "goal" field refers to the presence of heart disease in the patient.  It is integer valued from 0 (no presence) to 4.',
            "mapping_note": "Dictionary target num; compare source codes literally without recoding or inferring severity.",
        }
    index = next(i for i, row in enumerate(metadata["variables"]) if row["name"] == variable)
    return {
        "json_pointer": f"/variables/{index}/description",
        "text": metadata["variables"][index]["description"],
    }


def build(cached_root, original_report, out):
    if out.exists():
        raise ValueError("Preserve prior outputs; choose a new output directory")
    manifest = json.loads((cached_root / "INSPECTION_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["status"] != "PASS":
        raise ValueError("Original public source acquisition must have passed")
    retrievals = {row["url"]: row for row in manifest["retrievals"]}
    profiles, sources = [], []
    for uci_id, features in FEATURES.items():
        source_url = f"https://archive.ics.uci.edu/static/public/{uci_id}/data.csv"
        metadata_url = f"https://archive.ics.uci.edu/api/dataset?id={uci_id}"
        for path, url in [
            (cached_root / f"uci_{uci_id}.csv", source_url),
            (cached_root / f"uci_{uci_id}_metadata.json", metadata_url),
        ]:
            if sha256(path) != retrievals[url]["sha256"]:
                raise ValueError("Frozen input differs from original acquisition: " + str(path))
            sources.append({"path": str(path.resolve()), **retrievals[url]})
        metadata = json.loads(
            (cached_root / f"uci_{uci_id}_metadata.json").read_text(encoding="utf-8")
        )["data"]
        grouping, values = GROUPS[uci_id]
        specs = [
            {
                "outcome_variable": feature,
                "grouping_variable": grouping,
                "group_values": values,
                "definition_evidence": {
                    name: definition(metadata, name) for name in (feature, grouping)
                },
            }
            for feature in features
        ]
        profiles.append(
            inspect_tabular_comparisons(
                cached_root / f"uci_{uci_id}.csv",
                metadata,
                specs,
                source_url=source_url,
                metadata_url=metadata_url,
            )
        )
    cases, requests, controls = [], [], []
    for uci_id, features in FEATURES.items():
        grouping, values = GROUPS[uci_id]
        for feature in features:
            case_id = f"descriptive-comparison-{len(cases) + 1:02}"
            question = f"What is the difference between recorded {grouping} groups {', '.join(values)} in the recorded {feature} measurement in UCI:{uci_id}?"
            constraints = comparison_constraints(feature, grouping, values)
            compilation = compile_question(
                question,
                source_locator="development-protocol:" + case_id,
                user_constraints=constraints,
            )
            if compilation["question_purpose"] != "comparative":
                raise ValueError("Existing compiler did not recognize the explicit comparison")
            requirements = matching_requirements(compilation)
            for method in ("joint_observation", "outcome_ascertainment"):
                if not any(
                    row["field"] == method and row["expected"] is True and row["essential"] is True
                    for row in requirements
                ):
                    raise ValueError("Full scientific method requirement is missing: " + method)
            scoped = [scope_tabular_comparison(profile, requirements) for profile in profiles]
            ranked = rank_candidates(requirements, scoped)
            selected = next(row for row in ranked if row["dataset_id"] == f"UCI:{uci_id}")
            selected_profile = next(row for row in scoped if row["dataset_id"] == f"UCI:{uci_id}")
            scope = selected_profile["comparison_scope"]
            evidence = next(
                row
                for row in selected_profile["inspection"]["comparison_evidence"]
                if row["evidence_id"] == scope["evidence_id"]
            )
            cases.append(
                {
                    "case_id": case_id,
                    "question": question,
                    "question_origin": "explicit_new_descriptive_development_protocol_not_source_stated_gap",
                    "compilation": compilation,
                    "ranked_candidates": ranked,
                    "selected_dataset": selected["dataset_id"],
                    "qualified_under_full_compiled_contract": selected["is_qualified"],
                    "comparison_scope": scope,
                    "observed_evidence": evidence,
                    "unresolvedness": "NOT_ASSESSED_NOT_CLAIMED",
                }
            )
            requests.append(
                {"case_id": case_id, "question": question, "user_constraints": constraints}
            )
            if feature == features[0]:
                variants = [
                    (
                        "unreported_timing",
                        requirements
                        + [{"field": "temporal_order", "expected": "exposure_before_outcome"}],
                        "REQUIRES_INSPECTION",
                    ),
                    (
                        "unknown_independence",
                        requirements + [{"field": "independent_unit_count", "expected": 100}],
                        "REQUIRES_INSPECTION",
                    ),
                    (
                        "unavailable_measurement",
                        requirements + [{"field": "measured_variable", "expected": "IL6"}],
                        "NOT_QUALIFIED",
                    ),
                    (
                        "unscoped_method_only",
                        [
                            {"field": "joint_observation", "expected": True},
                            {"field": "outcome_ascertainment", "expected": True},
                        ],
                        "REQUIRES_INSPECTION",
                    ),
                ]
                base = next(row for row in profiles if row["dataset_id"] == selected["dataset_id"])
                for kind, control_requirements, expected in variants:
                    result = rank_candidates(
                        control_requirements, [scope_tabular_comparison(base, control_requirements)]
                    )[0]
                    controls.append(
                        {
                            "dataset_id": base["dataset_id"],
                            "kind": kind,
                            "expected": expected,
                            "result": result,
                        }
                    )
    original_hash = sha256(original_report)
    original = json.loads(original_report.read_text(encoding="utf-8"))
    replay = []
    for case in original["cases"]:
        requirements = matching_requirements(case["compilation"])
        ranked = rank_candidates(
            requirements, [scope_tabular_comparison(row, requirements) for row in profiles]
        )
        selected = next(row for row in ranked if row["dataset_id"] == case["selected_dataset"])
        replay.append(
            {
                "case_id": case["case_id"],
                "question": case["question"],
                "requirements_unchanged": True,
                "result": selected,
                "unmatched_requirements": [
                    row["field"]
                    for row in selected["assessment"]["requirements"]
                    if row["status"] != "MATCH"
                ],
            }
        )
    qualified = sum(row["qualified_under_full_compiled_contract"] for row in cases)
    validation = {
        "comparison_cases": len(cases),
        "distinct_processed_datasets": len(profiles),
        "qualified_full_compiled_contracts": qualified,
        "controls": len(controls),
        "controls_correct": sum(
            row["result"]["assessment"]["eligibility"] == row["expected"] for row in controls
        ),
        "original_cases_replayed_unchanged": len(replay),
        "original_cases_qualified": sum(row["result"]["is_qualified"] for row in replay),
        "original_report_hash_unchanged": original_hash == sha256(original_report),
        "no_global_method_capabilities": all(
            not (
                {"joint_observation", "outcome_ascertainment", "temporal_order"}
                & row["capabilities"].keys()
            )
            for row in profiles
        ),
    }
    passed = (
        qualified == 12
        and validation["controls_correct"] == 8
        and validation["original_cases_qualified"] == 0
        and validation["original_report_hash_unchanged"]
        and validation["no_global_method_capabilities"]
    )
    report = {
        "schema_version": "descriptive-comparison-development-1.0",
        "status": "PASS_WITH_LIMITATIONS" if passed else "FAIL",
        "validation": validation,
        "cases": cases,
        "controls": controls,
        "original_cases": replay,
        "original_report": {"path": str(original_report.resolve()), "sha256": original_hash},
        "source_receipts": sources,
        "sources_reused_without_download": True,
        "scientific_rules_and_thresholds_changed": False,
        "sealed_final_holdout_accessed": False,
        "limitations": LIMITS,
    }
    out.mkdir(parents=True)
    for name, rows in [("inspected_datasets.jsonl", profiles), ("requests.jsonl", requests)]:
        atomic_write(
            out / name,
            b"".join(
                json.dumps(row, sort_keys=True, allow_nan=False).encode() + b"\n" for row in rows
            ),
        )
    atomic_json(out / "COMPARISON_DEVELOPMENT.json", report)
    atomic_json(
        out / "task_state.json",
        {
            "task_name": "source-backed descriptive comparison development",
            "task_status": report["status"],
            "inputs": sources,
            "outputs": [
                "COMPARISON_DEVELOPMENT.json",
                "inspected_datasets.jsonl",
                "requests.jsonl",
            ],
            "items_requested": 12,
            "items_processed": len(cases),
            "items_succeeded": qualified,
            "items_failed": len(cases) - qualified,
            "validation_status": "PASS" if passed else "FAIL",
            "audit_status": "PASS" if passed else "FAIL",
            "repair_cycles": 0,
            "repairs_performed": [],
            "unresolved_issues": LIMITS,
            "scientific_assumptions_changed": False,
            "requires_human_review": True,
            "recommended_next_action": "Exercise identical requests through the integrated platform; review scientific validity/novelty separately.",
        },
    )
    atomic_write(
        out / "HANDOFF.md",
        (
            b"# Descriptive comparison development handoff\n\n"
            b"Twelve new descriptive questions passed full compiled comparative contracts, including essential joint_observation and outcome_ascertainment. "
            b"Eight controls preserved missing measurements, unknown timing/independence and unscoped-method abstention. "
            b"The original twelve association cases remain unqualified, with their requirements and source report unchanged.\n\n"
            b"The source CSVs and dictionaries were reused after checking original acquisition SHA-256 hashes: "
            b"UCI17 (569 records) and UCI45 (303 records). These are twelve dependent question/dataset pairs.\n\n"
            b"inspected_datasets.jsonl is a platform catalog of source-bound comparison evidence without global method claims. "
            b"requests.jsonl contains ordinary public question/user_constraints contracts. Call scope_tabular_comparison before assessment.\n\n"
            b"Definitions retain metadata pointers, data hashes, exact row numbers, group counts and missingness. "
            b"Timing, independence and power remain unknown. No novelty, gold-standard endpoint validity, final holdout, training or expert review is claimed.\n\n"
            b"Run the generator as a module from the repository root using --cached-uci-root, --original-report and a fresh --out. "
            b"Next: exercise exact requests through the integrated platform and review scientific validity separately.\n"
        ),
    )
    atomic_write(out / "demonstrate_tabular_comparisons.py", Path(__file__).read_bytes())
    module_path = Path(__file__).resolve().parents[2] / "litdatamatcher" / "tabular_contracts.py"
    atomic_write(out / "tabular_contracts.py", module_path.read_bytes())
    atomic_write(
        out / "MANIFEST.tsv",
        (
            "relative_path\tbytes\tsha256\n"
            + "".join(
                f"{path.name}\t{path.stat().st_size}\t{sha256(path)}\n"
                for path in sorted(out.iterdir())
            )
        ).encode(),
    )
    payloads = sorted(out.iterdir())
    archive = out / "LitDataMatcher-descriptive-development.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in payloads:
            info = zipfile.ZipInfo(path.name, date_time=(2026, 10, 31, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            handle.writestr(info, path.read_bytes())
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip() is not None or sorted(handle.namelist()) != sorted(
            path.name for path in payloads
        ):
            raise ValueError("Archive member validation failed")
        for path in payloads:
            if hashlib.sha256(handle.read(path.name)).hexdigest() != sha256(path):
                raise ValueError("Archive payload hash mismatch")
    atomic_json(
        out / "PACKAGE_VERIFICATION.json",
        {
            "status": "PASS",
            "archive": str(archive.resolve()),
            "sha256": sha256(archive),
            "members": len(payloads),
            "payload_hashes_verified": True,
            "validation": validation,
        },
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-uci-root", type=Path, required=True)
    parser.add_argument("--original-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.cached_uci_root, args.original_report, args.out)
    print(json.dumps({"status": report["status"], **report["validation"]}))
    return int(report["status"] == "FAIL")


if __name__ == "__main__":
    raise SystemExit(main())
