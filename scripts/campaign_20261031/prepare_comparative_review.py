"""Create an additive blank review worksheet for descriptive development cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

from litdatamatcher.data_plane import atomic_json, atomic_write
from litdatamatcher.expert_review import LABEL_VALUES

REVIEW_FIELDS = (
    "reviewer_id",
    "reviewed_at_utc",
    "question_validity",
    "full_feasibility",
    "missing_requirement_assessment",
    "missing_requirements",
    "source_definition_assessment",
    "independence_and_power_assessment",
    "access_and_reuse_assessment",
    "unresolvedness",
    "novelty",
    "literature_coverage",
    "justification",
    "additional_source_locators",
    "adjudication_needed",
)
PROTOCOL = """# Additive descriptive development review

This packet covers 12 NEW descriptive question/dataset pairs using two existing
public tables. It supplements the frozen 42-case review packet without modifying
it. All reviewer fields are blank. These are development examples, not untouched
final evaluation, independent studies or newly discovered unresolved questions.

Review the worksheet and context before opening the machine-decision sidecar.
Formal blinding is not claimed: compiled contracts and source metadata remain
visible. Save a separate copy per genuine reviewer. Do not fill labels by copying
machine eligibility. The development feasibility matches do not require invented
expert acceptance; independent scientific validation does require real judgments.

Assess TWO distinct dimensions:

1. Descriptive feasibility: Are the exact source columns, literal group codes,
   joint rows, source definitions and complete compiled requirements sufficient
   for the stated recorded-value comparison? Assess missing requirements,
   measurement validity, source access and reuse. Timing is not established.
   Counted rows do not establish participant independence, power or precision.
2. Unansweredness/novelty: Is this exact question already answered within a
   documented literature search? A feasible table cannot answer this judgment.
   Record coverage, date, searches and supporting citations. The questions were
   explicitly authored for development; they are not source-stated research gaps.

Use existing LABEL_VOCABULARY.json. full_feasibility uses dataset_compatibility;
unresolvedness and novelty use novelty. Other assessment fields accept reviewer
prose. No defaults or suggested labels have been prefilled. Explain judgments,
including uncertainty or missing information, and retain disagreement for genuine
adjudication rather than automatically forcing consensus.

Independent inputs still needed:
- Real domain reviewer identity, review date and an adjudication process.
- Inspection of the hash-identified source CSV/dictionary and exact definition
  mappings; provided pointers locate the evidence but do not replace judgment.
- A study-specific estimand, inferential plan, missingness policy, independence
  assessment and precision/power rationale when judging inferential adequacy.
- Coverage-bounded literature evidence for unansweredness/novelty, including
  search date, corpus/query scope, source citations and prior related analyses.
- Source-specific access/reuse interpretation for the intended analysis.

SOURCE_MANIFEST.json and per-case evidence pointers identify external source
files without duplicating raw data. The machine sidecar contains only the prior
development execution decisions; current integrated API acceptance is separate.
No model training, expert labeling, threshold change or sealed evaluation occurs
here. This packet must never be presented as final-evaluation evidence. The
original association cases and frozen review packet remain unchanged.
"""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path, role):
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "role": role,
    }


def build(comparison_root, frozen_review_zip, out):
    if out.exists():
        raise ValueError("Use a fresh output directory; review packets are immutable")
    source_report = comparison_root / "COMPARISON_DEVELOPMENT.json"
    with (comparison_root / "MANIFEST.tsv").open(encoding="utf-8", newline="") as handle:
        manifest = {row["relative_path"]: row for row in csv.DictReader(handle, delimiter="\t")}
    inputs = []
    for name in ("COMPARISON_DEVELOPMENT.json", "inspected_datasets.jsonl", "requests.jsonl"):
        path = comparison_root / name
        entry = manifest[name]
        if sha256(path) != entry["sha256"] or path.stat().st_size != int(entry["bytes"]):
            raise ValueError("Development input differs from its manifest: " + name)
        inputs.append(record(path, "development_evidence"))
    report = json.loads(source_report.read_text(encoding="utf-8"))
    if len(report["cases"]) != 12 or len({case["case_id"] for case in report["cases"]}) != 12:
        raise ValueError("Expected twelve distinct descriptive development cases")
    for source in report["source_receipts"]:
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError("Original source input differs from acquisition receipt")
        inputs.append({**source, "role": "external_raw_source_not_duplicated"})
    frozen = record(frozen_review_zip, "frozen_prior_42_case_packet_not_modified")
    rows, contexts, machine = [], [], []
    for index, case in enumerate(report["cases"]):
        evidence = case["observed_evidence"]
        scope = {
            key: evidence[key] for key in ("outcome_variable", "grouping_variable", "group_values")
        }
        pointer = str(source_report.resolve()) + f"#/cases/{index}"
        dataset_id = case["selected_dataset"]
        blank = {field: "" for field in REVIEW_FIELDS}
        rows.append(
            {
                "case_id": case["case_id"],
                "question": case["question"],
                "dataset_id": dataset_id,
                "exact_scope_json": json.dumps(scope, sort_keys=True),
                "evidence_pointer": pointer + "/observed_evidence",
                "source_data_locator": evidence["source_locator"],
                "compilation_pointer": pointer + "/compilation",
                **blank,
            }
        )
        contexts.append(
            {
                "case_id": case["case_id"],
                "question": case["question"],
                "dataset_id": dataset_id,
                "question_origin": case["question_origin"],
                "scope": scope,
                "source_dataset_url": "https://archive.ics.uci.edu/dataset/"
                + dataset_id.split(":")[1],
                "full_compilation": case["compilation"],
                "evidence_pointer": pointer + "/observed_evidence",
                "source_sha256": evidence["source_sha256"],
                "metadata_sha256": evidence["metadata_sha256"],
                "definition_evidence": evidence["definition_evidence"],
                "joint_rows_by_group": evidence["joint_rows_by_group"],
                "joint_observed_rows": evidence["joint_observed_rows"],
                "selected_variable_unobserved_cells": evidence[
                    "selected_variable_unobserved_cells"
                ],
                "source_row_indices_pointer": pointer
                + "/observed_evidence/source_data_row_numbers",
                "source_row_number_convention": evidence["source_row_number_convention"],
                "reviewer_fields": blank,
            }
        )
        selected = next(row for row in case["ranked_candidates"] if row["dataset_id"] == dataset_id)
        machine.append(
            {
                "case_id": case["case_id"],
                "decision_origin": "descriptive_development_execution_not_expert_label",
                "result": selected,
            }
        )
    out.mkdir(parents=True)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(out / "reviewer_worksheet.csv", stream.getvalue().encode())
    for name, data in [("review_context.jsonl", contexts), ("machine_decisions.jsonl", machine)]:
        atomic_write(
            out / name,
            b"".join(
                json.dumps(row, sort_keys=True, allow_nan=False).encode() + b"\n" for row in data
            ),
        )
    atomic_write(out / "PROTOCOL.md", PROTOCOL.encode())
    atomic_json(
        out / "LABEL_VOCABULARY.json", {key: sorted(value) for key, value in LABEL_VALUES.items()}
    )
    atomic_json(out / "SOURCE_MANIFEST.json", {"inputs": inputs, "frozen_prior_packet": frozen})
    validation = {
        "worksheet_rows": len(rows),
        "context_rows": len(contexts),
        "machine_decision_rows": len(machine),
        "distinct_datasets": len({row["dataset_id"] for row in rows}),
        "reviewer_fields_per_row": len(REVIEW_FIELDS),
        "filled_reviewer_fields": 0,
        "frozen_prior_packet_unchanged": sha256(frozen_review_zip) == frozen["sha256"],
        "input_hashes_verified": len(inputs),
        "raw_data_duplicated": False,
    }
    with (out / "reviewer_worksheet.csv").open(encoding="utf-8", newline="") as handle:
        checked = list(csv.DictReader(handle))
    if len(checked) != 12 or any(row[field] for row in checked for field in REVIEW_FIELDS):
        raise ValueError("Worksheet is incomplete or contains a reviewer label")
    if not validation["frozen_prior_packet_unchanged"]:
        raise ValueError("Frozen packet changed during generation")
    if any(sha256(item["path"]) != item["sha256"] for item in inputs):
        raise ValueError("An input changed during review preparation")
    atomic_json(out / "VALIDATION.json", {"status": "PASS", **validation})
    atomic_json(
        out / "task_state.json",
        {
            "task_name": "additive descriptive review worksheet",
            "task_status": "PASS_WITH_LIMITATIONS",
            "inputs": inputs,
            "outputs": ["reviewer_worksheet.csv", "review_context.jsonl", "PROTOCOL.md"],
            "items_requested": 12,
            "items_processed": 12,
            "items_succeeded": 12,
            "items_failed": 0,
            "validation_status": "PASS",
            "audit_status": "PASS",
            "repair_cycles": 0,
            "repairs_performed": [],
            "unresolved_issues": [
                "Reviewer assignment, scientific judgments, literature coverage and adjudication remain pending"
            ],
            "scientific_assumptions_changed": False,
            "requires_human_review": True,
            "recommended_next_action": "Assign genuine reviewers; preserve descriptive feasibility separately from unansweredness and novelty",
        },
    )
    atomic_write(out / "prepare_comparative_review.py", Path(__file__).read_bytes())
    atomic_write(
        out / "HANDOFF.md",
        b"# Review addendum handoff\n\nTwelve blank descriptive-development reviews supplement the frozen prior packet. Read PROTOCOL.md, then reviewer_worksheet.csv and review_context.jsonl. External source files are hash-identified in SOURCE_MANIFEST.json; no raw tables are duplicated. Machine decisions are separate from blank reviewer fields. Technical preparation passed; independent reviewers, literature coverage and scientific decisions remain pending. Reproduce with the included generator using --comparison-root, --frozen-review-zip and a fresh --out. No final evaluation was performed.\n",
    )
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
    archive = out / "LitDataMatcher-descriptive-review-addendum.zip"
    with zipfile.ZipFile(archive, "w") as handle:
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
                raise ValueError("Archive member hash mismatch")
    receipt = {
        "status": "PASS_WITH_LIMITATIONS",
        "validation": validation,
        "archive": record(archive, "review_addendum"),
        "members": len(payloads),
        "frozen_prior_packet": frozen,
    }
    atomic_json(out / "PACKAGE_VERIFICATION.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--frozen-review-zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.comparison_root, args.frozen_review_zip, args.out)))


if __name__ == "__main__":
    main()
