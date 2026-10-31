#!/usr/bin/env python3
"""Package existing development evidence for human scientific review, without labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from litdatamatcher.data_plane import atomic_json, atomic_write, digest
from litdatamatcher.expert_review import LABEL_VALUES, REVIEW_STATUS
from scripts.campaign_20261031.compare_case_batches import Snapshot, inside

VERSION = "development-scientific-review-1.0"
QUESTION_FIELDS = (
    "reviewer_id",
    "reviewed_at_utc",
    "question_validity",
    "unresolvedness",
    "requirement_assessment",
    "missing_requirements",
    "justification",
    "additional_source_locators",
)
CANDIDATE_FIELDS = (
    "reviewer_id",
    "reviewed_at_utc",
    "full_feasibility",
    "missing_requirement_assessment",
    "missing_requirements",
    "access_and_reuse_assessment",
    "justification",
    "additional_source_locators",
)
RESERVED = "GSE279879"


def _blank(fields):
    return {key: "" for key in fields}


def _write_rows(path, rows):
    atomic_write(
        path,
        b"".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True, allow_nan=False).encode() + b"\n"
            for row in rows
        ),
    )


def _write_csv(path, rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, stream.getvalue().encode("utf-8-sig"))


def validate_reviews(question_rows, candidate_rows):
    for rows, blank_fields in (
        (question_rows, QUESTION_FIELDS),
        (candidate_rows, CANDIDATE_FIELDS),
    ):
        keys = [row["review_item_id"] for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate review item identity")
        for row in rows:
            if any(row.get(key) != "" for key in blank_fields):
                raise ValueError("New review packet must not contain completed human fields")
            if not row.get("question", "").strip():
                raise ValueError("Exact review question missing")


def _dataset_evidence(row):
    allowed = (
        "dataset_id",
        "title",
        "source",
        "description",
        "summary",
        "url",
        "source_locator",
        "source_provenance",
        "metadata",
        "organisms",
        "assay_types",
        "capabilities",
        "availability",
        "access_type",
        "license",
        "independent_units",
        "independent_unit_status",
        "inspection",
        "inspection_status",
        "split_context",
    )
    result = {key: row[key] for key in allowed if key in row}
    if RESERVED in json.dumps(result):
        raise ValueError("Reserved evaluation study cannot enter the development review packet")
    return result


def _protocol():
    return """# Development scientific review protocol — unfilled

This packet contains 30 previously used development research cases and 12
processed-variable availability examples. It is not an untouched evaluation set.
The availability examples reuse two UCI tables and are not 12 independent datasets
or established unanswered questions. Historical machine requirements and decisions
are retained without recomputation, threshold changes, or scientific acceptance.
No final holdout is included or authorized to run.

Start with `review_cases.jsonl`, `dataset_evidence.jsonl`, `question_review.csv` and
`candidate_review.csv`. Each reviewer must save a separate copy, enter their real
identity and review date, inspect the source evidence, and fill the blank fields.
Record uncertainty instead of inventing unavailable measurements. A question's
source-paper association does not establish that it is unanswered. Original
PubMedQA/Evidence Inference development prompts may already have source answers.
Machine-generated requirements are objects for review, not reference truth.

Before consulting `machine_decisions.jsonl`, record judgments about question
validity, unresolvedness within the available literature coverage, necessary
experimental requirements, missing requirements and dataset feasibility. This
source-preserving packet is not a formally blinded trial: compiled requirements
are visible and are themselves under review. The sidecar separates machine
decisions to reduce avoidable anchoring. Preserve initial judgments before any
unblinded comparison or adjudication.

For full feasibility, use the project's existing compatibility vocabulary:
`exact_fit`, `directly_answerable`, `partial_fit`, `indirect_support`,
`requires_additional_data`, `incompatible`, `unknown`. For unresolvedness, use
`unresolved_in_coverage`, `answered_in_coverage`, `insufficient_coverage`, `uncertain`.
For question validity, use `valid`, `invalid`, `uncertain`. Requirement assessment
and missing-requirement assessment are free text; identify the exact requirement,
whether it is necessary/preferred/unsupported, the relevant source location, and
why it affects answerability. Do not change the machine's stored requirements.

Assess organism/system, assay/feature measurement, comparison groups, covariates,
sample units, repeated observations, temporal ordering, outcome ascertainment,
access/reuse restrictions and any claim of absence. A metadata record or public
file link is not proof of usable participant-level measurements. Sample columns
are not automatically independent subjects. Adequacy/power and causal claims
need their own evidence; unknowns are legitimate judgments.

Human decisions still needed for scientific validation:

1. Assign genuine domain reviewers and define an adjudication process before
   labeling. No reviewer identities or labels have been supplied by this generator.
2. Supply and justify independent case/pair judgments, including disagreements
   with source-stated gaps, inferred requirements and machine compatibility.
3. Decide how unresolved requirements, correlated cases, reused datasets and
   uncertain labels enter the evaluation denominator. Record these decisions;
   do not retrospectively redefine labels to improve metrics.
4. Finalize the prospective evaluation protocol, eligible reference cases,
   endpoints, source coverage, blinding/adjudication and acceptance criteria
   before accessing a genuinely untouched final set. This development packet
   cannot serve as that final set. The reserved GSE279879 remains untouched.

Generating or validating this packet establishes technical completeness only.
It does not establish expert validity, calibrated scores, publication readiness,
or scientific acceptance. Existing thresholds are unchanged; any future change
requires explicit scientific authorization and versioned documentation.
"""


def build(cases_root, batch_root, availability_root, out):
    if out.exists():
        raise ValueError("Choose a fresh output directory; existing reviews must be preserved")
    snapshot = Snapshot()
    manifest = snapshot.read(cases_root / "CASES_MANIFEST.json")
    expected = {row["path"]: row["sha256"] for row in manifest["outputs"]}
    cases = snapshot.read(cases_root / "cases.jsonl", expected["cases.jsonl"], lines=True)
    if len(cases) != 30 or any(case.get("split") != "development" for case in cases):
        raise ValueError("Expected exactly the 30 preserved development cases")
    state = snapshot.read(batch_root / "BATCH_STATE.json")
    if set(state["cases"]) != {case["case_id"] for case in cases}:
        raise ValueError("Native batch and frozen case identities differ")
    reviews, decisions, questions, pairs, evidence = [], [], [], [], {}
    native_candidate_count = native_question_count = 0

    def add_question(case_id, question_id, text, cohort):
        identifier = "review-question:" + digest([case_id, question_id, text])[:24]
        questions.append(
            {
                "review_item_id": identifier,
                "case_id": case_id,
                "question_id": question_id,
                "cohort": cohort,
                "question": text,
                **_blank(QUESTION_FIELDS),
            }
        )
        return identifier

    def add_candidate(case_id, question_id, text, row, dataset, cohort):
        dataset_id = row["dataset_id"]
        observed = _dataset_evidence(dataset)
        evidence_id = "dataset-evidence:" + digest(observed)[:24]
        evidence.setdefault(evidence_id, {"evidence_id": evidence_id, **observed})
        identifier = "review-candidate:" + digest([case_id, question_id, dataset_id])[:24]
        pairs.append(
            {
                "review_item_id": identifier,
                "case_id": case_id,
                "question_id": question_id,
                "cohort": cohort,
                "question": text,
                "dataset_id": dataset_id,
                "dataset_evidence_id": evidence_id,
                **_blank(CANDIDATE_FIELDS),
            }
        )
        return {
            "review_item_id": identifier,
            "question_id": question_id,
            "dataset_id": dataset_id,
            "dataset_evidence_id": evidence_id,
        }

    for case in cases:
        case_id, cohort = case["case_id"], "source_linked_development"
        entry = state["cases"][case_id]
        if not entry.get("terminal") or not entry.get("run_dir"):
            raise ValueError("Native case is not a closed preserved result")
        run_dir = inside(batch_root, entry["run_dir"])
        receipts = {row["path"]: row["sha256"] for row in entry["outputs"]}

        def load(name, run_dir=run_dir, receipts=receipts):
            path = run_dir / name
            relative = path.relative_to(batch_root).as_posix()
            if relative not in receipts:
                raise ValueError("Native artifact lacks a recorded output hash: " + relative)
            return snapshot.read(path, receipts[relative], lines=True)

        native_questions, compilations, matches = (
            load("questions.jsonl"),
            load("question_compilations.jsonl"),
            load("matches.jsonl"),
        )
        outcomes, bundles = load("candidate_outcomes.jsonl"), load("evidence_bundles.jsonl")
        selected_ids = {row["dataset_id"] for row in matches}
        dataset_rows = load("datasets.jsonl")
        selected = {
            row["dataset_id"]: row for row in dataset_rows if row["dataset_id"] in selected_ids
        }
        if set(selected) != selected_ids:
            raise ValueError("A native candidate lacks its observed dataset record")
        native_by_id = {row["question_id"]: row for row in native_questions}
        if not any(
            row["question"] == case["question"] and row.get("origin") == "user"
            for row in native_questions
        ):
            raise ValueError("Preserved caller question differs from frozen case")
        if {row["question_id"] for row in compilations} != set(native_by_id):
            raise ValueError("Native questions and full compilations differ")
        question_refs = [
            add_question(case_id, row["question_id"], row["question"], cohort)
            for row in native_questions
        ]
        candidate_refs = [
            add_candidate(
                case_id,
                row["question_id"],
                native_by_id[row["question_id"]]["question"],
                row,
                selected[row["dataset_id"]],
                cohort,
            )
            for row in matches
        ]
        source = case["source_document"]
        if not source.get("id") or not source.get("text"):
            raise ValueError("Frozen source identity/text missing")
        reviews.append(
            {
                "case_id": case_id,
                "cohort": cohort,
                "exact_input_question": case["question"],
                "input_question_origin": case["source_corpus"],
                "source_document": source,
                "question_records": [
                    {
                        key: row[key]
                        for key in (
                            "question_id",
                            "question",
                            "origin",
                            "source_document_id",
                            "evidence_span",
                            "source_locator",
                            "text",
                            "quote",
                        )
                        if key in row
                    }
                    for row in native_questions
                ],
                "full_compiled_requirements": compilations,
                "question_review_ids": question_refs,
                "candidate_refs": candidate_refs,
                "review_status": REVIEW_STATUS,
            }
        )
        decisions.append(
            {
                "case_id": case_id,
                "cohort": cohort,
                "run_status": entry["status"],
                "machine_question_records": native_questions,
                "candidate_decisions": matches,
                "candidate_outcomes": outcomes,
                "evidence_bundles_with_machine_interpretations": bundles,
                "origin": "preserved_native_source_literal_run_not_expert_labels",
            }
        )
        native_candidate_count += len(matches)
        native_question_count += len(native_questions)
    if native_candidate_count != 140:
        raise ValueError("Expected the 140 preserved native candidate rows")

    available = snapshot.read(availability_root / "QUALIFICATION_DEMONSTRATION.json")
    processed = snapshot.read(availability_root / "inspected_datasets.jsonl", lines=True)
    by_id = {row["dataset_id"]: row for row in processed}
    if len(available["cases"]) != 12:
        raise ValueError("Expected exactly twelve processed-availability cases")
    for case in available["cases"]:
        case_id, cohort = case["case_id"], "processed_variable_availability_development"
        question_id = "availability-question:" + digest(case["question"])[:24]
        selected_id = case["selected_dataset"]
        row = next(row for row in case["ranked_candidates"] if row["dataset_id"] == selected_id)
        question_ref = add_question(case_id, question_id, case["question"], cohort)
        candidate_ref = add_candidate(
            case_id, question_id, case["question"], row, by_id[selected_id], cohort
        )
        reviews.append(
            {
                "case_id": case_id,
                "cohort": cohort,
                "exact_input_question": case["question"],
                "input_question_origin": case["question_origin"],
                "source_document": None,
                "source_note": "No literature-unresolvedness claim; this is an explicit processed-variable availability example",
                "full_compiled_requirements": [case["compilation"]],
                "question_review_ids": [question_ref],
                "candidate_refs": [candidate_ref],
                "review_status": REVIEW_STATUS,
            }
        )
        decisions.append(
            {
                "case_id": case_id,
                "cohort": cohort,
                "candidate_decisions": [row],
                "availability_only_decisions": case["availability_only_candidates"],
                "matched_source_rows": case["matched_source_rows"],
                "origin": "deterministic_source_variable_availability_not_expert_labels",
            }
        )
    validate_reviews(questions, pairs)
    if len(reviews) != 42 or len(pairs) != 152 or len(questions) != native_question_count + 12:
        raise ValueError("Review packet counts do not reconcile")
    snapshot.unchanged()
    out.mkdir(parents=True)
    _write_rows(out / "review_cases.jsonl", reviews)
    _write_rows(
        out / "dataset_evidence.jsonl",
        sorted(evidence.values(), key=lambda row: row["evidence_id"]),
    )
    _write_rows(out / "machine_decisions.jsonl", decisions)
    _write_csv(out / "question_review.csv", questions, list(questions[0]))
    _write_csv(out / "candidate_review.csv", pairs, list(pairs[0]))
    atomic_write(out / "PROTOCOL.md", _protocol().encode("utf-8"))
    atomic_write(out / "prepare_scientific_review.py", Path(__file__).read_bytes())
    atomic_json(
        out / "LABEL_VOCABULARY.json", {key: sorted(values) for key, values in LABEL_VALUES.items()}
    )
    atomic_json(
        out / "SOURCE_MANIFEST.json",
        [
            {"path": str(path), "sha256": checksum, "bytes": path.stat().st_size}
            for path, checksum in sorted(snapshot.hashes.items())
        ],
    )
    validation = {
        "schema_version": VERSION,
        "technical_status": "PASS",
        "review_status": REVIEW_STATUS,
        "cases": len(reviews),
        "case_cohorts": dict(Counter(row["cohort"] for row in reviews)),
        "question_review_rows": len(questions),
        "candidate_review_rows": len(pairs),
        "native_questions": native_question_count,
        "native_candidate_rows": native_candidate_count,
        "distinct_dataset_evidence_records": len(evidence),
        "distinct_dataset_accessions": len({row["dataset_id"] for row in evidence.values()}),
        "source_files_hash_verified": len(snapshot.hashes),
        "reviewer_fields_filled": 0,
        "labels_generated": 0,
        "sealed_holdout_accessed": False,
        "scientific_thresholds_changed": False,
        "models_rerun": False,
        "source_inputs_modified": False,
        "limitations": [
            "Development-only review; not independent final evaluation.",
            "42 cases are not 42 independent data assets.",
            "Native machine decisions and zero full qualifications are not reference truth.",
            "Reviewer assignment, labels, adjudication and prospective final-evaluation protocol remain human work.",
        ],
    }
    atomic_json(out / "PACKET_VALIDATION.json", validation)
    atomic_json(
        out / "task_state.json",
        {
            "task_name": "development scientific review packet",
            "task_status": "PASS_WITH_LIMITATIONS",
            "inputs": ["SOURCE_MANIFEST.json"],
            "outputs": ["MANIFEST.tsv"],
            "items_requested": 42,
            "items_processed": 42,
            "items_succeeded": 42,
            "items_failed": 0,
            "validation_status": "PASS",
            "audit_status": "PASS",
            "repair_cycles": 0,
            "repairs_performed": [],
            "unresolved_issues": [
                "Genuine reviewer assignment, labels and adjudication remain pending",
                "Prospective final-evaluation protocol has not been finalized or executed",
            ],
            "scientific_assumptions_changed": False,
            "requires_human_review": True,
            "recommended_next_action": "Assign genuine reviewers and complete blank development review forms before prospective final-evaluation protocol approval.",
        },
    )
    payload = sorted(path for path in out.iterdir() if path.is_file())
    lines = ["relative_path\tfile_size_bytes\tsha256\trole\tstatus"]
    for path in payload:
        lines.append(
            f"{path.name}\t{path.stat().st_size}\t{hashlib.sha256(path.read_bytes()).hexdigest()}\treview_packet\tVALIDATED"
        )
    atomic_write(out / "MANIFEST.tsv", ("\n".join(lines) + "\n").encode())
    payload.append(out / "MANIFEST.tsv")
    archive = out / "LitDataMatcher-development-scientific-review.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in payload:
            bundle.write(path, path.name)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None or set(bundle.namelist()) != {
            path.name for path in payload
        }:
            raise ValueError("Review archive failed integrity or membership validation")
        for path in payload:
            if (
                hashlib.sha256(bundle.read(path.name)).hexdigest()
                != hashlib.sha256(path.read_bytes()).hexdigest()
            ):
                raise ValueError("Archive content differs from validated packet")
    result = {
        **validation,
        "archive": str(archive),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "archive_bytes": archive.stat().st_size,
        "archive_members": len(payload),
        "archive_verified": True,
    }
    atomic_json(out / "PACKAGE_VERIFICATION.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--availability", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.cases, args.batch, args.availability, args.out)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "technical_status",
                    "cases",
                    "question_review_rows",
                    "candidate_review_rows",
                    "archive",
                    "archive_sha256",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
