#!/usr/bin/env python3
"""Measure v0.4 development baselines without touching the sealed holdout.

The source labels used here are only question-paper edges and Evidence Inference
annotation fields.  They are never converted into fabricated full requirement
labels.  The model proposal interpreter remains a reviewed, verbatim-only
proposal baseline; it is not training data or an automatic product decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from litdatamatcher.question_compiler import compile_question
from litdatamatcher.semantic_runtime import LocalSemanticRuntime, RuntimeConfig


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def append(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def stable_sample(rows: list[dict[str, Any]], per_source: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_corpus"]].append(row)
    selected = []
    for source in sorted(grouped):
        selected.extend(sorted(grouped[source], key=lambda row: hashlib.sha256(row["question_id"].encode()).hexdigest())[:per_source])
    return sorted(selected, key=lambda row: row["question_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--split-assignments", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proposal-sample-per-source", type=int, default=4)
    parser.add_argument("--reuse-proposal-receipt", type=Path, help="Reuse a validated prior proposal-interpreter aggregate without rerunning local inference.")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite baseline receipt: {args.out}")
    if args.proposal_sample_per_source < 1:
        raise ValueError("proposal sample must be positive")

    question_path = args.corpus_root / "normalized" / "questions" / "questions.parquet"
    questions = {row["question_id"]: row for row in pq.read_table(question_path).to_pylist()}
    assignments = pq.read_table(args.split_assignments).to_pylist()
    development = [questions[row["question_id"]] for row in assignments if row["split"] == "development"]
    if not development:
        raise ValueError("Development allocation is empty")
    args.out.mkdir(parents=True)

    deterministic_path = args.out / "deterministic_compiler.jsonl"
    status_counts: Counter[str] = Counter()
    explicit_role_counts: Counter[str] = Counter()
    for row in development:
        compilation = compile_question(row["question_text"], source_locator=f"corpus-question:{row['question_id']}")
        status_counts[compilation["compilation_status"]] += 1
        # QUESTION_UNDERDETERMINED is an intentional abstention and does not
        # carry a role list.  Count it rather than treating abstention as an
        # execution failure.
        for role in compilation.get("entity_roles", []):
            explicit_role_counts[role["field"]] += 1
        append(deterministic_path, {"question_id": row["question_id"], "source_corpus": row["source_corpus"], "compilation": compilation})

    # Source label availability is a denominator/provenance report only.  It is
    # intentionally not a claim that relation annotations are gold requirements.
    relation_path = args.corpus_root / "normalized" / "evidence_relations" / "evidence_relations.parquet"
    relations = pq.read_table(relation_path, columns=["question_id", "intervention", "comparator", "outcome"]).to_pylist()
    development_ids = {row["question_id"] for row in development}
    source_label_availability = {
        field: len({row["question_id"] for row in relations if row["question_id"] in development_ids and (row.get(field) or "").strip()})
        for field in ("intervention", "comparator", "outcome")
    }

    if args.reuse_proposal_receipt:
        prior = json.loads(args.reuse_proposal_receipt.read_text(encoding="utf-8"))
        proposal_baseline = dict(prior["baselines"]["existing_proposal_interpreter"])
        proposal_baseline["reused_from"] = str(args.reuse_proposal_receipt)
        proposal_baseline["reuse_rationale"] = "The proposal interpreter and fixed development sample are unchanged; only the deterministic compiler was repaired."
    else:
        proposal_rows = stable_sample(development, args.proposal_sample_per_source)
        proposal_path = args.out / "proposal_interpreter.jsonl"
        progress_path = args.out / "progress.json"
        runtime = LocalSemanticRuntime(args.model, RuntimeConfig(device="cpu", max_new_tokens=128, max_attempts=1, cpu_threads=4))
        proposal_status: Counter[str] = Counter()
        for index, row in enumerate(proposal_rows, start=1):
            try:
                result = runtime.interpret_question(row["question_text"])
                status = result["inference_manifest"]["status"]
                payload: dict[str, Any] = {"question_id": row["question_id"], "source_corpus": row["source_corpus"], "question_sha256": hashlib.sha256(row["question_text"].encode()).hexdigest(), "result": result}
            except Exception as error:  # Preserve an exact failure class rather than retrying for a preferred answer.
                status = "ERROR_" + type(error).__name__
                payload = {"question_id": row["question_id"], "source_corpus": row["source_corpus"], "question_sha256": hashlib.sha256(row["question_text"].encode()).hexdigest(), "error_type": type(error).__name__, "error": str(error)}
            proposal_status[status] += 1
            append(proposal_path, payload)
            progress_path.write_text(json.dumps({"updated_utc": now(), "completed": index, "total": len(proposal_rows), "latest_question_id": row["question_id"], "proposal_status_counts": proposal_status}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        proposal_baseline = {"questions": len(proposal_rows), "sample_rule": f"stable {args.proposal_sample_per_source} questions per source corpus", "status_counts": dict(proposal_status), "scientific_status": "VERBATIM_PROPOSALS_REQUIRE_REVIEW"}

    receipt = {
        "schema_version": "v0.4-development-baselines-1.0",
        "created_utc": now(),
        "status": "PASS_WITH_LIMITATIONS",
        "scope": "development split only; sealed final holdout untouched",
        "inputs": {
            "questions": str(question_path.relative_to(args.corpus_root)),
            "split_assignments": str(args.split_assignments),
            "proposal_model": str(args.model),
        },
        "baselines": {
            "v0_3_question_only": {"generated_matching_contracts": 0, "interpretation": "Pre-compiler question-only input has no automatically generated requirement contract."},
            "v0_4_deterministic_compiler": {"questions": len(development), "compilation_status_counts": dict(status_counts), "source_explicit_role_counts": dict(explicit_role_counts)},
            "existing_proposal_interpreter": proposal_baseline,
            "source_annotation_availability": {"Evidence Inference relation-bearing development question counts": source_label_availability, "interpretation": "Availability only; annotation fields are not promoted to complete requirement labels."},
        },
        "limitations": [
            "This is a development measurement, not a held-out result.",
            "No learned retriever or compiler adapter is evaluated by this receipt.",
            "Question-paper relevance and relation fields do not constitute gold full data requirements.",
        ],
    }
    (args.out / "BASELINE_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "out": str(args.out), "proposal_questions": proposal_baseline.get("questions")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
