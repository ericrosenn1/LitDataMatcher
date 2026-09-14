#!/usr/bin/env python3
"""Measure compiler consistency with verbatim structured PICO source fields.

This is a development-only source-consistency audit.  Evidence Inference
relation annotations are used as an external-to-the-compiler comparison table,
not converted into requirements labels or treated as independent validation.
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


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical(value: str | None) -> str:
    # The compiler preserves a source span but removes terminal sentence
    # punctuation from a field value. Match that documented normalization only.
    return " ".join((value or "").casefold().split()).rstrip("?.;")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--split-assignments", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.out}")
    questions = {row["question_id"]: row for row in pq.read_table(args.corpus_root / "normalized" / "questions" / "questions.parquet").to_pylist()}
    assignments = pq.read_table(args.split_assignments).to_pylist()
    development_ids = {row["question_id"] for row in assignments if row["split"] == "development"}
    relations_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pq.read_table(args.corpus_root / "normalized" / "evidence_relations" / "evidence_relations.parquet").to_pylist():
        if row["question_id"] in development_ids:
            relations_by_question[row["question_id"]].append(row)

    args.out.mkdir(parents=True)
    detail_path = args.out / "structured_pico_consistency.jsonl"
    status_counts: Counter[str] = Counter()
    eligible = matched_questions = 0
    matched_fields: Counter[str] = Counter()
    explicit_fields: Counter[str] = Counter()
    with detail_path.open("w", encoding="utf-8") as handle:
        for question_id in sorted(development_ids):
            question = questions[question_id]
            compilation = compile_question(question["question_text"], source_locator=f"corpus-question:{question_id}")
            status_counts[compilation["compilation_status"]] += 1
            observed = {role["field"]: role["expected"] for role in compilation.get("entity_roles", []) if role["field"] in {"intervention", "comparator", "outcome"}}
            for field in observed:
                explicit_fields[field] += 1
            relations = relations_by_question.get(question_id, [])
            field_matches: dict[str, bool] = {}
            if relations:
                eligible += 1
                for field in ("intervention", "comparator", "outcome"):
                    field_matches[field] = field in observed and any(canonical(observed[field]) == canonical(row.get(field)) for row in relations)
                    if field_matches[field]:
                        matched_fields[field] += 1
                if all(field_matches.values()):
                    matched_questions += 1
            handle.write(json.dumps({"question_id": question_id, "question_sha256": hashlib.sha256(question["question_text"].encode()).hexdigest(), "compilation_status": compilation["compilation_status"], "source_explicit_roles": observed, "relation_count": len(relations), "relation_consistency": field_matches}, sort_keys=True) + "\n")

    receipt = {
        "schema_version": "v0.4-structured-pico-consistency-1.0",
        "created_utc": now(),
        "status": "PASS_WITH_LIMITATIONS",
        "scope": "development allocation only; sealed final holdout untouched",
        "compiler_input": "verbatim question text only; relation annotations were not passed to the compiler",
        "comparison": "Relation annotations are a source-consistency comparison, not full requirement gold labels or independent scientific validation.",
        "counts": {
            "development_questions": len(development_ids),
            "relation_eligible_questions": eligible,
            "compilation_statuses": dict(status_counts),
            "source_explicit_role_counts": dict(explicit_fields),
            "all_three_fields_match_any_source_relation": matched_questions,
            "field_match_counts": dict(matched_fields),
        },
        "limitations": [
            "Does not assess study-design compatibility, data availability, or candidate data-set suitability.",
            "Exact string agreement does not establish semantic equivalence beyond the source fields.",
            "No evaluation_candidate or sealed-holdout record was used for this development receipt.",
        ],
    }
    (args.out / "COMPILER_SOURCE_CONSISTENCY.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "counts": receipt["counts"], "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
