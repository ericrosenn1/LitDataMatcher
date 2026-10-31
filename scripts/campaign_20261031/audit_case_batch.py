"""Verify native batch artifacts and literal source grounding independently.

This checks computational provenance and spans, not expert scientific validity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        result = [strict_json(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in result):
        raise ValueError("JSONL records must be objects")
    return result


def strict_json(text):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"Duplicate JSON key: {key}")
            value[key] = item
        return value

    def reject(value):
        raise ValueError(f"Invalid JSON number: {value}")

    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite JSON number")
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)

    value = json.loads(text, object_pairs_hook=unique, parse_constant=reject)
    finite(value)
    return value


def audit(batch: Path, inputs: Path, out: Path):
    batch = batch.resolve()
    state_path = batch / "BATCH_STATE.json"
    state = strict_json(state_path.read_text(encoding="utf-8"))
    if state["inputs"]["input"]["sha256"] != file_sha(inputs):
        raise ValueError("Original batch input differs from the execution receipt")
    original = rows(inputs)
    expected_ids = {row["case_id"] for row in original}
    if not original or len(expected_ids) != len(original) or set(state["cases"]) != expected_ids:
        raise ValueError("Batch cases differ from the frozen input set")
    if len({str(value).casefold() for value in expected_ids}) != len(expected_ids):
        raise ValueError("Case IDs differ only in letter case")
    for case_id in expected_ids:
        if (not isinstance(case_id, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", case_id)
                or case_id.endswith(".")
                or case_id.split(".", 1)[0].upper() in
                {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)],
                 *[f"LPT{i}" for i in range(1, 10)]}):
            raise ValueError("Unsafe case ID")
    expected_sources = {}
    for row in original:
        source = row.get("source_document", row)
        expected_sources[row["case_id"]] = (
            source.get("document_id") or source.get("id") or source.get("source_id"),
            source.get("text") or source.get("abstract"),
        )
    results, checked_artifacts = [], 0
    for case_id, entry in state["cases"].items():
        if not entry.get("terminal"):
            raise ValueError(f"Case execution is incomplete: {case_id}")
        run = (batch / entry["run_dir"]).resolve()
        if not run.is_relative_to(batch / "cases" / case_id):
            raise ValueError("Case output path escapes the batch")
        covered = set()
        for receipt in entry.get("outputs", []):
            path = (batch / receipt["path"]).resolve()
            if not path.is_relative_to(run) or file_sha(path) != receipt["sha256"]:
                raise ValueError(f"Case artifact checksum failure: {case_id}")
            if path in covered:
                raise ValueError(f"Duplicate artifact receipt: {case_id}")
            covered.add(path)
            checked_artifacts += 1
        payloads = {run / name for name in ("claims.jsonl", "questions.jsonl", "matches.jsonl", "candidate_outcomes.jsonl")}
        required = payloads | {run / "RUN_MANIFEST.json", run / ".workflow/state.json"}
        if not required <= covered:
            raise ValueError(f"Required case artifact receipts missing: {case_id}")
        manifest = strict_json((run / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
        manifest_covered = set()
        for item in manifest["artifacts"]:
            path = (run / item["path"]).resolve()
            if not path.is_relative_to(run) or file_sha(path) != item["sha256"]:
                raise ValueError(f"Native manifest checksum failure: {case_id}")
            if path in manifest_covered:
                raise ValueError(f"Duplicate native artifact receipt: {case_id}")
            manifest_covered.add(path)
        if not payloads <= manifest_covered:
            raise ValueError(f"Required native payload receipts missing: {case_id}")
        workflow = strict_json((run / ".workflow/state.json").read_text(encoding="utf-8"))
        if workflow.get("status") != "COMPLETE" or not workflow.get("completed"):
            raise ValueError(f"Workflow completion is unverified: {case_id}")
        stages = {}
        for name, receipt in workflow["stages"].items():
            path = (run / ".workflow" / receipt["file"]).resolve()
            if path.parent != run / ".workflow" or file_sha(path) != receipt["sha256"]:
                raise ValueError(f"Stage checksum failure: {case_id}/{name}")
            stages[name] = strict_json(path.read_text(encoding="utf-8"))
        ingestion = stages.get("local_ingestion", {})
        for receipt in ingestion.get("outputs", []):
            path = (run / "local_ingestion" / receipt["name"]).resolve()
            if (path.parent != run / "local_ingestion"
                    or file_sha(path) != receipt["identity"]["sha256"]):
                raise ValueError(f"Ingestion checksum failure: {case_id}")
        source_documents = rows(run / "local_ingestion/literature.jsonl")
        if ingestion.get("records") != source_documents or not ingestion.get("outputs"):
            raise ValueError(f"Ingestion checkpoint is incomplete: {case_id}")
        expected_id, expected_text = expected_sources[case_id]
        if (len(source_documents) != 1 or source_documents[0].get("document_id") != expected_id
                or source_documents[0].get("text") != expected_text):
            raise ValueError(f"Ingested source differs from frozen input: {case_id}")
        by_id = {row["document_id"]: row["text"] for row in source_documents}
        claims = rows(run / "claims.jsonl")
        for claim in claims:
            text = by_id[claim["source_document_id"]]
            span = claim["evidence_span"]
            if (type(span.get("start")) is not int or type(span.get("end")) is not int
                    or not 0 <= span["start"] < span["end"] <= len(text)):
                raise ValueError(f"Claim source span bounds invalid: {case_id}")
            if text[span["start"]:span["end"]] != span["text"] or span["text"] != claim["quote"]:
                raise ValueError(f"Claim source span failure: {case_id}")
            cursor = 0
            for role in ("subject", "predicate", "object"):
                phrase = claim[role]
                if not isinstance(phrase, str) or not phrase.strip():
                    raise ValueError(f"Claim role must be nonempty text: {case_id}")
                start = claim["quote"].find(phrase, cursor)
                if not phrase or start < cursor:
                    raise ValueError(f"Claim role absent or out of order: {case_id}")
                cursor = start + len(phrase)
        questions = rows(run / "questions.jsonl")
        matches = rows(run / "matches.jsonl")
        outcomes = rows(run / "candidate_outcomes.jsonl")
        results.append({"case_id": case_id, "status": manifest["execution_status"],
                        "accepted_claims": len(claims), "questions": len(questions),
                        "matches": len(matches), "failures": manifest["failures"],
                        "outcomes": dict(Counter(row["classification"] for row in outcomes)),
                        "manifest_sha256": file_sha(run / "RUN_MANIFEST.json")})
    report = {"status": "PASS_ARTIFACT_AND_SPAN_AUDIT", "cases": len(results),
              "artifacts_checked": checked_artifacts,
              "input_sha256": file_sha(inputs), "batch_state_sha256": file_sha(state_path),
              "script_sha256": file_sha(__file__),
              "execution_statuses": dict(Counter(row["status"] for row in results)),
              "accepted_claims": sum(row["accepted_claims"] for row in results),
              "case_results": results,
              "limitations": "Literal/source/byte validation only; no expert accuracy or dataset suitability claim."}
    if out.exists():
        raise ValueError("Audit output exists; preserve it and choose a new path")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.batch, args.inputs, args.out)
    print(json.dumps({key: report[key] for key in ("status", "cases", "artifacts_checked", "execution_statuses", "accepted_claims")}))


if __name__ == "__main__":
    main()
