"""Compare closed development batch snapshots without running models or promoting one.

Counts describe accepted model output, not accuracy. Caller questions and
deterministic source extraction are excluded from model coverage. Missing
inference observations remain missing, including in otherwise closed batches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path, PureWindowsPath

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from litdatamatcher.analysis_batch import load_batch_inputs  # noqa: E402
from litdatamatcher.data_plane import canonical_json, digest  # noqa: E402
from scripts.campaign_20261031.audit_case_batch import strict_json  # noqa: E402


class Snapshot:
    """Hash the bytes actually read, then reject changes before publication."""

    def __init__(self):
        self.hashes = {}

    def read(self, path, expected=None, *, lines=False):
        path = Path(path).resolve(strict=True)
        raw = path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        if expected is not None and checksum != expected:
            raise ValueError(f"Artifact checksum mismatch: {path}")
        if path in self.hashes and self.hashes[path] != checksum:
            raise ValueError(f"Input changed during comparison: {path}")
        self.hashes[path] = checksum
        text = raw.decode("utf-8-sig")
        value = (
            [strict_json(line) for line in text.split("\n") if line.strip()]
            if lines
            else strict_json(text)
        )
        if lines and any(not isinstance(row, dict) for row in value):
            raise ValueError(f"Expected JSONL object records: {path}")
        return value

    def unchanged(self):
        for path, expected in self.hashes.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Input changed during comparison: {path}")


def inside(root, relative):
    if not isinstance(relative, str) or "\\" in relative or PureWindowsPath(relative).drive:
        raise ValueError("Unsafe relative artifact path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe relative artifact path")
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Artifact escapes its root")
    return target


def median(values):
    return statistics.median(values) if values else None


def seconds(value):
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("Inference timing must be finite, nonnegative seconds or missing")
    return value


def variants(values):
    return [value for _, value in sorted({digest(value): value for value in values}.items())]


def model_records(records, key, inference_ids):
    found = {}
    for row in records:
        nested = row.get("source_question_records", [])
        candidates = [row, *nested]
        for item in candidates:
            if item.get("origin") == "user" or not item.get("inference_fingerprint"):
                continue
            if digest(item["inference_fingerprint"]) not in inference_ids:
                raise ValueError("Model record has no matching returned inference")
            identifier = item.get(key)
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("Model record lacks an identity")
            # A merged question may retain both its enriched public record and
            # the original model record. Count the source identity once.
            identity = {
                name: item.get(name)
                for name in (
                    "inference_fingerprint",
                    "evidence_span",
                    "source_document_id",
                    "source_view_id",
                )
            }
            if identifier in found and found[identifier] != identity:
                raise ValueError("Conflicting duplicate model record")
            found[identifier] = identity
    return len(found)


def read_case(batch, case, entry, snapshot):
    case_id = case["case_id"]
    result = {
        "status": entry["status"],
        "terminal": entry.get("terminal", False),
        "retryable": not entry.get("terminal", False),
        "attempts": [
            {
                k: a[k]
                for k in ("status", "terminal", "error", "started_utc", "finished_utc", "resume")
                if k in a
            }
            for a in entry.get("attempts", [])
        ],
        "manifest_status": None,
        "workflow_status": None,
        "attempt_errors": [a["error"] for a in entry.get("attempts", []) if "error" in a],
        "technical_failures": [],
        "guard_failures": [],
        "model_claims": None,
        "model_questions": None,
        "fully_observed_empty": None,
        "no_accepted_model_records": None,
        "views": None,
        "published_claims": None,
        "published_question_origins": None,
        "models": [],
        "backend_identities": [],
    }
    materialized = batch / "inputs" / f"{case_id}.jsonl"
    snapshot.read(materialized, lines=True)
    if materialized.read_bytes() != canonical_json(case["document"]) + b"\n":
        raise ValueError(f"Materialized input differs from frozen case: {case_id}")
    if not entry.get("outputs"):
        return result
    run = inside(batch, entry["run_dir"])
    if not run.is_relative_to(batch / "cases" / case_id):
        raise ValueError("Run directory escapes its case")
    receipts = {}
    for receipt in entry["outputs"]:
        path = inside(batch, receipt["path"])
        if not path.is_relative_to(run) or path in receipts:
            raise ValueError("Invalid or duplicate case artifact receipt")
        receipts[path] = receipt["sha256"]

    def payload(name, lines=False):
        path = inside(run, name)
        if path not in receipts:
            raise ValueError(f"Missing artifact receipt: {case_id}/{name}")
        return snapshot.read(path, receipts[path], lines=lines)

    manifest = payload("RUN_MANIFEST.json")
    workflow = payload(".workflow/state.json")
    if workflow.get("fingerprint") != digest(workflow["inputs"]):
        raise ValueError("Workflow input fingerprint mismatch")
    if workflow["inputs"]["document"]["sha256"] != snapshot.hashes[materialized.resolve()]:
        raise ValueError("Workflow is bound to a different case input")
    if manifest["execution_status"] != entry["status"]:
        raise ValueError("Batch and native execution statuses disagree")
    if bool(workflow.get("completed")) != entry["terminal"]:
        raise ValueError("Batch and checkpoint completion markers disagree")
    result["manifest_status"] = manifest["execution_status"]
    result["workflow_status"] = workflow.get("status")
    result["models"] = manifest.get("models", [])
    failures = manifest.get("failures", [])
    result["guard_failures"] = [item for item in failures if item.get("stage") == "source_guard"]
    result["technical_failures"] = [
        item for item in failures if item.get("stage") != "source_guard"
    ]
    inferences = payload("inferences.jsonl", True)
    claims = payload("claims.jsonl", True)
    questions = payload("questions.jsonl", True)
    views = payload("source_views.jsonl", True)
    result["published_claims"] = len(claims)
    result["published_question_origins"] = dict(
        Counter(q.get("origin", "UNKNOWN") for q in questions)
    )
    native_receipts = {}
    for item in manifest["artifacts"]:
        path = inside(run, item["path"])
        if path in native_receipts:
            raise ValueError("Duplicate native artifact receipt")
        native_receipts[path] = item["sha256"]
    for name in ("inferences.jsonl", "claims.jsonl", "questions.jsonl", "source_views.jsonl"):
        path = run / name
        if native_receipts.get(path) != receipts[path]:
            raise ValueError("Native and batch artifact receipts disagree")

    stages = workflow.get("stages", {})

    def stage(name):
        receipt = stages[name]
        return snapshot.read(inside(run / ".workflow", receipt["file"]), receipt["sha256"])

    returned = {digest(i["fingerprint"]): i for i in inferences}
    if len(returned) != len(inferences):
        raise ValueError("Duplicate returned inference identity")
    view_rows, seen = [], set()
    for view in views:
        view_id = view["view_id"]
        if view_id in seen or view["document_id"] != case["document"]["document_id"]:
            raise ValueError("Invalid source view identity")
        seen.add(view_id)
        row = {
            **view,
            "seconds": None,
            "origin": None,
            "returned": False,
            "model_claims": None,
            "model_questions": None,
        }
        if "view:" + view_id in stages:
            value = stage("view:" + view_id)
            inference = value["inference_manifest"]
            if returned.get(digest(inference["fingerprint"])) != inference:
                raise ValueError("View checkpoint and returned inference disagree")
            row.update(
                seconds=seconds(inference.get("elapsed_seconds")),
                origin=inference.get("origin"),
                returned=True,
                model_claims=len(value["claims"]),
                model_questions=len(value["questions"]),
            )
            identity = dict(inference["fingerprint"])
            identity.pop("input_sha256", None)
            identity["decoding"] = {
                k: v
                for k, v in identity.get("decoding", {}).items()
                if k not in {"schema_sha256", "sentence_map_sha256", "complete_sentence_choices"}
            }
            result["backend_identities"].append(identity)
        view_rows.append(row)
    if sum(row["returned"] for row in view_rows) != len(inferences):
        raise ValueError("Returned inference has no source view checkpoint")
    result["views"] = view_rows
    if "extraction" in stages:
        extraction = stage("extraction")
        if extraction["inferences"] != inferences:
            raise ValueError("Extraction checkpoint and published inferences disagree")
        result["model_claims"] = model_records(extraction["claims"], "claim_id", returned)
        result["model_questions"] = model_records(extraction["questions"], "question_id", returned)
        empty = result["model_claims"] == result["model_questions"] == 0
        result["no_accepted_model_records"] = empty
        result["fully_observed_empty"] = (
            empty and bool(views) and all(v["returned"] for v in view_rows)
        )
    return result


def read_batch(path, cases, input_sha, snapshot):
    batch = Path(path).resolve()
    if batch.name == "BATCH_STATE.json":
        batch = batch.parent
    state = snapshot.read(batch / "BATCH_STATE.json")
    if state.get("status") not in {"PASS", "PARTIAL", "FAIL"} or not state.get("finished_utc"):
        raise ValueError("Batch snapshot is still running or interrupted; final comparison refused")
    identity = state["inputs"]
    if state["fingerprint"] != digest(identity) or identity["input"]["sha256"] != input_sha:
        raise ValueError("Frozen input or batch fingerprint mismatch")
    expected = {case["case_id"]: case["input_sha256"] for case in cases}
    if identity["case_inputs"] != expected or set(state["cases"]) != set(expected):
        raise ValueError("Batch cases differ from the exact frozen case set")
    if any(
        e.get("status") == "RUNNING" or e.get("input_sha256") != expected[k]
        for k, e in state["cases"].items()
    ):
        raise ValueError("Running case or per-case fingerprint mismatch")
    if any(
        e.get("status") not in {"PASS", "PARTIAL", "FAIL", "INTERRUPTED"}
        or type(e.get("terminal")) is not bool
        for e in state["cases"].values()
    ):
        raise ValueError("Invalid case status or non-boolean terminal marker")
    result = {
        "path": str(batch),
        "batch_status": state["status"],
        "batch_fingerprint": state["fingerprint"],
        "batch_state_sha256": snapshot.hashes[batch / "BATCH_STATE.json"],
        "recorded_counts": state.get("counts", {}),
        "configuration": identity["options"],
        "shared_analysis": {
            key: identity["shared_analysis"].get(key)
            for key in ("libraries", "runtime_environment")
        },
        "shared_inputs_digest": digest(identity["shared_analysis"].get("inputs")),
        "shared_models_digest": digest(identity["shared_analysis"].get("models")),
        "implementation_digest": digest(identity["shared_analysis"].get("implementation")),
        "cases": {
            case["case_id"]: read_case(batch, case, state["cases"][case["case_id"]], snapshot)
            for case in cases
        },
    }
    result["models"] = variants(m for c in result["cases"].values() for m in c["models"])
    result["backend_identities"] = variants(
        i for c in result["cases"].values() for i in c["backend_identities"]
    )
    rows = list(result["cases"].values())
    view_rows = [v for c in rows for v in c["views"] or []]
    result["summary"] = {
        "cases": len(rows),
        "statuses": dict(Counter(c["status"] for c in rows)),
        "retryable_cases": sum(c["retryable"] for c in rows),
        "model_claims": sum(c["model_claims"] or 0 for c in rows),
        "model_questions": sum(c["model_questions"] or 0 for c in rows),
        "cases_with_model_claims": sum(bool(c["model_claims"]) for c in rows),
        "cases_with_model_questions": sum(bool(c["model_questions"]) for c in rows),
        "no_accepted_model_records_cases": sum(
            c["no_accepted_model_records"] is True for c in rows
        ),
        "fully_observed_empty_model_cases": sum(c["fully_observed_empty"] is True for c in rows),
        "cases_without_model_accounting": sum(c["model_claims"] is None for c in rows),
        "cases_without_view_inventory": sum(c["views"] is None for c in rows),
        "technical_failures": sum(len(c["technical_failures"]) for c in rows),
        "attempt_errors": sum(len(c["attempt_errors"]) for c in rows),
        "guard_failure_events": sum(len(c["guard_failures"]) for c in rows),
        "views": len(view_rows),
        "returned_views": sum(v["returned"] for v in view_rows),
        "timed_views": sum(v["seconds"] is not None for v in view_rows),
        "missing_view_timings": sum(v["seconds"] is None for v in view_rows),
        "median_returned_view_seconds": median(
            [v["seconds"] for v in view_rows if v["seconds"] is not None]
        ),
        "timing_origins": dict(Counter(v["origin"] or "MISSING" for v in view_rows)),
    }
    for name, actual in (
        ("passed", sum(c["status"] == "PASS" for c in rows)),
        ("partial", sum(c["status"] == "PARTIAL" for c in rows)),
        ("failed", sum(c["status"] in {"FAIL", "INTERRUPTED"} for c in rows)),
        ("retryable", result["summary"]["retryable_cases"]),
    ):
        if name in result["recorded_counts"] and result["recorded_counts"][name] != actual:
            raise ValueError("Recorded batch counts disagree with case statuses")
    return result


def compare(baseline, candidate, inputs):
    snapshot = Snapshot()
    input_path = Path(inputs).resolve()
    snapshot.read(input_path, lines=True)
    cases = load_batch_inputs(input_path)
    input_sha = snapshot.hashes[input_path]
    left, right = [read_batch(p, cases, input_sha, snapshot) for p in (baseline, candidate)]
    deltas, timed_pairs = [], []
    common_views = baseline_missing = candidate_missing = 0
    for case in cases:
        key = case["case_id"]
        a, b = left["cases"][key], right["cases"][key]
        av, bv = [{v["view_id"]: v for v in row["views"] or []} for row in (a, b)]
        common = set(av) & set(bv)
        common_views += len(common)
        baseline_missing += sum(av[v]["seconds"] is None for v in common)
        candidate_missing += sum(bv[v]["seconds"] is None for v in common)
        for view in sorted(common):
            if {k: av[view][k] for k in ("document_id", "start", "end", "section")} != {
                k: bv[view][k] for k in ("document_id", "start", "end", "section")
            }:
                raise ValueError("Same view ID has different source coordinates")
            if av[view]["seconds"] is not None and bv[view]["seconds"] is not None:
                timed_pairs.append(
                    {
                        "case_id": key,
                        "view_id": view,
                        "baseline_seconds": av[view]["seconds"],
                        "candidate_seconds": bv[view]["seconds"],
                        "baseline_origin": av[view]["origin"],
                        "candidate_origin": bv[view]["origin"],
                        "delta_seconds": bv[view]["seconds"] - av[view]["seconds"],
                    }
                )
        deltas.append(
            {
                "case_id": key,
                "input_sha256": case["input_sha256"],
                "baseline_status": a["status"],
                "candidate_status": b["status"],
                "model_claim_delta": None
                if a["model_claims"] is None or b["model_claims"] is None
                else b["model_claims"] - a["model_claims"],
                "model_question_delta": None
                if a["model_questions"] is None or b["model_questions"] is None
                else b["model_questions"] - a["model_questions"],
                "baseline_only_views": sorted(set(av) - set(bv)),
                "candidate_only_views": sorted(set(bv) - set(av)),
            }
        )
    fields = (
        "configuration",
        "shared_analysis",
        "shared_inputs_digest",
        "shared_models_digest",
        "implementation_digest",
        "models",
        "backend_identities",
    )
    differences = {
        key: {"baseline": left[key], "candidate": right[key]}
        for key in fields
        if left[key] != right[key]
    }
    snapshot.unchanged()
    return {
        "schema_version": "case-batch-comparison-1",
        "status": "COMPUTATIONAL_COMPARISON",
        "comparison_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": input_sha,
        "case_ids": [c["case_id"] for c in cases],
        "scientific_accuracy": "NOT_MEASURED",
        "automatic_promotion": False,
        "baseline": left,
        "candidate": right,
        "configuration_differences": differences,
        "case_deltas": deltas,
        "paired_view_timings": timed_pairs,
        "paired_timing_summary": {
            "pairs": len(timed_pairs),
            "common_views": common_views,
            "baseline_missing_on_common_views": baseline_missing,
            "candidate_missing_on_common_views": candidate_missing,
            "pairs_with_different_origins": sum(
                r["baseline_origin"] != r["candidate_origin"] for r in timed_pairs
            ),
            "median_baseline_seconds": median([r["baseline_seconds"] for r in timed_pairs]),
            "median_candidate_seconds": median([r["candidate_seconds"] for r in timed_pairs]),
            "median_delta_seconds": median([r["delta_seconds"] for r in timed_pairs]),
        },
        "limitations": [
            "Accepted output counts are not accuracy, completeness, or scientific validity.",
            "A closed batch can retain partial, failed, or retryable cases; all statuses remain visible.",
            "Counts cover observed model records only; caller and deterministic questions are excluded.",
            "Missing view timings are not zero. Returned inference elapsed time includes bounded attempts, not model load or total case time.",
            "Cache replay elapsed fields describe the stored original inference; timing origins must be consulted.",
            "Configuration, model, dependency, implementation or input differences prevent isolated causal claims.",
        ],
        "verified_input_files": {str(p): h for p, h in sorted(snapshot.hashes.items())},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "inputs", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.exists() or any(
        out.is_relative_to(p.resolve().parent if p.is_file() else p.resolve())
        for p in (args.baseline, args.candidate)
    ):
        raise ValueError("Preserve existing outputs; choose a new directory outside both batches")
    report = compare(args.baseline, args.candidate, args.inputs)
    out.mkdir(parents=True, exist_ok=False)
    (out / "COMPARISON.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Development batch comparison",
        "",
        "Accepted counts are not scientific accuracy. No automatic promotion.",
        "",
        "| Metric | Baseline | Candidate |",
        "|---|---:|---:|",
    ]
    for name in report["baseline"]["summary"]:
        lines.append(
            f"| {name} | {report['baseline']['summary'][name]} | {report['candidate']['summary'][name]} |"
        )
    lines += [
        "",
        "Full per-case statuses, failures, missing observations, paired timings and exact configuration differences are retained in COMPARISON.json.",
        "",
    ]
    lines += ["- " + value for value in report["limitations"]]
    (out / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "baseline": report["baseline"]["summary"],
                "candidate": report["candidate"]["summary"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
