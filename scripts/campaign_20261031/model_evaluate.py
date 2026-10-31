"""Paired engineering challenges and source-grounded development recompilation.

No model inference, training, dataset acquisition, or held-out reading occurs.
The previous compiler's development receipt is a comparison artifact, not gold.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

from litdatamatcher import question_compiler, semantic_runtime


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baseline_module(root: Path, revision: str, name: str):
    code = subprocess.check_output(
        ["git", "show", f"{revision}:litdatamatcher/{name}.py"],
        cwd=root, text=True, encoding="utf-8",
    )
    module = types.ModuleType(f"campaign_baseline_{name}")
    module.__package__ = "litdatamatcher"
    module.__file__ = str(root / "litdatamatcher" / f"{name}.py")
    sys.modules[module.__name__] = module
    exec(compile(code, module.__file__, "exec"), module.__dict__)
    return module, hashlib.sha256(code.encode()).hexdigest()


def relation_case(name, text, subject, predicate, target, direction, negated, expected):
    return {
        "name": name, "expected_accept": expected, "label_origin": "engineering_challenge",
        "document": {"document_id": f"synthetic:{name}", "text": text},
        "payload": {"claims": [{
            "quote": text, "subject": subject, "predicate": predicate, "object": target,
            "direction": direction, "negated": negated, "status": "direct_experiment",
            "context": None, "comparator": None,
        }], "questions": []},
    }


def relation_cases():
    contrast = "Diet increased sleep duration, whereas exercise produced no change in reaction time."
    return [
        relation_case("borrowed_no_change", contrast, "Diet", "increased", "sleep duration", "no_change", True, False),
        relation_case("correct_contrast_increase", contrast, "Diet", "increased", "sleep duration", "increase", False, True),
        relation_case("borrowed_negation", contrast, "Diet", "increased", "sleep duration", "unknown", True, False),
        relation_case("supported_no_change", "Diet produced no change in reaction time.", "Diet", "produced no change in", "reaction time", "no_change", True, True),
        relation_case("supported_did_not", "Treatment did not reduce IL6 in human cells.", "Treatment", "did not reduce", "IL6", "no_change", True, True),
        relation_case("supported_decrease", "Treatment reduced IL6 in human cells.", "Treatment", "reduced", "IL6", "decrease", False, True),
        relation_case("opposite_direction", "Treatment reduced IL6 in human cells.", "Treatment", "reduced", "IL6", "increase", False, False),
        relation_case("cross_clause_roles", "Diet increased sleep duration, but exercise changed reaction time.", "Diet", "changed", "reaction time", "unknown", False, False),
        relation_case("word_fragment_role", "Treatment increased IL6 in human cells.", "Treatment", "increased", "IL", "increase", False, False),
        relation_case("correct_association", "Diet was associated with sleep duration.", "Diet", "was associated with", "sleep duration", "association", False, True),
        relation_case("negated_association_elsewhere", "Diet was associated with sleep duration, but exercise was not measured.", "Diet", "was associated with", "sleep duration", "association", True, False),
        relation_case("correct_association_elsewhere", "Diet was associated with sleep duration, but exercise was not measured.", "Diet", "was associated with", "sleep duration", "association", False, True),
    ]


def compiler_cases():
    return [
        {"name": "risk_explicit_antecedent", "question": "Does drug A increase this risk in adults?", "context": "NAION risk was noted. BMI was measured as a covariate.", "status": "COMPLETE", "outcome": "risk of NAION"},
        {"name": "risk_no_antecedent", "question": "Does drug A increase this risk in adults?", "context": "NAION and BMI were measured.", "status": "QUESTION_UNDERDETERMINED"},
        {"name": "risk_two_antecedents", "question": "Does drug A increase this risk in adults?", "context": "The risk of NAION and the risk of VTE were measured.", "status": "QUESTION_UNDERDETERMINED"},
        {"name": "predictive_morphology", "question": "Are biomarkers predictive of survival in adults?", "status": "COMPLETE", "purpose": "prediction", "outcome": "survival"},
        {"name": "correlation_morphology", "question": "Is diet correlated with sleep quality in adults?", "status": "COMPLETE", "purpose": "associational", "outcome": "sleep quality"},
        {"name": "child_population", "question": "Does diet affect sleep quality in children?", "status": "COMPLETE", "species": "human", "outcome": "sleep quality"},
        {"name": "mixed_population", "question": "Does treatment reduce inflammation in humans and mice?", "status": "QUESTION_UNDERDETERMINED"},
        {"name": "bounded_comparator", "question": "Does diet affect sleep quality compared with placebo in children?", "status": "COMPLETE", "comparator": "placebo"},
        {"name": "plain_comparative", "question": "Does treatment reduce inflammation in humans?", "status": "COMPLETE", "outcome": "inflammation"},
        {"name": "structured_pico", "question": "Intervention: treatment A | Comparator: placebo | Outcome: sleep quality.", "status": "COMPLETE", "outcome": "sleep quality"},
        {"name": "target_roles_missing", "question": "Are clinical predictors adequate?", "status": "QUESTION_UNDERDETERMINED"},
        {"name": "adverb_not_exposure", "question": "Occupational exposure is not consistently associated with an increased risk of cancer.", "status": "QUESTION_UNDERDETERMINED"},
        {"name": "score_not_population", "question": "Intervention: treatment | Comparator: placebo | Outcome: Child-Pugh scores.", "status": "COMPLETE", "absent_roles": ["species"]},
        {"name": "nested_preposition", "question": "Does treatment reduce changes in mood in humans?", "status": "COMPLETE", "outcome": "changes in mood"},
    ]


def summarize_compilation(result):
    return {
        "status": result["compilation_status"], "purpose": result["question_purpose"],
        "roles": {row["field"]: row["expected"] for row in result["entity_roles"]},
        "diagnostics": result["diagnostics"],
    }


def compiler_pass(case, summary):
    if summary["status"] != case["status"]:
        return False
    if "purpose" in case and summary["purpose"] != case["purpose"]:
        return False
    return (
        all(summary["roles"].get(key) == case[key] for key in ("outcome", "species", "comparator") if key in case)
        and all(key not in summary["roles"] for key in case.get("absent_roles", []))
    )


def retry_challenge(module):
    text = "Treatment reduced IL6 in human cells."
    good = relation_case("retry", text, "Treatment", "reduced", "IL6", "decrease", False, True)["payload"]["claims"][0]
    runtime = module.LocalSemanticRuntime.__new__(module.LocalSemanticRuntime)
    runtime.config = module.RuntimeConfig()
    runtime.model_manifest = {"model_id": "synthetic-runtime-double", "revision": "0" * 40, "license": "test"}
    calls = []

    def generate(document, repair):
        calls.append(repair)
        if len(calls) == 1:
            return json.dumps({"claims": [good, {**good, "direction": "increase"}], "questions": []}), 10, 20
        raise OSError("engineered retry failure")

    runtime._generate = generate
    with patch.object(importlib.metadata, "version", return_value="synthetic-test-version"):
        try:
            result = runtime.extract({"document_id": "synthetic:retry", "text": text})
            return {"pass": len(result["claims"]) == 1, "accepted": len(result["claims"]), "manifest": result["inference_manifest"]}
        except (OSError, ValueError, RuntimeError) as error:
            return {"pass": False, "accepted": 0, "error": type(error).__name__, "reason": str(error)}


def independent_pico(question):
    parts = question.split("|")
    result = {}
    for part in parts:
        label, separator, value = part.partition(":")
        if separator and label.strip().casefold() in {"intervention", "comparator", "outcome"}:
            result[label.strip().casefold()] = " ".join(value.strip().rstrip("?.;").split())
    return result if len(result) == 3 else None


def span_errors(compilation, question):
    errors = []
    for role in compilation["entity_roles"]:
        span = role["evidence"]
        if question[span["start"]:span["end"]] != span["text"]:
            errors.append(role["field"])
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="e592407")
    parser.add_argument("--development-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(part in str(args.development_records).casefold() for part in ("heldout", "holdout", "evaluation_candidate")):
        raise ValueError("This command only evaluates declared development artifacts")
    previous_receipt = args.development_records.with_name("BASELINE_RECEIPT.json")
    receipt = json.loads(previous_receipt.read_text(encoding="utf-8"))
    if "development split only" not in receipt.get("scope", ""):
        raise ValueError("Expected an explicitly development-only baseline receipt")
    if args.output.exists():
        raise FileExistsError("Use a new output directory; prior receipts are immutable")
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    old_compiler, compiler_hash = baseline_module(root, args.baseline_ref, "question_compiler")
    old_semantic, semantic_hash = baseline_module(root, args.baseline_ref, "semantic_runtime")
    challenges = []
    for case in relation_cases():
        result = {"name": case["name"], "kind": "source_guard", "expected_accept": case["expected_accept"], "label_origin": case["label_origin"], "input": case}
        for name, module in (("baseline", old_semantic), ("candidate", semantic_runtime)):
            actual = module.validate_extraction(case["payload"], case["document"])
            result[name] = {"pass": bool(actual["claims"]) == case["expected_accept"], "accepted": len(actual["claims"]), "rejected": actual["rejected"]}
        challenges.append(result)
    for case in compiler_cases():
        result = {"name": case["name"], "kind": "compiler", "input": case, "label_origin": "engineering_challenge"}
        for name, module in (("baseline", old_compiler), ("candidate", question_compiler)):
            actual = module.compile_question(case["question"], source_text=case.get("context"))
            summary = summarize_compilation(actual)
            result[name] = {"pass": compiler_pass(case, summary), **summary}
        challenges.append(result)
    challenges.append({"name": "preserve_valid_partition_after_retry_error", "kind": "runtime_orchestration", "label_origin": "engineering_challenge", "baseline": retry_challenge(old_semantic), "candidate": retry_challenge(semantic_runtime)})
    (args.output / "round1_challenges.json").write_text(json.dumps(challenges, indent=2) + "\n", encoding="utf-8")

    counts = {name: collections.Counter() for name in ("baseline", "candidate")}
    source_counts = collections.Counter()
    transitions = collections.Counter()
    differences = []
    failures = []
    total = 0
    started = time.perf_counter()
    with args.development_records.open(encoding="utf-8") as source, (args.output / "round2_candidate_compilations.jsonl").open("w", encoding="utf-8") as output:
        for line in source:
            row = json.loads(line)
            total += 1
            before = row["compilation"]
            question = before["original_gap_statement"]
            after = question_compiler.compile_question(question, source_locator=f"corpus-question:{row['question_id']}")
            output.write(json.dumps({**row, "compilation": after}, ensure_ascii=False, sort_keys=True) + "\n")
            source_counts[row["source_corpus"]] += 1
            transitions[f"{before['compilation_status']} -> {after['compilation_status']}"] += 1
            for name, compilation in (("baseline", before), ("candidate", after)):
                counts[name][compilation["compilation_status"]] += 1
                errors = span_errors(compilation, question)
                counts[name]["source_span_errors"] += len(errors)
                pico = independent_pico(question)
                if pico:
                    counts[name]["structured_pico_questions"] += 1
                    roles = {role["field"]: role["expected"] for role in compilation["entity_roles"]}
                    matched = all(roles.get(key) == value for key, value in pico.items())
                    counts[name]["structured_pico_consistent"] += int(matched)
                    if not matched and name == "candidate":
                        failures.append({"question_id": row["question_id"], "kind": "structured_pico", "expected": pico, "observed": roles})
                if errors and name == "candidate":
                    failures.append({"question_id": row["question_id"], "kind": "source_span", "fields": errors})
            before_summary, after_summary = summarize_compilation(before), summarize_compilation(after)
            if before_summary != after_summary:
                differences.append({"question_id": row["question_id"], "source_corpus": row["source_corpus"], "question": question, "baseline": before_summary, "candidate": after_summary})
    duration = time.perf_counter() - started
    (args.output / "round2_differences.json").write_text(json.dumps(differences, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "schema_version": "model-hardening-comparison-1.0",
        "status": "PASS_WITH_LIMITATIONS" if not failures and all(item["candidate"]["pass"] for item in challenges) else "FAIL",
        "baseline_ref": args.baseline_ref,
        "baseline_source_hashes": {"question_compiler.py": compiler_hash, "semantic_runtime.py": semantic_hash},
        "candidate_source_hashes": {name: sha256(root / "litdatamatcher" / name) for name in ("question_compiler.py", "semantic_runtime.py")},
        "round1": {"label_origin": "engineering_challenge", "cases": len(challenges), "baseline_pass": sum(item["baseline"]["pass"] for item in challenges), "candidate_pass": sum(item["candidate"]["pass"] for item in challenges)},
        "round2": {"label_origin": "source_explicit_structure_only", "records": total, "source_corpora": dict(source_counts), "counts": {key: dict(value) for key, value in counts.items()}, "transitions": dict(transitions), "changed_compilations": len(differences), "candidate_seconds": duration, "failures": failures},
        "inputs": {str(args.development_records): sha256(args.development_records), str(previous_receipt): sha256(previous_receipt)},
        "outputs": {
            path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in args.output.iterdir() if path.is_file()
        },
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "limitations": [
            "Engineered expected behaviors are not clinical or biological accuracy labels.",
            "Development questions and structured PICO fields support source-string consistency, not full requirement correctness or dataset answerability.",
            "Existing baseline outputs are preserved; the Qwen proposal interpreter was not rerun.",
            "No training, GPU inference, source acquisition, held-out reading, or calibration occurred.",
            "Improved extraction acceptance in a crafted case is not evidence of higher recall on published literature.",
        ],
    }
    (args.output / "MODEL_COMPARISON.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
