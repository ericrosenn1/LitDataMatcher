"""Compare rules with a preserved model result on exactly its saved source views.

This is a descriptive engineering comparison, without independent utility labels.
It never invokes a model or downloads literature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

from litdatamatcher.data_plane import atomic_json
from litdatamatcher.question_discovery import discover_questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-job", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.saved_job.read_bytes()
    saved = json.loads(raw)
    result = saved.get("result", saved)
    assert result["request"]["extraction_mode"] == "model"
    assert result["processing_status"] == "COMPLETE"
    assert result["questions"]["extraction_method"] == "local_model_source_guard"
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Choose a new output directory; completed comparisons are preserved")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rules = discover_questions(result["documents"], max_documents=80, max_questions=40)
    rules_seconds = time.perf_counter() - started
    model = result["questions"]
    documents = {d["document_id"]: d for d in result["documents"]}
    summaries = {}
    ids = {}
    for name, extracted in (("rules", rules), ("model", model)):
        ids[name] = {q["question_id"] for q in extracted["questions"]}
        assert len(ids[name]) == len(extracted["questions"])
        for question in extracted["questions"]:
            doc = documents[question["source_document_id"]]
            span = question["evidence_span"]
            assert doc["text"][span["start"] : span["end"]] == span["text"]
            assert question["question"] == span["text"]
        summaries[name] = {
            "coverage": extracted["coverage"],
            "question_ids": sorted(ids[name]),
            "compilation_status_counts": dict(
                Counter(q["compilation"]["compilation_status"] for q in extracted["questions"])
            ),
            "exact_source_spans_pass": True,
        }
    report = {
        "status": "PASS",
        "source_file": str(args.saved_job.resolve()),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "saved_job_id": saved.get("id"),
        "source_documents": sorted(documents),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_inference_repeated": False,
        "rules_runtime_seconds": rules_seconds,
        "model_timing_scope": "Historical per-document inference manifests retained in source job; not comparable to fresh end-to-end timing.",
        "modes": summaries,
        "common_question_ids": sorted(ids["rules"] & ids["model"]),
        "rules_only_question_ids": sorted(ids["rules"] - ids["model"]),
        "model_only_question_ids": sorted(ids["model"] - ids["rules"]),
        "saved_model_pipeline_candidates": len(result["candidates"]),
        "scientific_utility_labels": None,
        "limitations": [
            "One saved two-document example; no general model superiority or accuracy inference.",
            "Literal fidelity and compilation status do not establish scientific usefulness or unansweredness.",
            "Rules were freshly executed; model output and its original software/configuration are preserved from the historical job.",
            "No dataset retrieval, qualification, training or final evaluation was repeated.",
        ],
    }
    (args.out / "SOURCE_JOB.json").write_bytes(raw)
    atomic_json(args.out / "RULES_RESULT.json", rules)
    atomic_json(args.out / "COMPARISON.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
