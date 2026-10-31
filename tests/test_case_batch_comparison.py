import hashlib
import json
from pathlib import Path

import pytest

from litdatamatcher.analysis_batch import load_batch_inputs
from litdatamatcher.data_plane import canonical_json, digest
from scripts.campaign_20261031 import compare_case_batches as module


def write(path, value, *, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        b"".join(canonical_json(row) + b"\n" for row in value) if lines else canonical_json(value)
    )
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def fixture(
    tmp_path, *, candidate=False, missing=False, timing=3.0, status="PARTIAL", terminal=False
):
    inputs = tmp_path / "inputs.jsonl"
    if not inputs.exists():
        write(
            inputs,
            [
                {
                    "case_id": "case1",
                    "source_document": {"document_id": "doc1", "text": "A increased B."},
                    "question": "What is reported?",
                }
            ],
            lines=True,
        )
    case = load_batch_inputs(inputs)[0]
    root = tmp_path / ("candidate" if candidate else "baseline")
    root.mkdir()
    document = root / "inputs/case1.jsonl"
    document_hash = write(document, [case["document"]], lines=True)
    run = root / "cases/case1/attempt-0001"
    mode = "source_literal" if candidate else "source_schema"
    fp = {
        "input_sha256": "source-hash",
        "model_manifest_sha256": "model-b" if candidate else "model-a",
        "decoding": {"mode": mode},
        "config_sha256": "config",
    }
    inference = {"fingerprint": fp, "elapsed_seconds": timing, "origin": "fresh_local_inference"}
    model_claims = (
        [{"claim_id": "claim1", "source_view_id": "view1", "inference_fingerprint": fp}]
        if candidate
        else []
    )
    model_questions = [
        {"question_id": "model-q", "origin": "explicit_unresolved", "inference_fingerprint": fp}
    ]
    questions = model_questions + [
        {"question_id": "caller", "origin": "user"},
        {"question_id": "source-q", "origin": "explicit_unresolved_source"},
    ]
    view_result = {
        "claims": model_claims,
        "questions": model_questions,
        "inference_manifest": inference,
    }
    inferences = [] if missing else [inference]
    if missing:
        model_claims, model_questions = [], []
        questions = questions[1:]
    extraction = {"claims": model_claims, "questions": questions, "inferences": inferences}
    stage_items = {"extraction": extraction}
    if not missing:
        stage_items["view:view1"] = view_result
    stages = {}
    for name, value in stage_items.items():
        filename = digest(name) + ".json"
        stages[name] = {
            "file": filename,
            "sha256": write(run / ".workflow" / filename, value),
            "parents": {},
        }
    workflow_inputs = {"document": {"path": str(document), "sha256": document_hash}}
    workflow = {
        "inputs": workflow_inputs,
        "fingerprint": digest(workflow_inputs),
        "stages": stages,
        "completed": terminal,
        "status": "COMPLETE" if terminal else "RETRYABLE",
    }
    payloads = {
        ".workflow/state.json": (workflow, False),
        "claims.jsonl": (model_claims, True),
        "questions.jsonl": (questions, True),
        "inferences.jsonl": (inferences, True),
        "source_views.jsonl": (
            [
                {
                    "view_id": "view1",
                    "document_id": "doc1",
                    "start": 0,
                    "end": 14,
                    "section": "abstract",
                }
            ],
            True,
        ),
    }
    artifacts = []
    for name, (value, lines) in payloads.items():
        checksum = write(run / name, value, lines=lines)
        artifacts.append({"path": name, "sha256": checksum})
    failures = (
        [{"stage": "inference", "view_id": "view1", "description": "fixture generation error"}]
        if missing
        else [
            {
                "stage": "source_guard",
                "view_id": "view1",
                "rejections": [{"kind": "claim", "reason": "fixture unsupported direction"}],
            }
        ]
    )
    manifest = {
        "execution_status": status,
        "failures": failures,
        "models": [
            {
                "id": "model-b" if candidate else "model-a",
                "revision": "fixture",
                "runtime": "fixture",
            }
        ],
        "artifacts": artifacts[1:],
    }
    artifacts.append(
        {"path": "RUN_MANIFEST.json", "sha256": write(run / "RUN_MANIFEST.json", manifest)}
    )
    entry = {
        "status": status,
        "terminal": terminal,
        "input_sha256": case["input_sha256"],
        "run_dir": run.relative_to(root).as_posix(),
        "outputs": [
            {"path": (run / r["path"]).relative_to(root).as_posix(), "sha256": r["sha256"]}
            for r in artifacts
        ],
        "attempts": [
            {"status": status, "terminal": terminal, "finished_utc": "2026-01-01T00:00:01Z"}
        ],
    }
    identity = {
        "input": {"sha256": hashlib.sha256(inputs.read_bytes()).hexdigest()},
        "case_inputs": {"case1": case["input_sha256"]},
        "options": {"runtime_config": {"decoding_mode": mode}, "chunks": 1},
        "shared_analysis": {
            "models": [{"path": "model"}],
            "inputs": [],
            "implementation": [{"sha256": "code"}],
            "libraries": {"python": "fixture"},
        },
    }
    state = {
        "status": status,
        "finished_utc": "2026-01-01T00:00:01Z",
        "inputs": identity,
        "fingerprint": digest(identity),
        "counts": {
            "passed": int(status == "PASS"),
            "partial": int(status == "PARTIAL"),
            "failed": int(status == "FAIL"),
            "retryable": int(not terminal),
        },
        "cases": {"case1": entry},
    }
    write(root / "BATCH_STATE.json", state)
    return root, inputs


def pair(tmp_path, **kwargs):
    baseline, inputs = fixture(tmp_path, **kwargs)
    candidate, _ = fixture(tmp_path, candidate=True, status="PASS", terminal=True, timing=5.0)
    return baseline, candidate, inputs


def rewrite_state(root, change, *, fingerprint=False):
    path = root / "BATCH_STATE.json"
    state = json.loads(path.read_text())
    change(state)
    if fingerprint:
        state["fingerprint"] = digest(state["inputs"])
    write(path, state)


def test_model_coverage_excludes_caller_and_deterministic_questions(tmp_path):
    baseline, candidate, inputs = pair(tmp_path)
    result = module.compare(baseline, candidate, inputs)
    a, b = [result[k]["summary"] for k in ("baseline", "candidate")]
    assert a["model_questions"] == 1
    assert result["baseline"]["cases"]["case1"]["published_question_origins"] == {
        "user": 1,
        "explicit_unresolved": 1,
        "explicit_unresolved_source": 1,
    }
    assert (a["model_claims"], b["model_claims"]) == (0, 1)
    assert a["retryable_cases"] == 1 and result["baseline"]["batch_status"] == "PARTIAL"
    assert result["case_deltas"][0]["model_claim_delta"] == 1
    assert (
        result["configuration_differences"]["configuration"]["baseline"]["runtime_config"][
            "decoding_mode"
        ]
        == "source_schema"
    )
    assert "models" in result["configuration_differences"]
    assert result["paired_timing_summary"]["pairs"] == 1
    assert result["paired_timing_summary"]["median_delta_seconds"] == 2.0
    assert result["paired_timing_summary"]["baseline_missing_on_common_views"] == 0
    assert (
        result["automatic_promotion"] is False and result["scientific_accuracy"] == "NOT_MEASURED"
    )


def test_missing_inference_is_not_timed_or_declared_fully_observed_empty(tmp_path):
    baseline, candidate, inputs = pair(tmp_path, missing=True)
    report = module.compare(baseline, candidate, inputs)
    summary = report["baseline"]["summary"]
    assert summary["no_accepted_model_records_cases"] == 1
    assert summary["fully_observed_empty_model_cases"] == 0
    assert summary["missing_view_timings"] == 1
    assert summary["median_returned_view_seconds"] is None
    assert summary["technical_failures"] == 1
    assert report["paired_timing_summary"]["median_delta_seconds"] is None


def test_returned_view_with_missing_time_stays_missing(tmp_path):
    baseline, candidate, inputs = pair(tmp_path, timing=None)
    result = module.compare(baseline, candidate, inputs)
    assert result["baseline"]["summary"]["returned_views"] == 1
    assert result["baseline"]["summary"]["timed_views"] == 0
    assert result["paired_timing_summary"]["pairs"] == 0


@pytest.mark.parametrize("value", [True, -1, float("inf")])
def test_invalid_timing_is_rejected(value):
    with pytest.raises(ValueError, match="timing"):
        module.seconds(value)


def test_running_candidate_cannot_be_reported_as_final(tmp_path):
    baseline, candidate, inputs = pair(tmp_path)
    rewrite_state(candidate, lambda state: state.update(status="RUNNING"))
    with pytest.raises(ValueError, match="still running"):
        module.compare(baseline, candidate, inputs)


def test_frozen_input_bytes_and_case_digests_are_bound(tmp_path):
    baseline, candidate, inputs = pair(tmp_path)
    inputs.write_bytes(inputs.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Frozen input"):
        module.compare(baseline, candidate, inputs)


def test_forged_case_digest_is_rejected_even_with_updated_batch_fingerprint(tmp_path):
    baseline, candidate, inputs = pair(tmp_path)
    rewrite_state(
        candidate,
        lambda state: state["inputs"]["case_inputs"].update(case1="other"),
        fingerprint=True,
    )
    with pytest.raises(ValueError, match="exact frozen"):
        module.compare(baseline, candidate, inputs)


def test_corrupt_artifact_and_escape_are_rejected(tmp_path):
    baseline, candidate, inputs = pair(tmp_path)
    (candidate / "cases/case1/attempt-0001/claims.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        module.compare(baseline, candidate, inputs)
    rewrite_state(candidate, lambda state: state["cases"]["case1"].update(run_dir="../outside"))
    with pytest.raises(ValueError, match="Unsafe"):
        module.compare(baseline, candidate, inputs)


def test_mutation_during_read_is_rejected(tmp_path, monkeypatch):
    baseline, candidate, inputs = pair(tmp_path)
    original = module.read_batch

    def mutate(*args, **kwargs):
        value = original(*args, **kwargs)
        if Path(args[0]) == candidate:
            inputs.write_bytes(inputs.read_bytes() + b"\n")
        return value

    monkeypatch.setattr(module, "read_batch", mutate)
    with pytest.raises(ValueError, match="changed during comparison"):
        module.compare(baseline, candidate, inputs)


def test_output_is_new_and_source_artifacts_are_preserved(tmp_path):
    baseline, candidate, inputs = pair(tmp_path)
    before = {
        str(p): p.read_bytes()
        for root in (baseline, candidate)
        for p in root.rglob("*")
        if p.is_file()
    }
    out = tmp_path / "report"
    argv = [
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--inputs",
        str(inputs),
        "--out",
        str(out),
    ]
    assert module.main(argv) == 0
    report = json.loads((out / "COMPARISON.json").read_text())
    assert report["baseline"]["summary"]["retryable_cases"] == 1
    assert "not scientific accuracy" in (out / "COMPARISON.md").read_text()
    with pytest.raises(ValueError, match="Preserve"):
        module.main(argv)
    assert all(Path(p).read_bytes() == data for p, data in before.items())


def test_enriched_duplicate_question_counts_once_and_forged_provenance_fails():
    fp = {"input_sha256": "source"}
    original = {"question_id": "q", "origin": "explicit_unresolved", "inference_fingerprint": fp}
    enriched = {**original, "requirements": [], "source_question_records": [original]}
    assert module.model_records([enriched], "question_id", {digest(fp)}) == 1
    with pytest.raises(ValueError, match="matching returned inference"):
        module.model_records([enriched], "question_id", set())


def test_failed_case_without_outputs_keeps_unknown_model_counts(tmp_path):
    baseline, candidate, inputs = pair(tmp_path)

    def fail(state):
        state["status"] = "FAIL"
        state["counts"] = {"failed": 1, "retryable": 1}
        entry = state["cases"]["case1"]
        entry.update(status="FAIL", outputs=[])
        entry["attempts"][-1].update(status="FAIL", error={"type": "OSError", "message": "fixture"})

    rewrite_state(baseline, fail)
    result = module.compare(baseline, candidate, inputs)
    summary = result["baseline"]["summary"]
    assert summary["cases_without_model_accounting"] == 1
    assert summary["fully_observed_empty_model_cases"] == 0
    assert summary["attempt_errors"] == 1
    assert result["case_deltas"][0]["model_claim_delta"] is None


def test_jsonl_unicode_line_separators_are_inside_records(tmp_path):
    path = tmp_path / "records.jsonl"
    write(path, [{"quote": "Alpha\u2028Beta\u2029Gamma."}], lines=True)
    assert module.Snapshot().read(path, lines=True) == [{"quote": "Alpha\u2028Beta\u2029Gamma."}]


@pytest.mark.parametrize("updates", [{"status": "UNKNOWN"}, {"terminal": "false"}])
def test_invalid_status_and_boolean_marker_are_rejected(tmp_path, updates):
    baseline, candidate, inputs = pair(tmp_path)
    rewrite_state(candidate, lambda state: state["cases"]["case1"].update(updates))
    with pytest.raises(ValueError, match="Invalid case status"):
        module.compare(baseline, candidate, inputs)
