"""Synthetic fault injection; no model downloads or scientific acceptance claims."""

import hashlib
import json
import sys

import pytest

from litdatamatcher.controller import Controller, OwnershipConflict
from litdatamatcher.strict_artifacts import strict_json_loads
from litdatamatcher.v2 import analyze, read_rows, validate_run_artifact, write_rows
from litdatamatcher.workflow_checkpoint import CheckpointRun, initial_manifest


@pytest.mark.parametrize(
    "payload", ['{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{"x":1e999}', '{"x":1,"x":2}']
)
def test_strict_json_rejects_nonfinite_overflow_and_duplicate_keys(payload):
    with pytest.raises(ValueError):
        strict_json_loads(payload)


@pytest.mark.parametrize(
    "suffix,content",
    [
        (".json", '{"x":NaN}'),
        (".jsonl", '{"x":1e999}\n'),
        (".jsonl", "[]\n"),
        (".jsonl", "broken\n"),
    ],
)
def test_native_validator_and_controller_reject_bad_artifacts(tmp_path, suffix, content):
    output = tmp_path / ("bad" + suffix)
    controller = Controller(tmp_path / "controller")
    try:
        code = f"from pathlib import Path; Path({str(output)!r}).write_text({content!r})"
        controller.register("bad", [sys.executable, "-c", code], tmp_path, [output])
        assert controller.run_next()["status"] == "FAILED"
        assert validate_run_artifact(output) == "FAIL"
    finally:
        controller.close()


def test_explicit_transient_retry_succeeds_and_permanent_failure_stays_failed(tmp_path):
    controller = Controller(tmp_path / "controller")
    try:
        code = (
            "from pathlib import Path\nimport sys\np=Path('attempt')\n"
            "if not p.exists(): p.write_text('first'); sys.exit(75)\n"
            "Path('result.json').write_text('{\"ok\":true}')"
        )
        controller.register(
            "transient",
            [sys.executable, "-c", code],
            tmp_path,
            ["result.json"],
            retryable_exit_codes=[75],
            max_retries=2,
        )
        assert controller.run_next()["status"] == "RETRY_PENDING"
        assert controller.run_next()["status"] == "SUCCEEDED"
        assert controller.jobs()[0]["attempts"] == 2
        controller.register(
            "permanent",
            [sys.executable, "-c", "raise SystemExit(7)"],
            tmp_path,
            [],
            retryable_exit_codes=[75],
            max_retries=2,
        )
        assert controller.run_next()["status"] == "FAILED"
        assert controller.run_next()["status"] == "IDLE"
    finally:
        controller.close()


def test_retry_exhaustion_and_backoff_are_bounded(tmp_path):
    controller = Controller(tmp_path / "controller")
    try:
        controller.register(
            "exhaust",
            [sys.executable, "-c", "raise SystemExit(75)"],
            tmp_path,
            [],
            retryable_exit_codes=[75],
            max_retries=2,
        )
        assert [controller.run_next()["status"] for _ in range(3)] == [
            "RETRY_PENDING",
            "RETRY_PENDING",
            "FAILED",
        ]
        assert controller.run_next()["status"] == "IDLE"
        controller.register(
            "wait",
            [sys.executable, "-c", "raise SystemExit(75)"],
            tmp_path,
            [],
            retryable_exit_codes=[75],
            max_retries=1,
            retry_backoff=30,
        )
        assert controller.run_next()["status"] == "RETRY_PENDING"
        assert controller.run_next()["status"] == "WAITING_RETRY"
    finally:
        controller.close()


def test_checkpoint_ownership_resume_corruption_and_fingerprint_guard(tmp_path):
    out, calls = tmp_path / "run", []
    with pytest.raises(KeyboardInterrupt), CheckpointRun(out, {"input": "v1"}, False) as run:
        run.cached("extract", lambda: calls.append(1) or {"valid": True})
        with pytest.raises(OwnershipConflict), CheckpointRun(out, {"input": "v1"}, True):
            pass
        raise KeyboardInterrupt()
    assert json.loads((out / ".workflow/state.json").read_text())["status"] == "INTERRUPTED"
    with (
        pytest.raises(ValueError, match="fingerprint mismatch"),
        CheckpointRun(out, {"input": "changed"}, True),
    ):
        pass
    with CheckpointRun(out, {"input": "v1"}, True) as run:
        assert run.cached("extract", lambda: calls.append(2)) == {"valid": True}
        stage = run.directory / run.state["stages"]["extract"]["file"]
        stage.write_text("corrupted")
        assert run.cached("extract", lambda: {"repaired": True}) == {"repaired": True}
        assert list((run.directory / "quarantine").iterdir())
    assert calls == [1]


def native_fixture(tmp_path, monkeypatch, failure=None):
    import litdatamatcher.semantic_runtime as runtime_module

    executions, retrievals = [], []

    class Runtime:
        model_manifest = {"model_id": "synthetic-test", "revision": "fixture", "license": "fixture"}

        def __init__(self, *args, **kwargs):
            pass

        def extract(self, view, *args, **kwargs):
            executions.append(view["text"])
            if failure and len(executions) == 2:
                raise failure("Injected second-view failure")
            return {
                "claims": [],
                "questions": [],
                "rejected": [],
                "inference_manifest": {
                    "origin": "synthetic_test_stub",
                    "fingerprint": {"prompt_sha256": "fixture"},
                },
            }

    class Index:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, records):
            self.records = records
            retrievals.append("fit")

        def search(self, *args, **kwargs):
            return [{"id": row["id"], "score": 0.5} for row in self.records]

    monkeypatch.setattr(runtime_module, "LocalSemanticRuntime", Runtime)
    monkeypatch.setattr(runtime_module, "PretrainedSemanticIndex", Index)
    root = tmp_path / "data"
    documents = [
        {
            "document_id": f"d{n}",
            "title": f"Synthetic source {n}",
            "text": f"Future studies should investigate clinical response in humans {n}.",
            "source_locator": f"fixture:d{n}",
            "split_context": "development",
        }
        for n in (1, 2)
    ]
    write_rows(root / "catalog/literature.jsonl", documents)
    write_rows(
        root / "catalog/studies.jsonl",
        [
            {
                "dataset_id": "s1",
                "title": "Human clinical response",
                "organisms": ["Homo sapiens"],
                "source_provenance": {"source_url": "fixture:s1"},
            }
        ],
    )
    write_rows(root / "catalog/processed_inspections.jsonl", [])
    parameters = dict(
        question="What is clinical response in humans?",
        requirements=[{"field": "species", "expected": "human"}],
        limit=2,
        chunks=1,
    )

    def run(out, **kwargs):
        return analyze(
            root, out, tmp_path / "model", tmp_path / "embedding", **parameters, **kwargs
        )

    return run, executions, retrievals, root


def test_interrupted_native_analysis_reuses_valid_views_and_matches_fresh_result(
    tmp_path, monkeypatch
):
    run, executions, retrievals, root = native_fixture(tmp_path, monkeypatch, KeyboardInterrupt)
    out = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        run(out)
    assert (out / "RUN_MANIFEST.json").is_file()
    assert len(executions) == 2
    result = run(out, resume=True)
    assert result["status"] == "PASS"
    assert len(executions) == 3
    assert len(retrievals) == 1
    files = [
        "claims",
        "questions",
        "datasets",
        "matches",
        "evidence_bundles",
        "question_compilations",
    ]
    resumed = {name: read_rows(out / f"{name}.jsonl") for name in files}
    baseline = tmp_path / "fresh"
    run(baseline)
    assert resumed == {name: read_rows(baseline / f"{name}.jsonl") for name in files}
    counts = len(executions), len(retrievals)
    assert run(out, resume=True) == result
    assert (len(executions), len(retrievals)) == counts
    (out / "matches.jsonl").write_text("corrupt")
    run(out, resume=True)
    assert read_rows(out / "matches.jsonl") == resumed["matches"]
    assert (len(executions), len(retrievals)) == counts
    (root / "catalog/studies.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run(out, resume=True)


def test_retryable_native_inference_failure_resumes_only_failed_view(tmp_path, monkeypatch):
    run, executions, _, _ = native_fixture(tmp_path, monkeypatch, RuntimeError)
    out = tmp_path / "partial"
    assert run(out)["status"] == "PARTIAL"
    assert len(executions) == 2
    assert run(out, resume=True)["status"] == "PASS"
    assert len(executions) == 3


def test_native_resume_rejects_model_or_parameter_drift(tmp_path, monkeypatch):
    run, _, _, root = native_fixture(tmp_path, monkeypatch)
    out = tmp_path / "run"
    run(out)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        analyze(
            root,
            out,
            tmp_path / "model",
            tmp_path / "embedding",
            question="Changed question",
            resume=True,
        )
    (tmp_path / "model").mkdir()
    (tmp_path / "model/config.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run(out, resume=True)


def test_interrupted_compilation_reuses_extraction(tmp_path, monkeypatch):
    import litdatamatcher.v2 as native

    run, executions, _, _ = native_fixture(tmp_path, monkeypatch)
    original = native.compile_question

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("compiler interruption")

    monkeypatch.setattr(native, "compile_question", interrupted)
    out = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        run(out)
    assert len(executions) == 2
    monkeypatch.setattr(native, "compile_question", original)
    assert run(out, resume=True)["status"] == "PASS"
    assert len(executions) == 2


def test_corrupt_matching_stage_repairs_without_reextracting(tmp_path, monkeypatch):
    run, executions, retrievals, _ = native_fixture(tmp_path, monkeypatch)
    out = tmp_path / "run"
    run(out)
    original = (out / "matches.jsonl").read_bytes()
    state = json.loads((out / ".workflow/state.json").read_text())
    (out / ".workflow" / state["stages"]["matching"]["file"]).write_text("corrupt")
    (out / "matches.jsonl").write_text("corrupt")
    assert run(out, resume=True)["status"] == "PASS"
    assert (out / "matches.jsonl").read_bytes() == original
    assert len(executions) == 2
    assert len(retrievals) == 2
    assert list((out / ".workflow/quarantine").iterdir())


def test_input_changed_during_run_cannot_be_accepted(tmp_path, monkeypatch):
    import litdatamatcher.semantic_runtime as runtime

    run, _, _, root = native_fixture(tmp_path, monkeypatch)
    original = runtime.PretrainedSemanticIndex.fit

    def mutate(self, records):
        original(self, records)
        with (root / "catalog/studies.jsonl").open("a") as handle:
            handle.write("\n")

    monkeypatch.setattr(runtime.PretrainedSemanticIndex, "fit", mutate)
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="changed during execution"):
        run(out)
    state = json.loads((out / ".workflow/state.json").read_text())
    assert state["status"] == "FAILED" and "completed" not in state
    assert json.loads((out / "RUN_MANIFEST.json").read_text())["execution_status"] == "FAIL"


def test_journal_write_failure_releases_run_lease(tmp_path, monkeypatch):
    out = tmp_path / "run"
    with (
        pytest.raises(OSError, match="injected disk failure"),
        CheckpointRun(out, {"input": "v1"}, False) as run,
    ):

        def failed_write():
            raise OSError("injected disk failure")

        monkeypatch.setattr(run, "_write", failed_write)
    with CheckpointRun(out, {"input": "v1"}, True):
        pass


def test_corrupt_checkpoint_cannot_move_an_external_file(tmp_path):
    external = tmp_path / "original.txt"
    external.write_text("preserve")
    with CheckpointRun(tmp_path / "run", {"input": "v1"}, False) as run:
        run.state["stages"]["tampered"] = {"file": "../../original.txt", "sha256": "wrong"}
        with pytest.raises(ValueError, match="outside this run"):
            run.load("tampered")
    assert external.read_text() == "preserve"


def test_legacy_run_without_checkpoint_is_preserved(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "result.json").write_text('{"old":true}')
    before = list(legacy.iterdir())
    with (
        pytest.raises(ValueError, match="no resumable checkpoint"),
        CheckpointRun(legacy, {}, True),
    ):
        pass
    assert list(legacy.iterdir()) == before
    assert (legacy / "result.json").read_text() == '{"old":true}'


def test_empty_jsonl_preserves_no_result_but_empty_json_fails(tmp_path):
    controller = Controller(tmp_path / "controller")
    try:
        for name, expected in (("empty.jsonl", "SUCCEEDED"), ("empty.json", "FAILED")):
            controller.register(
                name,
                [sys.executable, "-c", f"from pathlib import Path; Path({name!r}).write_text('')"],
                tmp_path,
                [name],
            )
            assert controller.run_next()["status"] == expected
    finally:
        controller.close()


def test_log_failure_does_not_report_success(tmp_path):
    controller = Controller(tmp_path / "controller")
    try:
        log = controller.root / "logs" / (hashlib.sha256(b"job").hexdigest() + ".log")
        log.mkdir(parents=True)
        controller.register(
            "job",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('result.json').write_text('{}')",
            ],
            tmp_path,
            ["result.json"],
        )
        result = controller.run_next()
        assert result["status"] == "FAILED" and "Output drain failed" in result["error"]
    finally:
        controller.close()


def test_early_manifest_matches_existing_schema_and_failed_run_is_not_pass(tmp_path, monkeypatch):
    from pathlib import Path

    import jsonschema

    schema = json.loads(
        (
            Path(__file__).parents[1] / "litdatamatcher/schemas_v2/RUN_MANIFEST.schema.json"
        ).read_text()
    )
    jsonschema.validate(initial_manifest("run", 0), schema)
    run, _, _, _ = native_fixture(tmp_path, monkeypatch, KeyboardInterrupt)
    out = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        run(out)
    manifest = json.loads((out / "RUN_MANIFEST.json").read_text())
    jsonschema.validate(manifest, schema)
    assert manifest["execution_status"] == "INTERRUPTED"


def test_completed_native_manifest_validates_the_entire_packaged_contract(tmp_path, monkeypatch):
    import jsonschema

    import litdatamatcher.v2 as native

    run, _, _, _ = native_fixture(tmp_path, monkeypatch)
    output = tmp_path / "completed"
    assert run(output)["status"] == "PASS"
    manifest = json.loads((output / "RUN_MANIFEST.json").read_text())
    jsonschema.Draft202012Validator(native._run_manifest_schema()).validate(manifest)
    assert manifest["question_compilation"]["compiled_questions"] > 0
    assert manifest["artifacts"]


@pytest.mark.parametrize(
    "value",
    [
        {"compiled_questions": -1, "assessable_questions": 0, "underdetermined_questions": 0},
        {"compiled_questions": 1, "assessable_questions": 0.5, "underdetermined_questions": 0},
        {"compiled_questions": 1, "assessable_questions": False, "underdetermined_questions": 0},
        {"compiled_questions": 1, "assessable_questions": 0},
        {
            "compiled_questions": 1,
            "assessable_questions": 0,
            "underdetermined_questions": 0,
            "extra": 1,
        },
    ],
)
def test_question_compilation_contract_rejects_invalid_counts(value):
    import jsonschema

    import litdatamatcher.v2 as native

    manifest = initial_manifest("fixture", 0)
    manifest["question_compilation"] = value
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(native._run_manifest_schema()).validate(manifest)


def test_final_writer_refuses_stale_schema_before_publishing_success(tmp_path, monkeypatch):
    import jsonschema

    import litdatamatcher.v2 as native

    run, _, _, _ = native_fixture(tmp_path, monkeypatch)
    stale = native._run_manifest_schema()
    stale["properties"].pop("question_compilation")
    monkeypatch.setattr(native, "_run_manifest_schema", lambda: stale)
    output = tmp_path / "stale-schema"
    with pytest.raises(jsonschema.ValidationError, match="question_compilation"):
        run(output)
    manifest = json.loads((output / "RUN_MANIFEST.json").read_text())
    assert manifest["execution_status"] == "FAIL"
    state = json.loads((output / ".workflow/state.json").read_text())
    assert state["status"] == "FAILED" and "completed" not in state


def test_packaged_manifest_binds_the_runtime_schema_bytes():
    import hashlib
    from pathlib import Path

    import litdatamatcher.v2 as native

    root = Path(native.__file__).parent / "schemas_v2"
    schema = root / "RUN_MANIFEST.schema.json"
    package = json.loads((root / "PACKAGE_MANIFEST.json").read_text())
    entry = next(
        item for item in package["files"] if item["path"] == "templates/RUN_MANIFEST.schema.json"
    )
    assert entry["sha256"] == hashlib.sha256(schema.read_bytes()).hexdigest()
    assert entry["size_bytes"] == schema.stat().st_size
