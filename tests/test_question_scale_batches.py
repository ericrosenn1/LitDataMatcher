"""Engineering regressions for the bounded scale validator, using synthetic texts."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).parents[1] / "scripts" / "validate_question_discovery_batches.py"
    spec = importlib.util.spec_from_file_location("question_scale_batches", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest(tmp_path, runner, batches=3):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    entries = []
    for number in range(1, batches + 1):
        docs = []
        for item in range(2):
            pmid = str(number * 10 + item)
            text = "Further research is needed to understand glucose uptake."
            docs.append(
                {
                    "document_id": "PMID:" + pmid,
                    "pmid": pmid,
                    "source": "pubmed",
                    "text": text,
                    "abstract": text,
                    "source_locator": "https://example.org/" + pmid,
                    "source_provenance": {"fixture": True},
                }
            )
        path = inputs / f"batch-{number}.jsonl"
        path.write_text("".join(json.dumps(doc) + "\n" for doc in docs), encoding="utf-8")
        entries.append({"batch": number, "documents": 2, "input": runner.file_identity(path)})
    path = inputs / "INPUT_MANIFEST.json"
    runner.atomic_json(
        path,
        {"documents": batches * 2, "batches": entries, "sampling_scope": "Synthetic test only"},
    )
    return path


def test_resume_reuses_successful_chunks_without_extraction_or_rewrite(
    tmp_path, runner, monkeypatch
):
    input_manifest = manifest(tmp_path, runner)
    out = tmp_path / "run"
    original = runner.compute
    calls = []

    def count(documents):
        calls.append(documents[0]["document_id"])
        return original(documents)

    monkeypatch.setattr(runner, "compute", count)
    with pytest.raises(SystemExit) as stop:
        runner.run_batches(input_manifest, out, stop_after=2)
    assert stop.value.code == 75
    assert runner.read(out / ".workflow/state.json")["status"] == "INTERRUPTED"
    assert len(calls) == 2
    result = runner.run_batches(input_manifest, out, resume=True)
    assert len(calls) == 3
    assert result["documents"] == 6
    assert result["resume_validation"]["loaded_without_extraction_this_invocation"] == [1, 2]
    assert result["resume_validation"]["pre_stop_stage_hashes_and_mtimes_unchanged"] is True
    assert runner.read(out / ".workflow/state.json")["status"] == "COMPLETE"
    assert runner.run_batches(input_manifest, out, resume=True) == result
    assert len(calls) == 3


def test_changed_input_cannot_resume_existing_successful_chunks(tmp_path, runner):
    input_manifest = manifest(tmp_path, runner)
    out = tmp_path / "run"
    with pytest.raises(SystemExit):
        runner.run_batches(input_manifest, out, stop_after=1)
    path = Path(runner.read(input_manifest)["batches"][0]["input"]["path"])
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen input changed"):
        runner.run_batches(input_manifest, out, resume=True)


def test_validated_seed_is_imported_without_extraction(tmp_path, runner, monkeypatch):
    input_manifest = manifest(tmp_path, runner, batches=1)
    data = runner.read(input_manifest)
    docs = runner.rows(data["batches"][0]["input"]["path"])
    value = runner.compute(docs)
    seed = tmp_path / "seed.json"
    receipt = tmp_path / "seed-receipt.json"
    runner.atomic_json(seed, value["result"])
    runner.atomic_json(
        receipt,
        {
            "implementation_sha256": runner.implementation(),
            "completed_utc": value["created_utc"],
            "extraction_and_clustering_seconds": value["duration_seconds"],
            "memory": value["memory"],
        },
    )
    data["batches"][0].update(
        seed_result=runner.file_identity(seed), seed_receipt=runner.file_identity(receipt)
    )
    runner.atomic_json(input_manifest, data)

    def forbidden(_documents):
        raise AssertionError("Previously successful seed must not be computed")

    monkeypatch.setattr(runner, "compute", forbidden)
    result = runner.run_batches(input_manifest, tmp_path / "run")
    assert result["prior_completed_batches_reused"] == 1
    assert result["new_extraction_batches_total"] == 0
    handoff = (tmp_path / "run" / "HANDOFF.md").read_text(encoding="utf-8")
    assert "did not demonstrate stop/resume" in handoff
    assert "hashes and timestamps prove reuse" not in handoff


def test_fidelity_rejects_a_title_substituted_for_native_abstract(tmp_path, runner):
    input_manifest = manifest(tmp_path, runner, batches=1)
    docs = runner.rows(runner.read(input_manifest)["batches"][0]["input"]["path"])
    result = runner.compute(docs)["result"]
    docs[0]["abstract"] = ""
    with pytest.raises(ValueError, match="unchanged nonempty native abstract"):
        runner.validate_result(docs, result)


def test_actual_rows_must_equal_declared_batch_count(tmp_path, runner):
    input_manifest = manifest(tmp_path, runner, batches=1)
    data = runner.read(input_manifest)
    data["documents"] = data["batches"][0]["documents"] = 1
    runner.atomic_json(input_manifest, data)
    with pytest.raises(ValueError, match="Actual input rows differ"):
        runner.run_batches(input_manifest, tmp_path / "run")


def test_dependency_change_blocks_resume(tmp_path, runner, monkeypatch):
    input_manifest = manifest(tmp_path, runner)
    output = tmp_path / "run"
    with pytest.raises(SystemExit):
        runner.run_batches(input_manifest, output, stop_after=1)
    original = runner.file_identity

    def changed(path):
        identity = original(path)
        if Path(path).name == "literature_integrity.py":
            identity["sha256"] = "changed"
        return identity

    monkeypatch.setattr(runner, "file_identity", changed)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        runner.run_batches(input_manifest, output, resume=True)


def test_native_continuation_reuses_all_prior_payloads_above_ten_batches(
    tmp_path, runner, monkeypatch
):
    source_manifest = manifest(tmp_path, runner, batches=11)
    prior = tmp_path / "prior"
    prior_summary = runner.run_batches(source_manifest, prior)
    original_state = (prior / ".workflow/state.json").read_bytes()
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    entries, identifiers, _, proof = runner.import_prior_batches(prior, prepared)
    continued_manifest = prepared / "INPUT_MANIFEST.json"
    runner.atomic_json(
        continued_manifest,
        {
            "documents": 22,
            "batches": entries,
            "prior_run_proof": proof,
            "sampling_scope": "Synthetic test only",
        },
    )

    def forbidden(_):
        raise AssertionError("Native completed batches must not be recomputed")

    monkeypatch.setattr(runner, "compute", forbidden)
    result = runner.run_batches(continued_manifest, tmp_path / "continued")
    assert len(identifiers) == result["documents"] == 22
    assert result["prior_completed_batches_reused"] == 11
    assert result["new_extraction_batches_total"] == 0
    assert result["new_extraction_seconds_total"] == 0
    assert result["retained_questions"] == prior_summary["retained_questions"]
    assert (prior / ".workflow/state.json").read_bytes() == original_state
    stage = runner.read(prior / ".workflow/state.json")["stages"]["extract-01"]
    path = prior / ".workflow" / stage["file"]
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="native checkpoint changed"):
        runner.run_batches(continued_manifest, tmp_path / "corrupt-import")


def test_coverage_question_count_must_match_retained_payload(tmp_path, runner):
    source_manifest = manifest(tmp_path, runner, batches=1)
    docs = runner.rows(runner.read(source_manifest)["batches"][0]["input"]["path"])
    result = runner.compute(docs)["result"]
    result["coverage"]["retained_questions"] += 1
    with pytest.raises(ValueError, match="Retained-question count differs"):
        runner.validate_result(docs, result)
