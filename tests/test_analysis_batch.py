"""Serial batching and fault injection use stubs, never scientific gold labels."""

import json
import threading
from copy import deepcopy

import pytest

from litdatamatcher.analysis_batch import load_batch_inputs, run_batch
from litdatamatcher.analysis_session import AnalysisSession
from litdatamatcher.controller import OwnershipConflict
from litdatamatcher.scientific_v2 import assess_requirements
from litdatamatcher.strict_artifacts import validate_artifact
from litdatamatcher.v2 import analyze, normalize_dataset, qualified_runtime, read_rows, write_rows


@pytest.fixture
def native(tmp_path, monkeypatch):
    import litdatamatcher.semantic_runtime as runtime

    counts = {"runtime": 0, "index": 0, "fit": 0, "extract": [], "interrupt": False}

    class Runtime:
        model_manifest = {"model_id": "fixture", "revision": "fixture", "license": "fixture"}

        def __init__(self, *args, **kwargs):
            counts["runtime"] += 1

        def extract(self, view, *args, **kwargs):
            counts["extract"].append(view["text"])
            if counts["interrupt"] and "number 2" in view["text"]:
                raise KeyboardInterrupt("Fixture interruption")
            return {"claims": [], "questions": [], "rejected": [],
                    "inference_manifest": {"origin": "fixture", "fingerprint": {"prompt_sha256": "fixture"}}}

    class Index:
        def __init__(self, *args, **kwargs):
            counts["index"] += 1

        def fit(self, records):
            counts["fit"] += 1
            self.records = records

        def search(self, *args, **kwargs):
            return [{"id": row["id"], "score": 0.5} for row in self.records]

    monkeypatch.setattr(runtime, "LocalSemanticRuntime", Runtime)
    monkeypatch.setattr(runtime, "PretrainedSemanticIndex", Index)
    root = tmp_path / "data"
    write_rows(root / "catalog/literature.jsonl", [])
    write_rows(root / "catalog/studies.jsonl", [{"dataset_id": "study", "title": "Clinical response in humans"}])
    write_rows(root / "catalog/processed_inspections.jsonl", [])
    inputs = tmp_path / "input.jsonl"
    write_rows(inputs, [{"case_id": f"case{i}", "document_id": f"pmid:{i}", "title": f"Source {i}",
                         "text": f"Further studies should examine clinical response in humans number {i}."} for i in (1, 2)])
    return root, inputs, counts


def test_two_cases_share_constructors_and_resume_skips_verified_outputs(native, tmp_path):
    root, inputs, counts = native
    out = tmp_path / "batch"
    result = run_batch(root, out, tmp_path / "model", tmp_path / "embeddings", inputs)
    assert result["passed"] == 2
    assert (counts["runtime"], counts["index"], counts["fit"]) == (1, 1, 1)
    assert len(counts["extract"]) == 2
    replay = run_batch(root, out, tmp_path / "model", tmp_path / "embeddings", inputs, resume=True)
    assert replay["verified_skips"] == 2
    assert (counts["runtime"], counts["index"], len(counts["extract"])) == (1, 1, 2)


def test_interrupted_case_preserves_completed_case_and_resumes(native, tmp_path):
    root, inputs, counts = native
    out = tmp_path / "batch"
    counts["interrupt"] = True
    with pytest.raises(KeyboardInterrupt):
        run_batch(root, out, tmp_path / "model", tmp_path / "embeddings", inputs)
    state = json.loads((out / "BATCH_STATE.json").read_text())
    assert state["status"] == "INTERRUPTED"
    assert state["cases"]["case1"]["terminal"] is True
    assert list((out / "failures").glob("*.json"))
    counts["interrupt"] = False
    result = run_batch(root, out, tmp_path / "model", tmp_path / "embeddings", inputs, resume=True)
    assert result["passed"] == 2 and result["verified_skips"] == 1
    assert sum("number 1" in text for text in counts["extract"]) == 1


def test_failed_case_can_continue_and_failure_remains(native, tmp_path, monkeypatch):
    import litdatamatcher.v2 as v2

    root, inputs, counts = native
    original = v2.analyze
    def fail_first(*args, **kwargs):
        if "case1" in str(kwargs["document_path"]):
            raise RuntimeError("Fixture technical failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(v2, "analyze", fail_first)
    result = run_batch(root, tmp_path / "batch", tmp_path / "model", tmp_path / "embeddings", inputs)
    assert result["failed"] == 1 and result["passed"] == 1 and result["status"] == "PARTIAL"


def test_changed_batch_input_or_runtime_config_is_rejected(native, tmp_path):
    root, inputs, counts = native
    out = tmp_path / "batch"
    run_batch(root, out, tmp_path / "model", tmp_path / "embeddings", inputs)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run_batch(root, out, tmp_path / "model", tmp_path / "embeddings", inputs, resume=True, runtime_config={"max_new_tokens": 100})
    inputs.write_text(inputs.read_text() + "\n")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run_batch(root, out, tmp_path / "model", tmp_path / "embeddings", inputs, resume=True)
    assert counts["runtime"] == 1


@pytest.mark.parametrize("case_id", ["../x", "a/b", "a\\b", "CON", "NUL.txt", "x:", "", ".", "x."])
def test_unsafe_case_ids_are_rejected(tmp_path, case_id):
    path = tmp_path / "input.jsonl"
    write_rows(path, [{"case_id": case_id, "document_id": "id", "text": "Text."}])
    with pytest.raises(ValueError):
        load_batch_inputs(path)


def test_case_ids_are_case_insensitively_unique_and_unicode_text_is_preserved(tmp_path):
    path = tmp_path / "input.jsonl"
    row = {"case_id": "A", "document_id": "id", "text": "First\u2028second."}
    write_rows(path, [row])
    assert read_rows(path)[0]["text"] == row["text"]
    write_rows(path, [row, dict(row, case_id="a")])
    with pytest.raises(ValueError, match="Duplicate"):
        load_batch_inputs(path)


def test_session_rejects_reentrant_foreign_thread_and_mutated_config(native, tmp_path):
    root, inputs, counts = native
    session = AnalysisSession(root, tmp_path / "model", tmp_path / "embeddings")
    parameters = {"device": "cpu", "vector_cache": None, "runtime_config": session.runtime_config}
    with (
        session.claim(root, tmp_path / "model", tmp_path / "embeddings", parameters),
        pytest.raises(OwnershipConflict),
        session.claim(root, tmp_path / "model", tmp_path / "embeddings", parameters),
    ):
        pass
    errors = []
    def other_thread():
        try:
            with session.claim(root, tmp_path / "model", tmp_path / "embeddings", parameters):
                pass
        except OwnershipConflict:
            errors.append(True)
    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join()
    assert errors == [True]
    session.runtime_config["max_new_tokens"] = 3
    with pytest.raises(ValueError, match="configuration changed"):
        session.validate_unchanged()


def test_document_resume_reuses_ingestion_stage(native, tmp_path, monkeypatch):
    import litdatamatcher.v2 as v2

    root, inputs, counts = native
    document = tmp_path / "single.jsonl"
    write_rows(document, [read_rows(inputs)[0]])
    out = tmp_path / "single-run"
    original = v2.compile_question
    monkeypatch.setattr(v2, "compile_question", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        analyze(root, out, tmp_path / "model", tmp_path / "embeddings", document_path=document)
    monkeypatch.setattr(v2, "compile_question", original)
    assert analyze(root, out, tmp_path / "model", tmp_path / "embeddings", document_path=document, resume=True)["status"] == "PASS"
    assert len(counts["extract"]) == 1


def test_flat_inferred_fields_need_source_specific_evidence():
    raw = {"dataset_id": "x", "organisms": ["human"], "assay_types": ["RNA-seq"], "source_provenance": {"source_url": "fixture:source"}}
    unknown = normalize_dataset(raw)
    assert unknown["modality_contract"]["organism"] == "UNKNOWN"
    assert not unknown["capabilities"]
    raw["field_evidence"] = {"organisms": {"status": "observed", "value": ["human"], "source_locator": "fixture:source#organisms"},
                             "assay_types": {"status": "observed", "value": ["RNA-seq"], "source_locator": "fixture:source#assay_types"}}
    actual = normalize_dataset(raw)
    assert assess_requirements([{"field": "species", "expected": "human"}], actual)["eligibility"] == "DIRECT_FIT"
    assert actual["capabilities"]["assay"]["source_locator"] == "fixture:source#assay_types"


def test_unknown_caps_are_filled_only_by_declared_field_proof():
    raw = {"organisms": ["Homo sapiens"], "assay_types": ["RNA-seq"],
           "capabilities": {"species": {"value": None, "status": "unknown"}}}
    assert normalize_dataset(raw)["capabilities"]["species"]["status"] == "unknown"
    raw["field_evidence"] = {"organisms": {"value": raw["organisms"], "status": "observed", "source_locator": "fixture:sample#/organism"}}
    assert normalize_dataset(raw)["capabilities"]["species"]["status"] == "observed"
    raw["capabilities"]["species"] = {"value": None, "status": "absent", "source_locator": "fixture:absence"}
    assert normalize_dataset(raw)["capabilities"]["species"]["status"] == "absent"


def test_geo_raw_fields_require_matching_adapter_provenance_and_keep_ambiguity():
    raw = {"dataset_id": "GSE123", "organisms": ["inferred human"], "assay_types": ["RNA-seq"],
           "metadata": {"uid": "200000123", "accession": "GSE123", "taxon": "Homo sapiens; Mus musculus",
                        "gdstype": "Expression profiling by high throughput sequencing"}}
    assert normalize_dataset(raw)["modality_contract"]["organism"] == "UNKNOWN"
    raw["source_provenance"] = {"source_type": "geo", "adapter_name": "geo", "adapter_version": "ncbi_geo_esummary_source_fidelity_v2",
                                "raw_record_id": "GSE123", "acquisition_method": "ncbi_eutilities"}
    result = normalize_dataset(raw)
    assert result["modality_contract"]["organisms"] == ["Homo sapiens; Mus musculus"]
    assert result["modality_contract"]["modality"] == ["UNKNOWN"]
    assert result["capabilities"]["assay"]["value"] == [raw["metadata"]["gdstype"]]
    assert result["capabilities"]["species"]["source_locator"].endswith("#/result/200000123/taxon")
    assert assess_requirements([{"field": "species", "expected": "human"}], result)["eligibility"] == "REQUIRES_INSPECTION"
    raw["source_provenance"]["raw_record_id"] = "GSE999"
    assert normalize_dataset(raw)["modality_contract"]["organism"] == "UNKNOWN"


def test_registered_study_design_does_not_promote_planned_measurements():
    raw = {"dataset_id": "registry", "organisms": ["human"], "assay_types": ["clinical registry"],
           "source_provenance": {"source_locator": "fixture:registry"},
           "capabilities": {"study_design": {"value": None, "status": "unknown"}},
           "metadata": {"protocolSection": {"designModule": {"studyType": "INTERVENTIONAL", "enrollmentInfo": {"count": 90}},
                                               "outcomesModule": {"primaryOutcomes": [{"measure": "Response"}]}}}}
    caps = normalize_dataset(raw)["capabilities"]
    assert caps["study_design"]["value"]["study_type"] == "INTERVENTIONAL"
    assert "species" not in caps and "outcome" not in caps and "donor_count" not in caps


def test_unicode_scientific_text_survives_native_artifacts_and_resume(native, tmp_path):
    root, inputs, counts = native
    text = "The concentration was 5 μM.\u2028Further studies should assess β-cell response.\u2029Uncertainty remains."
    row = read_rows(inputs)[0]
    row["text"] = text
    write_rows(inputs, [row])
    assert validate_artifact(inputs) == "PASS"
    result = run_batch(root, tmp_path / "unicode", tmp_path / "model", tmp_path / "embeddings", inputs)
    assert result["passed"] == 1
    assert counts["extract"] == [text]
    replay = run_batch(root, tmp_path / "unicode", tmp_path / "model", tmp_path / "embeddings", inputs, resume=True)
    assert replay["verified_skips"] == 1


def test_document_resume_quarantines_changed_ingestion(native, tmp_path, monkeypatch):
    import litdatamatcher.v2 as v2

    root, inputs, counts = native
    document = tmp_path / "single.jsonl"
    write_rows(document, [read_rows(inputs)[0]])
    out = tmp_path / "single-run"
    original = v2.compile_question
    monkeypatch.setattr(v2, "compile_question", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        analyze(root, out, tmp_path / "model", tmp_path / "embeddings", document_path=document)
    (out / "local_ingestion/literature.jsonl").write_text("corrupt", encoding="utf-8")
    monkeypatch.setattr(v2, "compile_question", original)
    assert analyze(root, out, tmp_path / "model", tmp_path / "embeddings", document_path=document, resume=True)["status"] == "PASS"
    quarantined = list((out / ".workflow/quarantine").glob("ingestion-*/literature.jsonl"))
    assert len(quarantined) == 1 and quarantined[0].read_text() == "corrupt"
    assert len(counts["extract"]) == 1


def test_stop_on_retryable_native_result_persists_failure(native, tmp_path, monkeypatch):
    import litdatamatcher.analysis_batch as batch
    import litdatamatcher.v2 as v2

    root, inputs, counts = native
    monkeypatch.setattr(v2, "analyze", lambda *a, **k: {"status": "PARTIAL"})
    monkeypatch.setattr(batch, "_case_receipt", lambda *a: (False, []))
    out = tmp_path / "batch"
    with pytest.raises(RuntimeError, match="stopped by policy"):
        run_batch(root, out, tmp_path / "model", tmp_path / "embeddings", inputs, stop_on_error=True)
    state = json.loads((out / "BATCH_STATE.json").read_text())
    assert set(state["cases"]) == {"case1"}
    receipt = json.loads(next((out / "failures").glob("*.json")).read_text())
    assert receipt["finished_utc"] and receipt["error"]["type"] == "RuntimeError"


def test_session_rejects_backend_access_outside_owner_claim_and_changed_sources(native, tmp_path):
    root, inputs, counts = native
    session = AnalysisSession(root, tmp_path / "model", tmp_path / "embeddings")
    with pytest.raises(OwnershipConflict):
        _ = session.runtime
    with pytest.raises(OwnershipConflict):
        session.retriever([])
    write_rows(root / "catalog/studies.jsonl", [{"dataset_id": "changed"}])
    with pytest.raises(ValueError, match="changed"):
        session.validate_unchanged()


def qualification_fixture():
    from litdatamatcher.semantic_runtime import digest

    config = {"device": "cpu", "dtype": "float32", "decoding_mode": "source_schema", "max_new_tokens": 384}
    backend = {"model_manifest_sha256": "a" * 64, "config_sha256": digest(config),
               "prompt_sha256": "b" * 64, "prompt_version": "fixture-prompt", "schema_version": "fixture-schema",
               "implementation_sha256": "c" * 64, "torch_version": "fixture", "transformers_version": "fixture",
               "decoding": {"mode": "source_schema", "implementation_sha256": "d" * 64}}
    expected = {**deepcopy(backend), "input_sha256": "e" * 64}
    expected["decoding"].update(schema_sha256="f" * 64, complete_sentence_choices=3)
    report = {"qualification_schema": "runtime-qualification-v2.3", "status": "PASS",
              "backend_identity": deepcopy(backend), "expected_identity": expected,
              "configuration": config, "replay_identity": deepcopy(expected), "replay_configuration": deepcopy(config),
              "replay_origin": "cache_replay", "network_control": {"blocked_probe": ["fixture-blocked-socket"]},
              "fresh": {"claims": [{"fixture": True}], "questions": [], "inference_manifest": {
                  "origin": "fresh_local_inference", "model_revision": "fixture-revision",
                  "fingerprint": deepcopy(expected), "configuration": deepcopy(config)}}}
    class Runtime:
        model_manifest = {"revision": "fixture-revision"}

        def backend_identity(self):
            return deepcopy(backend)

        def inference_fingerprint(self, document):
            raise AssertionError("Backend qualification must not require the new document's dynamic schema")

    return Runtime(), report


def test_qualification_uses_static_backend_and_preserves_full_trial_identity(tmp_path):
    runtime, report = qualification_fixture()
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    accepted, checksum = qualified_runtime(path, runtime)
    assert accepted and len(checksum) == 64


@pytest.mark.parametrize("mutation", ["backend", "replay", "fresh", "configuration", "source", "revision", "empty", "schema", "static_identity"])
def test_qualification_rejects_identity_and_configuration_drift(tmp_path, mutation):
    runtime, report = qualification_fixture()
    if mutation == "backend":
        report["backend_identity"]["decoding"]["mode"] = "unconstrained"
    elif mutation == "replay":
        report["replay_identity"]["input_sha256"] = "0" * 64
    elif mutation == "fresh":
        report["fresh"]["inference_manifest"]["fingerprint"]["decoding"]["schema_sha256"] = "0" * 64
    elif mutation == "configuration":
        report["replay_configuration"]["max_new_tokens"] = 1024
    elif mutation == "source":
        for identity in (report["expected_identity"], report["replay_identity"], report["fresh"]["inference_manifest"]["fingerprint"]):
            identity["input_sha256"] = ""
    elif mutation == "revision":
        report["fresh"]["inference_manifest"]["model_revision"] = "different"
    elif mutation == "empty":
        report["fresh"]["claims"] = []
    elif mutation == "schema":
        report["qualification_schema"] = "runtime-qualification-v2.2"
    elif mutation == "static_identity":
        for identity in (report["expected_identity"], report["replay_identity"], report["fresh"]["inference_manifest"]["fingerprint"]):
            identity["prompt_sha256"] = "0" * 64
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert qualified_runtime(path, runtime)[0] is False


def test_decoder_environment_change_invalidates_session(native, tmp_path, monkeypatch):
    import litdatamatcher.analysis_session as sessions

    root, inputs, counts = native
    monkeypatch.setattr(sessions, "runtime_environment", lambda: {"lm_format_enforcer": "fixture-1"})
    session = AnalysisSession(root, tmp_path / "model", tmp_path / "embeddings")
    monkeypatch.setattr(sessions, "runtime_environment", lambda: {"lm_format_enforcer": "fixture-2"})
    with pytest.raises(ValueError, match="decoder environment changed"):
        session.validate_unchanged()


def test_cli_forwards_explicit_decoding_configuration(monkeypatch):
    import litdatamatcher.analysis_batch as batches
    import litdatamatcher.v2 as v2

    captured = {}
    def capture(*args, **kwargs):
        captured.update(kwargs)
        return {"status": "PASS", "failed": 0, "retryable": 0}
    monkeypatch.setattr(batches, "run_batch", capture)
    assert v2.main(["batch", "--root", "root", "--out", "out", "--model", "model", "--embeddings", "embed",
                    "--inputs", "cases.jsonl", "--decoding-mode", "source_schema", "--max-new-tokens", "384"]) == 0
    assert captured["runtime_config"] == {"decoding_mode": "source_schema", "max_new_tokens": 384}
