"""Independent runtime follow-up fixtures; no model loading or scientific execution."""

import json

import pytest

from litdatamatcher.semantic_runtime import parse_model_json, validate_extraction
from litdatamatcher.v2 import analyze, read_rows, write_rows


@pytest.fixture(autouse=True)
def deny_network_and_model_loading(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Runtime review attempted network transport or model loading")

    for target in ("requests.sessions.Session.request", "socket.create_connection", "socket.socket.connect", "litdatamatcher.semantic_runtime.LocalSemanticRuntime._load", "litdatamatcher.semantic_runtime.PretrainedSemanticIndex.__init__"):
        monkeypatch.setattr(target, forbidden)


def claim(quote, subject="We", predicate="observed", object_text="signal", status="interpretation"):
    return {"quote": quote, "subject": subject, "predicate": predicate, "object": object_text,
            "direction": "unknown", "negated": False, "status": status, "context": None, "comparator": None}


@pytest.mark.parametrize("quote,subject,predicate,object_text", [
    ("We reviewed 12 studies showing reduced mortality after treatment.", "We", "reviewed", "12 studies"),
    ("This meta-analysis included 12 randomized trials of the treatment.", "This meta-analysis", "included", "12 randomized trials"),
])
def test_explicit_review_findings_cannot_be_promoted_to_current_experiments(quote, subject, predicate, object_text):
    result = validate_extraction({"claims": [claim(quote, subject, predicate, object_text, "direct_experiment")], "questions": []}, {"document_id": "fixture-review", "text": quote})
    assert not result["claims"]
    assert result["rejected"]


@pytest.mark.parametrize("extra", [{"novelty_claim": "The first ever result in this field"}, {"expert_validated": True, "calibrated_probability": 0.99}])
def test_extra_scientific_claim_fields_cannot_bypass_source_guard(extra):
    quote = "We observed a signal of 4 units."
    candidate = {**claim(quote), **extra}
    result = validate_extraction({"claims": [candidate], "questions": []}, {"document_id": "fixture-extra", "text": quote})
    assert not result["claims"]
    assert result["rejected"]


@pytest.mark.parametrize("prefix,suffix", [("", ""), ("```json\n", "\n```"), ("```\n", "\n```"), (" \n```json\n", "\n```\t ")])
def test_complete_transport_fence_preserves_json_content(prefix, suffix):
    payload = {"claims": [], "questions": [{"quote": "Is the mechanism unknown?"}]}
    assert parse_model_json(prefix + json.dumps(payload) + suffix) == payload


@pytest.mark.parametrize("raw", [
    '```json\n{"claims": [], "questions": []}',
    '```json\n{"claims": [], "questions": []}\n```\ntrailing prose',
    'prefix prose\n```json\n{"claims": [], "questions": []}\n```',
    '```python\n{"claims": [], "questions": []}\n```',
    '{"claims": [], "questions": []} {"claims": [], "questions": []}',
    '```json\n{"claims": [], "questions": [],}\n```',
])
def test_transport_handling_does_not_repair_non_json_content(raw):
    with pytest.raises(ValueError):
        parse_model_json(raw)


def source_bound_run(tmp_path, monkeypatch, source_id, *, excluded=False):
    import litdatamatcher.semantic_runtime as runtime_module

    executed = []

    class Runtime:
        model_manifest = {"model_id": "runtime-followup-fixture", "revision": "fixture", "license": "fixture"}

        def __init__(self, *args, **kwargs):
            pass

        def extract(self, view, *args, **kwargs):
            executed.append(view)
            valid = validate_extraction({"claims": [claim(view["text"])], "questions": []}, view)
            return {**valid, "inference_manifest": {"origin": "synthetic_test_stub", "fingerprint": {"prompt_sha256": "fixture"}}}

    class Index:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, records):
            self.records = records

        def search(self, *args, **kwargs):
            return [{"id": row["id"], "score": 0.5} for row in self.records]

    monkeypatch.setattr(runtime_module, "LocalSemanticRuntime", Runtime)
    monkeypatch.setattr(runtime_module, "PretrainedSemanticIndex", Index)
    root, out = tmp_path / "root", tmp_path / "run"
    document = {"document_id": "fixture-source", "title": "Synthetic source", "text": "We observed a signal of 4 units.", "source_locator": "fixture:source", "split_context": "development"}
    if excluded:
        document["version_relationships"] = {"is-retracted-by": [{"id": "fixture-notice"}]}
    write_rows(root / "catalog/literature.jsonl", [document])
    write_rows(root / "catalog/studies.jsonl", [{"dataset_id": "fixture-study", "title": "Synthetic catalog", "organisms": ["Homo sapiens"], "source_provenance": {"source_url": "fixture:study"}}])
    write_rows(root / "catalog/processed_inspections.jsonl", [])
    result = analyze(root, out, tmp_path / "unused-model", tmp_path / "unused-embedding", question="Which catalogs are relevant?", question_source_id=source_id,
                     requirements=[{"field": "species", "expected": "human"}], limit=1, chunks=1)
    return result, out, executed


def test_explicit_source_binding_links_context_without_claiming_direct_support(tmp_path, monkeypatch):
    result, out, executed = source_bound_run(tmp_path, monkeypatch, "fixture-source")
    assert len(executed) == 1
    bundle = read_rows(out / "evidence_bundles.jsonl")[0]
    assert bundle["gap_status"] == "insufficient-coverage"
    assert bundle["evidence_items"][0]["integration_mode"] == "CONTEXT_ONLY_OR_UNRESOLVED"
    assert bundle["evidence_items"][0]["answers_question"] is False
    assert bundle["novelty_claim"] == "Limited to recorded searched coverage; no global novelty assertion"
    dossier = read_rows(out / "scientific_dossiers.jsonl")[0]
    assert dossier["question"]["source_document_id"] == "fixture-source"
    assert dossier["review_status"] == "SOURCE_ASSISTED_PENDING_EXPERT_REVIEW"
    assert result["status"] == "PASS"


def test_no_source_binding_does_not_create_unrelated_context(tmp_path, monkeypatch):
    _, out, _ = source_bound_run(tmp_path, monkeypatch, None)
    assert read_rows(out / "evidence_bundles.jsonl")[0]["evidence_items"] == []
    assert read_rows(out / "scientific_dossiers.jsonl") == []


@pytest.mark.parametrize("source_id,excluded", [("not-selected", False), ("fixture-source", True)])
def test_question_source_binding_cannot_reintroduce_unselected_or_retracted_documents(tmp_path, monkeypatch, source_id, excluded):
    with pytest.raises(ValueError, match="selected active documents"):
        source_bound_run(tmp_path, monkeypatch, source_id, excluded=excluded)
