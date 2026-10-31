"""Saved literal question selection is a verified binding, not browser provenance."""

import json
from copy import deepcopy

import pytest

from litdatamatcher.controller import Controller
from litdatamatcher.data_plane import atomic_json
from litdatamatcher.platform_server import PlatformService
from litdatamatcher.platform_workflow import run_workflow, validate_request

SOURCE = {
    "document_id": "PMID:123",
    "pmid": "123",
    "title": "Synthetic context fixture",
    "text": "NAION is a serious optic neuropathy. The risk of NAION needs investigation. Can semaglutide increase this risk in adults?",
    "source_locator": "https://example.org/source",
    "source_provenance": {"status": "test_fixture_not_scientific_evidence"},
}


def raw(**changes):
    return {
        "query": "semaglutide risk",
        "question": "",
        "requirements": [],
        "literature_sources": [],
        "dataset_sources": [],
        "inspection_limit": 0,
        **changes,
    }


@pytest.fixture
def setup(tmp_path):
    catalog = tmp_path / "catalog-root"
    (catalog / "catalog").mkdir(parents=True)
    (catalog / "catalog/literature.jsonl").write_text(json.dumps(SOURCE) + "\n", encoding="utf-8")
    (catalog / "catalog/studies.jsonl").write_text("", encoding="utf-8")
    service = PlatformService(
        tmp_path / "service", configuration={"catalog_root": str(catalog), "offline": True}
    )
    return service, catalog


def execute(service):
    controller = Controller(service.queue)
    try:
        executed = controller.run_next(timeout=60)
        assert executed["status"] == "SUCCEEDED", executed
    finally:
        controller.close()


def test_verified_selection_preserves_context_and_modified_lineage_without_model(
    setup, tmp_path, monkeypatch
):
    service, catalog = setup
    first = service.submit(raw(workflow_mode="questions"))
    execute(service)
    saved = service.job(first["id"])["result"]
    original = saved["questions"]["questions"][0]
    reference = {"job_id": first["id"], "question_id": original["question_id"]}
    prior_bytes = (service._job_path(first["id"]) / "run/result.json").read_bytes()
    with pytest.raises(ValueError, match="not uniquely present"):
        service.preview({"question_reference": dict(reference, question_id="rq_" + "0" * 24)})
    controller = Controller(service.queue)
    try:
        receipt_path = controller._receipt(first["id"])
    finally:
        controller.close()
    receipt_bytes = receipt_path.read_bytes()
    receipt_path.unlink()
    with pytest.raises(FileNotFoundError):
        service.preview({"question_reference": reference})
    receipt_path.write_bytes(receipt_bytes)
    real_check = Controller._check_outputs

    def replace_after_verification(controller, row, against_receipt=False):
        verified = real_check(controller, row, against_receipt)
        atomic_json(
            service._job_path(first["id"]) / "run/result.json", {"tampered": "between reads"}
        )
        return verified

    with monkeypatch.context() as patch:
        patch.setattr(Controller, "_check_outputs", replace_after_verification)
        with pytest.raises(ValueError, match="changed during read"):
            service.preview({"question_reference": reference})
    (service._job_path(first["id"]) / "run/result.json").write_bytes(prior_bytes)
    assert service.export(first["id"], "result.json") == prior_bytes
    for filename in ("result.json", "dossier.md", "candidates.csv"):
        export_path = service._job_path(first["id"]) / "run" / filename
        export_bytes = export_path.read_bytes()

        def replace_export_after_check(
            controller, row, against_receipt=False, target=export_path, original_bytes=export_bytes
        ):
            verified = real_check(controller, row, against_receipt)
            target.write_bytes(original_bytes + b"\n")
            return verified

        with monkeypatch.context() as patch:
            patch.setattr(Controller, "_check_outputs", replace_export_after_check)
            with pytest.raises(ValueError, match="changed during read"):
                service.export(first["id"], filename)
        export_path.write_bytes(export_bytes)

    constraints = [{"field": "paired", "expected": False, "essential": False}]
    preview = service.preview({"question_reference": reference, "requirements": constraints})
    assert preview["question_selection"]["question_id"] == original["question_id"]
    assert (
        next(r for r in preview["matching_requirements"] if r["field"] == "outcome")["expected"]
        == "risk of NAION"
    )
    assert (
        next(r for r in preview["matching_requirements"] if r["field"] == "paired")["expected"]
        is False
    )

    # Reusing a saved model extraction does not need a local model or invoke inference.
    selected = service.submit(raw(question_reference=reference, extraction_mode="model"))
    bound = selected["request"]
    assert bound["source_question_binding"]["question"] == original
    assert bound["source_question_binding"]["document"]["text"] == SOURCE["text"]
    execute(service)
    result = service.job(selected["id"])["result"]
    for key in (
        "question_id",
        "citation",
        "evidence_span",
        "source_provenance",
        "origin",
        "source_text_sha256",
    ):
        assert result["question"][key] == original[key]
    assert result["questions"]["coverage"]["reused_documents"] == 1
    assert result["question"]["selection_reference"] == reference
    assert (
        result["compilation"]["matching_requirements"]
        == service.preview({"question_reference": reference})["matching_requirements"]
    )

    modified_text = "Can semaglutide decrease this risk in adults?"
    modified = service.preview({"question": modified_text, "question_reference": reference})
    assert modified["question_selection"]["provenance_category"] == "USER_MODIFIED"
    assert modified["question_selection"]["source_stated"] is False
    assert "evidence_span" not in modified["question_selection"]
    assert modified["question_selection"]["derived_from"] == original
    context_evidence = next(r for r in modified["entity_roles"] if r["field"] == "outcome")[
        "antecedent_evidence"
    ]
    assert context_evidence["source_locator"].startswith(SOURCE["source_locator"] + "#chars=")
    assert context_evidence["source_text_sha256"] == original["source_text_sha256"]
    assert (
        SOURCE["text"][context_evidence["start"] : context_evidence["end"]]
        == context_evidence["text"]
    )

    # Freeze at submission. Later source edits and prior-output corruption cannot
    # alter the canonical binding or silently rerun the original extraction.
    (catalog / "catalog/literature.jsonl").write_text(
        json.dumps(dict(SOURCE, text="Updated unrelated text.")) + "\n", encoding="utf-8"
    )
    atomic_json(service._job_path(first["id"]) / "run/result.json", {"tampered": True})
    with pytest.raises(ValueError, match="receipt mismatch"):
        service.submit(raw(question_reference=reference))
    from litdatamatcher import semantic_runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved extraction must not load a model")

    monkeypatch.setattr(semantic_runtime, "LocalSemanticRuntime", forbidden)
    replay = run_workflow(
        dict(bound, question=modified_text), service.configuration, tmp_path / "frozen"
    )
    assert replay["documents"][0]["text"] == SOURCE["text"]
    assert replay["question"]["derived_from"] == original
    assert replay["compilation"]["entity_roles"] == modified["entity_roles"]
    assert "user modification" in (tmp_path / "frozen/dossier.md").read_text(encoding="utf-8")
    assert prior_bytes != (service._job_path(first["id"]) / "run/result.json").read_bytes()
    (catalog / "catalog/literature.jsonl").write_text(
        json.dumps(dict(SOURCE, text="Updated unrelated text.", lifecycle_status="RETRACTED"))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lifecycle/version review"):
        run_workflow(bound, service.configuration, tmp_path / "retracted")


def test_extraction_only_never_reads_dataset_catalog_or_runs_dataset_stages(
    setup, tmp_path, monkeypatch
):
    service, catalog = setup
    (catalog / "catalog/studies.jsonl").write_text("invalid dataset JSON", encoding="utf-8")
    from litdatamatcher import platform_workflow

    def forbidden(*args, **kwargs):
        raise AssertionError("Question-only mode must not run dataset stages")

    monkeypatch.setattr(platform_workflow, "_inspect", forbidden)
    monkeypatch.setattr(platform_workflow, "_qualify", forbidden)
    checked = validate_request(raw(workflow_mode="questions", ranking="dense"))
    result = run_workflow(checked, service.configuration, tmp_path / "questions")
    assert result["workflow_mode"] == "questions"
    assert result["status"] == "QUESTIONS_DISCOVERED"
    assert result["datasets"] == result["candidates"] == []
    assert result["question"] is result["compilation"] is None
    stages = json.loads((tmp_path / "questions/.workflow/state.json").read_text())["stages"]
    assert set(stages) == {"local_discovery", "discovery", "questions"}
    (catalog / "catalog/studies.jsonl").write_text(
        "changed unused dataset catalog", encoding="utf-8"
    )
    assert run_workflow(checked, service.configuration, tmp_path / "questions") == result
    assert (
        "Dataset discovery, inspection, and qualification were not requested"
        in (tmp_path / "questions/dossier.md").read_text()
    )
    # Unused dense ranking does not require an embedding model at submission.
    assert service.submit(raw(workflow_mode="questions", ranking="dense"))["status"] == "QUEUED"


@pytest.mark.parametrize(
    "change",
    [
        {"source_question_binding": {}},
        {"citation": {}},
        {"question_reference": {"job_id": "../invalid", "question_id": "rq_" + "a" * 24}},
        {
            "question_reference": {
                "job_id": "a" * 32,
                "question_id": "rq_" + "a" * 24,
                "source": SOURCE,
            }
        },
        {"workflow_mode": "questions", "dataset_sources": ["geo"]},
        {"workflow_mode": "questions", "inspection_limit": 1},
    ],
)
def test_client_provenance_and_mixed_extraction_modes_rejected(change):
    with pytest.raises(ValueError):
        validate_request(raw(**change))


def test_missing_or_unverified_question_context_is_rejected(setup, tmp_path, monkeypatch):
    service, _ = setup
    result = run_workflow(
        validate_request(raw(workflow_mode="questions")),
        service.configuration,
        tmp_path / "original",
    )
    q = result["questions"]["questions"][0]
    ref = {"job_id": "a" * 32, "question_id": q["question_id"]}
    with pytest.raises(FileNotFoundError):
        service.preview({"question_reference": ref})
    for mutation in (
        "queued",
        "missing_question",
        "duplicate_question",
        "missing_document",
        "duplicate_document",
        "span",
        "citation",
        "sha",
        "provenance",
        "lifecycle",
    ):
        altered = deepcopy(result)
        job = {"status": "SUCCEEDED", "result": altered}
        if mutation == "queued":
            job["status"] = "QUEUED"
        elif mutation == "missing_question":
            altered["questions"]["questions"] = []
        elif mutation == "duplicate_question":
            altered["questions"]["questions"] *= 2
        elif mutation == "missing_document":
            altered["documents"] = []
        elif mutation == "duplicate_document":
            altered["documents"] *= 2
        elif mutation == "span":
            altered["questions"]["questions"][0]["evidence_span"]["start"] += 1
        elif mutation == "citation":
            altered["questions"]["questions"][0]["citation"]["pmid"] = "999"
        elif mutation == "sha":
            altered["questions"]["questions"][0]["source_text_sha256"] = "0" * 64
        elif mutation == "provenance":
            altered["questions"]["questions"][0]["source_provenance"] = {"source_type": "forged"}
        else:
            altered["documents"][0]["lifecycle_status"] = "RETRACTED"
        monkeypatch.setattr(service, "job", lambda _, saved=job: saved)
        with pytest.raises(ValueError):
            service.preview({"question_reference": ref})


def test_pasted_text_only_extraction_and_legacy_full_default(tmp_path):
    checked = validate_request(
        raw(
            query="",
            workflow_mode="questions",
            documents=[{"title": "Pasted", "text": "Future research is needed on insulin."}],
        )
    )
    result = run_workflow(checked, {}, tmp_path / "pasted")
    assert result["questions"]["questions"]
    assert result["documents"][0]["source_type"] == "user_submitted_unverified"
    assert validate_request(raw())["workflow_mode"] == "full"
