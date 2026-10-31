"""Integrated application jobs: scientific abstention, evidence, recovery, HTTP boundaries."""

import json
import threading

import pytest
import requests

from litdatamatcher.controller import Controller
from litdatamatcher.data_plane import atomic_json
from litdatamatcher.platform_server import PlatformService, make_server
from litdatamatcher.platform_workflow import (
    _merge_documents,
    _select,
    run_workflow,
    validate_request,
)
from litdatamatcher.v2 import document_lifecycle_status, normalize_dataset


@pytest.fixture
def catalog(tmp_path):
    root = tmp_path / "input"
    (root / "catalog").mkdir(parents=True)
    document = {
        "document_id": "test:paper",
        "title": "Age and diabetes",
        "text": "Whether age predicts diabetes remains unclear.",
        "source_locator": "https://example.org/paper",
    }

    def study(key, value):
        return {
            "dataset_id": key,
            "title": "Age diabetes measurements",
            "summary": "Age and diabetes observations",
            "capabilities": {
                "feature": {
                    "value": value,
                    "status": "observed",
                    "source_locator": "https://example.org/data#columns",
                }
            },
            "availability": "PUBLIC",
        }

    for file, rows in (
        ("literature.jsonl", [document]),
        (
            "studies.jsonl",
            [
                study("positive", "age"),
                study("negative", "BMI"),
                {"dataset_id": "unknown", "title": "Age diabetes metadata", "capabilities": {}},
            ],
        ),
    ):
        (root / "catalog" / file).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    return root


def request(**overrides):
    return {
        "query": "age diabetes",
        "question": "Age and diabetes measurements",
        "requirements": [{"field": "feature", "expected": "age", "essential": True}],
        "literature_sources": [],
        "dataset_sources": [],
        "inspection_limit": 0,
        **overrides,
    }


def test_end_to_end_qualification_and_resume_preserve_evidence(catalog, tmp_path):
    out = tmp_path / "output"
    checked = validate_request(request())
    result = run_workflow(checked, {"catalog_root": str(catalog), "offline": True}, out)
    assert [r["dataset_id"] for r in result["candidates"]] == ["positive", "unknown", "negative"]
    assert [r["assessment"]["eligibility"] for r in result["candidates"]] == [
        "DIRECT_FIT",
        "REQUIRES_INSPECTION",
        "NOT_QUALIFIED",
    ]
    assert result["candidates"][0]["assessment"]["statistical_adequacy"] == "UNKNOWN"
    assert "https://example.org/data#columns" in (out / "dossier.md").read_text(encoding="utf-8")
    checkpoint = out / ".workflow/state.json"
    hashes = json.loads(checkpoint.read_text())["stages"]
    replay = run_workflow(checked, {"catalog_root": str(catalog), "offline": True}, out)
    assert replay == result
    assert hashes == json.loads(checkpoint.read_text())["stages"]
    (out / "dossier.md").write_text("damaged", encoding="utf-8")
    repaired = run_workflow(checked, {"catalog_root": str(catalog), "offline": True}, out)
    assert repaired["result_id"] == result["result_id"]
    assert "damaged" not in (out / "dossier.md").read_text()
    assert json.loads(checkpoint.read_text())["repairs"]
    (catalog / "catalog/studies.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run_workflow(checked, {"catalog_root": str(catalog), "offline": True}, out)


def test_underdetermined_question_cannot_rank(catalog, tmp_path):
    result = run_workflow(
        validate_request(request(question="Could this matter?", requirements=[])),
        {"catalog_root": str(catalog)},
        tmp_path / "run",
    )
    assert result["status"] == "QUESTION_UNDERDETERMINED"
    assert result["candidates"] == []
    assert result["datasets"]


def test_observed_variables_are_searchable_without_title_overlap():
    row = {
        "dataset_id": "study",
        "title": "Unrelated title",
        "capabilities": {
            "measured_variable": {
                "status": "observed",
                "source_locator": "https://example.org/columns",
                "value": ["radius1", "diagnosis"],
            }
        },
    }
    assert _select([row], "radius1 diagnosis", 5, "dataset_id") == [row]


def test_workflow_comparison_scope_stays_bound_to_one_question(catalog, tmp_path):
    from litdatamatcher.tabular_contracts import comparison_constraints, inspect_tabular_comparisons

    table = tmp_path / "measurements.csv"
    table.write_text("value,group\n1,A\n2,B\n", encoding="utf-8")
    metadata = {
        "dataset_id": "measured",
        "num_instances": 2,
        "variables": [
            {
                "name": "value",
                "role": "Feature",
                "type": "Continuous",
                "description": "Recorded concentration",
            },
            {
                "name": "group",
                "role": "Target",
                "type": "Categorical",
                "description": "Recorded source group A or B",
            },
        ],
    }
    specification = {
        "outcome_variable": "value",
        "grouping_variable": "group",
        "group_values": ["A", "B"],
        "definition_evidence": {
            row["name"]: {"json_pointer": f"/variables/{i}/description", "text": row["description"]}
            for i, row in enumerate(metadata["variables"])
        },
    }
    profile = inspect_tabular_comparisons(
        table,
        metadata,
        [specification],
        source_url="https://example.org/table.csv",
        metadata_url="https://example.org/definitions",
    )
    (catalog / "catalog/studies.jsonl").write_text(json.dumps(profile) + "\n", encoding="utf-8")
    constraints = comparison_constraints("value", "group", ["A", "B"])
    question = "What is the difference between group A and group B in recorded value?"
    result = run_workflow(
        validate_request(request(query="value group", question=question, requirements=constraints)),
        {"catalog_root": str(catalog)},
        tmp_path / "comparison",
    )
    assert result["candidates"][0]["assessment"]["eligibility"] == "DIRECT_FIT"
    assessed = {r["field"]: r for r in result["candidates"][0]["assessment"]["requirements"]}
    assert assessed["joint_observation"]["status"] == "MATCH"
    assert assessed["outcome_ascertainment"]["status"] == "MATCH"
    assert result["candidates"][0]["dataset"]["comparison_scope"]["outcome_variable"] == "value"
    assert "comparison_scope" not in result["datasets"][0]
    assert "joint_observation" not in result["datasets"][0]["capabilities"]
    alternate = run_workflow(
        validate_request(request(query="value group", question=question, requirements=[])),
        {"catalog_root": str(catalog)},
        tmp_path / "unbound",
    )
    assert alternate["status"] == "QUESTION_UNDERDETERMINED"
    assert alternate["candidates"] == []


def test_same_pmid_preserves_alternate_source_and_lifecycle():
    first = {"document_id": "PMID:123", "text": "Future work is needed.", "source": "pubmed"}
    second = {
        "document_id": "pubmed:123",
        "pmid": "123",
        "text": "Updated source.",
        "source": "pubmed",
        "version_relationships": {"retracted": True},
    }
    rows = _merge_documents([first, second])
    assert len(rows) == 1
    assert rows[0]["document_id"] == "PMID:123"
    assert rows[0]["metadata"]["merged_source_records"][0]["text"] == "Updated source."
    assert document_lifecycle_status(rows[0]) == "RETRACTED"


def test_normalization_preserves_verified_processed_file_access():
    sha = "a" * 64
    url = "https://example.org/data.csv"
    row = {
        "dataset_id": "study",
        "availability": "PUBLIC_FILE_INSPECTED",
        "inspection_status": "INSPECTED_MEASUREMENTS",
        "inspection": {"source_sha256": sha, "source_url": url},
        "capabilities": {
            "processed_measurements": {
                "value": True,
                "status": "observed",
                "source_locator": url + "#sha256=" + sha,
            }
        },
    }
    assert normalize_dataset(row)["availability"] == "PUBLIC_FILE_INSPECTED"
    row["inspection"]["source_sha256"] = "invalid"
    assert normalize_dataset(row)["availability"] == "UNKNOWN"


def test_exact_supported_assay_does_not_require_an_extra_ontology_alias():
    from litdatamatcher.scientific_v2 import assess_requirements

    study = {
        "dataset_id": "array",
        "capabilities": {
            "assay": {
                "value": ["microarray"],
                "status": "observed",
                "source_locator": "https://example.org/assay",
            }
        },
    }
    requirement = {"field": "assay", "expected": "microarray"}
    assert assess_requirements([requirement], study)["requirements"][0]["status"] == "MATCH"
    requirement["expected"] = "RNA sequencing"
    assert (
        assess_requirements([requirement], normalize_dataset(study))["requirements"][0]["status"]
        == "MISMATCH"
    )
    study["capabilities"]["assay"]["status"] = "derived"
    requirement["expected"] = "microarray"
    assert assess_requirements([requirement], study)["requirements"][0]["status"] == "UNKNOWN"


@pytest.mark.parametrize(
    "changes",
    [
        {"limit": True},
        {"limit": 21},
        {"dataset_sources": ["unknown"]},
        {"model_dir": "remote"},
        {"requirements": [{"field": "feature", "expected": "age", "essential": "yes"}]},
        {"documents": [{"text": "source", "url": "file:///secret"}]},
    ],
)
def test_invalid_unbounded_or_path_injection_requests(changes):
    with pytest.raises(ValueError):
        validate_request(request(**changes))


def test_controller_executes_real_worker_and_receipt_protects_exports(catalog, tmp_path):
    service = PlatformService(
        tmp_path / "service", configuration={"catalog_root": str(catalog), "offline": True}
    )
    submitted = service.submit(request())
    controller = Controller(service.queue)
    try:
        executed = controller.run_next(timeout=60)
        assert executed["status"] == "SUCCEEDED", executed
    finally:
        controller.close()
    result = service.job(submitted["id"])
    assert result["result"]["candidates"][0]["dataset_id"] == "positive"
    controller = Controller(service.queue)
    try:
        controller._receipt(submitted["id"]).unlink()
        recovered = controller.run_next(timeout=60)
        assert recovered["status"] == "SUCCEEDED", recovered
    finally:
        controller.close()
    atomic_json(service._job_path(submitted["id"]) / "run/result.json", {"tampered": True})
    with pytest.raises(ValueError, match="receipt mismatch"):
        service.job(submitted["id"])


def test_source_failures_are_explicit_and_retryable(tmp_path):
    service = PlatformService(tmp_path / "service", configuration={"offline": True})
    submitted = service.submit(request(dataset_sources=["geo"]))
    controller = Controller(service.queue)
    try:
        assert controller.run_next(timeout=60)["status"] == "SUCCEEDED"
    finally:
        controller.close()
    result = service.job(submitted["id"])
    assert result["result"]["processing_status"] == "PARTIAL_RETRYABLE"
    assert result["result"]["retryable_stages"] == ["discovery"]
    assert result["retryable"] is True
    assert service.retry(submitted["id"])["status"] == "QUEUED"


def test_missing_configured_catalog_is_not_empty_search(tmp_path):
    with pytest.raises(FileNotFoundError, match="catalog is missing"):
        run_workflow(
            validate_request(request()),
            {"catalog_root": str(tmp_path / "missing")},
            tmp_path / "run",
        )


@pytest.fixture
def api(tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><p>App</p>")
    service = PlatformService(tmp_path / "service", web_root=web)
    server = make_server(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", service
    server.shutdown()
    server.server_close()
    thread.join(5)


def test_http_rejects_cross_origin_dns_rebinding_and_invalid_contract(api):
    url, service = api
    assert requests.get(url + "/api/health", timeout=5).json()["status"] == "ok"
    page = requests.get(url + "/", timeout=5)
    assert page.status_code == 200
    # The existing lightweight workflow calls both public endpoints in the browser.
    policy = page.headers["Content-Security-Policy"]
    assert "https://api.datacite.org" in policy
    assert "https://www.ebi.ac.uk" in policy
    assert (
        requests.post(
            url + "/api/jobs",
            json=request(),
            headers={"Origin": "https://untrusted.example"},
            timeout=5,
        ).status_code
        == 403
    )
    assert (
        requests.get(url + "/api/health", headers={"Host": "evil.example"}, timeout=5).status_code
        == 403
    )
    assert (
        requests.post(
            url + "/api/jobs",
            data=json.dumps(request()),
            headers={"Content-Type": "text/plain"},
            timeout=5,
        ).status_code
        == 400
    )
    assert requests.post(url + "/api/jobs", json=request(limit=1000), timeout=5).status_code == 400
    assert service.jobs()["jobs"] == []
    response = requests.post(
        url + "/api/requirements", json={"question": "Could this matter?"}, timeout=5
    )
    assert response.json()["matching_requirements"] == []
    assert requests.post(url + "/api/pause", json={}, timeout=5).json()["mode"] == "PAUSED_BY_USER"
    assert requests.post(url + "/api/resume", json={}, timeout=5).json()["mode"] == "RUNNING"


def test_selected_document_ids_keep_original_source_and_reject_missing(catalog, tmp_path):
    selected = validate_request(request(query="no matching words", document_ids=["test:paper"]))
    result = run_workflow(selected, {"catalog_root": str(catalog)}, tmp_path / "selected")
    assert result["documents"][0]["document_id"] == "test:paper"
    assert result["documents"][0]["source_locator"] == "https://example.org/paper"
    assert result["documents"][0].get("source_type") != "user_submitted_unverified"
    with pytest.raises(ValueError, match="Selected documents are unavailable"):
        run_workflow(
            validate_request(request(document_ids=["missing"])),
            {"catalog_root": str(catalog)},
            tmp_path / "missing",
        )
