import json

from litdatamatcher.scientific_v2 import dependence_groups
from litdatamatcher.v2 import (
    explicit_unresolved_questions,
    normalize_dataset,
    qualified_runtime,
    rebase_runtime_item,
    render_report,
    source_chunks,
    validate_run_artifact,
)


def test_explicit_chunks_keep_parent_offsets():
    text = "Background text.\nTreatment reduced TNF. Further work is needed."
    d = {
        "document_id": "d",
        "text": text,
        "sections": [
            {"start": 0, "end": 16, "text": text[:16], "section": "Introduction"},
            {"start": 17, "end": len(text), "text": text[17:], "section": "Results"},
        ],
    }
    chunks = source_chunks(d, max_chars=25, max_chunks=2)
    assert chunks[0]["parent_start"] == 17
    assert all(text[c["parent_start"] : c["parent_end"]] == c["text"] for c in chunks)


def test_capability_migration_retains_unknown_and_source():
    raw = {
        "dataset_id": "GSE1",
        "capabilities": {
            "paired": {"status": "unknown", "value": None, "reason": "not reported"},
            "species": {
                "status": "known",
                "value": "human",
                "source_locator": ["sample:a", "sample:b"],
            },
        },
    }
    migrated = normalize_dataset(raw)
    assert migrated["capabilities"]["paired"]["value"] is None
    assert migrated["capabilities"]["species"]["source_locator"] == "sample:a; sample:b"
    assert raw["capabilities"]["species"]["status"] == "known"


def test_aggregator_multiple_citations_join_primary_lineage():
    rows = [
        {"evidence_id": "paper", "publication_id": "PMID:123"},
        {"evidence_id": "curation", "primary_publication_ids": ["PMID123", "PMID456"]},
    ]
    assert len(dependence_groups(rows)) == 1


def test_report_escapes_hostile_metadata(tmp_path):
    (tmp_path / "RUN_MANIFEST.json").write_text(
        json.dumps(
            {"run_id": "<script>alert(1)</script>", "execution_status": "PARTIAL", "coverage": {}}
        )
    )
    for name in ["questions", "matches", "evidence_bundles"]:
        (tmp_path / (name + ".jsonl")).write_text("")
    report = render_report(tmp_path).read_text()
    assert "<script>alert" not in report and "&lt;script&gt;" in report
    assert "default-src 'none'" in report


def test_explicit_gap_question_is_source_linked_without_novelty_claim():
    text = "Observed result. Further research is needed to resolve dose effects."
    document = {"document_id": "d1", "text": text, "source_locator": "https://example.test/d1"}
    view = source_chunks(document)[0]
    result = explicit_unresolved_questions(document, view)
    assert len(result) == 1
    question = result[0]
    assert (
        text[question["evidence_span"]["start"] : question["evidence_span"]["end"]]
        == question["question"]
    )
    assert question["gap_status"] == "insufficient-coverage"
    assert question["novelty_claim"].startswith("None")


def test_rebased_runtime_ids_use_parent_offsets():
    text = "Treatment reduced TNF. Treatment reduced TNF."
    document = {"document_id": "d1", "text": text, "source_locator": "source"}
    item = {
        "claim_id": "local",
        "evidence_span": {"start": 0, "end": 22, "text": "Treatment reduced TNF."},
        "source_document_id": "view",
    }
    first = rebase_runtime_item(item, document, {"parent_start": 0}, "v1")
    second = rebase_runtime_item(item, document, {"parent_start": 23}, "v2")
    assert first["claim_id"] != second["claim_id"]
    assert second["source_document_id"] == "d1"


def test_runtime_qualification_requires_matching_fresh_model(tmp_path):
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    runtime = type("Runtime", (), {"model_manifest": {"revision": "real-revision"}})()
    assert qualified_runtime(path, runtime) == (False, None)
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "fresh": {
                    "inference_manifest": {
                        "origin": "fresh_local_inference",
                        "model_revision": "real-revision",
                    }
                },
                "replay_origin": "cache_replay",
                "network_control": {"blocked_probe": ["socket.connect"]},
            }
        ),
        encoding="utf-8",
    )
    assert qualified_runtime(path, runtime)[0] is True


def test_artifact_validation_parses_content(tmp_path):
    good = tmp_path / "records.jsonl"
    bad = tmp_path / "bad.jsonl"
    good.write_text('{"id": 1}\n', encoding="utf-8")
    bad.write_text("{broken}\n", encoding="utf-8")
    assert validate_run_artifact(good) == "PASS"
    assert validate_run_artifact(bad) == "FAIL"
