"""Synthetic infrastructure fixtures; these are never counted as real acquisition."""

from __future__ import annotations

import json

import pytest

from litdatamatcher.data_plane import Catalog, digest
from litdatamatcher.phase2_benchmark import (
    benchmark_point,
    build_queries,
    check_recovery,
    file_sha256,
    ranking_metrics,
    reference_label,
    run_benchmark,
    unique_records,
    verify_acquisition_input,
)


def source_record(identity, organism="human", assay="clinical study registry metadata"):
    return {
        "dataset_id": identity,
        "title": "Synthetic unit test only",
        "source": "TEST_FIXTURE",
        "organisms": [organism] if organism else [],
        "assay_types": [assay] if assay else [],
        "metadata": {
            "source_provenance": {
                "source_locator": "https://example.invalid/unit-fixture/" + identity
            }
        },
    }


def write_records(path, records):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )


def test_input_identity_and_receipt_integrity_are_enforced(tmp_path):
    path = tmp_path / "datasets.jsonl"
    row = source_record("fixture-id")
    write_records(path, [row, row])
    records, metadata = unique_records(path, "dataset")
    assert records == [row]
    assert metadata["identical_duplicate_rows"] == 1
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({"status": "PASS", "files": {str(path): {"sha256": file_sha256(path)}}}),
        encoding="utf-8",
    )
    assert verify_acquisition_input(path, receipt)["status"] == "PASS"
    qualification = tmp_path / "qualification.json"
    qualification.write_text(
        json.dumps(
            {"status": "PASS", "artifact": {"path": path.name, "sha256": file_sha256(path)}}
        ),
        encoding="utf-8",
    )
    assert verify_acquisition_input(path, qualification)["status"] == "PASS"
    write_records(path, [row, dict(row, title="conflicting input")])
    with pytest.raises(ValueError, match="Conflicting"):
        unique_records(path, "dataset")
    with pytest.raises(ValueError, match="matching acquisition hash"):
        verify_acquisition_input(path, receipt)
    with pytest.raises(ValueError, match="matching acquisition hash"):
        verify_acquisition_input(path, qualification)


def test_reference_labels_preserve_unknown_and_observed_negatives():
    requirements = [
        {"field": "species", "expected": "Homo sapiens"},
        {"field": "modality", "expected": "clinical_registry"},
    ]
    assert reference_label(requirements, source_record("a"))["label"] == "OBSERVED_FIT"
    assert (
        reference_label(requirements, source_record("b", organism="mouse"))["label"]
        == "OBSERVED_MISMATCH"
    )
    assert (
        reference_label(requirements, source_record("c", assay="WGS"))["label"]
        == "OBSERVED_MISMATCH"
    )
    assert reference_label(requirements, source_record("d", organism=None))["label"] == "UNKNOWN"
    assert (
        reference_label(requirements, source_record("e", organism="soil metagenome"))["label"]
        == "UNKNOWN"
    )
    missing = source_record("f")
    missing["metadata"] = {}
    assert reference_label(requirements, missing)["label"] == "UNKNOWN"


def test_ranking_metrics_keep_unknown_out_of_negative_denominator():
    labels = {
        "a": {"label": "UNKNOWN"},
        "b": {"label": "OBSERVED_FIT"},
        "c": {"label": "OBSERVED_MISMATCH"},
    }
    metrics = ranking_metrics(["a", "b", "c"], labels)
    assert metrics["precision_at_5_denominator"] == 2
    assert metrics["judged_precision_at_5"] == 0.5
    assert metrics["unknown_in_top5"] == 1
    assert metrics["unknown_top"] is True
    assert metrics["confirmed_invalid_top"] is False
    with pytest.raises(ValueError, match="complete candidate universe"):
        ranking_metrics(["a", "b", "b"], labels)


def test_query_selection_is_source_selected_and_order_invariant():
    rows = [source_record("a"), source_record("b", organism="mouse", assay="WGS")]
    queries = build_queries(rows)
    assert queries == build_queries(list(reversed(rows)))
    assert {row["anchor_dataset_id"] for row in queries} == {"a", "b"}
    assert all(row["label_origin"] == "source_determined" for row in queries)


def test_real_process_interruption_resume_preserves_exact_payloads(tmp_path):
    path = tmp_path / "datasets.jsonl"
    write_records(path, [source_record(f"fixture-{index}") for index in range(4)])
    expected, _ = unique_records(path, "dataset")
    report = check_recovery(path, tmp_path, expected)
    assert report["status"] == "PASS"
    first, last = report["child_runs"]
    assert first["exit_code"] == 71 and last["exit_code"] == 0
    assert first["process_id"] != last["process_id"]
    assert first["inserted_count"] == last["inserted_count"] == 2
    catalog = Catalog(tmp_path / "recovery_catalog")
    try:
        assert (
            digest(sorted(catalog.records("dataset"), key=lambda row: row["dataset_id"]))
            == last["final_payload_digest"]
        )
    finally:
        catalog.close()


def test_point_measures_real_catalog_paths_without_scientific_promotion(tmp_path):
    datasets = [source_record("a"), source_record("b", organism="mouse", assay="WGS")]
    literature = [{"document_id": "test-document", "title": "Synthetic infrastructure test only"}]
    receipt = benchmark_point(
        tmp_path, literature, datasets, build_queries(datasets), "fixture-only.jsonl"
    )
    assert all(receipt["checks"].values())
    assert receipt["dataset_count"] == 2
    assert receipt["cache_replay"]["dataset"]["reinserted_records"] == 0
    assert receipt["cache_replay"]["dataset"]["hits"] == 2
    assert receipt["fts_queries"]["n"] >= 20
    assert receipt["disk_bytes"] > 0


def test_complete_receipt_refuses_missing_real_challenge_coverage(tmp_path):
    literature = tmp_path / "literature.jsonl"
    datasets = tmp_path / "datasets.jsonl"
    write_records(literature, [{"document_id": "fixture-literature", "title": "Unit fixture"}])
    write_records(datasets, [source_record("fixture-dataset")])
    acquisition = tmp_path / "acquisition.json"
    acquisition.write_text(
        json.dumps(
            {
                "status": "PASS",
                "files": {
                    str(path): {"sha256": file_sha256(path)} for path in [literature, datasets]
                },
            }
        ),
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol.md"
    protocol.write_text(
        "Synthetic unit fixture only; no real acquisition claims.", encoding="utf-8"
    )
    output = tmp_path / "output"
    receipt = run_benchmark(literature, datasets, acquisition, output, protocol)
    assert receipt["engineering_status"] == "PASS"
    assert receipt["status"] == "FAIL_OR_MISSING_COVERAGE"
    assert receipt["metadata_evaluation_checks"]["wrong_organism_negatives"] is False
    assert receipt["metadata_evaluation_checks"]["wrong_modality_negatives"] is False
    assert receipt["recovery"]["checks"]["equal_to_clean_catalog"] is True
    assert receipt["calibration_status"] == "UNCALIBRATED_HEURISTIC"
    assert receipt["expert_validation"] == "PENDING_EXPERT_REVIEW"
    assert all(
        file_sha256(output / name) == expected
        for name, expected in receipt["artifact_hashes"].items()
    )
    with pytest.raises(ValueError, match="new empty versioned"):
        run_benchmark(literature, datasets, acquisition, output, protocol)
