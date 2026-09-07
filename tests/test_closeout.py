import hashlib
import json

from litdatamatcher.closeout import run_closeout_audit


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _status(report, target, kind):
    return next(
        item["status"]
        for item in report["evidence"]
        if item["target"] == target and item["kind"] == kind
    )


def test_empty_root_never_promotes_presence_or_prose_to_pass(tmp_path):
    report = run_closeout_audit(tmp_path / "data", tmp_path / "source")

    assert report["summary"].get("PASS", 0) == 0
    assert report["gates"]["G02"]["status"] == "FAIL"
    assert _status(report, "G01", "clean_install") == "NOT_RUN"
    assert report["pre_holdout_ready"] is False


def test_padded_catalog_is_insufficient_without_snapshot_and_parse_evidence(tmp_path):
    data = tmp_path / "data"
    rows = [{"document_id": f"PMID:{index}", "title": "prose says pass"} for index in range(200)]
    catalog = data / "catalog" / "literature.jsonl"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = run_closeout_audit(data, tmp_path / "source")

    assert _status(report, "G02", "literature_coverage") == "PASS"
    assert _status(report, "G02", "source_snapshot") == "FAIL"
    assert _status(report, "G02", "fulltext_parse") == "FAIL"
    assert report["gates"]["G02"]["status"] == "FAIL"


def test_historical_partial_run_is_not_a_designated_final_failure(tmp_path):
    run = tmp_path / "data" / "runs" / "partial" / "RUN_MANIFEST.json"
    _write(run, {"execution_status": "PARTIAL", "failures": [{"stage": "source_guard"}]})

    report = run_closeout_audit(tmp_path / "data", tmp_path / "source")

    assert _status(report, "G05", "no_prewritten_or_regex_substitution") == "NOT_RUN"
    assert _status(report, "G16", "final_real_run") == "NOT_RUN"


def test_tampered_declared_artifact_fails_final_run_integrity(tmp_path):
    run_dir = tmp_path / "data" / "runs" / "final-real-run-01"
    artifact = run_dir / "inferences.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"actual":"content"}\n', encoding="utf-8")
    _write(
        run_dir / "RUN_MANIFEST.json",
        {
            "execution_status": "PASS",
            "failures": [],
            "run_id": "final-real-run-01",
            "evaluation": {"split_role": "VALIDATION", "holdout_exposed_to_tuning": False},
            "commands": [{"exit_code": 0, "log_reference": "inferences.jsonl"}],
            "artifacts": [
                {
                    "path": "inferences.jsonl",
                    "validation": "PASS",
                    "sha256": hashlib.sha256(b"different").hexdigest(),
                    "size_bytes": len(b"different"),
                }
            ],
        },
    )

    report = run_closeout_audit(tmp_path / "data", tmp_path / "source")

    assert _status(report, "G16", "final_real_run") == "FAIL"
    assert report["gates"]["G16"]["status"] == "FAIL"


def test_audit_output_is_deterministic_for_unchanged_evidence(tmp_path):
    first = run_closeout_audit(tmp_path / "data", tmp_path / "source")
    second = run_closeout_audit(tmp_path / "data", tmp_path / "source")

    assert first == second


def test_designated_final_pass_supersedes_historical_partial_runs(tmp_path):
    data = tmp_path / "data"
    _write(data / "runs" / "old" / "RUN_MANIFEST.json", {"execution_status": "PARTIAL"})
    _final_run(data, "final-real-run-01", source_disjointness=None, final_integrated=True)

    report = run_closeout_audit(data, tmp_path / "source")

    assert _status(report, "G05", "no_prewritten_or_regex_substitution") == "PASS"
    assert _status(report, "G16", "final_real_run") == "PASS"


def test_unknown_replacement_overlap_never_passes(tmp_path):
    data = tmp_path / "data"
    source = tmp_path / "source"
    _reservation(source)
    _final_run(
        data,
        "final",
        source_disjointness={
            "selected_accession": "GSE282859",
            "status": "PROVEN_SOURCE_DISJOINT",
            "unknown_overlap_count": 1,
        },
    )

    report = run_closeout_audit(data, source)

    assert _status(report, "G10", "study_grouped_holdout") == "FAIL"
    assert _status(report, "G10", "source_disjoint_test") == "FAIL"


def test_proven_replacement_final_holdout_can_replace_retired_exposed_family(tmp_path):
    data = tmp_path / "data"
    source = tmp_path / "source"
    _reservation(source)
    _final_run(
        data,
        "final",
        source_disjointness={
            "selected_accession": "GSE282859",
            "status": "PROVEN_SOURCE_DISJOINT",
            "unknown_overlap_count": 0,
        },
    )

    report = run_closeout_audit(data, source)

    assert _status(report, "G10", "study_grouped_holdout") == "PASS"
    assert _status(report, "G10", "source_disjoint_test") == "PASS"


def test_closeout_prefers_v4_reservation_when_present(tmp_path):
    data = tmp_path / "data"
    source = tmp_path / "source"
    _reservation(source)
    _write(
        source / "benchmarks" / "v2" / "final_holdout_reservation_v4.json",
        {
            "status": "RESERVED_PENDING_ONE_TIME_FINAL_HOLDOUT_AUTHORIZATION",
            "selected_accession": "GSE279879",
            "verification": {
                "all_exact_identifier_intersections_zero": True,
                "candidate_publication_pmc_complete": True,
                "candidate_ena_sra_complete": True,
                "base_official_relation_errors": 0,
                "base_official_relation_truncations": 0,
            },
            "sealed_evaluation_state": {
                "prediction_status": "NOT_RUN",
                "ranking_status": "NOT_RUN",
            },
        },
    )
    _final_run(
        data,
        "final-v4",
        source_disjointness={
            "selected_accession": "GSE279879",
            "status": "PROVEN_SOURCE_DISJOINT",
            "unknown_overlap_count": 0,
        },
    )

    report = run_closeout_audit(data, source)

    assert _status(report, "G10", "source_disjoint_test") == "PASS"


def test_junit_receipt_credits_only_exact_unskipped_contract_tests(tmp_path):
    receipt = tmp_path / "data" / "tests" / "post-acceptance-full.xml"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        "<testsuites><testsuite tests='2' failures='0' errors='0' skipped='0'>"
        "<testcase name='test_omitted_negation_rejected'/>"
        "<testcase name='test_negated_direction_rejected_even_when_quote_exists'/>"
        "</testsuite></testsuites>",
        encoding="utf-8",
    )

    report = run_closeout_audit(tmp_path / "data", tmp_path / "source")

    assert _status(report, "G06", "negation_direction_context") == "PASS"


def test_refinement_metrics_require_structured_comparable_methods(tmp_path):
    root = tmp_path / "data" / "evaluation" / "refinement"
    _write(root / "round3.json", _round())
    _write(root / "round4.json", _round())

    report = run_closeout_audit(tmp_path / "data", tmp_path / "source")

    assert _status(report, "G10", "baseline_hybrid_compatibility_comparison") == "PASS"
    assert _status(report, "G10", "label_provenance") == "PASS"
    assert _status(report, "G14", "integrated_refinement_round") == "NOT_RUN"
    assert report["refinement"] == {
        "structured_rounds": 2,
        "comparable_rounds": 2,
        "two_latest_metric_identical": True,
    }


def _final_run(data, name, source_disjointness, *, final_integrated=False):
    run = data / "runs" / name
    inference = run / "inferences.jsonl"
    inference.parent.mkdir(parents=True)
    inference.write_text(
        json.dumps(
            {
                "origin": "fresh_local_inference",
                "model_revision": "revision",
                "runtime": "transformers",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payload = inference.read_bytes()
    evaluation = {
        "split_role": "VALIDATION" if final_integrated else "FINAL_HOLDOUT",
        "holdout_exposed_to_tuning": False,
    }
    if source_disjointness is not None:
        evaluation["source_disjointness"] = source_disjointness
    _write(
        run / "RUN_MANIFEST.json",
        {
            "run_id": name,
            "execution_status": "PASS",
            "failures": [],
            "evaluation": evaluation,
            "commands": [{"exit_code": 0, "log_reference": "inferences.jsonl"}],
            "artifacts": [
                {
                    "path": "inferences.jsonl",
                    "validation": "PASS",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
            ],
        },
    )


def _reservation(source):
    _write(
        source / "benchmarks" / "v2" / "final_holdout_reservation_v3.json",
        {
            "status": "RESERVED_PENDING_ONE_TIME_FINAL_HOLDOUT_AUTHORIZATION",
            "selected_accession": "GSE282859",
            "verification": {
                "all_exact_identifier_intersections_zero": True,
                "candidate_publication_pmc_complete": True,
                "candidate_ena_sra_complete": True,
                "base_official_relation_errors": 0,
                "base_official_relation_truncations": 0,
            },
            "sealed_evaluation_state": {
                "prediction_status": "NOT_RUN",
                "ranking_status": "NOT_RUN",
            },
        },
    )


def _round():
    return {
        "protocol_id": "EP-1",
        "catalog_sha256": "a" * 64,
        "label_origin": "source_determined",
        "primary": {
            "metrics": {
                name: {"queries": 10, "invalid_top_match": 0}
                for name in ("lexical", "minilm_hybrid", "compatibility_aware")
            },
            "capability_audit": {"denominator": 40},
        },
        "gate_assessment": {"overall": "NOT_PRODUCT_APPROVAL"},
    }
