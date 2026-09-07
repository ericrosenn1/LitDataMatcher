import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "benchmarks" / "v2" / "evaluate_final_holdout.py"
SPEC = importlib.util.spec_from_file_location("evaluate_final_holdout", SCRIPT)
EVALUATOR = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(EVALUATOR)


def _write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _reservation(audit_sha):
    return {
        "protocol_id": "EP-test",
        "reservation_version": "v3_complete_official_relation_status",
        "status": "RESERVED_PENDING_ONE_TIME_FINAL_HOLDOUT_AUTHORIZATION",
        "selected_accession": "GSE999999",
        "audit": {
            "sha256": audit_sha,
            "decision": "RESERVED_PENDING_ONE_TIME_FINAL_HOLDOUT_AUTHORIZATION",
        },
        "verification": {
            "all_exact_identifier_intersections_zero": True,
            "candidate_publication_pmc_complete": True,
            "candidate_ena_sra_complete": True,
            "base_official_relation_errors": 0,
            "base_official_relation_truncations": 0,
        },
        "sealed_evaluation_state": {
            "title_summary_outcome_status": "UNINSPECTED",
            "label_status": "UNINSPECTED",
            "prediction_status": "NOT_RUN",
            "ranking_status": "NOT_RUN",
        },
    }


def _audit():
    return {
        "decision": "RESERVED_PENDING_ONE_TIME_FINAL_HOLDOUT_AUTHORIZATION",
        "candidate": {"accession": "GSE999999"},
        "identifier_overlap": {
            "series": [],
            "bioproject": [],
            "geo_samples": [],
            "pubmed": [],
            "pmc": [],
            "ena_sra": {"study_accession": [], "secondary_study_accession": [], "sample_accession": [], "run_accession": []},
        },
        "candidate_relation_status": {
            "publication_complete": True,
            "pmc_complete": True,
            "ena_sra_complete": True,
            "raw_files": {"PRJNA-test": {"rows": 3, "complete_under_limit": True}},
        },
        "official_relation_status": {
            "gds_to_pubmed": {"queried_series": 1, "indexed_links": 1, "explicit_no_indexed_links": [], "errors": [], "truncated": []},
            "pubmed_to_pmc": {"queried_pubmed": 1, "indexed_links": 1, "explicit_no_indexed_links": [], "errors": [], "truncated": []},
            "ena_sra": {"queried_bioproject_or_sra": 1, "indexed_links": 1, "explicit_no_indexed_links": [], "errors": [], "truncated": [], "maximum_rows_returned": 3, "raw_files": {"PRJNA-test": {"rows": 3, "complete_under_limit": True}}},
        },
    }


def _sealed_paths(tmp_path):
    audit_path = tmp_path / "audit.json"
    _write(audit_path, _audit())
    reservation_path = tmp_path / "reservation.json"
    _write(reservation_path, _reservation(hashlib.sha256(audit_path.read_bytes()).hexdigest().upper()))
    return reservation_path, audit_path


def test_sealed_reservation_requires_hash_complete_relations_and_not_run_states(tmp_path):
    reservation_path, audit_path = _sealed_paths(tmp_path)

    reservation, audit = EVALUATOR.validate_preconditions(reservation_path, audit_path)
    assert reservation["selected_accession"] == audit["candidate"]["accession"]

    changed = json.loads(reservation_path.read_text(encoding="utf-8"))
    changed["sealed_evaluation_state"]["ranking_status"] = "RAN"
    _write(reservation_path, changed)
    with pytest.raises(ValueError, match="ranking_status"):
        EVALUATOR.validate_preconditions(reservation_path, audit_path)


def test_tampered_audit_and_overlap_each_refuse_before_snapshot_access(tmp_path):
    reservation_path, audit_path = _sealed_paths(tmp_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["identifier_overlap"]["series"] = ["GSE1"]
    _write(audit_path, audit)
    with pytest.raises(ValueError, match="audit hash"):
        EVALUATOR.validate_preconditions(reservation_path, audit_path)

    _write(reservation_path, _reservation(hashlib.sha256(audit_path.read_bytes()).hexdigest().upper()))
    with pytest.raises(ValueError, match="overlap"):
        EVALUATOR.validate_preconditions(reservation_path, audit_path)


def test_consumption_ledger_is_exclusive_and_cannot_be_reused(tmp_path):
    reservation_path, audit_path = _sealed_paths(tmp_path)
    reservation, _ = EVALUATOR.validate_preconditions(reservation_path, audit_path)
    ledger = tmp_path / "final-holdout-consumed.json"

    record = EVALUATOR.reserve_consumption(ledger, reservation, audit_path)
    assert record["status"] == "RUNNING"
    with pytest.raises(RuntimeError, match="already consumed"):
        EVALUATOR.reserve_consumption(ledger, reservation, audit_path)


def test_selected_snapshot_constructs_source_determined_case_only(tmp_path):
    snapshot = tmp_path / "selected-source.json"
    _write(
        snapshot,
        {
            "result": {
                "uids": ["200999999"],
                "200999999": {
                    "accession": "GSE999999",
                    "title": "Synthetic source-described study",
                    "summary": "Synthetic acquisition metadata",
                    "taxon": "Homo sapiens",
                    "gdsType": "Expression profiling by high throughput sequencing",
                    "n_samples": 3,
                },
            }
        },
    )
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()

    record = EVALUATOR.source_record(snapshot, "GSE999999", digest)
    profile, retrieval_text = EVALUATOR.profile_from_source(
        record, "GSE999999", digest, "https://example.test/GSE999999"
    )

    assert profile["capabilities"]["comparator"]["status"] == "unknown"
    assert profile["independent_units"] is None
    assert "GSE999999" not in retrieval_text


def test_final_holdout_manifest_requires_proven_disjointness_with_zero_unknowns(tmp_path):
    schema = Path(__file__).parents[1] / "litdatamatcher" / "schemas_v2" / "RUN_MANIFEST.schema.json"
    manifest = {
        "schema_version": "2.0",
        "run_id": "final-test",
        "execution_status": "PASS",
        "started_at": "2026-09-07T00:00:00+00:00",
        "finished_at": "2026-09-07T00:01:00+00:00",
        "source": {"repository": "ericrosenn1/LitDataMatcher", "commit": "a", "working_tree_digest": "b", "spec_digest": "c", "config_digest": "d"},
        "environment": {"python": "test", "platform": "test", "dependency_lock_digest": "test", "hardware_record": "test"},
        "models": [{"id": "model", "revision": "rev", "runtime": "test", "license_status": "test", "prompt_version": "test"}],
        "source_snapshots": [{"source": "test", "snapshot_id": "test", "retrieved_at": "2026-09-07T00:00:00+00:00", "manifest_digest": "test"}],
        "commands": [{"command": "test", "cwd": "test", "started_at": "2026-09-07T00:00:00+00:00", "exit_code": 0, "log_reference": "metrics.json"}],
        "evaluation": {"protocol_version": "EP-test", "split_id": "GSE999999", "split_role": "FINAL_HOLDOUT", "label_origins": ["source_determined"], "holdout_exposed_to_tuning": False, "source_disjointness": {"status": "PROVEN_SOURCE_DISJOINT", "selected_accession": "GSE999999", "unknown_overlap_count": 0}},
        "coverage": {"unique_literature_records": 0, "parsed_full_texts": 0, "unique_accession_studies": 1, "sample_profiled_studies": 0, "inspected_processed_studies": 0, "external_structured_resources": 0, "distinct_pilot_contexts": 1, "case_dossiers": 1},
        "artifacts": [{"path": "metrics.json", "sha256": "0" * 64, "size_bytes": 1, "kind": "test", "validation": "PASS"}],
        "network": {"mode": "OFFLINE", "offline_block_test": True, "external_requests_observed": 0},
        "inference": {"fresh_calls": 1, "cache_replays": 0, "backend_qualified": True},
        "failures": [],
    }
    EVALUATOR.validate_manifest(manifest, schema)
    manifest["evaluation"]["source_disjointness"]["unknown_overlap_count"] = 1
    with pytest.raises(ValueError, match="schema validation"):
        EVALUATOR.validate_manifest(manifest, schema)
