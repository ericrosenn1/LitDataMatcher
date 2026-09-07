import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from litdatamatcher.acceptance import ACCEPTANCE_SCHEMA_PATH, validate_acceptance
from litdatamatcher.v2 import main


def test_missing_ledger_is_schema_valid_and_honestly_not_run(tmp_path):
    report = validate_acceptance(tmp_path / "missing-ledger.json")

    assert report["product_status"] == "NOT_READY"
    assert report["validator"]["executed"] is True
    assert {gate["status"] for gate in report["gates"].values()} == {"NOT_RUN"}
    assert (
        list(
            Draft202012Validator(json.loads(ACCEPTANCE_SCHEMA_PATH.read_text())).iter_errors(report)
        )
        == []
    )


def test_file_only_evidence_cannot_promote_a_gate(tmp_path):
    _write_run(tmp_path, execution_status="PASS")
    _write_ledger(tmp_path, checks=[])

    report = validate_acceptance(tmp_path / "ACCEPTANCE_EVIDENCE.json")

    assert report["gates"]["G05"]["status"] == "NOT_RUN"


def test_validated_single_run_populates_report_run_id(tmp_path):
    _write_run(tmp_path, execution_status="PASS")
    _write_ledger(tmp_path, checks=[])

    report = validate_acceptance(tmp_path / "ACCEPTANCE_EVIDENCE.json")

    assert report["run_id"] == "run-1"
    assert report["product_status"] == "NOT_READY"


def test_forged_artifact_digest_fails_the_claimed_gate(tmp_path):
    _write_run(tmp_path, forged_proof_digest=True)
    _write_ledger(tmp_path, checks=_g05_checks())

    report = validate_acceptance(tmp_path / "ACCEPTANCE_EVIDENCE.json")

    assert report["gates"]["G05"]["status"] == "FAIL"
    assert report["gates"]["G05"]["evidence"] == []
    assert report["product_status"] == "NOT_READY"


def test_hashed_arbitrary_json_cannot_promote_a_gate(tmp_path):
    _write_run(tmp_path, semantic_proof=False)
    _write_ledger(tmp_path, checks=_g05_checks())

    report = validate_acceptance(tmp_path / "ACCEPTANCE_EVIDENCE.json")

    assert report["gates"]["G05"]["status"] == "FAIL"
    assert "structured audit" in report["gates"]["G05"]["reason"]


def test_stale_observation_cannot_be_reused_as_successful_evidence(tmp_path):
    _write_run(tmp_path)
    checks = _g05_checks()
    for check in checks:
        check["observed_at"] = "2026-09-07T10:59:59+00:00"
    _write_ledger(tmp_path, checks=checks)

    report = validate_acceptance(tmp_path / "ACCEPTANCE_EVIDENCE.json")

    assert report["gates"]["G05"]["status"] == "FAIL"
    assert "stale evidence" in report["gates"]["G05"]["reason"]


def test_hashed_executed_receipts_can_validate_a_complete_gate_and_cli_writes_report(
    tmp_path, capsys
):
    _write_run(tmp_path)
    _write_ledger(tmp_path, checks=_g05_checks())
    output = tmp_path / "ACCEPTANCE_REPORT.json"

    result = main(
        [
            "acceptance",
            "--evidence",
            str(tmp_path / "ACCEPTANCE_EVIDENCE.json"),
            "--out",
            str(output),
        ]
    )
    printed = json.loads(capsys.readouterr().out)

    assert result == 0
    assert printed["gates"]["G05"]["status"] == "PASS"
    assert printed["product_status"] == "NOT_READY"
    assert json.loads(output.read_text())["gates"]["G05"]["status"] == "PASS"


def _g05_checks():
    kinds = [
        "fresh_application_process",
        "previously_unprocessed_input",
        "runtime_model_revision",
        "no_prewritten_or_regex_substitution",
    ]
    return [
        {
            "id": f"g05-{kind}",
            "target": "G05",
            "kind": kind,
            "command_index": 0,
            "artifacts": ["proof.json"],
            "observed_at": "2026-09-07T12:00:00+00:00",
        }
        for kind in kinds
    ]


def _write_ledger(root: Path, *, checks):
    (root / "ACCEPTANCE_EVIDENCE.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "build_id": "test-build",
                "runs": [{"run_manifest": "run/RUN_MANIFEST.json", "checks": checks}],
                "open_issues": [],
                "optional_backlog": [],
                "stop_reason": None,
            }
        ),
        encoding="utf-8",
    )


def _write_run(
    root: Path,
    *,
    execution_status="PASS",
    forged_proof_digest=False,
    semantic_proof=True,
):
    run = root / "run"
    run.mkdir()
    (run / "commands.log").write_text("command completed with exit code 0\n", encoding="utf-8")
    proof = {"executed": True}
    if semantic_proof:
        proof = {
            "runner": {"implementation_version": "closeout-evidence-audit-test"},
            "evidence": [
                {
                    "target": "G05",
                    "kind": kind,
                    "status": "PASS",
                    "reason": "Executed test observation.",
                    "artifacts": [{"path": "source.json", "sha256": "a" * 64}],
                }
                for kind in (
                    "fresh_application_process",
                    "previously_unprocessed_input",
                    "runtime_model_revision",
                    "no_prewritten_or_regex_substitution",
                )
            ],
        }
    (run / "proof.json").write_text(json.dumps(proof) + "\n", encoding="utf-8")
    proof_digest = _hash(run / "proof.json")
    if forged_proof_digest:
        proof_digest = "0" * 64
    manifest = {
        "schema_version": "2.0",
        "run_id": "run-1",
        "execution_status": execution_status,
        "started_at": "2026-09-07T11:00:00+00:00",
        "finished_at": "2026-09-07T12:00:00+00:00",
        "source": {
            "repository": "ericrosenn1/LitDataMatcher",
            "commit": "a" * 40,
            "working_tree_digest": "b" * 64,
            "spec_digest": "c" * 64,
            "config_digest": "d" * 64,
        },
        "environment": {
            "python": "3.12",
            "platform": "test",
            "dependency_lock_digest": None,
            "hardware_record": None,
        },
        "models": [
            {
                "id": "local-model",
                "revision": "r1",
                "runtime": "test",
                "license_status": "ok",
                "prompt_version": "p1",
            }
        ],
        "source_snapshots": [],
        "commands": [
            {
                "command": "pytest test",
                "cwd": str(run),
                "started_at": "2026-09-07T11:00:00+00:00",
                "exit_code": 0,
                "log_reference": "commands.log",
            }
        ],
        "evaluation": {
            "protocol_version": "EP-20260907-1",
            "split_id": "development",
            "split_role": "DEVELOPMENT",
            "label_origins": [],
            "holdout_exposed_to_tuning": False,
        },
        "coverage": {
            "unique_literature_records": 0,
            "parsed_full_texts": 0,
            "unique_accession_studies": 0,
            "sample_profiled_studies": 0,
            "inspected_processed_studies": 0,
            "external_structured_resources": 0,
            "distinct_pilot_contexts": 0,
            "case_dossiers": 0,
        },
        "artifacts": [
            {
                "path": "commands.log",
                "sha256": _hash(run / "commands.log"),
                "size_bytes": (run / "commands.log").stat().st_size,
                "kind": "command_log",
                "validation": "PASS",
            },
            {
                "path": "proof.json",
                "sha256": proof_digest,
                "size_bytes": (run / "proof.json").stat().st_size,
                "kind": "test_result",
                "validation": "PASS",
            },
        ],
        "network": {"mode": "OFFLINE", "offline_block_test": True, "external_requests_observed": 0},
        "inference": {"fresh_calls": 1, "cache_replays": 0, "backend_qualified": True},
        "failures": [],
    }
    (run / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
