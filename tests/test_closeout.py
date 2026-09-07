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


def test_partial_run_is_a_runtime_failure_not_a_pass_by_file_presence(tmp_path):
    run = tmp_path / "data" / "runs" / "partial" / "RUN_MANIFEST.json"
    _write(run, {"execution_status": "PARTIAL", "failures": [{"stage": "source_guard"}]})

    report = run_closeout_audit(tmp_path / "data", tmp_path / "source")

    assert _status(report, "G05", "no_prewritten_or_regex_substitution") == "FAIL"
    assert report["gates"]["G05"]["status"] == "FAIL"


def test_tampered_declared_artifact_fails_final_run_integrity(tmp_path):
    run_dir = tmp_path / "data" / "runs" / "candidate"
    artifact = run_dir / "inferences.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"actual":"content"}\n', encoding="utf-8")
    _write(
        run_dir / "RUN_MANIFEST.json",
        {
            "execution_status": "PASS",
            "failures": [],
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
