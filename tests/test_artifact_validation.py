import csv
import json
import sqlite3

from litdatamatcher.artifact_validation import validate_run_artifacts
from litdatamatcher.cli import main
from litdatamatcher.storage import write_jsonl


def test_validate_run_artifacts_reports_review_ready_run(tmp_path):
    run_dir = _write_minimal_run(tmp_path / "run")
    out_dir = tmp_path / "validation"

    summary = validate_run_artifacts(run_dir, out_dir=out_dir)

    assert summary["status"] == "pass"
    assert summary["artifact_counts"]["required_missing"] == 0
    assert summary["provenance_audit"]["curated_catalog_provenance_seen"] is True
    assert summary["advisory_artifact_audit"]["appears_advisory"] is True
    assert summary["review_readiness"]["review_csv_rows"] == 1
    assert (out_dir / "artifact_validation_report.md").exists()
    assert (out_dir / "artifact_validation_summary.json").exists()


def test_validate_run_artifacts_collects_malformed_jsonl_without_crashing(tmp_path):
    run_dir = _write_minimal_run(tmp_path / "run")
    (run_dir / "questions.jsonl").write_text('{"question_id": "ok"}\n{bad json\n', encoding="utf-8")

    summary = validate_run_artifacts(run_dir)

    assert summary["status"] == "fail"
    assert any(issue["category"] == "malformed_jsonl" for issue in summary["issues"])


def test_validate_artifacts_cli_writes_outputs(tmp_path, capsys):
    run_dir = _write_minimal_run(tmp_path / "run")
    out_dir = tmp_path / "cli_validation"

    result = main(["validate-artifacts", "--run-dir", str(run_dir), "--out", str(out_dir)])
    captured = json.loads(capsys.readouterr().out)

    assert result == 0
    assert captured["status"] == "pass"
    assert captured["required_missing"] == 0
    assert (out_dir / "artifact_validation_report.md").exists()


def test_validate_artifacts_treats_empty_provenance_json_as_empty(tmp_path):
    run_dir = _write_minimal_run(tmp_path / "run")
    with (run_dir / "review_sheet.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "match_id",
                "source_caveats",
                "source_provenance_json",
                "dataset_source_caveats",
                "dataset_source_provenance_json",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "match_id": "match-1",
                "source_caveats": "",
                "source_provenance_json": "[]",
                "dataset_source_caveats": "",
                "dataset_source_provenance_json": "{}",
            }
        )

    summary = validate_run_artifacts(run_dir)
    review_csv = summary["provenance_audit"]["surfaces"]["review_csv"]

    assert review_csv["with_provenance"] == 0
    assert review_csv["empty_provenance"] == 1
    assert any(
        issue["category"] == "review_dataset_provenance_hidden"
        for issue in summary["issues"]
    )


def test_validate_artifacts_reports_malformed_provenance_json(tmp_path):
    run_dir = _write_minimal_run(tmp_path / "run")
    with (run_dir / "review_sheet.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["match_id", "dataset_source_provenance_json"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "match_id": "match-1",
                "dataset_source_provenance_json": "{not valid json",
            }
        )

    summary = validate_run_artifacts(run_dir)
    review_csv = summary["provenance_audit"]["surfaces"]["review_csv"]

    assert review_csv["malformed_provenance"] == 1
    assert any(issue["category"] == "malformed_provenance" for issue in summary["issues"])


def test_validate_artifacts_warns_on_sqlite_jsonl_count_mismatch(tmp_path):
    run_dir = _write_minimal_run(tmp_path / "run")
    conn = sqlite3.connect(run_dir / "litdatamatcher.sqlite")
    try:
        conn.executescript(
            """
            CREATE TABLE questions (question_id TEXT PRIMARY KEY, question TEXT, payload TEXT);
            CREATE TABLE datasets (dataset_id TEXT PRIMARY KEY, title TEXT, source TEXT, payload TEXT);
            CREATE TABLE syntheses (cluster_id TEXT PRIMARY KEY, payload TEXT);
            CREATE TABLE matches (match_id TEXT PRIMARY KEY, question_id TEXT, dataset_id TEXT, combined_score REAL, payload TEXT);
            INSERT INTO questions VALUES ('question-1', 'one', '{}');
            INSERT INTO questions VALUES ('question-stale', 'stale', '{}');
            INSERT INTO datasets VALUES ('dataset-1', 'one', 'fixture', '{}');
            INSERT INTO syntheses VALUES ('synthesis-1', '{}');
            INSERT INTO matches VALUES ('match-1', 'question-1', 'dataset-1', 0.5, '{}');
            """
        )
        conn.commit()
    finally:
        conn.close()

    summary = validate_run_artifacts(run_dir)

    assert summary["status"] == "needs_review"
    assert summary["sqlite_consistency"]["tables"]["questions"]["sqlite_records"] == 2
    assert summary["sqlite_consistency"]["tables"]["questions"]["artifact_records"] == 1
    assert any(
        issue["category"] == "sqlite_artifact_count_mismatch"
        for issue in summary["issues"]
    )


def _write_minimal_run(run_dir):
    run_dir.mkdir()
    provenance = {
        "source_type": "curated_biomedical_catalog",
        "content_scope": "dataset_metadata",
        "acquisition_method": "bundled_curated_catalog",
        "status": "warning",
        "warnings": [
            "Offline curated catalog metadata should be checked against the source repository before publication use."
        ],
        "limitations": [
            "Catalog variables and counts are curated summaries, not downloaded or analyzed source datasets."
        ],
    }
    question = {
        "question_id": "question-1",
        "question": "Can microbiome recovery be validated?",
        "metadata": {"source_provenance": provenance},
    }
    dataset = {
        "dataset_id": "dataset-1",
        "title": "Curated microbiome metadata",
        "source": "Curated",
        "metadata": {"source_provenance": provenance},
    }
    match = {
        "match_id": "match-1",
        "question": question,
        "dataset": dataset,
        "score": {"combined": 0.8},
        "assessments": {},
    }
    review = {
        "match_id": "match-1",
        "source_provenance": [provenance],
        "source_caveats": ["Default offline catalog metadata should be verified against source repositories before publication use."],
        "dataset_source_provenance": [provenance],
        "dataset_source_caveats": ["Default offline catalog metadata should be verified against source repositories before publication use."],
        "match": match,
    }
    write_jsonl(run_dir / "questions.jsonl", [question])
    write_jsonl(run_dir / "datasets.jsonl", [dataset])
    write_jsonl(run_dir / "matches.jsonl", [match])
    write_jsonl(run_dir / "review_sheet.jsonl", [review])
    write_jsonl(run_dir / "syntheses.jsonl", [])
    write_jsonl(run_dir / "metrics.jsonl", [{"questions": 1, "datasets": 1, "matches": 1}])
    (run_dir / "source_provenance_summary.json").write_text(
        json.dumps(
            {
                "records_with_provenance": 2,
                "records_without_provenance": 0,
                "source_types": {"curated_biomedical_catalog": 1},
                "content_scopes": {"dataset_metadata": 1},
                "acquisition_methods": {"bundled_curated_catalog": 1},
                "review_caveats": {"metadata-only records should not be treated as direct evidence": 1},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "module_boundary_map.json").write_text(
        json.dumps({"litdatamatcher.provenance": {"responsibility": "diagnostics"}}),
        encoding="utf-8",
    )
    (run_dir / "provenance_transfer_check.json").write_text(
        json.dumps({"status": "needs_review", "issues": [{"severity": "warning"}]}),
        encoding="utf-8",
    )
    (run_dir / "summary.md").write_text("# Summary\n\nReview-ready summary.\n", encoding="utf-8")
    (run_dir / "publication_report.md").write_text(
        (
            "# Report\n\n"
            "This is not validated, not analysis-ready, not downloaded, and not computed. "
            "Curated metadata-only records and derived capability claims require review."
        ),
        encoding="utf-8",
    )
    with (run_dir / "review_sheet.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "match_id",
                "source_caveats",
                "source_provenance_json",
                "dataset_source_caveats",
                "dataset_source_provenance_json",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "match_id": "match-1",
                "source_caveats": "Default offline catalog metadata should be verified",
                "source_provenance_json": json.dumps([provenance]),
                "dataset_source_caveats": "Default offline catalog metadata should be verified",
                "dataset_source_provenance_json": json.dumps([provenance]),
            }
        )
    return run_dir
