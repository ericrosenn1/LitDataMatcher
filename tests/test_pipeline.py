import json
import sqlite3

from litdatamatcher.pipeline import run_pipeline
from litdatamatcher.storage import read_jsonl


def test_pipeline_writes_reproducible_artifacts(tmp_path):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "title": "Microbiome recovery",
                "abstract": (
                    "Further research should examine whether antibiotic exposure and "
                    "longitudinal microbiome data explain recovery."
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = run_pipeline(input_path, tmp_path / "out", top_n=5)

    assert metrics["documents"] == 1
    assert metrics["questions"] >= 1
    assert metrics["matches"] >= 1
    assert (tmp_path / "out" / "questions.jsonl").exists()
    assert (tmp_path / "out" / "matches.jsonl").exists()
    assert (tmp_path / "out" / "review_sheet.csv").exists()
    assert (tmp_path / "out" / "review_sheet.jsonl").exists()
    assert (tmp_path / "out" / "publication_report.md").exists()
    assert (tmp_path / "out" / "source_provenance_summary.json").exists()
    assert (tmp_path / "out" / "module_boundary_map.json").exists()
    assert (tmp_path / "out" / "provenance_transfer_check.json").exists()
    assert (tmp_path / "out" / "litdatamatcher.sqlite").exists()

    review_records = read_jsonl(tmp_path / "out" / "review_sheet.jsonl")
    assert review_records[0]["match"]["assessments"]["feasibility"]["recommended_design"]
    assert "score_components" in review_records[0]
    assert "match_relevance" in review_records[0]

    report = (tmp_path / "out" / "publication_report.md").read_text(encoding="utf-8")
    assert "Recommended Design" in report

    provenance_summary = json.loads(
        (tmp_path / "out" / "source_provenance_summary.json").read_text(encoding="utf-8")
    )
    transfer_check = json.loads(
        (tmp_path / "out" / "provenance_transfer_check.json").read_text(encoding="utf-8")
    )
    assert provenance_summary["source_types"]["curated_biomedical_catalog"] >= 1
    assert transfer_check["stages"]["dataset_records"]["with_provenance"] >= 1


def test_pipeline_rerun_resets_sqlite_run_tables(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    first_rows = [
        {
            "title": "Microbiome recovery",
            "abstract": "Further research should examine microbiome recovery.",
        },
        {
            "title": "Metabolomics response",
            "abstract": "Further research should examine metabolomics response.",
        },
    ]
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in first_rows),
        encoding="utf-8",
    )
    run_pipeline(input_path, output_dir, top_n=5)

    input_path.write_text(
        json.dumps(
            {
                "title": "Microbiome recovery",
                "abstract": "Further research should examine microbiome recovery.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = run_pipeline(input_path, output_dir, top_n=5)

    conn = sqlite3.connect(output_dir / "litdatamatcher.sqlite")
    try:
        sqlite_counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("questions", "datasets", "syntheses", "matches")
        }
    finally:
        conn.close()

    assert sqlite_counts["questions"] == metrics["questions"]
    assert sqlite_counts["syntheses"] == metrics["syntheses"]
    assert sqlite_counts["datasets"] == metrics["datasets"]
    assert sqlite_counts["matches"] == metrics["matches"]
