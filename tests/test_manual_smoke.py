import json

from litdatamatcher.cli import main
from litdatamatcher.manual_smoke import filter_manual_smoke_input_files, run_manual_smoke_test


def test_manual_smoke_prepare_only_creates_dirs_and_notes(tmp_path):
    input_dir = tmp_path / "inputs"
    out_dir = tmp_path / "smoke"

    summary = run_manual_smoke_test(input_dir, out_dir, prepare_only=True)

    assert summary["status"] == "prepared"
    assert input_dir.exists()
    assert (out_dir / "corpus").exists()
    assert (out_dir / "pipeline").exists()
    assert (out_dir / "manual_review_notes.md").exists()
    assert (out_dir / "smoke_test_summary.json").exists()
    assert "Place 3-5 supported files" in summary["next_action"]


def test_manual_smoke_runs_ingest_and_pipeline_for_small_real_files(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "paper_one.txt").write_text(
        (
            "Antibiotic Microbiome Recovery\n\n"
            "Future studies should examine whether antibiotic exposure and "
            "longitudinal microbiome data explain clinical recovery."
        ),
        encoding="utf-8",
    )
    (input_dir / "paper_two.md").write_text(
        (
            "# Treatment Response Study\n\n"
            "Further research should examine whether transcriptomics predicts "
            "treatment response and remission."
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "smoke"

    summary = run_manual_smoke_test(input_dir, out_dir, top_n=10)

    assert summary["status"] == "completed"
    assert summary["input_file_count"] == 2
    assert summary["records_ingested"] == 2
    assert summary["questions"] >= 2
    assert summary["matches"] >= 1
    assert (out_dir / "corpus" / "literature.ingestion_report.md").exists()
    assert (out_dir / "pipeline" / "review_sheet.csv").exists()
    assert (out_dir / "smoke_test_summary.md").exists()


def test_manual_smoke_excludes_local_helper_files(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    paper = input_dir / "article_notes.md"
    readme = input_dir / "README.md"
    inventory = input_dir / "input_inventory.jsonl"
    instructions = input_dir / "smoke-test-instructions.txt"
    for path in (paper, readme, inventory, instructions):
        path.write_text("Future studies should examine microbiome recovery.", encoding="utf-8")

    kept, excluded = filter_manual_smoke_input_files(
        [paper, readme, inventory, instructions]
    )

    assert kept == [paper]
    assert sorted(path.name for path in excluded) == [
        "README.md",
        "input_inventory.jsonl",
        "smoke-test-instructions.txt",
    ]


def test_manual_smoke_cli_prepare_only(tmp_path, capsys):
    input_dir = tmp_path / "inputs"
    out_dir = tmp_path / "smoke"

    result = main(
        [
            "manual-smoke",
            "--input-dir",
            str(input_dir),
            "--out",
            str(out_dir),
            "--prepare-only",
        ]
    )
    captured = json.loads(capsys.readouterr().out)

    assert result == 0
    assert captured["status"] == "prepared"
    assert (out_dir / "manual_review_notes.md").exists()
