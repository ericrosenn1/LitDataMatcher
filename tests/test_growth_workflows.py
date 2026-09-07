import csv
import json

from litdatamatcher.cli import main
from litdatamatcher.calibration import calibrate_ranking_threshold
from litdatamatcher.pipeline import run_demo
from litdatamatcher.review_queue import write_daily_review_queue
from litdatamatcher.schemas import QuestionDataMatchLabel
from litdatamatcher.stress import synthetic_literature_records
from litdatamatcher.storage import read_jsonl, write_jsonl


def test_daily_review_queue_writes_top_matches(tmp_path):
    run_dir = tmp_path / "demo"
    run_demo(run_dir)
    out_path = tmp_path / "daily_queue.csv"

    metrics = write_daily_review_queue(
        run_dir, out_path, limit=3, reviewer_id="reviewer-a", review_date="2026-06-15"
    )
    with out_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert metrics["queued_matches"] == 3
    assert len(rows) == 3
    assert {row["review_status"] for row in rows} == {"pending"}
    assert {row["reviewer_id"] for row in rows} == {"reviewer-a"}


def test_synthetic_literature_records_are_deterministic():
    first = synthetic_literature_records(3)
    second = synthetic_literature_records(3)

    assert first == second
    assert len(first) == 3
    assert all(row["source_id"].startswith("source_") for row in first)


def test_stress_demo_cli_runs_small_corpus(tmp_path, capsys):
    out_dir = tmp_path / "stress"

    result = main(["stress-demo", "--out", str(out_dir), "--documents", "3", "--top-n", "5"])
    captured = json.loads(capsys.readouterr().out)

    assert result == 0
    assert captured["documents_requested"] == 3
    assert captured["documents"] == 3
    assert captured["questions"] >= 3
    assert (out_dir / "synthetic_literature.jsonl").exists()
    assert (out_dir / "run" / "matches.jsonl").exists()


def test_calibrate_ranking_threshold_reports_best_threshold(tmp_path):
    run_dir = tmp_path / "demo"
    run_demo(run_dir)
    matches = read_jsonl(run_dir / "matches.jsonl")[:3]
    labels = [
        QuestionDataMatchLabel(
            match_id=matches[0]["match_id"],
            question_id=matches[0]["question"]["question_id"],
            dataset_id=matches[0]["dataset"]["dataset_id"],
            annotator_id="reviewer-a",
            relevance_score=1,
        ).to_dict(),
        QuestionDataMatchLabel(
            match_id=matches[1]["match_id"],
            question_id=matches[1]["question"]["question_id"],
            dataset_id=matches[1]["dataset"]["dataset_id"],
            annotator_id="reviewer-a",
            relevance_score=0,
        ).to_dict(),
    ]
    labels_path = tmp_path / "labels.jsonl"
    write_jsonl(labels_path, labels)

    report = calibrate_ranking_threshold(run_dir / "matches.jsonl", labels_path)

    assert report["labeled_matches"] == 2
    assert report["positive_labels"] == 1
    assert "best_threshold_metrics" in report
    assert len(report["calibration_bins"]) == 10


def test_calibrate_ranking_cli_writes_report(tmp_path, capsys):
    run_dir = tmp_path / "demo"
    run_demo(run_dir)
    match = read_jsonl(run_dir / "matches.jsonl")[0]
    labels_path = tmp_path / "labels.jsonl"
    out_path = tmp_path / "calibration" / "ranking_calibration.json"
    write_jsonl(
        labels_path,
        [
            QuestionDataMatchLabel(
                match_id=match["match_id"],
                question_id=match["question"]["question_id"],
                dataset_id=match["dataset"]["dataset_id"],
                annotator_id="reviewer-a",
                relevance_score=1,
            ).to_dict()
        ],
    )

    result = main(
        [
            "calibrate-ranking",
            "--matches",
            str(run_dir / "matches.jsonl"),
            "--labels",
            str(labels_path),
            "--out",
            str(out_path),
        ]
    )
    captured = json.loads(capsys.readouterr().out)

    assert result == 0
    assert out_path.exists()
    assert captured["labeled_matches"] == 1
