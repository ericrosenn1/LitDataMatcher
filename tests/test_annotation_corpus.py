import csv
import json

from litdatamatcher.annotation_agreement import (
    AGREEMENT_QA_LIMITATIONS,
    AGREEMENT_SCHEMA_VERSION,
)
from litdatamatcher.annotation_manifest import ANNOTATION_CORPUS_SCHEMA_VERSION
from litdatamatcher.annotations import export_annotation_corpus, load_review_corpus_rows
from litdatamatcher.cli import main
from litdatamatcher.storage import read_jsonl


def test_annotation_corpus_export_writes_normalized_training_files(tmp_path):
    csv_path = tmp_path / "completed_review.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "match_id",
                "question_id",
                "dataset_id",
                "score",
                "match_relevance",
                "expert_question_quality",
                "expert_data_match_quality",
                "expert_notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "rank": "1",
                "match_id": "match-1",
                "question_id": "question-1",
                "dataset_id": "dataset-1",
                "score": "0.75",
                "match_relevance": "1",
                "expert_question_quality": "4",
                "expert_data_match_quality": "5",
                "expert_notes": "Clear match.",
            }
        )
        writer.writerow(
            {
                "rank": "2",
                "match_id": "match-blank",
                "question_id": "question-blank",
                "dataset_id": "dataset-blank",
                "score": "0.25",
            }
        )

    jsonl_path = tmp_path / "completed_review.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "match": {
                    "match_id": "match-2",
                    "question": {"question_id": "question-2"},
                    "dataset": {"dataset_id": "dataset-2"},
                },
                "expert_relevance": "0",
                "expert_question_quality": "2",
                "expert_data_match_quality": "1",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = export_annotation_corpus(
        [csv_path, jsonl_path],
        tmp_path / "training_labels",
        annotator_id="expert-a",
    )

    match_labels = read_jsonl(tmp_path / "training_labels" / "question_data_match_labels.jsonl")
    quality_scores = read_jsonl(tmp_path / "training_labels" / "question_quality_scores.jsonl")
    summary = json.loads(
        (tmp_path / "training_labels" / "annotation_corpus_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["summary"]["source_rows"] == 3
    assert manifest["schema_version"] == ANNOTATION_CORPUS_SCHEMA_VERSION
    assert manifest["validation"]["source_rows"] == 3
    assert manifest["source_files"][0]["sha256"]
    assert (tmp_path / "training_labels" / "warnings.jsonl").exists()
    assert (tmp_path / "training_labels" / "skipped_rows.jsonl").exists()
    assert (tmp_path / "training_labels" / "duplicates.jsonl").exists()
    assert (tmp_path / "training_labels" / "conflicts.jsonl").exists()
    assert (tmp_path / "training_labels" / "agreement_summary.json").exists()
    assert (tmp_path / "training_labels" / "adjudication_needed.jsonl").exists()
    report = (tmp_path / "training_labels" / "annotation_corpus_report.md").read_text(
        encoding="utf-8"
    )
    assert manifest["outputs"]["report"].endswith("annotation_corpus_report.md")
    assert "Annotation Corpus Report" in report
    assert "No validation issues were recorded." in report
    assert manifest["summary"]["valid_rows"] == 3
    assert manifest["summary"]["exported_label_rows"] == 4
    assert manifest["summary"]["relevance_distribution"]["relevant"] == 1
    assert manifest["summary"]["relevance_distribution"]["not_relevant"] == 1
    assert manifest["summary"]["question_quality_distribution"]["4"] == 1
    assert manifest["summary"]["labels_per_reviewer"]["expert-a"] == 4
    assert "Training Readiness" in report
    assert "ready for exploratory training" in report
    assert len(match_labels) == 2
    assert len(quality_scores) == 2
    assert match_labels[0]["metadata"]["source_review_file"] == str(csv_path)
    assert match_labels[0]["metadata"]["raw_match_relevance"] == "1"
    assert match_labels[0]["relevance_score"] == 1.0
    assert match_labels[1]["label"] == "not_relevant"
    assert summary["question_data_match_labels"] == 2
    assert summary["question_quality_scores"] == 2


def test_annotation_export_reports_reviewer_agreement(tmp_path):
    labels_path = tmp_path / "agreeing_reviewers.jsonl"
    rows = [
        {
            "match_id": "match-1",
            "question_id": "question-1",
            "dataset_id": "dataset-1",
            "match_relevance": "1",
            "annotator_id": "reviewer-a",
            "source_id": "source-1",
        },
        {
            "match_id": "match-1",
            "question_id": "question-1",
            "dataset_id": "dataset-1",
            "match_relevance": "1",
            "annotator_id": "reviewer-b",
            "source_id": "source-1",
        },
    ]
    labels_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest = export_annotation_corpus([labels_path], tmp_path / "labels_out")

    agreement = json.loads((tmp_path / "labels_out" / "agreement_summary.json").read_text())
    adjudication = read_jsonl(tmp_path / "labels_out" / "adjudication_needed.jsonl")
    report = (tmp_path / "labels_out" / "annotation_corpus_report.md").read_text(
        encoding="utf-8"
    )

    assert agreement["reviewer_count"] == 2
    assert agreement["schema_version"] == AGREEMENT_SCHEMA_VERSION
    assert agreement["metric_limitations"] == AGREEMENT_QA_LIMITATIONS
    assert agreement["multi_reviewed_target_count"] == 1
    assert agreement["reviewer_pairs"][0]["overlap_count"] == 1
    assert agreement["reviewer_pairs"][0]["agreement_count"] == 1
    assert agreement["reviewer_pairs"][0]["disagreement_count"] == 0
    assert agreement["adjudication_needed_count"] == 0
    assert adjudication == []
    assert manifest["reviewer_overlap_counts"] == {"reviewer-a|reviewer-b": 1}
    assert manifest["unresolved_adjudication_count"] == 0
    assert "Agreement And Adjudication QA" in report
    assert AGREEMENT_QA_LIMITATIONS in report


def test_annotation_export_reports_reviewer_disagreement_for_adjudication(tmp_path):
    labels_path = tmp_path / "disagreeing_reviewers.jsonl"
    rows = [
        {
            "match_id": "match-1",
            "question_id": "question-1",
            "dataset_id": "dataset-1",
            "match_relevance": "1",
            "annotator_id": "reviewer-a",
            "primary_source_id": "source-1",
            "expert_notes": "Looks answerable.",
        },
        {
            "match_id": "match-1",
            "question_id": "question-1",
            "dataset_id": "dataset-1",
            "match_relevance": "0",
            "annotator_id": "reviewer-b",
            "primary_source_id": "source-1",
            "expert_notes": "Dataset does not answer it.",
        },
    ]
    labels_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest = export_annotation_corpus([labels_path], tmp_path / "labels_out")

    agreement = json.loads((tmp_path / "labels_out" / "agreement_summary.json").read_text())
    adjudication = read_jsonl(tmp_path / "labels_out" / "adjudication_needed.jsonl")
    report = (tmp_path / "labels_out" / "annotation_corpus_report.md").read_text(
        encoding="utf-8"
    )

    assert agreement["reviewer_pairs"][0]["overlap_count"] == 1
    assert agreement["reviewer_pairs"][0]["disagreement_count"] == 1
    assert agreement["adjudication_needed_count"] == 1
    assert adjudication[0]["disagreement_type"] == "cross_reviewer_disagreement"
    assert adjudication[0]["primary_source_id"] == "source-1"
    assert set(adjudication[0]["labels_by_reviewer"]) == {"reviewer-a", "reviewer-b"}
    assert manifest["unresolved_adjudication_count"] == 1
    assert "Adjudication-needed records: 1" in report
    assert "not ready for training" in report


def test_annotation_export_single_reviewer_reports_no_overlap(tmp_path):
    labels_path = tmp_path / "single_reviewer.jsonl"
    labels_path.write_text(
        json.dumps(
            {
                "match_id": "match-1",
                "question_id": "question-1",
                "dataset_id": "dataset-1",
                "match_relevance": "1",
                "annotator_id": "reviewer-a",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    export_annotation_corpus([labels_path], tmp_path / "labels_out")

    agreement = json.loads((tmp_path / "labels_out" / "agreement_summary.json").read_text())
    adjudication = read_jsonl(tmp_path / "labels_out" / "adjudication_needed.jsonl")

    assert agreement["reviewer_count"] == 1
    assert agreement["reviewer_pair_count"] == 0
    assert agreement["total_pair_overlap_count"] == 0
    assert adjudication == []


def test_annotation_export_unlabeled_corpus_writes_empty_agreement(tmp_path):
    labels_path = tmp_path / "unlabeled.jsonl"
    labels_path.write_text(
        json.dumps(
            {
                "match_id": "match-1",
                "question_id": "question-1",
                "dataset_id": "dataset-1",
                "annotator_id": "reviewer-a",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = export_annotation_corpus([labels_path], tmp_path / "labels_out")

    agreement = json.loads((tmp_path / "labels_out" / "agreement_summary.json").read_text())
    adjudication = read_jsonl(tmp_path / "labels_out" / "adjudication_needed.jsonl")

    assert manifest["summary"]["exported_label_rows"] == 0
    assert agreement["target_count"] == 0
    assert agreement["adjudication_needed_count"] == 0
    assert adjudication == []


def test_load_review_corpus_rows_tracks_source_file(tmp_path):
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(
        json.dumps(
            {
                "match_id": "match-1",
                "question_id": "question-1",
                "dataset_id": "dataset-1",
                "match_relevance": "1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_review_corpus_rows([labels_path])

    assert rows[0]["_source_review_file"] == str(labels_path)
    assert rows[0]["_source_row_number"] == 1
    assert rows[0]["_source_format"] == "jsonl"


def test_annotation_export_cli_writes_manifest(tmp_path, capsys):
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(
        json.dumps(
            {
                "match_id": "match-1",
                "question_id": "question-1",
                "dataset_id": "dataset-1",
                "match_relevance": "1",
                "expert_question_quality": "4",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "labels_out"

    result = main(
        [
            "annotation-export",
            "--labels",
            str(labels_path),
            "--out",
            str(out_dir),
            "--annotator-id",
            "expert-a",
        ]
    )
    captured = json.loads(capsys.readouterr().out)

    assert result == 0
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "annotation_corpus_report.md").exists()
    assert captured["summary"]["question_data_match_labels"] == 1


def test_annotation_export_skips_malformed_and_missing_labeled_rows(tmp_path):
    labels_path = tmp_path / "bad_labels.jsonl"
    rows = [
        {
            "match_id": "match-good",
            "question_id": "question-good",
            "dataset_id": "dataset-good",
            "match_relevance": "1",
        },
        {
            "match_id": "match-bad-score",
            "question_id": "question-bad-score",
            "dataset_id": "dataset-bad-score",
            "match_relevance": "not-a-number",
        },
        {
            "match_id": "",
            "question_id": "question-missing",
            "dataset_id": "dataset-missing",
            "match_relevance": "1",
        },
        {
            "match_id": "match-out-of-range",
            "question_id": "question-out-of-range",
            "dataset_id": "dataset-out-of-range",
            "expert_question_quality": "7",
        },
    ]
    labels_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest = export_annotation_corpus(
        [labels_path],
        tmp_path / "labels_out",
        annotator_id="reviewer-a",
    )

    match_labels = read_jsonl(tmp_path / "labels_out" / "question_data_match_labels.jsonl")
    skipped = read_jsonl(tmp_path / "labels_out" / "skipped_rows.jsonl")
    report = (tmp_path / "labels_out" / "annotation_corpus_report.md").read_text(
        encoding="utf-8"
    )
    skip_codes = {row["code"] for row in skipped}

    assert len(match_labels) == 1
    assert manifest["summary"]["skipped_rows"] == 3
    assert "malformed_numeric_score" in skip_codes
    assert "missing_match_id" in skip_codes
    assert "score_out_of_range" in skip_codes
    assert "Skipped rows: 3" in report
    assert "malformed_numeric_score" in report
    assert "score_out_of_range" in report
    assert "not ready for training" in report


def test_annotation_export_detects_duplicates_and_conflicts(tmp_path):
    labels_path = tmp_path / "reviewers.jsonl"
    rows = [
        {
            "match_id": "match-1",
            "question_id": "question-1",
            "dataset_id": "dataset-1",
            "match_relevance": "1",
            "annotator_id": "reviewer-a",
        },
        {
            "match_id": "match-1",
            "question_id": "question-1",
            "dataset_id": "dataset-1",
            "match_relevance": "1",
            "annotator_id": "reviewer-a",
        },
        {
            "match_id": "match-1",
            "question_id": "question-1",
            "dataset_id": "dataset-1",
            "match_relevance": "0",
            "annotator_id": "reviewer-a",
        },
        {
            "match_id": "match-1",
            "question_id": "question-1",
            "dataset_id": "dataset-1",
            "match_relevance": "0",
            "annotator_id": "reviewer-b",
        },
    ]
    labels_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest = export_annotation_corpus([labels_path], tmp_path / "labels_out")

    match_labels = read_jsonl(tmp_path / "labels_out" / "question_data_match_labels.jsonl")
    duplicates = read_jsonl(tmp_path / "labels_out" / "duplicates.jsonl")
    conflicts = read_jsonl(tmp_path / "labels_out" / "conflicts.jsonl")
    skipped = read_jsonl(tmp_path / "labels_out" / "skipped_rows.jsonl")
    report = (tmp_path / "labels_out" / "annotation_corpus_report.md").read_text(
        encoding="utf-8"
    )

    assert len(match_labels) == 2
    assert duplicates[0]["code"] == "duplicate_row"
    assert {row["code"] for row in conflicts} == {
        "conflicting_label",
        "cross_reviewer_disagreement",
    }
    assert {row["code"] for row in skipped} == {"duplicate_row", "conflicting_label"}
    assert manifest["validation"]["duplicates"] == 1
    assert manifest["validation"]["conflicts"] == 2
    assert manifest["validation"]["reviewer_count"] == 2
    assert manifest["validation"]["valid_rows"] == 2
    assert manifest["summary"]["exported_label_rows"] == 2
    assert manifest["summary"]["labels_per_reviewer"] == {"reviewer-a": 1, "reviewer-b": 1}
    assert manifest["unresolved_adjudication_count"] == 2
    assert {label["annotator_id"] for label in match_labels} == {"reviewer-a", "reviewer-b"}
    adjudication = read_jsonl(tmp_path / "labels_out" / "adjudication_needed.jsonl")
    assert {row["disagreement_type"] for row in adjudication} == {
        "conflicting_label",
        "cross_reviewer_disagreement",
    }
    assert "Duplicate Rows" in report
    assert "cross_reviewer_disagreement" in report
    assert "reviewer-a, reviewer-b" in report
    assert "not ready for training" in report
