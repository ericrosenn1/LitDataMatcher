import json

from litdatamatcher.annotations import export_annotation_corpus
from litdatamatcher.cli import main
from litdatamatcher.storage import read_jsonl


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_split_rows(out_dir):
    return {
        name: read_jsonl(out_dir / "splits" / f"{name}.jsonl")
        for name in ("train", "validation", "test")
    }


def _assert_group_not_leaked(split_rows, key):
    seen_by_group = {}
    for split_name, rows in split_rows.items():
        for row in rows:
            group = row["metadata"][key]
            assert seen_by_group.setdefault(group, split_name) == split_name


def test_annotation_export_writes_grouped_question_splits(tmp_path):
    labels_path = tmp_path / "labels.jsonl"
    rows = [
        {
            "match_id": f"match-{question}-{dataset}",
            "question_id": question,
            "dataset_id": dataset,
            "match_relevance": "1",
            "expert_question_quality": "4",
            "annotator_id": "reviewer-a",
        }
        for question in ("question-1", "question-2", "question-3", "question-4")
        for dataset in ("dataset-a", "dataset-b")
    ]
    _write_jsonl(labels_path, rows)

    manifest = export_annotation_corpus(
        [labels_path],
        tmp_path / "labels_out",
        split_strategy="by_question_id",
        split_fractions=(0.5, 0.25, 0.25),
        split_seed=7,
    )

    split_rows = _read_split_rows(tmp_path / "labels_out")
    split_total = sum(len(rows) for rows in split_rows.values())
    report = (tmp_path / "labels_out" / "annotation_corpus_report.md").read_text(
        encoding="utf-8"
    )

    assert manifest["split_strategy"] == "by_question_id"
    assert manifest["split_seed"] == 7
    assert manifest["split_output_files"]["train"].endswith("splits\\train.jsonl") or manifest[
        "split_output_files"
    ]["train"].endswith("splits/train.jsonl")
    assert split_total == manifest["summary"]["exported_label_rows"]
    assert split_rows["train"]
    assert split_rows["test"]
    assert {row["label_type"] for rows in split_rows.values() for row in rows} == {
        "question_data_match",
        "question_quality",
    }
    assert all("_split_group" not in row for rows in split_rows.values() for row in rows)
    assert all(
        row["metadata"]["split_strategy"] == "by_question_id"
        for rows in split_rows.values()
        for row in rows
    )
    _assert_group_not_leaked(split_rows, "split_group")
    assert "Split Summary" in report
    assert "Strategy: by_question_id" in report


def test_source_split_preserves_document_groups_when_available(tmp_path):
    labels_path = tmp_path / "source_labels.jsonl"
    rows = [
        {
            "match": {
                "match_id": f"match-{question}",
                "question": {"question_id": question, "source_ids": [source]},
                "dataset": {"dataset_id": "dataset-1"},
            },
            "match_relevance": "1",
            "annotator_id": "reviewer-a",
        }
        for source, questions in {
            "paper-a": ("question-1", "question-2"),
            "paper-b": ("question-3", "question-4"),
            "paper-c": ("question-5", "question-6"),
        }.items()
        for question in questions
    ]
    _write_jsonl(labels_path, rows)

    manifest = export_annotation_corpus(
        [labels_path],
        tmp_path / "labels_out",
        split_strategy="by_source_id",
        split_fractions=(1, 1, 1),
        split_seed=13,
    )

    split_rows = _read_split_rows(tmp_path / "labels_out")
    source_to_split = {}
    for split_name, rows in split_rows.items():
        for row in rows:
            source = row["metadata"]["source_ids"][0]
            assert source_to_split.setdefault(source, split_name) == split_name

    assert manifest["split_strategy"] == "by_source_id"
    assert manifest["splits"]["split_grouping_field_counts"]
    assert not manifest["splits"]["warnings"]


def test_source_split_warns_and_falls_back_when_source_ids_are_missing(tmp_path):
    labels_path = tmp_path / "flat_labels.jsonl"
    rows = [
        {
            "match_id": f"match-{index}",
            "question_id": f"question-{index}",
            "dataset_id": "dataset-1",
            "match_relevance": "1",
            "annotator_id": "reviewer-a",
        }
        for index in range(4)
    ]
    _write_jsonl(labels_path, rows)

    manifest = export_annotation_corpus(
        [labels_path],
        tmp_path / "labels_out",
        split_strategy="by_source_id",
        split_fractions=(1, 1, 1),
        split_seed=5,
    )

    split_rows = _read_split_rows(tmp_path / "labels_out")
    grouping_fields = {
        row["metadata"]["split_grouping_field"]
        for rows in split_rows.values()
        for row in rows
    }

    assert grouping_fields == {"question_id"}
    assert any("fell back to question_id" in item for item in manifest["splits"]["warnings"])


def test_split_seed_is_deterministic(tmp_path):
    labels_path = tmp_path / "labels.jsonl"
    rows = [
        {
            "match_id": f"match-{index}",
            "question_id": f"question-{index}",
            "dataset_id": "dataset-1",
            "match_relevance": "1",
            "annotator_id": "reviewer-a",
        }
        for index in range(8)
    ]
    _write_jsonl(labels_path, rows)

    export_annotation_corpus(
        [labels_path],
        tmp_path / "first",
        split_strategy="by_question_id",
        split_seed=31,
    )
    export_annotation_corpus(
        [labels_path],
        tmp_path / "second",
        split_strategy="by_question_id",
        split_seed=31,
    )

    assert _read_split_rows(tmp_path / "first") == _read_split_rows(tmp_path / "second")


def test_zero_label_split_reports_not_training_ready(tmp_path):
    labels_path = tmp_path / "unlabeled.jsonl"
    _write_jsonl(
        labels_path,
        [
            {
                "match_id": "match-1",
                "question_id": "question-1",
                "dataset_id": "dataset-1",
            }
        ],
    )

    manifest = export_annotation_corpus(
        [labels_path],
        tmp_path / "labels_out",
        annotator_id="reviewer-a",
        split_strategy="by_question_id",
    )

    split_rows = _read_split_rows(tmp_path / "labels_out")
    report = (tmp_path / "labels_out" / "annotation_corpus_report.md").read_text(
        encoding="utf-8"
    )

    assert sum(len(rows) for rows in split_rows.values()) == 0
    assert manifest["summary"]["exported_label_rows"] == 0
    assert manifest["training_readiness"]["status"] == "not ready for training"
    assert "no exported labels" in manifest["training_readiness"]["blocking_issues"]
    assert "no labels were available for splitting" in manifest["splits"]["warnings"]
    assert "not ready for training" in report
    assert "Complete review labels" in report


def test_annotation_export_cli_can_generate_splits(tmp_path, capsys):
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(
        labels_path,
        [
            {
                "match_id": "match-1",
                "question_id": "question-1",
                "dataset_id": "dataset-1",
                "match_relevance": "1",
                "expert_question_quality": "5",
            }
        ],
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
            "reviewer-a",
            "--split-strategy",
            "by_question_id",
        ]
    )
    captured = json.loads(capsys.readouterr().out)

    assert result == 0
    assert (out_dir / "splits" / "train.jsonl").exists()
    assert captured["split_strategy"] == "by_question_id"
    assert captured["summary"]["exported_label_rows"] == 2
