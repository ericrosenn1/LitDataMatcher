import json

from litdatamatcher.annotations import export_annotation_corpus
from litdatamatcher.review import match_review_records, match_review_rows
from litdatamatcher.schemas import (
    DatasetRecord,
    DatasetVariable,
    Evidence,
    MatchCandidate,
    MatchScore,
    QuestionCandidate,
)
from litdatamatcher.storage import read_jsonl


def _example_match() -> MatchCandidate:
    question_provenance = {
        "source_type": "local_text",
        "content_scope": "full_text",
        "acquisition_method": "local_file",
        "status": "ok",
    }
    dataset_provenance = {
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
    question = QuestionCandidate(
        question_id="question-1",
        question="Does antibiotic exposure alter microbiome recovery?",
        source_ids=["paper-a"],
        evidence=[
            Evidence(
                text="Future studies should examine microbiome recovery.",
                source_id="paper-a",
                title="Paper A",
                doi="10.0000/paper-a",
                section="discussion",
                sentence_index=3,
            )
        ],
        required_variables=["antibiotic_exposure", "microbiome_composition"],
        metadata={"source_provenance": question_provenance},
    )
    dataset = DatasetRecord(
        dataset_id="dataset-1",
        title="Dataset One",
        source="curated",
        variables=[
            DatasetVariable(name="microbiome_composition"),
            DatasetVariable(name="predictor", category="derived_or_proxy_capability"),
        ],
        metadata={
            "source_provenance": dataset_provenance,
            "capability_caveats": ["Proxy feature support requires source inspection."],
            "capability_annotations": [
                {
                    "capability": "predictor",
                    "capability_type": "derived_or_proxy_capability",
                    "support": "proxy",
                }
            ],
        },
    )
    score = MatchScore(
        variable_overlap=0.8,
        semantic_relevance=0.7,
        population_fit=0.6,
        data_quality=0.7,
        sample_adequacy=0.8,
        significance=0.6,
        feasibility=0.7,
        uncertainty_penalty=0.2,
        governance=0.7,
        design_fit=0.7,
        combined=0.75,
    )
    return MatchCandidate(
        match_id="match-1",
        question=question,
        dataset=dataset,
        score=score,
        rationale=["source metadata test"],
        assessments={
            "feasibility": {"recommended_design": "longitudinal model"},
            "capability_support": {
                "direct_capabilities": ["microbiome_composition"],
                "proxy_capabilities": ["predictor"],
                "missing_capabilities": ["antibiotic_exposure"],
                "answerability_class": "proxy",
                "dataset_capability_categories": ["derived_or_proxy_capability"],
                "capability_caveats": ["Proxy feature support requires source inspection."],
            },
        },
    )


def test_review_exports_include_source_and_evidence_metadata():
    match = _example_match()

    csv_rows = match_review_rows([match])
    jsonl_records = match_review_records([match])

    assert csv_rows[0]["primary_source_id"] == "paper-a"
    assert csv_rows[0]["source_ids"] == "paper-a"
    assert csv_rows[0]["evidence_dois"] == "10.0000/paper-a"
    assert csv_rows[0]["evidence_titles"] == "Paper A"
    assert csv_rows[0]["evidence_sections"] == "discussion"
    assert csv_rows[0]["evidence_sentence_indices"] == "3"
    assert csv_rows[0]["question_source_types"] == "local_text"
    assert csv_rows[0]["dataset_source_types"] == "curated_biomedical_catalog"
    assert csv_rows[0]["dataset_source_content_scopes"] == "dataset_metadata"
    assert "Default offline catalog metadata" in csv_rows[0]["dataset_source_caveats"]
    assert csv_rows[0]["proxy_capability_support"] == "predictor"
    assert csv_rows[0]["missing_capability_support"] == "antibiotic_exposure"
    assert csv_rows[0]["match_answerability_class"] == "proxy"
    assert "Proxy feature support" in csv_rows[0]["dataset_capability_caveats"]
    assert jsonl_records[0]["source_ids"] == ["paper-a"]
    assert jsonl_records[0]["evidence_dois"] == ["10.0000/paper-a"]
    assert jsonl_records[0]["question_source_provenance"][0]["source_type"] == "local_text"
    assert (
        jsonl_records[0]["dataset_source_provenance"][0]["source_type"]
        == "curated_biomedical_catalog"
    )
    assert jsonl_records[0]["proxy_capability_support"] == ["predictor"]
    assert jsonl_records[0]["match_answerability_class"] == "proxy"


def test_annotation_export_preserves_review_source_metadata(tmp_path):
    labels_path = tmp_path / "review.jsonl"
    record = match_review_records([_example_match()])[0]
    record["match_relevance"] = "1"
    record["expert_question_quality"] = "4"
    record["annotator_id"] = "reviewer-a"
    labels_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    manifest = export_annotation_corpus(
        [labels_path],
        tmp_path / "labels_out",
        split_strategy="by_source_id",
        split_fractions=(1, 1, 1),
    )

    match_labels = read_jsonl(tmp_path / "labels_out" / "question_data_match_labels.jsonl")
    split_rows = [
        row
        for split_name in ("train", "validation", "test")
        for row in read_jsonl(tmp_path / "labels_out" / "splits" / f"{split_name}.jsonl")
    ]

    assert match_labels[0]["metadata"]["primary_source_id"] == "paper-a"
    assert match_labels[0]["metadata"]["source_ids"] == ["paper-a"]
    assert match_labels[0]["metadata"]["evidence_dois"] == ["10.0000/paper-a"]
    assert (
        match_labels[0]["metadata"]["dataset_source_provenance"][0]["source_type"]
        == "curated_biomedical_catalog"
    )
    assert manifest["splits"]["split_grouping_field_counts"]
    assert {row["metadata"]["split_group"] for row in split_rows} == {"source:paper-a"}


def test_source_split_does_not_use_review_file_path_as_group(tmp_path):
    labels_path = tmp_path / "review_without_source_ids.jsonl"
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

    manifest = export_annotation_corpus(
        [labels_path],
        tmp_path / "labels_out",
        split_strategy="by_source_id",
    )
    split_rows = read_jsonl(tmp_path / "labels_out" / "splits" / "test.jsonl")
    split_rows.extend(read_jsonl(tmp_path / "labels_out" / "splits" / "train.jsonl"))
    split_rows.extend(read_jsonl(tmp_path / "labels_out" / "splits" / "validation.jsonl"))

    assert any("fell back to question_id" in item for item in manifest["splits"]["warnings"])
    assert {row["metadata"]["split_group"] for row in split_rows} == {"question:question-1"}
    assert str(labels_path) not in next(iter(row["metadata"]["split_group"] for row in split_rows))
