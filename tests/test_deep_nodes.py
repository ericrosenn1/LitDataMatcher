import json

from litdatamatcher.datasets import discover_datasets_for_question
from litdatamatcher.evaluation import evaluate_question_extraction, evaluate_ranking
from litdatamatcher.feasibility import assess_pair_feasibility
from litdatamatcher.literature import analyze_literature_records
from litdatamatcher.meta_analysis import run_meta_analysis_node, synthesis_index
from litdatamatcher.ontology import (
    explain_variable_match,
    infer_concepts_from_text,
    normalize_variable_name,
    normalize_entity,
)
from litdatamatcher.ranking import rank_matches
from litdatamatcher.review import (
    export_review_csv,
    export_review_jsonl,
    match_review_records,
    match_review_rows,
    review_rows_to_match_labels,
    review_rows_to_question_quality_scores,
    review_rows_to_training_labels,
    summarize_review_labels,
)
from litdatamatcher.storage import read_jsonl


def _demo_matches():
    questions = analyze_literature_records(
        [
            {
                "title": "Antibiotics and microbiome recovery",
                "abstract": (
                    "Future studies should examine whether antibiotic exposure and "
                    "longitudinal microbiome data predict IBD remission."
                ),
            }
        ]
    )
    datasets = []
    for question in questions:
        datasets.extend(discover_datasets_for_question(question))
    syntheses = synthesis_index(run_meta_analysis_node(questions))
    return questions, rank_matches(questions, datasets, syntheses, top_n=5)


def test_ontology_normalizes_synonyms():
    assert normalize_variable_name("antimicrobial exposure") == "antibiotic_exposure"
    assert normalize_variable_name("RNA-seq") == "transcriptomics"


def test_entity_contract_preserves_ambiguity_deprecation_and_source_failure():
    assert normalize_entity("p53", "gene_protein")["candidates"] == ["HGNC:11998"]
    assert normalize_entity("cd3", "gene_protein")["status"] == "AMBIGUOUS"
    assert normalize_entity("gut", "tissue_cell_type")["status"] == "AMBIGUOUS"
    assert normalize_entity("HGNC:OLDTP53", "gene_protein")["status"] == "DEPRECATED"
    assert normalize_entity("TP53", "gene_protein", source_available=False)["status"] == "SOURCE_UNAVAILABLE"


def test_ontology_handles_noisy_variable_names_and_empty_requirements():
    assert normalize_variable_name("Baseline visit time point") == "timepoint"
    assert normalize_variable_name("patient antimicrobial exposure status") == (
        "antibiotic_exposure"
    )

    empty = explain_variable_match([], {"age"})

    assert empty["coverage"] == 0.45
    assert empty["required_normalized"] == []


def test_ontology_concept_inference_is_deduplicated_by_concept_scan():
    hits = infer_concepts_from_text("RNA-seq transcriptomic data with remission outcome")

    assert "transcriptomics" in hits
    assert "outcome" in hits
    assert len(hits) == len(set(hits))


def test_feasibility_assessment_has_interpretable_caveats():
    questions, matches = _demo_matches()
    assessment = assess_pair_feasibility(questions[0], matches[0].dataset)

    assert 0 <= assessment.overall <= 1
    assert assessment.recommended_design
    assert isinstance(assessment.missing_variables, list)


def test_ranking_embeds_governance_and_feasibility_payloads():
    _questions, matches = _demo_matches()

    assert matches[0].assessments["feasibility"]["recommended_design"]
    assert "reuse_score" in matches[0].assessments["governance"]
    assert matches[0].score.governance >= 0


def test_evaluation_metrics_smoke():
    questions, matches = _demo_matches()
    extraction = evaluate_question_extraction(questions, [{"question": questions[0].question}])
    ranking = evaluate_ranking(matches, [{"match_id": matches[0].match_id, "relevance": 1}], k=3)

    assert extraction.true_positives == 1
    assert ranking.mean_reciprocal_rank == 1.0


def test_review_rows_and_label_summary():
    _questions, matches = _demo_matches()
    rows = match_review_rows(matches)
    rows[0]["match_relevance"] = "1"
    rows[0]["expert_question_quality"] = "4"
    rows[0]["expert_data_match_quality"] = "3"
    summary = summarize_review_labels(rows)

    assert rows[0]["recommended_design"]
    assert summary["labeled"] == 1
    assert summary["relevant"] == 1
    assert summary["mean_question_quality"] == 4.0
    assert summary["mean_data_match_quality"] == 3.0


def test_review_exports_preserve_interpretability_payloads(tmp_path):
    _questions, matches = _demo_matches()
    rows = match_review_rows(matches)
    records = match_review_records(matches)

    assessments = json.loads(rows[0]["assessments_json"])
    assert "score_variable_overlap" in rows[0]
    assert assessments["feasibility"]["recommended_design"]
    assert records[0]["score_components"]["combined"] == matches[0].score.combined
    assert records[0]["match"]["assessments"]["governance"]["reuse_score"] >= 0

    csv_path = tmp_path / "review_sheet.csv"
    jsonl_path = tmp_path / "review_sheet.jsonl"
    export_review_csv(matches, csv_path)
    export_review_jsonl(matches, jsonl_path)

    assert "score_components_json" in csv_path.read_text(encoding="utf-8").splitlines()[0]
    exported = read_jsonl(jsonl_path)
    assert exported[0]["assessments"]["feasibility"]["recommended_design"]
    assert exported[0]["match"]["score"]["combined"] == matches[0].score.combined


def test_review_rows_convert_to_training_label_schemas():
    _questions, matches = _demo_matches()
    rows = match_review_rows(matches)
    rows[0]["match_relevance"] = "5"
    rows[0]["expert_question_quality"] = "4"
    rows[0]["expert_data_match_quality"] = "3"
    rows[0]["expert_notes"] = "Useful match for review."

    match_labels = review_rows_to_match_labels(rows, annotator_id="expert-a")
    quality_scores = review_rows_to_question_quality_scores(rows, annotator_id="expert-a")
    training = review_rows_to_training_labels(rows, annotator_id="expert-a")

    assert match_labels[0].label == "relevant"
    assert match_labels[0].relevance_score == 1.0
    assert match_labels[0].question_quality_score == 4.0
    assert quality_scores[0].overall_score == 4.0
    assert training["question_data_match_labels"][0]["match_id"] == rows[0]["match_id"]
    assert training["question_quality_scores"][0]["question_id"] == rows[0]["question_id"]


def test_review_jsonl_records_can_convert_to_training_labels_with_nested_fallback():
    _questions, matches = _demo_matches()
    record = {
        "match": match_review_records(matches)[0]["match"],
        "match_relevance": "1",
        "expert_question_quality": "5",
        "expert_data_match_quality": "4",
    }

    labels = review_rows_to_match_labels([record], annotator_id="expert-a")

    assert labels[0].match_id == record["match"]["match_id"]
    assert labels[0].question_id == record["match"]["question"]["question_id"]
    assert labels[0].dataset_id == record["match"]["dataset"]["dataset_id"]


def test_legacy_relevance_aliases_still_convert():
    _questions, matches = _demo_matches()
    rows = match_review_rows(matches)
    rows[0]["expert_match_relevance"] = "1"
    legacy_rows = match_review_rows(matches)
    legacy_rows[0]["expert_relevance"] = "1"

    labels = review_rows_to_match_labels(rows, annotator_id="expert-a")
    legacy_labels = review_rows_to_match_labels(legacy_rows, annotator_id="expert-a")
    summary = summarize_review_labels(legacy_rows)

    assert labels[0].relevance_score == 1.0
    assert legacy_labels[0].relevance_score == 1.0
    assert summary["relevant"] == 1
