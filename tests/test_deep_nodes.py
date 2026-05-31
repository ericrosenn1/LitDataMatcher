from litdatamatcher.datasets import discover_datasets_for_question
from litdatamatcher.evaluation import evaluate_question_extraction, evaluate_ranking
from litdatamatcher.feasibility import assess_pair_feasibility
from litdatamatcher.literature import analyze_literature_records
from litdatamatcher.meta_analysis import run_meta_analysis_node, synthesis_index
from litdatamatcher.ontology import normalize_variable_name
from litdatamatcher.ranking import rank_matches
from litdatamatcher.review import match_review_rows, summarize_review_labels


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
    rows[0]["expert_relevance"] = "1"
    summary = summarize_review_labels(rows)

    assert rows[0]["recommended_design"]
    assert summary["labeled"] == 1
    assert summary["relevant"] == 1
