from litdatamatcher.datasets import discover_datasets_for_question
from litdatamatcher.literature import analyze_literature_records
from litdatamatcher.meta_analysis import run_meta_analysis_node, synthesis_index
from litdatamatcher.ranking import rank_matches


def test_open_question_identification_and_ranking_end_to_end():
    records = [
        {
            "title": "Antibiotic perturbation and IBD microbiome recovery",
            "abstract": (
                "Future studies should examine whether longitudinal gut microbiome and "
                "antibiotic exposure data predict remission after treatment."
            ),
            "text": (
                "Limitations. The small sample size and incomplete diet metadata limit "
                "causal interpretation."
            ),
            "doi": "10.0000/example",
        }
    ]

    questions = analyze_literature_records(records)
    syntheses = run_meta_analysis_node(questions)
    datasets = []
    for question in questions:
        datasets.extend(discover_datasets_for_question(question))
    matches = rank_matches(questions, datasets, synthesis_index(syntheses), top_n=10)

    assert questions
    assert syntheses
    assert matches
    assert matches[0].score.combined > 0
    assert matches[0].rationale


def test_questions_preserve_evidence_provenance():
    questions = analyze_literature_records(
        [
            {
                "title": "A study",
                "abstract": "It remains unclear whether metabolomics improves IBD remission prediction.",
                "doi": "10.0000/provenance",
            }
        ]
    )

    assert questions[0].evidence[0].doi == "10.0000/provenance"
    assert questions[0].source_ids
