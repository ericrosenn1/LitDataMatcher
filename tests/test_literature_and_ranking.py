from litdatamatcher.datasets import discover_datasets_for_question
from litdatamatcher.literature import (
    analyze_literature_records,
    classify_sentence,
    deduplicate_questions,
    statement_to_question,
)
from litdatamatcher.meta_analysis import run_meta_analysis_node, synthesis_index
from litdatamatcher.ranking import rank_matches, score_question_dataset
from litdatamatcher.schemas import (
    DatasetRecord,
    DatasetVariable,
    EvidenceSynthesis,
    QuestionCandidate,
)


def _question_for_ranking():
    return QuestionCandidate(
        question_id="q_rank",
        question="Does antibiotic exposure alter microbiome recovery?",
        domain_terms=["antibiotic", "microbiome", "recovery"],
        required_variables=[
            "antibiotic_exposure",
            "microbiome_composition",
            "timepoint",
        ],
        population="human",
        significance_score=0.4,
        extraction_confidence=0.8,
    )


def _dataset_with_variables(
    dataset_id,
    variables,
    sample_size=1000,
    quality_score=0.8,
):
    return DatasetRecord(
        dataset_id=dataset_id,
        title="Human microbiome antibiotic cohort",
        source="fixture",
        description=(
            "Human gut microbiome antibiotic exposure with longitudinal recovery metadata."
        ),
        variables=[DatasetVariable(name=variable) for variable in variables],
        populations=["human"],
        assay_types=["16S rRNA sequencing"],
        sample_size=sample_size,
        quality_score=quality_score,
        license="public",
        access_type="public",
    )


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


def test_literature_ignores_future_language_in_methods_section():
    questions = analyze_literature_records(
        [
            {
                "title": "Methods-only future statement",
                "text": (
                    "Methods\nFuture studies should examine whether antibiotic "
                    "exposure changes gut microbiome recovery."
                ),
                "doi": "10.0000/methods-only",
            }
        ]
    )

    assert questions == []


def test_literature_ignores_reference_title_questions():
    questions = analyze_literature_records(
        [
            {
                "title": "Prediction model study",
                "text": (
                    "Discussion\nFuture studies should examine prediction performance.\n"
                    "References\n"
                    "Ldl cholesterol lowering in type 2 diabetes: what is the optimum approach?"
                ),
            }
        ]
    )

    assert questions
    assert all("optimum approach" not in question.question.lower() for question in questions)


def test_literature_preserves_limitation_and_infers_fallback_fields():
    questions = analyze_literature_records(
        [
            {
                "title": "Data-driven approach to predicting clinical outcomes",
                "abstract": (
                    "The study uses predictors, labeled samples, and model performance "
                    "to evaluate clinical outcome prediction in a patient cohort."
                ),
                "text": (
                    "Discussion\n"
                    "This is partly due to the lack of large number of observations in "
                    "the data, with total number of samples at 8,459, and also as a "
                    "result of a high degree of imbalanced data with negative label "
                    "versus positive label samples."
                ),
            }
        ]
    )

    assert len(questions) == 1
    question = questions[0]
    assert question.question == (
        "Can the reported finding be validated with larger, better-balanced "
        "labeled samples?"
    )
    assert "sample_size" in question.required_variables
    assert "class_label" in question.required_variables
    assert "predictor" in question.required_variables
    assert question.population == "human"
    assert "prediction_performance" in question.outcomes
    assert question.metadata["variable_inference"]["required_variables_reason"]


def test_statement_to_question_and_classification_keep_traceable_signals():
    question = statement_to_question(
        "It remains unclear whether antibiotic exposure changes remission."
    )
    question_origin, extraction_confidence, signals = classify_sentence(
        "Future studies should examine whether antibiotic exposure changes remission?"
    )

    assert question == "Does antibiotic exposure changes remission?"
    assert question_origin == "future_direction"
    assert extraction_confidence > 0.5
    assert "explicit_rq" in signals
    assert "future_direction" in signals


def test_deduplicate_questions_merges_exact_normalized_questions():
    first = QuestionCandidate(
        question_id="q1",
        question="Does antibiotic exposure affect remission?",
        source_ids=["a"],
        extraction_confidence=0.5,
    )
    second = QuestionCandidate(
        question_id="q2",
        question="Does antibiotic exposure affect remission?",
        source_ids=["b"],
        extraction_confidence=0.9,
    )

    merged = deduplicate_questions([first, second])

    assert len(merged) == 1
    assert sorted(merged[0].source_ids) == ["a", "b"]
    assert merged[0].extraction_confidence == 0.9


def test_question_candidate_loads_legacy_extraction_field_names():
    question = QuestionCandidate.from_dict(
        {
            "question_id": "q_legacy",
            "question": "Does antibiotic exposure affect remission?",
            "extraction_type": "open_question",
            "confidence": 0.8,
            "answerability_hint": 0.6,
            "evidence": [
                {
                    "text": "Future studies should examine remission after exposure.",
                    "confidence": 0.7,
                }
            ],
        }
    )

    assert question.question_origin == "future_direction"
    assert question.extraction_confidence == 0.8
    assert question.answerability == 0.6
    assert question.evidence[0].extraction_confidence == 0.7


def test_score_question_dataset_penalizes_missing_variables_and_reports_rationale():
    question = _question_for_ranking()
    complete = _dataset_with_variables(
        "complete",
        ["antibiotic_exposure", "microbiome_composition", "timepoint"],
    )
    sparse = _dataset_with_variables(
        "sparse",
        ["microbiome_composition"],
        sample_size=50,
        quality_score=0.5,
    )

    complete_score, _complete_rationale, complete_missing, _ = score_question_dataset(
        question, complete
    )
    sparse_score, sparse_rationale, sparse_missing, _ = score_question_dataset(
        question, sparse
    )

    assert complete_missing == []
    assert "antibiotic_exposure" in sparse_missing
    assert "timepoint" in sparse_missing
    assert complete_score.combined > sparse_score.combined
    assert any("missing variables" in item for item in sparse_rationale)


def test_score_question_dataset_reports_proxy_capability_support():
    question = QuestionCandidate(
        question_id="q_ml",
        question="Can the reported finding be validated with labeled samples?",
        required_variables=["sample_size", "class_label", "predictor"],
        population="human",
    )
    dataset = DatasetRecord(
        dataset_id="proxy",
        title="Proxy model-ready metadata",
        source="fixture",
        variables=[
            DatasetVariable(name="sample_size", category="study_design_feature"),
            DatasetVariable(name="predictor", category="derived_or_proxy_capability"),
        ],
        populations=["human"],
        sample_size=1000,
        metadata={"capability_caveats": ["proxy capability requires source inspection"]},
    )

    _score, rationale, missing, assessments = score_question_dataset(question, dataset)

    assert missing == ["class_label"]
    assert any("proxy capability support: predictor" in item for item in rationale)
    assert any("answerability class: proxy" in item for item in rationale)
    assert assessments["capability_support"]["proxy_capabilities"] == ["predictor"]
    assert assessments["capability_support"]["missing_capabilities"] == ["class_label"]


def test_score_question_dataset_returns_stable_assessment_payload():
    question = _question_for_ranking()
    dataset = _dataset_with_variables(
        "reviewable",
        ["antibiotic_exposure", "microbiome_composition", "timepoint"],
    )

    _score, _rationale, _missing, assessments = score_question_dataset(
        question, dataset
    )

    assert set(assessments) == {
        "feasibility",
        "governance",
        "capability_support",
        "modality_contract",
    }
    assert {"overall", "variable_coverage", "recommended_design"} <= set(
        assessments["feasibility"]
    )
    assert {"reuse_score", "license_score", "risk_flags"} <= set(
        assessments["governance"]
    )
    assert {"direct_capabilities", "missing_capabilities", "answerability_class"} <= set(
        assessments["capability_support"]
    )
    assert {"status", "contract", "required_modality"} <= set(
        assessments["modality_contract"]
    )


def test_ranking_uses_synthesis_strength_and_respects_top_n():
    question = _question_for_ranking()
    datasets = [
        _dataset_with_variables(
            "strong",
            ["antibiotic_exposure", "microbiome_composition", "timepoint"],
            sample_size=2000,
            quality_score=0.9,
        ),
        _dataset_with_variables(
            "weak",
            ["microbiome_composition"],
            sample_size=25,
            quality_score=0.3,
        ),
    ]
    synthesis = EvidenceSynthesis(
        cluster_id="c1",
        question_ids=[question.question_id],
        summary="Recurring antibiotic recovery question.",
        evidence_strength=0.95,
        recurrence_score=0.8,
        uncertainty=0.1,
    )

    matches = rank_matches(
        [question], datasets, {question.question_id: synthesis}, top_n=1
    )

    assert len(matches) == 1
    assert matches[0].dataset.dataset_id == "strong"
    assert matches[0].score.significance == 0.95
    assert any("literature recurrence" in item for item in matches[0].rationale)
