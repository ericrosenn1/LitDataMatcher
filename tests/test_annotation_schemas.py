from litdatamatcher.schemas import (
    DatasetCapability,
    DerivedCapabilityLabel,
    DerivedVariableRule,
    EvidenceSpanLabel,
    ExpertPaperAnnotation,
    QuestionDataMatchLabel,
    QuestionLabel,
    QuestionQualityScore,
)


def test_paper_annotation_schema_round_trips_nested_labels():
    question_label = QuestionLabel(
        question_id="q1",
        annotator_id="expert-a",
        source_id="paper-1",
        label="accepted",
        is_valid_open_question=True,
    )
    span_label = EvidenceSpanLabel(
        question_id="q1",
        source_id="paper-1",
        text="Future studies should examine longitudinal microbiome recovery.",
        section="discussion",
        start_char=12,
        end_char=72,
        label="supporting",
        confidence=0.8,
    )

    annotation = ExpertPaperAnnotation(
        source_id="paper-1",
        annotator_id="expert-a",
        title="Microbiome recovery",
        doi="10.0000/example",
        question_labels=[question_label],
        evidence_span_labels=[span_label],
    )
    restored = ExpertPaperAnnotation.from_dict(annotation.to_dict())

    assert restored.annotation_id == annotation.annotation_id
    assert restored.question_labels[0].label == "accepted"
    assert restored.evidence_span_labels[0].label == "supporting"
    assert restored.evidence_span_labels[0].confidence == 0.8


def test_question_and_match_training_labels_validate_scores():
    quality = QuestionQualityScore(
        question_id="q1",
        annotator_id="expert-a",
        clarity_score="4",
        overall_score=6,
    )
    match_label = QuestionDataMatchLabel(
        match_id="m1",
        question_id="q1",
        dataset_id="d1",
        annotator_id="expert-a",
        relevance_score=0.9,
        question_quality_score=quality.overall_score,
        data_match_quality_score=5.5,
    )
    restored = QuestionDataMatchLabel.from_dict(match_label.to_dict())

    assert quality.clarity_score == 4.0
    assert quality.overall_score == 5.0
    assert restored.label == "relevant"
    assert restored.data_match_quality_score == 5.0


def test_dataset_capability_and_derived_label_round_trip():
    rule = DerivedVariableRule(
        output_variable="body_mass_index",
        input_variables=["height", "weight"],
        expression="weight_kg / height_m ** 2",
        assumptions=["height is measured in meters", "weight is measured in kilograms"],
        confidence=0.75,
    )
    capability = DatasetCapability(
        dataset_id="dataset-1",
        variable_name="body_mass_index",
        capability_type="derived",
        source_variable_names=["height", "weight"],
        derivation_rule_id=rule.rule_id,
        confidence=0.7,
    )
    label = DerivedCapabilityLabel(
        capability_id=capability.capability_id,
        dataset_id=capability.dataset_id,
        rule_id=rule.rule_id,
        annotator_id="expert-a",
        is_plausible="yes",
        usefulness_score=4,
        evidence_quality_score=3,
    )

    assert DerivedVariableRule.from_dict(rule.to_dict()).rule_id == rule.rule_id
    assert DatasetCapability.from_dict(capability.to_dict()).capability_type == "derived"
    assert DerivedCapabilityLabel.from_dict(label.to_dict()).is_plausible is True
