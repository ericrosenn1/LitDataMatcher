"""Engineering challenges, not independently annotated scientific accuracy."""

import json

import pytest

from litdatamatcher.question_compiler import compile_question, matching_requirements
from litdatamatcher.semantic_runtime import validate_extraction
from tests.test_semantic_runtime import case, claim, stub_runtime


def relation(text, subject, predicate, target, direction, negated=False):
    return {
        "quote": text,
        "subject": subject,
        "predicate": predicate,
        "object": target,
        "direction": direction,
        "negated": negated,
        "status": "direct_experiment",
        "context": None,
        "comparator": None,
    }


def check_relation(record):
    return validate_extraction({"claims": [record], "questions": []}, case(record["quote"]))


def test_no_change_cannot_borrow_the_other_clause():
    text = "Diet increased sleep duration, whereas exercise produced no change in reaction time."
    result = check_relation(
        relation(text, "Diet", "increased", "sleep duration", "no_change", True)
    )
    assert not result["claims"]
    assert result["rejected"]


def test_unrelated_negation_does_not_negate_selected_relation():
    text = "Diet increased sleep duration, whereas exercise produced no change in reaction time."
    result = check_relation(relation(text, "Diet", "increased", "sleep duration", "increase"))
    assert result["claims"][0]["negated"] is False


@pytest.mark.parametrize(
    "text,predicate",
    [
        ("Treatment did not reduce IL6 in human cells.", "did not reduce"),
        ("Treatment had no effect on IL6 in human cells.", "had no effect on"),
    ],
)
def test_source_bound_no_change_survives(text, predicate):
    assert check_relation(relation(text, "Treatment", predicate, "IL6", "no_change", True))[
        "claims"
    ]


def test_negation_cannot_be_borrowed_for_association():
    text = "Diet was associated with sleep duration, but exercise was not measured."
    assert not check_relation(
        relation(text, "Diet", "was associated with", "sleep duration", "association", True)
    )["claims"]
    assert check_relation(
        relation(text, "Diet", "was associated with", "sleep duration", "association")
    )["claims"]


def test_predicate_and_object_cannot_be_swapped_or_cross_clauses():
    text = "Diet increased sleep duration, but exercise changed reaction time."
    assert not check_relation(relation(text, "Diet", "changed", "reaction time", "unknown"))[
        "claims"
    ]
    assert not check_relation(relation(text, "Diet", "sleep duration", "increased", "unknown"))[
        "claims"
    ]


def test_embedded_word_is_not_a_source_role():
    text = "Treatment increased IL6 in human cells."
    assert not check_relation(relation(text, "Treatment", "increased", "IL", "increase"))["claims"]


@pytest.mark.parametrize(
    "failure",
    [
        OSError("retry unavailable"),
        RuntimeError("retry memory pressure"),
        ValueError("retry too long"),
    ],
)
def test_generation_retry_keeps_accepted_partition_and_cache(monkeypatch, tmp_path, failure):
    first = json.dumps({"claims": [claim(), claim(direction="increase")], "questions": []})
    runtime = stub_runtime(monkeypatch, [first])
    original = runtime._generate
    calls = []

    def generate(document, repair="", system_prompt=None):
        calls.append(repair)
        if len(calls) == 1:
            return original(document, repair, system_prompt)
        raise failure

    runtime._generate = generate
    result = runtime.extract(case(), tmp_path)
    assert len(result["claims"]) == 1
    assert result["inference_manifest"]["generation_failures"] == 1
    assert (
        result["inference_manifest"]["attempts"][-1]["generation_error"] == type(failure).__name__
    )
    assert result["inference_manifest"]["status"] == "validated_with_rejections"
    assert runtime.extract(case(), tmp_path)["claims"] == result["claims"]


def test_initial_generation_failure_is_not_faked_as_success(monkeypatch):
    runtime = stub_runtime(monkeypatch, [])

    def generate(*args, **kwargs):
        raise OSError("initial unavailable")

    runtime._generate = generate
    with pytest.raises(OSError, match="initial unavailable"):
        runtime.extract(case())


def test_risk_coreference_requires_an_explicit_antecedent_and_records_it():
    question = "Does drug A increase this risk in adults?"
    context = "NAION risk was noted. BMI was measured as a covariate."
    result = compile_question(question, source_text=context, source_locator="source:development")
    outcome = next(row for row in result["requirements"] if row["field"] == "outcome")
    assert outcome["expected"] == "risk of NAION"
    evidence = outcome["provenance"]["antecedent_evidence"]
    assert context[evidence["start"] : evidence["end"]] == evidence["text"] == "NAION risk"
    assert evidence["source_locator"] == "source:development#context"


@pytest.mark.parametrize(
    "context",
    [
        "NAION and BMI were measured.",
        "The risk of NAION was noted; the risk of VTE was also elevated.",
    ],
)
def test_ambiguous_or_unsupported_risk_reference_abstains(context):
    result = compile_question("Does drug A increase this risk in adults?", source_text=context)
    assert result["compilation_status"] == "QUESTION_UNDERDETERMINED"
    assert matching_requirements(result) == []
    assert any(item["code"] == "UNRESOLVED_RISK_REFERENCE" for item in result["diagnostics"])


def test_explicit_user_outcome_resolves_ambiguous_reference_without_inventing_one():
    result = compile_question(
        "Does drug A increase this risk in adults?",
        source_text="NAION and BMI were measured.",
        user_constraints=[{"field": "outcome", "expected": "risk of NAION"}],
    )
    assert result["compilation_status"] == "COMPLETE"
    assert (
        next(row for row in matching_requirements(result) if row["field"] == "outcome")["expected"]
        == "risk of NAION"
    )


@pytest.mark.parametrize(
    "question,purpose",
    [
        ("Are biomarkers predictive of survival in adults?", "prediction"),
        ("Is diet correlated with sleep quality in adults?", "associational"),
        ("Are biomarkers associated with survival in adults?", "associational"),
        ("Does stimulation alter sleep quality in adults?", "perturbational_mechanistic"),
    ],
)
def test_bounded_morphology_recognizes_question_purpose(question, purpose):
    result = compile_question(question)
    assert result["question_purpose"] == purpose
    assert result["compilation_status"] == "COMPLETE"


def test_population_is_not_swallowed_by_outcome_or_comparator():
    result = compile_question("Does diet affect sleep quality compared with placebo in children?")
    roles = {role["field"]: role for role in result["entity_roles"]}
    assert roles["species"]["expected"] == "human"
    assert roles["species"]["observed_text"] == "children"
    assert roles["outcome"]["expected"] == "sleep quality"
    assert roles["comparator"]["expected"] == "placebo"


def test_mixed_species_are_not_silently_reduced_to_one():
    result = compile_question("Does treatment reduce inflammation in humans and mice?")
    assert result["compilation_status"] == "QUESTION_UNDERDETERMINED"
    assert {role["expected"] for role in result["entity_roles"] if role["field"] == "species"} == {
        "human",
        "mouse",
    }
    assert (
        result["answer_specification"]["target_relationship_or_quantity"]["population_or_species"]
        is None
    )
    assert matching_requirements(result) == []


def test_matching_user_constraint_resolves_once_without_double_counting():
    result = compile_question(
        "Does treatment reduce inflammation in humans and mice?",
        user_constraints=[{"field": "species", "expected": "human"}],
    )
    assert result["compilation_status"] == "COMPLETE"
    species = [row for row in matching_requirements(result) if row["field"] == "species"]
    assert len(species) == 1


def test_raw_question_and_multiline_spans_are_preserved():
    question = "Does  drug A\nreduce fever in children?"
    result = compile_question(question)
    assert result["original_gap_statement"] == question
    assert result["normalized_research_question"] == "Does drug A reduce fever in children?"
    for role in result["entity_roles"]:
        span = role["evidence"]
        assert question[span["start"] : span["end"]] == span["text"]
    assert {row["expected"] for row in matching_requirements(result)} >= {
        "drug A",
        "fever",
        "human",
    }


def test_duplicate_user_field_is_an_explicit_input_error():
    with pytest.raises(ValueError, match="only once"):
        compile_question(
            "Does diet reduce inflammation?",
            user_constraints=[
                {"field": "species", "expected": "human"},
                {"field": "species", "expected": "mouse"},
            ],
        )


def test_task_word_without_target_roles_does_not_form_complete_contract():
    result = compile_question("Are clinical predictors adequate?")
    assert result["question_purpose"] == "prediction"
    assert result["compilation_status"] == "QUESTION_UNDERDETERMINED"
    assert matching_requirements(result) == []


def test_declarative_copula_adverbs_are_not_misparsed_as_exposure():
    result = compile_question(
        "Occupational exposure is not consistently associated with an increased risk of cancer."
    )
    assert not any(role["expected"] == "not consistently" for role in result["entity_roles"])
    assert result["compilation_status"] == "QUESTION_UNDERDETERMINED"


def test_child_pugh_score_is_not_a_population():
    result = compile_question(
        "Intervention: treatment | Comparator: placebo | Outcome: Child-Pugh scores."
    )
    assert not any(role["field"] == "species" for role in result["entity_roles"])


def test_nested_in_phrase_remains_part_of_outcome():
    result = compile_question("Does treatment reduce changes in mood in humans?")
    outcome = next(role for role in result["entity_roles"] if role["field"] == "outcome")
    assert outcome["expected"] == "changes in mood"
