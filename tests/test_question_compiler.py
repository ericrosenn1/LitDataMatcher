import json

from litdatamatcher.question_compiler import compile_question, matching_requirements
from litdatamatcher.scientific_v2 import assess_requirements
from litdatamatcher.v2 import main, requirement_aware_candidates


def _capability(value):
    return {"value": value, "status": "observed", "source_locator": "fixture:metadata"}


def test_risk_question_compiles_estimable_contract_without_predicting_direction():
    result = compile_question(
        "Does semaglutide increase risk of NAION in adults with type 2 diabetes?",
        source_locator="question:semaglutide",
    )
    assert result["compilation_status"] == "COMPLETE"
    assert result["question_purpose"] == "associational"
    assert result["answer_specification"]["predicted_value_or_direction"] is None
    requirements = matching_requirements(result)
    assert {row["field"] for row in requirements} >= {
        "species", "intervention", "outcome", "joint_observation", "temporal_order", "outcome_ascertainment"
    }
    assert next(row for row in result["requirements"] if row["field"] == "temporal_order")["provenance"]["category"] == "METHOD_DERIVED"


def test_compiled_contract_can_qualify_observed_joint_candidate():
    result = compile_question("Does LPS increase IL6 in humans?", source_locator="question")
    requirements = matching_requirements(result)
    dataset = {
        "dataset_id": "usable-cohort",
        "capabilities": {
            "species": _capability("human"),
            "intervention": _capability("LPS"),
            "outcome": _capability("IL6"),
            "joint_observation": _capability(True),
            "temporal_order": _capability("exposure_before_outcome"),
            "outcome_ascertainment": _capability(True),
        },
    }
    assert assess_requirements(requirements, dataset)["eligibility"] == "DIRECT_FIT"


def test_missing_metadata_is_inspection_not_confirmed_absence():
    result = compile_question("Does semaglutide increase risk of NAION in adults?", source_locator="question")
    assessment = assess_requirements(matching_requirements(result), {"dataset_id": "metadata-thin", "capabilities": {}})
    assert assessment["eligibility"] == "REQUIRES_INSPECTION"
    assert any(row["status"] == "UNKNOWN" for row in assessment["requirements"])


def test_ambiguous_question_cannot_produce_a_best_match_contract():
    result = compile_question("What should scientists investigate next?", source_locator="question")
    assert result["compilation_status"] == "QUESTION_UNDERDETERMINED"
    assert matching_requirements(result) == []
    assert any(item["code"] == "NO_ASSESSABLE_NECESSITIES" for item in result["diagnostics"])


def test_user_override_preserves_conflict_and_wins_effective_mapping():
    result = compile_question(
        "Does semaglutide increase risk of NAION in adults?",
        source_locator="question",
        user_constraints=[{"field": "species", "expected": "mouse", "essential": True}],
    )
    species = [item for item in result["requirements"] if item["field"] == "species"]
    assert len(species) == 2
    assert any(item["provenance"]["category"] == "USER_SPECIFIED" and item["effective"] for item in species)
    assert {item["expected"] for item in matching_requirements(result) if item["field"] == "species"} == {"mouse"}
    assert any(item["code"] == "USER_OVERRIDE_CONFLICT" for item in result["diagnostics"])


def test_bounded_coreference_uses_context_and_preserves_the_question_span():
    result = compile_question(
        "Can semaglutide increase this risk in adults?",
        source_text="NAION is a serious optic neuropathy. The risk of NAION needs investigation.",
        source_locator="MED:40810985",
    )
    outcome = next(item for item in result["requirements"] if item["field"] == "outcome")
    assert outcome["expected"] == "risk of NAION"
    assert outcome["provenance"]["category"] == "NORMALIZED"
    assert outcome["provenance"]["evidence"]["text"] == "this risk"


def test_compile_cli_writes_machine_readable_artifact(tmp_path):
    output = tmp_path / "compiled.json"
    assert main(["compile", "--question", "Does semaglutide increase risk of NAION in adults?", "--out", str(output)]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == "question-requirements-compiler-1.0"
    assert artifact["matching_requirements"]


def test_relevance_screen_does_not_promote_shared_species_to_inspection_candidate():
    datasets = [
        {"dataset_id": "unrelated", "title": "Human UVB skin expression", "summary": "Public risk metadata", "capabilities": {}},
        {"dataset_id": "relevant-but-thin", "title": "Semaglutide exposure cohort", "summary": "Outcome metadata pending", "capabilities": {}},
    ]
    assert [row["dataset_id"] for row in requirement_aware_candidates(datasets, ["semaglutide", "risk of NAION"])] == ["relevant-but-thin"]
