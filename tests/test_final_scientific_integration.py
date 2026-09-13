"""Regressions for composed Phase 2 scientific interfaces, using declared fixtures."""

import copy

import pytest

from litdatamatcher.modality_contract import compatibility
from litdatamatcher.scientific_dossier import build_dossier, validate_dossier
from litdatamatcher.scientific_v2 import assess_requirements, compile_evidence


def profile(assay="RNA-seq", species="Homo sapiens"):
    return {
        "dataset_id": "fixture-study",
        "assay_types": [assay],
        "organisms": [species],
        "capabilities": {
            "assay": {"value": assay, "status": "observed", "source_locator": "fixture:assay"},
            "species": {"value": species, "status": "observed", "source_locator": "fixture:species"},
        },
    }


@pytest.mark.parametrize("expected", ["RNA-seq", "rna sequencing"])
def test_assay_name_and_qualified_synonym_survive_family_guard(expected):
    result = assess_requirements([{"field": "assay", "expected": expected}], profile())
    assert result["eligibility"] == "DIRECT_FIT"


def test_organism_synonym_survives_adapter_guard():
    result = assess_requirements([{"field": "species", "expected": "human"}], profile())
    assert result["eligibility"] == "DIRECT_FIT"
    assert compatibility("RNA-seq", "human", profile()) == "PARTIAL"


def test_known_different_assays_are_not_equated_by_broad_family():
    result = assess_requirements([{"field": "assay", "expected": "RNA-seq"}], profile("microarray"))
    assert result["eligibility"] == "NOT_QUALIFIED"


def test_unknown_assay_has_no_false_incompatibility_from_string_family_comparison():
    result = assess_requirements([{"field": "assay", "expected": "unqualified assay alias"}], profile())
    assert result["eligibility"] == "REQUIRES_INSPECTION"


def compiled_fixture():
    question = {"question_id": "fixture-q", "question": "What does the recorded evidence establish?", "proposition_id": "fixture-p", "source_evidence_ids": ["fixture-e"]}
    evidence = [{"evidence_id": "fixture-e", "proposition_id": "fixture-p", "source_locator": "fixture:paragraph:1", "role": "background", "direction": "supports"}]
    bundle = compile_evidence(question, evidence, "2026-09-13", [{"source": "fixture", "status": "success"}])
    assessment = assess_requirements([{"field": "species", "expected": "Homo sapiens"}], profile())
    return question, bundle, assessment, profile(), ["Observed organism; other measurements need review."]


def test_compiler_output_is_accepted_by_dossier_without_novelty_overclaim():
    dossier = build_dossier(*compiled_fixture())
    assert validate_dossier(dossier)
    assert "no global novelty assertion" in dossier["unresolvedness"]["novelty_claim"]


@pytest.mark.parametrize("mutation", ["unlinked_question", "wrong_candidate", "missing_locator", "unknown_contradiction"])
def test_dossier_rejects_broken_scientific_lineage(mutation):
    args = copy.deepcopy(compiled_fixture())
    if mutation == "unlinked_question":
        args[0]["source_evidence_ids"] = ["not-in-bundle"]
    elif mutation == "wrong_candidate":
        args[3]["dataset_id"] = "different-study"
    elif mutation == "missing_locator":
        args[1]["evidence_items"][0].pop("source_locator")
    else:
        args[1]["contradictory_evidence_ids"] = ["not-in-bundle"]
    with pytest.raises(ValueError):
        build_dossier(*args)

