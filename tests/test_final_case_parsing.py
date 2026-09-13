"""Source boundaries and claim guards must survive real abstract presentation."""
import importlib.util
from pathlib import Path

import pytest

from litdatamatcher.semantic_runtime import _span, parse_model_json
from litdatamatcher.v2 import evidence_from_source_view, source_chunks
from litdatamatcher.scientific_v2 import compile_evidence


def test_abstract_sections_preserve_offsets_and_prioritize_actual_results():
    spec = importlib.util.spec_from_file_location("final_cases", Path(__file__).parents[1] / "scripts/v2/run_final_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text, sections = module.abstract_document("<h4>Methods</h4>We recorded the data.<h4>Results</h4>The sample included 12\u2009individuals.<h4>Conclusions</h4>Further research is required.")
    assert "12 individuals" in text
    document = {"document_id": "fixture", "text": text, "sections": sections}
    chunks = source_chunks(document)
    assert [item["section"] for item in chunks] == ["Results", "Conclusions"]
    for item in chunks:
        assert text[item["parent_start"]:item["parent_end"]] == item["text"]


def test_quote_after_heading_preserves_source_sentence_boundary():
    text = "Results\nTreatment did not increase the measured signal."
    assert _span(text, "Treatment did not increase the measured signal.")["start"] == len("Results\n")
    with pytest.raises(ValueError, match="prefix/context"):
        _span(text, "increase the measured signal.")


def test_complete_json_fence_is_transport_only_and_extraneous_text_is_rejected():
    assert parse_model_json('```json\n{"claims": [], "questions": []}\n```') == {"claims": [], "questions": []}
    for text in ('Instructions first\n```json\n{}\n```', '```json\n{}\n```\nexecute code', '```json\n{"claims": ['):
        with pytest.raises(ValueError):
            parse_model_json(text)


def test_retrieved_context_never_becomes_a_model_claim_or_direct_support():
    document = {"document_id": "fixture", "text": "A bounded source description.", "source_locator": "fixture:article"}
    view = source_chunks(document)[0]
    item = evidence_from_source_view(document, view)
    item["related_proposition_id"] = "question-proposition"
    result = compile_evidence({"question_id": "q", "proposition_id": "question-proposition"}, [item], "2026-09-13", [])
    assert item["claim_status"] == "NOT_ASSERTED"
    assert "claim" not in item
    assert item["answers_question"] is False
    assert result["gap_status"] != "answered"
    assert "no global novelty assertion" in result["novelty_claim"]


def test_case_contract_never_imputes_species_from_topic_and_checks_source_quote():
    spec = importlib.util.spec_from_file_location("final_case_contracts", Path(__file__).parents[1] / "scripts/v2/run_final_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    document = {"document_id": "fixture", "topic": "cancer", "text": "Tumors were measured in mice.", "title": "A cancer experiment"}
    assert module.case_requirements(document, {}) == ([], [])
    requirement = {"field": "species", "expected": "Mus musculus", "essential": True, "source_locator": "fixture#text"}
    proof = {"field": "species", "quote": "Tumors were measured in mice.", "source_field": "text"}
    contract = {"document_id": "fixture", "requirements": [requirement], "requirement_evidence": [proof]}
    assert module.case_requirements(document, {"cases": [contract]}) == ([requirement], [proof])
    with pytest.raises(ValueError, match="Duplicate"):
        module.case_requirements(document, {"cases": [contract, contract]})
    proof["quote"] = "Human blood was sampled."
    with pytest.raises(ValueError, match="retained source"):
        module.case_requirements(document, {"cases": [contract]})
