"""Source boundaries and claim guards must survive real abstract presentation."""
import importlib.util
from pathlib import Path

import pytest

from litdatamatcher.semantic_runtime import _span
from litdatamatcher.v2 import source_chunks


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
