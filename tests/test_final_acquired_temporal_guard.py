"""Acquired source dates must constrain both asserted and contextual evidence."""

import pytest

from litdatamatcher.scientific_v2 import compile_evidence
from litdatamatcher.v2 import evidence_from_claim, evidence_from_source_view, source_chunks


def evidence(kind, document):
    if kind == "context":
        item = evidence_from_source_view(document, source_chunks(document)[0])
    else:
        item = evidence_from_claim({
            "claim_id": "fixture-claim", "subject": "We", "predicate": "observed",
            "object": "a signal", "status": "interpretation", "direction": "unknown",
            "source_locator": "fixture:date-source#chars=0:21", "statement": document["text"],
        }, document)
    item["related_proposition_id"] = "fixture-proposition"
    return item


def document(**changes):
    return {
        "document_id": "fixture-europepmc-record", "source": "europepmc",
        "source_locator": "fixture:date-source", "text": "We observed a signal.",
        "metadata": {"first_publication_date": "2025-01-02"}, **changes,
    }


def compile_item(item, as_of):
    return compile_evidence({"question_id": "fixture-question", "proposition_id": "fixture-proposition"}, [item], as_of, [])


@pytest.mark.parametrize("kind", ["claim", "context"])
def test_qualified_acquired_date_rejects_past_assessment_and_preserves_source(kind):
    item = evidence(kind, document())
    with pytest.raises(ValueError, match="postdates"):
        compile_item(item, "2024-12-31")
    for as_of in ("2025-01-02", "2026-09-13"):
        bundle = compile_item(item, as_of)
        retained = bundle["evidence_items"][0]
        assert retained["publication_date"] == "2025-01-02"
        assert retained["publication_date_provenance"]["source_field"] == "metadata.first_publication_date"
        assert retained["publication_date_provenance"]["source_locator"] == "fixture:date-source"
        assert retained["answers_question"] is False
        assert bundle["gap_status"] == "insufficient-coverage"


@pytest.mark.parametrize("kind", ["claim", "context"])
@pytest.mark.parametrize("changes", [
    {"source": "generic"}, {"source": "crossref"}, {"source": None},
    {"metadata": None}, {"metadata": {}},
    *[{"metadata": {"first_publication_date": value}} for value in (
        "2025", "2025-01", "2025-02-30", "2025-01-02T00:00:00", "", None, 2025,
    )],
])
def test_unqualified_missing_partial_or_invalid_dates_remain_unknown(kind, changes):
    item = evidence(kind, document(**changes))
    assert item["publication_date"] is None
    assert "publication_date_provenance" not in item
    assert compile_item(item, "2024-12-31")["gap_status"] == "insufficient-coverage"


@pytest.mark.parametrize("kind", ["claim", "context"])
def test_explicit_document_date_retains_precedence(kind):
    item = evidence(kind, document(publication_date="2023-01-01"))
    assert compile_item(item, "2024-12-31")["evidence_items"][0]["publication_date"] == "2023-01-01"
