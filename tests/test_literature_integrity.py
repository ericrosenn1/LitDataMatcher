from litdatamatcher.literature_integrity import (
    consolidate_literature_rows,
    evidence_eligible_literature,
    invalidate_affected_derivations,
)


def test_retraction_and_cross_source_duplicate_are_not_silent_evidence():
    row = {
        "source_id": "pubmed:1", "source": "pubmed", "doi": "10/x",
        "source_provenance": {"source_type": "pubmed", "retrieval_time_utc": "2026-09-08T00:00:00Z"},
        "version_relationships": {"is-retracted-by": [{"id": "2"}]},
        "metadata": {"alternate_source_ids": ["crossref:10/x"], "alternate_source_provenance": [{"source_type": "crossref", "retrieval_time_utc": "2026-09-08T00:01:00Z"}]},
    }
    normalized = consolidate_literature_rows([row], [{"source": "pubmed", "status": "OBSERVED"}, {"source": "crossref", "status": "OBSERVED"}])[0]
    integrity = normalized["metadata"]["literature_integrity"]
    assert integrity["lifecycle_status"] == "RETRACTED"
    assert len(integrity["source_snapshots"]) == 2
    assert integrity["evidence_eligibility"] == "INELIGIBLE_REQUIRES_VERSION_REVIEW"
    assert not evidence_eligible_literature(normalized)


def test_missing_fulltext_and_source_failure_remain_unknown_and_invalidate():
    old = consolidate_literature_rows([{"source_id": "europepmc:MED:1", "source": "europepmc"}])[0]
    new = consolidate_literature_rows([{"source_id": "europepmc:MED:1", "source": "europepmc", "version_relationships": {"is-correction-of": [{"id": "old"}]}}], [{"source": "europepmc", "status": "UNKNOWN_RETRIEVAL_OR_SCHEMA_FAILURE"}])[0]
    integrity = new["metadata"]["literature_integrity"]
    assert integrity["fulltext_status"] == "UNKNOWN"
    assert integrity["source_statuses"][0]["status"] == "UNKNOWN_RETRIEVAL_OR_SCHEMA_FAILURE"
    invalidation = invalidate_affected_derivations(old, new, ["claim:1", "claim:1", "question:1"])
    assert invalidation["status"] == "INVALIDATED"
    assert invalidation["derivation_ids"] == ["claim:1", "question:1"]
