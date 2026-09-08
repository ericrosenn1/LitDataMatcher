"""Deterministic cross-source regression receipt for Phase 2 contract layers."""

from __future__ import annotations

from .data_plane import digest
from .expert_review import build_blinded_review_packet
from .literature_integrity import consolidate_literature_rows
from .ontology import normalize_entity
from .scientific_v2 import assess_requirements, compile_evidence


def build_cross_source_receipt() -> dict:
    lifecycle = consolidate_literature_rows([{"source_id": "pubmed:fixture", "source": "pubmed", "source_provenance": {"source_type": "pubmed"}, "version_relationships": {"is-retracted-by": [{"id": "x"}]}, "metadata": {"alternate_source_ids": ["crossref:fixture"]}}])[0]
    profile = {"dataset_id": "ena:fixture", "assay_types": ["WGS"], "organisms": ["Homo sapiens"], "metadata": {"dependence": {"technical_run_count": 8, "donor_links": "AMBIGUOUS_NOT_INFERRED"}}, "capabilities": {"biological_sample_count": {"value": 8, "status": "observed", "source_locator": "technical runs"}}}
    wrong_modality = assess_requirements([{"field": "modality", "expected": "bulk_transcriptomics"}], profile)
    units = assess_requirements([{"field": "biological_sample_count", "expected": 8}], profile)
    question = {"question_id": "q", "proposition_id": "p", "conditions": {"organism": "human"}}
    paper = {"evidence_id": "paper", "proposition_id": "p", "role": "direct_test", "direction": "supports", "source_id": "PMID:1", "study_id": "GSE:1", "conditions": question["conditions"], "measurement_type": "observation", "scope_match": "exact", "answers_question": True, "publication_date": "2026-01-01", "source_locator": "fixture:paper"}
    repository = {"evidence_id": "repository", "proposition_id": "p", "role": "metadata", "direction": "supports", "source_id": "GSE:1", "study_id": "GSE:1", "conditions": question["conditions"], "measurement_type": "metadata", "scope_match": "exact", "publication_date": "2026-01-01", "source_locator": "fixture:repo"}
    kg = {"evidence_id": "kg", "proposition_id": "p", "role": "curation", "direction": "contradicts", "source_id": "KG:1", "source_of_source": "GSE:1", "conditions": question["conditions"], "measurement_type": "curation", "scope_match": "related", "publication_date": "2026-01-01", "source_locator": "fixture:kg"}
    bundle = compile_evidence(question, [paper, repository, kg], "2026-09-08", [{"source": "fixture", "status": "success"}])
    packet = build_blinded_review_packet([{"match_id": "m", "question": {"question": "fixture", "evidence": [{"source_locator": "fixture:span"}]}, "dataset": {"dataset_id": "d"}, "score": 1.0}], ["a"])["packet"]
    item = packet["items"][0]
    outcomes = {"retracted_duplicate_ineligible": lifecycle["metadata"]["literature_integrity"]["evidence_eligibility"], "wrong_modality": wrong_modality["compatibility_status"], "technical_runs_not_donors": units["compatibility_status"], "ambiguous_tissue": normalize_entity("gut", "tissue_cell_type")["status"], "paper_repository_kg_groups": len(bundle["dependence_groups"]), "independent_support": bundle["independent_support_count"], "contradiction_retained": bundle["contradictory_evidence_ids"], "packet_score_masked": "score" not in item}
    return {"schema_version": "v2_cross_source_adversarial_v1", "fixture_scope": "synthetic/local regression fixture only", "outcomes": outcomes, "input_digest": digest([lifecycle, profile, paper, repository, kg]), "validation_status": "PASS" if outcomes == {"retracted_duplicate_ineligible": "INELIGIBLE_REQUIRES_VERSION_REVIEW", "wrong_modality": "INCOMPATIBLE", "technical_runs_not_donors": "UNKNOWN", "ambiguous_tissue": "AMBIGUOUS", "paper_repository_kg_groups": 1, "independent_support": None, "contradiction_retained": ["kg"], "packet_score_masked": True} else "FAIL"}
