"""Provenance-complete, scoped scientific dossier rendering and validation."""
from __future__ import annotations
import html
import json
from .data_plane import digest


def build_dossier(question: dict, bundle: dict, assessment: dict, candidate: dict, rationale: list[str]) -> dict:
    if not question.get("question_id") or not question.get("question") or not question.get("source_evidence_ids"):
        raise ValueError("Dossier question requires source-determined identity and evidence IDs")
    if not bundle.get("evidence_items") or "gap_status" not in bundle or not bundle.get("novelty_claim"):
        raise ValueError("Dossier requires scoped evidence bundle")
    if "global novelty" in str(bundle["novelty_claim"]).casefold():
        raise ValueError("Dossier cannot assert global novelty")
    if not candidate.get("dataset_id") or "compatibility_status" not in assessment:
        raise ValueError("Dossier candidate requires compatibility assessment")
    return {"schema_version": "scientific_dossier_v1", "dossier_id": digest([question, bundle, assessment, candidate])[:24], "question": question, "unresolvedness": {"gap_status": bundle["gap_status"], "as_of": bundle.get("as_of", "UNKNOWN"), "novelty_claim": bundle["novelty_claim"]}, "source_evidence": bundle["evidence_items"], "experimental_requirements": assessment.get("requirements", []), "candidate_dataset": candidate, "compatibility": {"status": assessment["compatibility_status"], "eligibility": assessment.get("eligibility"), "missing_fields": [item["field"] for item in assessment.get("requirements", []) if item["status"] == "UNKNOWN"]}, "dependence": bundle.get("dependence_groups", []), "contradictions": bundle.get("contradictory_evidence_ids", []), "ranking_rationale": rationale, "review_status": "SOURCE_ASSISTED_PENDING_EXPERT_REVIEW", "limitations": "Source-assisted dossier; no global novelty, expert validation, causal conclusion, or experiment claim."}


def validate_dossier(dossier: dict) -> bool:
    return bool(dossier.get("question", {}).get("source_evidence_ids") and dossier.get("source_evidence") and dossier.get("candidate_dataset", {}).get("dataset_id") and dossier.get("compatibility", {}).get("status") and "global novelty" not in str(dossier.get("unresolvedness", {}).get("novelty_claim", "")).casefold() and dossier.get("review_status") == "SOURCE_ASSISTED_PENDING_EXPERT_REVIEW")


def render_dossier(dossier: dict) -> str:
    if not validate_dossier(dossier):
        raise ValueError("Invalid scientific dossier")
    return "<article><h1>Scientific dossier</h1><h2>{}</h2><p>{}</p><pre>{}</pre></article>".format(html.escape(dossier["question"]["question"]), html.escape(dossier["review_status"]), html.escape(json.dumps(dossier, sort_keys=True)))
