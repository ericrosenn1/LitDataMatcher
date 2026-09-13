"""Provenance-complete, scoped scientific dossier rendering and validation."""
from __future__ import annotations

import html
import json

from .data_plane import digest

SCOPED_NOVELTY_STATEMENTS = frozenset({
    "Limited to searched coverage",
    "Limited to recorded searched coverage; no global novelty assertion",
})


def _valid_evidence(question: dict, items: list, contradictions: list) -> bool:
    if not isinstance(items, list) or not items:
        return False
    if not all(isinstance(item, dict) and item.get("evidence_id") and str(item.get("source_locator", "")).strip() for item in items):
        return False
    ids = {item["evidence_id"] for item in items}
    return len(ids) == len(items) and set(question.get("source_evidence_ids", [])) <= ids and set(contradictions) <= ids


def build_dossier(question: dict, bundle: dict, assessment: dict, candidate: dict, rationale: list[str]) -> dict:
    if not question.get("question_id") or not question.get("question") or not question.get("source_evidence_ids"):
        raise ValueError("Dossier question requires source-determined identity and evidence IDs")
    if not bundle.get("evidence_items") or "gap_status" not in bundle or not bundle.get("novelty_claim"):
        raise ValueError("Dossier requires scoped evidence bundle")
    if bundle["novelty_claim"] not in SCOPED_NOVELTY_STATEMENTS:
        raise ValueError("Dossier requires a recognized scoped statement; global novelty is unsupported")
    if not _valid_evidence(question, bundle["evidence_items"], bundle.get("contradictory_evidence_ids", [])):
        raise ValueError("Dossier evidence IDs and source locators must resolve within its bundle")
    if bundle.get("question_id", question["question_id"]) != question["question_id"]:
        raise ValueError("Dossier question does not match its evidence bundle")
    if not candidate.get("dataset_id") or "compatibility_status" not in assessment:
        raise ValueError("Dossier candidate requires compatibility assessment")
    if assessment.get("dataset_id", candidate["dataset_id"]) != candidate["dataset_id"]:
        raise ValueError("Dossier assessment refers to a different candidate")
    if not rationale or not all(isinstance(item, str) and item.strip() for item in rationale):
        raise ValueError("Dossier requires an explicit ranking rationale")
    return {"schema_version": "scientific_dossier_v1", "dossier_id": digest([question, bundle, assessment, candidate])[:24], "question": question, "unresolvedness": {"gap_status": bundle["gap_status"], "as_of": bundle.get("as_of", "UNKNOWN"), "novelty_claim": bundle["novelty_claim"]}, "source_evidence": bundle["evidence_items"], "experimental_requirements": assessment.get("requirements", []), "candidate_dataset": candidate, "compatibility": {"status": assessment["compatibility_status"], "eligibility": assessment.get("eligibility"), "missing_fields": [item["field"] for item in assessment.get("requirements", []) if item["status"] == "UNKNOWN"]}, "dependence": bundle.get("dependence_groups", []), "contradictions": bundle.get("contradictory_evidence_ids", []), "ranking_rationale": rationale, "review_status": "SOURCE_ASSISTED_PENDING_EXPERT_REVIEW", "limitations": "Source-assisted dossier; no global novelty, expert validation, causal conclusion, or experiment claim."}


def validate_dossier(dossier: dict) -> bool:
    return bool(dossier.get("question", {}).get("source_evidence_ids") and _valid_evidence(dossier.get("question", {}), dossier.get("source_evidence"), dossier.get("contradictions", [])) and dossier.get("candidate_dataset", {}).get("dataset_id") and dossier.get("compatibility", {}).get("status") and dossier.get("unresolvedness", {}).get("novelty_claim") in SCOPED_NOVELTY_STATEMENTS and dossier.get("ranking_rationale") and dossier.get("review_status") == "SOURCE_ASSISTED_PENDING_EXPERT_REVIEW")


def render_dossier(dossier: dict) -> str:
    if not validate_dossier(dossier):
        raise ValueError("Invalid scientific dossier")
    return "<article><h1>Scientific dossier</h1><h2>{}</h2><p>{}</p><pre>{}</pre></article>".format(html.escape(dossier["question"]["question"]), html.escape(dossier["review_status"]), html.escape(json.dumps(dossier, sort_keys=True)))
