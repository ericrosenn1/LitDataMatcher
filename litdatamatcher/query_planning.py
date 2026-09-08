"""Deterministic fail-closed source planning from declared adapter contracts."""
from __future__ import annotations


def plan_sources(requirements: dict, sources: list[dict]) -> dict:
    selected, decisions = [], []
    required_modality = requirements.get("modality")
    complete = bool(requirements.get("complete_candidate_universe", False))
    offline = bool(requirements.get("offline_required", False))
    for source in sources:
        name = str(source.get("name", "UNKNOWN"))
        reasons = []
        status = "ELIGIBLE"
        if not source.get("supported", False): status, reasons = "NOT_QUALIFIED", ["unsupported_route"]
        elif required_modality and required_modality not in source.get("modalities", []): status, reasons = "NOT_QUALIFIED", ["explicit_modality_mismatch"]
        elif source.get("access") in {"restricted", "unknown"}: status, reasons = "UNKNOWN", ["access_or_license_unavailable"]
        elif source.get("metadata_completeness") != "OBSERVED": status, reasons = "UNKNOWN", ["metadata_completeness_unknown"]
        elif offline and not source.get("offline_cache_available", False): status, reasons = "UNKNOWN", ["offline_cache_unavailable"]
        elif complete and source.get("candidate_universe_status") != "COMPLETE_CANDIDATE_UNIVERSE": status, reasons = "NOT_QUALIFIED", ["partial_candidate_universe"]
        if status == "ELIGIBLE": selected.append(name)
        decisions.append({"source": name, "status": status, "reasons": reasons, "query_permitted": status == "ELIGIBLE"})
    return {"schema_version": "v2_multisource_query_plan_v1", "selected_sources": selected, "decisions": decisions, "coverage_claim": "DECLARED_ELIGIBLE_SOURCES_ONLY", "limitations": "Planning uses declared local source contracts only; it does not query or inflate coverage."}
