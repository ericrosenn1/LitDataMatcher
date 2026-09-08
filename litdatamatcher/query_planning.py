"""Deterministic fail-closed source planning from declared adapter contracts."""
from __future__ import annotations


def plan_sources(requirements: dict, sources: list[dict]) -> dict:
    selected, decisions = [], []
    required_modality = requirements.get("modality")
    complete = bool(requirements.get("complete_candidate_universe", False))
    offline = bool(requirements.get("offline_required", False))
    required_access = str(requirements.get("required_access_class", "") or "")
    for source in sources:
        name = str(source.get("name", "UNKNOWN"))
        reasons = []
        status = "ELIGIBLE"
        if not source.get("supported", False): status, reasons = "NOT_QUALIFIED", ["unsupported_route"]
        elif required_modality and required_modality not in source.get("modalities", []): status, reasons = "NOT_QUALIFIED", ["explicit_modality_mismatch"]
        elif _access_status(source)["status"] != "OPEN": status, reasons = ("NOT_QUALIFIED" if required_access else "UNKNOWN"), [_access_status(source)["reason"]]
        elif required_access and source.get("access_class") != required_access: status, reasons = "NOT_QUALIFIED", ["unsupported_access_class"]
        elif source.get("metadata_completeness") != "OBSERVED": status, reasons = "UNKNOWN", ["metadata_completeness_unknown"]
        elif offline and not source.get("offline_cache_available", False): status, reasons = "UNKNOWN", ["offline_cache_unavailable"]
        elif complete and source.get("candidate_universe_status") != "COMPLETE_CANDIDATE_UNIVERSE": status, reasons = "NOT_QUALIFIED", ["partial_candidate_universe"]
        if status == "ELIGIBLE": selected.append(name)
        decisions.append({"source": name, "status": status, "reasons": reasons, "query_permitted": status == "ELIGIBLE"})
    return {"schema_version": "v2_multisource_query_plan_v1", "selected_sources": selected, "decisions": decisions, "coverage_claim": "DECLARED_ELIGIBLE_SOURCES_ONLY", "limitations": "Planning uses declared local source contracts only; it does not query or inflate coverage."}


def _access_status(source: dict) -> dict:
    """Classify declared source terms without inferring open rights from absence."""
    access = str(source.get("access", "unknown") or "unknown").casefold()
    license_value = str(source.get("license", "") or "").strip()
    context = str(source.get("terms_context", "") or "").strip()
    if access in {"restricted", "embargoed"}: return {"status":"RESTRICTED","reason":f"{access}_access"}
    if not license_value or not context or access == "unknown": return {"status":"UNKNOWN","reason":"access_or_license_unavailable"}
    if source.get("license_conflict", False): return {"status":"UNKNOWN","reason":"conflicting_license_declarations"}
    return {"status":"OPEN","reason":"declared_open_access"}
