"""Conservative modality and sample-unit compatibility contracts for live adapters."""

from __future__ import annotations

from .ontology import normalize_entity
from .schemas import JsonDict

MODALITY_FAMILIES = {
    "bulk_transcriptomics": {"rna-seq", "microarray", "transcriptomics"},
    "single_cell_transcriptomics": {"single-cell rna-seq", "scrna-seq"},
    "sequencing_genomics": {"wgs", "whole genome sequencing", "genomics"},
    "clinical_registry": {"clinical study registry metadata", "clinical registry"},
    "microbiome_metagenomics": {"metagenomics", "shotgun metagenomics", "16s rrna sequencing"},
    "proteomics": {"proteomics", "mass spectrometry proteomics"},
    "metabolomics": {"metabolomics", "mass spectrometry metabolomics"},
}


def modality_families(value: object) -> set[str]:
    """Resolve only declared assay terms/families and qualified local synonyms."""
    term = str(value).strip().casefold()
    if term in MODALITY_FAMILIES:
        return {term}
    mapping = normalize_entity(term, "assay")
    terms = {term}
    if mapping["status"] == "RESOLVED":
        terms.update(
            assay
            for assays in MODALITY_FAMILIES.values()
            for assay in assays
            if normalize_entity(assay, "assay")["candidates"] == mapping["candidates"]
        )
    return {name for name, values in MODALITY_FAMILIES.items() if terms & values}


def same_organism(expected: object, observed: object) -> bool | None:
    left = normalize_entity(str(expected), "organism")
    right = normalize_entity(str(observed), "organism")
    if left["status"] == right["status"] == "RESOLVED":
        return left["candidates"] == right["candidates"]
    return None


def modality_contract(record: JsonDict) -> JsonDict:
    """Return explicit observed/unknown modality and unit semantics for one record."""

    assays = {str(x).strip().lower() for x in record.get("assay_types", []) if str(x).strip()}
    families = sorted({family for assay in assays for family in modality_families(assay)})
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
    dependence = metadata.get("dependence", {}) if isinstance(metadata.get("dependence"), dict) else {}
    omics = metadata.get("omics_contract", {}) if isinstance(metadata.get("omics_contract"), dict) else {}
    temporal = metadata.get("temporal_contract", {}) if isinstance(metadata.get("temporal_contract"), dict) else {}
    biological_unit = "UNKNOWN"
    capabilities = record.get("capabilities", {})
    if isinstance(capabilities, dict) and dependence.get("donor_links") != "AMBIGUOUS_NOT_INFERRED":
        for field in ("biological_sample", "donor", "independent_unit"):
            observation = capabilities.get(field, {})
            if (isinstance(observation, dict) and observation.get("status") == "observed"
                    and observation.get("value") is not None and observation.get("source_locator")
                    and observation.get("mapping_type", "exact") in {"exact", "synonym"}):
                biological_unit = "OBSERVED"
    return {
        "modality": families or ["UNKNOWN"],
        "observed_assays": sorted(assays),
        "organism": "OBSERVED" if record.get("organisms") else "UNKNOWN",
        # Preserve the observed values so downstream eligibility can reject an
        # explicit, incompatible organism without inventing a synonym mapping.
        "organisms": [str(item) for item in record.get("organisms", []) if str(item).strip()],
        "specimen": "OBSERVED" if metadata.get("specimen") or metadata.get("biome") else "UNKNOWN",
        "biological_unit": biological_unit,
        "technical_units": int(dependence.get("technical_run_count", 0) or 0),
        "feature_type": str(omics.get("feature_type", "UNKNOWN") or "UNKNOWN"),
        "feature_unit": str(omics.get("feature_unit", "UNKNOWN") or "UNKNOWN"),
        "quantification": str(omics.get("quantification", "UNKNOWN") or "UNKNOWN"),
        "normalization": str(omics.get("normalization", "UNKNOWN") or "UNKNOWN"),
        "metadata_availability": str(omics.get("metadata_availability", "UNKNOWN") or "UNKNOWN"),
        "temporal_design": str(temporal.get("design", "UNKNOWN") or "UNKNOWN"),
        "baseline_timing": str(temporal.get("baseline_timing", "UNKNOWN") or "UNKNOWN"),
        "followup_window": str(temporal.get("followup_window", "UNKNOWN") or "UNKNOWN"),
        "intervention_timing": str(temporal.get("intervention_timing", "UNKNOWN") or "UNKNOWN"),
        "repeated_measure_unit": str(temporal.get("repeated_measure_unit", "UNKNOWN") or "UNKNOWN"),
        "access": str(record.get("access_type", "unknown") or "unknown"),
    }


def compatibility(required_modality: str, required_organism: str, record: JsonDict) -> str:
    """Return INCOMPATIBLE, PARTIAL, or UNKNOWN without treating absent metadata as failure."""

    contract = modality_contract(record)
    modalities = set(contract["modality"])
    required_families = modality_families(required_modality)
    if required_families and modalities != {"UNKNOWN"} and not required_families & modalities:
        return "INCOMPATIBLE"
    organisms = {str(x).lower() for x in record.get("organisms", [])}
    if required_organism and organisms:
        comparisons = [same_organism(required_organism, value) for value in organisms]
        if all(value is False for value in comparisons):
            return "INCOMPATIBLE"
        if not any(value is True for value in comparisons):
            return "UNKNOWN"
    if not required_families or contract["modality"] == ["UNKNOWN"] or contract["organism"] == "UNKNOWN":
        return "UNKNOWN"
    return "PARTIAL"
