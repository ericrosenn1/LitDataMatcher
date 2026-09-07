"""Conservative modality and sample-unit compatibility contracts for live adapters."""

from __future__ import annotations

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


def modality_contract(record: JsonDict) -> JsonDict:
    """Return explicit observed/unknown modality and unit semantics for one record."""

    assays = {str(x).strip().lower() for x in record.get("assay_types", []) if str(x).strip()}
    families = sorted(name for name, values in MODALITY_FAMILIES.items() if assays & values)
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
    dependence = metadata.get("dependence", {}) if isinstance(metadata.get("dependence"), dict) else {}
    return {
        "modality": families or ["UNKNOWN"],
        "organism": "OBSERVED" if record.get("organisms") else "UNKNOWN",
        "specimen": "OBSERVED" if metadata.get("specimen") or metadata.get("biome") else "UNKNOWN",
        "biological_unit": "UNKNOWN" if dependence.get("donor_links") == "AMBIGUOUS_NOT_INFERRED" else "OBSERVED",
        "technical_units": int(dependence.get("technical_run_count", 0) or 0),
        "access": str(record.get("access_type", "unknown") or "unknown"),
    }


def compatibility(required_modality: str, required_organism: str, record: JsonDict) -> str:
    """Return INCOMPATIBLE, PARTIAL, or UNKNOWN without treating absent metadata as failure."""

    contract = modality_contract(record)
    modalities = set(contract["modality"])
    if required_modality and modalities != {"UNKNOWN"} and required_modality not in modalities:
        return "INCOMPATIBLE"
    organisms = {str(x).lower() for x in record.get("organisms", [])}
    if required_organism and organisms and required_organism.lower() not in organisms:
        return "INCOMPATIBLE"
    if contract["modality"] == ["UNKNOWN"] or contract["organism"] == "UNKNOWN":
        return "UNKNOWN"
    return "PARTIAL"
