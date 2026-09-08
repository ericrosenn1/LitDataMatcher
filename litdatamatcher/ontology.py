"""Domain ontology and variable harmonization utilities.

The ontology in this module is deliberately lightweight and auditable. It is not
intended to replace UMLS, MeSH, EFO, MONDO, or OBI. Instead, it provides a stable
local concept layer so early pipeline runs can normalize variable names, report
why two records matched, and later swap in larger external ontologies.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True, slots=True)
class OntologyConcept:
    """A canonical concept used for variable and metadata matching."""

    concept_id: str
    label: str
    category: str
    synonyms: tuple[str, ...]
    description: str = ""

    def all_terms(self) -> tuple[str, ...]:
        """Return the label plus synonyms as lowercase terms."""

        return (self.label.lower(), *(term.lower() for term in self.synonyms))


CONCEPTS: tuple[OntologyConcept, ...] = (
    # Concept labels are used downstream as stable variable names.
    OntologyConcept(
        "LDM:AGE",
        "age",
        "demographic",
        ("adult", "infant", "newborn", "neonate", "pediatric", "paediatric", "elderly"),
        "Participant age or age stratum.",
    ),
    OntologyConcept(
        "LDM:SEX",
        "sex",
        "demographic",
        ("gender", "male", "female", "biological sex"),
        "Participant sex or gender metadata.",
    ),
    OntologyConcept(
        "LDM:ANTIBIOTIC_EXPOSURE",
        "antibiotic_exposure",
        "exposure",
        ("antibiotic", "antibiotics", "antimicrobial", "antimicrobial exposure"),
        "Antibiotic or antimicrobial exposure history.",
    ),
    OntologyConcept(
        "LDM:BODY_SITE",
        "body_site",
        "specimen",
        ("sample site", "body site", "stool", "fecal", "faecal", "gut", "intestinal", "mucosal"),
        "Sample body site or biospecimen location.",
    ),
    OntologyConcept(
        "LDM:MICROBIOME_COMPOSITION",
        "microbiome_composition",
        "omics",
        ("microbiome", "microbiota", "16s", "16s rrna", "metagenomic", "metagenomics", "taxonomic profile"),
        "Microbial community composition or metagenomic profile.",
    ),
    OntologyConcept(
        "LDM:METABOLOMICS",
        "metabolomics",
        "omics",
        ("metabolomic", "metabolite", "metabolites", "lc-ms", "gc-ms"),
        "Metabolite abundance or metabolomics assay data.",
    ),
    OntologyConcept(
        "LDM:TRANSCRIPTOMICS",
        "transcriptomics",
        "omics",
        ("transcriptomic", "rna-seq", "rnaseq", "gene expression", "microarray"),
        "Transcript abundance or host gene-expression data.",
    ),
    OntologyConcept(
        "LDM:DIET",
        "diet",
        "exposure",
        ("dietary", "nutrition", "nutrient", "food frequency", "caloric intake"),
        "Dietary or nutrition exposure metadata.",
    ),
    OntologyConcept(
        "LDM:TIMEPOINT",
        "timepoint",
        "study_design",
        ("longitudinal", "time point", "timepoint", "visit", "follow-up", "baseline"),
        "Longitudinal sampling or visit metadata.",
    ),
    OntologyConcept(
        "LDM:DISEASE_ACTIVITY",
        "disease_activity",
        "phenotype",
        ("ibd", "crohn", "crohn's", "ulcerative colitis", "colitis", "inflammation", "remission", "relapse"),
        "Disease status, activity, severity, remission, or relapse phenotype.",
    ),
    OntologyConcept(
        "LDM:TREATMENT",
        "treatment",
        "clinical",
        ("therapy", "drug", "intervention", "medication", "biologic", "steroid"),
        "Therapeutic intervention or treatment metadata.",
    ),
    OntologyConcept(
        "LDM:OUTCOME",
        "outcome",
        "clinical",
        ("endpoint", "response", "recovery", "remission", "relapse", "survival"),
        "Clinical, biological, or patient-centered outcome.",
    ),
)

CANONICAL_BY_LABEL = {concept.label: concept for concept in CONCEPTS}
TERM_TO_LABEL = {
    # Synonym lookup keeps raw metadata comparable to canonical labels.
    term: concept.label
    for concept in CONCEPTS
    for term in concept.all_terms()
}

# Deliberately small local contracts.  These are fixtures/interfaces, not a
# substitute for live ontology services or a claim of comprehensive coverage.
ENTITY_CONTRACTS = {
    "gene_protein": {"tp53": ("HGNC:11998", "exact"), "p53": ("HGNC:11998", "synonym"), "il6": ("HGNC:6018", "exact"), "cd3": (("CD3D", "CD3E"), "ambiguous")},
    "disease_condition": {"crohn disease": ("MONDO:0005011", "exact"), "ibd": (("MONDO:0005011", "MONDO:0005101"), "ambiguous")},
    "intervention_chemical": {"lipopolysaccharide": ("CHEBI:16412", "exact"), "lps": ("CHEBI:16412", "synonym")},
    "tissue_cell_type": {"macrophage": ("CL:0000235", "exact"), "gut": (("UBERON:0001155", "UBERON:0002107"), "ambiguous")},
    "organism": {"homo sapiens": ("NCBITaxon:9606", "exact"), "human": ("NCBITaxon:9606", "synonym"), "mus musculus": ("NCBITaxon:10090", "exact"), "mouse": ("NCBITaxon:10090", "synonym")},
    "assay": {"rna-seq": ("EFO:0002772", "exact"), "rna sequencing": ("EFO:0002772", "synonym")},
    "experimental_condition": {"lps": ("CHEBI:16412", "synonym"), "untreated": ("LDM:UNTREATED_CONTROL", "exact")},
}
DEPRECATED_ENTITY_IDS = {"HGNC:OLDTP53": "HGNC:11998"}


def normalize_entity(value: str, category: str, *, source_available: bool = True) -> dict[str, object]:
    """Return an explicit local identifier mapping; ambiguity never picks a winner."""
    if not source_available:
        return {"status": "SOURCE_UNAVAILABLE", "mapping_type": "unresolved", "source": "local_contract_v1", "candidates": []}
    raw = str(value or "").strip()
    if raw in DEPRECATED_ENTITY_IDS:
        return {"status": "DEPRECATED", "mapping_type": "unresolved", "source": "local_contract_v1", "candidates": [DEPRECATED_ENTITY_IDS[raw]], "deprecated_id": raw}
    entry = ENTITY_CONTRACTS.get(category, {}).get(raw.casefold())
    if not entry:
        return {"status": "UNRESOLVED", "mapping_type": "unresolved", "source": "local_contract_v1", "candidates": []}
    candidate, mapping_type = entry
    candidates = list(candidate) if isinstance(candidate, tuple) else [candidate]
    return {"status": "AMBIGUOUS" if mapping_type == "ambiguous" else "RESOLVED", "mapping_type": mapping_type, "source": "local_contract_v1", "candidates": candidates}


def normalize_token(value: str) -> str:
    """Normalize a concept or metadata token for matching."""

    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_variable_name(value: str) -> str:
    """Map a raw variable name or synonym to a canonical concept label."""

    raw = str(value or "").strip().lower()
    normalized = normalize_token(raw)
    if normalized in CANONICAL_BY_LABEL:
        return normalized
    if raw in TERM_TO_LABEL:
        return TERM_TO_LABEL[raw]
    spaced = normalized.replace("_", " ")
    if spaced in TERM_TO_LABEL:
        return TERM_TO_LABEL[spaced]
    for term, label in TERM_TO_LABEL.items():
        # Substring matching catches noisy field names but should stay well tested.
        if term in raw or normalize_token(term) in normalized:
            return label
    return normalized


def concept_for_variable(value: str) -> OntologyConcept | None:
    """Return the ontology concept for a variable if one is known."""

    return CANONICAL_BY_LABEL.get(normalize_variable_name(value))


def infer_concepts_from_text(text: str) -> list[str]:
    """Infer canonical concept labels from free text."""

    lowered = str(text or "").lower()
    hits: list[str] = []
    for concept in CONCEPTS:
        # Text inference is recall-oriented; later nodes handle feasibility and caveats.
        if any(term in lowered for term in concept.all_terms()):
            hits.append(concept.label)
    return hits


def explain_variable_match(required: list[str], available: set[str]) -> dict[str, object]:
    """Return coverage and traceable missing/present variable information."""

    required_norm = [normalize_variable_name(item) for item in required]
    available_norm = {normalize_variable_name(item) for item in available}
    present = sorted(set(required_norm) & available_norm)
    missing = sorted(set(required_norm) - available_norm)
    # Questions without explicit variables receive a neutral low-information score.
    coverage = len(present) / max(1, len(set(required_norm))) if required_norm else 0.45
    return {
        "coverage": round(coverage, 3),
        "present": present,
        "missing": missing,
        "required_normalized": sorted(set(required_norm)),
    }


def concept_table() -> list[dict[str, str]]:
    """Return ontology concepts as dictionaries for docs or reports."""

    return [
        {
            "concept_id": concept.concept_id,
            "label": concept.label,
            "category": concept.category,
            "synonyms": "; ".join(concept.synonyms),
            "description": concept.description,
        }
        for concept in CONCEPTS
    ]
