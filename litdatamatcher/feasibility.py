"""Design and data-feasibility assessment for ranked opportunities."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math

from .governance import assess_governance
from .ontology import explain_variable_match
from .schemas import DatasetRecord, QuestionCandidate


@dataclass(slots=True)
class FeasibilityAssessment:
    """Interpretability payload for a question-dataset pair."""

    variable_coverage: float
    population_fit: float
    sample_adequacy: float
    longitudinal_fit: float
    assay_fit: float
    governance_reuse: float
    overall: float
    recommended_design: str
    present_variables: list[str]
    missing_variables: list[str]
    caveats: list[str]

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""

        return asdict(self)


def sample_adequacy(sample_size: int, minimum: int = 200, target: int = 2000) -> float:
    """Return a smooth 0..1 sample adequacy score."""

    sample_size = max(0, int(sample_size or 0))
    if sample_size == 0:
        return 0.0
    if sample_size <= minimum:
        return round(0.2 * sample_size / minimum, 3)
    return round(min(1.0, math.log(sample_size / minimum + 1) / math.log(target / minimum + 1)), 3)


def population_fit(question: QuestionCandidate, dataset: DatasetRecord) -> float:
    """Score coarse population compatibility."""

    requested = question.population.lower().strip()
    available = {item.lower() for item in dataset.populations}
    if not requested:
        return 0.55
    if requested in available:
        return 1.0
    if requested == "human" and {"adult", "pediatric", "infant"} & available:
        return 0.8
    if requested in {"adult", "pediatric", "infant"} and "human" in available:
        return 0.75
    return 0.25 if available else 0.4


def assay_fit(question: QuestionCandidate, dataset: DatasetRecord) -> float:
    """Score whether dataset assays plausibly measure required data types."""

    required = {item.lower() for item in question.required_variables}
    assays = " ".join(dataset.assay_types).lower()
    if "microbiome_composition" in required and any(term in assays for term in ("16s", "metagenomic")):
        return 1.0
    if "transcriptomics" in required and any(term in assays for term in ("rna", "microarray")):
        return 1.0
    if "metabolomics" in required and any(term in assays for term in ("ms", "metabol")):
        return 1.0
    return 0.65 if assays else 0.45


def recommended_design(question: QuestionCandidate, dataset: DatasetRecord) -> str:
    """Return a concise suggested analysis design."""

    text = f"{question.question} {' '.join(question.required_variables)}".lower()
    if "timepoint" in question.required_variables or "longitudinal" in text:
        return "longitudinal mixed-effects model or time-to-event analysis"
    if "treatment" in question.required_variables:
        return "propensity-adjusted observational treatment-response analysis"
    if "microbiome_composition" in question.required_variables:
        return "multivariable association model with compositional-data sensitivity analysis"
    return "matched observational analysis with covariate adjustment"


def assess_pair_feasibility(
    question: QuestionCandidate, dataset: DatasetRecord
) -> FeasibilityAssessment:
    """Assess feasibility and produce interpretable caveats."""

    variable_info = explain_variable_match(question.required_variables, dataset.variable_aliases())
    pop_fit = population_fit(question, dataset)
    sample_fit = sample_adequacy(dataset.sample_size)
    longitudinal = 1.0 if "timepoint" in dataset.variable_aliases() else 0.45
    if "timepoint" not in [item.lower() for item in question.required_variables]:
        longitudinal = max(longitudinal, 0.65)
    assay = assay_fit(question, dataset)
    governance = assess_governance(dataset)
    caveats = list(governance.risk_flags)
    if variable_info["missing"]:
        caveats.append("missing_required_variables")
    if sample_fit < 0.5:
        caveats.append("limited_sample_size")
    if pop_fit < 0.5:
        caveats.append("population_mismatch")

    overall = round(
        0.28 * float(variable_info["coverage"])
        + 0.18 * pop_fit
        + 0.18 * sample_fit
        + 0.12 * longitudinal
        + 0.12 * assay
        + 0.12 * governance.reuse_score,
        3,
    )
    return FeasibilityAssessment(
        variable_coverage=float(variable_info["coverage"]),
        population_fit=round(pop_fit, 3),
        sample_adequacy=round(sample_fit, 3),
        longitudinal_fit=round(longitudinal, 3),
        assay_fit=round(assay, 3),
        governance_reuse=governance.reuse_score,
        overall=overall,
        recommended_design=recommended_design(question, dataset),
        present_variables=list(variable_info["present"]),
        missing_variables=list(variable_info["missing"]),
        caveats=sorted(set(caveats)),
    )
