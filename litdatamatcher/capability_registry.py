"""Dataset capability and derived-variable registry helpers."""

from __future__ import annotations

from collections.abc import Iterable

from .ontology import normalize_token, normalize_variable_name
from .provenance import source_profile
from .schemas import DatasetCapability, DatasetRecord, DerivedVariableRule, JsonDict

CAPABILITY_VOCABULARY: dict[str, JsonDict] = {
    "sample_size": {
        "category": "study_design_feature",
        "aliases": ["cohort_size", "number_of_samples"],
        "interpretation": "Study-level sample or participant count metadata.",
    },
    "cohort_size": {
        "category": "study_design_feature",
        "aliases": ["sample_size"],
        "interpretation": "Study-level cohort or sample count metadata.",
    },
    "class_label": {
        "category": "supervised_learning_label",
        "aliases": ["label", "outcome_label", "case_control_status", "study_arm"],
        "interpretation": "Potential supervised-learning label metadata, not a fitted model output.",
    },
    "label": {
        "category": "supervised_learning_label",
        "aliases": ["class_label", "outcome_label"],
        "interpretation": "Generic label metadata that requires source-specific inspection.",
    },
    "outcome_label": {
        "category": "evaluation_outcome",
        "aliases": ["endpoint", "response", "label"],
        "interpretation": "Outcome or endpoint field that may support supervised labels.",
    },
    "predictor": {
        "category": "derived_or_proxy_capability",
        "aliases": ["feature", "covariate"],
        "interpretation": "Potential model input feature after source-specific processing.",
    },
    "feature": {
        "category": "derived_or_proxy_capability",
        "aliases": ["predictor"],
        "interpretation": "Potential model input feature, not evidence that a matrix is analysis-ready.",
    },
    "covariate": {
        "category": "metadata_variable",
        "aliases": ["metadata", "adjustment_variable"],
        "interpretation": "Metadata that may support adjustment or stratification.",
    },
    "timepoint": {
        "category": "temporal_structure",
        "aliases": ["visit", "longitudinal_time"],
        "interpretation": "Repeated-measure or visit timing metadata.",
    },
    "longitudinal_time": {
        "category": "temporal_structure",
        "aliases": ["timepoint", "follow_up"],
        "interpretation": "Longitudinal timing structure or follow-up metadata.",
    },
    "validation_cohort": {
        "category": "derived_or_proxy_capability",
        "aliases": ["held_out_cohort", "validation_set"],
        "interpretation": "Possible validation partition; not assumed unless explicitly present.",
    },
    "prediction_performance": {
        "category": "evaluation_outcome",
        "aliases": ["auc", "accuracy", "model_performance"],
        "interpretation": "Performance metric metadata or derivable target; not a computed result.",
    },
}


DEFAULT_DERIVED_RULES: tuple[DerivedVariableRule, ...] = (
    DerivedVariableRule(
        output_variable="body_mass_index",
        input_variables=["height", "weight"],
        expression="weight_kg / height_m^2",
        description="BMI can be derived when height and weight are present.",
        assumptions=["height and weight use compatible units"],
        confidence=0.9,
    ),
    DerivedVariableRule(
        output_variable="treatment_response",
        input_variables=["treatment", "outcome"],
        expression="outcome grouped by treatment exposure",
        description="Treatment-response labels can often be derived from treatment and outcome fields.",
        assumptions=["outcome timing is compatible with treatment exposure"],
        confidence=0.7,
    ),
    DerivedVariableRule(
        output_variable="survival_outcome",
        input_variables=["outcome", "timepoint"],
        expression="time-to-event or status over follow-up",
        description="Survival-style endpoints may be derivable from outcome and follow-up fields.",
        assumptions=["outcome encodes event/status and timepoint encodes follow-up timing"],
        confidence=0.55,
    ),
    DerivedVariableRule(
        output_variable="longitudinal_change",
        input_variables=["timepoint", "outcome"],
        expression="follow-up outcome minus baseline outcome",
        description="Longitudinal change can be derived when outcome and repeated timepoints are present.",
        assumptions=["baseline and follow-up measurements are comparable"],
        confidence=0.65,
    ),
)

DERIVED_OUTPUT_NAMES = {
    normalize_token(rule.output_variable) for rule in DEFAULT_DERIVED_RULES
}

CAPABILITY_EVIDENCE_POLICY = "source-evidence-v2"


def _variable_evidence_status(variable) -> str:
    """Separate source-reported metadata from absent or text-inferred fields."""
    if variable.completeness <= 0 or variable.category == "absent":
        return "ABSENT"
    if variable.category in {"inferred", "derived", "proxy", "derived_or_proxy_capability"}:
        return "INFERRED"
    if variable.category == "unknown":
        return "UNKNOWN"
    return "OBSERVED_METADATA"


def infer_dataset_capabilities(
    dataset: DatasetRecord,
    derived_rules: Iterable[DerivedVariableRule] = DEFAULT_DERIVED_RULES,
) -> list[DatasetCapability]:
    """Infer observed and plausibly derived capabilities for one dataset."""

    observed = _observed_variable_map(dataset)
    capabilities: list[DatasetCapability] = []
    for variable in dataset.variables:
        normalized = normalize_variable_name(variable.name)
        vocabulary = CAPABILITY_VOCABULARY.get(normalized, {})
        evidence_status = _variable_evidence_status(variable)
        observed_metadata = evidence_status == "OBSERVED_METADATA"
        capabilities.append(
            DatasetCapability(
                dataset_id=dataset.dataset_id,
                variable_name=normalized,
                capability_type="observed" if observed_metadata else "unknown",
                source_variable_names=[variable.name],
                confidence=variable.completeness if evidence_status != "ABSENT" else 0.0,
                evidence=(f"Observed variable in dataset metadata: {variable.name}" if observed_metadata
                          else f"{evidence_status.lower()} variable; source-level presence is not established: {variable.name}"),
                limitations=(["Metadata presence does not establish measured, analysis-ready values."]
                             if observed_metadata else ["Not an observed capability or a derivation prerequisite."])
                            + ([] if variable.completeness >= 0.7 else ["variable completeness may be limited"]),
                metadata={
                    "category": variable.category,
                    "capability_category": vocabulary.get("category", variable.category),
                    "capability_interpretation": vocabulary.get("interpretation", ""),
                    "observed_count": variable.observed_count,
                    "completeness": variable.completeness,
                    "evidence_status": evidence_status,
                    "evidence_policy": CAPABILITY_EVIDENCE_POLICY,
                    "source_profile": source_profile("capability_registry"),
                    "interpretation": "Observed capability from dataset metadata; not a completed downstream analysis.",
                },
            )
        )

    for rule in derived_rules:
        required = [normalize_variable_name(value) for value in rule.input_variables]
        if not all(item in observed for item in required):
            continue
        source_names = [observed[item] for item in required]
        capabilities.append(
            DatasetCapability(
                dataset_id=dataset.dataset_id,
                variable_name=_capability_variable_name(rule.output_variable),
                capability_type="derived",
                source_variable_names=source_names,
                derivation_rule_id=rule.rule_id,
                confidence=rule.confidence,
                evidence=rule.description,
                limitations=rule.assumptions,
                metadata={
                    "expression": rule.expression,
                    "source_profile": source_profile("capability_registry"),
                    "interpretation": "Derived capability is plausible from available fields; it has not been computed.",
                },
            )
        )
    return _dedupe_capabilities(capabilities)


def capability_summary(capabilities: Iterable[DatasetCapability]) -> JsonDict:
    """Summarize capabilities for manifests, reports, or CLI output."""

    capabilities = list(capabilities)
    by_type: dict[str, int] = {}
    variables: list[str] = []
    for capability in capabilities:
        by_type[capability.capability_type] = by_type.get(capability.capability_type, 0) + 1
        variables.append(capability.variable_name)
    return {
        "capabilities": len(capabilities),
        "capabilities_by_type": dict(sorted(by_type.items())),
        "variables": sorted(set(variables)),
        "vocabulary_terms": sorted(CAPABILITY_VOCABULARY),
    }


def _observed_variable_map(dataset: DatasetRecord) -> dict[str, str]:
    """Map normalized observed variables to source metadata names."""

    observed: dict[str, str] = {}
    for variable in dataset.variables:
        if _variable_evidence_status(variable) != "OBSERVED_METADATA":
            continue
        normalized = normalize_variable_name(variable.name)
        observed.setdefault(normalized, variable.name)
        for synonym in variable.synonyms:
            observed.setdefault(normalize_variable_name(synonym), variable.name)
    return observed


def _capability_variable_name(value: str) -> str:
    """Preserve derived capability names that are broader than ontology concepts."""

    normalized = normalize_token(value)
    if normalized in DERIVED_OUTPUT_NAMES:
        return normalized
    return normalize_variable_name(value)


def _dedupe_capabilities(capabilities: list[DatasetCapability]) -> list[DatasetCapability]:
    """Deduplicate capabilities by dataset, variable, and capability type."""

    out: list[DatasetCapability] = []
    seen: set[tuple[str, str, str]] = set()
    for capability in capabilities:
        key = (capability.dataset_id, capability.variable_name, capability.capability_type)
        if key in seen:
            continue
        seen.add(key)
        out.append(capability)
    return out
