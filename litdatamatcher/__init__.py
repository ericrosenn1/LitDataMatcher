"""LitDataMatcher core package.

The package contains the reusable, testable parts of the project: validated
schemas, literature-question extraction, dataset cataloging, evidence synthesis,
ranking, storage, and CLI orchestration. The legacy top-level worker scripts are
kept as thin executable entrypoints around these modules.
"""

from .schemas import (
    DatasetRecord,
    DatasetVariable,
    DatasetCapability,
    DerivedCapabilityLabel,
    DerivedVariableRule,
    Evidence,
    EvidenceSynthesis,
    EvidenceSpanLabel,
    ExpertPaperAnnotation,
    MatchCandidate,
    MatchScore,
    QuestionCandidate,
    QuestionDataMatchLabel,
    QuestionLabel,
    QuestionQualityScore,
    stable_id,
)
from .capability_registry import DEFAULT_DERIVED_RULES, capability_summary, infer_dataset_capabilities
from .ontology import normalize_variable_name

__all__ = [
    "DEFAULT_DERIVED_RULES",
    "DatasetRecord",
    "DatasetVariable",
    "DatasetCapability",
    "DerivedCapabilityLabel",
    "DerivedVariableRule",
    "Evidence",
    "EvidenceSynthesis",
    "EvidenceSpanLabel",
    "ExpertPaperAnnotation",
    "MatchCandidate",
    "MatchScore",
    "QuestionCandidate",
    "QuestionDataMatchLabel",
    "QuestionLabel",
    "QuestionQualityScore",
    "capability_summary",
    "infer_dataset_capabilities",
    "stable_id",
    "normalize_variable_name",
]

__version__ = "0.1.0"
