"""LitDataMatcher core package.

The package contains the reusable, testable parts of the project: validated
schemas, literature-question extraction, dataset cataloging, evidence synthesis,
ranking, storage, and CLI orchestration. The legacy top-level worker scripts are
kept as thin executable entrypoints around these modules.
"""

from .schemas import (
    DatasetRecord,
    DatasetVariable,
    Evidence,
    EvidenceSynthesis,
    MatchCandidate,
    MatchScore,
    QuestionCandidate,
    stable_id,
)
from .ontology import normalize_variable_name

__all__ = [
    "DatasetRecord",
    "DatasetVariable",
    "Evidence",
    "EvidenceSynthesis",
    "MatchCandidate",
    "MatchScore",
    "QuestionCandidate",
    "stable_id",
    "normalize_variable_name",
]

__version__ = "0.1.0"
