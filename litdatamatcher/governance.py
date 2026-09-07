"""Dataset governance, access, and reuse-risk scoring."""

from __future__ import annotations

from dataclasses import dataclass, asdict

from .schemas import DatasetRecord


@dataclass(slots=True)
class GovernanceAssessment:
    """Structured governance assessment for a dataset."""

    access_score: float
    license_score: float
    privacy_score: float
    reuse_score: float
    risk_flags: list[str]

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""

        return asdict(self)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    """Return true when any term appears in text."""

    lowered = text.lower()
    return any(term in lowered for term in terms)


def assess_governance(dataset: DatasetRecord) -> GovernanceAssessment:
    """Score likely governance readiness for downstream reuse.

    The score is intentionally conservative: unclear access, restrictive terms,
    or human-subject indicators lower confidence and produce explicit flags.
    """

    access = f"{dataset.access_type} {dataset.license}".lower()
    text = f"{dataset.title} {dataset.description} {' '.join(dataset.populations)}".lower()
    flags: list[str] = []

    # Access terms estimate whether a researcher can realistically obtain the data.
    if _contains_any(access, ("public", "open")):
        access_score = 0.9
    elif _contains_any(access, ("controlled", "restricted", "application", "dbgap")):
        access_score = 0.35
        flags.append("controlled_or_restricted_access")
    else:
        access_score = 0.55
        flags.append("unclear_access_terms")

    # License terms stay separate from access because open metadata may still limit reuse.
    if _contains_any(access, ("public domain", "cc0", "open metadata", "public repository")):
        license_score = 0.9
    elif _contains_any(access, ("varies", "study-specific", "unknown")):
        license_score = 0.55
        flags.append("license_requires_manual_review")
    else:
        license_score = 0.7

    human = _contains_any(text, ("human", "patient", "participant", "clinical", "ibd"))
    controlled = "controlled_or_restricted_access" in flags
    # Human-subjects signals lower confidence unless reuse terms are clearly permissive.
    if human and controlled:
        privacy_score = 0.35
        flags.append("human_subjects_controlled_data")
    elif human:
        privacy_score = 0.65
        flags.append("human_subjects_review_needed")
    else:
        privacy_score = 0.9

    # The aggregate score remains interpretable because each component is exported.
    reuse_score = round(0.35 * access_score + 0.3 * license_score + 0.35 * privacy_score, 3)
    return GovernanceAssessment(
        access_score=round(access_score, 3),
        license_score=round(license_score, 3),
        privacy_score=round(privacy_score, 3),
        reuse_score=reuse_score,
        risk_flags=sorted(set(flags)),
    )
