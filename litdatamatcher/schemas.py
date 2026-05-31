"""Validated data contracts shared by all LitDataMatcher nodes.

These dataclasses intentionally use only the Python standard library so the
core pipeline remains runnable in constrained or offline environments. The
validation in ``__post_init__`` catches malformed records early while keeping
serialization transparent and publication-friendly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field as dataclass_field
from hashlib import sha1
from typing import Any


JsonDict = dict[str, Any]


def stable_id(prefix: str, *parts: object) -> str:
    """Return a deterministic identifier for a node output.

    Stable IDs make runs reproducible: the same input content produces the same
    question, dataset, synthesis, or match identifier across machines.
    """

    payload = "||".join(str(part or "").strip().lower() for part in parts)
    digest = sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _clamp01(value: float, field_name: str) -> float:
    """Validate and clamp a score into the inclusive 0..1 interval."""

    try:
        val = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric, got {value!r}") from exc
    return max(0.0, min(1.0, val))


def _clean_list(values: list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize a list of strings while preserving insertion order."""

    if not values:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value or "").strip()
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out


@dataclass(slots=True)
class Evidence:
    """A traceable text span or metadata record supporting an inference."""

    text: str
    source_id: str = ""
    title: str = ""
    doi: str = ""
    section: str = ""
    sentence_index: int = -1
    extraction_method: str = "rule"
    confidence: float = 0.5

    def __post_init__(self) -> None:
        self.text = " ".join(str(self.text or "").split())
        if not self.text:
            raise ValueError("Evidence text cannot be empty.")
        self.confidence = _clamp01(self.confidence, "Evidence.confidence")
        if self.sentence_index < -1:
            raise ValueError("Evidence.sentence_index must be -1 or greater.")

    def to_dict(self) -> JsonDict:
        """Serialize the evidence record to a JSON-compatible dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Evidence":
        """Deserialize an evidence record from a dictionary."""

        return cls(**data)


@dataclass(slots=True)
class DatasetVariable:
    """A normalized variable observed in a dataset."""

    name: str
    category: str = "unspecified"
    observed_count: int = 0
    completeness: float = 1.0
    synonyms: list[str] = dataclass_field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = str(self.name or "").strip()
        if not self.name:
            raise ValueError("DatasetVariable.name cannot be empty.")
        self.category = str(self.category or "unspecified").strip().lower()
        self.observed_count = max(0, int(self.observed_count or 0))
        self.completeness = _clamp01(self.completeness, "DatasetVariable.completeness")
        self.synonyms = _clean_list(self.synonyms)

    @property
    def normalized_name(self) -> str:
        """Canonical lowercase variable name used for matching."""

        from .ontology import normalize_variable_name

        return normalize_variable_name(self.name)

    def aliases(self) -> set[str]:
        """Return all normalized names that should count as this variable."""

        values = {self.normalized_name}
        values.update(s.lower().replace(" ", "_").replace("-", "_") for s in self.synonyms)
        return values

    def to_dict(self) -> JsonDict:
        """Serialize the variable to a JSON-compatible dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "DatasetVariable":
        """Deserialize a variable from a dictionary."""

        return cls(**data)


@dataclass(slots=True)
class DatasetRecord:
    """A normalized public dataset or data repository record."""

    dataset_id: str
    title: str
    source: str
    description: str = ""
    url: str = ""
    variables: list[DatasetVariable] = dataclass_field(default_factory=list)
    populations: list[str] = dataclass_field(default_factory=list)
    organisms: list[str] = dataclass_field(default_factory=list)
    assay_types: list[str] = dataclass_field(default_factory=list)
    sample_size: int = 0
    license: str = "unknown"
    access_type: str = "unknown"
    quality_score: float = 0.5
    metadata: JsonDict = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dataset_id = str(self.dataset_id or "").strip()
        if not self.dataset_id:
            self.dataset_id = stable_id("dataset", self.source, self.title, self.url)
        self.title = str(self.title or "").strip()
        self.source = str(self.source or "").strip()
        if not self.title:
            raise ValueError("DatasetRecord.title cannot be empty.")
        if not self.source:
            raise ValueError("DatasetRecord.source cannot be empty.")
        self.sample_size = max(0, int(self.sample_size or 0))
        self.quality_score = _clamp01(self.quality_score, "DatasetRecord.quality_score")
        self.populations = _clean_list(self.populations)
        self.organisms = _clean_list(self.organisms)
        self.assay_types = _clean_list(self.assay_types)
        self.variables = [
            item if isinstance(item, DatasetVariable) else DatasetVariable.from_dict(item)
            for item in self.variables
        ]

    def variable_aliases(self) -> set[str]:
        """Return normalized variable names and synonyms present in the dataset."""

        aliases: set[str] = set()
        for variable in self.variables:
            aliases.update(variable.aliases())
        return aliases

    def searchable_text(self) -> str:
        """Return text used by lexical and embedding matchers."""

        parts = [
            self.title,
            self.description,
            self.source,
            " ".join(self.populations),
            " ".join(self.organisms),
            " ".join(self.assay_types),
            " ".join(v.name for v in self.variables),
        ]
        return " ".join(part for part in parts if part)

    def to_dict(self) -> JsonDict:
        """Serialize the dataset to a JSON-compatible dictionary."""

        data = asdict(self)
        data["variables"] = [variable.to_dict() for variable in self.variables]
        return data

    @classmethod
    def from_dict(cls, data: JsonDict) -> "DatasetRecord":
        """Deserialize a dataset record from a dictionary."""

        payload = dict(data)
        payload["variables"] = [
            item if isinstance(item, DatasetVariable) else DatasetVariable.from_dict(item)
            for item in payload.get("variables", [])
        ]
        return cls(**payload)


@dataclass(slots=True)
class QuestionCandidate:
    """A normalized open research question extracted from literature."""

    question_id: str
    question: str
    source_ids: list[str] = dataclass_field(default_factory=list)
    evidence: list[Evidence] = dataclass_field(default_factory=list)
    extraction_type: str = "open_question"
    field: str = "biomedical"
    domain_terms: list[str] = dataclass_field(default_factory=list)
    required_variables: list[str] = dataclass_field(default_factory=list)
    population: str = ""
    outcomes: list[str] = dataclass_field(default_factory=list)
    confidence: float = 0.5
    novelty_score: float = 0.5
    significance_score: float = 0.5
    answerability_hint: float = 0.5
    metadata: JsonDict = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.question = " ".join(str(self.question or "").split())
        if not self.question:
            raise ValueError("QuestionCandidate.question cannot be empty.")
        if not self.question_id:
            self.question_id = stable_id("question", self.question)
        self.source_ids = _clean_list(self.source_ids)
        self.domain_terms = _clean_list(self.domain_terms)
        self.required_variables = _clean_list(self.required_variables)
        self.outcomes = _clean_list(self.outcomes)
        self.evidence = [
            item if isinstance(item, Evidence) else Evidence.from_dict(item)
            for item in self.evidence
        ]
        self.confidence = _clamp01(self.confidence, "QuestionCandidate.confidence")
        self.novelty_score = _clamp01(self.novelty_score, "QuestionCandidate.novelty_score")
        self.significance_score = _clamp01(
            self.significance_score, "QuestionCandidate.significance_score"
        )
        self.answerability_hint = _clamp01(
            self.answerability_hint, "QuestionCandidate.answerability_hint"
        )

    @property
    def normalized_question(self) -> str:
        """Lowercase representation used for deduplication."""

        chars = [ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in self.question]
        return " ".join("".join(chars).split())

    def merge(self, other: "QuestionCandidate") -> "QuestionCandidate":
        """Merge another candidate that refers to the same underlying question."""

        if other.confidence > self.confidence:
            self.question = other.question
        self.source_ids = _clean_list([*self.source_ids, *other.source_ids])
        self.evidence = [*self.evidence, *other.evidence]
        self.domain_terms = _clean_list([*self.domain_terms, *other.domain_terms])
        self.required_variables = _clean_list(
            [*self.required_variables, *other.required_variables]
        )
        self.outcomes = _clean_list([*self.outcomes, *other.outcomes])
        self.confidence = max(self.confidence, other.confidence)
        self.novelty_score = max(self.novelty_score, other.novelty_score)
        self.significance_score = max(self.significance_score, other.significance_score)
        self.answerability_hint = max(self.answerability_hint, other.answerability_hint)
        self.metadata = {**other.metadata, **self.metadata}
        return self

    def to_dict(self) -> JsonDict:
        """Serialize the question to a JSON-compatible dictionary."""

        data = asdict(self)
        data["evidence"] = [item.to_dict() for item in self.evidence]
        data["normalized_question"] = self.normalized_question
        return data

    @classmethod
    def from_dict(cls, data: JsonDict) -> "QuestionCandidate":
        """Deserialize a question candidate from a dictionary."""

        payload = dict(data)
        payload.pop("normalized_question", None)
        payload["evidence"] = [
            item if isinstance(item, Evidence) else Evidence.from_dict(item)
            for item in payload.get("evidence", [])
        ]
        return cls(**payload)


@dataclass(slots=True)
class EvidenceSynthesis:
    """A cluster-level assessment of how strongly literature supports a question."""

    cluster_id: str
    question_ids: list[str]
    summary: str
    support_count: int = 0
    contradiction_count: int = 0
    recurrence_score: float = 0.0
    evidence_strength: float = 0.0
    uncertainty: float = 1.0
    metadata: JsonDict = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cluster_id:
            self.cluster_id = stable_id("cluster", *self.question_ids)
        self.question_ids = _clean_list(self.question_ids)
        self.summary = " ".join(str(self.summary or "").split())
        self.support_count = max(0, int(self.support_count or 0))
        self.contradiction_count = max(0, int(self.contradiction_count or 0))
        self.recurrence_score = _clamp01(self.recurrence_score, "EvidenceSynthesis.recurrence_score")
        self.evidence_strength = _clamp01(self.evidence_strength, "EvidenceSynthesis.evidence_strength")
        self.uncertainty = _clamp01(self.uncertainty, "EvidenceSynthesis.uncertainty")

    def to_dict(self) -> JsonDict:
        """Serialize the synthesis to a JSON-compatible dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "EvidenceSynthesis":
        """Deserialize a synthesis record from a dictionary."""

        return cls(**data)


@dataclass(slots=True)
class MatchScore:
    """Explainable component scores for a question-dataset match."""

    variable_overlap: float
    semantic_relevance: float
    population_fit: float
    data_quality: float
    sample_adequacy: float
    significance: float
    feasibility: float
    uncertainty_penalty: float
    combined: float
    governance: float = 0.5
    design_fit: float = 0.5

    def __post_init__(self) -> None:
        for field_name in (
            "variable_overlap",
            "semantic_relevance",
            "population_fit",
            "data_quality",
            "sample_adequacy",
            "significance",
            "feasibility",
            "uncertainty_penalty",
            "combined",
            "governance",
            "design_fit",
        ):
            setattr(self, field_name, _clamp01(getattr(self, field_name), field_name))

    def to_dict(self) -> JsonDict:
        """Serialize the score to a JSON-compatible dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "MatchScore":
        """Deserialize a score record from a dictionary."""

        return cls(**data)


@dataclass(slots=True)
class MatchCandidate:
    """A ranked opportunity linking a literature question to a dataset."""

    match_id: str
    question: QuestionCandidate
    dataset: DatasetRecord
    score: MatchScore
    rationale: list[str] = dataclass_field(default_factory=list)
    missing_variables: list[str] = dataclass_field(default_factory=list)
    assessments: JsonDict = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.question, QuestionCandidate):
            self.question = QuestionCandidate.from_dict(self.question)
        if not isinstance(self.dataset, DatasetRecord):
            self.dataset = DatasetRecord.from_dict(self.dataset)
        if not isinstance(self.score, MatchScore):
            self.score = MatchScore.from_dict(self.score)
        if not self.match_id:
            self.match_id = stable_id("match", self.question.question_id, self.dataset.dataset_id)
        self.rationale = _clean_list(self.rationale)
        self.missing_variables = _clean_list(self.missing_variables)

    def to_dict(self) -> JsonDict:
        """Serialize the match to a JSON-compatible dictionary."""

        return {
            "match_id": self.match_id,
            "question": self.question.to_dict(),
            "dataset": self.dataset.to_dict(),
            "score": self.score.to_dict(),
            "rationale": self.rationale,
            "missing_variables": self.missing_variables,
            "assessments": self.assessments,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "MatchCandidate":
        """Deserialize a match candidate from a dictionary."""

        return cls(
            match_id=data.get("match_id", ""),
            question=QuestionCandidate.from_dict(data["question"]),
            dataset=DatasetRecord.from_dict(data["dataset"]),
            score=MatchScore.from_dict(data["score"]),
            rationale=data.get("rationale", []),
            missing_variables=data.get("missing_variables", []),
            assessments=data.get("assessments", {}),
        )
