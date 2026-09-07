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


# These helpers keep IDs, scores, and list fields consistent across every schema.
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


def _optional_score(value: object, field_name: str, maximum: float = 5.0) -> float | None:
    """Validate an optional numeric label score."""

    if value is None or str(value).strip() == "":
        return None
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric or blank, got {value!r}") from exc
    return max(0.0, min(float(maximum), score))


def _optional_bool(value: object) -> bool | None:
    """Parse optional reviewer booleans from bools, numbers, or text."""

    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse optional boolean value {value!r}.")


def _normalize_question_origin(value: str) -> str:
    """Map older extraction labels onto current question-origin terms."""

    normalized = str(value or "unspecified").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "explicit_research_question": "explicit_question",
        "explicit_rq": "explicit_question",
        "research_question": "explicit_question",
        "open_question": "future_direction",
        "future_directions": "future_direction",
        "future_direction_question": "future_direction",
        "limitation": "limitation_derived",
        "limitation_derived_question": "limitation_derived",
        "": "unspecified",
        "none": "unspecified",
    }
    return aliases.get(normalized, normalized or "unspecified")


@dataclass(slots=True)
class SourceProvenance:
    """Review-facing provenance for one local or remote source record."""

    source_type: str
    source_locator: str = ""
    source_name: str = ""
    content_scope: str = "unknown"
    acquisition_method: str = "unknown"
    adapter_name: str = ""
    adapter_version: str = ""
    parser_name: str = ""
    parser_version: str = ""
    retrieval_time_utc: str = ""
    local_path: str = ""
    source_url: str = ""
    raw_record_id: str = ""
    source_sha256: str = ""
    source_size_bytes: int = 0
    source_modified_time_utc: str = ""
    record_count: int = 1
    status: str = "ok"
    warnings: list[str] = dataclass_field(default_factory=list)
    limitations: list[str] = dataclass_field(default_factory=list)
    next_handoff: str = ""
    schema_version: str = "source_provenance_v1"
    metadata: JsonDict = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source_type = str(self.source_type or "unknown").strip().lower()
        self.source_locator = str(self.source_locator or "").strip()
        self.source_name = str(self.source_name or "").strip()
        self.content_scope = str(self.content_scope or "unknown").strip().lower()
        self.acquisition_method = str(self.acquisition_method or "unknown").strip().lower()
        self.adapter_name = str(self.adapter_name or "").strip()
        self.adapter_version = str(self.adapter_version or "").strip()
        self.parser_name = str(self.parser_name or "").strip()
        self.parser_version = str(self.parser_version or "").strip()
        self.retrieval_time_utc = str(self.retrieval_time_utc or "").strip()
        self.local_path = str(self.local_path or "").strip()
        self.source_url = str(self.source_url or "").strip()
        self.raw_record_id = str(self.raw_record_id or "").strip()
        self.source_sha256 = str(self.source_sha256 or "").strip()
        self.source_size_bytes = max(0, int(self.source_size_bytes or 0))
        self.source_modified_time_utc = str(self.source_modified_time_utc or "").strip()
        self.record_count = max(0, int(self.record_count or 0))
        self.status = str(self.status or "ok").strip().lower()
        self.warnings = _clean_list(self.warnings)
        self.limitations = _clean_list(self.limitations)
        self.next_handoff = str(self.next_handoff or "").strip()
        self.schema_version = str(self.schema_version or "source_provenance_v1").strip()
        self.metadata = dict(self.metadata or {})
        if not self.source_locator:
            self.source_locator = self.source_url or self.local_path or self.raw_record_id

    def to_dict(self) -> JsonDict:
        """Serialize source provenance to a JSON-compatible dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "SourceProvenance":
        """Deserialize source provenance from a dictionary."""

        return cls(**dict(data or {}))


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
    extraction_confidence: float = 0.5

    def __post_init__(self) -> None:
        self.text = " ".join(str(self.text or "").split())
        if not self.text:
            raise ValueError("Evidence text cannot be empty.")
        self.extraction_confidence = _clamp01(
            self.extraction_confidence, "Evidence.extraction_confidence"
        )
        if self.sentence_index < -1:
            raise ValueError("Evidence.sentence_index must be -1 or greater.")

    def to_dict(self) -> JsonDict:
        """Serialize the evidence record to a JSON-compatible dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Evidence":
        """Deserialize an evidence record from a dictionary."""

        payload = dict(data)
        if "extraction_confidence" not in payload and "confidence" in payload:
            payload["extraction_confidence"] = payload.pop("confidence")
        else:
            payload.pop("confidence", None)
        return cls(**payload)


# Dataset records describe what a source appears to contain, before matching.
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
        # JSONL/SQLite reloads provide nested variables as dicts; coerce them here.
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


# Question candidates are the central literature output consumed by later nodes.
@dataclass(slots=True)
class QuestionCandidate:
    """A normalized open research question extracted from literature."""

    question_id: str
    question: str
    source_ids: list[str] = dataclass_field(default_factory=list)
    evidence: list[Evidence] = dataclass_field(default_factory=list)
    question_origin: str = "unspecified"
    field: str = "biomedical"
    domain_terms: list[str] = dataclass_field(default_factory=list)
    required_variables: list[str] = dataclass_field(default_factory=list)
    population: str = ""
    outcomes: list[str] = dataclass_field(default_factory=list)
    extraction_confidence: float = 0.5
    novelty_score: float = 0.5
    significance_score: float = 0.5
    answerability: float = 0.5
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
        self.question_origin = _normalize_question_origin(self.question_origin)
        self.extraction_confidence = _clamp01(
            self.extraction_confidence, "QuestionCandidate.extraction_confidence"
        )
        self.novelty_score = _clamp01(self.novelty_score, "QuestionCandidate.novelty_score")
        self.significance_score = _clamp01(
            self.significance_score, "QuestionCandidate.significance_score"
        )
        self.answerability = _clamp01(
            self.answerability, "QuestionCandidate.answerability"
        )

    @property
    def normalized_question(self) -> str:
        """Lowercase representation used for deduplication."""

        chars = [ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in self.question]
        return " ".join("".join(chars).split())

    def merge(self, other: "QuestionCandidate") -> "QuestionCandidate":
        """Merge another candidate that refers to the same underlying question."""

        # Keep the clearest/highest extraction-confidence phrasing while preserving provenance.
        if other.extraction_confidence > self.extraction_confidence:
            self.question = other.question
        self.source_ids = _clean_list([*self.source_ids, *other.source_ids])
        self.evidence = [*self.evidence, *other.evidence]
        self.domain_terms = _clean_list([*self.domain_terms, *other.domain_terms])
        self.required_variables = _clean_list(
            [*self.required_variables, *other.required_variables]
        )
        self.outcomes = _clean_list([*self.outcomes, *other.outcomes])
        self.extraction_confidence = max(
            self.extraction_confidence, other.extraction_confidence
        )
        self.novelty_score = max(self.novelty_score, other.novelty_score)
        self.significance_score = max(self.significance_score, other.significance_score)
        self.answerability = max(self.answerability, other.answerability)
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
        if "question_origin" not in payload and "extraction_type" in payload:
            payload["question_origin"] = payload.pop("extraction_type")
        else:
            payload.pop("extraction_type", None)
        if "extraction_confidence" not in payload and "confidence" in payload:
            payload["extraction_confidence"] = payload.pop("confidence")
        else:
            payload.pop("confidence", None)
        if "answerability" not in payload and "answerability_hint" in payload:
            payload["answerability"] = payload.pop("answerability_hint")
        else:
            payload.pop("answerability_hint", None)
        payload["evidence"] = [
            item if isinstance(item, Evidence) else Evidence.from_dict(item)
            for item in payload.get("evidence", [])
        ]
        return cls(**payload)


# Synthesis and matching objects carry questions forward into ranked opportunities.
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
        # Persisted matches reload as nested dictionaries; normalize them back to contracts.
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


# Annotation schemas make expert review outputs reusable as training data.
@dataclass(slots=True)
class QuestionQualityScore:
    """Expert quality ratings for an extracted or proposed question."""

    question_id: str
    annotator_id: str = ""
    label_id: str = ""
    clarity_score: float | None = None
    importance_score: float | None = None
    novelty_score: float | None = None
    actionability_score: float | None = None
    translational_score: float | None = None
    overall_score: float | None = None
    notes: str = ""
    metadata: JsonDict = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.question_id = str(self.question_id or "").strip()
        self.annotator_id = str(self.annotator_id or "").strip()
        if not self.question_id:
            raise ValueError("QuestionQualityScore.question_id cannot be empty.")
        if not self.label_id:
            self.label_id = stable_id("question_quality", self.question_id, self.annotator_id)
        for field_name in (
            "clarity_score",
            "importance_score",
            "novelty_score",
            "actionability_score",
            "translational_score",
            "overall_score",
        ):
            setattr(self, field_name, _optional_score(getattr(self, field_name), field_name))
        self.notes = str(self.notes or "").strip()
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> JsonDict:
        """Serialize the quality score to a JSON-compatible dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "QuestionQualityScore":
        """Deserialize a quality score from a dictionary."""

        return cls(**data)


@dataclass(slots=True)
class QuestionLabel:
    """Expert label for whether a candidate is a usable open question."""

    question_id: str
    annotator_id: str = ""
    source_id: str = ""
    label_id: str = ""
    label: str = "unlabeled"
    is_valid_open_question: bool | None = None
    error_types: list[str] = dataclass_field(default_factory=list)
    notes: str = ""
    metadata: JsonDict = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.question_id = str(self.question_id or "").strip()
        self.annotator_id = str(self.annotator_id or "").strip()
        self.source_id = str(self.source_id or "").strip()
        if not self.question_id:
            raise ValueError("QuestionLabel.question_id cannot be empty.")
        allowed = {"accepted", "rejected", "uncertain", "needs_revision", "unlabeled"}
        self.label = str(self.label or "unlabeled").strip().lower()
        if self.label not in allowed:
            raise ValueError(f"QuestionLabel.label must be one of {sorted(allowed)}.")
        self.is_valid_open_question = _optional_bool(self.is_valid_open_question)
        self.error_types = _clean_list(self.error_types)
        self.notes = str(self.notes or "").strip()
        self.metadata = dict(self.metadata or {})
        if not self.label_id:
            self.label_id = stable_id(
                "question_label", self.question_id, self.annotator_id, self.label
            )

    def to_dict(self) -> JsonDict:
        """Serialize the question label to a JSON-compatible dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "QuestionLabel":
        """Deserialize a question label from a dictionary."""

        return cls(**data)


@dataclass(slots=True)
class EvidenceSpanLabel:
    """Expert label for a text span used as evidence for a question."""

    question_id: str
    source_id: str
    text: str = ""
    annotator_id: str = ""
    label_id: str = ""
    section: str = ""
    start_char: int = -1
    end_char: int = -1
    label: str = "unlabeled"
    confidence: float = 0.5
    notes: str = ""
    metadata: JsonDict = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.question_id = str(self.question_id or "").strip()
        self.source_id = str(self.source_id or "").strip()
        self.text = " ".join(str(self.text or "").split())
        self.annotator_id = str(self.annotator_id or "").strip()
        self.section = str(self.section or "").strip()
        if not self.question_id:
            raise ValueError("EvidenceSpanLabel.question_id cannot be empty.")
        if not self.source_id:
            raise ValueError("EvidenceSpanLabel.source_id cannot be empty.")
        self.start_char = int(self.start_char or -1)
        self.end_char = int(self.end_char or -1)
        if self.start_char >= 0 and self.end_char >= 0 and self.end_char < self.start_char:
            raise ValueError("EvidenceSpanLabel.end_char cannot be before start_char.")
        allowed = {"supporting", "not_relevant", "ambiguous", "unlabeled"}
        self.label = str(self.label or "unlabeled").strip().lower()
        if self.label not in allowed:
            raise ValueError(f"EvidenceSpanLabel.label must be one of {sorted(allowed)}.")
        self.confidence = _clamp01(self.confidence, "EvidenceSpanLabel.confidence")
        self.notes = str(self.notes or "").strip()
        self.metadata = dict(self.metadata or {})
        if not self.label_id:
            self.label_id = stable_id(
                "evidence_label",
                self.question_id,
                self.source_id,
                self.start_char,
                self.end_char,
                self.annotator_id,
            )

    def to_dict(self) -> JsonDict:
        """Serialize the evidence label to a JSON-compatible dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "EvidenceSpanLabel":
        """Deserialize an evidence label from a dictionary."""

        return cls(**data)


@dataclass(slots=True)
class QuestionDataMatchLabel:
    """Expert label for how well a dataset can answer a question."""

    match_id: str
    question_id: str
    dataset_id: str
    annotator_id: str = ""
    label_id: str = ""
    label: str = "unlabeled"
    relevance_score: float | None = None
    question_quality_score: float | None = None
    data_match_quality_score: float | None = None
    answerability_score: float | None = None
    notes: str = ""
    metadata: JsonDict = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.match_id = str(self.match_id or "").strip()
        self.question_id = str(self.question_id or "").strip()
        self.dataset_id = str(self.dataset_id or "").strip()
        self.annotator_id = str(self.annotator_id or "").strip()
        if not self.match_id:
            raise ValueError("QuestionDataMatchLabel.match_id cannot be empty.")
        if not self.question_id:
            raise ValueError("QuestionDataMatchLabel.question_id cannot be empty.")
        if not self.dataset_id:
            raise ValueError("QuestionDataMatchLabel.dataset_id cannot be empty.")
        self.relevance_score = _optional_score(
            self.relevance_score, "QuestionDataMatchLabel.relevance_score", maximum=1.0
        )
        self.question_quality_score = _optional_score(
            self.question_quality_score,
            "QuestionDataMatchLabel.question_quality_score",
        )
        self.data_match_quality_score = _optional_score(
            self.data_match_quality_score,
            "QuestionDataMatchLabel.data_match_quality_score",
        )
        self.answerability_score = _optional_score(
            self.answerability_score, "QuestionDataMatchLabel.answerability_score"
        )
        allowed = {"relevant", "not_relevant", "uncertain", "unlabeled"}
        self.label = str(self.label or "unlabeled").strip().lower()
        if self.label == "unlabeled" and self.relevance_score is not None:
            self.label = "relevant" if self.relevance_score > 0 else "not_relevant"
        if self.label not in allowed:
            raise ValueError(f"QuestionDataMatchLabel.label must be one of {sorted(allowed)}.")
        self.notes = str(self.notes or "").strip()
        self.metadata = dict(self.metadata or {})
        if not self.label_id:
            self.label_id = stable_id(
                "match_label", self.match_id, self.annotator_id, self.label
            )

    def to_dict(self) -> JsonDict:
        """Serialize the match label to a JSON-compatible dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "QuestionDataMatchLabel":
        """Deserialize a match label from a dictionary."""

        return cls(**data)


@dataclass(slots=True)
class DerivedVariableRule:
    """Rule describing how a variable could be derived from available fields."""

    output_variable: str
    input_variables: list[str]
    rule_id: str = ""
    expression: str = ""
    description: str = ""
    assumptions: list[str] = dataclass_field(default_factory=list)
    evidence: str = ""
    confidence: float = 0.5
    metadata: JsonDict = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_variable = str(self.output_variable or "").strip()
        if not self.output_variable:
            raise ValueError("DerivedVariableRule.output_variable cannot be empty.")
        self.input_variables = _clean_list(self.input_variables)
        if not self.input_variables:
            raise ValueError("DerivedVariableRule.input_variables cannot be empty.")
        self.expression = str(self.expression or "").strip()
        self.description = str(self.description or "").strip()
        self.assumptions = _clean_list(self.assumptions)
        self.evidence = str(self.evidence or "").strip()
        self.confidence = _clamp01(self.confidence, "DerivedVariableRule.confidence")
        self.metadata = dict(self.metadata or {})
        if not self.rule_id:
            self.rule_id = stable_id(
                "derived_rule", self.output_variable, *self.input_variables
            )

    def to_dict(self) -> JsonDict:
        """Serialize the derived-variable rule to a JSON-compatible dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "DerivedVariableRule":
        """Deserialize a derived-variable rule from a dictionary."""

        return cls(**data)


@dataclass(slots=True)
class DatasetCapability:
    """Observed or derivable dataset capability relevant to matching."""

    dataset_id: str
    variable_name: str
    capability_type: str = "observed"
    capability_id: str = ""
    source_variable_names: list[str] = dataclass_field(default_factory=list)
    derivation_rule_id: str = ""
    confidence: float = 0.5
    evidence: str = ""
    limitations: list[str] = dataclass_field(default_factory=list)
    metadata: JsonDict = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dataset_id = str(self.dataset_id or "").strip()
        self.variable_name = str(self.variable_name or "").strip()
        if not self.dataset_id:
            raise ValueError("DatasetCapability.dataset_id cannot be empty.")
        if not self.variable_name:
            raise ValueError("DatasetCapability.variable_name cannot be empty.")
        allowed = {"observed", "derived", "linked", "unknown"}
        self.capability_type = str(self.capability_type or "observed").strip().lower()
        if self.capability_type not in allowed:
            raise ValueError(f"DatasetCapability.capability_type must be one of {sorted(allowed)}.")
        self.source_variable_names = _clean_list(self.source_variable_names)
        self.derivation_rule_id = str(self.derivation_rule_id or "").strip()
        self.confidence = _clamp01(self.confidence, "DatasetCapability.confidence")
        self.evidence = str(self.evidence or "").strip()
        self.limitations = _clean_list(self.limitations)
        self.metadata = dict(self.metadata or {})
        if not self.capability_id:
            self.capability_id = stable_id(
                "dataset_capability", self.dataset_id, self.variable_name, self.capability_type
            )

    def to_dict(self) -> JsonDict:
        """Serialize the dataset capability to a JSON-compatible dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "DatasetCapability":
        """Deserialize a dataset capability from a dictionary."""

        return cls(**data)


@dataclass(slots=True)
class DerivedCapabilityLabel:
    """Expert label for whether a derived dataset capability is plausible."""

    capability_id: str
    dataset_id: str
    annotator_id: str = ""
    rule_id: str = ""
    label_id: str = ""
    is_plausible: bool | None = None
    usefulness_score: float | None = None
    evidence_quality_score: float | None = None
    notes: str = ""
    metadata: JsonDict = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.capability_id = str(self.capability_id or "").strip()
        self.dataset_id = str(self.dataset_id or "").strip()
        self.annotator_id = str(self.annotator_id or "").strip()
        self.rule_id = str(self.rule_id or "").strip()
        if not self.capability_id:
            raise ValueError("DerivedCapabilityLabel.capability_id cannot be empty.")
        if not self.dataset_id:
            raise ValueError("DerivedCapabilityLabel.dataset_id cannot be empty.")
        self.is_plausible = _optional_bool(self.is_plausible)
        self.usefulness_score = _optional_score(
            self.usefulness_score, "DerivedCapabilityLabel.usefulness_score"
        )
        self.evidence_quality_score = _optional_score(
            self.evidence_quality_score, "DerivedCapabilityLabel.evidence_quality_score"
        )
        self.notes = str(self.notes or "").strip()
        self.metadata = dict(self.metadata or {})
        if not self.label_id:
            self.label_id = stable_id(
                "derived_capability_label",
                self.capability_id,
                self.dataset_id,
                self.annotator_id,
            )

    def to_dict(self) -> JsonDict:
        """Serialize the derived-capability label to a JSON-compatible dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "DerivedCapabilityLabel":
        """Deserialize a derived-capability label from a dictionary."""

        return cls(**data)


@dataclass(slots=True)
class ExpertPaperAnnotation:
    """Container for expert labels attached to one paper or text source."""

    source_id: str
    annotator_id: str = ""
    annotation_id: str = ""
    title: str = ""
    doi: str = ""
    question_labels: list[QuestionLabel] = dataclass_field(default_factory=list)
    evidence_span_labels: list[EvidenceSpanLabel] = dataclass_field(default_factory=list)
    notes: str = ""
    metadata: JsonDict = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source_id = str(self.source_id or "").strip()
        self.annotator_id = str(self.annotator_id or "").strip()
        self.title = " ".join(str(self.title or "").split())
        self.doi = str(self.doi or "").strip()
        if not self.source_id:
            self.source_id = stable_id("source", self.doi, self.title)
        self.question_labels = [
            item if isinstance(item, QuestionLabel) else QuestionLabel.from_dict(item)
            for item in self.question_labels
        ]
        self.evidence_span_labels = [
            item if isinstance(item, EvidenceSpanLabel) else EvidenceSpanLabel.from_dict(item)
            for item in self.evidence_span_labels
        ]
        self.notes = str(self.notes or "").strip()
        self.metadata = dict(self.metadata or {})
        if not self.annotation_id:
            self.annotation_id = stable_id(
                "paper_annotation", self.source_id, self.annotator_id, self.doi
            )

    def to_dict(self) -> JsonDict:
        """Serialize the paper annotation to a JSON-compatible dictionary."""

        data = asdict(self)
        data["question_labels"] = [item.to_dict() for item in self.question_labels]
        data["evidence_span_labels"] = [
            item.to_dict() for item in self.evidence_span_labels
        ]
        return data

    @classmethod
    def from_dict(cls, data: JsonDict) -> "ExpertPaperAnnotation":
        """Deserialize a paper annotation from a dictionary."""

        payload = dict(data)
        payload["question_labels"] = [
            item if isinstance(item, QuestionLabel) else QuestionLabel.from_dict(item)
            for item in payload.get("question_labels", [])
        ]
        payload["evidence_span_labels"] = [
            item
            if isinstance(item, EvidenceSpanLabel)
            else EvidenceSpanLabel.from_dict(item)
            for item in payload.get("evidence_span_labels", [])
        ]
        return cls(**payload)
