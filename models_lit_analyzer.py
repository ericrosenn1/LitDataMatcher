"""
Data models for literature extraction.

This module defines the core data structures used throughout the analyzer.
Each model represents a specific concept with strong typing and validation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

# =============================================================================
# ENUMS: Controlled Vocabularies
# =============================================================================

class ExtractionType(Enum):
    """
    Types of extractions we can make from papers.
    
    Using an Enum ensures only valid types are used.
    Prevents typos like "explict_rq" vs "explicit_rq".
    """
    EXPLICIT_RQ = "explicit_rq"
    FUTURE_DIRECTION = "future_direction"
    LIMITATION = "limitation"
    
    # Benefits of Enum:
    # 1. Autocomplete: ExtractionType.EXPLICIT_RQ
    # 2. Type checking: Can't pass invalid string
    # 3. Documentation: All valid values in one place


# =============================================================================
# BASIC BUILDING BLOCKS
# =============================================================================

@dataclass
class Evidence:
    """
    Represents WHERE we found something in the paper.
    
    This is a "value object" - it has no behavior, just data.
    It's immutable (frozen=True) because evidence shouldn't change.
    
    Attributes:
        text: The actual sentence or paragraph we extracted
        sent_index: Position in the document (for reference)
        section: Which section it came from (optional)
    """
    text: str
    sent_index: int
    section: Optional[str] = None
    
    # Why dataclass?
    # - Automatic __init__, __repr__, __eq__
    # - Type hints enforced
    # - Clean, readable definition
    
    def __post_init__(self):
        """
        Validation that runs after __init__.
        
        This ensures data is always valid when object is created.
        """
        if not self.text:
            raise ValueError("Evidence text cannot be empty")
        if self.sent_index < 0:
            raise ValueError(f"Sentence index must be >= 0, got {self.sent_index}")
    
    def preview(self, max_len: int = 100) -> str:
        """
        Get a shortened version of the evidence text.
        
        This is a "convenience method" - makes the object more useful.
        """
        if len(self.text) <= max_len:
            return self.text
        return self.text[:max_len-3] + "..."


@dataclass
class ExtractionSignals:
    """
    Tracks HOW we identified an extraction.
    
    This is crucial for debugging and understanding why something
    was extracted. In your code, you build these as dicts on the fly.
    
    Attributes:
        hook: Did it match a phrase hook? (e.g., "future work")
        semantic: Did semantic similarity identify it?
        modal: Does it contain modal verbs? (will, should, etc.)
        will_infinitive: Has "will + verb" pattern?
        classifier_score: ML classifier probability (if used)
    """
    hook: bool = False
    semantic: bool = False
    modal: bool = False
    will_infinitive: bool = False
    classifier_score: Optional[float] = None
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary for JSON serialization.
        
        This is the bridge between your type-safe code and JSON output.
        Notice we control exactly what gets serialized.
        """
        d = {
            "hook": self.hook,
            "semantic": self.semantic,
            "modal": self.modal,
            "will_inf": self.will_infinitive
        }
        if self.classifier_score is not None:
            d["classifier"] = f"ml:{self.classifier_score:.3f}"
        return d
    
    @property
    def strength(self) -> str:
        """
        Qualitative assessment of signal strength.
        
        This is a "computed property" - derived from other attributes.
        """
        if self.classifier_score and self.classifier_score > 0.85:
            return "strong"
        if self.will_infinitive or self.semantic:
            return "strong"
        if self.hook and self.modal:
            return "moderate"
        return "weak"
    
    @property
    def signal_count(self) -> int:
        """Count how many signals fired."""
        count = sum([
            self.hook,
            self.semantic,
            self.modal,
            self.will_infinitive
        ])
        if self.classifier_score and self.classifier_score > 0.7:
            count += 1
        return count


# =============================================================================
# EXTRACTION RESULTS
# =============================================================================

@dataclass
class ResearchQuestion:
    """
    An extracted research question from a paper.
    
    This replaces your dict-based approach:
    OLD: {"type": "explicit_rq", "question": "...", ...}
    NEW: ResearchQuestion(question="...", ...)
    
    Benefits:
    1. Can't misspell field names (IDE catches it)
    2. Can't use wrong types (type checker catches it)
    3. Rich object with methods, not just data
    4. Self-documenting
    """
    question: str
    evidence: Evidence
    confidence: float
    signals: ExtractionSignals
    type: ExtractionType = ExtractionType.EXPLICIT_RQ
    
    def __post_init__(self):
        """Validate data on creation."""
        if not self.question:
            raise ValueError("Question cannot be empty")
        
        if not 0 <= self.confidence <= 1:
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )
    
    def to_dict(self) -> dict:
        """
        Convert to dict for JSONL output.
        
        This is what gets written to explicit_rqs.jsonl.
        Notice the structure matches your current output format.
        """
        return {
            "type": self.type.value,
            "question": self.question,
            "evidence": {
                "text": self.evidence.text,
                "sent_index": self.evidence.sent_index,
                "section": self.evidence.section
            },
            "confidence": self.confidence,
            "signals": self.signals.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ResearchQuestion':
        """
        Create from dict (for loading saved data).
        
        This is the inverse of to_dict() - useful for reading
        previously saved results.
        """
        return cls(
            question=data["question"],
            evidence=Evidence(
                text=data["evidence"]["text"],
                sent_index=data["evidence"]["sent_index"],
                section=data["evidence"].get("section")
            ),
            confidence=data["confidence"],
            signals=ExtractionSignals(**data.get("signals", {})),
            type=ExtractionType(data.get("type", "explicit_rq"))
        )
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return f"RQ[{self.confidence:.2f}]: {self.question[:80]}..."
    
    def is_high_confidence(self) -> bool:
        """Check if this is a high-confidence extraction."""
        return self.confidence >= 0.8


@dataclass
class FutureDirection:
    """
    An identified future research direction.
    
    These are forward-looking statements about what should be done next.
    """
    text: str
    category: str  # mechanism, validation, generalization, etc.
    sent_index: int
    confidence: float
    signals: ExtractionSignals
    context: str = ""  # Surrounding sentences for context
    
    def __post_init__(self):
        """Validate on creation."""
        if not self.text:
            raise ValueError("Future direction text cannot be empty")
        
        if not 0 <= self.confidence <= 1:
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )
        
        # Validate category
        valid_categories = {
            "mechanism", "structure", "regulation",
            "validation", "generalization", "unspecified"
        }
        if self.category not in valid_categories:
            raise ValueError(
                f"Invalid category '{self.category}'. "
                f"Must be one of: {valid_categories}"
            )
    
    def to_dict(self) -> dict:
        """Convert to dict for JSONL output."""
        return {
            "type": "future_direction",
            "text": self.text,
            "category": self.category,
            "sent_index": self.sent_index,
            "confidence": self.confidence,
            "signals": self.signals.to_dict(),
            "context": self.context
        }
    
    def has_context(self) -> bool:
        """Check if context was attached."""
        return bool(self.context)
    
    def is_actionable(self) -> bool:
        """
        Determine if this is an actionable direction.
        
        Actionable = specific + high confidence
        """
        return (
            self.confidence >= 0.75 and
            self.category != "unspecified" and
            len(self.text) > 50  # Detailed enough
        )


@dataclass
class Limitation:
    """
    An acknowledged limitation from the paper.
    """
    text: str
    sent_index: int
    confidence: float
    limitation_type: str = "unspecified"  # sample_size, confounding, etc.
    
    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")
    
    def to_dict(self) -> dict:
        return {
            "type": "limitation",
            "text": self.text,
            "sent_index": self.sent_index,
            "confidence": self.confidence,
            "limitation_type": self.limitation_type
        }


@dataclass
class NextStep:
    """
    A generated next step based on RQs and limitations.
    
    These are suggestions for follow-up work.
    """
    question: str
    rationale: str
    suggested_approach: str
    expected_impact: float  # 0-1 score
    difficulty: float       # 0-1 score
    links: List[str] = field(default_factory=list)  # Related RQs/limitations
    
    def __post_init__(self):
        if not 0 <= self.expected_impact <= 1:
            raise ValueError("Expected impact must be 0-1")
        if not 0 <= self.difficulty <= 1:
            raise ValueError("Difficulty must be 0-1")
    
    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "rationale": self.rationale,
            "suggested_approach": self.suggested_approach,
            "expected_impact": self.expected_impact,
            "difficulty": self.difficulty,
            "links": self.links
        }
    
    @property
    def priority_score(self) -> float:
        """
        Calculate priority based on impact and difficulty.
        
        High impact + low difficulty = high priority
        """
        return self.expected_impact * (1 - self.difficulty)


# =============================================================================
# DOCUMENT MODEL
# =============================================================================

@dataclass
class Document:
    """
    Input document with metadata and processed text.
    
    This replaces the various dicts passed around in your code:
    OLD: {"title": "...", "abstract": "...", "text": "...", "doi": "..."}
    NEW: Document(title="...", abstract="...", ...)
    """
    title: str
    text: str
    abstract: str = ""
    doi: str = ""
    sections: Dict[str, str] = field(default_factory=dict)
    
    # Enrichment data (added later in pipeline)
    embedding: List[float] = field(default_factory=list)
    openalex_data: Dict = field(default_factory=dict)
    entities: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        """
        Post-initialization processing.
        
        If sections aren't provided, split them automatically.
        """
        if not self.title and not self.doi:
            raise ValueError("Document must have either title or DOI")
        
        if not self.text:
            raise ValueError("Document must have text content")
        
        # Auto-split sections if not provided
        if not self.sections:
            from processing.sectioning import split_sections
            full_text = f"{self.abstract}\n{self.text}".strip()
            self.sections = split_sections(full_text)
    
    @property
    def identifier(self) -> str:
        """Get best identifier (DOI preferred, fallback to title)."""
        return self.doi if self.doi else self.title
    
    def get_section(self, section_name: str) -> str:
        """
        Safely get a section.
        
        Returns empty string if section doesn't exist.
        """
        return self.sections.get(section_name, "")
    
    def get_relevant_text_for_rqs(self) -> str:
        """
        Get text most likely to contain research questions.
        
        This encapsulates the domain knowledge of where RQs appear.
        """
        relevant = [
            self.get_section("introduction"),
            self.get_section("related"),
        ]
        return "\n".join(s for s in relevant if s)
    
    def get_relevant_text_for_future(self) -> str:
        """Get text most likely to contain future directions."""
        relevant = [
            self.get_section("discussion"),
            self.get_section("conclusion"),
            self.get_section("limitations"),
            self.get_section("future")
        ]
        return "\n".join(s for s in relevant if s)
    
    def has_openalex_data(self) -> bool:
        """Check if OpenAlex enrichment was successful."""
        return bool(self.openalex_data)
    
    def citation_count(self) -> int:
        """Get citation count from OpenAlex data."""
        return self.openalex_data.get("cited_by_count", 0)


# =============================================================================
# RESULTS AGGREGATION
# =============================================================================

@dataclass
class DocumentResults:
    """
    All extraction results for a single document.
    
    This groups all the outputs together in a structured way.
    """
    document: Document
    research_questions: List[ResearchQuestion] = field(default_factory=list)
    limitations: List[Limitation] = field(default_factory=list)
    future_directions: List[FutureDirection] = field(default_factory=list)
    next_steps: List[NextStep] = field(default_factory=list)
    
    @property
    def total_extractions(self) -> int:
        """Total number of items extracted."""
        return (
            len(self.research_questions) +
            len(self.limitations) +
            len(self.future_directions)
        )
    
    def get_high_confidence_rqs(self, threshold: float = 0.8) -> List[ResearchQuestion]:
        """Filter to high-confidence research questions."""
        return [
            rq for rq in self.research_questions
            if rq.confidence >= threshold
        ]
    
    def get_actionable_future_directions(self) -> List[FutureDirection]:
        """Get future directions that are actionable."""
        return [fd for fd in self.future_directions if fd.is_actionable()]
    
    def get_priority_next_steps(self, top_n: int = 5) -> List[NextStep]:
        """Get top priority next steps."""
        sorted_steps = sorted(
            self.next_steps,
            key=lambda x: x.priority_score,
            reverse=True
        )
        return sorted_steps[:top_n]


@dataclass  
class AnalysisResults:
    """
    Complete results from analyzing all documents.
    
    This is what the main analyzer returns.
    """
    documents: List[DocumentResults] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    
    def add_document_results(self, results: DocumentResults):
        """Add results for one document."""
        self.documents.append(results)
    
    @property
    def total_documents(self) -> int:
        return len(self.documents)
    
    @property
    def total_rqs(self) -> int:
        return sum(len(d.research_questions) for d in self.documents)
    
    @property
    def total_future_directions(self) -> int:
        return sum(len(d.future_directions) for d in self.documents)
    
    @property
    def total_limitations(self) -> int:
        return sum(len(d.limitations) for d in self.documents)
    
    @property
    def avg_extractions_per_doc(self) -> float:
        """Average extractions per document."""
        if not self.documents:
            return 0.0
        return sum(d.total_extractions for d in self.documents) / len(self.documents)
    
    def get_summary_stats(self) -> Dict[str, any]:
        """Get summary statistics."""
        return {
            "documents": self.total_documents,
            "research_questions": self.total_rqs,
            "future_directions": self.total_future_directions,
            "limitations": self.total_limitations,
            "processing_time": self.processing_time_seconds,
            "avg_extractions_per_doc": self.avg_extractions_per_doc
        }