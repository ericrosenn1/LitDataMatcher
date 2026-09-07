"""Research question extraction module."""

from typing import List
import re
from models_lit_analyzer import ResearchQuestion, Evidence, ExtractionSignals, ExtractionType

# Import these from parent - we'll handle this at the end
# For now, these will be imported from lit_analyzer
RQ_HOOKS = [
    "we ask", "we investigate", "our research question", "our central question",
    "we aim to", "we intend to", "we seek to", "we strive to", "we attempt to",
    "we hypothesize", "we test whether", "we explore whether", "we evaluate whether",
    "we examine whether", "we examine the hypothesis", "this study asks",
    "this study investigates", "this study evaluates", "this paper investigates",
    "this paper examines", "this work explores", "this research aims to",
    "the purpose of this study", "the purpose of this work",
    "the aim of this study", "the goal of this study", "the objective of this study",
    "our main question", "this research focuses on", "we analyzed whether"
]


def normalize_question(s: str) -> str:
    """Normalize a question string."""
    q = s.strip()
    if q.endswith("."):
        q = q[:-1]
    if not q.endswith("?"):
        q = q + "?"
    return q


def find_research_questions(
    text: str,
    split_sents_fn,
    semantic_match_fn,
    predict_discourse_fn
) -> List[ResearchQuestion]:
    """
    Extract research questions using models.
    
    Args:
        text: Text to analyze
        split_sents_fn: Function to split text into sentences
        semantic_match_fn: Function for semantic matching
        predict_discourse_fn: Function for discourse prediction
        
    Returns:
        List of ResearchQuestion objects
    """
    results = []
    sents = split_sents_fn(text)

    def _norm(s: str) -> str:
        t = s.lower()
        t = re.sub(r"\(\s*\d{1,4}\s*\)|\[\s*\d{1,4}\s*\]", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    seen = set()
    for i, s in enumerate(sents):
        # Length filter
        if not s or len(s) < 30 or len(s) > 280:
            continue
        
        # Deduplication
        key = _norm(s)
        if key in seen:
            continue
        
        low = key
        
        # Signal detection
        has_hook = any(h in low for h in RQ_HOOKS)
        has_semantic = semantic_match_fn(s, "rq")
        discourse_label = predict_discourse_fn(s)
        has_discourse = (discourse_label == "rq")
        
        # Must have at least one signal
        hit = has_hook or has_semantic or has_discourse
        if not hit:
            continue
        
        # Additional quality filter
        if not (s.strip().endswith("?") or "we " in low or "aim" in low or "hypothesi" in low):
            continue
        
        seen.add(key)
        
        # Create signal object
        signals = ExtractionSignals(
            hook=has_hook,
            semantic=has_semantic
        )
        
        # Create evidence object
        evidence = Evidence(
            text=s.strip(),
            sent_index=i
        )
        
        # Create ResearchQuestion object
        rq = ResearchQuestion(
            question=normalize_question(s),
            evidence=evidence,
            confidence=0.8,
            signals=signals,
            type=ExtractionType.EXPLICIT_RQ
        )
        
        results.append(rq)
    
    return results