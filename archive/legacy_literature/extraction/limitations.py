"""Limitation extraction module."""

from typing import List
from models_lit_analyzer import Limitation

LIMIT_HOOKS = [
    "limitation", "limitations", "constraints", "restricted to", "limited by",
    "threats to validity", "small sample", "sample size", "confounding",
    "lack of", "absence of", "did not", "cannot", "could not", "may bias",
    "bias may", "potential bias", "uncertainty", "variability", "heterogeneity",
    "not generalizable", "scope of this study", "beyond the scope",
    "was not evaluated", "was not addressed", "future work",
    "should be interpreted with caution", "requires cautious interpretation",
    "missing data", "incomplete data", "measurement error"
]


def find_limitations(
    text: str,
    split_sents_fn,
    semantic_match_fn,
    predict_discourse_fn
) -> List[Limitation]:
    """
    Extract limitations using models.
    
    Args:
        text: Text to analyze
        split_sents_fn: Function to split text into sentences
        semantic_match_fn: Function for semantic matching
        predict_discourse_fn: Function for discourse prediction
        
    Returns:
        List of Limitation objects
    """
    results = []
    sents = split_sents_fn(text)
    
    for i, s in enumerate(sents):
        low = s.lower()
        
        # Signal detection
        hit_hook = any(h in low for h in LIMIT_HOOKS)
        hit_sem = semantic_match_fn(s, "limit")
        lab = predict_discourse_fn(s)
        
        # Check if this is a limitation
        if hit_hook or hit_sem or (lab == "limitation"):
            # Calculate confidence
            conf = 0.9 if (hit_sem or lab == "limitation") else 0.8
            
            # Create Limitation object
            limitation = Limitation(
                text=s.strip(),
                sent_index=i,
                confidence=conf,
                limitation_type="unspecified"
            )
            
            results.append(limitation)
    
    return results