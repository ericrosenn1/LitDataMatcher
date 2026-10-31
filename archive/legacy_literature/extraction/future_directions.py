"""Future direction extraction module."""

from typing import List, Optional
import os
import numpy as np
from models_lit_analyzer import FutureDirection, ExtractionSignals

FUTURE_HOOKS = [
    "future work", "future studies", "future experiments", "further work",
    "further study", "further studies", "further experiments",
    "remain to be determined", "remains to be determined", "remains unknown",
    "is unknown", "is unclear", "unclear whether", "not been examined",
    "not yet examined", "not yet understood", "not yet explored",
    "warrants investigation", "warrants further investigation",
    "needs to be investigated", "needs to be explored", "should be examined",
    "should be explored", "should be investigated", "should be addressed",
    "should be clarified", "should be validated", "should be replicated",
    "required to determine", "needed to determine", "will be important",
    "will be critical", "will be essential", "will clarify", "will elucidate",
    "next step", "next steps", "next direction", "next directions",
    "in the future", "in future work", "in future studies",
    "future research should", "further research should",
    "further investigation is warranted", "further investigation will",
    "remains an open question", "open question", "open questions",
    "outstanding question", "outstanding questions",
    "should be confirmed", "should be validated", "should be extended",
    "should be compared", "should be tested", "should be improved",
    "remains to be explored", "requires additional investigation",
    "calls for additional studies", "calls for future work",
    "merits further investigation", "needs confirmation",
    "yet to be elucidated", "yet to be determined"
]

WILL_VERBS = [
    "reveal", "clarify", "determine", "validate", "replicate",
    "elucidate", "quantify", "compare", "test", "assess",
    "examine", "characterize", "investigate", "map", "measure"
]


def has_will_infinitive(low: str) -> bool:
    """Check if text has will + infinitive pattern."""
    if "will " not in low:
        return False
    for v in WILL_VERBS:
        if ("will " + v) in low:
            return True
    return False


def label_future_item(s: str) -> str:
    """Categorize a future direction."""
    low = s.lower()
    if low.find("mechanism") != -1 or low.find("oligomer") != -1:
        return "mechanism"
    if low.find("structure") != -1 or low.find("structural") != -1:
        return "structure"
    if low.find("regulat") != -1 or low.find("post-translational") != -1:
        return "regulation"
    if low.find("validate") != -1 or low.find("replicate") != -1:
        return "validation"
    if low.find("other species") != -1 or low.find("generaliz") != -1:
        return "generalization"
    return "unspecified"


def find_future_directions(
    text: str,
    split_sents_fn,
    semantic_match_fn,
    predict_discourse_fn,
    build_features_fn,
    assemble_context_fn,
    ctx_index=None,
    future_clf=None,
    future_th=0.70,
    ctx_opts=None,
    log_dir="out_lit"
) -> List[FutureDirection]:
    """
    Extract future directions using models.
    
    Includes:
    - Option A: ML classifier filtering (if future_clf is set)
    - Option C: Context attachment (if ctx_index provided)
    
    Args:
        text: Text to analyze
        split_sents_fn: Function to split sentences
        semantic_match_fn: Function for semantic matching
        predict_discourse_fn: Function for discourse prediction
        build_features_fn: Function to build ML features
        assemble_context_fn: Function to assemble context
        ctx_index: Optional context indexer
        future_clf: Optional ML classifier
        future_th: Classifier threshold
        ctx_opts: Context options dict
        log_dir: Directory for logs
        
    Returns:
        List of FutureDirection objects
    """
    results = []
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "rejected_sentences.txt")

    if ctx_opts is None:
        ctx_opts = {"enabled": False}

    with open(log_path, "a", encoding="utf-8") as rej:
        sents = split_sents_fn(text)

        MODALS = [
            "will", "should", "could", "might", "may", "needed", "need to", "needs to",
            "remain", "remains", "warrant", "requires", "required", "require",
            "expected", "important", "critical"
        ]

        def _norm(s: str) -> str:
            import re
            t = s.lower()
            t = re.sub(r"\(\s*\d{1,4}\s*\)", " ", t)
            t = re.sub(r"\[\s*\d{1,4}\s*\]", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            return t

        def _looks_like_reference(s: str) -> bool:
            import re
            low = s.lower()
            if "doi:" in low or "https://" in low or "http://" in low:
                return True
            if re.search(r"\b(et al\.|vol\.|pp\.|issn|isbn)\b", low):
                return True
            if re.search(r"\(\d{4}\)", s):
                return True
            if len(re.findall(r"\d+", s)) >= 6:
                return True
            return False

        seen = set()
        for i, s in enumerate(sents):
            # Length filter
            if not s or len(s) < 40 or len(s) > 350:
                continue
            
            # Reference filter
            if _looks_like_reference(s):
                rej.write(f"[future_dir_reject_ref]\t{s.strip()}\n")
                continue

            # Deduplication
            low = _norm(s)
            if low in seen:
                continue
            seen.add(low)

            # Signal detection
            sig_hook = any(h in low for h in FUTURE_HOOKS)
            sig_modal = any(m in f" {low} " for m in MODALS)
            sig_will = has_will_infinitive(low)
            sig_sem = semantic_match_fn(s, "future")
            sig_lab = predict_discourse_fn(s)

            # Baseline rule-based gate
            rule_pass = (sig_hook and (sig_modal or sig_will)) or sig_sem or (sig_lab == "future")

            # ML classifier (Option A)
            ml_prob = None
            ml_pass = True
            if future_clf is not None:
                try:
                    X = build_features_fn(s)
                    if hasattr(future_clf, "predict_proba"):
                        ml_prob = float(future_clf.predict_proba(X)[0, 1])
                    else:
                        score = float(future_clf.decision_function(X)[0])
                        ml_prob = 1.0 / (1.0 + np.exp(-score))
                    ml_pass = (ml_prob >= future_th)
                except Exception:
                    ml_prob = None
                    ml_pass = True

            keep = (rule_pass and ml_pass) if future_clf is not None else rule_pass

            if keep:
                # Calculate confidence
                conf = 0.9 if (sig_will or sig_sem or sig_lab == "future" or (ml_prob and ml_prob >= 0.85)) else 0.8

                # Option C: attach context
                context_str = ""
                if ctx_index is not None and assemble_context_fn is not None:
                    context_str = assemble_context_fn(i, ctx_index, ctx_opts)

                # Create signals object
                signals = ExtractionSignals(
                    hook=sig_hook,
                    semantic=sig_sem,
                    modal=sig_modal,
                    will_infinitive=sig_will,
                    classifier_score=ml_prob
                )
                
                # Categorize
                category = label_future_item(s)
                
                # Create FutureDirection object
                direction = FutureDirection(
                    text=s.strip(),
                    category=category,
                    sent_index=i,
                    confidence=conf,
                    signals=signals,
                    context=context_str
                )
                
                results.append(direction)
            else:
                rej.write(f"[future_dir_reject]\t{s.strip()}\n")

    return results