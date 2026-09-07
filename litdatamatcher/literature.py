"""Literature analysis and open-question identification node."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Iterable

from .schemas import Evidence, QuestionCandidate, stable_id
from .text import (
    extract_domain_terms,
    infer_outcomes,
    infer_population,
    infer_required_variables,
    lexical_similarity,
    normalize_text,
    split_sections,
    split_sentences,
)


# Cue banks are intentionally transparent so false positives can be reviewed.
FUTURE_CUES = (
    "future studies",
    "future work",
    "further research",
    "further studies",
    "remains unknown",
    "remain unknown",
    "remains unclear",
    "is unclear",
    "not yet understood",
    "not yet examined",
    "warrants investigation",
    "requires additional",
    "should be investigated",
    "should be examined",
    "should be tested",
    "needs to be investigated",
    "open question",
    "outstanding question",
)

RQ_CUES = (
    "we ask",
    "we investigate",
    "we examine",
    "we test whether",
    "we hypothesize",
    "we aim to",
    "this study asks",
    "this study investigates",
    "the purpose of this study",
    "the objective of this study",
)

LIMITATION_CUES = (
    "limitation",
    "limited by",
    "small sample",
    "lack of",
    "absence of",
    "could not",
    "did not",
    "not generalizable",
    "missing data",
    "incomplete data",
    "should be interpreted with caution",
)

SIGNIFICANCE_CUES = (
    "clinical",
    "causal",
    "mechanism",
    "biomarker",
    "therapeutic",
    "treatment",
    "prediction",
    "risk",
    "disease",
    "longitudinal",
    "multi-omic",
    "metagenomic",
)


def document_id(record: dict) -> str:
    """Return a stable document identifier from DOI, title, or text."""

    return stable_id(
        "doc",
        record.get("doi", ""),
        record.get("title", ""),
        record.get("abstract", ""),
    )


def statement_to_question(sentence: str) -> str:
    """Convert a future-direction or limitation statement into a question."""

    sentence = normalize_text(sentence).strip()
    if not sentence:
        return ""
    if sentence.endswith("?"):
        return sentence

    # Prefer narrow rewrites for common future-direction phrasing.
    lowered = sentence.lower().rstrip(".")
    replacements = [
        (r"^future studies should examine\s+", "How does "),
        (r"^future work should examine\s+", "How does "),
        (r"^further research should examine\s+", "How does "),
        (r"^future studies are needed to determine\s+", "Can available data determine "),
        (r"^further studies are needed to determine\s+", "Can available data determine "),
        (r"^it remains unclear whether\s+", "Does "),
        (r"^it is unclear whether\s+", "Does "),
        (r"^whether\s+", "Does "),
    ]
    for pattern, prefix in replacements:
        if re.search(pattern, lowered):
            tail = re.sub(pattern, "", lowered).strip(" .")
            if tail:
                question = prefix + tail
                return question[0].upper() + question[1:].rstrip(".") + "?"

    if "small sample" in lowered:
        return "Can the reported finding be validated in a larger independent cohort?"
    if (
        "lack of" in lowered
        and any(term in lowered for term in ("sample", "observation", "observations"))
        and any(term in lowered for term in ("imbalanced", "positive", "negative", "label"))
    ):
        return (
            "Can the reported finding be validated with larger, better-balanced "
            "labeled samples?"
        )
    if "missing data" in lowered or "incomplete data" in lowered:
        return "Would more complete public data resolve the reported limitation?"

    # The fallback keeps questionable statements visible for expert review.
    return f"What data would be needed to evaluate whether {lowered}?"


def classify_sentence(sentence: str) -> tuple[str, float, list[str]]:
    """Return question origin, extraction confidence, and fired rule signals."""

    lowered = sentence.lower()
    fired: list[str] = []
    question_origin = ""

    # Multiple cues can fire; ``fired`` preserves why the sentence was kept.
    if sentence.strip().endswith("?") or any(cue in lowered for cue in RQ_CUES):
        question_origin = "explicit_question"
        fired.append("explicit_rq")
    if any(cue in lowered for cue in FUTURE_CUES):
        question_origin = "future_direction"
        fired.append("future_direction")
    if any(cue in lowered for cue in LIMITATION_CUES):
        if not question_origin:
            question_origin = "limitation_derived"
        fired.append("limitation")

    if not question_origin:
        return "", 0.0, []

    extraction_confidence = min(0.95, 0.5 + 0.15 * len(fired))
    if "future_direction" in fired and "explicit_rq" in fired:
        extraction_confidence += 0.05
    return question_origin, min(extraction_confidence, 0.95), fired


def score_significance(text: str, evidence_count: int = 1) -> float:
    """Estimate scientific significance from lexical cues and recurrence."""

    lowered = text.lower()
    cue_score = sum(1 for cue in SIGNIFICANCE_CUES if cue in lowered) / len(SIGNIFICANCE_CUES)
    recurrence = min(1.0, evidence_count / 5.0)
    specificity = min(1.0, len(extract_domain_terms(text, max_terms=20)) / 12.0)
    return round(0.35 + 0.35 * cue_score + 0.2 * recurrence + 0.1 * specificity, 3)


def extract_question_candidates(record: dict) -> list[QuestionCandidate]:
    """Extract candidate open questions from one literature record.

    Parameters
    ----------
    record:
        Dictionary with at least one of ``title``, ``abstract``, or ``text``.

    Returns
    -------
    list[QuestionCandidate]
        Candidate questions with evidence, required variables, and scores.
    """

    title = normalize_text(record.get("title", ""))
    abstract = normalize_text(record.get("abstract", ""))
    body = str(record.get("text", "") or "")
    doi = normalize_text(record.get("doi", ""))
    source_id = record.get("source_id") or document_id(record)
    # Preserve body newlines so section headings remain visible to split_sections().
    full_text = "\n".join(part for part in (abstract, body) if normalize_text(part))
    if not full_text:
        full_text = title
    sections = split_sections(full_text)

    candidates: list[QuestionCandidate] = []
    for section_name, section_text in sections.items():
        # Methods text often describes procedures rather than open scientific questions.
        if section_name == "methods":
            continue
        for idx, sentence in enumerate(split_sentences(section_text)):
            if len(sentence) < 35 or len(sentence) > 450:
                continue
            question_origin, extraction_confidence, signals = classify_sentence(sentence)
            if not question_origin:
                continue
            question = statement_to_question(sentence)
            if not question:
                continue
            if should_reject_question_candidate(
                sentence,
                question,
                section_name=section_name,
                signals=signals,
            ):
                continue

            inference_text = _inference_context(title, abstract, sentence)
            combined_text = f"{title} {sentence}"
            # Variables and terms are the handoff from literature extraction to matching.
            variables = infer_required_variables(inference_text)
            terms = extract_domain_terms(combined_text)
            population = infer_population(inference_text)
            outcomes = infer_outcomes(inference_text)
            inference_notes = _variable_inference_notes(
                inference_text,
                variables=variables,
                population=population,
                outcomes=outcomes,
            )
            evidence = Evidence(
                text=sentence,
                source_id=source_id,
                title=title,
                doi=doi,
                section=section_name,
                sentence_index=idx,
                extraction_method="+".join(signals),
                extraction_confidence=extraction_confidence,
            )
            question_id = stable_id("question", question, source_id)
            # QuestionCandidate is the canonical object consumed by downstream nodes.
            candidates.append(
                QuestionCandidate(
                    question_id=question_id,
                    question=question,
                    source_ids=[source_id],
                    evidence=[evidence],
                    question_origin=question_origin,
                    field=record.get("field", "biomedical"),
                    domain_terms=terms,
                    required_variables=variables,
                    population=population,
                    outcomes=outcomes,
                    extraction_confidence=extraction_confidence,
                    novelty_score=0.55 if question_origin == "future_direction" else 0.45,
                    significance_score=score_significance(combined_text),
                    answerability=0.55 if variables else 0.35,
                    metadata={
                        "signals": signals,
                        "title": title,
                        "doi": doi,
                        "document_id": record.get("document_id", ""),
                        "variable_inference": inference_notes,
                        # Content scope follows the question into review so abstract-only
                        # candidates are not interpreted like full-text-derived ones.
                        "source_provenance": _record_source_provenance(record),
                    },
                )
            )
    return deduplicate_questions(candidates)


def should_reject_question_candidate(
    sentence: str,
    question: str,
    section_name: str = "",
    signals: list[str] | None = None,
) -> bool:
    """Return true for conservative, review-driven question false positives."""

    signals = signals or []
    if section_name in {"references", "bibliography"}:
        return True
    if _looks_like_reference_title_question(sentence, question, signals):
        return True
    if _has_no_meaningful_question_context(sentence, signals):
        return True
    return False


def _looks_like_reference_title_question(
    sentence: str, question: str, signals: list[str]
) -> bool:
    """Detect bare citation-title questions without suppressing grounded cues."""

    if signals != ["explicit_rq"]:
        return False
    text = normalize_text(sentence).rstrip(".")
    question_text = normalize_text(question)
    lowered = text.lower()
    if any(cue in lowered for cue in RQ_CUES + FUTURE_CUES + LIMITATION_CUES):
        return False
    if ":" in text and text.endswith("?") and len(text.split()) <= 18:
        return True
    if re.match(r"^[A-Z][A-Za-z0-9 ,;/()'-]{12,120}\?$", text):
        return True
    if text == question_text and len(text.split()) <= 14:
        return True
    return False


def _has_no_meaningful_question_context(sentence: str, signals: list[str]) -> bool:
    """Reject explicit-only fragments that read like standalone titles."""

    if signals != ["explicit_rq"]:
        return False
    lowered = normalize_text(sentence).lower()
    if any(cue in lowered for cue in RQ_CUES):
        return False
    if len(lowered.split()) < 6:
        return True
    return False


def _inference_context(title: str, abstract: str, sentence: str) -> str:
    """Build deterministic context used only for fallback variable inference."""

    return normalize_text(f"{title} {abstract} {sentence}")


def _variable_inference_notes(
    text: str,
    variables: list[str],
    population: str,
    outcomes: list[str],
) -> dict[str, object]:
    """Record why fallback inference did or did not emit structured fields."""

    notes: dict[str, object] = {
        "method": "title_abstract_evidence_lexical_fallback",
        "context_terms": extract_domain_terms(text, max_terms=8),
    }
    if variables:
        notes["required_variables_reason"] = "lexical_or_ontology_cues_found"
    else:
        notes["required_variables_reason"] = "no_defensible_local_cues_found"
    notes["population_reason"] = (
        "population_cue_found" if population else "no_defensible_population_cue_found"
    )
    notes["outcomes_reason"] = (
        "outcome_cue_found" if outcomes else "no_defensible_outcome_cue_found"
    )
    return notes


def analyze_literature_records(records: Iterable[dict]) -> list[QuestionCandidate]:
    """Run open-question identification over an iterable of literature records."""

    all_candidates: list[QuestionCandidate] = []
    for record in records:
        all_candidates.extend(extract_question_candidates(record))
    return deduplicate_questions(all_candidates)


def _record_source_provenance(record: dict) -> dict:
    """Return source provenance from a literature record or nested metadata."""

    provenance = record.get("source_provenance", {})
    if isinstance(provenance, dict) and provenance:
        return provenance
    metadata = record.get("metadata", {})
    if isinstance(metadata, dict):
        nested = metadata.get("source_provenance", {})
        if isinstance(nested, dict):
            return nested
    return {}


def deduplicate_questions(
    candidates: Iterable[QuestionCandidate], similarity_threshold: float = 0.78
) -> list[QuestionCandidate]:
    """Merge near-duplicate questions while preserving provenance."""

    merged: list[QuestionCandidate] = []
    by_norm: dict[str, QuestionCandidate] = {}
    for candidate in candidates:
        norm = candidate.normalized_question
        if norm in by_norm:
            by_norm[norm].merge(candidate)
            continue
        matched = None
        # Near-duplicate questions are merged so repeated evidence strengthens one record.
        for existing in merged:
            if lexical_similarity(existing.normalized_question, norm) >= similarity_threshold:
                matched = existing
                break
        if matched:
            matched.merge(candidate)
            by_norm[norm] = matched
        else:
            merged.append(candidate)
            by_norm[norm] = candidate

    for candidate in merged:
        evidence_count = max(1, len(candidate.evidence))
        candidate.significance_score = max(
            candidate.significance_score,
            score_significance(candidate.question, evidence_count=evidence_count),
        )
    return merged


def analyze_topic_signal(topic: str, context: str = "") -> dict:
    """Create a lightweight literature signal for streaming worker mode."""

    text = normalize_text(f"{topic} {context}")
    terms = extract_domain_terms(text, max_terms=8)
    variables = infer_required_variables(text)
    significance = score_significance(text, evidence_count=1)
    specificity = min(1.0, len(terms) / 8.0)
    return {
        "topic": topic,
        "counts": [max(1, int(20 * (idx + 1) * specificity)) for idx in range(5)],
        "significance_score": round(max(significance, 0.35 + 0.25 * specificity), 3),
        "top_terms": terms,
        "required_variables": variables,
    }


def group_questions_by_variable(questions: Iterable[QuestionCandidate]) -> dict[str, list[str]]:
    """Return a variable-to-question index for inspection and reporting."""

    index: dict[str, list[str]] = defaultdict(list)
    for question in questions:
        for variable in question.required_variables:
            index[variable].append(question.question_id)
    return dict(index)
