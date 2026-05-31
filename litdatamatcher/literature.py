"""Literature analysis and open-question identification node."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Iterable

from .schemas import Evidence, QuestionCandidate, stable_id
from .text import (
    extract_domain_terms,
    infer_population,
    infer_required_variables,
    lexical_similarity,
    normalize_text,
    split_sections,
    split_sentences,
)


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
    if "missing data" in lowered or "incomplete data" in lowered:
        return "Would more complete public data resolve the reported limitation?"

    return f"What data would be needed to evaluate whether {lowered}?"


def classify_sentence(sentence: str) -> tuple[str, float, list[str]]:
    """Classify a sentence as explicit RQ, future direction, limitation, or none."""

    lowered = sentence.lower()
    fired: list[str] = []
    extraction_type = ""

    if sentence.strip().endswith("?") or any(cue in lowered for cue in RQ_CUES):
        extraction_type = "explicit_research_question"
        fired.append("explicit_rq")
    if any(cue in lowered for cue in FUTURE_CUES):
        extraction_type = "open_question"
        fired.append("future_direction")
    if any(cue in lowered for cue in LIMITATION_CUES):
        if not extraction_type:
            extraction_type = "limitation_derived_question"
        fired.append("limitation")

    if not extraction_type:
        return "", 0.0, []

    confidence = min(0.95, 0.5 + 0.15 * len(fired))
    if "future_direction" in fired and "explicit_rq" in fired:
        confidence += 0.05
    return extraction_type, min(confidence, 0.95), fired


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
    body = normalize_text(record.get("text", ""))
    doi = normalize_text(record.get("doi", ""))
    source_id = record.get("source_id") or document_id(record)
    full_text = "\n".join(part for part in (abstract, body) if part)
    if not full_text:
        full_text = title
    sections = split_sections(full_text)

    candidates: list[QuestionCandidate] = []
    for section_name, section_text in sections.items():
        if section_name == "methods":
            continue
        for idx, sentence in enumerate(split_sentences(section_text)):
            if len(sentence) < 35 or len(sentence) > 450:
                continue
            extraction_type, confidence, signals = classify_sentence(sentence)
            if not extraction_type:
                continue
            question = statement_to_question(sentence)
            if not question:
                continue

            combined_text = f"{title} {sentence}"
            variables = infer_required_variables(combined_text)
            terms = extract_domain_terms(combined_text)
            population = infer_population(combined_text)
            evidence = Evidence(
                text=sentence,
                source_id=source_id,
                title=title,
                doi=doi,
                section=section_name,
                sentence_index=idx,
                extraction_method="+".join(signals),
                confidence=confidence,
            )
            question_id = stable_id("question", question, source_id)
            candidates.append(
                QuestionCandidate(
                    question_id=question_id,
                    question=question,
                    source_ids=[source_id],
                    evidence=[evidence],
                    extraction_type=extraction_type,
                    field=record.get("field", "biomedical"),
                    domain_terms=terms,
                    required_variables=variables,
                    population=population,
                    confidence=confidence,
                    novelty_score=0.55 if extraction_type == "open_question" else 0.45,
                    significance_score=score_significance(combined_text),
                    answerability_hint=0.55 if variables else 0.35,
                    metadata={"signals": signals, "title": title, "doi": doi},
                )
            )
    return deduplicate_questions(candidates)


def analyze_literature_records(records: Iterable[dict]) -> list[QuestionCandidate]:
    """Run open-question identification over an iterable of literature records."""

    all_candidates: list[QuestionCandidate] = []
    for record in records:
        all_candidates.extend(extract_question_candidates(record))
    return deduplicate_questions(all_candidates)


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
