"""Question-to-data ranking node."""

from __future__ import annotations

import math
from typing import Iterable

from .feasibility import assess_pair_feasibility
from .governance import assess_governance
from .schemas import DatasetRecord, EvidenceSynthesis, MatchCandidate, MatchScore, QuestionCandidate, stable_id
from .text import extract_domain_terms, lexical_similarity


def _sample_adequacy(sample_size: int) -> float:
    """Map sample size to a bounded adequacy score."""

    if sample_size <= 0:
        return 0.0
    return min(1.0, math.log10(sample_size + 1) / 4.0)


def _population_fit(question: QuestionCandidate, dataset: DatasetRecord) -> float:
    """Score compatibility between requested and available population metadata."""

    if not question.population:
        return 0.5
    pops = {item.lower() for item in dataset.populations}
    if question.population.lower() in pops:
        return 1.0
    if question.population == "human" and {"adult", "pediatric", "infant"} & pops:
        return 0.8
    return 0.2 if pops else 0.4


def _variable_overlap(question: QuestionCandidate, dataset: DatasetRecord) -> tuple[float, list[str]]:
    """Return required-variable coverage and missing variables."""

    feasibility = assess_pair_feasibility(question, dataset)
    return feasibility.variable_coverage, feasibility.missing_variables


def score_question_dataset(
    question: QuestionCandidate,
    dataset: DatasetRecord,
    synthesis: EvidenceSynthesis | None = None,
) -> tuple[MatchScore, list[str], list[str]]:
    """Compute an explainable composite score for one question-dataset pair."""

    variable_overlap, missing = _variable_overlap(question, dataset)
    feasibility_assessment = assess_pair_feasibility(question, dataset)
    governance = assess_governance(dataset)
    semantic_relevance = max(
        lexical_similarity(question.question, dataset.searchable_text()),
        lexical_similarity(" ".join(question.domain_terms), dataset.searchable_text()),
    )
    dataset_terms = set(extract_domain_terms(dataset.searchable_text(), max_terms=20))
    if question.domain_terms:
        semantic_relevance = max(
            semantic_relevance,
            len(set(question.domain_terms) & dataset_terms) / max(1, len(set(question.domain_terms))),
        )
    population_fit = feasibility_assessment.population_fit
    sample_adequacy = feasibility_assessment.sample_adequacy
    significance = question.significance_score
    if synthesis:
        significance = max(significance, synthesis.evidence_strength)
        uncertainty = synthesis.uncertainty
    else:
        uncertainty = 0.35
    feasibility = max(
        feasibility_assessment.overall,
        0.35 * variable_overlap
        + 0.2 * population_fit
        + 0.2 * dataset.quality_score
        + 0.15 * sample_adequacy
        + 0.1 * semantic_relevance,
    )
    uncertainty_penalty = min(0.6, 0.35 * uncertainty + 0.15 * len(missing))
    combined = (
        0.32 * significance
        + 0.28 * feasibility
        + 0.16 * variable_overlap
        + 0.1 * semantic_relevance
        + 0.08 * dataset.quality_score
        + 0.06 * sample_adequacy
        - 0.18 * uncertainty_penalty
    )
    score = MatchScore(
        variable_overlap=round(variable_overlap, 3),
        semantic_relevance=round(semantic_relevance, 3),
        population_fit=round(population_fit, 3),
        data_quality=round(dataset.quality_score, 3),
        sample_adequacy=round(sample_adequacy, 3),
        significance=round(significance, 3),
        feasibility=round(feasibility, 3),
        uncertainty_penalty=round(uncertainty_penalty, 3),
        combined=round(combined, 3),
        governance=round(governance.reuse_score, 3),
        design_fit=round(
            0.5 * feasibility_assessment.assay_fit
            + 0.5 * feasibility_assessment.longitudinal_fit,
            3,
        ),
    )
    rationale = [
        f"variable overlap {score.variable_overlap:.2f}",
        f"semantic relevance {score.semantic_relevance:.2f}",
        f"population fit {score.population_fit:.2f}",
        f"dataset quality {score.data_quality:.2f}",
        f"sample adequacy {score.sample_adequacy:.2f}",
        f"governance reuse {score.governance:.2f}",
        f"recommended design: {feasibility_assessment.recommended_design}",
    ]
    if synthesis:
        rationale.append(
            f"literature recurrence {synthesis.recurrence_score:.2f} with uncertainty {synthesis.uncertainty:.2f}"
        )
    if missing:
        rationale.append(f"missing variables: {', '.join(missing)}")
    assessments = {
        "feasibility": feasibility_assessment.to_dict(),
        "governance": governance.to_dict(),
    }
    return score, rationale, missing, assessments


def rank_matches(
    questions: Iterable[QuestionCandidate],
    datasets: Iterable[DatasetRecord],
    syntheses_by_question: dict[str, EvidenceSynthesis] | None = None,
    top_n: int = 100,
) -> list[MatchCandidate]:
    """Rank all question-dataset pairs and return the top opportunities."""

    syntheses_by_question = syntheses_by_question or {}
    matches: list[MatchCandidate] = []
    for question in questions:
        synthesis = syntheses_by_question.get(question.question_id)
        for dataset in datasets:
            score, rationale, missing, assessments = score_question_dataset(question, dataset, synthesis)
            if score.combined <= 0:
                continue
            matches.append(
                MatchCandidate(
                    match_id=stable_id("match", question.question_id, dataset.dataset_id),
                    question=question,
                    dataset=dataset,
                    score=score,
                    rationale=rationale,
                    missing_variables=missing,
                    assessments=assessments,
                )
            )
    matches.sort(key=lambda item: item.score.combined, reverse=True)
    return matches[:top_n]
