"""Meta-analysis style clustering and evidence synthesis node."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .schemas import EvidenceSynthesis, QuestionCandidate, stable_id
from .text import lexical_similarity


def cluster_questions(
    questions: Iterable[QuestionCandidate], similarity_threshold: float = 0.62
) -> list[list[QuestionCandidate]]:
    """Cluster questions that appear to address the same research gap."""

    clusters: list[list[QuestionCandidate]] = []
    for question in questions:
        placed = False
        for cluster in clusters:
            representative = cluster[0]
            # Use the first question as a stable representative for deterministic clustering.
            similarity = lexical_similarity(
                question.normalized_question, representative.normalized_question
            )
            shared_variables = set(question.required_variables) & set(
                representative.required_variables
            )
            # Shared required variables can join moderately similar wording into one gap.
            if similarity >= similarity_threshold or (similarity >= 0.35 and shared_variables):
                cluster.append(question)
                placed = True
                break
        if not placed:
            clusters.append([question])
    return clusters


def synthesize_question_cluster(cluster: list[QuestionCandidate]) -> EvidenceSynthesis:
    """Create a cluster-level evidence synthesis record."""

    question_ids = [question.question_id for question in cluster]
    all_evidence = [evidence for question in cluster for evidence in question.evidence]
    source_ids = {source for question in cluster for source in question.source_ids}
    terms: dict[str, int] = defaultdict(int)
    for question in cluster:
        for term in question.domain_terms:
            terms[term] += 1
    # Top terms make the synthesis summary inspectable without re-reading every sentence.
    top_terms = [term for term, _ in sorted(terms.items(), key=lambda item: item[1], reverse=True)[:5]]

    support_count = len(all_evidence)
    # Recurrence is a source-diversity signal, not a formal meta-analytic effect size.
    recurrence_score = min(1.0, len(source_ids) / 5.0)
    mean_extraction_confidence = (
        sum(question.extraction_confidence for question in cluster) / len(cluster)
        if cluster
        else 0.0
    )
    # Evidence strength blends extraction confidence with independent-source recurrence.
    evidence_strength = min(1.0, 0.5 * mean_extraction_confidence + 0.5 * recurrence_score)
    uncertainty = max(0.05, 1.0 - evidence_strength)
    summary = (
        f"{len(cluster)} question candidate(s) across {len(source_ids)} source(s)"
        + (f" involving {', '.join(top_terms)}." if top_terms else ".")
    )
    return EvidenceSynthesis(
        cluster_id=stable_id("cluster", *question_ids),
        question_ids=question_ids,
        summary=summary,
        support_count=support_count,
        contradiction_count=0,
        recurrence_score=round(recurrence_score, 3),
        evidence_strength=round(evidence_strength, 3),
        uncertainty=round(uncertainty, 3),
        metadata={"top_terms": top_terms, "source_count": len(source_ids)},
    )


def run_meta_analysis_node(
    questions: Iterable[QuestionCandidate], similarity_threshold: float = 0.62
) -> list[EvidenceSynthesis]:
    """Cluster questions and return evidence-synthesis records."""

    return [
        synthesize_question_cluster(cluster)
        for cluster in cluster_questions(questions, similarity_threshold=similarity_threshold)
    ]


def synthesis_index(syntheses: Iterable[EvidenceSynthesis]) -> dict[str, EvidenceSynthesis]:
    """Map each question ID to its synthesis cluster."""

    index: dict[str, EvidenceSynthesis] = {}
    for synthesis in syntheses:
        for question_id in synthesis.question_ids:
            # Ranking looks up each question's cluster-level evidence through this index.
            index[question_id] = synthesis
    return index
