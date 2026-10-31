"""Source-linked question discovery and conservative cross-document organization.

The deterministic mode selects exact sentences; it is never reported as model
inference. A supplied LocalSemanticRuntime keeps its own source guard and cache.
Clustering and addressing checks are review aids, not scientific novelty labels.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .data_plane import digest
from .question_compiler import compile_question
from .semantic_runtime import validate_extraction
from .structured_decoding import complete_source_sentences

VERSION = "question-discovery-1.0"
_UNRESOLVED = re.compile(
    r"\b(?:future (?:studies|research|work|investigations|trials)|"
    r"(?:remains?|is) (?:unclear|unknown|unresolved|uncertain)|not (?:yet )?known|"
    r"further (?:studies|research)|warrants? (?:further )?(?:investigation|study|research))\b",
    re.I,
)
_LIMITATION = re.compile(
    r"\b(?:limitations?|limited by|unable to (?:assess|determine|establish)|"
    r"could not (?:assess|determine|establish)|not generalizable)\b",
    re.I,
)
_INSTRUCTION = re.compile(
    r"\b(?:ignore (?:all|previous|prior)|system prompt|reveal (?:secrets|tokens)|"
    r"send (?:credentials|password)|execute (?:code|command))\b",
    re.I,
)
_RESULT = re.compile(
    r"\b(?:found|observed|demonstrated|showed|reported|associated|correlated|"
    r"increased|decreased|reduced|improved|no (?:significant )?(?:difference|effect))\b",
    re.I,
)
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "this",
        "to",
        "was",
        "were",
        "whether",
        "which",
        "with",
        "would",
        "should",
        "further",
        "future",
        "study",
        "studies",
        "research",
        "remains",
        "remain",
        "unknown",
        "unclear",
        "investigate",
        "examine",
        "needed",
    ]
)
_DIRECTION = re.compile(
    r"\b(?:no|not|without|increase\w*|decrease\w*|reduc\w*|improv\w*|"
    r"higher|lower|inhibit\w*|activat\w*|\d+(?:\.\d+)?)\b",
    re.I,
)


def _bounded_int(value: int, name: str, maximum: int) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _document(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Each document must be an object")
    identifier = raw.get("document_id") or raw.get("id") or raw.get("source_id")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("Each document needs a stable document_id, id, or source_id")
    text = raw.get("text") or raw.get("abstract")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Each document needs nonempty text or abstract")
    if len(text) > 200_000:
        raise ValueError("Supply a bounded source view of at most 200,000 characters")
    provenance = raw.get("source_provenance") or {}
    if not isinstance(provenance, dict):
        raise ValueError("source_provenance must be an object")
    locator = (
        raw.get("source_locator")
        or raw.get("url")
        or provenance.get("source_locator")
        or identifier
    )
    if not isinstance(locator, str) or not locator.strip():
        raise ValueError("source_locator must be nonempty text")
    return {
        **raw,
        "document_id": identifier,
        "text": text,
        "source_locator": locator,
        "source_provenance": deepcopy(provenance),
    }


def _sentences(document: dict) -> list[dict]:
    text = document["text"]
    return [
        {
            "text": sentence,
            "start": text.index(sentence),
            "end": text.index(sentence) + len(sentence),
        }
        for sentence in complete_source_sentences(text)
    ]


def _category(text: str) -> str | None:
    if _INSTRUCTION.search(text):
        return None
    if _UNRESOLVED.search(text):
        return "SOURCE_STATED_UNRESOLVED"
    if _LIMITATION.search(text) and not re.search(
        r"\b(?:no limitations?|without limitations?|not limited)\b", text, re.I
    ):
        return "SOURCE_STATED_LIMITATION"
    if text.endswith("?"):
        return "SOURCE_RESEARCH_QUESTION"
    return None


def normalize_question(text: str) -> str:
    """Normalize presentation only; negation, entities and numbers are preserved."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Nonempty question text required")
    return " ".join(text.split()).casefold().rstrip(".?! ")


def _terms(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[\w-]+", text.casefold())
        if word not in _STOP and len(word) > 1
    }


def _designs(compilation: dict) -> list[dict]:
    """Offer reviewable alternatives without activating new requirements."""
    templates = {
        "comparative": [
            (
                "parallel-group",
                "Between-group contrast",
                ["Group assignment and confounding require review."],
            ),
            (
                "paired-comparison",
                "Within-unit contrast",
                ["Verified unit pairing and appropriate ordering are required."],
            ),
        ],
        "associational": [
            (
                "prospective-cohort",
                "Prospective exposure-outcome association",
                ["Exposure must precede outcome; confounders require review."],
            ),
            (
                "retrospective-cohort",
                "Retrospective exposure-outcome association",
                ["Reliable exposure timing, outcome ascertainment and confounders require review."],
            ),
        ],
        "prediction": [
            (
                "held-out-cohort",
                "Independent cohort validation",
                [
                    "Predictors must precede the target; independent validation requires verification."
                ],
            ),
            (
                "nested-validation",
                "Nested resampling within a cohort",
                [
                    "Partitions must respect subject dependence and prevent feature-selection leakage."
                ],
            ),
        ],
        "longitudinal_change": [
            (
                "paired-change",
                "Paired baseline-to-follow-up change",
                ["The same independent units must be measured at both times."],
            ),
            (
                "repeated-trajectory",
                "Repeated-measure trajectory",
                ["Ordered times, repeated unit identifiers and an appropriate model are required."],
            ),
        ],
    }
    return [
        {
            "design_id": design_id,
            "description": description,
            "status": "PROPOSED_UNASSESSED",
            "provenance_category": "METHOD_DERIVED",
            "assumptions_requiring_review": assumptions,
            "changes_active_requirements": False,
        }
        for design_id, description, assumptions in templates.get(
            compilation["question_purpose"], []
        )
    ]


def compile_research_question(
    question: str,
    *,
    source_text: str | None = None,
    source_locator: str = "user question",
    user_constraints: list[dict[str, Any]] | None = None,
) -> dict:
    """Expose the existing scientific contract plus explicit soft preferences.

    The established compiler's essential requirements and status are unchanged.
    A preference is assessable but cannot turn an essential mismatch into a fit.
    Proposed designs are unassessed options, never silently selected designs.
    """
    if user_constraints is not None:
        if not isinstance(user_constraints, list) or len(user_constraints) > 64:
            raise ValueError("user_constraints must be a list of at most 64 requirements")
        # Fail before accepting NaN/Infinity or non-JSON Python objects as evidence.
        json.dumps(user_constraints, allow_nan=False)
    compilation = compile_question(
        question,
        source_text=source_text,
        source_locator=source_locator,
        user_constraints=user_constraints,
    )
    assessable = []
    if compilation["compilation_status"] != "QUESTION_UNDERDETERMINED":
        for item in compilation["requirements"]:
            if item["effective"]:
                assessable.append(
                    {key: item[key] for key in ("field", "expected", "essential", "source_locator")}
                )
    return {
        **compilation,
        "matching_requirements": assessable,
        "proposed_designs": _designs(compilation),
        "design_selection_status": "USER_REVIEW_REQUIRED"
        if _designs(compilation)
        else "NO_ALTERNATIVES_PROPOSED",
    }


def _record(document: dict, span: dict, method: str, manifest: dict | None = None) -> dict:
    text = span["text"]
    category = _category(text)
    if category is None:
        raise ValueError("No qualifying scientific question or limitation cue")
    source_hash = _text_sha(document["text"])
    identifier = "rq_" + digest([document["document_id"], source_hash, span])[:24]
    locator = f"{document['source_locator']}#chars={span['start']}:{span['end']}"
    return {
        "question_id": identifier,
        "question": text,
        "text": text,
        "normalized_question": normalize_question(text),
        "origin": category,
        "category": category,
        "extraction_method": method,
        "source_document_id": document["document_id"],
        "source_locator": locator,
        "source_text_sha256": source_hash,
        "source_provenance": document["source_provenance"],
        "evidence_span": {**span, "source_locator": document["source_locator"]},
        "citation": {
            key: document.get(key) for key in ("title", "doi", "pmid", "pmcid", "url", "year")
        },
        "unanswered_status": "SOURCE_STATES_UNRESOLVED"
        if category == "SOURCE_STATED_UNRESOLVED"
        else "NOT_ESTABLISHED",
        "novelty_status": "NOT_ESTABLISHED",
        "model_inferred": False,
        "requires_review": True,
        "compilation": compile_research_question(
            text, source_text=document["text"], source_locator=locator
        ),
        "inference_manifest": deepcopy(manifest),
    }


def _cluster_signature(question: dict) -> tuple:
    roles = question.get("compilation", {}).get("entity_roles", [])
    return tuple(
        sorted((role["field"], normalize_question(str(role["expected"]))) for role in roles)
    )


def cluster_questions(questions: list[dict], *, similarity_threshold: float = 0.9) -> list[dict]:
    """Group exact/near duplicates without replacing source-specific questions.

    The numeric threshold is an explicit engineering review threshold, not a
    validated scientific equivalence or probability. Representative-only
    assignment avoids a chain of weak similarities merging distinct questions.
    """
    if type(similarity_threshold) not in (int, float) or not 0.8 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be in 0.8..1")
    if not isinstance(questions, list) or len(questions) > 1000:
        raise ValueError("At most 1000 questions can be clustered per request")
    groups: list[list[dict]] = []
    for question in sorted(
        questions, key=lambda row: (normalize_question(row["text"]), row["question_id"])
    ):
        normalized = normalize_question(question["text"])
        for group in groups:
            representative = group[0]
            other = normalize_question(representative["text"])
            if question["category"] != representative["category"]:
                continue
            exact = normalized == other
            same_roles = _cluster_signature(question) == _cluster_signature(representative)
            same_polarity = _DIRECTION.findall(normalized) == _DIRECTION.findall(other)
            if exact or (
                same_roles
                and same_polarity
                and SequenceMatcher(None, normalized, other, autojunk=False).ratio()
                >= similarity_threshold
            ):
                group.append(question)
                break
        else:
            groups.append([question])
    return [
        {
            "cluster_id": "qcl_" + digest(sorted(q["question_id"] for q in group))[:24],
            "representative_question_id": group[0]["question_id"],
            "question_ids": [q["question_id"] for q in group],
            "source_document_ids": sorted({q["source_document_id"] for q in group}),
            "relation": "EXACT_NORMALIZED"
            if len({normalize_question(q["text"]) for q in group}) == 1
            else "NEAR_DUPLICATE_REVIEW",
            "scientific_equivalence": "NOT_ESTABLISHED",
            "independent_evidence_count": None,
            "similarity_threshold": similarity_threshold,
        }
        for group in groups
    ]


def discover_questions(
    documents: list[dict],
    *,
    runtime=None,
    cache_root: str | Path | None = None,
    max_documents: int = 100,
    max_questions: int = 200,
) -> dict:
    """Extract, compile and group bounded literature views with honest coverage.

    A model failure is retained as a failure and does not trigger an unreported
    heuristic fallback. Empty accepted output is distinct from failed extraction.
    Text views exceeding the source selector's 128-sentence bound need explicit
    upstream windows; failures and global result truncation appear in coverage.
    """
    _bounded_int(max_documents, "max_documents", 1000)
    _bounded_int(max_questions, "max_questions", 1000)
    if not isinstance(documents, list) or len(documents) > max_documents:
        raise ValueError("documents must be a list within max_documents")
    prepared = [_document(raw) for raw in documents]
    ids = [doc["document_id"] for doc in prepared]
    if len(ids) != len(set(ids)):
        raise ValueError("Document IDs must be unique; resolve revisions before discovery")
    method = (
        "local_model_source_guard" if runtime is not None else "deterministic_source_sentences_v1"
    )
    questions, statuses = [], []
    for document in prepared:
        identifier = document["document_id"]
        try:
            spans = _sentences(document)
            manifest = None
            rejected = []
            if runtime is None:
                selected = [span for span in spans if _category(span["text"]) is not None]
            else:
                result = runtime.extract(document, cache_root=cache_root)
                if not isinstance(result, dict) or not isinstance(result.get("questions"), list):
                    raise ValueError("Runtime did not return a questions list")
                if not all(isinstance(row, dict) for row in result["questions"]):
                    raise ValueError("Runtime question entries must be objects")
                manifest = result.get("inference_manifest")
                if not isinstance(manifest, dict) or not manifest.get("model_id"):
                    raise ValueError(
                        "Model extraction requires an inference manifest and model identity"
                    )
                quotes = [
                    {"quote": row.get("text", row.get("quote"))} for row in result["questions"]
                ]
                guarded = validate_extraction({"claims": [], "questions": quotes}, document)
                selected = [
                    row["evidence_span"]
                    for row in guarded["questions"]
                    if _category(row["text"]) is not None
                ]
                rejected = [*result.get("rejected", []), *guarded["rejected"]]
            records = [_record(document, span, method, manifest) for span in selected]
            records = list({row["question_id"]: row for row in records}.values())
            remaining = max_questions - len(questions)
            questions.extend(records[:remaining])
            statuses.append(
                {
                    "document_id": identifier,
                    "status": "SUCCESS" if records else "EMPTY",
                    "source_text_sha256": _text_sha(document["text"]),
                    "candidate_count": len(records),
                    "retained_count": min(remaining, len(records)),
                    "rejected": rejected,
                    "inference_manifest": manifest,
                }
            )
        except (ValueError, TypeError, KeyError, RuntimeError, OSError) as exc:
            statuses.append(
                {
                    "document_id": identifier,
                    "status": "EXTRACTION_FAILURE",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "retained_count": 0,
                }
            )
    candidate_count = sum(status.get("candidate_count", 0) for status in statuses)
    failures = sum(status["status"] == "EXTRACTION_FAILURE" for status in statuses)
    return {
        "schema_version": VERSION,
        "extraction_method": method,
        "questions": questions,
        "clusters": cluster_questions(questions),
        "document_statuses": statuses,
        "coverage": {
            "requested_documents": len(documents),
            "processed_documents": len(statuses) - failures,
            "failed_documents": failures,
            "candidate_questions": candidate_count,
            "retained_questions": len(questions),
            "truncated": candidate_count > len(questions),
            "source_scope": "Caller-supplied source views only",
            "global_novelty_established": False,
        },
        "status": "PARTIAL" if failures or candidate_count > len(questions) else "SUCCESS",
    }


def check_addressed_evidence(
    question_record: dict | str, documents: list[dict], *, limit: int = 5
) -> dict:
    """Return answer-bearing related passages for review, without deciding novelty.

    This lexical screen never asserts that a related finding answers the exact
    question. Same-source findings are useful: an interrogative title can be
    answered within the article that contains it.
    """
    _bounded_int(limit, "limit", 100)
    if not isinstance(documents, list) or len(documents) > 1000:
        raise ValueError("At most 1000 provided documents can be checked")
    question = (
        question_record
        if isinstance(question_record, str)
        else question_record.get("text") or question_record.get("question")
    )
    terms = _terms(normalize_question(question))
    candidates, failures, checked = [], [], []
    for raw in documents:
        doc = _document(raw)
        try:
            spans = _sentences(doc)
        except ValueError as exc:
            failures.append({"document_id": doc["document_id"], "error": str(exc)})
            continue
        checked.append(doc["document_id"])
        for span in spans:
            if _category(span["text"]) is not None or not _RESULT.search(span["text"]):
                continue
            overlap = terms & _terms(span["text"])
            if len(overlap) < 2:
                continue
            candidates.append(
                {
                    "source_document_id": doc["document_id"],
                    "source_locator": f"{doc['source_locator']}#chars={span['start']}:{span['end']}",
                    "evidence_span": span,
                    "source_text_sha256": _text_sha(doc["text"]),
                    "matched_terms": sorted(overlap),
                    "lexical_coverage": len(overlap) / len(terms),
                    "interpretation": "Related result passage; exact-question coverage requires scientific review.",
                }
            )
    candidates.sort(
        key=lambda row: (
            -row["lexical_coverage"],
            row["source_document_id"],
            row["evidence_span"]["start"],
        )
    )
    return {
        "status": "POSSIBLY_ADDRESSED_NEEDS_REVIEW"
        if candidates
        else "INCOMPLETE_SEARCH"
        if failures
        else "NO_ADDRESSING_EVIDENCE_IN_SEARCHED_DOCUMENTS",
        "candidates": candidates[:limit],
        "candidate_count": len(candidates),
        "searched_document_ids": checked,
        "failures": failures,
        "search_complete": not failures,
        "global_novelty_established": False,
        "question_answered": None,
        "method": "bounded_lexical_result_passages_v1",
    }
