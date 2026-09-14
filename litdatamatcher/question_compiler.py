"""Deterministic, provenance-preserving question-to-data compilation.

This module deliberately separates words observed in a question from concept
normalization and from requirements introduced by a versioned analysis rule.
It is a local product component: it neither calls Codex nor predicts a result.
"""

from __future__ import annotations

import json
import re
from importlib import resources
from typing import Any

SCHEMA_VERSION = "question-requirements-compiler-1.0"
RULESET_VERSION = "question-method-rules-1.0"
_SPACE = re.compile(r"\s+")


def _clean(value: str) -> str:
    return _SPACE.sub(" ", value).strip()


def _span(text: str, value: str, *, source_locator: str) -> dict[str, Any]:
    start = text.casefold().find(value.casefold())
    if start < 0:
        raise ValueError("Source-explicit value is not present in its source text")
    end = start + len(value)
    if text[start:end] != value:
        # Preserve the verbatim source slice, not a normalized/case-folded copy.
        value = text[start:end]
    return {"source_locator": source_locator, "start": start, "end": end, "text": value}


def _rulebook() -> dict[str, Any]:
    with resources.files(__package__).joinpath("question_compiler_rules_v1.json").open(
        encoding="utf-8"
    ) as handle:
        return json.load(handle)


def _question_type(question: str) -> str:
    value = question.casefold()
    if re.search(r"\b(literature|published evidence|review (?:the )?evidence)\b", value):
        return "literature_only"
    if re.search(r"\b(predict|prediction|prognos|classif)\b", value):
        return "prediction"
    if re.search(r"\b(over time|longitudinal|trajectory|change from baseline|follow-?up)\b", value):
        return "longitudinal_change"
    if re.search(r"\b(knockdown|knock-out|knockout|crispr|perturb|stimulat|inhibit|treated with|administered)\b", value):
        return "perturbational_mechanistic"
    if re.search(r"\b(which|what) (?:genes|features|biomarkers|entities)\b", value):
        return "entity_discovery"
    if re.search(r"\b(risk|associated|association|correlat)\b", value):
        return "associational"
    if re.search(r"\b(compared with|versus|vs\.?|difference between|does .+ (?:alter|affect|reduce|increase))\b", value):
        return "comparative"
    return "QUESTION_UNDERDETERMINED"


def _normal_question(gap: str) -> str:
    """Remove a future-work preamble without claiming a new scientific meaning."""
    result = _clean(gap)
    result = re.sub(
        r"^(?:further (?:studies|research) (?:are |is )?needed (?:to )?|"
        r"it remains (?:unclear|unknown) (?:whether|if) )",
        "",
        result,
        flags=re.I,
    )
    return result[:1].upper() + result[1:] if result else gap


def _role_matches(question: str, context: str, source_locator: str) -> list[dict[str, Any]]:
    """Extract only bounded lexical roles; normalized values retain their source span."""
    roles: list[dict[str, Any]] = []

    def add(field: str, expected: str, observed: str, category: str = "SOURCE_EXPLICIT") -> None:
        roles.append(
            {
                "field": field,
                "expected": expected,
                "observed_text": observed,
                "provenance_category": category,
                "evidence": _span(question, observed, source_locator=source_locator),
            }
        )

    species = [
        (r"\b(?:human|humans)\b", "human"),
        (r"\b(?:adult(?:s)?|patients?|participants?|people)\b", "human"),
        (r"\b(?:mouse|mice|murine)\b", "mouse"),
        (r"\b(?:rat|rats)\b", "rat"),
    ]
    for pattern, normalized in species:
        found = re.search(pattern, question, re.I)
        if found:
            add("species", normalized, found.group(0), "NORMALIZED" if normalized != found.group(0).casefold() else "SOURCE_EXPLICIT")
            break

    tissue = re.search(r"\b(?:in|from) ((?:human |mouse |murine |rat )?[A-Za-z][A-Za-z -]{1,60}?(?:cells?|tissue|tissues|organoids?|blood|plasma|serum))\b", question, re.I)
    if tissue:
        add("tissue", _clean(tissue.group(1)), tissue.group(1))

    comparator = re.search(r"\b(?:compared with|versus|vs\.?|than) ([^?.;,]+)", question, re.I)
    if comparator:
        observed = _clean(comparator.group(1))
        add("comparator", observed, observed)

    time = re.search(r"\b(?:over|after|at|within) ([0-9]+\s*(?:hours?|days?|weeks?|months?|years?))\b", question, re.I)
    if time:
        add("time", _clean(time.group(1)), time.group(1))

    # An intervention is a grammatical subject only when it precedes a relationship verb.
    intervention = re.search(
        r"(?:does|can|will|whether)\s+([A-Za-z0-9][A-Za-z0-9 -]{1,70}?)\s+"
        r"(?:increase|decrease|reduce|alter|affect|improve|worsen|predict|cause|inhibit)",
        question,
        re.I,
    )
    if not intervention:
        intervention = re.search(r"\b(?:treated with|exposed to|administered)\s+([A-Za-z0-9][A-Za-z0-9 -]{1,70}?)(?:\s+in\b|\s+compared\b|[?.;,])", question, re.I)
    if intervention:
        observed = _clean(intervention.group(1))
        add("intervention", observed, observed)

    outcome = re.search(
        r"\b(?:increase|decrease|reduce|alter|affect|improve|worsen|predict|cause|inhibit)\s+"
        r"(?:the )?(.+?)(?=\s+in\s+(?:humans?|mouse|mice|murine|rat|adults?|patients?|participants?)\b|\s+(?:compared with|versus|vs\.?)\b|\s+(?:over|after|within)\b|[?.;,])",
        question,
        re.I,
    )
    if outcome:
        observed = _clean(outcome.group(1))
        if observed and len(observed) <= 100:
            add("outcome", observed, observed)

    # Bounded co-reference: resolve only a named risk target in supplied context.
    if re.search(r"\bthis risk\b", question, re.I):
        acronyms = re.findall(r"\b[A-Z][A-Z0-9-]{2,}\b", context)
        if acronyms:
            observed = re.search(r"\bthis risk\b", question, re.I).group(0)
            roles = [role for role in roles if role["field"] != "outcome"]
            add("outcome", f"risk of {acronyms[-1]}", observed, "NORMALIZED")

    return roles


def _requirement(
    field: str,
    expected: Any,
    essential: bool,
    category: str,
    *,
    evidence: dict[str, Any] | None = None,
    rule_id: str | None = None,
    dependencies: list[str] | None = None,
    alternatives: list[Any] | None = None,
    unresolved_parameters: list[str] | None = None,
    effective: bool = True,
) -> dict[str, Any]:
    if expected is None:
        raise ValueError("Compiled requirements must not use null expected values")
    provenance: dict[str, Any] = {"category": category}
    if evidence:
        provenance["evidence"] = evidence
    if rule_id:
        provenance["rule_id"] = rule_id
        provenance["ruleset_version"] = RULESET_VERSION
    return {
        "field": field,
        "expected": expected,
        "operator": "EQUALS",
        "essential": essential,
        "provenance": provenance,
        "source_locator": (evidence or {}).get("source_locator", f"method-rule:{rule_id}" if rule_id else "user input"),
        "dependencies": dependencies or [],
        "alternative_acceptable_realizations": alternatives or [],
        "unresolved_parameters": unresolved_parameters or [],
        "effective": effective,
    }


def _method_requirements(question_type: str, rulebook: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for rule in rulebook["question_types"].get(question_type, {}).get("requirements", []):
        result.append(
            _requirement(
                rule["field"], rule["expected"], bool(rule["essential"]), "METHOD_DERIVED",
                rule_id=rule["rule_id"], dependencies=rule.get("dependencies", []),
                alternatives=rule.get("alternatives", []),
                unresolved_parameters=rule.get("unresolved_parameters", []),
            )
        )
    return result


def _answer_specification(question_type: str, roles: list[dict[str, Any]]) -> dict[str, Any]:
    values = {item["field"]: item["expected"] for item in roles}
    outputs = {
        "comparative": "contrast_estimate",
        "associational": "association_or_risk_estimate",
        "longitudinal_change": "within_unit_change_estimate",
        "prediction": "validated_predictive_performance",
        "perturbational_mechanistic": "perturbation_effect_estimate",
        "entity_discovery": "ranked_entity_or_feature_set",
        "literature_only": "evidence_synthesis",
    }
    return {
        "output_type": outputs.get(question_type, "UNRESOLVED"),
        "target_relationship_or_quantity": {
            "intervention_or_exposure": values.get("intervention"),
            "comparator": values.get("comparator"),
            "outcome": values.get("outcome"),
            "population_or_species": values.get("species"),
        },
        "predicted_value_or_direction": None,
        "interpretation": "The compiler specifies estimable information, not an answer or a direction of effect.",
    }


def _validate_user_requirement(raw: dict[str, Any]) -> tuple[str, Any, bool]:
    if not isinstance(raw, dict):
        raise ValueError("Each user requirement must be an object")
    field, expected = raw.get("field"), raw.get("expected")
    if not isinstance(field, str) or not field.strip() or expected is None:
        raise ValueError("User requirement needs a nonempty field and a non-null expected value")
    essential = raw.get("essential", True)
    if type(essential) is not bool:
        raise ValueError("User requirement essential must be Boolean")
    return field, expected, essential


def compile_question(
    question: str,
    *,
    source_text: str | None = None,
    source_locator: str = "question",
    user_constraints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compile an assessable contract from a question and optional bounded context.

    The result remains useful when partial, but callers must use
    :func:`matching_requirements` and never rank a candidate for an empty
    contract.
    """
    if not isinstance(question, str) or not _clean(question):
        raise ValueError("A nonempty question is required")
    if not isinstance(source_locator, str) or not source_locator:
        raise ValueError("source_locator is required")
    if source_text is not None and not isinstance(source_text, str):
        raise ValueError("source_text must be text when provided")
    question = _clean(question)
    context = source_text or question
    question_type = _question_type(question)
    roles = _role_matches(question, context, source_locator)
    requirements = [
        _requirement(
            role["field"], role["expected"], True, role["provenance_category"], evidence=role["evidence"]
        )
        for role in roles
    ]
    rulebook = _rulebook()
    requirements.extend(_method_requirements(question_type, rulebook))
    diagnostics: list[dict[str, Any]] = []
    if user_constraints:
        for raw in user_constraints:
            field, expected, essential = _validate_user_requirement(raw)
            for item in requirements:
                if item["field"] == field and item["effective"] and item["expected"] != expected:
                    item["effective"] = False
                    diagnostics.append({"code": "USER_OVERRIDE_CONFLICT", "field": field, "compiled": item["expected"], "user": expected})
            requirements.append(_requirement(field, expected, essential, "USER_SPECIFIED"))
    effective = [item for item in requirements if item["effective"] and item["essential"]]
    has_effective_user_constraint = any(
        item["effective"] and item["provenance"]["category"] == "USER_SPECIFIED"
        for item in requirements
    )
    status = (
        "COMPLETE"
        if has_effective_user_constraint or question_type == "literature_only"
        else "COMPLETE"
        if question_type != "QUESTION_UNDERDETERMINED" and effective
        else "QUESTION_UNDERDETERMINED"
    )
    if question_type == "QUESTION_UNDERDETERMINED":
        diagnostics.append({"code": "UNSUPPORTED_OR_AMBIGUOUS_QUESTION_TYPE", "message": "Provide a target relationship, outcome, or supported question family."})
    if not effective:
        diagnostics.append({"code": "NO_ASSESSABLE_NECESSITIES", "message": "No candidate-data ranking will be performed."})
    return {
        "schema_version": SCHEMA_VERSION,
        "ruleset_version": RULESET_VERSION,
        "compilation_status": status,
        "original_gap_statement": question,
        "normalized_research_question": _normal_question(question),
        "question_purpose": question_type,
        "entity_roles": roles,
        "answer_specification": _answer_specification(question_type, roles),
        "analysis_plans": rulebook["question_types"].get(question_type, {}).get("analysis_plans", []),
        "requirements": requirements,
        "ambiguities": [item for item in diagnostics if item["code"] != "USER_OVERRIDE_CONFLICT"],
        "diagnostics": diagnostics,
        "transformation_trace": [
            {"step": "preserve_original", "value": question},
            {"step": "bounded_normalization", "value": _normal_question(question)},
            {"step": "ruleset", "value": RULESET_VERSION},
        ],
    }


def matching_requirements(
    compilation: dict[str, Any], *, expert_override: bool = False
) -> list[dict[str, Any]]:
    """Return the strict legacy assessor view after validating compiler output."""
    if compilation.get("compilation_status") == "QUESTION_UNDERDETERMINED":
        return []
    rows = []
    for item in compilation.get("requirements", []):
        if expert_override and item.get("provenance", {}).get("category") != "USER_SPECIFIED":
            continue
        if item.get("effective") and item.get("essential"):
            expected = item.get("expected")
            if expected is None or type(item.get("essential")) is not bool:
                raise ValueError("Invalid compiler-to-assessor requirement mapping")
            rows.append({
                "field": item["field"], "expected": expected,
                "essential": item["essential"], "source_locator": item["source_locator"],
            })
    return rows
