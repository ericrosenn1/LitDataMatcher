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

SCHEMA_VERSION = "question-requirements-compiler-1.1"
COMPILER_VERSION = "source-bound-roles-1.1"
RULESET_VERSION = "question-method-rules-1.0"
_SPACE = re.compile(r"\s+")
_POPULATION_WORDS = (
    r"(?:humans?|mouse|mice|murine|rats?|adults?|patients?|participants?|people|"
    r"children|child|p[ae]ediatric|adolescents?|infants?|neonates?)"
)
_ROLE_END = (
    rf"(?=\s+in\s+(?:(?!in\b|from\b|of\b|and\b|or\b)[A-Za-z][A-Za-z-]*\s+){{0,4}}{_POPULATION_WORDS}\b|\s+(?:compared with|versus|vs\.?)\b|"
    r"\s+(?:over|after|within)\b|\s+at\s+\d+\s*(?:hours?|days?|weeks?|months?|years?)\b|[?.;,]|$)"
)


def _clean(value: str) -> str:
    return _SPACE.sub(" ", value).strip()


def _span(text: str, value: str, *, source_locator: str) -> dict[str, Any]:
    start = text.find(value)
    if start < 0:
        raise ValueError("Source-explicit value is not present in its source text")
    end = start + len(value)
    return {"source_locator": source_locator, "start": start, "end": end, "text": value}


def _rulebook() -> dict[str, Any]:
    with resources.files(__package__).joinpath("question_compiler_rules_v1.json").open(
        encoding="utf-8"
    ) as handle:
        return json.load(handle)


def _question_type(question: str) -> str:
    value = question.casefold()
    if all(re.search(rf"\b{field}\s*:", value) for field in ("intervention", "comparator", "outcome")):
        # A structured PICO-style source states a comparison even if it is not
        # written as an interrogative sentence. This recognizes source text;
        # it does not infer roles from a study annotation.
        return "comparative"
    if re.search(r"\b(literature|published evidence|review (?:the )?evidence)\b", value):
        return "literature_only"
    if re.search(r"\b(predict(?:s|ed|ing|ion|ions|ive|or|ors)?|prognos(?:is|tic)|classif(?:y|ies|ied|ying|ication|ications|ier|iers))\b", value):
        return "prediction"
    if re.search(r"\b(over time|longitudinal|trajectory|change from baseline|follow-?up)\b", value):
        return "longitudinal_change"
    if re.search(r"\b(knockdown|knock-out|knockout|crispr|perturb(?:s|ed|ing|ation|ations)?|stimulat(?:e|es|ed|ing|ion|ions|ory)|inhibit(?:s|ed|ing|ion|ions|ory)?|treated with|administered)\b", value):
        return "perturbational_mechanistic"
    if re.search(r"\b(which|what) (?:genes|features|biomarkers|entities)\b", value):
        return "entity_discovery"
    if re.search(r"\b(risks?|associat(?:e|es|ed|ing|ion|ions)|correlat(?:e|es|ed|ing|ion|ions)|related to)\b", value):
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


def _role_matches(
    question: str, context: str, source_locator: str,
    diagnostics: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Extract only bounded lexical roles; normalized values retain their source span."""
    roles: list[dict[str, Any]] = []
    diagnostics = diagnostics if diagnostics is not None else []

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

    # Some sources present a question as explicit PICO fields. Preserve their
    # source strings and spans exactly, without consulting relation annotations.
    for field, label in (
        ("intervention", "Intervention"),
        ("comparator", "Comparator"),
        ("outcome", "Outcome"),
    ):
        match = re.search(
            rf"\b{label}\s*:\s*(.+?)(?=\s*\|\s*(?:Intervention|Comparator|Outcome)\s*:|$)",
            question,
            re.I | re.S,
        )
        if match:
            observed = match.group(1).strip().rstrip("?.;")
            if observed:
                add(field, _clean(observed), observed)
    structured_fields = {role["field"] for role in roles}

    species = [
        (r"\b(?:human|humans)\b", "human"),
        (r"\b(?:adult(?:s)?|patients?|participants?|people|children|child(?!-)|p[ae]ediatric|adolescents?|infants?|neonates?)\b", "human"),
        (r"\b(?:mouse|mice|murine)\b", "mouse"),
        (r"\b(?:rat|rats)\b", "rat"),
    ]
    species_roles = {}
    for pattern, normalized in species:
        found = re.search(pattern, question, re.I)
        if found and normalized not in species_roles:
            add("species", normalized, found.group(0), "NORMALIZED" if normalized != found.group(0).casefold() else "SOURCE_EXPLICIT")
            species_roles[normalized] = found.group(0)
    if len(species_roles) > 1:
        diagnostics.append({
            "code": "AMBIGUOUS_MULTIPLE_SPECIES", "field": "species",
            "observed": list(species_roles),
            "message": "Multiple species are explicit; choose the intended population before dataset qualification.",
        })

    tissue = re.search(r"\b(?:in|from) ((?:human |mouse |murine |rat )?[A-Za-z][A-Za-z -]{1,60}?(?:cells?|tissue|tissues|organoids?|blood|plasma|serum))\b", question, re.I)
    if tissue:
        add("tissue", _clean(tissue.group(1)), tissue.group(1))

    if "comparator" not in structured_fields:
        comparator = re.search(r"\b(?:compared with|versus|vs\.?|than)\s+(.+?)" + _ROLE_END, question, re.I | re.S)
        if comparator:
            observed = comparator.group(1).strip()
            add("comparator", _clean(observed), observed)

    time = re.search(r"\b(?:over|after|at|within) ([0-9]+\s*(?:hours?|days?|weeks?|months?|years?))\b", question, re.I)
    if time:
        add("time", _clean(time.group(1)), time.group(1))

    # An intervention is a grammatical subject only when it precedes a relationship verb.
    if "intervention" not in structured_fields:
        intervention = re.search(
            r"(?:does|can|will|whether)\s+([A-Za-z0-9][A-Za-z0-9 \t\r\n-]{1,70}?)\s+"
            r"(?:increase|decrease|reduce|alter|affect|improve|worsen|predict|cause|inhibit|stimulate|suppress|enhance)",
            question,
            re.I,
        )
        if not intervention:
            intervention = re.search(r"\b(?:treated with|exposed to|administered)\s+([A-Za-z0-9][A-Za-z0-9 -]{1,70}?)(?:\s+in\b|\s+compared\b|[?.;,])", question, re.I)
        if intervention:
            observed = intervention.group(1).strip()
            add("intervention", _clean(observed), observed)

    if "outcome" not in structured_fields:
        outcome = re.search(
            r"\b(?:increase|decrease|reduce|alter|affect|improve|worsen|predict|cause|inhibit|stimulate|suppress|enhance)\s+"
            r"(?:the )?(.+?)" + _ROLE_END,
            question,
            re.I | re.S,
        )
        if outcome:
            observed = outcome.group(1).strip()
            if observed and len(observed) <= 100:
                add("outcome", _clean(observed), observed)

    # Bounded copular relations cover ordinary predictive/associational wording
    # without changing the downstream capability field vocabulary.
    copular = re.search(
        r"(?:^|:\s*)(?:is|are|was|were)\s+(.{1,70}?)\s+"
        r"(?:predictive of|associated with|correlated with|related to)\s+(.+?)" + _ROLE_END,
        question, re.I | re.S,
    )
    if copular:
        for field, group in (("intervention", 1), ("outcome", 2)):
            if field not in structured_fields and not any(role["field"] == field for role in roles):
                observed = copular.group(group).strip()
                add(field, _clean(observed), observed)

    # Bounded co-reference: resolve only a named risk target in supplied context.
    if re.search(r"\bthis risk\b", question, re.I):
        # An arbitrary acronym is not an antecedent. Require an explicit risk
        # relation, retain its exact context span, and abstain on competing targets.
        mentions = []
        for pattern in (r"\b(?i:risk (?:of|for))\s+(?P<target>[A-Z][A-Z0-9-]{2,})\b", r"\b(?P<target>[A-Z][A-Z0-9-]{2,})\s+(?i:risk)\b"):
            mentions.extend(re.finditer(pattern, context))
        targets = {match.group("target") for match in mentions}
        roles = [role for role in roles if role["field"] != "outcome"]
        if len(targets) == 1:
            mention = min(mentions, key=lambda match: match.start())
            observed = re.search(r"\bthis risk\b", question, re.I).group(0)
            add("outcome", f"risk of {mention.group('target')}", observed, "NORMALIZED")
            roles[-1]["antecedent_evidence"] = {
                "source_locator": source_locator + "#context",
                "start": mention.start(), "end": mention.end(), "text": mention.group(0),
            }
        else:
            diagnostics.append({
                "code": "UNRESOLVED_RISK_REFERENCE", "field": "outcome",
                "candidate_targets": sorted(targets),
                "message": "A unique explicit risk antecedent is required; unrelated acronyms are not substitutes.",
            })

    return roles


def _requirement(
    field: str,
    expected: Any,
    essential: bool,
    category: str,
    *,
    evidence: dict[str, Any] | None = None,
    antecedent_evidence: dict[str, Any] | None = None,
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
    if antecedent_evidence:
        provenance["antecedent_evidence"] = antecedent_evidence
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
    for field in values:
        if len({item["expected"] for item in roles if item["field"] == field}) > 1:
            values[field] = None
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
    # Source spans and original_gap_statement are relative to the unchanged
    # supplied question. Whitespace normalization is a separate, visible view.
    context = source_text or question
    question_type = _question_type(_clean(question))
    diagnostics: list[dict[str, Any]] = []
    roles = _role_matches(question, context, source_locator, diagnostics)
    requirements = [
        _requirement(
            role["field"], role["expected"], True, role["provenance_category"],
            evidence=role["evidence"], antecedent_evidence=role.get("antecedent_evidence"),
        )
        for role in roles
    ]
    rulebook = _rulebook()
    requirements.extend(_method_requirements(question_type, rulebook))
    user_fields: set[str] = set()
    if user_constraints:
        for raw in user_constraints:
            field, expected, essential = _validate_user_requirement(raw)
            if field in user_fields:
                raise ValueError("Each user requirement field must occur only once")
            user_fields.add(field)
            for item in requirements:
                if item["field"] == field and item["effective"]:
                    item["effective"] = False
                    if item["expected"] != expected:
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
    if any(
        item["code"] in {"UNRESOLVED_RISK_REFERENCE", "AMBIGUOUS_MULTIPLE_SPECIES"}
        and item["field"] not in user_fields
        for item in diagnostics
    ):
        status = "QUESTION_UNDERDETERMINED"
    # A recognized task word is not itself a named estimand. Preserve explicit
    # user contracts, but abstain when automatic rules lack their target roles.
    needed_roles = {
        "comparative": {"intervention", "outcome"},
        "associational": {"intervention", "outcome"},
        "prediction": {"intervention", "outcome"},
        "perturbational_mechanistic": {"intervention", "outcome"},
        "longitudinal_change": {"outcome"},
    }.get(question_type, set())
    missing_roles = sorted(needed_roles - {role["field"] for role in roles})
    if missing_roles and not has_effective_user_constraint:
        status = "QUESTION_UNDERDETERMINED"
        diagnostics.append({
            "code": "UNRESOLVED_TARGET_ROLES", "fields": missing_roles,
            "message": "Task family recognized, but automatic data qualification needs named source-grounded target roles.",
        })
    if question_type == "QUESTION_UNDERDETERMINED":
        diagnostics.append({"code": "UNSUPPORTED_OR_AMBIGUOUS_QUESTION_TYPE", "message": "Provide a target relationship, outcome, or supported question family."})
    if not effective:
        diagnostics.append({"code": "NO_ASSESSABLE_NECESSITIES", "message": "No candidate-data ranking will be performed."})
    return {
        "schema_version": SCHEMA_VERSION,
        "compiler_version": COMPILER_VERSION,
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
