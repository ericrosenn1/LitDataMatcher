"""Explicit experimental contracts and dependence-aware evidence synthesis.

Heuristic rankings are review priorities, never probabilities or power analyses.
No score can change the hard scientific eligibility decision.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from .data_plane import digest
from .modality_contract import modality_contract
from .schemas import stable_id


@dataclass(frozen=True)
class ExperimentalRequirement:
    field: str
    expected: Any
    essential: bool = True
    source_locator: str = "user question"

    def __post_init__(self):
        if not self.field or self.expected is None or type(self.essential) is not bool:
            raise ValueError("Requirement needs a field, expected value and Boolean essential flag")


@dataclass(frozen=True)
class ObservedCapability:
    value: Any
    status: str
    source_locator: str | None = None
    reason: str | None = None
    mapping_type: str = "exact"

    def __post_init__(self):
        if self.status not in {"observed", "unknown", "absent", "derived"}:
            raise ValueError("Unknown capability status")
        if self.status == "observed" and (self.value is None or not self.source_locator):
            raise ValueError("Observed capability requires value and field-level provenance")
        if self.status == "unknown" and self.value is not None:
            raise ValueError("Unknown value cannot simultaneously assert an observation")
        if self.status == "absent" and not self.source_locator:
            raise ValueError("Confirmed absence requires field-level provenance")
        if self.mapping_type not in {
            "exact",
            "synonym",
            "broader",
            "narrower",
            "ortholog",
            "related",
            "unresolved",
        }:
            raise ValueError("Unknown mapping relation")


def finite_number(value, name, minimum=None, maximum=None):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, not Boolean/string")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} out of range")
    return float(value)


def _normalized(value):
    if isinstance(value, str):
        aliases = {
            "homo sapiens": "human",
            "mus musculus": "mouse",
            "rna-seq": "rna sequencing",
            "rna sequencing": "rna sequencing",
        }
        value = " ".join(value.casefold().split())
        return aliases.get(value, value)
    return value


def _equivalent(expected, observed):
    # Python's True == 1 must never create evidence of pairing/control presence.
    if isinstance(expected, bool) or isinstance(observed, bool):
        return type(expected) is type(observed) and expected == observed
    return _normalized(expected) == _normalized(observed)


def assess_requirements(requirements: list[dict], dataset: dict) -> dict:
    contract = dataset.get("modality_contract")
    if not isinstance(contract, dict):
        # Adapter-shaped records can be evaluated directly as well as through
        # the V2 normalizer.  Derive only the conservative observed/unknown
        # facts; no inferred synonym or sample-unit mapping is introduced.
        contract = modality_contract(dataset)
    assessments = []
    for raw in requirements:
        req = ExperimentalRequirement(**raw)
        raw_cap = dataset.get("capabilities", {}).get(req.field)
        cap = (
            ObservedCapability(**raw_cap)
            if raw_cap
            else ObservedCapability(None, "unknown", reason="not reported")
        )
        contract_status = _contract_requirement_status(req.field, req.expected, contract)
        if contract_status:
            status = contract_status
        elif cap.status == "unknown":
            status = "UNKNOWN"
        elif cap.status == "absent":
            status = "MISMATCH"
        elif (
            not cap.source_locator
            or cap.status == "derived"
            or cap.mapping_type not in {"exact", "synonym"}
        ):
            status = "UNKNOWN"
        else:
            values = cap.value if isinstance(cap.value, list) else [cap.value]
            status = "MATCH" if any(_equivalent(req.expected, v) for v in values) else "MISMATCH"
        assessments.append(dict(asdict(req), status=status, observation=asdict(cap)))
    essentials = [x for x in assessments if x["essential"]]
    if not essentials:
        eligibility = "REQUIRES_INSPECTION"
    elif any(x["status"] == "MISMATCH" for x in essentials):
        eligibility = "NOT_QUALIFIED"
    elif any(x["status"] == "UNKNOWN" for x in essentials):
        eligibility = "REQUIRES_INSPECTION"
    else:
        eligibility = "DIRECT_FIT"
    matched = sum(x["status"] == "MATCH" for x in assessments)
    unknown = sum(x["status"] == "UNKNOWN" for x in assessments)
    if eligibility == "DIRECT_FIT" and any(x["status"] != "MATCH" for x in assessments):
        eligibility = "PARTIAL_FIT"
    units = dataset.get("independent_units")
    if units is not None and (type(units) is not int or units < 0):
        raise ValueError("Invalid independent units")
    return {
        "schema_version": "2.0",
        "dataset_id": dataset["dataset_id"],
        "eligibility": eligibility,
        "requirements": assessments,
        "matched_fields": matched,
        "unknown_fields": unknown,
        "coverage": matched / len(assessments) if assessments else 0.0,
        "independent_units": units,
        "statistical_adequacy": "UNKNOWN",
        "adequacy_reason": "Requires estimand, variance, effect-size and design-specific power assessment",
        "availability": dataset.get("availability", "UNKNOWN"),
        "compatibility_status": _compatibility_status(assessments, dataset),
    }


def _compatibility_status(assessments: list[dict], dataset: dict) -> str:
    """Map field-level evidence to expanded V2.6 vocabulary without inference."""
    essential = [item for item in assessments if item["essential"]]
    if any(item["status"] == "MISMATCH" for item in essential):
        return "INCOMPATIBLE"
    if not essential:
        return "REQUIRES_ADDITIONAL_DATA"
    mapping_types = {item["observation"]["mapping_type"] for item in assessments}
    if mapping_types - {"exact", "synonym"} or any(item["observation"]["status"] == "derived" for item in assessments):
        return "INDIRECT_SUPPORT"
    if any(item["status"] == "UNKNOWN" for item in essential):
        return "UNKNOWN"
    if any(item["status"] != "MATCH" for item in assessments):
        return "PARTIAL_FIT"
    evidence = set(dataset.get("evidence_classification", []))
    if "direct_perturbational_evidence" in evidence:
        return "DIRECTLY_ANSWERABLE"
    return "EXACT_FIT"


def _contract_requirement_status(field, expected, contract):
    """Apply hard adapter modality/unit facts before any semantic capability match."""
    field = str(field).casefold()
    if field in {"modality", "assay", "assay_modality"}:
        modalities = contract.get("modality", ["UNKNOWN"])
        if modalities != ["UNKNOWN"] and str(expected) not in modalities:
            return "MISMATCH"
    if field in {"organism", "species"}:
        observed = contract.get("organisms", [])
        if observed and str(expected).casefold() not in {str(x).casefold() for x in observed}:
            return "MISMATCH"
    if field in {
        "biological_sample",
        "biological_sample_count",
        "donor",
        "donor_count",
        "independent_unit",
        "independent_unit_count",
    } and contract.get("biological_unit") == "UNKNOWN":
        return "UNKNOWN"
    return ""


def rank_candidates(
    requirements: list[dict], datasets: list[dict], semantic_scores: dict[str, float] | None = None
) -> list[dict]:
    semantic_scores = semantic_scores or {}
    rows = []
    for dataset in datasets:
        assessment = assess_requirements(requirements, dataset)
        sem = finite_number(
            semantic_scores.get(dataset["dataset_id"], 0.0), "semantic score", -1, 1
        )
        components = {
            "requirement_coverage": assessment["coverage"],
            "semantic_relevance": (sem + 1) / 2,
        }
        score = 0.8 * components["requirement_coverage"] + 0.2 * components["semantic_relevance"]
        rows.append(
            {
                "dataset_id": dataset["dataset_id"],
                "assessment": assessment,
                "components": components,
                "score": score,
                "score_type": "UNCALIBRATED_HEURISTIC",
                "is_qualified": assessment["eligibility"] in {"DIRECT_FIT", "PARTIAL_FIT"},
            }
        )
    order = {"DIRECT_FIT": 0, "PARTIAL_FIT": 1, "REQUIRES_INSPECTION": 2, "NOT_QUALIFIED": 3}
    return sorted(
        rows, key=lambda r: (order[r["assessment"]["eligibility"]], -r["score"], r["dataset_id"])
    )


RELATION_TYPES = frozenset(
    {
        "same_underlying_evidence",
        "derivative_evidence",
        "duplicated_cohort",
        "replicated_evidence",
        "orthogonal_evidence",
        "direct_perturbational_evidence",
        "associative_evidence",
        "mechanistic_evidence",
        "indirect_evidence",
        "contradictory_evidence",
        "incompatible_evidence",
        "unknown_dependence",
    }
)
DEPENDENCE_LINK_TYPES = frozenset(
    {"same_underlying_evidence", "derivative_evidence", "duplicated_cohort"}
)


def _unique_evidence(items: list[dict]) -> list[dict]:
    """Return a deterministic evidence universe, rejecting identity conflicts."""
    unique = {}
    for item in items:
        eid = item["evidence_id"]
        if eid in unique and digest(unique[eid]) != digest(item):
            raise ValueError("Conflicting evidence identity")
        unique[eid] = item
    return [unique[eid] for eid in sorted(unique)]


def _lineage_value(field: str, value: object) -> str:
    value = str(value).strip()
    if field == "publication_id":
        value = re.sub(r"^PMID[:\\s]*", "", value, flags=re.I)
    return value.casefold()


def evidence_relation_graph(items: list[dict]) -> dict:
    """Build an auditable graph of declared and exact-lineage evidence relations.

    Automatic edges are limited to exact shared identifiers and explicit
    source-of-source references. Scientific relation types such as replication
    or mechanism must be source-located declarations; they are never inferred
    from matching prose, authors, or topic similarity.
    """
    rows = _unique_evidence(items)
    ids = {row["evidence_id"] for row in rows}
    edges: dict[tuple[str, str, str, str], dict] = {}

    def add(source: str, target: str, relation_type: str, basis: str, locator: str) -> None:
        if relation_type not in RELATION_TYPES:
            raise ValueError(f"Unsupported evidence relation type: {relation_type}")
        if source not in ids or target not in ids:
            raise ValueError("Evidence relation references an unknown evidence ID")
        if source == target:
            return
        if basis == "declared_relation" and not str(locator).strip():
            raise ValueError("Evidence relation requires a source locator")
        left, right = sorted((source, target))
        key = (left, right, relation_type, basis)
        edges.setdefault(
            key,
            {
                "source_evidence_id": left,
                "target_evidence_id": right,
                "relation_type": relation_type,
                "basis": basis,
                "source_locator": str(locator),
            },
        )

    seen: dict[tuple[str, str], str] = {}
    for row in rows:
        for field, relation_type in (
            ("cohort_id", "duplicated_cohort"),
            ("study_id", "same_underlying_evidence"),
            ("publication_id", "same_underlying_evidence"),
            ("source_id", "same_underlying_evidence"),
        ):
            if row.get(field):
                key = (field, _lineage_value(field, row[field]))
                if key in seen:
                    add(
                        row["evidence_id"],
                        seen[key],
                        relation_type,
                        f"shared_{field}",
                        str(row.get("source_locator", "") or f"{row['evidence_id']}:{field}"),
                    )
                else:
                    seen[key] = row["evidence_id"]
        for value in row.get("primary_publication_ids", []):
            key = ("publication_id", _lineage_value("publication_id", value))
            if key in seen:
                add(
                    row["evidence_id"],
                    seen[key],
                    "derivative_evidence",
                    "primary_publication_id",
                    str(row.get("source_locator", "") or f"{row['evidence_id']}:primary_publication_ids"),
                )
            else:
                seen[key] = row["evidence_id"]
        source_of_source = str(row.get("source_of_source", "")).strip()
        if source_of_source:
            for candidate in rows:
                if source_of_source in {
                    str(candidate.get("evidence_id", "")),
                    str(candidate.get("source_id", "")),
                    str(candidate.get("study_id", "")),
                }:
                    add(
                        row["evidence_id"],
                        candidate["evidence_id"],
                        "derivative_evidence",
                        "source_of_source",
                        str(row.get("source_locator", "") or f"{row['evidence_id']}:source_of_source"),
                    )
        for assertion in row.get("relation_assertions", []):
            if not isinstance(assertion, dict):
                raise ValueError("Evidence relation assertion must be an object")
            add(
                row["evidence_id"],
                str(assertion.get("target_evidence_id", "")),
                str(assertion.get("relation_type", "")),
                "declared_relation",
                str(assertion.get("source_locator", "")),
            )
    return {"schema_version": "evidence_relation_graph_v1", "nodes": sorted(ids), "edges": sorted(edges.values(), key=lambda x: (x["source_evidence_id"], x["target_evidence_id"], x["relation_type"], x["basis"]))}


def dependence_groups(items: list[dict]) -> list[dict]:
    """Connected known-dependence components; separation never proves independence."""
    rows = _unique_evidence(items)
    parent = list(range(len(rows)))
    positions = {row["evidence_id"]: index for index, row in enumerate(rows)}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for edge in evidence_relation_graph(rows)["edges"]:
        if edge["relation_type"] in DEPENDENCE_LINK_TYPES:
            parent[find(positions[edge["source_evidence_id"]])] = find(
                positions[edge["target_evidence_id"]]
            )
    groups = {}
    for i, item in enumerate(rows):
        groups.setdefault(find(i), []).append(item["evidence_id"])
    return sorted(
        [
            {
                "dependence_group_id": stable_id("dep", *sorted(ids)),
                "evidence_ids": sorted(ids),
                "between_group_independence": "UNKNOWN",
            }
            for ids in groups.values()
        ],
        key=lambda x: x["dependence_group_id"],
    )


def compile_evidence(
    question: dict, items: list[dict], as_of: str, search_coverage: list[dict]
) -> dict:
    cutoff = date.fromisoformat(as_of[:10])
    proposition = question.get("proposition_id")
    included = []
    context = []
    for item in items:
        if not item.get("source_locator"):
            raise ValueError("Evidence lacks source locator")
        if (
            item.get("publication_date")
            and date.fromisoformat(item["publication_date"][:10]) > cutoff
        ):
            raise ValueError("Evidence postdates as-of assessment")
        if item.get("proposition_id") == proposition and proposition:
            included.append(item)
        elif item.get("related_proposition_id") == proposition and proposition:
            context.append(dict(item, integration_mode="CONTEXT_ONLY_OR_UNRESOLVED"))
    unique = {
        x["evidence_id"]: x for x in sorted(included + context, key=lambda x: x["evidence_id"])
    }
    relation_graph = evidence_relation_graph(included + context)
    groups = dependence_groups(included + context)
    scope = question.get("conditions", {})
    direct = [
        x
        for x in included
        if x.get("role") in {"direct_test", "replication", "perturbational_observation"}
        and x.get("conditions") == scope
        and x.get("scope_match") == "exact"
        and x.get("measurement_type") == "observation"
    ]
    supports = [x for x in direct if x.get("direction") == "supports"]
    contradictions = [x for x in included if x.get("direction") == "contradicts"]
    exact_contradictions = [x for x in direct if x.get("direction") == "contradicts"]
    if supports and exact_contradictions:
        gap = "contradictory"
    elif any(x.get("answers_question") is True for x in direct):
        gap = "answered-in-scope"
    elif direct:
        gap = "partly answered"
    elif not search_coverage or any(x.get("status") != "success" for x in search_coverage):
        gap = "insufficient-coverage"
    else:
        gap = "unresolved-in-searched-coverage"
    return {
        "bundle_id": stable_id(
            "bundle", question["question_id"], digest(list(unique.values())), as_of
        ),
        "question_id": question["question_id"],
        "as_of": as_of,
        "gap_before": question.get("gap_status", "unassessed"),
        "gap_status": gap,
        "search_coverage": search_coverage,
        "evidence_items": list(unique.values()),
        "relation_graph": relation_graph,
        "dependence_groups": groups,
        "known_dependence_edge_count": sum(
            edge["relation_type"] in DEPENDENCE_LINK_TYPES
            for edge in relation_graph["edges"]
        ),
        "independent_support_count": None,
        "contradictory_evidence_ids": sorted({x["evidence_id"] for x in contradictions}),
        "integration_mode": "EVIDENCE_SYNTHESIS" if included else "CONTEXT_ONLY_OR_UNRESOLVED",
        "novelty_claim": "Limited to recorded searched coverage; no global novelty assertion",
        "change_reason": f"{len(direct)} scope-matched observations; {len(contradictions)} contradictory items retained",
    }


def discover_cross_document_gaps(claims: list[dict], as_of: str) -> list[dict]:
    propositions = {}
    for item in claims:
        if item.get("proposition_id") and item.get("source_locator"):
            propositions.setdefault(
                (item["proposition_id"], digest(item.get("conditions", {}))), []
            ).append(item)
    result = []
    for (prop, _), items in propositions.items():
        if len({x.get("source_document_id", x.get("source_id")) for x in items}) < 2:
            continue
        if not {"supports", "contradicts"} <= {x.get("direction") for x in items}:
            continue
        result.append(
            {
                "question_id": stable_id(
                    "question", "conflict", prop, digest(items[0].get("conditions", {}))
                ),
                "proposition_id": prop,
                "question": f"What explains the conflicting evidence for {prop}?",
                "origin": "cross_document_conflict",
                "conditions": items[0].get("conditions", {}),
                "gap_status": "contradictory",
                "as_of": as_of,
                "source_evidence_ids": [x["evidence_id"] for x in items],
                "missing_evidence": "A controlled replication or explanation of contextual heterogeneity",
            }
        )
    return result


def propose_combination(datasets: list[dict], requirements: list[dict]) -> dict:
    assessments = [assess_requirements(requirements, d) for d in datasets]
    # A union of variables alone never establishes within-unit joint observation.
    ids = [set(d.get("joint_unit_ids", [])) for d in datasets]
    shared = set.intersection(*ids) if ids and all(ids) else set()
    same_cohort = len({d.get("cohort_id") for d in datasets}) == 1 and all(
        d.get("cohort_id") for d in datasets
    )
    if not shared or not same_cohort:
        return {
            "mode": "EVIDENCE_SYNTHESIS",
            "jointly_sufficient": False,
            "assessments": assessments,
            "reason": "No verified common units and cohort; variable union does not establish joint observation",
        }
    return {
        "mode": "CONTEXT_ONLY_OR_UNRESOLVED",
        "jointly_sufficient": False,
        "shared_units": sorted(shared),
        "assessments": assessments,
        "reason": "Common IDs located; feature, units, design and linkage contract still require validation",
    }


def combine_effects(records: list[dict], contract: dict) -> dict:
    """Fixed-effect inverse-variance pooling only for a declared common estimand.

    Explicit cohort independence is required. Heterogeneity is reported; this is
    not a default merger for raw matrices, associations, or ortholog proxies.
    """
    required = ["estimand", "unit", "population", "design"]
    if (
        any(not contract.get(k) for k in required)
        or contract.get("method") != "fixed_effect_inverse_variance"
    ):
        return {"mode": "NOT_COMBINABLE", "reason": "Incomplete analysis contract"}
    if len(records) < 2:
        return {"mode": "NOT_COMBINABLE", "reason": "At least two estimates required"}
    cohorts = set()
    weights = []
    effects = []
    for row in records:
        if any(row.get(k) != contract[k] for k in required) or not row.get("source_locator"):
            return {
                "mode": "NOT_COMBINABLE",
                "reason": "Incompatible estimand, units, population, design or absent provenance",
            }
        if (
            row.get("independence_verified") is not True
            or not row.get("cohort_id")
            or row["cohort_id"] in cohorts
        ):
            return {"mode": "NOT_COMBINABLE", "reason": "Independence absent or overlapping cohort"}
        cohorts.add(row["cohort_id"])
        effect = finite_number(row["effect"], "effect")
        se = finite_number(row["standard_error"], "standard_error", 0)
        if se == 0:
            raise ValueError("Standard error must be positive")
        effects.append(effect)
        weights.append(1 / se**2)
    mean = sum(w * x for w, x in zip(weights, effects, strict=True)) / sum(weights)
    se = math.sqrt(1 / sum(weights))
    q = sum(w * (x - mean) ** 2 for w, x in zip(weights, effects, strict=True))
    return {
        "mode": "DIRECT_COMBINE",
        "method": contract["method"],
        "contract": contract,
        "effect": mean,
        "standard_error": se,
        "ci95": [mean - 1.96 * se, mean + 1.96 * se],
        "cochran_q": q,
        "degrees_of_freedom": len(records) - 1,
        "input_records": records,
        "assumptions": "Common true effect and verified independent estimates; inspect heterogeneity before interpretation",
    }
