"""Source-bound, question-scoped evidence for descriptive table comparisons.

Inspection never adds global joint-observation or outcome-ascertainment facts.
Call ``scope_tabular_comparison`` on each exact compiled requirement contract
immediately before assessment; do not cache its result as a general profile.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from copy import deepcopy
from urllib.parse import quote

from .dataset_inspection import _hash, _observed, inspect_tabular_dataset

VERSION = "tabular-comparison-evidence-1.0"
METHOD_FIELDS = {"joint_observation", "outcome_ascertainment"}


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _definition(metadata, variable, evidence, metadata_url):
    """Validate a curator-selected definition against unchanged source text.

    A text match verifies the citation, not the semantic mapping. The named
    variable mapping is explicit and remains inspectable by a reviewer.
    """
    pointer, text = evidence.get("json_pointer"), evidence.get("text")
    if not isinstance(pointer, str) or not pointer.startswith("/") or not text:
        raise ValueError("Source definition requires a JSON pointer and exact text")
    if pointer.split("/")[-1] not in {"description", "variable_info", "summary"}:
        raise ValueError("A column name alone is not an outcome definition")
    value = metadata
    try:
        for token in pointer[1:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            value = value[int(token)] if isinstance(value, list) else value[token]
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise ValueError("Source definition pointer is absent") from exc
    if not isinstance(value, str) or not isinstance(text, str) or text not in value:
        raise ValueError("Definition text does not occur at its source pointer")
    return {
        "variable": variable,
        "json_pointer": pointer,
        "text": text,
        "source_locator": metadata_url + "#" + pointer,
        "mapping_origin": "explicit_source_dictionary_mapping",
        "mapping_note": evidence.get("mapping_note", "Exact source variable definition"),
    }


def _is_observed(cell, definition, missing):
    if cell.casefold() in missing:
        return False, "missing"
    if definition.get("type") in {"Integer", "Continuous", "Real"} and not math.isfinite(
        float(cell)
    ):
        return False, "nonfinite"
    return True, None


def inspect_tabular_comparisons(path, metadata, comparisons, *, source_url, metadata_url):
    """Attach source-backed comparison evidence to an ordinary table profile.

    Each specification names an outcome column, grouping column, exact source
    group codes, and a source definition for both columns. No recoding, inferred
    timing, imputation, independence, statistical significance or power follows.
    """
    if not isinstance(comparisons, list) or not 1 <= len(comparisons) <= 100:
        raise ValueError("Provide 1..100 bounded comparison specifications")
    profile = inspect_tabular_dataset(
        path, metadata, source_url=source_url, metadata_url=metadata_url
    )
    source_hash = profile["inspection"]["source_sha256"]
    definitions = {row["name"]: row for row in metadata["variables"]}
    missing = {"", "?", "na", "n/a", "null"}
    if metadata.get("missing_values_symbol"):
        missing.add(str(metadata["missing_values_symbol"]).casefold())
    entries, seen = [], set()
    for spec in comparisons:
        outcome, grouping = spec.get("outcome_variable"), spec.get("grouping_variable")
        values = spec.get("group_values")
        if (
            outcome not in definitions
            or grouping not in definitions
            or outcome == grouping
            or any(definitions[name].get("role") == "ID" for name in (outcome, grouping))
        ):
            raise ValueError("Comparison requires two distinct source measurement columns")
        if (
            not isinstance(values, list)
            or not 2 <= len(values) <= 50
            or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError("Comparison requires 2..50 distinct literal source group codes")
        identity = (outcome, grouping, tuple(sorted(values)))
        if identity in seen:
            raise ValueError("Duplicate comparison scope")
        seen.add(identity)
        evidence = spec.get("definition_evidence", {})
        documented = [
            _definition(metadata, name, evidence.get(name, {}), metadata_url)
            for name in (outcome, grouping)
        ]
        if _hash(path) != source_hash:
            raise ValueError("Source table changed during inspection")
        rows, reasons, outside, counts = [], Counter(), 0, {value: 0 for value in values}
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for number, row in enumerate(csv.DictReader(handle), 1):
                observed = {}
                for name in (outcome, grouping):
                    cell = row[name].strip()
                    present, reason = _is_observed(cell, definitions[name], missing)
                    observed[name] = (cell, present)
                    if reason:
                        reasons[name + ":" + reason] += 1
                group, group_present = observed[grouping]
                if group_present and group not in counts:
                    outside += 1
                if group_present and group in counts and observed[outcome][1]:
                    counts[group] += 1
                    rows.append(number)
        if _hash(path) != source_hash:
            raise ValueError("Source table changed during inspection")
        entry = {
            "schema_version": VERSION,
            "outcome_variable": outcome,
            "grouping_variable": grouping,
            "group_values": sorted(values),
            "source_sha256": source_hash,
            "metadata_sha256": profile["inspection"]["metadata_sha256"],
            "definition_evidence": documented,
            "joint_rows_by_group": counts,
            "joint_observed_rows": len(rows),
            "source_data_row_numbers": rows,
            "source_row_number_convention": "one-based CSV data records, excluding header",
            "selected_variable_unobserved_cells": dict(reasons),
            "records_outside_requested_groups": outside,
            "total_source_rows": profile["inspection"]["rows"],
            "ascertainment_scope": "recorded_source_defined_measurement_only",
            "source_locator": source_url + "#sha256=" + source_hash,
            "limitations": [
                "Recorded measurements are not an adjudicated clinical endpoint or causal effect.",
                "Same-row observations do not establish independent participants or statistical power.",
                "Timing and confounding remain unknown; no temporal capability is supplied.",
            ],
        }
        entry["evidence_id"] = _fingerprint(entry)
        entries.append(entry)
    profile["inspection"]["comparison_evidence"] = entries
    return profile


def comparison_constraints(outcome_variable, grouping_variable, group_values):
    """Public compiler user-constraint shape for an exact source-column scope."""
    return [
        {"field": "outcome_variable", "expected": outcome_variable},
        {"field": "grouping_variable", "expected": grouping_variable},
        {"field": "measured_variable:" + outcome_variable, "expected": True},
        {"field": "measured_variable:" + grouping_variable, "expected": True},
        {"field": "processed_measurements", "expected": True},
        *[
            {"field": f"comparison_group:{grouping_variable}:{value}", "expected": True}
            for value in sorted(group_values)
        ],
    ]


def scope_tabular_comparison(dataset, requirements):
    """Project only an exact explicit question scope for one assessment call.

    Ordinary records and unmatched scopes retain unknown method capabilities.
    Previously projected records are reset first, preventing reuse leakage.
    """
    result = deepcopy(dataset)
    previous = result.pop("comparison_scope", None)
    if previous:
        caps = result.setdefault("capabilities", {})
        for field in previous.get("projected_fields", []):
            caps.pop(field, None)
    values = {}
    for row in requirements:
        if row["field"] in values:
            return result
        values[row["field"]] = row["expected"]
    outcome, grouping = values.get("outcome_variable"), values.get("grouping_variable")
    if not isinstance(outcome, str) or not isinstance(grouping, str):
        return result
    named_measurements = {
        field.removeprefix("measured_variable:")
        for field in values
        if field.startswith("measured_variable:")
    }
    if named_measurements != {outcome, grouping} or "measured_variable" in values:
        return result
    requested_groups = {
        field.removeprefix(f"comparison_group:{grouping}:")
        for field, expected in values.items()
        if field.startswith(f"comparison_group:{grouping}:") and expected is True
    }
    required = comparison_constraints(outcome, grouping, requested_groups)
    if (
        any(
            field.startswith("comparison_group:")
            and (not field.startswith(f"comparison_group:{grouping}:") or expected is not True)
            for field, expected in values.items()
        )
        or len(requested_groups) < 2
        or any(
            values.get(row["field"]) != row["expected"]
            or type(values.get(row["field"])) is not type(row["expected"])
            for row in required
        )
    ):
        return result
    inspection = result.get("inspection", {})
    for evidence in inspection.get("comparison_evidence", []):
        if (
            evidence["outcome_variable"] != outcome
            or evidence["grouping_variable"] != grouping
            or set(evidence["group_values"]) != requested_groups
            or evidence["source_sha256"] != inspection.get("source_sha256")
            or evidence["metadata_sha256"] != inspection.get("metadata_sha256")
        ):
            continue
        identity_payload = {key: value for key, value in evidence.items() if key != "evidence_id"}
        if _fingerprint(identity_payload) != evidence.get("evidence_id"):
            raise ValueError("Comparison evidence integrity mismatch")
        caps = result.setdefault("capabilities", {})
        locator = evidence["source_locator"] + "&comparison=" + evidence["evidence_id"]
        projected = [row["field"] for row in required if row["field"] not in caps]
        for row in required:
            if row["field"].startswith("comparison_group:"):
                group = row["field"].removeprefix(f"comparison_group:{grouping}:")
                caps[row["field"]] = _observed(evidence["joint_rows_by_group"][group] > 0, locator)
            elif row["field"] in {"outcome_variable", "grouping_variable"}:
                caps[row["field"]] = _observed(row["expected"], locator)
        caps["joint_observation"] = _observed(
            all(count > 0 for count in evidence["joint_rows_by_group"].values()), locator
        )
        outcome_definition = next(
            row for row in evidence["definition_evidence"] if row["variable"] == outcome
        )
        caps["outcome_ascertainment"] = _observed(
            evidence["joint_observed_rows"] > 0,
            outcome_definition["source_locator"] + "; " + locator + "&column=" + quote(outcome),
        )
        result["comparison_scope"] = {
            "evidence_id": evidence["evidence_id"],
            "requirement_fingerprint": _fingerprint(requirements),
            "outcome_variable": outcome,
            "grouping_variable": grouping,
            "group_values": sorted(requested_groups),
            "projected_fields": sorted(set(projected) | METHOD_FIELDS),
            "usage": "one exact compiled contract; do not cache as a general dataset profile",
        }
        return result
    return result
