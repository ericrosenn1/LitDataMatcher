from copy import deepcopy

import pytest

from litdatamatcher.question_compiler import compile_question, matching_requirements
from litdatamatcher.scientific_v2 import assess_requirements
from litdatamatcher.tabular_contracts import (
    comparison_constraints,
    inspect_tabular_comparisons,
    scope_tabular_comparison,
)


def build(tmp_path, text="value,group,other\n1,A,3\n2,B,4\n?,B,5\nNaN,A,6\n"):
    path = tmp_path / "source.csv"
    path.write_text(text)
    metadata = {
        "dataset_id": "source",
        "num_instances": 4,
        "variables": [
            {
                "name": "value",
                "role": "Feature",
                "type": "Continuous",
                "description": "Recorded assay concentration",
            },
            {
                "name": "group",
                "role": "Target",
                "type": "Categorical",
                "description": "Observed diagnosis code A or B",
            },
            {
                "name": "other",
                "role": "Feature",
                "type": "Integer",
                "description": "Another measurement",
            },
        ],
    }
    spec = {
        "outcome_variable": "value",
        "grouping_variable": "group",
        "group_values": ["A", "B"],
        "definition_evidence": {
            "value": {
                "json_pointer": "/variables/0/description",
                "text": "Recorded assay concentration",
            },
            "group": {
                "json_pointer": "/variables/1/description",
                "text": "Observed diagnosis code A or B",
            },
        },
    }
    profile = inspect_tabular_comparisons(
        path,
        metadata,
        [spec],
        source_url="https://example.org/source.csv",
        metadata_url="https://example.org/dict",
    )
    compiled = compile_question(
        "What is the difference between A and B records in recorded value?",
        user_constraints=comparison_constraints("value", "group", ["A", "B"]),
    )
    return profile, matching_requirements(compiled), path, metadata, spec


def test_exact_contract_uses_source_rows_and_missingness_without_global_promotion(tmp_path):
    dataset, requirements, *_ = build(tmp_path)
    before = deepcopy(dataset)
    assert assess_requirements(requirements, dataset)["eligibility"] == "REQUIRES_INSPECTION"
    scoped = scope_tabular_comparison(dataset, requirements)
    assert assess_requirements(requirements, scoped)["eligibility"] == "DIRECT_FIT"
    assert dataset == before
    evidence = dataset["inspection"]["comparison_evidence"][0]
    assert evidence["source_data_row_numbers"] == [1, 2]
    assert evidence["joint_rows_by_group"] == {"A": 1, "B": 1}
    assert evidence["selected_variable_unobserved_cells"] == {
        "value:missing": 1,
        "value:nonfinite": 1,
    }
    assert "temporal_order" not in scoped["capabilities"]
    assert scoped["independent_units"] is None


@pytest.mark.parametrize(
    "change",
    [
        "alternate_outcome",
        "extra_variable",
        "wrong_groups",
        "no_binding",
        "source_hash",
        "metadata_hash",
    ],
)
def test_unmatched_scope_remains_unknown(tmp_path, change):
    dataset, requirements, *_ = build(tmp_path)
    if change == "alternate_outcome":
        next(row for row in requirements if row["field"] == "outcome_variable")["expected"] = (
            "other"
        )
    elif change == "extra_variable":
        requirements.append({"field": "measured_variable:other", "expected": True})
    elif change == "wrong_groups":
        next(row for row in requirements if row["field"] == "comparison_group:group:B")["field"] = (
            "comparison_group:group:C"
        )
    elif change == "no_binding":
        requirements = [row for row in requirements if row["field"] != "outcome_variable"]
    else:
        dataset["inspection"]["source_sha256" if change == "source_hash" else "metadata_sha256"] = (
            "changed"
        )
    scoped = scope_tabular_comparison(dataset, requirements)
    assert "comparison_scope" not in scoped
    assert "joint_observation" not in scoped["capabilities"]
    assert assess_requirements(requirements, scoped)["eligibility"] != "DIRECT_FIT"


def test_reusing_projection_for_another_question_cannot_leak_joint_observation(tmp_path):
    dataset, requirements, *_ = build(tmp_path)
    scoped = scope_tabular_comparison(dataset, requirements)
    alternate = comparison_constraints("other", "group", ["A", "B"])
    alternate.append({"field": "joint_observation", "expected": True})
    reset = scope_tabular_comparison(scoped, alternate)
    assert "joint_observation" not in reset["capabilities"]
    assert "outcome_ascertainment" not in reset["capabilities"]


def test_group_with_no_joint_measured_rows_is_a_mismatch(tmp_path):
    dataset, requirements, *_ = build(tmp_path, "value,group,other\n1,A,3\n?,B,4\n?,B,5\nNaN,A,6\n")
    scoped = scope_tabular_comparison(dataset, requirements)
    assert assess_requirements(requirements, scoped)["eligibility"] == "NOT_QUALIFIED"


def test_existing_contradiction_is_preserved_without_projection(tmp_path):
    dataset, requirements, *_ = build(tmp_path)
    for field, value in [
        ("joint_observation", False),
        ("outcome_variable", "prior"),
        ("grouping_variable", "prior_group"),
    ]:
        dataset["capabilities"][field] = {
            "value": value,
            "status": "observed",
            "source_locator": "prior source",
        }
    original = deepcopy(dataset["capabilities"])
    scoped = scope_tabular_comparison(dataset, requirements)
    assert scoped["capabilities"] == original
    assert "comparison_scope" not in scoped
    assert scoped["comparison_scope_conflict"]["fields"] == [
        "grouping_variable",
        "joint_observation",
        "outcome_variable",
    ]
    assert assess_requirements(requirements, scoped)["eligibility"] == "NOT_QUALIFIED"


def test_reused_projection_restores_exact_original_capabilities(tmp_path):
    dataset, requirements, *_ = build(tmp_path)
    for field, value in [
        ("joint_observation", True),
        ("outcome_variable", "value"),
        ("grouping_variable", "group"),
        ("comparison_group:group:A", True),
    ]:
        dataset["capabilities"][field] = {
            "value": value,
            "status": "observed",
            "source_locator": "prior source",
            "reason": "original reason",
        }
    original = deepcopy(dataset["capabilities"])
    scoped = scope_tabular_comparison(dataset, requirements)
    assert scoped["comparison_scope"]
    reset = scope_tabular_comparison(scoped, comparison_constraints("other", "group", ["A", "B"]))
    assert reset["capabilities"] == original


def test_timing_and_independence_remain_unknown_with_real_comparison_evidence(tmp_path):
    dataset, requirements, *_ = build(tmp_path)
    for field, expected in [
        ("temporal_order", "exposure_before_outcome"),
        ("independent_unit_count", 20),
    ]:
        extended = requirements + [{"field": field, "expected": expected}]
        scoped = scope_tabular_comparison(dataset, extended)
        assessment = assess_requirements(extended, scoped)
        assert assessment["eligibility"] == "REQUIRES_INSPECTION"
        assert assessment["requirements"][-1]["status"] == "UNKNOWN"


@pytest.mark.parametrize(
    "failure", ["invented_text", "name_only", "id_column", "tampered_evidence"]
)
def test_source_definition_and_integrity_are_required(tmp_path, failure):
    dataset, requirements, path, metadata, spec = build(tmp_path)
    if failure == "tampered_evidence":
        dataset["inspection"]["comparison_evidence"][0]["joint_observed_rows"] += 1
        with pytest.raises(ValueError, match="integrity"):
            scope_tabular_comparison(dataset, requirements)
        return
    if failure == "invented_text":
        spec["definition_evidence"]["value"]["text"] = "invented concentration"
    elif failure == "name_only":
        spec["definition_evidence"]["value"] = {
            "json_pointer": "/variables/0/name",
            "text": "value",
        }
    else:
        metadata["variables"][0]["role"] = "ID"
    with pytest.raises(ValueError):
        inspect_tabular_comparisons(
            path,
            metadata,
            [spec],
            source_url="https://example.org/source.csv",
            metadata_url="https://example.org/dict",
        )
