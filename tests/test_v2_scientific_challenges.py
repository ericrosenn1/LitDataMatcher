"""Independent source-anchored contract challenges; not population validation.

Real source facts and constructed counterfactuals are distinguished in
benchmarks/v2/challenge_contracts.json. No expert annotations are claimed.
"""
import json
from pathlib import Path

import pytest

from litdatamatcher.scientific_v2 import (
    assess_requirements, combine_effects, compile_evidence, dependence_groups,
    evidence_relation_graph, propose_combination, rank_candidates,
)


def requirement(field="species", expected="Homo sapiens", essential=True):
    return dict(field=field, expected=expected, essential=essential, source_locator="explicit evaluation question")


def capability(value, locator="GEO:!Series_overall_design", status="observed"):
    return dict(value=value, status=status, source_locator=locator)


def dataset(identifier="GSE193336", species="Homo sapiens"):
    return dict(dataset_id=identifier, capabilities={"species": capability(species)},
                independent_units=4, study_lineage=["PRJNA795806"])


def question():
    return dict(question_id="DEV-ANSWERED-001", proposition_id="DEF_changes_IRG1_mRNA",
                conditions={"species":"human", "cell":"macrophage", "exposure":"DEF+LPS"},
                gap_status="unresolved-in-searched-coverage")


def evidence(identifier="paper", direction="supports", **updates):
    row=dict(evidence_id=identifier, proposition_id=question()["proposition_id"],
             role="direct_test", direction=direction, source_id="PMID31291584",
             publication_id="PMID31291584", study_id="GSE128885", cohort_id=None,
             conditions=question()["conditions"], measurement_type="observation",
             source_locator="PMC6635384:JATS:p42", scope_match="exact", answers_question=True,
             publication_date="2019-07-09")
    row.update(updates)
    return row


def compile_rows(rows):
    return compile_evidence(question(), rows, "2026-09-07", [{"source":"EuropePMC", "status":"success"}])


def test_source_selected_challenge_contract_has_honest_provenance():
    payload=json.loads((Path(__file__).parents[1]/"benchmarks/v2/challenge_contracts.json").read_text())
    assert len(payload["cases"]) == 8
    assert len({x["case_id"] for x in payload["cases"]}) == 8
    assert payload["label_origin"] == "source_determined"
    assert all(len(x["source_sha256"]) == 64 and x["locator"] and x["construction"] for x in payload["cases"])


def test_wrong_species_high_similarity_cannot_rescue_direct_fit():
    mouse=dataset("GSE99787", "Mus musculus")
    human=dataset()
    rows=rank_candidates([requirement()], [mouse,human], {"GSE99787":1.0,"GSE193336":-1.0})
    assert rows[0]["dataset_id"] == "GSE193336"
    assert rows[1]["assessment"]["eligibility"] == "NOT_QUALIFIED"
    assert not rows[1]["is_qualified"]


def test_lps_mention_does_not_establish_untreated_comparator():
    profile=dataset("GSE133844", "Mus musculus")
    profile["capabilities"]["exposure"]=capability("LPS")
    result=assess_requirements([requirement("comparator", "untreated")],profile)
    assert result["eligibility"] == "REQUIRES_INSPECTION"


def test_cells_libraries_are_not_donors_and_no_automatic_power_claim():
    profile=dataset("GSE95435")
    profile.update(independent_units=3, cell_libraries=92, bulk_libraries=3, empty_wells=1)
    result=assess_requirements([requirement()],profile)
    assert result["independent_units"] == 3
    assert result["statistical_adequacy"] == "UNKNOWN"


@pytest.mark.parametrize("bad", [True, 3.2, -1, "92"])
def test_invalid_independent_units_rejected(bad):
    profile=dataset();profile["independent_units"]=bad
    with pytest.raises(ValueError): assess_requirements([requirement()],profile)


def test_missing_null_and_explicit_false_are_distinct():
    profile=dataset()
    req=[requirement("paired", True)]
    assert assess_requirements(req,profile)["eligibility"] == "REQUIRES_INSPECTION"
    profile["capabilities"]["paired"]=capability(None,status="unknown")
    assert assess_requirements(req,profile)["eligibility"] == "REQUIRES_INSPECTION"
    profile["capabilities"]["paired"]=capability(False)
    assert assess_requirements(req,profile)["eligibility"] == "NOT_QUALIFIED"


def test_boolean_required_value_cannot_match_integer_one():
    profile=dataset();profile["capabilities"]["paired"]=capability(1)
    assert assess_requirements([requirement("paired",True)],profile)["eligibility"] != "DIRECT_FIT"


def test_absence_without_provenance_cannot_be_confirmed_mismatch():
    profile=dataset();profile["capabilities"]["paired"]=capability(None,locator=None,status="absent")
    try: result=assess_requirements([requirement("paired",True)],profile)
    except ValueError: return
    assert result["eligibility"] == "REQUIRES_INSPECTION"


def test_optional_missing_field_preserves_useful_partial_result():
    result=assess_requirements([requirement(),requirement("age",30,False)],dataset())
    assert result["eligibility"] == "PARTIAL_FIT"


def test_expanded_requirement_statuses_preserve_unknown_and_evidence_type():
    exact = dataset()
    assert assess_requirements([requirement()], exact)["compatibility_status"] == "EXACT_FIT"
    exact["evidence_classification"] = ["direct_perturbational_evidence"]
    assert assess_requirements([requirement()], exact)["compatibility_status"] == "DIRECTLY_ANSWERABLE"
    indirect = dataset(); indirect["capabilities"]["species"]["mapping_type"] = "ortholog"
    assert assess_requirements([requirement()], indirect)["compatibility_status"] == "INDIRECT_SUPPORT"
    missing = dataset(); missing["capabilities"] = {}
    assert assess_requirements([requirement()], missing)["compatibility_status"] == "UNKNOWN"
    wrong = dataset(species="Mus musculus")
    assert assess_requirements([requirement()], wrong)["compatibility_status"] == "INCOMPATIBLE"
    assert assess_requirements([], dataset())["compatibility_status"] == "REQUIRES_ADDITIONAL_DATA"


def test_orthology_is_not_identity():
    profile=dataset();profile["capabilities"]["species"]["mapping_type"]="ortholog"
    assert assess_requirements([requirement()],profile)["eligibility"] == "REQUIRES_INSPECTION"


def test_unambiguous_entity_ids_match_but_ambiguous_tissue_stays_unknown():
    profile = dataset(); profile["capabilities"] = {"species": capability("human"), "tissue": capability("gut")}
    assert assess_requirements([requirement("species", "Homo sapiens")], profile)["eligibility"] == "DIRECT_FIT"
    assert assess_requirements([requirement("tissue", "gut")], profile)["eligibility"] == "REQUIRES_INSPECTION"


def test_copied_paper_geo_graph_do_not_become_independent_votes():
    paper=evidence()
    geo=evidence("geo",source_id="GSE128885")
    copied=evidence("graph",source_id="KG:copied",source_of_source="GSE128885",publication_id=None,study_id=None)
    result=compile_rows([paper,geo,copied,copied])
    assert len(result["evidence_items"]) == 3
    assert len(result["dependence_groups"]) == 1
    assert result["independent_support_count"] is None
    assert result["known_dependence_edge_count"] >= 1
    assert {edge["relation_type"] for edge in result["relation_graph"]["edges"]} & {
        "same_underlying_evidence", "derivative_evidence"
    }


def test_relation_graph_preserves_explicit_scientific_classifications_without_inference():
    relation_types = [
        "replicated_evidence", "orthogonal_evidence", "direct_perturbational_evidence",
        "associative_evidence", "mechanistic_evidence", "indirect_evidence",
        "contradictory_evidence", "incompatible_evidence", "unknown_dependence",
    ]
    anchor = evidence("anchor", source_id="PMID:anchor", study_id="study-anchor")
    rows = [anchor]
    for index, relation_type in enumerate(relation_types):
        row = evidence(
            f"kind-{index}", source_id=f"source-{index}", study_id=f"study-{index}",
            publication_id=f"PMID:{index}",
        )
        row["relation_assertions"] = [{
            "target_evidence_id": "anchor", "relation_type": relation_type,
            "source_locator": f"fixture:{relation_type}",
        }]
        rows.append(row)
    graph = evidence_relation_graph(rows)
    assert {edge["relation_type"] for edge in graph["edges"]} == set(relation_types)
    assert len(dependence_groups(rows)) == len(rows)


def test_relation_graph_rejects_unlocated_or_unknown_relation_assertions():
    row = evidence("child", source_id="child", study_id="child-study")
    row["relation_assertions"] = [{"target_evidence_id": "missing", "relation_type": "replicated_evidence", "source_locator": "fixture"}]
    with pytest.raises(ValueError, match="unknown evidence ID"):
        evidence_relation_graph([evidence("anchor", source_id="anchor", study_id="anchor-study"), row])


def test_unknown_lineage_does_not_assert_independence():
    rows=[evidence("a",source_id="a",publication_id=None,study_id=None),evidence("b",source_id="b",publication_id=None,study_id=None)]
    assert all(g["between_group_independence"] == "UNKNOWN" for g in dependence_groups(rows))


def test_contradiction_retained_even_when_context_does_not_match():
    contrary=evidence("rat",direction="contradicts",conditions={"species":"rat"})
    result=compile_rows([evidence(),contrary])
    assert result["gap_status"] == "answered-in-scope"
    assert result["contradictory_evidence_ids"] == ["rat"]


def test_negative_result_can_answer_scoped_question_without_positive_vote():
    # Source p42 reports no apparent IRG1 transcript response; null is informative
    # about this experiment and must not become an untested/novel question.
    result=compile_rows([evidence(direction="contradicts")])
    assert result["gap_status"] == "answered-in-scope"


def test_background_cannot_close_gap():
    result=compile_rows([evidence(role="background")])
    assert result["gap_status"] != "answered-in-scope"


def test_future_observation_cannot_answer_as_of_past_date():
    try:
        result=compile_evidence(question(),[evidence(publication_date="2027-01-01")],"2026-09-07",[{"status":"success"}])
    except ValueError: return
    assert result["gap_status"] != "answered-in-scope"


def test_insufficient_search_never_becomes_global_novelty():
    result=compile_evidence(question(),[],"2026-09-07",[{"source":"EuropePMC","status":"failed"}])
    assert result["gap_status"] == "insufficient-coverage"


def test_complementary_union_without_joint_units_not_sufficient():
    first=dataset();second=dataset("GSE128885")
    first["capabilities"]["exposure"]=capability("LPS")
    second["capabilities"]["outcome"]=capability("IRG1")
    result=propose_combination([first,second],[requirement("exposure","LPS"),requirement("outcome","IRG1")])
    assert result["jointly_sufficient"] is False
    assert result["mode"] != "DIRECT_COMBINE"


def test_common_cohort_identifiers_alone_not_valid_numeric_join():
    first=dataset();second=dataset("same-cohort-assay")
    for item in (first,second): item.update(cohort_id="cohort",joint_unit_ids=["donor1"])
    assert propose_combination([first,second],[requirement()])["jointly_sufficient"] is False


@pytest.mark.parametrize("changed", [{"unit":"mg/dL"},{"population":"mouse"},{"estimand":"different"},{"cohort_id":"a"},{"independence_verified":False}])
def test_invalid_effect_pooling_abstains(changed):
    contract=dict(estimand="mean difference",unit="normalized expression",population="human",design="parallel",method="fixed_effect_inverse_variance")
    first=dict(contract, effect=1.,standard_error=.2,cohort_id="a",independence_verified=True,source_locator="test-only estimate A")
    second=dict(first,cohort_id="b",**changed) if "cohort_id" not in changed else dict(first,**changed)
    assert combine_effects([first,second],contract)["mode"] == "NOT_COMBINABLE"


def test_evidence_order_does_not_change_bundle_identity():
    rows=[evidence("a"),evidence("b",direction="contradicts")]
    assert compile_rows(rows)["bundle_id"] == compile_rows(list(reversed(rows)))["bundle_id"]
