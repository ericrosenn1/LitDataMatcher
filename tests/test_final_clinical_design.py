"""Independent clinical design contract controls; synthetic, offline, no inference."""
import copy

import pytest

from litdatamatcher.scientific_v2 import assess_requirements
from litdatamatcher.v2 import normalize_dataset


def registry_record(study_type=None, allocation=None, *, located=True):
    design = {}
    if study_type is not None:
        design["studyType"] = study_type
    if allocation is not None:
        design["designInfo"] = {"allocation": allocation}
    metadata = {"protocolSection": {"designModule": design}}
    if located:
        metadata["source_provenance"] = {"source_locator": "fixture:clinicaltrials/NCT00000001", "source_type": "clinicaltrials"}
    return {"dataset_id": "NCT00000001", "source": "clinicaltrials", "title": "Synthetic registry study", "metadata": metadata}


def assess(record, expected="randomized clinical trials"):
    return assess_requirements([{"field": "study_design", "expected": expected, "source_locator": "fixture:question#randomized-trials"}], normalize_dataset(record))


@pytest.mark.parametrize("study_type,allocation,expected", [
    ("OBSERVATIONAL", None, "MISMATCH"),
    ("OBSERVATIONAL", "RANDOMIZED", "MISMATCH"),
    ("INTERVENTIONAL", "NON_RANDOMIZED", "MISMATCH"),
    ("INTERVENTIONAL", "RANDOMIZED", "MATCH"),
    ("INTERVENTIONAL", None, "UNKNOWN"),
    (None, "RANDOMIZED", "UNKNOWN"),
    (None, None, "UNKNOWN"),
    ("INTERVENTIONAL", "UNKNOWN", "UNKNOWN"),
])
def test_registry_randomized_design_is_tristate_and_does_not_prove_statistical_adequacy(study_type, allocation, expected):
    record = registry_record(study_type, allocation)
    original = copy.deepcopy(record)
    result = assess(record)
    assert result["requirements"][0]["status"] == expected
    assert result["statistical_adequacy"] == "UNKNOWN"
    assert result["independent_units"] is None
    assert result["compatibility_status"] != "DIRECTLY_ANSWERABLE"
    assert record == original


def test_clinical_design_observation_retains_the_exact_protocol_field_locator():
    normalized = normalize_dataset(registry_record("OBSERVATIONAL"))
    capability = normalized["capabilities"]["study_design"]
    assert capability["status"] == "observed"
    assert capability["value"]["study_type"] == "OBSERVATIONAL"
    assert capability["source_locator"].startswith("fixture:clinicaltrials/NCT00000001#")
    assert "protocolSection" in capability["source_locator"]
    assert "designModule" in capability["source_locator"]


@pytest.mark.parametrize("study_type,allocation", [("OBSERVATIONAL", None), ("INTERVENTIONAL", "RANDOMIZED")])
def test_protocol_design_without_source_location_cannot_supply_an_observation(study_type, allocation):
    assert assess(registry_record(study_type, allocation, located=False))["requirements"][0]["status"] == "UNKNOWN"


@pytest.mark.parametrize("expected", ["randomly sampled cohort", "cluster randomized crossover superiority study", "not randomized clinical trials", "double blind clinical study"])
def test_unqualified_design_phrases_are_not_fuzzy_mapped_to_a_randomized_trial(expected):
    assert assess(registry_record("INTERVENTIONAL", "RANDOMIZED"), expected)["requirements"][0]["status"] == "UNKNOWN"


@pytest.mark.parametrize("status,mapping", [("derived", "exact"), ("observed", "broader"), ("observed", "related")])
def test_unqualified_design_capability_mapping_cannot_be_promoted_to_exact_fit(status, mapping):
    record = registry_record("INTERVENTIONAL", "RANDOMIZED")
    record["capabilities"] = {"study_design": {"value": {"study_type": "INTERVENTIONAL", "allocation": "RANDOMIZED"}, "status": status, "source_locator": "fixture:unqualified-design", "mapping_type": mapping}}
    assert assess(record)["requirements"][0]["status"] == "UNKNOWN"
