import random

from litdatamatcher.query_planning import plan_sources
from litdatamatcher.scientific_v2 import assess_requirements, rank_candidates


def test_seeded_fail_closed_invariants():
    rng=random.Random(20260908)
    for _ in range(24):
        wrong = {"dataset_id":"x","assay_types":["proteomics"],"organisms":["Homo sapiens"],"metadata":{"dependence":{"technical_run_count":rng.randint(1,20),"donor_links":"AMBIGUOUS_NOT_INFERRED"}},"capabilities":{}}
        ranked=rank_candidates([{"field":"modality","expected":"metabolomics"}],[wrong],{"x":1.0})[0]
        assert ranked["is_qualified"] is False
        assert assess_requirements([{"field":"biological_sample_count","expected":1}],wrong)["eligibility"] == "REQUIRES_INSPECTION"
        partial=plan_sources({"modality":"proteomics","complete_candidate_universe":True},[{"name":"p","supported":True,"modalities":["proteomics"],"access":"public","license":"CC0","terms_context":"fixture","metadata_completeness":"OBSERVED","candidate_universe_status":"PARTIAL_CANDIDATE_UNIVERSE_NOT_EVIDENCE_COMPLETE"}])
        assert partial["selected_sources"]==[]


def test_unknown_never_strengthens_claim():
    unknown={"dataset_id":"u","assay_types":[],"organisms":[],"metadata":{},"capabilities":{}}
    assert assess_requirements([{"field":"temporal_design","expected":"longitudinal"}],unknown)["eligibility"] == "REQUIRES_INSPECTION"
