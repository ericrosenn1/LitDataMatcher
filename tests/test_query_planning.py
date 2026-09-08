from litdatamatcher.query_planning import plan_sources


def test_planner_selects_only_complete_cached_eligible_source():
    sources=[{"name":"good","supported":True,"modalities":["proteomics"],"access":"public","metadata_completeness":"OBSERVED","offline_cache_available":True,"candidate_universe_status":"COMPLETE_CANDIDATE_UNIVERSE"},{"name":"partial","supported":True,"modalities":["proteomics"],"access":"public","metadata_completeness":"OBSERVED","offline_cache_available":True,"candidate_universe_status":"PARTIAL_CANDIDATE_UNIVERSE_NOT_EVIDENCE_COMPLETE"}]
    r=plan_sources({"modality":"proteomics","complete_candidate_universe":True,"offline_required":True},sources)
    assert r["selected_sources"]==["good"]
    assert r["decisions"][1]["status"]=="NOT_QUALIFIED"


def test_planner_never_queries_unsupported_wrong_or_unknown_routes():
    r=plan_sources({"modality":"metabolomics"},[{"name":"wrong","supported":True,"modalities":["proteomics"],"access":"public","metadata_completeness":"OBSERVED"},{"name":"unknown","supported":True,"modalities":["metabolomics"],"access":"unknown","metadata_completeness":"OBSERVED"},{"name":"off","supported":False}])
    assert r["selected_sources"]==[]
    assert [x["status"] for x in r["decisions"]]==["NOT_QUALIFIED","UNKNOWN","NOT_QUALIFIED"]
    assert not any(x["query_permitted"] for x in r["decisions"])
