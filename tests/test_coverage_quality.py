from litdatamatcher.coverage_quality import coverage_quality
def test_coverage_missing_partial_and_malformed_are_explicit():
 r=coverage_quality([{"source":"ENA","dataset_id":"x","title":"","access_type":"public","assay_types":["RNA-SEQ"],"organisms":["Homo sapiens"],"metadata":{"pagination":{"candidate_universe_status":"PARTIAL_CANDIDATE_UNIVERSE_NOT_EVIDENCE_COMPLETE"}}},None,{"source":"ENA","metadata":{}}]);s=r["source_coverage"]["ENA"];assert s["records"]==2 and s["missing"]["title"]==1 and s["universe"]["PARTIAL_CANDIDATE_UNIVERSE_NOT_EVIDENCE_COMPLETE"]==1
