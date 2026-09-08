from litdatamatcher.negative_results import interpret
def test_negative_result_is_scoped_and_no_evidence_is_distinct():
 n=interpret({'effect_observed':False,'assay':'RNA','context':'adult','power_limit':'low','coverage_limit':'partial'});assert n['status']=='NEGATIVE_RESULT_SCOPED' and not n['global_negative'] and not n['novelty_claim'];assert interpret({})['status']=='NO_EVIDENCE'
