from litdatamatcher.covariate_contract import assess_covariates
def test_missing_covariates_stay_unknown_not_incompatible_or_causal():
 r=assess_covariates(['age'],['sex'],{});assert r['compatibility']=='REQUIRES_INSPECTION' and r['covariates'][0]['availability']=='UNKNOWN' and r['causal_interpretation']=='NOT_CAUSAL_WITHOUT_DECLARED_ADJUSTMENT'
