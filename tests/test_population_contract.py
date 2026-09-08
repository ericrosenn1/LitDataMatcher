from litdatamatcher.population_contract import assess_population
def test_population_mismatch_unknown_and_no_inference():
 assert assess_population({'sex':'female'},{'sex':'male'})['applicability']=='NOT_APPLICABLE';assert assess_population({'age':'adult'},{})['applicability']=='REQUIRES_INSPECTION'
