from litdatamatcher.outcome_contract import assess_outcome
def test_outcome_mismatch_unknown_and_surrogate_guard():
 assert assess_outcome({'definition':'remission'},{'definition':'relapse'})['eligibility']=='NOT_QUALIFIED';assert assess_outcome({'unit':'score'},{})['eligibility']=='REQUIRES_INSPECTION';assert assess_outcome({'definition':'x'},{'definition':'x','measurement_status':'surrogate'})['eligibility']=='INDIRECT_ONLY'
