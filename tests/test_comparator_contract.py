from litdatamatcher.comparator_contract import assess_comparator
def test_comparator_explicit_mismatch_and_absence():
 assert assess_comparator('placebo',{'comparator_type':'vehicle'})['status']=='MISMATCH';assert assess_comparator('placebo',{})['status']=='UNKNOWN';assert assess_comparator('sham',{'comparator_type':'sham'})['validity']=='DECLARED'
