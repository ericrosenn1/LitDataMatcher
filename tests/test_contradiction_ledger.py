from litdatamatcher.contradiction_ledger import ledger
def test_ledger_retains_negative_contradiction_indirect_and_sourceless_unknown():
 r=ledger([{'claim_id':'a','state':'support','source_span':'s','relation_scope':'direct'},{'claim_id':'b','state':'contradiction','source_span':'x'},{'claim_id':'c','state':'indirect','source_span':'y'},{'claim_id':'d','state':'support'}]);assert [x['state'] for x in r['entries']]==['SUPPORT','CONTRADICTION','INDIRECT','UNKNOWN'] and r['net_vote'] is None and r['requires_comparison_context']
