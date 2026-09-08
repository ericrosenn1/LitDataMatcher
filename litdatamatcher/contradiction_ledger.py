"""Source-scoped scientific claim contradiction ledger without voting."""
from __future__ import annotations
VALID={'SUPPORT','CONTRADICTION','INDIRECT','UNKNOWN'}
def ledger(claims:list[dict])->dict:
 entries=[]
 for c in claims:
  state=str(c.get('state','UNKNOWN')).upper();source=c.get('source_span')
  if state not in VALID or not source:state='UNKNOWN'
  entries.append({'claim_id':c.get('claim_id'),'state':state,'source_span':source or None,'relation_scope':c.get('relation_scope') or 'UNKNOWN','comparison_context':c.get('comparison_context') or None})
 return {'schema_version':'contradiction_ledger_v1','entries':entries,'net_vote':None,'requires_comparison_context':any(e['state']=='CONTRADICTION' and not e['comparison_context'] for e in entries)}
