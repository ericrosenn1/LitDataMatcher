"""Outcome measurement compatibility without surrogate promotion."""
from __future__ import annotations
def assess_outcome(required:dict,observed:dict)->dict:
 fields=('definition','unit','ascertainment','estimand')
 rows=[]
 for f in fields:
  want=required.get(f);got=observed.get(f);rows.append({'field':f,'status':'UNKNOWN' if want is not None and got is None else ('MATCH' if want is None or want==got else 'MISMATCH')})
 mismatch=any(x['status']=='MISMATCH' for x in rows);unknown=any(x['status']=='UNKNOWN' for x in rows);surrogate=observed.get('measurement_status')=='surrogate'
 return {'schema_version':'outcome_contract_v1','fields':rows,'measurement_status':'SURROGATE' if surrogate else ('DIRECT' if observed.get('measurement_status')=='direct' else 'UNKNOWN'),'eligibility':'NOT_QUALIFIED' if mismatch else ('REQUIRES_INSPECTION' if unknown else ('INDIRECT_ONLY' if surrogate else 'DIRECT_FIT'))}
