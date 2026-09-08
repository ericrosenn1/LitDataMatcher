"""Explicit population applicability checks without demographic inference."""
from __future__ import annotations
def assess_population(required:dict,observed:dict)->dict:
 rows=[]
 for field in ('population','age','sex','eligibility'):
  want=required.get(field);got=observed.get(field)
  status='UNKNOWN' if want is not None and got is None else ('MATCH' if want is None or want==got else 'MISMATCH')
  rows.append({'field':field,'required':want,'observed':got,'status':status})
 return {'schema_version':'population_contract_v1','fields':rows,'applicability':'NOT_APPLICABLE' if any(x['status']=='MISMATCH' for x in rows) else ('REQUIRES_INSPECTION' if any(x['status']=='UNKNOWN' for x in rows) else 'SOURCE_SCOPED')}
