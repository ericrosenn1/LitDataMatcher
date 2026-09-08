"""Covariate availability and causal-limit contracts."""
from __future__ import annotations
def assess_covariates(required:list[str],optional:list[str],observed:dict)->dict:
 rows=[]
 for name in required:rows.append({'covariate':name,'required':True,'availability':'OBSERVED' if name in observed else 'UNKNOWN'})
 for name in optional:rows.append({'covariate':name,'required':False,'availability':'OBSERVED' if name in observed else 'UNKNOWN'})
 return {'schema_version':'covariate_contract_v1','covariates':rows,'confounder_availability':observed.get('confounders','UNKNOWN'),'adjustment_status':observed.get('adjustment_status','UNKNOWN'),'compatibility':'REQUIRES_INSPECTION' if any(x['availability']=='UNKNOWN' and x['required'] for x in rows) else 'PARTIAL','causal_interpretation':'NOT_CAUSAL_WITHOUT_DECLARED_ADJUSTMENT'}
