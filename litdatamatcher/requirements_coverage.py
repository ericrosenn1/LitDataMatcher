"""Machine-readable requirement evidence coverage without score substitution."""
from __future__ import annotations
def audit_requirements(requirements:list[dict],candidates:list[dict])->dict:
 rows=[]
 for req in requirements:
  field=req['field']; states=[]
  for c in candidates:
   cap=c.get('capabilities',{}).get(field);states.append('UNKNOWN' if not cap else str(cap.get('status','UNKNOWN')).upper())
  disposition='SUPPORTED' if states and all(x=='OBSERVED' for x in states) else ('PARTIAL' if 'OBSERVED' in states else ('UNSUPPORTED' if not states else 'UNKNOWN'))
  rows.append({'field':field,'disposition':disposition,'action':'collect_source_evidence' if disposition!='SUPPORTED' else 'none'})
 return {'schema_version':'requirements_coverage_v1','requirements':rows,'missing_fields':[r['field'] for r in rows if r['disposition']!='SUPPORTED'],'score_substitution':False}
