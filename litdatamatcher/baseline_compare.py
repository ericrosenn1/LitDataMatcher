"""Fail-closed comparison of future Phase 2 checkpoints to protected baselines."""
from __future__ import annotations
def compare(baseline:dict,current:dict)->dict:
 regressions=[];added=[];unknown=[]
 for key in ("protected_hashes","api_version","schema_version"):
  if key not in baseline or key not in current: unknown.append(key);continue
  if baseline[key]!=current[key]:regressions.append(key)
 for key in current.get("capabilities",[]):
  if key not in baseline.get("capabilities",[]):added.append(key)
 return {"schema_version":"phase2_baseline_compare_v1","status":"FAIL" if regressions else "PASS","regressions":regressions,"added_capabilities":added,"unknowns":unknown}
