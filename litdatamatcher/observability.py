"""Compact, redacted Phase 2 run observability receipts."""
from __future__ import annotations
FIELDS=("resource","backend","cache","network","retry","recovery","provenance")
def receipt(manifest:dict)->dict:
 exit_code=manifest.get("exit_code"); logs=str(manifest.get("logs","")).lower()
 success=exit_code==0 and manifest.get("status")=="PASS"
 if "api_key" in logs or "token=" in logs: success=False
 return {"schema_version":"operability_receipt_v1","status":"PASS" if success else "FAIL","exit_code":exit_code if isinstance(exit_code,int) else None,"metrics":{k:manifest.get("metrics",{}).get(k) if isinstance(manifest.get("metrics"),dict) else None for k in FIELDS},"cache_replay":manifest.get("cache_replay") if isinstance(manifest.get("cache_replay"),bool) else None,"provenance":manifest.get("provenance") if isinstance(manifest.get("provenance"),dict) else None,"logs_emitted":False}
