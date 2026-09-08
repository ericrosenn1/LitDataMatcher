"""Minimal Phase 2 receipt manifest audit."""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
def audit(manifest:dict,root:str)->dict:
 issues=[]; base=Path(root).resolve()
 for item in manifest.get("artifacts",[]):
  p=(base/str(item.get("path",""))).resolve()
  if base not in p.parents:issues.append("unsafe_path");continue
  if not p.exists():issues.append("missing_artifact");continue
  if item.get("sha256") and sha256(p.read_bytes()).hexdigest()!=item["sha256"]:issues.append("digest_mismatch")
 if manifest.get("exit_code") not in {0,None}:issues.append("nonzero_exit")
 if manifest.get("log_reference") and not (base/manifest["log_reference"]).exists():issues.append("missing_log_reference")
 return {"schema_version":"phase2_manifest_audit_v1","status":"PASS" if not issues else "FAIL","issues":issues,"model_observation":manifest.get("model") or None,"network_observation":manifest.get("network_mode") or None,"provenance_complete":bool(manifest.get("provenance"))}
