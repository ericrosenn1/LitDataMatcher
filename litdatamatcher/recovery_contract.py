"""Typed resumable recovery state for bounded multisource jobs."""
from __future__ import annotations

OUTCOMES={"TRANSIENT","RETRYABLE","PERMANENT","SCHEMA","CORRUPTION","CAPACITY","USER_PAUSE"}
def recover(state:dict,outcome:str,*,writer_id:str,max_retries:int=2)->dict:
    if outcome not in OUTCOMES:raise ValueError("unknown outcome")
    if state.get("writer_id") not in {None,writer_id}:return {**state,"status":"DUPLICATE_WRITER_BLOCKED"}
    retries=int(state.get("retries",0)); retryable=outcome in {"TRANSIENT","RETRYABLE"}
    retries=retries+1 if retryable else retries
    status="RESUMABLE" if retryable and retries<=max_retries else ("PAUSED" if outcome in {"CAPACITY","USER_PAUSE"} else "FAILED")
    return {**state,"writer_id":writer_id,"retries":retries,"last_outcome":outcome,"status":status,"offline_recovery":"CACHE_ONLY" if outcome in {"CORRUPTION","SCHEMA"} else state.get("offline_recovery","NONE")}
