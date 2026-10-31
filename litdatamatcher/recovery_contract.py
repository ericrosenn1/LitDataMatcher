"""Explicit bounded outcomes shared by source jobs and the local controller."""

from __future__ import annotations

OUTCOMES = {"TRANSIENT", "RETRYABLE", "PERMANENT", "SCHEMA", "CORRUPTION", "CAPACITY", "USER_PAUSE"}


def recover(state: dict, outcome: str, *, writer_id: str, max_retries: int = 2) -> dict:
    if outcome not in OUTCOMES:
        raise ValueError("unknown outcome")
    if not isinstance(writer_id, str) or not writer_id:
        raise ValueError("writer_id is required")
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    if state.get("writer_id") not in {None, writer_id}:
        return {**state, "status": "DUPLICATE_WRITER_BLOCKED"}
    retries = state.get("retries", 0)
    if type(retries) is not int or retries < 0:
        raise ValueError("retries must be a non-negative integer")
    retries += int(outcome in {"TRANSIENT", "RETRYABLE"})
    if outcome in {"TRANSIENT", "RETRYABLE"} and retries <= max_retries:
        status = "RESUMABLE"
    elif outcome in {"CAPACITY", "USER_PAUSE"}:
        status = "PAUSED"
    else:
        status = "FAILED"
    return {
        **state,
        "writer_id": writer_id,
        "retries": retries,
        "last_outcome": outcome,
        "status": status,
        "offline_recovery": "CACHE_ONLY"
        if outcome in {"CORRUPTION", "SCHEMA"}
        else state.get("offline_recovery", "NONE"),
    }
