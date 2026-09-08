"""Selective provenance-DAG invalidation for local Phase 2 artifacts."""
from __future__ import annotations
from .data_plane import digest

def invalidate_snapshot(nodes:list[dict], changed:set[str])->list[dict]:
    parents={n["id"]:set(n.get("parents",[])) for n in nodes}; stale=set(changed); progressed=True
    while progressed:
        progressed=False
        for node,refs in parents.items():
            if node not in stale and refs & stale: stale.add(node);progressed=True
    return [{**n,"state":"STALE" if n["id"] in stale else "VALID","identity":digest({k:n.get(k) for k in ("id","parents","source_hash","adapter","schema","config")})} for n in nodes]
