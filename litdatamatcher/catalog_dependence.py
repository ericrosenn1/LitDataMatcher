"""Declared-only cross-catalog dataset dependence reconciliation."""
from __future__ import annotations

from .data_plane import digest


def reconcile_catalog_records(records: list[dict]) -> dict:
    edges=[]
    for record in records:
        rid=str(record.get("dataset_id","")); meta=record.get("metadata",{}) if isinstance(record.get("metadata"),dict) else {}
        for target in meta.get("declared_related_accessions",[]) or []:
            edges.append({"from":rid,"to":str(target),"relation":"DECLARED_CROSS_ACCESSION","dependence":"UNKNOWN_DEPENDENCE"})
        for target in meta.get("same_cohort_accessions",[]) or []:
            edges.append({"from":rid,"to":str(target),"relation":"DECLARED_SAME_COHORT","dependence":"SAME_COHORT"})
        for target in meta.get("derivative_of_accessions",[]) or []:
            edges.append({"from":rid,"to":str(target),"relation":"DECLARED_DERIVATIVE","dependence":"DERIVATIVE"})
    dependent={edge["from"] for edge in edges if edge["dependence"] in {"SAME_COHORT","DERIVATIVE"}}
    return {"schema_version":"catalog_dependence_v1","edges":edges,"independent_dataset_count":len([r for r in records if str(r.get("dataset_id","")) not in dependent]),"unknown_dependence_count":sum(e["dependence"]=="UNKNOWN_DEPENDENCE" for e in edges),"input_digest":digest(records)}
