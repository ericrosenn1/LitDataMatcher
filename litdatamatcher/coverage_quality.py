"""Observed-only Phase 2 catalog coverage summaries."""
from __future__ import annotations
from collections import Counter,defaultdict

def coverage_quality(records:list[dict])->dict:
    sources=defaultdict(lambda:{"records":0,"missing":Counter(),"modality":Counter(),"organism":Counter(),"assay":Counter(),"access":Counter(),"lifecycle":Counter(),"dependence":Counter(),"offline_cache":Counter(),"universe":Counter()})
    for r in records:
        if not isinstance(r,dict): continue
        s=sources[str(r.get("source","UNKNOWN"))];s["records"]+=1;m=r.get("metadata",{}) if isinstance(r.get("metadata"),dict) else {}
        for f in ("dataset_id","title","access_type"):
            if not r.get(f):s["missing"][f]+=1
        for x in r.get("assay_types",[]) or []:s["assay"][str(x)]+=1
        for x in r.get("organisms",[]) or []:s["organism"][str(x)]+=1
        c=m.get("modality_contract",{}); c=c if isinstance(c,dict) else {}
        for x in c.get("modality",[]) or []:s["modality"][str(x)]+=1
        s["access"][str(r.get("access_type","UNKNOWN") or "UNKNOWN")]+=1
        s["lifecycle"][str(m.get("literature_integrity",{}).get("lifecycle_status","UNKNOWN") if isinstance(m.get("literature_integrity"),dict) else "UNKNOWN")]+=1
        s["dependence"][str(m.get("dependence_status","UNKNOWN"))]+=1;s["offline_cache"][str(bool(m.get("cache_snapshot")))]+=1
        p=m.get("pagination",{});s["universe"][str(p.get("candidate_universe_status","UNKNOWN") if isinstance(p,dict) else "UNKNOWN")]+=1
    return {"schema_version":"coverage_quality_v1","source_coverage":{k:{**v,**{n:dict(v[n]) for n in v if isinstance(v[n],Counter)}} for k,v in sources.items()},"limitations":"Observed records only; source universe completeness is reported separately and missingness is never zero-filled."}
