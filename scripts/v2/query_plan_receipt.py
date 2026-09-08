import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from litdatamatcher.data_plane import atomic_json
from litdatamatcher.query_planning import plan_sources
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
r=plan_sources({"modality":"proteomics","complete_candidate_universe":True,"offline_required":True},[{"name":"good","supported":True,"modalities":["proteomics"],"access":"public","metadata_completeness":"OBSERVED","offline_cache_available":True,"candidate_universe_status":"COMPLETE_CANDIDATE_UNIVERSE"},{"name":"partial","supported":True,"modalities":["proteomics"],"access":"public","metadata_completeness":"OBSERVED","offline_cache_available":True,"candidate_universe_status":"PARTIAL_CANDIDATE_UNIVERSE_NOT_EVIDENCE_COMPLETE"}]);atomic_json(a.out,{**r,"fixture_scope":"synthetic/local only","validation_status":"PASS" if r["selected_sources"]==["good"] else "FAIL"})
