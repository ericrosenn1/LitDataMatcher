import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from litdatamatcher.catalog_dependence import reconcile_catalog_records
from litdatamatcher.data_plane import atomic_json
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();r=reconcile_catalog_records([{"dataset_id":"GSE1","metadata":{"declared_related_accessions":["ERP1"]}},{"dataset_id":"ERP1","metadata":{}}]);atomic_json(a.out,{**r,"fixture_scope":"synthetic/local only","validation_status":"PASS" if r["unknown_dependence_count"]==1 else "FAIL"})
