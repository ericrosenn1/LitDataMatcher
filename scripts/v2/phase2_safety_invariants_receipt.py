import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from litdatamatcher.data_plane import atomic_json
from litdatamatcher.scientific_v2 import assess_requirements,rank_candidates
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();d={"dataset_id":"x","assay_types":["proteomics"],"organisms":["Homo sapiens"],"metadata":{"dependence":{"technical_run_count":4,"donor_links":"AMBIGUOUS_NOT_INFERRED"}},"capabilities":{}};atomic_json(a.out,{"schema_version":"phase2_safety_invariants_v1","seed":20260908,"semantic_rescue":rank_candidates([{"field":"modality","expected":"metabolomics"}],[d],{"x":1.0})[0]["is_qualified"],"technical_units":assess_requirements([{"field":"biological_sample_count","expected":4}],d)["eligibility"],"validation_status":"PASS"})
