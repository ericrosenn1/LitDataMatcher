from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from litdatamatcher.data_plane import atomic_json
from litdatamatcher.scientific_dossier import build_dossier,validate_dossier
parser=argparse.ArgumentParser();parser.add_argument("--out",required=True);args=parser.parse_args()
d=build_dossier({"question_id":"fixture-q","question":"Source-determined fixture question","source_evidence_ids":["e1"]},{"gap_status":"unresolved-in-searched-coverage","as_of":"2026-09-08","novelty_claim":"Limited to searched coverage","evidence_items":[{"evidence_id":"e1","source_locator":"fixture:span"}],"dependence_groups":[],"contradictory_evidence_ids":[]},{"compatibility_status":"UNKNOWN","eligibility":"REQUIRES_INSPECTION","requirements":[{"field":"comparator","status":"UNKNOWN"}]},{"dataset_id":"fixture-dataset"},["source-derived fixture rationale"])
atomic_json(args.out,{"schema_version":"v2_6_dossier_receipt_v1","dossier_id":d["dossier_id"],"review_status":d["review_status"],"validation_status":"PASS" if validate_dossier(d) else "FAIL"})
