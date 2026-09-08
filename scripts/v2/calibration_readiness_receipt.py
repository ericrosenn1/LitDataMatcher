from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from litdatamatcher.calibration_readiness import build_calibration_scorecard
from litdatamatcher.data_plane import atomic_json
parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True); args=parser.parse_args()
base={"record_status":"RETAINED","label_origin":"source_determined","label_provenance":{"source_locator":"fixture"},"split_family":"fixture-family","dimension":"dataset_compatibility","ablation":"full"}
calibrated=build_calibration_scorecard([{**base,"record_id":"a","label":1,"score":0.9},{**base,"record_id":"b","label":0,"score":0.1}],split_family="fixture-family")
pending=build_calibration_scorecard([{**base,"record_id":"c","label_origin":"pending_expert","label":1,"score":0.9},{**base,"record_id":"d","dimension":"novelty","label":1,"score":0.9}],split_family="fixture-family")
atomic_json(args.out,{"schema_version":"v2_4_calibration_readiness_receipt_v1","fixture_scope":"synthetic source-determined labels only","calibrated_status":calibrated["calibration_status"],"pending_status":pending["calibration_status"],"pending_metrics":pending["metrics"],"validation_status":"PASS" if calibrated["calibration_status"]=="CALIBRATED" and pending["calibration_status"]=="PENDING_EXPERT_REVIEW" and pending["metrics"] is None else "FAIL"})
