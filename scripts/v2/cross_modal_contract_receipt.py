from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from litdatamatcher.data_plane import atomic_json
from litdatamatcher.modality_contract import compatibility, modality_contract
from litdatamatcher.scientific_v2 import assess_requirements
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); a = p.parse_args()
r = {"dataset_id":"protein-fixture","assay_types":["proteomics"],"organisms":["Homo sapiens"],"metadata":{"omics_contract":{"feature_type":"protein","feature_unit":"peptide_intensity","quantification":"label_free","normalization":"median_scaled"},"dependence":{"donor_links":"AMBIGUOUS_NOT_INFERRED","technical_run_count":3}},"capabilities":{}}
o = {"modality":modality_contract(r)["modality"],"transcript_mismatch":compatibility("bulk_transcriptomics","Homo sapiens",r),"feature_mismatch":assess_requirements([{"field":"feature_type","expected":"metabolite"}],r)["eligibility"],"unit_unknown":assess_requirements([{"field":"biological_sample_count","expected":3}],r)["eligibility"]}
atomic_json(a.out,{"schema_version":"v2_cross_modal_contract_receipt_v1","fixture_scope":"synthetic/local only","outcomes":o,"validation_status":"PASS" if o=={"modality":["proteomics"],"transcript_mismatch":"INCOMPATIBLE","feature_mismatch":"NOT_QUALIFIED","unit_unknown":"REQUIRES_INSPECTION"} else "FAIL"})
