from hashlib import sha256
from litdatamatcher.manifest_audit import audit
def test_manifest_audit_handles_missing_model_and_rejects_bad_paths(tmp_path):
 f=tmp_path/'a';f.write_text('x');ok=audit({'artifacts':[{'path':'a','sha256':sha256(b'x').hexdigest()}],'exit_code':0,'provenance':{'x':1}},tmp_path);assert ok['status']=='PASS' and ok['model_observation'] is None
 assert 'unsafe_path' in audit({'artifacts':[{'path':'../bad'}]},tmp_path)['issues']
