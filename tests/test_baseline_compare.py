from litdatamatcher.baseline_compare import compare
def test_protected_hash_api_schema_regress_only_on_explicit_change():
 b={"protected_hashes":"x","api_version":"1","schema_version":"1","capabilities":["a"]};assert compare(b,{**b,"capabilities":["a","b"]})["status"]=="PASS";assert compare(b,{**b,"protected_hashes":"bad"})["regressions"]==["protected_hashes"]
