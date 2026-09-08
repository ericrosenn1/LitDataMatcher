from litdatamatcher.observability import receipt
def test_observability_fails_hidden_exit_and_secret_and_keeps_nulls():
 assert receipt({"status":"PASS","exit_code":1,"logs":"ok"})["status"]=="FAIL"
 assert receipt({"status":"PASS","exit_code":0,"logs":"api_key=x"})["status"]=="FAIL"
 r=receipt({"status":"PASS","exit_code":0,"metrics":{"cache":"hit"},"cache_replay":True});assert r["status"]=="PASS" and r["metrics"]["network"] is None and r["cache_replay"] is True
