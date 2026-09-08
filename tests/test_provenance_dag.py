from litdatamatcher.provenance_dag import invalidate_snapshot
def test_selective_invalidation_preserves_unrelated_artifacts():
 r=invalidate_snapshot([{"id":"source","source_hash":"a"},{"id":"derived","parents":["source"]},{"id":"other"}],{"source"});assert [x["state"] for x in r]==["STALE","STALE","VALID"]
