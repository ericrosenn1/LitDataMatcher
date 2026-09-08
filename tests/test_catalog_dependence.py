from litdatamatcher.catalog_dependence import reconcile_catalog_records


def test_declared_only_catalog_dependence_never_merges_title_similarity():
    rows=[{"dataset_id":"GSE1","title":"same title","metadata":{"declared_related_accessions":["ERP1"]}},{"dataset_id":"ERP1","title":"same title","metadata":{}},{"dataset_id":"NCT1","metadata":{"same_cohort_accessions":["GSE1"]}}]
    r=reconcile_catalog_records(rows)
    assert r["unknown_dependence_count"]==1
    assert any(e["dependence"]=="SAME_COHORT" for e in r["edges"])
    assert r["independent_dataset_count"]==2
