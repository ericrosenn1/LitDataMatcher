import json
from pathlib import Path
import pytest
from litdatamatcher.v2 import source_chunks,normalize_dataset,render_report
from litdatamatcher.scientific_v2 import dependence_groups


def test_explicit_chunks_keep_parent_offsets():
    text='Background text.\nTreatment reduced TNF. Further work is needed.'
    d={'document_id':'d','text':text,'sections':[{'start':0,'end':16,'text':text[:16],'section':'Introduction'},
      {'start':17,'end':len(text),'text':text[17:],'section':'Results'}]}
    chunks=source_chunks(d,max_chars=25,max_chunks=2)
    assert chunks[0]['parent_start']==17
    assert all(text[c['parent_start']:c['parent_end']]==c['text'] for c in chunks)


def test_capability_migration_retains_unknown_and_source():
    raw={'dataset_id':'GSE1','capabilities':{'paired':{'status':'unknown','value':None,'reason':'not reported'},'species':{'status':'known','value':'human','source_locator':['sample:a','sample:b']}}}
    migrated=normalize_dataset(raw)
    assert migrated['capabilities']['paired']['value'] is None
    assert migrated['capabilities']['species']['source_locator']=='sample:a; sample:b'
    assert raw['capabilities']['species']['status']=='known'


def test_aggregator_multiple_citations_join_primary_lineage():
    rows=[{'evidence_id':'paper','publication_id':'PMID:123'},
          {'evidence_id':'curation','primary_publication_ids':['PMID123','PMID456']}]
    assert len(dependence_groups(rows))==1


def test_report_escapes_hostile_metadata(tmp_path):
    (tmp_path/'RUN_MANIFEST.json').write_text(json.dumps({'run_id':'<script>alert(1)</script>','execution_status':'PARTIAL','coverage':{}}))
    for name in ['questions','matches','evidence_bundles']:(tmp_path/(name+'.jsonl')).write_text('')
    report=render_report(tmp_path).read_text()
    assert '<script>alert' not in report and '&lt;script&gt;' in report
    assert "default-src 'none'" in report
