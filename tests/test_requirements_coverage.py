from litdatamatcher.requirements_coverage import audit_requirements
def test_requirements_coverage_keeps_unknown_partial_unsupported():
 r=audit_requirements([{'field':'assay'},{'field':'outcome'}],[{'capabilities':{'assay':{'status':'observed'}}},{'capabilities':{}}]);assert r['requirements'][0]['disposition']=='PARTIAL' and r['requirements'][1]['disposition']=='UNKNOWN' and r['score_substitution'] is False
