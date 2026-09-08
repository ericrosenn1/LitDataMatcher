import pytest
from litdatamatcher.scientific_dossier import build_dossier, render_dossier, validate_dossier


def fixture():
    question={"question_id":"q","question":"<script>source question</script>","source_evidence_ids":["e1"]}
    bundle={"gap_status":"contradictory","as_of":"2026-09-08","novelty_claim":"Limited to searched coverage","evidence_items":[{"evidence_id":"e1","source_locator":"fixture"}],"dependence_groups":[],"contradictory_evidence_ids":["e1"]}
    assessment={"compatibility_status":"PARTIAL_FIT","eligibility":"PARTIAL_FIT","requirements":[{"field":"comparator","status":"UNKNOWN"}]}
    return question,bundle,assessment,{"dataset_id":"GSE1"},["fixture rationale"]


def test_dossier_preserves_required_scope_and_escapes_hostile_text():
    dossier=build_dossier(*fixture())
    assert validate_dossier(dossier) and dossier["contradictions"] == ["e1"]
    assert "&lt;script&gt;" in render_dossier(dossier)


def test_dossier_rejects_missing_provenance_and_global_novelty():
    question,bundle,assessment,candidate,rationale=fixture()
    question["source_evidence_ids"]=[]
    with pytest.raises(ValueError): build_dossier(question,bundle,assessment,candidate,rationale)
    question,bundle,assessment,candidate,rationale=fixture(); bundle["novelty_claim"]="Global novelty established"
    with pytest.raises(ValueError, match="global novelty"): build_dossier(question,bundle,assessment,candidate,rationale)
