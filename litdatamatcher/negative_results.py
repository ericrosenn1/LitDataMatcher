"""Scoped negative-result interpretation without global or novelty claims."""
from __future__ import annotations
def interpret(record:dict)->dict:
 has_effect=record.get('effect_observed')
 if has_effect is None:return {'status':'NO_EVIDENCE','compatibility':'UNKNOWN','novelty_claim':False}
 required=('assay','context','power_limit','coverage_limit')
 limits={k:record.get(k) or 'UNKNOWN' for k in required}
 return {'status':'NEGATIVE_RESULT_SCOPED' if has_effect is False else 'EFFECT_OBSERVED_SCOPED','compatibility':'SOURCE_SCOPED','assay':limits['assay'],'context':limits['context'],'limits':limits,'global_negative':False,'novelty_claim':False,'dossier_status':'SOURCE_ASSISTED_PENDING_EXPERT_REVIEW'}
