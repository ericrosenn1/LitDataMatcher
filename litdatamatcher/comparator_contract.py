"""Declared assay comparator metadata checks."""
from __future__ import annotations
VALID={'sham','placebo','vehicle','baseline','active_comparator','untreated'}
def assess_comparator(expected:str,observed:dict)->dict:
 value=str(observed.get('comparator_type','UNKNOWN') or 'UNKNOWN').casefold()
 if value=='unknown':return {'status':'UNKNOWN','control_type':'UNKNOWN','validity':'UNOBSERVED'}
 if value not in VALID:return {'status':'UNKNOWN','control_type':value,'validity':'UNSUPPORTED_DECLARATION'}
 return {'status':'MATCH' if value==expected.casefold() else 'MISMATCH','control_type':value,'validity':'DECLARED'}
