import pytest
from litdatamatcher.schemas import _clamp01, _optional_score
from litdatamatcher.resources import ResourceGovernor
from litdatamatcher.scientific_v2 import combine_effects


@pytest.mark.parametrize('value',[True,False,'0.3',float('nan'),float('inf'),-.1,1.1])
def test_invalid_numeric_scores_rejected(value):
    with pytest.raises(ValueError): _clamp01(value,'test')


def test_optional_score_csv_numeric_and_unknown():
    assert _optional_score('4','score')==4
    assert _optional_score('','score') is None
    with pytest.raises(ValueError): _optional_score('nan','score')


def test_governor_hysteresis():
    g=ResourceGovernor()
    assert g.admission(available_fraction=.19,cpu_count=32)['cpu_workers']==0
    assert g.admission(available_fraction=.23,cpu_count=32)['mode']=='PRESSURE'
    assert g.admission(available_fraction=.4,cpu_count=32)['mode']=='INTERACTIVE'
    assert g.admission(available_fraction=.4,cpu_count=32,idle_seconds=700)['cpu_workers']==22


def test_pooling_rejects_shared_units_and_incompatible_estimands():
    contract=dict(estimand='log fold change',unit='log2',population='human macrophage',design='paired',method='fixed_effect_inverse_variance')
    a=dict(contract,effect=1.,standard_error=.2,cohort_id='a',independence_verified=True,source_locator='fixture:a')
    b=dict(a,cohort_id='b',effect=1.2,source_locator='fixture:b')
    result=combine_effects([a,b],contract)
    assert result['mode']=='DIRECT_COMBINE' and result['effect']==pytest.approx(1.1)
    assert combine_effects([a,a],contract)['mode']=='NOT_COMBINABLE'
    assert combine_effects([a,dict(b,unit='counts')],contract)['mode']=='NOT_COMBINABLE'
    # This fixture tests mathematics; it cannot count as the real-data demonstration.
