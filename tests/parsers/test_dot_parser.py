import pytest
from chortools.lts import LTS

@pytest.mark.wip
def test_tsdot():
    lts : LTS = LTS.parse('tests/files/dotlts/test_0_ts5.dot')
    assert lts is not None

def test_lts_compliance():
    lts : LTS = LTS.parse('tests/files/dotlts/test_0_ts5.dot')

    assert not lts.is_success_configuration("q2_q3____C-Bb", [["q3"], ["q3"]])
    assert lts.is_success_configuration("q2_q3____C-Bb", [["q2"], ["q3"]])
    assert lts.is_compliant([["q3"], ["q3"]])
