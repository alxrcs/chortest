import pytest
from os.path import exists
from os import remove

from chortest.gchor import Participant
from chortest.cfsm import CommunicatingSystem

@pytest.fixture
def small_atm() -> str:
    return 'tests/files/gchors/fsa/atm_simple.fsa'

def test_small_atm_testgen(small_atm) -> None:
    atm = CommunicatingSystem.parse(small_atm)
    tests = list(atm.tests(Participant('A')))
    test_name = 'tmptest'
    for test in tests:
        test.to_fsa(test_name)
        assert exists(test_name)
        remove(test_name)
        remove('oracle.yaml')

    assert len(tests) > 0
