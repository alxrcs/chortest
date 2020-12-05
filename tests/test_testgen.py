import pytest
from os.path import exists
from os import remove

from chortools.gchor import Participant
from chortools.cfsm import CommunicatingSystem

@pytest.fixture
def large_atm() -> str:
    return 'examples/gchors/fsa/atm_fixed.fsa'

@pytest.fixture
def small_atm() -> str:
    return 'examples/gchors/fsa/atm_simple.fsa'

def test_small_atm_testgen(small_atm) -> None:
    atm = CommunicatingSystem.parse(small_atm)
    tests = list(atm.tests(Participant('A')))
    test_name = 'test.test'
    for test in tests:
        test.to_fsa(test_name)
        assert exists(test_name)
        remove(test_name)

    assert len(tests) > 0

def test_large_atm_testgen(large_atm) -> None:
    atm = CommunicatingSystem.parse(large_atm)
    assert len(list(atm.tests(Participant('A')))) > 0

