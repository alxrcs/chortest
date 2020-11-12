import pytest
from chortools.gchor import Participant

from src.chortools.cfsm import CommunicatingSystem

@pytest.fixture
def large_atm():
    return 'examples/gchors/fsa/atm_fixed.fsa'

@pytest.fixture
def small_atm():
    return 'examples/gchors/fsa/atm_simple.fsa'

def test_small_atm_testgen(small_atm):
    atm = CommunicatingSystem.parse(small_atm)
    assert len(list(atm.tests(Participant('A')))) > 0

def test_large_atm_testgen(large_atm):
    atm = CommunicatingSystem.parse(large_atm)
    assert len(list(atm.tests(Participant('A')))) > 0

