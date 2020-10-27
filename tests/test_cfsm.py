from chorparse.gchor import Participant
import pytest
from chorparse.cfsm import CFSM, CommunicatingSystem, TransitionLabel
from typing import Iterable, List, Set


@pytest.fixture
def small_bank_cfsm():
    return CFSM(
        states={f"B{i}" for i in range(1, 5)},
        initial="B1",
        transitions={
            "B1": {"AB?authW": "B2"},
            "B2": {"BA!allow": "B4", "BA!deny": "B3"},
        },
    )

@pytest.fixture
def small_bank_cfsm_new():
    return CFSM.new(
        [
            ("B1", "AB?authW", "B2"),
            ("B2", "BA!allow", "B4"),
            ("B2", "BA!deny", "B3"),
        ],
        "B1",
    )

@pytest.fixture
def simple_atm_cs() -> CommunicatingSystem:
    atm_m = CFSM.new(
        transitions=[
            ("A1", "CA?withdraw", "A2"),
            ("A2", "AB!authW", "A3"),
            ("A3", "BA?allow", "A6"),
            ("A3", "BA?deny", "A4"),
            ("A6", "AC!money", "A7"),
            ("A4", "AC!bye", "A5"),
        ],
        initial="A1",
    )

    bank_m = CFSM.new(
        transitions=[
            ("B1", "AB?authW", "B2"),
            ("B2", "BA!allow", "B4"),
            ("B2", "BA!deny", "B3"),
        ],
        initial="B1",
    )

    client_m = CFSM.new(
        transitions=[
            ("C1", "CA!withdraw", "C2"),
            ("C2", "AC?money", "C4"),
            ("C2", "AC?bye", "C3"),
        ],
        initial="C1",
    )

    cs = CommunicatingSystem(
        {
            Participant("A"): atm_m,
            Participant("B"): bank_m,
            Participant("C"): client_m,
        }
    )

    return cs

def test_create_cfsm(small_bank_cfsm):
    assert small_bank_cfsm.transitions["B1"]["AB?authW"] == "B2"
    assert small_bank_cfsm.transitions["B2"]["BA!allow"] == "B4"
    assert small_bank_cfsm.transitions["B2"]["BA!deny"] == "B3"
    assert len(small_bank_cfsm.states) == 4
    assert "B1" in small_bank_cfsm.states
    assert "B4" in small_bank_cfsm.states

def test_create_new_cfsm(small_bank_cfsm_new):
    # TODO: Refactor these indexings to use only a public api
    assert (
        small_bank_cfsm_new.transitions["B1"][TransitionLabel.new("AB?authW")]
        == "B2"
    )
    assert (
        small_bank_cfsm_new.transitions["B2"][TransitionLabel.new("BA!allow")]
        == "B4"
    )
    assert (
        small_bank_cfsm_new.transitions["B2"][TransitionLabel.new("BA!deny")]
        == "B3"
    )
    assert len(small_bank_cfsm_new.states) == 4
    assert "B1" in small_bank_cfsm_new.states
    assert "B4" in small_bank_cfsm_new.states

@pytest.mark.cfsm
def test_create_new_cs(simple_atm_cs: CommunicatingSystem) -> None:

    print()
    while True:
        for cfsm, a, t, b in simple_atm_cs.enabled_transitions():
            print("transition: ", t)
            simple_atm_cs.fire_transition(cfsm, t, a, b)
            break
        else:
            break

def test_non_deterministic_states(simple_atm_cs: CommunicatingSystem) -> None:
    nds = list(simple_atm_cs.non_deterministic_states())
    assert len(nds) == 1
    assert nds[0] == "B2"

def test_split(small_bank_cfsm_new: CFSM) -> None:
    machines = list(small_bank_cfsm_new.split())
    assert len(machines) == 2

def test_tests(simple_atm_cs: CommunicatingSystem) -> None:
    tests = simple_atm_cs.tests(Participant('A'))
    for i, test in enumerate(tests):
        print(f'Test #{i}')
        print(str(test))

@pytest.mark.wip
def test_to_fsa(simple_atm_cs: CommunicatingSystem) -> None:
    tests = simple_atm_cs.tests(Participant('A'), 'experiments/simple_atm/tests')
    for i, test in enumerate(tests):
        print(f'Test #{i}')
        print(test.to_fsa())