import os

import pytest
from chortest.cfsm import CFSM, CommunicatingSystem, TransitionLabel
from chortest.common import LocalMutationTypes
from chortest.gchor import Participant
from chortest.mutations import LocalMutator


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
        name="ATM",
    )

    bank_m = CFSM.new(
        transitions=[
            ("B1", "AB?authW", "B2"),
            ("B2", "BA!allow", "B4"),
            ("B2", "BA!deny", "B3"),
        ],
        initial="B1",
        name="Bank",
    )

    client_m = CFSM.new(
        transitions=[
            ("C1", "CA!withdraw", "C2"),
            ("C2", "AC?money", "C4"),
            ("C2", "AC?bye", "C3"),
        ],
        initial="C1",
        name="Client",
    )

    d = {
        Participant("A"): atm_m,
        Participant("B"): bank_m,
        Participant("C"): client_m,
    }

    return CommunicatingSystem(list(d.keys()), d)


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
        small_bank_cfsm_new.transitions["B1"][TransitionLabel.new("AB?authW")] == "B2"
    )
    assert (
        small_bank_cfsm_new.transitions["B2"][TransitionLabel.new("BA!allow")] == "B4"
    )
    assert small_bank_cfsm_new.transitions["B2"][TransitionLabel.new("BA!deny")] == "B3"
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
    tests = simple_atm_cs.tests(Participant("A"))
    for i, test in enumerate(tests):
        print(f"Test #{i}")
        print(str(test))


def test_to_fsa(simple_atm_cs: CommunicatingSystem) -> None:
    tests = simple_atm_cs.tests(Participant("A"))
    for i, test in enumerate(tests):
        print(f"Test #{i}")
        print(test.to_fsa())


@pytest.mark.wip
def test_to_networkx_cfsm(small_bank_cfsm_new: CFSM) -> None:
    g = small_bank_cfsm_new.to_networkx()
    assert ("B1", "B2") in g.edges
    assert ("B2", "B4") in g.edges
    assert ("B2", "B3") in g.edges


def test_write_dot(small_bank_cfsm_new: CFSM) -> None:
    small_bank_cfsm_new.to_dot("small_bank_cfsm_new.dot")
    assert os.path.exists("small_bank_cfsm_new.dot")
    os.remove("small_bank_cfsm_new.dot")


@pytest.mark.wip
def test_to_networkx_cs(simple_atm_cs: CommunicatingSystem) -> None:
    gs = simple_atm_cs.to_networkx()
    print(gs)


@pytest.mark.wip
def test_to_dot_cs(simple_atm_cs: CommunicatingSystem) -> None:
    gs = simple_atm_cs.to_dot()
    print(gs)


def test_mutate_A(simple_atm_cs: CommunicatingSystem) -> None:
    atm = Participant("A")

    old_cs = simple_atm_cs.to_networkx()
    LocalMutator.mutate_randomly(
        simple_atm_cs, LocalMutationTypes.REMOVE_RANDOM_OUTPUT, target_p=atm
    )

    new_cs = simple_atm_cs.to_networkx()
    assert old_cs.adj != new_cs.adj


def test_mutate_B(simple_atm_cs: CommunicatingSystem) -> None:
    atm = Participant("A")

    old_cs = simple_atm_cs.to_networkx()
    LocalMutator.mutate_randomly(
        simple_atm_cs,
        LocalMutationTypes.CHANGE_RANDOM_TRANSITION_MESSAGE_TYPE,
        target_p=atm,
    )

    new_cs = simple_atm_cs.to_networkx()
    assert old_cs.adj != new_cs.adj


def test_mutate_C(simple_atm_cs: CommunicatingSystem) -> None:
    atm = Participant("A")

    old_cs = simple_atm_cs.to_networkx()
    LocalMutator.mutate_randomly(
        simple_atm_cs, LocalMutationTypes.SWAP_INTERACTION_TYPE, target_p=atm
    )

    new_cs = simple_atm_cs.to_networkx()
    assert old_cs.adj != new_cs.adj

    # def test_mutate_D(simple_atm_cs: CommunicatingSystem) -> None:
    #     atm = Participant("A")

    #     old_cs = simple_atm_cs.to_networkx()
    #     simple_atm_cs.mutate(
    #         MutationTypes.SWAP_RANDOM_CONSECUTIVE_TRANSITIONS, target_p=atm
    #     )

    new_cs = simple_atm_cs.to_networkx()
    assert old_cs.adj != new_cs.adj
