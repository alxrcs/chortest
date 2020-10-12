from chorparse.gchor import Participant
import pytest
from chorparse.cfsm import CFSM, CommunicatingSystem, Transition
import inquirer

class TestCFSM:
    def test_create_cfsm(self):
        bank_m = CFSM(
            states={f"B{i}" for i in range(1, 5)},
            initial="B1",
            transitions={
                "B1": {"AB?authW": "B2"},
                "B2": {"BA!allow": "B4", "BA!deny": "B3"},
            },
        )

        assert bank_m.transitions["B1"]["AB?authW"] == "B2"
        assert bank_m.transitions["B2"]["BA!allow"] == "B4"
        assert bank_m.transitions["B2"]["BA!deny"] == "B3"
        assert len(bank_m.states) == 4
        assert "B1" in bank_m.states
        assert "B4" in bank_m.states

    def test_create_new_cfsm(self):
        bank_m = CFSM.new(
            [
                ("B1", "AB?authW", "B2"),
                ("B2", "BA!allow", "B4"),
                ("B2", "BA!deny", "B3"),
            ],
            "B1",
        )

        # TODO: Refactor these indexings to use only a public api
        assert bank_m.transitions["B1"][Transition.new("AB?authW")] == "B2"
        assert bank_m.transitions["B2"][Transition.new("BA!allow")] == "B4"
        assert bank_m.transitions["B2"][Transition.new("BA!deny")] == "B3"
        assert len(bank_m.states) == 4
        assert "B1" in bank_m.states
        assert "B4" in bank_m.states

    @pytest.fixture
    def simple_atm_cs(self):
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
                Participant("ATM"): atm_m,
                Participant("Bank"): bank_m,
                Participant("Client"): client_m,
            }
        )

        return cs

    @pytest.mark.wip
    @pytest.mark.cfsm
    def test_create_new_cs(self, simple_atm_cs):
        
        print()
        while True:
            for cfsm, a, t, b in simple_atm_cs.enabled_transitions():
                print('transition: ', t)
                simple_atm_cs.fire_transition(cfsm, t, a, b)
                break
            else:
                break

