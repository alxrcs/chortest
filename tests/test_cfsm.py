import pytest
from chorparse.cfsm import CFSM, CommunicatingSystem

# @pytest.mark.wip
class TestCFSM():
    def test_create_new_cfsm(self):
        bank_m = CFSM("Bank", 'B1')
        for i in range(4):
            bank_m.add_state(f"B{i+1}")
        bank_m.add_transition('B1', 'B2', 'AB?authW')
        bank_m.add_transition('B2', 'B4', 'BA!allow')
        bank_m.add_transition('B2', 'B3', 'BA!deny')

    def test_create_new_cs(self):
        atm_m = CFSM("ATM", 'A1')
        for i in range(7):
            atm_m.add_state(f"A{i+1}")
        atm_m.add_transition('A1', 'A2', 'CA?withdraw')
        atm_m.add_transition('A2', 'A3', 'AB!authW')
        atm_m.add_transition('A3', 'A6', 'BA?allow')
        atm_m.add_transition('A3', 'A4', 'BA?deny')
        atm_m.add_transition('A6', 'A7', 'AC!money')
        atm_m.add_transition('A4', 'A5', 'AC!bye')

        bank_m = CFSM("Bank", 'B1')
        for i in range(4):
            bank_m.add_state(f"B{i+1}")
        bank_m.add_transition('B1', 'B2', 'AB?authW')
        bank_m.add_transition('B2', 'B4', 'BA!allow')
        bank_m.add_transition('B2', 'B3', 'BA!deny')

        client_m = CFSM("Client", 'C1')
        for i in range(4):
            bank_m.add_state(f"C{i+1}")
        bank_m.add_transition('C1', 'C2', 'CA!withdraw')
        bank_m.add_transition('C2', 'C4', 'AC?money')
        bank_m.add_transition('C2', 'C3', 'AC?bye')

        cs = CommunicatingSystem([atm_m, bank_m, client_m])
        for t in cs.enabled_transitions():
            print(t)
        



