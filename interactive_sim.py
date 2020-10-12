from chorparse.gchor import Participant
from chorparse.cfsm import CFSM, CommunicatingSystem

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

cs.execute_interactively()