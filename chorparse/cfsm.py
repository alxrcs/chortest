from collections import defaultdict
from copy import Error
from dataclasses import dataclass
from typing import List, Mapping, Sequence, Set, Tuple
import networkx as nx  # type: ignore

from .gchor import Message, Participant

Node = str

@dataclass
class Transition:
    A: Participant
    B: Participant
    m: Message
    "Represents an message transition, either incoming or outgoing"

    @staticmethod
    def new(act):
        "Convenience method. Assumes a specific format (AB[?|!]msg)."
        # TODO: avoid using this method
        # or extend it for arbitrary sized participant names
        if act[2] == "!":
            return SendTransition(
                Participant(act[0]), Participant(act[1]), Message(act[3:])
            )
        elif act[2] == "?":
            return RecvTransition(
                Participant(act[0]), Participant(act[1]), Message(act[3:])
            )
        else:
            raise RuntimeError("Incorrect action format.")


@dataclass
class SendTransition(Transition):
    "Represents an outgoing message transition, e.g. AB!m"

    def __repr__(self) -> str:
        return f"{self.A}-{self.B}!{self.m}"


@dataclass
class RecvTransition(Transition):
    "Represents an incoming message transition, e.g. AB?m"

    def __repr__(self) -> str:
        return f"{self.A}-{self.B}?{self.m}"


class CFSM:
    # TODO: decide whether to leave this as just a name or
    # fix it as a Participant (which would disallow non-local machines)
    name: str
    dfa: nx.MultiDiGraph
    initial_state: Node
    current_state: Node

    def __init__(self, name: str, initial_state: Node) -> None:
        self.dfa = nx.MultiDiGraph()
        self.name = name
        self.initial_state = initial_state
        self.current_state = initial_state

    def add_state(self, state_name: str) -> None:
        self.dfa.add_node(state_name)

    def add_transition(self, source: Node, dest: Node, tr_label: str) -> None:
        self.dfa.add_edge(source, dest, key=tr_label, label=Transition.new(tr_label))
        # TODO: Check if the label is actually needed

    def transitions(self) -> Sequence[Tuple[Node, Node, Transition]]:
        for a,b,l in self.dfa.edges(self.current_state, data=True):
            yield a,b,l['label']

    def all_transitions(self) -> Sequence[Tuple[Node, Node, Transition]]:
        return self.dfa.edges.data("label")


class CommunicatingSystem:
    machines: Mapping[Participant, CFSM]
    messages: Mapping[Tuple[Participant, Participant], List[Message]]

    participants: Set[Participant]
    message_types: Set[Message]

    def __init__(self, cfsms: List[CFSM]) -> None:
        self.participants = set()
        self.message_types = set()
        self.machines = {}

        for cfsm in cfsms:
            self.machines[Participant(cfsm.name)] = cfsm
            self.participants.add(cfsm.name)
            for (_, _, l) in cfsm.dfa.edges.data("label"):
                self.message_types.add(str(l.m))

        self.messages = defaultdict(lambda: list())

    def is_enabled(self, t: Transition) -> bool:
        """
        A transition is enabled if it is either a send,
        or a receive and there is a message in the 
        corresponding buffer.
        """
        if isinstance(t, SendTransition):
            return True # Send transitions are always enabled
        # Receiving transitions are enabled if the right message is
        # in the corresponding buffer 
        # TODO: This is probably the place to change semantics to FIFO
        # if required
        return t.m in self.messages[(t.A, t.B)] 

    def enabled_transitions(
        self,
    ) -> Sequence[Tuple[CFSM, Participant, Participant, Transition]]:
        """
        Returns a list of all enabled transitions for the 
        machines in the system.
        """
        for cfsm in self.machines.values():
            for v1, v2, transition in cfsm.transitions():
                if self.is_enabled(transition):
                    yield cfsm, v1, v2, transition

    def fire_transition(self, cfsm: CFSM, t: Transition, v1: Node, v2: Node):
        "Fires a transition in the current communicating system."
        assert self.is_enabled(t, t.A, t.B)
        if isinstance(t, SendTransition):
            self.messages[(t.A, t.B)].append(t.m)
        elif isinstance(t, RecvTransition):
            self.messages[(t.A, t.B)].remove(t.m)
        else:
            raise Error("Invalid transition.")

        cfsm.current_state = v2

