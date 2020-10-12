from collections import defaultdict
from dataclasses import dataclass
from typing import Generator, List, Mapping, Sequence, Set, Tuple
import networkx as nx  # type: ignore

from .gchor import Message, Participant

State = str
TransitionStr = str

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

    def __hash__(self) -> int:
        return self.__repr__().__hash__()


@dataclass
class SendTransition(Transition):
    "Represents an outgoing message transition, e.g. AB!m"

    def __repr__(self) -> str:
        return f"{self.A}-{self.B}!{self.m}"

    
    def __hash__(self) -> int:
        return self.__repr__().__hash__()


@dataclass
class RecvTransition(Transition):
    "Represents an incoming message transition, e.g. AB?m"

    def __repr__(self) -> str:
        return f"{self.A}-{self.B}?{self.m}"


    def __hash__(self) -> int:
        return self.__repr__().__hash__()


AdjacencyList = Mapping[State, Mapping[Transition, State]]

class CFSM:
    states: Set[State]
    transitions: AdjacencyList

    initial: State
    current: State

    def __init__(
        self,
        states: Set[State],
        initial: State,
        transitions: AdjacencyList
    ) -> None:
        self.states = states
        self.initial = initial
        self.transitions = transitions

        self.current = initial

    @staticmethod
    def new(transitions: List[Tuple[State, TransitionStr, State]], initial: State):
        """
        Convenience method for building CFSMs. 
        Expects a list of triples <state> <transition> <state>.
        """
        states = set()
        _transitions : AdjacencyList = defaultdict(lambda: defaultdict(lambda: {}))

        for q0, t, q1 in transitions:
            states.add(q0)
            states.add(q1)
            _transitions[q0][Transition.new(t)] = q1

        return CFSM(states, initial, _transitions)


class CommunicatingSystem:
    machines: Mapping[Participant, CFSM]
    messages: Mapping[Tuple[Participant, Participant], List[Message]]
    participants: Set[Participant]
    fifo: bool

    def __init__(self, cfsms: Mapping[Participant, CFSM], fifo=False) -> None:
        self.participants = {participant for participant, _ in cfsms.items()}
        self.messages = defaultdict(lambda: list())
        self.machines = cfsms
        self.fifo = fifo

    def is_enabled(self, t: Transition) -> bool:
        """
        A transition is enabled if it is either a send,
        or a receive and there is a message in the 
        corresponding buffer.
        """
        if isinstance(t, SendTransition):
            # Send transitions are always enabled
            return True

        # Receiving transitions are enabled if the right message
        # is in the corresponding buffer
        if self.fifo:
            return t.m == self.messages[(t.A, t.B)][0]
        else:
            return t.m in self.messages[(t.A, t.B)]

    def enabled_transitions(
        self,
    ) -> Generator[Tuple[CFSM, State, Transition, State], None, None]:
        """
        Returns a list of all enabled transitions for the 
        machines in the system.
        """
        for cfsm in self.machines.values():
            for t in cfsm.transitions[cfsm.current]:
                if self.is_enabled(t):
                    next_state = cfsm.transitions[cfsm.current][t]
                    yield (cfsm, cfsm.current, t, next_state)


    def fire_transition(self, cfsm: CFSM, t: Transition, v1: State, v2: State):
        "Fires a transition in the current communicating system."

        assert self.is_enabled(t)
        assert cfsm.current == v1

        if isinstance(t, SendTransition):
            self.messages[(t.A, t.B)].append(t.m)
        elif isinstance(t, RecvTransition):
            self.messages[(t.A, t.B)].remove(t.m)
        else:
            raise ValueError("Invalid transition.")

        cfsm.current = v2

    def execute_interactively(self):
        import inquirer

        print('Starting interactive simulation... \n')

        available_choices = list(self.enabled_transitions())
        transitions = list(map(lambda x: x[2],available_choices))

        while len(available_choices) > 0:
            # while len(available_transitions) > 0:
            choice = inquirer.prompt(
                questions=[inquirer.List(name='action', message='Choose an action to fire', choices=transitions)]
            )

            # TODO: improve choice handling 
            # (perhaps remove v1 and v2 from the generator)
            # (and add the cfsm name to the choice list)

            cfsm, v1, t, v2 = available_choices[transitions.index(choice['action'])]

            self.fire_transition(cfsm, t=t, v1=v1, v2=v2)

            available_choices = list(self.enabled_transitions())
            transitions = list(map(lambda x: x[2],available_choices))

        print('Simulation finished.')
        


    def execute(self):
        class TransitionSystem:
            states: Mapping[Participant, State]
            messages: Mapping[Tuple[Participant, Participant], List[Message]]

            
        







