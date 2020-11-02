from io import StringIO
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import (
    AbstractSet,
    Dict,
    Generator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from attr import s
from lark import Lark, Transformer
from lark.lexer import Token
from lark.tree import Tree

from .helpers import select
from .gchor import Message, Participant

State = str
TransitionStr = str

import networkx as nx


@dataclass
class TransitionLabel:
    "Represents an message transition, either incoming or outgoing"

    _empty_instance = None

    @staticmethod
    def empty():
        if not TransitionLabel._empty_instance:
            TransitionLabel._empty_instance = TransitionLabel(
                Participant(""), Participant(""), ""
            )
        return TransitionLabel._empty_instance

    A: Participant
    B: Participant
    m: Message

    @staticmethod
    def new(act):
        "Convenience method. Assumes a specific format (AB[?|!]msg)."
        # TODO: avoid using this method
        # or extend it for arbitrary sized participant names
        if act[2] == "!":
            return OutTransitionLabel(
                Participant(act[0]), Participant(act[1]), Message(act[3:])
            )
        elif act[2] == "?":
            return InTransitionLabel(
                Participant(act[0]), Participant(act[1]), Message(act[3:])
            )
        else:
            raise RuntimeError("Incorrect action format.")

    def __hash__(self) -> int:
        return self.__repr__().__hash__()


@dataclass
class Transition:
    q1: State
    l: TransitionLabel
    q2: State


@dataclass
class OutTransitionLabel(TransitionLabel):
    "Represents an outgoing message transition, e.g. AB!m"

    def __repr__(self) -> str:
        return f"{self.A}-{self.B}!{self.m}"

    def __hash__(self) -> int:
        return self.__repr__().__hash__()


@dataclass
class InTransitionLabel(TransitionLabel):
    "Represents an incoming message transition, e.g. AB?m"

    def __repr__(self) -> str:
        return f"{self.A}-{self.B}?{self.m}"

    def __hash__(self) -> int:
        return self.__repr__().__hash__()


AdjacencyList = Dict[State, Dict[TransitionLabel, State]]


class CFSM:
    states: Set[State]
    transitions: AdjacencyList

    success: Set[State]

    initial: State
    current: State

    def __init__(
        self, states: Set[State], initial: State, transitions: AdjacencyList
    ) -> None:
        self.states = states
        self.initial = initial
        self.transitions = transitions

        self.success = self.default_oracle()

        self.current = initial

    def default_oracle(self):
        return set(s for s in list(self.states) if s not in self.transitions)

    @staticmethod
    def new(transitions: List[Tuple[State, TransitionStr, State]], initial: State):
        """
        Convenience method for building CFSMs.
        Expects a list of triples <state> <transition> <state>.
        """
        states = set()
        _transitions: AdjacencyList = defaultdict(lambda: defaultdict(lambda: ""))

        for q0, t, q1 in transitions:
            states.add(q0)
            states.add(q1)
            if isinstance(t, TransitionLabel):
                _transitions[q0][t] = q1
            else:
                _transitions[q0][TransitionLabel.new(t)] = q1

        return CFSM(states, initial, _transitions)

    def non_deterministic_states(self):
        for q in self.states:
            for l1, l2 in combinations(self.transitions[q], 2):
                q1, q2 = self.transitions[q][l1], self.transitions[q][l2]
                if l1 is l2 or q1 is q2:
                    continue
                if l1 == l2 or (
                    isinstance(l1, OutTransitionLabel)
                    or isinstance(l2, OutTransitionLabel)
                ):
                    yield q

    def split(self, q: State = None) -> Generator["CFSM", None, None]:
        if q is None:
            # split(M)
            nds = list(self.non_deterministic_states())
            if len(nds) == 0:  # if nds(M) is empty
                yield self.copy()
            else:
                # Union_{q\in{nds(M)}} split(M,q)
                for q in nds:
                    yield from self.split(q)
        else:
            # split(M, q)
            output_transitions = list(
                filter(
                    lambda x: isinstance(x, OutTransitionLabel),
                    self.transitions[q].keys(),
                )
            )
            # if M(q) has output transitions
            if len(output_transitions) > 0:
                for t in self.transitions[q]:
                    new_m = self.copy()
                    for ot in output_transitions:
                        if ot is not t:
                            del new_m.transitions[q][t]  # Check if this is correct
                    yield from new_m.split()
            else:
                for t1, t2 in combinations(self.transitions[q], 2):
                    q1 = self.transitions[q][t1]
                    q2 = self.transitions[q][t2]
                    if q1 != q2:
                        m = self.copy()
                        yield from (m - Transition(q, t1, q1)).split()
                    else:  # TODO: What happens if there are two input transitions to the same target state?
                        raise ValueError()

    # region operator overloads
    def __add__(self, t: Transition) -> "CFSM":
        "Returns a new machine with the transition"

        newcfsm = self.copy()
        newcfsm.states.add(t.q1)
        newcfsm.states.add(t.q2)
        newcfsm.transitions[t.q1][t.l] = t.q2

        return newcfsm

    def __sub__(self, t: Transition) -> "CFSM":
        newcfsm = self.copy()
        newcfsm.transitions[t.q1].pop(t.l)
        return newcfsm
        # TODO: Prune states

    # endregion

    def copy(self):
        return CFSM(
            states=self.states.copy(),
            initial=self.initial,
            transitions=deepcopy(self.transitions),
        )

    def __str__(self) -> str:
        s = "--------\n"
        for trans in self.transitions.values():
            for t in trans.keys():
                s += str(t) + "\n"
        return s + "\n"

    def to_networkx(self):
        edge_list = [
            (q0, self.transitions[q0][label], {"label": label})
            for q0 in self.transitions
            for label in self.transitions[q0]
        ]
        return nx.DiGraph(edge_list)

    def to_dot(self, path: Optional[str] = None) -> str:
        from networkx.drawing.nx_agraph import to_agraph

        a = to_agraph(self.to_networkx())
        if path is not None:
            a.write(path)
        return a.to_string()


class CommunicatingSystem:
    machines: Mapping[Participant, CFSM]
    messages: Mapping[Tuple[Participant, Participant], List[Message]]

    fifo: bool = False

    def __init__(self, cfsms: Mapping[Participant, CFSM], fifo=False) -> None:
        self.machines = cfsms
        self.messages = defaultdict(lambda: list())
        self.fifo = fifo

    def participants(self) -> AbstractSet[Participant]:
        return self.machines.keys()

    def is_enabled(self, t: TransitionLabel) -> bool:
        """
        A transition is enabled if it is either a send,
        or a receive and there is a message in the
        corresponding buffer.
        """
        if isinstance(t, OutTransitionLabel):
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
    ) -> Generator[Tuple[CFSM, State, TransitionLabel, State], None, None]:
        """
        Returns a list of all enabled transitions for the
        machines in the system.
        """
        for cfsm in self.machines.values():
            for t in cfsm.transitions[cfsm.current]:
                if self.is_enabled(t):
                    next_state = cfsm.transitions[cfsm.current][t]
                    yield (cfsm, cfsm.current, t, next_state)

    def fire_transition(self, cfsm: CFSM, t: TransitionLabel, v1: State, v2: State):
        "Fires a transition in the current communicating system."

        assert self.is_enabled(t)
        assert cfsm.current == v1

        if isinstance(t, OutTransitionLabel):
            self.messages[(t.A, t.B)].append(t.m)
        elif isinstance(t, InTransitionLabel):
            self.messages[(t.A, t.B)].remove(t.m)
        else:
            raise ValueError("Invalid transition.")

        cfsm.current = v2

    def tests(
        self, CUT: Participant, output_path: Optional[str] = None
    ) -> Generator["CommunicatingSystem", None, None]:
        assert CUT in self.machines, f"Invalid participant ({CUT.participant_name})"
        split_machines = dict(
            {
                p: list(self.machines[p].split())
                for p in self.participants()
                if p is not CUT
            }
        )
        split_machines[CUT] = [self.machines[CUT]]

        for i, test_cfsms in enumerate(select(list(split_machines.items()))):
            cs = CommunicatingSystem(dict(test_cfsms))
            if output_path is not None:
                path = str(Path(output_path) / f"test_{i}" / f"test_{i}.fsa")
                cs.to_fsa(path)
            yield cs

    def execute_interactively(self):
        import inquirer

        print("Starting interactive simulation... \n")

        available_choices = list(self.enabled_transitions())
        transitions = list(map(lambda x: x[2], available_choices))

        while len(available_choices) > 0:
            # while len(available_transitions) > 0:

            if len(available_choices) > 1:
                choice = inquirer.prompt(
                    questions=[
                        inquirer.List(
                            name="action",
                            message="Choose an action to fire",
                            choices=transitions,
                        )
                    ]
                )

                # TODO: improve choice handling
                # (perhaps remove v1 and v2 from the generator)
                # (and add the cfsm name to the choice list)
                i = transitions.index(choice["action"])
                cfsm, v1, t, v2 = available_choices[i]
            else:
                cfsm, v1, t, v2 = available_choices[0]

            print(f"Firing {t}")
            self.fire_transition(cfsm, t=t, v1=v1, v2=v2)

            available_choices = list(self.enabled_transitions())
            transitions = list(map(lambda x: x[2], available_choices))

        print("Simulation finished.")

    def non_deterministic_states(self) -> Generator[State, None, None]:
        for cfsm in self.machines.values():
            yield from cfsm.non_deterministic_states()

    # def execute(self):
    #     class TransitionSystem:
    #         states: Mapping[Participant, State]
    #         messages: Mapping[Tuple[Participant, Participant], List[Message]]

    @staticmethod
    def parse(cs_filename) -> 'CommunicatingSystem':

        grammarfile_path = Path("grammars") / "fsa.lark"
        fsa_parser = Lark.open(str(grammarfile_path))

        with open(cs_filename) as f:
            tree = fsa_parser.parse(f.read())

        transformer = CFSMBuilder()
        communicating_system = transformer.transform(tree)

        return communicating_system

    def to_fsa(self, output_filename: Optional[str] = None):
        import jinja2

        template_str = open("templates/cfsm2fsa.jinja").read()
        template = jinja2.Template(template_str)

        transitions: Dict[
            Participant, List[Tuple[State, int, str, Message, State]]
        ] = {}
        initial_states: Dict[Participant, State] = {}

        part_map = {p.participant_name: i for i, p in enumerate(self.participants())}


        for p in self.participants():
            l = []
            cfsm = self.machines[p]
            for q in cfsm.transitions:
                for t in cfsm.transitions[q]:
                    # TODO: Add tau transitions.
                    assert isinstance(t, InTransitionLabel) or isinstance(
                        t, OutTransitionLabel
                    )
                    symbol = "?" if isinstance(t, InTransitionLabel) else "!"
                    target = t.B if symbol is "!" else t.A
                    l.append(
                        (q, part_map[str(target)], symbol, t.m, cfsm.transitions[q][t])
                    )
            transitions[p] = l
            initial_states[p] = cfsm.initial

        text = template.render(
            participants=list(self.participants()),
            transitions=transitions,
            initial_states=initial_states,
        )

        if output_filename is not None:
            os.makedirs(Path(output_filename).parent, exist_ok=True)
            with open(output_filename, "wb") as f:
                f.write(text.encode())

        return text

    def __str__(self) -> str:
        s = ""
        s += "------------\n"
        for p, machine in self.machines.items():
            s += f"[{p.participant_name}]\n"
            s += str(machine)
        s += "------------"
        return s

    def to_networkx(self) -> List[nx.Graph]:
        return [self.machines[p].to_networkx() for p in self.machines]


class CFSMBuilder(Transformer):
    """
    Constructs a communicating system from a parse tree
    of an .fsa file.

    TODO?: Eliminate participant(_)?
    TODO: Change numbered participants in transitions for their actual name.
    """

    def __init__(self) -> None:
        # Global info
        self.cs: Mapping[Participant, CFSM] = {}

    def start(self, graphs):
        return CommunicatingSystem(self.cs)

    def graph(self, t):
        name = t[0]
        self.cs[name] = CFSM.new(transitions=t[1], initial=t[2])
        return (name, self.cs[name])

    def header(self, t):
        return Participant(str(t[0]))

    def edges(self, l):
        return l

    def markings(self, l):
        assert len(l) == 1  # There should be a single state marked as initial
        return str(l[0])

    def receive_msg(self, t: List[Union[Tree, Token]]):
        label = InTransitionLabel(
            Participant("_"), Participant(str(t[1])), Message(str(t[2]))
        )
        return (str(t[0]), label, str(t[3]))

    def send_msg(self, t: List[Union[Tree, Token]]):
        label = OutTransitionLabel(
            Participant("_"), Participant(str(t[1])), Message(str(t[2]))
        )
        return (str(t[0]), label, str(t[3]))

    def empty(self, t):
        return (str(t[0]), TransitionLabel.empty(), str(t[1]))


# An oracle scheme of G for a given projection operator is a function f(A, τ) ∈ # P × T(G) on a set of states of the CFSM proj(G,A)
