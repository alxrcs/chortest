import os
from collections import OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import (
    Dict,
    Generator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from lark import Lark, Transformer
from lark.lexer import Token
from lark.tree import Tree

from .gchor import Participant
from .helpers import select

State = str
TransitionStr = str
Message = str

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
        self, states: Set[State], initial: State, transitions: AdjacencyList, name: Optional[str] = None
    ) -> None:
        self.states = states
        self.initial = initial
        self.transitions = transitions

        self.success = self.default_oracle()

        self.current = initial
        self.name = name

    # An oracle scheme of G for a given projection operator is a function f(A, τ) ∈ # P × T(G) on a set of states of the CFSM proj(G,A)
    def default_oracle(self):
        """The default oracle marks as final states those
        which do not have outgoing transitions"""

        # TODO: remove 's in self.transitions'
        return {
            s
            for s in list(self.states)
            if s not in self.transitions or len(self.transitions[s]) == 0
        }

    @staticmethod
    def new(transitions: List[Tuple[State, TransitionStr, State]], initial: State, name: Optional[str] = None):
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

        return CFSM(states, initial, _transitions, name)

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
            if nds:
                # Union_{q\in{nds(M)}} split(M,q)
                for qnds in nds[:1]:
                    # for qnds in nds:
                    yield from self.split(qnds)
            else:  # if nds(M) is empty
                yield self.copy()
        else:
            # split(M, q)
            output_transitions = list(
                filter(
                    lambda x: isinstance(x, OutTransitionLabel),
                    self.transitions[q].keys(),
                )
            )
            # if M(q) has output transitions
            if output_transitions:
                for t in output_transitions:
                    new_m = self.copy()
                    # TODO: iterate over output_transitions - t to avoid the if
                    for ot in self.transitions[q]:
                        if ot is not t:
                            del new_m.transitions[q][ot]
                    yield from new_m.split()

                ### From yesterday
                # for ot in self.transitions[q]:
                #     if ot is not t and isinstance(ot, InTransitionLabel):
                #         ns1 = new_m.transitions[q][t]
                #         ns2 = new_m.transitions[q][ot]

                #         if new_m.transitions[ns1][ot] == new_m.transitions[ns2][t]:
                # del new_m.transitions[q][ot]
                ### End yesterday

            else:
                for t1, t2 in combinations(self.transitions[q], 2):
                    # TODO: Only split if the labels are the same but go to different states
                    q1 = self.transitions[q][t1]
                    q2 = self.transitions[q][t2]
                    if t1 == t2 and q1 != q2:
                        m = self.copy()
                        yield from (m - Transition(q, t1, q1)).split()
                        yield from (m - Transition(q, t2, q2)).split()

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
        return nx.DiGraph(incoming_graph_data=edge_list, name=self.name)

    def to_dot(self, path: Optional[str] = None) -> str:
        from networkx.drawing.nx_agraph import to_agraph

        nx = self.to_networkx()
        a = to_agraph(nx)

        if path is not None:
            a.write(path)
        return a.to_string()


class CommunicatingSystem:
    machines: Mapping[Participant, CFSM]
    messages: Mapping[Tuple[Participant, Participant], List[Message]]
    participants: List[
        Participant
    ]  # NOTE: The order matters for dot parsing and compliance checking
    fifo: bool = False

    def __init__(
        self,
        participants: List[Participant],
        cfsms: Mapping[Participant, CFSM],
        fifo=False,
    ) -> None:
        self.machines = cfsms
        self.participants = participants
        self.messages = defaultdict(lambda: list())
        self.fifo = fifo

    def is_enabled(self, t: TransitionLabel) -> bool:
        """
        A transition is enabled if it is either a send,
        or a receive and there is a message in the
        corresponding buffer.
        """
        if isinstance(t, OutTransitionLabel):
            # Send transitions are always enabled
            return True

        idx = (t.A, t.B)

        # Receiving transitions are enabled if the right message
        # is in the corresponding buffer
        if self.fifo:
            return t.m == self.messages[idx][0]
        else:
            return t.m in self.messages[idx]

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

        idx = (t.A, t.B)

        if isinstance(t, OutTransitionLabel):
            self.messages[idx].append(t.m)
        elif isinstance(t, InTransitionLabel):
            self.messages[idx].remove(t.m)
        else:
            raise ValueError("Invalid transition.")

        cfsm.current = v2

    def tests(self, CUT: Participant) -> Generator["CommunicatingSystem", None, None]:
        """
        Generates a set of tests for the given participant.
        """
        assert CUT in self.machines, f"Invalid participant ({CUT.participant_name})"
        split_machines = dict(
            {
                p: list(self.machines[p].split())
                for p in self.participants
                if p is not CUT
            }
        )
        split_machines[CUT] = [self.machines[CUT]]

        for test_cfsms in select(list(split_machines.items())):
            yield CommunicatingSystem(self.participants, dict(test_cfsms))

    def execute_interactively(self):
        import inquirer

        print("Starting interactive simulation... \n")

        available_choices = list(self.enabled_transitions())
        transitions = list(map(lambda x: x[2], available_choices))

        while available_choices:
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
        success = all(m.current in m.success for m in self.machines.values())
        status = "successful! 🎉" if success else "failed ☹️"
        print(f"Test {status} ")

    def non_deterministic_states(self) -> Generator[State, None, None]:
        for cfsm in self.machines.values():
            yield from cfsm.non_deterministic_states()

    @staticmethod
    def parse(cs_filename) -> "CommunicatingSystem":
        """
        Parses a communicating system from an .fsa file.
        """

        grammarfile_path = Path("grammars") / "fsa.lark"
        fsa_parser = Lark.open(str(grammarfile_path))

        with open(cs_filename) as f:
            tree = fsa_parser.parse(f.read())

        transformer = CFSMBuilder()
        return transformer.transform(tree)

    def to_fsa(self, output_filename: Optional[str] = None):
        import jinja2
        import yaml

        template_str = open("templates/cfsm2fsa.jinja").read()
        template = jinja2.Template(template_str)

        transitions: Dict[
            Participant, List[Tuple[State, int, str, Message, State]]
        ] = {}
        initial_states: Dict[Participant, State] = {}

        part_map = {p.participant_name: i for i, p in enumerate(self.participants)}

        for p in self.participants:
            l = []
            cfsm = self.machines[p]
            for q in cfsm.transitions:
                for t in cfsm.transitions[q]:
                    # TODO: Add tau transitions.
                    assert isinstance(t, (InTransitionLabel, OutTransitionLabel))
                    symbol = "?" if isinstance(t, InTransitionLabel) else "!"
                    target = t.B if symbol == "!" else t.A
                    l.append(
                        (q, part_map[str(target)], symbol, t.m, cfsm.transitions[q][t])
                    )
            transitions[p] = l
            initial_states[p] = cfsm.initial

        text = template.render(
            participants=self.participants,
            transitions=transitions,
            initial_states=initial_states,
        )

        if output_filename is not None:
            os.makedirs(Path(output_filename).parent, exist_ok=True)
            with open(output_filename, "wb") as f:
                f.write(text.encode())
            with open(Path(output_filename).parent.joinpath("oracle.yaml"), "w") as o:
                oracle = {
                    p.participant_name: list(self.machines[p].success)
                    for p in self.machines.keys()
                }
                yaml.dump(
                    {
                        "success_states": oracle,
                        "order": list(
                            map(lambda x: x.participant_name, self.participants)
                        ),
                    },
                    o,
                )

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

    def to_dot(self, output_folder : Optional[str]) -> List[str]:

        dot_machines = [self.machines[p].to_dot() for p in self.machines]

        if output_folder:
            p = Path(output_folder)
            p.mkdir(exist_ok=True)
            for i, m in enumerate(dot_machines):
                with open(p / f'machine_{i}.dot', 'w') as f:
                    f.write(m)

        return dot_machines


class CFSMBuilder(Transformer):
    """
    Constructs a communicating system from a parse tree
    of an .fsa file.
    """

    def __init__(self) -> None:
        # Global info
        self.cs: Dict[Participant, CFSM] = {}
        self.participants = []

    def start(self, cfsms: List[Tuple[Participant, CFSM]]):
        for i, t in enumerate(cfsms):
            p, cfsm = t
            for q in cfsm.transitions:
                trs = cfsm.transitions[q]
                trs_new: Dict[TransitionLabel, State] = {}
                for tr in trs:
                    if isinstance(tr, InTransitionLabel):
                        trs_new[
                            InTransitionLabel(
                                cfsms[int(tr.B.participant_name)][0], p, tr.m
                            )
                        ] = trs[tr]
                    elif isinstance(tr, OutTransitionLabel):
                        trs_new[
                            OutTransitionLabel(
                                p, cfsms[int(tr.B.participant_name)][0], tr.m
                            )
                        ] = trs[tr]
                    else:
                        raise Exception("Shouldn't happen.")
                cfsm.transitions[q] = trs_new

        return CommunicatingSystem(self.participants, self.cs)

    def graph(self, t):
        name = t[0]
        self.cs[name] = CFSM.new(transitions=t[1], initial=t[2], name=name)
        return (name, self.cs[name])

    def header(self, t):
        p = Participant(str(t[0]))
        self.participants.append(p)
        return p

    def edges(self, l):
        return l

    def markings(self, l):
        assert len(l) == 1  # There should be a single state marked as initial
        return str(l[0])

    def receive_msg(self, t: List[Union[Tree, Token]]):
        # _ is a temporary placeholder. Fixed in the 'start' rule.
        label = InTransitionLabel(
            Participant("_"), Participant(str(t[1])), Message(str(t[2]))
        )
        return (str(t[0]), label, str(t[3]))

    def send_msg(self, t: List[Union[Tree, Token]]):
        # _ is a temporary placeholder. Fixed in the 'start' rule.
        label = OutTransitionLabel(
            Participant("_"), Participant(str(t[1])), Message(str(t[2]))
        )
        return (str(t[0]), label, str(t[3]))

    def empty(self, t):
        return (str(t[0]), TransitionLabel.empty(), str(t[1]))
