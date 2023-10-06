import os
from collections import defaultdict
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import Dict, Generator, List, Dict, Optional, Set, TextIO, Tuple

import networkx as nx

from chortest.common import (
    InTransitionLabel,
    MessageType,
    OutTransitionLabel,
    Participant,
    State,
    Transition,
    TransitionLabel,
    TransitionStr,
)

from .helpers import select

BASE_PATH = Path(__file__).parent


AdjacencyList = Dict[State, Dict[TransitionLabel, State]]


class CFSM:
    states: Set[State]
    transitions: AdjacencyList

    initial: State
    current: State

    def __init__(
        self,
        states: Set[State],
        initial: State,
        transitions: AdjacencyList,
        name: Optional[str] = None,
    ) -> None:
        self.states = states
        self.initial = initial
        self.transitions = transitions

        self.current = initial
        self.name = name

    @staticmethod
    def new(
        transitions: List[Tuple[State, TransitionStr, State]],
        initial: State,
        name: Optional[str] = None,
    ):
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

    def copy(self):
        return CFSM(
            states=self.states.copy(),
            initial=self.initial,
            transitions=deepcopy(self.transitions),
        )

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

    def split(self, q: State | None = None) -> Generator["CFSM", None, None]:
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

    def default_oracle(
        self,
    ):  # TODO: add participant being tested as a parameter (optional)
        """
        An oracle scheme of G for a given projection operator is a function
        f(A, τ) ∈ # P × T(G) on a set of states of the CFSM = proj(G,A).

        The default oracle marks as final states those
        which do not have outgoing transitions.
        """

        return {
            s
            for s in list(self.states)
            if s not in self.transitions or len(self.transitions[s]) == 0
        }

    def all_transitions(
        self,
    ) -> Generator[Tuple[State, TransitionLabel, State], None, None]:
        for q0 in self.states:
            for tr in self.transitions[q0]:
                q1 = self.transitions[q0][tr]
                yield (q0, tr, q1)

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

    def __str__(self) -> str:
        s = "--------\n"
        for trans in self.transitions.values():
            for t in trans.keys():
                s += str(t) + "\n"
        return s + "\n"

    # region to_*

    def to_networkx(self):
        edge_list = [
            (q0, self.transitions[q0][label], {"label": label})
            for q0 in self.transitions
            for label in self.transitions[q0]
        ]
        return nx.MultiDiGraph(incoming_graph_data=edge_list, name=self.name)

    def to_dot(self, path: Optional[str] = None) -> str:
        from networkx.drawing.nx_agraph import to_agraph

        nx = self.to_networkx()
        a = to_agraph(nx)

        if path is not None:
            a.write(path)
        return a.to_string()

    # endregion


class CommunicatingSystem:
    machines: Dict[Participant, CFSM]
    messages: Dict[Tuple[Participant, Participant], List[MessageType]]
    participants: List[
        Participant
    ]  # NOTE: The order matters for dot parsing and compliance checking
    fifo: bool = False

    def __getitem__(self, participant: Participant) -> CFSM:
        return self.machines[participant]

    def __init__(
        self,
        participants: List[Participant],
        cfsms: Dict[Participant, CFSM],
        fifo=False,
    ) -> None:
        self.machines = cfsms
        self.participants = participants
        self.messages = defaultdict(lambda: list())
        self.fifo = fifo

    def copy(self):
        return CommunicatingSystem(
            deepcopy(self.participants), deepcopy(self.machines), self.fifo
        )

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
        self, ignore: Optional[Participant] = None
    ) -> Generator[Tuple[CFSM, State, TransitionLabel, State], None, None]:
        """
        Returns a list of all enabled transitions for the
        machines in the system.
        """
        for part in self.machines:
            cfsm = self.machines[part]
            if part and part == ignore:
                continue
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
        assert CUT in self.machines, f"Invalid participant ({CUT})"
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
        transitions = [x[2] for x in available_choices]

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

                assert choice is not None, "Invalid choice"

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
            transitions = [x[2] for x in available_choices]

        print("Simulation finished.")
        # TODO: Update oracle checking
        # success = all(m.current in m.success for m in self.machines.values())
        # status = "successful! 🎉" if success else "failed ☹️"
        # print(f"Test {status} ")

    def non_deterministic_states(self) -> Generator[State, None, None]:
        for cfsm in self.machines.values():
            yield from cfsm.non_deterministic_states()

    def all_message_types(self) -> Set[MessageType]:
        return set(tr.m for m in self.machines.values() for (q1, tr, q2) in m.all_transitions())

    def all_participant_types(self) -> Set[Participant]:
        return set(self.machines.keys())

    def all_states(self) -> Dict[Participant, Set[State]]:
        return {p: m.states for p, m in self.machines.items()}

    def to_fsa(
        self,
        output_filename: Optional[str] = None,
        part: Participant | None = None,
        output_oracle=True,
    ):
        """
        Outputs the given CS to the given path.
        Optionally accepts a specific participant to output.
        """
        import jinja2
        import yaml

        template = jinja2.Template(
            (BASE_PATH / "templates" / "cfsm2fsa.jinja").read_text()
        )

        ps = self.participants if not part else [part]

        transitions: Dict[
            Participant, List[Tuple[State, int, str, MessageType, State]]
        ] = {}
        initial_states: Dict[Participant, State] = {}

        part_map = {p: i for i, p in enumerate(self.participants)}

        for p in ps:
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
            participants=ps,
            transitions=transitions,
            initial_states=initial_states,
        )

        if output_filename is not None:
            output_path = Path(output_filename)
            os.makedirs(output_path.parent, exist_ok=True)
            Path(output_path).write_text(text)

            if output_oracle:
                with open(f"{output_filename}.oracle.yaml", "w") as o:
                    yaml.dump(
                        {
                            "success_states": {
                                p: list(self.machines[p].default_oracle())
                                for p in self.machines.keys()
                            },
                            "order": list(ps),
                        },
                        o,
                    )

        return text

    def __str__(self) -> str:
        s = ""
        s += "------------\n"
        for p, machine in self.machines.items():
            s += f"[{p}]\n"
            s += str(machine)
        s += "------------"
        return s

    def to_networkx(self, simple_names=False) -> nx.Graph:
        """
        Turns the current CFSM into a single networkX
        graph, renaming the node names according to each machine name.
        """
        graphs = [self.machines[p].to_networkx() for p in self.machines]
        # merge graphs into one, using union
        names = [part for part in self.machines.keys()]

        graph: nx.MultiDiGraph = nx.union_all(graphs, rename=names)
        graph.name = "CFSM"

        if simple_names:
            graph = nx.convert_node_labels_to_integers(graph)
            nx.relabel_nodes

        return graph

    def to_dot(self, output_file: Optional[TextIO] = None) -> str:
        """
        Writes the current communicating system to the given
        stream, in dot format.
        """
        nx_graph = self.to_networkx()
        from networkx.drawing.nx_agraph import to_agraph

        dot_graph = to_agraph(nx_graph)

        output_str = dot_graph.to_string()
        if output_file:
            output_file.write(output_str)
        return output_str
