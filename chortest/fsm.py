from dataclasses import dataclass
from pathlib import Path
from typing import List

from lark import Transformer


@dataclass
class Parameter:
    name: str
    domain_cardinality: int
    domain_name: str
    domain_values: List[str]


@dataclass
class Transition:
    source_state: int
    target_state: int
    transition_label: str


@dataclass
class State:
    values: List[int]


@dataclass
class LTS:
    parameters: List[Parameter]
    configurations: List[State]
    transitions: List[Transition]


class FSMTransformer(Transformer):
    # TODO: Finish proper FSM parser
    def parameter(self, t):
        return Parameter(
            name=t[0],
            domain_cardinality=int(t[1]),
            domain_name=t[2],
            domain_values=t[3:-1],
        )

    def transition(self, t):
        return Transition(int(t[0]), int(t[1]), t[2])

    def state(self, t):
        return State(list(map(int, t[:-1])))

    def start(self, t):
        return LTS(parameters=t[0], configurations=t[1], transitions=t[2])

