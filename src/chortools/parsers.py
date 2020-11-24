from os import name
from dataclasses import dataclass
from lark.lexer import Token
from lark.visitors import Transformer
from typing import List


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
    states: List[State]
    transitions: List[Transition]

class FSATransformer(Transformer):
    def parameter(self, t):
        return Parameter(
            name=t[0], domain_cardinality=int(t[1]), 
            domain_name=t[2], domain_values=t[3:-1]
        )

    def transition(self, t):
        return Transition(int(t[0]), int(t[1]), t[2])

    def state(self, t):
        return State(list(map(int, t[:-1])))

    def start(self, t):
        t = list(filter(lambda x: x is not Token, t))
        return LTS(parameters=t[0], states=t[1], transitions=t[2])