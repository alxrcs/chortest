from dataclasses import dataclass
from pathlib import Path
from typing import List

from chortools.gchor import GTransformer


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


class FSATransformer(GTransformer):
    # TODO: Finish proper FSA parser
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


class FSACombiner:
    def parse(
        self,
        input_fsa_filename: str,
        replacement_fsa_filename: str,
        output_filename: str = None,
    ):
        import re

        output_filename = output_filename or str(
            Path(input_fsa_filename).with_suffix(".fsa.test")
        )

        with open(input_fsa_filename, "r") as input_fsa_file, open(
            replacement_fsa_filename, "r"
        ) as replacement_fsa_file, open(output_filename, "w") as output_fsa_file:

            parse_fsm = lambda txt: {
                f[1]: f[0]
                for f in re.findall(r"(\.outputs (\w+).*?\.end)", txt, re.DOTALL)
            }

            updated_fsm = parse_fsm(input_fsa_file.read())
            updated_fsm.update(parse_fsm(replacement_fsa_file.read()))

            txt = "\n\n".join(updated_fsm.values())
            output_fsa_file.write(txt)


if __name__ == "__main__":
    FSACombiner().parse(
        "examples/gchors/fsa/ex_parallel.fsa",
        "examples/gchors/fsa/ex_parallel_changed.fsa",
        "examples/gchors/fsa/ex_parallel_output.fsa",
    )
