from dataclasses import dataclass
from lark.visitors import Transformer


@dataclass
class Participant:
    "Represents a participant in a g-choregraphy."
    participant_name: str

    def __repr__(self) -> str:
        return self.participant_name

    def __hash__(self) -> int:
        return hash(self.participant_name)

    def __eq__(self, o: object) -> bool:
        return (
            self.participant_name == o.participant_name
            if isinstance(o, Participant)
            else self.participant_name == o
        )

    def __str__(self) -> str:
        return self.participant_name

    def __lt__(self, o: "Participant") -> int:
        return self.participant_name < o.participant_name


@dataclass
class Message:
    "Represents a message in a g-choregraphy."
    msg: str

    def __repr__(self) -> str:
        return self.msg

    def __hash__(self) -> int:
        return hash(self.msg)


class GChor:
    "Base class for g-choregraphies."

    id_count: int = 0

    def __post_init__(self) -> None:
        GChor.id_count += 1
        self.id = GChor.id_count


@dataclass
class EmptyC(GChor):
    "Represents the empty g-choregraphy."

    def __repr__(self) -> str:
        return "(o)"


@dataclass
class InteractionC(GChor):
    "Represents an interaction in a g-choregraphy:"

    a: Participant
    b: Participant
    msg: Message

    def __str__(self) -> str:
        return f"{self.a}->{self.b}:{self.msg}"

    def __repr__(self) -> str:
        return f"{self.a}->{self.b}:{self.msg}" + f"({self.id})" if __debug__ else ""


@dataclass
class ForkC(GChor):

    a: GChor
    b: GChor

    def __repr__(self) -> str:
        return f"{self.a} | {self.b}"


@dataclass
class ChoiceC(GChor):

    a: GChor
    b: GChor

    def __repr__(self) -> str:
        return f"{{{self.a} + {self.b}}}"


@dataclass
class SeqC(GChor):

    a: GChor
    b: GChor

    def __repr__(self) -> str:
        return f"{self.a};{self.b}"


@dataclass
class IterC(GChor):
    g: GChor

    def __repr__(self) -> str:
        return f"repeat {{{self.g}}}"


class GTransformer(Transformer):

    # region Token rules
    def part(self, p):
        (p,) = p
        return p

    def msg(self, m):
        (m,) = m
        return m

    # endregion

    # region Parser rules
    def interaction(self, i: list):
        return InteractionC(i[0], i[1], i[2])
        # print(i)

    def sequential(self, s: list):
        return SeqC(s[0], s[1])

    def choice(self, c):
        return ChoiceC(c[0], c[1])

    # endregion
