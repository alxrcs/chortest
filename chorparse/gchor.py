from dataclasses import dataclass


@dataclass
class Participant:
    "Represents a participant in a g-choregraphy."
    participant_name: str

    def __str__(self) -> str:
        return self.participant_name

    def __hash__(self) -> int:
        return hash(self.participant_name)

    def __eq__(self, o: object) -> bool:
        return (
            self.participant_name == o.participant_name if o is Participant else False
        )


@dataclass
class Message:
    "Represents a message in a g-choregraphy."
    msg: str

    def __str__(self) -> str:
        return self.msg

    def __hash__(self) -> int:
        return hash(self.msg)


class GChor:
    "Base class for g-choregraphies."
    pass


@dataclass
class EmptyC(GChor):
    "Represents the empty g-choregraphy."

    def __str__(self) -> str:
        return "(o)"


@dataclass
class InteractionC(GChor):
    "Represents an interaction in a g-choregraphy:"

    a: Participant
    b: Participant
    msg: Message

    def __str__(self) -> str:
        return f"{self.a}->{self.b}:{self.msg}"


@dataclass
class ForkC(GChor):

    a: GChor
    b: GChor

    def __str__(self) -> str:
        return f"{self.a} | {self.b}"


@dataclass
class ChoiceC(GChor):

    a: GChor
    b: GChor

    def __str__(self) -> str:
        return f"{{{self.a} + {self.b}}}"


@dataclass
class SeqC(GChor):

    a: GChor
    b: GChor

    def __str__(self) -> str:
        return f"{self.a};{self.b}"


@dataclass
class IterC(GChor):
    g: GChor

    def __str__(self) -> str:
        return f"repeat {{{self.g}}}"

