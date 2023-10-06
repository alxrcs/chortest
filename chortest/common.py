from dataclasses import dataclass
from enum import Enum

State = str
TransitionStr = str
MessageType = str
Participant = str


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
    m: MessageType

    @staticmethod
    def new(act):
        "Convenience method. Assumes a specific format (AB[?|!]msg)."
        # TODO: avoid using this method
        # or extend it for arbitrary sized participant names
        if act[2] == "!":
            return OutTransitionLabel(
                Participant(act[0]), Participant(act[1]), MessageType(act[3:])
            )
        elif act[2] == "?":
            return InTransitionLabel(
                Participant(act[0]), Participant(act[1]), MessageType(act[3:])
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


class LocalMutationTypes(Enum):
    NONE = 0
    REMOVE_RANDOM_OUTPUT = 1
    SWAP_INTERACTION_TYPE = 2  # (flipping ? with !)
    # SWAP_RANDOM_CONSECUTIVE_TRANSITIONS = 3
    CHANGE_RANDOM_TRANSITION_MESSAGE_TYPE = 4


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
