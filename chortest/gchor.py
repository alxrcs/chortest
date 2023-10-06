import itertools
from abc import abstractmethod
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Generator, List, Optional, Set

from lark import Token

from chortest.common import MessageType, Participant

BASE_PATH = Path(__file__).parent


@dataclass
class GlobalMutation:
    mutant: "GChor"
    mutation_type: str
    target_id: int


class GChor:
    "Base class for g-choregraphies."

    id_count: int = 0

    def __post_init__(self) -> None:
        GChor.id_count += 1
        self.id = GChor.id_count

    @abstractmethod
    def paths(self, part: Participant) -> List["GChor"]:
        ...

    @abstractmethod
    def participants(self) -> Set[Participant]:
        ...

    @abstractmethod
    def message_types(self) -> Set[MessageType]:
        ...

    @abstractmethod
    def copy(self) -> "GChor":
        ...

    def mutate(self, p: Participant) -> Generator[GlobalMutation, None, None]:
        # If this isn't redefined, will yield an empty generator
        yield from ()


@dataclass
class EmptyC(GChor):
    "Represents the empty g-choregraphy."

    def __repr__(self) -> str:
        return "(o)"

    def paths(self, part: Participant) -> List[GChor]:
        return []

    def participants(self) -> Set[Participant]:
        return set()

    def message_types(self) -> Set[MessageType]:
        return set()

    def copy(self) -> "EmptyC":
        return EmptyC()


@dataclass
class InteractionC(GChor):
    "Represents an interaction in a g-choregraphy:"

    src: Participant
    dst: Participant
    msg: MessageType
    comment: Optional[str] = None

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"{self.src}->{self.dst}:{self.msg}"

    def paths(self, part: Participant):
        return [self]

    def participants(self) -> Set[Participant]:
        return set([self.src, self.dst])

    def message_types(self) -> Set[MessageType]:
        return set(self.msg)

    def copy(self) -> "InteractionC":
        return InteractionC(self.src, self.dst, self.msg)

    def mutate(self, p: Participant) -> Generator[GlobalMutation, None, None]:
        # TODO: Determine whether should we include this condition
        if p == self.src or p == self.dst:
            line, col = (
                self.src.line if isinstance(self.src, Token) else 0,
                self.src.column if isinstance(self.src, Token) else 0,
            )
            mut = self.copy()
            mut.msg += "_broken"
            yield GlobalMutation(mut, f"Change message type [{line,col}]", self.id)
            mut = self.copy()
            mut.src, mut.dst = mut.dst, mut.src
            yield GlobalMutation(mut, f"Swap src and dest [{line,col}]", self.id)


@dataclass
class ForkC(GChor):

    gs: List[GChor]
    comment: Optional[str] = None

    def __str__(self) -> str:
        body = "|".join([f"\n{str(g)}" for g in self.gs])
        return f"{body}\n"

    def paths(self, part: Participant):
        # TODO: Extend to arbitrary number of forkings
        paths_a = self.gs[0].paths(part)
        paths_b = self.gs[1].paths(part)

        return [ForkC([a, b]) for a in paths_a for b in paths_b]

    def participants(self) -> Set[Participant]:
        return reduce(
            lambda a, b: a.union(b), (g.participants() for g in self.gs), set()
        )

    def message_types(self) -> Set[MessageType]:
        return reduce(
            lambda a, b: a.union(b), (g.message_types() for g in self.gs), set()
        )

    def copy(self) -> "ForkC":
        return ForkC([gc.copy() for gc in self.gs])

    def mutate(self, p: Participant) -> Generator[GlobalMutation, None, None]:

        p_is_in_multiple_children = sum(p in x.participants() for x in self.gs) > 1
        if p_is_in_multiple_children:
            mutant = ChoiceC(p, [GGChor(x) for x in self.copy().gs])
            yield GlobalMutation(mutant, "Change Fork for Choice", self.id)

        # FIXME: Wrapping cause of the choice nesting parser bug

        # TODO: Fix the active participant
        # Recursively check the first interaction?

        A = [g.mutate(p) for g in self.gs]
        B = self.gs
        rng = range(len(A))

        # Take one branch of mutants at a time and compose it with the rest of the tree.
        # A = [['aa','ab'], ['b'], ['c']]
        # B = [1, 2, 3]
        # out = [['aa', 2, 3], ['ab', 2, 3], [1, 'b', 3], [1, 2, 'c']]
        rec_children = ((x if i == j else B[i] for i in rng) for j in rng for x in A[j])

        # Recursively mutate
        for children in rec_children:
            children_l = []
            mut_type, mut_id = "", -1
            for child in children:
                if isinstance(child, GlobalMutation):
                    mut_type = child.mutation_type
                    mut_id = child.target_id
                    children_l.append(child.mutant)
                elif isinstance(child, GChor):
                    children_l.append(child)
                else:
                    raise Exception("shouldn't happen")
            yield GlobalMutation(ForkC(children_l), mut_type, mut_id)


@dataclass
class ChoiceC(GChor):

    active: Participant
    gs: List[GChor]
    comment: Optional[str] = None

    def __str__(self) -> str:
        body = "\n+\n".join([f"{str(g)}" for g in self.gs])
        return f"sel {str(self.active)} {{\n{body}\n}}"

    def paths(self, part: Participant) -> List[GChor]:
        rec: List[GChor] = sum((g.paths(part) for g in self.gs), [])

        if part == self.active:
            return [ChoiceC(self.active, rec)]
        else:
            return rec

    def participants(self) -> Set[Participant]:
        return reduce(
            lambda a, b: a.union(b), (g.participants() for g in self.gs), set()
        )

    def message_types(self) -> Set[MessageType]:
        return reduce(
            lambda a, b: a.union(b), (g.message_types() for g in self.gs), set()
        )

    def copy(self) -> "ChoiceC":
        return ChoiceC(self.active, [gc.copy() for gc in self.gs])

    def mutate(self, p: Participant) -> Generator[GlobalMutation, None, None]:

        # Change Choice to Fork
        p_is_in_multiple_children = sum(p in x.participants() for x in self.gs) > 1
        if p_is_in_multiple_children:
            yield GlobalMutation(
                GGChor(ForkC(self.copy().gs)), "Change Choice to Fork", self.id
            )

        A = [g.mutate(p) for g in self.gs]
        B = self.gs
        rng = range(len(A))

        # Take one branch of mutants at a time and compose it with the rest of the tree.
        # A = [['aa','ab'], ['b'], ['c']]
        # B = [1, 2, 3]
        # out = [['aa', 2, 3], ['ab', 2, 3], [1, 'b', 3], [1, 2, 'c']]
        rec_children = ((x if i == j else B[i] for i in rng) for j in rng for x in A[j])

        # Recursively mutate
        for children in rec_children:
            children_l = []
            mut_type, mut_id = "", -1
            for child in children:
                if isinstance(child, GlobalMutation):
                    mut_type = child.mutation_type
                    mut_id = child.target_id
                    children_l.append(child.mutant)
                elif isinstance(child, GChor):
                    children_l.append(child)
                else:
                    raise Exception("shouldn't happen")
            yield GlobalMutation(ChoiceC(self.active, children_l), mut_type, mut_id)


@dataclass
class SeqC(GChor):

    gs: List[GChor]
    comment: Optional[str] = None

    def __str__(self) -> str:
        return ";\n".join([str(gc) for gc in self.gs])

    def __repr__(self) -> str:
        return super().__repr__() + f"{self.gs}"

    def paths(self, part: Participant) -> List[GChor]:
        sub_paths = [sub_gc.paths(part) for sub_gc in self.gs]
        return list(itertools.product(*sub_paths))  # type: ignore

    def participants(self) -> Set[Participant]:
        return reduce(
            lambda a, b: a.union(b), (g.participants() for g in self.gs), set()
        )

    def message_types(self) -> Set[MessageType]:
        return reduce(
            lambda a, b: a.union(b), (g.message_types() for g in self.gs), set()
        )

    def copy(self) -> "SeqC":
        return SeqC([gc.copy() for gc in self.gs])

    def mutate(self, p: Participant) -> Generator[GlobalMutation, None, None]:
        for i, (sub_gc, next_gc) in enumerate(zip(self.gs, self.gs[1:])):
            # Check if two consecutive children are interactions
            if not (
                isinstance(sub_gc, InteractionC) and isinstance(next_gc, InteractionC)
            ):
                continue
            # Check if the CUT participates in both interactions at some point
            if not p in sub_gc.participants() or not p in next_gc.participants():
                continue

            # If the above conditions are satisfied, swap these two interactions.
            mut = self.copy()
            mut.gs[i], mut.gs[i + 1] = (
                mut.gs[i + 1],
                mut.gs[i],
            )
            yield GlobalMutation(mut, f"Swap Interaction Order {i, i+1}", self.id)

        # When two children or more in a sequence, randomly remove one
        if len(self.gs) > 2:
            for i, _ in enumerate(self.gs):
                # Only remove them if they're interactions
                if not isinstance(self.gs[i], InteractionC):
                    continue
                # Only remove them if the CUT is involved
                if p not in self.gs[i].participants():
                    continue
                mut = self.copy()
                del mut.gs[i]
                yield GlobalMutation(mut, f"Remove child in sequence ({i})", self.id)

        # Recursively mutate
        A = [g.mutate(p) for g in self.gs]
        B = self.gs
        rng = range(len(A))
        rec_children = ((x if i == j else B[i] for i in rng) for j in rng for x in A[j])
        for children in rec_children:
            children_l = []
            mut_type, mut_id = "", -1
            for child in children:
                if isinstance(child, GlobalMutation):
                    mut_type = child.mutation_type
                    mut_id = child.target_id
                    children_l.append(child.mutant)
                elif isinstance(child, GChor):
                    children_l.append(child)
                else:
                    raise Exception("shouldn't happen")
            yield GlobalMutation(SeqC(children_l), mut_type, mut_id)


@dataclass
class IterC(GChor):
    g: GChor
    comment: Optional[str] = None

    def __repr__(self) -> str:
        return f"repeat {{ \n{self.g}\n }}"

    def paths(self, part: Participant) -> List[GChor]:
        raise NotImplementedError()

    def participants(self) -> Set[Participant]:
        return self.g.participants()

    def message_types(self) -> Set[MessageType]:
        return self.g.message_types()

    def copy(self) -> "IterC":
        return IterC(self.g.copy())

    def mutate(self, p: Participant) -> Generator[GlobalMutation, None, None]:

        # Recursively mutate
        for mut in self.g.mutate(p):
            yield GlobalMutation(IterC(mut.mutant), mut.mutation_type, mut.target_id)


@dataclass
class GGChor(GChor):
    g: GChor
    comment: Optional[str] = None

    def __str__(self) -> str:
        return "{\n" + str(self.g) + "\n}"

    def paths(self, part: Participant) -> List[GChor]:
        return [GGChor(g) for g in self.g.paths(part)]

    def participants(self) -> Set[Participant]:
        return self.g.participants()

    def message_types(self) -> Set[MessageType]:
        return self.g.message_types()

    def copy(self) -> "GGChor":
        return GGChor(self.g.copy())

    def mutate(self, p: Participant) -> Generator[GlobalMutation, None, None]:
        # Recursively mutate
        for g_mut in self.g.mutate(p):
            yield GlobalMutation(
                GGChor(g_mut.mutant), g_mut.mutation_type, g_mut.target_id
            )
