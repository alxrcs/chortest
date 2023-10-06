from collections import defaultdict
from distutils.log import debug
from pathlib import Path
from typing import Any, Dict, List, Protocol, Tuple, Union
from lark import Lark, Token, Transformer, Tree
from chortest.cfsm import CFSM, CommunicatingSystem
from chortest.common import (
    InTransitionLabel,
    MessageType,
    OutTransitionLabel,
    Participant,
    State,
    TransitionLabel,
)
from chortest.gchor import ChoiceC, ForkC, GChor, GGChor, InteractionC, IterC, SeqC
from chortest.lts import LTS

BASE_PATH = Path(__file__).parent


class CFSMBuilder(Transformer):
    """
    Constructs a communicating system from a parse tree
    of an .fsa file.
    """

    def __init__(self) -> None:
        # Global info
        self.cs: Dict[Participant, CFSM] = {}
        self.participants: List[Participant] = []

    def start(self, cfsms: List[Tuple[Participant, CFSM]]):
        for i, t in enumerate(cfsms):
            p, cfsm = t
            for q in cfsm.transitions:
                trs = cfsm.transitions[q]
                trs_new: Dict[TransitionLabel, State] = {}
                for tr in trs:
                    if isinstance(tr, InTransitionLabel):
                        trs_new[InTransitionLabel(cfsms[int(tr.B)][0], p, tr.m)] = trs[
                            tr
                        ]
                    elif isinstance(tr, OutTransitionLabel):
                        trs_new[OutTransitionLabel(p, cfsms[int(tr.B)][0], tr.m)] = trs[
                            tr
                        ]
                    else:
                        raise Exception("Shouldn't happen.")  # TODO: Fix this
                        # Spoiler alert: it happens.
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
            Participant("_"), Participant(str(t[1])), MessageType(str(t[2]))
        )
        return (str(t[0]), label, str(t[3]))

    def send_msg(self, t: List[Union[Tree, Token]]):
        # _ is a temporary placeholder. Fixed in the 'start' rule.
        label = OutTransitionLabel(
            Participant("_"), Participant(str(t[1])), MessageType(str(t[2]))
        )
        return (str(t[0]), label, str(t[3]))

    def empty(self, t):
        return (str(t[0]), TransitionLabel.empty(), str(t[1]))

    @staticmethod
    def parse(cs_filename) -> CommunicatingSystem:
        """
        Parses a communicating system from an .fsa file.
        """

        grammarfile_path = BASE_PATH / Path("grammars") / "fsa.lark"
        fsa_parser = Lark.open(
            str(grammarfile_path), transformer=CFSMBuilder(), parser="lalr"
        )

        return fsa_parser.parse(Path(cs_filename).read_text())  # type: ignore


class DOTBuilder(Transformer):
    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, str]] = defaultdict(lambda: {})
        self.edges: Dict[Tuple[str, str], Dict[str, str]] = {}

    def node_stmt(self, t):
        """
        node_id [ attr_list ]
        """
        node_name = str(t[0])
        if len(t) > 1:
            attrs = t[1] if isinstance(t[1], dict) else {}
            self.nodes[node_name].update(attrs)
        return node_name

    def edge_stmt(self, t):
        """
        (node_id | subgraph) edge_rhs [ attr_list ]
        """
        self.edges[(str(t[0]), str(t[1]))] = t[2]
        return str(t[0]), str(t[1]), t[2]

    def node_id(self, t):
        """
        ID [ port ]
        """
        name = t[0]
        return name if name[0] != '"' else name[1:-1]

    def attr_stmt(self, t):
        """
        ("graph" | node_stmt | edge_stmt) attr_list
        """
        return t

    def assignment(self, t):
        """
        ID "=" ID (part of stmt)
        """
        return {str(t[0]): str(t[1])}

    def stmt_list(self, t):
        """
        [ stmt [ ";" ] stmt_list ]
        """
        l = [t[0]] if t[0] is not None else []
        if len(t) > 1:
            l += t[1] if t[1] is not None else []
        return l

    def a_list(self, t):
        """
        ID "=" ID [ (";" | ",") ] [ a_list ]
        """
        ret = {str(t[0]): str(t[1])}
        if len(t) > 2 and t[2] is not None:
            ret.update(t[2])
        return ret

    def attr_list(self, t):
        """
        "[" a_list? "]" [ attr_list ]
        """
        attr_l = {k: v for d in t if d != None for k, v in d.items()}
        return attr_l

    def edge_rhs(self, t):
        """
        edge_op (node_id | subgraph) [ edge_rhs ]
        """
        return str(t[1])

    def graph(self, t):
        """
        [ "strict" ] ("graph" | "digraph") [ ID ] "{" stmt_list "}"
        """
        return LTS(self.nodes, self.edges)

    @staticmethod
    def parse(filename: str) -> "LTS":
        """
        Parses an LTS.

        Supported input formats:
        *.dot

        # TODO: Complete .fsa support
        """
        assert Path(filename).exists(), "File does not exist."
        if filename.endswith(".dot"):
            fsa_parser = Lark.open(
                str(Path(__file__).parent / "grammars/tsdot.lark"),
                start="graph",
                parser="lalr",
                transformer=DOTBuilder(),
            )
            dot_text = open(filename).read()
            return fsa_parser.parse(dot_text)  # type: ignore
        else:
            raise ValueError(
                "Unsupported file format for LTS: ", str(Path(filename).suffix)
            )


class GCBuilder(Transformer):
    @staticmethod
    def parse(path: str) -> "GChor":
        parser = Lark.open(
            str(BASE_PATH / "grammars" / "gchor.lark"),
            start="gg",
            debug=True,
            # transformer=GCBuilder(),
        )
        text = Path(path).read_text()
        parsed = parser.parse(text)
        return GCBuilder().transform(parsed)

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

    def sequential(self, s):
        seq = []
        for _, t in enumerate(s):
            if isinstance(t, SeqC):
                seq += t.gs
            else:
                seq.append(t)
        return SeqC(seq)

    def choice(self, c):
        return ChoiceC(c[0], c[1:])

    def fork(self, c):
        return ForkC(c)

    def iteration(self, c):
        return IterC(c[1])

    def nested_gg(self, c):
        return GGChor(c[0])


class Parsers:

    m = {
        ".fsa": CFSMBuilder,
        ".dot": DOTBuilder,
        ".gc": GCBuilder,
    }

    @staticmethod
    def parseFile(filename: Union[str, Path]) -> Any:
        return Parsers.m[Path(filename).suffix].parse(str(filename))  # type: ignore

    @staticmethod
    def parseFSA(filename: str) -> CommunicatingSystem:
        assert Path(filename).suffix == ".fsa"
        return CFSMBuilder.parse(filename)

    @staticmethod
    def parseDOT(filename: str) -> LTS:
        assert Path(filename).suffix == ".dot"
        return DOTBuilder.parse(filename)

    @staticmethod
    def parseGC(filename: str) -> GChor:
        assert Path(filename).suffix == ".gc"
        return GCBuilder.parse(filename)

    # endregion
