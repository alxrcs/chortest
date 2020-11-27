from collections import defaultdict
from dataclasses import dataclass
from lark.visitors import Transformer
from typing import Dict, List, Tuple

State = str
Participant = str
ParticipantPair = Tuple[Participant, Participant]
MessageType = str

@dataclass
class LTSConfig:
    states: List[State]
    messages: Dict[ParticipantPair, List[MessageType]]

    @staticmethod
    def parse(s: str):
        """
        example: "q5&bull;q2&bull;q1\n\nAC<bye>\n\nCA<bye>"
        """

        l = s.split(r"\n\n")
        ms: List[str] = l[1:]
        states = l[0].split("&bull;")
        messages = defaultdict(lambda: list())
        for m in ms:
            messages[(m[0], m[1])].append(
                m[3:-2]
            )
            # TODO: Fix fragile method 
            # (a separator is needed for participant names from chorgram's output)
        return LTSConfig(states, messages)
    
    def is_stable(self):
        for ppair in self.messages:
            if len(self.messages[ppair]) > 0:
                return False
        return True


@dataclass
class LTSTransitionLabel:
    src: Participant
    dest: Participant
    msg_type: MessageType
    is_send: bool

    @staticmethod
    def parse(s: str):
        """
        example: A&middot;B ! authW
        """
        ls = max(s.split("!"), s.split("?"), key=len)
        participants = ls[0].split("&middot;")
        message_type = ls[1]
        is_send = "!" in s
        assert is_send or "?" in s

        return LTSTransitionLabel(
            src=participants[0].strip(),
            dest=participants[1].strip(),
            msg_type=message_type.strip(),
            is_send=is_send,
        )

@dataclass
class LTSTransition:
    src: LTSConfig
    dest: LTSConfig
    label: LTSTransitionLabel

class LTS:
    configurations: Dict[str, LTSConfig]
    transitions: List[LTSTransition]

    initial: str

    def __init__(self, nodes, edges) -> None:

        nodes = {n: nodes[n] for n in nodes if n[0] == '"' and n != '"__start"'}
        edges = {e: edges[e] for e in edges if e[0] != '"__start"'}

        self.configurations = {n: LTSConfig.parse(nodes[n]["label"]) for n in nodes}
        self.transitions = []

        for e in edges:
            for l in edges[e]:
                tl = LTSTransitionLabel.parse(l["label"])
                t = LTSTransition(src=self.configurations[e[0]], dest=self.configurations[e[1]], label=tl)
                self.transitions.append(t)

        pass
        # TODO: Set initial state from "__start"

    
    def is_success_configuration(self, conf : str, final_configurations : List[List[str]]):
        """
        Checks whether a state in the lts is a success state.

        state: str representing the state (e.g. "q2_q1_q0")
        final_configurations: A list with the success states for each machine (e.g. [["q1", "q2"], "q1"])
        """
        c = self.configurations[conf]
        
        for i, s in enumerate(c.states):
            if s not in c.states[i]:
                return False
        
        return True
    
    def is_compliant(self):
        """
        Returns whether M x T is "compliant", which basically means that
        every finite run of the system contains a stable configuration 
        in which all machines are in a success state.

        Notice that the CUT is not a parameter, since the whole system
        is already run in parallel here.
        """

        pass





class DOTTransformer(Transformer):
    def __init__(self) -> None:
        self.graph_attributes: Dict[str, str] = {}
        self.nodes: Dict[str, Dict[str, str]] = {}
        self.edges: Dict[Tuple[str, str], Dict[str, str]] = {}

    def stmt(self, t):
        return t[0]

    def subgraph(self, t):
        return t

    def node_stmt(self, t):
        node_name = str(t[0])
        if node_name == "graph":
            self.graph_attributes = t[1]
        if len(t) > 1:
            self.nodes[str(t[0])] = t[1]
            return str(t[0]), t[1]

    def edge_stmt(self, t):
        self.edges[(str(t[0]), str(t[1]))] = t[2:]
        return str(t[0]), str(t[1]), t[2:]

    def attr_stmt(self, t):
        return t

    def assignment(self, t):
        return {str(t[0]): str(t[1])}

    def stmt_list(self, t):
        if len(t) == 0 or t[0] is None:
            return None
        return t[0]

    def a_list(self, t):
        return dict([(str(t[0]), str(t[1]))])

    def attr_list(self, t):
        return {k: v for d in t for k, v in d.items()}

    def edge_rhs(self, t):
        return str(t[1])

    def graph(self, t):
        return LTS(self.nodes, self.edges)


import pytest
import logging
from lark import logger

logger.setLevel(logging.DEBUG)


@pytest.mark.wip
def test_tsdot():
    from lark import Lark

    fsa_parser = Lark.open("grammars/tsdot.lark", debug=True)
    text = open("tests/files/dotlts/test_1_ts5.dot").read()
    tree = fsa_parser.parse(text)
    t = DOTTransformer()
    lts = t.transform(tree)
    print(lts)


if __name__ == "__main__":
    test_tsdot()