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
        
        states = l[0].split("&bull;")
        states[0] = states[0][1:] # remove first "

        # split channels by line breaks
        ms: List[str] = l[1].split(r'\n') if len(l) == 2 else []
        if len(ms) > 0: ms[-1] = ms[-1][:-1] # remove last char of last msg

        messages = defaultdict(lambda: list())
        for b in ms: # b for buffer
            src, dest = b.split("-")[0], b.split("-")[1].split("[")[0]
            for m in b[b.index('[') + 1:-1].split(','):
                messages[(src, dest)].append(m)
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

@pytest.mark.wip
def test_tsdot():
    import logging
    from lark import Lark, logger
    logger.setLevel(logging.DEBUG)

    fsa_parser = Lark.open("grammars/tsdot.lark", start='graph', debug=True)
    text = open("tests/files/dotlts/test_0_ts5.dot").read()
    tree = fsa_parser.parse(text)
    t = DOTTransformer()
    lts = t.transform(tree)
    print(lts)


if __name__ == "__main__":
    test_tsdot()