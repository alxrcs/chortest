from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union
from itertools import groupby

from chortest.common import MessageType, Participant, State

ParticipantPair = Tuple[Participant, Participant]


@dataclass
class LTSConfig:
    states: List[State]
    messages: Dict[ParticipantPair, List[MessageType]]
    str_rep: str

    @staticmethod
    def parse(s: str):
        """
        example: "q5&bull;q2&bull;q1\n\nAC<bye>\n\nCA<bye>"
        """

        l = s.split(r"\n\n")

        states = l[0].split("&bull;")
        if states[0][0] == '"':
            states[0] = states[0][1:]  # remove first "
        if states[-1][-1] == '"':
            states[-1] = states[-1][:-1]  # remove last "

        # split channels by line breaks
        ms: List[str] = l[1].split(r"\n") if len(l) == 2 else []

        messages = defaultdict(lambda: list())
        for b in ms:  # b for buffer
            src, dest = b.split("-")[0], b.split("-")[1].split("[")[0]
            for m in b[b.index("[") + 1 : -1].split(","):
                if m[-1] in ']"':
                    m = m[:-1]
                messages[(src, dest)].append(m)

        return LTSConfig(states, messages, s)

    def __str__(self) -> str:
        return self.str_rep

    def is_stable(self):
        return all(len(self.messages[ppair]) == 0 for ppair in self.messages)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, LTSConfig) and self.str_rep == o.str_rep

    def __hash__(self) -> int:
        return self.str_rep.__hash__()


@dataclass(eq=True, frozen=True)
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
        if s[0] == '"' == s[-1]:
            s = s[1:-1].strip()
        ls = max(s.split("!"), s.split("?"), key=len)
        participants = [str.strip(p) for p in ls[0].split("&middot;")]
        message_type = ls[1].strip()
        is_send = "!" in s
        assert is_send or "?" in s

        return LTSTransitionLabel(
            src=participants[0].strip(),
            dest=participants[1].strip(),
            msg_type=message_type.strip(),
            is_send=is_send,
        )

    def __repr__(self) -> str:
        op = "!" if self.is_send else "?"
        return f"{self.src}_{self.dest}{op}{self.msg_type}"

    def __str__(self) -> str:
        return self.__repr__()

    def subject(self) -> Participant:
        return self.src if self.is_send else self.dest


LangType = Set[Tuple[LTSTransitionLabel, ...]]


@dataclass(eq=True, frozen=True)
class LTSTransition:
    src: LTSConfig
    dest: LTSConfig
    label: LTSTransitionLabel


class LTS:
    configurations: Dict[str, LTSConfig]
    transitions: List[LTSTransition]

    initial: LTSConfig

    cache: Dict[LTSConfig, bool] = {}
    failing_states: List[LTSConfig]
    START_STATE = "__start"

    def __init__(self, nodes, edges) -> None:

        initial: str = ""
        for e in edges:
            if e[0] == LTS.START_STATE:
                initial = e[1]

        assert initial != "", "Could not find initial state."

        nodes = {n: nodes[n] for n in nodes if n != LTS.START_STATE}
        edges = {e: edges[e] for e in edges if e[0] != LTS.START_STATE}

        self.configurations = {n: LTSConfig.parse(nodes[n]["label"]) for n in nodes}
        self.transitions: List[LTSTransition] = []
        self.attributes = nodes

        for edge, attribs in edges.items():
            tl = LTSTransitionLabel.parse(attribs["label"])
            t = LTSTransition(
                src=self.configurations[edge[0]],
                dest=self.configurations[edge[1]],
                label=tl,
            )
            self.transitions.append(t)

        self.initial = self.configurations[initial]
        self.failing_states = []

    def has_a_deadlock(self):
        """
        Deadlocks are signaled by chorgram as states with attr style=filled
        """
        return any(x.get("style") == "filled" for x in self.attributes.values())

    def is_success_configuration(
        self,
        conf: Union[str, LTSConfig],
        final_configurations: List[List[str]],
        order: List[Participant],
        cut_name: str,
    ):
        """
        Checks whether a state in the lts is a success state.

        conf: str or LTSConfig representing the configuration (e.g. "q2_q1_q0")
        final_configurations: A list with the success states for each machine (e.g. [["q1", "q2"], "q1"])
        cut_name: The name of the component under test. Its state will be ignored.
        """
        if isinstance(conf, str):
            states = self.configurations[conf].states
            messages = self.configurations[conf].messages
        else:
            states = conf.states
            messages = conf.messages

        cut_idx = order.index(cut_name)
        return (
            all(
                s in final_configurations[i]
                for i, s in enumerate(states)
                if i != cut_idx
            )
            and len(messages) == 0
        )

    def is_compliant(
        self,
        final_configurations: List[List[State]],
        order: List[Participant],
        cut_name: Participant,
        curr: Optional[LTSConfig] = None,
    ):
        """
        Returns whether M x T is "compliant", which basically means that
        every finite run of the system contains a stable configuration
        in which all machines (except the CUT) are in a success state.
        """

        if final_configurations == []:
            # If no oracle was given, just check if there are no deadlocks.
            return not self.has_a_deadlock()

        if curr is None:
            curr = self.initial
            LTS.cache = {}

        if curr in LTS.cache:
            return LTS.cache[curr]

        curr_is_success = self.is_success_configuration(
            curr, final_configurations, order, cut_name
        )

        if curr_is_success:
            ret = True
        else:
            enabled_transitions = [t for t in self.transitions if t.src == curr]
            next_configs = [t.dest for t in enabled_transitions]
            is_an_intermediate_state = len(next_configs) > 0
            all_subsequent_states_are_compliant = all(
                self.is_compliant(final_configurations, order, cut_name, c)
                for c in next_configs
            )

            if is_an_intermediate_state:
                ret = all_subsequent_states_are_compliant
            else:
                if not curr_is_success:
                    self.failing_states.append(curr)
                ret = curr_is_success

        LTS.cache[curr] = ret
        return ret

    def get_failing_states(self):
        """
        Returns a string representation of all states disturbing compliance of the
        current LTS. NOTE: This must be invoked after checking for compliance at least once
        through `is_compliant`.
        """
        assert (
            self.failing_states
        ), "Current LTS is either compliant or compliance has not yet been checked."

        return [", ".join(c.states) for c in self.failing_states]

    def language(
        self, restricted: Optional[Participant] = None
    ) -> Tuple[LangType, LangType]:
        """
        Do a DFS on the LTS to obtain
        all possible words in the language generated by this system.
        """

        lang: LangType = set()

        # available transitions
        actions = {
            k: list(v) for k, v in groupby(self.transitions, key=lambda x: x.src)
        }
        targets = {(t.src, t.label): t.dest for t in self.transitions}

        # DFS
        def dfs(conf, path):
            if conf in actions:
                for label in actions[conf]:
                    new_conf = targets[(conf, label.label)]
                    dfs(new_conf, path + [label.label])
            else:
                assert len(path) >= 0
                lang.add(tuple(path))

        dfs(self.initial, [])

        if restricted:
            restricted_lang: LangType = set()
            for word in lang:
                restricted_word: List[LTSTransitionLabel] = [
                    label
                    for label in word
                    # if label.subject() != restricted
                    if label.is_send or label.subject() != restricted
                ]
                restricted_lang.add(tuple(restricted_word))
            return lang, restricted_lang

        return lang, set()
