from lark import Lark
from lark.visitors import Transformer
from chorparse.gchor import *
from pytest import fixture  # type: ignore
import pytest


class TestChorsAST:
    def test_interaction(self):
        e1 = InteractionC(Participant("A"), Participant("B"), Message("msg"))
        assert str(e1) == "A->B:msg"

    def test_seq(self):
        e1 = InteractionC(Participant("A"), Participant("B"), Message("msg"))
        e2 = InteractionC(Participant("B"), Participant("A"), Message("msg"))
        assert (str(SeqC(e1, e2))) == "A->B:msg;B->A:msg"

    def test_choice(self):
        e1 = InteractionC(Participant("A"), Participant("B"), Message("msg"))
        e2 = InteractionC(Participant("B"), Participant("A"), Message("msg"))
        assert (str(ChoiceC(e1, e2))) == "{A->B:msg + B->A:msg}"


@fixture
def gg_parser():
    return Lark.open("grammars/gchor.lark", start="gg")


@fixture
def atm_simple():
    return open("examples/gchors/atm_simple.gg").read()


class TestGChorParser:
    def test_comment(self, gg_parser):
        text = "A -> C: money;  // comment \n B -> A: money"
        gg_parser.parse(text)

    @pytest.mark.gg
    @pytest.mark.parsers
    def test_parse(self, gg_parser, atm_simple):
        tree = gg_parser.parse(atm_simple)
        print(tree.pretty())
        print(GTransformer().transform(tree))

