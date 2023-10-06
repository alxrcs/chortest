from importlib.resources import read_text
from io import StringIO
from pathlib import Path

import pytest
from chortest.gchor import *
from chortest.defaults import CHORTEST_BASE_PATH, DEFAULT_CHORGRAM_BASE_PATH
from pytest import fixture

from chortest.parsing import Parsers  # type: ignore

class TestChorsAST:
    def test_interaction(self):
        e1 = InteractionC(Participant("A"), Participant("B"), MessageType("msg"))
        assert str(e1) == "A->B:msg"

    def test_seq(self):
        e1 = InteractionC(Participant("A"), Participant("B"), MessageType("msg"))
        e2 = InteractionC(Participant("B"), Participant("A"), MessageType("msg"))
        assert (str(SeqC([e1, e2]))) == "A->B:msg;\nB->A:msg"

    def test_choice(self):
        e1 = InteractionC(Participant("A"), Participant("B"), MessageType("msg"))
        e2 = InteractionC(Participant("A"), Participant("B"), MessageType("msg2"))
        s = str(ChoiceC("A", [e1, e2]))
        assert "sel A" in s and "A->B:msg" in s and "A->B:msg2" in s


@fixture
def atm_simple(shared_datadir):
    return Parsers.parseFile(shared_datadir / "gchors" / "atm_simple.gc")

@fixture
def multi_choice_select(shared_datadir):
    return Parsers.parseFile(shared_datadir / "gchors" / "multi_choice_select.gc")

def test_gcsplit(multi_choice_select):
    gc = multi_choice_select
    paths = gc.paths(Participant("A"))
    assert len(paths) == 1
    paths = gc.paths(Participant("B"))
    assert len(paths) == 3


class TestGChorParser:
    @pytest.mark.gc
    @pytest.mark.parsers
    def test_parse_and_determinize_gc_1(self, shared_datadir):
        gc_tree = Parsers.parseFile(
            shared_datadir / "gchors" / "atm_simple.gc"
        )

        paths = gc_tree.paths("A")
        assert len(paths) == 2

    def test_parse_and_determinize_gc_2(self, shared_datadir):
        gc_tree = Parsers.parseFile(
            shared_datadir / "gchors" / "parallel_choice.gc"
        )

        paths = gc_tree.paths("A")
        assert len(paths) == 4
