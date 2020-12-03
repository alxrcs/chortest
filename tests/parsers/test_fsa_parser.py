import pytest

from lark import Lark
from chortools.fsa import FSACombiner

def test_fsa():
    # TODO: finish parser
    fsa_parser = Lark.open('grammars/fsa.lark')
    text = open('examples/fsa/cfsmA.fsa').read()
    tree = fsa_parser.parse(text)
    assert tree is not None

def test_fsa_combiner():
    fsa = FSACombiner().combine_fsa(
        "examples/gchors/fsa/ex_parallel.fsa",
        "examples/gchors/fsa/ex_parallel_changed.fsa",
        "examples/gchors/fsa/ex_parallel_output.fsa",
    )