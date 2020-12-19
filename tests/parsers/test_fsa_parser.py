import pytest

from lark import Lark
from chortest.cfsm import CFSMBuilder
from chortest.helpers import combine_fsa

# def test_fsa_parser():
#     fsa_parser = Lark.open('grammars/fsa.lark')
#     text = open('examples/fsa/cfsmA.fsa').read()
#     tree = fsa_parser.parse(text)
#     cfsm = CFSMBuilder().transform(tree)
#     assert cfsm is not None

def test_fsa_combiner():
    combine_fsa(
        "examples/gchors/fsa/ex_parallel.fsa",
        "examples/gchors/fsa/ex_parallel_changed.fsa",
        "examples/gchors/fsa/ex_parallel_output.fsa",
    )