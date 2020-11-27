import pytest

# @pytest.mark.wip
def test_fsa():
    from lark import Lark

    fsa_parser = Lark.open('grammars/fsa.lark')
    text = open('examples/fsa/cfsmA.fsa').read()
    tree = fsa_parser.parse(text)
    print(tree.pretty())