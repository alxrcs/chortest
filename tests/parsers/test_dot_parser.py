# import pytest

# from chortools.parsers import DOTTransformer

# @pytest.mark.wip
# def test_tsdot():
#     from lark import Lark

#     fsa_parser = Lark.open('grammars/tsdot.lark')
#     text = open('tests/files/dotlts/test_1_ts5.dot').read()
#     tree = fsa_parser.parse(text)
#     t = DOTTransformer()
#     lts = t.transform(tree)
#     print(lts)
