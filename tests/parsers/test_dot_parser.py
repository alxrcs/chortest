import pytest

from chortools.dot import DOTTransformer, LTS

@pytest.mark.wip
def test_tsdot():
    import logging
    from lark import Lark, logger

    logger.setLevel(logging.DEBUG)

    fsa_parser = Lark.open("grammars/tsdot.lark", start="graph", debug=True)
    tree = fsa_parser.parse(open("tests/files/dotlts/test_0_ts5.dot").read())
    lts: LTS = DOTTransformer().transform(tree)

    assert not lts.is_success_configuration("q2_q3____C-Bb", [["q3"], ["q3"]])
    assert lts.is_success_configuration("q2_q3____C-Bb", [["q2"], ["q3"]])
    assert lts.is_compliant([["q3"], ["q3"]])
