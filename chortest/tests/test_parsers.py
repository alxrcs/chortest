import pytest
from chortest.lts import LTS
from pathlib import Path

from chortest.helpers import combine_fsa
from chortest.parsing import Parsers

BASE_PATH = Path(__file__).parent.parent


@pytest.mark.wip
def test_tsdot(shared_datadir):
    lts: LTS = Parsers.parseFile(shared_datadir / "dotlts" / "test_0_ts5.dot")
    assert lts is not None


def test_lts_compliance(shared_datadir):
    lts: LTS = Parsers.parseFile(shared_datadir / "dotlts/test_0_ts5.dot")

    assert not lts.is_success_configuration(
        "q2_q3____C-Bb", [["q3"], ["q3"]], order=["B", "C"], cut_name="C"
    )
    assert lts.is_success_configuration(
        "q2_q2", [["q2"], ["q3"]], order=["B", "C"], cut_name="C"
    )
    # TODO: Add more assertions for compliance checking.
    assert lts.is_compliant([["q3"], ["q3"]], order=["B", "C"], cut_name="B")


def test_fsa_combiner(shared_datadir):
    BASE = Path(__file__).parent.parent.parent
    combine_fsa(
        shared_datadir / "gchors" / "fsa" / "ex_parallel.fsa",
        shared_datadir / "gchors" / "fsa" / "ex_parallel_changed.fsa",
        shared_datadir / "gchors" / "fsa" / "ex_parallel_output.fsa",
    )
    assert Path(shared_datadir / "gchors" / "fsa" / "ex_parallel_output.fsa").exists()


# from chortest.parsers import FSATransformer

# @pytest.mark.wip
# def test_fsm():
#     from lark import Lark

#     fsa_parser = Lark.open('grammars/fsm.lark')
#     text = open('examples/fsm/ex1.fsm').read()
#     tree = fsa_parser.parse(text)
#     t = FSATransformer()
#     lts = t.transform(tree)
#     print(lts)


# def test_fsa_parser():
#     fsa_parser = Lark.open('grammars/fsa.lark')
#     text = open('examples/fsa/cfsmA.fsa').read()
#     tree = fsa_parser.parse(text)
#     cfsm = CFSMBuilder().transform(tree)
#     assert cfsm is not None
