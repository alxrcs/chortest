from pathlib import Path
from chortest.defaults import CHORTEST_BASE_PATH, DEFAULT_CHORGRAM_BASE_PATH
import pytest
from os.path import exists
from os import remove

from chortest.gchor import Participant
from chortest.parsing import Parsers


@pytest.fixture
def small_atm(shared_datadir) -> str:
    return str(shared_datadir / "gchors" / "fsa" / "atm_simple.fsa")


def test_small_atm_testgen(small_atm) -> None:
    atm = Parsers.parseFile(small_atm)
    tests = list(atm.tests(Participant("A")))
    test_name = "tmptest"
    for test in tests:
        test.to_fsa(test_name)
        assert exists(test_name)
        remove(test_name)
        remove(f"{test_name}.oracle.yaml")

    assert len(tests) > 0
