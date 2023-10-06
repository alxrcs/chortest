import os
from typer.testing import CliRunner
from pathlib import Path
import pytest

runner = CliRunner()

from chortest.__main__ import app

BASE_PATH = Path(__file__).parent


@pytest.mark.chorgram
@pytest.mark.cli
def test_project(shared_datadir):
    result = runner.invoke(app, ["project", str(shared_datadir / "gchors" /"atm_simple.gc")])
    DEFAULT_OUTPUT_FILENAME =  shared_datadir / "gchors" /"fsa"/"atm_simple.fsa"
    assert result.exit_code == 0
    assert os.path.exists(DEFAULT_OUTPUT_FILENAME)
    assert os.stat(DEFAULT_OUTPUT_FILENAME).st_size > 0


@pytest.mark.cli
def test_gentests_small(shared_datadir):
    result = runner.invoke(app, ["project", str(shared_datadir / "gchors" / "atm_simple.gc")])
    result = runner.invoke(app, ["gentests", str(shared_datadir / "gchors" / "fsa/atm_simple.fsa")])
    DEFAULT_TESTS_OUTPUT_PATH = shared_datadir / "gchors"/ "fsa"/"atm_simple_tests"
    assert result.exit_code == 0
    assert os.path.exists(DEFAULT_TESTS_OUTPUT_PATH)
    assert len(os.listdir(DEFAULT_TESTS_OUTPUT_PATH)) > 0
    suite = os.listdir(DEFAULT_TESTS_OUTPUT_PATH)[-1]
    p = os.path.join(DEFAULT_TESTS_OUTPUT_PATH, suite)
    assert len(os.listdir(p)) > 0


@pytest.mark.cli
def test_gentests_large(shared_datadir):
    result = runner.invoke(
        app, ["gentests", str(shared_datadir / "gchors/fsa/atm_fixed.fsa")]
    )
    DEFAULT_TESTS_OUTPUT_PATH = str(shared_datadir / "gchors/fsa/atm_fixed_tests")
    assert result.exit_code == 0
    assert os.path.exists(DEFAULT_TESTS_OUTPUT_PATH)
    assert len(os.listdir(DEFAULT_TESTS_OUTPUT_PATH)) > 0
    suite = os.listdir(DEFAULT_TESTS_OUTPUT_PATH)[-1]
    p = os.path.join(DEFAULT_TESTS_OUTPUT_PATH, suite)
    assert len(os.listdir(p)) > 0


@pytest.mark.cli
@pytest.mark.wip
# @pytest.mark.skip
def test_main_full(shared_datadir):
    result = runner.invoke(app, ["project", str(shared_datadir / "gchors/ex_parallel.gc")])

    assert result.exit_code == 0
    result = runner.invoke(
        app, ["gentests", str(shared_datadir / "gchors/fsa/ex_parallel.fsa")]
    )
    assert result.exit_code == 0
    result = runner.invoke(
        app,
        [
            "genlts",
            str(shared_datadir / "gchors/fsa/ex_parallel_tests/B/test_0/test_0.fsa"),
        ],
    )
    assert result.exit_code == 0
    result = runner.invoke(
        app,
        [
            "genlts",
            str(shared_datadir / "gchors/fsa/ex_parallel_tests/B/test_0/test_0.fsa"),
            "--cut-filename",
            str(shared_datadir / "gchors/fsa/ex_parallel_changed.fsa"),
        ],
    )
    assert result.exit_code == 0
    result = runner.invoke(
        app,
        [
            "checklts",
            str(shared_datadir / "gchors/fsa/ex_parallel_tests/B/test_0/test_0_ts5.dot"),
            # str(shared_datadir / "gchors/fsa/ex_parallel_tests/B/test_0/test_0.fsa.oracle.yaml"),
            "B",
        ],
    )
    assert result.exit_code == 0
