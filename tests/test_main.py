import os
from typer.testing import CliRunner
import pytest

runner = CliRunner()

from chortools.__main__ import app


@pytest.mark.chorgram
@pytest.mark.cli
def test_project():
    DEFAULT_OUTPUT_FILENAME = 'examples/gchors/fsa/atm_simple.fsa'
    result = runner.invoke(app, ['project', 'examples/gchors/atm_simple.gg'])
    assert result.exit_code == 0
    assert os.path.exists(DEFAULT_OUTPUT_FILENAME)
    assert os.stat(DEFAULT_OUTPUT_FILENAME).st_size > 0

@pytest.mark.cli
def test_gentests_small():
    result = runner.invoke(app, ['gentests', 'examples/gchors/fsa/atm_simple.fsa'])
    DEFAULT_TESTS_OUTPUT_PATH = 'examples/gchors/fsa/atm_simple_tests'
    assert result.exit_code == 0
    assert os.path.exists(DEFAULT_TESTS_OUTPUT_PATH)
    assert len(os.listdir(DEFAULT_TESTS_OUTPUT_PATH)) > 0
    suite = os.listdir(DEFAULT_TESTS_OUTPUT_PATH)[-1]
    p = os.path.join(DEFAULT_TESTS_OUTPUT_PATH, suite)
    assert len(os.listdir(p)) > 0

@pytest.mark.cli
def test_gentests_large():
    result = runner.invoke(app, ['gentests', 'examples/gchors/fsa/atm_fixed.fsa'])
    DEFAULT_TESTS_OUTPUT_PATH = 'examples/gchors/fsa/atm_fixed_tests'
    assert result.exit_code == 0
    assert os.path.exists(DEFAULT_TESTS_OUTPUT_PATH)
    assert len(os.listdir(DEFAULT_TESTS_OUTPUT_PATH)) > 0
    suite = os.listdir(DEFAULT_TESTS_OUTPUT_PATH)[-1]
    p = os.path.join(DEFAULT_TESTS_OUTPUT_PATH, suite)
    assert len(os.listdir(p)) > 0

@pytest.mark.cli
@pytest.mark.wip
def test_main_full():
    result = runner.invoke(app, ['project', 'tests/files/gchors/ex_parallel.gg'])
    assert result.exit_code == 0
    result = runner.invoke(app, ['gentests', 'tests/files/gchors/fsa/ex_parallel.fsa'])
    assert result.exit_code == 0
    result = runner.invoke(app, ['genlts', 'tests/files/gchors/fsa/ex_parallel_tests/B/test_0/test_0.fsa'])
    assert result.exit_code == 0
    result = runner.invoke(app, ['genlts', 'tests/files/gchors/fsa/ex_parallel_tests/B/test_0/test_0.fsa', '--cut-filename', 'tests/files/gchors/fsa/ex_parallel_changed.fsa'])
    assert result.exit_code == 0
    result = runner.invoke(app, ['checklts', 'tests/files/gchors/fsa/ex_parallel_tests/B/test_0/test_0_ts5.dot'])
    assert result.exit_code == 0
    result = runner.invoke(app, ['checklts', 'tests/files/gchors/fsa/ex_parallel_tests/B/test_0__ex_parallel_changed/test_0__ex_parallel_changed_ts5.dot'])
    assert result.exit_code == 0

# TODO: Check how to test interactively
# @pytest.mark.cli
# def test_run():
#     EXAMPLE_CFSM_PATH = 'examples/gchors/fsa/atm_simple.fsa'
#     result = runner.invoke(app, ['run', EXAMPLE_CFSM_PATH])
#     assert result.exit_code == 0
