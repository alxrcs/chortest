import os
from pathlib import Path
from typer.testing import CliRunner
import pytest

runner = CliRunner()

from src.chortools.__main__ import app

@pytest.mark.chorgram
@pytest.mark.cli
def test_project():
    DEFAULT_OUTPUT_FILENAME = 'examples/gchors/fsa/atm_simple.fsa'
    result = runner.invoke(app, ['project', 'examples/gchors/atm_simple.gg'])
    assert result.exit_code == 0
    assert os.path.exists(DEFAULT_OUTPUT_FILENAME)
    assert os.stat(DEFAULT_OUTPUT_FILENAME).st_size > 0

@pytest.mark.cli
@pytest.mark.wip
def test_gentests():
    result = runner.invoke(app, ['gentests', 'examples/gchors/fsa/atm_simple.fsa'])
    assert result.exit_code == 0

@pytest.mark.cli
def test_run():
    EXAMPLE_CFSM_PATH = 'examples/gchors/fsa/atm_simple.fsa'
    result = runner.invoke(app, ['run', EXAMPLE_CFSM_PATH])
    assert result.exit_code == 0

# def test_default_oracle():
#     raise NotImplementedError()
