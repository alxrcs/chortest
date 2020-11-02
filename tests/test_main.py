from os import remove
from pathlib import Path
from typer.testing import CliRunner
import pytest

from src.chortools.main import app

runner = CliRunner()

@pytest.mark.chorgram
@pytest.mark.cli
def test_project():
    result = runner.invoke(app, ['project', 'examples/gchors/atm_simple.gg'])
    assert result.exit_code == 0

@pytest.mark.cli
@pytest.mark.wip
def test_gentests():
    result = runner.invoke(app, ['gentests', 'examples/gchors/fsa/atm_simple.fsa'])
    assert result.exit_code == 0

@pytest.mark.cli
def test_run():
    result = runner.invoke(app, ['run', 'examples/'])
    assert result.exit_code == 0

# def test_default_oracle():
#     raise NotImplementedError()
