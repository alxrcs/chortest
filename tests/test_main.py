from os import remove
from pathlib import Path
from typer.testing import CliRunner
import pytest

from chorparse.main import app

runner = CliRunner()

def test_project():
    result = runner.invoke(app, ['project', 'examples/gchors/atm_simple.gg'])
    assert result.exit_code == 0
    assert Path('examples/gchors/fsa/atm_simple.fsa').is_file()
    # remove(Path('examples/gchors/fsa/atm_simple.fsa').resolve())

@pytest.mark.wip
def test_project_and_parse():
    result = runner.invoke(app, ['project', 'examples/gchors/atm_simple.gg'])
    assert result.exit_code == 0
    result = runner.invoke(app, ['parse', 'examples/gchors/fsa/atm_simple.fsa'])
    assert result.exit_code == 0

# def test_default_oracle():
#     raise NotImplementedError()
