import os
from os import makedirs
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .gchor import Participant
from .cfsm import CommunicatingSystem

app = typer.Typer()

console = Console()

CHORGRAM_BASE_PATH = Path("chorgram")
PROJECTION_COMMAND = "gg2fsa"
FSA_OUTPUT_DEFAULT_FOLDER = "fsa"


@app.command()
def project(gchor_filename: str, output_folder: str = None):
    """
    Projects a g-choreography into a communicating system
    """
    gchor_path = Path(gchor_filename)

    if output_folder == None:
        output_path = gchor_path.parent / FSA_OUTPUT_DEFAULT_FOLDER
        makedirs(output_path, exist_ok=True)
        output_folder = str(output_path)

    output_filepath = Path(str(output_folder)) / Path(gchor_path.name).with_suffix(
        ".fsa"
    )

    with open(output_filepath, "wb") as outfile:
        retcode = subprocess.call(
            [CHORGRAM_BASE_PATH / PROJECTION_COMMAND, gchor_filename], stdout=outfile
        )

        assert (
            retcode == 0
        ), "Could not invoke chorgram. Check that dependencies are correctly installed."

        console.print(f"Projections saved to {output_filepath}")


@app.command()
def gentests(cs_filename: str, participant_name: Optional[str] = None):
    """
    Generates tests for a given communicating system.
    Expects the system to be in a single .fsa file.
    """
    cs = CommunicatingSystem.parse(cs_filename)
    tests_path = Path(cs_filename).parent / "tests"

    if participant_name is not None:
        cs.tests(Participant(participant_name), str(tests_path / participant_name))
    else:
        for p in cs.participants():
            cs.tests(p, str(tests_path / p.participant_name))


@app.command()
def run(cs_filename: str):
    """
    Executes the given communicating system interactively.
    """
    cs = CommunicatingSystem.parse(cs_filename)
    cs.execute_interactively()


if __name__ == "__main__":
    try:
        app(prog_name="chorparse")
    except Exception as e:
        console.print(f'⚠️  The command failed with message:\n"{str(e)}".')
        console.print_exception()

        # if console.input("❓ Do you want to inspect the traceback? \[y/N] ") == "y":
        # console.print("Check the traceback below.")
