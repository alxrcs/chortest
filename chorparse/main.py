import os
from os import mkdir
import subprocess
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer()

console = Console()

CHORGRAM_BASE_PATH = Path("chorgram")
PROJECTION_COMMAND = "gg2fsa"
FSA_OUTPUT_DEFAULT_FOLDER = "fsa"


@app.command()
def project(gchor_filename: str, output_folder: str = None):
    """
    Projects a g-choreography into a Communicating System

    (i.e. a set of CFSMs).
    """
    gchor_path = Path(gchor_filename)
    if output_folder == None:
        output_path = gchor_path.parent / FSA_OUTPUT_DEFAULT_FOLDER
        if not output_path.exists():
            os.mkdir(output_path)
        output_folder = str(output_path)

    output_filepath = Path(str(output_folder)) / Path(gchor_path.name).with_suffix(
        ".fsa"
    )

    with open(output_filepath, "wb") as outfile:
        retcode = subprocess.call(
            [CHORGRAM_BASE_PATH / PROJECTION_COMMAND, gchor_filename], stdout=outfile
        )

    console.print(f"Projections saved to {output_filepath}")

@app.command()
def parse(cs_filename: str):
    """
    Parses a Communicating System.

    Supported formats:
    - .fsa
    """
    from .cfsm import CommunicatingSystem

    cs = CommunicatingSystem.parse(cs_filename)

    console.print(cs)


if __name__ == "__main__":
    try:
        app(prog_name="chorparse")
    except Exception as e:
        console.print(f'⚠️  The command failed with message:\n"{str(e)}".')
        console.print_exception()

        # if console.input("❓ Do you want to inspect the traceback? \[y/N] ") == "y":
        # console.print("Check the traceback below.")
