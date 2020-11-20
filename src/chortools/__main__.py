from os import makedirs
from pathlib import Path
from subprocess import call
from typing import Optional
from datetime import datetime

from rich.console import Console
from typer import Typer

from .cfsm import CommunicatingSystem
from .gchor import Participant

app = Typer()
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
        retcode = call(
            [CHORGRAM_BASE_PATH / PROJECTION_COMMAND, gchor_filename], stdout=outfile
        )

        assert (
            retcode == 0
        ), "Could not invoke chorgram. Check that dependencies are correctly installed."

        console.print(f"Projections saved to {output_filepath}")


@app.command()
def gentests(
    cs_filename: str,
    participant_name: Optional[str] = None,
    output_path: Optional[str] = None,
):
    """
    Generates tests for a given communicating system.
    Expects the system to be in a single .fsa file.
    """
    cs = CommunicatingSystem.parse(cs_filename)
    output_foldername = str(datetime.now().isoformat(sep="_").replace(":", ""))
    tests_path = Path(cs_filename).parent / "tests" / output_foldername
    if output_path is not None:
        tests_path = Path(output_path)

    if participant_name is not None:
        list(
            cs.tests(Participant(participant_name), str(tests_path / participant_name))
        )
    else:
        for p in cs.participants():
            list(cs.tests(p, str(tests_path / p.participant_name)))


@app.command(no_args_is_help=True)
def genlts(fsa_filename: str, output_folder: Optional[str] = None, buffer_size=5):
    """
    Generates the labeled transition system
    for a given communicating system.
    """
    output_path = (
        Path(fsa_filename).parent.parent
        if output_folder is None
        else Path(output_folder)
    )
    output_path.mkdir(exist_ok=True)

    # invoke the transition system builder
    retcode = call(
        [
            str((CHORGRAM_BASE_PATH / "cfsm2gg.py").absolute()),
            "-ts",
            Path(fsa_filename).absolute(),
            "-dir",
            Path(output_path).absolute(),
            "-b",
            str(buffer_size),
        ],
        cwd=CHORGRAM_BASE_PATH,
    )
    assert retcode == 0

    # output png graphic from dot diagram
    for dot in Path(fsa_filename).parent.glob("*.dot"):
        with open(str(dot.with_suffix(".png")), "wb") as outfile:
            retcode = call(["dot", dot.absolute(), "-Tpng"], stdout=outfile)


@app.command()
def run(cs_filename: str):
    """
    Executes the given communicating system interactively.
    """
    cs = CommunicatingSystem.parse(cs_filename)
    cs.execute_interactively()


def main():
    # try:
    app(
        prog_name="chortools",
    )
    # except Exception as e:
    # console.print(f'⚠️  The command failed with message:\n"{str(e)}".')
    # console.print_exception()

    # if console.input("❓ Do you want to inspect the traceback? \[y/N] ") == "y":
    # console.print("Check the traceback below.")


if __name__ == "__main__":
    main()
