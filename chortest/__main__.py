import os
import subprocess
import sys
from datetime import datetime
from logging import FileHandler, Formatter, basicConfig, getLogger
from os import makedirs
from pathlib import Path
from shutil import copy
from time import perf_counter, time
from typing import List, Optional

import typer
import yaml
from rich.logging import RichHandler
from typer import Typer

from chortest.helpers import combine_fsa
from chortest.lts import LTS

from .cfsm import CommunicatingSystem, State
from .gchor import Participant

app = Typer()
L = getLogger()

CHORGRAM_BASE_PATH = Path("chorgram")
PROJECTION_COMMAND = "gc2fsa"
LTSGEN_COMMAND = "cfsm2gc.py"
FSA_OUTPUT_DEFAULT_FOLDER = "fsa"
CHORGRAM_INVOKE_ERROR_MSG = (
    "Could not invoke chorgram. Check that dependencies are correctly installed."
)
DOT_INVOKE_ERROR_MSG = (
    "Could not invoke dot. Check that graphviz is properly installed."
)
ORACLE_DEFAULT_FILENAME = "oracle.yaml"

def call(cmd, *args, **kwargs):
    before = time()
    cmd_str = " ".join(map(lambda x: str(x), cmd))
    L.info(f"Calling {str(cmd_str)}")
    ret = subprocess.call(cmd, *args, **kwargs)
    L.info(f"Done in {(time()-before):.3f}s!")
    return 0 

@app.command(no_args_is_help=True)
def project(gchor_filename: str, output_folder: str = None):
    """
    Projects a g-choreography into a communicating system.
    """
    gchor_path = Path(gchor_filename)

    if output_folder is None:
        output_path = gchor_path.parent / FSA_OUTPUT_DEFAULT_FOLDER
        makedirs(output_path, exist_ok=True)
        output_folder = str(output_path)

    output_filepath = Path(str(output_folder)) / Path(gchor_path.name).with_suffix(
        ".fsa"
    )

    with open(output_filepath, "wb") as outfile:
        retcode = call(
            [CHORGRAM_BASE_PATH / PROJECTION_COMMAND, gchor_filename],
            stdout=outfile,
            stderr=outfile,
        )
        
        assert retcode == 0, CHORGRAM_INVOKE_ERROR_MSG + f' Check {output_filepath} for more details.'

        L.info(f"Projections saved to {output_filepath}")


@app.command(no_args_is_help=True)
def gentests(
    cs_filename: str,
    participant: Optional[str] = None,
    output: Optional[str] = None,
    include_timestamp: Optional[bool] = False,
):
    """
    Generates tests for a given communicating system.
    Expects the system to be in a single .fsa file.
    """

    L.info(f'Parsing {cs_filename}')
    cs = CommunicatingSystem.parse(cs_filename)
    L.info(f'Finished parsing {cs_filename}')

    output_foldername = str(datetime.now().isoformat(sep="_").replace(":", ""))
    fsa_f = Path(cs_filename)
    tests_path = fsa_f.parent / f"{fsa_f.stem}_tests"

    if include_timestamp:
        tests_path = tests_path / output_foldername

    if output is not None:
        tests_path = Path(output)

    if participant is None:
        L.info(f'No participant specified. Will generate tests for all participants.')
        participants = list(cs.participants)
    else:
        participants = [Participant(participant)]

    start_time = perf_counter()

    for p in participants:
        L.info(f'Generating tests for participant {str(p)}')
        tests = list(cs.tests(p))
        for i, test in enumerate(tests):
            test.to_fsa(
                str(tests_path / p.participant_name / f"test_{i}" / f"test_{i}.fsa")
            )
        L.info(f'{len(tests)} tests saved to "{str(tests_path / p.participant_name)}"')

    elapsed_time = perf_counter() - start_time
    L.info(f"Tests generated in {elapsed_time}s")
    return tests_path


@app.command(no_args_is_help=True)
def genlts(
    fsa_filename: str,
    output_folder: Optional[str] = None,
    buffer_size: int = 5,
    fifo_semantics: bool = False,
    cut_filename: str = None,
    gen_pngs: bool = False,
):
    """
    Generates the labeled transition system
    for a given communicating system.
    """
    if not fsa_filename.endswith('.fsa') or not os.path.exists(fsa_filename):
        typer.echo('Please provide a valid .fsa file.')
        raise typer.Exit(1)

    output_path = (
        Path(
            fsa_filename
        ).parent.parent  # This is needed since chorgram generates the lts in a folder with the same name as the fsa file.
        if output_folder is None
        else Path(output_folder)
    )
    os.makedirs(output_path, exist_ok=True)

    if cut_filename is not None:
        combined_foldername = f"{Path(fsa_filename).stem}__{Path(cut_filename).stem}"
        combined_filename = str(
            output_path / combined_foldername / (combined_foldername + ".fsa")
        )
        combine_fsa(fsa_filename, cut_filename, combined_filename)
        copy(  # oracle
            Path(fsa_filename).parent / "oracle.yaml", output_path / combined_foldername
        )
    else:
        combined_filename = fsa_filename

    # invoke the transition system builder
    start_time = perf_counter()

    with open("chorgram_output.log", "w") as l:
        retcode = call(
            [
                str((CHORGRAM_BASE_PATH / LTSGEN_COMMAND).absolute()),
                "-ts",
                Path(combined_filename).absolute(),
                "-dir",
                Path(output_path).absolute(),
                "-b",
                str(buffer_size),
                "-nf" if not fifo_semantics else "",
                "-sn",  # Do not shorten state names
            ],
            stderr=l,
            stdout=l,
            cwd=CHORGRAM_BASE_PATH,
        )

    elapsed_time = perf_counter() - start_time
    assert retcode == 0, CHORGRAM_INVOKE_ERROR_MSG
    L.info(f'LTS saved to "{output_path}/{Path(combined_filename).stem}"')
    L.info(f"Time to generate LTS: {elapsed_time}s")

    # output png graphic from dot diagram
    if gen_pngs:
        for dot in Path(combined_filename).parent.glob("*.dot"):
            output_filename = str(dot.with_suffix(".png"))
            with open(output_filename, "wb") as outfile:
                retcode = call(["dot", dot.absolute(), "-Tpng"], stdout=outfile)
                assert retcode == 0, DOT_INVOKE_ERROR_MSG
                L.info(f'PNG file saved at "{output_filename}"')

    return combined_filename


@app.command(no_args_is_help=True)
def checklts(
    lts_filename: str,
    parsed_lts=typer.Argument(
        None,
        hidden=True,
    ),
) -> bool:
    """
    Checks compliance of the given CS as a dot.
    """

    start = perf_counter()

    lts = parsed_lts or LTS.parse(lts_filename)

    if not parsed_lts:
        L.info(f"LTS parsing check took {perf_counter() - start} seconds")
        start = perf_counter()

    oracle_filename = ORACLE_DEFAULT_FILENAME

    with open(str(Path(lts_filename).parent / oracle_filename), "r") as oracle_f:
        oracle = yaml.load(oracle_f, Loader=yaml.FullLoader)

    final_confs: List[List[State]] = [
        oracle["success_states"][p] for p in oracle["order"]
    ]
    compliant = lts.is_compliant(final_confs)

    L.info(f"Compliance check took {perf_counter() - start} seconds")

    if compliant:
        L.info(f"{lts_filename} is compliant.")
    else:
        L.warning(f"{lts_filename} is NOT compliant!")

    return compliant


@app.command(no_args_is_help=True)
def run(cs_filename: str):
    """
    Executes the given communicating system interactively.
    """
    cs = CommunicatingSystem.parse(cs_filename)
    cs.execute_interactively()


@app.command(no_args_is_help=True)
def getdot(cs_filename: str):
    """
    Converts the given CS to a dot diagram.
    """
    cs = CommunicatingSystem.parse(cs_filename)
    fp = Path(cs_filename)
    of = fp.parent

    cs.to_dot(str(of))
    L.info(f"Machines saved to {str(of)}")


def main():
    LOG_FILENAME = "chortest.log"
    log_file_handler = FileHandler(LOG_FILENAME)
    log_file_handler.setFormatter(
        Formatter("[%(asctime)s] - %(levelname)s - %(message)s")
    )
    rich_handler = RichHandler()
    rich_handler.setFormatter(Formatter("%(message)s"))

    basicConfig(
        level="DEBUG", datefmt="[%X]", handlers=[rich_handler, log_file_handler]
    )

    # try:
    app(
        prog_name="chortest",
    )
    # except Exception as e:
    # L.info(f'⚠️  The command failed with message:\n"{str(e)}".')
    # L.info_exception()

    # if console.input("❓ Do you want to inspect the traceback? \[y/N] ") == "y":
    # L.info("Check the traceback below.")


if __name__ == "__main__":
    main()
