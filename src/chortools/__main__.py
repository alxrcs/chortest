from datetime import datetime
from logging import FileHandler, Formatter, basicConfig, getLogger
from os import makedirs
from pathlib import Path
from subprocess import call
from time import perf_counter
from typing import Optional
from shutil import copy

from rich.logging import RichHandler
from typer import Typer

from chortools.helpers import combine_fsa
from chortools.lts import LTS

from .cfsm import CommunicatingSystem
from .gchor import Participant

app = Typer()
L = getLogger()

CHORGRAM_BASE_PATH = Path("chorgram")
PROJECTION_COMMAND = "gg2fsa"
FSA_OUTPUT_DEFAULT_FOLDER = "fsa"
CHORGRAM_INVOKE_ERROR_MSG = (
    "Could not invoke chorgram. Check that dependencies are correctly installed."
)
DOT_INVOKE_ERROR_MSG = (
    "Could not invoke dot. Check that graphviz is properly installed."
)
ORACLE_DEFAULT_FILENAME = "oracle.yaml"


@app.command()
def project(gchor_filename: str, output_folder: str = None):
    """
    Projects a g-choreography into a communicating system
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

        assert retcode == 0, CHORGRAM_INVOKE_ERROR_MSG

        L.info(f"Projections saved to {output_filepath}")


@app.command()
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
    cs = CommunicatingSystem.parse(cs_filename)
    output_foldername = str(datetime.now().isoformat(sep="_").replace(":", ""))
    fsa_f = Path(cs_filename)
    tests_path = fsa_f.parent / f"{fsa_f.stem}_tests"
    if include_timestamp:
        tests_path = tests_path / output_foldername

    if output is not None:
        tests_path = Path(output)

    if participant is not None:
        participants = [Participant(participant)]
    else:
        participants = list(cs.participants)

    start_time = perf_counter()

    for p in participants:
        tests = list(cs.tests(p))
        for i, test in enumerate(tests):
            test.to_fsa(
                str(tests_path / p.participant_name / f"test_{i}" / f"test_{i}.fsa")
            )
        L.info(f'{len(tests)} tests saved to "{str(tests_path / p.participant_name)}"')

    elapsed_time = perf_counter() - start_time
    L.info(f"Tests generated in {elapsed_time}s")


@app.command(no_args_is_help=True)
def genlts(
    fsa_filename: str,
    output_folder: Optional[str] = None,
    buffer_size: int = 5,
    fifo_semantics: bool = False,
    cut_filename: str = None,
):
    """
    Generates the labeled transition system
    for a given communicating system.
    """
    output_path = (
        Path(
            fsa_filename
        ).parent.parent  # This is needed since chorgram generates the lts in a folder with the same name as the fsa file.
        if output_folder is None
        else Path(output_folder)
    )
    output_path.mkdir(exist_ok=True)

    if cut_filename is not None:
        combined_foldername = f"{Path(fsa_filename).stem}__{Path(cut_filename).stem}"
        combined_filename = str(
            output_path / combined_foldername / (combined_foldername + ".fsa")
        )
        combine_fsa(fsa_filename, cut_filename, combined_filename)
        copy( # oracle
            Path(fsa_filename).parent / "oracle.yaml", output_path / combined_foldername
        )  
    else:
        combined_filename = fsa_filename

    # invoke the transition system builder
    start_time = perf_counter()
    retcode = call(
        [
            str((CHORGRAM_BASE_PATH / "cfsm2gg.py").absolute()),
            "-ts",
            Path(combined_filename).absolute(),
            "-dir",
            Path(output_path).absolute(),
            "-b",
            str(buffer_size),
            "-nf" if not fifo_semantics else "",
            "-sn",  # Do not shorten state names
        ],
        cwd=CHORGRAM_BASE_PATH,
    )
    elapsed_time = perf_counter() - start_time
    assert retcode == 0, CHORGRAM_INVOKE_ERROR_MSG
    L.info(f'LTS saved to "{output_path}/{Path(combined_filename).stem}"')
    L.info(f"Time to generate LTS: {elapsed_time}s")

    # output png graphic from dot diagram
    for dot in Path(combined_filename).parent.glob("*.dot"):
        output_filename = str(dot.with_suffix(".png"))
        with open(output_filename, "wb") as outfile:
            retcode = call(["dot", dot.absolute(), "-Tpng"], stdout=outfile)
            assert retcode == 0, DOT_INVOKE_ERROR_MSG
            L.info(f'PNG file saved at "{output_filename}"')


@app.command()
def checklts(lts_filename: str) -> bool:
    """
    Checks compliance of the given CS as a dot.
    """

    start = perf_counter()

    lts: LTS = LTS.parse(lts_filename)
    import yaml

    oracle_filename = ORACLE_DEFAULT_FILENAME

    with open(str(Path(lts_filename).parent / oracle_filename), "r") as oracle_f:
        oracle = yaml.load(oracle_f, Loader=yaml.FullLoader)

    final_confs = [oracle["success_states"][p] for p in oracle["order"]]
    compliant = lts.is_compliant(final_configurations=final_confs)

    L.info(f"Compliance check took {perf_counter() - start} seconds")

    if compliant:
        L.info(f"{lts_filename} is compliant.")
        return True
    else:
        L.warning(f"{lts_filename} is NOT compliant!")
        return False


@app.command()
def run(cs_filename: str):
    """
    Executes the given communicating system interactively.
    """
    cs = CommunicatingSystem.parse(cs_filename)
    cs.execute_interactively()


def main():
    LOG_FILENAME = "chortools.log"
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
        prog_name="chortools",
    )
    # except Exception as e:
    # L.info(f'⚠️  The command failed with message:\n"{str(e)}".')
    # L.info_exception()

    # if console.input("❓ Do you want to inspect the traceback? \[y/N] ") == "y":
    # L.info("Check the traceback below.")


if __name__ == "__main__":
    main()
