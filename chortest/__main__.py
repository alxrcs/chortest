import logging
import os
import subprocess
import sys
from datetime import datetime
from glob import glob
from logging import FileHandler, Formatter, basicConfig, getLogger
from os import makedirs
from pathlib import Path
from shutil import copy
from time import perf_counter, process_time, sleep, strftime, time
from typing import Any, Dict, List, Optional

import typer
import yaml
from rich.logging import RichHandler
from typer import Typer

from .cfsm import State
from .common import Participant
from .defaults import DEFAULT_CHORGRAM_BASE_PATH
from .errors import CHORGRAM_INVOKE_ERROR_MSG, DOT_INVOKE_ERROR_MSG
from .helpers import combine_fsa
from .lts import LTS
from .metagen import generate_concrete_tests
from .mutations import LocalMutator
from .parsing import Parsers
from .utils import fail_unless

app = Typer(add_completion=False, add_help_option=True, no_args_is_help=True)
L = getLogger("chortest")

PROJECTION_COMMAND = "project"
UNFOLD_COMMAND = "gcunfold"
LTSGEN_COMMAND = "gents"


def call(cmd, *args, **kwargs):
    before = time()
    cmd_str = " ".join(str(s) for s in cmd)
    L.info(f"Calling {str(cmd_str)}")
    if not Path(cmd[0]).is_absolute():
        cmd[0] = Path(DEFAULT_CHORGRAM_BASE_PATH).resolve() / cmd[0]
    ret = subprocess.run(cmd, *args, **kwargs)
    L.info(f"Done in {(time()-before):.3f}s!")
    return ret.returncode


def timeit(func, d: Dict[Any, Any], param: str):
    def inner(*args, **kwargs):
        t1 = process_time()
        f = func(*args, **kwargs)
        t2 = process_time()
        d[param] = t2 - t1
        return f

    return inner


@app.command(no_args_is_help=True)
def project(
    gc_filename_or_folder: str,
    output_folder: Optional[str] = None,
    unbound_length: int = 2,
    # determinize: bool = False, TODO: Add parameter and remove determinize command
):
    """
    Projects a g-choreography into a communicating system.

    Optionally a folder can also be passed, which will
    process all .gc files in that folder.
    """

    gc_path = Path(gc_filename_or_folder)

    if not gc_path.exists():
        L.error("Invalid path or path does not exist!")
        return

    if gc_path.suffix != ".gc" and not gc_path.is_dir():
        L.error("Invalid file extension! Make sure you're projecting a .gc file or a folder containing .gc files.")
        return
    if gc_path.is_file():
        gc_paths = [gc_path]
    elif gc_path.is_dir():
        gc_paths = list(gc_path.glob("*.gc"))
    else:
        L.error(f"Unknown issue reading {gc_path}")
        return

    before = perf_counter()
    output_filepath = None

    for gc_path in gc_paths:
        L.info(f"Processing {gc_path} ...")

        if output_folder is None:
            output_path = gc_path.parent
            makedirs(output_path, exist_ok=True)
        else:
            output_path = Path(output_folder)

        output_filepath = (output_path / f"{gc_path.name}").with_suffix(".fsa")

        with open(output_filepath, "wb") as outfile:
            retcode = call(
                [
                    PROJECTION_COMMAND,
                    "-u",
                    str(unbound_length),
                    "-D",
                    "min",
                    str(gc_path),
                ],
                stdout=outfile,
                stderr=outfile,
            )

            assert retcode == 0, (
                CHORGRAM_INVOKE_ERROR_MSG
                + f" Check {output_filepath} for more details."
            )

            L.info(f"Projections saved to {output_filepath}")

    L.info(f"Projections done in {(perf_counter()-before):.3f}s!")
    return output_filepath


@app.command(no_args_is_help=True)
def gentests(
    cs_filename_or_folder: str,
    participant: Optional[str] = None,
    output: Optional[str] = None,
    include_timestamp: Optional[bool] = False,
):
    """
    Generates tests for a given communicating system.
    Expects the system to be in a single .fsa file.
    """

    cs_path = Path(cs_filename_or_folder)
    if cs_path.is_file() and cs_path.suffix == ".fsa":
        cs_paths = [cs_path]
    elif cs_path.is_dir():
        cs_paths = list(cs_path.glob("*.fsa"))
    else:
        L.error(f"Cannot recognize given path: {cs_filename_or_folder}")
        L.error("Expected .fsa file or folder containing .fsa files.")
        return

    counter = 0

    all_output_tests = []

    for cs_filename in cs_paths:
        cs = Parsers.parseFile(cs_filename)

        fsa_f = Path(cs_filename)
        tests_path = fsa_f.parent / f"{fsa_f.stem}_tests"

        if include_timestamp:
            tests_path = tests_path / str(datetime.now().isoformat(sep="_").replace(":", ""))

        if output is not None:
            tests_path = Path(output)

        if participant is None and cs_path.is_file():
            L.info(
                f"No participant specified. Will generate tests for all participants."
            )
            participants = list(cs.participants)
        elif participant is None and cs_path.is_dir():
            # Generate only for the participant in the filename
            participants = [Participant(str(cs_filename).split("_")[-2])]
        else:
            participants = [Participant(str(participant))]

        start_time = process_time()

        for p in participants:
            L.info(f"Generating tests for participant {str(p)}")
            tests = cs.tests(p)
            for i, test in enumerate(tests):
                counter += 1
                L.info(f"Generating test #{i}")
                output_test_filename = str(tests_path / p / f"test_{i}" / f"test_{i}.fsa")
                test.to_fsa(output_test_filename)
                all_output_tests.append(output_test_filename)
            L.info(f'Tests saved to "{str(tests_path / p)}"')

        elapsed_time = process_time() - start_time
        L.info(f"Tests generated in {elapsed_time}s")

    L.info(f"Generated a total of {counter} tests")

    return all_output_tests


@app.command(no_args_is_help=True)
def genlts(
    fsa_filename: str,
    output_filename: Optional[str] = None,
    buffer_size: int = 5,
    fifo_semantics: bool = False,
    cut_filename: Optional[str] = None,
    gen_pngs: bool = False,
):
    """
    Generates the LTS
    for a given communicating system.
    """
    # if not output_filename:
    #     L.disabled = True

    if not str(fsa_filename).endswith(".fsa") or not os.path.exists(fsa_filename):
        typer.echo("Please provide a valid .fsa file.")
        raise typer.Exit(1)

    output_path = Path(
        fsa_filename
    ).parent.parent  # This is needed since chorgram generates the lts in a folder with the same name as the fsa file.

    os.makedirs(output_path, exist_ok=True)

    if cut_filename is not None:
        combined_foldername = f"{Path(cut_filename).stem}__{Path(fsa_filename).stem}"
        combined_filename = str(
            output_path / combined_foldername / (combined_foldername + ".fsa")
        )
        combine_fsa(fsa_filename, cut_filename, combined_filename)
        copy(  # oracle
            Path(fsa_filename).with_suffix(".fsa.oracle.yaml"),
            output_path / combined_foldername,
        )
    else:
        combined_filename = fsa_filename

    # invoke the transition system builder
    start_time = perf_counter()

    combined_p = Path(combined_filename)
    dot_file_filename = f"{combined_p.stem}_ts{buffer_size}.dot"
    output_path = combined_p.parent / dot_file_filename

    output_file = (
        open(output_filename, "w") if output_filename else open(output_path, "w")
    )

    retcode = call(
        [
            LTSGEN_COMMAND,
            "-b",
            str(buffer_size),
            "-nf" if not fifo_semantics else "",
            Path(combined_filename).absolute(),
        ],
        stdout=output_file,
        cwd=DEFAULT_CHORGRAM_BASE_PATH,
    )

    if output_file:
        L.info(f'LTS saved to "{str(output_filename or output_path)}"')
        output_file.close()

    elapsed_time = perf_counter() - start_time
    assert retcode == 0, CHORGRAM_INVOKE_ERROR_MSG
    L.info(f"Time to generate LTS: {elapsed_time}s")

    # output png graphic from dot diagram
    if gen_pngs:
        for dot in Path(combined_filename).parent.glob("*.dot"):
            png_filename = str(dot.with_suffix(".png"))
            with open(png_filename, "wb") as outfile:
                retcode = call(["dot", dot.absolute(), "-Tpng"], stdout=outfile)
                assert retcode == 0, DOT_INVOKE_ERROR_MSG
                L.info(f'PNG file saved at "{png_filename}"')

    return str(output_path)


@app.command(no_args_is_help=True)
def checklts(
    lts_filename: str,
    part_name: str,
    oracle_filename: Optional[str] = None,
    parsed_lts=typer.Argument(
        None,
        hidden=True,
    ),
    use_exit_codes=False,
) -> bool:
    """
    Checks compliance of the given CS as a dot.
    The state of the given participant will be ignored when
    checking the LTS.
    """

    sys.setrecursionlimit(2000)  # Patch for larger LTSs

    start_t = process_time()
    lts_path = Path(lts_filename)

    lts: LTS = parsed_lts or Parsers.parseDOT(lts_filename)

    if not parsed_lts:
        L.info(f"LTS parsing check took {process_time() - start_t} seconds")
        start_t = process_time()

    # TODO: Extract this into parsers
    if not oracle_filename:
        oracle_paths = glob(str(Path(lts_path.parent)) + '/*.oracle.yaml')
        fail_unless(len(oracle_paths) == 1, f"Expected exactly one oracle file in {lts_path}")
        oracle_path = Path(oracle_paths[0])
    else:
        oracle_path = Path(oracle_filename)

    fail_unless(oracle_path.exists(), f"Oracle file {oracle_path} does not exist")
    with open(oracle_path, "r") as oracle_f:
        oracle = yaml.load(oracle_f, Loader=yaml.FullLoader)
        L.info(f"Using {oracle_path} as oracle.")

    final_confs: List[List[State]] = [
        oracle["success_states"][p] for p in oracle["order"]
    ]

    compliant = lts.is_compliant(final_confs, oracle["order"], part_name)

    L.info(f"Compliance check took {perf_counter() - start_t} seconds")

    if compliant:
        L.info(f"{lts_filename} is compliant.")
    else:
        L.warning(f"{lts_filename} is NOT compliant!")

    if not use_exit_codes:
        return compliant

    raise typer.Exit(code=1 if not compliant else 0)


@app.command(no_args_is_help=True)
def execute(cs_filename: str):
    """
    Executes the given communicating system interactively.
    """
    cs = Parsers.parseFile(cs_filename)
    cs.execute_interactively()


@app.command(no_args_is_help=True)
def getdot(cs_filename: str, output_filename: Optional[str] = None):
    """
    Converts the given CS to a dot diagram.
    """
    cs = Parsers.parseFile(cs_filename)
    output = open(output_filename, "w") if output_filename else sys.stdout
    cs.to_dot(output)


@app.command(no_args_is_help=True)
def mutate(
    fsa_filename: str,
    output_folder: Optional[str] = None,
    participant: Optional[str] = None,
    seed: Optional[int] = 1,
    mutate_randomly: bool = False,
    number_of_mutants: int = 1,
):
    """
    Mutates a given communicating system
    """
    source_cs = Parsers.parseFile(fsa_filename)
    fsa_path = Path(fsa_filename)

    if not participant:
        part = Participant(source_cs.participants[0])
    else:
        part = Participant(participant)

    L.info(f"Mutating for participant {str(part)}")

    if not mutate_randomly:
        mutants = LocalMutator.mutate_systematically(source_cs, part)
        output_folder_path = (
            Path(output_folder) if output_folder else fsa_path.with_suffix("")
        )
        for i, (mutant, change) in enumerate(mutants):
            output_path = (output_folder_path / fsa_path.name).with_suffix(
                f".mut_{part}.{i}.fsa"
            )
            mutant.to_fsa(str(output_path), part, output_oracle=False)
            L.info(f"Mutant saved to {str(output_path)}")
        
        source_saved_path = (output_folder_path / fsa_path.name).with_suffix(
            f".original.fsa"
        )
        source_cs.to_fsa(str(source_saved_path), output_oracle=False)
        L.info(f"Original saved to {str(source_saved_path)}")
    else:
        # TODO: Check that the random mutations still work
        for i in range(number_of_mutants):
            cs = source_cs.copy()
            mutation_type = LocalMutator.mutate_randomly(
                cs, seed=seed or 1, target_p=part
            )

            output_folder_path = (
                Path(output_folder) if output_folder else fsa_path.with_suffix("")
            )

            output_path = (output_folder_path / fsa_path.name).with_suffix(
                f".mutated.{i}.fsa"
            )

            cs.to_fsa(str(output_path), part=participant, output_oracle=False)
            L.info(f"Mutant saved to {str(output_path)} with mutation {mutation_type}")


@app.command(no_args_is_help=True)
def unfold(gc_filename: str, n: int, output_filename: Optional[str] = None):
    """
    Unfolds a given communicating system.
    """

    output_filepath = (
        Path(output_filename)
        if output_filename
        else Path(Path(gc_filename).stem + f"_unfolded_{n}.gc")
    )

    with open(output_filepath, "wb") as outfile:
        retcode = call(
            [
                UNFOLD_COMMAND,
                str(n),
                gc_filename,
            ],
            stdout=outfile,
        )
        if retcode:
            typer.Exit(retcode)

    L.info(f"Saved unfolded gc to {output_filepath}")
    return output_filepath


@app.command(no_args_is_help=True)
def splitgc(gc_filename: str, output_folder: Optional[str] = None, participant: Optional[str] = None):
    """
    Removes choices from a choreography,
    resulting in multiple choice-free choreographies as output

    NOTE: Expects the gchor to have the active choice participants
    explicit in the syntax.
    """
    chor = Parsers.parseFile(gc_filename)
    gc_path = Path(gc_filename)

    if not output_folder:
        output_path = (gc_path.parent / gc_path.stem).absolute()
        makedirs(output_path, exist_ok=True)
    else:
        output_path = Path(output_folder).absolute()
        makedirs(output_path, exist_ok=True)

    if participant is None:
        participants = chor.participants()
    else:
        participants = set([Participant(participant)])

    total_gcs = 0
    for part in participants:
        paths = chor.paths(part)
        for i, simpler_gc in enumerate(paths):
            filename = f"{Path(gc_filename).stem}_{part}_{i}.gc"
            file_path = output_path / filename
            file_path.write_text(str(simpler_gc))

            L.info(f"Wrote to {file_path}")

        L.info(f"Saved {len(paths)} g-choreographies for Participant {part}.")
        total_gcs += len(paths)

    L.info(f"Saved {total_gcs} choreographies in total.")


@app.command(no_args_is_help=True, hidden=True)
def paths(gc_filename: str):
    gc = Parsers.parseFile(gc_filename)
    gc_path = Path(gc_filename)
    L.info(str(gc))
    for p in gc.participants():
        l = gc.paths(p)
        L.info(f"Found {len(l)} paths for CUT {str(p)}")
        for i, path in enumerate(l):
            L.info(f"Path {i}:")
            L.info(str(path))

            output_folder = gc_path.parent / gc_path.stem
            output_folder.mkdir(exist_ok=True)
            (output_folder / f"{gc_path.stem}_{p}_{i}.gc").write_text(str(path))


def setup_logging():
    logger = getLogger("chortest")
    logger.setLevel(logging.DEBUG)

    log_folder = Path("~/.chortest_logs").expanduser()
    log_folder.mkdir(exist_ok=True)

    timestr = strftime("%Y%m%d-%H%M%S")
    LOG_FILENAME = log_folder / f"chortest-{timestr}.log"
    log_file_handler = FileHandler(LOG_FILENAME)
    log_file_handler.setLevel(logging.DEBUG)
    log_file_handler.setFormatter(
        Formatter("[%(asctime)s] - %(levelname)s - %(message)s")
    )

    rich_handler = RichHandler()
    rich_handler.setFormatter(Formatter("%(message)s"))

    logger.addHandler(log_file_handler)
    logger.addHandler(rich_handler)
        
    return logger

@app.command(no_args_is_help=True)
def gencode(model_path: str, watch: bool = False):
    """
    Generates executable tests from a given model.
    """
    model_name = Path(model_path).stem
    # check that the model is an .fsa file 
    if not model_path.endswith(".fsa"):
        raise Exception(f"Model file {model_path} must be an .fsa file")
    cfsm = Parsers.parseFSA(str(model_path))
    while True: 
        try:
            output_path = generate_concrete_tests(model_name, model_path, cfsm)
            if not watch:
                print(f"Generated tests in {output_path}")
                break
            sleep(1.)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            sleep(1.)



def main():
    logger = setup_logging()

    from chortest.experiments import app as experiments_app
    app.add_typer(experiments_app, name="experiments", help="Run experiments")

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
