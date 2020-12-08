import os
from logging import Formatter, basicConfig, getLogger
from pathlib import Path
from time import perf_counter
from typing import List, Optional

from rich.logging import RichHandler

from chortools import __main__ as cli
from typer import Typer

app = Typer()
L = getLogger(__name__)


def mainCLI():
    pass
    # result = runner.invoke(app, [
    #     'project', 'scripts/jlamp2021/ATM/atm_simple.gg'
    # ])

    # result = runner.invoke(app, [
    #     'gentests', 'scripts/jlamp2021/ATM/fsa/atm_simple.fsa'
    # ])

    # result = runner.invoke(app, [
    #     'genlts', 'scripts/jlamp2021/ATM/fsa/atm_simple_tests/A/test_0/test_0.fsa'
    # ])

    # result = runner.invoke(app, [
    #     'checklts', 'scripts/jlamp2021/ATM/fsa/atm_simple_tests/A/test_0/test_0_ts5.dot'
    # ])


data = {}
import time


def timeit(func, param):
    def inner(*args, **kwargs):
        t1 = time.perf_counter()
        f = func(*args, **kwargs)
        t2 = time.perf_counter()
        data[param] = t2 - t1
        return f

    return inner


def get_test_paths(tests_dir) -> List[Path]:
    test_paths = []
    for root, dirs, files in os.walk(tests_dir):
        if not files:
            continue
        for f in files:
            if f.endswith(".fsa") and "tmp" not in f:
                test_path = Path(root) / f
                test_paths.append(test_path)
    return test_paths


def setup_log():
    rich_handler = RichHandler()
    rich_handler.setFormatter(Formatter("%(message)s"))

    basicConfig(level="DEBUG", datefmt="[%X]", handlers=[rich_handler])


def run_experiment(gchor: Optional[str] = None, substitute_fsa: Optional[str] = None):

    if gchor is None:
        gchor_path = Path(".") / "scripts" / "jlamp2021" / "ATM" / "atm_simple.gg"
    else:
        gchor_path = Path(gchor)

    BASE_DIR = gchor_path.parent
    GCHOR_FNAME = gchor_path.name

    project = timeit(cli.project, "Time to project")
    project((str(BASE_DIR / GCHOR_FNAME)))

    cli.gentests(str(BASE_DIR / "fsa" / Path(GCHOR_FNAME).with_suffix(".fsa")))

    TESTS_BASE_DIR = BASE_DIR / "fsa" / (Path(GCHOR_FNAME).stem + "_tests")
    test_paths = get_test_paths(TESTS_BASE_DIR)

    for test_path in test_paths:
        L.info(f"Generating LTS for {test_path}")
        cli.genlts(str(test_path), cut_filename=substitute_fsa)

        lts_path = test_path.parent / (test_path.stem + "_ts5.dot")
        L.info(f"Checking projection test compliance...")
        cli.checklts(str(lts_path))


def experiment_0():
    run_experiment("scripts/jlamp2021/ATM/atm_simple.gg")


def experiment_1():
    run_experiment(
        gchor="scripts/jlamp2021/ATM/atm_full.gg",
        substitute_fsa="scripts/jlamp2021/ATM/fsa/atm_full_no_quit_ATM2Bank.fsa",
    )


def main():
    experiment_0()
    print(data)


if __name__ == "__main__":
    setup_log()
    main()
