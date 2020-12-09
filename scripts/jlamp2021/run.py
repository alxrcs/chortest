import os
from collections import defaultdict
from logging import Formatter, basicConfig, getLogger
from pathlib import Path
from time import perf_counter, process_time
from typing import DefaultDict, List, Optional, Union
import json

from chortools import __main__ as cli
from rich.logging import RichHandler
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


data: DefaultDict[str, Union[List[float], int]] = defaultdict(lambda: list())
import time


def timeit(func, param):
    def inner(*args, **kwargs):
        t1 = time.process_time()
        f = func(*args, **kwargs)
        t2 = time.process_time()
        data['param'].append(t2 - t1)

        # t1 = time.perf_counter()
        # f = func(*args, **kwargs)
        # t2 = time.perf_counter()
        # data[param + '(perf counter)'].append(t2 - t1)
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

    gentests = timeit(cli.gentests, "Time to generate tests")
    gentests(str(BASE_DIR / "fsa" / Path(GCHOR_FNAME).with_suffix(".fsa")))

    TESTS_BASE_DIR = BASE_DIR / "fsa" / (Path(GCHOR_FNAME).stem + "_tests")
    test_paths = get_test_paths(TESTS_BASE_DIR)

    data["Number of tests"] = len(test_paths)
    data["Compliant tests"] = 0
    for test_path in test_paths:
        L.info(f"Generating LTS for {test_path}")
        genlts = timeit(cli.genlts, "Time to generate LTS")
        genlts(str(test_path), cut_filename=substitute_fsa)

        lts_path = test_path.parent / (test_path.stem + "_ts5.dot")

        L.info(f"Checking projection test compliance...")
        checklts = timeit(cli.checklts, "Time to check compliance")
        compliant = checklts(str(lts_path))
        data["Compliant tests"] += compliant

    data["Counterexamples"] = data["Number of tests"] - data["Compliant tests"]
    data["Total time for LTS generation"] = sum(data['Time to generate LTS'])
    data['Average time for LTS generation'] = data["Total time for LTS generation"] / data["Number of tests"]

    log_filename = gchor_path.with_suffix('.log') if substitute_fsa is None else Path(substitute_fsa).with_suffix('.log')

    with open(log_filename, 'w') as j:
        json.dump(data, j)



def experiment_0():
    run_experiment("scripts/jlamp2021/ATM/atm_simple.gg")


def experiment_1():
    run_experiment(
        gchor="scripts/jlamp2021/ATM/atm_full.gg",
        substitute_fsa="scripts/jlamp2021/ATM/fsa/atm_full_01_no_quit_ATM2Bank.fsa",
    )

def experiment_2():
    run_experiment(
        gchor="scripts/jlamp2021/ATM/atm_full.gg",
        substitute_fsa="scripts/jlamp2021/ATM/fsa/atm_full_02_no_checkBalanceATM.fsa",
    )

def experiment_3():
    run_experiment(
        gchor="scripts/jlamp2021/ATM/atm_full.gg",
        substitute_fsa="scripts/jlamp2021/ATM/fsa/atm_full_03_no_ATM_balance_always_0.fsa",
    )

def main():
    # experiment_0()
    experiment_1()
    experiment_2()
    experiment_3()
    print(data)


if __name__ == "__main__":
    setup_log()
    main()
