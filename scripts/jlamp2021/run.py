import json
import os
import time
from collections import defaultdict
from logging import Formatter, basicConfig, getLogger
from pathlib import Path
from typing import DefaultDict, List, Optional, Union

import pandas as pd
from chortools import __main__ as cli
from chortools.lts import LTS
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


def timeit(func, d: DefaultDict, param: str):
    def inner(*args, **kwargs):
        t1 = time.process_time()
        f = func(*args, **kwargs)
        t2 = time.process_time()
        if isinstance(d[param], list):
            d[param].append(t2 - t1)
        else:
            d[param] = t2 - t1

        return f

    return inner


def get_test_paths(tests_dir) -> List[Path]:
    test_paths = []
    for root, dirs, files in os.walk(tests_dir):
        if not files or "__" in root:
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

    general_data: DefaultDict[str, Union[float, int]] = defaultdict(lambda: 0)
    specific_data: DefaultDict = defaultdict(lambda: list())

    # Project global choreography
    project = timeit(cli.project, general_data, "Time to project")
    project((str(BASE_DIR / GCHOR_FNAME)))

    # Generate tests
    gentests = timeit(cli.gentests, general_data, "Time to generate tests")
    gentests(str(BASE_DIR / "fsa" / Path(GCHOR_FNAME).with_suffix(".fsa")))

    TESTS_BASE_DIR = BASE_DIR / "fsa" / (Path(GCHOR_FNAME).stem + "_tests")
    test_paths = get_test_paths(TESTS_BASE_DIR)

    general_data["Number of tests"] = len(test_paths)
    general_data["Compliant tests"] = 0
    for test_path in test_paths:
        L.info(f"Generating LTS for {test_path}")
        genlts = timeit(cli.genlts, specific_data, "Time to generate LTS")
        lts_path = genlts(str(test_path), cut_filename=substitute_fsa)
        lts_path = Path(lts_path).parent / (Path(lts_path).stem + "_ts5.dot")

        lts = LTS.parse(str(lts_path))
        specific_data["Number of nodes"].append(len(lts.configurations))
        specific_data["Number of transitions"].append(len(lts.transitions))
        specific_data["CUT"].append(test_path.parent.parent.stem)

        L.info(f"Checking projection test compliance...")
        checklts = timeit(cli.checklts, specific_data, "Time to check compliance")
        compliant = checklts(str(lts_path))

        specific_data["Path"].append(str(lts_path))
        specific_data["Pass"].append(compliant)

        general_data["Compliant tests"] += compliant

    general_data["Failed tests"] = (
        general_data["Number of tests"] - general_data["Compliant tests"]
    )

    total_time_for_lts_generation = sum(specific_data["Time to generate LTS"])
    total_time_for_compliance_check = sum(specific_data["Time to check compliance"])

    general_data["Total time for LTS generation"] = total_time_for_lts_generation
    general_data["Total time for compliance check"] = total_time_for_compliance_check

    general_data[
        "Average time for LTS generation"
    ] = total_time_for_lts_generation / len(test_paths)
    general_data[
        "Average time for compliance check"
    ] = total_time_for_compliance_check / len(test_paths)

    log_path = gchor_path if substitute_fsa is None else Path(substitute_fsa)

    with open(log_path.with_suffix(".summary.log"), "w") as j:
        json.dump(general_data, j)
    with open(log_path.with_suffix(".pertest.log"), "w") as j:
        json.dump(specific_data, j)

    pd.DataFrame(specific_data).to_excel(log_path.with_suffix(".pertest.log.xlsx"))
    pd.DataFrame(general_data, index=[0]).to_csv(
        log_path.with_suffix(".summary.log.csv")
    )


def experiment_0():
    """
    Just a small choreography for sanity checking.
    """
    run_experiment("scripts/jlamp2021/ATM/atm_simple.gg")


def experiment_1_0():
    """
    A larger choreography for the ATM example.
    """
    run_experiment("scripts/jlamp2021/ATM/atm_full.gg")


def experiment_1_1():
    """
    An example where the quit message from the ATM to the Bank is incorrectly removed.
    """
    run_experiment(
        gchor="scripts/jlamp2021/ATM/atm_full.gg",
        substitute_fsa="scripts/jlamp2021/ATM/fsa/atm_full_01_no_quit_ATM2Bank.fsa",
    )


def experiment_1_2():
    """
    An example where the ATM does not support the checkBalance message.
    """
    run_experiment(
        gchor="scripts/jlamp2021/ATM/atm_full.gg",
        substitute_fsa="scripts/jlamp2021/ATM/fsa/atm_full_02_no_checkBalanceATM.fsa",
    )


def experiment_1_3():
    """
    An example where the ATM does not interact with the bank when asked for a customer's balance.
    """
    run_experiment(
        gchor="scripts/jlamp2021/ATM/atm_full.gg",
        substitute_fsa="scripts/jlamp2021/ATM/fsa/atm_full_03_no_ATM_balance_always_0.fsa",
    )


def experiment_2_0():
    """
    A correct implementation of the shipping example.
    """
    run_experiment(gchor="scripts/jlamp2021/shipping/shipping.sgg")


def experiment_2_1():
    """
    Uses wrong implementation of P from Isola paper.
    """
    pass


def experiment_2_2():
    """
    Test P, where the order of cancel in the left thread of the right branch is swapped.
    (note: there should be no counterexample for this one)
    """
    pass


def experiment_2_3():
    """
    Test T, where you swap the input/output in the right thread or right branch
    """
    pass


def experiment_2_4():
    """
    The client sends Shipments Details and places the order before ever receiving a quote.
    """
    pass


def main():
    experiment_0()
    experiment_1_0()
    experiment_1_1()
    experiment_1_2()
    experiment_1_3()
    # experiment_2_0()


if __name__ == "__main__":
    setup_log()
    main()
