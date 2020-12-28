import os
import shutil
import sys
import time
from collections import defaultdict
from glob import glob
from logging import Formatter, basicConfig, getLogger
from pathlib import Path
from typing import DefaultDict, List, Optional, Union

import lark
import pandas as pd
from chortest import __main__ as cli
from chortest.lts import LTS
from rich.logging import RichHandler
from rich.progress import track
from typer import Typer

app = Typer()
L = getLogger(__name__)


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
        if not files or "__" in root:  # Avoid using tests from other experiments
            continue
        for f in files:
            if f.endswith(".fsa") and "tmp" not in f:
                test_path = Path(root) / f
                test_paths.append(test_path)
    return test_paths


def setup():
    sys.setrecursionlimit(2000)  # Default is ~900

    rich_handler = RichHandler()
    rich_handler.setFormatter(Formatter("%(message)s"))

    basicConfig(level="DEBUG", datefmt="[%X]", handlers=[rich_handler])


def run_experiment(gchor: Optional[str] = None, substitute_fsa: Optional[str] = None):

    if gchor is None:
        gchor_path = Path(".") / "scripts" / "jlamp2021" / "ATM" / "atm_simple.gc"
    else:
        gchor_path = Path(gchor)

    substitute_name = "All"
    if substitute_fsa is not None:
        grammarfile_path = Path("grammars") / "fsa.lark"
        fsa_parser = lark.Lark.open(str(grammarfile_path))

        with open(substitute_fsa) as f:
            tree = fsa_parser.parse(f.read())

        try:
            substitute_name = tree.children[0].children[0].children[0].value  # type: ignore
        except AttributeError:
            raise Exception("Incorrect fsa format")

        assert (
            substitute_name is not None and substitute_name != ""
        ), "The participant name in the fsa file should be valid"

    BASE_DIR = gchor_path.parent
    GCHOR_FNAME = gchor_path.name
    TESTS_BASE_DIR = BASE_DIR / "fsa" / (Path(GCHOR_FNAME).stem + "_tests")
    for tdir in glob(str(TESTS_BASE_DIR) + "/*/test_*"):
        if "__" not in tdir:  # Avoid removing tests from other experiments
            shutil.rmtree(tdir)

    summary_data: DefaultDict[str, Union[float, int]] = defaultdict(lambda: 0)
    test_data: DefaultDict = defaultdict(lambda: list())

    def log(value, description):
        test_data[description].append(value)

    # Project global choreography
    project = timeit(cli.project, summary_data, "Time to project")
    project((str(BASE_DIR / GCHOR_FNAME)))

    # Generate tests
    gentests = timeit(cli.gentests, summary_data, "Time to generate tests")
    gentests(str(BASE_DIR / "fsa" / Path(GCHOR_FNAME).with_suffix(".fsa")))

    test_paths = get_test_paths(TESTS_BASE_DIR)

    for test_path in track(test_paths, description="Checking tests..."):
        L.info(f"Generating LTS for {test_path}")

        # Generate the LTS for the current test
        genlts = timeit(cli.genlts, test_data, "Time to generate LTS")

        lts_path = genlts(str(test_path), cut_filename=substitute_fsa)
        lts_path = Path(lts_path).parent / (Path(lts_path).stem + "_ts5.dot")

        L.info(f"Parsing LTS {lts_path}...")
        start = time.perf_counter()
        lts = LTS.parse(str(lts_path))
        t = time.perf_counter() - start
        L.info(f"Parsing took {t}")

        log(t, "LTS parsing time")
        log(len(lts.configurations), "Number of nodes")
        log(len(lts.transitions), "Number of transitions")
        log(test_path.parent.parent.stem, "CUT")

        L.info(f"Checking projection test compliance...")
        checklts = timeit(cli.checklts, test_data, "Time to check compliance")
        compliant = checklts(str(lts_path), lts)

        if not compliant:
            fails = lts.get_failing_states()
            L.info(f"Failing configuration: {fails}")
            log(fails, "Failing configuration")
        else:
            log("None", "Failing configuration")

        log(str(lts_path), "Path")
        log(compliant, "Pass")

    log_path = gchor_path if substitute_fsa is None else Path(substitute_fsa)

    tdf = pd.DataFrame(test_data)

    # Save unnagregated, unfiltered info
    tdf.to_csv(log_path.with_suffix(".pertest.csv"), index=False)
    tdf.to_latex(log_path.with_suffix(".pertest.tex"), index=False)
    tdf.to_json(log_path.with_suffix(".pertest.json"))

    # Filter info for aggregation if the experiment
    # pertains a specific participant
    tdf = tdf[tdf["CUT"] == substitute_name] if substitute_fsa is not None else tdf

    summary = {
        "Number of tests": len(tdf),
        "Total time for LTS generation": tdf["Time to generate LTS"].sum(),
        "Total time for compliance check": tdf["Time to check compliance"].sum(),
        "CUT": substitute_name,
        "Failed tests": len(tdf) - tdf["Pass"].sum(),
        "Average time for LTS generation": tdf["Time to generate LTS"].sum() / len(tdf),
        "Average time for compliance check": tdf["Time to check compliance"].sum()
        / len(tdf),
    }

    summary.update(summary_data)

    gdf = pd.DataFrame(summary, index=[0])
    gdf.to_csv(log_path.with_suffix(".summary.csv"), index=False)
    gdf.to_latex(log_path.with_suffix(".summary.tex"), index=False)
    gdf.to_json(log_path.with_suffix(".summary.json"))


def experiment_0():
    """
    Just a small choreography for sanity checking.
    """
    run_experiment("scripts/jlamp2021/ATM/atm_simple.gc")


def experiment_1_0():
    """
    A larger choreography for the ATM example.
    """
    run_experiment("scripts/jlamp2021/ATM/atm_full.gc")


def experiment_1_1():
    """
    An example where the quit message from the ATM to the Bank is incorrectly removed.
    """
    run_experiment(
        gchor="scripts/jlamp2021/ATM/atm_full.gc",
        substitute_fsa="scripts/jlamp2021/ATM/fsa/atm_full_01_no_quit_ATM2Bank.fsa",
    )


def experiment_1_2():
    """
    An example where the ATM does not support the checkBalance message.
    """
    run_experiment(
        gchor="scripts/jlamp2021/ATM/atm_full.gc",
        substitute_fsa="scripts/jlamp2021/ATM/fsa/atm_full_02_no_checkBalanceATM.fsa",
    )


def experiment_1_3():
    """
    An example where the ATM does not interact with the bank when asked for a customer's balance.
    """
    run_experiment(
        gchor="scripts/jlamp2021/ATM/atm_full.gc",
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
    run_experiment(
        gchor="scripts/jlamp2021/shipping/shipping.sgg",
        substitute_fsa="scripts/jlamp2021/shipping/fsa/shipping_01_faulty_provider.fsa",
    )


def experiment_2_2():
    """
    Test P, where the order of cancel in the left thread of the right branch is swapped.
    (note: there should be no counterexample for this one)
    """
    run_experiment(
        gchor="scripts/jlamp2021/shipping/shipping.sgg",
        substitute_fsa="scripts/jlamp2021/shipping/fsa/shipping_02_provider_swaps_cancel_order_correctly.fsa",
    )


def experiment_2_3():
    """
    Test T, where you swap the input/output in the right thread or right branch
    """
    run_experiment(
        gchor="scripts/jlamp2021/shipping/shipping.sgg",
        substitute_fsa="scripts/jlamp2021/shipping/fsa/shipping_03_truck_swaps_cancel_order_wrongly.fsa",
    )


def experiment_2_4():
    """
    The client sends Shipments Details and places the order before ever receiving a quote.
    """
    run_experiment(
        gchor="scripts/jlamp2021/shipping/shipping.sgg",
        substitute_fsa="scripts/jlamp2021/shipping/fsa/shipping_04_client_details_order_before_quote.fsa",
    )


def main():
    experiment_0()
    experiment_1_0()
    experiment_1_1()
    experiment_1_2()
    experiment_1_3()
    experiment_2_0()
    experiment_2_1()
    experiment_2_2()
    experiment_2_3()
    experiment_2_4()


if __name__ == "__main__":
    setup()
    main()
