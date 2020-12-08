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

def get_test_paths(tests_dir) -> List[Path]:
    test_paths = []
    for root, dirs, files in os.walk(tests_dir):
        if not files:
            continue
        for f in files:
            if f.endswith('.fsa'):
                test_path = Path(root) / f
                test_paths.append(test_path)
    return test_paths
        
def setup_log():
    rich_handler = RichHandler()
    rich_handler.setFormatter(Formatter("%(message)s"))

    basicConfig(level="DEBUG", datefmt="[%X]", handlers=[rich_handler])

def run_experiment(
    gchor: Optional[str] = None,
    substitute_fsa: Optional[str] = None
    ):

    if gchor is None:
        gchor_path = Path(".") / "scripts" / "jlamp2021" / "ATM" / "atm_simple.gg"
    else:
        gchor_path = Path(gchor)

    BASE_DIR = gchor_path.parent
    GCHOR_FNAME = gchor_path.name

    cli.project(str(BASE_DIR / GCHOR_FNAME))
    cli.gentests(str(BASE_DIR / "fsa" / Path(GCHOR_FNAME).with_suffix(".fsa")))

    TESTS_BASE_DIR = BASE_DIR / "fsa" / (Path(GCHOR_FNAME).stem + "_tests")
    test_paths = get_test_paths(TESTS_BASE_DIR)

    for test_path in test_paths:

        if substitute_fsa:
            cli.combine_fsa(
                input_fsa_filename=str(test_path),
                replacement_fsa_filename=substitute_fsa,
                output_filename=str(test_path.with_suffix('.tmp'))
            )
            test_path = test_path.with_suffix('.tmp')

        L.info(f'Generating LTS for {test_path}')
        cli.genlts(str(test_path))

        lts_path = test_path.parent / (test_path.stem + '_ts5.dot')
        L.info(f'Checking projection test compliance...')
        cli.checklts(str(lts_path))


def main():
    run_experiment('scripts/jlamp2021/ATM/fsa/atm_full_no_quit_ATM2Bank.fsa')

if __name__ == "__main__":
    setup_log()
    main()
