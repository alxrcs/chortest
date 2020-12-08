import os
from typer.testing import CliRunner
from logging import getLogger, basicConfig
from sys import argv
from time import perf_counter

runner = CliRunner()

from chortools.__main__ import *

def get_test_paths(tests_dir):
    test_paths = []
    for root, dirs, files in os.walk(tests_dir):
        if not files:
            continue
        for f in files:
            if f.endswith('.fsa'):
                test_path = Path(root) / f
                test_paths.append(test_path)
    return test_paths
        

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

def setup_log():
    rich_handler = RichHandler()
    rich_handler.setFormatter(Formatter("%(message)s"))

    basicConfig(level="DEBUG", datefmt="[%X]", handlers=[rich_handler])

def main():
    BASE_DIR = Path(".") / "scripts" / "jlamp2021" / "ATM"
    GCHOR_FNAME = Path(argv[1] if len(argv) > 1 else "atm_simple.gg")

    project(str(BASE_DIR / GCHOR_FNAME))
    gentests(str(BASE_DIR / "fsa" / GCHOR_FNAME.with_suffix(".fsa")))

    TESTS_BASE_DIR = BASE_DIR / "fsa" / (GCHOR_FNAME.stem + "_tests")
    test_paths = get_test_paths(TESTS_BASE_DIR)

    for test_path in test_paths:
        L.info(f'Generating LTS for {test_path}')
        genlts(test_path)

        lts_path = test_path.parent / (test_path.stem + '_ts5.dot')
        L.info(f'Checking projection test compliance...')
        checklts(str(lts_path))



if __name__ == "__main__":
    setup_log()
    main()
