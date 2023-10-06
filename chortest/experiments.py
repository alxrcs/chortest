import json
import logging
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from logging import (FileHandler, Formatter, StreamHandler, basicConfig,
                     getLogger)
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

import pandas as pd
from rich.logging import RichHandler
from rich.progress import track
from typer import Typer

import chortest.__main__ as cli
from chortest.cfsm import CommunicatingSystem
from chortest.common import Participant
from chortest.gchor import GChor
from chortest.lts import LTS
from chortest.mutations import LocalMutator
from chortest.parsing import Parsers
from chortest.utils import LangJSONEncoder, TimeoutError, fail_unless, timeout

app = Typer(add_completion=False, add_help_option=True, no_args_is_help=True)
L = getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

STATS_KILLED_MUTANTS = "killed"
STATS_SURVIVING_MUTANTS = "survived"

timestr = time.strftime("%Y%m%d-%H%M%S")
DEBUG = True


def setup(seed=1):
    sys.setrecursionlimit(2000)  # Default is ~900

    random.seed = seed

    rich_handler = RichHandler()
    rich_handler.setFormatter(Formatter("%(message)s"))
    poor_handler = StreamHandler()

    LOG_FILENAME = f"logs/chortest-{timestr}.log"
    log_file_handler = FileHandler(LOG_FILENAME)

    basicConfig(
        level="DEBUG",
        datefmt="[%X]",
        handlers=[poor_handler if DEBUG else rich_handler, log_file_handler],
    )


def timeit(func):
    def inner(*args, **kwargs):
        t1 = time.process_time()
        f = func(*args, **kwargs)
        t2 = time.process_time()
        return f, t2 - t1

    return inner


def create_experiment_folder(gc_filename: str, part_name: str, split: bool = False):
    gc_path = Path(gc_filename)
    split_text = "_with_split" if split else ""
    output_path: Path = (
        gc_path.parent / "results_mut" / f"ex_{gc_path.stem}{split_text}_{part_name}"
    )
    os.makedirs(str(output_path), exist_ok=True)
    shutil.copy(gc_filename, output_path)

    return output_path


# region logging info declarations
@dataclass
class ExperimentStats:
    input_gc_path: str
    participant_to_test: str
    total_tests: int
    total_mutants: int
    default_oracle_survivors: List[str]
    default_oracle_survivor_count: int
    restricted_lang_oracle_survivors: int
    actually_equivalent_mutants: int
    time_to_project: float
    time_to_generate_original_lts: float
    max_lts_gen_timeout: float
    time_to_gentests: float
    original_lts_conf_count: int
    original_lts_tr_count: int
    total_experiment_time: float
    error_count: int
    split: bool


@dataclass
class MutantStats:
    mutant_path: str
    cut: str
    first_failed_test_path: str
    time_to_first_fail: float
    mutation_type: str
    changed_node_id: int  # TODO: Change to str later (line and column)
    time_to_genlts: float
    time_to_checklts: float
    mutant_lts_node_count: int
    mutant_lts_transition_count: int
    mutant_lang_is_subset: bool
    mutant_lang_is_superset: bool  # TODO: Remove this later
    restricted_mutant_lang_is_subset: bool
    restricted_mutant_lang_is_superset: bool  # TODO: Remove, we don't care about this
    # number_of_failed_tests: int # TODO: Implement an option to run all tests
    # total_time_to_test: float # TODO: Implement an option to run all tests


def save_stats(json_stats, filename):
    Path(filename).write_text(
        json.dumps(json_stats, indent=4, sort_keys=True, cls=LangJSONEncoder)
    )


# def save_stats(
#     ex_stats: ExperimentStats,
#     mutant_stats: List[MutantStats],
#     langs: Tuple[Set, ...],
#     mut_langs: List[Tuple[Set, ...]],
#     ex_stats_filename: str,
#     mut_stats_filename: str,
# ):
#     Path(ex_stats_filename).write_text(
#         json.dumps(asdict(ex_stats), indent=4, sort_keys=True)
#     )
#     pd.DataFrame(mutant_stats).to_csv(mut_stats_filename)

#     lang, lang_restricted = langs
#     L.info(f"Saving original projections' language...")

#     Path(ex_stats.input_gc_path).with_suffix(".lang.json").write_text(
#         json.dumps(
#             [
#                 {
#                     "lang": list(lang),
#                     "lang_restricted": list(lang_restricted),
#                 }
#             ],
#             cls=LangJSONEncoder,
#             indent=4,
#         )
#     )

#     for i, (mut_lang, mut_lang_restricted) in enumerate(mut_langs):
#         if mut_lang and len(mut_lang) > 1000:
#             L.info(f"Skipping language dump for mutant {i} (language too large)")
#             continue

#         L.info(f"Saving language for mutant {i}...")
#         Path(mutant_stats[i].mutant_path).with_suffix(".lang.json").write_text(
#             json.dumps(
#                 [
#                     {
#                         "mut_lang": list(mut_lang) if mut_lang else None,
#                         "mut_lang_restricted": list(mut_lang_restricted)
#                         if mut_lang_restricted
#                         else None,
#                     }
#                 ],
#                 cls=LangJSONEncoder,
#                 indent=4,
#             )
#         )


def get_language_relations(lang, mut_lang, lang_restricted, mut_lang_restricted):
    mutant_lang_is_subset, mutant_lang_is_superset = (
        (mut_lang.issubset(lang), mut_lang.issuperset(lang))
        if lang and mut_lang
        else (None, None)
    )
    restricted_mutant_lang_is_subset, restricted_mutant_lang_is_superset = (
        (
            mut_lang_restricted.issubset(lang_restricted),
            mut_lang_restricted.issuperset(lang_restricted),
        )
        if mut_lang_restricted and lang_restricted
        else (None, None)
    )
    return (
        mutant_lang_is_subset,
        mutant_lang_is_superset,
        restricted_mutant_lang_is_subset,
        restricted_mutant_lang_is_superset,
    )


# endregion

# region mutation_local

@app.command()
def mutation_local_ex():
    setup()

    number_of_experiments = 1
    chor_path = Path('chortest/chortest/examples/gchors/ATM/atm_fixed.gc')

    for i in range(number_of_experiments):
        L.info(f"Experiment {i}")
        stats = mutate_locally(
            str(chor_path),
        )

        # save stats
        save_stats(stats, chor_path.parent / 'results_mut' / f"{chor_path.stem}_stats_{i}.json")


def mutate_locally(
    gc_filename: str, 
    part_name: Optional[str] = None, 
    ) -> dict[str, Any]:

    setup()
    fail_unless(os.path.exists(gc_filename), f"Cannot open {gc_filename}")

    # Init stats
    error_count = 0
    ex_start_time = time.process_time()

    # Decorate functions to time them
    timed_project = timeit(cli.project)
    timed_gentests = timeit(cli.gentests)
    timed_genlts = timeit(cli.genlts)
    timed_checklts = timeit(cli.checklts)

    # Parse the original GChoreography 
    gc: GChor = Parsers.parseGC(gc_filename)
    gc_path = Path(gc_filename)

    participants: list[str] = list(map(str,gc.participants())) if part_name is None else [part_name]
    assert len(participants) > 0, "No participants found."

    surviving_mutants = []

    ex_stats = {}
    
    for part_name in participants:
        # Create output folder
        output_folder_path = create_experiment_folder(gc_filename, part_name)
        
        # Check if the participant exists
        fail_unless(part_name in gc.participants(), f"({part_name}) not found.")

        # Project the G-chor and parse the projection
        projections_path, time_to_project = timed_project(
            str(gc_filename), 
            str(output_folder_path)
        )

        # Parse the projection
        source_cs: CommunicatingSystem = Parsers.parseFSA(projections_path)

        # Generate tests from the projections (for the given participant)
        tests_path, time_to_gentests = timed_gentests(
            projections_path, participant=part_name
        )

        # all_tests = list(tests_path.glob(f"{part_name}/test_*/*.fsa"))
        all_tests = tests_path
        all_mutant_stats = []
        surviving_mutants = []

        # For each possible mutant...
        for i, (cs, mut_info) in enumerate(
            LocalMutator.mutate_systematically(source_cs, Participant(part_name))
        ):

            # Write the current mutant to disk
            mutant_filename = str(gc_path.parent / "mutants" / f"{gc_path.stem}_mut_{part_name}_{i}.fsa")
            cs.to_fsa(mutant_filename, part=Participant(part_name), output_oracle=False)
            L.info(f"Mutant saved to {mutant_filename}")

            mutant_check_start_t = time.process_time()
            first_failed_test_path = ""
            time_to_first_fail = 0.0

            # Run the previously generated tests with the current mutant
            (   first_failed_test_path,
                time_to_first_fail,
                survived, 
                time_to_genlts,
                time_to_checklts
            ) = test_mutant(timed_genlts, timed_checklts, all_tests, i, mutant_filename, mutant_check_start_t, surviving_mutants, part_name)

            mutant_stats = {
                "mutant_path": mutant_filename,
                "mutant_part": part_name,
                "mutant_id": i,
                "participant_to_mutate": part_name,
                "survived": survived,
                "first_failed_test": first_failed_test_path,
                "time_to_genlts": time_to_genlts,
                "time_to_checklts": time_to_checklts,
                "time_to_first_fail": time_to_first_fail,
                "total_time_to_check": time.process_time() - mutant_check_start_t,
                "mutant_change": mut_info,
            }

            all_mutant_stats.append(mutant_stats)

        ex_stats[part_name] = {
            "input_gc_path": gc_filename,
            "total_tests": len(all_tests),
            "total_mutants": len(all_mutant_stats),
            "time_to_project": time_to_project,
            "time_to_gentests": time_to_gentests,
            "total_experiment_time": time.process_time() - ex_start_time,
            "error_count": error_count,
            "split": False,
            "mutants": all_mutant_stats,
            "surviving_mutants_count": len(surviving_mutants),
        }

    return ex_stats

def test_mutant(timed_genlts, timed_checklts, all_tests, i, mutant_filename, mutant_check_start_t, surviving_mutants, part_name):
    survived = False
    time_to_genlts = 0.0
    time_to_checklts = 0.0

    for test_fsa in track(all_tests):
        L.info(f"checking mutant #{i+1} with {test_fsa}")

        # Generate the labeled transition system and check it
        lts_path, time_to_genlts = timed_genlts(
                test_fsa, cut_filename=str(mutant_filename)
            )
        test_ok, time_to_checklts = timed_checklts(str(lts_path), part_name, oracle_filename=None, parsed_lts=None)
        if not test_ok:
            first_failed_test_path = test_fsa
            time_to_first_fail = time.process_time() - mutant_check_start_t
            break

    else:
        L.warn("Mutant survived!")
        surviving_mutants.append(mutant_filename)
        first_failed_test_path = "None"
        time_to_first_fail = 0
        survived = True
    return first_failed_test_path,time_to_first_fail,survived,time_to_genlts,time_to_checklts


# endregion


@app.command()
def mutation_global_ex(
    gc_filename: str,
    part_name: Optional[str] = None,
    output_folder: Optional[str] = None,
    stop_at_first_fail: bool = True,
    split: bool = False,
    timeout_secs: int = 40,
    unfold_n: int = 2,
):
    """
    Workflow:
    # TODO: Update workflow
    1. Split the original GC
        2. Project each of the split GCs and use those as the test suite (with or w/o splitting locally?)
            - w/o, for contrast with the local method and to compare with larger tests
    3. Mutate (either the original GC or the simpler split GC?)
        - The original GC (otherwise there'd be a lot of redundant mutants) -- is this true?
        4. Project the mutant GCs
        5. Generate and check the labeled transition system of the mutated CFSM with the original split projections
    """
    setup()
    fail_unless(os.path.exists(gc_filename), f"Cannot open {gc_filename}")

    timed_project = timeit(cli.project)
    timed_gentests = timeit(cli.gentests)
    timed_genlts = timeit(cli.genlts)
    timed_checklts = timeit(cli.checklts)

    # Remove loops from the original GC
    # unfolded_gc_filename = cli.unfold(gc_filename=gc_filename, n=unfold_n)
    # Parse and project the unfolded GChoreography
    # gc: GChor = Parsers.parseGC(str(unfolded_gc_filename))
    # gc_name = Path(unfolded_gc_filename).stem

    gc: GChor = Parsers.parseGC(gc_filename)
    gc_name = Path(gc_filename).stem

    all_participants = list(gc.participants())
    parts = [part_name] if part_name else all_participants

    if part_name:
        fail_unless(
            part_name in all_participants, f"Participant ({part_name}) not found."
        )
    else:
        L.info(f"Participant not specified, running for all participants")

    for part_name in parts:
        output_folder_path = create_experiment_folder(
            str(unfolded_gc_filename), part_name, split
        )

        mutant_stats: List[MutantStats] = []
        skipped_lang_count = 0
        ex_start_time = time.process_time()

        # Project the original GC
        original_projections_p, time_to_project = timed_project(
            unfolded_gc_filename, output_folder_path
        )

        # Load the projections and generate their language.
        original_fsa = Parsers.parseFSA(original_projections_p)
        lts_filename, original_genlts_time = timed_genlts(original_projections_p)
        original_lts: LTS = Parsers.parseDOT(lts_filename)
        try:
            language_with_timeout = timeout(timeout_secs)(LTS.language)
            lang, lang_restricted = language_with_timeout(original_lts, part_name)
        except TimeoutError:
            L.error("Language generation for original projections timed out")
            skipped_lang_count += 1
            lang, lang_restricted = None, None

        time_to_project, tests = generate_tests_from_global(
            part_name, split, gc, gc_name, output_folder_path, original_projections_p, timed_project
        )

        mutants_folder_path = output_folder_path / f"mutants_{part_name}"
        os.makedirs(mutants_folder_path, exist_ok=True)

        mut_langs: list = []

        total_mutants = 0
        surviving_mutants = []
        # 3. Mutate the original GC
        for i, mut_info in enumerate(gc.mutate(Participant(part_name))):
            mut_lang: Optional[Set] = set()
            mut_lang_restricted: Optional[Set] = set()
            mut_gc, mut_type, mut_target_id = (
                mut_info.mutant,
                mut_info.mutation_type,
                mut_info.target_id,
            )

            # setup a few counters
            mutant_check_start_t = time.process_time()
            first_failed_test_path, time_to_first_fail = "", 0.0

            # write the current mutant to disk
            mutant_path = mutants_folder_path / f"{gc_name}_mut_{i}.gc"
            mutant_path.write_text(str(mut_gc))

            # 4. Project the current mutant and get the wrong projection to check
            mutant_fsa_path, _ = timed_project(str(mutant_path))
            mutant_fsa: CommunicatingSystem = Parsers.parseFSA(mutant_fsa_path)
            mutant_machine = mutant_fsa.machines.get(Participant(part_name), None)

            if not mutant_machine:
                # This might mean the mutant was completely removed.
                # We'll ignore this case and continue with other mutants
                continue

            total_mutants += 1
            assert len(tests) > 0, "No tests found for the original GC"
            # 4. Try running the mutants on the test suite formed previously
            for j, test_fsa_path in enumerate(tests):

                L.info(f"Checking mutant #{i+1} with {test_fsa_path}")
                test_fsa_mut_path = merge_test_and_mutant(
                    part_name,
                    gc_name,
                    mutants_folder_path,
                    i,
                    mutant_machine,
                    j,
                    test_fsa_path,
                )
                L.info(f"Saved mutant to {test_fsa_mut_path}")

                # 5. Generate the labeled transition system and check it
                lts_path, time_to_genlts = timed_genlts(test_fsa_mut_path)
                mutant_lts: LTS = Parsers.parseDOT(lts_path)
                L.info("Generating language for current mutant...")

                # mut_lang, mut_lang_restricted = mutant_lts.language(part_name)
                try:
                    language_with_timeout = timeout(timeout_secs)(LTS.language)
                    # lang, lang_restricted = original_lts.language(part_name)
                    mut_lang, mut_lang_restricted = language_with_timeout(
                        mutant_lts, part_name
                    )
                except TimeoutError:
                    L.error(f"Language generation for mutant LTS {lts_path} timed out")
                    skipped_lang_count += 1
                    mut_lang, mut_lang_restricted = None, None

                L.info("Language for mutant generated successfully.")

                mut_langs.append((mut_lang, mut_lang_restricted))

                test_ok, time_to_checklts = timed_checklts(
                    str(lts_path),
                    oracle_filename=test_fsa_mut_path.with_suffix(".fsa.oracle.yaml"),
                    part_name=part_name,
                    parsed_lts=None,
                    use_exit_codes=False,
                )

                if not test_ok:
                    first_failed_test_path = str(test_fsa_path)
                    time_to_first_fail = time.process_time() - mutant_check_start_t
                    break
            else:
                L.warning(f"Mutant {mutant_fsa_path} survived!")
                surviving_mutants.append(str(mutant_fsa_path))
                first_failed_test_path = "None"
                time_to_first_fail = 0

            (
                mutant_lang_is_subset,
                mutant_lang_is_superset,
                restricted_mutant_lang_is_subset,
                restricted_mutant_lang_is_superset,
            ) = get_language_relations(
                lang, mut_lang, lang_restricted, mut_lang_restricted
            )

            mutant_stats.append(
                MutantStats(
                    mutant_path=str(mutant_path),
                    cut=part_name,
                    first_failed_test_path=first_failed_test_path,
                    time_to_first_fail=time_to_first_fail,
                    mutation_type=mut_type,
                    changed_node_id=mut_target_id,
                    mutant_lts_node_count=len(mutant_lts.configurations) if mutant_lts else -1,
                    mutant_lts_transition_count=len(mutant_lts.transitions),
                    time_to_genlts=time_to_genlts,
                    time_to_checklts=time_to_checklts,
                    # Remember that this makes sense only if there is no splitting
                    mutant_lang_is_subset=mutant_lang_is_subset,
                    mutant_lang_is_superset=mutant_lang_is_superset,
                    restricted_mutant_lang_is_subset=restricted_mutant_lang_is_subset,
                    restricted_mutant_lang_is_superset=restricted_mutant_lang_is_superset,
                )
            )

        non_detectable = sum(
            x.restricted_mutant_lang_is_subset or 0 for x in mutant_stats
        )
        equivalent_langs = sum(
            (x.mutant_lang_is_subset and x.mutant_lang_is_superset) or 0
            for x in mutant_stats
        )

        ex_stats = ExperimentStats(
            input_gc_path=str(Path(gc_filename)),
            participant_to_test=part_name,
            total_tests=len(tests),
            total_mutants=total_mutants,
            default_oracle_survivors=surviving_mutants,
            default_oracle_survivor_count=len(surviving_mutants),
            restricted_lang_oracle_survivors=non_detectable,
            actually_equivalent_mutants=equivalent_langs,
            time_to_project=time_to_project,
            time_to_generate_original_lts=original_genlts_time,
            max_lts_gen_timeout=timeout_secs,
            time_to_gentests=0,
            original_lts_conf_count=len(original_lts.configurations),
            original_lts_tr_count=len(original_lts.transitions),
            total_experiment_time=time.process_time() - ex_start_time,
            error_count=skipped_lang_count,
            split=split,
        )

        L.info(f"Original projections: {str(original_projections_p)}")
        if surviving_mutants:
            L.warning("List of surviving mutants: ")
            for s_mut in surviving_mutants:
                L.warning(s_mut)

        save_stats(
            
        )

        L.info(f"Using {part_name} as CUT.")

        assert len(mutant_stats) == total_mutants, "mutant counts do not match"

        if len(mutant_stats):
            L.info(
                f"{len(surviving_mutants)}/{total_mutants} survive with the default oracle."
            )
        else:
            L.info(
                f"No mutants available for this choreography when testing participant {part_name}."
            )

        L.info(
            f"{non_detectable}/{total_mutants} survive with the restricted language subsetting oracle."
        )
        L.info(f"{equivalent_langs}/{total_mutants} mutants are completely equivalent.")

        if skipped_lang_count:
            L.warning(f"Some languages were not generated, check the logs.")


def merge_test_and_mutant(
    part_name, gc_name, mutants_folder_path, i, mutant_machine, j, test_fsa_path
):
    test_fsa = Parsers.parseFSA(str(test_fsa_path))

    # Merge test and mutant, and save to disk
    test_fsa.machines[Participant(part_name)] = mutant_machine
    test_fsa_mut_path = mutants_folder_path / f"{gc_name}_mut_{i}_test_{j}.fsa"
    test_fsa.to_fsa(str(test_fsa_mut_path))
    return test_fsa_mut_path


def generate_tests_from_global(
    part_name, split, gc, gc_name, output_folder_path, original_projections_p, project_fn
):
    tests: List[Path] = []

    # Split the GChoreography into smaller ones (optional)
    total_project_time = 0
    if split:
        for i, split_gc in enumerate(gc.paths(part_name)):
            split_gc_path = output_folder_path / f"{gc_name}_path_{i}.gc"
            split_gc_path.write_text(str(split_gc))

            # Project the G-chor and use them as the test suite (w/o splitting locally)
            projections_path, time_to_project = project_fn(
                str(split_gc_path), output_folder_path
            )
            total_project_time += time_to_project

            tests.append(projections_path)
    else:
        # If no split is required, we'll just use the original projections
        tests.append(original_projections_p)
        time_to_project = 0

    return total_project_time, tests

import pytest


@app.command()
def codegen():
    """
    Run the codegen experiments.
    """

    # max_examples_l = [1, 2, 5, 10, 50, 100, 500, 1000]
    # max_examples_l = [1, 2, 5, 10, 25, 50, 75, 100]
    # max_examples_l = [1, 2, 5, 10, 25, 50, 75, 100]
    # max_examples_l = [1] + list(range(10, 100, 10)) + [100]
    max_examples_l = [100]

    for max_examples in max_examples_l:
        for run_number in range(1, 31):
            # Specify the test file path and any additional arguments
            arguments = [
                'chortest/chortest/examples/gchors/ATM/atm_fixed_code/atm_fixed_tests.py', 
                '-v', 
                '-s',
                "--hypothesis-show-statistics",
                "--hypothesis-verbosity",
                "verbose",
                # "--custom-coverage",
                "--max-examples",
                str(max_examples),
                "--run-number",
                str(run_number),
            ]

            # Run pytest with the specified arguments
            exit_code = pytest.main(arguments)

            # Exit with the exit code from pytest so that CI fails correctly
            if exit_code != 0:
                sys.exit(exit_code)


if __name__ == "__main__":
    app()
