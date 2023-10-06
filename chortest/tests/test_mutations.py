import pytest
from chortest.__main__ import genlts, project
from chortest.cfsm import CommunicatingSystem
from chortest.common import Participant
from chortest.gchor import GChor
from chortest.lts import LTS
from chortest.mutations import LocalMutator
from chortest.parsing import Parsers
from chortest.tests.test_cfsm import simple_atm_cs
from tempfile import NamedTemporaryFile


def test_mutate_systematically_locally(simple_atm_cs: CommunicatingSystem):
    atm = Participant("B")
    mutants = list(LocalMutator.mutate_systematically(simple_atm_cs, atm))
    print(list(mutants))


def test_mutate_globally_choice(shared_datadir):
    atm_simple: GChor = Parsers.parseFile(shared_datadir / "gchors/atm_simple.gc")
    atm = Participant("B")
    for gc in atm_simple.mutate(atm):
        print(gc)


def test_mutate_globally_nested_choice(shared_datadir):
    nested_choice: GChor = Parsers.parseFile(
        shared_datadir / "gchors/nested_choice.gc"
    )
    atm = Participant("B")
    for gc in nested_choice.mutate(atm):
        print(gc)


@pytest.mark.parametrize(
    "gc_filename, part",
    [
        ("gchors/nested_choice.gc", "B"),
        ("gchors/req_reply.gc", "B"),
        ("gchors/par_choice_longer.gc", "receiver"),
    ],
)
def test_mutate_globally_and_check_restricted_language_equivalence(gc_filename, part, shared_datadir):
    gc: GChor = Parsers.parseFile(shared_datadir / gc_filename)

    projection_filename = project(shared_datadir / gc_filename)
    original_fsa = Parsers.parseFSA(str(projection_filename))
    lts_filename = genlts(shared_datadir / projection_filename)
    lts = Parsers.parseDOT(lts_filename)
    lang, lang_restricted = lts.language(part)

    for gc_mut in gc.mutate(part):
        with NamedTemporaryFile(
            "wt", suffix=".gc", delete=False
        ) as temp_gc_file, NamedTemporaryFile(
            "wt", suffix=".fsa", delete=False
        ) as temp_fsa_file:
            temp_gc_file.write(str(gc_mut.mutant))
            temp_gc_file.flush()

            mutant_fsa = Parsers.parseFSA(project(temp_gc_file.name))

            # Switch the mutant in the original projections
            original_fsa.machines[part] = mutant_fsa.machines[part]
            original_fsa.to_fsa(temp_fsa_file.name)

            # Generate the LTS for the (mutant | projections) system
            lts: LTS = Parsers.parseDOT(genlts(temp_fsa_file.name))

            mut_lang, mut_lang_restricted = lts.language(part)

            print(mut_lang.issubset(lang))
            print(mut_lang_restricted.issubset(lang_restricted))
