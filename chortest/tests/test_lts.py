from chortest.__main__ import genlts, project
from chortest.parsing import Parsers


def test_subset(shared_datadir):
    gc_a_filename = shared_datadir / "gchors/atm_simple.gc"
    gc_b_filename = shared_datadir / "gchors/atm_simple_insecure.gc"

    gc_a = Parsers.parseGC(gc_a_filename)
    gc_b = Parsers.parseGC(gc_b_filename)

    fsa_filename_a = project(gc_a_filename)
    fsa_filename_b = project(gc_b_filename)

    assert fsa_filename_a is not None
    assert fsa_filename_b is not None

    fsa_a = Parsers.parseFSA(str(fsa_filename_a))
    fsa_b = Parsers.parseFSA(str(fsa_filename_b))

    lts_a_filename = genlts(str(fsa_filename_a))
    lts_b_filename = genlts(str(fsa_filename_b))

    lts_a = Parsers.parseDOT(lts_a_filename)
    lts_b = Parsers.parseDOT(lts_b_filename)

    lang_a, _ = lts_a.language()
    lang_b, _ = lts_b.language()

    assert lang_a.issuperset(lang_b)
    assert lang_b.issubset(lang_a)
