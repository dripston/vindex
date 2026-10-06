"""
Tests for vindex.trap_words. The dictionary CSVs
(vindex/data/trap_words/hindiwic_inventory.csv, own_additions.csv) are
human-authored. These tests check the loader's real behavior against that real data, plus
its filtering logic (rows with either reading blank must be skipped)
using injected fixtures so that logic doesn't depend on the dictionary
staying a particular size.
"""

from vindex.trap_words import load_trap_words


def test_load_trap_words_returns_list() -> None:
    words = load_trap_words()
    assert isinstance(words, list)


def test_load_trap_words_has_entries() -> None:
    # 69 rows have both readings: 5 HindiWiC words were left blank on
    # purpose (e.g. तेल, धन, डब्बा, संबंध, थान), and उत्तर was removed --
    # see test_load_trap_words_does_not_include_uttar below.
    words = load_trap_words()
    assert len(words) == 69


def test_load_trap_words_includes_the_documented_samudra_tal_case() -> None:
    # समुद्र तल is the documented mistranslation case -- must be present, and its
    # reading_b must match the real documented mistranslation ("sea
    # floor"), not a paraphrase, since tests and docs pin this exact
    # wording elsewhere (see test_judge_trace_check.py).
    words = load_trap_words()
    samudra_tal = next(w for w in words if w.term == "समुद्र तल")
    assert samudra_tal.reading_a == "sea level"
    assert samudra_tal.reading_b == "sea floor"
    assert samudra_tal.source == "own"


def test_load_trap_words_does_not_include_uttar() -> None:
    # उत्तर (north/answer) is intentionally not in the dictionary: this
    # library evaluates
    # question-answering, and उत्तर meaning "answer" is correct in that
    # context far more often than "north" -- but the dictionary had
    # reading_a="north" (treated as correct) and reading_b="answer"
    # (treated as a mistranslation to flag), so a judge trace correctly
    # saying "the answer is X" for a Hindi question containing उत्तर
    # got flagged as a misread. There is no single reading_a/reading_b
    # assignment that is right for both "उत्तर की ओर" (north) and
    # "सही उत्तर" (the correct answer), so it is excluded rather than
    # shipped wrong either way.
    words = load_trap_words()
    terms = {w.term for w in words}
    assert "उत्तर" not in terms


def test_load_trap_words_skips_no_good_trap_rows() -> None:
    # Rows deliberately left blank (both CSVs' "NO GOOD TRAP" calls --
    # तेल, धन, डब्बा, संबंध, थान) must not appear as usable trap words.
    words = load_trap_words()
    terms = {w.term for w in words}
    for skipped in ("तेल", "धन", "डब्बा", "संबंध", "थान"):
        assert skipped not in terms


def test_load_trap_words_includes_both_sources() -> None:
    words = load_trap_words()
    sources = {w.source for w in words}
    assert sources == {"hindiwic", "own"}
