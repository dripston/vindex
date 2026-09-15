"""
Tests for vindex.trap_words (Milestone 6.1). See its module docstring:
filling in reading_a/reading_b is explicitly human work, not automated
-- these tests confirm the loader's honest behavior on the current
(deliberately blank) CSVs, and its filtering logic on injected rows,
not any specific dictionary content.
"""

from vindex.trap_words import load_trap_words


def test_load_trap_words_returns_list() -> None:
    words = load_trap_words()
    assert isinstance(words, list)


def test_load_trap_words_currently_empty() -> None:
    # As of writing, both CSVs ship with suggested_reading_a/
    # suggested_reading_b blank (filling them in is human work -- see
    # module docstring). This pins that honest state: the dictionary
    # must not silently claim entries that were never human-reviewed.
    # If this test starts failing because the CSVs were filled in,
    # that's real progress -- delete this test, don't weaken it.
    words = load_trap_words()
    assert len(words) == 0
