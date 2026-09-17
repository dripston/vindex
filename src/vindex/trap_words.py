"""
Trap-word dictionary loader for judge_trace_check (Milestone 6.1).

A "trap word" is a Hindi term with two readings that could plausibly
be confused: reading_a is the correct sense in context, reading_b is
the mistranslation risk (e.g. समुद्र तल: reading_a "sea level",
reading_b "sea floor" -- the exact Phase 0 failure this project has
direct evidence for; see experiments/FINDINGS.md and
src/vindex/judge_rubric.py). Loaded from two CSVs shipped as package
data at vindex/data/trap_words/ (see PACKAGING below):
hindiwic_inventory.csv (60 polysemous Hindi nouns from the HindiWiC
dataset -- word list only, no context sentences; see
data/trap_words/NOTICE.md at the repo root for the licensing
rationale) and own_additions.csv (hand-authored: misleading compounds,
tense-flipping time adverbs, fractional numbers, Indian large-number
words).

PACKAGING (fixed after a real bug): earlier versions computed the CSV
paths as three `dirname()` hops from `__file__`, pointing at a
repo-relative `data/trap_words/` directory. That only exists in a git
checkout -- a `pip install`'d copy of this package has no such
directory anywhere near it, so `load_trap_words()` silently returned
`[]` for every installed user, and `check_trace()` silently reported
`passed=True, score=1.0, dictionary_size=0` on every call: a green
check that checked nothing. Fixed by shipping the CSVs as real package
data under `vindex/data/trap_words/` (declared in pyproject.toml) and
loading them via `importlib.resources`, which works identically
whether running from source or from an installed wheel/sdist.

70 of the ~75 rows across both CSVs have suggested_reading_a/
suggested_reading_b filled in by a human (Milestone 6.1's own
instruction: "about an hour, and it's your work, not your agent's").
This module loads whatever rows currently have both readings filled
in and ignores the rest -- so the trap-word set is exactly as large as
the human-reviewed portion of the dictionary, no more, no less. See
judge_trace_check.py's module docstring for why the claim this
dictionary backs ("detects mistranslation of N known ambiguous
terms") must never overstate N.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib import resources

_PACKAGE_DATA = "vindex.data.trap_words"
_HINDIWIC_FILENAME = "hindiwic_inventory.csv"
_OWN_ADDITIONS_FILENAME = "own_additions.csv"


@dataclass(frozen=True, slots=True)
class TrapWord:
    """term       : the Hindi source term (e.g. "समुद्र तल").
    reading_a  : the correct sense in the intended context (e.g. "sea
                 level"). Human-authored, not inferred.
    reading_b  : the plausible mistranslation (e.g. "sea floor").
                 Human-authored, not inferred.
    source     : "hindiwic" or "own", for provenance/citation purposes.
    """

    term: str
    reading_a: str
    reading_b: str
    source: str


def _load_csv_rows(filename: str) -> list[dict[str, str]]:
    """Load a CSV shipped as package data under vindex/data/trap_words/
    -- works identically from an installed wheel/sdist or from source,
    unlike a repo-relative filesystem path (see this module's
    PACKAGING note)."""
    ref = resources.files(_PACKAGE_DATA).joinpath(filename)
    if not ref.is_file():
        return []
    with resources.as_file(ref) as path, open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_trap_words() -> list[TrapWord]:
    """Load every trap word with BOTH reading_a and reading_b filled in
    across both CSVs. Rows with either reading blank are skipped --
    they are not yet human-reviewed and must not be treated as usable
    trap words. Returns [] if neither CSV has any completed rows yet."""
    words: list[TrapWord] = []

    for row in _load_csv_rows(_HINDIWIC_FILENAME):
        reading_a = (row.get("suggested_reading_a") or "").strip()
        reading_b = (row.get("suggested_reading_b") or "").strip()
        if reading_a and reading_b:
            words.append(
                TrapWord(
                    term=row["target_word"],
                    reading_a=reading_a,
                    reading_b=reading_b,
                    source="hindiwic",
                )
            )

    for row in _load_csv_rows(_OWN_ADDITIONS_FILENAME):
        reading_a = (row.get("suggested_reading_a") or "").strip()
        reading_b = (row.get("suggested_reading_b") or "").strip()
        if reading_a and reading_b:
            words.append(
                TrapWord(
                    term=row["target_word"],
                    reading_a=reading_a,
                    reading_b=reading_b,
                    source="own",
                )
            )

    return words
