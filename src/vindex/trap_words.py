"""
Trap-word dictionary loader for judge_trace_check (Milestone 6.1).

A "trap word" is a Hindi term with two readings that could plausibly
be confused: reading_a is the correct sense in context, reading_b is
the mistranslation risk (e.g. समुद्र तल: reading_a "sea level",
reading_b "sea floor" -- the exact Phase 0 failure this project has
direct evidence for; see experiments/FINDINGS.md and
src/vindex/judge_rubric.py). Loaded from two CSVs at
data/trap_words/: hindiwic_inventory.csv (60 polysemous Hindi nouns
from the HindiWiC dataset -- word list only, no context sentences; see
data/trap_words/NOTICE.md for the licensing rationale) and
own_additions.csv (hand-authored: misleading compounds, tense-flipping
time adverbs, fractional numbers, Indian large-number words).

BOTH CSVs ship with suggested_reading_a/suggested_reading_b BLANK.
Filling them in is explicitly human work, not something this module or
an agent does: BUILD_PLAN.md 6.1 says so directly ("Fill in reading_a
/ reading_b by hand -- about an hour, and it's your work, not your
agent's"). This module loads whatever rows currently have both
readings filled in and ignores the rest -- so the trap-word set is
exactly as large as the human-reviewed portion of the dictionary, no
more, no less. As of this module's writing, both CSVs are entirely
blank in those columns, so load_trap_words() returns an empty list
until a human fills some in. This is the correct, honest behavior, not
a bug: see judge_trace_check.py's module docstring for why the claim
this dictionary backs ("detects mistranslation of N known ambiguous
terms") must never overstate N.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "trap_words",
)
HINDIWIC_INVENTORY_PATH = os.path.join(_DATA_DIR, "hindiwic_inventory.csv")
OWN_ADDITIONS_PATH = os.path.join(_DATA_DIR, "own_additions.csv")


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


def _load_csv_rows(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_trap_words() -> list[TrapWord]:
    """Load every trap word with BOTH reading_a and reading_b filled in
    across both CSVs. Rows with either reading blank are skipped --
    they are not yet human-reviewed and must not be treated as usable
    trap words. Returns [] if neither CSV has any completed rows yet."""
    words: list[TrapWord] = []

    for row in _load_csv_rows(HINDIWIC_INVENTORY_PATH):
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

    for row in _load_csv_rows(OWN_ADDITIONS_PATH):
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
