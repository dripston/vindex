"""Trap-word dictionary used by judge_trace_check.

A trap word is a Hindi term with two plausible readings: ``reading_a`` is
the correct sense in context and ``reading_b`` is the likely
mistranslation (e.g. समुद्र तल: "sea level" vs "sea floor"). Both readings
are human-authored, not inferred.

The dictionary is shipped as package data in ``vindex/data/trap_words/``:

- ``hindiwic_inventory.csv``: polysemous Hindi nouns from the HindiWiC
  dataset (word list only, no context sentences). See
  ``vindex/data/NOTICE.md`` for attribution and licensing.
- ``own_additions.csv``: hand-authored entries (misleading compounds,
  tense-ambiguous time adverbs, fractional numbers, Indian large-number
  words).

Only rows with both readings filled in are loaded, so the dictionary
contains exactly the human-reviewed entries (currently 69).
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
    """One ambiguous Hindi term and its two readings.

    Attributes:
        term: The Hindi source term, e.g. ``"समुद्र तल"``.
        reading_a: The correct sense in context, e.g. ``"sea level"``.
        reading_b: The plausible mistranslation, e.g. ``"sea floor"``.
        source: ``"hindiwic"`` or ``"own"``, for provenance.
    """

    term: str
    reading_a: str
    reading_b: str
    source: str


def _load_csv_rows(filename: str) -> list[dict[str, str]]:
    """Load rows from a CSV in ``vindex/data/trap_words/``.

    Returns an empty list if the file is missing.
    """
    ref = resources.files(_PACKAGE_DATA).joinpath(filename)
    if not ref.is_file():
        return []
    with resources.as_file(ref) as path, open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_trap_words() -> list[TrapWord]:
    """Load every human-reviewed trap word from both dictionary CSVs.

    Rows missing either reading are skipped.

    Returns:
        The trap words, HindiWiC entries first.
    """
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
