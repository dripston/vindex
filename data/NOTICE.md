# NOTICE — data/

## `results_clean.json`

**PINNED. Never regenerate.** This is the frozen output of
`experiments/scripts/regenerate_dataset.py`'s pinned generation run
(the model, `openai/gpt-oss-20b` via Groq, is not deterministic even at
temperature 0 -- confirmed empirically, roughly a third of cases reword
on a re-run). Every downstream number, every hand-authored negative in
`experiments/testcases.py`, and every script in `experiments/scripts/`
that scores against this file assumes its content never changes.

29 of 30 cases are script-adherent. One case
(`father_of_nation_india__hi`) is marked `script_adherence_failure: true`
and excluded from scoring by every script that reads this file -- it is
kept in the file for the record, not deleted, per the "nothing deleted"
rule.

If the dataset genuinely needs to change, produce a new file
(`results_clean_v2.json` or similar) as a separately reviewed artifact.
Never overwrite this one in place.

## `trap_words/`

Extracted from the third-party HindiWiC dataset (gitignored local clone
at `data/external/HindiWiC/`, never committed) plus hand-authored
additions. See `trap_words/NOTICE.md` for licensing and citation --
that file covers what was and wasn't extracted from HindiWiC and why.
Earmarked as Milestone 5's mistranslation-dictionary seed.
