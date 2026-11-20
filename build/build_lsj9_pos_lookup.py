#!/usr/bin/env python3
"""Build lsj9_pos_lookup.json from lsj9 headword POS data.

Reads lsj9_headword_pos.json (from the lsj9 repo) which maps headwords to
UPOS tags, then cross-references with ag_lookup.json to find all inflected
forms for those headwords. Only includes forms that are genuinely useful for
POS disambiguation (i.e., NOUN and ADJ entries, since these are the
grammar-derived categories from lsj9).

Output: data/lsj9_pos_lookup.json
Format: {form: {UPOS: headword}, ...}

LSJ9 keys a head divided at its stem as printed (ἀασι-φρονία), while the
lookup files the head joined (ἀασιφρονία, #137), so the headword POS keys
are joined before they are matched. A lookup row whose form or lemma still
has a hyphen or dash is not read: no token is spelled that way and no
citation form is (#149).

Usage:
    python build/build_lsj9_pos_lookup.py [--lsj9-dir DIR] [--ag-lookup PATH]
                                          [--output PATH]
"""

import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dilemma.citation import HEAD_HYPHENS, hyphenated_lemma  # noqa: E402

DATA_DIR = SCRIPT_DIR / "data"
AG_LOOKUP_PATH = DATA_DIR / "ag_lookup.json"
# LSJ9 exports: ``$DILEMMA_LSJ9_DIR`` (or ``--lsj9-dir``), else ~/Documents/lsj9.
LSJ9_DIR = Path(os.environ.get("DILEMMA_LSJ9_DIR")
                or Path.home() / "Documents" / "lsj9")
HEADWORD_POS_PATH = LSJ9_DIR / "lsj9_headword_pos.json"
OUTPUT_PATH = DATA_DIR / "lsj9_pos_lookup.json"


def _strip_diacritics(s: str) -> str:
    """Strip all combining diacritics for accent-free comparison."""
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if not unicodedata.combining(c))


def _strip_length_marks(s: str) -> str:
    """Strip combining breve (U+0306) and macron (U+0304)."""
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(c for c in nfd if ord(c) not in (0x0306, 0x0304)))


def joined_head(headword: str) -> str:
    """LSJ9's head with the hyphens of its stem division removed."""
    if not hyphenated_lemma(headword):
        return headword
    return unicodedata.normalize("NFC", "".join(
        char for char in headword if char not in HEAD_HYPHENS))


def join_headword_pos(headword_pos: dict) -> dict:
    """``{joined head: UPOS}``. Where a divided and an undivided key join to
    one spelling, the undivided key's POS stands; among divided keys, the
    first in the export's order."""
    joined: dict[str, str] = {}
    for head, upos in headword_pos.items():
        if not hyphenated_lemma(head):
            joined[head] = upos
    for head, upos in headword_pos.items():
        if hyphenated_lemma(head):
            joined.setdefault(joined_head(head), upos)
    return joined


def has_hyphen(text: str) -> bool:
    return any(char in HEAD_HYPHENS for char in text)


def build(ag_lookup: dict, headword_pos: dict) -> dict:
    """``{form: {UPOS: headword}}`` for the NOUN and ADJ heads of LSJ9."""
    # Only include NOUN and ADJ - these are the grammar-derived categories
    # that provide genuine disambiguation value. VERB entries from lsj9 are
    # heuristic-based (ending detection) and less reliable.
    target_pos = {"NOUN", "ADJ"}
    hw_pos_filtered = {hw: upos for hw, upos in join_headword_pos(headword_pos).items()
                       if upos in target_pos}
    pos_lookup: dict[str, dict[str, str]] = {}
    for form, lemma in ag_lookup.items():
        if has_hyphen(form) or hyphenated_lemma(lemma):
            continue
        lemma_clean = _strip_length_marks(lemma)
        upos = hw_pos_filtered.get(lemma_clean)
        if not upos:
            continue
        pos_lookup.setdefault(form, {})[upos] = lemma_clean
        plain = _strip_diacritics(form)
        if plain != form:
            pos_lookup.setdefault(plain, {})[upos] = lemma_clean
    return pos_lookup


def main(argv=None):
    global AG_LOOKUP_PATH, HEADWORD_POS_PATH, OUTPUT_PATH
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lsj9-dir", type=Path, default=None,
                        help="LSJ9 export directory (default: $DILEMMA_LSJ9_DIR, "
                             "else ~/Documents/lsj9)")
    parser.add_argument("--ag-lookup", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.lsj9_dir:
        HEADWORD_POS_PATH = args.lsj9_dir / "lsj9_headword_pos.json"
    AG_LOOKUP_PATH = args.ag_lookup or AG_LOOKUP_PATH
    OUTPUT_PATH = args.output or OUTPUT_PATH
    print("Building lsj9_pos_lookup.json")

    # Load headword -> UPOS mapping from lsj9
    if not HEADWORD_POS_PATH.exists():
        print(f"  Error: {HEADWORD_POS_PATH} not found")
        print("  Run build_exports.py in the lsj9 repo first.")
        return

    with open(HEADWORD_POS_PATH, encoding="utf-8") as f:
        headword_pos = json.load(f)
    print(f"  Loaded {len(headword_pos):,} headword POS entries")


    # Load ag_lookup (form -> headword)
    if not AG_LOOKUP_PATH.exists():
        print(f"  Error: {AG_LOOKUP_PATH} not found")
        return

    with open(AG_LOOKUP_PATH, encoding="utf-8") as f:
        ag_lookup = json.load(f)
    print(f"  Loaded {len(ag_lookup):,} ag_lookup entries")

    pos_lookup = build(ag_lookup, headword_pos)

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pos_lookup, f, ensure_ascii=False, separators=(",", ":"))

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"  Output: {len(pos_lookup):,} form entries ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
