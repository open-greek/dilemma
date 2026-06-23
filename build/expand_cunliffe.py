#!/usr/bin/env python3
"""Expand Cunliffe Homeric Lexicon headwords into inflected forms.

Reads a Cunliffe lexicon export, extracts headwords with
grammar info, and expands via grc-decl/grc-conj templates. Merges
expanded forms into ag_lookup.json.

Also populates cunliffe_headwords.json for convention mapping.

Usage:
    python expand_cunliffe.py --analyze       # show stats
    python expand_cunliffe.py --expand        # expand and merge into ag_lookup
    python expand_cunliffe.py --headwords     # just populate cunliffe_headwords.json
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

# Import shared infrastructure from expand_lsj
from expand_lsj import (
    get_wtp, expand_noun, expand_verb, strip_length_marks, strip_diacritics,
    infer_genitive, load_wiktionary_genitives,
    DATA_DIR,
)

# Cunliffe Homeric lexicon export; set CUNLIFFE_JSON to your copy.
CUNLIFFE_PATH = Path(os.environ.get(
    "CUNLIFFE_JSON", Path.home() / "Documents" / "cunliffe" / "cunliffe.json"))
AG_LOOKUP = DATA_DIR / "ag_lookup.json"
HEADWORDS_OUT = DATA_DIR / "cunliffe_headwords.json"

_GK = r"\u0370-\u03FF\u1F00-\u1FFF"

# Gender from short definition or inline grammar
GENDER_MAP = {"ὁ": "m", "ἡ": "f", "τό": "n", "τὸ": "n"}

# Detect article at start of definition
_ARTICLE_START = re.compile(rf"^(ὁ|ἡ|τό|τὸ)\b")

# Detect adjective endings in definition
_ADJ_PATTERNS = re.compile(
    r"^(?:ον|ές|εῖα|η,?\s*ον|α,?\s*ον|ος,?\s*η,?\s*ον)\b"
)


def load_cunliffe() -> dict[str, dict]:
    """Load Cunliffe entries from the lexicon export."""
    if not CUNLIFFE_PATH.exists():
        print(f"Cunliffe not found at {CUNLIFFE_PATH}")
        sys.exit(1)

    with open(CUNLIFFE_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # Extract headword + grammar info
    entries = {}
    for hw, entry in data.items():
        # Clean headword: strip prefix markers
        hw_clean = hw.lstrip("*†¨ʽ\u0313\u0314 ")
        if not hw_clean or not re.match(rf"[{_GK}]", hw_clean):
            continue

        hw_clean = strip_length_marks(hw_clean)

        # Try to extract gender from the short definition
        short = entry.get("short", "")
        defs = entry.get("defs", [])

        gender = ""
        # Check if first def starts with article
        if defs:
            first_def = ""
            if isinstance(defs[0], dict):
                first_def = defs[0].get("t", "")
            elif isinstance(defs[0], str):
                first_def = defs[0]

            m = _ARTICLE_START.match(first_def.strip())
            if m:
                gender = GENDER_MAP.get(m.group(1), "")

        entries[hw_clean] = {
            "headword": hw_clean,
            "gender": gender,
            "short": short,
        }

    return entries


def analyze(entries: dict, ag_lookup: dict):
    """Show coverage stats."""
    in_lookup = sum(1 for hw in entries if hw in ag_lookup)
    missing = sum(1 for hw in entries if hw not in ag_lookup)
    with_gender = sum(1 for e in entries.values() if e["gender"])

    print(f"Cunliffe entries: {len(entries)}")
    print(f"  In AG lookup: {in_lookup} ({100*in_lookup/len(entries):.0f}%)")
    print(f"  Missing: {missing} ({100*missing/len(entries):.0f}%)")
    print(f"  With gender: {with_gender}")

    # Categorize missing by type
    nouns = verbs = adjs = other = 0
    for hw, e in entries.items():
        if hw in ag_lookup:
            continue
        if e["gender"]:
            nouns += 1
        elif hw.endswith("ω") or hw.endswith("μι") or hw.endswith("μαι"):
            verbs += 1
        elif hw.endswith("ος") or hw.endswith("ής") or hw.endswith("ύς"):
            adjs += 1
        else:
            other += 1

    print(f"\n  Missing breakdown:")
    print(f"    Nouns (with gender): {nouns}")
    print(f"    Verbs (-ω/-μι/-μαι): {verbs}")
    print(f"    Adj-like (-ος/-ής/-ύς): {adjs}")
    print(f"    Other: {other}")


def expand_entries(entries: dict, ag_lookup: dict):
    """Expand missing Cunliffe headwords and merge into ag_lookup."""
    wtp = get_wtp()
    wikt_genitives = load_wiktionary_genitives()

    added = 0
    expanded_nouns = 0
    expanded_verbs = 0
    failed = 0

    for hw, e in entries.items():
        if hw in ag_lookup:
            continue

        # Always add the headword itself as a self-map
        ag_lookup[hw] = hw
        added += 1

        gender = e["gender"]

        if gender:
            # Noun: try to expand
            genitive = infer_genitive(hw, gender, wikt_genitives)
            forms, err = expand_noun(wtp, hw, gender, genitive)
            if forms:
                for form in forms:
                    form_clean = strip_length_marks(form)
                    if form_clean not in ag_lookup:
                        ag_lookup[form_clean] = hw
                        added += 1
                expanded_nouns += 1
            else:
                failed += 1

        elif hw.endswith("ω") or hw.endswith("μι") or hw.endswith("μαι"):
            # Verb: try to expand
            forms, err = expand_verb(wtp, hw)
            if forms:
                for form in forms:
                    form_clean = strip_length_marks(form)
                    if form_clean not in ag_lookup:
                        ag_lookup[form_clean] = hw
                        added += 1
                expanded_verbs += 1
            else:
                failed += 1

    print(f"\nExpansion results:")
    print(f"  Nouns expanded: {expanded_nouns}")
    print(f"  Verbs expanded: {expanded_verbs}")
    print(f"  Failed: {failed}")
    print(f"  Total new forms added: {added}")

    return ag_lookup


def save_headwords(entries: dict):
    """Save Cunliffe headword list for convention mapping."""
    headwords = sorted(entries.keys())
    with open(HEADWORDS_OUT, "w", encoding="utf-8") as f:
        json.dump(headwords, f, ensure_ascii=False, indent=1)
    print(f"Saved {len(headwords)} headwords to {HEADWORDS_OUT}")


def main():
    parser = argparse.ArgumentParser(
        description="Expand Cunliffe headwords into Dilemma's AG lookup")
    parser.add_argument("--analyze", action="store_true",
                        help="Show coverage stats only")
    parser.add_argument("--expand", action="store_true",
                        help="Expand missing headwords and merge into ag_lookup")
    parser.add_argument("--headwords", action="store_true",
                        help="Populate cunliffe_headwords.json only")
    args = parser.parse_args()

    entries = load_cunliffe()

    if args.headwords or args.analyze or not args.expand:
        save_headwords(entries)

    # Load AG lookup
    ag_lookup = {}
    if AG_LOOKUP.exists():
        with open(AG_LOOKUP, encoding="utf-8") as f:
            ag_lookup = json.load(f)
        print(f"AG lookup: {len(ag_lookup):,} entries")

    if args.analyze or (not args.expand and not args.headwords):
        analyze(entries, ag_lookup)
        return

    if args.expand:
        ag_lookup = expand_entries(entries, ag_lookup)

        # Save updated lookup
        with open(AG_LOOKUP, "w", encoding="utf-8") as f:
            json.dump(ag_lookup, f, ensure_ascii=False, separators=(",", ":"))
        size_mb = AG_LOOKUP.stat().st_size / (1024 * 1024)
        print(f"Updated AG lookup: {len(ag_lookup):,} entries ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
