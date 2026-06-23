#!/usr/bin/env python3
"""Extract indeclinable words (adverbs, prepositions, conjunctions, particles)
from lsj9 data and save as form->lemma pairs for the lookup table.

Reads lsj9_indeclinables.json (from an upstream LSJ9 export) which
contains headword -> POS category mappings detected from the raw LSJ entry
text.

Detection strategy (in the upstream LSJ9 export):
- STRONG markers: "Particle", "Prep.", "Conj.", "Interj.", or "exclamation"
  in the entry text (after stripping parenthetical content)
- WEAK markers: "Adv." in entry text for entries without adjective-like or
  verb-like endings
- Ending-based: entries with adverb-like suffixes (-θεν, -δε, -δόν, -δην,
  -τί, -κις, etc.) that also have "Adv." in the entry text

Output: data/indeclinable_pairs.json with form->lemma mappings (both accented
and accent-stripped variants), excluding forms already in ag_lookup.json.

Usage:
    python extract_indeclinables.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from expand_lsj import strip_diacritics, LSJ9_DIR

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LSJ9_INDECLINABLES = LSJ9_DIR / "lsj9_indeclinables.json"


def extract_indeclinables():
    """Load indeclinable entries from lsj9_indeclinables.json."""
    print("Loading indeclinables from lsj9 export...")

    if not LSJ9_INDECLINABLES.exists():
        print(f"  {LSJ9_INDECLINABLES} not found. Run the upstream LSJ9 export first.")
        return {}, Counter()

    with open(LSJ9_INDECLINABLES, encoding="utf-8") as f:
        indeclinables = json.load(f)

    stats = Counter(indeclinables.values())
    return indeclinables, stats


def main():
    indeclinables, stats = extract_indeclinables()

    print(f"\nFound {len(indeclinables):,} indeclinable entries:")
    for cat, count in stats.most_common():
        if cat.startswith('skipped'):
            continue
        print(f"  {cat}: {count}")

    for cat in ['adverb', 'preposition', 'conjunction', 'particle',
                'interjection']:
        examples = [hw for hw, c in indeclinables.items() if c == cat][:10]
        if examples:
            print(f"\n  {cat} examples: {', '.join(examples)}")

    # Load existing lookup
    ag_lookup_path = DATA_DIR / "ag_lookup.json"
    print(f"\nLoading {ag_lookup_path}...")
    with open(ag_lookup_path, encoding="utf-8") as f:
        existing_lookup = json.load(f)
    print(f"  {len(existing_lookup):,} existing entries")

    # Build new pairs, filtering already-present entries
    new_pairs = {}
    already_present = 0
    for hw, category in indeclinables.items():
        if hw not in existing_lookup:
            new_pairs[hw] = hw
        else:
            already_present += 1

        plain = strip_diacritics(hw)
        if plain != hw and plain not in existing_lookup:
            new_pairs[plain] = hw

    print(f"\n  Indeclinable headwords already in lookup: {already_present}")
    print(f"  New form->lemma pairs to add: {len(new_pairs):,}")

    new_by_cat = Counter()
    for hw, cat in indeclinables.items():
        if (hw not in existing_lookup or
                (strip_diacritics(hw) != hw and
                 strip_diacritics(hw) not in existing_lookup)):
            new_by_cat[cat] += 1
    print(f"  New entries by category:")
    for cat, count in new_by_cat.most_common():
        print(f"    {cat}: {count}")

    # Save
    out_path = DATA_DIR / "indeclinable_pairs.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(new_pairs, f, ensure_ascii=False, indent=None)
    size_kb = out_path.stat().st_size / 1024
    print(f"\nSaved {len(new_pairs):,} pairs to {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
