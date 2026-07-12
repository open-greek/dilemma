#!/usr/bin/env python3
"""Gold-match evaluation against the held-out Gorman treebanks.

The Gorman treebanks (Vanessa Gorman, CC BY-SA 4.0, hand-annotated,
github.com/perseids-publications/gorman-trees) are this project's
HELD-OUT GOLD corpus: they are deliberately never ingested into
lookup.db, treebank_pos_lookup.json, or any other shipped artifact
(build_lookup_db.py and build/build_treebank_pos_lookup.py enforce
this, and tests/test_new_features.py::TestGormanHoldout guards it).
That makes agreement with their lemma annotation a genuinely
independent accuracy measure across 18 classical authors - including
Herodotus, the project's only held-out Ionic gold.

Usage:
    GORMAN_TREES=/path/to/gorman-trees python eval/eval_gorman_gold.py
    ... --author herodotus        # single author
    ... --min-tokens 5000         # skip small authors in the table

Scoring matches the multi-period benchmarks: equiv-adjusted (accent-
stripped, case-folded, lemma equivalence groups from
data/lemma_equivalences.json), via lemmatize_batch with
resolve_articles=True (the treebank lemmatizes articles).
"""
import argparse
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "eval"))

from eval_holdout import GORMAN, AUTHOR_PREFIXES, load_treebank_pairs  # noqa: E402

EQUIV_PATH = SCRIPT_DIR / "data" / "lemma_equivalences.json"


def strip_accents(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize(
        "NFC", "".join(c for c in nfd if unicodedata.category(c) != "Mn"))


def load_equivalences() -> dict:
    with open(EQUIV_PATH, encoding="utf-8") as f:
        data = json.load(f)
    equiv = {}
    for group in data["groups"]:
        group_set = set(group)
        for lemma in group:
            equiv[lemma] = equiv.get(lemma, set()) | group_set
    return equiv


def are_equivalent(pred: str, gold: str, equiv: dict) -> bool:
    pa, ga = strip_accents(pred).lower(), strip_accents(gold).lower()
    if pa == ga:
        return True
    for e in equiv.get(gold, set()):
        if strip_accents(e).lower() == pa:
            return True
    for e in equiv.get(pred, set()):
        if strip_accents(e).lower() == ga:
            return True
    return False


def author_of(stem: str) -> str | None:
    for pref, name in AUTHOR_PREFIXES.items():
        if stem.startswith(pref):
            return name
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Gold-match eval vs the held-out Gorman treebanks")
    parser.add_argument("--author", type=str, default=None,
                        help="Evaluate a single author (e.g. herodotus)")
    parser.add_argument("--min-tokens", type=int, default=0,
                        help="Hide authors with fewer tokens from the table")
    args = parser.parse_args()

    if not GORMAN.exists():
        sys.exit(f"Gorman trees not found: {GORMAN}\n"
                 "Set GORMAN_TREES to a clone of "
                 "github.com/perseids-publications/gorman-trees (or the "
                 "vgorman1/Greek-Dependency-Trees XML files).")

    by_author = defaultdict(list)
    for f in sorted(GORMAN.glob("*.xml")):
        author = author_of(f.stem)
        if author is None:
            continue
        if args.author and author != args.author.lower():
            continue
        by_author[author].extend(load_treebank_pairs([f]))

    if not by_author:
        sys.exit("No matching treebank files found.")

    from dilemma import Dilemma
    d = Dilemma(lang="all", resolve_articles=True)
    equiv = load_equivalences()

    total_n = total_ok = 0
    print(f"{'Author':<18} {'Tokens':>8}  {'Gold':>6}")
    print("-" * 36)
    for author in sorted(by_author):
        pairs = by_author[author]
        preds = d.lemmatize_batch([form for form, _ in pairs])
        ok = sum(1 for (form, gold), p in zip(pairs, preds)
                 if p and are_equivalent(p, gold, equiv))
        total_n += len(pairs)
        total_ok += ok
        if len(pairs) >= args.min_tokens:
            print(f"{author:<18} {len(pairs):>8}  {100 * ok / len(pairs):5.1f}%")
    print("-" * 36)
    print(f"{'OVERALL':<18} {total_n:>8}  {100 * total_ok / total_n:5.1f}%")


if __name__ == "__main__":
    main()
