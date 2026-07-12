#!/usr/bin/env python3
"""Evaluate Dilemma with author-holdout benchmarking.

Builds the AG lookup from all Gorman treebank authors EXCEPT the one
being tested, then measures lemmatization accuracy on the held-out author.
This gives a fair test of whether treebank data from other authors helps
generalize to unseen text.

Also tests with and without LSJ/Cunliffe headwords in the model's beam
filter, so you can see the effect of each data source independently.

Usage:
    python eval_holdout.py                          # test all authors
    python eval_holdout.py --author herodotus       # test one author
    python eval_holdout.py --author xenophon-cyr --top 3000
    python eval_holdout.py --list                   # list available authors
"""

import argparse
import json
import os
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
# Holdout eval sources; set GORMAN_TREES / ILIAD_TREEBANK to your copies.
GORMAN = Path(os.environ.get(
    "GORMAN_TREES", Path.home() / "Documents" / "gorman-trees"))
ILIAD_TB = Path(os.environ.get(
    "ILIAD_TREEBANK", Path.home() / "Documents" / "iliad" / "treebank.xml"))
LSJ_PATH = DATA_DIR / "lsj_headwords.json"
CUNLIFFE_PATH = DATA_DIR / "cunliffe_headwords.json"

# Author prefix mapping (filename stem -> canonical author name)
AUTHOR_PREFIXES = {
    "xen-cyr": "xenophon-cyr",
    "xen-hell": "xenophon-hell",
    "hdt": "herodotus",
    "thuc": "thucydides",
    "dem": "demosthenes",
    "demosthenes": "demosthenes",
    "lysias": "lysias",
    "plato": "plato",
    "polybius": "polybius",
    "polybius1": "polybius",
    "plut": "plutarch",
    "plutarch": "plutarch",
    "athen": "athenaeus",
    "diodsic": "diodorus",
    "dion-hal": "dionysius-hal",
    "josephus": "josephus",
    "appian": "appian",
    "aeschines": "aeschines",
    "antiphon": "antiphon",
    "aristotle": "aristotle",
    "ps-xen": "ps-xenophon",
}


def file_to_author(path: Path) -> str:
    """Map a treebank filename to its canonical author name."""
    stem = path.stem
    # Try longest prefix first
    for prefix in sorted(AUTHOR_PREFIXES.keys(), key=len, reverse=True):
        if stem.startswith(prefix):
            return AUTHOR_PREFIXES[prefix]
    return stem.split("-")[0]


def load_treebank_pairs(xml_files) -> list[tuple[str, str]]:
    """Extract (form, lemma) pairs from treebank XML files."""
    pairs = []
    for f in xml_files:
        try:
            tree = ET.parse(f)
            for w in tree.findall('.//word'):
                form = w.get('form', '').strip()
                lemma = w.get('lemma', '').strip()
                if (form and lemma and
                    any('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF'
                        for c in form)):
                    pairs.append((form, lemma))
        except ET.ParseError:
            pass
    return pairs


def strip_length(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        ''.join(c for c in nfd if ord(c) not in (0x0306, 0x0304)))


def in_lsj(lemma: str, lsj_set: set) -> bool:
    """Check if a lemma is in LSJ (with normalization variants)."""
    variants = [lemma, lemma.lower(), lemma[0].upper() + lemma[1:],
                strip_length(lemma), strip_length(lemma).lower()]
    # Add monotonic variants
    from dilemma import to_monotonic
    variants.extend([to_monotonic(lemma), to_monotonic(lemma).lower()])
    for v in variants:
        if v in lsj_set:
            return True
    return False


# Homeric ὅτι does not elide (it is written ὅττι / ὅ τι), so an elided ὅτ'/ὅθ'
# can only be temporal ὅτε or locative ὅθι. Some treebanks (AGDT included)
# mislabel elided ὅτ'/ὅθ' as ὅτι; a lemmatizer that returns ὅτε is CORRECT,
# so accept ὅτε/ὅθι for a gold ὅτι on an elided ὅτ-/ὅθ- form. Keyed by the
# accent-stripped, monotonic base spelling (see _norm_lemma).
_ELISION_GOLD_FIXUPS = {
    ("οτε", "οτι"),   # pred ὅτε vs gold ὅτι
    ("οθι", "οτι"),   # pred ὅθι vs gold ὅτι
}


def _norm_lemma(s: str) -> str:
    """Normalize a lemma for gold comparison: lowercase, monotonic,
    accent- and length-stripped. Keeps base letters (so ὅτε != ὅτι) while
    tolerating accent/breathing/case convention differences between the
    treebank gold and Dilemma's Wiktionary headword spellings."""
    from dilemma import to_monotonic, strip_accents
    return strip_accents(to_monotonic(strip_length(s).lower()))


def gold_match(pred: str, gold: str, form: str = "") -> bool:
    """Is the predicted lemma the RIGHT lemma (matches gold), not merely a
    valid LSJ headword? This is the accuracy the in_lsj check cannot see:
    ὅτι, καί, ὁ, εἷς are all valid headwords, so a wrong elision resolution
    still passes in_lsj but fails here.

    For elided ὅτ-/ὅθ- forms it accepts ὅτε/ὅθι where the gold says ὅτι,
    because the gold, not the lemmatizer, is wrong there (_ELISION_GOLD_FIXUPS)."""
    if pred is None:
        return False
    np, ng = _norm_lemma(pred), _norm_lemma(gold)
    if np == ng:
        return True
    return (np, ng) in _ELISION_GOLD_FIXUPS


def build_ag_lookup_without(holdout_author: str, all_files: dict[str, list],
                            base_lookup: dict) -> dict:
    """Build AG lookup augmented with treebank data, excluding one author."""
    lookup = dict(base_lookup)

    # Load headword sets for validation
    all_hw = {k for k, v in lookup.items() if k == v}
    if LSJ_PATH.exists():
        all_hw |= set(json.load(open(LSJ_PATH)))
    if CUNLIFFE_PATH.exists():
        all_hw |= set(json.load(open(CUNLIFFE_PATH)))

    added = 0
    for author, files in all_files.items():
        if author == holdout_author:
            continue
        for form, lemma in load_treebank_pairs(files):
            if form not in lookup and (lemma in all_hw or lemma in lookup):
                lookup[form] = lemma
                added += 1
            if lemma not in lookup and lemma in all_hw:
                lookup[lemma] = lemma

    # Also add Iliad treebank (always safe - not in Gorman holdout)
    if ILIAD_TB.exists() and holdout_author != "homer-iliad":
        for form, lemma in load_treebank_pairs([ILIAD_TB]):
            if form not in lookup and (lemma in all_hw or lemma in lookup):
                lookup[form] = lemma
                added += 1

    return lookup, added


# Apostrophe codepoints that mark elision (matches dilemma.core._ELISION_MARKS
# spacing members plus the combining psili); used to slice out elided forms.
_ELISION_CHARS = set("’ʼ᾽'`ʹ̓")


def _is_elided(form: str) -> bool:
    return any(c in _ELISION_CHARS for c in form)


def evaluate(test_pairs, lookup, lsj_set, headwords, top_n=3000,
             include_frequent=False):
    """Run lemmatization on test pairs and return accuracy stats.

    Reports TWO accuracies:
      * ``pct``      - in_lsj: is the output a valid LSJ headword. Loose; it
                       cannot see a wrong-but-valid lemma (ὅτι for ὅτε).
      * ``gold_pct`` - gold_match: is the output the RIGHT lemma. This is the
                       one that exposes the elision bug.
    Plus the same two over the elided-form subset (``elided_*``), where the
    elided function words live. Set ``include_frequent`` (or ``top_n=0``) to
    stop excluding the most common forms, so those function words are scored.

    Uses lemmatize_batch for speed - all model inference happens in
    one batched forward pass instead of per-word.
    """
    from dilemma import Dilemma

    if include_frequent or top_n <= 0:
        uncommon = list(test_pairs)
    else:
        form_counts = Counter(f for f, _ in test_pairs)
        common = {f for f, _ in form_counts.most_common(top_n)}
        uncommon = [(f, l) for f, l in test_pairs if f not in common]

    if not uncommon:
        return {"total": 0, "success": 0, "pct": 0.0, "gold_success": 0,
                "gold_pct": 0.0, "elided_total": 0, "elided_lsj": 0,
                "elided_gold": 0, "failures": [], "gold_failures": []}

    d = Dilemma(lang='all', resolve_articles=True)
    d._lookup = lookup
    d._ag_lookup = dict(lookup)

    # Pre-set headwords so model filter uses our augmented set
    d._headwords = headwords

    # Batch lemmatize all forms at once
    forms = [f for f, _ in uncommon]
    results = d.lemmatize_batch(forms)

    success = gold_success = 0
    elided_total = elided_lsj = elided_gold = 0
    failures = []
    gold_failures = []
    for i, (form, gold) in enumerate(uncommon):
        pred = results[i]
        lsj_ok = in_lsj(pred, lsj_set)
        gold_ok = gold_match(pred, gold, form)
        success += lsj_ok
        gold_success += gold_ok
        if not lsj_ok:
            failures.append((form, gold, pred))
        if not gold_ok:
            gold_failures.append((form, gold, pred))
        if _is_elided(form):
            elided_total += 1
            elided_lsj += lsj_ok
            elided_gold += gold_ok

    return {
        "total": len(uncommon),
        "success": success,
        "pct": 100 * success / len(uncommon),
        "gold_success": gold_success,
        "gold_pct": 100 * gold_success / len(uncommon),
        "elided_total": elided_total,
        "elided_lsj": elided_lsj,
        "elided_gold": elided_gold,
        "failures": failures[:10],
        "gold_failures": gold_failures[:10],
    }


def main():
    parser = argparse.ArgumentParser(description="Holdout evaluation for Dilemma")
    parser.add_argument("--author", type=str, default=None,
                        help="Author to hold out (default: test all)")
    parser.add_argument("--top", type=int, default=3000,
                        help="Exclude top N most common forms (default: 3000)")
    parser.add_argument("--list", action="store_true",
                        help="List available authors and exit")
    parser.add_argument("--failures", action="store_true",
                        help="Print failure details")
    parser.add_argument("--include-frequent", action="store_true",
                        help="Do not exclude the top-N most frequent forms "
                             "(elided function words live there)")
    parser.add_argument("--elision", action="store_true",
                        help="Print the elided-form subset accuracy and its "
                             "gold-mismatch failures")
    args = parser.parse_args()

    # Group files by author
    author_files = defaultdict(list)
    for f in sorted(GORMAN.glob("*.xml")):
        author_files[file_to_author(f)].append(f)

    if args.list:
        print(f"{'Author':<20} {'Files':>5}")
        print("-" * 28)
        for author in sorted(author_files):
            print(f"{author:<20} {len(author_files[author]):>5}")
        return

    # Load base AG lookup (Wiktionary only, no treebank augmentation).
    # We cache the clean Wiktionary-only lookup to avoid rebuilding every run.
    # Run `python build_data.py` then `cp data/ag_lookup.json data/ag_lookup_wikt.json`
    # to create/update the cache.
    wikt_cache = DATA_DIR / "ag_lookup_wikt.json"
    if wikt_cache.exists():
        print("Loading cached Wiktionary-only AG lookup...", end=" ", flush=True)
        with open(wikt_cache) as f:
            base_lookup = json.load(f)
    else:
        print("Building base lookup from Wiktionary (first run)...", flush=True)
        import subprocess
        subprocess.run([sys.executable, "build_data.py"], capture_output=True)
        with open(DATA_DIR / "ag_lookup.json") as f:
            base_lookup = json.load(f)
        # Cache it
        import shutil
        shutil.copy(DATA_DIR / "ag_lookup.json", wikt_cache)
        print("  Cached to ag_lookup_wikt.json for future runs.")
    print(f"{len(base_lookup):,} entries")

    # Load LSJ for validation
    lsj_set = set()
    if LSJ_PATH.exists():
        lsj_set = set(json.load(open(LSJ_PATH)))

    # Build full headword set for model filter
    base_hw = {k for k, v in base_lookup.items() if k == v}
    full_hw = set(base_hw)
    if LSJ_PATH.exists():
        full_hw |= lsj_set
    if CUNLIFFE_PATH.exists():
        full_hw |= set(json.load(open(CUNLIFFE_PATH)))

    authors_to_test = [args.author] if args.author else sorted(author_files.keys())

    # Two accuracies per author: in_lsj (headword-valid) and gold (right lemma).
    print(f"\n{'Author':<18} {'Words':>6} {'LSJ base':>9} {'LSJ+tb':>8} "
          f"{'Gold base':>10} {'Gold+tb':>8} {'Gain':>6}")
    print("-" * 70)

    for author in authors_to_test:
        if author not in author_files:
            print(f"Unknown author: {author}")
            continue

        test_pairs = load_treebank_pairs(author_files[author])
        if len(test_pairs) < 50:
            continue

        # Baseline: Wiktionary + LSJ/Cunliffe headwords, no treebank
        base_result = evaluate(test_pairs, base_lookup, lsj_set, full_hw,
                               top_n=args.top,
                               include_frequent=args.include_frequent)

        # With treebank from other authors
        aug_lookup, added = build_ag_lookup_without(author, author_files,
                                                     base_lookup)
        aug_result = evaluate(test_pairs, aug_lookup, lsj_set, full_hw,
                              top_n=args.top,
                              include_frequent=args.include_frequent)

        gain = aug_result["gold_pct"] - base_result["gold_pct"]
        print(f"{author:<18} {base_result['total']:>6} "
              f"{base_result['pct']:>8.1f}% {aug_result['pct']:>7.1f}% "
              f"{base_result['gold_pct']:>9.1f}% {aug_result['gold_pct']:>7.1f}% "
              f"{gain:>+5.1f}pp")

        if args.elision and aug_result["elided_total"]:
            et = aug_result["elided_total"]
            print(f"    elided forms: {et}  "
                  f"in_lsj {100*aug_result['elided_lsj']/et:.1f}%  "
                  f"gold {100*aug_result['elided_gold']/et:.1f}%")
            for form, gold, got in aug_result["gold_failures"]:
                if _is_elided(form):
                    print(f"      {form} -> {got} (gold: {gold})")

        if args.failures and aug_result["gold_failures"]:
            for form, gold, got in aug_result["gold_failures"][:5]:
                print(f"  {form} -> {got} (gold: {gold})")


if __name__ == "__main__":
    main()
