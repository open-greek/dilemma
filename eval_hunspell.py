#!/usr/bin/env python3
"""Evaluate the compact Hunspell artifact against full Dilemma.

We want to know two things about the shipped dictionaries:

    1. Can they correct typos at the level we need for a keyboard?
    2. How much does the compression cost us vs the full
       suggest_spelling() on lookup.db?

Method
------

- Pick a held-out word sample per variant. For MG we take the top-N
  highest-frequency monotonic forms from mg_form_freq.json; for AG we
  take the top-N highest-frequency forms from corpus_freq.json intersected
  with the lookup.db polytonic vocabulary (since corpus_freq is stripped).
- Generate edit-distance 1 and edit-distance 2 perturbations.
- For each (typo, target_word) pair, ask Hunspell for top-1 and top-5
  suggestions. Compare to dilemma.suggest_spelling() if available.
- Report top-1 and top-5 correction accuracy.

Output
------

A small table to stdout and written to build/hunspell/eval_results.txt.

Usage
-----

    python3 eval_hunspell.py                  # fast eval, 100 typos per variant
    python3 eval_hunspell.py --n 500          # bigger sample
    python3 eval_hunspell.py --variant el     # only MG
    python3 eval_hunspell.py --deep           # also run spylls.suggest fallback
                                              # (slow, ~3s per miss)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "build" / "hunspell"

try:
    from spylls.hunspell import Dictionary
except ImportError:
    print("ERROR: spylls not installed. pip install spylls", file=sys.stderr)
    sys.exit(1)

# The exporter's own inclusion rule, imported rather than restated.
# The sample below is only meaningful if it picks target words the
# shipped dictionary actually accepts, so the two must not drift: a
# restated copy that tested only combining marks would report elided
# forms such as 'δ᾽' as unmarked and quietly skip them.
from export_hunspell import has_any_diacritic


def strip_accents(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize(
        "NFC", "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    )


def strip_key(s: str) -> str:
    """Stripped-lowercase key matching corpus_freq.json form keys."""
    nfd = unicodedata.normalize("NFD", s).lower()
    return unicodedata.normalize(
        "NFC", "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    )


POLYTONIC = {0x0313, 0x0314, 0x0342, 0x0345, 0x0300}


def has_polytonic(s: str) -> bool:
    return any(ord(c) in POLYTONIC for c in unicodedata.normalize("NFD", s))


GREEK_LETTERS = "αβγδεζηθικλμνξοπρστυφχψωάέήίόύώϊϋΐΰᾳῃῳ"


def edits1(word: str) -> list[str]:
    """Greek-alphabet edit-distance-1 variants of word."""
    letters = GREEK_LETTERS
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    out = list(set(deletes + transposes + replaces + inserts))
    return [w for w in out if w and w != word]


def sample_target_words(variant: str, n: int, seed: int = 42) -> list[str]:
    """Sample high-to-mid frequency attested words for the variant.

    We avoid the top-10 (too trivial, Hunspell accepts) and the long tail
    (too rare, no corpus-grounded typos to test against) by picking from
    frequency rank 100..10000.
    """
    if variant == "el":
        with open(DATA / "mg_form_freq.json", encoding="utf-8") as f:
            raw = json.load(f)
        # Keep only monotonic forms with decent freq (>= 100)
        candidates = [
            w for w, c in raw.items()
            if c >= 100 and not has_polytonic(w)
            and len(w) >= 3 and len(w) <= 20
        ]
    elif variant == "grc":
        with open(DATA / "corpus_freq.json", encoding="utf-8") as f:
            raw = json.load(f)
        forms = raw.get("forms", {})
        # corpus_freq keys are stripped-lowercase. We use them to pick
        # target stripped forms, then reconstruct an accented form by
        # indexing lookup.db src='grc' by computed stripped(form). The
        # shipped grc_polytonic dict contains polytonic + acute-only
        # forms (but not fully-stripped forms), so we must return an
        # accented sibling - not the stripped key itself - or Hunspell
        # will never accept the target and accuracy collapses to 0.
        candidates = [
            w for w, counts in forms.items()
            if counts and counts[0] >= 100 and 3 <= len(w) <= 20
        ]
        # Build stripped -> [accented forms] index once.
        import sqlite3
        from collections import defaultdict
        conn = sqlite3.connect(str(DATA / "lookup.db"))
        idx: dict[str, list[str]] = defaultdict(list)
        for (f,) in conn.execute("SELECT form FROM lookup WHERE src='grc'"):
            idx[strip_key(f)].append(f)
        reconstructed: list[str] = []
        for stripped in candidates:
            matches = idx.get(stripped, [])
            # Prefer polytonic (breathing / circumflex / grave / iota sub)
            poly = [m for m in matches if has_polytonic(m)]
            if poly:
                reconstructed.append(poly[0])
                continue
            # Fall back to acute-only (still in the dict per inclusion rules)
            acute = [m for m in matches if has_any_diacritic(m)]
            if acute:
                reconstructed.append(acute[0])
                continue
            # No accented sibling attested: skip (testing the stripped form
            # against a polytonic-only dict is not a meaningful correction
            # task, it's a coverage miss).
        candidates = reconstructed
    else:
        raise ValueError(variant)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    # Skip the very top (too trivial) and pick from positions 100..
    pool = candidates[100:]
    return pool[:n]


def gen_typos(word: str, seed: int = 0) -> tuple[str, str]:
    """Return (ed1_typo, ed2_typo) deterministically for a word."""
    rng = random.Random(f"{word}:{seed}")
    e1 = rng.choice(edits1(word))
    e2_pool = edits1(e1)
    e2 = rng.choice(e2_pool) if e2_pool else e1
    return e1, e2


def dict_top_n(d: Dictionary, typo: str, n: int, deep: bool) -> list[str]:
    """Get up to N suggestions from a spylls Hunspell dictionary.

    Strategy:
      1. If the typo is a dictionary word, return it as-is.
      2. Otherwise enumerate ED1 candidates over the Greek alphabet and
         keep any that the dictionary accepts. Cheap, ~1s for a 10-char
         word against our ~772K-entry dict.
      3. With --deep, also fall back to spylls' d.suggest() which implements
         the full Hunspell suggest pipeline (replacements, phonetic, ED2
         ngram). Slow (~3s per call), but matches what an actual consumer
         would do for misses at ED2.
    """
    if d.lookup(typo):
        return [typo]

    seen: list[str] = []
    seen_set: set[str] = set()

    for cand in edits1(typo):
        if cand in seen_set:
            continue
        if d.lookup(cand):
            seen.append(cand)
            seen_set.add(cand)
            if len(seen) >= n:
                return seen

    if deep and len(seen) < n:
        try:
            for cand in d.suggest(typo):
                if cand in seen_set or cand == typo:
                    continue
                if " " in cand:
                    continue
                seen.append(cand)
                seen_set.add(cand)
                if len(seen) >= n:
                    break
        except Exception:
            pass
    return seen


def evaluate_variant(variant: str, n: int, compare_full: bool = False, deep: bool = False) -> dict:
    if variant == "el":
        dic_name = "el_GR_monotonic"
    else:
        dic_name = "grc_polytonic"

    d = Dictionary.from_files(str(OUT / dic_name))

    targets = sample_target_words(variant, n)
    if not targets:
        return {"variant": variant, "error": "no targets"}

    top1_ed1 = top5_ed1 = top1_ed2 = top5_ed2 = 0
    total_ed1 = total_ed2 = 0

    full_top1_ed1 = full_top5_ed1 = full_top1_ed2 = full_top5_ed2 = 0

    # Lazy full dilemma only if compare_full
    full = None
    if compare_full:
        try:
            from dilemma import Dilemma
            lang = "grc" if variant == "grc" else "el"
            full = Dilemma(lang=lang)
        except Exception as e:
            print(f"  Warning: couldn't load Dilemma for comparison: {e}",
                  file=sys.stderr)
            full = None

    for i, target in enumerate(targets):
        e1, e2 = gen_typos(target, seed=i)

        total_ed1 += 1
        suggs = dict_top_n(d, e1, 5, deep=deep)
        if suggs:
            if suggs[0] == target:
                top1_ed1 += 1
            if target in suggs:
                top5_ed1 += 1

        total_ed2 += 1
        suggs = dict_top_n(d, e2, 5, deep=deep)
        if suggs:
            if suggs[0] == target:
                top1_ed2 += 1
            if target in suggs:
                top5_ed2 += 1

        if full is not None:
            try:
                s1 = [f for f, _ in full.suggest_spelling(e1, max_distance=2)[:5]]
                if s1 and s1[0] == target:
                    full_top1_ed1 += 1
                if target in s1:
                    full_top5_ed1 += 1
                s2 = [f for f, _ in full.suggest_spelling(e2, max_distance=2)[:5]]
                if s2 and s2[0] == target:
                    full_top1_ed2 += 1
                if target in s2:
                    full_top5_ed2 += 1
            except Exception:
                pass

    result = {
        "variant": variant,
        "n_targets": len(targets),
        "hunspell_ed1_top1": top1_ed1,
        "hunspell_ed1_top5": top5_ed1,
        "hunspell_ed2_top1": top1_ed2,
        "hunspell_ed2_top5": top5_ed2,
        "total_ed1": total_ed1,
        "total_ed2": total_ed2,
    }
    if full is not None:
        result.update({
            "full_ed1_top1": full_top1_ed1,
            "full_ed1_top5": full_top5_ed1,
            "full_ed2_top1": full_top1_ed2,
            "full_ed2_top5": full_top5_ed2,
        })
    return result


def fmt_pct(num: int, denom: int) -> str:
    if denom == 0:
        return "  --%"
    return f"{100 * num / denom:5.1f}%"


def print_report(results: list[dict]) -> str:
    lines = []
    lines.append("Hunspell spelling correction eval")
    lines.append("=" * 70)
    has_full = any("full_ed1_top1" in r for r in results)

    if has_full:
        header = (
            f"{'variant':<8} {'n':>5}  | "
            f"{'hs e1@1':>8} {'hs e1@5':>8} {'hs e2@1':>8} {'hs e2@5':>8} | "
            f"{'fl e1@1':>8} {'fl e1@5':>8} {'fl e2@1':>8} {'fl e2@5':>8}"
        )
    else:
        header = (
            f"{'variant':<8} {'n':>5}  | "
            f"{'hs e1@1':>8} {'hs e1@5':>8} {'hs e2@1':>8} {'hs e2@5':>8}"
        )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        if "error" in r:
            lines.append(f"{r['variant']:<8}  {r['error']}")
            continue
        row = (
            f"{r['variant']:<8} {r['n_targets']:>5}  | "
            f"{fmt_pct(r['hunspell_ed1_top1'], r['total_ed1']):>8} "
            f"{fmt_pct(r['hunspell_ed1_top5'], r['total_ed1']):>8} "
            f"{fmt_pct(r['hunspell_ed2_top1'], r['total_ed2']):>8} "
            f"{fmt_pct(r['hunspell_ed2_top5'], r['total_ed2']):>8}"
        )
        if "full_ed1_top1" in r:
            row += (
                " | "
                f"{fmt_pct(r['full_ed1_top1'], r['total_ed1']):>8} "
                f"{fmt_pct(r['full_ed1_top5'], r['total_ed1']):>8} "
                f"{fmt_pct(r['full_ed2_top1'], r['total_ed2']):>8} "
                f"{fmt_pct(r['full_ed2_top5'], r['total_ed2']):>8}"
            )
        lines.append(row)

    lines.append("")
    lines.append("Legend:")
    lines.append("  hs = compact Hunspell artifact, fl = full Dilemma")
    lines.append("  e1@1 = top-1 accuracy at edit distance 1")
    lines.append("  e1@5 = top-5 accuracy at edit distance 1")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100,
                    help="Number of target words per variant")
    ap.add_argument("--variant", choices=["el", "grc", "both"],
                    default="grc")
    ap.add_argument("--compare-full", action="store_true",
                    help="Also run Dilemma.suggest_spelling() as baseline. "
                         "Slow, requires dilemma lookup.db loaded.")
    ap.add_argument("--deep", action="store_true",
                    help="Enable spylls.suggest() fallback for words missed "
                         "by the fast ED1 scan. ~3s per miss.")
    args = ap.parse_args()

    variants = ["el", "grc"] if args.variant == "both" else [args.variant]
    results = []
    for v in variants:
        print(f"Evaluating {v}...")
        r = evaluate_variant(v, args.n,
                             compare_full=args.compare_full,
                             deep=args.deep)
        results.append(r)

    report = print_report(results)
    print()
    print(report)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval_results.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\nResults written to {OUT / 'eval_results.txt'}")


if __name__ == "__main__":
    main()
