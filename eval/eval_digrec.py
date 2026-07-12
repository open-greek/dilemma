#!/usr/bin/env python3
"""Evaluate Dilemma against DiGreC treebank gold standard.

DiGreC (DIachrony of GREek Case) is a manually reviewed treebank
spanning Homer through 15th century Byzantine Greek.

Usage:
    python eval_digrec.py                       # all sources, auto-detect scale
    python eval_digrec.py --scale 3             # specific scale
    python eval_digrec.py --period byzantine    # filter by period
    python eval_digrec.py --digrec /path/to/digrec.xml
"""

import argparse
import json
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DIGREC = Path.home() / "Documents" / "digrec" / "data" / "digrec.xml"
EQUIV_PATH = SCRIPT_DIR / "data" / "lemma_equivalences.json"

# TLG ID ranges for rough period classification
# Based on TLG numbering conventions + known sources
BYZANTINE_IDS = {
    # Vernacular Byzantine
    "digenis", "achilleis", "ilias_byzantina",
    # Known Byzantine TLG IDs (approximate ranges)
}

# Date-based classification from printed-text-date element
def classify_period(source_elem):
    """Classify a source by period based on date metadata."""
    date_elem = source_elem.find("printed-text-date")
    title_elem = source_elem.find("title")
    author_elem = source_elem.find("author")
    title = title_elem.text if title_elem is not None else ""
    author = author_elem.text if author_elem is not None else ""
    date = date_elem.text if date_elem is not None else ""

    # Try to parse a year from the date field
    import re
    year = None
    if date:
        m = re.search(r"(\d{1,4})", date)
        if m:
            year = int(m.group(1))

    # Classify by year
    if year is not None:
        if year < 0 or year <= 300:
            return "classical"
        elif year <= 600:
            return "late_antique"
        elif year <= 1100:
            return "early_byzantine"
        elif year <= 1500:
            return "late_byzantine"
        else:
            return "post_byzantine"

    # Fallback: classify by TLG ID range
    sid = source_elem.get("id", "")
    try:
        tlg_num = int(sid.split(".")[0].replace("tlg", ""))
    except (ValueError, IndexError):
        return "unknown"

    if tlg_num <= 100:
        return "classical"
    elif tlg_num <= 500:
        return "classical"  # most early TLG
    elif tlg_num <= 2100:
        return "classical"
    elif tlg_num <= 3000:
        return "late_antique"
    elif tlg_num <= 4500:
        return "byzantine"
    else:
        return "unknown"


def to_monotonic(s):
    _strip = {0x0313, 0x0314, 0x0345, 0x0306, 0x0304}
    _to_acute = {0x0300, 0x0342}
    nfd = unicodedata.normalize("NFD", s)
    out = []
    for ch in nfd:
        cp = ord(ch)
        if cp in _strip:
            continue
        if cp in _to_acute:
            out.append("\u0301")
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def strip_accents(s):
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn"))


def parse_digrec(xml_path):
    """Parse DiGreC XML and return list of (form, gold_lemma, pos, source_id, period)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    tokens = []
    for source in root.findall(".//source"):
        sid = source.get("id", "")
        period = classify_period(source)
        author = (source.findtext("author") or "").strip()

        for token in source.findall(".//token"):
            form = token.get("form", "")
            lemma = token.get("lemma", "")
            pos = token.get("part-of-speech", "")
            if form and lemma:
                tokens.append({
                    "form": form,
                    "gold_lemma": lemma,
                    "pos": pos,
                    "source": sid,
                    "period": period,
                    "author": author,
                })

    return tokens


def evaluate(tokens, dilemma_instance, greedy=True):
    """Run Dilemma on all tokens and compare with gold lemmas.

    Uses greedy decoding by default for speed (beam search is 10-50x
    slower and only matters for the model fallback path).
    """
    results = []

    # For greedy mode, set num_beams=1 on the existing _predict method
    if greedy:
        orig_predict = dilemma_instance._predict
        def _greedy_predict(words, num_beams=1):
            return orig_predict(words, num_beams=1)
        dilemma_instance._predict = _greedy_predict

    # Batch lemmatize
    forms = [t["form"] for t in tokens]
    batch_size = 500
    predicted = []
    for i in range(0, len(forms), batch_size):
        batch = forms[i:i+batch_size]
        predicted.extend(dilemma_instance.lemmatize_batch(batch))
        done = min(i + batch_size, len(forms))
        if done % 5000 < batch_size:
            print(f"  {done}/{len(forms)}...", flush=True)

    if greedy:
        dilemma_instance._predict = orig_predict

    # Load lemma equivalence table
    equiv_map = {}  # lemma -> set of equivalent lemmas
    if EQUIV_PATH.exists():
        with open(EQUIV_PATH, encoding="utf-8") as f:
            equiv_data = json.load(f)
        for group in equiv_data["groups"]:
            group_set = set(group)
            for lemma in group:
                equiv_map[lemma] = group_set
                equiv_map[strip_accents(lemma.lower())] = {strip_accents(l.lower()) for l in group}

    def is_equiv(pred, gold):
        """Check if pred and gold are equivalent lemmas."""
        stripped_pred = strip_accents(pred.lower())
        stripped_gold = strip_accents(gold.lower())
        if stripped_pred in equiv_map:
            return stripped_gold in equiv_map[stripped_pred]
        return False

    for t, pred in zip(tokens, predicted):
        gold = t["gold_lemma"]

        # Compare at multiple normalization levels
        strict = (pred == gold)
        mono = (to_monotonic(pred).lower() == to_monotonic(gold).lower())
        stripped = (strip_accents(pred.lower()) == strip_accents(gold.lower()))
        equiv = stripped or is_equiv(pred, gold)

        results.append({
            **t,
            "predicted": pred,
            "strict": strict,
            "mono": mono,
            "stripped": stripped,
            "equiv": equiv,
        })

    return results


def print_results(results, label=""):
    """Print accuracy breakdown."""
    total = len(results)
    if total == 0:
        print(f"  {label}: no tokens")
        return

    strict = sum(1 for r in results if r["strict"])
    mono = sum(1 for r in results if r["mono"])
    stripped = sum(1 for r in results if r["stripped"])
    equiv = sum(1 for r in results if r.get("equiv", r["stripped"]))

    print(f"  {label:30s}  {total:>6} tokens  "
          f"strict={strict/total:5.1%}  mono={mono/total:5.1%}  "
          f"stripped={stripped/total:5.1%}  equiv={equiv/total:5.1%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Dilemma on DiGreC")
    parser.add_argument("--digrec", type=str, default=str(DEFAULT_DIGREC),
                        help="Path to digrec.xml")
    parser.add_argument("--scale", type=int, default=None,
                        help="Dilemma model scale")
    parser.add_argument("--period", type=str, default=None,
                        help="Filter by period (classical, late_antique, byzantine, etc.)")
    parser.add_argument("--author", type=str, default=None,
                        help="Filter by source author (e.g. Herodotus)")
    parser.add_argument("--exclude-author", action="append", default=[],
                        help="Drop sources by author; repeatable. Used to "
                             "exclude the PROIEL-derived sections (Herodotus, "
                             "New Testament, Septuagint, Pseudo-Sphrantzes): "
                             "the PROIEL treebank is CC BY-NC-SA and project "
                             "policy avoids it entirely, evaluation included.")
    parser.add_argument("--errors", type=int, default=20,
                        help="Number of error examples to show")
    args = parser.parse_args()

    print(f"Parsing DiGreC: {args.digrec}")
    tokens = parse_digrec(args.digrec)
    print(f"Total tokens: {len(tokens)}")

    # Period breakdown
    period_counts = Counter(t["period"] for t in tokens)
    print(f"\nPeriod breakdown:")
    for period, count in period_counts.most_common():
        print(f"  {period:20s} {count:>6}")

    if args.period:
        tokens = [t for t in tokens if t["period"] == args.period]
        print(f"\nFiltered to {args.period}: {len(tokens)} tokens")

    if args.author:
        tokens = [t for t in tokens
                  if t["author"].lower() == args.author.lower()]
        print(f"\nFiltered to author {args.author}: {len(tokens)} tokens")

    if args.exclude_author:
        drop = {a.lower() for a in args.exclude_author}
        tokens = [t for t in tokens if t["author"].lower() not in drop]
        print(f"\nExcluded {sorted(drop)}: {len(tokens)} tokens remain")

    if not tokens:
        print("No tokens to evaluate.")
        return

    # Load Dilemma
    sys.path.insert(0, str(SCRIPT_DIR))
    from dilemma import Dilemma
    d = Dilemma(scale=args.scale, resolve_articles=True)
    print(f"\nDilemma loaded (scale={args.scale}, resolve_articles=True)")

    # Evaluate
    print(f"\nEvaluating...")
    results = evaluate(tokens, d)

    # Overall
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print_results(results, "Overall")

    # By period
    print(f"\nBy period:")
    by_period = defaultdict(list)
    for r in results:
        by_period[r["period"]].append(r)
    for period in ["classical", "late_antique", "early_byzantine",
                    "late_byzantine", "byzantine", "post_byzantine", "unknown"]:
        if period in by_period:
            print_results(by_period[period], period)

    # By POS
    print(f"\nBy POS:")
    pos_names = {"V-": "verb", "Nb": "noun", "A-": "adj", "S-": "article",
                 "R-": "prep", "Df": "adverb", "C-": "conj", "Ne": "proper",
                 "Pp": "pers.pron", "Pd": "dem.pron", "Px": "indef.pron",
                 "Pi": "interr.pron"}
    by_pos = defaultdict(list)
    for r in results:
        by_pos[r["pos"]].append(r)
    for pos in sorted(by_pos, key=lambda p: -len(by_pos[p])):
        name = pos_names.get(pos, pos)
        print_results(by_pos[pos], f"{name} ({pos})")

    # Error examples
    errors = [r for r in results if not r["stripped"]]
    if errors and args.errors > 0:
        print(f"\nError examples (first {min(args.errors, len(errors))}):")
        for r in errors[:args.errors]:
            print(f"  {r['form']:20s} gold={r['gold_lemma']:20s} "
                  f"pred={r['predicted']:20s} [{r['pos']}] ({r['period']})")


if __name__ == "__main__":
    main()
