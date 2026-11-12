#!/usr/bin/env python3
"""Evaluate Dilemma against DBBE gold standard (Swaelens et al.).

DBBE (Database of Byzantine Book Epigrams) provides 10K tokens of
unedited Byzantine Greek with gold lemmas and POS tags.

Usage:
    python eval_dbbe.py                       # default settings
    python eval_dbbe.py --scale 3             # specific model scale
    python eval_dbbe.py --errors 50           # show more error examples
    python eval_dbbe.py --pos V-              # filter by POS
    python eval_dbbe.py --use-pos gold        # use gold POS tags for lemmatization
"""

import argparse
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DBBE = SCRIPT_DIR / "data" / "dbbe" / "lingAnn_GS_medievalGreek.tsv"
EQUIV_PATH = SCRIPT_DIR / "data" / "lemma_equivalences.json"

# DBBE POS category (first char) -> UPOS mapping
DBBE_TO_UPOS = {
    "n": "NOUN",
    "v": "VERB",
    "a": "ADJ",
    "l": "DET",
    "g": "PART",
    "d": "ADV",
    "p": "PRON",
    "r": "ADP",
    "c": "CCONJ",
    "m": "NUM",
    "i": "INTJ",
    "e": "INTJ",
}


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


def parse_dbbe(tsv_path):
    """Parse DBBE TSV and return list of token dicts."""
    tokens = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            form, pos, lemma = parts
            # Skip punctuation
            if pos.startswith("u"):
                continue
            # DBBE POS: first char is category (n,v,a,l,g,d,p,r,c,m,i,e)
            # second char is person for verbs/pronouns, so group by first char
            pos_cat = pos[0] if pos else "?"
            tokens.append({
                "form": form,
                "gold_lemma": lemma,
                "pos": pos_cat,
                "pos_full": pos,
            })
    return tokens


def evaluate(tokens, dilemma_instance, greedy=True, use_pos=None):
    """Run Dilemma on all tokens and compare with gold lemmas.

    Args:
        use_pos: None (no POS), "gold" (use gold DBBE POS tags mapped to UPOS).
    """
    if greedy:
        orig_predict = dilemma_instance._predict
        def _greedy_predict(words, num_beams=1):
            return orig_predict(words, num_beams=1)
        dilemma_instance._predict = _greedy_predict

    forms = [t["form"] for t in tokens]
    batch_size = 500
    predicted = []

    if use_pos == "gold":
        upos_tags = [DBBE_TO_UPOS.get(t["pos"], "X") for t in tokens]
        for i in range(0, len(forms), batch_size):
            batch_forms = forms[i:i+batch_size]
            batch_upos = upos_tags[i:i+batch_size]
            predicted.extend(dilemma_instance.lemmatize_batch_pos(
                batch_forms, batch_upos))
            done = min(i + batch_size, len(forms))
            if done % 5000 < batch_size:
                print(f"  {done}/{len(forms)}...", flush=True)
    else:
        for i in range(0, len(forms), batch_size):
            batch = forms[i:i+batch_size]
            predicted.extend(dilemma_instance.lemmatize_batch(batch))
            done = min(i + batch_size, len(forms))
            if done % 5000 < batch_size:
                print(f"  {done}/{len(forms)}...", flush=True)

    if greedy:
        dilemma_instance._predict = orig_predict

    # Load equivalences
    equiv_map = {}
    if EQUIV_PATH.exists():
        with open(EQUIV_PATH, encoding="utf-8") as f:
            equiv_data = json.load(f)
        for group in equiv_data["groups"]:
            group_set = set(group)
            for lemma in group:
                equiv_map[lemma] = group_set
                equiv_map[strip_accents(lemma.lower())] = {strip_accents(l.lower()) for l in group}

    def is_equiv(pred, gold):
        stripped_pred = strip_accents(pred.lower())
        stripped_gold = strip_accents(gold.lower())
        if stripped_pred in equiv_map:
            return stripped_gold in equiv_map[stripped_pred]
        return False

    results = []
    for t, pred in zip(tokens, predicted):
        gold = t["gold_lemma"]
        pred = pred or ""   # a token the lemmatizer declines counts as wrong
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
    parser = argparse.ArgumentParser(description="Evaluate Dilemma on DBBE")
    parser.add_argument("--dbbe", type=str, default=str(DEFAULT_DBBE),
                        help="Path to DBBE TSV file")
    parser.add_argument("--scale", type=int, default=None,
                        help="Dilemma model scale")
    parser.add_argument("--errors", type=int, default=20,
                        help="Number of error examples to show")
    parser.add_argument("--pos", type=str, default=None,
                        help="Filter by POS prefix (e.g. V-, Nb, A-)")
    parser.add_argument("--use-pos", type=str, default=None, choices=["gold"],
                        help="Use POS tags for lemmatization: 'gold' uses DBBE gold tags")
    args = parser.parse_args()

    print(f"Parsing DBBE: {args.dbbe}")
    tokens = parse_dbbe(args.dbbe)
    print(f"Total tokens: {len(tokens)} (punctuation excluded)")

    # POS breakdown
    pos_counts = Counter(t["pos"] for t in tokens)
    print(f"\nPOS breakdown:")
    pos_names = {"v": "verb", "n": "noun", "a": "adj", "l": "article",
                 "r": "prep", "d": "adverb", "c": "conj", "p": "pronoun",
                 "m": "numeral", "i": "interj", "g": "particle", "e": "exclam"}
    for pos, count in pos_counts.most_common():
        name = pos_names.get(pos, pos)
        print(f"  {name:20s} ({pos})  {count:>6}")

    if args.pos:
        tokens = [t for t in tokens if t["pos"] == args.pos]
        print(f"\nFiltered to POS={args.pos}: {len(tokens)} tokens")

    if not tokens:
        print("No tokens to evaluate.")
        return

    # Load Dilemma
    sys.path.insert(0, str(SCRIPT_DIR))
    from dilemma import Dilemma
    d = Dilemma(scale=args.scale, resolve_articles=True)
    pos_label = f", use_pos={args.use_pos}" if args.use_pos else ""
    print(f"\nDilemma loaded (scale={args.scale}, resolve_articles=True{pos_label})")

    # Evaluate
    print(f"\nEvaluating...")
    results = evaluate(tokens, d, use_pos=args.use_pos)

    # Overall
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print_results(results, "Overall")

    # By POS
    print(f"\nBy POS:")
    by_pos = defaultdict(list)
    for r in results:
        by_pos[r["pos"]].append(r)
    for pos in sorted(by_pos, key=lambda p: -len(by_pos[p])):
        name = pos_names.get(pos, pos)
        print_results(by_pos[pos], f"{name} ({pos})")

    # Error analysis
    errors = [r for r in results if not r["stripped"]]
    if errors and args.errors > 0:
        # Categorize errors
        no_lookup = [r for r in errors if r["predicted"] == r["form"]]
        wrong_lemma = [r for r in errors if r["predicted"] != r["form"]]
        print(f"\nError breakdown:")
        print(f"  Wrong lemma: {len(wrong_lemma)} ({len(wrong_lemma)/len(results):.1%})")
        print(f"  No lookup:   {len(no_lookup)} ({len(no_lookup)/len(results):.1%})")

        print(f"\nError examples (first {min(args.errors, len(errors))}):")
        import io, sys as _sys
        out = io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")
        for r in errors[:args.errors]:
            tag = "MISS" if r["predicted"] == r["form"] else "WRONG"
            out.write(f"  [{tag:5s}] {r['form']:20s} gold={r['gold_lemma']:20s} "
                      f"pred={r['predicted']:20s} [{r['pos']}]\n")
        out.flush()
        out.detach()  # don't close underlying stdout


if __name__ == "__main__":
    main()
