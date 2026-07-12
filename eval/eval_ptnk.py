#!/usr/bin/env python3
"""Septuagint / Koine gold eval against UD_Ancient_Greek-PTNK.

PTNK (CC BY-SA 4.0, github.com/UniversalDependencies/UD_Ancient_Greek-PTNK)
is the Septuagint per Codex Alexandrinus - Genesis and Ruth - with MANUAL
native lemma and morphology annotation (only its syntax was projected from
Hebrew). No PROIEL involvement, and it is not ingested into any dilemma
artifact, so this is an independent Koine gold benchmark.

Caveat printed with the results: the underlying TEXT is inside dilemma's
corpora (GLAUx annotates Genesis via Pedalion, Diorisis has the LXX), so
this measures convention alignment on seen text more than unseen-Koine
generalization.

PTNK's lemma conventions differ systematically from dilemma's Wiktionary
convention (pronoun case-form lemmas σε/σός/ἐμός, εἶπον as an aorist
lemma, ἐνώπιον as its own headword, comparatives as headwords). Those are
neutralized by a LOCAL equivalence table below - deliberately NOT added to
data/lemma_equivalences.json, which would loosen every other benchmark.

Usage:
    PTNK_DIR=~/Documents/UD_Ancient_Greek-PTNK python eval/eval_ptnk.py
"""
import os
import sys
import unicodedata
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "eval"))

from eval_gorman_gold import load_equivalences, are_equivalent, strip_accents  # noqa: E402

PTNK_DIR = Path(os.environ.get(
    "PTNK_DIR", Path.home() / "Documents" / "UD_Ancient_Greek-PTNK"))

# PTNK-convention pairs (gold <-> dilemma/Wiktionary), applied ONLY here.
# Accent-stripped, lowercased comparison keys.
_PTNK_EQUIV = {
    ("σε", "συ"), ("σος", "συ"), ("εμος", "εγω"),
    ("ειπον", "λεγω"), ("ενωπιον", "ενωπιος"),
    ("νεωτερος", "νεος"), ("πρεσβυτερος", "πρεσβυς"),
    ("γινομαι", "γιγνομαι"), ("εναντιον", "εναντιος"),
    ("εως", "ηως"),
}


def _key(s: str) -> str:
    return strip_accents(unicodedata.normalize("NFC", s)).lower()


def _ptnk_match(pred: str, gold: str, equiv: dict) -> bool:
    if are_equivalent(pred, gold, equiv):
        return True
    pair = (_key(gold), _key(pred))
    return pair in _PTNK_EQUIV or pair[::-1] in _PTNK_EQUIV


def load_split(name: str):
    path = PTNK_DIR / f"grc_ptnk-ud-{name}.conllu"
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 4 or "-" in cols[0] or "." in cols[0]:
                continue
            if cols[3] == "PUNCT":
                continue
            pairs.append((unicodedata.normalize("NFC", cols[1]),
                          unicodedata.normalize("NFC", cols[2])))
    return pairs


def main():
    if not PTNK_DIR.exists():
        sys.exit(f"PTNK not found: {PTNK_DIR}\nSet PTNK_DIR to a clone of "
                 "github.com/UniversalDependencies/UD_Ancient_Greek-PTNK")

    from dilemma import Dilemma
    d = Dilemma(lang="all", resolve_articles=True)
    equiv = load_equivalences()

    print("PTNK: Septuagint (Codex Alexandrinus), manual lemmas.")
    print("NB: the text (not the annotation) is inside dilemma's corpora;")
    print("this measures convention alignment more than OOV generalization.\n")

    total_n = total_ok = 0
    all_misses = Counter()
    for split in ("train", "dev", "test"):
        pairs = load_split(split)
        preds = d.lemmatize_batch([f for f, _ in pairs])
        ok = 0
        for (form, gold), pred in zip(pairs, preds):
            if pred and _ptnk_match(pred, gold, equiv):
                ok += 1
            else:
                all_misses[(gold, pred)] += 1
        total_n += len(pairs)
        total_ok += ok
        print(f"{split:<6} {len(pairs):>7} tokens  {100 * ok / len(pairs):5.1f}%")
    print("-" * 30)
    print(f"{'ALL':<6} {total_n:>7} tokens  {100 * total_ok / total_n:5.1f}%")
    print("\ntop disagreements (gold <- pred):")
    for (gold, pred), c in all_misses.most_common(15):
        print(f"  {c:5d}  {gold} <- {pred}")


if __name__ == "__main__":
    main()
