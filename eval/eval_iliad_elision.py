#!/usr/bin/env python3
"""Elided-token lemma agreement against the AGDT Iliad treebank.

Homeric Greek elides constantly (over 10K elided tokens in the Iliad),
and an elided form is exactly where a frequency-ranked fallback picks
the wrong homograph (the historic bugs: elided ὅτ' -> the frequent
ὅτι instead of ὅτε, μ' -> εἷς instead of ἐγώ). This eval scores every
treebank token that carries an elision mark.

Data: the Ancient Greek Dependency Treebank (AGDT / PerseusDL
treebank_data, CC BY-SA) Iliad file, tlg0012.tlg001. Point
ILIAD_TREEBANK at your copy of the treebank XML:

    ILIAD_TREEBANK=/path/to/tlg0012.tlg001...tb.xml \
        python eval/eval_iliad_elision.py

Counting rule: AGDT labels its 35 elided ὅτ'/ὅθ' tokens as ὅτι, but
Homeric ὅτι never elides (it appears as ὅττι or ὅ τι), so a prediction
of ὅτε/ὅθι against gold ὅτι on an elided ὅτ-/ὅθ- form is counted as a
WIN. Both the raw and the adjusted agreement are reported.
"""
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from dilemma import Dilemma, to_monotonic, strip_accents  # noqa: E402

ILIAD_TB = Path(os.environ.get(
    "ILIAD_TREEBANK", Path.home() / "Documents" / "iliad" / "treebank.xml"))

# Every mark that encodes elision in the wild: combining psili (AGDT),
# the spacing apostrophe codepoints, and the koronis.
MARKS = set("̓’ʼ'᾽`ʹ᾿")


def norm(s: str) -> str:
    s = re.sub(r"\d+$", "", s)  # AGDT homograph numbers (ὅς1)
    s = unicodedata.normalize("NFC", s)
    return strip_accents(to_monotonic(s.lower()))


def main():
    if not ILIAD_TB.exists():
        sys.exit(f"Treebank not found: {ILIAD_TB}\n"
                 "Set ILIAD_TREEBANK to your AGDT Iliad tb.xml "
                 "(PerseusDL treebank_data, tlg0012.tlg001).")

    txt = ILIAD_TB.read_text(encoding="utf-8")
    pairs = [(unicodedata.normalize("NFC", m.group(1)), m.group(2))
             for m in re.finditer(r'form="([^"]*)" lemma="([^"]*)"', txt)]
    elided = [(f, g) for f, g in pairs
              if g and any(c in MARKS for c in f)]
    print(f"tokens: {len(pairs)}  elided: {len(elided)}  "
          f"distinct elided forms: {len({f for f, _ in elided})}")

    d = Dilemma(lang="all", resolve_articles=True)
    preds = d.lemmatize_batch([f for f, _ in elided])

    raw = adjusted = 0
    misses = Counter()
    for (form, gold), pred in zip(elided, preds):
        if pred is None:
            misses[(form, gold, pred)] += 1
            continue
        np_, ng = norm(pred), norm(gold)
        if np_ == ng:
            raw += 1
            adjusted += 1
        elif (ng == "οτι" and np_ in ("οτε", "οθι")
                and norm(form)[:2] in ("οτ", "οθ")):
            adjusted += 1  # gold is wrong: Homeric ὅτι never elides
        else:
            misses[(form, gold, pred)] += 1

    n = len(elided)
    print(f"raw agreement:      {raw}/{n} = {100 * raw / n:.1f}%")
    print(f"with ὅτι adjustment: {adjusted}/{n} = {100 * adjusted / n:.1f}%")
    print("top disagreements (form -> pred, gold):")
    for (form, gold, pred), c in misses.most_common(20):
        print(f"  {c:5d}  {form} -> {pred}  (gold: {gold})")


if __name__ == "__main__":
    main()
