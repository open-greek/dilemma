#!/usr/bin/env python3
"""Extract form -> lemma + POS pairs from the openly-licensed Nestle 1904
lowfat Greek New Testament (Clear-Bible/macula-greek, CC BY 4.0; the Nestle
1904 base text is public domain, 1904).

This is the openly-licensed replacement for the dropped CC BY-NC-SA PROIEL NT,
supplying Koine NT form -> lemma evidence. Output: data/nt_pairs.json in the
same shape as glaux_pairs.json / diorisis_pairs.json ([{form, lemma, pos}]),
so it feeds both build_lookup_db.py (gap-fill) and
build/build_treebank_pos_lookup.py (POS-keyed disambiguation).

Run:  python build/extract_nt.py [--lowfat DIR]
The lowfat XML comes from a sparse checkout of Clear-Bible/macula-greek:
  git clone --filter=blob:none --sparse https://github.com/Clear-Bible/macula-greek.git
  cd macula-greek && git sparse-checkout set Nestle1904/lowfat
"""
import argparse
import json
import unicodedata
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

DEFAULT_LOWFAT = (Path.home() / "Documents" / "macula-greek"
                  / "Nestle1904" / "lowfat")
OUT = Path(__file__).resolve().parent.parent / "data" / "nt_pairs.json"

# lowfat @class -> the glaux/diorisis pos vocabulary (mapped to UPOS downstream)
_CLASS_TO_POS = {
    "noun": "noun", "verb": "verb", "adj": "adj", "adv": "adv",
    "pron": "pron", "det": "article", "art": "article", "prep": "prep",
    "conj": "conj", "ptcl": "particle", "particle": "particle",
    "num": "num", "intj": "intj", "interj": "intj",
}


def extract(lowfat_dir: Path):
    """Return a deduped list of {form, lemma, pos} from the lowfat XML."""
    seen = OrderedDict()
    for xml in sorted(lowfat_dir.glob("*.xml")):
        for w in ET.parse(xml).getroot().iter("w"):
            form = (w.get("unicode") or w.text or "")
            lemma = (w.get("lemma") or "")
            pos = _CLASS_TO_POS.get((w.get("class") or "").strip().lower())
            if not form or not lemma or not pos:
                continue
            form = unicodedata.normalize("NFC", form.strip())
            lemma = unicodedata.normalize("NFC", lemma.strip())
            if form and lemma:
                seen[(form, lemma, pos)] = True
    return [{"form": f, "lemma": l, "pos": p} for (f, l, p) in seen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lowfat", type=Path, default=DEFAULT_LOWFAT)
    args = ap.parse_args()
    if not args.lowfat.exists():
        raise SystemExit(f"lowfat dir not found: {args.lowfat}")
    pairs = extract(args.lowfat)
    json.dump(pairs, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"wrote {len(pairs):,} NT form->lemma+pos pairs to {OUT}")


if __name__ == "__main__":
    main()
