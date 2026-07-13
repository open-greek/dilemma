#!/usr/bin/env python3
"""Extract form -> lemma + POS pairs from cog's standardized Pedalion export.

Pedalion Trees (CC BY-SA 4.0) are manually corrected AGDT gold dependency
trees from the GLAUx team. cog's export already DROPS the PROIEL (PRO1/PRO2)
rows; the GORMAN rows survive tagged provenance_tag='gorman' (tag-don't-delete),
and we skip them here for held-out-gold hygiene (Gorman is dilemma's gold).

The value is ~78K manual-gold tokens whose works are NOT in GLAUx/Diorisis
(Sextus Empiricus, Menander, papyri, the Pedalion example-sentence collections,
Mimnermus, Semonides). Because it is AGDT-tagset manual gold - the same tagset
as the treebank POS originals - it feeds BOTH lookup gap-fill and the treebank
POS table (weighted like GLAUx), unlike all-auto OGA which is attestation-only.

Homograph-disambiguation digits (ξένος2) are stripped from the lemma to match
the attestation/eval keying. Output: data/pedalion_pairs.json in the shared
[{form, lemma, pos}] shape.

Run:  python build/extract_pedalion.py
"""
import json
import sys
import unicodedata
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cog_annotations as C  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "pedalion_pairs.json"

# AGDT postag position-1 char -> the glaux/diorisis corpus pos vocabulary.
# u (punct), x (unknown), b (Pedalion experimental) and None are left
# unmapped: the pair still feeds lookup gap-fill, but carries no POS edge.
_AGDT_TO_CORPUS = {
    "n": "noun", "v": "verb", "a": "adj", "p": "pron", "d": "adv",
    "c": "conj", "r": "prep", "l": "article", "g": "intj", "m": "num",
    "i": "intj",
}


def extract(export):
    manifest = C.load_manifest(export)
    seen = OrderedDict()
    kept = gorman = skipped = 0
    for w in manifest["works"]:
        for rec in C.iter_work_tokens(export, w):
            if rec.get("provenance_tag") == "gorman":
                gorman += 1
                continue
            form = rec.get("form") or ""
            raw_lemma = rec.get("lemma") or ""
            if not form or not raw_lemma:
                skipped += 1
                continue
            form = unicodedata.normalize("NFC", form.strip())
            lemma = C.strip_homograph_digits(
                unicodedata.normalize("NFC", raw_lemma.strip()))
            if not form or not C.is_clean_lemma(lemma):
                skipped += 1
                continue
            pos = _AGDT_TO_CORPUS.get(rec.get("pos") or "")
            seen[(form, lemma, pos)] = True
            kept += 1
    pairs = [{"form": f, "lemma": l, "pos": p} for (f, l, p) in seen]
    return pairs, manifest, kept, gorman, skipped


def main():
    export = C.export_dir("pedalion")
    if export is None:
        sys.exit("Pedalion export not found under "
                 f"{C.ANNOTATIONS_ROOT}/pedalion; set DILEMMA_COG_ANNOTATIONS")
    pairs, manifest, kept, gorman, skipped = extract(export)
    json.dump(pairs, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"wrote {len(pairs):,} Pedalion form->lemma+pos pairs to {OUT}")
    print(f"  ({kept:,} tokens kept, {gorman:,} gorman-tagged skipped, "
          f"{skipped:,} unlemmatized skipped)")
    print(f"  pin: {C.pin_line(manifest)}")


if __name__ == "__main__":
    main()
