#!/usr/bin/env python3
"""Extract form -> lemma + POS pairs from cog's standardized PTNK export.

PTNK (UD_Ancient_Greek-PTNK, CC BY-SA 4.0) is the Septuagint (Genesis, Ruth)
per the Codex Alexandrinus, with MANUAL native lemma + morphology annotation
(only its syntax was projected from Hebrew - no PROIEL). It supplies Koine /
LXX form -> lemma evidence, notably the Hebrew proper nouns dilemma garbles.

Only the TRAIN split is extracted; dev + test are held out so eval/eval_ptnk.py
stays an honest LXX benchmark. Output: data/ptnk_pairs.json in the shared
[{form, lemma, pos}] shape (pos in the glaux/diorisis corpus vocabulary), so it
feeds build_lookup_db.py (gap-fill) and build/build_treebank_pos_lookup.py.

The lemma is PTNK's native convention verbatim (cog only encoding-normalizes).
Because the gap-fill block in build_lookup_db.py adds a form only when it is
absent (`if form not in ag`), importing all train pairs never overrides a
Wiktionary/LSJ/GLAUx/Diorisis entry - it only fills genuine gaps. PTNK is
manual gold, so its POS edges are weighted like GLAUx/NT (not like all-auto OGA).

Run:  python build/extract_ptnk.py
"""
import json
import sys
import unicodedata
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cog_annotations as C  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "ptnk_pairs.json"

# UD UPOS -> the glaux/diorisis corpus pos vocabulary (mapped to UPOS again
# downstream). PROPN folds to noun (the POS table has no PROPN slot; the
# proper-noun value survives in the lookup gap-fill, which ignores pos).
_UD_TO_CORPUS = {
    "NOUN": "noun", "PROPN": "noun", "VERB": "verb", "AUX": "verb",
    "ADJ": "adj", "ADV": "adv", "PRON": "pron", "DET": "article",
    "ADP": "prep", "CCONJ": "conj", "SCONJ": "conj", "PART": "particle",
    "NUM": "num", "INTJ": "intj",
}


def extract(export):
    manifest = C.load_manifest(export)
    seen = OrderedDict()
    kept = skipped = 0
    for w in manifest["works"]:
        for rec in C.iter_work_tokens(export, w):
            if rec.get("split") != "train":
                continue
            lemma = rec.get("lemma")
            form = rec.get("form")
            pos = _UD_TO_CORPUS.get(rec.get("pos") or "")
            if not lemma or not form or not pos:
                skipped += 1
                continue
            form = unicodedata.normalize("NFC", form.strip())
            lemma = unicodedata.normalize("NFC", lemma.strip())
            if not form or not C.is_clean_lemma(lemma):
                skipped += 1
                continue
            seen[(form, lemma, pos)] = True
            kept += 1
    pairs = [{"form": f, "lemma": l, "pos": p} for (f, l, p) in seen]
    return pairs, manifest, kept, skipped


def main():
    export = C.export_dir("ptnk")
    if export is None:
        sys.exit("PTNK export not found under "
                 f"{C.ANNOTATIONS_ROOT}/ptnk; set DILEMMA_COG_ANNOTATIONS")
    pairs, manifest, kept, skipped = extract(export)
    json.dump(pairs, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"wrote {len(pairs):,} PTNK train form->lemma+pos pairs to {OUT}")
    print(f"  ({kept:,} train tokens kept, {skipped:,} skipped)")
    print(f"  pin: {C.pin_line(manifest)}")


if __name__ == "__main__":
    main()
