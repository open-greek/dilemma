#!/usr/bin/env python3
"""Extract form -> lemma pairs from cog's standardized TAGNT export.

TAGNT (STEPBible Translators Amalgamated Greek NT, CC BY 4.0; attribute to
STEPBible.org) amalgamates the Greek NT over 8 editions with TBESG dictionary
lemmas and Robinson morphology. It is a lookup GAP-FILL source: the value is
NT variant spellings (itacisms, TR/Byzantine readings) that map to a known
lemma, plus NT proper nouns. It is NOT used for POS edges - Robinson's native
tagset is not the AGDT/UD tagset the treebank POS table trusts - and NOT a
lemmatization eval (TBESG/Strongs lemma conventions differ from Wiktionary).

TBESG multi-form headwords ("Δαυείδ, Δαυίδ, Δαβίδ") are split on comma; each
variant is emitted so build_lookup_db.py's headword validation keeps whichever
spelling is a real AG headword. As with every gap-fill source, a form is added
only when no earlier source has it, so nothing here overrides Wiktionary/LSJ/
GLAUx/Diorisis/NT.

Output: data/tagnt_pairs.json in the shared [{form, lemma, pos}] shape (pos
best-effort in the corpus vocabulary; ignored by the lookup gap-fill).

Run:  python build/extract_tagnt.py
"""
import json
import re
import sys
import unicodedata
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cog_annotations as C  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "tagnt_pairs.json"

# Robinson category prefix -> the glaux/diorisis corpus pos vocabulary.
_ROBINSON_TO_CORPUS = {
    "N": "noun", "V": "verb", "A": "adj", "ADV": "adv", "PREP": "prep",
    "CONJ": "conj", "T": "article", "P": "pron", "R": "pron", "C": "pron",
    "D": "pron", "K": "pron", "I": "pron", "X": "pron", "Q": "pron",
    "F": "pron", "S": "pron", "PRT": "particle", "INJ": "intj", "NUM": "num",
}
# A single Greek word (drops punctuation-only or THGNT-marker fragments).
_GREEK_WORD = re.compile(r"[Ͱ-Ͽἀ-῿̀-ͯ’]+")


def _corpus_pos(robinson: str) -> str | None:
    if not robinson:
        return None
    head = robinson.split("-")[0]
    return _ROBINSON_TO_CORPUS.get(head)


def extract(export):
    manifest = C.load_manifest(export)
    seen = OrderedDict()
    kept = skipped = 0
    for w in manifest["works"]:
        for rec in C.iter_work_tokens(export, w):
            form = rec.get("form") or ""
            # strip any attached THGNT punctuation/pilcrow around the word
            m = _GREEK_WORD.search(form)
            form = unicodedata.normalize("NFC", m.group(0)) if m else ""
            pos = _corpus_pos(rec.get("pos") or "")
            raw_lemma = rec.get("lemma") or ""
            if not form or not raw_lemma:
                skipped += 1
                continue
            emitted = False
            for variant in raw_lemma.split(","):
                lemma = unicodedata.normalize("NFC", variant.strip())
                if (not lemma or not _GREEK_WORD.fullmatch(lemma)
                        or not C.is_clean_lemma(lemma)):
                    continue
                seen[(form, lemma, pos or "noun")] = True
                emitted = True
            kept += emitted
            skipped += not emitted
    pairs = [{"form": f, "lemma": l, "pos": p} for (f, l, p) in seen]
    return pairs, manifest, kept, skipped


def main():
    export = C.export_dir("tagnt")
    if export is None:
        sys.exit("TAGNT export not found under "
                 f"{C.ANNOTATIONS_ROOT}/tagnt; set DILEMMA_COG_ANNOTATIONS")
    pairs, manifest, kept, skipped = extract(export)
    json.dump(pairs, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"wrote {len(pairs):,} TAGNT form->lemma pairs to {OUT}")
    print(f"  ({kept:,} tokens kept, {skipped:,} skipped)")
    print(f"  pin: {C.pin_line(manifest)}")


if __name__ == "__main__":
    main()
