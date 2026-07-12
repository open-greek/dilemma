"""Build POS-indexed disambiguation lookup from GLAUx corpus data.

GLAUx (Greek Learner Corpus, Keersmaekers 2021) provides 644K form-lemma pairs
with AGDT morphological tags. This script extracts genuinely ambiguous forms
(where different POS tags lead to different lemmas) into a POS lookup table.

Output: data/glaux_pos_lookup.json
Format: {form: {UPOS: lemma, ...}, ...}

This supplements the treebank_pos_lookup.json (gold, 1.8K entries) and
ag_pos_lookup.json (Wiktionary, 28K entries) with corpus-derived evidence.
"""

import json
import unicodedata
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
GLAUX_PAIRS = SCRIPT_DIR / "data" / "glaux_pairs.json"
OUTPUT_PATH = SCRIPT_DIR / "data" / "glaux_pos_lookup.json"

# Map GLAUx/Wiktionary POS to UPOS
WIKT_TO_UPOS = {
    "noun": "NOUN",
    "verb": "VERB",
    "adj": "ADJ",
    "adv": "ADV",
    "pron": "PRON",
    "num": "NUM",
    "prep": "ADP",
    "conj": "CCONJ",
    "intj": "INTJ",
    "article": "DET",
}

_POLYTONIC_STRIP = {0x0313, 0x0314, 0x0345, 0x0306, 0x0304}
_POLYTONIC_TO_ACUTE = {0x0300, 0x0342}


def to_monotonic(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    out = []
    for ch in nfd:
        cp = ord(ch)
        if cp in _POLYTONIC_STRIP:
            continue
        if cp in _POLYTONIC_TO_ACUTE:
            out.append("\u0301")
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def strip_accents(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(c for c in nfd if unicodedata.category(c) != "Mn"))


def build_lookup():
    print(f"Loading GLAUx pairs from {GLAUX_PAIRS}")
    with open(GLAUX_PAIRS, encoding="utf-8") as f:
        pairs = json.load(f)
    print(f"Total pairs: {len(pairs)}")

    # Collect form_upos_lemmas[form][upos][lemma] = count
    form_upos_lemmas = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    skipped = 0

    _junk_finals = tuple("᾽'’ʼ`ʹ")
    for item in pairs:
        form = item.get("form", "")
        lemma = item.get("lemma", "")
        pos = item.get("pos", "")
        if not form or not lemma or not pos:
            skipped += 1
            continue
        # Junk lemma values are corpus artifacts, not headwords: elided
        # fragments (ἀλλ᾽) and abbreviation overlines (οὐδ̅, U+0305).
        # lemmatize_pos trusts POS-table entries, so junk here becomes
        # output (matches _clean_lemma in build_treebank_pos_lookup.py).
        if lemma.endswith(_junk_finals) \
                or "̅" in unicodedata.normalize("NFD", lemma):
            skipped += 1
            continue
        upos = WIKT_TO_UPOS.get(pos)
        if not upos:
            skipped += 1
            continue
        # Check if it's a proper noun (capitalized lemma, noun POS)
        if upos == "NOUN" and lemma and lemma[0].isupper():
            # Add both NOUN and PROPN entries
            form_upos_lemmas[form]["PROPN"][lemma] += 1
        form_upos_lemmas[form][upos][lemma] += 1

    print(f"Skipped (no POS / no UPOS mapping): {skipped}")
    print(f"Unique forms: {len(form_upos_lemmas)}")

    # Filter to genuinely ambiguous forms
    lookup = {}
    for form, upos_dict in form_upos_lemmas.items():
        resolved = {}
        for upos, lemma_counts in upos_dict.items():
            best_lemma = max(lemma_counts, key=lemma_counts.get)
            resolved[upos] = best_lemma

        # Only keep forms where different UPOS tags map to different lemmas
        unique_lemmas = set(resolved.values())
        if len(unique_lemmas) < 2:
            continue

        lookup[form] = resolved

    print(f"Ambiguous forms (different UPOS -> different lemma): {len(lookup)}")

    # Add lowercase and monotonic variants
    extra = {}
    for form, upos_lemmas in list(lookup.items()):
        lower = form.lower()
        if lower != form and lower not in lookup:
            extra[lower] = upos_lemmas
        mono = to_monotonic(form.lower())
        if mono != form and mono != lower and mono not in lookup:
            extra[mono] = upos_lemmas

    lookup.update(extra)
    print(f"After adding lowercase/monotonic variants: {len(lookup)}")

    # Sort for stable output
    lookup = dict(sorted(lookup.items()))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False, indent=1)

    print(f"\nSaved to {OUTPUT_PATH}")
    return lookup


if __name__ == "__main__":
    build_lookup()
