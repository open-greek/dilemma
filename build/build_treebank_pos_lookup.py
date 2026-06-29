"""Build POS-indexed disambiguation lookup from gold treebank data.

Reads CoNLL-U files from UD_Ancient_Greek-Perseus, UD_Ancient_Greek-PROIEL,
DiGreC treebanks, and Gorman AGDT XML dependency trees. Extracts genuinely
ambiguous forms: same surface form maps to different lemmas depending on
UPOS tag.

Output: data/treebank_pos_lookup.json
Format: {form: {UPOS: lemma, ...}, ...}

Only forms that are genuinely ambiguous (2+ distinct UPOS->lemma mappings)
are included. Monotonic and lowercase variants are added for lookup cascade.
"""

import json
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

TREEBANKS_DIR = Path(__file__).parent.parent / "data" / "treebanks"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "treebank_pos_lookup.json"

# Openly licensed sources only. UD_Ancient_Greek-Perseus and -PROIEL are
# CC BY-NC-SA (NonCommercial) and are NOT used; the AGDT POS data comes from the
# Perseus original (.tb.xml, CC BY-SA 3.0 US) instead. DiGreC is dropped pending
# license verification.
import os
AGDT_DIR = Path(os.environ.get(
    "DILEMMA_AGDT_DIR", str(TREEBANKS_DIR / "treebank_data")))
TREEBANK_DIRS: list = []   # no CoNLL-U sources (the NC UD treebanks are excluded)

GORMAN_DIR = TREEBANKS_DIR / "Greek-Dependency-Trees" / "xml versions"

# AGDT POS code (position 1 of postag) -> UD UPOS
_AGDT_TO_UPOS = {
    "n": "NOUN", "v": "VERB", "a": "ADJ", "p": "PRON",
    "d": "ADV", "c": "CCONJ", "r": "ADP", "l": "DET",
    "g": "INTJ", "m": "NUM", "x": "X", "u": "PUNCT",
}

# Reuse Dilemma's monotonic conversion
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


def parse_conllu(path: Path):
    """Yield (form, lemma, upos) tuples from a CoNLL-U file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 4:
                continue
            tok_id = cols[0]
            # Skip multiword tokens (1-2) and empty nodes (1.1)
            if "-" in tok_id or "." in tok_id:
                continue
            form = cols[1]
            lemma = cols[2]
            upos = cols[3]
            # Skip punctuation
            if upos == "PUNCT":
                continue
            yield form, lemma, upos


def parse_gorman_xml(path: Path):
    """Yield (form, lemma, upos) from a Gorman AGDT XML treebank file."""
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return
    for word in tree.findall(".//word"):
        form = word.get("form", "")
        lemma = word.get("lemma", "")
        postag = word.get("postag", "")
        if not form or not lemma or not postag or len(postag) < 1:
            continue
        upos = _AGDT_TO_UPOS.get(postag[0], "")
        if not upos or upos == "PUNCT":
            continue
        yield form, lemma, upos


def build_lookup():
    # Collect all (form, upos) -> {lemma: count} from treebanks
    # form_upos_lemmas[form][upos][lemma] = count
    form_upos_lemmas = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    total_tokens = 0

    # CoNLL-U treebanks
    for treebank_dir in TREEBANK_DIRS:
        if not treebank_dir.exists():
            print(f"  Skipping {treebank_dir.name} (not found)")
            continue
        conllu_files = sorted(treebank_dir.glob("*.conllu"))
        for f in conllu_files:
            count = 0
            for form, lemma, upos in parse_conllu(f):
                form_upos_lemmas[form][upos][lemma] += 1
                count += 1
            total_tokens += count
            print(f"  {f.name}: {count} tokens")

    # AGDT original (.tb.xml, CC BY-SA 3.0 US) - the openly licensed Perseus
    # POS source, replacing the NonCommercial UD_Ancient_Greek-Perseus.
    agdt_files = []
    for _ver in ("v2.1", "v2.0", "v1.6"):
        _texts = AGDT_DIR / _ver / "Greek" / "texts"
        if _texts.exists():
            agdt_files = sorted(_texts.glob("*.tb.xml"))
            break
    if agdt_files:
        agdt_tokens = 0
        for f in agdt_files:
            for form, lemma, upos in parse_gorman_xml(f):
                form_upos_lemmas[form][upos][lemma] += 1
                agdt_tokens += 1
        total_tokens += agdt_tokens
        print(f"  AGDT (Perseus original): {len(agdt_files)} files, "
              f"{agdt_tokens:,} tokens")
    else:
        print("  Skipping AGDT (not found; set DILEMMA_AGDT_DIR)")

    # Gorman AGDT XML trees
    if GORMAN_DIR.exists():
        gorman_tokens = 0
        gorman_files = sorted(GORMAN_DIR.glob("*.xml"))
        for f in gorman_files:
            count = 0
            for form, lemma, upos in parse_gorman_xml(f):
                form_upos_lemmas[form][upos][lemma] += 1
                count += 1
            gorman_tokens += count
        total_tokens += gorman_tokens
        print(f"  Gorman trees: {len(gorman_files)} files, "
              f"{gorman_tokens:,} tokens")
    else:
        print(f"  Skipping Gorman trees (not found)")

    print(f"\nTotal tokens: {total_tokens}")
    print(f"Unique forms: {len(form_upos_lemmas)}")

    # Filter to genuinely ambiguous forms:
    # A form is ambiguous if it has multiple DISTINCT (upos -> lemma) mappings,
    # meaning different UPOS tags lead to different lemmas.
    # For each UPOS, pick the most frequent lemma.
    lookup = {}
    for form, upos_dict in form_upos_lemmas.items():
        # For each UPOS, pick the most frequent lemma
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
