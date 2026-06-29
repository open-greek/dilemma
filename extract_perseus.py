#!/usr/bin/env python3
"""Extract form->lemma pairs from the Ancient Greek Dependency Treebank (AGDT).

Sourced from the AGDT ORIGINAL (the `.tb.xml` files in PerseusDL/treebank_data),
which is CC BY-SA 3.0 US -- openly licensed. The Universal Dependencies
repackaging (UD_Ancient_Greek-Perseus) is an auto-conversion of this same data
but is relicensed CC BY-NC-SA, so we deliberately do NOT use it; the form+lemma
content here is the permissive AGDT original. Authors include Sophocles,
Aeschylus, Homer, Hesiod, Herodotus, Thucydides, Plutarch, Polybius, Athenaeus
(the 33 Greek works, ~550K tokens).

Source: https://github.com/PerseusDL/treebank_data  (v2.1/Greek/texts/*.tb.xml)
License: CC BY-SA 3.0 US

Outputs:
    data/perseus_pairs.json - list of {form, lemma, pos} dicts

Usage:
    # clone the treebank, then point DILEMMA_AGDT_DIR at it:
    DILEMMA_AGDT_DIR=/path/to/treebank_data python extract_perseus.py
"""

import json
import os
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from dilemma.form_sanitize import sanitize_form  # noqa: E402
from convert_treebank import convert_postag       # noqa: E402

DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_PATH = DATA_DIR / "perseus_pairs.json"

# AGDT original (PerseusDL/treebank_data). Override with $DILEMMA_AGDT_DIR.
AGDT_DIR = Path(os.environ.get(
    "DILEMMA_AGDT_DIR", str(DATA_DIR / "treebanks" / "treebank_data")))

# UPOS tags whose form->lemma pairs we skip (not lexical Greek vocabulary).
SKIP_UPOS = {"PUNCT", "NUM", "X", "SYM"}

# Map UPOS to simpler POS labels (matching GLAUx/Diorisis pair format).
UPOS_TO_POS = {
    "NOUN": "noun", "VERB": "verb", "ADJ": "adj", "ADV": "adv",
    "PRON": "pron", "DET": "det", "ADP": "prep", "CCONJ": "conj",
    "SCONJ": "conj", "PART": "particle", "INTJ": "intj", "AUX": "verb",
    "PROPN": "noun",
}


def _is_greek(s: str) -> bool:
    return any("Ͱ" <= c <= "Ͽ" or "ἀ" <= c <= "῿" for c in s)


def _normalize_nfc(s: str) -> str:
    # NFC + fix misplaced combining breathings (AGDT encodes elision/aphaeresis
    # with a combining psili U+0313); see dilemma/form_sanitize.
    return sanitize_form(unicodedata.normalize("NFC", s))


def _greek_treebank_files(agdt_dir: Path) -> list[Path]:
    """The 33 Greek AGDT works, preferring the newest release present."""
    for ver in ("v2.1", "v2.0", "v1.6"):
        texts = agdt_dir / ver / "Greek" / "texts"
        files = sorted(texts.glob("*.tb.xml")) if texts.exists() else []
        if files:
            return files
    # fallback: any *.tb.xml under a Greek/ dir, deduped by filename stem
    seen, out = set(), []
    for p in sorted(agdt_dir.glob("**/Greek/**/*.tb.xml")):
        if p.name not in seen:
            seen.add(p.name)
            out.append(p)
    return out


def parse_agdt_xml(path: Path) -> list[dict]:
    """Yield {form, lemma, pos, upos} from one AGDT .tb.xml file.

    AGDT word elements look like:
      <word id="1" form="Κάδμου" lemma="Κάδμος" postag="n-s---mg-" .../>
    The 9-char postag is mapped to UD via convert_treebank.convert_postag
    (which also flags capitalized nouns as PROPN and punctuation as PUNCT).
    """
    pairs = []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return pairs
    for w in root.iter("word"):
        form = w.get("form")
        lemma = w.get("lemma")
        postag = w.get("postag") or ""
        if not form or not lemma or lemma == "_":
            continue
        form = _normalize_nfc(form)
        lemma = _normalize_nfc(lemma)
        if not _is_greek(form):
            continue
        upos, _feats = convert_postag(postag, lemma)
        if upos in SKIP_UPOS:
            continue
        pos = UPOS_TO_POS.get(upos, upos.lower())
        pairs.append({"form": form, "lemma": lemma, "pos": pos, "upos": upos})
    return pairs


def extract_perseus_pairs(agdt_dir: Path = AGDT_DIR) -> list[dict]:
    """Deduplicated form-lemma pairs across all Greek AGDT works."""
    files = _greek_treebank_files(agdt_dir)
    if not files:
        print(f"Error: no Greek *.tb.xml files found under {agdt_dir}")
        return []

    all_pairs = []
    for path in files:
        raw = parse_agdt_xml(path)
        all_pairs.extend(raw)
    print(f"  {len(files)} works, {len(all_pairs):,} tokens (post-filter)")

    # Deduplicate: per (form, lemma) keep the most common POS; per form keep
    # the most frequent lemma.
    pair_counts: Counter = Counter()
    pair_pos: dict = {}
    for p in all_pairs:
        key = (p["form"], p["lemma"])
        pair_counts[key] += 1
        pair_pos.setdefault(key, Counter())[p["pos"]] += 1

    form_best: dict = {}
    for (form, lemma), count in pair_counts.items():
        best_pos = pair_pos[(form, lemma)].most_common(1)[0][0]
        if form not in form_best or count > form_best[form][2]:
            form_best[form] = (lemma, best_pos, count)

    result = [{"form": form, "lemma": lemma, "pos": pos}
              for form, (lemma, pos, _c) in sorted(form_best.items())]

    n_propn = sum(1 for p in all_pairs if p["upos"] == "PROPN")
    print(f"  Proper nouns in corpus: {n_propn:,}")
    print(f"  Unique form->lemma pairs: {len(result):,}")
    return result


def main():
    if not AGDT_DIR.exists():
        print(f"Error: {AGDT_DIR} not found")
        print("Clone the AGDT original from "
              "https://github.com/PerseusDL/treebank_data and set "
              "$DILEMMA_AGDT_DIR (CC BY-SA 3.0 US, openly licensed).")
        return

    print(f"AGDT treebank ({AGDT_DIR}):")
    pairs = extract_perseus_pairs()
    if not pairs:
        return

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=0)
    print(f"  -> {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
