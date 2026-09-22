#!/usr/bin/env python3
"""Extract form->lemma pairs from Diorisis Ancient Greek corpus.

Diorisis (Vatri & McGillivray, 2018) is a 10M-token lemmatized corpus
of Ancient Greek (Homer through 5th c. AD). Lemma accuracy is ~91.4%
(vs GLAUx's 98.8%), so pairs from this corpus should be treated as
lower-confidence and only used when they don't conflict with existing
entries from Wiktionary, LSJ, or GLAUx.

Each token has:
- form: Beta Code surface form
- lemma entry: Unicode Greek lemma (dictionary headword)
- POS: noun, verb, adjective, pronoun, etc.
- morph: morphological analysis string(s)

Output: data/diorisis_pairs.json - list of {form, lemma, pos} dicts,
matching the format of data/glaux_pairs.json.

Usage:
    python build/build_diorisis_pairs.py
    python build/build_diorisis_pairs.py --diorisis ~/path/to/xml
    python build/build_diorisis_pairs.py --stats  # show stats only
"""

import argparse
import json
import unicodedata
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from beta_code import beta_code_to_greek
from corpus_freq_key import beta_trailing_paren_is_elision

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_DIORISIS_DIR = DATA_DIR / "diorisis" / "xml"
OUTPUT_PATH = DATA_DIR / "diorisis_pairs.json"

# Map Diorisis POS labels to Wiktionary-style POS (matching GLAUx pairs format)
DIORISIS_POS_MAP = {
    "noun": "noun",
    "verb": "verb",
    "adjective": "adj",
    "adverb": "adv",
    "pronoun": "pron",
    "article": "article",
    "particle": "particle",
    "preposition": "prep",
    "conjunction": "conj",
    "interjection": "intj",
    "proper": "noun",       # proper nouns -> noun
    "numeral": "num",
}


def nfc(s):
    return unicodedata.normalize("NFC", s)


def is_greek(s):
    return any('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF' for c in s)


def is_pure_greek(s):
    """Check that every character is Greek letter or combining mark."""
    for c in s:
        cat = unicodedata.category(c)
        if cat.startswith("L"):
            if not ('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF'):
                return False
        elif cat.startswith("M"):
            pass  # combining marks ok
        else:
            return False
    return len(s) > 0


def strip_non_greek(s):
    """Remove non-Greek, non-combining-mark characters (quotes, apostrophes)."""
    return "".join(c for c in s
                   if unicodedata.category(c).startswith("M")
                   or ('\u0370' <= c <= '\u03FF')
                   or ('\u1F00' <= c <= '\u1FFF'))


def betacode_to_unicode(bc):
    """Convert Beta Code form to NFC-normalized Unicode Greek.

    Strips any non-Greek residue (apostrophes from elision, quotes). A final
    ``)`` that Diorisis writes for elision (``di)``, ``par)``) is dropped
    before conversion; converted, it would be a breathing on the last vowel
    or consonant (``δἰ`` for δι’).
    """
    try:
        if beta_trailing_paren_is_elision(bc):
            bc = bc[:-1]
        uni = beta_code_to_greek(bc)
        uni = nfc(uni)
        # Strip non-Greek characters (apostrophes, quotes from elision)
        cleaned = strip_non_greek(uni)
        return cleaned if cleaned else ""
    except Exception:
        return ""


def parse_morph(morph_str, pos):
    """Parse Diorisis morph string into Dilemma-style tag list.

    Diorisis morph strings look like:
        'masc acc pl'
        'pres subj act 1st sg (attic epic doric)'
        'indeclform (particle)'
        'fem gen sg (attic doric aeolic)'

    We extract the main features, ignoring dialect markers in parentheses.
    """
    if not morph_str:
        return []

    # Strip dialect info in parentheses
    clean = morph_str
    if "(" in clean:
        clean = clean[:clean.index("(")].strip()

    parts = clean.lower().split()
    tags = []

    # Gender
    for p in parts:
        if p in ("masc", "masc/neut"):
            tags.append("masculine")
            break
        elif p == "fem":
            tags.append("feminine")
            break
        elif p == "neut":
            tags.append("neuter")
            break

    # Number
    for p in parts:
        if p == "sg":
            tags.append("singular")
            break
        elif p == "pl":
            tags.append("plural")
            break
        elif p == "dual":
            tags.append("dual")
            break

    # Case
    case_map = {
        "nom": "nominative", "gen": "genitive", "dat": "dative",
        "acc": "accusative", "voc": "vocative",
        "nom/voc/acc": "nominative",  # take first
        "nom/voc": "nominative",
        "nom/acc": "nominative",
        "gen/dat": "genitive",
    }
    for p in parts:
        if p in case_map:
            tags.append(case_map[p])
            break

    # Verbal features: tense
    tense_map = {
        "pres": "present", "imperf": "imperfect", "aor": "aorist",
        "fut": "future", "perf": "perfect", "plup": "pluperfect",
    }
    for p in parts:
        if p in tense_map:
            tags.append(tense_map[p])
            break

    # Mood
    mood_map = {
        "ind": "indicative", "subj": "subjunctive", "opt": "optative",
        "imperat": "imperative", "inf": "infinitive", "part": "participle",
    }
    for p in parts:
        if p in mood_map:
            tags.append(mood_map[p])
            break

    # Voice
    voice_map = {
        "act": "active", "mid": "middle", "pass": "passive",
        "mp": "middle",  # medio-passive
    }
    for p in parts:
        if p in voice_map:
            tags.append(voice_map[p])
            break

    # Person
    person_map = {
        "1st": "first-person", "2nd": "second-person", "3rd": "third-person",
    }
    for p in parts:
        if p in person_map:
            tags.append(person_map[p])
            break

    return tags


def extract_diorisis(diorisis_dir, stats_only=False):
    """Extract form->lemma pairs from Diorisis XML files."""
    xml_files = sorted(Path(diorisis_dir).glob("*.xml"))
    if not xml_files:
        print(f"No XML files found in {diorisis_dir}")
        print("Download from: https://figshare.com/articles/dataset/"
              "The_Diorisis_Ancient_Greek_Corpus/6187256")
        return []

    print(f"Processing {len(xml_files)} Diorisis files...")

    t0 = time.time()
    pairs = []
    seen = set()  # (form, lemma) dedup
    total_tokens = 0
    skipped_no_lemma = 0
    skipped_non_greek = 0
    skipped_convert = 0
    skipped_dup = 0

    pos_counts = Counter()
    has_nominal = 0
    has_verbal = 0

    for i, xml_file in enumerate(xml_files):
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError:
            continue

        for word in tree.findall(".//word"):
            total_tokens += 1

            bc_form = word.get("form", "")
            lemma_el = word.find("lemma")

            if lemma_el is None:
                skipped_no_lemma += 1
                continue

            entry = lemma_el.get("entry", "")
            raw_pos = lemma_el.get("POS", "")

            if not bc_form or not entry:
                skipped_no_lemma += 1
                continue

            # Convert Beta Code form to Unicode
            form = betacode_to_unicode(bc_form)
            if not form:
                skipped_convert += 1
                continue

            # NFC normalize lemma
            lemma = nfc(entry)

            if not is_greek(form) or not is_greek(lemma):
                skipped_non_greek += 1
                continue

            # Map POS
            pos = DIORISIS_POS_MAP.get(raw_pos)

            # Dedup by (form, lemma)
            key = (form, lemma)
            if key in seen:
                skipped_dup += 1
                continue
            seen.add(key)

            # Parse morphological tags from first analysis
            tags = []
            analyses = lemma_el.findall("analysis")
            if analyses:
                morph_str = analyses[0].get("morph", "")
                tags = parse_morph(morph_str, pos)

            pair = {"form": form, "lemma": lemma}
            if pos:
                pair["pos"] = pos
                pos_counts[pos] += 1
            if tags:
                pair["tags"] = tags

                # Count morphological coverage
                tag_set = set(tags)
                if tag_set & {"masculine", "feminine", "neuter"}:
                    has_nominal += 1
                if tag_set & {"present", "imperfect", "aorist", "future",
                              "perfect", "pluperfect"}:
                    has_verbal += 1

            pairs.append(pair)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(xml_files)} files, "
                  f"{len(pairs):,} pairs", flush=True)

    elapsed = time.time() - t0
    print(f"\nTotal tokens: {total_tokens:,} ({elapsed:.1f}s)")
    print(f"Skipped: {skipped_no_lemma:,} no lemma, "
          f"{skipped_convert:,} conversion failures, "
          f"{skipped_non_greek:,} non-Greek, "
          f"{skipped_dup:,} duplicates")
    print(f"Unique pairs: {len(pairs):,}")

    print(f"\nPOS distribution:")
    for pos, count in pos_counts.most_common():
        print(f"  {pos:10s}: {count:,}")

    print(f"\nMorphological coverage:")
    print(f"  With nominal features (G+N+C): {has_nominal:,}")
    print(f"  With verbal features (T+M+V):  {has_verbal:,}")

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Extract Diorisis form-lemma pairs")
    parser.add_argument("--diorisis", type=str,
                        default=str(DEFAULT_DIORISIS_DIR),
                        help="Path to Diorisis xml/ directory")
    parser.add_argument("--stats", action="store_true",
                        help="Show stats only, don't save")
    parser.add_argument("--output", type=str,
                        default=str(OUTPUT_PATH),
                        help="Output path")
    args = parser.parse_args()

    pairs = extract_diorisis(args.diorisis, stats_only=args.stats)

    if not args.stats and pairs:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False)
        size_mb = out.stat().st_size / (1024 * 1024)
        print(f"\nSaved {len(pairs):,} pairs to {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
