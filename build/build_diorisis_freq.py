#!/usr/bin/env python3
"""Extract token frequencies from Diorisis Ancient Greek corpus, by genre.

Diorisis (Vatri & McGillivray, 2018) is a 10M-token lemmatized corpus
of Ancient Greek (Homer through 5th c. AD). Each text has genre/date
metadata. Token forms are in Beta Code and must be converted to Unicode.

Reads all Diorisis XML files and counts how many times each accent-stripped
form appears, with per-genre breakdowns matching the glaux_freq.json format.

Output: data/diorisis_freq.json
    {"_total_tokens": N, "_genres": [...],
     "_n_forms": N,
     "forms": {"ανθρωπος": [total, genre1, genre2, ...], ...}}

Usage:
    python build/build_diorisis_freq.py
    python build/build_diorisis_freq.py --diorisis ~/path/to/xml
    python build/build_diorisis_freq.py --stats  # show stats only
"""

import argparse
import json
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from beta_code import beta_code_to_greek
from corpus_freq_key import (
    ELISION_MARKS,
    beta_trailing_paren_is_elision,
    corpus_freq_key,
)

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_DIORISIS_DIR = DATA_DIR / "diorisis" / "xml"
OUTPUT_PATH = DATA_DIR / "diorisis_freq.json"

# Map Diorisis genre labels to the same categories used in glaux_freq.json.
# Diorisis has 10 genres; we collapse to the same ~10 categories as GLAUx.
GENRE_MAP = {
    "Philosophy": "philosophy",
    "Poetry": "poetry",
    "Tragedy": "poetry",
    "Comedy": "poetry",
    "Oratory": "oratory",
    "Narrative": "history",       # Diorisis "Narrative" ~ historical/biographical prose
    "Technical": "science",       # Diorisis "Technical" = medicine, science, engineering
    "Essays": "philosophy",       # Diorisis "Essays" = Plutarch, Lucian, etc.
    "Letters": "epistles",
    "Religion": "religion",
}

# Same genre order as glaux_freq.json for compatibility
GENRE_ORDER = [
    "philosophy", "poetry", "history", "oratory", "science",
    "narrative", "epistles", "religion", "commentary", "other",
]


def is_greek(s):
    """Check if string contains at least one Greek character."""
    return any('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF' for c in s)


def strip_non_greek(s):
    """Remove non-Greek residue while preserving Greek elision marks."""
    return "".join(c for c in s
                   if unicodedata.category(c).startswith("M")
                   or c in ELISION_MARKS
                   or ('\u0370' <= c <= '\u03FF')
                   or ('\u1F00' <= c <= '\u1FFF'))


def betacode_to_unicode(bc):
    """Convert Beta Code form to NFC-normalized Unicode Greek.

    Diorisis stores token forms in Beta Code (e.g. 'qeou\\s' for 'θεούς').
    Strips non-Greek residue while retaining apostrophes used for elision.
    Returns empty string on conversion failure.
    """
    try:
        trailing_elision = beta_trailing_paren_is_elision(bc)
        uni = beta_code_to_greek(bc[:-1] if trailing_elision else bc)
        uni = unicodedata.normalize("NFC", uni)
        cleaned = strip_non_greek(uni)
        if trailing_elision and cleaned:
            cleaned += "\u1fbd"
        return cleaned if cleaned else ""
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="Extract Diorisis token frequencies")
    parser.add_argument("--diorisis", type=Path, default=DEFAULT_DIORISIS_DIR,
                        help="Path to Diorisis xml/ directory")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help="Output path")
    parser.add_argument("--stats", action="store_true",
                        help="Show stats only, don't save")
    args = parser.parse_args()

    t0 = time.time()

    genre_to_idx = {g: i for i, g in enumerate(GENRE_ORDER)}
    n_genres = len(GENRE_ORDER)

    # form_counts[stripped] = [total, philosophy, poetry, history, ...]
    form_counts = defaultdict(lambda: [0] * (1 + n_genres))
    total_tokens = 0
    skipped_non_greek = 0
    skipped_convert = 0
    files_by_genre = Counter()

    xml_files = sorted(args.diorisis.glob("*.xml"))
    if not xml_files:
        print(f"No XML files found in {args.diorisis}")
        print("Download from: https://figshare.com/articles/dataset/"
              "The_Diorisis_Ancient_Greek_Corpus/6187256")
        return

    print(f"Processing {len(xml_files)} Diorisis XML files...")

    for i, xml_file in enumerate(xml_files):
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError:
            continue

        root = tree.getroot()

        # Extract genre from <xenoData><genre>
        genre_el = root.find(".//genre")
        raw_genre = genre_el.text.strip() if genre_el is not None and genre_el.text else ""
        genre = GENRE_MAP.get(raw_genre, "other")
        genre_idx = genre_to_idx.get(genre, genre_to_idx["other"])
        files_by_genre[genre] += 1

        for word in root.findall(".//word"):
            bc_form = word.get("form", "")
            if not bc_form:
                continue

            # Skip empty POS (unlemmatized tokens, often proper nouns or
            # corrupt forms) - they still count as tokens for frequency
            # but we need the Unicode form
            uni_form = betacode_to_unicode(bc_form)
            if not uni_form:
                skipped_convert += 1
                continue

            if not is_greek(uni_form):
                skipped_non_greek += 1
                continue

            stripped = corpus_freq_key(uni_form)
            form_counts[stripped][0] += 1           # total
            form_counts[stripped][1 + genre_idx] += 1  # genre-specific
            total_tokens += 1

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(xml_files)} files, "
                  f"{total_tokens:,} tokens, "
                  f"{len(form_counts):,} unique forms", flush=True)

    elapsed = time.time() - t0
    print(f"\nTotal: {total_tokens:,} tokens, "
          f"{len(form_counts):,} unique stripped forms "
          f"({elapsed:.1f}s)")
    print(f"Skipped: {skipped_convert:,} conversion failures, "
          f"{skipped_non_greek:,} non-Greek")

    print(f"\nGenre distribution:")
    for g in GENRE_ORDER:
        total_g = sum(v[1 + genre_to_idx[g]] for v in form_counts.values())
        print(f"  {g:15s}: {files_by_genre.get(g, 0):>4d} texts, "
              f"{total_g:>10,} tokens")

    if args.stats:
        return

    # Write output
    print(f"\nWriting {args.output}...", end=" ", flush=True)
    output = {
        "_total_tokens": total_tokens,
        "_genres": GENRE_ORDER,
        "_n_forms": len(form_counts),
        "forms": dict(form_counts),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))
    size_mb = args.output.stat().st_size / 1e6
    print(f"{size_mb:.0f} MB ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
