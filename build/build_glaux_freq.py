#!/usr/bin/env python3
"""Extract token frequencies from GLAUx corpus, optionally by genre.

Reads all GLAUx XML files and counts how many times each accent-stripped
form appears. Outputs a JSON file mapping stripped forms to counts,
plus per-genre breakdowns.

Output: data/glaux_freq.json
    {"_total": N, "_genres": ["Philosophy", ...],
     "forms": {"ανθρωπος": [total, philosophy, history, ...], ...}}

Usage:
    python build/build_glaux_freq.py
    python build/build_glaux_freq.py --glaux ~/Documents/glaux/xml
"""

import argparse
import csv
import json
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_GLAUX_DIR = Path.home() / "Documents" / "glaux" / "xml"
DEFAULT_METADATA = Path.home() / "Documents" / "glaux" / "metadata.txt"
OUTPUT_PATH = DATA_DIR / "glaux_freq.json"

# Genre groups for the Vesuvius use case.
# Collapse 40+ fine-grained genres into ~10 useful categories.
GENRE_MAP = {
    "Philosophy": "philosophy",
    "Philosophic Dialogue": "philosophy",
    "Epic poetry": "poetry",
    "Lyric poetry": "poetry",
    "Tragedy": "poetry",
    "Comedy": "poetry",
    "Scientific Poetry": "poetry",
    "Religious Poetry": "poetry",
    "History": "history",
    "Religious History": "history",
    "Biography": "history",
    "Oratory": "oratory",
    "Rhetoric": "oratory",
    "Medicine": "science",
    "Mathematics": "science",
    "Engineering": "science",
    "Physics": "science",
    "Biology": "science",
    "Astronomy/Astrology": "science",
    "Alchemy": "science",
    "Geography": "science",
    "Narrative": "narrative",
    "Religious Narrative": "narrative",
    "Mythography": "narrative",
    "Paradoxography": "narrative",
    "Epistolography": "epistles",
    "Religious Epistle": "epistles",
    "Theology": "religion",
    "Religious Wisdom": "religion",
    "Religious Prophecy": "religion",
    "Oracle": "religion",
    "Commentary": "commentary",
    "Language": "commentary",
    "Dialogue": "philosophy",
    "Polyhistory": "other",
    "Art": "other",
    "Military": "other",
    "Music": "other",
    "Oneirocritic": "other",
}

GENRE_ORDER = [
    "philosophy", "poetry", "history", "oratory", "science",
    "narrative", "epistles", "religion", "commentary", "other",
]


def strip_accents(s):
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(c for c in nfd if unicodedata.category(c) != "Mn"))


def is_greek(s):
    return any('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF' for c in s)


def load_metadata(path):
    """Load TLG -> genre mapping from metadata.txt."""
    tlg_to_genre = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tlg = row.get("TLG", "").strip()
            genre = row.get("GENRE_STANDARD", "").strip()
            if tlg and genre:
                tlg_to_genre[tlg] = GENRE_MAP.get(genre, "other")
    return tlg_to_genre


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glaux", type=Path, default=DEFAULT_GLAUX_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output (default glaux_freq.json, or "
                             "glaux_freq_commercial.json with --exclude-nc)")
    parser.add_argument("--exclude-nc", action="store_true",
                        help="Drop NonCommercial GLAUx texts (commercial-safe)")
    args = parser.parse_args()
    if args.output is None:
        args.output = DATA_DIR / ("glaux_freq_commercial.json"
                                  if args.exclude_nc else "glaux_freq.json")

    t0 = time.time()

    # Load genre metadata
    print("Loading metadata...", end=" ", flush=True)
    tlg_to_genre = load_metadata(args.metadata)
    print(f"{len(tlg_to_genre)} texts")

    nc_stems = frozenset()
    if args.exclude_nc:
        from nc_filter import nc_glaux_stems
        nc_stems = nc_glaux_stems(args.metadata)

    genre_to_idx = {g: i for i, g in enumerate(GENRE_ORDER)}
    n_genres = len(GENRE_ORDER)

    # Count frequencies
    # form_counts[stripped] = [total, philosophy, poetry, history, ...]
    form_counts = defaultdict(lambda: [0] * (1 + n_genres))
    total_tokens = 0
    files_by_genre = Counter()

    xml_files = sorted(args.glaux.glob("*.xml"))
    if nc_stems:
        before = len(xml_files)
        xml_files = [x for x in xml_files if x.stem not in nc_stems]
        print(f"Commercial-safe: excluded {before - len(xml_files)} "
              f"NonCommercial GLAUx text(s)")
    print(f"Processing {len(xml_files)} XML files...")

    for i, xml_file in enumerate(xml_files):
        # Map filename to TLG ID
        tlg = xml_file.stem  # e.g. "0012-001"
        genre = tlg_to_genre.get(tlg, "other")
        genre_idx = genre_to_idx.get(genre, genre_to_idx["other"])
        files_by_genre[genre] += 1

        try:
            tree = ET.parse(xml_file)
        except ET.ParseError:
            continue

        for word in tree.findall(".//word"):
            postag = word.get("postag", "")
            if postag and postag[0] == "u":
                continue  # skip punctuation

            form = word.get("form", "")
            if not form or not is_greek(form):
                continue

            stripped = strip_accents(
                unicodedata.normalize("NFC", form).lower())
            form_counts[stripped][0] += 1           # total
            form_counts[stripped][1 + genre_idx] += 1  # genre-specific
            total_tokens += 1

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(xml_files)} files, "
                  f"{total_tokens:,} tokens, "
                  f"{len(form_counts):,} unique forms", flush=True)

    print(f"\nTotal: {total_tokens:,} tokens, "
          f"{len(form_counts):,} unique stripped forms")
    print(f"\nGenre distribution:")
    for g in GENRE_ORDER:
        total_g = sum(v[1 + genre_to_idx[g]] for v in form_counts.values())
        print(f"  {g:15s}: {files_by_genre[g]:>4d} texts, "
              f"{total_g:>10,} tokens")

    # Write output
    print(f"\nWriting {args.output}...", end=" ", flush=True)
    output = {
        "_total_tokens": total_tokens,
        "_genres": GENRE_ORDER,
        "_n_forms": len(form_counts),
        "forms": {k: v for k, v in form_counts.items()},
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))
    size_mb = args.output.stat().st_size / 1e6
    print(f"{size_mb:.0f} MB ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
