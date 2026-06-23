#!/usr/bin/env python3
"""Build data/lsj10_headwords.json from the LSJ10 headword database.

Reads the canonical LSJ 10th ed. (Liddell-Scott-Jones) headword list shipped in
an LSJ10 headword SQLite database (table
`headwords`, 122,572 rows) and emits a flat list of clean, single-token, NFC
polytonic Ancient Greek headwords for dilemma to fold into the AG lookup as
self-maps.

The shape mirrors data/wip_headwords.json and data/vlg_headwords.json: a JSON
array of strings. build_lookup_db.py adds the entries not already present as
AG self-maps (headword -> itself), exactly like the VLG and WiP blocks.

Source (`headwords.sqlite`, table `headwords`):
    columns: id, canonical_key, headword, text, fold, initial, clean
We read the `headword` column (the display lemma) for EVERY row. The `clean`
column is an INTEGER boolean, NOT a cleaned-text field, and it is derived from
the `canonical_key`, not the `headword`: clean=1 means the *key* has no leading
hyphen / editorial brackets. It does not reliably describe the `headword`:
  - 121 clean=1 rows carry a bracketed or gloss `headword`
    ("()σαλακας", "δάνας (B)", full Hesychius glosses with spaces);
  - 18 clean=0 rows have an already-repaired `headword`
    ("δηριάζομαι" with key "[δηρ]ιάζομαι", "τυπογράφος", "ἐπιπολᾷ", ...).
Filtering clean=1 would silently drop those 18 good lemmas while still admitting
the 121 dirty ones, so we ignore the flag and clean every `headword` ourselves.

The `headword` carries LSJ typography: an internal morpheme-boundary hyphen
("Αἰγί-πᾱν"), vowel-quantity macron/breve ("Αἰγῑν-αῖος"), inline editorial
restoration markers ([...]/<...>/() around supplied or uncertain letters,
"γυναικ(ε)ιαριος", "αἰσ<ιμ>ώματα"), trailing homograph suffixes "(A)"/"(B)",
and a minority of rows that are full glosses with spaces.

Cleaning, in order (each step matched to a real LSJ convention; the dilemma key
convention is bare NFC polytonic Greek):
  - NFC normalize.
  - Strip a trailing homograph suffix "(A)"/"(B)" ("δάνας (B)" -> "δάνας").
  - Repair inline editorial markup: remove the bracket/paren characters and KEEP
    the supplied letters, which are part of the lemma ("[δηρ]ιάζομαι" ->
    "δηριάζομαι", "γυναικ(ε)ιαριος" -> "γυναικειαριος", "αἰσ<ιμ>ώματα" ->
    "αἰσιμώματα", "()σαλακας" -> "σαλακας").
  - Strip trailing punctuation (Greek raised dot, comma, full stop, quotes).
  - Drop multi-token forms (any whitespace): the gloss rows and the
    "(A) mow down" style entries fall out here.
  - Require at least one Greek code point; drop anything still carrying a Latin
    letter (residual OCR contamination).
  - Drop leading-dash suffix/termination forms ("-φόντης", "-ίνδα", 13 rows):
    dilemma's AG keys are full lemmas, not bound terminations.
  - Strip the LSJ internal morpheme-boundary hyphen ("Αἰγυπτι-άζω" ->
    "Αἰγυπτιάζω", "Βου-ζύγης" -> "Βουζύγης").
  - Strip vowel-quantity diacritics (COMBINING MACRON / BREVE and precomposed
    ᾱᾰῑῐῡῠ); dilemma's AG keys never carry metrical marks (ag_lookup / vlg / wip
    have zero), so keeping them would create unreachable self-maps and false
    "new" duplicates.
  - Drop bare single consonants with no vowel (letter-name / fragment rows like
    "μ", "ς"); single-letter *words* (always a vowel: ἀ, ὦ, ὁ, ἤ, η, ...) stay.
  - Dedup.

Output is byte-stable: sorted unique strings, with a sibling meta block in
data/lsj10_headwords.meta.json recording the source SHA-256 and per-reason drop
counts (no wall-clock), so a rebuild is reproducible and diffable.

Usage:
    python build/build_lsj10_headwords.py                 # default db path
    python build/build_lsj10_headwords.py --stats         # report, no write
    python build/build_lsj10_headwords.py \\
        --db /path/to/headwords.sqlite \\
        --output data/lsj10_headwords.json
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import unicodedata
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"

# The LSJ10 headword SQLite database (table `headwords`) lives outside this
# repo; point at it with --db or the LSJ10_HEADWORDS_DB env var.
DEFAULT_DB = os.environ.get("LSJ10_HEADWORDS_DB")
OUTPUT_PATH = DATA_DIR / "lsj10_headwords.json"
META_PATH = DATA_DIR / "lsj10_headwords.meta.json"

GREEK_RE = re.compile(r"[Ͱ-Ͽἀ-῿]")
LATIN_RE = re.compile(r"[A-Za-z]")
# Bare NFC polytonic Greek (optional leading suffix dash), nothing else.
GREEK_ONLY_RE = re.compile(r"^-?[Ͱ-Ͽἀ-῿̀-ͯ]+$")
# Trailing homograph marker, e.g. "δάνας (B)" -> "δάνας".
HOMOGRAPH_RE = re.compile(r"\s*\([A-Z]\)\s*$")
# Inline editorial restoration markers: drop the delimiters, keep the letters.
EDITORIAL_CHARS = str.maketrans("", "", "[]<>()")
# Trailing punctuation to peel off (Greek raised dot, comma, full stop, quotes).
TRAILING_PUNCT = ".,··;:’'\")]}"

COMBINING_MACRON = "̄"
COMBINING_BREVE = "̆"
# Precomposed vowel-quantity letters (α/ι/υ with macron or breve).
QUANTITY_PRECOMPOSED = set("ᾱᾰῑῐῡῠ")  # ᾱᾰῑῐῡῠ
# A "vowel" for the single-consonant test: any Greek vowel letter, with or
# without diacritics, detected on the accent-stripped lowercase base.
VOWEL_BASES = set("αειηουω")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_quantity(s: str) -> str:
    """Remove vowel-quantity (macron/breve) marks, keep all other diacritics."""
    if not (QUANTITY_PRECOMPOSED & set(s) or COMBINING_MACRON in s
            or COMBINING_BREVE in s):
        return s
    decomposed = unicodedata.normalize("NFD", s)
    decomposed = decomposed.replace(COMBINING_MACRON, "").replace(
        COMBINING_BREVE, "")
    return unicodedata.normalize("NFC", decomposed)


def base_letters(s: str) -> str:
    """Accent-stripped lowercase base letters (for the single-vowel test)."""
    out = []
    for ch in unicodedata.normalize("NFD", s.lower()):
        if unicodedata.combining(ch):
            continue
        out.append(ch)
    return "".join(out)


def clean_headword(raw: str, drops: Counter) -> str | None:
    """Clean one db `headword` string into a dilemma AG headword, or None.

    On rejection, bumps the matching counter in `drops` and returns None.
    """
    h = unicodedata.normalize("NFC", raw).strip()
    # Trailing homograph suffix, then repair inline editorial markup (keep the
    # supplied letters), then any trailing punctuation.
    h = HOMOGRAPH_RE.sub("", h)
    h = h.translate(EDITORIAL_CHARS)
    h = h.strip().strip(TRAILING_PUNCT)
    if not h or any(c.isspace() for c in h):
        drops["multi_token_or_empty"] += 1
        return None
    if not GREEK_RE.search(h):
        drops["no_greek"] += 1
        return None
    if LATIN_RE.search(h):
        drops["latin_contamination"] += 1
        return None
    if h.startswith("-"):
        drops["leading_dash_suffix"] += 1
        return None
    # Strip LSJ internal morpheme-boundary hyphens.
    h = h.replace("-", "")
    h = strip_quantity(unicodedata.normalize("NFC", h))
    h = unicodedata.normalize("NFC", h)
    if not h:
        drops["empty_after_clean"] += 1
        return None
    if not GREEK_ONLY_RE.match(h):
        drops["nongreek_residue"] += 1
        return None
    # Drop bare single consonants (letter-name rows / fragments). Keep
    # single-letter words, which always contain a vowel (ἀ, ὦ, ὁ, ἤ, η, ...).
    if len(h) == 1 and not (set(base_letters(h)) & VOWEL_BASES):
        drops["bare_single_consonant"] += 1
        return None
    return h


def load_headwords(db_path: Path) -> list[str]:
    """The `headword` column of every row (the `clean` flag is not a filter)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return [r[0] for r in con.execute(
            "SELECT headword FROM headwords WHERE headword IS NOT NULL")]
    finally:
        con.close()


def build(db_path: Path):
    raw = load_headwords(db_path)
    drops: Counter = Counter()
    cleaned: set[str] = set()
    for r in raw:
        c = clean_headword(r, drops)
        if c is not None:
            cleaned.add(c)

    headwords = sorted(cleaned)
    stats = {
        "source_rows": len(raw),
        "distinct_raw_headword": len(set(raw)),
        "clean_headwords": len(headwords),
        "dropped_total": sum(drops.values()),
        "dropped_by_reason": dict(sorted(drops.items())),
    }
    return headwords, stats


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path,
                    default=(Path(DEFAULT_DB) if DEFAULT_DB else None),
                    help="LSJ10 headword sqlite (table `headwords`); "
                         "defaults to $LSJ10_HEADWORDS_DB")
    ap.add_argument("--output", type=Path, default=OUTPUT_PATH,
                    help="output JSON (flat list of headwords)")
    ap.add_argument("--stats", action="store_true",
                    help="print stats only; do not write output")
    args = ap.parse_args(argv)

    if args.db is None:
        sys.exit("error: no LSJ10 headword db given; pass --db or set "
                 "LSJ10_HEADWORDS_DB")
    if not args.db.exists():
        sys.exit(f"error: LSJ10 headword db not found: {args.db}")

    headwords, stats = build(args.db)

    print("LSJ10 headwords:")
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv:,}")
        else:
            print(f"  {k}: {v:,}")
    print(f"  sample: {headwords[:8]}")

    if args.stats:
        return

    meta = {
        "source": "LSJ10 headwords.sqlite (table headwords, `headword` column, "
                  "all rows)",
        "db_sha256": sha256_file(args.db),
        "stats": stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(headwords, f, ensure_ascii=False, indent=0)
        f.write("\n")
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {len(headwords):,} headwords -> {args.output}")
    print(f"wrote meta -> {META_PATH}")


if __name__ == "__main__":
    main()
