#!/usr/bin/env python3
"""Extract the Modern Greek dialect sections of the Greek Wiktionary.

The Greek Wiktionary heads some Modern Greek pages as Cretan (``el-crt``) or
Cypriot (``el-cyp``). The extract files those entries as languages of their
own and publishes no file for either, so a page with no ``el`` section never
reached ``build_data.py`` (μπαλωθιά). They are only in the edition's raw
extract of every language.

This reads that raw extract and writes the dialect entries, in source order
and byte for byte, to ``kaikki.org-el-dictionary-GreekDialects.jsonl`` beside
the other dumps, with a ``.meta.json`` that records the input and output
checksums and the entries per language code. ``build_data.py`` reads the
file as part of the Greek edition when it is there.

Usage:
    python build/extract_el_dialect_sections.py --download --kaikki KAIKKI
    python build/extract_el_dialect_sections.py RAW.jsonl.gz --kaikki KAIKKI
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

RAW_URL = "https://kaikki.org/elwiktionary/raw-wiktextract-data.jsonl.gz"
RAW_NAME = "kaikki.org-el-raw-wiktextract-data.jsonl.gz"
OUTPUT_NAME = "kaikki.org-el-dictionary-GreekDialects.jsonl"
# The Greek Wiktionary's section codes for varieties of Modern Greek, and the
# tag the build gives their pairs. Pontic, Italiot, Tsakonian and Cappadocian
# have language codes of their own there and are not Modern Greek sections.
DIALECT_CODES = {"el-crt": "Cretan", "el-cyp": "Cypriot"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dialect_lines(lines):
    """The lines of the raw extract that are dialect-section entries."""
    markers = tuple(f'"lang_code": "{code}"' for code in DIALECT_CODES)
    for line in lines:
        if not any(marker in line for marker in markers):
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        # The marker also matches a translation or a descendant's code.
        if entry.get("lang_code") in DIALECT_CODES and entry.get("word"):
            yield entry["lang_code"], line if line.endswith("\n") else line + "\n"


def extract(raw: Path, output: Path) -> dict:
    opener = gzip.open if raw.suffix == ".gz" else open
    counts = Counter()
    with opener(raw, "rt", encoding="utf-8") as lines, \
            output.open("w", encoding="utf-8", newline="\n") as out:
        for code, line in dialect_lines(lines):
            counts[code] += 1
            out.write(line)
    meta = {
        "schema_version": 1,
        "source": {"name": raw.name, "url": RAW_URL, "bytes": raw.stat().st_size,
                   "sha256": sha256(raw)},
        "license": "CC BY-SA 4.0 (Greek Wiktionary)",
        "language_codes": DIALECT_CODES,
        "entries": {code: counts[code] for code in sorted(DIALECT_CODES)},
        "output": {"name": output.name, "sha256": sha256(output)},
    }
    output.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw", nargs="?", type=Path,
                        help="the Greek edition's raw extract (.jsonl or .jsonl.gz)")
    parser.add_argument("--kaikki", type=Path, required=True,
                        help="dump directory; the dialect file is written here")
    parser.add_argument("--download", action="store_true",
                        help=f"fetch the raw extract into the dump directory ({RAW_URL})")
    args = parser.parse_args(argv)
    args.kaikki.mkdir(parents=True, exist_ok=True)
    raw = args.raw or args.kaikki / RAW_NAME
    if args.download and not raw.exists():
        print(f"Downloading {RAW_URL}")
        urllib.request.urlretrieve(RAW_URL, raw)
    if not raw.exists():
        parser.error(f"raw extract not found: {raw} (pass a path or --download)")
    meta = extract(raw, args.kaikki / OUTPUT_NAME)
    print(f"{raw.name}: sha256 {meta['source']['sha256']}")
    for code, count in meta["entries"].items():
        print(f"  {code} ({DIALECT_CODES[code]}): {count:,} entries")
    print(f"  -> {args.kaikki / OUTPUT_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
