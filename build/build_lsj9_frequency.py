#!/usr/bin/env python3
"""Build data/lsj9_frequency.json from the LSJ9 export of the same name.

The LSJ9 export counts the references of each entry under its head as
printed: divided at the stem (ψῡχ-ή, κατα-λύω) and with vowel-length
marks (πᾰρά). Up to 1.3.28 the file was a byte copy of that export, so
33,870 of its 55,495 keys had a hyphen between two letters and 17,045 a
breve or macron, and no lemma the lookup returns matched them: the runtime
reads the table under the lemma spelling (Dilemma.suggest_spelling ranks
candidates by it), and lemmas carry neither mark (#149).

This script keys each count under the citation spelling: the stem
division and the length marks removed, NFC. Where two entries join to one
spelling (τροχ-ός and τροχός, the two entries of a homograph) their counts
are added. A hyphen at an edge (ἀ-, a prefix entry) stays.
data/changes/lsj9_frequency_keys.json records the export's sha256 (the
export itself is the inverse), how many keys changed for each reason, and
every merge with the counts it added.

    python build/build_lsj9_frequency.py [--lsj9-dir DIR]   # export -> data/
    python build/build_lsj9_frequency.py --check            # exit 1 if stale
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dilemma.citation import HEAD_HYPHENS, hyphenated_lemma  # noqa: E402

LSJ9_DIR = Path(os.environ.get("DILEMMA_LSJ9_DIR")
                or Path.home() / "Documents" / "lsj9")
OUTPUT_PATH = ROOT / "data" / "lsj9_frequency.json"
RECORD_PATH = ROOT / "data" / "changes" / "lsj9_frequency_keys.json"
LENGTH_MARKS = (0x0304, 0x0306)  # combining macron, breve


def citation_key(head: str) -> str:
    """The head as the lookup spells a lemma: no stem division, no length
    marks, NFC. A hyphen at an edge is kept."""
    if hyphenated_lemma(head):
        head = "".join(char for char in head if char not in HEAD_HYPHENS)
    nfd = unicodedata.normalize("NFD", head)
    return unicodedata.normalize(
        "NFC", "".join(char for char in nfd if ord(char) not in LENGTH_MARKS))


def build(export: dict) -> tuple[dict, dict]:
    """``(table, changes)``: the counts under citation keys, in the export's
    order of first appearance, and ``{new key: [[old key, count], ...]}``
    for every key that is not the export's own."""
    table: dict[str, int] = {}
    sources: dict[str, list] = {}
    for head, count in export.items():
        key = citation_key(head)
        table[key] = table.get(key, 0) + int(count)
        sources.setdefault(key, []).append([head, int(count)])
    changes = {key: heads for key, heads in sources.items()
               if heads != [[key, table[key]]]}
    return table, changes


def serialize(table: dict) -> bytes:
    return json.dumps(table, ensure_ascii=False, indent=1).encode("utf-8")


def record(export_bytes: bytes, export: dict, table: dict, changes: dict) -> dict:
    out = serialize(table)
    return {
        "schema_version": 1,
        "table": "data/lsj9_frequency.json",
        "rule": "build/build_lsj9_frequency.py:citation_key",
        "reason": ("LSJ9 counts references under the head as printed, divided "
                   "at the stem and with vowel-length marks; the lookup's "
                   "lemmas have neither, so those keys matched no lemma. Each "
                   "count is keyed under the citation spelling; entries that "
                   "join to one spelling are added."),
        "source": {"path": "lsj9_frequency.json (LSJ9 export)",
                   "sha256": hashlib.sha256(export_bytes).hexdigest(),
                   "size": len(export_bytes), "entries": len(export)},
        "result": {"sha256": hashlib.sha256(out).hexdigest(),
                   "size": len(out), "entries": len(table)},
        "changed_keys": len(changes),
        "changed_for_stem_division": sum(
            1 for heads in changes.values()
            if any(hyphenated_lemma(head) for head, _ in heads)),
        "changed_for_length_marks_only": sum(
            1 for heads in changes.values()
            if not any(hyphenated_lemma(head) for head, _ in heads)),
        "merged_keys": sum(1 for heads in changes.values() if len(heads) > 1),
        "merges": {key: heads for key, heads in sorted(changes.items())
                   if len(heads) > 1},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lsj9-dir", type=Path, default=None,
                        help="LSJ9 export directory (default: $DILEMMA_LSJ9_DIR, "
                             "else ~/Documents/lsj9)")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--record", type=Path, default=RECORD_PATH)
    parser.add_argument("--check", action="store_true",
                        help="write nothing; exit 1 unless the table and the "
                             "record are what the export builds")
    args = parser.parse_args(argv)
    source = (args.lsj9_dir or LSJ9_DIR) / "lsj9_frequency.json"
    export_bytes = source.read_bytes()
    export = json.loads(export_bytes)
    table, changes = build(export)
    entry = record(export_bytes, export, table, changes)
    if args.check:
        current = args.output.read_bytes() if args.output.exists() else b""
        recorded = (json.loads(args.record.read_text(encoding="utf-8"))
                    if args.record.exists() else None)
        if current != serialize(table) or recorded != entry:
            print(f"{args.output.name} is not what {source} builds; run "
                  f"build/build_lsj9_frequency.py", file=sys.stderr)
            return 1
        print(f"{args.output.name}: up to date ({len(table):,} keys)")
        return 0
    args.output.write_bytes(serialize(table))
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(entry, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
    print(f"{args.output.name}: {len(export):,} -> {len(table):,} keys, "
          f"{len(changes):,} keys changed ({entry['merged_keys']:,} merged); "
          f"recorded in {args.record.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
