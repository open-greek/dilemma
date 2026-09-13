#!/usr/bin/env python3
"""Audit Dilemma lookup lemmas for citation-form hygiene issues.

This is a Dilemma-side artifact audit: it scans data/lookup.db's distinct lemma
strings and reports marks that should not normally appear in dictionary
citation forms (grave accent, final elision marks, overline abbreviations,
leading combining marks, and multiple tonal accents). It does not read OGC
cache files.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dilemma.core import DATA_DIR, LOOKUP_DB_PATH, Dilemma  # noqa: E402


TONAL_MARKS = {"\u0301", "\u0300", "\u0342"}
SPACING_ELISION_MARKS = {"\u2019", "\u02bc", "'", "\u1fbd", "`", "\u02b9"}


def _flags(lemma: str) -> list[str]:
    nfd = unicodedata.normalize("NFD", lemma)
    flags: list[str] = []
    if "\u0300" in nfd:
        flags.append("grave")
    if lemma and unicodedata.combining(lemma[0]):
        flags.append("leading_combining")
    if lemma and lemma[-1] in SPACING_ELISION_MARKS:
        flags.append("final_elision_mark")
    if "\u0305" in nfd:
        flags.append("overline")
    if sum(1 for ch in nfd if ch in TONAL_MARKS) > 1:
        flags.append("multiple_tonal_accents")
    return flags


def _iter_lemmas(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    try:
        for (lemma,) in conn.execute("SELECT text FROM lemmas"):
            yield lemma
    finally:
        conn.close()


def audit(db_path: Path, example_limit: int) -> dict:
    d = Dilemma(lang="grc", citation_policy="strict_ag", skip_pos=True)
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict]] = defaultdict(list)
    total = 0

    for lemma in _iter_lemmas(db_path):
        total += 1
        for flag in _flags(lemma):
            counts[flag] += 1
            if len(examples[flag]) < example_limit:
                status = d.citation_status(lemma, lang="grc")
                examples[flag].append({
                    "lemma": lemma,
                    "normalized": status["normalized"],
                    "ok": status["ok"],
                    "reason": status["reason"],
                })

    return {
        "lookup_db": str(db_path),
        "data_dir": str(DATA_DIR),
        "total_lemmas": total,
        "counts": dict(sorted(counts.items())),
        "examples": dict(examples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookup-db", type=Path, default=LOOKUP_DB_PATH)
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON only")
    args = parser.parse_args()

    report = audit(args.lookup_db, args.examples)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print(f"lookup_db: {report['lookup_db']}")
    print(f"total lemmas: {report['total_lemmas']:,}")
    for flag, count in report["counts"].items():
        print(f"{flag}: {count:,}")
        for item in report["examples"].get(flag, []):
            normalized = item["normalized"] or "<rejected>"
            print(f"  {item['lemma']} -> {normalized} ({item['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
