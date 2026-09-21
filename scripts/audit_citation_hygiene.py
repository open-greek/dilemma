#!/usr/bin/env python3
"""Audit Dilemma lookup lemmas for citation-form hygiene issues.

This is a Dilemma-side artifact audit: it scans data/lookup.db's distinct lemma
strings and reports marks that should not normally appear in dictionary
citation forms (grave accent, final elision marks, overline abbreviations,
leading combining marks, editorial sigla, and multiple tonal accents).
Numeral/keraia residue is classified separately from true elision marks. This
script does not read OGC cache files.
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
from dilemma.citation import TONAL_MARKS, malformed_tonal_reason  # noqa: E402
from dilemma.nonlexical import classify_nonlexical  # noqa: E402


SPACING_ELISION_MARKS = {"\u2019", "\u02bc", "'", "\u1fbd", "`"}
KERAIA_OR_PRIME_MARKS = {"\u0374", "\u02b9"}
EDITORIAL_SIGLA = frozenset("[]()")


def _flags(lemma: str) -> list[str]:
    nfd = unicodedata.normalize("NFD", lemma)
    nonlexical = classify_nonlexical(lemma)
    flags: list[str] = []
    if "\u0300" in nfd:
        flags.append("grave")
    if lemma and unicodedata.combining(lemma[0]):
        flags.append("leading_combining")
    if lemma and lemma[-1] in KERAIA_OR_PRIME_MARKS:
        if nonlexical == "numeral":
            flags.append("greek_numeral")
        elif nonlexical:
            flags.append("nonlexical_keraia_or_prime")
        else:
            flags.append("final_keraia_or_prime")
    elif lemma and lemma[-1] in SPACING_ELISION_MARKS:
        if nonlexical == "numeral":
            flags.append("greek_numeral")
        elif nonlexical:
            flags.append("nonlexical_elision_mark")
        else:
            flags.append("final_elision_mark")
    if "\u0305" in nfd:
        flags.append("overline")
    if any(char in lemma for char in EDITORIAL_SIGLA):
        flags.append("editorial_siglum")
    malformed_tonal = malformed_tonal_reason(lemma)
    if malformed_tonal:
        flags.append(malformed_tonal)
    if sum(1 for ch in nfd if ch in TONAL_MARKS) > 1:
        flags.append("multiple_tonal_accents")
    return flags


def _multiple_tonal_shape(lemma: str) -> str:
    """Classify broad multi-accent diagnostics without declaring them invalid."""
    for char in lemma:
        if char.isspace() or unicodedata.category(char).startswith("P"):
            return "multiword_or_punctuated"
    return "single_token"


def _flagged_lemmas(db_path: Path) -> tuple[int, list[dict], int]:
    """Return flagged lemmas with the lookup-source languages that expose them.

    ``lemmas`` is shared by the AG and MG lookup tables, so auditing every row
    as Ancient Greek misclassifies valid monotonic multiword MG headwords.  The
    lookup's ``src`` column records which language supplied each mapping.  A
    temporary ID table lets SQLite recover that provenance with one scan of the
    large lookup table without adding a runtime index to the shipped artifact.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        total = conn.execute("SELECT COUNT(*) FROM lemmas").fetchone()[0]
        flagged = []
        for lemma_id, lemma in conn.execute("SELECT id, text FROM lemmas"):
            flags = _flags(lemma)
            if flags:
                flagged.append({
                    "id": lemma_id,
                    "lemma": lemma,
                    "flags": flags,
                    "languages": set(),
                })

        if flagged:
            conn.execute(
                "CREATE TEMP TABLE audit_lemma_ids "
                "(id INTEGER PRIMARY KEY) WITHOUT ROWID"
            )
            conn.executemany(
                "INSERT INTO audit_lemma_ids (id) VALUES (?)",
                ((item["id"],) for item in flagged),
            )
            by_id = {item["id"]: item for item in flagged}
            for lemma_id, source in conn.execute(
                "SELECT DISTINCT k.lemma_id, k.src "
                "FROM lookup k JOIN audit_lemma_ids a ON a.id = k.lemma_id"
            ):
                if source in {"grc", "el"}:
                    by_id[lemma_id]["languages"].add(source)

        editorial_forms = conn.execute(
            "SELECT COUNT(DISTINCT form) FROM lookup "
            "WHERE instr(form, '[') OR instr(form, ']') "
            "OR instr(form, '(') OR instr(form, ')')"
        ).fetchone()[0]
        return total, flagged, editorial_forms
    finally:
        conn.close()


def audit(db_path: Path, example_limit: int) -> dict:
    d = Dilemma(lang="grc", citation_policy="strict_ag", skip_pos=True)
    counts: Counter[str] = Counter()
    accepted: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict]] = defaultdict(list)
    language_reports = {
        language: {
            "counts": Counter(),
            "accepted_counts": Counter(),
            "rejected_counts": Counter(),
            "reason_counts": defaultdict(Counter),
            "examples": defaultdict(list),
            "multiple_tonal_classes": defaultdict(Counter),
        }
        for language in ("grc", "el")
    }
    total, flagged_lemmas, editorial_forms = _flagged_lemmas(db_path)

    for item in flagged_lemmas:
        lemma = item["lemma"]
        nonlexical = classify_nonlexical(lemma)
        status_by_language = {}
        for language in sorted(item["languages"]):
            status = d.citation_status(
                lemma,
                lang=language,
                source="nonlexical" if nonlexical else "",
            )
            status_by_language[language] = status

        # Unreferenced lemma rows cannot be emitted by lookup, but retaining
        # them in the global structural counts exposes artifact bloat.  Their
        # citation status is deliberately neutral rather than guessed as AG.
        statuses = list(status_by_language.values())
        globally_ok = all(status["ok"] for status in statuses)
        if statuses:
            failed_reasons = sorted({
                status["reason"] for status in statuses if not status["ok"]
            })
            successful_reasons = sorted({
                status["reason"] for status in statuses if status["ok"]
            })
            global_reason = ",".join(failed_reasons or successful_reasons)
        else:
            global_reason = "unreferenced"

        for flag in item["flags"]:
            counts[flag] += 1
            if globally_ok:
                accepted[flag] += 1
            else:
                rejected[flag] += 1
            reasons[flag][global_reason] += 1
            if len(examples[flag]) < example_limit:
                examples[flag].append({
                    "lemma": lemma,
                    "ok": globally_ok,
                    "reason": global_reason,
                    "nonlexical": nonlexical,
                    "languages": sorted(item["languages"]),
                    "status_by_language": status_by_language,
                })

            for language, status in status_by_language.items():
                language_report = language_reports[language]
                language_report["counts"][flag] += 1
                outcome = ("accepted_counts" if status["ok"]
                           else "rejected_counts")
                language_report[outcome][flag] += 1
                language_report["reason_counts"][flag][status["reason"]] += 1
                if flag == "multiple_tonal_accents":
                    shape = _multiple_tonal_shape(lemma)
                    shape_counts = language_report["multiple_tonal_classes"][shape]
                    shape_counts["count"] += 1
                    shape_counts["accepted" if status["ok"] else "rejected"] += 1
                    shape_counts[f"reason:{status['reason']}"] += 1
                if len(language_report["examples"][flag]) < example_limit:
                    language_report["examples"][flag].append({
                        "lemma": lemma,
                        "normalized": status["normalized"],
                        "ok": status["ok"],
                        "reason": status["reason"],
                        "nonlexical": nonlexical,
                    })

    by_language = {}
    for language, language_report in language_reports.items():
        by_language[language] = {
            "counts": dict(sorted(language_report["counts"].items())),
            "accepted_counts": dict(sorted(
                language_report["accepted_counts"].items())),
            "rejected_counts": dict(sorted(
                language_report["rejected_counts"].items())),
            "reason_counts": {
                flag: dict(sorted(reason_counts.items()))
                for flag, reason_counts in sorted(
                    language_report["reason_counts"].items())
            },
            "examples": dict(language_report["examples"]),
            "multiple_tonal_classes": {
                shape: dict(sorted(shape_counts.items()))
                for shape, shape_counts in sorted(
                    language_report["multiple_tonal_classes"].items())
            },
        }

    return {
        "lookup_db": str(db_path),
        "data_dir": str(DATA_DIR),
        "total_lemmas": total,
        "editorial_form_count": editorial_forms,
        "counts": dict(sorted(counts.items())),
        "accepted_counts": dict(sorted(accepted.items())),
        "rejected_counts": dict(sorted(rejected.items())),
        "reason_counts": {
            flag: dict(sorted(reason_counts.items()))
            for flag, reason_counts in sorted(reasons.items())
        },
        "examples": dict(examples),
        "by_language": by_language,
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
    print(f"editorial surface forms: {report['editorial_form_count']:,}")
    for flag, count in report["counts"].items():
        rejected = report["rejected_counts"].get(flag, 0)
        accepted = report["accepted_counts"].get(flag, 0)
        print(f"{flag}: {count:,} ({rejected:,} rejected, "
              f"{accepted:,} accepted/classified)")
        reasons = ", ".join(
            f"{reason}={n:,}"
            for reason, n in report["reason_counts"].get(flag, {}).items()
        )
        if reasons:
            print(f"  reasons: {reasons}")
        for item in report["examples"].get(flag, []):
            nonlex = (f", nonlexical={item['nonlexical']}"
                      if item.get("nonlexical") else "")
            languages = ",".join(item.get("languages", [])) or "unreferenced"
            print(f"  {item['lemma']} ({item['reason']}; "
                  f"languages={languages}{nonlex})")
    for language, language_report in report["by_language"].items():
        print(f"{language} source:")
        for flag, count in language_report["counts"].items():
            rejected = language_report["rejected_counts"].get(flag, 0)
            accepted = language_report["accepted_counts"].get(flag, 0)
            print(f"  {flag}: {count:,} ({rejected:,} rejected, "
                  f"{accepted:,} accepted/classified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
