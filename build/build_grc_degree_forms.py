#!/usr/bin/env python3
"""Write data/grc_degree_forms.json: attested regular degree forms.

For every Ancient adjective a lexicon classes (data/grc_second_declension.json,
data/grc_ikos_adjectives.json, data/grc_es_adjectives.json, and the LSJ 9
heads whose entry states a regular comparison), dilemma/grc_degree_forms.py
gives the comparative and superlative cells of the formation its stem or
LSJ's statement settles. A cell is kept only when ``form_profile.db``
attests its spelling (grave folded to acute, case folded, as the output gate
of dilemma/_attest_db.py compares forms), and only when no other listed
adjective gives the same cell. The record lists each kept cell with its
attestation, and each lemma with its class evidence and the evidence of its
formation.

    python build/build_grc_degree_forms.py --lsj9-dir DIR
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import sys
import unicodedata

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "build"), str(ROOT)]

from dilemma._attest_db import norm_key  # noqa: E402
from dilemma.grc_adjective_gaps import read_es_adjectives  # noqa: E402
from dilemma.grc_degree_forms import (  # noqa: E402
    degree_cells, plain, rule_formations, statement_formations)
from dilemma.grc_noun_gaps import is_plain_lemma, strip_accents  # noqa: E402
from dilemma.grc_os_gaps import NOT_DECLINED, read_classes  # noqa: E402
from find_grc_second_declension import sha256  # noqa: E402

DATA = ROOT / "data"
OUTPUT = DATA / "grc_degree_forms.json"
RULE = "grc_degree_forms_v1"
CLASS_FILES = ("grc_second_declension.json", "grc_ikos_adjectives.json",
               "grc_es_adjectives.json")
LSJ9_FILES = ("lsj9_headwords.json", "lsj9_glosses.jsonl")
FORM_PROFILE = "form_profile.db"


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def lexicon_adjectives(data: Path) -> dict[str, list[str]]:
    """lemma -> its class evidence, from the class lists."""
    evidence: dict[str, list[str]] = defaultdict(list)
    classes = read_classes(json.loads(
        (data / CLASS_FILES[0]).read_text(encoding="utf-8")))
    for lemma, cls in classes.items():
        if cls.startswith("adjective"):
            evidence[lemma].append(f"grc_second_declension:{cls}")
    for row in json.loads((data / CLASS_FILES[1]).read_text(
            encoding="utf-8"))["lemmas"]:
        evidence[_nfc(row["lemma"])].append("grc_ikos_adjectives")
    for lemma in read_es_adjectives(json.loads(
            (data / CLASS_FILES[2]).read_text(encoding="utf-8"))):
        evidence[lemma].append("grc_es_adjectives")
    return evidence


def lsj9_entries(lsj9_dir: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """(lemma -> its headword with LSJ's length marks, lemma -> entry texts)."""
    marked: dict[str, str] = {}
    for entry in json.loads((lsj9_dir / "lsj9_headwords.json").read_text(
            encoding="utf-8")):
        head = _nfc(entry["headword"])
        lemma = plain(head)
        if head != lemma:
            # Homographs may mark different vowels; keep the first.
            marked.setdefault(lemma, head)
    texts: dict[str, list[str]] = defaultdict(list)
    with (lsj9_dir / "lsj9_glosses.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            if entry.get("headword"):
                texts[plain(_nfc(entry["headword"]))].append(entry.get("text", ""))
    return marked, texts


def is_candidate(lemma: str) -> bool:
    bare = strip_accents(lemma)
    return (is_plain_lemma(lemma) and lemma not in NOT_DECLINED
            and bare.endswith(("ος", "ης"))
            and not bare.endswith(("οος", "εος")) and len(bare) >= 5)


class Attestation:
    """Attestation of a spelling in form_profile.db, under its gate key."""

    def __init__(self, path: Path):
        self.connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key = 'content_hash'").fetchone()
        self.content_hash = row[0] if row else None

    def __call__(self, form: str) -> dict | None:
        total, sources = 0, Counter()
        for count, counts in self.connection.execute(
                "SELECT p.total_count, p.source_counts_json FROM forms f "
                "JOIN form_profile p ON p.form_id = f.form_id "
                "WHERE f.form_norm = ?", (norm_key(form),)):
            total += count or 0
            sources.update(json.loads(counts or "{}"))
        if not sources and not total:
            return None
        return {"total": total, "sources": dict(sorted(sources.items()))}


def build(lsj9_dir: Path, data: Path = DATA,
          attestation: Attestation | None = None) -> dict:
    attestation = attestation or Attestation(data / FORM_PROFILE)
    evidence = lexicon_adjectives(data)
    marked, texts = lsj9_entries(lsj9_dir)
    formations: dict[str, tuple[tuple[str, ...], str]] = {}
    for lemma in sorted(set(evidence) | set(texts)):
        if not is_candidate(lemma):
            continue
        stated = statement_formations(lemma, texts.get(lemma, ()))
        if stated:
            if lemma not in evidence:
                evidence[lemma].append("lsj9_statement")
            formations[lemma] = (stated, "lsj9_statement")
        elif lemma in evidence:
            ruled = rule_formations(marked.get(lemma, lemma))
            if ruled:
                formations[lemma] = (
                    ruled, "stem_rule_lsj9_marks" if lemma in marked
                    else "stem_rule")
    # form -> {lemma: (cell, formation)}
    claims: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    for lemma, (kinds, _) in formations.items():
        for formation in kinds:
            for form, cell in degree_cells(lemma, formation):
                claims[form].setdefault(lemma, (cell, formation))
    lemmas: dict[str, dict] = {}
    ambiguous: dict[str, list[str]] = {}
    counts: Counter = Counter()
    for form in sorted(claims):
        found = attestation(form)
        if found is None:
            continue
        rows = claims[form]
        if len(rows) > 1:
            ambiguous[form] = sorted(rows)
            continue
        (lemma, (cell, formation)), = rows.items()
        kinds, how = formations[lemma]
        record = lemmas.setdefault(lemma, {
            "class": sorted(evidence[lemma]), "formations": list(kinds),
            "formation_evidence": how, "cells": []})
        record["cells"].append([form, cell, found["total"], found["sources"]])
        counts["cells"] += 1
        counts[cell.split()[0]] += 1
    for record in lemmas.values():
        record["cells"].sort()
    counts["lemmas"] = len(lemmas)
    counts["ambiguous"] = len(ambiguous)
    counts["candidate_lemmas"] = len(formations)
    inputs = {name: sha256(data / name) for name in CLASS_FILES}
    inputs.update({name: sha256(lsj9_dir / name) for name in LSJ9_FILES})
    inputs[FORM_PROFILE] = {"content_hash": attestation.content_hash}
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count:,}")
    return {"schema_version": 1, "rule": RULE, "inputs": inputs,
            "counts": dict(sorted(counts.items())),
            "lemmas": dict(sorted(lemmas.items())),
            "ambiguous": ambiguous}


def read_cells(record: dict) -> dict[str, list[tuple[str, str]]]:
    """lemma -> (form, cell) of the kept cells."""
    return {lemma: [(row[0], row[1]) for row in entry["cells"]]
            for lemma, entry in record["lemmas"].items()}


def main(argv=None) -> int:
    import expand_lsj
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lsj9-dir", type=Path, default=None,
                        help="LSJ9 export directory (default: $DILEMMA_LSJ9_DIR, "
                             "else ~/Documents/lsj9)")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    expand_lsj.configure_inputs(lsj9_dir=args.lsj9_dir)
    expand_lsj.verify_lsj_glosses()
    if not (args.data / FORM_PROFILE).exists():
        raise SystemExit(f"{args.data / FORM_PROFILE} is missing: the degree "
                         "forms are kept only where it attests them "
                         "(python -m dilemma download --with-attestation)")
    record = build(expand_lsj.LSJ9_DIR, args.data)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=1)
        stream.write("\n")
    print(f"  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
