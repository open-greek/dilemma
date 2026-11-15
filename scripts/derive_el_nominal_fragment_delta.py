#!/usr/bin/env python3
"""Cross-check a Modern Greek fragment rule against the published tables.

``build_data.py`` skips a Modern Greek noun or adjective table cell that
has no stem of its word (``is_modern_nominal_fragment``: αρχαΐζων's ούσης,
όντων and ουσών, αισώπειος's ος and ε, βρόντημα's γ). The published Modern
Greek tables are a pinned snapshot, so the change reaches them as a recorded
delta: ``data/changes/el_nominal_fragments.json`` is the difference between
the old and the new generator over the dumps pinned in
``data/kaikki_inputs.json`` (``scripts/refresh_el_generator_delta.py diff``).

This script is the removal-only approximation for when those dumps are not
available. It reads the Wiktionary dumps and applies the generator's own
rule to every Modern Greek form cell. For each fragment it removes, from the
pinned ``mg_lookup.json`` and ``mg_pos_lookup.json``, only the readings the
cell could have put there: one of the cell's keys (its spelling, lowercase,
monotonic and accentless forms) whose value is the fragment's headword, or
the word that headword resolves to in the same snapshot (αρχαΐζων to
αρχαΐζω). Nothing is added. The output uses the delta format of
``scripts/refresh_el_generator_delta.py``, which applies, audits and undoes
it; write it to a scratch path, not over the committed record::

    python scripts/derive_el_nominal_fragment_delta.py --dumps DUMP_DIR \
        --base data --manifest data/hf_manifest.json --out CHECK.json

A full ``build_data.py --lang el`` differs from this approximation where a
removed key has another claimant (the next claim wins), where a POS entry is
left with one reading (the table keeps only ambiguous forms) and in the
training pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_data import (DUMPS, EL_DIALECT_CODES, EL_DIALECT_DUMP, WIKT_TO_UPOS,
                        _is_greek, strip_accents, strip_length_marks,
                        to_monotonic)
from dilemma.nominal_header_forms import is_modern_nominal_fragment

SCHEMA = 1
RULE = "el_nominal_fragment_repair_v1"
FILES = ("mg_lookup.json", "mg_pos_lookup.json")
GENERATOR = ROOT / "dilemma" / "nominal_header_forms.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def letters(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text or "")
                   if unicodedata.category(c) != "Mn").lower()


def keys(form: str) -> list[str]:
    """The keys build_data's _add_lookup and _add_pos_map give a form."""
    out = []
    for key in (form, form.lower(), to_monotonic(form),
                to_monotonic(form).lower(), strip_accents(form.lower())):
        if key and key not in out:
            out.append(key)
    return out


def fragment_cells(entries):
    """The (pos, word, form, tags, source) of each fragment cell, in order.

    ``entries`` yields (dump name, kaikki entry). The cell filters are the
    ones build_data applies before the fragment check: table headings and
    romanizations are skipped, and a cell must be one Greek word.
    """
    skip_tags = {"romanization", "table-tags", "inflection-template", "class"}
    seen = set()
    for dump, entry in entries:
        if entry.get("lang_code") not in {"el", *EL_DIALECT_CODES}:
            continue
        for cell in entry.get("forms") or ():
            tags = cell.get("tags") or []
            if any(tag in skip_tags for tag in tags):
                continue
            if not is_modern_nominal_fragment(entry, cell):
                continue
            form = strip_length_marks(cell.get("form", ""))
            if (not form or " " in form or not any(c.isalpha() for c in form)
                    or not _is_greek(form)):
                continue
            row = (entry["pos"], entry["word"], form)
            if row in seen:
                continue
            seen.add(row)
            yield {"dump": dump, "pos": entry["pos"], "lemma": entry["word"],
                   "form": form, "tags": sorted(tags),
                   "source": cell.get("source")}


def read_dumps(dump_dir: Path):
    names = list(DUMPS["el"].values()) + [EL_DIALECT_DUMP]
    for name in names:
        path = dump_dir / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                yield name, json.loads(line)


def derive(cells, lookup: dict, pos_lookup: dict) -> dict:
    """The per-file ``set`` changes that remove the cells' readings."""
    lookup_set, pos_set = {}, {}
    for cell in cells:
        lemma = cell["lemma"]
        targets = {letters(lemma)}
        for spelling in (lemma, to_monotonic(lemma)):
            if isinstance(lookup.get(spelling), str):
                targets.add(letters(lookup[spelling]))
        upos = WIKT_TO_UPOS.get(cell["pos"])
        for key in keys(cell["form"]):
            value = lookup.get(key)
            if (isinstance(value, str) and key not in lookup_set
                    and letters(value) in targets):
                lookup_set[key] = [value, None]
            before = pos_set[key][0] if key in pos_set else pos_lookup.get(key)
            current = pos_set[key][1] if key in pos_set else before
            if (upos and isinstance(current, dict) and upos in current
                    and letters(current[upos]) in targets):
                after = {u: l for u, l in current.items() if u != upos} or None
                pos_set[key] = [before, after]
    files = {}
    if lookup_set:
        files["mg_lookup.json"] = {"set": dict(sorted(lookup_set.items()))}
    if pos_set:
        files["mg_pos_lookup.json"] = {"set": dict(sorted(pos_set.items()))}
    return files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dumps", required=True, type=Path,
                        help="directory holding the Modern Greek kaikki dumps")
    parser.add_argument("--base", required=True, type=Path,
                        help="directory holding the pinned mg_lookup.json "
                             "and mg_pos_lookup.json")
    parser.add_argument("--manifest", type=Path,
                        help="refuse base files that differ from this manifest")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    pinned = (json.loads(args.manifest.read_text(encoding="utf-8"))["files"]
              if args.manifest else {})
    inputs = {}
    for name in FILES:
        digest = sha256(args.base / name)
        want = pinned.get(f"data/{name}", {}).get("sha256")
        if want and digest != want:
            sys.exit(f"base {name} differs from the pinned manifest")
        inputs[name] = digest
    dumps = {path.name: sha256(path) for path in sorted(args.dumps.iterdir())
             if path.name in set(DUMPS["el"].values()) | {EL_DIALECT_DUMP}}
    if not dumps:
        sys.exit(f"no Modern Greek dump in {args.dumps}")

    cells = list(fragment_cells(read_dumps(args.dumps)))
    lookup = json.loads((args.base / "mg_lookup.json").read_text(encoding="utf-8"))
    pos_lookup = json.loads(
        (args.base / "mg_pos_lookup.json").read_text(encoding="utf-8"))
    files = derive(cells, lookup, pos_lookup)
    record = {
        "schema_version": SCHEMA, "language": "el", "rule": RULE,
        "generator_sha256": sha256(GENERATOR),
        "dumps": dumps, "inputs": inputs,
        "cells": cells, "files": files,
    }
    args.out.write_text(json.dumps(record, ensure_ascii=False, indent=1,
                                   sort_keys=True) + "\n", encoding="utf-8")
    print(f"  fragment cells: {len(cells):,}")
    for name, delta in files.items():
        print(f"  {name}: {len(delta['set']):,} keys changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
