#!/usr/bin/env python3
"""Refresh a pinned lookup/spelling snapshot with reviewed Modern Greek gaps.

This bounded refresh uses the same rules as build_lookup_db.py. Existing
Modern, Ancient, and combined readings are preserved. The base files must
match the supplied manifest, and every added row is recorded in JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dilemma.el_paradigm_gaps import complete_lookup, stripped


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh(base, output, manifest):
    files = json.loads(manifest.read_text())["files"]
    inputs = {}
    for name in ("lookup.db", "spell_index.db", "mg_lookup.json"):
        path = base / name
        digest = sha256(path)
        if digest != files[f"data/{name}"]["sha256"]:
            raise ValueError(f"base {name} differs from the pinned manifest")
        inputs[name] = digest
    lookup = json.loads((base / "mg_lookup.json").read_text())
    additions = complete_lookup(lookup)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("lookup.db", "spell_index.db"):
        if (base / name).resolve() == (output / name).resolve():
            raise ValueError("base and output must be separate")
        shutil.copyfile(base / name, output / name)
    changes = []
    with sqlite3.connect(output / "lookup.db") as db, sqlite3.connect(output / "spell_index.db") as spell:
        for cell in additions:
            form, lemma = cell["form"], cell["lemma"]
            lemma_row = db.execute("SELECT id FROM lemmas WHERE text=?", (lemma,)).fetchone()
            if lemma_row is None:
                raise ValueError(f"missing independently defined head {lemma}")
            rows = db.execute("SELECT lemma_id,src,lang FROM lookup WHERE form=?", (form,)).fetchall()
            if any(src == "el" and lang in {"el", "all"} for _, src, lang in rows):
                continue
            language = "el" if any(lang == "all" for _, _, lang in rows) else "all"
            db.execute("INSERT INTO lookup VALUES (?,?,?,?)", (form, lemma_row[0], "el", language))
            key = stripped(form)
            old = spell.execute("SELECT forms FROM spell WHERE stripped=?", (key,)).fetchone()
            entries = old[0].splitlines() if old else []
            if not any(entry.split("\t")[0] == form for entry in entries):
                entries.append(f"{form}\tel")
            new = "\n".join(entries)
            spell.execute("INSERT OR REPLACE INTO spell VALUES (?,?)", (key, new))
            changes.append({**cell, "lookup_row": [form, lemma_row[0], "el", language],
                            "spell_key": key, "spell_before": old[0] if old else None,
                            "spell_after": new})
    audit = {"schema_version": 1, "language": "el", "input_sha256": inputs,
             "generator_sha256": {p: sha256(ROOT / p) for p in
                                   ("dilemma/el_paradigm_gaps.py", "scripts/complete_el_paradigms.py")},
             "changes": changes,
             "output_sha256": {name: sha256(output / name) for name in ("lookup.db", "spell_index.db")}}
    path = output / "changes" / "el_paradigm_completion.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    return audit


def unapply_refresh(data, audit):
    """Restore the original logical lookup/spelling rows, in reverse order."""
    for name, digest in audit["output_sha256"].items():
        if sha256(data / name) != digest:
            raise ValueError(f"{name} changed after this refresh; refusing to undo unrelated work")
    with sqlite3.connect(data / "lookup.db") as db, sqlite3.connect(data / "spell_index.db") as spell:
        for change in reversed(audit["changes"]):
            db.execute("DELETE FROM lookup WHERE form=? AND lemma_id=? AND src=? AND lang=?",
                       change["lookup_row"])
            if change["spell_before"] is None:
                spell.execute("DELETE FROM spell WHERE stripped=?", (change["spell_key"],))
            else:
                spell.execute("UPDATE spell SET forms=? WHERE stripped=?",
                              (change["spell_before"], change["spell_key"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--undo", type=Path, help="Undo the exact refresh recorded by this audit JSON")
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/hf_manifest.json")
    args = parser.parse_args()
    if args.undo:
        unapply_refresh(args.output, json.loads(args.undo.read_text()))
        print("Restored the original logical rows")
        return
    if args.base is None:
        parser.error("--base is required unless --undo is used")
    report = refresh(args.base, args.output, args.manifest)
    print(f"Added {len(report['changes'])} source-backed lookup rows")


if __name__ == "__main__":
    main()
