#!/usr/bin/env python3
"""Apply the builder's present completion to a manifest-pinned data snapshot.

This preserves all existing artifacts' readings. It avoids regenerating
unrelated Ancient, Medieval, corpus and model inputs for a bounded Modern fix.
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
import build_data as B
from dilemma.el_present_paradigms import complete_present_pairs, unapply_present_pairs
from dilemma.el_paradigm_gaps import stripped

INPUTS = ("mg_pairs.json", "mg_lookup.json", "lookup.db", "spell_index.db")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_heads(kaikki):
    records, hashes = {}, {}
    for source, filename in B.DUMPS["el"].items():
        path = B.resolve_dump(filename, kaikki)
        hashes[source] = sha(path)
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if '"verb"' not in line:
                    continue
                row = json.loads(line)
                if B.is_independent_finite_entry(row, source):
                    word = B.strip_length_marks(row["word"])
                    records.setdefault(word, []).append(B.verb_head_evidence(row, source))
    return records, hashes


def refresh(base, output, manifest, heads, source_hashes):
    if base.resolve() == output.resolve():
        raise ValueError("The pinned base and output must be separate")
    pins = json.loads(manifest.read_text())["files"]
    hashes = {name: sha(base / name) for name in INPUTS}
    for name, digest in hashes.items():
        if digest != pins["data/" + name]["sha256"]:
            raise ValueError(f"Base {name} differs from the manifest")
    pairs = json.loads((base / "mg_pairs.json").read_text())
    lookup = {form: (lemma, 1) for form, lemma in
              json.loads((base / "mg_lookup.json").read_text()).items()}
    audit = complete_present_pairs(pairs, lookup, heads)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("lookup.db", "spell_index.db"):
        shutil.copyfile(base / name, output / name)
    db_rows, spell_rows = [], []
    with sqlite3.connect(output / "lookup.db") as db, sqlite3.connect(output / "spell_index.db") as spell:
        for change in audit["lookup_additions"]:
            form, lemma = change["form"], change["lemma"]
            lemma_row = db.execute("SELECT id FROM lemmas WHERE text=?", (lemma,)).fetchone()
            if lemma_row is None:
                raise ValueError(f"Independent head is absent from lookup.db: {lemma}")
            existing = db.execute("SELECT lemma_id,src,lang FROM lookup WHERE form=?", (form,)).fetchall()
            if any(src == "el" and lang in {"el", "all"} for _, src, lang in existing):
                continue
            language = "el" if any(lang == "all" for _, _, lang in existing) else "all"
            row = [form, lemma_row[0], "el", language]
            db.execute("INSERT INTO lookup (form, lemma_id, src, lang) VALUES (?,?,?,?)", row)
            db_rows.append({"original": None, "corrected": row})
        for change in audit["changes"]:
            form = change["corrected"]["form"]
            key = stripped(form)
            old = spell.execute("SELECT forms FROM spell WHERE stripped=?", (key,)).fetchone()
            entries = old[0].splitlines() if old else []
            if any(entry.split("\t")[0] == form for entry in entries):
                continue
            entries.append(form + "\tel")
            value = "\n".join(entries)
            spell.execute("INSERT OR REPLACE INTO spell VALUES (?,?)", (key, value))
            spell_rows.append({"key": key, "original": old[0] if old else None,
                               "corrected": value})
    (output / "mg_pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "mg_lookup.json").write_text(
        json.dumps({form: value[0] for form, value in lookup.items()},
                   ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    audit.update({"input_sha256": hashes, "source_sha256": source_hashes,
                  "generator_sha256": {name: sha(ROOT / name) for name in (
                      "dilemma/el_present_paradigms.py", "scripts/complete_el_present_snapshot.py")},
                  "lookup_rows": db_rows, "spell_rows": spell_rows,
                  "output_sha256": {name: sha(output / name) for name in INPUTS}})
    changes = output / "changes"
    changes.mkdir(exist_ok=True)
    (changes / "el_present_completion.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def unapply(output, audit):
    for name, digest in audit["output_sha256"].items():
        if sha(output / name) != digest:
            raise ValueError(f"{name} changed after completion; refusing to undo unrelated work")
    pairs = json.loads((output / "mg_pairs.json").read_text())
    lookup = {f: (lemma, 1) for f, lemma in
              json.loads((output / "mg_lookup.json").read_text()).items()}
    unapply_present_pairs(pairs, lookup, audit)
    with sqlite3.connect(output / "lookup.db") as db, sqlite3.connect(output / "spell_index.db") as spell:
        for change in reversed(audit["lookup_rows"]):
            db.execute("DELETE FROM lookup WHERE form=? AND lemma_id=? AND src=? AND lang=?",
                       change["corrected"])
        for change in reversed(audit["spell_rows"]):
            if change["original"] is None:
                spell.execute("DELETE FROM spell WHERE stripped=?", (change["key"],))
            else:
                spell.execute("UPDATE spell SET forms=? WHERE stripped=?",
                              (change["original"], change["key"]))
    (output / "mg_pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "mg_lookup.json").write_text(
        json.dumps({f: v[0] for f, v in lookup.items()}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/hf_manifest.json")
    parser.add_argument("--kaikki", type=Path)
    parser.add_argument("--undo", type=Path, help="Unapply the exact recorded snapshot completion")
    args = parser.parse_args()
    if args.undo:
        unapply(args.output, json.loads(args.undo.read_text()))
        print("Restored the original logical rows")
        return
    if args.base is None or args.kaikki is None:
        parser.error("--base and --kaikki are required without --undo")
    heads, hashes = source_heads(args.kaikki)
    audit = refresh(args.base, args.output, args.manifest, heads, hashes)
    print(json.dumps({"added_pairs": len(audit["changes"]),
                      "added_lookup_keys": len(audit["lookup_additions"]),
                      "independent_heads": len(audit["head_evidence"]),
                      "lookup_rows": len(audit["lookup_rows"]),
                      "unsupported_heads": audit["unsupported_heads"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
