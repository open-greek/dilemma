#!/usr/bin/env python3
"""Repair a pinned Modern Greek snapshot using the builder's source rules.

Archive the exact source fields first, then replay from that archive. The
archive pins its original Hub revision and four base hashes, so subsequent
rebuilds do not need a changing Wiktionary dump or overwrite other languages.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_data as B
from dilemma.el_imperfect_paradigms import complete_imperfect_pairs, unapply_imperfect_pairs
from dilemma.el_paradigm_gaps import stripped

INPUTS = ("mg_pairs.json", "mg_lookup.json", "lookup.db", "spell_index.db")
SOURCE_FILE = "el_verb_repair_sources.json.gz"
AUDIT_FILE = "el_verb_data_repairs.json"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def archive_sources(kaikki, destination, pins, base_revision):
    result = {"schema_version": 1, "language": "el", "base_repo": "ciscoriordan/dilemma",
              "base_revision": base_revision,
              "input_sha256": {name: pins["data/" + name]["sha256"] for name in INPUTS},
              "source_sha256": {}, "heads": {}, "unaccented_heads": {},
              "lemma_pos": {}, "tense_cells": []}
    for source, filename in B.DUMPS["el"].items():
        path = B.resolve_dump(filename, kaikki)
        result["source_sha256"][source] = sha(path)
        with path.open(encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                word = B.strip_length_marks(e.get("word", "")); pos = e.get("pos", "")
                if not word or " " in word or not B._is_greek(word):
                    continue
                if pos not in result["lemma_pos"].setdefault(word, []):
                    result["lemma_pos"][word].append(pos)
                if B.is_independent_finite_entry(e, source):
                    result["heads"].setdefault(word, []).append(B.verb_head_evidence(e, source))
                if (word == B.strip_accents(word.lower())
                        and B.is_independent_lexical_entry(e, source)):
                    result["unaccented_heads"].setdefault(word, []).append({
                        "source": source, "headword": word, "pos": pos,
                        "tags": e.get("tags") or [],
                        "senses": [{k: s.get(k) or [] for k in ("glosses", "tags", "form_of", "alt_of")}
                                   for s in e.get("senses") or []]})
                if pos != "verb":
                    continue
                page_dialect = next((t for s in e.get("senses", []) for t in s.get("tags", [])
                                     if t in B.DIALECT_PREFIXES), "")
                header, dialect = "", page_dialect
                for cell in e.get("forms", []):
                    tags = cell.get("tags", [])
                    if "table-tags" in tags:
                        header = B._parse_verb_tense(cell.get("form", ""))
                        dialect = page_dialect or B._parse_dialect(cell.get("form", ""))
                        continue
                    if any(t in tags for t in ("romanization", "inflection-template", "class")):
                        continue
                    original = [t for t in tags if t != "canonical"]
                    if dialect and dialect not in original:
                        original.append(dialect)
                    corrected = B.verb_cell_tense_tags(original, header)
                    if not (header and header not in original and B._VERB_AXES.intersection(tags)):
                        continue
                    original.append(header)
                    if set(original) == set(corrected):
                        continue
                    form = B.strip_length_marks(cell.get("form", ""))
                    forms = [form]
                    m = re.fullmatch(r"(.+?)\((.+?)\)", form)
                    if m:
                        forms = [m[1], m[1] + m[2]]
                    for form in dict.fromkeys(f for v in forms for f in (v, B.to_monotonic(v))):
                        if not B._is_greek(form) or " " in form:
                            continue
                        result["tense_cells"].append({"source": source, "source_lemma": word,
                            "form": form, "table_header": header, "source_cell": cell,
                            "original_tags": original, "corrected_tags": corrected})
    payload = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as f:
        with gzip.GzipFile(filename="", fileobj=f, mode="wb", mtime=0) as g:
            g.write(payload)
    return result


def pair_key(r):
    return r["form"], r["lemma"], r.get("pos"), tuple(sorted(set(r.get("tags", ()))))


def repair_pairs(pairs, lookup, sources, head_changes):
    index = {}
    for proof in sources["tense_cells"]:
        # The archive preserves input evidence; current code determines
        # the correction when replaying it.
        explicit = [t for t in proof["original_tags"] if t != proof["table_header"]]
        proof = {**proof, "corrected_tags": B.verb_cell_tense_tags(explicit, proof["table_header"])}
        lemma = proof["source_lemma"]
        for target in {lemma, lookup.get(lemma, (lemma,))[0]}:
            key = (proof["form"], target, "verb", tuple(sorted(set(proof["original_tags"]))))
            index.setdefault(key, []).append(proof)
    recovered = {c["form"]: c for c in head_changes}
    changes = {}
    for i, row in enumerate(pairs):
        corrected = dict(row); evidence = []
        if pair_key(row) in index:
            proofs = index[pair_key(row)]
            variants = {tuple(p["corrected_tags"]) for p in proofs}
            if len({frozenset(v) for v in variants}) != 1:
                raise ValueError("Conflicting source tense corrections: " + row["form"])
            corrected["tags"] = list(sorted(next(iter(variants))))
            evidence.extend(proofs)
        recovery = recovered.get(row["form"])
        if (recovery and row["lemma"] == recovery["original"][0]
                and row.get("pos") in {p["pos"] for p in recovery["evidence"]}
                and not {"alt-of", "form-of"} & set(row.get("tags", ()))):
            corrected["lemma"] = row["form"]
            evidence.extend(recovery["evidence"])
        if corrected != row:
            changes[i] = {"index": i, "original": row, "corrected": corrected, "evidence": evidence}
    seen = {pair_key(row) for i, row in enumerate(pairs) if i not in changes}
    result = []
    for i, row in enumerate(pairs):
        if i in changes:
            c = changes[i]; key = pair_key(c["corrected"])
            if key in seen:
                c["corrected"] = None
                continue
            seen.add(key); row = c["corrected"]
        result.append(row)
    return result, list(changes.values())


def refresh(base, output, sources):
    if base.resolve() == output.resolve():
        raise ValueError("Base and output must be separate")
    hashes = {name: sha(base / name) for name in INPUTS}
    if hashes != sources["input_sha256"]:
        raise ValueError("Base inputs differ from the pinned source archive")
    pairs = json.loads((base / "mg_pairs.json").read_text())
    original_count = len(pairs)
    lookup = {f: (v, 1) for f, v in json.loads((base / "mg_lookup.json").read_text()).items()}
    head_changes = B.recover_unaccented_heads(lookup, sources["unaccented_heads"], sources["lemma_pos"])
    pairs, pair_changes = repair_pairs(pairs, lookup, sources, head_changes)
    imperfect = complete_imperfect_pairs(pairs, lookup, sources["heads"])
    imperfect["source_sha256"] = sources["source_sha256"]
    output.mkdir(parents=True, exist_ok=True)
    for name in ("lookup.db", "spell_index.db"):
        shutil.copyfile(base / name, output / name)
    sql_changes, added_lemmas, spell_changes = [], [], []
    with sqlite3.connect(output / "lookup.db") as db, sqlite3.connect(output / "spell_index.db") as spell:
        def lemma_id(lemma):
            found = db.execute("SELECT id FROM lemmas WHERE text=?", (lemma,)).fetchone()
            if found:
                return found[0]
            cursor = db.execute("INSERT INTO lemmas(text) VALUES (?)", (lemma,))
            added_lemmas.append([cursor.lastrowid, lemma]); return cursor.lastrowid
        for c in head_changes:
            form = c["form"];target = lemma_id(form)
            rows = db.execute("SELECT form,lemma_id,src,lang FROM lookup WHERE form=? AND src='el'", (form,)).fetchall()
            for old in rows:
                new = [form, target, old[2], old[3]]
                db.execute("UPDATE lookup SET lemma_id=? WHERE form=? AND src=? AND lang=?",
                           (target, form, old[2], old[3]))
                sql_changes.append({"original": list(old), "corrected": new})
        for c in imperfect["lookup_additions"]:
            form, lemma = c["form"], c["corrected"][0]
            rows = db.execute("SELECT lemma_id,src,lang FROM lookup WHERE form=?", (form,)).fetchall()
            if any(src == "el" for _, src, _ in rows):
                continue
            language = "el" if any(lang == "all" for _, _, lang in rows) else "all"
            new = [form, lemma_id(lemma), "el", language]
            db.execute("INSERT INTO lookup VALUES (?,?,?,?)", new)
            sql_changes.append({"original": None, "corrected": new})
        for c in imperfect["changes"]:
            form = c["corrected"]["form"];key = stripped(form)
            old = spell.execute("SELECT forms FROM spell WHERE stripped=?", (key,)).fetchone()
            entries = old[0].splitlines() if old else []
            if any(e.split("\t")[0] == form for e in entries):
                continue
            entries.append(form + "\tel"); value = "\n".join(entries)
            spell.execute("INSERT OR REPLACE INTO spell VALUES (?,?)", (key, value))
            spell_changes.append({"key": key, "original": old[0] if old else None, "corrected": value})
    (output / "mg_pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "mg_lookup.json").write_text(json.dumps({f: v[0] for f, v in lookup.items()}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    audit = {"schema_version": 1, "language": "el", "rule": "source_scoped_verb_repairs_v1",
             "original_pair_count": original_count, "input_sha256": hashes,
             "source_sha256": sources["source_sha256"], "base_revision": sources["base_revision"],
             "pair_changes": pair_changes, "head_changes": head_changes,
             "lookup_rows": sql_changes, "added_lemmas": added_lemmas, "spell_rows": spell_changes,
             "generator_sha256": {name: sha(ROOT / name) for name in (
                 "build_data.py", "dilemma/el_imperfect_paradigms.py", "scripts/repair_el_verb_snapshot.py")},
             "output_sha256": {name: sha(output / name) for name in INPUTS}}
    save(output / "changes" / AUDIT_FILE, audit)
    save(output / "changes/el_imperfect_completion.json", imperfect)
    return audit, imperfect


def unapply(output, audit, imperfect):
    for name, expected in audit["output_sha256"].items():
        if sha(output / name) != expected:
            raise ValueError("Snapshot changed after generation: " + name)
    pairs = json.loads((output / "mg_pairs.json").read_text())
    lookup = {f: (v, 1) for f, v in json.loads((output / "mg_lookup.json").read_text()).items()}
    # The serialized flat lookup does not carry confidence values.
    for c in imperfect["lookup_additions"]:
        if lookup.get(c["form"], (None,))[0] != c["corrected"][0]:
            raise ValueError("Lookup changed after generation")
        lookup[c["form"]] = tuple(c["corrected"])
    unapply_imperfect_pairs(pairs, lookup, imperfect)
    changes = {c["index"]: c for c in audit["pair_changes"]}; restored = []; cursor = 0
    for i in range(audit["original_pair_count"]):
        c = changes.get(i)
        if c:
            if c["corrected"] is not None:
                if pairs[cursor] != c["corrected"]:
                    raise ValueError("Pair changed after repair")
                cursor += 1
            restored.append(c["original"])
        else:
            restored.append(pairs[cursor]);cursor += 1
    if cursor != len(pairs):
        raise ValueError("Unexpected extra pairs")
    for c in audit["head_changes"]:
        lookup[c["form"]] = tuple(c["original"])
    with sqlite3.connect(output / "lookup.db") as db, sqlite3.connect(output / "spell_index.db") as spell:
        for c in reversed(audit["lookup_rows"]):
            db.execute("DELETE FROM lookup WHERE form=? AND lemma_id=? AND src=? AND lang=?", c["corrected"])
            if c["original"] is not None:
                db.execute("INSERT INTO lookup VALUES (?,?,?,?)", c["original"])
        for row in reversed(audit["added_lemmas"]):
            db.execute("DELETE FROM lemmas WHERE id=? AND text=?", row)
        for c in reversed(audit["spell_rows"]):
            if c["original"] is None:
                spell.execute("DELETE FROM spell WHERE stripped=?", (c["key"],))
            else:
                spell.execute("UPDATE spell SET forms=? WHERE stripped=?", (c["original"], c["key"]))
    (output / "mg_pairs.json").write_text(json.dumps(restored, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "mg_lookup.json").write_text(json.dumps({f: v[0] for f, v in lookup.items()}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sources", type=Path)
    p.add_argument("--kaikki", type=Path)
    p.add_argument("--download-base", action="store_true")
    p.add_argument("--undo", action="store_true")
    args = p.parse_args()
    if args.undo:
        unapply(args.output, json.loads((args.output / "changes" / AUDIT_FILE).read_text()),
                json.loads((args.output / "changes/el_imperfect_completion.json").read_text()))
        print("Restored original logical rows");return
    source_path = args.sources or args.output / SOURCE_FILE
    if args.kaikki:
        if source_path.exists():
            raise ValueError("Refusing to replace an existing source archive")
        from huggingface_hub import HfApi
        archive_sources(args.kaikki, source_path, json.loads((ROOT / "data/hf_manifest.json").read_text())["files"],
                        HfApi().repo_info("ciscoriordan/dilemma").sha)
    else:
        pin = json.loads((ROOT / "data/hf_manifest.json").read_text())["files"].get("data/" + SOURCE_FILE)
        if pin and sha(source_path) != pin["sha256"]:
            raise ValueError("Source archive differs from the manifest; fetch the pinned source version")
    with gzip.open(source_path, "rt", encoding="utf-8") as f:
        sources = json.load(f)
    if args.download_base:
        from huggingface_hub import hf_hub_download
        args.base.mkdir(parents=True, exist_ok=True)
        for name in INPUTS:
            target = args.base / name
            if not target.exists():
                shutil.copyfile(hf_hub_download(sources["base_repo"], "data/" + name,
                                               revision=sources["base_revision"]), target)
    audit, imperfect = refresh(args.base, args.output, sources)
    print(json.dumps({"repaired_pairs": len(audit["pair_changes"]), "recovered_exact_heads": len(audit["head_changes"]),
                      "added_imperfect_pairs": len(imperfect["changes"]), "affected_imperfect_heads": len(imperfect["head_evidence"])}))


if __name__ == "__main__":
    main()
