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

# These old English table rows are absent from the pinned Kaikki dump.
# The rendered, revision-pinned tables explicitly label the cells below.
# Preserve the snapshot's canonical lemma; only remove its extra heading.
REVIEWED_TABLE_READINGS = (
    ('present', ('εξασθενίζομε', 'εξασθενίζοντας'),
     'https://en.wiktionary.org/w/index.php?title=εξασθενίζω&oldid=91814395'),
    ('present', ('εξασθενώ', 'εξασθενείς', 'εξασθενεί', 'εξασθενούμε',
                 'εξασθενείτε', 'εξασθενούν', 'εξασθενούνε', 'εξασθενώντας'),
     'https://en.wiktionary.org/w/index.php?title=εξασθενώ&oldid=93555953'),
    ('imperfect', ('εξασθενούσα', 'εξασθενούσες', 'εξασθενούσε', 'εξασθενούσαμε',
                   'εξασθενούσατε', 'εξασθενούσαν', 'εξασθενούσανε'),
     'https://en.wiktionary.org/w/index.php?title=εξασθενώ&oldid=93555953'),
)


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def archive_sources(kaikki, destination, pins, base_revision, base_pairs=()):
    specific = set(B.VERB_TENSE_LABELS.values())
    affected = {r['form'] for r in base_pairs if r.get('pos') == 'verb'
                and len(set(r.get('tags', ())) & specific) > 1}
    result = {"schema_version": 1, "language": "el", "base_repo": "ciscoriordan/dilemma",
              "base_revision": base_revision,
              "input_sha256": {name: pins["data/" + name]["sha256"] for name in INPUTS},
              "source_sha256": {}, "heads": {}, "unaccented_heads": {},
              "lemma_pos": {}, "tense_cells": [], "tense_witnesses": []}
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
                for sense in e.get("senses", []):
                    original = sorted(set(e.get("tags", [])) | set(sense.get("tags", [])) | {"form-of"})
                    corrected = B.modern_verb_source_tags(original, sense)
                    if (set(original) == set(corrected)
                            and len(B.separate_modern_verb_tenses(corrected)) == 1):
                        continue
                    for ref in sense.get("form_of", []):
                        result["tense_cells"].append({"source": source,
                            "source_lemma": B.strip_length_marks(ref.get("word", "")),
                            "form": word, "table_header": "", "source_sense": sense,
                            "original_tags": original})
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
                    corrected = B.verb_cell_tense_tags(B.modern_verb_source_tags(original, cell), header)
                    form = B.strip_length_marks(cell.get("form", ""))
                    if (form in affected and cell.get('source') == 'conjugation'
                            and len(set(corrected) & specific) == 1):
                        result['tense_witnesses'].append({'source':source, 'source_lemma':word,
                            'form':form, 'source_cell':cell, 'corrected_tags':corrected})
                    if header and header not in original and B._VERB_AXES.intersection(tags):
                        original.append(header)
                    if (set(original) == set(corrected)
                            and len(B.separate_modern_verb_tenses(corrected)) == 1):
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
    raw_keys = {(p['form'], tuple(sorted(set(p['original_tags'])))) for p in sources['tense_cells']}
    matching_rows = {}
    for row in pairs:
        key = (row['form'], tuple(sorted(set(row.get('tags', [])))))
        if row.get('pos') == 'verb' and key in raw_keys:
            matching_rows.setdefault(key, []).append(row)
    source_owners = {}
    for proof in sources['tense_cells']:
        key = (proof['form'], tuple(sorted(set(proof['original_tags']))))
        for row in matching_rows.get(key, ()):
            source_owners.setdefault(proof['source_lemma'], {}).setdefault(row['lemma'], row)
    head_readings = {}
    for row in pairs:
        if (row.get("pos") == "verb"
                and {"first-person", "singular"} <= set(row.get("tags", []))):
            head_readings.setdefault(row["form"], {}).setdefault(row["lemma"], row)
    for proof in sources["tense_cells"]:
        # The archive preserves input evidence; current code determines
        # the correction when replaying it.
        record = proof.get("source_sense") or proof.get("source_cell") or {}
        if "source_sense" in proof:
            corrected = B.modern_verb_source_tags(proof["original_tags"], record)
        else:
            explicit = [t for t in proof["original_tags"] if t != proof["table_header"]]
            corrected = B.verb_cell_tense_tags(B.modern_verb_source_tags(explicit, record), proof["table_header"])
        proof = {**proof, "corrected_readings": B.separate_modern_verb_tenses(corrected)}
        lemma = proof["source_lemma"]
        # The published pairs have already canonicalized spelling and voice
        # variants. Retain that owner, with its original first-person row as
        # evidence, when matching the source page's table.
        agreed_owner = source_owners.get(lemma, {})
        if len(agreed_owner) != 1:
            agreed_owner = {}
        for target in {lemma, lookup.get(lemma, (lemma,))[0], *head_readings.get(lemma, {}), *agreed_owner}:
            evidence = proof
            if target in head_readings.get(lemma, {}):
                evidence = {**proof, "published_head_reading": head_readings[lemma][target]}
            elif target in agreed_owner:
                evidence = {**proof, "published_table_owner": agreed_owner[target]}
            key = (proof["form"], target, "verb", tuple(sorted(set(proof["original_tags"]))))
            index.setdefault(key, []).append(evidence)
    recovered = {c["form"]: c for c in head_changes}
    witnesses = {}
    for proof in sources.get('tense_witnesses', []):
        witnesses.setdefault(proof['form'], []).append(proof)
    specific = set(B.VERB_TENSE_LABELS.values())
    axes = ({'first-person','second-person','third-person'}, {'singular','plural','dual'},
            {'indicative','subjunctive','imperative','optative'}, {'active','passive','middle'})
    changes = {}
    for i, row in enumerate(pairs):
        corrected = dict(row); evidence = []
        if pair_key(row) in index:
            proofs = index[pair_key(row)]
            variants = {tuple(sorted(tuple(sorted(t)) for t in p["corrected_readings"])) for p in proofs}
            if len(variants) != 1:
                raise ValueError("Conflicting source tense corrections: " + row["form"])
            tags = next(iter(variants))
            corrected["tags"] = list(tags[0])
            evidence.extend(proofs)
            extra = [{**row, "tags": list(t)} for t in tags[1:]]
        else:
            extra = []
            old_tags = set(row.get('tags', []))
            if row.get('pos') == 'verb' and len(old_tags & specific) > 1:
                votes = {}
                for proof in witnesses.get(row['form'], []):
                    tags = set(proof['corrected_tags'])
                    if (proof['source_lemma'] != row['lemma']
                            or not (tags & axes[0]) or not (tags & axes[1])
                            or any(tags & a and old_tags & a and tags & a != old_tags & a for a in axes)):
                        continue
                    tense = tags & specific & old_tags
                    if len(tense) == 1:
                        votes.setdefault(next(iter(tense)), []).append(proof)
                if votes:
                    readings = [{**row,'tags':sorted((old_tags - specific) | {t})} for t in sorted(votes)]
                    corrected = readings[0]; extra = readings[1:]
                    evidence.extend(p for v in votes.values() for p in v)
                elif row['lemma'] == 'εξασθενίζω':
                    for tense, forms, url in REVIEWED_TABLE_READINGS:
                        if row['form'] in forms and old_tags & specific == {tense, 'perfect'}:
                            corrected['tags'] = sorted(old_tags - {'perfect'})
                            evidence.append({'source':'en', 'revision_url':url,
                                'source_lemma':'εξασθενώ' if '93555953' in url else 'εξασθενίζω',
                                'form':row['form'], 'confirmed_tense':tense,
                                'evidence':'Reviewed exact cell in the conjugation table; perfect is periphrastic.'})
        recovery = recovered.get(row["form"])
        if (recovery and row["lemma"] == recovery["original"][0]
                and row.get("pos") in {p["pos"] for p in recovery["evidence"]}
                and not {"alt-of", "form-of"} & set(row.get("tags", ()))):
            corrected["lemma"] = row["form"]
            evidence.extend(recovery["evidence"])
        if corrected != row:
            changes[i] = {"index": i, "original": row, "corrected": corrected,
                          "additional_readings": extra, "evidence": evidence}
    seen = {pair_key(row) for i, row in enumerate(pairs) if i not in changes}
    result = []
    for i, row in enumerate(pairs):
        if i in changes:
            c = changes[i]
            readings = [c["corrected"], *c["additional_readings"]]
            kept = []
            for reading in readings:
                key = pair_key(reading)
                if key not in seen:
                    seen.add(key); kept.append(reading)
            c["corrected"] = kept[0] if kept else None
            c["additional_readings"] = kept[1:]
            result.extend(kept)
            continue
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
            db.execute("INSERT INTO lookup (form, lemma_id, src, lang) VALUES (?,?,?,?)", new)
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
            for reading in ([c["corrected"]] if c["corrected"] is not None else []) + c.get("additional_readings", []):
                if pairs[cursor] != reading:
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
                db.execute("INSERT INTO lookup (form, lemma_id, src, lang) VALUES (?,?,?,?)", c["original"])
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
                        HfApi().repo_info("ciscoriordan/dilemma").sha,
                        json.loads((args.base / 'mg_pairs.json').read_text()))
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
