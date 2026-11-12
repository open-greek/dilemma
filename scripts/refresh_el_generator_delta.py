#!/usr/bin/env python3
"""Carry a Modern Greek generator change onto a pinned published snapshot.

The published Modern Greek tables are a pinned snapshot with recorded repairs
(scripts/repair_el_verb_snapshot.py and its siblings), so a generator change
reaches them as a bounded refresh rather than a full rebuild.

``diff`` runs on the outputs of the old and the new ``build_data.py --lang el``
over the same dumps, so every difference it records was made by the code
change. It writes one delta record: the dumps' checksums, both generator
commits, and each changed key, POS reading, pair, head and participle cell.

``apply`` checks the base files against a manifest, then makes each recorded
change only where the base still holds the old generator's value; anything
else is reported as a conflict and left as it is. It writes the refreshed
files and an audit with every applied change and the input and output hashes.
``undo`` restores the base from that audit and refuses files changed since.

    python scripts/refresh_el_generator_delta.py diff OLD_DATA NEW_DATA \
        --dumps DUMP_DIR --old-commit SHA --new-commit SHA --out DELTA.json
    python scripts/refresh_el_generator_delta.py apply --base DIR \
        --manifest data/hf_manifest.json --delta DELTA.json --output DIR
    python scripts/refresh_el_generator_delta.py undo --data DIR --audit AUDIT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

SCHEMA = 1
RULE = "el_generator_delta_refresh_v1"
# name -> (shape, key inside the file holding the records, record identity)
FILES = {
    "mg_lookup.json": ("map", None, None),
    "mg_pos_lookup.json": ("map", None, None),
    "mg_pairs.json": ("records", None, None),
    "el_participles.json": ("records", "cells", None),
    "el_noun_heads.json": ("records", "heads", ("lemma", "pos")),
    "el_adjective_heads.json": ("records", "heads", ("lemma", "pos")),
    "el_verb_heads.json": ("records", "heads", ("lemma", "pos")),
    "el_closed_class_heads.json": ("records", "heads", ("lemma", "pos")),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


# The layout build_data.py writes each file in, so a refreshed file differs
# from the published one only where its content does.
COMPACT = {"mg_lookup.json", "mg_pos_lookup.json"}
TRAILING_NEWLINE = {"el_participles.json", "el_noun_heads.json",
                    "el_adjective_heads.json", "el_verb_heads.json",
                    "el_closed_class_heads.json"}


def dump(value, path: Path) -> None:
    if path.name in COMPACT:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2)
    if path.name in TRAILING_NEWLINE:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def identity(record: dict, fields) -> str:
    if fields is None:
        return canonical(record)
    return canonical([record.get(field) for field in fields])


def records_of(document, inner):
    return document if inner is None else document[inner]


def diff_file(name: str, old, new) -> dict:
    shape, inner, fields = FILES[name]
    if shape == "map":
        changes = {key: [old.get(key), new.get(key)]
                   for key in sorted(old.keys() | new.keys())
                   if old.get(key) != new.get(key)}
        return {"set": changes} if changes else {}
    old_records = {identity(r, fields): r for r in records_of(old, inner)}
    new_records = {identity(r, fields): r for r in records_of(new, inner)}
    remove = [old_records[k] for k in sorted(old_records)
              if k not in new_records or old_records[k] != new_records[k]]
    add = [new_records[k] for k in sorted(new_records)
           if k not in old_records or old_records[k] != new_records[k]]
    return {key: value for key, value in (("remove", remove), ("add", add)) if value}


def cmd_diff(args) -> int:
    old_dir, new_dir = Path(args.old), Path(args.new)
    files = {}
    for name in FILES:
        present = [(d / name).exists() for d in (old_dir, new_dir)]
        if not any(present):
            continue                     # another step writes this file
        if not all(present):
            sys.exit(f"{name} is missing from one of the generator outputs")
        delta = diff_file(name, load(old_dir / name), load(new_dir / name))
        if delta:
            files[name] = delta
    dumps = {path.name: sha256(path) for path in sorted(Path(args.dumps).iterdir())
             if path.suffix == ".jsonl"}
    record = {
        "schema_version": SCHEMA, "language": "el", "rule": RULE,
        "generator": {"old": args.old_commit, "new": args.new_commit},
        "dumps": dumps,
        "inputs": {name: sha256(Path(args.new) / name) for name in args.input},
        "files": files,
    }
    Path(args.out).write_text(json.dumps(record, ensure_ascii=False, indent=1,
                                         sort_keys=True), encoding="utf-8")
    for name, delta in files.items():
        print(f"  {name}: " + ", ".join(f"{len(v):,} {k}" for k, v in delta.items()))
    return 0


def apply_file(name: str, document, delta: dict, log: list):
    shape, inner, fields = FILES[name]
    if shape == "map":
        for key, (before, after) in delta.get("set", {}).items():
            have = document.get(key)
            if have == after:
                status = "already"
            elif have == before:
                status = "applied"
                if after is None:
                    del document[key]
                else:
                    document[key] = after
            else:
                status = "conflict"
            log.append({"file": name, "key": key, "base": have,
                        "before": before, "after": after, "status": status})
        return document
    records = records_of(document, inner)
    present = {identity(r, fields): i for i, r in enumerate(records)}
    drop = set()
    for record in delta.get("remove", []):
        index = present.get(identity(record, fields))
        if index is not None and records[index] == record:
            drop.add(index)
            status = "applied"
        else:
            status = "absent"
        log.append({"file": name, "remove": record, "status": status})
    kept = [r for i, r in enumerate(records) if i not in drop]
    have = {identity(r, fields) for r in kept}
    for record in delta.get("add", []):
        if identity(record, fields) in have:
            status = "already"
        else:
            kept.append(record)
            have.add(identity(record, fields))
            status = "applied"
        log.append({"file": name, "add": record, "status": status})
    if inner is None:
        return kept
    document[inner] = kept
    return document


def cmd_apply(args) -> int:
    base, output = Path(args.base), Path(args.output)
    if base.resolve() == output.resolve():
        sys.exit("base and output must be separate directories")
    pinned = load(Path(args.manifest))["files"]
    delta = load(Path(args.delta))
    output.mkdir(parents=True, exist_ok=True)
    inputs, outputs, log = {}, {}, []
    for name, file_delta in delta["files"].items():
        digest = sha256(base / name)
        want = pinned.get(f"data/{name}", {}).get("sha256")
        if want and digest != want:
            sys.exit(f"base {name} differs from the pinned manifest")
        inputs[name] = digest
        document = apply_file(name, load(base / name), file_delta, log)
        dump(document, output / name)
        outputs[name] = sha256(output / name)
    counts = {}
    for row in log:
        key = f"{row['file']}: {row['status']}"
        counts[key] = counts.get(key, 0) + 1
    audit = {"schema_version": SCHEMA, "language": "el", "rule": RULE,
             "delta_sha256": sha256(Path(args.delta)), "inputs": inputs,
             "outputs": outputs, "counts": dict(sorted(counts.items())),
             "changes": [row for row in log if row["status"] != "already"]}
    (output / "el_generator_refresh_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=1), encoding="utf-8")
    for key, count in sorted(counts.items()):
        print(f"  {key}: {count:,}")
    conflicts = sum(n for k, n in counts.items() if k.endswith("conflict"))
    print(f"  conflicts left as published: {conflicts:,}")
    return 0


def cmd_undo(args) -> int:
    data, audit = Path(args.data), load(Path(args.audit))
    for name, digest in audit["outputs"].items():
        if sha256(data / name) != digest:
            sys.exit(f"{name} changed after the refresh; not undoing")
    for name in audit["outputs"]:
        shape, inner, fields = FILES[name]
        document = load(data / name)
        rows = [r for r in audit["changes"]
                if r["file"] == name and r["status"] == "applied"]
        if shape == "map":
            for row in rows:
                if row["before"] is None:
                    document.pop(row["key"], None)
                else:
                    document[row["key"]] = row["before"]
        else:
            records = records_of(document, inner)
            added = {identity(r["add"], fields) for r in rows if "add" in r}
            records = [r for r in records if identity(r, fields) not in added]
            records += [r["remove"] for r in rows if "remove" in r]
            document = records if inner is None else {**document, inner: records}
        dump(document, data / name)
    print("restored the refreshed files' logical content; record order may differ")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("diff")
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--dumps", required=True)
    p.add_argument("--old-commit", required=True)
    p.add_argument("--new-commit", required=True)
    p.add_argument("--input", action="append", default=[],
                   help="a further input file in NEW to record by checksum")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_diff)
    p = sub.add_parser("apply")
    p.add_argument("--base", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--delta", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_apply)
    p = sub.add_parser("undo")
    p.add_argument("--data", required=True)
    p.add_argument("--audit", required=True)
    p.set_defaults(func=cmd_undo)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
