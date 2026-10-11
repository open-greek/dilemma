#!/usr/bin/env python3
"""Record independent finite heads that a flat lookup represents by an alias.

Run after build_data.py with the same Modern Greek Kaikki inputs. This writes
only source evidence, never changes training pairs or runtime lookup tables.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from build_data import (DEFAULT_DUMP_DIR, DUMPS, resolve_dump, strip_length_marks,
                        validate_verb_head_inventory, verb_head_evidence, is_independent_finite_entry)


def required_verb_heads(pairs, lookup):
    return {row["lemma"] for row in pairs if row.get("pos") == "verb"
            and row["lemma"] in lookup and lookup[row["lemma"]] != row["lemma"]}


def build_inventory(required, paths):
    grouped, hashes = {}, {}
    for source, path in sorted(paths.items()):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for raw in stream:
                digest.update(raw)
                entry = json.loads(raw)
                lemma = strip_length_marks(entry.get("word", ""))
                if lemma not in required or entry.get("pos") != "verb":
                    continue
                if not is_independent_finite_entry(entry, source):
                    continue
                evidence = verb_head_evidence(entry, source)
                key = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
                grouped.setdefault(lemma, {})[key] = evidence
        hashes[source] = digest.hexdigest()
    missing = required - grouped.keys()
    if missing:
        raise ValueError("No independent finite verb evidence: " + ", ".join(sorted(missing)))
    inventory = {"schema_version": 1, "language": "el", "source_sha256": hashes,
                 "heads": [{"lemma": lemma, "evidence": [rows[key] for key in sorted(rows)]}
                           for lemma, rows in sorted(grouped.items())]}
    assert validate_verb_head_inventory(inventory) == required
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaikki", type=Path, default=DEFAULT_DUMP_DIR)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    with (args.data / "mg_pairs.json").open(encoding="utf-8") as stream:
        pairs = json.load(stream)
    with (args.data / "mg_lookup.json").open(encoding="utf-8") as stream:
        lookup = json.load(stream)
    required = required_verb_heads(pairs, lookup)
    del pairs, lookup
    inventory = build_inventory(required, {source: resolve_dump(name, args.kaikki)
                                          for source, name in DUMPS["el"].items()})
    output = args.data / "el_verb_heads.json"
    with output.open("w", encoding="utf-8") as stream:
        json.dump(inventory, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Independent finite verb heads: {len(inventory['heads'])}")
    print(f"  -> {output}")


if __name__ == "__main__":
    main()
