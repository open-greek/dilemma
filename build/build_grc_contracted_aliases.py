#!/usr/bin/env python3
"""Keep explicit contracted-verb references separate from lexical definitions."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build.build_grc_lexical_heads import normalize_headword

SOURCES = {
    "wiktionary_en_grc": "en-el/kaikki.org-dictionary-AncientGreek.jsonl",
    "wiktionary_el_grc": "el/kaikki.org-dictionary-AncientGreek.jsonl",
}


def pin(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size": path.stat().st_size, "sha256": digest.hexdigest()}


def contracted_verb_targets(entry):
    if entry.get("pos") != "verb":
        return set()
    targets = set()
    for sense in entry.get("senses", []):
        if "contracted" in set(entry.get("tags", [])) | set(sense.get("tags", [])):
            targets.update(normalize_headword(row.get("word", ""))
                           for row in sense.get("form_of", []))
        for gloss in sense.get("glosses", []):
            match = re.match(r"^(?:συνηρημένη μορφή του|contracted form of)\s+([^\s(,;:]+)",
                             gloss.strip(), re.I)
            if match:
                targets.add(normalize_headword(match[1]))
    return targets - {None}


def build_aliases(kaikki, lexical_path):
    lexical = json.loads(lexical_path.read_bytes())
    aliases = defaultdict(lambda: defaultdict(set))
    inputs = {}
    for source, relative in SOURCES.items():
        path = kaikki / relative
        inputs[source] = pin(path)
        assert inputs[source] == lexical["inputs"][source], source
        for line in path.open(encoding="utf-8"):
            entry = json.loads(line)
            if entry.get("lang_code") != "grc":
                continue
            alias = normalize_headword(entry.get("word", ""))
            if alias:
                for target in contracted_verb_targets(entry):
                    if target in lexical["heads"]:
                        aliases[alias][target].add(source)
    return {"schema_version": 1, "language": "grc",
            "builder_sha256": pin(Path(__file__))["sha256"],
            "lexical_heads": pin(lexical_path), "inputs": inputs,
            "aliases": {alias: [{"lemma": target, "sources": sorted(sources)}
                                for target, sources in sorted(targets.items())]
                        for alias, targets in sorted(aliases.items())}}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kaikki", type=Path, required=True)
    p.add_argument("--lexical-heads", type=Path, default=ROOT / "data/grc_lexical_heads.json")
    p.add_argument("--output", type=Path, default=ROOT / "data/grc_contracted_verb_aliases.json")
    a = p.parse_args()
    result = build_aliases(a.kaikki, a.lexical_heads)
    a.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")) + "\n")
    print(f"Contracted references: {len(result['aliases']):,}")
