#!/usr/bin/env python3
"""List the -ικός adjectives whose lookup paradigm has no feminine.

Writes data/grc_ikos_adjectives.json, the reviewed input of
dilemma/grc_adjective_gaps.py. A lemma is listed when it is an oxytone,
lowercase -ικός head in ag_lookup.json, none of its keys is feminine, and at
least one independent source calls it an adjective:

  * sophocles: Sophocles' entry gives three endings (μοναχικός, ή, όν),
    read from data/sophocles/chunks with build/expand_sophocles.py's parser;
  * lsj9: the LSJ 9 POS table has an ADJ reading for the headword;
  * corpus: the corpus attestation's dominant POS for the lemma is adjective;
  * review: data/grc_ikos_review.json decides it is one, with a reason. The
    review covers the heads no other source settles, and is the only way a
    capitalized head (Πλατωνικός) is listed. Its `exclude` decisions keep a
    lemma out whatever the other sources say.

A lemma whose Sophocles entry gives only two endings is left out, and so is
one with characters other than Greek letters (OCR debris such as *σχολαστικός
or µαντευτικός with a micro sign).

    python build/find_grc_ikos_adjectives.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "build"), str(ROOT)]

from dilemma.grc_adjective_gaps import SUFFIX  # noqa: E402

DATA = ROOT / "data"
OUTPUT = DATA / "grc_ikos_adjectives.json"
REVIEW = "grc_ikos_review.json"
FEMININE = ("ικη", "ικης", "ικην", "ικαι", "ικας")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_greek_word(word: str) -> bool:
    """Only Greek letters and their diacritics (no OCR asterisks, micro signs)."""
    return all(unicodedata.combining(ch) or "GREEK" in unicodedata.name(ch, "")
               for ch in unicodedata.normalize("NFD", word))


def bare(word: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", word)
                   if not unicodedata.combining(ch)).lower()


def sophocles_endings(chunks: Path) -> dict[str, str]:
    import expand_sophocles
    expand_sophocles.SOPH_DIR = chunks
    return {headword: entry["adjective_endings"]
            for headword, entry in expand_sophocles.parse_sophocles_entries().items()
            if entry["adjective_endings"]}


def find(data: Path = DATA) -> dict:
    lookup = json.loads((data / "ag_lookup.json").read_text(encoding="utf-8"))
    keys = defaultdict(set)
    for form, lemma in lookup.items():
        keys[lemma].add(form)
    pos = json.loads((data / "lsj9_pos_lookup.json").read_text(encoding="utf-8"))
    attested = json.loads((data / "lemma_attestation.json").read_text(
        encoding="utf-8"))["lemmas"]
    sophocles = sophocles_endings(data / "sophocles" / "chunks")
    review = {entry["lemma"]: entry["decision"] for entry in json.loads(
        (data / REVIEW).read_text(encoding="utf-8"))["entries"]}

    listed, two_ending = [], 0
    for lemma in sorted(keys):
        if ((lemma != lemma.lower() and review.get(lemma) != "adjective")
                or review.get(lemma) == "exclude" or not lemma.endswith(SUFFIX)
                or not is_greek_word(lemma)
                or lookup.get(lemma) != lemma
                or any(bare(form).endswith(FEMININE) for form in keys[lemma])):
            continue
        if sophocles.get(lemma) == "two":
            two_ending += 1
            continue
        evidence = []
        if sophocles.get(lemma) == "three":
            evidence.append("sophocles")
        if "ADJ" in pos.get(lemma, {}):
            evidence.append("lsj9")
        if attested.get(lemma, {}).get("dominant_pos") in ("adj", "adjective"):
            evidence.append("corpus")
        if review.get(lemma) == "adjective":
            evidence.append("review")
        if evidence:
            listed.append({"lemma": lemma, "evidence": evidence})
    inputs = {name: sha256(data / name) for name in
              ("ag_lookup.json", "lsj9_pos_lookup.json", "lemma_attestation.json",
               REVIEW)}
    inputs["sophocles/chunks"] = hashlib.sha256(b"".join(
        sha256(path).encode() for path in
        sorted((data / "sophocles" / "chunks").glob("*.xml")))).hexdigest()
    print(f"  listed {len(listed):,}; left out {two_ending:,} two-ending entries")
    return {"schema_version": 1, "rule": "grc_ikos_feminine_cells_v1",
            "inputs": inputs, "lemmas": listed}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    record = find(args.data)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
    print(f"  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
