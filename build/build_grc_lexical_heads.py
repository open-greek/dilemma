#!/usr/bin/env python3
"""Build positive lexical-head evidence for sparse Ancient corpus admission.

Run with ``--kaikki DIR --lsj9-headwords FILE`` after refreshing the source
extracts. Pure inflection/alternative-form entries and generated headword
unions are not independent definitions. The output records only headwords,
portable source identifiers, and hashes, never dictionary definitions.
Modern Greek definitions are stored separately for the existing edited
vernacular fallback; they do not authorize sparse Ancient corpus inflections.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dilemma.form_sanitize import nonlexical_generated_lemma, sanitize_form
from build_data import _is_verb_inflection_gloss, _is_ancient_reference_gloss

KAIKKI_INPUTS = (
    ("wiktionary_en_grc", "en-el/kaikki.org-dictionary-AncientGreek.jsonl", "grc"),
    ("wiktionary_el_grc", "el/kaikki.org-dictionary-AncientGreek.jsonl", "grc"),
    ("wiktionary_el_medieval", "el/kaikki.org-dictionary-MedievalGreek.jsonl", "gkm"),
)
VERNACULAR_INPUTS = (
    ("wiktionary_en_el", "en-el/kaikki.org-dictionary-Greek-words.jsonl", "el"),
    ("wiktionary_el_el", "el/kaikki.org-dictionary-Greek.jsonl", "el"),
)
EDITED_INPUTS = (
    "dge_headwords.json", "cunliffe_headwords.json", "lgpn_names.json",
    "pd_headwords.json", "vlg_headwords.json", "wip_headwords.json",
    "lbg_headwords.json",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest()


def normalize_headword(text: str) -> str | None:
    text = sanitize_form(text).strip()
    # Lexicon syllable boundaries and quantity marks are notation. Retain
    # case, breathing, diaeresis, subscript, and every tonal mark.
    text = unicodedata.normalize("NFC", "".join(
        char for char in unicodedata.normalize("NFD", text)
        if char not in {"-", "\u0304", "\u0306"}
    ))
    if (not text or nonlexical_generated_lemma(text)
            or any(char.isupper() for char in text[1:])):
        return None
    if not all(unicodedata.combining(char)
               or "GREEK" in unicodedata.name(char, "")
               for char in unicodedata.normalize("NFD", text)):
        return None
    return text


def is_reference_gloss(gloss: str, pos: str | None = None) -> bool:
    """Reject grammatical/reference glosses without rejecting lexical 'of'."""
    return (_is_verb_inflection_gloss(gloss)
            or _is_ancient_reference_gloss(gloss, pos) or bool(re.match(
        r"^(?:(?:(?:nominative|accusative|genitive|dative|vocative|singular|plural|dual|"
        r"masculine|feminine|neuter|comparative|superlative|positive)\s+)+of\b|"
        r"(?:(?:alternative|alternate|obsolete|archaic|rare|variant|nonstandard)\s+)*"
        r"(?:form|spelling|misspelling|variant) of\b|"
        r"(?:ονομαστική|γενική|αιτιατική|κλητική|δοτική|αρσενικό|θηλυκό|ουδέτερο)\b|"
        r"(?:εναλλακτική|άλλη|παλαιότερη|πολυτονική) γραφή\b)",
        gloss.strip().lower())))


def has_definition(entry: dict, source: str = "en") -> bool:
    """Require at least one actual lexical sense, not a form-of redirect."""
    if entry.get("pos") in {"prefix", "suffix", "infix", "character", "symbol"}:
        return False
    if source == "en" and (entry.get("form_of") or entry.get("alt_of")):
        return False
    for sense in entry.get("senses", []):
        tags = set(entry.get("tags", [])) | set(sense.get("tags", []))
        if (sense.get("form_of") or sense.get("alt_of")
                or tags & {"form-of", "alt-of", "inflection-of", "alternative", "misspelling", "no-gloss"}):
            continue
        if any(gloss.strip() and not is_reference_gloss(gloss, entry.get("pos"))
               for gloss in sense.get("glosses", [])
               if isinstance(gloss, str)):
            return True
    return False


def build_heads(kaikki: Path, data: Path, lsj9_headwords: Path) -> dict:
    heads: dict[str, set[str]] = defaultdict(set)
    vernacular_heads: dict[str, set[str]] = defaultdict(set)
    inputs, counts = {}, Counter()

    def pin(source: str, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"required lexical source missing: {source}")
        inputs[source] = {"sha256": sha256(path), "size": path.stat().st_size}

    def add(text: str, source: str, target: dict[str, set[str]] = heads) -> None:
        word = normalize_headword(text)
        if word:
            target[word].add(source)
            counts[source] += 1

    for source, relative, language in KAIKKI_INPUTS + VERNACULAR_INPUTS:
        path = kaikki / relative
        pin(source, path)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                entry = json.loads(line)
                if (entry.get("lang_code") == language
                        and has_definition(entry, "en" if source.startswith("wiktionary_en_") else "el")):
                    add(entry["word"], source,
                        vernacular_heads if language == "el" else heads)

    for name in EDITED_INPUTS:
        source = name.removesuffix(".json")
        path = data / name
        pin(source, path)
        for entry in json.loads(path.read_text(encoding="utf-8")):
            add(entry if isinstance(entry, str) else entry["lemma"], source)

    pin("lsj9_headwords", lsj9_headwords)
    for entry in json.loads(lsj9_headwords.read_text(encoding="utf-8")):
        add(entry["headword"], "lsj9_headwords")

    chunks = sorted((data / "sophocles" / "chunks").glob("*.xml"))
    if not chunks:
        raise FileNotFoundError("required lexical source missing: sophocles/chunks")
    for path in chunks:
        source = f"sophocles/{path.name}"
        pin(source, path)
        text = path.read_text(encoding="utf-8")
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            root = ET.fromstring("<wrap>" + text + "</wrap>")
        for entry in root.findall(".//entry"):
            orth = entry.find("orth")
            if orth is not None and orth.text:
                add(orth.text, source)

    return {"schema_version": 2, "builder_sha256": sha256(Path(__file__)),
            "source_sha256": {name: sha256(ROOT / name) for name in
                              ("build_data.py", "dilemma/form_sanitize.py")},
            "inputs": dict(sorted(inputs.items())),
            "source_head_counts": dict(sorted(counts.items())),
            "heads": {word: sorted(heads[word]) for word in sorted(heads)},
            "vernacular_heads": {word: sorted(vernacular_heads[word])
                                 for word in sorted(vernacular_heads)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaikki", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--lsj9-headwords", type=Path,
                        default=Path.home() / "Documents" / "lsj9" / "lsj9_headwords.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data" / "grc_lexical_heads.json")
    args = parser.parse_args()
    result = build_heads(args.kaikki, args.data, args.lsj9_headwords)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Lexical heads: {len(result['heads']):,}; "
          f"vernacular heads: {len(result['vernacular_heads']):,}; "
          f"inputs: {len(result['inputs'])}")
    print(f"Output SHA-256: {sha256(args.output)}")


if __name__ == "__main__":
    main()
