#!/usr/bin/env python3
"""Map regular Ancient Greek adverbs in -ως to the adjective they come from.

Writes data/grc_adverb_heads.json, which dilemma/core.py reads to lemmatize a
regular adverb to its adjective (καλῶς -> καλός, σαφῶς -> σαφής, ταχέως ->
ταχύς) in Ancient Greek, in the tagged and the untagged calls alike. That is
how the annotated corpora (Diorisis, the AGDT, DBBE) and the gate fixtures
lemmatize them; see docs/usage.md, "Adverbs in -ως".

Keys are lowercase polytonic adverbs in -ως/-ῶς, taken from the four Ancient
POS tables (ag_pos, lsj9_pos, treebank_pos, glaux_pos). The adjective is the
first of these cells that passes every check: ag_pos ADJ, lsj9_pos ADJ,
treebank ADV, treebank ADJ, GLAUx ADV, GLAUx ADJ. A cell passes when its value

  * is lowercase, a headword of ag/lsj/lsj10/dge/cunliffe, and an ADJ value in
    one of the POS tables;
  * derives exactly from the adverb's stem s (the unaccented adverb less -ως):
    s+ος, s+ης, s+οος, s+ους; s-ε+υς or s-ε+ης when s ends in ε; s-ον+ων;
    s-εντ+εις;
  * agrees in accent: an oxytone -ός adjective gives -ῶς and any other -ος
    adjective gives -ως; a contracted -οος/-ους adjective gives -ῶς. This keeps
    ὅμως from ὁμός and εὔχρως from εὔχρους.

Adverb lemmas of lookup.db that no POS cell covers are mapped when exactly one
headword passes the same checks among the ADJ values of ag_pos and lsj9_pos and
the -ικός headwords (ὑποτυπωτικῶς -> ὑποτυπωτικός); contracted targets are not
tried there.

A key is left out when it is:

  * itself a lowercase NOUN or ADJ in a POS table (σῶς, ἅλως);
  * shorter than five letters without its accents;
  * monotonic: it ends in -ώς, or starts with a vowel or ρ that has no
    breathing (Modern Greek keeps its adverbs as headwords);
  * accented on the antepenult: an Attic-declension adjective or noun
    (ἔκπλεως, ἀνάπλεως, κατάχρεως), not an adverb;
  * a dental-stem noun: lookup.db gives it a form in -ωτ- that is not a
    comparative or superlative (καταγέλως, ἀνίδρως);
  * kept by data/grc_adverb_review.json, which lists the lexicalized adverbs
    that the references keep as headwords (ἴσως, ἰδίως, ἄλλως, ...), each with a
    reason. The output repeats that list as `keep`.

    python build/build_grc_adverb_heads.py

Run it after build_lookup_db.py and build/build_treebank_pos_lookup.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import unicodedata

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT_NAME = "grc_adverb_heads.json"
REVIEW_NAME = "grc_adverb_review.json"
RULE = "grc_adverb_heads_v1"

# Order is precedence: the first passing cell names the adjective.
POS_TABLES = ("ag_pos_lookup.json", "lsj9_pos_lookup.json",
              "treebank_pos_lookup.json", "glaux_pos_lookup.json")
CELL_ORDER = (("ag_pos_lookup.json", "ADJ"), ("lsj9_pos_lookup.json", "ADJ"),
              ("treebank_pos_lookup.json", "ADV"),
              ("treebank_pos_lookup.json", "ADJ"),
              ("glaux_pos_lookup.json", "ADV"), ("glaux_pos_lookup.json", "ADJ"))
# Fallback adjectives come only from the dictionary-derived tables.
FALLBACK_TABLES = ("ag_pos_lookup.json", "lsj9_pos_lookup.json")
HEADWORD_LISTS = ("ag_headwords.json", "lsj_headwords.json",
                  "lsj10_headwords.json", "dge_headwords.json",
                  "cunliffe_headwords.json")

ADVERB_ENDINGS = ("ως", "ῶς", "ώς")
MIN_BARE_LENGTH = 5
VOWELS = set("αεηιουω")
DIPHTHONGS = {"αι", "ει", "οι", "υι", "αυ", "ευ", "ου", "ηυ", "ωυ"}
ACCENTS = {"́", "̀", "͂"}
BREATHINGS = {"̓", "̔"}
DIAERESIS = "̈"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bare(word: str) -> str:
    """Lowercase, without diacritics."""
    return "".join(ch for ch in unicodedata.normalize("NFD", word.lower())
                   if not unicodedata.combining(ch))


def accented(word: str) -> bool:
    return any(ch in ACCENTS for ch in unicodedata.normalize("NFD", word))


def accent_from_end(word: str) -> int | None:
    """Syllable carrying the accent, counted from the end (0 = ultima)."""
    nuclei: list[str] = []
    accent = None
    previous_vowel = ""
    for ch in unicodedata.normalize("NFD", word.lower()):
        if unicodedata.combining(ch):
            if ch in ACCENTS and nuclei:
                accent = len(nuclei) - 1
            elif ch == DIAERESIS and nuclei and len(nuclei[-1]) == 2:
                # ἀΐδιος: the diaeresis splits the pair back into two vowels.
                nuclei[-1:] = list(nuclei[-1])
            continue
        if ch in VOWELS:
            if previous_vowel and previous_vowel + ch in DIPHTHONGS \
                    and len(nuclei[-1]) == 1:
                nuclei[-1] += ch
            else:
                nuclei.append(ch)
            previous_vowel = ch
        else:
            previous_vowel = ""
    return None if accent is None else len(nuclei) - 1 - accent


def is_monotonic_key(word: str) -> bool:
    """Spelled as Modern Greek: final -ώς, or no breathing on a vowel or ρ."""
    if word.endswith("ώς"):
        return True
    nfd = unicodedata.normalize("NFD", word.lower())
    if not nfd or (nfd[0] not in VOWELS and nfd[0] != "ρ"):
        return False
    index = 1
    if nfd[0] in VOWELS and len(nfd) > 1 and nfd[0] + nfd[1] in DIPHTHONGS:
        index = 2  # a diphthong carries its breathing on the second vowel
    marks = set()
    for ch in nfd[1:index + 3]:
        if unicodedata.combining(ch):
            marks.add(ch)
    return not marks & BREATHINGS


def adjective_shapes(adverb: str, contracted: bool = True) -> set[str]:
    """Unaccented adjectives whose regular adverb is ``adverb``."""
    stem = bare(adverb)[:-2]
    shapes = {stem + "ος", stem + "ης"}
    if contracted:
        shapes |= {stem + "οος", stem + "ους"}
    if stem.endswith("ε"):
        shapes |= {stem[:-1] + "υς", stem[:-1] + "ης"}
    if stem.endswith("ον"):
        shapes.add(stem[:-2] + "ων")
    if stem.endswith("εντ"):
        shapes.add(stem[:-3] + "εις")
    return shapes


def accent_agrees(adverb: str, adjective: str) -> bool:
    """An oxytone -ός adjective gives -ῶς; other -ος adjectives give -ως."""
    shape = bare(adjective)
    if shape.endswith(("οος", "ους")):
        return adverb.endswith("ῶς")
    if shape.endswith("ος"):
        return adverb.endswith("ῶς") == (accent_from_end(adjective) == 0)
    return True


def is_dental_noun(adverb: str, forms) -> bool:
    """A form in -ωτ- that is not -ωτερ-/-ωτατ-: καταγέλως, καταγέλωτος."""
    stem = bare(adverb)[:-1] + "τ"
    for form in forms:
        shape = bare(form)
        if shape.startswith(stem) and not shape.startswith(
                (stem + "ερ", stem + "ατ")):
            return True
    return False


def candidate_key(word: str) -> bool:
    return (word[:1].islower() and word == word.lower()
            and word.endswith(ADVERB_ENDINGS) and accented(word))


def build_heads(tables: dict[str, dict], headwords: set[str],
                adverb_forms: dict[str, list[str]],
                keep: set[str]) -> dict:
    """The adverb -> adjective map, from in-memory inputs.

    ``tables`` maps each name in POS_TABLES to its {form: {UPOS: lemma}}
    table, ``headwords`` is the union of HEADWORD_LISTS, ``adverb_forms`` maps
    each lowercase adverb lemma of lookup.db in -ως to its forms, and ``keep``
    holds the reviewed adverbs that stay their own headword.
    """
    adjectives = set()
    for table in tables.values():
        for cell in table.values():
            if "ADJ" in cell:
                adjectives.add(cell["ADJ"])
    nominal_selves = set()
    keys = set()
    for table in tables.values():
        for key, cell in table.items():
            if not key.endswith(ADVERB_ENDINGS):
                continue
            if candidate_key(key):
                keys.add(key)
            for upos in ("NOUN", "ADJ"):
                value = cell.get(upos)
                if value and value[:1].islower() and bare(value) == bare(key):
                    nominal_selves.update((key, value))

    def eligible(key: str) -> bool:
        return (key not in nominal_selves and key not in keep
                and len(bare(key)) >= MIN_BARE_LENGTH
                and not is_monotonic_key(key)
                and (accent_from_end(key) or 0) < 2
                and not is_dental_noun(key, adverb_forms.get(key, ())))

    heads: dict[str, str] = {}
    evidence: dict[str, int] = {}
    for key in sorted(keys):
        if not eligible(key):
            continue
        shapes = adjective_shapes(key)
        for table_name, upos in CELL_ORDER:
            value = tables[table_name].get(key, {}).get(upos)
            if (value and value[:1].islower() and value in headwords
                    and value in adjectives and bare(value) in shapes
                    and accent_agrees(key, value)):
                heads[key] = value
                label = f"{table_name.removesuffix('_lookup.json')}.{upos}"
                evidence[label] = evidence.get(label, 0) + 1
                break

    fallback_adjectives = {cell["ADJ"] for name in FALLBACK_TABLES
                           for cell in tables[name].values() if "ADJ" in cell}
    fallback_adjectives |= {word for word in headwords if word.endswith("ικός")}
    by_shape: dict[str, set[str]] = {}
    for adjective in fallback_adjectives & headwords:
        if adjective[:1].islower():
            by_shape.setdefault(bare(adjective), set()).add(adjective)
    fallback = 0
    for key in sorted(adverb_forms):
        if key in heads or not candidate_key(key) or not eligible(key):
            continue
        hits = set()
        for shape in adjective_shapes(key, contracted=False):
            hits |= {adj for adj in by_shape.get(shape, ())
                     if accent_agrees(key, adj)}
        if len(hits) == 1:
            heads[key] = hits.pop()
            fallback += 1
    evidence["lookup_headword_fallback"] = fallback
    return {"heads": dict(sorted(heads.items())),
            "evidence": dict(sorted(evidence.items()))}


def read_review(path: Path) -> dict[str, str]:
    review = json.loads(path.read_text(encoding="utf-8"))
    entries = {}
    for entry in review["entries"]:
        if entry.get("decision") != "keep" or not entry.get("reason", "").strip():
            raise ValueError(f"{path.name}: every entry needs decision 'keep' "
                             f"and a reason: {entry!r}")
        entries[unicodedata.normalize("NFC", entry["adverb"])] = entry["reason"]
    return entries


def read_adverb_forms(lookup_db: Path) -> dict[str, list[str]]:
    """Each lowercase -ως lemma of lookup.db with its forms."""
    connection = sqlite3.connect(f"file:{lookup_db}?mode=ro", uri=True)
    try:
        forms: dict[str, list[str]] = {}
        rows = connection.execute(
            "SELECT m.text, l.form FROM lemmas m JOIN lookup l"
            " ON l.lemma_id = m.id"
            " WHERE m.text LIKE '%ως' OR m.text LIKE '%ῶς' OR m.text LIKE '%ώς'")
        for lemma, form in rows:
            if candidate_key(lemma):
                forms.setdefault(lemma, []).append(form)
        return {lemma: sorted(set(values)) for lemma, values in forms.items()}
    finally:
        connection.close()


def build(data: Path = DATA) -> dict:
    review = read_review(data / REVIEW_NAME)
    tables = {name: json.loads((data / name).read_text(encoding="utf-8"))
              for name in POS_TABLES}
    headwords: set[str] = set()
    for name in HEADWORD_LISTS:
        headwords.update(json.loads((data / name).read_text(encoding="utf-8")))
    result = build_heads(tables, headwords, read_adverb_forms(data / "lookup.db"),
                         set(review))
    inputs = {name: sha256(data / name)
              for name in (*POS_TABLES, *HEADWORD_LISTS, "lookup.db", REVIEW_NAME)}
    return {"schema_version": 1, "rule": RULE, "inputs": inputs,
            "evidence": result["evidence"], "keep": sorted(review),
            "heads": result["heads"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=DATA / OUTPUT_NAME)
    args = parser.parse_args(argv)
    record = build(args.data)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=0) + "\n",
                           encoding="utf-8")
    print(f"  {len(record['heads']):,} adverbs, {len(record['keep'])} kept; "
          f"evidence {record['evidence']}")
    print(f"  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
