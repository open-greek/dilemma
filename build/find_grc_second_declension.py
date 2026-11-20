#!/usr/bin/env python3
"""List the Ancient -ος words whose second-declension class a lexicon settles.

Writes data/grc_second_declension.json, the input of dilemma/grc_os_gaps.py:
every plain lowercase -ος headword (one word, one accent) with the class
that tells which regular cells it has. The class is read from lexicon
entries, never from the lookup tables, so the list does not change when the
tables do:

  * lsj9: the LSJ 9 exports. The entry's head text opens with the endings
    (``ή, όν`` and ``η, ον`` are three endings with the feminine in -η,
    ``ά, όν``, ``α, ον`` and ``ία, ιον`` three with the feminine in -α,
    ``όν`` and ``ον`` two); the ``grammar`` field gives a noun's article or
    ``ον`` for an adjective whose text begins after the ending; the headword
    POS table says ADJ for sub-entries the headword export leaves out
    (αὐτοσύστατος).
  * lbg: the Byzantine lexicon's headword list. It gives every noun its
    article, so an -ος headword without one is an adjective; its feminine is
    not recorded. An oxytone in -μός without an article is an action noun
    whose article was not captured and is left out.
  * sophocles: the Sophocles lexicon's entry gives the adjective's endings
    (θεοτήρητος, ον; θεαρχικός, ή, όν), read with build/expand_sophocles.py's
    parser after its head rules, which read a capitalized head that the
    other dictionaries list in lowercase as the common word. Its nouns are
    declined by the Sophocles expansion itself.
  * suffix: the feminine of an adjective that a lexicon lists without one,
    where the suffix decides it: oxytone -ικός and -ιακός take -ή (2,999 of
    3,008 such LSJ adjectives), the verbal adjective in -τέος takes -α (all
    181). The suffix alone never lists a word.

Classes: ``noun`` (ὁ or ἡ: the cells are the same), ``adjective`` (two
endings, or the feminine is not settled: masculine and neuter cells only),
``adjective_eta`` and ``adjective_alpha`` (three endings). A word any source
calls a neuter (τό: the s-stem neuters, τεῖχος) is left out, and so is a
stem in -ο (εὔνοος contracts). When the sources give both feminines, or a
noun beside an adjective, the adjective class with the fewer cells is kept.

    python build/find_grc_second_declension.py --lsj9-dir DIR
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "build"), str(ROOT)]

from dilemma.grc_noun_gaps import is_plain_lemma, strip_accents  # noqa: E402
from dilemma.grc_noun_gaps import _needs_breathing_and_lacks_it  # noqa: E402
from dilemma.grc_os_gaps import (  # noqa: E402
    CLASSES, accent_type, has_only_initial_breathing, os_cells, read_classes)

DATA = ROOT / "data"
OUTPUT = DATA / "grc_second_declension.json"
LBG_HEADWORDS = "lbg_headwords.json"
LSJ9_FILES = ("lsj9_headwords.json", "lsj9_glosses.jsonl",
              "lsj9_headword_pos.json")
RULE = "grc_second_declension_v2"

NEUTER = "neuter"
_ARTICLES = {"ὁ": "noun", "ἡ": "noun", "τό": NEUTER}
# Quantity notes that open a head text: "[ῐ], α, ον".
_QUANTITY = re.compile(r"^(?:\[[^\]]{1,14}\][,.]?\s*)+")
_TOKENS = re.compile(r"^([^\s,;:()\[\]]{1,5}),\s*([^\s,;:()\[\]]{1,5})(?=[\s,;:(.]|$)")
_FIRST = re.compile(r"^([^\s,;:()\[\]]{1,4})(?=[\s,;:(.]|$)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_length_marks(word: str) -> str:
    return unicodedata.normalize("NFC", "".join(
        ch for ch in unicodedata.normalize("NFD", word)
        if ch not in "̄̆"))


def bare(word: str) -> str:
    return strip_accents(word).lower()


def is_greek_word(word: str) -> bool:
    """Greek letters and their diacritics only (no micro sign, no Latin)."""
    return all(unicodedata.combining(ch) or "GREEK" in unicodedata.name(ch, "")
               for ch in unicodedata.normalize("NFD", word))


def is_candidate(lemma: str) -> bool:
    """A plain lowercase Greek -ος word that ``os_cells`` can decline, with
    the breathing its first letter needs and no other."""
    return (bare(lemma).endswith("ος") and is_plain_lemma(lemma)
            and is_greek_word(lemma) and accent_type(lemma) is not None
            and not _needs_breathing_and_lacks_it(lemma)
            and has_only_initial_breathing(lemma)
            and bool(os_cells(lemma, "noun")))


def head_text_class(lemma: str, text: str) -> str | None:
    """The class the opening of an LSJ head text gives, or None.

    ``ή, όν`` / ``η, ον`` -> adjective_eta; ``ά, όν`` / ``α, ον`` and the
    spelled-out ``ία, ιον`` / ``έα, έον`` -> adjective_alpha; ``όν`` / ``ον``
    -> adjective; ``ὁ`` / ``ἡ`` -> noun; ``τό`` and ``εος, τό`` -> neuter.
    """
    text = _QUANTITY.sub("", strip_length_marks(text).lstrip())
    pair = _TOKENS.match(text)
    if pair:
        first, second = bare(pair.group(1)), bare(pair.group(2))
        if second.endswith("ον") and first[-1:] in ("η", "α"):
            shared = first[:-1]
            if (second[:-2] == shared and len(shared) <= 2
                    and bare(lemma).endswith(shared + "ος")):
                return "adjective_eta" if first[-1] == "η" else "adjective_alpha"
        if first == "εος" and second == "το":
            return NEUTER
    single = _FIRST.match(text)
    if single:
        token = single.group(1)
        if bare(token) == "ον":
            return "adjective"
        if token in _ARTICLES:
            return _ARTICLES[token]
    return None


def lsj9_classes(lsj9_dir: Path) -> dict[str, set[str]]:
    """lemma -> the classes its LSJ 9 entries give (homographs pooled)."""
    classes: dict[str, set[str]] = {}
    grammar: dict[str, set[str]] = {}
    for entry in json.loads((lsj9_dir / "lsj9_headwords.json").read_text(
            encoding="utf-8")):
        lemma = strip_length_marks(entry["headword"])
        if is_candidate(lemma):
            grammar.setdefault(lemma, set()).add(entry.get("grammar") or "")
    texts: dict[str, list[str]] = {}
    with (lsj9_dir / "lsj9_glosses.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            if "level" in entry or "number" in entry or not entry.get("headword"):
                continue
            lemma = strip_length_marks(entry["headword"])
            if is_candidate(lemma):
                texts.setdefault(lemma, []).append(entry.get("text", ""))
    for lemma in sorted(set(grammar) | set(texts)):
        found = {head_text_class(lemma, text) for text in texts.get(lemma, ())}
        found.discard(None)
        adjective = any(cls.startswith("adjective") for cls in found)
        for value in grammar.get(lemma, ()):
            if value in _ARTICLES:
                # An article beside the endings is a substantive use of the
                # adjective (ἀπαιτητικός, ή, όν: -κόν, τό).
                if not adjective:
                    found.add(_ARTICLES[value])
            elif value == "ον" and not adjective:
                found.add("adjective")
        if found:
            classes[lemma] = found
    pos = json.loads((lsj9_dir / "lsj9_headword_pos.json").read_text(
        encoding="utf-8"))
    for headword, upos in pos.items():
        lemma = strip_length_marks(headword).replace("-", "")
        if upos == "ADJ" and lemma not in classes and is_candidate(lemma):
            classes[lemma] = {"adjective"}
    return classes


def sophocles_classes(entries) -> dict[str, set[str]]:
    """lemma -> the adjective class Sophocles' entry gives
    (``adjective_class`` of build/expand_sophocles.py)."""
    classes: dict[str, set[str]] = {}
    for lemma, entry in entries.items():
        lemma = unicodedata.normalize("NFC", lemma)
        cls = entry.get("adjective_class")
        if cls in CLASSES and is_candidate(lemma):
            classes.setdefault(lemma, set()).add(cls)
    return classes


def lbg_classes(path: Path) -> dict[str, set[str]]:
    """lemma -> the classes the Byzantine headword list gives."""
    classes: dict[str, set[str]] = {}
    for entry in json.loads(path.read_text(encoding="utf-8")):
        lemma = unicodedata.normalize("NFC", entry.get("lemma") or "")
        if not is_candidate(lemma):
            continue
        gender = entry.get("gender")
        if gender in _ARTICLES:
            cls = _ARTICLES[gender]
        elif gender is None and not lemma.endswith("μός"):
            cls = "adjective"
        else:
            continue
        classes.setdefault(lemma, set()).add(cls)
    return classes


_VOWELS = frozenset("αεηιουω")


def suffix_feminine(lemma: str) -> str | None:
    """The three-ending class an adjective's suffix settles, or None.

    The suffix needs a stem with a vowel before it, and the ι of -ικός and
    -ιακός must be a syllable of its own: after a consonant (τακτικός) or
    with the diaeresis (ἀρχαϊκός). δικός and ἀεικός do not have the suffix.
    """
    for suffix, cls in (("ικός", "adjective_eta"), ("ιακός", "adjective_eta"),
                        ("τέος", "adjective_alpha")):
        if not lemma.endswith(suffix):
            continue
        stem = bare(lemma[:-len(suffix)])
        if not _VOWELS & set(stem):
            continue
        if suffix == "τέος" or stem[-1] not in _VOWELS:
            return cls
    if lemma.endswith("ϊκός") and _VOWELS & set(bare(lemma[:-4])):
        return "adjective_eta"
    return None


def settle(found: set[str], lemma: str) -> tuple[str | None, bool]:
    """(class, whether the suffix supplied the feminine) for pooled evidence."""
    if NEUTER in found:
        return None, False
    three = {cls for cls in found if cls in ("adjective_eta", "adjective_alpha")}
    if len(three) == 1:
        return three.pop(), False
    if three or "adjective" in found:
        if not three:
            by_suffix = suffix_feminine(lemma)
            if by_suffix:
                return by_suffix, True
        return "adjective", False
    return ("noun", False) if "noun" in found else (None, False)


def find(lsj9_dir: Path, data: Path = DATA, sophocles=None) -> dict:
    """The record; ``sophocles`` is the parsed Sophocles entries
    (``sophocles_entries``), or None to leave that lexicon out."""
    sources = {"lsj9": lsj9_classes(lsj9_dir),
               "lbg": lbg_classes(data / LBG_HEADWORDS)}
    if sophocles is not None:
        sources["sophocles"] = sophocles_classes(sophocles)
    listed: dict[str, dict[str, list[str]]] = {cls: {} for cls in CLASSES}
    counts: Counter = Counter()
    for lemma in sorted(set().union(*sources.values())):
        found = set().union(*(sources[name].get(lemma, ()) for name in sources))
        cls, by_suffix = settle(found, lemma)
        if cls is None:
            counts["left out: a source calls it a neuter"
                   if NEUTER in found else "left out"] += 1
            continue
        evidence = [name for name in sources if lemma in sources[name]]
        if by_suffix:
            evidence.append("suffix")
        key = "+".join(evidence)
        listed[cls].setdefault(key, []).append(lemma)
        counts[cls] += 1
    inputs = {name: sha256(lsj9_dir / name) for name in LSJ9_FILES}
    inputs[LBG_HEADWORDS] = sha256(data / LBG_HEADWORDS)
    if sophocles is not None:
        from find_grc_es_adjectives import sophocles_inputs
        inputs.update(sophocles_inputs(data))
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count:,}")
    return {"schema_version": 1, "rule": RULE, "inputs": inputs,
            "counts": {cls: counts[cls] for cls in CLASSES},
            "classes": listed}


def load_classes(path: Path = OUTPUT) -> dict[str, str]:
    """lemma -> class from the written list."""
    return read_classes(json.loads(path.read_text(encoding="utf-8")))


def main(argv=None) -> int:
    import expand_lsj
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lsj9-dir", type=Path, default=None,
                        help="LSJ9 export directory (default: $DILEMMA_LSJ9_DIR, "
                             "else ~/Documents/lsj9)")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    expand_lsj.configure_inputs(lsj9_dir=args.lsj9_dir)
    # The class of an LSJ word is read from the pinned glosses export.
    expand_lsj.verify_lsj_glosses()
    from find_grc_es_adjectives import sophocles_entries
    record = find(expand_lsj.LSJ9_DIR, args.data, sophocles_entries())
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=1)
        stream.write("\n")
    print(f"  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
