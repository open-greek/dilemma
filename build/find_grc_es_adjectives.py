#!/usr/bin/env python3
"""List the Ancient s-stem adjectives in -ής and -ης that a lexicon classes.

Writes data/grc_es_adjectives.json, an input of dilemma/grc_adjective_gaps.py
(their contracted cells) and of build/build_grc_degree_forms.py (-έστερος,
-έστατος). The class is read from lexicon entries, never from the lookup
tables:

  * lsj9: the LSJ 9 exports. The ``grammar`` field or the opening of the
    entry's head text gives the neuter ``ές`` or ``ες`` (ἀληθής, ές;
    συνήθης, ες, gen. εος); the headword POS table says ADJ for a head the
    other fields leave open.
  * sophocles: the Sophocles entry gives the neuter (βληχηματώδης, ες), read
    with build/expand_sophocles.py's parser after its head rules.
  * lbg: the Byzantine lexicon's headword list gives every noun its article,
    so a head in -ής or -ης without one is an adjective. A head in -της
    without one is left to the other sources: the masculine nouns of the
    first declension in -ίτης and -ώτης are the larger part of them.

A head that any source gives an article (ὁ or ἡ) is left out, and so is one
that is no plain lowercase word with one accent, the breathing its first
letter needs and no other, or whose accent is not on the ending (oxytone) or
on the syllable before it (barytone).

    python build/find_grc_es_adjectives.py --lsj9-dir DIR
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

from dilemma.grc_adjective_gaps import es_cells  # noqa: E402
from dilemma.grc_noun_gaps import (  # noqa: E402
    _needs_breathing_and_lacks_it, is_plain_lemma, strip_accents)
from dilemma.grc_os_gaps import has_only_initial_breathing  # noqa: E402
from find_grc_second_declension import (  # noqa: E402
    LBG_HEADWORDS, LSJ9_FILES, _QUANTITY, is_greek_word, sha256,
    strip_length_marks)

DATA = ROOT / "data"
OUTPUT = DATA / "grc_es_adjectives.json"
RULE = "grc_es_adjectives_v1"
ADJECTIVE, NOUN = "es", "noun"
_ARTICLES = {"ὁ": NOUN, "ἡ": NOUN, "τό": NOUN}
_FIRST = re.compile(r"^([^\s,;:()\[\]]{1,4})(?=[\s,;:(.]|$)")


def is_candidate(lemma: str) -> bool:
    """A plain lowercase Greek word in -ής or -ης whose contracted cells
    ``es_cells`` can give."""
    return (strip_accents(lemma).endswith("ης") and is_plain_lemma(lemma)
            and is_greek_word(lemma)
            and not _needs_breathing_and_lacks_it(lemma)
            and has_only_initial_breathing(lemma)
            and len(strip_accents(lemma)) >= 5
            and bool(es_cells(lemma)))


def head_text_class(text: str) -> str | None:
    """``es`` when an LSJ head text opens with the neuter ending, ``noun``
    when it opens with an article, else None."""
    text = _QUANTITY.sub("", strip_length_marks(text).lstrip())
    first = _FIRST.match(text)
    if not first:
        return None
    token = first.group(1)
    if strip_accents(token) == "ες":
        return ADJECTIVE
    return _ARTICLES.get(token)


def lsj9_classes(lsj9_dir: Path) -> dict[str, set[str]]:
    classes: dict[str, set[str]] = {}
    for entry in json.loads((lsj9_dir / "lsj9_headwords.json").read_text(
            encoding="utf-8")):
        lemma = strip_length_marks(entry["headword"])
        if not is_candidate(lemma):
            continue
        grammar = entry.get("grammar") or ""
        if strip_accents(grammar) == "ες":
            classes.setdefault(lemma, set()).add(ADJECTIVE)
        elif grammar in _ARTICLES:
            classes.setdefault(lemma, set()).add(NOUN)
    with (lsj9_dir / "lsj9_glosses.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            if "level" in entry or "number" in entry or not entry.get("headword"):
                continue
            lemma = strip_length_marks(entry["headword"])
            if is_candidate(lemma):
                found = head_text_class(entry.get("text", ""))
                if found:
                    classes.setdefault(lemma, set()).add(found)
    pos = json.loads((lsj9_dir / "lsj9_headword_pos.json").read_text(
        encoding="utf-8"))
    for headword, upos in pos.items():
        lemma = strip_length_marks(headword).replace("-", "")
        if upos == "ADJ" and lemma not in classes and is_candidate(lemma):
            classes[lemma] = {ADJECTIVE}
    return classes


def lbg_classes(path: Path) -> dict[str, set[str]]:
    classes: dict[str, set[str]] = {}
    for entry in json.loads(path.read_text(encoding="utf-8")):
        lemma = unicodedata.normalize("NFC", entry.get("lemma") or "")
        if not is_candidate(lemma):
            continue
        gender = entry.get("gender")
        if gender in _ARTICLES:
            classes.setdefault(lemma, set()).add(NOUN)
        elif gender is None and not strip_accents(lemma).endswith("της"):
            classes.setdefault(lemma, set()).add(ADJECTIVE)
    return classes


def sophocles_classes(entries) -> dict[str, set[str]]:
    """lemma -> classes from parsed Sophocles entries (``adjective_class``
    ``es``, or a stated article)."""
    classes: dict[str, set[str]] = {}
    for lemma, entry in entries.items():
        lemma = unicodedata.normalize("NFC", lemma)
        if not is_candidate(lemma):
            continue
        if entry.get("adjective_class") == "es":
            classes.setdefault(lemma, set()).add(ADJECTIVE)
        elif entry.get("gender_stated"):
            classes.setdefault(lemma, set()).add(NOUN)
    return classes


def sophocles_entries():
    """The Sophocles entries after the head rules, keyed by head."""
    import expand_sophocles
    audit = expand_sophocles.new_head_audit()
    return expand_sophocles.settle_heads(
        expand_sophocles.parse_sophocles_entries(audit=audit),
        expand_sophocles.load_lexical_heads(), audit)


def sophocles_inputs(data: Path) -> dict[str, str]:
    chunks = data / "sophocles" / "chunks"
    return {
        "sophocles/chunks": hashlib.sha256(b"".join(
            sha256(path).encode() for path in
            sorted(chunks.glob("*.xml")))).hexdigest(),
        "sophocles_head_corrections.json":
            sha256(data / "sophocles_head_corrections.json"),
    }


def find(lsj9_dir: Path, data: Path = DATA, sophocles=None) -> dict:
    sources = {"lsj9": lsj9_classes(lsj9_dir),
               "lbg": lbg_classes(data / LBG_HEADWORDS)}
    if sophocles is not None:
        sources["sophocles"] = sophocles_classes(sophocles)
    listed: dict[str, list[str]] = {}
    counts: Counter = Counter()
    for lemma in sorted(set().union(*sources.values())):
        found = set().union(*(sources[name].get(lemma, ()) for name in sources))
        if found != {ADJECTIVE}:
            counts["left out: a source gives an article"] += 1
            continue
        evidence = "+".join(name for name in sources
                            if lemma in sources[name])
        listed.setdefault(evidence, []).append(lemma)
        counts["oxytone" if lemma.endswith("ής") else "barytone"] += 1
    inputs = {name: sha256(lsj9_dir / name) for name in LSJ9_FILES}
    inputs[LBG_HEADWORDS] = sha256(data / LBG_HEADWORDS)
    if sophocles is not None:
        inputs.update(sophocles_inputs(data))
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count:,}")
    return {"schema_version": 1, "rule": RULE, "inputs": inputs,
            "counts": {"oxytone": counts["oxytone"],
                       "barytone": counts["barytone"]},
            "lemmas": dict(sorted(listed.items()))}


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
    expand_lsj.verify_lsj_glosses()
    record = find(expand_lsj.LSJ9_DIR, args.data, sophocles_entries())
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=1)
        stream.write("\n")
    print(f"  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
