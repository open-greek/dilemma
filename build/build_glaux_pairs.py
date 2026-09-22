#!/usr/bin/env python3
"""Extract form->lemma pairs with morphological tags from GLAUx corpus.

GLAUx (Keersmaekers, 2021) is a 17M-token automatically annotated corpus
of Ancient Greek (8th c. BC - 4th c. AD). Each token has AGDT-style
9-position morphological tags.

Extracts training pairs for Dilemma's multi-task learning heads:
- POS tag (position 1)
- Nominal group: Gender + Number + Case (positions 7, 3, 8)
- Verbal group: Tense + Mood + Voice (positions 4, 5, 6)

Output: data/glaux_pairs.json in Dilemma's training pair format.

Usage:
    python build_glaux_pairs.py                    # extract all
    python build_glaux_pairs.py --stats            # show stats only
    python build_glaux_pairs.py --glaux ~/path     # custom GLAUx path
"""

import argparse
import csv
import json
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_GLAUX = Path.home() / "Documents" / "glaux" / "xml"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dilemma.form_sanitize import has_editorial_sigla  # noqa: E402

# What GLAUx puts in the lemma field of an editorial gap, where the manuscript
# is damaged or unreadable. The surviving letters stay in the form, so the
# token still reads as Greek.
GAP_LEMMA = "G"

# Lemma values that are annotation placeholders rather than lemmas: the gap
# marker above, the CoNLL-U empty field, an ellipsis, and the empty string.
PLACEHOLDER_LEMMAS = frozenset({GAP_LEMMA, "_", "...", ""})

# AGDT position 1 -> Wiktionary POS
AGDT_POS = {
    "n": "noun", "v": "verb", "a": "adj", "p": "pron",
    "d": "adv", "c": "conj", "r": "prep", "l": "article",
    "g": "intj", "m": "num",
}

# AGDT single-char codes -> Dilemma tag labels
AGDT_NUMBER = {"s": "singular", "p": "plural", "d": "dual"}
AGDT_GENDER = {"m": "masculine", "f": "feminine", "n": "neuter"}
AGDT_CASE = {"n": "nominative", "g": "genitive", "d": "dative",
             "a": "accusative", "v": "vocative"}
AGDT_TENSE = {"p": "present", "i": "imperfect", "a": "aorist",
              "f": "future", "r": "perfect", "l": "pluperfect",
              "t": "future-perfect"}
AGDT_MOOD = {"i": "indicative", "s": "subjunctive", "o": "optative",
             "m": "imperative", "n": "infinitive", "p": "participle"}
AGDT_VOICE = {"a": "active", "m": "middle", "p": "passive"}
AGDT_PERSON = {"1": "first-person", "2": "second-person", "3": "third-person"}


def nfc(s):
    return unicodedata.normalize("NFC", s)


def is_greek(s):
    return any('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF' for c in s)


def parse_postag(postag):
    """Parse 9-position AGDT morphological tag into Dilemma tag list.

    Returns (pos, tags) where pos is a Wiktionary POS string and
    tags is a list of morphological feature labels.
    """
    if not postag or len(postag) < 9:
        return None, []

    pos_code = postag[0]
    pos = AGDT_POS.get(pos_code)
    if not pos:
        return None, []

    tags = []

    # Person (position 2)
    person = AGDT_PERSON.get(postag[1])
    if person:
        tags.append(person)

    # Number (position 3)
    number = AGDT_NUMBER.get(postag[2])
    if number:
        tags.append(number)

    # Tense (position 4)
    tense = AGDT_TENSE.get(postag[3])
    if tense:
        tags.append(tense)

    # Mood (position 5)
    mood = AGDT_MOOD.get(postag[4])
    if mood:
        tags.append(mood)

    # Voice (position 6)
    voice = AGDT_VOICE.get(postag[5])
    if voice:
        tags.append(voice)

    # Gender (position 7)
    gender = AGDT_GENDER.get(postag[6])
    if gender:
        tags.append(gender)

    # Case (position 8)
    case = AGDT_CASE.get(postag[7])
    if case:
        tags.append(case)

    return pos, tags


# GLAUx's DIALECT column against the dialect tag the paradigm builders read.
# Attic, Attic/Koine and Koine stay in the default (Attic) slice: Koine is the
# continuation of Attic and textbook paradigms include its forms. The marked
# dialects are the ones whose forms must not fill an Attic paradigm slot
# (Doric δωσῶ, Ionic τιθεῖς, Epic λῦσε).
GLAUX_DIALECT_TAGS = {
    "Ionic/Epic": "Epic",
    "Ionic": "Ionic",
    "Doric": "Doric",
    "Aeolic": "Aeolic",
    "Attic": "Attic",
    "Attic/Koine": "Attic",
    "Koine": "Attic",
}


def glaux_dialects(metadata_path) -> dict:
    """Map GLAUx file stem to a dialect tag, '' for the Attic default."""
    path = Path(metadata_path)
    if not path.exists():
        return {}
    out = {}
    text = path.read_text(encoding="utf-8")
    for row in csv.DictReader(text.splitlines(), delimiter="\t"):
        stem = (row.get("TLG") or "").strip()
        if stem:
            out[stem] = GLAUX_DIALECT_TAGS.get(
                (row.get("DIALECT") or "").strip(), ""
            )
    return out


def extract_glaux(glaux_dir, stats_only=False, metadata_path=None):
    """Extract form->lemma pairs from GLAUx XML files.

    NonCommercial GLAUx source texts are always dropped (the build is openly
    licensed by default); see build/nc_filter.py.
    """
    xml_files = sorted(Path(glaux_dir).glob("*.xml"))
    if not xml_files:
        print(f"No XML files found in {glaux_dir}")
        return []

    from nc_filter import excluded_glaux_stems, gorman_glaux_stems
    meta = metadata_path or (Path(glaux_dir).parent / "metadata.txt")
    nc = excluded_glaux_stems(meta)
    gorman = gorman_glaux_stems(meta)
    dialects = glaux_dialects(meta)
    before = len(xml_files)
    xml_files = [x for x in xml_files if x.stem not in nc]
    print(f"Excluded {before - len(xml_files)} "
          f"GLAUx text(s) (NonCommercial or PROIEL-derived)")
    print(f"Skipping manual (Gorman-derived) sentences in "
          f"{sum(1 for x in xml_files if x.stem in gorman)} text(s)")

    print(f"Processing {len(xml_files)} GLAUx files...")

    pairs = []
    seen = set()  # (form, lemma) dedup
    entry_by_key = {}
    dialect_votes = defaultdict(set)
    total_tokens = 0
    skipped_punct = 0
    skipped_no_lemma = 0
    skipped_gap = 0
    skipped_editorial = 0
    skipped_non_greek = 0
    skipped_dup = 0

    pos_counts = Counter()
    has_nominal = 0
    has_verbal = 0
    tense_counts = Counter()

    for i, xml_file in enumerate(xml_files):
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError:
            continue

        # Gorman-derived works: the manual sentences ARE Gorman's trees
        # (held-out gold, never ingested); only the auto sentences pass.
        skip_manual = xml_file.stem in gorman
        file_dialect = dialects.get(xml_file.stem, "")
        for sentence in tree.findall(".//sentence"):
            if skip_manual and sentence.get("analysis") == "manual":
                continue
            words = sentence.findall(".//word")
            if not words:
                continue
            for word in words:
                form = word.get("form", "")
                lemma = word.get("lemma", "")
                postag = word.get("postag", "")

                total_tokens += 1

                # Skip punctuation
                if postag and postag[0] == "u":
                    skipped_punct += 1
                    continue

                # Skip the gaps the treebank marks itself. Where a
                # manuscript is damaged or unreadable the editor records a
                # lacuna, and the annotation says so three ways at once:
                # relation="GAP", postag "z" (unanalyzable), and the lemma
                # field set to the gap marker. The surviving letters are
                # still carried in `form`, so "Gdou" below is what is left
                # of an occurrence of the genitive of Hades after its first
                # two letters were lost.
                #
                # The `is_greek` test further down does not catch these,
                # because it asks whether ANY character is Greek and the
                # surviving letters are. Reading the annotation is both
                # exact and cheaper than guessing from the characters.
                # The lemma field is checked as well as the relation and the
                # postag, because an annotator sometimes assigns a part of
                # speech to a token whose text is mostly gap: `τGῆς` is tagged
                # a feminine genitive numeral and `G?ἔπλει` a verb. Both still
                # carry the gap marker as their lemma, and all 113 such rows
                # in GLAUx do, so the lemma is the reliable signal. Real Greek
                # proper nouns transliterated from Latin keep Latin homoglyphs
                # in their own lemmas (`Σιλουίαν` under `σιλvια`, `Οὐολουμνίαν`
                # under `Vολυμνια`), so a blanket Latin-letter test would throw
                # those away; this one does not touch them.
                if (postag[:1] == "z" or word.get("relation") == "GAP"
                        or lemma in PLACEHOLDER_LEMMAS):
                    skipped_gap += 1
                    continue

                # "_" is the CoNLL-U marker for a field left empty. It is a
                # plain string, so a bare truthiness test lets it through:
                # GLAUx carries one such row, for διατείνας, and Diorisis has
                # the same form under its real lemma διατείνω.
                if not form or lemma in PLACEHOLDER_LEMMAS:
                    skipped_no_lemma += 1
                    continue

                # NFC normalize (GLAUx uses NFD)
                form = nfc(form)
                lemma = nfc(lemma)

                # Leiden brackets and parentheses describe an editor's
                # reconstruction; they are not part of the Greek spelling.
                # Reject both fields here so every consumer of the shared
                # GLAUx pair artifact inherits the same hygiene policy.
                if has_editorial_sigla(form) or has_editorial_sigla(lemma):
                    skipped_editorial += 1
                    continue

                if not is_greek(form):
                    skipped_non_greek += 1
                    continue

                # Parse morphological tag
                pos, tags = parse_postag(postag)

                # Dedup. A (form, lemma) pair is written once, but every
                # work that attests it votes on its dialect below.
                key = (form, lemma)
                dialect_votes[key].add(file_dialect)
                if key in seen:
                    skipped_dup += 1
                    continue
                seen.add(key)

                entry = {"form": form, "lemma": lemma}
                if pos:
                    entry["pos"] = pos
                    pos_counts[pos] += 1
                if tags:
                    entry["tags"] = tags

                    # Count morphological coverage
                    tag_set = set(tags)
                    if tag_set & {"masculine", "feminine", "neuter"}:
                        has_nominal += 1
                    if tag_set & {"present", "imperfect", "aorist", "future",
                                  "perfect", "pluperfect"}:
                        has_verbal += 1
                        for t in tag_set & {"present", "imperfect", "aorist",
                                            "future", "perfect", "pluperfect"}:
                            tense_counts[t] += 1

                pairs.append(entry)
                entry_by_key[key] = entry

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(xml_files)} files, {len(pairs):,} pairs", flush=True)

    # A form keeps the Attic default if any work attesting it is explicitly
    # Attic, Attic/Koine or Koine, or if no work attesting it records a
    # dialect at all. A form attested only in dialect works carries that
    # dialect, so the paradigm builders route it to that dialect's slice.
    dialect_counts = Counter()
    for key, votes in dialect_votes.items():
        entry = entry_by_key.get(key)
        if entry is None or "Attic" in votes or votes == {""}:
            continue
        marked = [d for d in ("Epic", "Ionic", "Doric", "Aeolic")
                  if d in votes]
        if not marked:
            continue
        entry.setdefault("tags", []).append(marked[0])
        dialect_counts[marked[0]] += 1
    if dialect_counts:
        print("\nDialect-tagged pairs (attested only in that dialect): "
              + ", ".join(f"{d} {n:,}" for d, n in dialect_counts.most_common()))

    print(f"\nTotal tokens: {total_tokens:,}")
    print(f"Skipped: {skipped_punct:,} punct, {skipped_gap:,} editorial gaps, "
          f"{skipped_editorial:,} editorial sigla, "
          f"{skipped_no_lemma:,} no lemma, "
          f"{skipped_non_greek:,} non-Greek, {skipped_dup:,} duplicates")
    print(f"Unique pairs: {len(pairs):,}")

    print(f"\nPOS distribution:")
    for pos, count in pos_counts.most_common():
        print(f"  {pos:10s}: {count:,}")

    print(f"\nMorphological coverage:")
    print(f"  With nominal features (G+N+C): {has_nominal:,}")
    print(f"  With verbal features (T+M+V):  {has_verbal:,}")

    print(f"\nTense distribution (verbs):")
    total_tense = sum(tense_counts.values())
    for t, c in tense_counts.most_common():
        print(f"  {t:15s}: {c:,} ({100*c/total_tense:.1f}%)")

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Extract GLAUx training pairs")
    parser.add_argument("--glaux", type=str, default=str(DEFAULT_GLAUX),
                        help="Path to GLAUx xml/ directory")
    parser.add_argument("--stats", action="store_true",
                        help="Show stats only, don't save")
    parser.add_argument("--output", type=str,
                        default=str(DATA_DIR / "glaux_pairs.json"),
                        help="Output path (default: glaux_pairs.json)")
    args = parser.parse_args()

    # Always drop the NonCommercial GLAUx source texts: glaux_pairs.json is a
    # openly licensed artifact (see build/nc_filter.py and NOTICE).
    pairs = extract_glaux(args.glaux, stats_only=args.stats)

    if not args.stats and pairs:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False)
        size_mb = out.stat().st_size / (1024 * 1024)
        print(f"\nSaved {len(pairs):,} pairs to {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
