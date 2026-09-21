#!/usr/bin/env python3
"""Expand Sophocles lexicon headwords into inflected forms.

Extracts headwords from Sophocles TEI XML chunks, infers declension
from inline grammatical info and ending patterns, then expands via
grc-decl/grc-conj templates.

Usage:
    python expand_sophocles.py --analyze       # show stats, don't modify lookup
    python expand_sophocles.py --expand        # expand nouns and merge into ag_lookup.json
    python expand_sophocles.py --expand-verbs  # expand verbs and merge into ag_lookup.json
"""

import argparse
import json
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

# Import shared infrastructure from expand_lsj
from expand_lsj import (
    get_wtp, expand_noun, expand_verb, strip_length_marks, strip_diacritics,
    mark_alpha_length, AG_LOOKUP, DATA_DIR, ARTICLES,
)
from dilemma.form_sanitize import (  # noqa: E402
    has_editorial_sigla,
    resolve_editorial_form,
)

SOPH_DIR = DATA_DIR / "sophocles" / "chunks"

# Article -> gender
GENDER_MAP = {"ὁ": "m", "ἡ": "f", "τό": "n", "τὸ": "n", "τά": "n", "οἱ": "m"}

# Genitive ending patterns for inferring declension from inline text
# Matches things like "ου, ὁ" or "ατος, τό" or "εως, ἡ"
INLINE_GEN_RE = re.compile(
    r'^[,\s]*'
    r'([α-ωά-ώἀ-ᾧΑ-ΩῬ]+)'  # genitive ending
    r'[,\s]+'
    r'(ὁ|ἡ|τό|τὸ|τά|οἱ)'     # article = gender
)

# For entries without inline genitive, infer from ending
ENDING_RULES = [
    # (nom_ending, gender, gen_ending) - high confidence patterns only
    # 2nd declension
    ("ος", "m", "ου"),
    ("ος", "f", "ου"),
    ("ον", "n", "ου"),
    # 1st declension
    ("η", "f", "ης"),
    ("ής", "m", "οῦ"),
    # -μα neuters (very reliable)
    ("μα", "n", "ματος"),
    # -ις feminines (assume -εως, most common)
    ("σις", "f", "σεως"),
    ("ξις", "f", "ξεως"),
    ("ψις", "f", "ψεως"),
    # -εύς masculines
    ("εύς", "m", "έως"),
    # -τής agent nouns
    ("τής", "m", "τοῦ"),
    ("στής", "m", "στοῦ"),
    # -ία feminines (long alpha after ι)
    ("ία", "f", "ίας"),
    # -τωρ/-τήρ
    ("τωρ", "m", "τορος"),
    ("τήρ", "m", "τῆρος"),
    # More 1st declension
    ("ρα", "f", "ρας"),   # -ρα long alpha
    ("ρη", "f", "ρης"),
    # -ιον neuters
    ("ιον", "n", "ίου"),
    ("ειον", "n", "είου"),
    # -τις feminines (3rd decl -ιδος pattern)
    ("τις", "f", "τεως"),
]

# Adverb endings (indeclinable, just add to lookup directly)
ADVERB_ENDINGS = ("ως", "δόν", "δην", "θεν", "θι", "σε", "ζε")


def is_greek(s):
    return any('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF' for c in s)


def has_accent(s):
    """Check if a Greek string has any accent mark."""
    nfd = unicodedata.normalize("NFD", s)
    return any(ord(c) in (0x0301, 0x0300, 0x0342) for c in nfd)


def add_penult_accent(s):
    """Add acute accent on the penultimate vowel (default for proper nouns).

    Greek proper nouns without accents are typically accented on the
    penultimate syllable (paroxytone).
    """
    vowels = "αεηιουωάέήίόύώὰὲὴὶὸὺὼᾶῆῖῦῶ"
    nfd = unicodedata.normalize("NFD", s)
    # Find positions of vowels (base characters)
    vowel_positions = []
    base_idx = 0
    for i, c in enumerate(nfd):
        if not unicodedata.combining(c):
            if c.lower() in vowels:
                vowel_positions.append(i)
            base_idx += 1

    if len(vowel_positions) < 2:
        # Monosyllable or no vowels - accent the last vowel
        if vowel_positions:
            pos = vowel_positions[-1]
            return unicodedata.normalize("NFC",
                nfd[:pos+1] + "\u0301" + nfd[pos+1:])
        return s

    # Add acute after the penultimate vowel
    pos = vowel_positions[-2]
    return unicodedata.normalize("NFC",
        nfd[:pos+1] + "\u0301" + nfd[pos+1:])


def clean_headword(raw):
    """Clean OCR artifacts from a headword."""
    if not raw:
        return ""
    # Take first word only
    hw = raw.strip().split(",")[0].strip().split(" ")[0].strip()
    # Strip punctuation
    hw = hw.strip(".,;:()[]{}—–-·\"' ")
    # Remove Latin characters (OCR errors)
    hw = ''.join(c for c in hw if not ('A' <= c <= 'Z' or 'a' <= c <= 'z'))
    # Normalize Unicode
    hw = unicodedata.normalize("NFC", hw)
    hw = strip_length_marks(hw)
    if not hw or len(hw) < 2 or not is_greek(hw):
        return ""
    # Add accent if missing (common in OCR'd Byzantine proper nouns)
    if not has_accent(hw) and hw[0].isupper():
        hw = add_penult_accent(hw)
    return hw


def parse_sophocles_entries():
    """Parse all Sophocles XML chunks, extract headwords + grammatical info."""
    entries = {}

    for xml_file in sorted(SOPH_DIR.glob("*.xml")):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except ET.ParseError:
            with open(xml_file, encoding="utf-8") as f:
                content = f.read()
            try:
                root = ET.fromstring("<wrap>" + content + "</wrap>")
            except ET.ParseError:
                continue

        for entry_elem in root.findall(".//entry"):
            orth = entry_elem.find("orth")
            if orth is None or not orth.text:
                continue

            hw = clean_headword(orth.text)
            if not hw:
                continue

            # Get full entry text (after the headword)
            full_text = ET.tostring(entry_elem, encoding="unicode", method="text")
            # Strip the headword from the beginning
            after_hw = full_text[len(orth.text):] if orth.text in full_text else full_text

            # Try to extract genitive + gender from inline text
            gender = ""
            genitive = ""
            m = INLINE_GEN_RE.match(after_hw)
            if m:
                gen_ending = m.group(1).strip()
                article = m.group(2).strip()
                gender = GENDER_MAP.get(article, "")

                if gender and gen_ending:
                    # Build full genitive from headword + ending
                    gen_ending_plain = strip_diacritics(gen_ending)
                    hw_plain = strip_diacritics(hw)

                    # If the ending is a full word (contains the stem), use as-is
                    if len(gen_ending) > 3:
                        genitive = gen_ending
                    else:
                        # It's a suffix - figure out how much of hw to keep
                        # Try matching the ending overlap
                        for strip_n in range(1, min(4, len(hw_plain))):
                            test_stem = hw_plain[:-strip_n]
                            if gen_ending_plain.startswith(test_stem[-1:]):
                                genitive = hw[:-strip_n] + gen_ending
                                break
                        if not genitive:
                            genitive = hw[:-1] + gen_ending
            else:
                # Check for just an article without genitive
                article_m = re.search(r'[,\s]+(ὁ|ἡ|τό|τὸ|τά)[,\s.]', after_hw[:30])
                if article_m:
                    gender = GENDER_MAP.get(article_m.group(1), "")

            # Check for adjective markers
            is_adj = bool(re.match(r'^[,\s]*(ές|όν|ον|ή)\b', after_hw))
            # Check for "see X" cross-references
            is_xref = bool(re.match(r'^[,\s]*(see|v\.|=)\s', after_hw))
            # Check for indeclinable
            is_indecl = "indeclinable" in after_hw[:80] or "indecl" in after_hw[:80]

            if hw not in entries:
                entries[hw] = {
                    "headword": hw,
                    "gender": gender,
                    "genitive": genitive,
                    "is_adj": is_adj,
                    "is_xref": is_xref,
                    "is_indecl": is_indecl,
                }

    return entries


def infer_from_ending(hw):
    """Infer gender and genitive from headword ending pattern."""
    hw_plain = strip_diacritics(hw)
    for nom_end, gender, gen_end in ENDING_RULES:
        nom_end_plain = strip_diacritics(nom_end)
        if hw_plain.endswith(nom_end_plain):
            stem = hw[:-len(nom_end)]
            return gender, stem + gen_end
    return "", ""


def analyze():
    """Show stats about Sophocles entries."""
    print("Parsing Sophocles entries...")
    entries = parse_sophocles_entries()
    print(f"Total entries: {len(entries)}")

    # Load existing lookup
    print(f"Loading {AG_LOOKUP}...")
    with open(AG_LOOKUP, encoding="utf-8") as f:
        lookup = json.load(f)

    # Categorize
    already_covered = 0
    has_gender_gen = 0
    inferable = 0
    adjectives = 0
    xrefs = 0
    indecl = 0
    adverbs = 0
    verbs = 0
    unknown = 0
    expandable = []      # nouns to expand via grc-decl
    adverb_list = []     # adverbs to add directly
    verb_list = []       # verbs (tracked for future expansion)

    # Verb ending detection
    VERB_ENDINGS = ("ω", "ομαι", "μαι", "μι", "ειμι")

    for hw, entry in entries.items():
        if has_editorial_sigla(hw):
            continue
        plain = strip_diacritics(hw)
        if hw in lookup or plain in lookup:
            already_covered += 1
            continue

        if entry["is_xref"]:
            xrefs += 1
            continue
        if entry["is_adj"]:
            adjectives += 1
            continue
        if entry["is_indecl"]:
            indecl += 1
            continue

        # Check for adverbs (-ως etc.)
        if any(plain.endswith(strip_diacritics(e)) for e in ADVERB_ENDINGS):
            # Only if it looks like an adverb (not a noun ending in -ος)
            if plain.endswith("ως") and len(plain) > 3:
                adverbs += 1
                adverb_list.append(hw)
                continue

        # Check for verbs
        if any(plain.endswith(strip_diacritics(e)) for e in VERB_ENDINGS):
            verbs += 1
            verb_list.append(hw)
            continue

        if entry["gender"] and entry["genitive"]:
            has_gender_gen += 1
            expandable.append(hw)
        else:
            # Try ending inference
            g, gen = infer_from_ending(hw)
            if g:
                inferable += 1
                entry["gender"] = g
                entry["genitive"] = gen
                expandable.append(hw)
            elif entry["gender"]:
                inferable += 1
                expandable.append(hw)
            else:
                unknown += 1

    print(f"\nAlready in lookup: {already_covered}")
    print(f"Cross-references: {xrefs}")
    print(f"Adjectives (skip for now): {adjectives}")
    print(f"Indeclinable: {indecl}")
    print(f"Adverbs (-ως etc.): {adverbs}")
    print(f"Verbs (future expansion): {verbs}")
    print(f"Has gender+genitive from text: {has_gender_gen}")
    print(f"Inferable from ending: {inferable}")
    print(f"Unknown/unhandled: {unknown}")
    print(f"\nTotal expandable nouns: {len(expandable)}")
    print(f"Adverbs to add directly: {len(adverb_list)}")

    return entries, expandable, adverb_list


def expand_all():
    """Expand Sophocles headwords and merge into ag_lookup.json."""
    entries, expandable, adverb_list = analyze()

    print(f"\nLoading lookup...")
    with open(AG_LOOKUP, encoding="utf-8") as f:
        lookup = json.load(f)
    original_size = len(lookup)

    wtp = get_wtp()

    stats = {"expanded": 0, "failed": 0, "new_forms": 0, "collisions": 0}
    t0 = time.time()

    for i, hw in enumerate(expandable):
        entry = entries[hw]
        gender = entry["gender"]
        genitive = entry["genitive"]

        forms, err = expand_noun(wtp, hw, gender, genitive)

        if err or not forms:
            stats["failed"] += 1
            continue

        stats["expanded"] += 1

        for raw_form in forms:
            for form in resolve_editorial_form(raw_form):
                if form not in lookup:
                    lookup[form] = hw
                    stats["new_forms"] += 1
                elif lookup[form] != hw:
                    stats["collisions"] += 1

                plain = strip_diacritics(form)
                if plain != form:
                    if plain not in lookup:
                        lookup[plain] = hw
                        stats["new_forms"] += 1
                    elif lookup[plain] != hw:
                        stats["collisions"] += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(expandable) - i - 1) / rate
            print(f"  {i+1:,}/{len(expandable):,} "
                  f"({stats['expanded']:,} ok, {stats['failed']:,} fail, "
                  f"{stats['new_forms']:,} new forms) "
                  f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Expanded: {stats['expanded']:,} / {len(expandable):,}")
    print(f"  Failed: {stats['failed']:,}")
    print(f"  New forms added: {stats['new_forms']:,}")
    print(f"  Collisions (kept existing): {stats['collisions']:,}")
    print(f"  Lookup size: {original_size:,} -> {len(lookup):,}")

    # Add adverbs directly (no inflection needed)
    adv_added = 0
    for hw in adverb_list:
        if has_editorial_sigla(hw):
            continue
        if hw not in lookup:
            lookup[hw] = hw
            adv_added += 1
        plain = strip_diacritics(hw)
        if plain != hw and plain not in lookup:
            lookup[plain] = hw
            adv_added += 1
    print(f"  Adverbs added: {adv_added}")

    print(f"\nSaving to {AG_LOOKUP}...")
    with open(AG_LOOKUP, "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False)
    size_mb = AG_LOOKUP.stat().st_size / (1024 * 1024)
    print(f"  {size_mb:.1f} MB written")


def find_verb_candidates(entries, lookup):
    """Find Sophocles verb entries not already in ag_lookup.json."""
    VERB_ENDINGS = ("ω", "εω", "αω", "οω", "μαι")
    candidates = []

    for hw, entry in entries.items():
        if has_editorial_sigla(hw):
            continue
        plain = strip_diacritics(hw)
        # Skip if already covered
        if hw in lookup or plain in lookup:
            continue
        # Skip non-verb categories
        if entry["is_xref"] or entry["is_adj"] or entry["is_indecl"]:
            continue
        # Check verb endings
        if any(plain.endswith(e) for e in VERB_ENDINGS):
            candidates.append(hw)

    return candidates


def expand_verbs():
    """Expand Sophocles verb headwords and merge into ag_lookup.json."""
    print("Parsing Sophocles entries...")
    entries = parse_sophocles_entries()
    print(f"Total entries: {len(entries)}")

    print(f"\nLoading {AG_LOOKUP}...")
    with open(AG_LOOKUP, encoding="utf-8") as f:
        lookup = json.load(f)
    original_size = len(lookup)
    print(f"  {original_size:,} existing entries")

    candidates = find_verb_candidates(entries, lookup)
    print(f"Sophocles verbs to expand: {len(candidates):,}")

    wtp = get_wtp()

    stats = {"expanded": 0, "failed": 0, "mi_verbs": 0, "new_forms": 0, "collisions": 0}
    t0 = time.time()

    for i, hw in enumerate(candidates):
        forms, err = expand_verb(wtp, hw)

        if err or not forms:
            stats["failed"] += 1
            if err == "mi-verb":
                stats["mi_verbs"] += 1
            continue

        stats["expanded"] += 1

        for raw_form in forms:
            for form in resolve_editorial_form(raw_form):
                if form not in lookup:
                    lookup[form] = hw
                    stats["new_forms"] += 1
                elif lookup[form] != hw:
                    stats["collisions"] += 1

                plain = strip_diacritics(form)
                if plain != form:
                    if plain not in lookup:
                        lookup[plain] = hw
                        stats["new_forms"] += 1
                    elif lookup[plain] != hw:
                        stats["collisions"] += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(candidates) - i - 1) / rate
            print(f"  {i+1:,}/{len(candidates):,} "
                  f"({stats['expanded']:,} ok, {stats['failed']:,} fail, "
                  f"{stats['new_forms']:,} new forms) "
                  f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Expanded: {stats['expanded']:,} / {len(candidates):,} verbs")
    print(f"  Failed: {stats['failed']:,} (of which {stats['mi_verbs']:,} -μι verbs)")
    print(f"  New forms added: {stats['new_forms']:,}")
    print(f"  Collisions (kept existing): {stats['collisions']:,}")
    print(f"  Lookup size: {original_size:,} -> {len(lookup):,}")

    print(f"\nSaving to {AG_LOOKUP}...")
    with open(AG_LOOKUP, "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False)
    size_mb = AG_LOOKUP.stat().st_size / (1024 * 1024)
    print(f"  {size_mb:.1f} MB written")


def main():
    parser = argparse.ArgumentParser(description="Expand Sophocles lexicon headwords")
    parser.add_argument("--analyze", action="store_true",
                        help="Show stats without modifying lookup")
    parser.add_argument("--expand", action="store_true",
                        help="Expand nouns and merge into ag_lookup.json")
    parser.add_argument("--expand-verbs", action="store_true",
                        help="Expand verbs and merge into ag_lookup.json")
    args = parser.parse_args()

    if args.analyze:
        analyze()
    elif args.expand:
        expand_all()
    elif args.expand_verbs:
        expand_verbs()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
