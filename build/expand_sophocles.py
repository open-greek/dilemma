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
import hashlib
import json
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# Import shared infrastructure from expand_lsj
from expand_lsj import (
    get_wtp, expand_noun, expand_verb, strip_length_marks, strip_diacritics,
    mark_alpha_length, AG_LOOKUP, DATA_DIR, ARTICLES,
    load_wiktionary_genitives, resolve_noun_genitive, _genitive_is_junk,
    noun_generation_input_hashes,
    noun_expansion_error,
    add_input_arguments, configure_inputs,
)
from dilemma.form_sanitize import (  # noqa: E402
    has_editorial_sigla,
    resolve_editorial_form,
    sanitize_form,
)

SOPH_DIR = DATA_DIR / "sophocles" / "chunks"
# build/build_grc_lexical_heads.py decides the breathings the OCR lost
# (build/sophocles_breathing.py) and records them in this file.
LEXICAL_HEADS = DATA_DIR / "grc_lexical_heads.json"
_DEFAULT = object()


def load_breathing_review(path: Path | None = None, *, required: bool = False):
    """The ``sophocles_breathing`` record of grc_lexical_heads.json, or None.
    ``required`` refuses a missing record (the expansion runs)."""
    path = path or LEXICAL_HEADS
    review = (json.loads(path.read_text(encoding="utf-8")).get("sophocles_breathing")
              if path.exists() else None)
    if review is None and required:
        raise SystemExit(f"{path} has no sophocles_breathing record; rebuild it "
                         "with build/build_grc_lexical_heads.py first")
    return review


def _apply_breathing(raw: str, hw: str, review) -> str | None:
    """``hw`` with the breathing the review restored for the source head
    ``raw``, unchanged, or None when the review withheld it."""
    if not review:
        return hw
    from dilemma_build.build_grc_lexical_heads import normalize_headword
    from dilemma_build.sophocles_breathing import add_breathing, breathing_of, needs_breathing
    key = normalize_headword(raw)
    if key in review["withheld"]:
        return None
    row = review["corrected"].get(key)
    if row and needs_breathing(hw):
        return add_breathing(hw, breathing_of(row["head"]))
    return hw

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

# An article states the entry's gender when it is the first field after the
# head or follows one field, the genitive (which the OCR may garble):
# "καισάριης, ἡ, the Latin caesaries", "ἀτρεπτότης, ητος; ἡ". An article
# further on belongs to the definition and is only a guess.
STATED_ARTICLE_RE = re.compile(
    r'^[,\s]*(?:[^\s,;()=—–-]+[,;]?\s+)?(ὁ|ἡ|τό|τὸ|τά)[,\s.]')
# ... unless the neuter ending follows it: "ἀρθρικός, ἡ. ὄν" is a three-ending
# adjective whose feminine ή the OCR read as the article.
_NEUTER_ENDING_NEXT = re.compile(r"^[\s.,;]*ον\b")

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


def accent_count(s):
    """How many accent marks (acute, grave, circumflex) a Greek string has."""
    nfd = unicodedata.normalize("NFD", s)
    return sum(ord(c) in (0x0301, 0x0300, 0x0342) for c in nfd)


def has_accent(s):
    """Check if a Greek string has any accent mark."""
    return accent_count(s) > 0


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


# Punctuation the OCR leaves around a printed head.
_EDGE_PUNCTUATION = ".,;:()[]{}—–-·\"' "
# A full stop, colon or semicolon (or the raised dot) closes the head; so does
# the lexicon's equals sign, which the OCR gives as a dash or a run of hyphens.
_HEAD_CLOSERS = (".", ";", ":", "·", "—", "–", "=", "--")
# Spacing breathings and quotation marks the OCR puts before a capital.
_LEADING_MARKS = "᾿᾽'‘’“\""


def _starts_greek_word(token):
    """Whether a token inside an <orth> opens with a Greek letter (the OCR
    also writes the micro sign for μ)."""
    token = token.lstrip(_LEADING_MARKS)
    return bool(token) and (token[0] == "µ" or (
        is_greek(token[0]) and unicodedata.category(token[0]).startswith("L")))


def read_orth(raw):
    """Read an <orth>: ``(first word, reason, rest)``.

    The first word is cleaned as a headword ("" when it holds no Greek word)
    and ``rest`` is the element's text after it. ``reason`` is "" when the
    first word is the entry's head, that is when the text before the first
    comma ends there or goes on with entry text: after a full stop, colon,
    semicolon or the equals sign (a dash, or two or more hyphens), or with a
    parenthesis, a digit or a Latin word (``or``/``for`` before a variant, a
    citation, an ending the OCR read as Latin).

    ``reason`` is "multi-word" when another Greek word follows the first one
    directly. The head is then a phrase (Νέα Ῥώμη, ἀνθ᾽ ὧν) or a word the
    OCR split (κατα- φθάνω), and its first word is not the entry.
    """
    if not raw:
        return "", "", ""
    text = raw.strip()
    segment = text.split(",")[0].strip()
    first = segment.split(" ")[0].strip()
    rest = text[len(first):]
    # Strip punctuation
    hw = first.strip(_EDGE_PUNCTUATION)
    # Remove Latin characters (OCR errors)
    hw = ''.join(c for c in hw if not ('A' <= c <= 'Z' or 'a' <= c <= 'z'))
    # Normalize Unicode
    hw = unicodedata.normalize("NFC", hw)
    hw = strip_length_marks(hw)
    if not hw or len(hw) < 2 or not is_greek(hw):
        return "", "", rest
    # Add accent if missing (common in OCR'd Byzantine proper nouns)
    if not has_accent(hw) and hw[0].isupper():
        hw = add_penult_accent(hw)
    following = segment[len(first):].split()
    if (following and not first.endswith(_HEAD_CLOSERS)
            and _starts_greek_word(following[0])):
        return hw, "multi-word", rest
    return hw, "", rest


def clean_headword(raw):
    """The head an <orth> prints, cleaned of OCR artifacts; "" when it has
    none or prints several words (see ``read_orth``)."""
    hw, reason, _ = read_orth(raw)
    return "" if reason else hw


# The entry text after a headword, folded for the OCR: accents, breathings and
# case are dropped, and the Latin look-alikes the OCR gives for ο and ν are
# mapped back (ἀγγελτικός, ή, όν; ἀδικητικός, ἦ, ov; ἀβυσσικός, ή, 6v).
_OCR_LOOKALIKES = str.maketrans({"o": "ο", "v": "ν", "6": "ο", "d": "ο",
                                 "e": "ε", "s": "ς", "a": "α"})
_THREE_ENDINGS = re.compile(r"^\s*,\s*[ηα]\s*,\s*ο?ν\b")
_TWO_ENDINGS = re.compile(r"^\s*,\s*(ον|ες)\b")


def _fold_ocr(text: str) -> str:
    return strip_diacritics(text).lower().translate(_OCR_LOOKALIKES)


def adjective_endings(after_headword: str) -> str | None:
    """'three' or 'two' when the entry gives an adjective's endings, else None."""
    folded = _fold_ocr(after_headword[:24])
    if _THREE_ENDINGS.match(folded):
        return "three"
    if _TWO_ENDINGS.match(folded):
        return "two"
    return None


def new_head_audit():
    """An empty record of one parse's head decisions (``head_rules`` in the
    noun generation audit)."""
    return {"printed": {}, "multi_word_heads": {}, "stated_gender": {},
            "malformed_heads": {}, "repaired_heads": {}, "lowercased_heads": {}}


def load_lexical_heads(path: Path | None = None):
    """The heads of grc_lexical_heads.json that a source other than this
    lexicon lists, or None without the file. The file's own Sophocles heads
    are the ones being judged, so they are no evidence here."""
    path = path or LEXICAL_HEADS
    if not path.exists():
        return None
    heads = json.loads(path.read_text(encoding="utf-8")).get("heads") or {}
    return frozenset(head for head, sources in heads.items()
                     if any(not source.startswith("sophocles/") for source in sources))


# Hyphens and dashes left inside a printed head: the lexicon divides some
# compounds (καθ-ηδύνω) and the OCR keeps line-break hyphens (Σινα-ῖτις).
_INNER_HYPHENS = "-—–"


def lowercase_twin(head, lexical_heads):
    """The lowercase spelling of a capitalized head when the other lexical
    sources list that spelling and not the capitalized one, else "". The
    OCR prints some capitals' breathing as a separate mark (᾿Αθεότης); the
    twin is that of the head with the mark joined (Ἀθεότης -> ἀθεότης)."""
    spelled = sanitize_form(head)
    lowered = spelled.lower()
    if (lexical_heads and spelled[:1].isupper() and spelled not in lexical_heads
            and lowered in lexical_heads):
        return lowered
    return ""


def settle_head(printed, lexical_heads=None):
    """``(head, rule)``: the head a printed head is generated under.

    ``rule`` is "" when the printed head stands. A head is malformed, and
    ``head`` is "", when it carries two accents ("two-accents": Ἔρετριάς,
    οἴηματικός) or an inner hyphen ("inner-hyphen": Σινα-ῖτις,
    ὕπο-κατασκευή). The one repair is to drop the hyphens, and only when the
    joined word has a single accent and another lexical source lists it
    ("joined": καθ-ηδύνω -> καθηδύνω).

    A capitalized head is a common word the lexicon prints with a capital
    when the other sources list only its lowercase spelling ("lowercased":
    ᾿Αθεότης -> ἀθεότης, Ἔκπτυξις -> ἔκπτυξις). A capitalized head another
    source lists (Πέτρα beside πέτρα), or one with no lowercase twin, is a
    name and stands.
    """
    if any(mark in printed for mark in _INNER_HYPHENS):
        joined = "".join(ch for ch in printed if ch not in _INNER_HYPHENS)
        if accent_count(joined) <= 1:
            if joined in (lexical_heads or ()):
                return joined, "joined"
            lowered = lowercase_twin(joined, lexical_heads)
            if lowered:
                return lowered, "joined"
        return "", "inner-hyphen"
    if accent_count(printed) > 1:
        return "", "two-accents"
    lowered = lowercase_twin(printed, lexical_heads)
    if lowered:
        return lowered, "lowercased"
    return printed, ""


def settle_heads(entries, lexical_heads, audit=None):
    """The parsed entries keyed by the head each is generated under.

    Malformed heads are dropped; repaired and lowercased ones are re-keyed
    in place (see ``settle_head``), and an entry the lexicon also prints
    under the new head keeps its own text. ``audit`` records the three
    groups by printed head.
    """
    audit = audit if audit is not None else new_head_audit()
    malformed = audit.setdefault("malformed_heads", {})
    changed = {"joined": audit.setdefault("repaired_heads", {}),
               "lowercased": audit.setdefault("lowercased_heads", {})}
    settled = {}
    for printed, entry in entries.items():
        head, rule = settle_head(printed, lexical_heads)
        if not head:
            malformed[printed] = rule
            continue
        if head != printed:
            changed[rule][printed] = head
            if head in entries:
                continue
            entry = dict(entry, headword=head, printed=printed)
        settled.setdefault(head, entry)
    return settled


def parse_sophocles_entries(breathing=_DEFAULT, audit=None):
    """Parse all Sophocles XML chunks, extract headwords + grammatical info.

    ``breathing`` is the lexical heads' breathing review (default: read from
    grc_lexical_heads.json when present); withheld heads are skipped.
    ``audit`` (``new_head_audit()``) receives the heads the rules of
    ``read_orth`` set aside, and under ``printed`` every first word as the
    lexicon prints it, entry or not."""
    entries = {}
    if breathing is _DEFAULT:
        breathing = load_breathing_review()
    if audit is None:
        audit = new_head_audit()
    printed_heads = audit.setdefault("printed", {})
    multi_word = audit.setdefault("multi_word_heads", {})

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

            printed, reason, _rest = read_orth(orth.text)
            if printed:
                printed = _apply_breathing(orth.text, printed, breathing)
            if not printed:
                continue

            # Get full entry text (after the headword)
            full_text = ET.tostring(entry_elem, encoding="unicode", method="text")
            # The text right after the headword (it need not open the entry)
            at = full_text.find(orth.text)
            after_hw = full_text[at + len(orth.text):] if at >= 0 else full_text

            # Try to extract genitive + gender from inline text
            gender = ""
            genitive = ""
            gender_stated = False
            m = INLINE_GEN_RE.match(after_hw)
            if m:
                gen_ending = m.group(1).strip()
                article = m.group(2).strip()
                gender = GENDER_MAP.get(article, "")
                gender_stated = bool(gender)

                if gender and gen_ending:
                    # Preserve the exact source field. The shared resolver
                    # distinguishes a complete form from a supported suffix.
                    genitive = gen_ending
            else:
                # Check for just an article without genitive
                stated_m = STATED_ARTICLE_RE.match(after_hw)
                article_m = stated_m or re.search(
                    r'[,\s]+(ὁ|ἡ|τό|τὸ|τά)[,\s.]', after_hw[:30])
                if article_m:
                    gender = GENDER_MAP.get(article_m.group(1), "")
                    gender_stated = stated_m is not None and not (
                        _NEUTER_ENDING_NEXT.match(_fold_ocr(
                            after_hw[stated_m.end(1):stated_m.end(1) + 8])))

            # Check for adjective markers
            endings = adjective_endings(after_hw)
            is_adj = endings is not None
            # Check for "see X" cross-references
            is_xref = bool(re.match(r'^[,\s]*(see|v\.|=)\s', after_hw))
            # Check for indeclinable
            is_indecl = "indeclinable" in after_hw[:80] or "indecl" in after_hw[:80]

            entry = {
                "headword": printed,
                "gender": gender,
                "gender_stated": gender_stated,
                "genitive": genitive,
                "supplied_genitive": genitive,
                "is_adj": is_adj,
                "adjective_endings": endings,
                "is_xref": is_xref,
                "is_indecl": is_indecl,
            }
            # The first word as printed, whether or not it is an entry: the
            # noun audit names superseded lemmas by it.
            printed_heads.setdefault(printed, entry)

            if reason == "multi-word":
                multi_word.setdefault(
                    " ".join(orth.text.strip().split(",")[0].split()), printed)
                continue

            hw = printed
            if hw not in entries:
                entries[hw] = entry

    return entries


def infer_from_ending(hw, gender=""):
    """Infer gender and genitive from headword ending pattern.

    ``gender`` is the gender the entry states: only a class of that gender
    is returned, and ("", "") when the ending has none."""
    hw_plain = strip_diacritics(hw)
    for nom_end, rule_gender, gen_end in ENDING_RULES:
        if gender and rule_gender != gender:
            continue
        nom_end_plain = strip_diacritics(nom_end)
        if hw_plain.endswith(nom_end_plain):
            stem = hw[:-len(nom_end)]
            return rule_gender, stem + gen_end
    return "", ""


def add_plain_key(lookup, plain, hw, withheld):
    """Write the accent-stripped key ``plain`` of a form of ``hw`` unless the
    table has it. True when the key was written.

    A capitalized key is withheld (and recorded in ``withheld``) when its
    lowercase spelling already reads another lemma: unaccented input folds
    to that key, and Εστιας -> Ἑστιάς would take the reading of εστιας ->
    ἑστία.
    """
    if plain in lookup:
        return False
    lowered = plain.lower()
    other = lookup.get(lowered) if lowered != plain else None
    if other is not None and other != hw:
        withheld[plain] = {"lemma": hw, "lowercase_reading": other}
        return False
    lookup[plain] = hw
    return True


def withheld_keys_record(withheld):
    return {"count": len(withheld), "keys": dict(sorted(withheld.items()))}


def analyze(audit=None):
    """Show stats about Sophocles entries."""
    print("Parsing Sophocles entries...")
    entries = settle_heads(parse_sophocles_entries(audit=audit),
                           load_lexical_heads(), audit)
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
            # Try ending inference. A stated gender wins: the ending gives
            # the genitive only through a class of that gender, and without
            # one the shared resolver decides (or withholds the noun).
            stated = entry["gender"] if entry.get("gender_stated") else ""
            g, gen = infer_from_ending(hw, stated)
            if stated and audit is not None:
                by_ending = infer_from_ending(hw)[0]
                if by_ending and by_ending != stated:
                    audit.setdefault("stated_gender", {})[hw] = {
                        "stated": stated, "ending_class": by_ending}
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
    load_breathing_review(required=True)
    input_hashes = sophocles_noun_input_hashes()
    head_audit = new_head_audit()
    entries, expandable, adverb_list = analyze(head_audit)

    print(f"\nLoading lookup...")
    with open(AG_LOOKUP, encoding="utf-8") as f:
        lookup = json.load(f)
    original_size = len(lookup)

    wtp = get_wtp()
    wikt_genitives = load_wiktionary_genitives()
    superseded = set()
    resolutions = {}
    # Superseded lemmas are named as the lexicon prints them: the historical
    # layer keyed its rows that way, also for a first word that is no entry.
    for hw, entry in (head_audit["printed"] or entries).items():
        supplied = entry.get("supplied_genitive", "")
        if not entry["gender"]:
            continue
        resolved, error = resolve_noun_genitive(hw, entry["gender"], entry["genitive"],
                                               wikt_genitives)
        error = error or noun_expansion_error(hw, resolved)
        if not supplied and error != "unsupported-irregular-noun":
            continue
        if error or resolved != supplied:
            resolutions[hw] = {"supplied": supplied, "resolved": resolved,
                               "reason": error or "completed-genitive"}
            if error == "unsupported-irregular-noun" or not _genitive_is_junk(supplied):
                superseded.add(hw)

    stats = {"expanded": 0, "failed": 0, "new_forms": 0, "collisions": 0}
    failures = Counter()
    withheld_keys = {}
    t0 = time.time()

    for i, hw in enumerate(expandable):
        entry = entries[hw]
        gender = entry["gender"]
        genitive = entry["genitive"]

        forms, err = expand_noun(wtp, hw, gender, genitive, wikt_genitives)
        if hw in head_audit["stated_gender"]:
            head_audit["stated_gender"][hw]["result"] = (
                "expanded" if forms and not err else err or "empty-noun-paradigm")

        if err or not forms:
            stats["failed"] += 1
            failures[err or "empty-noun-paradigm"] += 1
            continue

        stats["expanded"] += 1

        for raw_form in sorted(forms):
            for form in resolve_editorial_form(raw_form):
                if form not in lookup:
                    lookup[form] = hw
                    stats["new_forms"] += 1
                elif lookup[form] != hw:
                    stats["collisions"] += 1

                plain = strip_diacritics(form)
                if plain != form:
                    if add_plain_key(lookup, plain, hw, withheld_keys):
                        stats["new_forms"] += 1
                    elif lookup.get(plain, hw) != hw:
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
    print(f"  Noun failure reasons: {dict(sorted(failures.items()))}")
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
        if plain != hw and add_plain_key(lookup, plain, hw, withheld_keys):
            adv_added += 1
    print(f"  Adverbs added: {adv_added}")
    print(f"  Capitalized unaccented keys withheld: {len(withheld_keys):,}")

    print(f"\nSaving to {AG_LOOKUP}...")
    with open(AG_LOOKUP, "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False)
    size_mb = AG_LOOKUP.stat().st_size / (1024 * 1024)
    print(f"  {size_mb:.1f} MB written")
    AG_LOOKUP.with_name("ag_sophocles_noun_generation_audit.json").write_text(
        json.dumps({
            "schema_version": 1,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "shared_generator_sha256": hashlib.sha256(
                Path(__file__).with_name("expand_lsj.py").read_bytes()).hexdigest(),
            "input_sha256": input_hashes,
            "superseded_lemmas": sorted(superseded),
            "failures": dict(sorted(failures.items())),
            "resolutions": dict(sorted(resolutions.items())),
            "head_rules": head_rules_record(head_audit),
            "withheld_capitalized_keys": withheld_keys_record(withheld_keys),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def head_rules_record(head_audit):
    """The audit's ``head_rules``: each rule's heads and their count. The
    rules apply to every entry, so the verb step reads the same heads."""
    return {name: {"count": len(rows), "heads": dict(sorted(rows.items()))}
            for name, rows in sorted(head_audit.items()) if name != "printed"}


def sophocles_noun_input_hashes():
    hashes = noun_generation_input_hashes(include_lsj=False)
    review = json.dumps(load_breathing_review(), ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"))
    hashes["sophocles_breathing"] = hashlib.sha256(review.encode("utf-8")).hexdigest()
    heads = load_lexical_heads()
    listed = "null" if heads is None else "\n".join(sorted(heads))
    hashes["lexical_heads"] = hashlib.sha256(listed.encode("utf-8")).hexdigest()
    for path in sorted(SOPH_DIR.glob("*.xml")):
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
            hashes["sophocles/" + path.name] = digest.hexdigest()
    return hashes


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
    load_breathing_review(required=True)
    print("Parsing Sophocles entries...")
    entries = settle_heads(parse_sophocles_entries(), load_lexical_heads())
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
    withheld_keys = {}
    t0 = time.time()

    for i, hw in enumerate(candidates):
        forms, err = expand_verb(wtp, hw)

        if err or not forms:
            stats["failed"] += 1
            if err == "mi-verb":
                stats["mi_verbs"] += 1
            continue

        stats["expanded"] += 1

        for raw_form in sorted(forms):
            for form in resolve_editorial_form(raw_form):
                if form not in lookup:
                    lookup[form] = hw
                    stats["new_forms"] += 1
                elif lookup[form] != hw:
                    stats["collisions"] += 1

                plain = strip_diacritics(form)
                if plain != form:
                    if add_plain_key(lookup, plain, hw, withheld_keys):
                        stats["new_forms"] += 1
                    elif lookup.get(plain, hw) != hw:
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
    print(f"  Capitalized unaccented keys withheld: {len(withheld_keys):,}")
    print(f"  Lookup size: {original_size:,} -> {len(lookup):,}")

    print(f"\nSaving to {AG_LOOKUP}...")
    with open(AG_LOOKUP, "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False)
    size_mb = AG_LOOKUP.stat().st_size / (1024 * 1024)
    print(f"  {size_mb:.1f} MB written")
    AG_LOOKUP.with_name("ag_sophocles_verb_generation_audit.json").write_text(
        json.dumps({
            "schema_version": 1,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "withheld_capitalized_keys": withheld_keys_record(withheld_keys),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Expand Sophocles lexicon headwords")
    parser.add_argument("--analyze", action="store_true",
                        help="Show stats without modifying lookup")
    parser.add_argument("--expand", action="store_true",
                        help="Expand nouns and merge into ag_lookup.json")
    parser.add_argument("--expand-verbs", action="store_true",
                        help="Expand verbs and merge into ag_lookup.json")
    add_input_arguments(parser)
    args = parser.parse_args()
    configure_inputs(args.kaikki, args.lsj9_dir)

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
