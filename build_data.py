#!/usr/bin/env python3
"""Build training data for Dilemma from Wiktionary kaikki JSONL dumps.

Scans both EN and EL Wiktionary dumps for Greek entries, extracts every
inflected form -> lemma pair from inflection tables, form_of/alt_of
references, and dialect-tagged paradigms. Produces:
  - data/mg_pairs.json: Modern Greek form->lemma training pairs
  - data/el_participles.json: source-owned standalone participle morphology
  - data/el_noun_heads.json: source evidence for independent noun owners
  - data/el_adjective_heads.json: source evidence for independent adjective owners
  - data/el_closed_class_heads.json: POS-specific determiner/pronoun head evidence
  - data/ag_pairs.json: Ancient Greek form->lemma training pairs
  - data/mg_lookup.json: flat lookup table {form: lemma}
  - data/ag_lookup.json: flat lookup table {form: lemma}
  - data/mg_lookup_scored.json: scored lookup {form: {lemma, confidence}}
  - data/ag_lookup_scored.json: scored lookup {form: {lemma, confidence}}
  - data/mg_pos_lookup.json: POS-indexed disambiguation table
  - data/ag_pos_lookup.json: POS-indexed disambiguation table

The kaikki dumps are from https://kaikki.org/dictionary/ and contain
complete Wiktionary entries in JSONL format with inflection paradigms.

Usage:
    python build_data.py --download              # download dumps if missing
    python build_data.py --kaikki /path/to/dumps  # specify dump directory
    python build_data.py                          # auto-detect dump locations
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import NamedTuple

from dilemma.verb_source import is_periphrastic_auxiliary
from dilemma.nominal_header_forms import is_abbreviated_nominal_header
from dilemma.verb_table_fragments import (
    retain_complete_passive_aorist_readings, stemless_passive_aorist_dialects,
    retain_complete_aorist_nonpast_readings, stemless_aorist_nonpast_voices,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

# Default kaikki dump directory. Set KAIKKI_DIR env var to override.
# Supports two layouts:
#   flat:   <dir>/kaikki.org-en-dictionary-Greek.jsonl
#   nested: <dir>/en-el/kaikki.org-dictionary-Greek-words.jsonl
DEFAULT_DUMP_DIR = Path(os.environ.get(
    "KAIKKI_DIR", DATA_DIR))

DUMPS = {
    "el": {
        "en": "kaikki.org-en-dictionary-Greek.jsonl",
        "el": "kaikki.org-el-dictionary-Greek.jsonl",
    },
    "grc": {
        "en": "kaikki.org-en-dictionary-AncientGreek.jsonl",
        "el": "kaikki.org-el-dictionary-AncientGreek.jsonl",
    },
    "mgr": {
        "el": "kaikki.org-el-dictionary-MedievalGreek.jsonl",
    },
}

_GENDER_TAGS = frozenset({"masculine", "feminine", "neuter"})

# Alternate filenames/paths for the nested directory layout
_NESTED_MAP = {
    "kaikki.org-en-dictionary-Greek.jsonl":
        "en-el/kaikki.org-dictionary-Greek-words.jsonl",
    "kaikki.org-en-dictionary-AncientGreek.jsonl":
        "en-el/kaikki.org-dictionary-AncientGreek.jsonl",
    "kaikki.org-el-dictionary-Greek.jsonl":
        "el/kaikki.org-dictionary-Greek.jsonl",
    "kaikki.org-el-dictionary-AncientGreek.jsonl":
        "el/kaikki.org-dictionary-AncientGreek.jsonl",
    "kaikki.org-el-dictionary-MedievalGreek.jsonl":
        "el/kaikki.org-dictionary-MedievalGreek.jsonl",
}


def resolve_dump(filename: str, dump_dir: Path) -> Path:
    """Resolve a dump filename, checking flat and nested layouts."""
    direct = dump_dir / filename
    if direct.exists():
        return direct
    if filename in _NESTED_MAP:
        nested = dump_dir / _NESTED_MAP[filename]
        if nested.exists():
            return nested
    return direct

DOWNLOAD_URLS = {
    "kaikki.org-en-dictionary-Greek.jsonl":
        "https://kaikki.org/dictionary/Greek/kaikki.org-dictionary-Greek.jsonl",
    "kaikki.org-el-dictionary-Greek.jsonl":
        "https://kaikki.org/elwiktionary/Greek/kaikki.org-dictionary-Greek.jsonl",
    "kaikki.org-en-dictionary-AncientGreek.jsonl":
        "https://kaikki.org/dictionary/Ancient%20Greek/kaikki.org-dictionary-AncientGreek.jsonl",
    "kaikki.org-el-dictionary-AncientGreek.jsonl":
        "https://kaikki.org/elwiktionary/Ancient%20Greek/kaikki.org-dictionary-AncientGreek.jsonl",
    "kaikki.org-el-dictionary-MedievalGreek.jsonl":
        "https://kaikki.org/elwiktionary/Medieval%20Greek/words/kaikki.org-dictionary-MedievalGreek-words.jsonl",
}

# Dialect prefixes found in table-tags (e.g. "Epic declension-1")
DIALECT_PREFIXES = {
    "Epic", "Attic", "Ionic", "Doric", "Aeolic", "Koine",
    "Homeric", "Laconian", "Boeotian", "Arcadocypriot",
}

# Tense labels found in verb table headers. Kaikki keeps the header as a
# synthetic ``table-tags`` form but often omits its tense from the cells.
VERB_TENSE_LABELS = {
    "present": "present",
    "imperfect": "imperfect",
    "future": "future",
    "aorist": "aorist",
    "perfect": "perfect",
    "pluperfect": "pluperfect",
    "future perfect": "future-perfect",
    "contracted present": "present",
    "contracted imperfect": "imperfect",
    "contracted future": "future",
    "contracted aorist": "aorist",
    "contracted perfect": "perfect",
    "contracted pluperfect": "pluperfect",
}

_VERB_AXES = {
    "active", "middle", "passive", "mediopassive",
    "indicative", "subjunctive", "optative", "imperative",
    "infinitive", "participle",
    "first-person", "second-person", "third-person",
}

# Wiktionary POS field -> UPOS mapping (for POS-indexed lookup tables)
WIKT_TO_UPOS = {
    "noun": "NOUN", "verb": "VERB", "adj": "ADJ", "name": "PROPN",
    "adv": "ADV", "pron": "PRON", "det": "DET", "num": "NUM",
    "prep": "ADP", "conj": "CCONJ", "particle": "PART",
    "article": "DET", "intj": "INTJ",
}


def strip_length_marks(s: str) -> str:
    """Strip vowel length marks (breve/macron) from headwords."""
    nfd = unicodedata.normalize("NFD", s)
    out = []
    for ch in nfd:
        cp = ord(ch)
        if cp in (0x0306, 0x0304):  # combining breve, combining macron
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


# Polytonic combining marks to strip for monotonic conversion
_POLYTONIC_STRIP = {
    0x0313,  # COMBINING COMMA ABOVE (smooth breathing)
    0x0314,  # COMBINING REVERSED COMMA ABOVE (rough breathing)
    0x0345,  # COMBINING GREEK YPOGEGRAMMENI (iota subscript)
    0x0306,  # COMBINING BREVE
    0x0304,  # COMBINING MACRON
}
_POLYTONIC_TO_ACUTE = {
    0x0300,  # COMBINING GRAVE ACCENT -> acute
    0x0342,  # COMBINING GREEK PERISPOMENI (circumflex) -> acute
}


def to_monotonic(s: str) -> str:
    """Convert polytonic Greek to monotonic (strip breathings, normalize accents)."""
    nfd = unicodedata.normalize("NFD", s)
    out = []
    for ch in nfd:
        cp = ord(ch)
        if cp in _POLYTONIC_STRIP:
            continue
        if cp in _POLYTONIC_TO_ACUTE:
            out.append("\u0301")  # combining acute
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def strip_accents(s: str) -> str:
    """Strip all accents and diacritics."""
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn"))


def _is_greek(s: str) -> bool:
    """Check if string is entirely Greek characters + diacriticals."""
    return all(
        "\u0370" <= c <= "\u03FF"       # Greek and Coptic
        or "\u1F00" <= c <= "\u1FFF"    # Greek Extended
        or "\u0300" <= c <= "\u036F"    # Combining diacriticals
        for c in s
    )


def _parse_dialect(table_tag: str) -> str:
    """Extract dialect name from a table-tags value like 'Epic declension-1'."""
    for dialect in DIALECT_PREFIXES:
        if table_tag.startswith(dialect):
            return dialect
    return ""


def _parse_verb_tense(table_tag: str) -> str:
    """Extract a canonical tense from a Wiktionary table header."""
    if not table_tag:
        return ""
    rest = table_tag
    for dialect in DIALECT_PREFIXES:
        if rest.startswith(dialect + " "):
            rest = rest[len(dialect) + 1:]
            break
        if rest == dialect:
            return ""
    normalized = rest.lower()
    if normalized in VERB_TENSE_LABELS:
        return VERB_TENSE_LABELS[normalized]
    for word in normalized.split():
        if word in VERB_TENSE_LABELS:
            return VERB_TENSE_LABELS[word]
    return ""


def download_dump(filename: str, dest_dir: Path):
    """Download a kaikki dump if missing."""
    url = DOWNLOAD_URLS.get(filename)
    if not url:
        print(f"  No download URL for {filename}")
        return False

    dest = dest_dir / filename
    if dest.exists():
        print(f"  {filename} already exists")
        return True

    print(f"  Downloading {filename}...")
    import urllib.request
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  Downloaded {size_mb:.0f} MB")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def _add_lookup(lookup: dict, form: str, lemma: str, confidence: int = 1,
                proper_noun: bool = False, closed_class_capital: bool = False,
                capitals_last: bool = False):
    """Add a form to the lookup under original, lowercase, monotonic, and stripped keys.

    Confidence levels (assigned during merge):
        5 = both EN + EL have pages for this form
        4 = EN Wiktionary has a page for this form (no EL page)
        3 = EL Wiktionary has a page for this form (no EN page)
        2 = EN + EL table entries agree on lemma (corroborated)
        1 = single source, table-only (default)

    Higher confidence always wins over lower.

    When proper_noun=True, stripped/lowercase keys that differ from the
    original form get reduced confidence (confidence - 1, min 0). This
    prevents proper noun forms (e.g. Φᾶσιν -> Φᾶσις) from winning over
    common words (e.g. φασίν -> φημί) on accent-stripped keys.

    When capitals_last=True (the Modern Greek dumps), a capitalized
    headword's lowercase keys (Ερωτώ's ερωτώ, ΔΕΣ's δες) get confidence 0:
    they fill a key only when nothing lowercase claims it.

    When closed_class_capital=True (a capitalized closed-class entry
    like the formal MG pronouns Αυτού, Αυτής), the mapping is NOT
    propagated to the lowercase / stripped keys. Otherwise the formal
    pronoun Αυτής (self-map, conf 3) would block the regular pronoun
    resolution αυτής -> αυτός (form-of, conf 1) on the key αυτής. The
    capitalized entry is still accessible via its own exact key. This
    only fires for closed-class POS tags (pron, det, adj) where the
    uppercase / lowercase distinction is semantically meaningful;
    general nouns or adjectives stay with propagation.
    """
    capitalized = form != form.lower()
    for key in (form, form.lower(), to_monotonic(form), to_monotonic(form).lower(),
                strip_accents(form.lower())):
        if not key:
            continue
        if closed_class_capital and key != form:
            continue
        conf = confidence
        if proper_noun and key != form:
            conf = max(0, confidence - 1)
        if capitals_last and capitalized and key != form and key == key.lower():
            # A capitalized headword reaches its lowercase keys only where
            # no lowercase page or table claims them: the female name
            # Ερωτώ must not take ερωτώ from the verb, nor the abbreviation
            # ΔΕΣ, the name Πέις or Λες take δες, πεις or λες from βλέπω
            # and λέω, whose form-of pages have the lowest confidence.
            conf = 0
        existing = lookup.get(key)
        if existing is None or conf > existing[1]:
            lookup[key] = (lemma, conf)


# The Greek Wiktionary's page for a word's polytonic spelling says so in its
# gloss, and kaikki often gives it no alt_of reference (μιλῶ, "πολυτονική
# γραφή του μιλώ").
_POLYTONIC_SPELLING_GLOSS = re.compile(
    r"^πολυτονική γραφή (?:του|της) (\S+?)[.,]?$")


_GREEK_VOWELS = frozenset("αεηιουωΑΕΗΙΟΥΩ")
# The case tags of an inflected form's sense ("dative singular of σοφία").
_CASE_TAGS = frozenset({"nominative", "genitive", "dative", "accusative",
                        "vocative"})


def vowel_groups(word: str) -> int:
    """The runs of vowels in a word, a vowel with a diaeresis starting a run
    of its own: 1 for a monosyllable, synizesis included (γιος, γεια, μια),
    2 for χαϊδεύω's χαϊ."""
    groups = 0
    previous_vowel = False
    for ch in unicodedata.normalize("NFD", word):
        if unicodedata.category(ch) == "Mn":
            if ch == "\u0308" and previous_vowel:
                groups += 1
            continue
        vowel = ch in _GREEK_VOWELS
        if vowel and not previous_vowel:
            groups += 1
        previous_vowel = vowel
    return groups


def spelling_target(lemma: str, senses: list) -> str | None:
    """The word a page only gives another spelling of, or None.

    That word is the page's own word without the marks monotonic writing
    drops: μουστακάτος for μουστακᾶτος, αναφτερουγιάζω for ἀναφτερουγιάζω,
    and for a monosyllable (one run of vowels: γιος, γεια) without its
    accent too, γιος for γιός. Every sense of the page must refer to it, by
    an alt_of or form_of reference or by a gloss that calls the page its
    polytonic spelling. A page that refers to a word of other letters is a
    form of that word (ἐτρώθη of τιτρώσκω), a form_of reference of a sense
    tagged with a case names the word the page is a form of (σοφίᾳ, the
    dative of σοφία), and a page with a sense of its own is a word of its
    own."""
    mono = to_monotonic(lemma)
    spellings = {mono}
    if vowel_groups(mono) == 1:
        spellings.add(strip_accents(mono))
    spellings.discard(lemma)
    target = None
    for sense in senses:
        refs = list(sense.get("alt_of") or [])
        if not _CASE_TAGS.intersection(sense.get("tags") or []):
            refs += sense.get("form_of") or []
        words = [strip_length_marks(ref.get("word", "")) for ref in refs]
        gloss = (sense.get("glosses") or [""])[0]
        named = _POLYTONIC_SPELLING_GLOSS.match(gloss)
        if named:
            words.append(named.group(1))
        found = [w for w in words if w in spellings]
        if not found:
            return None
        target = target or found[0]
    return target


def _add_pos_map(pos_map: dict, form: str, lemma: str, wikt_pos: str):
    """Record a form->lemma mapping under its UPOS tag for POS disambiguation.

    Only forms with mappable POS tags are recorded. The pos_map accumulates
    {form: {upos: lemma}} entries. After all sources are processed, forms
    where all UPOS keys map to the same lemma are filtered out (not ambiguous).
    """
    upos = WIKT_TO_UPOS.get(wikt_pos)
    if not upos:
        return
    for key in (form, form.lower(), to_monotonic(form), to_monotonic(form).lower(),
                strip_accents(form.lower())):
        if not key:
            continue
        if key not in pos_map:
            pos_map[key] = {}
        # First POS mapping wins (same convention as _add_lookup)
        if upos not in pos_map[key]:
            pos_map[key][upos] = lemma


def _add_verb_reference(references: dict, form: str, lemma: str):
    """Keep every explicit verb target separate from inflection-table claims."""
    if references is None:
        return
    for key in (form, form.lower(), to_monotonic(form), to_monotonic(form).lower(),
                strip_accents(form.lower())):
        if key and key != lemma:
            references.setdefault(key, set()).add(lemma)


def _is_participle_sense(sense: dict, has_ordinary_sense: bool) -> bool:
    return ("participle" in sense.get("tags", [])
            and not (has_ordinary_sense and "no-gloss" in sense.get("tags", [])
                     and sense.get("related")
                     and not sense.get("form_of") and not sense.get("alt_of")))


def _is_synonym_sense(sense: dict) -> bool:
    """A synonym label is not an inflection or spelling reference."""
    return (bool({"synonym", "synonym-of"}.intersection(sense.get("tags", [])))
            or any(re.match(r"^(?:συνώνυμ\w*|synonyms? of)\b", gloss.strip().lower())
                   for gloss in sense.get("glosses", [])))


def merge_pos_maps(target: dict, source: dict, prefer_verb_refs=False):
    """Retain a second source's verb reference over a self-map.

    One dictionary can describe a participle as its own verb headword
    (ισχύων), while the other names its verb (ισχύω). Follow that explicit
    reference before the unqualified lookup chooses the noun ισχύς.
    """
    for form, entries in source.items():
        current = target.setdefault(form, {})
        for pos, lemma in entries.items():
            if (pos not in current or
                    (prefer_verb_refs and pos == "VERB"
                     and current[pos] == form and lemma != form)):
                current[pos] = lemma


class Abbreviations(NamedTuple):
    """What a Modern Greek dump says of its abbreviations. An abbreviation is
    no form of the word it abbreviates, and its reference must not move a
    key that a page presents as a word of its own: the English Wiktionary
    gives δις and δ as abbreviations of δεσποινίς, and following them made
    δις ("twice", "billion"; 20,825 monotonic tokens) and the letter δ forms
    of δεσποινίς. A key no lowercase word claims still follows its
    abbreviation (κα, "κυρία", whose Greek Wiktionary page is an abbreviation
    too; μαρ, "Μαρτίου", beside the surname Μαρ), and so does a contraction
    that is no word (στο, "σε + το", which the English Wiktionary tags as an
    abbreviation as well)."""
    keys: set       # keys whose form-of target is an abbreviation's
    words: set      # keys of lowercase words with a sense of their own


def drop_abbreviation_targets(targets: dict, abbreviations: Abbreviations,
                              words: set) -> int:
    """Drop from ``targets`` the abbreviation targets of the keys in
    ``words``, the words of both Wiktionaries (``Abbreviations``), and
    return how many were dropped."""
    dropped = abbreviations.keys & words & set(targets)
    for key in dropped:
        del targets[key]
    return len(dropped)


_FORM_OF_PATTERNS = (
    # EN Wiktionary patterns
    " of ", " singular of ", " plural of ",
    " form of ", " active of ", " passive of ",
    " participle of ", " indicative of ",
    # EL case/number
    " πρόσωπο ", " ενικού ", " πληθυντικού ",
    " γενική ", " αιτιατική ", " ονομαστική ",
    " κλητική ", " δοτική ",
    # EL gender
    " αρσενικού ", " θηλυκού ", " ουδέτερου ",
    # EL tense
    " αορίστου ", " ενεστώτα ", " μέλλοντα ",
    " παρατατικού ", " παρακειμένου ",
    " στιγμιαίου ", " εξαρτημένου ", " συντελεσμένου ",
    # EL mood
    " οριστικής ", " υποτακτικής ", " προστακτικής ",
    # EL voice
    " ενεργητικού ", " παθητικού ",
    " ρήματος ",
    # EL verb/participle forms
    " μετοχή ", " απαρέμφατο ",
    # EL voice descriptions (gloss-initial)
    "παθητική φωνή ", "μεσοπαθητική φωνή ",
    "ενεργητική φωνή ",
    # EL variant/alternative form
    "μορφή του ", "μορφή τού ",
    # EL diminutive/augmentative
    "υποκοριστικό του ", "μεγεθυντικό του ",
)

_NOUN_REFERENCE_TAGS = frozenset({"form-of", "alternative", "misspelling"})


def _is_adjective_derived_form_gloss(gloss: str) -> bool:
    """Degree, contraction, or participle descriptions do not define a head."""
    return bool(re.match(
        r"^(?:(?:comparative|superlative)(?: (?:degree|form))? of\b|"
        r"(?:(?:masculine|feminine|neuter|common|nominative|genitive|dative|accusative|vocative|singular|plural|dual)[ /-]+)+of\b|"
        r"(?:contracted|uncontracted)(?: form)? of\b|"
        r"(?:a )?(?:rare|more formal|less formal|older|old|alternative|alternate) "
        r"(?:form|variant|pronunciation|spelling) of\b|"
        r"(?:polytonic spelling|monotonic (?:script|spelling))[: ]|"
        r"etymological (?:spelling|speeling) of\b|"
        r"(?:βλέπε|δες)\s+\S+|"
        r"(?:(?:με άρθρο,? )?σχετικός υπερθετικός|συγκριτικός|υπερθετικός)"
        r"(?: βαθμός)? (?:του|της|των|με περίφραση)\b|"
        r"(?:(?:παθητική|ενεργητική|μέση|μεσοπαθητική|ενεστώτα|αορίστου|παρακειμένου) )+μετοχή\b|"
        r"(?:συνηρημένος|συνηρημένη|συνηρημένο|ασυναίρετος|ασυναίρετη|ασυναίρετο) "
        r"(?:τύπος|μορφή) (?:του|της)\b)", gloss.strip().lower()))


def _has_adjective_morphology(entry: dict) -> bool:
    """Require adjective morphology beyond a potentially duplicated POS label."""
    if any(template.get("name") == "el-adj"
           or (template.get("name") == "head"
               and isinstance(template.get("args", {}), dict)
               and template.get("args", {}).get("1") == "el"
               and template.get("args", {}).get("2") in {"adjective", "adjectives"})
           for template in entry.get("head_templates") or []
           if isinstance(template, dict)):
        return True
    forms = entry.get("forms") or []
    genders = {tag for form in forms for tag in form.get("tags") or []
               if tag in {"masculine", "feminine", "neuter"}}
    return len(genders) >= 2 or any(
        form.get("source") == "header"
        and {"invariable", "indeclinable"}.intersection(form.get("tags") or [])
        for form in forms)


def is_independent_adjective_entry(entry: dict, source: str) -> bool:
    """Require an adjective definition, rather than a grammatical form label."""
    reference_tags = _NOUN_REFERENCE_TAGS | {"alt-of"}
    if (entry.get("pos") != "adj" or source not in {"en", "el"}
            or not _is_greek(strip_length_marks(entry.get("word", "")))
            or reference_tags.intersection(entry.get("tags") or [])
            or (source != "el" and (entry.get("form_of") or entry.get("alt_of")))
            or any(template.get("name") == "head"
                   and isinstance(template.get("args", {}), dict)
                   and template.get("args", {}).get("1") == "el"
                   and template.get("args", {}).get("2") in {
                       "adjective form", "adjective forms", "adjf", "noun form", "noun forms"}
                   for template in entry.get("head_templates") or []
                   if isinstance(template, dict))
            or not _has_adjective_morphology(entry)):
        return False
    senses = entry.get("senses") or []
    if (not senses or any(s.get("form_of") or s.get("alt_of")
                          or reference_tags.intersection(s.get("tags") or [])
                          for s in senses)):
        return False
    word = strip_length_marks(entry.get("word", ""))
    return any("no-gloss" not in (sense.get("tags") or [])
               and not {"comparative", "superlative"}.intersection(sense.get("tags") or [])
               and not any(_is_adjective_derived_form_gloss(gloss)
                           for gloss in sense.get("glosses") or [])
               and any(gloss.strip() and not _is_closed_class_inflection_gloss(gloss)
                       and strip_accents(gloss.strip().lower()) != strip_accents(word.lower())
                       and not re.fullmatch(
                           r"το (?:αρσενικό|θηλυκό|ουδέτερο) "
                           r"(?:ονομαστική|γενική|δοτική|αιτιατική|κλητική) "
                           r"(?:ενικού|πληθυντικού|δυϊκού) του \S+[.;]?",
                           gloss.strip().lower())
                       for gloss in sense.get("glosses") or [])
               for sense in senses)


def adjective_head_evidence(entry: dict, source: str) -> dict:
    """Keep the source definition and declared head type for adjective proof."""
    return {
        "source": source, "headword": entry["word"], "pos": entry["pos"],
        "tags": sorted(set(entry.get("tags") or [])),
        "form_of": entry.get("form_of") or [], "alt_of": entry.get("alt_of") or [],
        "head_templates": entry.get("head_templates") or [],
        "forms": [{"form": form.get("form", ""), "tags": form.get("tags") or [],
                   "source": form.get("source", "")}
                  for form in entry.get("forms") or []],
        "senses": [{key: sense.get(key) or []
                    for key in ("glosses", "tags", "form_of", "alt_of")}
                   for sense in entry.get("senses") or []],
    }



def noun_table_conflicts_with_form_page(entry: dict) -> bool:
    """A nonlexical noun form page cannot authorize a different nominal table."""
    cases = {"nominative", "genitive", "dative", "accusative", "vocative"}
    numbers = {"singular", "plural", "dual"}

    def cells(tags):
        tags = set(tags or [])
        return {(case, number) for case in tags & cases for number in tags & numbers}

    def grammatical(gloss):
        return _is_closed_class_inflection_gloss(gloss) or bool(re.fullmatch(
            r"το (?:αρσενικό|θηλυκό|ουδέτερο) "
            r"(?:ονομαστική|γενική|δοτική|αιτιατική|κλητική) "
            r"(?:ενικού|πληθυντικού|δυϊκού) του \S+[.;]?",
            gloss.strip().lower()))

    source_cells = set()
    for sense in entry.get("senses") or []:
        # A spelling-variant lemma may legitimately supply its own table.
        if sense.get("alt_of"):
            return False
        if sense.get("form_of"):
            source_cells.update(cells(sense.get("tags")))
        if ("no-gloss" not in (sense.get("tags") or [])
                and any(not grammatical(gloss)
                        for gloss in sense.get("glosses") or [])):
            # Check every gloss independently: one grammatical description
            # does not erase an ordinary meaning in the same sense.
            return False
    if not source_cells:
        return False
    headword = strip_length_marks(entry.get("word", ""))
    table_cells = set()
    has_nominal_table = False
    for row in entry.get("forms") or []:
        row_cells = cells(row.get("tags"))
        has_nominal_table |= bool(row_cells)
        if strip_length_marks(row.get("form", "")) == headword:
            table_cells.update(row_cells)
    return has_nominal_table and not source_cells.intersection(table_cells)

def _numbered_etymology(entry: dict) -> int | None:
    number = str(entry.get("etymology_number", ""))
    return int(number) if number.isascii() and number.isdecimal() and int(number) > 0 else None


def extract_pairs(jsonl_path: Path, lang: str,
                  skip_name_plurals: set = None, lemma_pos: dict = None,
                  participle_refs: dict = None, verb_heads: set = None,
                  participle_heads: set = None, verb_refs: dict = None,
                  participle_verb_refs: dict = None,
                  participle_cells: list = None, noun_heads: list = None,
                  closed_class_heads: dict = None, closed_head_records: list = None,
                  lexical_heads: dict = None,
                  typed_references: dict = None,
                  canonical_references: dict = None,
                  closed_class_form_refs: dict = None,
                  adjective_heads: set = None, adjective_head_records: list = None,
                  ) -> tuple[list[dict], dict, set, dict, dict, Abbreviations]:
    """Extract form->lemma pairs from a kaikki JSONL dump.

    Extracts from three sources per entry:
      1. forms[] array (inflection table cells)
      2. form_of[] (this entry is a form of another headword)
      3. alt_of[] (this entry is an alternative form of another headword)

    Also extracts dialect tags from table-tags headers (e.g. "Epic declension-1")
    and propagates them to forms that follow.

    Args:
        skip_name_plurals: if provided, skip plural forms of proper nouns unless
            the form is in this set (filters EL template-generated garbage while
            keeping EN-corroborated plurals).

    Returns:
        pairs: list of {form, lemma, pos, tags} dicts (for training)
        lookup: {form: (lemma, confidence)} dict
        headwords: set of headwords that have their own page in this dump
        pos_map: {form: {upos: lemma}} dict for POS disambiguation
        form_of_targets: {key_variant: target_lemma} for cross-source
            self-map resolution after lookups are merged
        abbreviations: the keys whose target is an abbreviation's and the
            keys of words of their own (``Abbreviations``), for the
            Modern Greek dumps; empty for the others
    """
    if not jsonl_path.exists():
        print(f"  {lang}: not found at {jsonl_path}")
        return [], {}, set(), {}, {}, Abbreviations(set(), set())

    pairs = []
    page_headwords = set()
    lookup = {}
    modern_greek = lang.split("-", 1)[0] == "el"
    # Modern Greek pages in sentence case (names, abbreviations, headings)
    # must not take the lowercase keys of common words (_add_lookup).
    capitals_last = modern_greek

    def _add_lookup_here(*args, **kwargs):
        return _add_lookup(*args, capitals_last=capitals_last, **kwargs)
    # form_of/alt_of targets: {key_variant: target_lemma}
    # Used in pass 2 to resolve self-map artifacts.
    form_of_targets = {}
    # The keys whose target is an abbreviation's, and the keys of words
    # with a sense of their own (``Abbreviations``; Modern Greek only).
    abbreviated_keys: set[str] = set()
    word_keys: set[str] = set()
    pos_map = {}  # {form: {upos: lemma}} for POS disambiguation
    skip_tags = {"romanization", "table-tags", "inflection-template", "class"}
    # Articles dump all genders/persons into each headword.
    # Pronouns have cross-contamination (εσύ lists εγώ as a "form").
    # NOTE: "det" used to be in skip_all_forms_pos, but that also
    # catches demonstratives (οὗτος, ὅδε, ἐκεῖνος) which have
    # legitimate paradigms. Instead, we target article headwords
    # specifically in the skip logic below.
    skip_all_forms_pos = {"article", "phrase"}
    _article_headwords = {
        "ὁ", "ἡ", "τό", "ο", "η", "το", "the",
    }
    filter_cross_forms_pos = {"pron"}

    # Personal-pronoun templates that mix first-/second-/third-person forms
    # into one shared table. When the headword is any personal pronoun
    # (εγώ, εσύ, αυτός, τα, ...), the same table gets dumped on its entry,
    # so for example αυτό appears as a "form" of τα, and εσύ appears as
    # a form of αυτός. Extracting these as form -> headword mappings
    # is pure contamination. Skip the forms[] table for these entries
    # (headword still self-maps).
    # Detected by the inflection-template identifier kaikki emits.
    _SHARED_PERSON_TEMPLATES = {
        # EN Wiktionary Modern Greek personal-pronoun grid (τα pron)
        "g",
        # EN Wiktionary MG weak/strong personal-pronoun grid (του, αυτός, αμφότεροι pron)
        "l-self",
        # EL Wiktionary Modern Greek personal-pronoun grid
        "προσωπική αντωνυμία",
        "el-προσωπική αντωνυμία",
        # EL Wiktionary Ancient Greek personal-pronoun grid
        "grc-προσωπική αντωνυμία",
        # EN Wiktionary Ancient Greek personal-pronoun grid
        "grc-decl",
    }

    # Modern Greek personal-pronoun headword (accent-stripped) -> grammatical
    # person. The shared MG conjugation table dumps all three persons onto each
    # headword, so εγώ's forms[] lists the third-person αυτός forms (αυτό,
    # αυτόν, αυτά, ...) tagged 'third-person'. Without filtering, those map
    # αυτό -> εγώ. Keep only forms whose person tag matches the headword.
    _PRON_PERSON = {
        "εγω": "first-person", "εμεις": "first-person",
        "εσυ": "second-person", "εσεις": "second-person",
        "αυτος": "third-person", "αυτη": "third-person", "αυτο": "third-person",
        "αυτοι": "third-person", "αυτες": "third-person", "αυτα": "third-person",
    }
    _PERSON_TAGS = {"first-person", "second-person", "third-person"}
    # The Modern Greek weak personal pronouns. The Greek Wiktionary's μόνος
    # shows those it takes on its headword line (μόνος μου, σου, μας, σας),
    # which kaikki gives as untagged forms.
    _WEAK_PRONOUNS = {"μου", "σου", "του", "της", "μας", "σας", "τους"}

    # First pass: collect closed-class headwords and article forms.
    # Article forms leak into Katharevousa noun/adj declension templates
    # (e.g. γωνιακός lists τῶν as its genitive plural "form").
    closed_class_headwords = set()
    article_forms = set()
    # An alternative spelling in one numbered etymology must not rename
    # another etymology's independently defined word and inflection table.
    lexical_etymologies = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            pos = entry.get("pos", "")
            etymology = _numbered_etymology(entry)
            if (lang.startswith("grc-") and etymology is not None
                    and is_independent_lexical_entry(entry, lang.split("-", 1)[1])):
                head = strip_length_marks(entry.get("word", ""))
                lexical_etymologies.setdefault((head, pos), set()).add(etymology)
            if pos in skip_all_forms_pos | filter_cross_forms_pos:
                hw = strip_length_marks(entry.get("word", ""))
                if hw:
                    closed_class_headwords.add(hw)
                    closed_class_headwords.add(hw.lower())
            if pos in skip_all_forms_pos:
                # Collect all forms from article/det entries, including
                # polytonic variants that Katharevousa templates use
                for fe in entry.get("forms", []):
                    form = strip_length_marks(fe.get("form", ""))
                    if form and _is_greek(form):
                        article_forms.add(form)
                        article_forms.add(form.lower())
                        article_forms.add(to_monotonic(form))
                        article_forms.add(to_monotonic(form).lower())
                        article_forms.add(strip_accents(form.lower()))

    # Add all known article forms (monotonic, polytonic, accented, stripped).
    # MG Wiktionary article entries only have unaccented monotonic forms
    # but Katharevousa templates use accented/polytonic variants.
    _all_article_forms = {
        # Monotonic (unaccented as in Wiktionary)
        "ο", "η", "το", "τον", "την", "του", "της", "τους", "τις", "τα",
        "των", "οι", "τη",
        # Monotonic (accented, as in Katharevousa templates)
        "τόν", "τήν", "τού", "τής", "τούς", "τό", "τά", "τών", "τώ",
        "τή", "τοί", "ταί",
        # Polytonic
        "ὁ", "ἡ", "τό", "τόν", "τήν", "τοῦ", "τῆς", "τούς", "τάς",
        "τά", "τῶν", "τῷ", "τῇ", "τοῖς", "ταῖς", "τοῖν", "ταῖν",
        "αἱ", "οἱ", "τώ",
    }
    article_forms.update(_all_article_forms)
    for f in _all_article_forms:
        article_forms.add(to_monotonic(f))
        article_forms.add(to_monotonic(f).lower())
        article_forms.add(strip_accents(f.lower()))

    scanned = 0
    entries_with_data = 0
    form_of_count = 0
    alt_of_count = 0
    dialect_tagged = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            scanned += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            word = entry.get("word", "")
            pos = entry.get("pos", "")
            if not word:
                continue

            lemma = strip_length_marks(word)
            if not _is_greek(lemma.replace(" ", "")):
                continue

            # Skip multi-word entries (phrases) from lookup
            if " " in lemma:
                continue

            if lemma_pos is not None:
                lemma_pos.setdefault(lemma, set()).add(pos)

            page_headwords.add(lemma)
            page_headwords.add(lemma.lower())

            # EL Wiktionary uses entry-level top "form_of"/"alt_of" as an
            # etymology / register / spelling-variant pointer template
            # (kaikki maps the {{form of}} / {{ετυμολογία}} template into
            # this field), NOT as an inflection-form-of relationship.
            # Real inflection-form-of refs in EL show up under
            # senses[].form_of with a matching gloss like "ονομαστική
            # πληθυντικού του X". A survey of the current dumps shows
            # 31,795 EL entries with top-level form_of where the gloss
            # is a substantive lemma definition (etymology), vs only ~14
            # where the gloss is genuinely inflectional - and even those
            # are spelling-variant pointers ("άλλη μορφή του Y") rather
            # than canonical case/number forms.
            #
            # Treating the top-level field as inflectional caused a
            # regression on Demotic Modern Greek headwords whose EL
            # Wiktionary page lists the Katharevousa/Ancient ancestor
            # via this template: e.g. ποτήρι has top form_of=[ποτήριον,
            # αρχαιοπρεπές] (literally "archaicizing"), άντρας has
            # form_of=[άνδρας, ανήρ, αρχαιοπρεπές], σεντόνι has
            # form_of=[σινδών, αρχαιοπρεπές]. These are real Demotic
            # headwords with full noun inflection tables; the ancestor
            # word should not surface as the lemma for their inflected
            # plural / oblique forms.
            #
            # EN Wiktionary never uses entry-level top form_of (0 entries
            # across both Greek and Ancient Greek dumps), so this only
            # filters EL data.
            wikt_source = lang.split("-", 1)[1] if "-" in lang else lang
            if wikt_source == "el":
                top_form_of_for_detection = []
                top_alt_of_for_detection = []
            else:
                top_form_of_for_detection = list(entry.get("form_of") or [])
                top_alt_of_for_detection = list(entry.get("alt_of") or [])

            # Headword self-mapping. True lemma entries (their own
            # dictionary definition) get confidence 3 so they can't be
            # overridden by another entry's inflection table. Form-of
            # pages (e.g. θεοί saying "plural of θεός") get confidence 1;
            # these are resolved in pass 2 using form_of/alt_of targets.
            sense_has_form_of = any(
                s.get("form_of") or s.get("alt_of")
                for s in entry.get("senses") or [])
            if lexical_heads is not None and is_independent_lexical_entry(entry, wikt_source):
                lexical_heads.setdefault(pos, set()).add(lemma)
            is_form_of_page = bool(top_form_of_for_detection
                                   or top_alt_of_for_detection
                                   or sense_has_form_of)
            if not is_form_of_page:
                # Check gloss for "X of Y" pattern (kaikki often doesn't
                # populate form_of for AG form-of pages)
                senses = entry.get("senses", [])
                if senses:
                    gloss = (senses[0].get("glosses") or [""])[0].lower()
                    _form_of_patterns = _FORM_OF_PATTERNS
                    if any(p in gloss for p in _form_of_patterns):
                        is_form_of_page = True
            if (verb_heads is not None and pos == "verb" and not sense_has_form_of
                    and not top_form_of_for_detection and not top_alt_of_for_detection
                    and "form-of" not in (entry.get("tags") or [])
                    and not any("form-of" in (s.get("tags") or [])
                                for s in entry.get("senses") or [])
                    and is_independent_finite_entry(entry, wikt_source)):
                # A verb-tagged inflection/alternate page is not an
                # independent lexical verb parent for another participle.
                verb_heads.add(lemma)
            hw_confidence = 1 if is_form_of_page else 3
            # A page that only gives another spelling of a word (γιός, the
            # Greek Wiktionary's "πολυτονική γραφή του γιος") is no lemma:
            # its monotonic and accentless keys are that word's own, and a
            # self-map would take them from it. γιος lost its whole
            # paradigm so, since its own page, whose gloss names a male
            # ("αρσενικού φύλου"), is read as a form-of page and could not
            # take the key back. Its keys go to the word with its alt_of
            # reference below. In the Ancient Greek dumps a page whose
            # monotonic spelling is the word it names is a form of it, not
            # a spelling (the dative φωνῇ of φωνή), and is read as before.
            spelling_of = (spelling_target(lemma, entry.get("senses") or [])
                           if modern_greek else None)
            # A lexical noun's own table does not belong to an inflectional
            # homograph of its headword. Keep that ownership through pair
            # cleanup even when the flat lookup chooses the other reading.
            noun_owner = (modern_greek and pos == "noun"
                          and not is_form_of_page and spelling_of is None
                          and not _NOUN_REFERENCE_TAGS.intersection(entry.get("tags") or [])
                          and not any(_NOUN_REFERENCE_TAGS.intersection(s.get("tags") or [])
                                      for s in entry.get("senses", []))
                          and any(s.get("glosses")
                                  and "no-gloss" not in s.get("tags", [])
                                  for s in entry.get("senses", [])))
            adjective_owner = (modern_greek
                               and is_independent_adjective_entry(entry, wikt_source)
                               and spelling_of is None)
            if adjective_owner:
                if adjective_heads is not None:
                    adjective_heads.add(lemma)
                if adjective_head_records is not None:
                    adjective_head_records.append({
                        "lemma": lemma, **adjective_head_evidence(entry, wikt_source)})
            if (modern_greek and pos in {"det", "pron"}
                    and is_independent_closed_class_entry(entry, wikt_source)):
                if closed_class_heads is not None:
                    closed_class_heads.setdefault(pos, set()).add(lemma)
                if closed_head_records is not None:
                    closed_head_records.append({"lemma": lemma,
                                                **closed_class_head_evidence(entry, wikt_source)})
            if noun_owner and noun_heads is not None:
                noun_heads.append({
                    "lemma": lemma, "source": wikt_source,
                    "headword": entry["word"], "pos": pos,
                    "tags": sorted(set(entry.get("tags") or [])),
                    "form_of": entry.get("form_of") or [],
                    "alt_of": entry.get("alt_of") or [],
                    "senses": [
                        {key: sense.get(key) or []
                         for key in ("glosses", "tags", "form_of", "alt_of")}
                        for sense in entry.get("senses") or []
                    ],
                })
            # Closed-class capitals like Αυτής (formal pronoun) must not
            # hijack the lowercase αυτής key from regular αυτής -> αυτός.
            _cc_cap = (pos in ("pron", "det", "adj")
                       and lemma and lemma != lemma.lower())
            if spelling_of is None:
                _add_lookup_here(lookup, lemma, lemma,
                                 confidence=hw_confidence,
                                 proper_noun=(pos == "name"),
                                 closed_class_capital=_cc_cap)
            # The keys of a lowercase word with a sense of its own (the
            # adverb δις, "twice"; the letter δ), which an abbreviation's
            # reference does not move (``Abbreviations``). An abbreviation
            # is no such word: the Greek Wiktionary's pages of part of
            # speech "abbrev" (κα, "κυρία") and the senses the English one
            # tags "abbreviation". Nor is a capitalized page, which comes
            # last for a key as in _add_lookup: the surname Μαρ leaves μαρ
            # to Μάρτιος, whose abbreviation it is.
            if (modern_greek and hw_confidence == 3 and spelling_of is None
                    and lemma == lemma.lower() and pos != "abbrev"
                    and not any("abbreviation" in (s.get("tags") or [])
                                for s in entry.get("senses") or [])):
                for key in (lemma, to_monotonic(lemma),
                            strip_accents(lemma)):
                    if key:
                        word_keys.add(key)

            has_data = False
            ref_targets = []

            # --- Source 1: form_of (this page is a form of another word) ---
            # Check both top-level form_of AND senses[].form_of.
            # AG entries in EN kaikki store form-of refs in senses,
            # not at the top level.
            #
            # For EL Wiktionary, top-level form_of is an etymology /
            # register / variant pointer (see is_form_of_page comment
            # above), NOT inflection. We skip it entirely here so that
            # real Demotic headwords like ποτήρι, άντρας, σεντόνι don't
            # get rewired to their Katharevousa ancestor.
            #
            # Ordering for non-EL sources: sense-level refs come first.
            # EL Wiktionary occasionally lists spurious double-gen
            # variants at the top level (αυτών -> αυτωνών) while the
            # correct parent lemma shows up at the sense level. Sense
            # refs are consistently better-curated, so we use them as
            # the primary source for pass-2 form_of_targets.
            sense_form_of = []
            form_of_rows = []
            named_verb_roots = set()
            entry_participle = "participle" in entry.get("tags", [])
            has_ordinary_sense = any(s.get("glosses") and "participle" not in s.get("tags", [])
                                     for s in entry.get("senses", []))
            is_participle = (modern_greek and pos == "verb"
                            and (entry_participle or any(
                                _is_participle_sense(s, has_ordinary_sense)
                                for s in entry.get("senses", []))))
            if is_participle:
                # These references are checked only after both dictionaries'
                # headword POS evidence is collected. Ordinary noun/adj
                # etymology pointers remain excluded.
                if participle_heads is not None:
                    participle_heads.add(lemma)
                if participle_cells is not None:
                    head_tags = [set(entry.get("tags", []))]
                    if not entry_participle:
                        head_tags = [set(entry.get("tags", [])) | set(s.get("tags", []))
                                     for s in entry.get("senses", [])
                                     if _is_participle_sense(s, has_ordinary_sense)]
                    for tags in head_tags:
                        if tags & _PARTICIPLE_FINITE_TAGS:
                            continue
                        participle_cells.append({
                            "form": lemma, "lemma": lemma, "pos": pos,
                            "kind": "headword", "tags": sorted(tags | {"participle"}),
                            "source_form": word, "source_lemma": word,
                            "source": lang.split("-", 1)[-1],
                        })
                for ref in entry.get("form_of", []):
                    target = strip_length_marks(ref.get("word", ""))
                    if participle_refs is not None and target and _is_greek(target):
                        participle_refs.setdefault(lemma, set()).add(target)
            for sense in entry.get("senses", []):
                sense_form_of.extend(sense.get("form_of", []))
                if closed_class_form_refs is not None:
                    if modern_greek and pos in {"det", "pron"}:
                        # Own sense references precede a weak form's table
                        # alias (εσένα belongs to εσύ, not to σε).
                        for ref in sense.get("form_of") or []:
                            target = strip_length_marks(ref.get("word", ""))
                            if target and _is_greek(target):
                                closed_class_form_refs.setdefault((lemma, pos), set()).add(target)
                    named = _named_weak_pronoun_reference(entry, sense, wikt_source)
                    if named:
                        closed_class_form_refs.setdefault((lemma, pos), set()).add(named)
                sense_verb = None
                # A malformed head page may have the wrong POS even when
                # its inflection pages explicitly identify it as a verb:
                # δηώ is labeled romanization, but δήωσα says "ρήματος δηώ".
                # That positive evidence preserves the verb; a nominal
                # case reference alone does not license a verb lemma.
                if modern_greek and pos == "verb" and sense.get("form_of"):
                    explicit_refs = {
                        to_monotonic(strip_length_marks(ref.get("word", "")))
                        for ref in sense["form_of"]
                    }
                    named_verbs = {
                        to_monotonic(word)
                        for gloss in sense.get("glosses", [])
                        for word in re.findall(r"του\s+ρήματος\s+(\w+)", gloss.lower())
                        if _is_greek(word)
                    }
                    named_verb_roots.update(named_verbs)
                    sense_verb = next(iter(named_verbs)) if len(named_verbs) == 1 else None
                    for ref in sense["form_of"]:
                        target = strip_length_marks(ref.get("word", ""))
                        if target and _is_greek(target):
                            _add_verb_reference(verb_refs, lemma, sense_verb or target)
                            if (entry_participle
                                    or _is_participle_sense(sense, has_ordinary_sense)):
                                _add_verb_reference(participle_verb_refs, lemma,
                                                    sense_verb or target)
                    if lemma_pos is not None:
                        # Only an explicit reference licenses correcting a
                        # head's POS. A parenthetical verb in a case-form
                        # gloss can still resolve through its existing POS.
                        for target in named_verbs & explicit_refs:
                            lemma_pos.setdefault(target, set()).add("verb")
                form_of_rows.extend((ref, set(entry.get("tags", [])) | set(sense.get("tags", [])),
                                     sense_verb) for ref in sense.get("form_of", []))
            verb_target = next(iter(named_verb_roots)) if len(named_verb_roots) == 1 else None
            form_of_refs = sense_form_of + list(top_form_of_for_detection)
            form_of_rows.extend((ref, set(entry.get("tags", [])), verb_target)
                                for ref in top_form_of_for_detection)

            for ref, source_tags, source_verb in form_of_rows:
                ref_word = strip_length_marks(ref.get("word", ""))
                if not ref_word or not _is_greek(ref_word.replace(" ", "")):
                    continue
                if " " in ref_word:  # skip multi-word
                    continue
                # This entry (lemma) is a form of ref_word
                _add_lookup_here(lookup, lemma, ref_word, proper_noun=(pos == "name"))
                # Record for pass-2 self-map resolution (skip proper nouns
                # since variant names like Βησσαρίων should self-map). A
                # spelling page records no target for its word's own key:
                # γιός's accentless key is γιος, and μιά's is μια, whose own
                # page must record that it is a form of ένας.
                if pos != "name" and lemma != ref_word:
                    for key in (lemma, lemma.lower(),
                                to_monotonic(lemma), to_monotonic(lemma).lower(),
                                strip_accents(lemma.lower())):
                        if key and typed_references is not None:
                            typed_references.setdefault(key, {}).setdefault(pos, set()).add(ref_word)
                        if (key and key not in form_of_targets
                                and not key == ref_word == spelling_of):
                            form_of_targets[key] = ref_word
                _add_pos_map(pos_map, lemma, verb_target or ref_word, pos)
                ref_pair = {
                    "form": lemma,
                    "lemma": ref_word,
                    "pos": pos,
                    "tags": (sorted(source_tags | {"form-of"})
                             if modern_greek else ["form-of"]),
                }
                if source_verb:
                    ref_pair["via"] = source_verb
                pairs.append(ref_pair)
                form_of_count += 1
                ref_targets.append(ref_word)
                has_data = True

            # --- Source 2: alt_of (alternative form of another word) ---
            # For EL Wiktionary, top-level alt_of has the same etymology
            # / variant-pointer semantics as top-level form_of (see above
            # comment). Use sense-level alt_of refs as the primary signal
            # and the (already filtered) top_alt_of_for_detection list
            # for non-EL sources.
            sense_alt_of = []
            alt_of_rows = []
            # The words that only senses tagged "abbreviation" name, in the
            # Modern Greek dumps (the English Wiktionary's δις and δ,
            # "abbreviation of δεσποινίς"; ``Abbreviations``).
            abbreviation_of: set[str] = set()
            named_otherwise: set[str] = set()
            for sense in entry.get("senses", []):
                sense_alt_of.extend(sense.get("alt_of", []))
                alt_of_rows.extend((ref, set(entry.get("tags", [])) | set(sense.get("tags", [])))
                                   for ref in sense.get("alt_of", []))
                if modern_greek and pos == "verb" and not _is_synonym_sense(sense):
                    for ref in sense.get("alt_of", []):
                        target = strip_length_marks(ref.get("word", ""))
                        if target and _is_greek(target):
                            _add_verb_reference(verb_refs, lemma, target)
                            if (entry_participle
                                    or _is_participle_sense(sense, has_ordinary_sense)):
                                _add_verb_reference(participle_verb_refs, lemma, target)
                named = {strip_length_marks(ref.get("word", ""))
                         for ref in sense.get("alt_of") or []}
                if "abbreviation" in (sense.get("tags") or []):
                    abbreviation_of |= named
                else:
                    named_otherwise |= named
            if not modern_greek:
                abbreviation_of.clear()
            abbreviation_of -= named_otherwise
            # The other spellings a page of one abbreviation lists as
            # "alternative" abbreviate the same word, and are forms of that
            # word rather than of the page's headword: the English
            # Wiktionary's δ lists δις, both "abbreviation of δεσποινίς".
            # Filed under δ, the letter that keeps its key, δις would be a
            # noun of lemma δ.
            abbreviated_word = None
            if (len(abbreviation_of) == 1 and not named_otherwise
                    and all("abbreviation" in (s.get("tags") or [])
                            for s in entry.get("senses") or [])):
                abbreviated_word = next(iter(abbreviation_of))
                if (" " in abbreviated_word
                        or not _is_greek(abbreviated_word)):
                    abbreviated_word = None
            alt_of_refs = sense_alt_of + list(top_alt_of_for_detection)
            alt_of_rows.extend((ref, set(entry.get("tags", [])))
                               for ref in top_alt_of_for_detection)
            # A polytonic spelling page known only by its gloss (μιλῶ) is
            # an alternative form of the word it names.
            if spelling_of and spelling_of not in {
                    strip_length_marks(ref.get("word", ""))
                    for ref in form_of_refs + alt_of_refs}:
                alt_of_refs.append({"word": spelling_of})
                alt_of_rows.append(({"word": spelling_of}, set(entry.get("tags", []))))
                if modern_greek and pos == "verb":
                    _add_verb_reference(verb_refs, lemma, spelling_of)
                    if is_participle:
                        _add_verb_reference(participle_verb_refs, lemma, spelling_of)

            for ref, source_tags in alt_of_rows:
                ref_word = strip_length_marks(ref.get("word", ""))
                if not ref_word or not _is_greek(ref_word.replace(" ", "")):
                    continue
                if " " in ref_word:
                    continue
                _add_lookup_here(lookup, lemma, ref_word, proper_noun=(pos == "name"))
                own_etymologies = lexical_etymologies.get((lemma, pos), set())
                etymology = _numbered_etymology(entry)
                separate_etymology = (bool(own_etymologies) and etymology is not None
                                      and etymology not in own_etymologies)
                if pos != "name" and lemma != ref_word:
                    for key in (lemma, lemma.lower(),
                                to_monotonic(lemma), to_monotonic(lemma).lower(),
                                strip_accents(lemma.lower())):
                        if key and typed_references is not None:
                            typed_references.setdefault(key, {}).setdefault(pos, set()).add(ref_word)
                        if key and canonical_references is not None and not separate_etymology:
                            canonical_references.setdefault(key, {}).setdefault(pos, set()).add(ref_word)
                        if (key and key not in form_of_targets
                                and not key == ref_word == spelling_of
                                and not separate_etymology):
                            form_of_targets[key] = ref_word
                            if ref_word in abbreviation_of:
                                abbreviated_keys.add(key)
                _add_pos_map(pos_map, lemma, ref_word, pos)
                pairs.append({
                    "form": lemma,
                    "lemma": ref_word,
                    "pos": pos,
                    "tags": (sorted(source_tags | {"alt-of"})
                             if modern_greek else ["alt-of"]),
                })
                alt_of_count += 1
                ref_targets.append(ref_word)
                has_data = True

            # --- Source 3: forms[] (inflection table) ---
            forms = entry.get("forms", [])
            if not forms:
                if has_data:
                    entries_with_data += 1
                continue

            if (modern_greek and pos == "noun"
                    and noun_table_conflicts_with_form_page(entry)):
                # Its exact form reference was already retained above. A
                # different same-word noun or adjective owns this table.
                if has_data:
                    entries_with_data += 1
                continue

            entries_with_data += 1

            # For articles: add forms as self-mappings (so they block
            # contamination from Katharevousa templates) but don't
            # create form->lemma training pairs. Skip article headwords
            # specifically, not all det (which includes demonstratives).
            if pos in skip_all_forms_pos or (
                pos == "det"
                and strip_accents(lemma.lower()) in _article_headwords
            ):
                for fe in entry.get("forms", []):
                    form = strip_length_marks(fe.get("form", ""))
                    if form and _is_greek(form) and " " not in form:
                        _add_lookup_here(lookup, form, form, confidence=3)
                continue

            # Personal-pronoun template detection. If a pron entry uses
            # a shared-person template (see _SHARED_PERSON_TEMPLATES above),
            # skip its forms[] table entirely so that cross-person forms
            # like αυτό don't map to unrelated headwords like τα or εσύ.
            # The headword self-map from pass 1 above is still in place.
            if pos == "pron":
                uses_shared_person_template = False
                for fe in forms:
                    if "inflection-template" in fe.get("tags", []):
                        tmpl = fe.get("form", "")
                        if tmpl in _SHARED_PERSON_TEMPLATES:
                            uses_shared_person_template = True
                        break
                # The grid is also told by its persons, since the dumps
                # rename its template (EN's τα came as the generic
                # inflection-table-top in the 2026-09 dump): a pronoun's
                # own table has one person.
                persons = {t for fe in forms for t in fe.get("tags", [])
                           if t in _PERSON_TAGS}
                if uses_shared_person_template or len(persons) > 1:
                    continue

            # Track current dialect from table-tags headers.
            # page_dialect is the entry-level dialect from sense tags
            # (e.g. νοῦμμος is a "Doric spelling of νόμος"), and it
            # initializes current_dialect so the entry's own forms get
            # tagged with the page's dialect even when the table-tags
            # header is missing or says something generic like
            # "Attic declension-2" (grc-decl's default).
            page_dialect = ""
            for sense in entry.get("senses", []):
                for tag in sense.get("tags") or []:
                    if tag in DIALECT_PREFIXES:
                        page_dialect = tag
                        break
                if page_dialect:
                    break
            current_dialect = page_dialect
            current_tense = ""
            stemless_passive_tables = (
                stemless_passive_aorist_dialects(entry)
                if lang.startswith("grc-") and pos == "verb" else frozenset())
            stemless_nonpast_tables = (
                stemless_aorist_nonpast_voices(entry)
                if lang.startswith("grc-") and pos == "verb" else {})

            for f_entry in forms:
                tags = f_entry.get("tags", [])
                form_text = f_entry.get("form", "")

                # Drop cross-person forms from a personal-pronoun table (εγώ's
                # shared conjugation lists the third-person αυτός forms, which
                # would mislemmatize αυτό -> εγώ).
                if pos == "pron":
                    hw_person = _PRON_PERSON.get(strip_accents(lemma.lower()))
                    if hw_person:
                        fp = _PERSON_TAGS.intersection(tags)
                        if fp and hw_person not in fp:
                            continue
                    # A weak pronoun on another pronoun's headword line is no
                    # form of it. As the first claim on its key it kept μας
                    # and σας from their own pages, forms of εγώ and εσύ, and
                    # made them forms of μόνος.
                    elif (modern_greek and not tags and form_text != lemma
                          and form_text in _WEAK_PRONOUNS):
                        continue

                # Update dialect context from table-tags. If a page-level
                # dialect is set (from the entry's own sense tags), it
                # wins over the table-tags header. Page dialect is a
                # semantic claim about the entry (e.g. νοῦμμος is tagged
                # Doric in its sense list), while table-tags only names
                # the inflection template (e.g. "Attic declension-2" for
                # grc-decl - a template name, not a content claim).
                # Without this, Doric alt-forms get tagged 'Attic' and
                # then leak into the parent lemma's Attic paradigm via
                # the chain-break step.
                if "table-tags" in tags:
                    parsed = _parse_dialect(form_text)
                    current_dialect = page_dialect or parsed
                    current_tense = (
                        _parse_verb_tense(form_text)
                        if pos == "verb" else ""
                    )
                    continue

                if any(t in skip_tags for t in tags):
                    continue

                if (lang.startswith("grc-")
                        and is_abbreviated_nominal_header(entry, f_entry)):
                    continue

                form = strip_length_marks(form_text)
                if not form or not any(c.isalpha() for c in form):
                    continue
                if " " in form:
                    continue

                # Handle parenthetical optional suffixes: ἐστί(ν) -> ἐστί + ἐστίν
                extra_form = None
                paren = re.match(r'^(.+?)\((.+?)\)$', form)
                if paren:
                    base, suffix = paren.groups()
                    extra_form = base + suffix  # full form with suffix
                    form = base                 # base form without suffix

                if not _is_greek(form):
                    continue

                # Pronoun cross-contamination: confidence handles this.
                # Forms that are headwords of other entries self-map at
                # confidence 3, so table forms at confidence 1 can't
                # override them. E.g., εσύ self-maps at conf 3, so
                # ἐγώ's table trying to add εσύ→ἐγώ at conf 1 loses.
                # But μοι→ἐγώ works because μοι's self-mapping is also
                # conf 3 (a tie), so first-wins applies and the self-map
                # from μοι's own page wins. This is correct for alignment
                # (self-mapping), and resolve_articles handles eval.

                # Article form leaking into noun/adj Katharevousa templates.
                # Check all variants (original, monotonic, stripped) since
                # _add_lookup creates keys for all of them.
                if form != lemma:
                    form_variants = {form, form.lower(), to_monotonic(form),
                                     to_monotonic(form).lower(), strip_accents(form.lower())}
                    if form_variants & article_forms:
                        continue

                # Proper noun plural filter
                if pos == "name" and "plural" in tags and skip_name_plurals is not None:
                    if form not in skip_name_plurals:
                        continue

                morph_tags = [t for t in tags if t not in ("canonical",)]
                # Another abbreviation of the word a page abbreviates is a
                # form of that word (``abbreviated_word``).
                form_lemma = (abbreviated_word
                              if abbreviated_word and "alternative" in tags
                              else lemma)

                # Add dialect tag if we're inside a dialect-specific table section
                if current_dialect and current_dialect not in morph_tags:
                    morph_tags.append(current_dialect)
                    dialect_tagged += 1

                if (pos == "verb" and current_tense
                        and current_tense not in morph_tags
                        and any(tag in _VERB_AXES for tag in tags)):
                    morph_tags.append(current_tense)

                # An isolated εἰμί auxiliary in a periphrastic table is not
                # a form of the table's main verb. Filter before producing
                # training pairs, their monotonic copies, or lookup claims.
                if (lang.startswith("grc-") and pos == "verb"
                        and is_periphrastic_auxiliary(form, form_lemma, morph_tags)):
                    continue
                # Empty passive stems yield endings without a word. Retain any
                # separately identified voice reading before lookup/training.
                if lang.startswith("grc-") and pos == "verb":
                    morph_tags = retain_complete_passive_aorist_readings(
                        form, morph_tags,
                        stemless_table=current_dialect in stemless_passive_tables)
                    if morph_tags is None:
                        continue
                    morph_tags = retain_complete_aorist_nonpast_readings(
                        form, morph_tags,
                        stemless_voices=stemless_nonpast_tables.get(current_dialect, frozenset()))
                    if morph_tags is None:
                        continue

                pair = {
                    "form": form,
                    "lemma": form_lemma,
                    "pos": pos,
                    "tags": morph_tags,
                }
                if noun_owner:
                    pair["noun_owner"] = True
                if adjective_owner:
                    pair["adjective_owner"] = True
                if (is_participle and participle_cells is not None
                        and _is_nominal_participle_cell(morph_tags)):
                    participle_cells.append({
                        "form": form, "lemma": lemma, "pos": pos,
                        "kind": "inflection",
                        "tags": sorted(set(morph_tags) | {"participle"}),
                        "source_form": form_text, "source_lemma": word,
                        "source": lang.split("-", 1)[-1],
                    })
                    if extra_form and _is_greek(extra_form):
                        participle_cells.append({
                            **participle_cells[-1], "form": extra_form,
                        })
                # A form page's head line belongs to the referenced word,
                # even when a noun homograph wins the page's lookup key.
                # Inflected participle tables belong to that verb too.
                if ((modern_greek or lexical_heads is not None) and pos != "name"
                        and (ref_targets or verb_target)
                        and ((form == lemma and set(tags) <= _GENDER_TAGS)
                             or pos == "verb" or lexical_heads is not None)):
                    pair["via"] = verb_target or ref_targets[0]
                pairs.append(pair)

                # Also add monotonic version as a training pair
                mono = to_monotonic(form)
                if mono != form:
                    pairs.append({**pair, "form": mono})

                # Lookup: original, lowercase, monotonic, accent-stripped
                _form_cc_cap = (pos in ("pron", "det", "adj")
                                and form and form != form.lower())
                _add_lookup_here(lookup, form, form_lemma, proper_noun=(pos == "name"),
                            closed_class_capital=_form_cc_cap)
                _add_pos_map(pos_map, form, pair.get("via", form_lemma), pos)

                # Also add the parenthetical-expanded form (e.g. ἐστίν from ἐστί(ν))
                if extra_form and _is_greek(extra_form):
                    _extra_cc_cap = (pos in ("pron", "det", "adj")
                                     and extra_form
                                     and extra_form != extra_form.lower())
                    _add_lookup_here(lookup, extra_form, form_lemma,
                                proper_noun=(pos == "name"),
                                closed_class_capital=_extra_cc_cap)
                    _add_pos_map(pos_map, extra_form, pair.get("via", form_lemma), pos)
                    pairs.append({**pair, "form": extra_form})

    # --- Pass 2: resolve self-map artifacts using form_of/alt_of targets ---
    # For each lookup entry that is BOTH a self-map AND was detected as a
    # form-of page (confidence 1), replace it with the form_of/alt_of target.
    # This fixes processing-order artifacts where form-of pages self-map
    # because they're processed before their parent's inflection table.
    #
    # Safety: only overrides self-maps (key == lemma) at confidence 1
    # (form-of pages). True headwords at confidence 3 are protected.
    # Non-self mappings (correct table entries like λέγεται -> λέω)
    # are never touched. Proper nouns are excluded via form_of_targets.
    selfmap_resolved = 0
    # An abbreviation's reference moves no key of a word with a sense of
    # its own (``Abbreviations``); main drops those of the other dump's
    # words before the cross-source pass.
    abbreviations = Abbreviations(abbreviated_keys, word_keys)
    drop_abbreviation_targets(form_of_targets, abbreviations, word_keys)
    moves = {}
    for key in list(lookup.keys()):
        lemma, conf = lookup[key]
        if key != lemma:
            continue  # not a self-map, already has a correct mapping
        if conf >= 3:
            continue  # true headword, protected
        # Article forms (ο, η, το, τα, etc.) stay as self-maps. EL
        # Wiktionary lists them as "form of ὁ" via form_of, but for
        # MG lemmatization purposes the monotonic article form is
        # the lemma; downstream consumers (the dilemma tagger, etc.)
        # render it directly. The polytonic ὁ would surface as an AG
        # leak in the MG UI.
        if key in article_forms:
            continue
        if key not in form_of_targets:
            # No form_of/alt_of target available. Closed-class
            # headwords without any form-of evidence stay as self-maps
            # (εγώ, εσύ, τις, ...). They're real lemmas.
            continue
        # There is a form_of target. For pronoun/article inflected forms
        # (αυτό, αυτές, αυτοί), the form_of target is the correct lemma
        # (αυτός) and we should prefer it over the self-map, even though
        # the form appears as a pronoun headword in its own right.
        target = form_of_targets[key]
        if target != key:
            moves[key] = target
    # In the Modern Greek dumps a form-of reference that closes a loop is
    # not followed: two pages that each name the other (αγχωτικός and
    # αγχώδης, each "συνώνυμο του" the other, which kaikki reads as alt_of;
    # δεσποινίς and δεσποινίδα), or a target whose key already maps back
    # here. Following it makes a cycle that chain-breaking deletes along
    # with both paradigms, as in resolve_cross_source, so both words stay
    # lemmas. αγχωτικός kept its paradigm only while the polytonic page
    # ἀγχωτικός took its key as a lemma of its own.
    for key, target in moves.items():
        if modern_greek and (moves.get(target) == key
                             or lookup.get(target, (None,))[0] == key):
            continue
        lookup[key] = (target, lookup[key][1])
        selfmap_resolved += 1

    print(f"  {lang}: scanned {scanned:,} entries, "
          f"{entries_with_data:,} with data, "
          f"{len(pairs):,} pairs, {len(lookup):,} lookup entries")
    if form_of_count:
        print(f"    form_of: {form_of_count:,} pairs")
    if alt_of_count:
        print(f"    alt_of: {alt_of_count:,} pairs")
    if dialect_tagged:
        print(f"    dialect-tagged forms: {dialect_tagged:,}")
    if selfmap_resolved:
        print(f"    self-maps resolved via form_of/alt_of: {selfmap_resolved:,}")
    return (pairs, lookup, page_headwords, pos_map, form_of_targets,
            abbreviations)


def merge_source_lookups(all_lookup: dict, lookups: list,
                         capitals_last: bool = False, lemma_pos: dict = None) -> None:
    """Merge source lookups, leaving weak capital-derived keys until last.

    Each source already lowers a name's lowercase keys to confidence zero.
    Preserve that priority across sources so a surname in one dictionary
    cannot replace the other dictionary's ordinary verb form.
    """
    # A missing lowercase lemma is not a reference to a same-letter name.
    # κυριακή's adjective page names κυριακός, but these dumps only contain
    # the name Κυριακός. Following that fallback displaced the noun Κυριακή.
    # Leave such a reference out so the independently attested noun wins.
    weak_names, stronger = set(), set()
    if capitals_last:
        for lookup in lookups:
            for key, (lemma, confidence) in lookup.items():
                if confidence > 0:
                    stronger.add(key)
                elif key == key.lower() and lemma != lemma.lower():
                    weak_names.add(key)
        weak_names -= stronger
    tiers = (False, True) if capitals_last else (None,)
    for weak in tiers:
        for lookup in lookups:
            for key, (lemma, confidence) in lookup.items():
                if weak is not None and (confidence == 0) != weak:
                    continue
                if (lemma == lemma.lower() and lemma in weak_names and key != lemma
                        and (lemma_pos is None
                             or "verb" in lemma_pos.get(key, set())
                             or ("adj" in lemma_pos.get(key, set())
                                 and any(key in other and other[key][0] != lemma
                                         for other in lookups)))):
                    continue
                all_lookup.setdefault(key, (lemma, 1))


def deduplicate_pairs(pairs: list, preserve_cells: bool = False) -> list:
    """Keep every distinct Modern Greek morphological cell.

    A genitive and an accusative may have the same monotonic form and
    lemma but different polytonic realizations. Tag order is not a cell
    distinction; POS, tags and a form page's resolution target are.
    """
    seen, result = set(), []
    for pair in pairs:
        key = (pair["form"], pair["lemma"])
        if preserve_cells:
            key += (pair.get("pos", ""), tuple(sorted(set(pair.get("tags", [])))),
                    pair.get("via"), bool(pair.get("noun_owner")),
                    bool(pair.get("adjective_owner")))
        if key not in seen:
            seen.add(key)
            result.append(pair)
    return result


_PARTICIPLE_FINITE_TAGS = frozenset({
    "first-person", "second-person", "third-person", "indicative",
    "subjunctive", "imperative", "optative", "infinitive", "finite",
    "dependent",
})


def _is_nominal_participle_cell(tags) -> bool:
    """An explicitly recorded nominal cell, never a finite verbal form."""
    tags = set(tags)
    return (bool(tags & {"nominative", "genitive", "accusative", "vocative", "dative"})
            and bool(tags & {"singular", "plural", "dual"})
            and not tags & _PARTICIPLE_FINITE_TAGS)


def standalone_participle_inventory(cells: list, unresolved: set) -> dict:
    """Preserve source participles separately from finite-verb training rows.

    A missing or ambiguous finite parent does not invalidate a dictionary's
    own participle headword or nominal inflection table. Keep the exact source
    spellings and cells under that participle, without inventing a finite
    lemma or an adjective reading. Edition provenance is merged only for
    identical rows; different readings remain distinct.
    """
    grouped = {}
    for row in cells:
        if row["lemma"] not in unresolved:
            continue
        key = (row["form"], row["lemma"], row["pos"], row["kind"],
               tuple(sorted(set(row["tags"]))), row["source_form"],
               row["source_lemma"])
        grouped.setdefault(key, set()).add(row["source"])
    return {
        "schema_version": 1,
        "language": "el",
        "cells": [
            {"form": form, "lemma": lemma, "pos": pos, "kind": kind,
             "tags": list(tags), "source_form": source_form,
             "source_lemma": source_lemma, "sources": sorted(sources)}
            for (form, lemma, pos, kind, tags, source_form, source_lemma), sources
            in sorted(grouped.items())
        ],
    }



def noun_head_inventory(records: list, lookup: dict, source_hashes: dict) -> dict:
    """Document independent noun heads not representable by one flat reading."""
    grouped = {}
    for record in records:
        lemma = record["lemma"]
        if lookup.get(lemma, (None,))[0] == lemma:
            continue
        evidence = {key: value for key, value in record.items() if key != "lemma"}
        key = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
        grouped.setdefault(lemma, {})[key] = evidence
    return {
        "schema_version": 1, "language": "el",
        "source_sha256": dict(sorted(source_hashes.items())),
        "heads": [{"lemma": lemma, "evidence": [rows[key] for key in sorted(rows)]}
                  for lemma, rows in sorted(grouped.items())],
    }



def _is_verb_inflection_gloss(gloss: str) -> bool:
    """Recognize grammatical form descriptions without matching lexical 'of'."""
    return bool(re.match(
        r"^(?:(?:first|second|third)[ -]person\b|"
        r"(?:singular|plural|dual) (?:present|past|aorist|perfect|future|indicative|subjunctive|imperative)\b|"
        r"(?:inflection|conjugation|form|(?:alternative|alternate) (?:form|spelling)) of\b|"
        r"(?:active|passive|middle)(?: nonfinite form)? of\b|"
        r"(?:(?:present|past|aorist|perfect|future|passive|active|middle) )+"
        r"(?:participle|indicative|subjunctive|imperative|tense)\b|"
        r"(?:(?:άκλιτη|κλιτή) )?μετοχ[ήη]\b|"
        r"(?:πρώτο|δεύτερο|τρίτο) πρόσωπο\b|"
        r"[αβγ][΄'ʹ]?[ .-]+(?:ενικό|πληθυντικό|πρόσωπο)\b)", gloss.strip().lower()))


def _is_closed_class_inflection_gloss(gloss: str) -> bool:
    """Recognize nominal form descriptions, preserving ordinary lexical 'of'."""
    return _is_verb_inflection_gloss(gloss) or bool(re.match(
        r"^(?:(?:(?:masculine|feminine|neuter|common|nominative|genitive|dative|accusative|vocative|singular|plural|dual) )+of\b|"
        r"(?:(?:alternative|alternate) (?:form|spelling)|misspelling) of\b|"
        r"(?:γενική|αιτιατική|ονομαστική|κλητική|δοτική|αρσενικό|θηλυκό|ουδέτερο)\b|"
        r"(?:πολυτονική|εναλλακτική|παλαιότερη|παλιά) γραφή\b)", gloss.strip().lower()))

def _declared_closed_class_form(entry: dict) -> bool:
    """A structured English head template explicitly identifies a form page."""
    return any(template.get("name") == "head"
               and isinstance(template.get("args", {}), dict)
               and template.get("args", {}).get("1") == "el"
               and template.get("args", {}).get("2") in ("pronoun form", "determiner form")
               for template in entry.get("head_templates") or []
               if isinstance(template, dict))


def _closed_class_person_case_gloss(gloss: str) -> bool:
    """Translations followed by explicit grammatical person/case metadata."""
    return bool(re.fullmatch(
        r"[^()]*\(\s*[1-3](?:st|nd|rd)[ -]+person\b[^()]*"
        r"\b(?:nominative|genitive|dative|accusative|vocative)\b[^()]*\)\s*[.;]?",
        gloss.strip().lower()))


def _named_weak_pronoun_reference(entry: dict, sense: dict, source: str) -> str | None:
    """An exact Greek form description names its personal-pronoun parent."""
    if (source != "el" or entry.get("pos") != "pron"
            or "form-of" not in set(entry.get("tags", [])) | set(sense.get("tags", []))
            or sense.get("form_of") or sense.get("alt_of")):
        return None
    targets = set()
    for gloss in sense.get("glosses", []):
        match = re.fullmatch(
            r"αδύνατος τύπος της προσωπικής αντωνυμίας [αβγ]['’΄] προσώπου:\s*(\w+)[.;]?",
            gloss.strip().lower())
        if match and _is_greek(match[1]):
            targets.add(strip_length_marks(match[1]))
    return next(iter(targets)) if len(targets) == 1 else None


def apply_closed_class_form_references(pos_map: dict, references: dict, heads: dict) -> dict:
    """Follow exact typed references to lexical heads without choosing conflicts."""
    def roots(form, pos, seen):
        if form in heads.get(pos, set()):
            return {form}
        if form in seen:
            return set()
        seen = seen | {form}
        targets = references.get((form, pos))
        if targets:
            # Own sense references take precedence over another page's table.
            return set().union(*(roots(target, pos, seen) for target in targets))
        target = pos_map.get(form, {}).get(WIKT_TO_UPOS[pos])
        return roots(target, pos, seen) if target else set()

    selected = {}
    supported_roots = {}
    for form, pos in references:
        supported = roots(form, pos, set())
        supported_roots[form, pos] = supported
        if len(supported) == 1:
            selected[form, pos] = next(iter(supported))
    # Resolve against one snapshot, then collect every exact form's projection.
    # Different accented forms may share a key but retain distinct lexical roots.
    projected = {}
    for (form, pos), supported in supported_roots.items():
        for key in (form, form.lower(), to_monotonic(form), to_monotonic(form).lower(),
                    strip_accents(form.lower())):
            projected.setdefault((key, WIKT_TO_UPOS[pos]), set()).update(supported)
    for (key, upos), supported in projected.items():
        if len(supported) == 1:
            pos_map.setdefault(key, {})[upos] = next(iter(supported))
    return selected


def _validate_head_inventory(inventory: dict, pos: str) -> set:
    """Return positively supported heads, rejecting malformed or form evidence.

    The input hashes identify the source editions. Evidence retains their
    exact headword and senses, so a noun exception cannot be authorized by
    an inflection, spelling reference, empty definition, or another head.
    """
    def fail(message):
        raise ValueError(f"Invalid {pos}-head evidence: " + message)

    reference_tags = _NOUN_REFERENCE_TAGS | ({"alt-of"} if pos in {"det", "pron", "adj"} else set())

    if (not isinstance(inventory, dict) or inventory.get("schema_version") != 1
            or inventory.get("language") != "el"):
        fail("expected schema_version 1 and language el")
    hashes = inventory.get("source_sha256")
    if (not isinstance(hashes, dict) or not hashes
            or any(source not in {"en", "el"} or not isinstance(value, str)
                   or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for source, value in hashes.items())):
        fail("missing source edition hashes")
    rows = inventory.get("heads")
    if not isinstance(rows, list):
        fail("heads must be a list")
    result = set()
    for row in rows:
        if not isinstance(row, dict):
            fail("head must be an object")
        lemma, evidence = row.get("lemma"), row.get("evidence")
        if not isinstance(lemma, str) or not lemma or lemma in result:
            fail("missing or duplicate lemma")
        if not isinstance(evidence, list) or not evidence:
            fail("missing independent evidence for " + lemma)
        for entry in evidence:
            if (not isinstance(entry, dict)
                    or set(entry) != ({"source", "headword", "pos", "tags", "senses",
                                      "form_of", "alt_of"}
                                     | ({"head_templates"} if pos in {"det", "pron", "adj"} else set())
                                     | ({"forms"} if pos == "adj" else set()))):
                fail("incomplete source entry for " + lemma)
            source, word = entry["source"], entry["headword"]
            if (source not in hashes or entry["pos"] != pos
                    or not isinstance(word, str) or strip_length_marks(word) != lemma):
                fail("wrong source head or part of speech for " + lemma)
            tags, senses = entry["tags"], entry["senses"]
            if (not isinstance(tags, list) or not all(isinstance(t, str) for t in tags)
                    or reference_tags.intersection(tags)
                    or not isinstance(senses, list) or not senses):
                fail("nonlexical noun evidence for " + lemma)
            if (not isinstance(entry["form_of"], list)
                    or not isinstance(entry["alt_of"], list)
                    or (source != "el" and (entry["form_of"] or entry["alt_of"]))):
                fail("reference-only head for " + lemma)
            declared_form = False
            if pos in {"det", "pron", "adj"}:
                templates = entry["head_templates"]
                if (not isinstance(templates, list)
                        or not all(isinstance(t, dict)
                                   and isinstance(t.get("args", {}), dict)
                                   and all(isinstance(k, str) and isinstance(v, str)
                                           for k, v in t.get("args", {}).items())
                                   for t in templates)):
                    fail("malformed head templates for " + lemma)
                declared_form = pos in {"det", "pron"} and _declared_closed_class_form(entry)
            if pos == "adj":
                forms = entry["forms"]
                if (not isinstance(forms, list) or not all(
                        isinstance(form, dict) and set(form) == {"form", "tags", "source"}
                        and isinstance(form["form"], str)
                        and isinstance(form["source"], str)
                        and isinstance(form["tags"], list)
                        and all(isinstance(tag, str) for tag in form["tags"])
                        for form in forms)):
                    fail("malformed adjective morphology for " + lemma)
            substantive = False
            for sense in senses:
                if (not isinstance(sense, dict)
                        or set(sense) != ({"glosses", "tags", "form_of", "alt_of", "related"}
                                          if pos == "verb" else {"glosses", "tags", "form_of", "alt_of"})
                        or not all(isinstance(sense[k], list) for k in sense)
                        or not all(isinstance(g, str) and g.strip() for g in sense["glosses"])
                        or not all(isinstance(t, str) for t in sense["tags"])
                        or sense["form_of"] or sense["alt_of"]
                        or reference_tags.intersection(sense["tags"])):
                    fail("form or malformed sense for " + lemma)
                substantive |= bool("no-gloss" not in sense["tags"] and any(
                    (not _is_verb_inflection_gloss(gloss) if pos == "verb" else
                     not _is_closed_class_inflection_gloss(gloss)
                     and not (declared_form and _closed_class_person_case_gloss(gloss))
                     if pos in {"det", "pron"} else True)
                    for gloss in sense["glosses"]))
            first_gloss = (senses[0]["glosses"] or [""])[0].lower()
            has_ordinary_sense = any(s["glosses"] and "participle" not in s["tags"]
                                     for s in senses)
            participle = (pos == "verb" and ("participle" in tags or any(
                _is_participle_sense(s, has_ordinary_sense) for s in senses)))
            reference_gloss = (pos == "noun" and any(
                p in first_gloss for p in _FORM_OF_PATTERNS))
            if (not substantive or reference_gloss or participle
                    or spelling_target(lemma, senses) is not None):
                fail("no independent lexical sense for " + lemma)
        result.add(lemma)
    return result



def validate_noun_head_inventory(inventory: dict) -> set:
    """Return exact source-backed noun exceptions to the flat-head rule."""
    return _validate_head_inventory(inventory, "noun")


def adjective_head_inventory(records: list, lookup: dict, source_hashes: dict) -> dict:
    """Record exact adjective definitions whose flat head belongs elsewhere."""
    return noun_head_inventory(records, lookup, source_hashes)


def validate_adjective_head_inventory(inventory: dict) -> set:
    """Authorize only positively defined, exact Modern adjective heads."""
    heads = _validate_head_inventory(inventory, "adj")
    for row in inventory["heads"]:
        for evidence in row["evidence"]:
            entry = {**evidence, "word": evidence["headword"]}
            if not is_independent_adjective_entry(entry, evidence["source"]):
                raise ValueError("Invalid adj-head evidence: no independent adjective definition for "
                                 + row["lemma"])
    return heads



def verb_head_evidence(entry: dict, source: str) -> dict:
    """Retain the exact source fields used to identify a finite verb head."""
    return {
        "source": source, "headword": entry.get("word", ""), "pos": entry.get("pos", ""),
        "tags": sorted(set(entry.get("tags") or [])),
        "form_of": entry.get("form_of") or [], "alt_of": entry.get("alt_of") or [],
        "senses": [{key: sense.get(key) or []
                    for key in ("glosses", "tags", "form_of", "alt_of", "related")}
                   for sense in entry.get("senses") or []],
    }


def is_independent_finite_entry(entry: dict, source: str) -> bool:
    """Require a positive lexical definition, not a header or form reference."""
    if entry.get("pos") != "verb":
        return False
    word = entry.get("word", "")
    if not word:
        return False
    probe = {"schema_version": 1, "language": "el", "source_sha256": {source: "0" * 64},
             "heads": [{"lemma": strip_length_marks(word),
                        "evidence": [verb_head_evidence(entry, source)]}]}
    try:
        _validate_head_inventory(probe, "verb")
    except ValueError:
        return False
    return True


_ANCIENT_GREEK_WORD = r"[\u0370-\u03ff\u1f00-\u1fff\u0300-\u036f-]+"
_ANCIENT_REFERENCE_ARTICLE = r"(?:του|των|της\s+λέξης)\b"
_ANCIENT_REFERENCE_TARGET_ITEM = (
    rf"(?:{_ANCIENT_REFERENCE_ARTICLE}\s+)?"
    r"(?:(?:ρήματος|επιθέτου|επιρρήματος|αρχαίου|αρχαίας|ελληνιστικού|ελληνιστικής|μεταγενέστερου)\s+)*"
    + r"\*?" + _ANCIENT_GREEK_WORD
)
_ANCIENT_REFERENCE_TARGET = (
    _ANCIENT_REFERENCE_TARGET_ITEM +
    r"(?:\s*(?:/|&|και|ή|ἤ|ἠ|-)\s*" + _ANCIENT_REFERENCE_TARGET_ITEM + r")*"
    r"(?:\s+\((?:(?:και|αλλά\s+και\s+των\s+ομόρριζων)\s+" +
    _ANCIENT_GREEK_WORD + r"(?:\s*,\s*" + _ANCIENT_GREEK_WORD + r")*|"
    r"συνηρημένη\s+μορφή\s+του\s+" + _ANCIENT_GREEK_WORD + r")\))?"
)
_ANCIENT_REFERENCE_MODIFIER = (
    r"(?:επικός|ποιητικός|ιωνικός|(?:παλαιός\s+)?αττικός|δωρικός|αιολικός|"
    r"βοιωτικός|θεσσαλικός|αρκαδοκυπριακός|κρητικός|λακωνικός|"
    r"εναλλακτικός|άλλος|παλαιότερος|μεταγενέστερος|σπανιότερος|συνηρημένος)"
)
_ANCIENT_REFERENCE_SEPARATOR = r"(?:\s+(?:και|&)\s+|\s*[,/]\s*)"
_ANCIENT_REFERENCE_TYPE = (
    rf"(?:{_ANCIENT_REFERENCE_MODIFIER}"
    rf"(?:{_ANCIENT_REFERENCE_SEPARATOR}{_ANCIENT_REFERENCE_MODIFIER})*\s+)?"
    r"τύπος"
)
_ANCIENT_REFERENCE_GLOSS = re.compile(
    rf"(?:{_ANCIENT_REFERENCE_TYPE}"
    rf"(?:{_ANCIENT_REFERENCE_SEPARATOR}{_ANCIENT_REFERENCE_TYPE})*\s+{_ANCIENT_REFERENCE_ARTICLE}|"
    rf"(?:συνηρημένη|εναλλακτική|άλλη|παλαιότερη|πολυτονική)\s+μορφή\s+{_ANCIENT_REFERENCE_ARTICLE})\s*:?\s*"
    + _ANCIENT_REFERENCE_TARGET + r"\s*[.,;]?"
)
_ANCIENT_GRAMMATICAL_FORM_GLOSS = re.compile(
    rf"^(?:{_ANCIENT_REFERENCE_TYPE}\s*:\s*[αβγ][΄'ʹ]?\s+πρόσωπο\b|"
    r"απαρέμφατο\b)"
)
_ANCIENT_COMPARATIVE_REFERENCE_GLOSS = re.compile(
    _ANCIENT_REFERENCE_TYPE + r"\s+του\s+" + _ANCIENT_GREEK_WORD +
    r"\s*\(" + _ANCIENT_GREEK_WORD + r"\)\s*[·;]\s*"
    r"συγκριτικός\s+βαθμός\s+του\s+" + _ANCIENT_GREEK_WORD + r"\s*[.;]?"
)
_ANCIENT_DEGREE_REFERENCE_GLOSS = re.compile(
    r"(?:συγκριτικός|υπερθετικός)(?:\s+βαθμός)?\s+(?:του|της|των)\s+"
    + _ANCIENT_GREEK_WORD + r"(?=\s|[:,.;]|$)"
)
_ANCIENT_EN_REFERENCE_MODIFIER = (
    r"(?:Attic|Ionic|Doric|Aeolic|Homeric|Epic|Delphic|Pamphylian|"
    r"post-Epic|poetic|Tragic|contracted|uncontracted|alternative|alternate|"
    r"obsolete|archaic|rare|variant|nonstandard)"
)
_ANCIENT_EN_REFERENCE_GLOSS = re.compile(
    rf"^(?:{_ANCIENT_EN_REFERENCE_MODIFIER}"
    rf"(?:(?:\s*,?\s+(?:and|&)\s+|\s*[,/]\s*){_ANCIENT_EN_REFERENCE_MODIFIER})*\s+)?"
    r"(?:form|spelling|variant) of\s+[\u0370-\u03ff\u1f00-\u1fff]",
    re.IGNORECASE,
)
_ANCIENT_UNQUALIFIED_FORM_GLOSS = re.compile(
    r"μορφή\s+του\s+" + _ANCIENT_GREEK_WORD +
    r"(?:\s+\((?:και|αλλά\s+και\s+των\s+ομόρριζων)\s+" +
    _ANCIENT_GREEK_WORD + r"(?:\s*,\s*" + _ANCIENT_GREEK_WORD + r")*\))?\s*[.;]?"
)
_ANCIENT_VOICE_REFERENCE_GLOSS = re.compile(
    r"(?:(?:μεσοπαθητική|παθητική|μέση(?:\s*-\s*παθητική|\s+και\s+παθητική)?)\s+φωνή|"
    r"(?:μέσος(?:\s+και\s+παθητικός)?|παθητικός)\s+τύπος)\s+του\s+(?:ρήματος\s+)?"
    + _ANCIENT_GREEK_WORD +
    r"(?:\s*\((?:συνηρημένη\s+μορφή\s+του\s+" + _ANCIENT_GREEK_WORD +
    r"|παθητική\s+διάθεση)\)|\s*,\s*συνηρημένη\s+μορφή\s+του\s+" +
    _ANCIENT_GREEK_WORD + r")?\s*[.;]?"
)
_ANCIENT_VOICE_NOTE = re.compile(
    r"(?:η\s+μέση(?:\s+φωνή)?\s+εκφέρεται\s+και\s+αναλυτικά\s+με\s+την\s+"
    r"ενεργητική\s+φωνή:\s*[\"«]" + _ANCIENT_GREEK_WORD +
    r"(?:\s+" + _ANCIENT_GREEK_WORD + r")*[\"»]|"
    r"ο\s+μέσος\s+μέλλων\s*\(" + _ANCIENT_GREEK_WORD +
    r"\)\s+συνήθως\s+σύνθετος|"
    r"ο\s+παθητικός\s+αόριστος\s*[\"«]" + _ANCIENT_GREEK_WORD +
    r"[\"»]\s+είναι\s+και\s+μέσος)"
)
_ANCIENT_REFERENCE_NOTE = re.compile(
    r"(?:χρειάζεται\s+επεξεργασία|στον\s+όμηρο|"
    r"(?:συνηρημένος|ασυναίρετος)\s+τύπος|συνηρημένο|"
    r"μέσω\s+κράσης\s+με\s+το\s+" + _ANCIENT_GREEK_WORD + r"|"
    r"επίσης:\s*" + _ANCIENT_GREEK_WORD +
    r"(?:\s*(?:,|και|ή|ἤ|ἠ|&)\s*" + _ANCIENT_GREEK_WORD + r")*|"
    r"συχνά\s+σε\s+ελληνιστικά\s+κείμενα|ελληνιστική\s+κοινή|"
    r"στην\s+κεντρική\s+κρήτη|στη\s+γόρτυνα|των\s+μεγαρέων|στους\s+σικελιώτες|"
    r"α\s+στερ\.?\s*\+\s*" + _ANCIENT_GREEK_WORD + r"|"
    r"εκ\s+του\s+" + _ANCIENT_GREEK_WORD + r"|"
    r"από\s+τον\s+αόριστο\s+του\s+οποίου,\s*δηλ\.\s*το\s+" +
    _ANCIENT_GREEK_WORD + r",\s*θεωρείται\s+ότι\s+προέκυψε\s+το\s+" +
    _ANCIENT_GREEK_WORD + r"|"
    r"κλίση\s+" + _ANCIENT_GREEK_WORD + r"\s*-\s*απαντούν\s+συνηρημένοι\s+τύποι|"
    r"απαντούν\s+τύποι\s+από\s+την\s+κλίση\s+" + _ANCIENT_GREEK_WORD + r"|"
    r"αποθετικού\s+ρήματος|κυρίως\s+" + _ANCIENT_REFERENCE_TYPE + r"|"
    r"όπως\s+και\s+το\s+ποιητικό\s+" + _ANCIENT_GREEK_WORD + r"|"
    r"μαρτυρείται\s+στο\s+απαρέμφατο\s+ενεστώτα\s+" + _ANCIENT_GREEK_WORD + r")"
)
# Bare target lists and parentheticals can also look like real translations.
# These reviewed full source glosses are references; any added meaning remains
# evidence, and no rule depends on the entry's headword or chosen endpoint.
_ANCIENT_REVIEWED_REFERENCE_GLOSSES = frozenset({
    "κρητικός τύπος του χαλκός, χαλκοῦς",
    "δωρικός τύπος του γᾶ (γῆ)",
    "αιολικός & δωρικός τύπος του ἐκεῖνος, κεῖνος",
    "αιολικός τύπος του ἁμός, -ά, -όν (ἡμέτερος)",
    "αιολικός τύπος του ἐν μέσῳ",
    "άλλη μορφή της λέξης γλοιός, γλία",
    "άλλη μορφή του ορυκτού κίσσηρη",
    'άλλη μορφή του επιθέτου πορτογαλικός. θεωρείται λιγότερο επίσημη, '
    'χρησιμοποιείται κυρίως γιατί το "πορτογαλικός" προκαλεί σύγχυση με το '
    'δεύτερο συνθετικό του "γαλικός".',
    "επικός τύπος του ὁδός, μόνο στην οδύσσεια (σε αιτιατική): "
    "※ 8ος αιώνας πκε ⌘ ὅμηρος, ὀδύσσεια, 17 "
    "(ρ. τηλεμάχου καὶ ὀδυσσέως ἐπάνοδος εἰς ἰθάκην.), στίχ. 196 (195-196 )",
})


def _is_ancient_voice_reference(gloss: str) -> bool:
    if _ANCIENT_VOICE_REFERENCE_GLOSS.fullmatch(gloss):
        return True
    first, separator, remainder = gloss.partition(".")
    if not separator or not _ANCIENT_VOICE_REFERENCE_GLOSS.fullmatch(first):
        return False
    # A voice reference may append paradigm notes. Consume only complete
    # grammatical clauses; any translation or other prose remains evidence.
    remainder = remainder.strip()
    if not remainder:
        return False
    while remainder:
        note = _ANCIENT_VOICE_NOTE.match(remainder)
        if not note:
            return False
        remainder = remainder[note.end():].lstrip(" .;")
    return True


def _is_ancient_reference_note(note: str) -> bool:
    """Recognize explicit grammatical metadata, never arbitrary Greek prose."""
    note = note.strip().rstrip(".;").strip()
    # Some source pages omit the space after a person abbreviation.
    note = re.sub(r"^([αβγ][΄'ʹ]?)\s*(?=πρόσωπο\b)", r"\1 ", note)
    return bool(_is_verb_inflection_gloss(note)
                or _ANCIENT_REFERENCE_NOTE.fullmatch(note)
                or _ANCIENT_REFERENCE_GLOSS.fullmatch(note)
                or _is_ancient_voice_reference(note))


def _is_ancient_form_reference(text: str) -> bool:
    """Allow complete references with bounded grammar or source notes."""
    if _ANCIENT_REFERENCE_GLOSS.fullmatch(text):
        return True
    parenthetical = re.search(r"\s*\(([^()]*)\)\s*[.;]?$", text)
    if parenthetical and _is_ancient_reference_note(parenthetical[1]):
        return _is_ancient_form_reference(text[:parenthetical.start()].rstrip())
    # A colon or comma may introduce either grammar or a real meaning. Only
    # consume a recognized grammatical continuation after a complete reference.
    for separator in re.finditer(r"\s*[:,]\s*|\s+-\s+", text):
        if (_is_ancient_reference_note(text[separator.end():])
                and _is_ancient_form_reference(text[:separator.start()].rstrip())):
            return True
    for suffix in (" των μεγαρέων", " στους σικελιώτες"):
        if text.rstrip(".;").endswith(suffix):
            return _is_ancient_form_reference(text.rstrip(".;")[:-len(suffix)].rstrip())
    return False


def _is_ancient_reference_gloss(gloss: str, pos: str | None = None) -> bool:
    """Identify grammatical references without discarding lexical meanings."""
    text = " ".join(gloss.split())
    return bool(text.lower() in _ANCIENT_REVIEWED_REFERENCE_GLOSSES
                or _is_ancient_form_reference(text.lower())
                or _ANCIENT_COMPARATIVE_REFERENCE_GLOSS.fullmatch(text.lower())
                # A degree header and its translation describe the base
                # adjective's comparison, as English structured form_of
                # records do. A separate lexical gloss can still qualify.
                or (pos == "adj" and _ANCIENT_DEGREE_REFERENCE_GLOSS.match(text.lower()))
                or _ANCIENT_GRAMMATICAL_FORM_GLOSS.match(text.lower())
                or _ANCIENT_EN_REFERENCE_GLOSS.match(text)
                or _is_ancient_voice_reference(text.lower())
                # A bare 'form of' can define a noun's physical shape. Only
                # a complete verb/adjective reference is unambiguous here.
                or (pos in {"verb", "adj"}
                    and _ANCIENT_UNQUALIFIED_FORM_GLOSS.fullmatch(text.lower())))


def is_independent_lexical_entry(entry: dict, source: str) -> bool:
    """Require a definition of this word, excluding inflection-only pages.

    Ancient homographs can have separate lexical and form-of entries even
    within one part of speech. Their own tables remain separate readings.
    """
    if entry.get("pos") not in WIKT_TO_UPOS:
        # Letter names, symbols, and phrase headers do not establish a
        # lexical POS owner for an inflected word with the same spelling.
        return False
    if entry.get("pos") == "verb" and not is_independent_finite_entry(entry, source):
        return False
    senses = entry.get("senses") or []
    reference_tags = _NOUN_REFERENCE_TAGS | {"alt-of"}
    if (reference_tags.intersection(entry.get("tags") or [])
            or (source != "el" and (entry.get("form_of") or entry.get("alt_of")))
            or any(s.get("form_of") or s.get("alt_of")
                   or reference_tags.intersection(s.get("tags") or []) for s in senses)):
        return False
    return any("no-gloss" not in (s.get("tags") or [])
               # English form senses retain their grammatical header and
               # nested translation in one list. The child translation
               # does not define another head. Greek entries can instead
               # cite a present first-person head before its own meanings.
               and not (source == "en" and entry.get("pos") == "verb"
                        and s.get("glosses")
                        and _is_verb_inflection_gloss(s["glosses"][0]))
               and any(g.strip() and not _is_closed_class_inflection_gloss(g)
                       and not _is_ancient_reference_gloss(g, entry.get("pos"))
                       for g in s.get("glosses") or []) for s in senses)

def validate_verb_head_inventory(inventory: dict) -> set:
    """Return independently defined finite heads, excluding participles."""
    return _validate_head_inventory(inventory, "verb")


def training_pair_headword_errors(pairs: list, lookup: dict,
                                  noun_heads: set = frozenset(),
                                  verb_heads: set = frozenset(),
                                  closed_class_heads: dict = None,
                                  lexical_heads: dict = None,
                                  adjective_heads: set = frozenset()) -> tuple:
    """Require a flat head or independent evidence for this exact POS."""
    count, examples = 0, []
    for pair in pairs:
        lemma = pair["lemma"]
        needs_proof = lemma in lookup and lookup[lemma] != lemma
        if lexical_heads is not None and lemma not in lookup:
            needs_proof = True
        if (needs_proof
                and not (pair.get("pos") == "noun" and lemma in noun_heads)
                and not (pair.get("pos") == "verb" and lemma in verb_heads)
                and not (pair.get("pos") == "adj" and lemma in adjective_heads)
                and not (pair.get("pos") in {"det", "pron"}
                         and lemma in (closed_class_heads or {}).get(pair.get("pos"), set()))
                and lemma not in (lexical_heads or {}).get(pair.get("pos"), set())):
            count += 1
            if len(examples) < 5:
                examples.append(f"{pair['form']} -> {lemma} -> {lookup.get(lemma, '<absent>')}")
    return count, examples


def ancient_reading_head_inventory(pairs: list, lookup: dict, sources: dict) -> dict:
    """Keep source evidence for Ancient readings outside the flat head map."""
    import hashlib

    wanted = {(p["lemma"], p.get("pos", "")) for p in pairs
              if lookup.get(p["lemma"], (None,))[0] != p["lemma"]}
    evidence = {key: {} for key in wanted}
    hashes = {}
    for source, path in sorted(sources.items()):
        if not path.exists():
            continue
        with path.open('rb') as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
            hashes[source] = digest.hexdigest()
        with path.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (strip_length_marks(entry.get('word', '')), entry.get('pos', ''))
                if key not in wanted or not is_independent_lexical_entry(entry, source):
                    continue
                record = {'source': source, 'source_line': line_number, 'entry': {
                    'word': entry['word'], 'pos': entry['pos'],
                    'tags': entry.get('tags') or [],
                    'form_of': entry.get('form_of') or [],
                    'alt_of': entry.get('alt_of') or [],
                    'senses': [{field: sense.get(field) or [] for field in
                                ('glosses', 'tags', 'form_of', 'alt_of', 'related')}
                               for sense in entry.get('senses') or []]}}
                evidence[key][json.dumps(record, ensure_ascii=False, sort_keys=True)] = record
    missing = [key for key, records in evidence.items() if not records]
    if missing:
        raise ValueError('Unproven Ancient reading heads: ' + repr(sorted(missing)))
    return {'schema_version': 1, 'language': 'grc', 'source_sha256': hashes,
            'heads': [{'lemma': lemma, 'pos': pos,
                       'evidence': [records[key] for key in sorted(records)]}
                      for (lemma, pos), records in sorted(evidence.items())]}


def validate_ancient_reading_head_inventory(inventory: dict) -> dict:
    """Recheck exact positive source definitions for non-flat Ancient heads."""
    if (not isinstance(inventory, dict)
            or set(inventory) != {'schema_version', 'language', 'source_sha256', 'heads'}
            or inventory['schema_version'] != 1 or inventory['language'] != 'grc'
            or not isinstance(inventory['heads'], list)
            or not isinstance(inventory['source_sha256'], dict)
            or not inventory['source_sha256']
            or any(source not in {'en', 'el'} or not isinstance(sha, str)
                   or re.fullmatch(r'[0-9a-f]{64}', sha) is None
                   for source, sha in inventory['source_sha256'].items())):
        raise ValueError('Invalid Ancient reading-head inventory')
    result = {}
    for row in inventory['heads']:
        if (not isinstance(row, dict) or set(row) != {'lemma', 'pos', 'evidence'}
                or not isinstance(row['lemma'], str) or not row['lemma']
                or not isinstance(row['pos'], str) or row['pos'] not in WIKT_TO_UPOS
                or row['lemma'] in result.get(row['pos'], set())
                or not isinstance(row['evidence'], list) or not row['evidence']):
            raise ValueError('Invalid Ancient reading-head row')
        for proof in row['evidence']:
            if (not isinstance(proof, dict) or set(proof) != {'source', 'source_line', 'entry'}
                    or not isinstance(proof['source'], str)
                    or proof['source'] not in inventory['source_sha256']
                    or type(proof['source_line']) is not int or proof['source_line'] < 1):
                raise ValueError('Invalid Ancient reading-head source')
            entry = proof['entry']
            if (not isinstance(entry, dict)
                    or set(entry) != {'word', 'pos', 'tags', 'form_of', 'alt_of', 'senses'}
                    or not isinstance(entry['word'], str)
                    or strip_length_marks(entry['word']) != row['lemma']
                    or entry['pos'] != row['pos']
                    or any(not isinstance(entry[field], list)
                           for field in ('tags', 'form_of', 'alt_of', 'senses'))
                    or any(not isinstance(tag, str) for tag in entry['tags'])
                    or any(not isinstance(sense, dict)
                           or set(sense) != {'glosses', 'tags', 'form_of', 'alt_of', 'related'}
                           or any(not isinstance(sense[field], list) for field in sense)
                           or any(not isinstance(value, str)
                                  for field in ('glosses', 'tags') for value in sense[field])
                           for sense in entry['senses'])
                    or not is_independent_lexical_entry(entry, proof['source'])):
                raise ValueError('No independent Ancient reading-head evidence')
        result.setdefault(row['pos'], set()).add(row['lemma'])
    return result


def closed_class_head_evidence(entry: dict, source: str) -> dict:
    """Retain exact same-POS source evidence, without inferring a reading."""
    evidence = verb_head_evidence(entry, source)
    evidence["head_templates"] = entry.get("head_templates") or []
    for sense in evidence["senses"]:
        del sense["related"]
    return evidence


def is_independent_closed_class_entry(entry: dict, source: str) -> bool:
    pos, word = entry.get("pos"), entry.get("word")
    if pos not in {"det", "pron"} or not word:
        return False
    probe = {"schema_version": 1, "language": "el", "source_sha256": {source: "0" * 64},
             "heads": [{"lemma": strip_length_marks(word),
                        "evidence": [closed_class_head_evidence(entry, source)]}]}
    try:
        _validate_head_inventory(probe, pos)
    except ValueError:
        return False
    return True


def closed_class_head_inventory(records: list, lookup: dict, source_hashes: dict) -> dict:
    grouped = {}
    for record in records:
        lemma, pos = record["lemma"], record["pos"]
        if lookup.get(lemma, (lemma,))[0] == lemma:
            continue
        evidence = {key: value for key, value in record.items() if key != "lemma"}
        key = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
        grouped.setdefault((lemma, pos), {})[key] = evidence
    return {"schema_version": 1, "language": "el",
            "source_sha256": dict(sorted(source_hashes.items())),
            "heads": [{"lemma": lemma, "pos": pos,
                       "evidence": [rows[key] for key in sorted(rows)]}
                      for (lemma, pos), rows in sorted(grouped.items())]}


def validate_closed_class_head_inventory(inventory: dict, pos_lookup: dict = None) -> dict:
    """Require independent exact DET/PRON evidence and matching typed targets."""
    if not isinstance(inventory, dict) or not isinstance(inventory.get("heads"), list):
        raise ValueError("Invalid closed-class-head evidence")
    result = {"det": set(), "pron": set()}
    for row in inventory["heads"]:
        if (not isinstance(row, dict) or set(row) != {"lemma", "pos", "evidence"}
                or row["pos"] not in result or not isinstance(row["lemma"], str)
                or row["lemma"] in result[row["pos"]]):
            raise ValueError("Invalid closed-class-head row")
        probe = {**inventory, "heads": [{"lemma": row["lemma"], "evidence": row["evidence"]}]}
        heads = _validate_head_inventory(probe, row["pos"])
        if pos_lookup is not None and (not isinstance(pos_lookup, dict)
                or not isinstance(pos_lookup.get(row["lemma"]), dict)
                or pos_lookup[row["lemma"]].get(WIKT_TO_UPOS[row["pos"]]) != row["lemma"]):
            raise ValueError("Missing same-POS self-target for " + row["lemma"])
        result[row["pos"]].update(heads)
    # Validate the envelope even when no exceptions are necessary.
    _validate_head_inventory({**inventory, "heads": []}, "det")
    return result

def apply_participle_references(pos_map: dict, references: dict,
                               verb_heads: set,
                               participle_heads: set = frozenset(),
                               lemma_pos: dict = None,
                               verb_refs: dict = None,
                               participle_verb_refs: dict = None) -> set:
    """Resolve participle pointers to lexical verbs from one graph snapshot.

    Explicit sense references take priority over entry-level pointers, which
    take priority over inflection-table evidence. A successful tier retains
    all its supported endpoints; ambiguity is not resolved by source order.
    """
    finite_heads = verb_heads - participle_heads - references.keys()

    def eligible(target):
        return not lemma_pos or target not in lemma_pos or "verb" in lemma_pos[target]

    table = {word: {target} for word, values in pos_map.items()
             if (target := values.get("VERB")) and target != word and eligible(target)}
    # The default preserves this helper's original direct-call convention.
    # Production extraction always supplies the separate explicit multimap.
    sense = table if verb_refs is None else {
        word: {target for target in targets if target != word and eligible(target)}
        for word, targets in verb_refs.items()}
    if participle_verb_refs is not None:
        # A finite-form homograph's references do not describe a separately
        # tagged participle. Keep the latter's own entry provenance.
        for word in participle_heads | participle_verb_refs.keys():
            sense[word] = {target for target in participle_verb_refs.get(word, ())
                           if target != word and eligible(target)}
    # Top-level etymology metadata can name a nominal label such as λόγιο.
    # An unrelated verb's form table does not make that label a verb alias.
    references = {word: {target for target in targets if eligible(target)}
                  for word, targets in references.items()}

    pending, reachable = list(references), set()
    while pending:
        word = pending.pop()
        if word in reachable or word in finite_heads:
            continue
        reachable.add(word)
        for tier in (sense, references, table):
            pending.extend(tier.get(word, ()))

    # Strongly connected components of the preferred explicit-reference
    # graph. A cycle has no preferred member; its outgoing evidence must
    # identify one agreed finite verb before any member can be rewritten.
    order, visited = [], set()
    for start in reachable:
        if start in visited:
            continue
        stack = [(start, False)]
        while stack:
            word, finish = stack.pop()
            if finish:
                order.append(word)
            elif word not in visited:
                visited.add(word)
                stack.append((word, True))
                stack.extend((target, False) for target in sense.get(word, ())
                             if target in reachable and target not in visited)
    reverse = {}
    for word in reachable:
        for target in sense.get(word, ()):
            if target in reachable:
                reverse.setdefault(target, set()).add(word)
    cycles, assigned = {}, set()
    for start in reversed(order):
        if start in assigned:
            continue
        group, pending = set(), [start]
        while pending:
            word = pending.pop()
            if word in assigned:
                continue
            assigned.add(word)
            group.add(word)
            pending.extend(reverse.get(word, ()))
        if len(group) > 1:
            component = frozenset(group)
            cycles.update((word, component) for word in group)

    def roots(word, path=frozenset()):
        if word in finite_heads:
            return {word}
        marker = cycles.get(word, word)
        if marker in path:
            return set()
        path = path | {marker}
        members = cycles.get(word, {word})
        for tier in (sense, references, table):
            targets = {target for member in members for target in tier.get(member, ())
                       if target not in members}
            supported = {root for target in targets for root in roots(target, path)}
            if supported:
                return supported
        return set()

    replacements = {}
    for lemma in references:
        supported = roots(lemma)
        if len(supported) == 1:
            replacements[lemma] = next(iter(supported))
    # Evaluate all paths before mutation. A finite endpoint must stay terminal
    # even if an inflection table elsewhere supplied a reverse POS edge.
    for root in set(replacements.values()):
        pos_map.setdefault(root, {})["VERB"] = root
    for lemma, target in replacements.items():
        pos_map.setdefault(lemma, {})["VERB"] = target
    return set(references) - replacements.keys()


def resolve_pair_lemma(lemma: str, pos: str, lookup: dict,
                       pos_map: dict = None, lemma_pos: dict = None,
                       participle_roots: dict = None,
                       unresolved_verbs: set = frozenset(),
                       closed_class_heads: dict = None,
                       lexical_heads: dict = None,
                       canonical_references: dict = None,
                       adjective_heads: set = frozenset(), *,
                       reference_target: bool = False):
    """Follow a verb's own references before a homograph's flat lookup.

    A polytonic participle can first name its monotonic spelling, whose
    unqualified key belongs to an adjective or noun. The verb references
    still name its actual verb. A known non-verb cannot be a verb lemma.
    """
    original = lemma
    if pos == "adj" and lemma in adjective_heads:
        # This exact target is a defined adjective, not a noun homograph.
        return lemma
    if lexical_heads is not None:
        # A same-POS reference ends at a positively defined lexical head,
        # before the untyped lookup can follow another homograph's edge.
        # Do not treat an inflection-only POS label as lexical evidence.
        upos = WIKT_TO_UPOS.get(pos)
        seen = set()
        while lemma not in seen:
            lexical = lemma in lexical_heads.get(pos, set())
            pending = list((canonical_references or {}).get(lemma, {}).get(pos, set()))
            if lexical or (reference_target and pending):
                # A separate inflected homograph does not replace this
                # lexical reading. Only explicit alternative spellings
                # can move its own table to another lexical head.
                # An explicit source target follows the same chain before
                # another homograph's table can claim it. A scalar POS map
                # has no source-entry context and keeps its prior fallback.
                if not pending:
                    return lemma
                visited, endpoints = {lemma}, set()
                while pending:
                    candidate = pending.pop()
                    if candidate in visited:
                        continue
                    visited.add(candidate)
                    targets = (canonical_references or {}).get(candidate, {}).get(pos, set())
                    if targets:
                        pending.extend(targets - visited)
                    elif candidate in lexical_heads.get(pos, set()):
                        endpoints.add(candidate)
                if len(endpoints) == 1:
                    return next(iter(endpoints))
                if lexical:
                    return lemma
            seen.add(lemma)
            target = (pos_map or {}).get(lemma, {}).get(upos)
            if not target or target == lemma:
                break
            lemma = target
        lemma = original
    if pos in {"det", "pron"} and pos_map is not None and closed_class_heads:
        upos = WIKT_TO_UPOS[pos]
        seen = set()
        while lemma not in seen:
            if lemma in closed_class_heads.get(pos, set()):
                return lemma
            seen.add(lemma)
            target = pos_map.get(lemma, {}).get(upos)
            if not target or target == lemma:
                break
            lemma = target
        # An unproven closed-class chain retains the existing flat behavior.
        lemma = original
    if pos == "verb" and pos_map is not None:
        seen = set()
        while lemma not in seen:
            if lemma in unresolved_verbs:
                return None
            if participle_roots and lemma in participle_roots:
                # This endpoint is supported by the verb reference graph;
                # a flat noun or finite-form homograph must not replace it.
                return participle_roots[lemma]
            seen.add(lemma)
            target = pos_map.get(lemma, {}).get("VERB")
            if not target or target == lemma:
                break
            lemma = target
    resolved = lookup.get(lemma, (lemma,))[0]
    if pos == "verb" and participle_roots and resolved in participle_roots:
        return participle_roots[resolved]
    if pos == "verb" and resolved in unresolved_verbs:
        return None
    if (pos == "verb" and lookup.get(resolved, (None,))[0] != resolved
            and lookup.get(original, (None,))[0] == original
            and lemma_pos and "verb" in lemma_pos.get(original, set())):
        # A broken reference must not remove a separately attested verb
        # head and its own paradigm (e.g. ημαρτημένος). This does not
        # rescue a reference that actually resolves to a nominal head.
        resolved = original
    if pos == "verb" and (pos_map is not None or lemma_pos is not None):
        if lemma == lemma.lower() and resolved != resolved.lower():
            return None
        if lemma_pos and resolved in lemma_pos and "verb" not in lemma_pos[resolved]:
            return None
    return resolved


def preserve_finite_verb_heads(pos_map: dict, lookup: dict, finite_heads: set,
                               participle_roots: dict, unresolved_verbs: set):
    """Keep lexical verbs whose flat or table alias hits an unresolved form.

    A participle can name two independently defined verbs. Omitting that
    ambiguous participle must not also omit either finite verb and its own
    paradigm merely because an inflection table supplied the reverse alias.
    """
    protected = set()
    for head in finite_heads:
        word, seen = head, set()
        while word not in seen:
            if (word in unresolved_verbs
                    or lookup.get(word, (word,))[0] in unresolved_verbs):
                protected.add(head)
                break
            seen.add(word)
            target = pos_map.get(word, {}).get("VERB")
            if not target or target == word:
                break
            word = target
    for head in protected:
        participle_roots[head] = head
        pos_map.setdefault(head, {})["VERB"] = head


def include_participle_endpoints(pairs: list, lookup: dict, pos_map: dict,
                                lemma_pos: dict, references: dict,
                                participle_heads: set) -> dict:
    """Check a surviving tagged participle through its own source references.

    A standalone participle page may have no top-level pointer. If ordinary
    pair cleanup would keep it as a verb lemma, it still needs a finite verb
    endpoint. Do not turn every inflected participle page into a candidate.
    """
    candidates = {lemma for pair in pairs if pair.get("pos") == "verb"
                  for lemma in (pair.get("via"), pair["lemma"]) if lemma}
    endpoints = set()
    for lemma in candidates:
        target = resolve_pair_lemma(lemma, "verb", lookup, pos_map, lemma_pos)
        if (target in participle_heads and target not in references
                and lookup.get(target, (None,))[0] == target):
            endpoints.add(target)
    return {**references, **{word: set() for word in sorted(endpoints)}}


def clean_training_pairs(pairs: list, lookup: dict, preserve_cells=False,
                         pos_map: dict = None, lemma_pos: dict = None,
                         participle_roots: dict = None,
                         unresolved_verbs: set = frozenset(),
                         closed_class_heads: dict = None,
                         lexical_heads: dict = None,
                         canonical_references: dict = None,
                         adjective_heads: set = frozenset()) -> list:
    """Resolve page references before lookup homographs; omit internal metadata."""
    headwords = {key for key, (lemma, _) in lookup.items() if key == lemma}
    graph_heads = set((participle_roots or {}).values())
    result = []
    preserve_readings = preserve_cells or lexical_heads is not None
    for pair in deduplicate_pairs(pairs, preserve_readings):
        noun_owner = (preserve_cells and pair.get("pos") == "noun"
                      and pair.get("noun_owner") and not pair.get("via"))
        adjective_owner = (preserve_cells and pair.get("pos") == "adj"
                           and pair.get("adjective_owner") and not pair.get("via"))
        own_nominal = noun_owner or adjective_owner
        own_root = (preserve_cells and pair.get("pos") == "verb"
                    and pair.get("via") in unresolved_verbs
                    and pair["lemma"] in (participle_roots or {}))
        if (preserve_cells and pair.get("pos") == "verb"
                and not own_root
                and (pair.get("via") or pair["lemma"]) in unresolved_verbs):
            continue
        # A resolved source head owns its table even when another spelling
        # pointer is unresolved. Without that positive graph evidence the
        # unresolved alias remains a reason to omit the verb reading.
        candidates = ([pair["via"]] if pair.get("via") and not own_root
                      else [pair["lemma"]])
        if (pair.get("via") and not own_root and preserve_cells
                and pair.get("pos") == "verb"):
            # Some references name a missing verb spelling (κυμαίνω),
            # while another source supplies the actual verb (κυμαίνομαι).
            # The POS check below makes this fallback safe from homographs.
            candidates.append(pair["lemma"])
        for lemma in candidates:
            resolved = pair["lemma"] if own_nominal else resolve_pair_lemma(
                lemma, pair.get("pos", "") if preserve_readings else "name", lookup,
                pos_map if preserve_readings else None,
                lemma_pos if preserve_cells else None,
                participle_roots if preserve_cells else None,
                unresolved_verbs if preserve_cells else frozenset(),
                closed_class_heads if preserve_cells else None, lexical_heads,
                canonical_references, adjective_heads if preserve_cells else frozenset(),
                reference_target=(bool(pair.get("via"))
                                  or bool({"alt-of", "form-of"}
                                          & set(pair.get("tags", [])))))
            if (not own_nominal and resolved not in headwords
                    and not (pair.get("pos") == "verb" and resolved in graph_heads)
                    and not (preserve_cells and resolved in (closed_class_heads or {}).get(
                        pair.get("pos"), set()))
                    and not (preserve_cells and pair.get("pos") == "adj"
                             and resolved in adjective_heads)
                    and resolved not in (lexical_heads or {}).get(pair.get("pos"), set())):
                continue
            result.append({"form": pair["form"], "lemma": resolved,
                           "pos": pair.get("pos", ""), "tags": pair.get("tags", [])})
            break
    return deduplicate_pairs(result, True) if preserve_readings else result


def _cross_pos_alias_shadows_head(word: str, target: str,
                                  lexical_heads: dict, typed_references: dict,
                                  canonical_references: dict = None) -> bool:
    """An explicit other-POS reference cannot consume an independent head.

    The selected alias must be positively identified as another part of
    speech. A same-POS alternative spelling from the original word, or a
    same-POS reference from the selected target, retains canonicalization.
    An inflected homograph cannot consume the original word's own meaning.
    """
    head_pos = {pos for pos, heads in lexical_heads.items() if word in heads}
    references = typed_references.get(word, {})
    reference_pos = {pos for pos, targets in references.items() if target in targets}
    if not head_pos or not reference_pos or head_pos & reference_pos:
        return False
    for pos in head_pos:
        # Alternative spellings may canonicalize a lexical head. A form_of
        # reading has a different role: δίω is both an independent verb and
        # an inflected form of δείδω, so it must retain both readings.
        for seeds, graph in (
                ([target], typed_references),
                ((canonical_references or {}).get(word, {}).get(pos, set()),
                 canonical_references or {})):
            pending = list(seeds)
            seen = {word}
            while pending:
                candidate = pending.pop()
                if candidate in seen:
                    continue
                seen.add(candidate)
                if candidate in lexical_heads.get(pos, set()):
                    return False
                pending.extend(graph.get(candidate, {}).get(pos, set()) - seen)
    return True


def recover_cross_pos_heads(all_lookup: dict, lexical_heads: dict,
                            typed_references: dict,
                            canonical_references: dict = None) -> int:
    """Recover proven heads already displaced during per-source extraction.

    Only an explicit other-POS alias with no supported same-POS endpoint
    qualifies. A derived lookup key follows its head only when it still
    has that exact displaced owner and no competing recovery claims it.
    """
    if lexical_heads is None or typed_references is None:
        return 0
    heads = set().union(*lexical_heads.values()) if lexical_heads else set()
    proposals = {}
    for word in sorted(heads):
        target = all_lookup.get(word, (None,))[0]
        if (not target or target == word or not _cross_pos_alias_shadows_head(
                word, target, lexical_heads, typed_references,
                canonical_references)):
            continue
        for key in {word, word.lower(), to_monotonic(word),
                    to_monotonic(word).lower(), strip_accents(word.lower())}:
            if (all_lookup.get(key, (None,))[0] == target
                    and (key not in heads or key == word)):
                proposals.setdefault(key, set()).add(word)
    recovered = 0
    for key, words in proposals.items():
        if len(words) == 1:
            all_lookup[key] = (next(iter(words)), all_lookup[key][1])
            recovered += 1
    return recovered


def resolve_cross_source(all_lookup: dict, en_form_of_targets: dict,
                         el_form_of_targets: dict, en_headwords: set,
                         skip: set = frozenset(), preserve_cycles=False,
                         lexical_heads: dict = None, typed_references: dict = None,
                         canonical_references: dict = None) -> int:
    """Point each self-mapped key of the merged lookup at the lemma EN's
    form-of reference gives it, else EL's where EN has no page for the key
    (see main), and return how many moved. A target that maps back to the
    key is left alone: the cycle would make chain-breaking delete both
    lemmas with all their pairs (EN files ρωτώ as a form of ρωτάω, while
    EL's ρωτώ table lists ρωτάω, and both conjugations were lost)."""
    # Compute all proposed moves before changing any lookup. Otherwise two
    # reciprocal references that survived the per-source pass collapse onto
    # whichever synonym happens to occur second in the merged dictionary.
    moves = {}
    for k, (lemma, _) in list(all_lookup.items()):
        if lemma != k:
            continue  # not a self-map
        if k in skip:
            continue
        target = en_form_of_targets.get(k)
        if target is None and k not in en_headwords:
            # EN says nothing; only fall back to EL's target if EN
            # also doesn't claim this key as a real lemma headword.
            target = el_form_of_targets.get(k)
        if target and target != k:
            if (lexical_heads is not None and typed_references is not None
                    and _cross_pos_alias_shadows_head(k, target, lexical_heads,
                                                      typed_references,
                                                      canonical_references)):
                continue
            moves[k] = target
    cross_resolved = 0
    for k, target in moves.items():
        seen = {k}
        next_target = target
        while preserve_cycles and next_target in moves and next_target not in seen:
            seen.add(next_target)
            next_target = moves[next_target]
        if preserve_cycles and next_target in seen:
            continue
        if all_lookup.get(target, (None,))[0] != k:
            all_lookup[k] = (target, 1)
            cross_resolved += 1
    return cross_resolved


def published_pos_readings(resolved: dict, previous: dict, scalar_target: str,
                           closed_target_differs: bool = False,
                           adjective_heads: set = frozenset()) -> dict:
    """Publish an adjective override without exposing unrelated POS fallbacks.

    Previously published ambiguous or closed-class readings keep every POS.
    An adjective-only correction adds its own field to an otherwise omitted
    key; it does not newly certify that key's other internal table mappings.
    """
    if len(set(previous.values())) > 1 or closed_target_differs:
        return resolved
    adjective = resolved.get("ADJ")
    if adjective in adjective_heads and adjective != scalar_target:
        return {"ADJ": adjective}
    return {}


def main():
    parser = argparse.ArgumentParser(description="Build Dilemma training data")
    parser.add_argument("--kaikki", type=str, default=None,
                        help="Path to kaikki dump directory")
    parser.add_argument("--download", action="store_true",
                        help="Download kaikki dumps if missing")
    parser.add_argument("--lang", type=str, default="all",
                        choices=["el", "grc", "mgr", "all"],
                        help="Which language to build (default: all)")
    args = parser.parse_args()

    dump_dir = Path(args.kaikki) if args.kaikki else DEFAULT_DUMP_DIR

    if args.download:
        print("Checking/downloading kaikki dumps...")
        for lang_dumps in DUMPS.values():
            for filename in lang_dumps.values():
                download_dump(filename, dump_dir)
        print()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    langs = ["el", "grc", "mgr"] if args.lang == "all" else [args.lang]

    for lang in langs:
        lang_name = {"el": "Modern Greek", "grc": "Ancient Greek", "mgr": "Medieval Greek"}[lang]
        prefix = {"el": "mg", "grc": "ag", "mgr": "med"}[lang]
        print(f"\n{'='*50}")
        print(f"Building {lang_name} ({lang})")
        print(f"{'='*50}")

        all_pairs = []
        all_lookup = {}  # {form: (lemma, confidence)}
        all_pos_map = {}  # {form: {upos: lemma}} for POS disambiguation
        all_lemma_pos = {}
        closed_class_heads = {}
        closed_head_records = []
        lexical_heads = {} if lang == "grc" else None
        typed_references = {} if lang == "grc" else None
        canonical_references = {} if lang == "grc" else None
        closed_class_form_refs = {}
        participle_refs, verb_heads = {}, set()
        participle_heads = set()
        verb_refs = {}
        participle_verb_refs = {}
        participle_cells = []
        noun_heads, noun_source_hashes = [], {}
        adjective_heads, adjective_head_records = set(), []
        participle_roots, unresolved_verbs = {}, set()
        source_lookups = []  # (wikt_lang, lookup, headwords)

        # Per-source form_of_targets, kept separately so we can prefer
        # EN's resolution when a key is a real lemma headword in EN but
        # only a "dialect variant" listing in EL. After the per-source
        # passes have merged into all_lookup, we run a final pass-2
        # over the merged lookup using EN's form_of_targets first, then
        # EL's only when EN doesn't claim the key as a real headword.
        # This catches αυτούς (EL has no explicit form_of, EN has
        # senses[].form_of=αυτός) without trashing θεός (EN has a real
        # noun page, EL has misleading top-level form_of=[θεύς]).
        en_form_of_targets: dict[str, str] = {}
        el_form_of_targets: dict[str, str] = {}
        en_abbreviations = Abbreviations(set(), set())
        el_abbreviations = Abbreviations(set(), set())

        # Extract EN first to collect proper noun forms for corroboration
        en_name_forms = set()
        if "en" in DUMPS[lang]:
            en_path = resolve_dump(DUMPS[lang]["en"], dump_dir)
            print(f"\nScanning en Wiktionary: {DUMPS[lang]['en']}")
            pairs, lookup, headwords, pos_map, fot, abbr = extract_pairs(
                en_path, f"{lang}-en", lemma_pos=all_lemma_pos,
                participle_refs=participle_refs, verb_heads=verb_heads,
                participle_heads=participle_heads, verb_refs=verb_refs,
                participle_verb_refs=participle_verb_refs,
                participle_cells=participle_cells if lang == "el" else None,
                noun_heads=noun_heads if lang == "el" else None,
                adjective_heads=adjective_heads if lang == "el" else None,
                adjective_head_records=adjective_head_records if lang == "el" else None,
                closed_class_heads=closed_class_heads if lang == "el" else None,
                closed_head_records=closed_head_records if lang == "el" else None,
                lexical_heads=lexical_heads,
                typed_references=typed_references,
                canonical_references=canonical_references,
                closed_class_form_refs=closed_class_form_refs if lang == "el" else None)
            en_form_of_targets.update(fot)
            en_abbreviations = abbr
            all_pairs.extend(pairs)
            source_lookups.append(("en", lookup, headwords))
            # Merge POS map (first source wins per form+upos)
            merge_pos_maps(all_pos_map, pos_map, prefer_verb_refs=(lang == "el"))
            # Collect all proper noun forms from EN for corroboration
            if en_path.exists():
                with open(en_path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            e = json.loads(line)
                        except:
                            continue
                        if e.get("pos") == "name":
                            for fe in e.get("forms", []):
                                en_name_forms.add(fe.get("form", ""))

        # Extract remaining sources (EL, etc.) with proper noun plural filter
        for wikt_lang, filename in DUMPS[lang].items():
            if wikt_lang == "en":
                continue  # already processed
            path = resolve_dump(filename, dump_dir)
            print(f"\nScanning {wikt_lang} Wiktionary: {path.name}")
            pairs, lookup, headwords, pos_map, fot, abbr = extract_pairs(
                path, f"{lang}-{wikt_lang}",
                skip_name_plurals=en_name_forms if en_name_forms else None,
                lemma_pos=all_lemma_pos,
                participle_refs=participle_refs, verb_heads=verb_heads,
                participle_heads=participle_heads, verb_refs=verb_refs,
                participle_verb_refs=participle_verb_refs,
                participle_cells=participle_cells if lang == "el" else None,
                noun_heads=noun_heads if lang == "el" else None,
                adjective_heads=adjective_heads if lang == "el" else None,
                adjective_head_records=adjective_head_records if lang == "el" else None,
                closed_class_heads=closed_class_heads if lang == "el" else None,
                closed_head_records=closed_head_records if lang == "el" else None,
                lexical_heads=lexical_heads,
                typed_references=typed_references,
                canonical_references=canonical_references,
                closed_class_form_refs=closed_class_form_refs if lang == "el" else None)
            for k, v in fot.items():
                if k not in el_form_of_targets:
                    el_form_of_targets[k] = v
                    if k in abbr.keys:
                        el_abbreviations.keys.add(k)
            el_abbreviations.words.update(abbr.words)
            all_pairs.extend(pairs)
            source_lookups.append((wikt_lang, lookup, headwords))
            # Merge POS map
            merge_pos_maps(all_pos_map, pos_map, prefer_verb_refs=(lang == "el"))

        if lang == "el":
            import hashlib
            for source, _, _ in source_lookups:
                path = resolve_dump(DUMPS[lang][source], dump_dir)
                with path.open("rb") as stream:
                    digest = hashlib.sha256()
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                    noun_source_hashes[source] = digest.hexdigest()

        # Collect headword sets by source for confidence scoring
        en_headwords = set()
        el_headwords = set()
        for wikt_lang, _, headwords in source_lookups:
            if wikt_lang == "en":
                en_headwords = headwords
            else:
                el_headwords |= headwords  # merge all non-EN sources

        # First pass: merge all lookups (first wins for same confidence).
        # For MG, prefer EL Wiktionary lemma forms over EN - EL uses the
        # modern contracted forms (τρώω, λέω) that reflect actual usage,
        # while EN tends toward fuller morphological stems (τρώγω, λέγω).
        merge_order = source_lookups
        if lang == "el":
            merge_order = sorted(source_lookups, key=lambda x: (x[0] == "en",))
        merge_source_lookups(all_lookup, [lookup for _, lookup, _ in merge_order],
                             capitals_last=(lang == "el"),
                             lemma_pos=all_lemma_pos)

        # Cross-source self-map resolution. After per-source pass-2, any
        # remaining self-maps (key == lemma) at the merged level got
        # there because no individual source had a form_of/alt_of ref.
        # Apply EN's form_of_target first (sense-level form-of refs
        # from EN are the highest-quality "this is a form of X" signal),
        # then EL's only when EN didn't have a target for the key. This
        # fixes αυτούς (EN sense form_of=αυτός fires, EL had only a
        # gloss match without a ref) without trashing θεός (EN has no
        # form_of_target, so the bogus EL top-level form_of=[θεύς] is
        # never applied; θεός stays as the EN-attested headword).
        #
        # Article forms (ο, η, το, ...) are skipped: we want the
        # monotonic MG form to stay as the surface lemma. The polytonic
        # AG ancestor (ὁ) is reachable via etymology bridges for
        # callers that want it but should not surface as the MG lemma.
        _MG_ARTICLE_FORMS = {
            "ο", "η", "το", "τον", "την", "του", "της",
            "τους", "τις", "τα", "των", "τη", "οι",
        }
        # An abbreviation's reference moves no key that a page of either
        # Wiktionary presents as a word of its own (``Abbreviations``):
        # EN's δ, "abbreviation of δεσποινίς", leaves EL's letter δ alone.
        words = en_abbreviations.words | el_abbreviations.words
        for targets, abbreviations in (
                (en_form_of_targets, en_abbreviations),
                (el_form_of_targets, el_abbreviations)):
            dropped = drop_abbreviation_targets(targets, abbreviations, words)
            if dropped:
                print(f"Abbreviation targets dropped for words: {dropped:,}")
        # Reciprocal form references also arise between Ancient noun and
        # verb homographs (ἔργον / ἔργω). Keep their independent heads;
        # processing order must not collapse one into the other.
        recovered_heads = recover_cross_pos_heads(all_lookup, lexical_heads,
                                                   typed_references,
                                                   canonical_references)
        if recovered_heads:
            print(f"Independent head lookup keys recovered: {recovered_heads:,}")
        cross_resolved = resolve_cross_source(
            all_lookup, en_form_of_targets, el_form_of_targets,
            en_headwords, _MG_ARTICLE_FORMS, preserve_cycles=True,
            lexical_heads=lexical_heads, typed_references=typed_references,
            canonical_references=canonical_references)
        if cross_resolved:
            print(f"Cross-source form-of resolutions: {cross_resolved:,}")

        # Second pass: assign confidence tiers based on page presence
        # 5 = both EN + EL have pages for this form
        # 4 = EN page only
        # 3 = EL page only
        # 2 = EN + EL table entries agree on lemma (no page for form)
        # 1 = single source, table-only
        for k in list(all_lookup.keys()):
            lemma = all_lookup[k][0]
            in_en = k in en_headwords
            in_el = k in el_headwords
            if in_en and in_el:
                all_lookup[k] = (lemma, 5)
            elif in_en:
                all_lookup[k] = (lemma, 4)
            elif in_el:
                all_lookup[k] = (lemma, 3)
            else:
                # Check if both sources have this form in tables with same lemma
                en_lemma = None
                el_lemma = None
                for wikt_lang, lookup, _ in source_lookups:
                    if k in lookup:
                        if wikt_lang == "en":
                            en_lemma = lookup[k][0]
                        else:
                            el_lemma = lookup[k][0]
                if en_lemma and el_lemma and en_lemma == el_lemma:
                    all_lookup[k] = (lemma, 2)
                # else stays at 1

        # Deduplicate pairs
        unique_pairs = deduplicate_pairs(all_pairs, preserve_cells=(lang in {"el", "grc"}))
        if lang == "el":
            apply_closed_class_form_references(all_pos_map, closed_class_form_refs,
                                                closed_class_heads)
            for pos, heads in closed_class_heads.items():
                for head in heads:
                    all_pos_map.setdefault(head, {})[WIKT_TO_UPOS[pos]] = head

        # Break chained lookups: if form->lemma->X, the mapping is suspect.
        headwords = {k for k, (v, _) in all_lookup.items() if k == v}
        chains_broken = 0
        for k in list(all_lookup.keys()):
            lemma, conf = all_lookup[k]
            if lemma != k and lemma in all_lookup and all_lookup[lemma][0] != lemma:
                seen_chain = {k, lemma}
                target = all_lookup[lemma][0]
                depth = 0
                while target in all_lookup and all_lookup[target][0] != target and depth < 5:
                    if target in seen_chain:
                        break
                    seen_chain.add(target)
                    target = all_lookup[target][0]
                    depth += 1
                if target in headwords:
                    all_lookup[k] = (target, conf)
                else:
                    del all_lookup[k]
                chains_broken += 1
        print(f"Chained lookups fixed: {chains_broken}")

        # Recompute headwords after chain-breaking
        headwords = {k for k, (v, _) in all_lookup.items() if k == v}

        if lang == "el":
            # Lookup cleanup does not alter POS/reference evidence. Resolve
            # the original graph after identifying standalone participles
            # that would otherwise remain verb lemmas in the final pairs.
            participle_refs = include_participle_endpoints(
                unique_pairs, all_lookup, all_pos_map, all_lemma_pos,
                participle_refs, participle_heads)
            unresolved_verbs = apply_participle_references(
                all_pos_map, participle_refs, verb_heads, participle_heads,
                all_lemma_pos, verb_refs, participle_verb_refs)
            participle_roots = {word: all_pos_map[word]["VERB"]
                                for word in participle_refs
                                if word not in unresolved_verbs}
            preserve_finite_verb_heads(
                all_pos_map, all_lookup,
                verb_heads - participle_heads - participle_refs.keys(),
                participle_roots, unresolved_verbs)
            noun_inventory = noun_head_inventory(noun_heads, all_lookup, noun_source_hashes)
            validate_noun_head_inventory(noun_inventory)
            noun_heads_path = DATA_DIR / "el_noun_heads.json"
            with noun_heads_path.open("w", encoding="utf-8") as f:
                json.dump(noun_inventory, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print(f"Independent noun heads: {len(noun_inventory['heads']):,}")
            print(f"  -> {noun_heads_path}")
            adjective_inventory = adjective_head_inventory(
                adjective_head_records, all_lookup, noun_source_hashes)
            validate_adjective_head_inventory(adjective_inventory)
            adjective_path = DATA_DIR / "el_adjective_heads.json"
            adjective_path.write_text(json.dumps(adjective_inventory, ensure_ascii=False,
                                                  indent=2) + "\n", encoding="utf-8")
            print(f"Independent adjective heads: {len(adjective_inventory['heads']):,}")
            print(f"  -> {adjective_path}")
            closed_inventory = closed_class_head_inventory(
                closed_head_records, all_lookup, noun_source_hashes)
            validate_closed_class_head_inventory(closed_inventory)
            closed_path = DATA_DIR / "el_closed_class_heads.json"
            with closed_path.open("w", encoding="utf-8") as f:
                json.dump(closed_inventory, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print(f"Independent closed-class heads: {len(closed_inventory['heads']):,}")
            print(f"  -> {closed_path}")
            inventory = standalone_participle_inventory(
                participle_cells, unresolved_verbs)
            participles_path = DATA_DIR / "el_participles.json"
            with open(participles_path, "w", encoding="utf-8") as f:
                json.dump(inventory, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print(f"Standalone participle cells: {len(inventory['cells']):,}")
            print(f"  -> {participles_path}")

        # Fix training pairs: rewrite lemmas to match the cleaned lookup,
        # drop pairs whose lemma can't be resolved to a headword.
        unique_pairs = clean_training_pairs(unique_pairs, all_lookup,
                                             preserve_cells=(lang == "el"),
                                             pos_map=all_pos_map,
                                             lemma_pos=all_lemma_pos,
                                             participle_roots=participle_roots,
                                             unresolved_verbs=unresolved_verbs,
                                             closed_class_heads=closed_class_heads if lang == "el" else None,
                                             lexical_heads=lexical_heads,
                                             canonical_references=canonical_references,
                                             adjective_heads=adjective_heads if lang == "el" else frozenset())

        if lang == "grc":
            inventory = ancient_reading_head_inventory(
                unique_pairs, all_lookup, {source: resolve_dump(filename, dump_dir)
                                          for source, filename in DUMPS[lang].items()})
            validate_ancient_reading_head_inventory(inventory)
            head_path = DATA_DIR / "grc_reading_heads.json"
            head_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
            print(f"Independent non-flat Ancient reading heads: {len(inventory['heads']):,}")

        # Save training pairs (after cleanup)
        pairs_path = DATA_DIR / f"{prefix}_pairs.json"
        with open(pairs_path, "w", encoding="utf-8") as f:
            json.dump(unique_pairs, f, ensure_ascii=False, indent=2)
        size_mb = pairs_path.stat().st_size / (1024 * 1024)
        print(f"Training pairs: {len(unique_pairs)} ({size_mb:.1f} MB)")
        print(f"  -> {pairs_path}")

        # Confidence stats
        conf_counts = Counter(c for _, c in all_lookup.values())
        print(f"Confidence tiers:")
        for tier, label in [(5, "both pages"), (4, "EN page"), (3, "EL page"),
                             (2, "corroborated"), (1, "table-only")]:
            if conf_counts[tier]:
                print(f"  {tier}: {conf_counts[tier]:>10,}  {label}")

        # Dialect tag stats (from training pairs)
        dialect_counts = Counter()
        for p in unique_pairs:
            for t in p.get("tags", []):
                if t in DIALECT_PREFIXES:
                    dialect_counts[t] += 1
        if dialect_counts:
            print(f"Dialect-tagged pairs:")
            for dialect, count in dialect_counts.most_common():
                print(f"  {dialect:20s} {count:>8,}")

        # Save lookup table (flatten to {form: lemma})
        flat_lookup = {k: v[0] for k, v in all_lookup.items()}
        lookup_path = DATA_DIR / f"{prefix}_lookup.json"
        with open(lookup_path, "w", encoding="utf-8") as f:
            json.dump(flat_lookup, f, ensure_ascii=False, separators=(",", ":"))
        size_mb = lookup_path.stat().st_size / (1024 * 1024)
        print(f"Lookup table: {len(flat_lookup)} entries ({size_mb:.1f} MB)")
        print(f"  -> {lookup_path}")

        # Save scored lookup table (preserves confidence tiers for downstream consumers)
        scored_lookup = {k: {"lemma": v[0], "confidence": v[1]}
                         for k, v in all_lookup.items()}
        scored_path = DATA_DIR / f"{prefix}_lookup_scored.json"
        with open(scored_path, "w", encoding="utf-8") as f:
            json.dump(scored_lookup, f, ensure_ascii=False, separators=(",", ":"))
        scored_mb = scored_path.stat().st_size / (1024 * 1024)
        print(f"Scored lookup: {len(scored_lookup)} entries ({scored_mb:.1f} MB)")
        print(f"  -> {scored_path}")

        # Also write to SQLite for fast loading by build_lookup_db.py
        raw_db_path = DATA_DIR / "raw_lookups.db"
        import sqlite3 as _sql
        rdb = _sql.connect(str(raw_db_path))
        table = {"ag": "ag", "mg": "mg", "med": "med"}.get(prefix, prefix)
        rdb.execute(f"DROP TABLE IF EXISTS {table}")
        rdb.execute(f"CREATE TABLE {table} (form TEXT NOT NULL, lemma TEXT NOT NULL)")
        rdb.executemany(f"INSERT INTO {table} (form, lemma) VALUES (?, ?)",
                        flat_lookup.items())
        rdb.execute(f"CREATE INDEX idx_{table}_form ON {table} (form)")
        rdb.commit()
        rdb.close()
        print(f"  -> {raw_db_path} (table: {table})")

        # Filter POS map to only genuinely ambiguous forms (different
        # lemmas for different POS tags). Also resolve lemmas through
        # the cleaned lookup to use final headword forms.
        ambiguous_pos = {}
        for form, upos_lemmas in all_pos_map.items():
            # Resolve each lemma through the flat lookup
            resolved = {}
            for upos, lemma in upos_lemmas.items():
                if (lang == "el" and upos == "VERB"
                        and (form in unresolved_verbs or lemma in unresolved_verbs)):
                    continue
                source_lemma = (form if lang == "el" and upos == "VERB"
                                and form in participle_roots else lemma)
                if source_lemma in all_lookup or source_lemma in participle_roots:
                    target = resolve_pair_lemma(
                        source_lemma, next((p for p, u in WIKT_TO_UPOS.items() if u == upos), "")
                        if lang in {"el", "grc"} else "name", all_lookup,
                        all_pos_map if lang in {"el", "grc"} else None,
                        all_lemma_pos if lang == "el" else None,
                        participle_roots if lang == "el" else None,
                        unresolved_verbs if lang == "el" else frozenset(),
                        closed_class_heads if lang == "el" else None, lexical_heads,
                        canonical_references, adjective_heads if lang == "el" else frozenset())
                    if target is not None:
                        resolved[upos] = target
                else:
                    resolved[upos] = lemma
            # Only keep if there are multiple distinct lemmas
            # Agreement between typed readings does not make the scalar
            # fallback correct: τη/την can both point to ο while the
            # unqualified key still points to the form itself.
            closed_target_differs = lang == "el" and any(
                upos in resolved and resolved[upos] != all_lookup.get(form, (form,))[0]
                for upos in ("DET", "PRON"))
            previous = dict(resolved)
            if lang == "el" and resolved.get("ADJ") in adjective_heads:
                # Modern adjective resolution previously used only this flat
                # target. Preserve the old publication of all other POS fields.
                previous["ADJ"] = resolve_pair_lemma(
                    upos_lemmas["ADJ"], "adj", all_lookup)
            published = published_pos_readings(
                resolved, previous, all_lookup.get(form, (form,))[0],
                closed_target_differs, adjective_heads if lang == "el" else frozenset())
            if published:
                ambiguous_pos[form] = published

        pos_lookup_path = DATA_DIR / f"{prefix}_pos_lookup.json"
        with open(pos_lookup_path, "w", encoding="utf-8") as f:
            json.dump(ambiguous_pos, f, ensure_ascii=False, separators=(",", ":"))
        size_kb = pos_lookup_path.stat().st_size / 1024
        print(f"POS lookup: {len(ambiguous_pos)} ambiguous forms ({size_kb:.0f} KB)")
        print(f"  -> {pos_lookup_path}")

        # Stats
        unique_lemmas = len(set(v[0] for v in all_lookup.values()))
        print(f"Unique lemmas: {unique_lemmas}")


if __name__ == "__main__":
    main()
