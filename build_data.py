#!/usr/bin/env python3
"""Build training data for Dilemma from Wiktionary kaikki JSONL dumps.

Scans both EN and EL Wiktionary dumps for Greek entries, extracts every
inflected form -> lemma pair from inflection tables, form_of/alt_of
references, and dialect-tagged paradigms. Produces:
  - data/mg_pairs.json: Modern Greek form->lemma training pairs
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
                proper_noun: bool = False, closed_class_capital: bool = False):
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
    for key in (form, form.lower(), to_monotonic(form), to_monotonic(form).lower(),
                strip_accents(form.lower())):
        if not key:
            continue
        if closed_class_capital and key != form:
            continue
        conf = confidence
        if proper_noun and key != form:
            conf = max(0, confidence - 1)
        existing = lookup.get(key)
        if existing is None or conf > existing[1]:
            lookup[key] = (lemma, conf)


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


def extract_pairs(jsonl_path: Path, lang: str,
                   skip_name_plurals: set = None) -> tuple[list[dict], dict, set, dict, dict]:
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
    """
    if not jsonl_path.exists():
        print(f"  {lang}: not found at {jsonl_path}")
        return [], {}, set(), {}

    pairs = []
    page_headwords = set()
    lookup = {}
    # form_of/alt_of targets: {key_variant: target_lemma}
    # Used in pass 2 to resolve self-map artifacts.
    form_of_targets = {}
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

    # First pass: collect closed-class headwords and article forms.
    # Article forms leak into Katharevousa noun/adj declension templates
    # (e.g. γωνιακός lists τῶν as its genitive plural "form").
    closed_class_headwords = set()
    article_forms = set()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            pos = entry.get("pos", "")
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
            is_form_of_page = bool(top_form_of_for_detection
                                   or top_alt_of_for_detection
                                   or sense_has_form_of)
            if not is_form_of_page:
                # Check gloss for "X of Y" pattern (kaikki often doesn't
                # populate form_of for AG form-of pages)
                senses = entry.get("senses", [])
                if senses:
                    gloss = (senses[0].get("glosses") or [""])[0].lower()
                    _form_of_patterns = (
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
                    if any(p in gloss for p in _form_of_patterns):
                        is_form_of_page = True
            hw_confidence = 1 if is_form_of_page else 3
            # Closed-class capitals like Αυτής (formal pronoun) must not
            # hijack the lowercase αυτής key from regular αυτής -> αυτός.
            _cc_cap = (pos in ("pron", "det", "adj")
                       and lemma and lemma != lemma.lower())
            _add_lookup(lookup, lemma, lemma, confidence=hw_confidence,
                        proper_noun=(pos == "name"),
                        closed_class_capital=_cc_cap)

            has_data = False

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
            for sense in entry.get("senses", []):
                sense_form_of.extend(sense.get("form_of", []))
            form_of_refs = sense_form_of + list(top_form_of_for_detection)

            for ref in form_of_refs:
                ref_word = strip_length_marks(ref.get("word", ""))
                if not ref_word or not _is_greek(ref_word.replace(" ", "")):
                    continue
                if " " in ref_word:  # skip multi-word
                    continue
                # This entry (lemma) is a form of ref_word
                _add_lookup(lookup, lemma, ref_word, proper_noun=(pos == "name"))
                # Record for pass-2 self-map resolution (skip proper nouns
                # since variant names like Βησσαρίων should self-map)
                if pos != "name" and lemma != ref_word:
                    for key in (lemma, lemma.lower(),
                                to_monotonic(lemma), to_monotonic(lemma).lower(),
                                strip_accents(lemma.lower())):
                        if key and key not in form_of_targets:
                            form_of_targets[key] = ref_word
                _add_pos_map(pos_map, lemma, ref_word, pos)
                pairs.append({
                    "form": lemma,
                    "lemma": ref_word,
                    "pos": pos,
                    "tags": ["form-of"],
                })
                form_of_count += 1
                has_data = True

            # --- Source 2: alt_of (alternative form of another word) ---
            # For EL Wiktionary, top-level alt_of has the same etymology
            # / variant-pointer semantics as top-level form_of (see above
            # comment). Use sense-level alt_of refs as the primary signal
            # and the (already filtered) top_alt_of_for_detection list
            # for non-EL sources.
            sense_alt_of = []
            for sense in entry.get("senses", []):
                sense_alt_of.extend(sense.get("alt_of", []))
            alt_of_refs = sense_alt_of + list(top_alt_of_for_detection)

            for ref in alt_of_refs:
                ref_word = strip_length_marks(ref.get("word", ""))
                if not ref_word or not _is_greek(ref_word.replace(" ", "")):
                    continue
                if " " in ref_word:
                    continue
                _add_lookup(lookup, lemma, ref_word, proper_noun=(pos == "name"))
                if pos != "name" and lemma != ref_word:
                    for key in (lemma, lemma.lower(),
                                to_monotonic(lemma), to_monotonic(lemma).lower(),
                                strip_accents(lemma.lower())):
                        if key and key not in form_of_targets:
                            form_of_targets[key] = ref_word
                _add_pos_map(pos_map, lemma, ref_word, pos)
                pairs.append({
                    "form": lemma,
                    "lemma": ref_word,
                    "pos": pos,
                    "tags": ["alt-of"],
                })
                alt_of_count += 1
                has_data = True

            # --- Source 3: forms[] (inflection table) ---
            forms = entry.get("forms", [])
            if not forms:
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
                        _add_lookup(lookup, form, form, confidence=3)
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
                if uses_shared_person_template:
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
                    continue

                if any(t in skip_tags for t in tags):
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

                # Add dialect tag if we're inside a dialect-specific table section
                if current_dialect and current_dialect not in morph_tags:
                    morph_tags.append(current_dialect)
                    dialect_tagged += 1

                pairs.append({
                    "form": form,
                    "lemma": lemma,
                    "pos": pos,
                    "tags": morph_tags,
                })

                # Also add monotonic version as a training pair
                mono = to_monotonic(form)
                if mono != form:
                    pairs.append({
                        "form": mono,
                        "lemma": lemma,
                        "pos": pos,
                        "tags": morph_tags,
                    })

                # Lookup: original, lowercase, monotonic, accent-stripped
                _form_cc_cap = (pos in ("pron", "det", "adj")
                                and form and form != form.lower())
                _add_lookup(lookup, form, lemma, proper_noun=(pos == "name"),
                            closed_class_capital=_form_cc_cap)
                _add_pos_map(pos_map, form, lemma, pos)

                # Also add the parenthetical-expanded form (e.g. ἐστίν from ἐστί(ν))
                if extra_form and _is_greek(extra_form):
                    _extra_cc_cap = (pos in ("pron", "det", "adj")
                                     and extra_form
                                     and extra_form != extra_form.lower())
                    _add_lookup(lookup, extra_form, lemma,
                                proper_noun=(pos == "name"),
                                closed_class_capital=_extra_cc_cap)
                    _add_pos_map(pos_map, extra_form, lemma, pos)
                    pairs.append({
                        "form": extra_form,
                        "lemma": lemma,
                        "pos": pos,
                        "tags": morph_tags,
                    })

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
            lookup[key] = (target, conf)
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
    return pairs, lookup, page_headwords, pos_map, form_of_targets


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

        # Extract EN first to collect proper noun forms for corroboration
        en_name_forms = set()
        if "en" in DUMPS[lang]:
            en_path = resolve_dump(DUMPS[lang]["en"], dump_dir)
            print(f"\nScanning en Wiktionary: {DUMPS[lang]['en']}")
            pairs, lookup, headwords, pos_map, fot = extract_pairs(en_path, f"{lang}-en")
            en_form_of_targets.update(fot)
            all_pairs.extend(pairs)
            source_lookups.append(("en", lookup, headwords))
            # Merge POS map (first source wins per form+upos)
            for form, upos_lemmas in pos_map.items():
                if form not in all_pos_map:
                    all_pos_map[form] = {}
                for upos, lemma in upos_lemmas.items():
                    if upos not in all_pos_map[form]:
                        all_pos_map[form][upos] = lemma
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
            pairs, lookup, headwords, pos_map, fot = extract_pairs(
                path, f"{lang}-{wikt_lang}",
                skip_name_plurals=en_name_forms if en_name_forms else None)
            for k, v in fot.items():
                if k not in el_form_of_targets:
                    el_form_of_targets[k] = v
            all_pairs.extend(pairs)
            source_lookups.append((wikt_lang, lookup, headwords))
            # Merge POS map
            for form, upos_lemmas in pos_map.items():
                if form not in all_pos_map:
                    all_pos_map[form] = {}
                for upos, lemma in upos_lemmas.items():
                    if upos not in all_pos_map[form]:
                        all_pos_map[form][upos] = lemma

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
        for _, lookup, _ in merge_order:
            for k, (lemma, _) in lookup.items():
                if k not in all_lookup:
                    all_lookup[k] = (lemma, 1)

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
        cross_resolved = 0
        for k, (lemma, _) in list(all_lookup.items()):
            if lemma != k:
                continue  # not a self-map
            if k in _MG_ARTICLE_FORMS:
                continue
            target = en_form_of_targets.get(k)
            if target is None and k not in en_headwords:
                # EN says nothing; only fall back to EL's target if EN
                # also doesn't claim this key as a real lemma headword.
                target = el_form_of_targets.get(k)
            if target and target != k:
                all_lookup[k] = (target, 1)
                cross_resolved += 1
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
        seen = set()
        unique_pairs = []
        for p in all_pairs:
            key = (p["form"], p["lemma"])
            if key not in seen:
                seen.add(key)
                unique_pairs.append(p)

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

        # Fix training pairs: rewrite lemmas to match the cleaned lookup,
        # drop pairs whose lemma can't be resolved to a headword.
        clean_pairs = []
        pairs_fixed = 0
        pairs_dropped = 0
        for p in unique_pairs:
            lemma = p["lemma"]
            if lemma in headwords:
                clean_pairs.append(p)
            elif lemma in all_lookup:
                resolved = all_lookup[lemma][0]
                if resolved in headwords:
                    clean_pairs.append({"form": p["form"], "lemma": resolved,
                                        "pos": p.get("pos", ""), "tags": p.get("tags", [])})
                    pairs_fixed += 1
                else:
                    pairs_dropped += 1
            else:
                pairs_dropped += 1
        if pairs_fixed or pairs_dropped:
            print(f"Training pairs: {pairs_fixed} lemmas resolved, {pairs_dropped} dropped")
            unique_pairs = clean_pairs

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
                if lemma in all_lookup:
                    resolved[upos] = all_lookup[lemma][0]
                else:
                    resolved[upos] = lemma
            # Only keep if there are multiple distinct lemmas
            if len(set(resolved.values())) > 1:
                ambiguous_pos[form] = resolved

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
