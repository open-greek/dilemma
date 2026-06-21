"""Dilemma - Greek lemmatizer.

Fast lookup table for known forms, custom transformer model for unknown forms.

Usage:
    from dilemma import Dilemma

    m = Dilemma()                        # loads lookup table + model
    m.lemmatize("πάθης")                # -> "παθαίνω"
    m.lemmatize("πολεμούσαν")           # -> "πολεμώ"
    m.lemmatize_batch(["δώση", "σκότωσε"])  # -> ["δίνω", "σκοτώνω"]

    # Elision expansion (uses Wiktionary lookup)
    m.lemmatize("ἀλλ̓")                  # -> "ἀλλά"
    m.lemmatize("ἔφατ̓")                 # -> "φημί"

    # Verbose mode: returns all candidates with metadata
    m.lemmatize_verbose("ἔριδι")
    # -> [LemmaCandidate(lemma="ἔρις", lang="grc", proper=False),
    #     LemmaCandidate(lemma="Ἔρις", lang="grc", proper=True)]

    m.lemmatize_verbose("πόλεμο")
    # -> [LemmaCandidate(lemma="πόλεμος", lang="el"),
    #     LemmaCandidate(lemma="πόλεμος", lang="grc")]

    # Convention remapping: LSJ dictionary headwords
    m_lsj = Dilemma(convention="lsj")
    m_lsj.lemmatize("αἰνῶς")        # -> "αἰνός" (adverb -> adjective)
    m_lsj.lemmatize("εἶπον")        # -> "λέγω" (aorist -> present stem)
"""

import json
import math
import os
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


def _resolve_data_dir() -> Path:
    """Return the directory holding Dilemma data files.

    Resolution order:
      1. $DILEMMA_DATA_DIR (if set and existing)
      2. ~/.cache/dilemma/data/ (if existing)
      3. <repo-root>/data/ (dev mode; repo root is the package's parent)
      4. <package>/data/ (if someone bundled data inside the install)
      5. Fallback: ~/.cache/dilemma/data/ even if it doesn't exist yet,
         so callers get a stable path to write to.
    """
    env = os.environ.get("DILEMMA_DATA_DIR")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    cache = Path.home() / ".cache" / "dilemma" / "data"
    if cache.exists():
        return cache
    dev = Path(__file__).resolve().parent.parent / "data"
    if dev.exists():
        return dev
    bundled = Path(__file__).resolve().parent / "data"
    if bundled.exists():
        return bundled
    return cache


def _resolve_model_dir() -> Path:
    """Return the directory holding trained model files. Same order as data."""
    env = os.environ.get("DILEMMA_MODEL_DIR")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    cache = Path.home() / ".cache" / "dilemma" / "model"
    if cache.exists():
        return cache
    dev = Path(__file__).resolve().parent.parent / "model"
    if dev.exists():
        return dev
    bundled = Path(__file__).resolve().parent / "model"
    if bundled.exists():
        return bundled
    return cache


DATA_DIR = _resolve_data_dir()
MODEL_DIR = _resolve_model_dir()
LOOKUP_DB_PATH = DATA_DIR / "lookup.db"
SPELL_INDEX_PATH = DATA_DIR / "spell_index.db"
LSJ9_POS_LOOKUP_PATH = DATA_DIR / "lsj9_pos_lookup.json"
LSJ9_INDECLINABLES_PATH = DATA_DIR / "lsj9_indeclinables.json"
LSJ9_FREQUENCY_PATH = DATA_DIR / "lsj9_frequency.json"
LOOKUP_PATH = DATA_DIR / "mg_lookup.json"
AG_LOOKUP_PATH = DATA_DIR / "ag_lookup.json"
MED_LOOKUP_PATH = DATA_DIR / "med_lookup.json"
MG_POS_LOOKUP_PATH = DATA_DIR / "mg_pos_lookup.json"
AG_POS_LOOKUP_PATH = DATA_DIR / "ag_pos_lookup.json"
TREEBANK_POS_LOOKUP_PATH = DATA_DIR / "treebank_pos_lookup.json"
GLAUX_POS_LOOKUP_PATH = DATA_DIR / "glaux_pos_lookup.json"
LSJ_HEADWORDS_PATH = DATA_DIR / "lsj_headwords.json"
CUNLIFFE_HEADWORDS_PATH = DATA_DIR / "cunliffe_headwords.json"
MG_HEADWORDS_PATH = DATA_DIR / "mg_headwords.json"
AG_HEADWORDS_PATH = DATA_DIR / "ag_headwords.json"
DGE_HEADWORDS_PATH = DATA_DIR / "dge_headwords.json"
LGPN_NAMES_PATH = DATA_DIR / "lgpn_names.json"
LEMMA_EQUIVALENCES_PATH = DATA_DIR / "lemma_equivalences.json"
CORPUS_FREQ_PATH = DATA_DIR / "corpus_freq.json"
ATTESTATION_PATH = DATA_DIR / "lemma_attestation.json"
CONVENTION_DIR = DATA_DIR

_VALID_CONVENTIONS = {None, "lsj", "cunliffe", "triantafyllidis", "wiktionary"}

# Map convention name -> headword file path for auto-derivation.
# Conventions not listed here use LSJ headwords as fallback.
_CONVENTION_HEADWORDS = {
    "lsj": LSJ_HEADWORDS_PATH,
    "cunliffe": CUNLIFFE_HEADWORDS_PATH,
    "triantafyllidis": MG_HEADWORDS_PATH,
}

# Conventions that output monotonic Greek (apply to_monotonic to all results).
_MONOTONIC_CONVENTIONS = {"triantafyllidis"}


_POLYTONIC_STRIP = {0x0313, 0x0314, 0x0345, 0x0306, 0x0304}
_POLYTONIC_TO_ACUTE = {0x0300, 0x0342}

# Elision mark: U+0313 COMBINING COMMA ABOVE (repurposed as apostrophe
# in polytonic Greek text). Also handle right single quote U+2019 and
# modifier letter apostrophe U+02BC.
_ELISION_MARKS = {"\u0313", "\u2019", "\u02BC", "'", "\u1FBD", "\u02B9", "\u1FBF"}

# Vowels to try when expanding elision (ordered by frequency in AG text)
_GREEK_VOWELS = "αεοιηυω"

# Article and pronoun resolution: maps forms to canonical lemma.
# Used when resolve_articles=True (for treebank evaluation).
_ARTICLE_LEMMA = "ὁ"
_ARTICLE_FORMS = {
    # Polytonic
    "ὁ", "ἡ", "τό", "τοῦ", "τῆς", "τῶν", "τόν", "τήν",
    "τά", "τοῖς", "ταῖς", "τῷ", "τῇ", "τούς", "τάς", "τοῖν", "ταῖν",
    "οἱ", "αἱ", "τώ",
    # Grave variants
    "τὸ", "τοὺς", "τὰ", "τὸν", "τὴν", "τὰς", "αἵ", "οἵ",
    # Monotonic
    "ο", "η", "το", "του", "της", "των", "τον", "την",
    "τα", "τους", "τοις", "οι", "αι",
    # Stripped (no accents/breathings)
    "τω", "ται",
}

_PRONOUN_LEMMAS = {
    # 1st person -> ἐγώ
    "μοι": "ἐγώ", "μοί": "ἐγώ", "μου": "ἐγώ", "με": "ἐγώ",
    "ἐμοί": "ἐγώ", "ἐμοῦ": "ἐγώ", "ἐμέ": "ἐγώ",
    "ἡμεῖς": "ἐγώ", "ἡμῶν": "ἐγώ", "ἡμῖν": "ἐγώ", "ἡμᾶς": "ἐγώ",
    # 2nd person -> σύ
    "σοι": "σύ", "σοί": "σύ", "σου": "σύ", "σε": "σύ",
    "σοῦ": "σύ",
    "ὑμεῖς": "σύ", "ὑμῶν": "σύ", "ὑμῖν": "σύ", "ὑμᾶς": "σύ",
}

# Modern Greek closed-class resolution (monotonic).
# Used when convention='triantafyllidis' to intercept MG function words
# before the AG lookup cascade, which otherwise misresolves them
# (e.g. στη -> ἵστημι, σε -> σύ, τις -> τις self-map).
_MG_ARTICLE_FORMS = {
    "ο", "η", "το", "τον", "την", "του", "της", "τα", "τους", "τις",
    "των", "τη", "οι",
}
_MG_ARTICLE_LEMMA = "ο"

# ---- Feature: Particle/enclitic suffix stripping ----
# Ancient Greek appends particles like -περ, -γε, -δε, and deictic -ι to
# words without changing the lemma. Ordered longest-first for greedy match.
_PARTICLE_SUFFIXES = ["περ", "γε", "δε"]

# Deictic -ι is only stripped from demonstrative pronoun stems.
# These are accent-stripped stems of common demonstratives (οὗτος, ὅδε, etc.)
_DEICTIC_STEMS = {
    "τουτο", "τουτω", "τουτοισ", "τουτου", "τουτων", "τουτοι",
    "ταυτη", "ταυτα", "ταυτησ", "ταυτων", "ταυται", "ταυταισ",
    "τουτουσ", "τουτοισι", "ταυτασ", "ταυτησι",
    "τοδ", "τηδ", "τωδ", "τονδ", "τηνδ",
    "ενθαδ",  # ἐνθάδε is a real word but ἐνθαδί also exists
}

# ---- Feature: Article-agreement disambiguation ----
# Maps article forms to (gender, number, case) features for ranking
# ambiguous lemma candidates. "m"=masculine, "f"=feminine, "n"=neuter;
# "s"=singular, "p"=plural, "d"=dual; case abbreviated.
_ARTICLE_FEATURES = {
    # Masculine
    "ὁ": ("m", "s", "nom"), "τοῦ": ("m", "s", "gen"),
    "τῷ": ("m", "s", "dat"), "τόν": ("m", "s", "acc"),
    "οἱ": ("m", "p", "nom"), "τῶν": ("m", "p", "gen"),
    "τοῖς": ("m", "p", "dat"), "τούς": ("m", "p", "acc"),
    # Feminine
    "ἡ": ("f", "s", "nom"), "τῆς": ("f", "s", "gen"),
    "τῇ": ("f", "s", "dat"), "τήν": ("f", "s", "acc"),
    "αἱ": ("f", "p", "nom"),
    "ταῖς": ("f", "p", "dat"), "τάς": ("f", "p", "acc"),
    # Neuter
    "τό": ("n", "s", "nom"), "τά": ("n", "p", "nom"),
    # Grave variants
    "τὸ": ("n", "s", "nom"), "τὰ": ("n", "p", "nom"),
    "τὸν": ("m", "s", "acc"), "τὴν": ("f", "s", "acc"),
    "τοὺς": ("m", "p", "acc"), "τὰς": ("f", "p", "acc"),
}

# Gender of common lemma endings (heuristic for AG nouns/adjectives)
_LEMMA_GENDER_HINTS = {
    # Masculine endings
    "ος": "m", "ής": "m", "εύς": "m", "ηρ": "m", "ων": "m",
    "ας": "m", "ής": "m",
    # Feminine endings
    "η": "f", "α": "f", "ις": "f", "ύς": "f",
    # Neuter endings
    "ον": "n", "ος": "m", "μα": "n", "ον": "n",
}

_MG_CLOSED_CLASS = {
    # Preposition contractions σε + article -> σε
    "στη": "σε", "στην": "σε", "στο": "σε", "στον": "σε",
    "στου": "σε", "στους": "σε", "στις": "σε", "στα": "σε",
    "στης": "σε", "στων": "σε",
    # Indefinite article -> ένας
    "ένα": "ένας", "μία": "ένας", "μια": "ένας",
    "ενός": "ένας", "μίας": "ένας", "μιας": "ένας",
    # Copula -> είμαι
    "είναι": "είμαι", "ήταν": "είμαι", "ήμουν": "είμαι",
    "ήσουν": "είμαι", "ήμαστε": "είμαι", "ήσαστε": "είμαι",
    # Common auxiliaries -> έχω
    "έχει": "έχω", "είχε": "έχω", "έχουν": "έχω",
    "είχαν": "έχω", "έχεις": "έχω", "είχες": "έχω",
    "έχουμε": "έχω", "είχαμε": "έχω",
    # Demonstrative/personal pronoun αυτός (MG Wiktionary has bad
    # mappings like αυτών->τα due to article/pronoun conflation)
    "αυτό": "αυτός", "αυτόν": "αυτός", "αυτήν": "αυτός",
    "αυτού": "αυτός", "αυτής": "αυτός", "αυτών": "αυτός",
    "αυτούς": "αυτός", "αυτές": "αυτός", "αυτοί": "αυτός",
    "αυτά": "αυτός",
}


@dataclass
class LemmaCandidate:
    """A lemma candidate with metadata for disambiguation."""
    lemma: str
    lang: str = ""       # "el" (SMG), "grc" (AG), "med" (Medieval), "" (unknown)
    proper: bool = False  # True if lemma is a proper noun (capitalized headword)
    source: str = ""      # "lookup", "elision", "crasis", "model", "article"
    score: float = 1.0    # confidence (1.0 for lookup, lower for model)
    via: str = ""         # how the lookup matched: "exact", "lower", "mono",
                          # "stripped", "elision:ε" (which vowel expanded), etc.


def to_monotonic(s: str) -> str:
    """Convert polytonic Greek to monotonic."""
    nfd = unicodedata.normalize("NFD", s)
    out = []
    for ch in nfd:
        cp = ord(ch)
        if cp in _POLYTONIC_STRIP:
            continue
        if cp in _POLYTONIC_TO_ACUTE:
            out.append("\u0301")
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def grave_to_acute(s: str) -> str:
    """Convert grave accents to acute, preserving all other diacritics.

    In Greek orthography, grave (βαρεῖα) is a positional variant of acute —
    it appears on the last syllable when followed by another word. So ὣς = ὡς,
    τὸν = τόν, etc. This is a lighter normalization than to_monotonic(), which
    also strips breathings and circumflex.
    """
    nfd = unicodedata.normalize("NFD", s)
    out = []
    for ch in nfd:
        if ord(ch) == 0x0300:  # COMBINING GRAVE ACCENT
            out.append("\u0301")  # COMBINING ACUTE ACCENT
        else:
            out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def strip_accents(s: str) -> str:
    """Strip all accents for fuzzy matching."""
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(c for c in nfd if unicodedata.category(c) != "Mn"))


def _is_self_map(form: str, lemma: str) -> bool:
    """Check if a lookup entry is a trivial self-map (form ≈ lemma)."""
    return (form == lemma
            or strip_accents(form.lower()) == strip_accents(lemma.lower()))


def _levenshtein(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(curr[j] + 1, prev[j + 1] + 1,
                            prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


# OCR confusion pairs: (char_a, char_b) -> substitution cost (0.0 to 1.0).
# Normal substitution costs 1.0. OCR-common confusions cost less.
# Built from GCV analysis of LSJ supplement OCR output.
_OCR_CONFUSIONS: dict[tuple[str, str], float] = {}


def _build_ocr_confusions():
    """Build OCR confusion cost matrix (lazy, called once)."""
    if _OCR_CONFUSIONS:
        return

    pairs = [
        # Greek/Latin script mixing (cost ~0, same glyph)
        ("ο", "o", 0.1), ("Ο", "O", 0.1),
        ("ρ", "p", 0.1), ("Ρ", "P", 0.1),
        ("ν", "v", 0.1), ("Ν", "N", 0.1),
        ("τ", "t", 0.1), ("Τ", "T", 0.1),
        ("κ", "k", 0.1), ("Κ", "K", 0.1),
        ("α", "a", 0.1), ("Α", "A", 0.1),
        ("η", "n", 0.2), ("Η", "H", 0.1),
        ("ε", "e", 0.1), ("Ε", "E", 0.1),
        ("ι", "i", 0.1),
        ("υ", "u", 0.1), ("Υ", "Y", 0.1),
        ("χ", "x", 0.1), ("Χ", "X", 0.1),
        ("ω", "w", 0.2),
        ("β", "B", 0.2),
        ("γ", "y", 0.3),  # GCV-specific
        ("δ", "d", 0.3),  # GCV-specific

        # Cyrillic contamination (GCV-specific, same glyph)
        ("ο", "\u043e", 0.0),  # Cyrillic о
        ("α", "\u0430", 0.0),  # Cyrillic а
        ("ε", "\u0435", 0.0),  # Cyrillic е
        ("υ", "\u0443", 0.0),  # Cyrillic у
        ("κ", "\u043a", 0.0),  # Cyrillic к
        ("ρ", "\u0440", 0.0),  # Cyrillic р

        # Common OCR letter confusions (similar shapes)
        ("ο", "σ", 0.5),  # round shapes
        ("θ", "δ", 0.5),  # similar with crossbar
        ("η", "π", 0.5),  # similar verticals
        ("ν", "μ", 0.7),  # similar verticals
        ("ρ", "β", 0.7),  # descender confusion
        ("ι", "ΐ", 0.3),  # diaeresis confusion
        ("υ", "ΰ", 0.3),

        # Number/letter confusions (GCV Roman numeral issue)
        ("1", "I", 0.1),
        ("1", "l", 0.1),

        # GCV descender confusions: J/j for ψ/ὑ
        ("ψ", "J", 0.2), ("ψ", "j", 0.2),
        ("υ", "J", 0.3), ("υ", "j", 0.3),
        ("ὑ", "J", 0.2), ("ὑ", "j", 0.2),
        # GCV: Q/q for θ/σ
        ("θ", "Q", 0.3), ("θ", "q", 0.3),
        ("σ", "Q", 0.3), ("σ", "q", 0.3),
    ]

    for a, b, cost in pairs:
        _OCR_CONFUSIONS[(a, b)] = cost
        _OCR_CONFUSIONS[(b, a)] = cost


def _weighted_levenshtein(a: str, b: str) -> float:
    """Weighted Levenshtein distance using OCR confusion costs.

    Decomposes to NFD first so that combining diacritics are handled
    separately from base characters:
    - Combining diacritics: insert/delete costs 0.1 (nearly free)
    - OCR-confused base character pairs: uses cost matrix (0.1-0.5)
    - Normal substitutions: cost 1.0

    This makes breathing/accent errors nearly free while correctly
    penalizing real letter substitutions.
    """
    _build_ocr_confusions()

    a_nfd = unicodedata.normalize("NFD", a)
    b_nfd = unicodedata.normalize("NFD", b)

    if len(a_nfd) < len(b_nfd):
        return _weighted_levenshtein(b, a)
    if not b_nfd:
        return float(len(a_nfd))

    def _char_cost(c: str) -> float:
        """Cost to insert or delete a character."""
        if unicodedata.combining(c):
            return 0.1  # diacritics are nearly free
        return 1.0

    prev = [0.0]
    for j in range(len(b_nfd)):
        prev.append(prev[-1] + _char_cost(b_nfd[j]))

    for i, ca in enumerate(a_nfd):
        ins_cost_a = _char_cost(ca)
        curr = [prev[0] + ins_cost_a]
        for j, cb in enumerate(b_nfd):
            if ca == cb:
                sub_cost = 0.0
            elif unicodedata.combining(ca) and unicodedata.combining(cb):
                sub_cost = 0.1  # swapping one diacritic for another
            else:
                sub_cost = _OCR_CONFUSIONS.get((ca, cb), 1.0)
            ins_cost_b = _char_cost(cb)
            curr.append(min(
                curr[j] + ins_cost_b,     # insert b[j]
                prev[j + 1] + ins_cost_a,  # delete a[i]
                prev[j] + sub_cost,        # substitute
            ))
        prev = curr
    return prev[-1]


def _strip_elision(word: str) -> str | None:
    """Strip trailing elision mark from an elided word form.

    Returns the consonant stem, or None if no elision detected.
    The elision mark in polytonic text is U+0313 (COMBINING COMMA ABOVE)
    attached to the final consonant, e.g. ἀλλ + U+0313 for ἀλλ̓.

    IMPORTANT: U+0313 also serves as smooth breathing at the START of
    polytonic words (ἐ = ε + U+0313). We only treat it as elision when
    it appears after the first base character cluster (i.e. not on the
    initial letter).
    """
    nfd = unicodedata.normalize("NFD", word)
    if len(nfd) < 2:
        return None

    # Count base (non-combining) characters
    base_count = sum(1 for ch in nfd if unicodedata.category(ch) != "Mn")

    if base_count <= 1:
        # Single base char (like δ̓, τ̓, γ̓): the U+0313 IS the elision
        # mark, not a breathing. Return the bare consonant as stem.
        if nfd[-1] in _ELISION_MARKS:
            stem = unicodedata.normalize("NFC", nfd[:-1])
            if stem:
                return stem
        return None

    # Multi-char word: U+0313 is only elision when it's on the LAST
    # base character. Anywhere else (initial letter, diphthong like ου)
    # it's a smooth breathing mark.
    #
    # Find the last base character, then check if U+0313 follows it
    # with no more base characters after.
    last_base_idx = -1
    for i in range(len(nfd) - 1, -1, -1):
        if unicodedata.category(nfd[i]) != "Mn":
            last_base_idx = i
            break

    if last_base_idx < 0:
        return None

    # Check combining marks after the last base char for elision mark
    for i in range(last_base_idx + 1, len(nfd)):
        if nfd[i] in _ELISION_MARKS:
            stem = unicodedata.normalize("NFC", nfd[:i])
            if stem:
                return stem

    # Also check non-combining elision marks (right quote, modifier apostrophe)
    # at the very end of the string
    if nfd[-1] in _ELISION_MARKS and unicodedata.category(nfd[-1]) != "Mn":
        stem = unicodedata.normalize("NFC", nfd[:-1])
        if stem:
            return stem

    return None


# LSJ9 indeclinable category -> UPOS tag mapping
_INDECL_TO_UPOS = {
    "adverb": "ADV",
    "preposition": "ADP",
    "conjunction": "CCONJ",
    "particle": "PART",
    "interjection": "INTJ",
}

# Subordinating conjunctions get SCONJ instead of CCONJ
_SUBORDINATING = {
    "ὅτι", "ὁτιή", "ἐπεί", "ἐπάν", "ὄφρα", "πρίν", "διότι",
    "μέχριπερ", "ἠΰτε",
}


def _indeclinables_to_pos(raw: dict[str, str]) -> dict[str, dict[str, str]]:
    """Convert {lemma: category} indeclinables dict to POS lookup format.

    Returns {form: {UPOS: lemma}} where form == lemma (indeclinable).
    For conjunctions, subordinating forms get SCONJ; others get CCONJ.
    """
    result: dict[str, dict[str, str]] = {}
    for lemma, category in raw.items():
        upos = _INDECL_TO_UPOS.get(category)
        if not upos:
            continue
        if category == "conjunction" and lemma in _SUBORDINATING:
            upos = "SCONJ"
        result[lemma] = {upos: lemma}
    return result


class LookupDB:
    """Dict-like wrapper around SQLite lookup table.

    Provides .get(key) that queries SQLite instead of loading the full
    12.5M-entry dict into memory. Supports lazy bulk-load into a dict
    for batch operations.

    The DB has two tables:
      - lemmas(id, text): deduplicated lemma strings (~700K)
      - lookup(form, lemma_id, lang): form->lemma mappings (~12.5M)
        lang='all' for combined, lang='grc' for AG-only overrides,
        lang='el' for MG-only overrides
    """

    def __init__(self, db_path, lang='all'):
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
        self._lang = lang
        self._dict = None  # lazy-loaded for batch ops
        self._cache = None  # per-query result cache (enabled via enable_cache)
        self._query = (
            "SELECT l.text FROM lookup k JOIN lemmas l ON k.lemma_id = l.id "
            "WHERE k.form = ? AND k.lang = ? LIMIT 1"
        )
        # Separate spell index (stripped -> form mapping)
        self._spell_conn = None
        if SPELL_INDEX_PATH.exists():
            self._spell_conn = sqlite3.connect(str(SPELL_INDEX_PATH))
            self._spell_conn.execute("PRAGMA mmap_size=134217728")  # 128MB mmap

    def get(self, key, default=None):
        """Dict-compatible .get() backed by SQLite."""
        if self._dict is not None:
            return self._dict.get(key, default)
        # Check per-query cache before hitting SQLite
        if self._cache is not None:
            _sentinel = object()
            cached = self._cache.get(key, _sentinel)
            if cached is not _sentinel:
                return cached if cached is not None else default
        row = self._conn.execute(self._query, (key, self._lang)).fetchone()
        if row:
            if self._cache is not None:
                self._cache[key] = row[0]
            return row[0]
        # AG-only table ('grc') falls through to combined ('all') for entries
        # where AG agrees with combined (not stored separately to save space).
        # Same logic for MG-only ('el').
        if self._lang in ('grc', 'el'):
            row = self._conn.execute(self._query, (key, 'all')).fetchone()
            result = row[0] if row else None
            if self._cache is not None:
                self._cache[key] = result
            return result if result is not None else default
        if self._cache is not None:
            self._cache[key] = None
        return default

    def __contains__(self, key):
        if self._dict is not None:
            return key in self._dict
        if self._cache is not None:
            _sentinel = object()
            cached = self._cache.get(key, _sentinel)
            if cached is not _sentinel:
                return cached is not None
        # Delegate to .get() which handles caching
        return self.get(key) is not None

    def __getitem__(self, key):
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def __bool__(self):
        return True

    def items(self):
        """Iterate all entries (loads nothing extra into memory)."""
        cursor = self._conn.execute(
            "SELECT k.form, l.text FROM lookup k JOIN lemmas l ON k.lemma_id = l.id "
            "WHERE k.lang = ?", (self._lang,))
        return cursor

    def __iter__(self):
        cursor = self._conn.execute(
            "SELECT form FROM lookup WHERE lang = ?", (self._lang,))
        for row in cursor:
            yield row[0]

    def __len__(self):
        row = self._conn.execute(
            "SELECT COUNT(*) FROM lookup WHERE lang = ?", (self._lang,)).fetchone()
        return row[0]

    def bulk_load(self):
        """Load entire table into a dict for fast batch operations."""
        if self._dict is not None:
            return
        self._dict = {}
        for form, lemma in self.items():
            self._dict[form] = lemma

    def enable_cache(self):
        """Enable per-query result caching for repeated lookups.

        Unlike bulk_load() which loads the entire table into memory,
        this only caches results for forms that are actually queried.
        Useful for large tables (e.g. lang='all' with 12M+ entries)
        where bulk_load would use too much memory but the working set
        of queried forms is much smaller.
        """
        if self._dict is not None:
            return  # already fully loaded, cache not needed
        if self._cache is None:
            self._cache = {}
            # Boost SQLite page cache for the initial cold lookups
            self._conn.execute("PRAGMA cache_size=-65536")  # 64MB

    @staticmethod
    def _parse_spell_forms(forms_blob: str,
                           src_filter: str | None = None) -> list[str]:
        """Parse the compact forms blob from spell_index.db.

        Each line is "form\\tsrc" or just "form". Returns the form
        strings, optionally filtered by src.
        """
        results = []
        for line in forms_blob.split("\n"):
            if "\t" in line:
                form, src = line.split("\t", 1)
                if src_filter and src != src_filter:
                    continue
            else:
                form = line
            results.append(form)
        return results

    def spell_lookup_stripped(self, candidates: set[str],
                             src_filter: str = None
                             ) -> dict[str, list[str]]:
        """Look up stripped forms, return {stripped: [original_forms]}.

        Queries the separate spell_index.db for fast batch lookup.
        Falls back to the main lookup table if spell_index.db is
        not available (legacy DBs with a stripped column).

        Args:
            candidates: Set of accent-stripped forms to look up.
            src_filter: If set, only return forms from this source
                (e.g., 'grc' for AG-sourced forms only).
        """
        if not candidates:
            return {}
        result: dict[str, list[str]] = {}
        candidate_list = list(candidates)

        if self._spell_conn is not None:
            for i in range(0, len(candidate_list), 900):
                batch = candidate_list[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                query = (
                    f"SELECT stripped, forms FROM spell "
                    f"WHERE stripped IN ({placeholders})"
                )
                for stripped, forms_blob in self._spell_conn.execute(
                        query, batch):
                    forms = self._parse_spell_forms(forms_blob, src_filter)
                    if forms:
                        result[stripped] = forms
            return result

        # Legacy: stripped column in main lookup table
        for i in range(0, len(candidate_list), 900):
            batch = candidate_list[i:i + 900]
            placeholders = ",".join("?" * len(batch))
            if src_filter:
                query = (
                    f"SELECT DISTINCT stripped, form FROM lookup "
                    f"WHERE stripped IN ({placeholders}) AND src = ?"
                )
                batch = batch + [src_filter]
            elif self._lang in ('grc', 'el'):
                query = (
                    f"SELECT DISTINCT stripped, form FROM lookup "
                    f"WHERE stripped IN ({placeholders}) "
                    f"AND lang IN ('{self._lang}', 'all')"
                )
            else:
                query = (
                    f"SELECT DISTINCT stripped, form FROM lookup "
                    f"WHERE stripped IN ({placeholders}) AND lang = ?"
                )
                batch = batch + [self._lang]
            for stripped, form in self._conn.execute(query, batch):
                if stripped not in result:
                    result[stripped] = []
                result[stripped].append(form)
        return result

    def has_stripped(self, stripped: str) -> bool:
        """Check if a stripped form exists."""
        if self._spell_conn is not None:
            row = self._spell_conn.execute(
                "SELECT 1 FROM spell WHERE stripped = ? LIMIT 1",
                (stripped,)).fetchone()
        elif self._lang in ('grc', 'el'):
            row = self._conn.execute(
                "SELECT 1 FROM lookup WHERE stripped = ? AND lang IN (?, 'all') LIMIT 1",
                (stripped, self._lang)).fetchone()
        else:
            row = self._conn.execute(
                "SELECT 1 FROM lookup WHERE stripped = ? AND lang = ? LIMIT 1",
                (stripped, self._lang)).fetchone()
        return row is not None

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


class Dilemma:
    def __init__(self, lang="all", device=None, scale=None,
                 resolve_articles=False, normalize=False, period=None,
                 dialect=None, convention=None, skip_pos=False):
        """Initialize Dilemma.

        Args:
            lang: "all" (default) for MG+AG+Medieval combined,
                  "el" for MG only, "grc" for AG only
            device: "cpu", "cuda", etc. Auto-detected if None.
            scale: Model scale (0-4). None auto-detects the best available.
                   Larger scales = more training data = better generalization
                   on unseen forms. Lookup table is the same for all scales.
            resolve_articles: if True, resolve article forms (τῆς, τόν,
                  etc.) to the canonical lemma ὁ, and pronoun clitics
                  (μοι, σοι) to their pronoun lemma (ἐγώ, σύ). Default
                  False, which keeps articles/pronouns as self-mappings
                  (better for alignment where you want surface-form
                  matching). Set True for evaluation against treebanks
                  like DiGreC/AGDT which use ὁ as the article lemma.
            normalize: if True, enable orthographic normalization for
                  Byzantine/papyrological texts. Generates candidate
                  spellings (fixing itacism, missing iota subscripta,
                  etc.) and checks them against the lookup table.
                  Also enabled implicitly when dialect is set.
            period: Historical period for normalization rule weights.
                  One of: "hellenistic", "late_antique", "byzantine",
                  "all" (default). Only used when normalize=True.
            dialect: Ancient Greek dialect normalization. Maps dialect
                  forms to Attic equivalents for lookup. One of:
                  "ionic" (Herodotus, Hippocrates, etc.),
                  "doric" (Pindar, Theocritus, etc.),
                  "aeolic" (Sappho, Alcaeus, etc.),
                  "koine" (NT, papyri - overlaps with period rules),
                  "auto" (try all dialect rules),
                  None (default, no dialect normalization).
                  Setting a dialect implicitly enables normalization.
            convention: Lemma convention for output remapping. Controls
                  which citation form is returned when multiple conventions
                  exist for the same word (e.g. LSJ vs Wiktionary headwords).
                  One of: None (default, no remapping), "wiktionary" (same
                  as None), "lsj" (remap to LSJ dictionary headwords),
                  "cunliffe" (remap to Cunliffe Homeric Lexicon headwords),
                  "triantafyllidis" (remap to Modern Greek monotonic
                  dictionary forms, using MG Wiktionary headwords as
                  reference).
        """
        if convention not in _VALID_CONVENTIONS:
            raise ValueError(
                f"Unknown convention {convention!r}. "
                f"Valid values: {sorted(c for c in _VALID_CONVENTIONS if c)}, or None."
            )
        if lang == "both":
            lang = "all"
        self.lang = lang
        self._scale = scale
        # Triantafyllidis convention always needs article resolution:
        # MG text lemmatizes articles to ο, pronouns to αυτός, etc.
        if convention == "triantafyllidis" and not resolve_articles:
            resolve_articles = True
        self._resolve_articles = resolve_articles
        self._convention_name = convention
        self._model = None
        self._vocab = None
        self._device = device
        self._normalizer = None
        # Setting a dialect implicitly enables normalization
        if normalize or dialect is not None:
            from .normalize import Normalizer
            self._normalizer = Normalizer(period=period, dialect=dialect)

        # Lookup tables: SQLite-backed (instant startup) or dict (JSON fallback)
        self._mg_lookup = {}
        self._med_lookup = {}
        self._ag_lookup = {}
        self._lookup = {}
        self._using_db = False

        # POS-indexed disambiguation table: {form: {upos: lemma}}
        self._pos_lookup: dict[str, dict[str, str]] = {}
        # AG-only POS lookup for polytonic-first disambiguation
        self._pos_ag_lookup: dict[str, dict[str, str]] = {}

        # Headword frequency table (lazy-loaded from lsj9_frequency.json)
        self._hw_frequency: dict[str, int] | None = None

        # Headword sets by convention name (lazy-loaded)
        self._headword_sets: dict[str, set[str]] = {}

        # GLAUx corpus frequency: stripped_form -> token_count (lazy-loaded)
        self._glaux_freq: dict[str, int] | None = None

        # Per-lemma corpus attestation profiles (lazy-loaded)
        self._attestation: dict[str, dict] | None = None

        self._load_lookups(skip_pos=skip_pos)
        self._convention_map = self._build_convention_map(convention)
        self._convention_monotonic = convention in _MONOTONIC_CONVENTIONS
        self._check_backend()

    def _check_backend(self):
        """Warn once at init if no model backend is available."""
        model_dir = self._find_model_dir()
        has_onnx_files = (model_dir / "encoder.onnx").exists()
        has_pt_file = (model_dir / "model.pt").exists()
        if has_onnx_files:
            try:
                import onnxruntime  # noqa: F401
                return
            except ImportError:
                pass
        if has_pt_file:
            try:
                import torch  # noqa: F401
                return
            except ImportError:
                pass
        import warnings
        warnings.warn(
            "No model backend available. Install onnxruntime (~50 MB) or "
            "torch (~2 GB) for unseen-word inference. Lookup table still "
            "works, but unknown forms will return unchanged. "
            "pip install onnxruntime",
            stacklevel=3,
        )

    def preload(self):
        """Optimize lookup tables for batch processing.

        Call this before processing a large corpus to avoid per-query
        SQLite overhead. Enables per-query caching on all SQLite-backed
        lookup tables so repeated lookups for the same form skip SQLite
        entirely. The Iliad's working set (~20K unique forms) stays
        small even though the full table has 12M+ entries.

        Safe to call multiple times (idempotent). Does not affect the
        public API - all existing methods work the same, just faster.
        """
        if not self._using_db:
            return  # JSON fallback already uses dicts
        for table in [self._lookup, self._ag_lookup,
                      self._mg_lookup, self._med_lookup]:
            if isinstance(table, LookupDB):
                table.enable_cache()

    def export_cache(self) -> dict[str, str]:
        """Export the full lookup table as a {form: lemma} dict.

        Useful for passing to downstream tools (e.g. Tagger's lemma_cache
        parameter) so they can skip Dilemma calls for known forms.
        Convention remapping is applied if a convention is set.
        """
        cache: dict[str, str] = {}
        if self._using_db:
            for form, lemma in self._lookup.items():
                cache[form] = self._apply_convention(lemma)
        else:
            for form, lemma in self._lookup.items():
                cache[form] = self._apply_convention(lemma)
        return cache

    def _load_lookups(self, skip_pos=False):
        """Load lookup tables from SQLite (instant) or JSON (fallback).

        SQLite path (lookup.db): pre-merged combined table with AG priority,
        plus AG-only overrides for polytonic disambiguation. Near-instant
        startup, 0.05ms/query, supports lazy bulk_load() for batch ops.

        JSON fallback: loads all three JSON files and merges at init (~11s).
        """
        if LOOKUP_DB_PATH.exists():
            # SQLite: instant startup for all language modes
            if self.lang == "grc":
                self._lookup = LookupDB(LOOKUP_DB_PATH, lang='grc')
                self._ag_lookup = self._lookup
            elif self.lang == "el":
                self._lookup = LookupDB(LOOKUP_DB_PATH, lang='all')
                self._ag_lookup = LookupDB(LOOKUP_DB_PATH, lang='grc')
                self._mg_lookup = LookupDB(LOOKUP_DB_PATH, lang='el')
            else:
                self._lookup = LookupDB(LOOKUP_DB_PATH, lang='all')
                self._ag_lookup = LookupDB(LOOKUP_DB_PATH, lang='grc')
                # Load MG lookup for triantafyllidis convention: MG entries
                # should take priority over AG for monotonic MG forms.
                if self._convention_name == "triantafyllidis":
                    self._mg_lookup = LookupDB(LOOKUP_DB_PATH, lang='el')
            self._using_db = True
        elif LOOKUP_PATH.exists() or AG_LOOKUP_PATH.exists():
            # JSON fallback (dev mode before lookup.db is built)
            self._load_lookups_json()
        else:
            raise FileNotFoundError(
                f"Dilemma data not found at {DATA_DIR}. "
                "Download with: python -m dilemma download  "
                "(or set DILEMMA_DATA_DIR to point at an existing copy)."
            )

        if not skip_pos:
            self._load_pos_lookups()

    def _load_lookups_json(self):
        """Fallback: load JSON lookup tables and merge (~11s)."""
        if LOOKUP_PATH.exists():
            with open(LOOKUP_PATH, encoding="utf-8") as f:
                self._mg_lookup = json.load(f)
        if MED_LOOKUP_PATH.exists():
            with open(MED_LOOKUP_PATH, encoding="utf-8") as f:
                self._med_lookup = json.load(f)
        if AG_LOOKUP_PATH.exists():
            with open(AG_LOOKUP_PATH, encoding="utf-8") as f:
                self._ag_lookup = json.load(f)

        def _is_self_map(form, lemma):
            return (form == lemma
                    or strip_accents(form.lower()) == strip_accents(lemma.lower()))

        if self.lang == "all":
            for data in [self._ag_lookup, self._med_lookup, self._mg_lookup]:
                for k, v in data.items():
                    if k not in self._lookup:
                        self._lookup[k] = v
                    elif _is_self_map(k, self._lookup[k]) and not _is_self_map(k, v):
                        self._lookup[k] = v
                    elif (_is_self_map(k, self._lookup[k])
                          and _is_self_map(k, v) and v == k
                          and self._lookup[k] != k):
                        self._lookup[k] = v
        elif self.lang == "el":
            for data in [self._mg_lookup, self._med_lookup]:
                for k, v in data.items():
                    if k not in self._lookup:
                        self._lookup[k] = v
                    elif _is_self_map(k, self._lookup[k]) and not _is_self_map(k, v):
                        self._lookup[k] = v
        elif self.lang == "grc":
            self._lookup = dict(self._ag_lookup)

    def _load_pos_lookups(self):
        """Load POS disambiguation tables from JSON.

        Builds two tables:
        - _pos_ag_lookup: AG-only sources (treebank, GLAUx, AG Wiktionary, LSJ9)
        - _pos_lookup: combined (AG sources + MG Wiktionary)

        For polytonic input (breathing marks, circumflex), lemmatize_pos() and
        lemmatize_batch_pos() check _pos_ag_lookup first before the combined
        table, mirroring the AG-first logic in the main lookup.

        Priority within each table: treebank (gold) > LSJ9 indeclinables
        (unambiguous POS) > GLAUx (corpus) > MG Wiktionary (combined only) >
        AG Wiktionary > LSJ9 grammar.
        """
        def _add_to(target, source_data, overwrite=False):
            """Merge source_data into target POS dict."""
            for form, upos_lemmas in source_data.items():
                if form not in target:
                    target[form] = {}
                if overwrite:
                    target[form].update(upos_lemmas)
                else:
                    for upos, lemma in upos_lemmas.items():
                        if upos not in target[form]:
                            target[form][upos] = lemma

        # 1. Treebank POS lookup (gold-annotated, highest priority for AG)
        if self.lang in ("all", "grc") and TREEBANK_POS_LOOKUP_PATH.exists():
            with open(TREEBANK_POS_LOOKUP_PATH, encoding="utf-8") as f:
                tb_pos = json.load(f)
            _add_to(self._pos_ag_lookup, tb_pos, overwrite=True)
            _add_to(self._pos_lookup, tb_pos, overwrite=True)

        # 1b. LSJ9 indeclinables (adverbs, prepositions, conjunctions,
        #     particles, interjections) - POS is unambiguous for these
        if self.lang in ("all", "grc") and LSJ9_INDECLINABLES_PATH.exists():
            with open(LSJ9_INDECLINABLES_PATH, encoding="utf-8") as f:
                indecl_raw = json.load(f)
            indecl_pos = _indeclinables_to_pos(indecl_raw)
            _add_to(self._pos_ag_lookup, indecl_pos)
            _add_to(self._pos_lookup, indecl_pos)

        # 2. GLAUx POS lookup (corpus-derived, 8.7K entries)
        if self.lang in ("all", "grc") and GLAUX_POS_LOOKUP_PATH.exists():
            with open(GLAUX_POS_LOOKUP_PATH, encoding="utf-8") as f:
                glaux_pos = json.load(f)
            _add_to(self._pos_ag_lookup, glaux_pos)
            _add_to(self._pos_lookup, glaux_pos)

        # 3. MG POS lookup (Wiktionary-derived, combined table only)
        if self.lang in ("all", "el") and MG_POS_LOOKUP_PATH.exists():
            with open(MG_POS_LOOKUP_PATH, encoding="utf-8") as f:
                mg_pos = json.load(f)
            _add_to(self._pos_lookup, mg_pos)  # combined only, not AG

        # 4. AG Wiktionary POS lookup (fills remaining gaps)
        if self.lang in ("all", "grc") and AG_POS_LOOKUP_PATH.exists():
            with open(AG_POS_LOOKUP_PATH, encoding="utf-8") as f:
                ag_pos = json.load(f)
            _add_to(self._pos_ag_lookup, ag_pos)
            _add_to(self._pos_lookup, ag_pos)

        # 5. LSJ9 grammar-derived POS (407K forms with NOUN/ADJ from
        #    the grammar field: ὁ/ἡ/τό -> NOUN, ον/ές -> ADJ)
        if self.lang in ("all", "grc") and LSJ9_POS_LOOKUP_PATH.exists():
            with open(LSJ9_POS_LOOKUP_PATH, encoding="utf-8") as f:
                lsj9_pos = json.load(f)
            _add_to(self._pos_ag_lookup, lsj9_pos)
            _add_to(self._pos_lookup, lsj9_pos)

    def _build_convention_map(self, convention: str | None) -> dict[str, str]:
        """Build a lemma remapping dict for the given convention.

        For "lsj"/"cunliffe", each equivalence group is resolved to the
        first member that appears in the convention's headword list. All
        other members map to it.

        For "triantafyllidis", the headword set is the MG lookup lemmas
        (Modern Greek Wiktionary forms). Members are also checked after
        to_monotonic() conversion. The _convention_monotonic flag ensures
        all output is converted to monotonic in _apply_convention().

        For None or "wiktionary", returns an empty dict (no remapping).

        After auto-deriving from equivalences, explicit overrides from
        data/convention_{name}.json are applied (if the file exists).
        The override file format is: {"mappings": {"from_lemma": "to_lemma"}}.
        """
        if convention is None or convention == "wiktionary":
            return {}

        remap = {}

        # Load the headword set for this convention.
        # For triantafyllidis, derive from MG lookup lemmas.
        # For lsj/cunliffe, load from the dedicated headword file.
        headwords = set()
        hw_path = _CONVENTION_HEADWORDS.get(convention)
        if hw_path and hw_path.exists():
            with open(hw_path, encoding="utf-8") as f:
                raw = json.load(f)
            headwords = set(raw)
            # LSJ/Cunliffe headwords may include vowel-length marks
            # (macron U+0304, breve U+0306) like βᾰρύς. Strip these
            # so they match our plain equivalence group members.
            for h in raw:
                nfd = unicodedata.normalize("NFD", h)
                stripped = "".join(
                    c for c in nfd if ord(c) not in (0x0304, 0x0306))
                stripped = unicodedata.normalize("NFC", stripped)
                if stripped != h:
                    headwords.add(stripped)
        elif convention == "triantafyllidis" and not headwords:
            # Fallback: derive from MG lookup JSON if no headword file.
            if LOOKUP_PATH.exists():
                with open(LOOKUP_PATH, encoding="utf-8") as f:
                    headwords = set(json.load(f).values())

        # Auto-derive from equivalence groups
        if LEMMA_EQUIVALENCES_PATH.exists():
            with open(LEMMA_EQUIVALENCES_PATH, encoding="utf-8") as f:
                equiv_data = json.load(f)

            is_monotonic = convention in _MONOTONIC_CONVENTIONS

            for group in equiv_data.get("groups", []):
                if len(group) < 2:
                    continue

                # Find the canonical for this convention: first member
                # in the headword list. Fall back to the first member
                # of the group if none match.
                canonical = group[0]
                found = False
                for member in group:
                    if member in headwords:
                        canonical = member
                        found = True
                        break

                # For monotonic conventions, also try to_monotonic()
                # on each member and check against the headword set.
                if not found and is_monotonic:
                    for member in group:
                        mono = to_monotonic(member)
                        if mono in headwords:
                            canonical = mono
                            found = True
                            break

                for member in group:
                    if member != canonical:
                        remap[member] = canonical

        # For monotonic conventions, add systematic morphological remappings
        # from AG lemma forms to their MG equivalents. These cover productive
        # patterns like neuter -ον -> -ο, abstract -σις -> -ση, and
        # contracted verbs -έω -> -ώ / -άω -> -ώ.
        if convention in _MONOTONIC_CONVENTIONS and headwords:
            # Load AG lemmas from pre-extracted headword file (fast) or
            # fall back to loading the full AG lookup JSON (slow).
            ag_lemmas = set()
            if AG_HEADWORDS_PATH.exists():
                with open(AG_HEADWORDS_PATH, encoding="utf-8") as f:
                    ag_lemmas = set(json.load(f))
            elif AG_LOOKUP_PATH.exists():
                with open(AG_LOOKUP_PATH, encoding="utf-8") as f:
                    ag_lemmas = set(json.load(f).values())
            patterns = [
                # (AG suffix, replacement, description)
                ("ον", "ο", "neuter -ον -> -ο"),
                ("σις", "ση", "abstract -σις -> -ση"),
                ("ξις", "ξη", "abstract -ξις -> -ξη"),
                ("ψις", "ψη", "abstract -ψις -> -ψη"),
                ("έω", "ώ", "contracted -έω -> -ώ"),
                ("άω", "ώ", "contracted -άω -> -ώ"),
                ("όω", "ώ", "contracted -όω -> -ώ"),
            ]
            for al in ag_lemmas:
                if al in remap:
                    continue  # already mapped by equivalence group
                mono = to_monotonic(al)
                for suffix, replacement, _desc in patterns:
                    if mono.endswith(suffix):
                        candidate = mono[:-len(suffix)] + replacement
                        if candidate in headwords and candidate != mono:
                            remap[al] = candidate
                            break

        # Apply explicit overrides from convention file
        override_path = CONVENTION_DIR / f"convention_{convention}.json"
        if override_path.exists():
            with open(override_path, encoding="utf-8") as f:
                overrides = json.load(f)
            for from_lemma, to_lemma in overrides.get("mappings", {}).items():
                remap[from_lemma] = to_lemma

        return remap

    def _apply_convention(self, lemma: str) -> str:
        """Remap a lemma according to the active convention.

        For monotonic conventions (e.g. triantafyllidis), the result is
        converted to monotonic Greek after any explicit remapping, so
        polytonic lemmas like ὁ become ο automatically.

        For the LSJ convention, adverbs (-ῶς/-ως) and neuter adjectives
        (-ον/-όν) that aren't LSJ headwords are mapped to their adjective
        headword, since LSJ files these as sub-entries under the adjective.
        """
        if self._convention_map:
            lemma = self._convention_map.get(lemma, lemma)
        if self._convention_name == "lsj":
            lemma = self._lsj_adverb_neuter_remap(lemma)
        if self._convention_monotonic:
            lemma = to_monotonic(lemma)
        return lemma

    def _lsj_adverb_neuter_remap(self, lemma: str) -> str:
        """Map adverbs and neuter adjectives to LSJ adjective headwords.

        LSJ doesn't give adverbs (δεινῶς) or neuter forms (δεινόν) their
        own headword entries - they appear under the adjective (δεινός).
        This remaps them automatically using accent-stripped matching
        against the LSJ adjective headword set.
        """
        lsj_hw = self._get_headword_set("lsj")
        if lemma in lsj_hw:
            return lemma

        # Lazy-load LSJ adjective set
        if not hasattr(self, "_lsj_adj_stripped"):
            lsj_pos_path = (Path(__file__).resolve().parent.parent.parent
                            / "lsj9" / "lsj9_headword_pos.json")
            if not lsj_pos_path.exists():
                lsj_pos_path = DATA_DIR / "lsj9_headword_pos.json"
            self._lsj_adj_stripped = {}
            if lsj_pos_path.exists():
                import json as _json
                pos_data = _json.load(open(lsj_pos_path, encoding="utf-8"))
                for hw, pos in pos_data.items():
                    if pos == "ADJ":
                        self._lsj_adj_stripped[
                            strip_accents(hw.lower())] = hw

        if not self._lsj_adj_stripped:
            return lemma

        # Adverb -ῶς/-ως -> adjective -ος/-ης/-υς
        if lemma.endswith("ῶς") or lemma.endswith("ως"):
            stem = strip_accents(lemma[:-2].lower())
            for suffix in ("ος", "ης", "υς", "ων", "ις"):
                candidate = stem + suffix
                if candidate in self._lsj_adj_stripped:
                    return self._lsj_adj_stripped[candidate]

        # Neuter -ον/-όν -> adjective -ος/-ης
        if lemma.endswith("ον") or lemma.endswith("όν"):
            stem = strip_accents(lemma[:-2].lower())
            for suffix in ("ος", "ης"):
                candidate = stem + suffix
                if candidate in self._lsj_adj_stripped:
                    return self._lsj_adj_stripped[candidate]

        return lemma

    def _get_headword_set(self, convention: str = "lsj") -> set[str]:
        """Lazy-load and cache the headword set for a given convention.

        Returns a set of headword strings. Strips vowel-length marks
        (macron/breve) so matching works against plain forms.
        """
        if convention in self._headword_sets:
            return self._headword_sets[convention]

        hw_path = _CONVENTION_HEADWORDS.get(convention)
        headwords: set[str] = set()
        if hw_path and hw_path.exists():
            with open(hw_path, encoding="utf-8") as f:
                raw = json.load(f)
            headwords = set(raw)
            # Strip vowel-length marks (macron U+0304, breve U+0306)
            for h in raw:
                nfd = unicodedata.normalize("NFD", h)
                stripped = "".join(
                    c for c in nfd if ord(c) not in (0x0304, 0x0306))
                stripped = unicodedata.normalize("NFC", stripped)
                if stripped != h:
                    headwords.add(stripped)

        self._headword_sets[convention] = headwords
        return headwords

    def is_headword(self, word: str, convention: str = "lsj") -> bool:
        """Check if a word is a known headword in the given convention.

        Args:
            word: The Greek word to check.
            convention: Which headword list to check against.
                "lsj" (default), "cunliffe", or "triantafyllidis".

        Returns:
            True if the word is in the convention's headword list.
        """
        return word in self._get_headword_set(convention)

    def _build_sorted_headword_index(self):
        """Build sorted (sort_key, headword) list from all headword sources.

        Lazy-loaded on first call to headwords_between(). Combines:
        LSJ, Cunliffe, DGE, LGPN, BrillDAG, and ag_headwords (Wiktionary-derived).
        """
        if hasattr(self, "_sorted_hw_index"):
            return
        all_hws = set()
        PD_HW_PATH = DATA_DIR / "pd_headwords.json"
        BRILLDAG_HW_PATH = DATA_DIR / "brilldag_headwords.json"
        for path in [LSJ_HEADWORDS_PATH, CUNLIFFE_HEADWORDS_PATH,
                     AG_HEADWORDS_PATH, DGE_HEADWORDS_PATH, LGPN_NAMES_PATH,
                     PD_HW_PATH, BRILLDAG_HW_PATH]:
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    all_hws.update(json.load(f))
        # Build sorted list by sort key
        def _sort_key(hw):
            nfkd = unicodedata.normalize("NFKD", hw.lower().replace("-", ""))
            return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
        self._sorted_hw_pairs = sorted(
            ((_sort_key(h), h) for h in all_hws if h),
            key=lambda x: x[0])
        self._sorted_hw_keys = [p[0] for p in self._sorted_hw_pairs]
        self._sorted_hw_index = True

    def headwords_between(self, hw_a: str, hw_b: str) -> list[str]:
        """Return all known headwords sorting alphabetically between hw_a and hw_b.

        Uses accent-stripped, case-folded sort keys for comparison.
        Returns original polytonic headword forms.

        Args:
            hw_a: Lower bound headword (exclusive).
            hw_b: Upper bound headword (exclusive).

        Returns:
            List of headwords between hw_a and hw_b, sorted alphabetically.
        """
        import bisect
        self._build_sorted_headword_index()

        def _sort_key(hw):
            nfkd = unicodedata.normalize("NFKD", hw.lower().replace("-", ""))
            return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")

        key_a = _sort_key(hw_a)
        key_b = _sort_key(hw_b)
        if key_a > key_b:
            key_a, key_b = key_b, key_a

        lo = bisect.bisect_right(self._sorted_hw_keys, key_a)
        hi = bisect.bisect_left(self._sorted_hw_keys, key_b)
        return [self._sorted_hw_pairs[i][1] for i in range(lo, hi)]

    def _get_lemma_set(self) -> set[str]:
        """Get the set of all lemmata (citation forms) in the lookup table.

        Unlike headword sets which come from specific dictionaries, this
        returns all unique lemma values from the lookup table itself -
        the right-hand side of form->lemma mappings. Cached after first call.
        """
        if hasattr(self, "_lemma_set"):
            return self._lemma_set

        lemmas: set[str] = set()
        if self._using_db:
            cursor = self._lookup._conn.execute(
                "SELECT DISTINCT text FROM lemmas")
            lemmas = {row[0] for row in cursor}
        else:
            lemmas = set(self._lookup.values()) if self._lookup else set()

        self._lemma_set = lemmas
        return self._lemma_set

    def _find_model_dir(self):
        """Find the best available model directory.

        Search order: {lang}/ (full model), then {lang}-s3/-s2/-s1 (legacy),
        then combined/ as fallback if no language-specific model exists.
        """
        lang_dir = {"el": "el", "grc": "grc", "all": "combined"}[self.lang]

        # Explicit scale requested
        if self._scale is not None:
            for prefix in [lang_dir, "combined"]:
                # New naming: {lang}-test or {lang} (full)
                if str(self._scale) == "test":
                    candidate = MODEL_DIR / f"{prefix}-test"
                else:
                    candidate = MODEL_DIR / prefix
                if (candidate / "encoder.onnx").exists() or (candidate / "model.pt").exists():
                    return candidate
                # Legacy naming: -s1, -s2, -s3
                candidate = MODEL_DIR / f"{prefix}-s{self._scale}"
                if (candidate / "encoder.onnx").exists() or (candidate / "model.pt").exists():
                    return candidate

        # Auto-detect: prefer {lang}/ (full), then legacy -s3/-s2/-s1
        for prefix in [lang_dir, "combined"]:
            candidate = MODEL_DIR / prefix
            if (candidate / "encoder.onnx").exists() or (candidate / "model.pt").exists():
                return candidate
            for s in [3, 2, 1]:
                candidate = MODEL_DIR / f"{prefix}-s{s}"
                if (candidate / "encoder.onnx").exists() or (candidate / "model.pt").exists():
                    return candidate

        return MODEL_DIR / lang_dir

    def _load_model(self):
        """Lazy-load the model on first use. Prefers ONNX, falls back to PyTorch."""
        if self._model is not None:
            return

        model_path = self._find_model_dir()

        # Try ONNX first (no PyTorch dependency, ~50MB vs ~2GB)
        if (model_path / "encoder.onnx").exists():
            self._load_onnx(model_path)
            return

        # Fall back to PyTorch
        self._load_pytorch(model_path)

    def _load_onnx(self, model_path):
        """Load ONNX model and lightweight vocab."""
        from .onnx_inference import OnnxLemmaModel, CharVocabLight
        vocab_path = model_path / "vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"No vocab.json at {vocab_path}. "
                f"Run: python export_onnx.py"
            )
        self._vocab = CharVocabLight(vocab_path)
        self._model = OnnxLemmaModel(model_path)
        self._device = "cpu"
        self._use_onnx = True

    def _load_pytorch(self, model_path):
        """Load PyTorch model (original path).

        Detects and loads morphology heads (POS, nominal, verbal) when
        present in the checkpoint. Head label mappings are stored in
        self._head_labels for inference use.
        """
        import torch
        from .model import CharVocab, LemmaTransformer

        pt_path = model_path / "model.pt"
        if not pt_path.exists():
            raise FileNotFoundError(
                f"No trained model at {pt_path}. "
                f"Run: python train.py --lang {self.lang}"
            )

        device = self._device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        checkpoint = torch.load(pt_path, map_location=device, weights_only=False)
        self._vocab = CharVocab()
        self._vocab.load_state_dict(checkpoint["vocab"])
        cfg = checkpoint["config"]

        # Detect morphology heads from state dict keys
        state = checkpoint["model_state_dict"]
        head_cfg = checkpoint.get("head_config", {})

        # Infer head dimensions from weights if head_config is missing
        num_pos = cfg.get("num_pos_tags", 0)
        if not num_pos and "pos_head.weight" in state:
            num_pos = state["pos_head.weight"].shape[0]
            cfg["num_pos_tags"] = num_pos

        self._model = LemmaTransformer(**cfg)

        # Create nom/verb heads if weights exist
        d_model = cfg.get("d_model", 256)
        if "nom_head.weight" in state:
            num_nom = state["nom_head.weight"].shape[0]
            import torch.nn as nn
            self._model.nom_head = nn.Linear(d_model, num_nom)
        if "verb_head.weight" in state:
            num_verb = state["verb_head.weight"].shape[0]
            import torch.nn as nn
            self._model.verb_head = nn.Linear(d_model, num_verb)

        self._model.load_state_dict(state)
        self._model.to(device)
        self._model.eval()
        self._use_onnx = False

        # Store label mappings for inference
        self._head_labels = {
            "pos": {int(k): v for k, v in head_cfg.get("pos_labels", {}).items()},
            "nom": {int(k): v for k, v in head_cfg.get("nom_labels", {}).items()},
            "verb": {int(k): v for k, v in head_cfg.get("verb_labels", {}).items()},
        }
        # Build fallback POS label map if not saved
        if num_pos and not self._head_labels["pos"]:
            _POS_FALLBACK = {
                0: "verb", 1: "noun", 2: "adj", 3: "adv", 4: "name",
                5: "pron", 6: "num", 7: "prep", 8: "article", 9: "character",
            }
            self._head_labels["pos"] = {
                i: _POS_FALLBACK.get(i, f"tag{i}") for i in range(num_pos)
            }

    def _resolve_closed_class(self, word: str) -> str | None:
        """Resolve articles/pronouns to canonical lemma if enabled.

        When convention='triantafyllidis', MG closed-class resolution
        takes priority: MG articles -> ο, preposition contractions
        (στη, στον, ...) -> σε, copula forms -> είμαι, etc. AG pronoun
        resolution (σε -> σύ, με -> ἐγώ) is skipped since those forms
        serve as prepositions/particles in MG.
        """
        if not self._resolve_articles:
            return None

        # MG closed-class resolution (triantafyllidis convention)
        if self._convention_name == "triantafyllidis":
            lower = word.lower()
            # MG articles -> ο
            if lower in _MG_ARTICLE_FORMS:
                return _MG_ARTICLE_LEMMA
            # MG preposition contractions, copula, auxiliaries, αυτός
            if lower in _MG_CLOSED_CLASS:
                return _MG_CLOSED_CLASS[lower]
            # For triantafyllidis, skip AG pronoun resolution entirely.
            # AG pronouns (σε -> σύ, με -> ἐγώ) conflict with MG
            # prepositions/particles. Fall through to AG article
            # resolution only for polytonic article forms.

        lower = word.lower()
        if (word in _ARTICLE_FORMS
                or lower in _ARTICLE_FORMS
                or to_monotonic(word) in _ARTICLE_FORMS
                or to_monotonic(lower) in _ARTICLE_FORMS):
            # Don't use strip_accents here - it's too aggressive for short
            # words (e.g. ἤ "or" becomes η which matches the article)
            return _ARTICLE_LEMMA

        # Skip AG pronoun resolution when triantafyllidis is active
        if self._convention_name == "triantafyllidis":
            return None

        if word in _PRONOUN_LEMMAS:
            return _PRONOUN_LEMMAS[word]
        mono = to_monotonic(word)
        if mono in _PRONOUN_LEMMAS:
            return _PRONOUN_LEMMAS[mono]
        return None

    def _lookup_word(self, word: str) -> str | None:
        """Try lookup cascade: exact -> lowercase -> grave_to_acute -> monotonic -> stripped.

        Grave-to-acute is tried before monotonic because it preserves
        breathings and circumflex (lighter normalization). ὣς → ὡς works
        here without losing the breathing that monotonic would strip.

        For polytonic input (breathings/circumflex present), AG lookup is
        tried first to avoid MG lemma forms (biblion vs biblion).

        For lang="el" (MG mode), MG-specific entries are checked first
        (monotonic input -> el lookup), then falls back to combined.

        Skips mono/stripped matches that are trivially short (1-2 chars)
        and map to themselves - these are usually false positives from
        accent stripping on elided or particle forms.
        """
        # For MG mode or triantafyllidis convention, try MG-specific
        # lookup first. MG entries take priority over AG when there's a
        # conflict (e.g. χρήσης -> χρήση in MG vs χράω in AG, or
        # κοντά -> κοντά in MG vs κοντός in AG).
        _prefer_mg = (self.lang == "el"
                      or self._convention_name == "triantafyllidis")
        if _prefer_mg and self._mg_lookup:
            tbl = self._mg_lookup
            hit = tbl.get(word) or tbl.get(word.lower())
            if not hit:
                mono = to_monotonic(word.lower())
                stripped = strip_accents(word.lower())
                for variant in [mono, stripped]:
                    hit = tbl.get(variant)
                    if hit and not (len(variant) <= 2 and hit == variant):
                        break
                    hit = None
            if hit:
                return hit

        # For polytonic input (breathings/circumflex), try AG-only lookup
        # first. Even though AG has priority in the combined table, the
        # normalization cascade (mono/stripped) can still land on MG entries.
        nfd = unicodedata.normalize("NFD", word)
        has_poly = any(ord(ch) in _POLYTONIC_STRIP | _POLYTONIC_TO_ACUTE
                       for ch in nfd)
        if has_poly and self.lang == "all" and self._ag_lookup:
            tbl = self._ag_lookup
            hit = tbl.get(word) or tbl.get(word.lower())
            if not hit:
                acute = grave_to_acute(word)
                if acute != word:
                    hit = tbl.get(acute) or tbl.get(acute.lower())
            if not hit:
                for variant in [to_monotonic(word.lower()),
                                strip_accents(word.lower())]:
                    hit = tbl.get(variant)
                    if hit and not (len(variant) <= 2 and hit == variant):
                        break
                    hit = None
            if hit:
                return hit

        lemma = self._lookup.get(word) or self._lookup.get(word.lower())
        if lemma:
            return lemma
        # Grave → acute (lightest normalization, preserves breathings)
        acute = grave_to_acute(word)
        if acute != word:
            lemma = self._lookup.get(acute) or self._lookup.get(acute.lower())
            if lemma:
                return lemma
        mono = to_monotonic(word.lower())
        stripped = strip_accents(word.lower())
        for variant in [mono, stripped]:
            hit = self._lookup.get(variant)
            if hit and not (len(variant) <= 2 and hit == variant):
                return hit
        return None


    def _expand_elision(self, word: str) -> str | None:
        """Try to resolve an elided form by expanding with vowels.

        Strips the elision mark, appends each Greek vowel, and checks
        if the expanded form is in the lookup table. Prefers expansions
        where the lemma is a real word (differs from the expanded form)
        over self-mapping headwords.
        """
        candidates = self._expand_elision_all(word)
        if not candidates:
            return None
        # _expand_elision_all returns results pre-sorted by vowel frequency
        return candidates[0][1]

    def _expand_elision_all(self, word: str) -> list[tuple[str, str, str]]:
        """Return ALL valid elision expansions as (expanded, lemma, vowel) triples.

        For polytonic input (has breathings/circumflex), prioritizes AG lookup.
        Elision is overwhelmingly an AG phenomenon; MG monotonic forms would
        pollute results with false matches.

        Results are sorted by vowel frequency in elision contexts (ε, α, ο
        most common), then by lemma length. This ensures both lemmatize()
        and lemmatize_verbose() get the same ranking.
        """
        stem = _strip_elision(word)
        if not stem:
            return []

        # Detect if input is polytonic (has AG-style diacritics).
        # Also treat any non-combining elision mark (U+1FBD KORONIS,
        # U+02BC, U+2019) as indicating AG context, since elision is
        # overwhelmingly an AG phenomenon.
        nfd = unicodedata.normalize("NFD", word)
        has_polytonic = (any(ord(ch) in _POLYTONIC_STRIP | _POLYTONIC_TO_ACUTE
                            for ch in nfd)
                         or any(ch in _ELISION_MARKS
                                and unicodedata.category(ch) != "Mn"
                                for ch in word))

        # Choose which lookups to search
        if has_polytonic:
            tables = [(self._ag_lookup, "grc")]
        else:
            tables = [(self._mg_lookup, "el"),
                      (self._med_lookup, "med"),
                      (self._ag_lookup, "grc")]

        results = []
        seen_lemmas = set()

        all_vowels = list(_GREEK_VOWELS)
        # Also try accented vowels
        accented = {"α": "ά", "ε": "έ", "ι": "ί", "ο": "ό",
                    "η": "ή", "υ": "ύ", "ω": "ώ"}
        all_candidates = [(v, v) for v in all_vowels]
        all_candidates += [(acc, v) for v, acc in accented.items()]

        # Consonant de-assimilation: before rough breathing, Greek
        # assimilates voiceless stops to aspirates (τ->θ, π->φ, κ->χ).
        # Reverse this to recover the original stem for prepositions:
        # καθ' -> κατ' (κατά), ἐφ' -> ἐπ' (ἐπί), ἀφ' -> ἀπ' (ἀπό)
        _DEASSIMILATE = {"θ": "τ", "φ": "π", "χ": "κ"}
        stems_to_try = [stem]
        if stem and stem[-1] in _DEASSIMILATE:
            stems_to_try.append(stem[:-1] + _DEASSIMILATE[stem[-1]])

        for try_stem in stems_to_try:
            for table, lang in tables:
                for suffix, vowel_name in all_candidates:
                    expanded = try_stem + suffix
                    # Try full normalization cascade
                    lemma = None
                    for variant in (expanded, expanded.lower(),
                                    grave_to_acute(expanded),
                                    to_monotonic(expanded.lower()),
                                    strip_accents(expanded.lower())):
                        lemma = table.get(variant)
                        if lemma:
                            break
                    if lemma and lemma not in seen_lemmas:
                        seen_lemmas.add(lemma)
                        results.append((expanded, lemma, suffix))

        # Common elided function words - these should always win over
        # content words when ambiguous. Includes both original stems
        # and assimilated forms (καθ from κατά, αφ from ἀπό, etc.)
        _ELISION_PRIORITY = {
            "αλλ", "μετ", "παρ", "κατ", "δι", "απ", "επ",
            "υπ", "αφ", "εφ", "υφ", "μηδ", "ουδ", "αντ", "περ",
            # Assimilated forms (θ<-τ, φ<-π, χ<-κ before rough breathing)
            "καθ", "εφ", "αφ", "υφ", "μεθ", "παρ", "ανθ",
        }
        stem_lower = strip_accents(stem.lower())

        # Sort by: (1) function word priority, (2) vowel frequency, (3) lemma length
        _VOWEL_RANK = {v: i for i, v in enumerate("εαοιηυω")}
        _ACC_VOWEL_RANK = {"ά": 1, "έ": 0, "ί": 4, "ό": 2,
                           "ή": 3, "ύ": 5, "ώ": 6}

        # Function word lemmas (prepositions, particles, conjunctions)
        _FUNCTION_LEMMAS = {
            "ἀλλά", "μετά", "παρά", "κατά", "διά", "ἀπό", "ἐπί",
            "ὑπό", "ἀπό", "ἐπί", "ὑπό", "μηδέ", "οὐδέ", "ἀντί",
            "περί", "δέ", "γε", "τε", "ἄρα", "ἔτι",
        }

        def _rank(item):
            expanded, lemma, vowel = item
            # Prefer function words when stem is a known elision stem
            is_function = 0 if (stem_lower in _ELISION_PRIORITY
                                and lemma in _FUNCTION_LEMMAS) else 1
            # Deprioritize proper nouns
            is_proper = 1 if lemma and lemma[0].isupper() else 0
            # Use corpus frequency: more frequent lemmas are more likely
            # to be the correct resolution (κατά >> κάθω)
            freq = self._get_glaux_freq(strip_accents(lemma.lower()))
            neg_freq = -freq  # negate so higher frequency sorts first
            vrank = _ACC_VOWEL_RANK.get(vowel, _VOWEL_RANK.get(vowel, 10))
            return (is_function, is_proper, neg_freq, vrank, len(lemma))

        results.sort(key=_rank)
        return results

    def _lang_of(self, form: str) -> str:
        """Determine which language table a form comes from."""
        if self._using_db:
            return self._lang_of_db(form)
        lower = form.lower()
        mono = to_monotonic(lower)
        stripped = strip_accents(lower)
        for variant in [form, lower, mono, stripped]:
            if variant in self._mg_lookup:
                return "el"
            if variant in self._med_lookup:
                return "med"
            if variant in self._ag_lookup:
                return "grc"
        return ""

    def _lang_of_db(self, form: str) -> str:
        """SQLite-backed language detection via the src column."""
        _SRC_TO_LANG = {'grc': 'grc', 'el': 'el', 'med': 'med'}
        conn = self._lookup._conn
        lower = form.lower()
        mono = to_monotonic(lower)
        stripped = strip_accents(lower)
        query = "SELECT src FROM lookup WHERE form = ? AND lang = 'all' LIMIT 1"
        for variant in [form, lower, mono, stripped]:
            row = conn.execute(query, (variant,)).fetchone()
            if row:
                return _SRC_TO_LANG.get(row[0], "")
        return ""

    def _is_proper(self, lemma: str) -> bool:
        """Check if a lemma is a proper noun (capitalized headword)."""
        return bool(lemma) and lemma[0].isupper()

    # ---- Morphology head inference ----

    # POS tag to UPOS mapping for POS-lookup integration
    _POS_TO_UPOS = {
        "verb": "VERB", "noun": "NOUN", "adj": "ADJ", "adv": "ADV",
        "name": "PROPN", "pron": "PRON", "num": "NUM", "prep": "ADP",
        "article": "DET", "character": "PROPN",
    }

    def predict_pos_tag(self, word: str) -> str:
        """Predict POS tag for a Greek word using the model's POS head.

        Returns a Wiktionary-style POS label ("verb", "noun", "adj", etc.)
        or "" if no POS head is available.

        Requires the model to be loaded (lazy-loads on first call).
        """
        self._load_model()
        if hasattr(self._model, 'has_pos_head') and self._model.has_pos_head:
            # ONNX path
            src, mask = self._encode_word(word)
            tags = self._model.predict_pos(src, mask)
            return tags[0]
        return ""

    def predict_pos_batch(self, words: list[str]) -> list[str]:
        """Predict POS tags for a batch of Greek words.

        Returns list of Wiktionary-style POS labels.
        """
        self._load_model()
        if hasattr(self._model, 'has_pos_head') and self._model.has_pos_head:
            src, mask = self._encode_words(words)
            return self._model.predict_pos(src, mask)
        return [""] * len(words)

    def _encode_word(self, word: str):
        """Encode a single word for ONNX inference. Returns (src, mask) arrays."""
        import numpy as np
        ids = self._vocab.encode(word)
        max_len = 48  # ONNX_MAX_LEN
        ids = ids + [0] * (max_len - len(ids))
        src = np.array([ids[:max_len]], dtype=np.int64)
        mask = (src == 0)
        return src, mask

    def _encode_words(self, words: list[str]):
        """Encode a batch of words for ONNX inference."""
        import numpy as np
        max_len = 48
        batch = []
        for w in words:
            ids = self._vocab.encode(w)
            ids = ids + [0] * (max_len - len(ids))
            batch.append(ids[:max_len])
        src = np.array(batch, dtype=np.int64)
        mask = (src == 0)
        return src, mask

    # ---- Byzantine normalization fallback ----

    # Iota adscript -> subscript mappings (Byzantine texts often write
    # the iota next to rather than under the vowel).
    _IOTA_ADSCRIPT_MAP = {
        'ωι': 'ῳ', 'ηι': 'ῃ',
        'ῶι': 'ῷ', 'ῆι': 'ῇ',
        'ώι': 'ῴ', 'ήι': 'ῄ',
        'ὼι': 'ῲ', 'ὴι': 'ῂ',
    }

    # Known Greek verbal/nominal prefixes for compound decomposition.
    _KNOWN_PREFIXES = [
        'ἀντι', 'ἐπι', 'ὑπερ', 'παρα', 'κατα', 'ἀπο', 'μετα',
        'περι', 'προσ', 'ἀνα', 'δια', 'ὑπο', 'ἐξ', 'ἐκ',
        'εἰσ', 'εἰς', 'συν', 'συμ', 'συγ', 'συλ', 'ἐν',
        'προ',
    ]

    def _normalize_byzantine(self, word: str,
                             use_normalizer: bool = True) -> str | None:
        """Try Byzantine-specific normalizations to find a lookup match.

        Handles common scribal/orthographic features of Byzantine Greek
        manuscripts that differ from the classical forms stored in the
        lookup table:
          - Final medial sigma (σ instead of ς at word end)
          - Iota adscript (ωι, ηι instead of subscript ῳ, ῃ)
          - Missing breathings on initial vowels
          - Hyphens and editorial marks in transcriptions

        When use_normalizer=True, also chains with the orthographic
        normalizer to handle itacism, geminate simplification, etc.
        This is more aggressive and can produce false positives.

        When use_normalizer=False, only the precise/safe normalizations
        above are applied.

        Called as a fallback after lookup, normalizer, and model have
        all failed. Returns a lemma if any normalization yields a
        lookup hit, otherwise None.
        """
        # 1. Final sigma: σ at end of word -> ς
        if word.endswith('σ'):
            fixed = word[:-1] + 'ς'
            lemma = self._lookup_word(fixed)
            if lemma:
                return lemma

        # 2. Iota adscript -> subscript
        fixed = word
        for src, dst in self._IOTA_ADSCRIPT_MAP.items():
            fixed = fixed.replace(src, dst)
        if fixed != word:
            lemma = self._lookup_word(fixed)
            if lemma:
                return lemma

        # 3. Hyphen removal (editorial hyphens in line-broken words)
        if '-' in word:
            fixed = word.replace('-', '')
            lemma = self._lookup_word(fixed)
            if lemma:
                return lemma

        # 4. Lowercase initial capital (Byzantine sentence-initial caps)
        if word and word[0].isupper():
            lowered = word[0].lower() + word[1:]
            lemma = self._lookup_word(lowered)
            if lemma:
                return lemma
            # Also try lowercase + iota adscript fix
            fixed = lowered
            for src, dst in self._IOTA_ADSCRIPT_MAP.items():
                fixed = fixed.replace(src, dst)
            if fixed != lowered:
                lemma = self._lookup_word(fixed)
                if lemma:
                    return lemma
            # Also try lowercase + final sigma fix
            if lowered.endswith('σ'):
                fixed = lowered[:-1] + 'ς'
                lemma = self._lookup_word(fixed)
                if lemma:
                    return lemma

        # 5. Missing breathing on initial vowel: try adding smooth or rough
        lower = word.lower()
        nfd = unicodedata.normalize("NFD", lower)
        if nfd and nfd[0] in 'αεηιουω':
            # Check if there's already a breathing mark
            has_breathing = (len(nfd) > 1
                            and ord(nfd[1]) in (0x0313, 0x0314))
            if not has_breathing:
                # Try smooth breathing (U+0313), then rough (U+0314)
                for breathing in ('\u0313', '\u0314'):
                    candidate = unicodedata.normalize(
                        "NFC", nfd[0] + breathing + nfd[1:])
                    lemma = self._lookup_word(candidate)
                    if lemma:
                        return lemma

        if not use_normalizer:
            return None

        # 6. Chain: apply Byzantine fixes, then run the orthographic
        #    normalizer on the result. This handles cases like
        #    ἀναγυνώσκοντα (ι/υ itacism + no iota adscript issue)
        #    where the normalizer alone would have worked if the
        #    form had standard orthographic features.
        #    Use existing normalizer if available, otherwise create a
        #    lightweight Byzantine normalizer on first use.
        #    Only applied to longer words (>= 6 chars) to avoid false
        #    matches on short common words.
        stripped_lower = strip_accents(word.lower())
        if len(stripped_lower) >= 6:
            if not hasattr(self, '_byz_normalizer'):
                from .normalize import Normalizer
                self._byz_normalizer = Normalizer(period="byzantine")
            norm = self._normalizer or self._byz_normalizer

            # Collect all forms to try: original + Byzantine-fixed variants
            forms_to_try = [word]
            # Final sigma fix
            if word.endswith('σ'):
                forms_to_try.append(word[:-1] + 'ς')
            # Iota adscript fix
            fixed = word
            for src, dst in self._IOTA_ADSCRIPT_MAP.items():
                fixed = fixed.replace(src, dst)
            if fixed != word:
                forms_to_try.append(fixed)
            # Lowercase
            if word and word[0].isupper():
                forms_to_try.append(word[0].lower() + word[1:])

            # Only run normalizer if it wasn't already run pre-model
            # (avoid double-normalizing). For pre-model normalizer,
            # only the original form was tried, so skip it here.
            start_idx = 1 if self._normalizer else 0
            for variant in forms_to_try[start_idx:]:
                for candidate in norm.normalize(variant):
                    lemma = self._lookup_word(candidate)
                    if lemma:
                        lemma_s = strip_accents(lemma.lower())
                        cand_s = strip_accents(candidate.lower())
                        # Guard: only accept if the lookup found a real
                        # lemma (different from the candidate form).
                        # This prevents accepting identity lookups where
                        # the normalizer just tweaked the form slightly.
                        if lemma_s != cand_s:
                            return lemma

        return None

    def _prefix_strip_lookup(self, word: str) -> str | None:
        """Try stripping known Greek prefixes and looking up the base.

        For compound verbs and adjectives where the prefixed form is not
        in the lookup table but the base form is. Only returns a result
        if the reconstructed prefix+base_lemma compound is itself a known
        form in the lookup table (very conservative to avoid false hits).
        """
        lower = word.lower()
        stripped = strip_accents(lower)

        for prefix in self._KNOWN_PREFIXES:
            prefix_s = strip_accents(prefix)
            if not stripped.startswith(prefix_s):
                continue
            if len(stripped) <= len(prefix_s) + 2:
                continue

            remainder = stripped[len(prefix_s):]
            base_lemma = self._lookup_word(remainder)
            if not base_lemma:
                continue

            base_lemma_s = strip_accents(base_lemma.lower())

            # Guard: skip identity lookups
            if base_lemma_s == remainder:
                continue

            # Guard: skip suspiciously short base lemmas
            if len(base_lemma_s) < 2:
                continue

            # Reconstruct compound: prefix + base_lemma
            compound = prefix_s + base_lemma_s

            # Validate: the reconstructed compound must itself be
            # a known form in the lookup table. This avoids producing
            # nonsense compounds like υπεραραρισκω.
            compound_lemma = self._lookup_word(compound)
            if compound_lemma:
                return compound_lemma

        return None

    # ---- Compound decomposition ----

    # Linking vowels at the junction of Greek compounds
    _COMPOUND_LINK_VOWELS = set("οιυ")
    _MIN_COMPOUND_PREFIX = 2   # e.g. εὐ-, τρι-
    _MIN_COMPOUND_BASE = 3     # need enough for inflection

    def _decompose_compound(self, word: str) -> str | None:
        """Try to lemmatize an unknown compound by splitting at linking vowels.

        Greek compounds: first-stem + linking-vowel (ο/ι/υ) + second-element.
        The second element inflects like its standalone form. Strategy: split
        at each linking vowel (left to right, preferring longer bases), look
        up the base, and reconstruct prefix + base_lemma.

        Returns the reconstructed compound lemma, or None.
        """
        lower = word.lower()
        stripped = strip_accents(lower)

        if len(stripped) < self._MIN_COMPOUND_PREFIX + self._MIN_COMPOUND_BASE + 1:
            return None

        # Try split points left to right (longest base first = most reliable)
        for i in range(self._MIN_COMPOUND_PREFIX - 1,
                       len(stripped) - self._MIN_COMPOUND_BASE):
            if stripped[i] not in self._COMPOUND_LINK_VOWELS:
                continue

            prefix = stripped[:i + 1]   # includes linking vowel
            base = stripped[i + 1:]

            if len(base) < self._MIN_COMPOUND_BASE:
                continue

            # Look up the base in the lookup table
            base_lemma = self._lookup_word(base)
            if not base_lemma:
                continue

            base_lemma_s = strip_accents(base_lemma.lower())

            # Guard: skip if lookup returned identity (no real lemmatization)
            if base_lemma_s == base:
                continue

            # Guard: skip if base_lemma is suspiciously short (false match)
            if len(base_lemma_s) < 2:
                continue

            # Guard: base_lemma should be shorter or equal to base
            # (lemmatization removes inflection, doesn't add length)
            if len(base_lemma_s) > len(base) + 2:
                continue

            # Reconstruct compound lemma
            return prefix + base_lemma_s

        return None

    def _decompose_compound_all(self, word: str) -> list[tuple[str, str, str]]:
        """Return ALL valid compound decompositions.

        Returns list of (compound_lemma, base_lemma, prefix) triples,
        ordered by base length descending (longest base = most specific).
        Used by lemmatize_verbose for multi-candidate output.
        """
        lower = word.lower()
        stripped = strip_accents(lower)
        results = []
        seen = set()

        if len(stripped) < self._MIN_COMPOUND_PREFIX + self._MIN_COMPOUND_BASE + 1:
            return results

        for i in range(self._MIN_COMPOUND_PREFIX - 1,
                       len(stripped) - self._MIN_COMPOUND_BASE):
            if stripped[i] not in self._COMPOUND_LINK_VOWELS:
                continue

            prefix = stripped[:i + 1]
            base = stripped[i + 1:]

            if len(base) < self._MIN_COMPOUND_BASE:
                continue

            base_lemma = self._lookup_word(base)
            if not base_lemma:
                continue

            base_lemma_s = strip_accents(base_lemma.lower())
            if base_lemma_s == base or len(base_lemma_s) < 2:
                continue
            if len(base_lemma_s) > len(base) + 2:
                continue

            compound = prefix + base_lemma_s
            if compound not in seen:
                seen.add(compound)
                results.append((compound, base_lemma, prefix))

        return results

    # ---- Feature: Particle/enclitic suffix stripping ----

    def _strip_particle_suffix(self, word: str) -> str | None:
        """Try stripping enclitic particles (-per, -ge, -de, deictic -i).

        Only fires as a fallback when the full form is not in the lookup
        table. Returns the base form's lemma (not the base form itself),
        or None if no valid stripping is found.

        Suffix priority: -per (safest, most common), then -ge, then -de.
        Deictic -i is only stripped from known demonstrative pronoun stems.
        For -de, only strips if the resulting base form is found in lookup
        (since -de is also a real word ending in many cases).
        """
        lower = word.lower()
        stripped_word = strip_accents(lower)

        # Try -per, -ge, -de suffixes
        for suffix in _PARTICLE_SUFFIXES:
            suffix_stripped = strip_accents(suffix)
            if not stripped_word.endswith(suffix_stripped):
                continue
            base = stripped_word[:-len(suffix_stripped)]
            if len(base) < 2:
                continue

            # For the original (accented) form, try cutting the suffix
            # from the actual word. We try multiple cuts since accent
            # position may shift. Order matters: accented variants are
            # tried before the accent-stripped fallback, since accented
            # forms disambiguate homographs like ἔμοι (dat. of ἐγώ) vs
            # εμοι (which the lookup also resolves to a verb of ἐμέω).
            # Use a list with explicit dedup rather than a set, since
            # set iteration order is hash-seed-dependent.
            candidates: list[str] = []
            def _add(c: str) -> None:
                if c and c not in candidates:
                    candidates.append(c)

            # Accented variants first (more specific)
            for suf_len in (len(suffix), len(suffix_stripped)):
                if len(lower) > suf_len:
                    _add(lower[:-suf_len])
            # Accent-stripped fallback
            _add(base)

            # After stripping a suffix, a medial sigma (σ) that was
            # word-internal may now be at the end of the base. Convert
            # to final sigma (ς) so lookup matches dictionary forms
            # (e.g. ὅσπερ -> ὅσ needs to find ὅς). Append in the same
            # priority order as the source candidates.
            for c in list(candidates):
                if c.endswith("σ"):
                    _add(c[:-1] + "ς")

            for candidate in candidates:
                lemma = self._lookup_word(candidate)
                if lemma:
                    return lemma

        # Deictic -i: only strip from demonstrative pronoun forms
        if stripped_word.endswith("ι") and len(stripped_word) > 2:
            base = stripped_word[:-1]
            if base in _DEICTIC_STEMS:
                # Try looking up the base (without the deictic -i)
                # Try the accent-stripped base and original minus last char
                for candidate in (base, lower[:-1] if len(lower) > 1 else ""):
                    if not candidate:
                        continue
                    lemma = self._lookup_word(candidate)
                    if lemma:
                        return lemma

        return None

    # ---- Feature: Verb morphology stripping (augment, reduplication) ----

    # Temporal augment mappings: augmented vowel -> original stem-initial vowel(s)
    _TEMPORAL_AUGMENT = {
        "η": ["ε", "α"],     # η- can be augment of ε- or α-initial stems
        "ω": ["ο"],           # ω- augment of ο-initial stems
        "ηυ": ["ευ"],         # ηυ- augment of ευ-initial stems
        "ῃ": ["ε", "α"],     # with iota subscript
        "ει": ["ε"],          # spurious diphthong augment
        "ηι": ["αι"],         # ηι- augment of αι-initial stems (rare)
    }

    def _strip_verb_morphology(self, word: str) -> str | None:
        """Try decomposing unknown verb forms by stripping augment or reduplication.

        Handles three patterns:
        1. Syllabic augment (ε-prefix): strip leading ε- and look up
        2. Temporal augment (η- for ε-stems, ω- for ο-stems, etc.)
        3. Reduplication (λε-λυ- type perfect forms): strip the reduplicated
           consonant+vowel prefix

        Only returns a result if the transformed form matches a known lemma
        in the lookup table. This is a conservative heuristic - it won't
        catch everything but avoids false positives.
        """
        lower = word.lower()
        stripped = strip_accents(lower)

        # Skip very short words
        if len(stripped) < 4:
            return None

        # 1. Syllabic augment: strip leading ε- (most common augment)
        # The augmented form starts with ε and the unaugmented stem
        # should be a recognizable verb form.
        nfd = unicodedata.normalize("NFD", lower)
        base_chars = [ch for ch in nfd if unicodedata.category(ch) != "Mn"]
        if base_chars and base_chars[0] == "ε":
            # Strip the initial epsilon (with any diacritics)
            # Find the end of the first character cluster (base + combiners)
            idx = 0
            for i, ch in enumerate(nfd):
                if unicodedata.category(ch) != "Mn":
                    if idx > 0:
                        break
                    idx = 1
                    cut_point = i + 1
                else:
                    cut_point = i + 1
            remainder = unicodedata.normalize("NFC", nfd[cut_point:])
            if len(remainder) >= 3:
                lemma = self._lookup_word(remainder)
                if lemma and not _is_self_map(remainder, lemma):
                    return lemma

        # 2. Temporal augment: try replacing augmented initial vowel
        # with the original stem vowel(s)
        for augmented, originals in self._TEMPORAL_AUGMENT.items():
            aug_stripped = strip_accents(augmented)
            if not stripped.startswith(aug_stripped):
                continue
            remainder_stripped = stripped[len(aug_stripped):]
            if len(remainder_stripped) < 3:
                continue
            for original in originals:
                candidate = original + remainder_stripped
                lemma = self._lookup_word(candidate)
                if lemma and not _is_self_map(candidate, lemma):
                    return lemma

        # 3. Reduplication: perfect forms have consonant+vowel prefix
        # that duplicates the stem-initial consonant. Pattern: CV-stem
        # where C matches the first consonant of the stem.
        if len(stripped) >= 5:
            # Check if first two chars look like reduplication (C + ε/vowel)
            first_char = stripped[0]
            if first_char.isalpha() and stripped[1] in "εα":
                rest = stripped[2:]
                # The stem should start with the same consonant
                if rest and rest[0] == first_char:
                    lemma = self._lookup_word(rest)
                    if lemma and not _is_self_map(rest, lemma):
                        return lemma

        return None

    # ---- Feature: Article-agreement disambiguation ----

    def _rank_by_article_agreement(self, candidates: list[LemmaCandidate],
                                   prev_word: str | None) -> list[LemmaCandidate]:
        """Re-rank candidates using article gender/number agreement.

        If prev_word is a Greek article, boost candidates whose lemma
        gender matches the article's gender. This only re-ranks, never
        excludes candidates.

        Args:
            candidates: List of LemmaCandidate objects to rank.
            prev_word: The preceding word (may be an article or None).

        Returns:
            Re-ranked list of candidates (same elements, possibly reordered).
        """
        if not prev_word or not candidates or len(candidates) < 2:
            return candidates

        # Normalize the article form (try exact, then grave-to-acute)
        features = _ARTICLE_FEATURES.get(prev_word)
        if not features:
            features = _ARTICLE_FEATURES.get(grave_to_acute(prev_word))
        if not features:
            return candidates

        article_gender = features[0]  # "m", "f", or "n"

        def _guess_gender(lemma: str) -> str | None:
            """Guess gender from lemma ending (heuristic)."""
            # Check longest suffixes first
            for suffix, gender in sorted(_LEMMA_GENDER_HINTS.items(),
                                         key=lambda x: -len(x[0])):
                if lemma.endswith(suffix):
                    return gender
            return None

        # Score each candidate: 0 if gender matches, 1 if not
        def _gender_score(c: LemmaCandidate) -> int:
            g = _guess_gender(strip_accents(c.lemma.lower()))
            if g == article_gender:
                return 0
            return 1

        # Stable sort: preserve original order within same gender score
        candidates_sorted = sorted(candidates,
                                   key=lambda c: (_gender_score(c), c.proper, -c.score))
        return candidates_sorted

    def lemmatize(self, word: str) -> str:
        """Lemmatize a single Greek word.

        Resolution order:
          1. Article/pronoun resolution (if resolve_articles=True)
          2. Crasis table (small, hand-curated)
          3. Lookup table (instant, 5M+ forms)
          4. Elision expansion (strip mark, try vowels against lookup)
          5. Particle suffix stripping (-per, -ge, -de, deictic -i)
          6. Verb morphology stripping (augment, reduplication)
          7. Normalizer (orthographic variants)
          8. Compound decomposition (split at linking vowel, look up base)
          9. Model with beam search + headword filter

        If a convention is set, the output lemma is remapped accordingly.
        """
        if not word:
            return word
        # Pass through tokens that are purely digits/punctuation (dates, numbers)
        if word.isdigit():
            return word

        # Resolve articles/pronouns to canonical lemma
        closed = self._resolve_closed_class(word)
        if closed is not None:
            return self._apply_convention(closed)

        # Check crasis first (before lookup, since crasis forms are
        # Wiktionary headwords that self-map in the lookup)
        from .crasis import resolve_crasis
        crasis_result = resolve_crasis(word) or resolve_crasis(to_monotonic(word))
        if crasis_result is not None:
            return self._apply_convention(crasis_result)

        # Lookup: exact -> lowercase -> monotonic -> accent-stripped
        lemma = self._lookup_word(word)
        if lemma:
            return self._apply_convention(lemma)

        # Elision expansion (after lookup, so known words like εἰ/οὐ
        # aren't falsely caught by smooth-breathing-as-elision)
        elision_lemma = self._expand_elision(word)
        if elision_lemma:
            return self._apply_convention(elision_lemma)

        # Particle suffix stripping (after lookup, before model)
        particle_lemma = self._strip_particle_suffix(word)
        if particle_lemma:
            return self._apply_convention(particle_lemma)

        # Verb morphology stripping (augment, reduplication - after
        # particle stripping, before model)
        verb_morph_lemma = self._strip_verb_morphology(word)
        if verb_morph_lemma:
            return self._apply_convention(verb_morph_lemma)

        # Normalizer: try orthographic variants against lookup
        if self._normalizer:
            for candidate in self._normalizer.normalize(word):
                lemma = self._lookup_word(candidate)
                if lemma:
                    return self._apply_convention(lemma)

        # Fall back to model
        try:
            self._load_model()
            pred = self._predict([word])[0]
        except (RuntimeError, IndexError, ImportError, FileNotFoundError):
            # Model inference can fail on unusual inputs (empty tensors,
            # single-char forms, etc.) or if no backend is installed.
            # Fall through to identity fallback.
            return self._apply_convention(word)

        # Fallback strategies when the model returns identity
        # (model couldn't lemmatize). Uses accent-stripped comparison since
        # the model may return slight accent variants of the input.
        if strip_accents(pred.lower()) == strip_accents(word.lower()):
            # 1. Light Byzantine normalization (final sigma, iota
            #    adscript, breathing, hyphen - very precise, few false
            #    positives)
            byz = self._normalize_byzantine(word, use_normalizer=False)
            if byz:
                return self._apply_convention(byz)

            # 2. Compound decomposition (linking vowel split)
            compound = self._decompose_compound(word)
            if compound:
                return self._apply_convention(compound)

            # 3. Prefix stripping (known prefixes + base lookup)
            prefix_hit = self._prefix_strip_lookup(word)
            if prefix_hit:
                return self._apply_convention(prefix_hit)

            # 4. Heavy Byzantine normalization (uses the orthographic
            #    normalizer to handle itacism, geminate simplification,
            #    etc. - more aggressive, may produce false positives)
            byz_heavy = self._normalize_byzantine(word, use_normalizer=True)
            if byz_heavy:
                return self._apply_convention(byz_heavy)

        return self._apply_convention(pred)

    def _pos_table_lookup(self, word: str, upos: str) -> str | None:
        """Look up POS-specific lemma from POS disambiguation tables.

        For polytonic input, checks the AG-only POS table first to avoid
        MG lemma overrides on Ancient Greek text. Returns None if no match.
        """
        lower = word.lower()
        acute = grave_to_acute(lower)
        mono = to_monotonic(lower)
        stripped = strip_accents(lower)
        variants = (word, lower, acute, mono, stripped)

        # For polytonic input, try AG-only POS first
        nfd = unicodedata.normalize("NFD", word)
        has_poly = any(ord(ch) in _POLYTONIC_STRIP | _POLYTONIC_TO_ACUTE
                       for ch in nfd)
        if has_poly and self.lang == "all" and self._pos_ag_lookup:
            for variant in variants:
                pos_entry = self._pos_ag_lookup.get(variant)
                if pos_entry and upos in pos_entry:
                    return pos_entry[upos]

        # Combined POS lookup
        for variant in variants:
            pos_entry = self._pos_lookup.get(variant)
            if pos_entry and upos in pos_entry:
                return pos_entry[upos]

        return None

    def _fix_mg_selfmap(self, word: str, candidates: list[LemmaCandidate],
                        upos: str | None = None) -> str | None:
        """Fix MG self-map problem for adjective/verb inflections.

        When _prefer_mg is true (triantafyllidis or lang="el"), the MG
        lookup sometimes returns self-maps for inflected forms instead
        of the citation form (masculine nominative for ADJ, infinitive
        for VERB). Examples:
          - ανθρώπινα -> ανθρώπινα (self-map), should be ανθρώπινος
          - περιέχει -> περιέχει (self-map), should be περιέχω

        When POS is known (from upos parameter), uses it directly.
        Adverbs and nouns keep their MG self-maps unchanged.

        Also fixes adjectives where MG returns a feminine nominative
        (e.g. παλαιολιθική) instead of the masculine nominative
        (παλαιολιθικός), by checking if a -ος form exists as a headword.

        Returns the corrected lemma, or None if no fix applies.
        """
        _prefer_mg = (self.lang == "el"
                      or self._convention_name == "triantafyllidis")
        if not _prefer_mg or not candidates:
            return None

        top = candidates[0]
        form_lower = word.lower()
        top_lower = top.lemma.lower()

        # ADJ/VERB self-map fix: when MG returns a self-map for an
        # inflected form, prefer a non-self-map citation form from
        # the combined/AG lookup.
        is_selfmap = _is_self_map(word, top.lemma)

        if is_selfmap and upos in ("ADJ", "VERB", None):
            # Find non-self-map candidates
            for c in candidates:
                if _is_self_map(word, c.lemma):
                    continue
                c_lower = c.lemma.lower()
                # ADJ: prefer masculine nominative (-ος, -ής, -ύς)
                if upos == "ADJ" or upos is None:
                    if (c_lower.endswith("ος") or c_lower.endswith("ής")
                            or c_lower.endswith("ύς")):
                        if not c.proper:
                            return c.lemma
                # VERB: prefer infinitive/1sg (-ω, -ώ, -μαι)
                if upos == "VERB" or upos is None:
                    if (c_lower.endswith("ω") or c_lower.endswith("ώ")
                            or c_lower.endswith("μαι")):
                        if not c.proper:
                            return c.lemma

        # ADJ feminine-to-masculine fix: when the top result is a
        # feminine nominative adjective (ending in -η/-ή/-ια/-α), check
        # if there is a masculine form (-ος) available as a headword.
        if upos == "ADJ" and not is_selfmap:
            mono_lemma = to_monotonic(top_lower)
            masc_form = None
            # Common feminine -> masculine adjective mappings
            if mono_lemma.endswith("ική") or mono_lemma.endswith("ικη"):
                masc_form = mono_lemma[:-1] + "ός"
            elif mono_lemma.endswith("ή"):
                masc_form = mono_lemma[:-1] + "ός"
            elif mono_lemma.endswith("η"):
                masc_form = mono_lemma[:-1] + "ος"
            elif mono_lemma.endswith("ια"):
                masc_form = mono_lemma[:-2] + "ος"
            elif mono_lemma.endswith("ιά"):
                masc_form = mono_lemma[:-2] + "ός"

            if masc_form:
                # Check if the masculine form exists as a headword
                # (self-mapping entry) in any table
                for tbl in (self._mg_lookup, self._ag_lookup, self._lookup):
                    hit = tbl.get(masc_form)
                    if hit and strip_accents(hit.lower()) == strip_accents(masc_form.lower()):
                        return self._apply_convention(hit)

        return None

    def lemmatize_pos(self, word: str, upos: str) -> str:
        """Lemmatize with POS-aware disambiguation.

        POS is used to disambiguate among multiple candidates, or to
        override the single candidate when curated POS tables indicate
        a different lemma for the given POS tag.

        Algorithm:
          1. Run regular lemmatize_verbose() to get all candidates.
          2. Check POS tables for a POS-specific lemma. If it matches
             any candidate, return that candidate.
          3. If there is only one candidate and the POS table suggests
             a different lemma, trust the POS table (curated sources:
             treebank, GLAUx, Wiktionary). This handles cases like
             σκέψει+NOUN -> σκέψις where the default lookup only has
             the verb mapping σκέπτομαι.
          4. For MG self-map fix: when _prefer_mg is true and POS is
             ADJ/VERB, check if the top candidate is an MG self-map and
             prefer a citation-form alternative from combined/AG lookup.
          5. If no POS match, return the top candidate (same as regular
             lookup).

        Args:
            word: Greek word form.
            upos: Universal POS tag (NOUN, VERB, ADJ, etc.).

        Returns:
            The lemma string.
        """
        # Get all candidates from regular lookup
        candidates = self.lemmatize_verbose(word)

        if not candidates:
            # Should not happen (verbose always adds identity), but be safe
            return self.lemmatize(word)

        # Use POS tables to disambiguate or override
        pos_lemma = self._pos_table_lookup(word, upos)
        if pos_lemma is not None:
            pos_lemma_conv = self._apply_convention(pos_lemma)
            # Check if any candidate matches the POS-specific lemma
            for c in candidates:
                if c.lemma == pos_lemma_conv:
                    return c.lemma
            # Also check with accent-stripped comparison (POS tables and
            # lookup tables may use slightly different accent conventions)
            pos_stripped = strip_accents(pos_lemma_conv.lower())
            for c in candidates:
                if strip_accents(c.lemma.lower()) == pos_stripped:
                    return c.lemma
            # POS lemma not among candidates (e.g., single candidate from
            # lookup maps to a different headword). Trust the POS table -
            # it comes from curated sources (treebank, GLAUx, Wiktionary).
            if len(candidates) == 1:
                return pos_lemma_conv
            # Also trust the POS table when every candidate is just a
            # case/accent variant of the input form (i.e. nothing in
            # the candidate list disagrees with the POS table). This
            # catches forms like αυτού (homograph: pronoun gen sg of
            # αυτός vs adverb "there") where the lookup has only the
            # adverb's self-map and the αυτός pron form-of resolution
            # got hidden under it.
            word_stripped = strip_accents(word.lower())
            if all(strip_accents(c.lemma.lower()) == word_stripped
                   for c in candidates):
                return pos_lemma_conv

        if len(candidates) == 1:
            return candidates[0].lemma

        # MG self-map fix: when the top candidate is an MG self-map for
        # an ADJ/VERB inflection, prefer the citation form from combined/AG.
        mg_fix = self._fix_mg_selfmap(word, candidates, upos=upos)
        if mg_fix is not None:
            return mg_fix

        # No POS match among candidates - return top candidate
        # (same result as regular lemmatize)
        return candidates[0].lemma

    def lemmatize_batch_pos(self, words: list[str], upos_tags: list[str]) -> list[str]:
        """Lemmatize a batch of words with POS-aware disambiguation.

        POS is used to disambiguate among multiple candidates, or to
        override the single candidate when curated POS tables indicate
        a different lemma. Preserves the batch model optimization from
        lemmatize_batch() while applying POS corrections.

        Algorithm:
          1. Run lemmatize_batch() to get baseline results (efficient,
             batches model inference for unknown words).
          2. For each word where POS tables suggest a different lemma,
             call lemmatize_verbose() to get all candidates and check
             if the POS-specific lemma is among them.
          3. If a single candidate doesn't match the POS lemma, trust
             the POS table (curated sources).
          4. For MG self-map fix: when _prefer_mg is true and POS is
             ADJ/VERB, check if the baseline result is an MG self-map
             and prefer a citation-form alternative.
          5. If a match is found, use it. Otherwise keep the baseline.

        Args:
            words: List of Greek word forms.
            upos_tags: List of UPOS tags, one per word.

        Returns:
            List of lemma strings.
        """
        assert len(words) == len(upos_tags), (
            f"words and upos_tags must have same length: {len(words)} vs {len(upos_tags)}"
        )

        # Step 1: Get baseline results from regular batch lemmatization
        results = self.lemmatize_batch(words)

        # Step 2: For each word, check if POS could improve the result
        for i, (word, upos) in enumerate(zip(words, upos_tags)):
            pos_lemma = self._pos_table_lookup(word, upos)
            if pos_lemma is not None:
                pos_lemma_conv = self._apply_convention(pos_lemma)
                if pos_lemma_conv == results[i]:
                    # POS agrees with baseline - no change needed
                    continue

                # POS suggests a different lemma than baseline. Check if
                # the POS lemma is among the valid candidates.
                candidates = self.lemmatize_verbose(word)

                # Check if any candidate matches the POS-specific lemma
                pos_stripped = strip_accents(pos_lemma_conv.lower())
                matched = False
                for c in candidates:
                    if c.lemma == pos_lemma_conv:
                        results[i] = c.lemma
                        matched = True
                        break
                if not matched:
                    # Try accent-stripped comparison
                    for c in candidates:
                        if strip_accents(c.lemma.lower()) == pos_stripped:
                            results[i] = c.lemma
                            matched = True
                            break
                if not matched and len(candidates) <= 1:
                    # Single candidate doesn't match POS lemma - trust the
                    # POS table (curated sources: treebank, GLAUx, Wiktionary)
                    results[i] = pos_lemma_conv
                    matched = True
                if matched:
                    continue

            # MG self-map fix: when the baseline result is an MG self-map
            # for an ADJ/VERB inflection, prefer the citation form.
            # Also handles ADJ feminine-to-masculine fix (not a self-map
            # but still a wrong lemma form).
            if results[i] is not None and upos in ("ADJ", "VERB"):
                candidates = self.lemmatize_verbose(word)
                if len(candidates) > 1:
                    mg_fix = self._fix_mg_selfmap(word, candidates, upos=upos)
                    if mg_fix is not None:
                        results[i] = mg_fix

        return results

    def lemmatize_verbose(self, word: str,
                          prev_word: str | None = None,
                          ) -> list[LemmaCandidate]:
        """Return all candidate lemmas with metadata.

        Unlike lemmatize(), this returns multiple candidates for
        ambiguous forms, tagged with language, proper noun status,
        and source. Useful for downstream tools that can use context
        to disambiguate.

        Args:
            word: The Greek word to lemmatize.
            prev_word: Optional preceding word. If this is a Greek article
                (e.g. ὁ, τήν, τῶν), it is used to rank candidates by
                gender/number agreement. Only affects ranking, not
                filtering - all candidates are still returned.

        Examples:
            lemmatize_verbose("ἔριδι")
            -> [LemmaCandidate(lemma="Ἔρις", lang="grc", proper=True, ...),
                LemmaCandidate(lemma="ἔρις", lang="grc", proper=False, ...)]

            lemmatize_verbose("πόλεμο")
            -> [LemmaCandidate(lemma="πόλεμος", lang="el", ...),
                LemmaCandidate(lemma="πόλεμος", lang="grc", ...)]

            lemmatize_verbose("ἀλλ̓")
            -> [LemmaCandidate(lemma="ἀλλά", lang="grc", source="elision", via="elision:ά")]
        """
        candidates = []
        seen = set()  # track (lemma_lower, lang) to avoid exact dupes

        def _add(lemma, lang="", source="", via="", score=1.0):
            key = (lemma, lang)
            if key not in seen:
                seen.add(key)
                candidates.append(LemmaCandidate(
                    lemma=lemma,
                    lang=lang or self._lang_of(lemma),
                    proper=self._is_proper(lemma),
                    source=source,
                    score=score,
                    via=via,
                ))

        # 0. Digit-only passthrough
        if word.isdigit():
            _add(word, source="identity")
            return candidates

        # 1. Article/pronoun
        if self._resolve_articles:
            closed = self._resolve_closed_class(word)
            if closed is not None:
                _add(closed, source="article")
                return candidates

        # 2. Crasis
        from .crasis import resolve_crasis
        cr = resolve_crasis(word) or resolve_crasis(to_monotonic(word))
        if cr:
            _add(cr, source="crasis")
            return candidates

        # 3. Elision expansion — collect ALL valid expansions (before
        #    lookup, since elided forms false-match letter headwords)
        elision_results = self._expand_elision_all(word)
        for expanded, lemma, vowel in elision_results:
            lang = self._lang_of(expanded) or self._lang_of(lemma)
            _add(lemma, lang=lang, source="elision", via=f"elision:{vowel}")

        # 4. Lookup — collect from ALL language tables
        lower = word.lower()
        mono = to_monotonic(lower)
        stripped = strip_accents(lower)
        variants = [
            (word, "exact"), (lower, "lower"),
            (mono, "mono"), (stripped, "stripped"),
        ]

        # Respect self.lang: when constructed with lang='el' (MG mode),
        # don't leak AG candidates (and vice versa). This matters for
        # lemmatize_pos() which returns the first matching candidate:
        # αυτό in MG mode must not match AG's αὐτός, and είναι (MG copula)
        # must not match AG's εἰμί.
        _tables: list[tuple[object, str]] = []
        if self.lang in ("all", "el"):
            _tables.append((self._mg_lookup, "el"))
            _tables.append((self._med_lookup, "med"))
        if self.lang in ("all", "grc"):
            _tables.append((self._ag_lookup, "grc"))
        for table, lang in _tables:
            for variant, via in variants:
                lemma = table.get(variant)
                if lemma:
                    # Skip trivial short self-mappings (accent artifacts)
                    if len(variant) <= 2 and lemma == variant and via in ("mono", "stripped"):
                        continue
                    _add(lemma, lang=lang, source="lookup", via=via)
                    # Also check if the OTHER case variant is a headword
                    # (Ἔρις the goddess vs ἔρις strife)
                    if lemma[0].isupper():
                        alt = lemma[0].lower() + lemma[1:]
                    else:
                        alt = lemma[0].upper() + lemma[1:]
                    # The alt must be a self-mapping headword (not just
                    # a form that maps elsewhere)
                    alt_lemma = table.get(alt)
                    if alt_lemma == alt:
                        _add(alt, lang=lang, source="lookup",
                             via=via + "+case_alt")
                    break  # first matching variant wins per language

        # 5. Particle suffix stripping (after lookup, before model)
        if not candidates:
            particle_lemma = self._strip_particle_suffix(word)
            if particle_lemma:
                lang = self._lang_of(particle_lemma) or "grc"
                _add(particle_lemma, lang=lang, source="particle_strip",
                     via="suffix_strip", score=0.9)

        # 6. Verb morphology stripping (augment, reduplication)
        if not candidates:
            verb_morph_lemma = self._strip_verb_morphology(word)
            if verb_morph_lemma:
                lang = self._lang_of(verb_morph_lemma) or "grc"
                _add(verb_morph_lemma, lang=lang, source="verb_morphology",
                     via="augment_strip", score=0.85)

        # 7. Normalizer: try orthographic variants against all tables
        if not candidates and self._normalizer:
            for norm_candidate in self._normalizer.normalize(word):
                norm_lower = norm_candidate.lower()
                norm_mono = to_monotonic(norm_lower)
                norm_stripped = strip_accents(norm_lower)
                norm_variants = [
                    (norm_candidate, "normalize"),
                    (norm_lower, "normalize+lower"),
                    (norm_mono, "normalize+mono"),
                    (norm_stripped, "normalize+stripped"),
                ]
                for table, lang in _tables:
                    for variant, via in norm_variants:
                        lemma = table.get(variant)
                        if lemma:
                            if len(variant) <= 2 and lemma == variant and via.endswith(("mono", "stripped")):
                                continue
                            _add(lemma, lang=lang, source="normalize", via=via)
                            break

        # 8. Model fallback (if no candidates yet)
        model_identity = False
        if not candidates:
            try:
                self._load_model()
                pred = self._predict([word])[0]
                if strip_accents(pred.lower()) != strip_accents(word.lower()):
                    _add(pred, source="model", score=0.5)
                else:
                    model_identity = True
            except (FileNotFoundError, RuntimeError, ImportError, IndexError):
                model_identity = True

        # 9. Light Byzantine normalization (only when model returned identity)
        if model_identity and not candidates:
            byz = self._normalize_byzantine(word, use_normalizer=False)
            if byz:
                _add(byz, source="byzantine_norm", score=0.7)

        # 10. Compound decomposition (only when model returned identity)
        if model_identity:
            for compound, base_lemma, prefix in self._decompose_compound_all(word):
                _add(compound, source="compound",
                     via=f"{prefix}+{base_lemma}", score=0.65)

        # 11. Prefix stripping (only when model returned identity)
        if model_identity and not candidates:
            prefix_hit = self._prefix_strip_lookup(word)
            if prefix_hit:
                _add(prefix_hit, source="prefix_strip", score=0.6)

        # 12. Heavy Byzantine normalization with normalizer
        if model_identity and not candidates:
            byz_heavy = self._normalize_byzantine(
                word, use_normalizer=True)
            if byz_heavy:
                _add(byz_heavy, source="byzantine_norm", score=0.5)

        # If still nothing, return the word itself
        if not candidates:
            _add(word, source="identity", score=0.0)

        # Refine lookup candidate scores using GLAUx corpus frequency.
        # Only activated when there are multiple lookup candidates to rank.
        lookup_candidates = [c for c in candidates if c.source == "lookup"]
        if len(lookup_candidates) > 1:
            # Collect frequencies for all lookup candidates' lemmas
            freq_pairs = []
            for c in lookup_candidates:
                stripped = strip_accents(c.lemma.lower())
                freq = self._get_glaux_freq(stripped)
                freq_pairs.append((c, freq))
            # Only refine if at least one candidate has frequency data
            has_freq = any(f > 0 for _, f in freq_pairs)
            if has_freq:
                max_log = max(math.log1p(f) for _, f in freq_pairs)
                for c, freq in freq_pairs:
                    if freq > 0:
                        # Scale log(1+freq) into [0.5, 1.0]
                        c.score = 0.5 + 0.5 * (math.log1p(freq) / max_log)
                    else:
                        # No corpus data: slight penalty vs attested forms
                        c.score = 0.55

        # Sort: non-proper before proper, then by score descending
        candidates.sort(key=lambda c: (c.proper, -c.score))

        # Apply convention remapping
        if self._convention_map:
            seen_remapped = set()
            remapped = []
            for c in candidates:
                c.lemma = self._apply_convention(c.lemma)
                key = (c.lemma, c.lang)
                if key not in seen_remapped:
                    seen_remapped.add(key)
                    remapped.append(c)
            candidates = remapped

        # Article-agreement disambiguation: if prev_word is a Greek article,
        # re-rank candidates by gender agreement (only re-ranks, never excludes)
        if prev_word is not None and len(candidates) > 1:
            candidates = self._rank_by_article_agreement(candidates, prev_word)

        return candidates

    def lemmatize_batch(self, words: list[str]) -> list[str]:
        """Lemmatize a batch of words. Uses model only for unknowns.

        If a convention is set, all output lemmas are remapped accordingly.
        """
        results = []
        model_indices = []
        model_words = []

        for i, word in enumerate(words):
            # Article/pronoun resolution
            closed = self._resolve_closed_class(word)
            if closed is not None:
                results.append(closed)
                continue

            # Crasis
            from .crasis import resolve_crasis
            cr = resolve_crasis(word) or resolve_crasis(to_monotonic(word))
            if cr:
                results.append(cr)
                continue

            # Elision expansion (before lookup — elided forms can
            # false-match letter headwords)
            elision_lemma = self._expand_elision(word)
            if elision_lemma:
                results.append(elision_lemma)
                continue

            # Lookup
            lemma = self._lookup_word(word)
            if lemma:
                results.append(lemma)
                continue

            # Particle suffix stripping
            particle_lemma = self._strip_particle_suffix(word)
            if particle_lemma:
                results.append(particle_lemma)
                continue

            # Verb morphology stripping (augment, reduplication)
            verb_morph_lemma = self._strip_verb_morphology(word)
            if verb_morph_lemma:
                results.append(verb_morph_lemma)
                continue

            # Normalizer: try orthographic variants against lookup
            if self._normalizer:
                norm_hit = None
                for candidate in self._normalizer.normalize(word):
                    norm_hit = self._lookup_word(candidate)
                    if norm_hit:
                        break
                if norm_hit:
                    results.append(norm_hit)
                    continue

            results.append(None)
            model_indices.append(i)
            model_words.append(word)

        if model_words:
            try:
                self._load_model()
                predictions = self._predict(model_words)
            except (RuntimeError, IndexError, ImportError, FileNotFoundError):
                predictions = model_words  # identity fallback
            for idx, word, pred in zip(model_indices, model_words, predictions):
                # Fallback strategies when model returns identity
                if strip_accents(pred.lower()) == strip_accents(word.lower()):
                    # 1. Light Byzantine normalization (safe/precise)
                    byz = self._normalize_byzantine(
                        word, use_normalizer=False)
                    if byz:
                        pred = byz
                    # 2. Compound decomposition (linking vowel split)
                    elif (compound := self._decompose_compound(word)):
                        pred = compound
                    # 3. Prefix stripping (known prefixes + base)
                    elif (pfx := self._prefix_strip_lookup(word)):
                        pred = pfx
                    # 4. Heavy Byzantine normalization (with normalizer)
                    else:
                        byz_heavy = self._normalize_byzantine(
                            word, use_normalizer=True)
                        if byz_heavy:
                            pred = byz_heavy
                results[idx] = pred

        # Apply convention remapping to all results
        if self._convention_map:
            results = [self._apply_convention(r) if r else r for r in results]

        return results

    # ---- Spelling correction ----

    # Greek lowercase letters for ED1 candidate generation
    _GREEK_LETTERS = "αβγδεζηθικλμνξοπρσςτυφχψω"

    def _build_spell_index(self):
        """Build the accent-stripped index for spelling correction.

        With SQLite backend: no-op (queries use the indexed 'stripped' column).
        With JSON fallback: builds in-memory norm map from dict.
        """
        if hasattr(self, "_spell_norm_map"):
            return
        if self._using_db:
            # SQLite path: no in-memory index needed
            self._spell_norm_map = None
            self._spell_norm_set = None
            return
        norm_map: dict[str, set[str]] = {}
        for form in self._lookup:
            stripped = strip_accents(form.lower())
            if stripped not in norm_map:
                norm_map[stripped] = set()
            norm_map[stripped].add(form)
        self._spell_norm_map = norm_map
        self._spell_norm_set = set(norm_map.keys())

    @staticmethod
    def _edits1(word: str) -> set[str]:
        """Generate all strings within edit distance 1 of word.

        Operations: deletes, transposes, replaces, inserts.
        Uses Greek lowercase alphabet for replacements and insertions.
        """
        letters = Dilemma._GREEK_LETTERS
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def suggest_spelling(self, word: str, max_distance: int = 2,
                         ocr_mode: bool = False,
                         headwords_only: str | None = None,
                         lemmata_only: bool = False,
                         ) -> list[tuple[str, int | float]]:
        """Suggest spelling corrections for an unknown Greek word.

        Returns a list of (correct_form, edit_distance) tuples, sorted
        by edit distance then alphabetically. Uses a two-layer approach:

        1. Strip diacritics from the input and the dictionary, reducing
           8-11M entries to ~1-3M unique base forms
        2. Find ED0/ED1/ED2 matches on the stripped forms
        3. Return the original polytonic forms, ranked by actual
           Levenshtein distance to the input

        This means diacritic errors (wrong accent, missing breathing)
        cost 0 in the first layer and are corrected for free, while
        letter-level errors (θ/δ, ρ/ν) use standard edit distance.

        Args:
            word: The possibly-misspelled Greek word.
            max_distance: Maximum edit distance (1 or 2). Default 2.
            ocr_mode: If True, use weighted Levenshtein distance that
                gives lower cost to OCR-common confusions (Greek/Latin
                script mixing, Cyrillic contamination, θ/δ, ο/σ).
                This produces better rankings for OCR post-correction.
            headwords_only: If set to a convention name (e.g. "lsj",
                "cunliffe"), only return forms that are known headwords
                in that dictionary. This filters out inflected forms and
                headwords from other dictionaries, reducing false
                positives when resolving to a specific lexicon.
            lemmata_only: If True, only return forms that are lemmata
                (citation forms) in the lookup table, filtering out
                inflected forms. Less restrictive than headwords_only
                since it includes all lemmata, not just those in a
                specific dictionary.

        Returns:
            List of (corrected_form, distance) tuples. Empty if no
            suggestions found within max_distance.
        """
        self._build_spell_index()
        query_stripped = strip_accents(word.lower())

        if self._using_db:
            results = self._suggest_spelling_db(word, query_stripped,
                                                max_distance, ocr_mode)
        else:
            results = self._suggest_spelling_mem(word, query_stripped,
                                                 max_distance, ocr_mode)

        if headwords_only:
            hw_set = self._get_headword_set(headwords_only)
            results = [(form, dist) for form, dist in results
                       if form in hw_set]
        elif lemmata_only:
            lemma_set = self._get_lemma_set()
            results = [(form, dist) for form, dist in results
                       if form in lemma_set]

        return results

    def _suggest_spelling_db(self, word: str, query_stripped: str,
                             max_distance: int, ocr_mode: bool = False
                             ) -> list[tuple[str, int | float]]:
        """SQLite-backed spelling suggestion. No in-memory index needed.

        Generates ED1/ED2 candidate strings, then batch-queries the
        indexed 'stripped' column. ~1000 candidates for ED1, checked
        in one SQL query.
        """
        # For AG mode, filter to AG-sourced forms only (src='grc')
        # to avoid suggesting monotonic MG forms
        src_filter = 'grc' if self.lang == 'grc' else None

        # Collect candidate stripped forms at each distance level
        candidates: set[str] = set()

        # ED0: just the query itself
        candidates.add(query_stripped)

        # ED1: all edits of the stripped query
        if max_distance >= 1:
            ed1 = self._edits1(query_stripped)
            candidates.update(ed1)

        # Look up which candidates actually exist in the DB
        hits = self._lookup.spell_lookup_stripped(candidates,
                                                  src_filter=src_filter)

        # ED2: if few hits so far, expand
        if max_distance >= 2 and len(hits) < 3:
            ed2_candidates: set[str] = set()
            for e1 in ed1:
                ed2_candidates.update(self._edits1(e1))
            # Remove already-checked candidates
            ed2_candidates -= candidates
            ed2_hits = self._lookup.spell_lookup_stripped(ed2_candidates,
                                                          src_filter=src_filter)
            hits.update(ed2_hits)

        if not hits:
            return []

        return self._rank_spell_results(word, query_stripped, hits, ocr_mode)

    def _suggest_spelling_mem(self, word: str, query_stripped: str,
                              max_distance: int, ocr_mode: bool = False
                              ) -> list[tuple[str, int | float]]:
        """In-memory spelling suggestion (JSON fallback)."""
        norm_hits: set[str] = set()

        if query_stripped in self._spell_norm_set:
            norm_hits.add(query_stripped)

        if not norm_hits or max_distance >= 1:
            for candidate in self._edits1(query_stripped):
                if candidate in self._spell_norm_set:
                    norm_hits.add(candidate)

        if max_distance >= 2 and len(norm_hits) < 3:
            for e1 in self._edits1(query_stripped):
                for candidate in self._edits1(e1):
                    if candidate in self._spell_norm_set:
                        norm_hits.add(candidate)

        if not norm_hits:
            return []

        # Convert to {stripped: [original_forms]} format
        hits = {n: list(self._spell_norm_map[n]) for n in norm_hits}
        return self._rank_spell_results(word, query_stripped, hits, ocr_mode)

    @staticmethod
    def _has_breathing(s: str) -> bool:
        """Check if a string contains Greek breathing marks (polytonic)."""
        nfd = unicodedata.normalize("NFD", s)
        return "\u0313" in nfd or "\u0314" in nfd

    def _get_frequency(self, headword: str) -> int:
        """Get reference count for a headword from lsj9 frequency data.

        Lazy-loads lsj9_frequency.json on first call. Returns 0 for
        unknown headwords.
        """
        if self._hw_frequency is None:
            self._hw_frequency = {}
            if LSJ9_FREQUENCY_PATH.exists():
                with open(LSJ9_FREQUENCY_PATH, encoding="utf-8") as f:
                    self._hw_frequency = json.load(f)
        return self._hw_frequency.get(headword, 0)

    def _get_glaux_freq(self, stripped_form: str) -> int:
        """Get corpus token count for a stripped form.

        Lazy-loads corpus_freq.json on first call (extracts only the total
        count per form, discarding genre breakdowns). Returns 0 for
        forms not in the corpus.
        """
        if self._glaux_freq is None:
            self._glaux_freq = {}
            if CORPUS_FREQ_PATH.exists():
                with open(CORPUS_FREQ_PATH, encoding="utf-8") as f:
                    data = json.load(f)
                # Index 0 of each value array is the total count
                self._glaux_freq = {
                    form: counts[0]
                    for form, counts in data.get("forms", {}).items()
                }
        return self._glaux_freq.get(stripped_form, 0)

    def attestation(self, lemma: str) -> dict | None:
        """Corpus attestation profile for an Ancient Greek lemma, or None.

        Returns the per-lemma record from ``data/lemma_attestation.json``:
        token counts stratified by ``by_source`` (glaux / diorisis),
        ``by_genre`` (10 bins), ``by_century`` (signed century integer, where
        ``-8`` = 8th c. BC and ``2`` = 2nd c. AD), and ``by_dialect`` (GLAUx
        only; present only when known), plus ``total`` and ``dominant_pos``.
        Returns None for a lemma not attested in GLAUx or Diorisis.

        The argument is matched (after NFC normalization) against the corpora's
        own polytonic lemma annotation; it is not accent-stripped. Because the
        corpora are not sense-disambiguated, there is one profile per lemma
        string -- LSJ (A)/(B) homographs are not separated; use
        ``dominant_pos`` to pick a noun-vs-verb homograph. The artifact's
        ``_meta`` documents the full schema. Lazily loaded on first call.
        """
        if self._attestation is None:
            self._attestation = {}
            if ATTESTATION_PATH.exists():
                with open(ATTESTATION_PATH, encoding="utf-8") as f:
                    self._attestation = json.load(f).get("lemmas", {})
        return self._attestation.get(unicodedata.normalize("NFC", lemma))

    def _rank_spell_results(self, word: str, query_stripped: str,
                            hits: dict[str, list[str]],
                            ocr_mode: bool = False
                            ) -> list[tuple[str, int | float]]:
        """Rank spelling suggestions by edit distance.

        Sorts by (stripped_dist, full_dist, -frequency, form) so that
        at the same stripped distance, forms closer to the original
        polytonic input are preferred, then more common headwords win.
        Polytonic forms (with breathing marks) are preferred over
        monotonic when the input is polytonic or when using AG mode.

        In ocr_mode, uses weighted Levenshtein that gives lower cost
        to OCR-common character confusions (Greek/Latin script mixing,
        Cyrillic contamination, theta/delta, omicron/sigma).
        """
        prefer_polytonic = self.lang == "grc" or self._has_breathing(word)
        dist_fn = _weighted_levenshtein if ocr_mode else _levenshtein

        results: list[tuple[str, float, float]] = []
        for norm, originals in hits.items():
            stripped_dist = _levenshtein(query_stripped, norm)
            for original in originals:
                full_dist = dist_fn(word.lower(), original.lower())
                results.append((original, stripped_dist, full_dist))

        # Deduplicate: keep best (lowest stripped_dist, then full_dist)
        best: dict[str, tuple[float, float]] = {}
        for form, sd, fd in results:
            if form not in best or (sd, fd) < best[form]:
                best[form] = (sd, fd)

        if prefer_polytonic:
            by_sd: dict[float, list[str]] = {}
            for form, (sd, fd) in best.items():
                by_sd.setdefault(sd, []).append(form)
            for sd, forms in by_sd.items():
                poly = [f for f in forms if self._has_breathing(f)]
                if poly:
                    for f in forms:
                        if not self._has_breathing(f):
                            del best[f]

        # Sort by: stripped_dist, full_dist, then prefer common headwords
        return sorted(
            [(form, sd) for form, (sd, fd) in best.items()],
            key=lambda x: (x[1], best[x[0]][1],
                           -self._get_frequency(x[0]), x[0]),
        )

    def _predict(self, words: list[str], num_beams=4) -> list[str]:
        """Run model inference with beam search + headword filtering.

        Generates multiple candidates via beam search. Picks the
        highest-scoring candidate that is a known headword in the
        lookup table. If no candidate is a headword, returns the
        input word unchanged (better than a confidently wrong answer).

        Works with both PyTorch and ONNX backends transparently.
        """
        if not words:
            return []

        # Build headword set on first use (Wiktionary self-maps + LSJ + Cunliffe)
        if not hasattr(self, "_headwords") or self._headwords is None:
            self._headwords = {k for k, v in self._lookup.items() if k == v}
            if LSJ_HEADWORDS_PATH.exists():
                with open(LSJ_HEADWORDS_PATH, encoding="utf-8") as f:
                    self._headwords |= set(json.load(f))
            if CUNLIFFE_HEADWORDS_PATH.exists():
                with open(CUNLIFFE_HEADWORDS_PATH, encoding="utf-8") as f:
                    self._headwords |= set(json.load(f))

        max_len = max(len(w) for w in words) + 1
        src_ids = []
        for w in words:
            ids = self._vocab.encode(w)
            ids = ids + [0] * (max_len - len(ids))
            src_ids.append(ids)

        if getattr(self, '_use_onnx', False):
            import numpy as np
            # ONNX MHA reshapes require consistent sequence lengths.
            # Pad all inputs to a fixed max to avoid shape mismatches.
            ONNX_MAX_LEN = 48
            padded = []
            for ids in src_ids:
                if len(ids) < ONNX_MAX_LEN:
                    ids = ids + [0] * (ONNX_MAX_LEN - len(ids))
                padded.append(ids[:ONNX_MAX_LEN])
            src = np.array(padded, dtype=np.int64)
            src_pad_mask = (src == 0)
            beam_results = self._model.generate(
                src, src_key_padding_mask=src_pad_mask, num_beams=num_beams)
        else:
            import torch
            src = torch.tensor(src_ids, dtype=torch.long, device=self._device)
            src_pad_mask = (src == 0)
            with torch.no_grad():
                beam_results = self._model.generate(
                    src, src_key_padding_mask=src_pad_mask, num_beams=num_beams)

        results = []
        greedy = (num_beams == 1)
        onnx_mode = getattr(self, '_use_onnx', False)
        for i, candidates in enumerate(beam_results):
            if greedy and not onnx_mode:
                # num_beams=1 with PyTorch: beam_results is (batch, seq_len)
                # tensor, each row is token IDs with no score
                decoded = [self._vocab.decode(candidates)]
            else:
                # ONNX always returns list of (ids, score) tuples;
                # PyTorch beam search (num_beams>1) does the same
                decoded = [self._vocab.decode(ids) for ids, score in candidates]
            chosen = None
            for d in decoded:
                # Check headword with normalization cascade
                if any(v in self._headwords for v in (
                    d, d.lower(), to_monotonic(d), to_monotonic(d).lower(),
                    d[0].upper() + d[1:] if d else d,
                ) if v):
                    chosen = d
                    break
            if chosen is None:
                chosen = words[i]
            results.append(chosen)

        return results
