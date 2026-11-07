"""Monotonic Modern Greek text normalization, sentence splitting and
tokenization for the monotonic Modern Greek next-word model and word list.

The tokens follow the word segmentation of a monotonic Modern Greek
keyboard, so the counts a model learns from them are counts of the words
the keyboard will look up:

* A word is a run of letters. Every other character breaks words: space,
  punctuation, digits, symbols, and the hyphen. A hyphenated compound
  (γερο-Ανέστης, ελληνο-τουρκικός) is therefore always two tokens, in
  order, and no token ever contains a hyphen. The one exception is the
  pronoun ``ό,τι`` (whatever, ``Ό,τι`` at the start of a sentence), one
  word written with a comma and no space: ``ό`` alone is no word, so it
  stays one token (``OTI``).
* The apostrophe U+2019 is part of a word, never a breaker. The elided
  word keeps it (σ’ αγαπώ is ``σ’`` ``αγαπώ``, απ’ το is ``απ’`` ``το``),
  an aphaeresis keeps it in front (θα ’ρθω is ``θα`` ``’ρθω``), and an
  apostrophe written between two letters with no space joins them into
  one token (σ’αγαπώ), exactly as the keyboard sees what was typed.
  U+0027, U+02BC, U+2018, U+0384, U+00B4, U+0060 and the Greek Extended
  koronis and psili, which writers use for the same mark, are folded to
  U+2019 when they act as an elision mark. When they act as quotation
  marks ('Ναι') they are dropped.
* Sentences end at the keyboard's sentence breakers: ``. ! ? ; · …`` and
  the line break. A period after a single letter (κ., Γ.), after a known
  abbreviation (σελ., δισ.) or inside a dotted abbreviation (π.χ., κ.λπ.,
  Τ.Ε.Ι.) does not end a sentence; the abbreviation becomes one token that
  keeps its periods. Such tokens are not words the keyboard can type and
  are easy to recognize by the period.
* A word that is not monotonic Greek becomes a placeholder: ``<NUM>`` for
  numbers (and Greek numerals such as Β΄), ``<FOREIGN>`` for Latin or other
  scripts, URLs and e-mail addresses. Placeholders keep the position of
  the material they replace, so no n-gram spans it without knowing.

Case is kept as written (names, sentence-initial capitals, acronyms).
:func:`lower_el` gives the lowercase form for counting, with the Greek
final sigma rule applied (Python's ``str.lower`` already applies it; this
function also keeps σ before the elision mark, σ’).

Normalizations applied to every document, each counted in the statistics
:func:`process_document` returns, so a run reports what it changed:

* NFC. This also maps the Greek Extended oxia letters (U+1F71 ...) to the
  tonos letters (U+03AC ...), U+037E to ``;`` and U+0387 to U+00B7.
* No-break spaces, zero-width characters and soft hyphens are removed or
  made spaces; the hyphen variants U+2010 and U+2011 become ``-``.
* A spacing tonos typed before a vowel (΄Οταν, the result of a dead key
  pressed in the wrong order) is combined with it (Όταν).
* Apostrophe variants are folded to U+2019 as described above.
* Latin letters that look like Greek letters inside a Greek word (τo with
  a Latin o, NOMOΣ) are replaced by the Greek letters when every Latin
  letter in the word has such a twin. Each repair is counted by type.
* A word-final σ becomes ς (not before U+2019, and not before a hyphen
  that joins the next word); ς before U+2019 becomes σ.

The filters drop text that is not monotonic Modern Greek. Each drop is
counted by its reason:

* documents: polytonic spelling, Katharevousa, mostly non-Greek words,
  typing without accents;
* sentences: any polytonic word, Katharevousa markers, all-capital
  headings (Greek capitals drop the tonos), mostly non-Greek words, typing
  without accents, no Greek word at all.

Nothing here reads files; ``el_corpus`` and ``extract_el_monotonic_lm.py``
do the input and output.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

APOSTROPHE = "\u2019"
# ό,τι (whatever): the one word with a comma inside, kept as one token.
OTI = "ό,τι"
NUM_TOKEN = "<NUM>"
FOREIGN_TOKEN = "<FOREIGN>"
PLACEHOLDERS = frozenset({NUM_TOKEN, FOREIGN_TOKEN})

# Characters writers use for the elision apostrophe. U+0384 and U+00B4 are
# the spacing tonos and acute, U+1FBD and U+1FBF the koronis and psili,
# U+0060 the grave accent; all of them stand in for U+2019 in typed text.
APOSTROPHE_VARIANTS = "'\u2019\u02bc\u2018\u0384\u00b4\u1fbd\u1fbf`"
# Signs that turn letters into a Greek numeral (Αʹ, Β΄ τάξη). U+0374
# normalizes to U+02B9 under NFC.
NUMERAL_SIGNS = "\u02b9\u2032"

_LOWER_VOWELS = "αεηιουω"
_UPPER_VOWELS = "ΑΕΗΙΟΥΩ"
_ACCENTED = dict(zip(_LOWER_VOWELS + _UPPER_VOWELS, "άέήίόύώΆΈΉΊΌΎΏ"))
_GREEK_CONSONANTS = frozenset("βγδζθκλμνξπρσςτφχψΒΓΔΖΘΚΛΜΝΞΠΡΣΤΦΧΨ")
# Single letters that elide before a vowel: σε, με, το/τα, να, θα.
_ELIDING_LETTERS = frozenset("σμτνθΣΜΤΝΘ")
# Words ending in a vowel that still take the elision mark (για, και).
_ELIDING_VOWEL_WORDS = frozenset({"γι", "κι"})
# Elided prepositions whose letters could also read as a Greek numeral
# (υπ’ is υπό, but Υ=400 and Π=80 also spell 480).
_ELIDED_NOT_NUMERALS = frozenset({"υπ", "απ", "επ", "αφ", "εφ", "υφ", "καθ"})

_MONOTONIC_WORD_RE = re.compile(
    "^[ΆΈ-ΊΌΎ-ΡΣ-ώ\u2019]+$"
)
_GREEK_CHAR_RE = re.compile("[\u0370-\u03ff\u1f00-\u1fff]")
_POLYTONIC_RE = re.compile("[\u1f00-\u1fff]")
_LATIN_LETTER_RE = re.compile("[A-Za-z\u00c0-\u024f]")
_OXIA_RE = re.compile(
    "[\u1f71\u1f73\u1f75\u1f77\u1f79\u1f7b\u1f7d"
    "\u1fbb\u1fc9\u1fcb\u1fdb\u1ff9\u1feb\u1ffb\u1fd3\u1fe3]"
)
_GREEK_OR_MARK_RE = re.compile(
    "^[\u0370-\u03ff\u1f00-\u1fff\u0300-\u036f\u2019]+$")

# Latin letters with a Greek twin of the same shape. Only words that mix
# the two scripts are repaired, and only when every Latin letter is here.
HOMOGLYPHS = {
    "A": "Α", "B": "Β", "E": "Ε", "H": "Η", "I": "Ι", "K": "Κ", "M": "Μ",
    "N": "Ν", "O": "Ο", "P": "Ρ", "T": "Τ", "X": "Χ", "Y": "Υ", "Z": "Ζ",
    "a": "α", "i": "ι", "k": "κ", "o": "ο", "p": "ρ", "u": "υ", "v": "ν",
    "w": "ω", "x": "χ",
    "\u00e1": "ά", "\u00e9": "έ", "\u00ed": "ί", "\u00f3": "ό", "\u00fa": "ύ",
    "\u00ef": "ϊ",
}

# Abbreviations that are written with a final period inside a sentence,
# compared in lowercase. Single letters (κ. for κύριος, initials such as
# Γ.) are handled by rule. The list was extended from the words that a
# lowercase word follows after a period in the extracted corpus (the
# abbreviation_candidates tables), keeping those that are not words.
KNOWN_ABBREVIATIONS = frozenset({
    "αγ", "αγγλ", "αι", "ακαδ", "αντιπρ", "απεβ", "αρ", "αριθ", "αριθμ",
    "άρθ", "αρχ", "αφ", "βλ", "γαλλ", "γεν", "γενν", "γερμ", "γραμ", "δηλ",
    "δημ", "δικ", "δις", "δισ", "δισεκ", "δολ", "δολλ", "δρ", "δραχ", "δρχ",
    "εδ", "εδαφ", "εδάφ", "εκ", "εκατ", "εκατομ", "εκδ", "έκδ", "ελ", "ελλ",
    "ελλην", "ενν", "επ", "επιστ", "ετ", "ευρ", "ημερ", "ημερομ", "ιδ",
    "ισλ", "ισπ", "ιταλ", "καθ", "κεφ", "κκ", "κλπ", "κτλ", "λατ", "λεωφ",
    "λιμεν", "λπ", "οδ", "οικ", "οικον", "ολλ", "παλ", "παρ", "πβ", "περ",
    "περιλ", "περιπτ", "πλ", "πληθ", "ποικ", "πορτ", "πρβλ", "πχ", "σελ",
    "σημ", "σιδηρ", "στ", "στοιχ", "στρ", "συμφ", "συντ", "σχ", "τετ",
    "τετρ", "τηλ", "τόμ", "τουρκ", "τρισ", "υπ", "υποπαρ", "υποπερ", "υφυπ",
    "φαιν", "φασμ", "φωτ", "χιλ", "χλμ", "χλστ", "χμ",
})
# Abbreviations that often end a sentence: followed by a capital, they do.
SENTENCE_FINAL_ABBREVIATIONS = frozenset({
    "κ.λπ.", "κλπ.", "κτλ.", "κ.ά.", "κ.ο.κ.", "κ.τ.λ.", "κ.λ.π.", "μ.χ.",
    "χλμ.", "χιλ.", "εκατ.", "εκατομ.", "δισ.", "δις.", "τρισ.", "δρχ.",
    "δραχ.", "δολ.", "λπ.",
})

# Monotonic spellings that only Katharevousa uses in running text.
KATHAREVOUSA_WORDS = frozenset({
    "ταις", "τω", "ούτω", "ουδέν", "ουδείς", "ουδεμία", "ουχί", "ήτο",
    "ήτον", "οίτινες", "αίτινες", "ήτις", "άτινα", "ειμή", "εστί", "εστίν",
})
# Infinitives in -ειν, datives in -οις, verbs in -ουσι(ν)/-ωσι(ν). Not -αις,
# which ends Modern verbs (κλαις, φταις); the article ταις is listed above.
_KATHAREVOUSA_SUFFIXES = ("ειν", "οις", "ουσι", "ουσιν", "ωσι", "ωσιν")
# Modern Greek words with one of those endings.
_NOT_KATHAREVOUSA = frozenset({"εντούτοις"})

# Unaccented proclitic particles joined to the next word with a hyphen
# (γερο-Ανέστης); unaccented by rule, so never evidence of missing accents.
_UNACCENTED_PROCLITICS = frozenset({
    "αγια", "γρια", "γερο", "θεια", "κυρα", "μαστρο", "μπαρμπα", "παπα",
})

SENTENCE_BREAK_CHARS = frozenset(".!?;·…\n")

REGISTER_WORDS = (
    "είμαι", "είσαι", "ξέρω", "εντάξει", "ευχαριστώ", "γεια", "μου", "σου",
    "εσύ", "θέλω", "επίσης", "άρθρο",
)

# Thresholds of the document and sentence filters.
DOC_MAX_POLYTONIC_SHARE = 0.02
DOC_MAX_KATHAREVOUSA_SHARE = 0.01
DOC_MIN_KATHAREVOUSA = 3            # and at least this many markers
DOC_MIN_POLYTONIC = 2               # and at least this many polytonic words
DOC_MAX_FOREIGN_SHARE = 0.5
DOC_MAX_UNACCENTED_SHARE = 0.25
SENT_MAX_KATHAREVOUSA = 1           # more than this many markers drops it
SENT_MAX_FOREIGN_SHARE = 0.5
SENT_MAX_UNACCENTED_SHARE = 0.25    # at or above this share drops it
SENT_ALL_CAPS_SHARE = 0.6

_CLEANUP = str.maketrans({
    "\u00a0": " ", "\u2007": " ", "\u202f": " ", "\u2009": " ", "\t": " ",
    "\u200b": None, "\u200c": None, "\u200d": None, "\u2060": None,
    "\ufeff": None, "\u00ad": None,
    "\u2010": "-", "\u2011": "-",
    "\r": "\n", "\u2028": "\n", "\u2029": "\n", "\u0085": "\n",
})

_SPACING_TONOS_RE = re.compile(
    "(?<![^\\W\\d_])[\u0384\u00b4]([" + _LOWER_VOWELS + _UPPER_VOWELS + "])"
)

_APOS = "[" + re.escape(APOSTROPHE_VARIANTS) + "]"
_LETTER = r"[^\W\d_]"
_TOKEN_RE = re.compile(
    r"(?P<url>(?:https?://|www\.)\S+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+)"
    r"|(?P<abbr>(?<![^\W\d_])(?:" + _LETTER + r"{1,4}\.){2,}(?![^\W\d_]))"
    r"|(?P<latindot>[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)+)"
    r"|(?P<num>\d(?:[\d.,:/]*\d)?" + _LETTER + r"*)"
    r"|(?P<oti>(?<![^\W\d_])[όΌ],τι(?![^\W\d_]))"
    r"|(?P<word>" + _APOS + r"?" + _LETTER
    + r"(?:" + _LETTER + r"|[\u0300-\u036f]|" + _APOS + r"(?=" + _LETTER
    + r"))*" + _APOS + r"?)"
    r"|(?P<brk>[.!?;·…\n]+)"
    r"|(?P<other>\S)"
)


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def lower_el(token: str) -> str:
    """Lowercase ``token`` the Greek way, for counting.

    Python's ``str.lower`` applies the final sigma rule (ΟΔΟΣ to οδος), but
    it treats U+2019 as transparent, so ΔΩΣ’ would become δως’. The elision
    mark keeps σ (σ’ is the elided σε), so ς before it becomes σ.
    """
    low = token.lower()
    if "ς\u2019" in low:
        low = low.replace("ς\u2019", "σ\u2019")
    return low


def sentence_key(tokens) -> int:
    """64-bit key of a sentence, case-folded, for duplicate detection.

    Punctuation is already gone and numbers are ``<NUM>``, so two sentences
    that differ only in those, or in case, share a key.
    """
    text = lower_el(" ".join(tokens))
    return int.from_bytes(
        hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little"
    )


def is_monotonic_word(token: str) -> bool:
    """True for a token made only of monotonic Greek letters and U+2019."""
    return bool(_MONOTONIC_WORD_RE.match(token))


def has_tonos(word: str) -> bool:
    return any(c in "άέήίόύώΐΰΆΈΉΊΌΎΏ" for c in word)


_DIGRAPHS = frozenset({"αι", "ει", "οι", "ου", "υι", "αυ", "ευ", "ηυ"})
_I_SOUNDS = frozenset({"ι", "η", "υ", "ει", "οι", "υι", "ϊ", "ϋ"})


def unaccented_nuclei(word: str) -> int:
    """Syllable nuclei of a lowercase word written without the tonos.

    Vowel digraphs (αι, ει, οι, ου, υι, αυ, ευ, ηυ) are one vowel, and an
    /i/ vowel before another vowel is a glide (synizesis), so μια, γεια,
    ποιος, δυο and πιο count one nucleus and stay valid without a tonos,
    while απο, ειναι and μπορει count two or more.
    """
    units: list[str] = []
    i = 0
    n = len(word)
    while i < n:
        c = word[i]
        if c in "αεηιουωϊϋ":
            pair = word[i:i + 2]
            if len(pair) == 2 and pair in _DIGRAPHS:
                units.append(pair)
                i += 2
                continue
            units.append(c)
        else:
            units.append("")
        i += 1
    nuclei = 0
    for k, u in enumerate(units):
        if not u:
            continue
        nxt = units[k + 1] if k + 1 < len(units) else ""
        if u in _I_SOUNDS and nxt and u not in ("ϊ", "ϋ"):
            continue  # glide before a vowel
        nuclei += 1
    return nuclei


def is_unaccented_polysyllable(word: str) -> bool:
    """A word that monotonic spelling requires to carry a tonos but lacks it.

    All-capital words are not judged (capitals drop the tonos), nor
    unaccented proclitics (γερο-), nor a word that lost its first syllable
    to aphaeresis, which keeps no tonos when that syllable carried it
    (τα ’θελες, θα ’λεγε).
    """
    if has_tonos(word) or word.isupper() or word.startswith(APOSTROPHE):
        return False
    low = lower_el(word).replace(APOSTROPHE, "")
    if low in _UNACCENTED_PROCLITICS:
        return False
    return unaccented_nuclei(low) >= 2


def is_katharevousa_marker(low: str, next_low: str = "") -> bool:
    """True for a lowercase monotonic word that marks Katharevousa."""
    if low in KATHAREVOUSA_WORDS:
        return True
    if low in _NOT_KATHAREVOUSA:
        return False
    if low == "τοις":
        # τοις εκατό (per cent) is standard Modern Greek.
        return next_low not in ("εκατό", "<NUM>")
    if len(low) >= 5 and low.endswith(_KATHAREVOUSA_SUFFIXES):
        return True
    return False


@dataclass
class Token:
    """One token of a sentence, before the filters."""

    text: str
    kind: str              # "word", "abbr", "num", "foreign", "polytonic"
    hyphen_prefix: bool = False   # directly followed by "-" and a letter


@dataclass
class ProcessedDocument:
    """What :func:`process_document` keeps of one document."""

    sentences: list[list[str]]
    drop_reason: str | None
    stats: Counter = field(default_factory=Counter)
    homoglyph_repairs: Counter = field(default_factory=Counter)
    abbreviation_candidates: Counter = field(default_factory=Counter)


def normalize_text(text: str, stats: Counter) -> str:
    """NFC plus the character cleanups described in the module docstring."""
    if _OXIA_RE.search(text):
        stats["norm_oxia_chars"] += len(_OXIA_RE.findall(text))
    out = nfc(text).translate(_CLEANUP)
    if "\u0384" in out or "\u00b4" in out:
        out, n = _SPACING_TONOS_RE.subn(lambda m: _ACCENTED[m.group(1)], out)
        if n:
            stats["norm_spacing_tonos_combined"] += n
    return out


def _repair_homoglyphs(word: str) -> str | None:
    """The Greek spelling of a word that mixes in Latin look-alikes."""
    if not _GREEK_CHAR_RE.search(word):
        return None
    out = []
    for c in word:
        if "\u0370" <= c <= "\u03ff" or c == APOSTROPHE:
            out.append(c)
        elif c in HOMOGLYPHS:
            out.append(HOMOGLYPHS[c])
        else:
            return None
    fixed = nfc("".join(out))
    return fixed if is_monotonic_word(fixed) else None


_NUMERAL_VALUES = {
    "α": 1, "β": 2, "γ": 3, "δ": 4, "ε": 5, "ϛ": 6, "ζ": 7, "η": 8, "θ": 9,
    "ι": 10, "κ": 20, "λ": 30, "μ": 40, "ν": 50, "ξ": 60, "ο": 70, "π": 80,
    "ϟ": 90, "ϙ": 90, "ρ": 100, "σ": 200, "τ": 300, "υ": 400, "φ": 500,
    "χ": 600, "ψ": 700, "ω": 800, "ϡ": 900,
}


def is_greek_numeral(letters: str) -> bool:
    """True when ``letters`` spell a Greek numeral (ΙΒ, ΣΤ, ρκγ).

    The letters must give at most one hundreds, one tens and one units
    value, in that order; ΣΤ (or στ) is the units digit 6. This tells the
    numerals that take a numeral sign (Β΄, ΙΑ΄) from elided words written
    in capitals (ΑΠ’, ΚΑΤ’), whose letters are not in that order.
    """
    low = letters.lower()
    if low.endswith("στ"):
        low = low[:-2] + "ϛ"
    values = []
    for c in low:
        v = _NUMERAL_VALUES.get(c)
        if v is None:
            return False
        values.append(v)
    places = [3 if v >= 100 else 2 if v >= 10 else 1 for v in values]
    return bool(values) and all(a > b for a, b in zip(places, places[1:]))


def _fold_apostrophes(raw: str, stats: Counter) -> tuple[str, bool]:
    """Resolve the apostrophe-like marks of one word match.

    Returns the word with U+2019 where a mark is an elision or aphaeresis
    mark and without the marks that are quotation marks, and whether the
    word is a Greek numeral (letters plus a numeral sign).
    """
    word = raw
    lead = word[0] in APOSTROPHE_VARIANTS
    trail = len(word) > 1 and word[-1] in APOSTROPHE_VARIANTS
    core = word[1 if lead else 0:len(word) - (1 if trail else 0)]
    if not core:
        return "", False
    if lead and trail:
        # 'Ναι': a quoted word, not an elision on both sides.
        lead = trail = False
        stats["norm_quote_marks_dropped"] += 2
    # Marks between two letters: an elision written without a space.
    if any(c in APOSTROPHE_VARIANTS for c in core):
        folded = "".join(APOSTROPHE if c in APOSTROPHE_VARIANTS else c
                         for c in core)
        if folded != core:
            stats["norm_apostrophe_folded"] += sum(
                1 for c in core if c in APOSTROPHE_VARIANTS and c != APOSTROPHE)
        stats["tok_joined_elision"] += 1
        core = folded
    out = core
    numeral = False
    if trail:
        mark = word[-1]
        letters = core.replace(APOSTROPHE, "")
        if len(letters) == 1:
            # β΄ and Β΄ are numerals; σ’ and Σ’ are the elided σε.
            numeral = letters not in _ELIDING_LETTERS
        elif (lower_el(letters) not in _ELIDED_NOT_NUMERALS
              and (core.isupper() or len(letters) <= 3)
              and is_greek_numeral(letters)):
            numeral = True            # ΣΤ΄, ΙΑ΄, ιβ΄
        if numeral:
            pass
        elif (core[-1] in _GREEK_CONSONANTS
              or lower_el(core) in _ELIDING_VOWEL_WORDS):
            out = core + APOSTROPHE
            stats["tok_elision"] += 1
            if mark != APOSTROPHE:
                stats["norm_apostrophe_folded"] += 1
        else:
            stats["norm_quote_marks_dropped"] += 1
    if lead and not numeral:
        mark = word[0]
        if core[0] in _GREEK_CONSONANTS:
            out = APOSTROPHE + out
            stats["tok_aphaeresis"] += 1
            if mark != APOSTROPHE:
                stats["norm_apostrophe_folded"] += 1
        else:
            stats["norm_quote_marks_dropped"] += 1
    return out, numeral


def _classify_word(word: str) -> str:
    """"word" (monotonic Greek), "num" (a Greek numeral), "polytonic" (Greek
    that is not monotonic: Greek Extended letters, archaic letters, stray
    combining marks) or "foreign" (any other script)."""
    if is_monotonic_word(word):
        return "word"
    if any(c in NUMERAL_SIGNS for c in word):
        return "num"
    if _POLYTONIC_RE.search(word) or _GREEK_OR_MARK_RE.match(word):
        return "polytonic"
    return "foreign"


def tokenize_text(
    text: str, stats: Counter, repairs: Counter, candidates: Counter,
    line_breaks: str = "sentence",
) -> list[tuple[list[Token], str]]:
    """Split normalized text into sentences of :class:`Token`.

    Returns ``(tokens, closer)`` pairs; ``closer`` is the break that ended
    the sentence: the punctuation, ``"\\n"`` for a line break, or ``""`` at
    the end of the text.

    ``line_breaks`` is ``"sentence"`` when every line break ends a sentence
    (the keyboard's rule, right for text typed in paragraphs) or ``"wrap"``
    when a single line break is layout and only a blank line ends a
    sentence.
    """
    if line_breaks == "wrap":
        text = re.sub(r"\n[ ]*\n+", "\u2029", text).replace("\n", " ")
        text = text.replace("\u2029", "\n")
    items = list(_TOKEN_RE.finditer(text))
    sentences: list[tuple[list[Token], str]] = []
    current: list[Token] = []

    def close(closer: str) -> None:
        nonlocal current
        if current:
            sentences.append((current, closer))
        current = []

    i = 0
    n = len(items)
    while i < n:
        m = items[i]
        kind = m.lastgroup
        raw = m.group()
        nxt = items[i + 1] if i + 1 < n else None
        adjacent = nxt is not None and nxt.start() == m.end()
        if kind == "oti":
            # ό,τι (whatever) is one word written with a comma; ό alone is
            # no word, so splitting it would leave a fragment.
            current.append(Token(raw, "word"))
            stats["tok_oti"] += 1
        elif kind == "word":
            word, numeral = _fold_apostrophes(raw, stats)
            if not word:
                i += 1
                continue
            if numeral:
                current.append(Token(NUM_TOKEN, "num"))
                i += 1
                continue
            wkind = _classify_word(word)
            if wkind == "foreign":
                fixed = _repair_homoglyphs(word)
                if fixed is not None:
                    repairs[(word, fixed)] += 1
                    stats["norm_homoglyph_repairs"] += 1
                    word, wkind = fixed, "word"
            # A period right after the word: abbreviation or sentence end.
            if (adjacent and nxt.lastgroup == "brk" and nxt.group() == "."
                    and wkind in ("word", "foreign")):
                letters = word.replace(APOSTROPHE, "")
                if len(letters) == 1 or lower_el(word) in KNOWN_ABBREVIATIONS:
                    abbr = word + "."
                    current.append(Token(
                        abbr if wkind == "word" else FOREIGN_TOKEN,
                        "abbr" if wkind == "word" else "foreign"))
                    stats["tok_abbreviation"] += 1
                    i += 2
                    after = items[i] if i < n else None
                    if (lower_el(abbr) in SENTENCE_FINAL_ABBREVIATIONS
                            and after is not None and after.lastgroup == "word"
                            and after.group()[:1].isupper()):
                        close(".")
                    continue
                if wkind == "word":
                    after = items[i + 2] if i + 2 < n else None
                    if (after is not None and after.lastgroup == "word"
                            and after.group()[:1].islower()):
                        if word.isupper() and is_greek_numeral(letters):
                            # ΙΑ. λαμβάνοντας: a numbered list item.
                            current.append(Token(NUM_TOKEN, "num"))
                            i += 2
                            continue
                        candidates[word] += 1
            if wkind == "word":
                hyphen_prefix = (
                    adjacent and nxt.group() == "-" and i + 2 < n
                    and items[i + 2].start() == nxt.end()
                    and items[i + 2].lastgroup == "word")
                if hyphen_prefix:
                    stats["tok_hyphen_split"] += 1
                if (word.endswith("σ") and len(word) > 1 and not hyphen_prefix):
                    word = word[:-1] + "ς"
                    stats["norm_final_sigma"] += 1
                if "ς\u2019" in word:
                    word = word.replace("ς\u2019", "σ\u2019")
                    stats["norm_sigma_before_apostrophe"] += 1
                current.append(Token(word, "word", hyphen_prefix))
            elif wkind == "num":
                current.append(Token(NUM_TOKEN, "num"))
            elif wkind == "polytonic":
                current.append(Token(word, "polytonic"))
            else:
                current.append(Token(FOREIGN_TOKEN, "foreign"))
        elif kind == "abbr":
            if _LATIN_LETTER_RE.search(raw) and not _GREEK_CHAR_RE.search(raw):
                current.append(Token(FOREIGN_TOKEN, "foreign"))
            else:
                current.append(Token(raw, "abbr"))
                stats["tok_abbreviation"] += 1
            after = nxt
            if (lower_el(raw) in SENTENCE_FINAL_ABBREVIATIONS
                    and after is not None and after.lastgroup == "word"
                    and after.group()[:1].isupper()):
                close(".")
        elif kind == "num":
            current.append(Token(NUM_TOKEN, "num"))
        elif kind in ("url", "latindot"):
            current.append(Token(FOREIGN_TOKEN, "foreign"))
        elif kind == "brk":
            # ".\n" was closed by its period; a bare line break by itself.
            close(raw.replace("\n", "")[-1:] or "\n")
        # "other": punctuation and symbols that do not end a sentence.
        i += 1
    close("")
    return sentences


@dataclass
class SentenceInfo:
    tokens: list[Token]
    closer: str
    greek: int = 0
    foreign: int = 0
    polytonic: int = 0
    katharevousa: int = 0
    judged: int = 0
    unaccented: int = 0
    caps: int = 0
    multi_letter: int = 0


def _measure(tokens: list[Token], closer: str) -> SentenceInfo:
    info = SentenceInfo(tokens, closer)
    lows = [lower_el(t.text) if t.kind == "word" else t.text for t in tokens]
    for k, t in enumerate(tokens):
        if t.kind == "word":
            info.greek += 1
            low = lows[k]
            nxt = lows[k + 1] if k + 1 < len(tokens) else ""
            if is_katharevousa_marker(low, nxt):
                info.katharevousa += 1
            letters = t.text.replace(APOSTROPHE, "")
            if len(letters) >= 2:
                info.multi_letter += 1
                if t.text.isupper():
                    info.caps += 1
                elif not t.hyphen_prefix:
                    info.judged += 1
                    if is_unaccented_polysyllable(t.text):
                        info.unaccented += 1
        elif t.kind == "polytonic":
            info.polytonic += 1
        elif t.kind == "foreign":
            info.foreign += 1
    return info


def sentence_drop_reason(info: SentenceInfo) -> str | None:
    """Why a sentence is not kept, or None."""
    if info.polytonic:
        return "polytonic"
    if info.greek == 0:
        return "no_greek"
    if info.katharevousa > SENT_MAX_KATHAREVOUSA:
        return "katharevousa"
    if info.foreign / (info.foreign + info.greek) > SENT_MAX_FOREIGN_SHARE:
        return "foreign"
    if info.multi_letter and (
            info.caps == info.multi_letter
            or (info.multi_letter >= 3
                and info.caps / info.multi_letter >= SENT_ALL_CAPS_SHARE)):
        return "all_caps"
    if info.unaccented and info.unaccented / info.judged >= SENT_MAX_UNACCENTED_SHARE:
        return "unaccented"
    return None


def document_drop_reason(infos: list[SentenceInfo]) -> str | None:
    """Why a whole document is not kept, or None."""
    greek = sum(s.greek for s in infos)
    poly = sum(s.polytonic for s in infos)
    foreign = sum(s.foreign for s in infos)
    if greek + poly == 0:
        return "no_greek"
    if poly >= DOC_MIN_POLYTONIC and poly / (greek + poly) > DOC_MAX_POLYTONIC_SHARE:
        return "polytonic"
    kath = sum(s.katharevousa for s in infos)
    if kath >= DOC_MIN_KATHAREVOUSA and kath / greek > DOC_MAX_KATHAREVOUSA_SHARE:
        return "katharevousa"
    if foreign / (foreign + greek + poly) > DOC_MAX_FOREIGN_SHARE:
        return "foreign"
    judged = sum(s.judged for s in infos)
    if judged >= 5 and sum(s.unaccented for s in infos) / judged > DOC_MAX_UNACCENTED_SHARE:
        return "unaccented"
    return None


def process_document(
    text: str,
    *,
    line_breaks: str = "sentence",
    require_terminal_punctuation: bool = False,
    stop_at_lines: frozenset[str] | set[str] = frozenset(),
) -> ProcessedDocument:
    """Normalize, split, tokenize and filter one document.

    ``require_terminal_punctuation`` drops sentences that a line break or
    the end of the text closed, not punctuation (headings, list items,
    category lines). ``stop_at_lines`` names lines (stripped) at which the
    useful text ends, such as a reference section heading.
    """
    stats: Counter = Counter()
    repairs: Counter = Counter()
    candidates: Counter = Counter()
    stats["docs_read"] = 1
    text = normalize_text(text or "", stats)
    if stop_at_lines:
        lines = text.split("\n")
        for k, line in enumerate(lines):
            if line.strip() in stop_at_lines:
                text = "\n".join(lines[:k])
                stats["doc_truncated_at_heading"] += 1
                break
    raw_sentences = tokenize_text(text, stats, repairs, candidates, line_breaks)
    infos = [_measure(toks, closer) for toks, closer in raw_sentences]
    for s in infos:
        stats["sents_read"] += 1
        stats["tokens_read"] += len(s.tokens)
    reason = document_drop_reason(infos)
    if reason is not None:
        stats["docs_dropped_" + reason] += 1
        stats["tokens_dropped_doc_" + reason] += sum(len(s.tokens) for s in infos)
        return ProcessedDocument([], reason, stats, Counter(), Counter())
    kept: list[list[str]] = []
    for s in infos:
        why = sentence_drop_reason(s)
        if why is None and require_terminal_punctuation and s.closer in ("", "\n"):
            why = "unterminated"
        if why is not None:
            stats["sents_dropped_" + why] += 1
            stats["tokens_dropped_sent_" + why] += len(s.tokens)
            continue
        kept.append([t.text for t in s.tokens])
    stats["docs_kept"] = 1 if kept else 0
    if not kept:
        stats["docs_empty_after_filters"] += 1
    return ProcessedDocument(kept, None, stats, repairs, candidates)


def count_greek_words(tokens) -> int:
    """Tokens that are words (not placeholders, not abbreviations)."""
    return sum(1 for t in tokens if t not in PLACEHOLDERS and not t.endswith("."))
