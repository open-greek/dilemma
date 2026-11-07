"""Spelling rules, syllable counting and frequency ratings for the monotonic
Modern Greek word list (``export_el_monotonic.py``) and its evaluation.

Nothing here reads files.

Syllables
---------
:func:`vowel_units` splits a word into vowel units: the digraphs αι, ει,
οι, ου, υι, αυ, ευ, ηυ (unless the first letter carries the tonos, as in
τσάι, or the second the dialytika, as in προϊόν) and the single vowels.
:func:`nuclei` then drops the units that synizesis makes a glide: an
unaccented ι, υ, ει or οι right before another vowel unit (μια, γεια,
ποιος, δυο, πιο, βοήθεια). So μια, ποιος and σπίτια count one and two
syllables, as they are pronounced, while μία (accented ί) and υιός (υι is
not a glide) keep theirs. The count only has to say whether a word is a
monosyllable and which syllable from the end carries the tonos, so the few
words where an unaccented ι before a vowel is a full syllable (ποιητής)
are miscounted harmlessly.

Spelling
--------
:func:`orthography_reason` returns None for a spelling the list may hold,
else the reason it may not:

* NFC, the monotonic letters of the Greek block only (no Greek Extended,
  no Latin look-alikes, no digits); U+2019 only as an elision mark at the
  end (απ’, είν’) or an aphaeresis mark at the start (’ναι, ’ρθει); the
  one comma is that of ``ό,τι``, which is always an entry;
* lowercase, or a capital initial for names; never all capitals (Greek
  capitals drop the tonos, so all-capital spellings go to the acronym
  list instead);
* ς only at the end, σ never at the end (σ’ keeps σ before the mark);
* the dialytika only on ι or υ, only after a vowel, and only where the
  two letters would otherwise read as a digraph (προϊόν, ταΐζω,
  Ευρωπαϊκή; not τσάϊ, πρωϊνό);
* at least one vowel; no letter three times in a row (σσσ, ναιιι);
* a polysyllable carries exactly one tonos, within the last three
  syllables; a monosyllable none, except ή, πού and πώς;
* so: no unaccented polysyllables (ειναι), no accented monosyllables (ναί,
  δέν, θά), no double accents (όνομά, the enclitic accent is added by the
  keyboard, not listed).

A monosyllable that synizesis made one (ποιός, μιά) gets its own reason,
``accented_synizesis``. The list rejects it when its unaccented spelling is
a word (ποιος, μια) and may keep it otherwise (ιός, a virus, is two
syllables).

Elided spellings keep the accent of the full word (είν’ from είναι, όλ’
from όλα), so an accented monosyllable is fine before the mark; one with
no tonos and two or more syllables is ``unaccented_elision``, which the
list keeps only when the full word carries its accent on the vowel that
was dropped (μεταξ’ from μεταξύ). An aphaeresis drops the first syllable,
and with it the accent when that syllable had it (’θελες from ήθελες).

Ratings
-------
:func:`bucket_for_rate` maps a rate per million words to the four ratings
of the keyboard's dictionary format: C at 37 per million and above, M at
3.7, R for any attested form below that, X for a form no frequency source
has. These are the per-million edges of the polytonic Modern Greek list
(``export_mg_polytonic.MONOTONIC_*``), so ratings mean the same in both.
"""

from __future__ import annotations

import unicodedata

APOSTROPHE = "’"
# ό,τι (whatever), the one word written with a comma (``el_text.OTI``).
OTI = "ό,τι"
C_PER_MILLION = 37.0
M_PER_MILLION = 3.7
BUCKETS = ("C", "M", "R", "X")
BUCKET_RANK = {"C": 3, "M": 2, "R": 1, "X": 0}

LOWER_PLAIN = "αβγδεζηθικλμνξοπρστυφχψως"
UPPER_PLAIN = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
LOWER_MARKED = "άέήίόύώϊϋΐΰ"
UPPER_MARKED = "ΆΈΉΊΌΎΏΪΫ"
LOWER_LETTERS = frozenset(LOWER_PLAIN + LOWER_MARKED)
UPPER_LETTERS = frozenset(UPPER_PLAIN + UPPER_MARKED)
LETTERS = LOWER_LETTERS | UPPER_LETTERS

# Each lowercase letter: (base letter, carries the tonos, carries the
# dialytika).
_MARKS = {
    "ά": ("α", True, False), "έ": ("ε", True, False), "ή": ("η", True, False),
    "ί": ("ι", True, False), "ό": ("ο", True, False), "ύ": ("υ", True, False),
    "ώ": ("ω", True, False), "ϊ": ("ι", False, True), "ϋ": ("υ", False, True),
    "ΐ": ("ι", True, True), "ΰ": ("υ", True, True),
}
VOWELS = frozenset("αεηιουω")
DIGRAPHS = frozenset({"αι", "ει", "οι", "ου", "υι", "αυ", "ευ", "ηυ"})
# Units that synizesis makes a glide before another vowel.
GLIDES = frozenset({"ι", "υ", "ει", "οι"})
# The monosyllables monotonic spelling accents (D10 of the keyboard's
# design): the disjunctive ή and the interrogatives πού and πώς.
ACCENTED_MONOSYLLABLES = frozenset({"ή", "πού", "πώς"})
_STRIP = str.maketrans({
    "ά": "α", "έ": "ε", "ή": "η", "ί": "ι", "ό": "ο", "ύ": "υ", "ώ": "ω",
    "ϊ": "ι", "ϋ": "υ", "ΐ": "ι", "ΰ": "υ", "ς": "σ",
})


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def lower_el(word: str) -> str:
    """Lowercase with the final sigma rule, keeping σ before U+2019."""
    low = word.lower()
    if "ς" + APOSTROPHE in low:
        low = low.replace("ς" + APOSTROPHE, "σ" + APOSTROPHE)
    return low


# The letter rows of the Greek keyboard layout, for slips onto a key beside
# the intended one.
KEYBOARD_ROWS = ("ςερτυθιοπ", "ασδφγηξκλ", "ζχψωβνμ")
NEIGHBOR_KEYS = {c: frozenset(row[j] for j in (k - 1, k + 1) if 0 <= j < len(row))
                 for row in KEYBOARD_ROWS for k, c in enumerate(row)}


def bare_key(word: str) -> str:
    """The accent-insensitive key: lowercase, no tonos, no dialytika, ς as
    σ. The keyboard's stripped key (``HunspellCompiler.strippedForm``)."""
    return lower_el(word).translate(_STRIP)


# Vowel digraphs and letters that are pronounced alike, for sound_key.
_SOUND_DIGRAPHS = {"αι": "ε", "ει": "ι", "οι": "ι", "υι": "ι", "ου": "u",
                   "αυ": "αv", "ευ": "εv", "ηυ": "ιv"}
_SOUND_LETTERS = {"η": "ι", "υ": "ι", "ω": "ο", "ς": "σ"}


def sound_key(word: str) -> str:
    """A key that spellings of the same pronunciation share, accents aside:
    ι, η, υ, ει, οι and υι are one letter, ο and ω another, ε and αι a
    third, and a doubled letter counts once. So κάθισε and κάθησε,
    γλυκιά and γλυκειά, συνέβη and συνέβει, νιώθω and νοιώθω share their
    key. A digraph is two letters whose first has no mark and whose second
    has no dialytika (τσάι and προϊόν have none)."""
    low = lower_el(word)
    out = []
    k = 0
    while k < len(low):
        c = low[k]
        base, accented, dialytika = _MARKS.get(c, (c, False, False))
        if k + 1 < len(low) and not accented and not dialytika:
            nxt, _, nxt_dialytika = _MARKS.get(low[k + 1], (low[k + 1], False, False))
            pair = _SOUND_DIGRAPHS.get(base + nxt)
            if pair and not nxt_dialytika:
                out.append(pair)
                k += 2
                continue
        out.append(_SOUND_LETTERS.get(base, base))
        k += 1
    key = "".join(out)
    return "".join(c for k, c in enumerate(key) if k == 0 or c != key[k - 1])


def tonos_count(word: str) -> int:
    return sum(1 for c in lower_el(word) if c in _MARKS and _MARKS[c][1])


def vowel_units(word: str) -> list[tuple[str, bool, str]]:
    """The vowel units of a lowercase word, in order, as ``(base letters,
    accented, next letter)``; the next letter is the base letter right
    after the unit, "" at the end. A letter with the dialytika keeps the
    mark in its unit (ϊ, ϋ), so it is never a glide. See the module
    docstring."""
    letters = [(_MARKS.get(c, (c, False, False))) for c in word]
    units: list[tuple[str, bool, str]] = []
    i = 0
    n = len(letters)
    while i < n:
        base, tonos, dial = letters[i]
        if base not in VOWELS:
            i += 1
            continue
        if (i + 1 < n and letters[i + 1][0] in VOWELS
                and base + letters[i + 1][0] in DIGRAPHS
                and not letters[i + 1][2] and not tonos and not dial):
            nxt = letters[i + 2][0] if i + 2 < n else ""
            units.append((base + letters[i + 1][0], letters[i + 1][1], nxt))
            i += 2
            continue
        nxt = letters[i + 1][0] if i + 1 < n else ""
        units.append((base if not dial else base + "̈", tonos, nxt))
        i += 1
    return units


def nuclei(word: str) -> list[tuple[str, bool]]:
    """The syllable nuclei of a word, counting synizesis: ``(unit,
    accented)`` from first to last.

    Two kinds of unit are glides, not nuclei: an unaccented ι, υ, ει or οι
    before another vowel (μια, ποιος, βοήθεια, Μάιος), and an unaccented ι
    after an accented vowel and before a consonant (κορόιδο, χάιδεψε,
    βόιδι), where the accent keeps the two letters from being a digraph
    but they still make one syllable. A final ι after an accented vowel is
    its own syllable (ρολόι, τσάι)."""
    low = lower_el(word).replace(APOSTROPHE, "")
    out = []
    after_accented_vowel = False    # the unit right before is an accented
    for unit, accented, nxt in vowel_units(low):   # single vowel, adjacent
        glide = (not accented and (
            (nxt in VOWELS and unit in GLIDES)
            or (unit == "ι" and after_accented_vowel and nxt
                and nxt not in VOWELS)))
        after_accented_vowel = accented and len(unit) == 1 and nxt == "ι"
        if glide:
            continue
        out.append((unit, accented))
    return out


def syllable_count(word: str) -> int:
    return len(nuclei(word))


def accent_position(word: str) -> int | None:
    """The syllable carrying the tonos, counted from the end (1 is the
    last), or None for no tonos or more than one."""
    marks = [k for k, (_, accented) in enumerate(nuclei(word)) if accented]
    if len(marks) != 1:
        return None
    return len(nuclei(word)) - marks[0]


def is_proparoxytone(word: str) -> bool:
    """Stressed on the third syllable from the end, counting synizesis
    (άνθρωπος, but not σπίτια). The keyboard adds the enclitic's second
    accent to these by rule."""
    return accent_position(word) == 3


def merged_by_synizesis(low: str) -> bool:
    """True when synizesis removed at least one syllable from ``low``."""
    return len(vowel_units(low)) > len(nuclei(low))


def case_shape(word: str) -> str:
    """"lower", "title" (capital initial, the rest lowercase), "upper" (two
    or more letters, all capitals) or "mixed"."""
    letters = [c for c in word if c in LETTERS]
    if not letters:
        return "mixed"
    if all(c in LOWER_LETTERS for c in letters):
        return "lower"
    if letters[0] in UPPER_LETTERS and all(c in LOWER_LETTERS for c in letters[1:]):
        return "title" if len(letters) > 1 else "upper"
    if len(letters) >= 2 and all(c in UPPER_LETTERS for c in letters):
        return "upper"
    return "mixed"


def orthography_reason(form: str, *, allow_capital: bool = True,
                       allow_elision: bool = True) -> str | None:
    """Why ``form`` may not be a list entry, or None. See the module
    docstring for the rules and the reasons."""
    if not form:
        return "empty"
    if nfc(form) != form:
        return "not_nfc"
    if form == OTI:
        return None
    lead = form.startswith(APOSTROPHE)
    trail = form.endswith(APOSTROPHE) and len(form) > 1
    core = form[1 if lead else 0:len(form) - (1 if trail else 0)]
    if not core:
        return "empty"
    if any(c not in LETTERS for c in core):
        if APOSTROPHE in core:
            return "apostrophe_inside"
        if any("a" <= c.lower() <= "z" for c in core):
            return "latin"
        return "not_monotonic_greek"
    if (lead or trail) and not allow_elision:
        return "elision_not_allowed"
    shape = case_shape(core)
    if shape == "upper" or (shape == "title" and len(core) == 1):
        return "all_caps"
    if shape == "mixed":
        return "mixed_case"
    if shape == "title" and not allow_capital:
        return "capital_not_allowed"
    low = lower_el(core)
    if "ς" in low[:-1]:
        return "medial_final_sigma"
    if trail and low.endswith("ς"):
        return "final_sigma_before_apostrophe"
    if not trail and low.endswith("σ"):
        return "final_sigma"
    letters = [_MARKS.get(c, (c, False, False)) for c in low]
    for k, (base, _, dial) in enumerate(letters):
        if not dial:
            continue
        if k == 0 or letters[k - 1][0] not in VOWELS:
            return "dialytika_without_vowel_before"
        before, before_accented, _ = letters[k - 1]
        if before_accented or before + base not in DIGRAPHS:
            # Nothing to keep apart: τσάι, ρολόι and πρωινό need no
            # dialytika (τσάϊ, ρολόϊ, πρωϊνό are old spellings).
            return "unneeded_dialytika"
    if not any(base in VOWELS for base, _, _ in letters):
        # σ’, τ’, μ’, ν’, θ’: a one-letter elision.
        return None if trail and not lead and len(low) == 1 else "no_vowel"
    for k in range(len(low) - 2):
        if low[k] == low[k + 1] == low[k + 2]:
            return "repeated_letters"
    marks = tonos_count(low)
    if marks > 1:
        return "double_accent"
    syllables = nuclei(low)
    if trail or lead:
        if marks == 0 and len(syllables) >= 2:
            return None if lead else "unaccented_elision"
        if marks == 1:
            position = accent_position(low)
            if position is None or position > 3:
                return "accent_too_early"
        return None
    if len(syllables) <= 1:
        if marks == 0 or low in ACCENTED_MONOSYLLABLES:
            return None
        return ("accented_synizesis" if merged_by_synizesis(low)
                else "accented_monosyllable")
    if marks == 0:
        return "unaccented_polysyllable"
    position = accent_position(low)
    if position is None or position > 3:
        return "accent_too_early"
    return None


def rate_per_million(count: float, words: int) -> float:
    return count * 1_000_000 / words if words else 0.0


def bucket_for_rate(rate: float) -> str:
    """The rating of a rate per million words (see the module docstring)."""
    if rate >= C_PER_MILLION:
        return "C"
    if rate >= M_PER_MILLION:
        return "M"
    if rate > 0:
        return "R"
    return "X"


def wilson_lower(n: float, total: float, z: float = 1.96) -> float:
    """The lower bound of the Wilson score interval of ``n`` / ``total``."""
    if total <= 0:
        return 0.0
    p = n / total
    zz = z * z
    center = p + zz / (2 * total)
    spread = z * (p * (1 - p) / total + zz / (4 * total * total)) ** 0.5
    return (center - spread) / (1 + zz / total)
