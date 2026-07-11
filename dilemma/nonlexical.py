"""Classify non-lexical tokens so they stop counting as lemmatization failures.

A large share of the "failures" in real (OCR'd) Greek corpora are not words at
all: they are editorial apparatus and typographic residue that leaked into the
text layer. A full-corpus census over the Open Greek Corpus found the worst
works (lexica, scholia, Patrologia Graeca) are dense with:

  - gamma-rho variant markers (``gr``, i.e. the abbreviation for GRAPHETAI)
    flagging a manuscript variant;
  - Greek-numeral / keraia tokens (kappa-zeta prime, rho prime, thousands-mark
    forms, bare lambda-stigma);
  - editorial bracket references (``[76]``, ``[49-59]``, ``(3)``);
  - Latin / citation abbreviations (``fr.``, ``p.``, ``Herod.``, ``Menand.``);
  - lone punctuation, sigla, and vowel-less consonant fragments.

None of these has a lemma. Sending them to the transformer fallback both wastes
the expensive layer and manufactures spurious "resolutions". This module
recognizes them structurally so a resolver can tag them NON-LEXICAL (UD POS
``X``) and a census-style consumer can exclude them from the failure
denominator, distinguishing "not a lexical word" from "failed to lemmatize a
real word".

Design constraint: it must NEVER misclassify a real word. Every rule is chosen
to be linguistically safe:

  - A lexical Greek word always contains a vowel, so a vowel-less Greek run is
    provably non-lexical.
  - Greek numerals are unaccented and their letters run in strictly descending
    place-value tiers (thousands > hundreds > tens > units); real words almost
    never satisfy both, and any that could are closed-class forms already
    resolved by the lookup layer before this classifier runs.
  - A token ending in an elision mark may be an elided monosyllable (delta +
    apostrophe = the elided form of the particle), so the vowel-less rule is
    suppressed there and single-letter apostrophe numerals are not claimed.

The classifier is pure stdlib (no lookup DB, no model). It is applied by
``dilemma.core`` just before the transformer fallback, and exposed publicly as
``Dilemma.classify_nonlexical`` / ``Dilemma.is_lexical`` plus the module-level
``classify_nonlexical`` / ``is_lexical`` functions.
"""
from __future__ import annotations

import unicodedata

# ---- Non-lexical class labels (also usable as a POS = "X" refinement) --------

VARIANT_MARK = "variant-mark"       # graphetai variant marker
NUMERAL = "numeral"                 # Greek alphabetic numeral (marked)
BRACKET_REF = "bracket-ref"        # editorial reference: [76], [49-59], (3)
ABBREVIATION = "abbreviation"      # Latin/citation abbr: fr., p., Herod.
SYMBOL = "symbol"                  # lone punctuation / siglum / bare number
CONSONANT_CLUSTER = "consonant-cluster"  # vowel-less Greek run (no lemma)

#: All non-lexical class labels, in classification-priority order.
NONLEXICAL_CLASSES = (
    VARIANT_MARK,
    NUMERAL,
    BRACKET_REF,
    ABBREVIATION,
    SYMBOL,
    CONSONANT_CLUSTER,
)

#: UD POS tag a non-lexical token maps to.
NONLEXICAL_POS = "X"

# ---- Character sets ----------------------------------------------------------

_GREEK_VOWELS = set("αεηιουω"  # a e h i o u w
                    "ΑΕΗΙΟΥΩ")

# Unambiguous numeral signs: the keraia (U+0374) and its NFC form, the modifier
# letter prime (U+02B9). A token bearing one is a numeral even if a single
# letter.
_KERAIA_MARKS = {"ʹ", "ʹ"}

# Apostrophe-like / breathing marks that double as the elision mark. A token
# ending in one of these may be an elided real monosyllable, so the vowel-less
# rule is suppressed and a single-letter apostrophe numeral is not claimed.
_ELISION_TAIL_MARKS = {
    "'",  # ' apostrophe
    "’",  # ' right single quote
    "‘",  # ' left single quote
    "ʼ",  # modifier letter apostrophe
    "ʽ",  # modifier letter reversed comma
    "᾽",  # koronis
    "᾿",  # Greek psili (spacing)
    "῾",  # Greek dasia (spacing)
    "̓",  # combining psili
    "̔",  # combining dasia
    "´",  # acute accent (spacing)
    "`",  # grave accent (spacing)
    "ˊ",  # modifier letter acute
    "ˋ",  # modifier letter grave
}
_NUMERAL_TAIL_MARKS = _KERAIA_MARKS | _ELISION_TAIL_MARKS
# Marks that prefix a Greek thousands numeral: the lower keraia (U+0375) or a
# comma glued to the leading letter (,a = 1000). A real word never starts with
# one of these fused to a letter.
_THOUSANDS_MARKS = {"͵", ","}

# Greek numeral place values. Tier 1 = units (1-9), 2 = tens (10-90),
# 3 = hundreds (100-900). Includes the numeral-only letters stigma (6),
# koppa (90) and sampi (900), and archaic digamma (a 6-glyph).
_NUM_TIER = {}
for _tier, _letters in (
    (1, "αβγδεϛϝζηθ"),  # 1-9
    (2, "ικλμνξοπϟϙ"),  # 10-90
    (3, "ρστυφχψωϡ"),        # 100-900
):
    for _ch in _letters:
        _NUM_TIER[_ch] = _tier

_BRACKET_OPEN = set("[({<⟨⟦")
_BRACKET_CLOSE = set("])}>⟩⟧")

_BRACKET_INNER_OK = set("-–—.,;· \t")


def _has_greek_letter(s: str) -> bool:
    for c in s:
        if ("Ͱ" <= c <= "Ͽ" or "ἀ" <= c <= "῿") \
                and unicodedata.category(c)[0] == "L":
            return True
    return False


def _has_latin_letter(s: str) -> bool:
    for c in s:
        if "a" <= c.lower() <= "z" and unicodedata.category(c)[0] == "L":
            return True
    return False


def _is_greek_numeral(token: str) -> bool:
    """True iff ``token`` is a well-formed, explicitly-marked Greek numeral.

    Strips an optional thousands prefix and an optional trailing keraia / prime
    / apostrophe, then requires the remaining base to be non-empty, carry no
    combining accents/breathings, contain only numeral letters, and run in
    strictly descending place-value tiers. The leading letter after a thousands
    mark is promoted a tier so a thousands+hundreds form parses.

    An explicit numeral marker is required. A single-letter numeral is accepted
    only with an unambiguous keraia/prime or thousands mark, never with a bare
    apostrophe, so an elided monosyllable is not taken for a numeral.
    """
    t = token
    thousands = False
    hard_mark = False   # keraia/prime or thousands mark (single-letter OK)
    soft_mark = False   # bare apostrophe (needs >=2 letters)
    while t and t[0] in _THOUSANDS_MARKS:
        thousands = True
        hard_mark = True
        t = t[1:]
    while t and t[-1] in _NUMERAL_TAIL_MARKS:
        if t[-1] in _KERAIA_MARKS:
            hard_mark = True
        else:
            soft_mark = True
        t = t[:-1]
    if not t or not (hard_mark or soft_mark):
        return False
    # No combining marks: numerals are bare unaccented letters.
    if any(unicodedata.category(c) == "Mn"
           for c in unicodedata.normalize("NFD", t)):
        return False
    if any(unicodedata.category(c)[0] != "L" for c in t):
        return False  # stray non-letter residue inside the base
    tiers = []
    for i, ch in enumerate(t):
        tier = _NUM_TIER.get(ch.lower())
        if tier is None:
            return False
        if thousands and i == 0:
            if tier != 1:
                return False
            tier = 4  # units glyph promoted to the thousands place
        tiers.append(tier)
    if len(t) < 2 and not hard_mark:
        return False  # single letter + bare apostrophe is not claimed
    return all(tiers[i] > tiers[i + 1] for i in range(len(tiers) - 1))


def _is_variant_mark(token: str) -> bool:
    """True for the graphetai variant marker (gamma-rho, optionally trailed by
    a period or apostrophe)."""
    t = token
    while t and t[-1] in ({".", ":"} | _NUMERAL_TAIL_MARKS):
        t = t[:-1]
    return t == "γρ"  # gamma rho


def _is_bracket_ref(token: str) -> bool:
    """True for an editorial reference like [76], [49-59], (3), <12>.

    The token must open with a bracket/paren and the interior must be a numeric
    reference (digits, with optional dash/comma/dot separators).
    """
    if len(token) < 2 or token[0] not in _BRACKET_OPEN:
        return False
    inner = token[1:]
    if inner and inner[-1] in _BRACKET_CLOSE:
        inner = inner[:-1]
    if not inner:
        return False
    saw_digit = False
    for c in inner:
        if c.isdigit():
            saw_digit = True
        elif c in _BRACKET_INNER_OK:
            continue
        else:
            return False
    return saw_digit


def classify_nonlexical(token: str) -> str | None:
    """Return a non-lexical class label for ``token``, or ``None`` if it looks
    like a real lexical word.

    The label is one of :data:`NONLEXICAL_CLASSES`. ``None`` means "this could
    be a real Greek word" - the caller should keep resolving it (lookup, rules,
    model). Empty input returns ``None`` (callers handle empties separately).
    """
    if not token:
        return None
    token = unicodedata.normalize("NFC", token)

    if not _has_greek_letter(token):
        # No Greek letter: editorial reference, foreign/citation abbreviation,
        # or bare punctuation/number.
        if _is_bracket_ref(token):
            return BRACKET_REF
        if _has_latin_letter(token):
            return ABBREVIATION
        return SYMBOL

    # Has a Greek letter. Order matters: the variant marker and numerals are
    # vowel-less too, so tag them by their specific class before the generic
    # consonant-cluster catch-all.
    if _is_variant_mark(token):
        return VARIANT_MARK
    if _is_greek_numeral(token):
        return NUMERAL
    # A token ending in an elision mark may be an elided monosyllable; leave it
    # to the resolver rather than calling it a consonant siglum.
    if token[-1] in _ELISION_TAIL_MARKS:
        return None
    # A lexical Greek word must contain a vowel. A Greek run with none is a
    # truncation / consonant siglum, never a word. Decompose first so accented
    # vowels are recognized by their base letter.
    nfd = unicodedata.normalize("NFD", token)
    greek_bases = [c for c in nfd
                   if unicodedata.category(c)[0] == "L"
                   and ("Ͱ" <= c <= "Ͽ" or "ἀ" <= c <= "῿")]
    if greek_bases and not any(c in _GREEK_VOWELS for c in greek_bases):
        return CONSONANT_CLUSTER
    return None


def is_lexical(token: str) -> bool:
    """True iff ``token`` is a candidate lexical word (has content and is not
    classified non-lexical). Empty/whitespace tokens are not lexical."""
    if not token or not token.strip():
        return False
    return classify_nonlexical(token) is None
