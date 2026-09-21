"""Shared normalization for accent-insensitive Ancient Greek corpus counts."""

from __future__ import annotations

import unicodedata


# U+02B9 MODIFIER LETTER PRIME is deliberately absent: the language model
# uses it for Greek numerals, not elision.
TEXTUAL_ELISION_MARKS = frozenset({"'", "`", "\u02bc", "\u1fbd", "\u2019"})
# Also exported for source cleaners which must retain spacing psili long enough
# for corpus_freq_key() to distinguish a breathing from an apostrophe.
ELISION_MARKS = TEXTUAL_ELISION_MARKS | {"\u1fbf"}
CANONICAL_ELISION = "\u2019"
GREEK_VOWELS = frozenset("αεηιουωΑΕΗΙΟΥΩ")
BETA_VOWELS = frozenset("aehiouw")


def beta_trailing_paren_is_elision(beta_code: str) -> bool:
    """Distinguish Diorisis final elision ``)`` from a breathing.

    In Beta Code a breathing on a one-vowel word or initial diphthong may be
    written at the very end (``ou)``). On a longer word, the same final marker
    is Diorisis's elision apostrophe (``di)``, ``par)``, ``a)ll)``).
    """
    if not beta_code.endswith(")"):
        return False
    letters = [
        char.lower()
        for char in beta_code[:-1]
        if char.isascii() and char.isalpha()
    ]
    return bool(letters) and not (
        len(letters) <= 2 and all(char in BETA_VOWELS for char in letters)
    )


def corpus_freq_key(text: str) -> str:
    """Return the lowercase, accentless corpus key without losing elision.

    NFC is applied before combining marks are removed. That composes a real
    breathing with its vowel, while a trailing combining psili on a consonant
    remains visible and can be preserved as an elision mark.
    """
    text = unicodedata.normalize("NFC", text).lower()
    if not text:
        return ""

    trailing_elision = text.endswith("\u0313") or text.endswith("\u0314")
    if trailing_elision:
        text = text[:-1] + CANONICAL_ELISION

    decomposed = unicodedata.normalize("NFD", text)
    normalized = []
    for index, char in enumerate(decomposed):
        if char == "\u1fbf":
            # A leading spacing psili before a vowel is a breathing, not an
            # apostrophe. Before a consonant it represents aphaeresis.
            next_char = (
                decomposed[index + 1] if index + 1 < len(decomposed) else ""
            )
            if index == 0 and next_char in GREEK_VOWELS:
                continue
            normalized.append(CANONICAL_ELISION)
        elif char in TEXTUAL_ELISION_MARKS:
            normalized.append(CANONICAL_ELISION)
        elif unicodedata.category(char) != "Mn":
            normalized.append(char)
    return unicodedata.normalize("NFC", "".join(normalized))
