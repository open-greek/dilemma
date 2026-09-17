"""Structural validation helpers for Greek citation forms."""

from __future__ import annotations

import unicodedata


TONAL_MARKS = frozenset({"\u0301", "\u0300", "\u0342"})
_GREEK_VOWELS = frozenset("αεηιουωΑΕΗΙΟΥΩ")


def _is_greek_letter(char: str) -> bool:
    if not char:
        return False
    codepoint = ord(char)
    return ((0x0370 <= codepoint <= 0x03FF
             or 0x1F00 <= codepoint <= 0x1FFF)
            and char.isalpha())


def malformed_tonal_reason(text: str) -> str | None:
    """Return a reason for structurally impossible tonal-mark placement.

    Multiple accents can be legitimate across a phrase or an
    enclitic-bearing lexical form. The safe failures are narrower: a tonal
    mark must attach to a Greek vowel, and one vowel cannot carry two tonal
    marks at once.
    """
    base = ""
    tonal_count = 0
    for char in unicodedata.normalize("NFD", text or ""):
        if not unicodedata.combining(char):
            base = char
            tonal_count = 0
            continue
        if char not in TONAL_MARKS:
            continue
        # Latin and other scripts have their own accent rules. This helper is
        # deliberately limited to malformed structure inside Greek text.
        if base not in _GREEK_VOWELS and (
                _is_greek_letter(base) or not base.isalpha()):
            return "orphaned_tonal_mark"
        tonal_count += 1
        if tonal_count > 1:
            return "duplicate_tonal_marks"
    return None
