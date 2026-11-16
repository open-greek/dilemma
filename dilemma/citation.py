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


# The hyphens and dashes a lexicon, its export, its OCR or a line break
# leaves inside a word; the soft hyphen is invisible.
HEAD_HYPHENS = "-\u00ad\u2014\u2013"


def hyphenated_lemma(lemma: str) -> bool:
    """Whether a lemma has a hyphen (also a soft one) or a dash between two
    of its letters.

    No Ancient citation form is spelled that way. A lexicon divides a
    compound or a derivative at its stem (Βακχ-εία, καθ-ηδύνω), OCR keeps a
    line-break hyphen (Σινα-ῖτις), and a corpus names a preverb apart
    (ἐπί-εἰσκομίζω) or lemmatizes a line-broken token as it stands (αὐ-τός).
    Accents and breathings are ignored (ἐμ\u0301-βλῴσκω); a hyphen at either
    end is not between letters (ὁτ-), and neither is one beside a metrical
    sign. Two marks in a row are still inside the word.
    """
    if not isinstance(lemma, str) or not any(
            mark in lemma for mark in HEAD_HYPHENS):
        return False
    letter_before = after_letter = False
    for char in unicodedata.normalize("NFD", lemma):
        if unicodedata.combining(char):
            continue
        if char in HEAD_HYPHENS:
            after_letter = after_letter or letter_before
        elif after_letter and char.isalpha():
            return True
        else:
            after_letter = False
            letter_before = char.isalpha()
    return False


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
