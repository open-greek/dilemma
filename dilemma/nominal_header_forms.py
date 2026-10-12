"""Distinguish abbreviated nominal header endings from complete forms."""

import unicodedata


# These genitive endings also appear with an explicit leading hyphen in
# Greek Wiktionary headers. Some headers omit that hyphen; the remaining
# suffix is still not an independently inflected word.
_GENITIVE_ENDINGS = frozenset({"ας", "ης", "ος", "ου", "ους", "εος", "εως",
                               "ιος", "ως", "ων", "γος"})
_HEADER_METADATA = frozenset({"masculine", "feminine", "neuter", "common",
                              "Attic", "Epic", "Ionic", "Doric", "Aeolic",
                              "Aeolian", "Koine", "Homeric"})


def _letters(text):
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn").lower()


def is_abbreviated_nominal_header(entry, form):
    """Reject only an unqualified suffix in a nominal header.

    Complete case-tagged forms, alternatives, the lemma itself, and
    independently marked vowel-initial words keep their source readings.
    """
    if (entry.get("pos") not in {"noun", "name"}
            or form.get("source") != "header"
            or set(form.get("tags") or []) - _HEADER_METADATA):
        return False
    text = form.get("form", "")
    letters = _letters(text)
    if letters == _letters(entry.get("word", "")) or letters not in _GENITIVE_ENDINGS:
        return False
    # A real breathing distinguishes, for example, ὅς or οὗ from a
    # genitive suffix with only an accent. Do not discard such evidence.
    return not any(c in unicodedata.normalize("NFD", text) for c in ("\u0313", "\u0314"))
