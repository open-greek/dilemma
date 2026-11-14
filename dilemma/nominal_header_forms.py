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


# The case tags of a declension-table cell, and the tags a Modern Greek
# headword line gives a form without a case (its gender and number).
_CASE_TAGS = frozenset({"nominative", "genitive", "accusative", "vocative",
                        "dative"})
_MODERN_HEADER_TAGS = frozenset({"masculine", "feminine", "neuter", "common",
                                 "singular", "plural"})


def is_modern_nominal_fragment(entry, form):
    """True for a Modern Greek noun or adjective table cell that is no word.

    Some Wiktionary tables lose a stem and list bare endings or the
    participle endings of another stem: αρχαΐζων's table gives ούσης, όντων
    and ουσών, αισώπειος's gives ος, ε and ους, and a headword line can end
    in a stray letter (βρόντημα 'γ'). Others are copied from another word's
    table (κρεατομηχανή lists μηχανή, μηχανής). A cell is such a fragment
    when its letters are shorter than the headword's and are either a bare
    ending (at most three letters, two fewer than the headword) or, for a
    case cell of a common noun or adjective, start with another letter.

    Longer suppletive cells (κακός, χειρότερος), short forms of short names
    (Ζευς, Δία) and header alternatives with another initial (αργιλές for
    ναργιλές) are kept.
    """
    pos = entry.get("pos")
    if pos not in {"adj", "noun", "name"}:
        return False
    tags = set(form.get("tags") or ())
    cell = bool(_CASE_TAGS & tags)
    header = form.get("source") == "header" and tags <= _MODERN_HEADER_TAGS
    if not (cell or header):
        return False
    word = entry.get("word", "")
    letters, word_letters = _letters(form.get("form", "")), _letters(word)
    if (not letters or not word_letters or letters == word_letters
            or " " in word or len(letters) >= len(word_letters)):
        return False
    if len(letters) <= 3 and len(letters) <= len(word_letters) - 2:
        return True
    return cell and pos != "name" and letters[0] != word_letters[0]
