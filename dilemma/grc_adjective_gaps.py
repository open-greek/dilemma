"""Missing feminine and neuter-plural cells of Ancient -ικός adjectives.

The pinned Sophocles expansion declined many three-ending adjectives as
masculine nouns, because its parser missed their endings (μοναχικός, ή, όν).
Their masculine cells are the adjective's, but the feminine and the neuter
plural are absent, so μοναχικήν has no reading.

An oxytone -ικός adjective declines regularly: -ικός, -ική, -ικόν, with η
throughout the feminine singular after κ. ``data/grc_ikos_adjectives.json``
lists the lemmas with independent evidence that they are such adjectives
(build/find_grc_ikos_adjectives.py); this module gives their missing cells.
Nothing here overwrites a reading: a cell fills an absent key only.
"""
from __future__ import annotations

import unicodedata

SUFFIX = "ικός"

# (ending, cell) for the cells a masculine-noun paradigm lacks.
_ENDINGS = (
    ("ική", "feminine nominative/vocative singular"),
    ("ικῆς", "feminine genitive singular"),
    ("ικῇ", "feminine dative singular"),
    ("ικήν", "feminine accusative singular"),
    ("ικαί", "feminine nominative/vocative plural"),
    ("ικῶν", "feminine genitive plural"),
    ("ικαῖς", "feminine dative plural"),
    ("ικάς", "feminine accusative plural"),
    ("ικά", "feminine nominative/accusative/vocative dual; neuter nominative/accusative/vocative plural"),
    ("ικαῖν", "feminine genitive/dative dual"),
)


def _strip_accents(word: str) -> str:
    return unicodedata.normalize("NFC", "".join(
        ch for ch in unicodedata.normalize("NFD", word)
        if not unicodedata.combining(ch)))


def ikos_cells(lemma: str) -> list[tuple[str, str]]:
    """The feminine and neuter-plural cells of an oxytone -ικός adjective."""
    lemma = unicodedata.normalize("NFC", lemma)
    if not lemma.endswith(SUFFIX):
        raise ValueError(f"not an oxytone -ικός adjective: {lemma!r}")
    stem = lemma[:-len(SUFFIX)]
    return [(unicodedata.normalize("NFC", stem + ending), cell)
            for ending, cell in _ENDINGS]


def complete_ikos_adjectives(lookup: dict, lemmas) -> list[dict]:
    """Fill absent keys of the listed adjectives; return one row per added key.

    Each cell is added under its accented form and its accentless key, the
    way the paradigm expansions key their forms. A capitalized lemma
    (Πλατωνικός) is also keyed in lowercase, and its accentless key is
    lowercase, as the capitalized heads of the lookup are (Ἀττικήν, ἀττικήν,
    αττικην). A lemma that is not a self-mapped head in ``lookup`` is skipped.
    """
    added = []
    for lemma in lemmas:
        if lookup.get(lemma) != lemma:
            continue
        for form, cell in ikos_cells(lemma):
            for key in dict.fromkeys((form, form.lower(), _strip_accents(form).lower())):
                if key not in lookup:
                    lookup[key] = lemma
                    added.append({"form": key, "lemma": lemma, "cell": cell})
    return added
