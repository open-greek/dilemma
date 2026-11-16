"""Missing cells of Ancient -ικός adjectives and of s-stem adjectives in -ής.

The pinned Sophocles expansion declined many three-ending adjectives as
masculine nouns, because its parser missed their endings (μοναχικός, ή, όν).
Their masculine cells are the adjective's, but the feminine and the neuter
plural are absent, so μοναχικήν has no reading.

An oxytone -ικός adjective declines regularly: -ικός, -ική, -ικόν, with η
throughout the feminine singular after κ. ``data/grc_ikos_adjectives.json``
lists the lemmas with independent evidence that they are such adjectives
(build/find_grc_ikos_adjectives.py); this module gives their missing cells.
Nothing here overwrites a reading: a cell fills an absent key only.

The s-stem adjectives in -ής (σαφής, δυσπενθής) have the opposite gap. The
dictionary expansion and the corpora give many of them Ionic or epic cells
only (δυσπενθέος, δυσπενθέϊ, δυσπενθέες), so the Attic contracted cells
(δυσπενθοῦς, δυσπενθεῖ, δυσπενθεῖς) have no reading. An oxytone -ής lemma
with an accented -έος row under it is such an adjective: no noun in -ής has
that genitive (ποιητής has -οῦ, ἐσθής -ῆτος). ``es_cells`` gives its
contracted cells and ``select_es_adjectives`` the lemmas; the keys are filled
by ``grc_noun_gaps.fill_gap_cells``.
"""
from __future__ import annotations

from collections.abc import Iterable
import unicodedata

from dilemma.grc_noun_gaps import (
    ANCIENT_TABLE_SOURCES, MODERN_SOURCES, _needs_breathing_and_lacks_it,
    is_plain_lemma, strip_accents)

SUFFIX = "ικός"
SOURCE = "grc_adjective_gaps"
# The kind ``es_paradigms`` gives ``grc_noun_gaps.fill_gap_cells``.
ES_KIND = "es"

# (ending, cell) of an oxytone s-stem adjective, contracted. The dual and
# the vocative are left out, as in the other generated paradigms.
_ES_ENDINGS = (
    ("οῦς", "genitive singular"),
    ("εῖ", "dative singular"),
    ("ῆ", "masculine/feminine accusative singular; "
           "neuter nominative/accusative/vocative plural"),
    ("ές", "neuter nominative/accusative/vocative singular"),
    ("εῖς", "masculine/feminine nominative/accusative/vocative plural"),
    ("ῶν", "genitive plural"),
    ("έσι", "dative plural"),
    ("έσιν", "dative plural"),
)
_VOWELS = frozenset("αεηιουω")
# First-declension endings (accentless, iota subscript dropped). An Ancient
# dictionary table that gives two of them under an -ής lemma declines it as
# a masculine in -ής, -οῦ.
_FIRST_DECLENSION_ENDINGS = frozenset(("ου", "η", "ην", "αι", "αις", "ας"))

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


def es_cells(lemma: str) -> list[tuple[str, str]]:
    """The Attic contracted cells of an oxytone s-stem adjective in -ής.

    After a vowel the accusative singular and the neuter plural end in -ᾶ
    as often as in -ῆ (ὑγιᾶ and ὑγιῆ, ἐνδεᾶ), so that cell is left out
    there. Empty for a word that is not an oxytone in -ής.
    """
    lemma = unicodedata.normalize("NFC", lemma)
    if not lemma.endswith("ής") or len(lemma) < 3:
        return []
    stem = lemma[:-2]
    after_vowel = strip_accents(stem)[-1:].lower() in _VOWELS
    return [(unicodedata.normalize("NFC", stem + ending), cell)
            for ending, cell in _ES_ENDINGS
            if not (after_vowel and ending == "ῆ")]


def select_es_adjectives(items: Iterable[tuple]) -> list[str]:
    """The s-stem adjectives in -ής of a form -> lemma table, sorted.

    ``items`` yields (form, lemma) or (form, lemma, source) tuples, read
    once; a row without a source counts as an Ancient dictionary row. A
    lemma is selected when it is a plain lowercase oxytone in -ής with the
    breathing its first letter needs, and one of its own forms, from a
    source outside ``MODERN_SOURCES``, is its accented genitive in -έος. It
    is not selected when an ``ANCIENT_TABLE_SOURCES`` table gives two
    first-declension cells under it (-οῦ, -ῇ, -ήν, -αί, -αῖς, -άς).
    """
    genitive: set[str] = set()
    first_declension: dict[str, set[str]] = {}
    for item in items:
        form, lemma = item[0], item[1]
        source = item[2] if len(item) > 2 else None
        if form == lemma or not lemma.endswith("ής"):
            continue
        bare_form = strip_accents(form).lower()
        stem = strip_accents(lemma[:-2]).lower()
        if not bare_form.startswith(stem):
            continue
        ending = bare_form[len(stem):]
        if ending in _FIRST_DECLENSION_ENDINGS:
            if source is None or source in ANCIENT_TABLE_SOURCES:
                first_declension.setdefault(lemma, set()).add(ending)
        elif (ending == "εος" and source not in MODERN_SOURCES
                and unicodedata.normalize("NFC", form)
                == unicodedata.normalize("NFC", lemma[:-2] + "έος")):
            genitive.add(lemma)
    return sorted(
        lemma for lemma in genitive
        if is_plain_lemma(lemma) and not _needs_breathing_and_lacks_it(lemma)
        and len(first_declension.get(lemma, ())) < 2)


def es_paradigms(lemmas: Iterable[str]) -> Iterable[tuple[str, str, list]]:
    """(kind, lemma, cells) for ``grc_noun_gaps.fill_gap_cells``."""
    for lemma in lemmas:
        yield ES_KIND, lemma, es_cells(lemma)
