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
by ``grc_noun_gaps.fill_gap_cells``, under the accented form only (its
``strict`` kinds).

An -ής or -ης adjective that a lexicon classes as one needs no -έος row:
data/grc_es_adjectives.json (build/find_grc_es_adjectives.py) lists the
lemmas whose lexicon entry gives the neuter -ες (LSJ, Sophocles) or that the
Byzantine list gives without an article. A barytone (συνήθης, εὐώδης) keeps
its accent where the ultima allows (συνήθους, συνήθεσι); its neuter singular
is left out, since its accent recedes (σύνηθες) except in some compounds
(εὐῶδες), and its genitive plural is given both ways (συνήθων, συνηθῶν).
"""
from __future__ import annotations

from collections.abc import Iterable
import unicodedata

from dilemma.grc_noun_gaps import (
    ACUTE, ANCIENT_TABLE_SOURCES, MODERN_SOURCES, _needs_breathing_and_lacks_it,
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
# (ending, cell) of a barytone s-stem adjective (συνήθης): the stem keeps
# its acute. The neuter singular is left out (σύνηθες, but εὐῶδες), and the
# genitive plural is given with the stem's accent too (συνήθων beside
# συνηθῶν, ``_ES_BARYTONE_UNACCENTED``).
_ES_BARYTONE_ENDINGS = (
    ("ους", "genitive singular"),
    ("ει", "dative singular"),
    ("η", "masculine/feminine accusative singular; "
          "neuter nominative/accusative/vocative plural"),
    ("εις", "masculine/feminine nominative/accusative/vocative plural"),
    ("ων", "genitive plural"),
    ("εσι", "dative plural"),
    ("εσιν", "dative plural"),
)
_ES_BARYTONE_UNACCENTED = (("ῶν", "genitive plural"),)
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


def _penult_acute(lemma: str) -> bool:
    """The lemma's one accent is an acute on the syllable before -ης."""
    nfd = unicodedata.normalize("NFD", lemma[:-2])
    marks = [ch for ch in nfd if ch in ("\u0301", "\u0300", "\u0342")]
    if marks != [ACUTE]:
        return False
    tail = nfd[nfd.index(ACUTE) + 1:]
    # No vowel between the accent and the ending (the vowel's second
    # letter of a diphthong carries the accent, so a later vowel is a
    # syllable of its own).
    return not any(ch.lower() in _VOWELS for ch in tail)


def es_cells(lemma: str) -> list[tuple[str, str]]:
    """The Attic contracted cells of an s-stem adjective in -ής or -ης.

    An oxytone takes the accent on the ending where it contracts
    (ἀληθοῦς, ἀληθεῖ). A barytone accented on the syllable before -ης
    keeps it (συνήθους, συνήθεσι); its neuter singular is not given, and
    its genitive plural is given both ways (``_ES_BARYTONE_ENDINGS``).
    After a vowel the accusative singular and the neuter plural end in -ᾶ
    as often as in -ῆ (ὑγιᾶ and ὑγιῆ, ἐνδεᾶ), so that cell is left out
    there. Empty for another word.
    """
    lemma = unicodedata.normalize("NFC", lemma)
    if len(lemma) < 3:
        return []
    stem = lemma[:-2]
    after_vowel = strip_accents(stem)[-1:].lower() in _VOWELS
    if lemma.endswith("ής"):
        return [(unicodedata.normalize("NFC", stem + ending), cell)
                for ending, cell in _ES_ENDINGS
                if not (after_vowel and ending == "ῆ")]
    if not lemma.endswith("ης") or not _penult_acute(lemma):
        return []
    bare_stem = unicodedata.normalize("NFC", "".join(
        ch for ch in unicodedata.normalize("NFD", stem) if ch != ACUTE))
    cells = [(unicodedata.normalize("NFC", stem + ending), cell)
             for ending, cell in _ES_BARYTONE_ENDINGS
             if not (after_vowel and ending == "η")]
    cells += [(unicodedata.normalize("NFC", bare_stem + ending), cell)
              for ending, cell in _ES_BARYTONE_UNACCENTED]
    return cells


def select_es_adjectives(items: Iterable[tuple],
                         listed: Iterable[str] = ()) -> list[str]:
    """The s-stem adjectives in -ής of a form -> lemma table, sorted.

    ``items`` yields (form, lemma) or (form, lemma, source) tuples, read
    once; a row without a source counts as an Ancient dictionary row. A
    lemma is selected when it is a plain lowercase oxytone in -ής with the
    breathing its first letter needs, and one of its own forms, from a
    source outside ``MODERN_SOURCES``, is its accented genitive in -έος.
    A lemma of ``listed`` (data/grc_es_adjectives.json: a lexicon classes
    it) needs no such row, may be a barytone in -ης, and is selected when
    its key reads itself or the name of the same spelling. No lemma is
    selected when an ``ANCIENT_TABLE_SOURCES`` table gives two
    first-declension cells under it (-οῦ, -ῇ, -ήν, -αί, -αῖς, -άς).
    """
    from dilemma.grc_os_gaps import is_name_twin
    listed = frozenset(unicodedata.normalize("NFC", lemma) for lemma in listed)
    genitive: set[str] = set()
    heads: set[str] = set()
    first_declension: dict[str, set[str]] = {}
    for item in items:
        form, lemma = item[0], item[1]
        source = item[2] if len(item) > 2 else None
        if form in listed and (form == lemma or is_name_twin(lemma, form)):
            heads.add(form)
        if form == lemma or not lemma.endswith(("ής", "ης")):
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
    genitive = {lemma for lemma in genitive if lemma.endswith("ής")}
    return sorted(
        lemma for lemma in genitive | (heads & listed)
        if is_plain_lemma(lemma) and not _needs_breathing_and_lacks_it(lemma)
        and len(first_declension.get(lemma, ())) < 2 and es_cells(lemma))


def es_paradigms(lemmas: Iterable[str]) -> Iterable[tuple[str, str, list]]:
    """(kind, lemma, cells) for ``grc_noun_gaps.fill_gap_cells``."""
    for lemma in lemmas:
        yield ES_KIND, lemma, es_cells(lemma)


def read_es_adjectives(record) -> list[str]:
    """The lemmas of data/grc_es_adjectives.json."""
    return sorted({lemma for lemmas in record["lemmas"].values()
                   for lemma in lemmas})
