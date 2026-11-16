"""Missing regular cells of Ancient -σις nouns and s-stem neuters.

Two kinds of noun reach the lookup with only part of their declension:

* Nouns in -σις (σύναρσις, ῥήτρευσις). Most Byzantine and later headwords
  come from lists that give the headword alone, and the Byzantine paradigm
  expansion (build/expand_lbg.py) skips every -ις noun because it cannot
  infer an -ις genitive. So συνάρσεως, συνάρσει and σύναρσιν have no reading.
* s-stem neuters in -ος (στέφος, τέκος, γλάγος). Their tables come from the
  dictionary expansion or from Homeric Wiktionary tables, which give the
  Ionic uncontracted cells only (στέφεος, στέφεα, στεφέων). The Attic
  contracted cells (στέφους, στέφη, στεφῶν) are absent.

Both declensions are regular, with an acute on the stem's last vowel
(συνάρσεως, στέφους, μεγέθη) except where the ending carries the accent
(συναρσέοιν, στεφῶν, στεφοῖν). This module gives those cells. Nothing here
overwrites a reading: a cell fills an absent key only, only for a plain lemma
(lowercase, one word, one accent), and not at all when its lowercase,
grave-to-acute or monotonic spelling already reads as another lemma, since
the runtime reaches a form's reading through those spellings too (λαθῶν
keeps λανθάνω through λαθών). A -σις noun must be a self-mapped head of the
table, with the breathing its first letter needs.
An s-stem neuter needs its own evidence in the table from an Ancient source:
an accented -εος row and a neuter plural (-εα, -εσι(ν) or -εων) row under the
lemma. A Modern Greek row is no evidence (υπέρβαρος), a lemma that begins
with a vowel or rho needs its breathing, a lemma an Ancient dictionary table
also declines in the second declension (two of -ου -ῳ -ον -οι -οις) is a
masculine or an adjective (λόχος, πρᾶος, λάϊνος), and the reviewed
``NOT_SSTEM_NEUTERS`` (φῖλος, whose -εος rows Wiktionary files beside the
adjective φίλος) are not declined either.

The vocative singular of a -σις noun (σύναρσι) is not generated: it is the
only cell that collides with corpus names (Κίμωσι, of Κίμων).

``fill_gap_cells`` is the one place where a generated cell becomes a key.
The second-declension cells of dilemma/grc_os_gaps.py and the -ής cells of
dilemma/grc_adjective_gaps.py go through it too, after the cells above.
"""
from __future__ import annotations

from collections.abc import Callable, Container, Iterable, Mapping, MutableMapping
import unicodedata

ACUTE, GRAVE, CIRCUMFLEX = "́", "̀", "͂"
_ACCENTS = frozenset((ACUTE, GRAVE, CIRCUMFLEX))
_VOWELS = frozenset("αεηιουω")

SOURCE = "grc_noun_gaps"

# (ending after the -ις-less stem, cell). A cell of None takes the stem with
# its accent removed (the ending carries the accent).
_SIS_ENDINGS = (
    ("εως", "genitive singular"),
    ("ει", "dative singular; nominative/accusative/vocative dual"),
    ("εις", "nominative/accusative/vocative plural"),
    ("εων", "genitive plural"),
    ("εσι", "dative plural"),
    ("εσιν", "dative plural"),
)
_SIS_DUAL = ("έοιν", "genitive/dative dual")
_SIS_ACCUSATIVE = ("ιν", "accusative singular")

_SSTEM_ENDINGS = (
    ("ους", "genitive singular"),
    ("ει", "dative singular; nominative/accusative/vocative dual"),
    ("η", "nominative/accusative/vocative plural"),
    ("εσι", "dative plural"),
    ("εσιν", "dative plural"),
)
_SSTEM_UNACCENTED = (
    ("ῶν", "genitive plural"),
    ("οῖν", "genitive/dative dual"),
)
# Neuter plural endings whose presence under a lemma shows that it declines
# as an s-stem neuter (the -εος row alone does not).
_SSTEM_PLURAL_EVIDENCE = ("εα", "εσι", "εσιν", "εων")
# Second-declension endings (accentless, iota subscript dropped). An Ancient
# dictionary table that gives two of them under an -ος lemma declines it as a
# masculine or an adjective, whatever -εος rows sit beside them.
_OSTEM_ENDINGS = frozenset(("ου", "ω", "ωι", "ον", "οι", "οις"))
# Lookup sources (registry codes, data/lookup_sources.json) of Modern and
# Medieval Greek readings: no evidence for an Ancient declension.
MODERN_SOURCES = frozenset((
    "wiktionary_el", "wiktionary_med", "el_rule_paradigms", "el_paradigm_gaps",
    "hnc"))
# Lookup sources whose rows are dictionary inflection tables of Ancient Greek.
# Corpus pairs are left out: a lemmatizer's slip (γλῆνον under γλῆνος) is no
# declension.
ANCIENT_TABLE_SOURCES = frozenset((
    "wiktionary_grc", "grc_merged", "grc_lexicon_expansion"))
_BREATHINGS = frozenset(("\u0313", "\u0314"))
# Reviewed -ος lemmas with s-stem evidence in the table that are not s-stem
# neuters. φῖλος is Homer's masculine "friend" (φίλος); its -εος/-εα rows in
# the Wiktionary tables are spellings of other words. ἀδράνεος is LSJ's
# adjective (η, ον, = ἀδρανής); the expansion layer declines it as a neuter
# (ἀδρανέεος, ἀδρανέεα). Its second-declension cells come from
# dilemma/grc_os_gaps.py, and the build takes every word that module
# declines out of the s-stem selection; the entry here covers a build
# without the second-declension list.
NOT_SSTEM_NEUTERS = frozenset(("φῖλος", "ἀδράνεος"))


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def strip_accents(word: str) -> str:
    """Remove every diacritic (accents, breathings, subscripts, diaeresis)."""
    return _nfc("".join(ch for ch in unicodedata.normalize("NFD", word)
                        if not unicodedata.combining(ch)))


def _unaccent(stem: str) -> str:
    """Remove the acute, grave and circumflex, keeping every other mark."""
    return _nfc("".join(ch for ch in unicodedata.normalize("NFD", stem)
                        if ch not in _ACCENTS))


def _has_accent(word: str) -> bool:
    return any(ch in _ACCENTS for ch in unicodedata.normalize("NFD", word))


def acute_last_vowel(stem: str) -> str | None:
    """The stem with its accents replaced by an acute on its last vowel.

    A diphthong takes the acute on its second letter (στεφ-, τειχ- give
    στέφ-, τείχ-). None when the stem has no vowel.
    """
    letters = list(unicodedata.normalize("NFD", _unaccent(stem)))
    for i in range(len(letters) - 1, -1, -1):
        if letters[i].lower() in _VOWELS:
            j = i + 1
            while j < len(letters) and unicodedata.combining(letters[j]):
                j += 1
            letters.insert(j, ACUTE)
            return _nfc("".join(letters))
    return None


def _accent_count(word: str) -> int:
    return sum(ch in _ACCENTS for ch in unicodedata.normalize("NFD", word))


def is_plain_lemma(lemma: str) -> bool:
    """A lowercase, single-word headword with one accent (no name, no fragment).

    A noun headword carries exactly one accent, so a doubly accented string
    (διερμήνεύσις, κάτοχύρωσις) is a dictionary artefact and is not declined.
    """
    return (bool(lemma) and lemma[0].islower() and lemma.isalpha()
            and _accent_count(lemma) == 1)


def sis_cells(lemma: str) -> list[tuple[str, str]]:
    """The regular Attic cells of a -σις noun (πόλις type), without the vocative."""
    lemma = _nfc(lemma)
    if not lemma.endswith("σις"):
        return []
    stem = lemma[:-2]
    accented = acute_last_vowel(stem)
    if accented is None:
        return []
    cells = [(_nfc(accented + ending), cell) for ending, cell in _SIS_ENDINGS]
    cells.append((_nfc(_unaccent(stem) + _SIS_DUAL[0]), _SIS_DUAL[1]))
    cells.append((_nfc(stem + _SIS_ACCUSATIVE[0]), _SIS_ACCUSATIVE[1]))
    return cells


def sstem_cells(lemma: str) -> list[tuple[str, str]]:
    """The Attic contracted cells of an s-stem neuter in -ος (γένος type)."""
    lemma = _nfc(lemma)
    if not lemma.endswith("ος"):
        return []
    stem = lemma[:-2]
    accented = acute_last_vowel(stem)
    if accented is None:
        return []
    cells = [(_nfc(accented + ending), cell) for ending, cell in _SSTEM_ENDINGS]
    bare = _unaccent(stem)
    cells += [(_nfc(bare + ending), cell) for ending, cell in _SSTEM_UNACCENTED]
    # Keep the paradigm order: genitive plural after the nominative plural.
    order = ("genitive singular", "dative singular; nominative/accusative/vocative dual",
             "nominative/accusative/vocative plural", "genitive plural",
             "dative plural", "genitive/dative dual")
    return sorted(cells, key=lambda item: order.index(item[1]))


def _needs_breathing_and_lacks_it(lemma: str) -> bool:
    """A lemma beginning with a vowel or rho written without its breathing."""
    nfd = unicodedata.normalize("NFD", lemma)
    if not nfd or nfd[0].lower() not in _VOWELS | {"ρ"}:
        return False
    head = []
    for ch in nfd:
        if not unicodedata.combining(ch) and head and \
                (len(head) >= 2 or ch.lower() not in _VOWELS):
            break
        head.append(ch)
    return not any(ch in _BREATHINGS for ch in head)


def select_noun_gap_lemmas(
        items: Iterable[tuple]) -> tuple[list[str], list[str]]:
    """The -σις nouns and s-stem neuters of a form -> lemma table.

    ``items`` yields (form, lemma) or (form, lemma, source) tuples, read
    once; ``source`` is the row's lookup source code, and a row without one
    counts as an Ancient dictionary row. A -σις noun is a plain lemma that
    maps to itself, with the breathing its first letter needs (όδευσις,
    beside ὅδευσις, is not one): for a headword-only noun that row is all
    the table knows. An s-stem neuter is a plain -ος lemma that has, among
    its own forms from a source outside ``MODERN_SOURCES``, an accented -εος form
    and an accented neuter-plural form (-εα, -εσι, -εσιν or -εων) on the same
    stem; its headword key may read as another lemma (γλάγος as γάλα) and
    the cells still go to the lemma its table is listed under. It is not
    selected when it lacks the breathing its first letter needs, when an
    ``ANCIENT_TABLE_SOURCES`` table gives two second-declension cells under
    it, or when it is one of the reviewed ``NOT_SSTEM_NEUTERS``. Both lists
    are sorted, so the result does not depend on the table's order.
    """
    sis: set[str] = set()
    genitive: set[str] = set()
    plural: set[str] = set()
    second_declension: dict[str, set[str]] = {}
    plural_endings = {ending: len(ending) for ending in _SSTEM_PLURAL_EVIDENCE}
    for item in items:
        form, lemma = item[0], item[1]
        source = item[2] if len(item) > 2 else None
        if form == lemma:
            if (lemma.endswith("σις") and is_plain_lemma(lemma)
                    and not _needs_breathing_and_lacks_it(lemma)):
                sis.add(lemma)
            continue
        if not lemma.endswith("ος"):
            continue
        bare_form = strip_accents(form).lower()
        stem = strip_accents(lemma[:-2])
        if not bare_form.startswith(stem):
            continue
        ending = bare_form[len(stem):]
        if ending in _OSTEM_ENDINGS:
            if source is None or source in ANCIENT_TABLE_SOURCES:
                second_declension.setdefault(lemma, set()).add(ending)
            continue
        if not _has_accent(form) or source in MODERN_SOURCES:
            continue
        if ending == "εος":
            genitive.add(lemma)
        elif ending in plural_endings:
            plural.add(lemma)
    sstem = {
        lemma for lemma in genitive & plural
        if is_plain_lemma(lemma)
        and not _needs_breathing_and_lacks_it(lemma)
        and len(second_declension.get(lemma, ())) < 2
        and lemma not in NOT_SSTEM_NEUTERS
    }
    return sorted(sis), sorted(sstem)


def _spellings(form: str) -> tuple[str, ...]:
    """The keys a cell is stored under: accented, lowercase, accentless."""
    return tuple(dict.fromkeys((form, form.lower(), strip_accents(form).lower())))


def reading_spellings(form: str) -> tuple[str, ...]:
    """The spellings whose reading a cell may not hide.

    The form, its lowercase, its grave-to-acute and its monotonic spelling:
    the runtime answers the cell's form from any of them, and they keep its
    accent's place. The accentless key is left out, since it serves
    unaccented input only and an accented cell is told apart by its accent
    (αἰτίωσιν beside the verb's αἰτιῶσιν, both αιτιωσιν without accents).
    """
    from dilemma.core import grave_to_acute, to_monotonic
    return tuple(dict.fromkeys((form, form.lower(), grave_to_acute(form),
                                to_monotonic(form.lower()))))


def fill_gap_cells(
        lookup: MutableMapping[str, str],
        paradigms: Iterable[tuple[str, str, Iterable[tuple[str, str]]]],
        *,
        taken: Mapping[str, object] | None = None,
        variants: Callable[[str], Iterable[str]] | None = None,
        strict: Container[str] = ()) -> list[dict]:
    """Fill absent keys with generated cells; return one row per added key.

    ``paradigms`` yields (kind, lemma, cells), ``cells`` being (form, cell
    name) pairs. Each cell is added under its accented form, its lowercase
    and its accentless lowercase key, the way the paradigm expansions key
    their forms. A key already in ``lookup`` (or in ``taken``, for keys a
    table left out on purpose, such as article forms) keeps its reading. A
    cell one of whose spellings (``variants``, by default
    ``reading_spellings``) already reads as another lemma is not added under
    any key: the new accented key would hide the reading the runtime gave
    the form through that spelling (λαθῶν, whose monotonic λαθών reads
    λανθάνω). For the kinds in ``strict`` the accentless key is weighed the
    same way: their lemmas are rare headwords, and a cell that differs only
    in its accent from a form of a known word is, in the texts, that word
    with a misprinted accent (δικῆς for δίκης, not the feminine of δικός).
    When two lemmas of one call give the same absent key, neither takes it:
    the key is ambiguous and the table has no evidence to choose. A later
    call finds the keys of an earlier one in ``lookup`` and leaves them.
    """
    taken = taken or {}
    variants = variants or reading_spellings
    claims: dict[str, list[tuple[str, str, str]]] = {}
    for kind, lemma, cells in paradigms:
        for form, cell in cells:
            spellings = list(variants(form))
            if kind in strict:
                spellings.append(strip_accents(form).lower())
            if any(lookup.get(spelling, lemma) != lemma for spelling in spellings):
                continue
            for key in _spellings(form):
                if key in lookup or key in taken:
                    continue
                rows = claims.setdefault(key, [])
                if all(row[0] != lemma for row in rows):
                    rows.append((lemma, cell, kind))
    added = []
    for key in sorted(claims):
        rows = claims[key]
        if len(rows) != 1:
            continue
        lemma, cell, kind = rows[0]
        lookup[key] = lemma
        added.append({"form": key, "lemma": lemma, "cell": cell, "kind": kind})
    return added


def noun_gap_paradigms(
        lookup: Mapping[str, str],
        sis_lemmas: Iterable[str],
        sstem_lemmas: Iterable[str]) -> Iterable[tuple[str, str, list]]:
    """(kind, lemma, cells) of the -σις nouns and s-stem neuters, for
    ``fill_gap_cells``. A -σις lemma must map to itself in ``lookup`` (see
    select_noun_gap_lemmas)."""
    for lemma in sis_lemmas:
        if lookup.get(lemma) == lemma:
            yield "sis", lemma, sis_cells(lemma)
    for lemma in sstem_lemmas:
        yield "sstem", lemma, sstem_cells(lemma)


def complete_noun_gaps(
        lookup: MutableMapping[str, str],
        sis_lemmas: Iterable[str],
        sstem_lemmas: Iterable[str],
        *,
        taken: Mapping[str, object] | None = None,
        variants: Callable[[str], Iterable[str]] | None = None) -> list[dict]:
    """Fill absent keys with the lemmas' cells; return one row per added key.

    The -σις and s-stem paradigms alone, through ``fill_gap_cells``.
    """
    return fill_gap_cells(
        lookup, list(noun_gap_paradigms(lookup, sis_lemmas, sstem_lemmas)),
        taken=taken, variants=variants)
