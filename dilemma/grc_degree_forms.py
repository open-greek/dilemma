"""Regular comparative and superlative cells of Ancient adjectives.

The degree forms in the lookup come from Wiktionary tables and the corpus
pairs (σοφώτερος, εἰδικώτερον), so an adjective with neither has none:
ὁλικωτάτων, ὁλικωτέρα and the comparative that LSJ cites for ὁλικός
(-ώτερον) had no reading. The formation is regular once the adjective is
known:

* an adjective in -ος takes -ότερος, -ότατος after a long syllable and
  -ώτερος, -ώτατος after a short one (σοφός, σοφώτερος; ἰσχυρός,
  ἰσχυρότερος). The last syllable of the stem is long when its vowel is η,
  ω, a diphthong, or marked long, or when two consonants or a double
  consonant follow it; it is short when its vowel is ε or ο, or α, ι, υ
  marked short, before at most one consonant. A stem vowel α, ι or υ that
  no mark settles, and a short vowel before a stop and a liquid, give both
  formations: only an attested cell is kept, so the texts decide. The -ι-
  of -ιος and of the suffix -ικός is short (ἀξιώτερος, ὁλικώτερος);
* an s-stem adjective in -ής or -ης takes -έστερος, -έστατος (ἀληθέστερος);
* where LSJ states the comparison (``Comp. -ώτερον``, ``Sup. -έστατος``),
  its formation is used and the rule is not.

The comparative declines like an adjective in -ος, -α, -ον (σοφωτέρα, the
feminine after ρ), the superlative like one in -ος, -η, -ον (σοφωτάτη);
both read as the positive adjective, as Wiktionary and the treebanks
lemmatize them (σοφώτερος, εἰδικώτερον). The cells are those of
dilemma/grc_os_gaps.py with the masculine nominative: no dual, no vocative
singular.

These are hypotheses: build/build_grc_degree_forms.py keeps only the cells
whose spelling ``form_profile.db`` attests and writes them, with their
evidence, to data/grc_degree_forms.json. build_lookup_db.py fills absent
keys from it through ``grc_noun_gaps.fill_gap_cells``.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import re
import unicodedata

from dilemma.grc_noun_gaps import ACUTE, CIRCUMFLEX, GRAVE, strip_accents
from dilemma.grc_os_gaps import _clusters, os_cells

SOURCE = "grc_degree_forms"
KIND = "degree"
# The formations, by the comparative's ending.
SHORT, LONG, S_STEM = "-ώτερος", "-ότερος", "-έστερος"
FORMATIONS = (LONG, SHORT, S_STEM)
_INFIX = {LONG: ("ότερος", "ότατος"), SHORT: ("ώτερος", "ώτατος"),
          S_STEM: ("έστερος", "έστατος")}
# Length marks a lexicon prints on a vowel: macron and breve.
MACRON, BREVE = "̄", "̆"
_SUBSCRIPT = "ͅ"
_DIAERESIS = "̈"
_VOWELS = frozenset("αεηιουω")
_DIPHTHONG_FIRST = {"ι": frozenset("αεου"), "υ": frozenset("αεοηω")}
_DOUBLE = frozenset("ζξψ")
_STOPS = frozenset("πβφτδθκγχ")
_LIQUIDS = frozenset("λρμν")
# A statement of comparison in an LSJ entry: the label, then the form or
# its ending within the next two words ("Comp. -ώτερον", "Sup. written
# ἀκινδυνότατα"). "Comp. of ..." names the positive of a comparative head.
_STATEMENT = re.compile(
    r"\b(Comp|Sup)\.\s*((?:[^\s,;:.()]+\s+){0,2}?)"
    r"(-?[Ͱ-Ͽἀ-῿]+)")


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _without(word: str, marks: Iterable[str]) -> str:
    marks = frozenset(marks)
    return _nfc("".join(ch for ch in unicodedata.normalize("NFD", word)
                        if ch not in marks))


def plain(lemma: str) -> str:
    """The lemma without length marks, as the lookup spells it."""
    return _without(lemma, (MACRON, BREVE))


def is_es_adjective(lemma: str) -> bool:
    return strip_accents(lemma).endswith("ης")


def stem(lemma: str) -> str:
    """The lemma without its ending (-ος, -ής, -ης), its accents and its
    length marks: the base of the degree forms (σοφ-, ὁλικ-, ἀληθ-)."""
    return _without(plain(lemma)[:-2], (ACUTE, GRAVE, CIRCUMFLEX))


def rule_formations(lemma: str) -> tuple[str, ...]:
    """The formations the stem's last syllable allows, by the rule above.

    ``lemma`` may carry a lexicon's length marks (ἀγᾰθός, ἀνιᾱρός).
    """
    lemma = _nfc(lemma)
    if is_es_adjective(lemma):
        return (S_STEM,)
    if not strip_accents(lemma).endswith("ος"):
        return ()
    clusters = _clusters(lemma)[:-2]
    vowels = [i for i, cluster in enumerate(clusters)
              if cluster[0].lower() in _VOWELS]
    if not vowels:
        return ()
    last = vowels[-1]
    consonants = [cluster[0].lower() for cluster in clusters[last + 1:]]
    marks = set(clusters[last][1:])
    base = clusters[last][0].lower()
    previous = clusters[last - 1] if last else None
    diphthong = (previous is not None and last - 1 in vowels
                 and base in _DIPHTHONG_FIRST
                 and previous[0].lower() in _DIPHTHONG_FIRST[base]
                 and _DIAERESIS not in marks
                 and not ({ACUTE, GRAVE, CIRCUMFLEX, "̓", "̔"}
                          & set(previous[1:])))
    long_vowel = (base in "ηω" or diphthong or _SUBSCRIPT in marks
                  or CIRCUMFLEX in marks or MACRON in marks)
    short_vowel = not long_vowel and (base in "εο" or BREVE in marks)
    weight = sum(2 if letter in _DOUBLE else 1 for letter in consonants)
    if weight >= 2:
        muta_cum_liquida = (len(consonants) == 2 and consonants[0] in _STOPS
                            and consonants[1] in _LIQUIDS)
        if muta_cum_liquida and not long_vowel:
            return (LONG, SHORT)
        return (LONG,)
    if long_vowel:
        return (LONG,)
    if short_vowel:
        return (SHORT,)
    # α, ι or υ with no mark: -ιος and the suffix -ικός have a short ι.
    bare = strip_accents(plain(lemma)).lower()
    if base == "ι" and (not consonants or bare.endswith("ικος")):
        return (SHORT,)
    return (LONG, SHORT)


def statement_formations(lemma: str, texts: Iterable[str]) -> tuple[str, ...]:
    """The formations that LSJ's statements of comparison give the lemma.

    ``texts`` are the lemma's entry texts. A statement counts when it gives
    an ending (-ώτερον) or a form of the lemma's own stem (ἀγαθώτερος under
    ἀγαθός); another stem (ἀμείνων) and "Comp. of" are no statement.
    """
    own = strip_accents(stem(lemma)).lower()
    found = []
    for text in texts:
        for match in _STATEMENT.finditer(text):
            if match.group(2).strip().startswith("of"):
                continue
            word = match.group(3)
            bare = strip_accents(word).lower()
            if not word.startswith("-") and not bare.startswith(own):
                continue
            for formation, markers in ((SHORT, ("ωτερ", "ωτατ")),
                                       (LONG, ("οτερ", "οτατ")),
                                       (S_STEM, ("εστερ", "εστατ"))):
                if any(marker in bare for marker in markers):
                    found.append(formation)
                    break
    return tuple(formation for formation in FORMATIONS if formation in found)


def degree_cells(lemma: str, formation: str) -> list[tuple[str, str]]:
    """(form, cell) of the comparative and the superlative of one formation.

    The cells of an adjective in -ος, -α, -ον (comparative) and in -ος, -η,
    -ον (superlative) with their masculine nominatives, named "comparative
    ..." and "superlative ...".
    """
    base = stem(lemma)
    if not base or formation not in _INFIX:
        return []
    comparative, superlative = (_nfc(base + infix) for infix in _INFIX[formation])
    cells = []
    for degree, word, declension in (
            ("comparative", comparative, "adjective_alpha"),
            ("superlative", superlative, "adjective_eta")):
        cells.append((word, f"{degree} masculine nominative singular"))
        cells += [(form, f"{degree} {cell}")
                  for form, cell in os_cells(word, declension)]
    return cells


def _nominative(word_cells: list[tuple[str, str]], degree: str) -> str:
    return next(form for form, cell in word_cells
                if cell == f"{degree} masculine nominative singular")


def degree_paradigms(
        lookup: Mapping[str, str],
        record: Mapping,
        same_lemma: Callable[[str, str], bool] | None = None,
        dropped: list | None = None) -> Iterable[tuple[str, str, list]]:
    """(``KIND``, lemma, cells) of the kept cells of data/grc_degree_forms.json.

    A lemma is used when its key reads the lemma or the name of the same
    spelling (``grc_os_gaps.reads_itself``). A comparative or superlative
    is another word's when its nominative or one of its cells already reads
    as another lemma (κατώτερος, of κάτω, beside the Byzantine κάτος): all
    its cells are left out then, and ``dropped`` receives (lemma, the degree
    word, the key, its reading). A name of the lemma's spelling
    (δικαιότερος under Δίκαιος) is no other lemma.
    """
    from dilemma.grc_os_gaps import NOT_DECLINED, is_name_twin, reads_itself
    same = same_lemma or (lambda reading, lemma: reading == lemma)
    for lemma, entry in sorted(record["lemmas"].items()):
        if lemma in NOT_DECLINED or not reads_itself(lookup, lemma):
            continue
        kept = {row[0] for row in entry["cells"]}
        cells = []
        for formation in entry["formations"]:
            generated = degree_cells(lemma, formation)
            for degree in ("comparative", "superlative"):
                group = [(form, cell) for form, cell in generated
                         if cell.startswith(degree + " ")]
                word = _nominative(group, degree)
                rival = next(
                    ((form, lookup[form]) for form, _ in group
                     if form in lookup and not same(lookup[form], lemma)
                     and not is_name_twin(lookup[form], lemma)), None)
                if rival:
                    if dropped is not None:
                        dropped.append((lemma, word, *rival))
                    continue
                cells += [(form, cell) for form, cell in group if form in kept]
        if cells:
            yield KIND, lemma, cells
