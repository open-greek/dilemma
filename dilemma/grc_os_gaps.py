"""Regular cells of Ancient second-declension nouns and adjectives in -ος.

Many -ος words reach the lookup as a headword and nothing else: the
Byzantine lexicon's adjectives (it gives nouns an article and adjectives
none, and build/expand_lbg.py declines only the nouns), LSJ adjectives whose
dictionary table was never generated or was generated in the wrong
declension (αὐτοσύστατος), and nouns whose expansion failed. So ἐμφλόγου,
κολοβόρινον and αὐτοσύστατοι have no reading.

The second declension is regular once the word's class is known, and the
class is read from a lexicon, never guessed from the spelling
(build/find_grc_second_declension.py writes data/grc_second_declension.json):

* ``noun``: a masculine or feminine noun (ὁ λόγος, ἡ ὁδός), seven cells;
* ``adjective``: an adjective of two endings, or one whose feminine the
  lexica do not settle. It gets the masculine cells and the neuter plural;
  the neuter singular is the masculine accusative;
* ``adjective_eta`` and ``adjective_alpha``: an adjective of three endings,
  with the feminine in -η or in -α (after ε, ι, ρ).

The accent is persistent. An oxytone takes the circumflex in every genitive
and dative (ἀγαθοῦ, ἀγαθῷ, ἀγαθῶν, ἀγαθοῖς, ἀγαθῆς). A proparoxytone moves
its acute to the penult before a long ultima (ἔμφλογος, ἐμφλόγου, but
ἔμφλογον, ἔμφλογοι, ἔμφλογα: final -οι and -αι count as short), and a
properispomenon takes the acute there (σπουδαῖος, σπουδαίου, σπουδαῖον).
The feminine genitive plural of an adjective is the masculine's (δικαίων).
An acute that lands on ι or υ right after a vowel that is a syllable of its
own takes the diaeresis (λάινος, λαΐνου; γήινος, γηΐνου); after a diphthong
it does not (εὔιος, εὐίου).

Two kinds of cell are left out. The dual: of 16,073 genitive-dative duals a
trial generated, the corpora attest 4, and the attested forms in -ω are
datives printed without their subscript or verb forms (ἐξολίσθω, ἐπαύρω).
The vocative singular: the corpora attest 49 of 15,224, and two in five of
those are other words (ἀεικέ and δυσχερέ are cut-off forms in -ές, πρόσειλε
is a verb), the same finding that keeps the -σις vocative out.

Stems in -ο (εὔνοος, ἀθρόος) contract in Attic and are not declined.
``select_os_lemmas`` decides which listed lemmas are declined and
``grc_noun_gaps.fill_gap_cells`` which of their cells fill a key. A cell is
stored under its accented form only: these are rare headwords, and the same
letters with another accent are as often another word or a misprint (οἴκτον
is οἶκτον, of οἶκτος, not a form of οἰκτός).

Accent doublets share cells: εὐπροσόδος, a Byzantine by-form, and εὐπρόσοδος
both have εὐπροσόδου, εὐπροσόδων. When both are declined here neither takes
such a cell, and the same holds when the other word is not declined here at
all, because its headword key reads as a name or as another lemma or because
no lexicon gives its class: ``os_rivals`` finds those words and their cells
claim keys without filling any (``RIVAL_KIND``).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import unicodedata

from dilemma.grc_noun_gaps import (
    ACUTE, ANCIENT_TABLE_SOURCES, CIRCUMFLEX, GRAVE, MODERN_SOURCES,
    _needs_breathing_and_lacks_it, is_plain_lemma, strip_accents)

CLASSES = ("noun", "adjective", "adjective_eta", "adjective_alpha")
# The kinds ``os_paradigms`` gives ``grc_noun_gaps.fill_gap_cells``, which
# weighs the accentless key of their cells too and stores no cell under it
# (its ``strict``).
KINDS = frozenset("os_" + name for name in CLASSES)
# The kind of the cells of ``os_rivals`` words, which claim keys and fill
# none (``claims_only`` of ``fill_gap_cells``).
RIVAL_KIND = "os_rival"
# Lookup sources of the generated rows, by class.
NOUN_SOURCE = "grc_noun_gaps"
ADJECTIVE_SOURCE = "grc_adjective_gaps"
# Lookup sources whose genitive row under a lemma shows that a paradigm of
# the word is in the table: the Ancient dictionary tables and the generated
# Byzantine paradigms.
DECLINING_SOURCES = ANCIENT_TABLE_SOURCES | {"lbg_generated"}
# The shortest stem that is declined, in letters (μίνσ-ος). A shorter word's
# cells are too often another word: of the 26 cells of the 49 shorter listed
# lemmas that the corpora attest, the commonest is λέγους (17 tokens: an
# elided λέγουσ᾽ or a misprint, not a form of λέγος), and ὕπου and ζῆναι
# are likewise other words that would read as ὕπος and ζῆνος.
MIN_STEM_LETTERS = 4
# Reviewed listed lemmas that are not declined, with the reason. Each is a
# Byzantine headword whose cells are forms of a pronominal or suppletive
# word that the tables decline under its own lemma.
NOT_DECLINED = {
    "μεγάλος": "the oblique stem of μέγας as a headword: the table reads "
               "μεγάλου and μεγάλοι as μέγας, and μεγάλον is its late "
               "accusative",
    "τοσαῦτος": "a by-form read off τοσαύτη, τοσαῦτα: in the texts "
                "τοσαύτων is the late feminine genitive plural of τοσοῦτος",
    "μισοτοιοῦτος": "a compound of τοιοῦτος, whose stem changes in the "
                    "declension (τοιαῦτα, τοιαύτη): the regular -οῦτα "
                    "does not exist",
}
# Reviewed cells that are not generated, with the reason: the corpora attest
# the form, and there it is a form of another word.
WITHHELD_CELLS = {
    "σιδηροπόδων": "genitive plural of σιδηρόπους; σιδηρόποδος is its "
                   "Byzantine by-form",
    "κακόγνωμον": "neuter of κακογνώμων; κακόγνωμος is its by-form",
    "καλόγνωμον": "neuter of καλογνώμων; καλόγνωμος is its by-form",
    "ὑψίνους": "the contracted ὑψίνοος, not an accusative plural of ὑψίνος",
    "φιλητέων": "Ionic genitive plural of φιλητής (φηλητής, a thief), not "
                "of the verbal adjective φιλητέος",
}

_VOWELS = frozenset("αεηιουω")
_BREATHINGS = frozenset(("̓", "̔"))
_DIAERESIS = "̈"
_ACCENTS = frozenset((ACUTE, GRAVE, CIRCUMFLEX))
# First letters that make a diphthong with a following ι or υ.
_DIPHTHONG_FIRST = {"ι": frozenset("αεου"), "υ": frozenset("αεοηω")}
# Vowels after which an accented ι or υ is written with the diaeresis
# (γηΐνου, πρωΐου as well as λαΐνου).
_HIATUS_FIRST = {"ι": frozenset("αεουηω"), "υ": frozenset("αεοηω")}

# (ending, oxytone ending, cell). Endings with a long ultima.
_LONG = (
    ("ου", "οῦ", "genitive singular"),
    ("ῳ", "ῷ", "dative singular"),
    ("ων", "ῶν", "genitive plural"),
    ("οις", "οῖς", "dative plural"),
    ("ους", "ούς", "accusative plural"),
)
# Endings with a short ultima (final -οι and -αι are short for the accent).
_SHORT = (
    ("ον", "όν", "accusative singular"),
    ("οι", "οί", "nominative/vocative plural"),
)
_NEUTER_SHORT = (
    ("α", "ά", "neuter nominative/accusative/vocative plural"),
)
_FEMININE_ETA_LONG = (
    ("η", "ή", "feminine nominative/vocative singular"),
    ("ης", "ῆς", "feminine genitive singular"),
    ("ῃ", "ῇ", "feminine dative singular"),
    ("ην", "ήν", "feminine accusative singular"),
)
_FEMININE_ALPHA_LONG = (
    ("α", "ά", "feminine nominative/vocative singular"),
    ("ας", "ᾶς", "feminine genitive singular"),
    ("ᾳ", "ᾷ", "feminine dative singular"),
    ("αν", "άν", "feminine accusative singular"),
)
_FEMININE_PLURAL_LONG = (
    ("αις", "αῖς", "feminine dative plural"),
    ("ας", "άς", "feminine accusative plural"),
)
_FEMININE_SHORT = (
    ("αι", "αί", "feminine nominative/vocative plural"),
)
_NEUTER_SINGULAR = "neuter nominative/accusative/vocative singular"
# Paradigm order of the cell names.
_ORDER = tuple(cell for group in (
    _LONG[:2], _SHORT, _LONG[2:], _NEUTER_SHORT,
    _FEMININE_ETA_LONG, _FEMININE_SHORT, _FEMININE_PLURAL_LONG)
    for _, _, cell in group) + (_NEUTER_SINGULAR,)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _clusters(word: str) -> list[list[str]]:
    """The word's letters, each with its combining marks: [base, mark, ...]."""
    clusters: list[list[str]] = []
    for ch in unicodedata.normalize("NFD", word):
        if unicodedata.combining(ch) and clusters:
            clusters[-1].append(ch)
        else:
            clusters.append([ch])
    return clusters


def _nuclei(clusters: list[list[str]]) -> list[list[int]]:
    """Cluster indexes of each syllable's vowel or diphthong, in order.

    ι or υ joins the vowel before it into a diphthong unless it carries a
    diaeresis or that vowel carries an accent or a breathing, which a
    diphthong writes on its second letter (λάινος and ἄυπνος have hiatus).
    """
    nuclei: list[list[int]] = []
    for i, cluster in enumerate(clusters):
        base = cluster[0].lower()
        if base not in _VOWELS:
            continue
        previous = clusters[i - 1] if i else None
        if (previous is not None and nuclei and nuclei[-1] == [i - 1]
                and base in _DIPHTHONG_FIRST
                and previous[0].lower() in _DIPHTHONG_FIRST[base]
                and _DIAERESIS not in cluster
                and not (_ACCENTS | _BREATHINGS) & set(previous[1:])):
            nuclei[-1].append(i)
        else:
            nuclei.append([i])
    return nuclei


def has_only_initial_breathing(lemma: str) -> bool:
    """No breathing except on the word's first vowel, diphthong or rho.

    A breathing inside a word is a lexicon's or an OCR's artefact
    (ἀνοἰκοδομητέος), and its cells would carry it.
    """
    clusters = _clusters(_nfc(lemma))
    nuclei = _nuclei(clusters)
    first = set(nuclei[0]) if nuclei else set()
    if clusters and clusters[0][0].lower() == "ρ":
        first.add(0)
    return all(i in first for i, cluster in enumerate(clusters)
               if _BREATHINGS & set(cluster[1:]))


def accent_type(lemma: str) -> str | None:
    """oxytone, paroxytone, properispomenon or proparoxytone; None otherwise.

    None for a word without exactly one acute or circumflex, with a
    circumflex anywhere but the penult, or accented before the antepenult.
    """
    clusters = _clusters(_nfc(lemma))
    nuclei = _nuclei(clusters)
    marked = [(n, mark) for n, nucleus in enumerate(nuclei) for i in nucleus
              for mark in clusters[i][1:] if mark in _ACCENTS]
    if len(marked) != 1 or len(nuclei) < 2:
        return None
    position, mark = marked[0]
    from_end = len(nuclei) - 1 - position
    return {(0, ACUTE): "oxytone", (1, ACUTE): "paroxytone",
            (1, CIRCUMFLEX): "properispomenon",
            (2, ACUTE): "proparoxytone"}.get((from_end, mark))


def _penult_acute_stem(lemma: str) -> str | None:
    """The stem (lemma without -ος) with its accent as an acute on its last
    syllable: the stem of the cells whose ending has a long ultima."""
    clusters = _clusters(lemma)[:-2]
    nuclei = _nuclei(clusters)
    if not nuclei:
        return None
    for cluster in clusters:
        cluster[1:] = [mark for mark in cluster[1:] if mark not in _ACCENTS]
    last = nuclei[-1]
    target = clusters[last[-1]]
    base = target[0].lower()
    if (len(last) == 1 and base in _HIATUS_FIRST
            and len(nuclei) > 1 and nuclei[-2] == [last[0] - 1]
            and clusters[last[0] - 1][0].lower() in _HIATUS_FIRST[base]
            and _DIAERESIS not in target):
        # The acute leaves the vowel that showed the hiatus. The vowel
        # before must be a syllable of its own: after a diphthong (εὔιος,
        # εὐίου) nothing can join the ι, and no diaeresis is written.
        target.append(_DIAERESIS)
    target.append(ACUTE)
    return _nfc("".join(ch for cluster in clusters for ch in cluster))


def os_cells(lemma: str, declension: str) -> list[tuple[str, str]]:
    """The regular cells of a second-declension -ος word of the given class.

    ``declension`` is one of ``CLASSES``. Empty for another class, for a
    word that is not an accented -ος word of at least two syllables, and
    for a stem in -ο (εὔνοος). Cells that share a form are merged, their
    names joined with "; ".
    """
    lemma = _nfc(lemma)
    if declension not in CLASSES:
        return []
    kind = accent_type(lemma)
    bare = _clusters(lemma)
    if (kind is None or len(bare) < 3 or bare[-1][0] != "ς"
            or bare[-2][0] != "ο" or bare[-3][0].lower() == "ο"):
        return []
    stem = lemma[:-2]
    if kind == "oxytone":
        short = long = stem
        accented = 1
    else:
        short = stem
        long = stem if kind == "paroxytone" else _penult_acute_stem(lemma)
        accented = 0
        if long is None:
            return []
    groups = [(long, _LONG), (short, _SHORT)]
    if declension != "noun":
        groups.append((short, _NEUTER_SHORT))
    if declension in ("adjective_eta", "adjective_alpha"):
        singular = (_FEMININE_ETA_LONG if declension == "adjective_eta"
                    else _FEMININE_ALPHA_LONG)
        groups += [(long, singular), (short, _FEMININE_SHORT),
                   (long, _FEMININE_PLURAL_LONG)]
    cells: dict[str, list[str]] = {}
    for base, endings in groups:
        for ending in endings:
            cells.setdefault(_nfc(base + ending[accented]), []).append(ending[2])
    if declension != "noun":
        # An adjective's -ον is its neuter singular too.
        cells[_nfc(short + _SHORT[0][accented])].append(_NEUTER_SINGULAR)
    rows = [(form, sorted(names, key=_ORDER.index)) for form, names in cells.items()]
    rows.sort(key=lambda row: _ORDER.index(row[1][0]))
    return [(form, "; ".join(names)) for form, names in rows]


def genitive_singular(lemma: str) -> str | None:
    """The word's genitive singular in -ου, the cell that shows whether a
    table declines it in the second declension."""
    cells = os_cells(lemma, "noun")
    return cells[0][0] if cells else None


def is_name_twin(reading: str | None, lemma: str) -> bool:
    """Whether ``reading`` is the capitalized spelling of ``lemma``: the
    name a lowercase headword's key reads as (ἐπάρατος, Ἐπάρατος)."""
    return (bool(reading) and reading != lemma and reading[0].isupper()
            and reading.lower() == lemma)


def reads_itself(lookup: Mapping[str, str], lemma: str) -> bool:
    """The lemma's key reads the lemma or the name of the same spelling.

    A lexicon lists the lowercase word, so its lowercase cells are its
    own; the name keeps its key and every capitalized form."""
    reading = lookup.get(lemma)
    return reading == lemma or is_name_twin(reading, lemma)


def select_os_lemmas(
        lookup: Mapping[str, str],
        listed: Mapping[str, str],
        source_of: Callable[[str], str | None] | None = None) -> dict[str, str]:
    """The listed lemmas that no table declines: lemma -> class.

    ``listed`` maps a lemma to its class (data/grc_second_declension.json).
    A lemma is declined here when it is a plain lowercase head that maps to
    itself in ``lookup`` (or to the name of the same spelling, which keeps
    its key: ``reads_itself``), has the breathing its first letter needs and no
    other, has a stem of at least ``MIN_STEM_LETTERS`` letters, is not one
    of the reviewed ``NOT_DECLINED``, and no table gives its genitive
    singular in -ου: a table that gives the genitive declines the word
    already, and its gaps are its own business.

    ``source_of`` gives a key's lookup source code. With it, only a genitive
    from a ``DECLINING_SOURCES`` row counts as a table's: a corpus pair or a
    Modern Greek row beside the headword (ψευδηγόρου from a treebank) leaves
    the rest of the paradigm unread. Without it every genitive row counts.
    """
    selected = {}
    for lemma in sorted(listed):
        if (not reads_itself(lookup, lemma) or not is_plain_lemma(lemma)
                or lemma in NOT_DECLINED
                or _needs_breathing_and_lacks_it(lemma)
                or not has_only_initial_breathing(lemma)
                or len(strip_accents(lemma)) - 2 < MIN_STEM_LETTERS):
            continue
        genitive = genitive_singular(lemma)
        if genitive is None:
            continue
        if lookup.get(genitive) == lemma and (
                source_of is None
                or source_of(genitive) in DECLINING_SOURCES | {None}):
            continue
        selected[lemma] = listed[lemma]
    return selected


def read_classes(record: Mapping) -> dict[str, str]:
    """lemma -> class from the record in data/grc_second_declension.json."""
    return {lemma: name for name, groups in record["classes"].items()
            if name in CLASSES
            for lemmas in groups.values() for lemma in lemmas}


def os_paradigms(selected: Mapping[str, str]) -> Iterable[tuple[str, str, list]]:
    """(kind, lemma, cells) for ``grc_noun_gaps.fill_gap_cells``, without
    the reviewed ``WITHHELD_CELLS``."""
    for lemma in sorted(selected):
        yield "os_" + selected[lemma], lemma, [
            (form, cell) for form, cell in os_cells(lemma, selected[lemma])
            if form not in WITHHELD_CELLS]


def os_rivals(
        lookup: Mapping[str, str],
        listed: Mapping[str, str],
        selected: Mapping[str, str],
        source_of: Callable[[str], str | None] | None = None,
        ) -> dict[str, str | None]:
    """Words with a selected lemma's letters that are not declined here:
    word -> class, None when no lexicon gives it.

    A rival is a plain lowercase word, another accentuation or spelling of
    a selected lemma's letters, that the sources know as a word: a listed
    word that was not selected (εὐπρόσοδος, whose headword key reads as the
    name Εὐπρόσοδος), or a self-mapped head of ``lookup`` from a source
    outside ``MODERN_SOURCES`` (``source_of`` gives a key's source code;
    without it every self-mapped head counts). A Modern Greek head is no
    rival: its monotonic spelling of an Ancient cell (μετωπιαίοις) is that
    cell. Which cells the two words share is left to ``fill_gap_cells``.
    """
    letters = {strip_accents(lemma).lower() for lemma in selected}

    def is_rival(word: str) -> bool:
        return (word not in selected and is_plain_lemma(word)
                and strip_accents(word).lower() in letters)

    rivals = {word for word in listed if is_rival(word)}
    for key, lemma in lookup.items():
        if (key == lemma and key.endswith("ς") and key not in rivals
                and is_rival(key)
                and (source_of is None
                     or source_of(key) not in MODERN_SOURCES)):
            rivals.add(key)
    return {word: listed.get(word) for word in sorted(rivals)}


def rival_paradigms(
        rivals: Mapping[str, str | None]) -> Iterable[tuple[str, str, list]]:
    """(``RIVAL_KIND``, word, cells) of the ``os_rivals`` words.

    The cells of the word's class. A word no lexicon classes claims the
    cells of an adjective of three endings, which include those of every
    other class: the feminine in -α after ε, ι or ρ, in -η otherwise.
    """
    for word in sorted(rivals):
        declension = rivals[word] or (
            "adjective_alpha"
            if strip_accents(word)[-3:-2] in ("ε", "ι", "ρ")
            else "adjective_eta")
        yield RIVAL_KIND, word, os_cells(word, declension)


_VERB_CITATION_ENDINGS = ("ω", "ομαι", "αμαι", "ημι", "ωμι", "υμι")


def is_verb_citation(lemma: str | None) -> bool:
    """Whether ``lemma`` is spelled as a verb's citation form: lowercase, in
    -ω (not an oxytone noun in -ώ or a dative in -ῳ), -μαι or -μι."""
    if not lemma or lemma != lemma.lower():
        return False
    nfd = unicodedata.normalize("NFD", lemma)
    if "ͅ" in nfd or nfd.endswith(ACUTE):
        return False
    return strip_accents(lemma).endswith(_VERB_CITATION_ENDINGS)


def cells_beside_a_verb(
        lookup: Mapping[str, str],
        paradigms: Iterable[tuple[str, str, list]],
        attestation: Callable[[str], tuple[int, str | None] | None],
        ) -> dict[str, dict]:
    """Generated cells whose accentless key reads a verb, that the texts
    show to be the noun's or the adjective's: form -> evidence.

    A cell such as θεοφόροις has the letters of a verb form (θεοφοροῖς, of
    θεοφορέω), and the strict gate of ``fill_gap_cells`` keeps it out,
    because a spelling that differs from a known word's only in its accent
    is often that word misprinted (παραίνων, 66 tokens, is the participle
    παραινῶν). The cell is let in here when ``attestation`` (form -> (count,
    dominant part of speech) of the texts, None when unattested) shows its
    own spelling more often than every spelling of the verb with the same
    letters, and the annotated texts do not tag it a verb.
    """
    candidates: dict[str, dict[str, str]] = {}
    for _, lemma, cells in paradigms:
        for form, _ in cells:
            bare = strip_accents(form).lower()
            reading = lookup.get(bare)
            if (reading is not None and reading != lemma
                    and is_verb_citation(reading)):
                candidates.setdefault(bare, {})[form] = reading
    if not candidates:
        return {}
    lengths = {len(bare) for bare in candidates}
    spellings: dict[str, set[str]] = {}
    for key, reading in lookup.items():
        if len(key) not in lengths or not is_verb_citation(reading):
            continue
        bare = strip_accents(key).lower()
        if bare in candidates and key != bare:
            spellings.setdefault(bare, set()).add(key)
    kept = {}
    for bare in sorted(candidates):
        for form, verb in sorted(candidates[bare].items()):
            found = attestation(form)
            if not found or found[1] == "verb":
                continue
            rivals = {key: (attestation(key) or (0, None))[0]
                      for key in sorted(spellings.get(bare, ()))
                      if lookup.get(key) == verb}
            if all(count < found[0] for count in rivals.values()):
                kept[form] = {"verb": verb, "count": found[0],
                              "verb_spellings": rivals}
    return kept
