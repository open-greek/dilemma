#!/usr/bin/env python3
"""Export a word list of attested polytonic Modern Greek spellings.

The grc dictionary (``build/hunspell/grc_polytonic.dic``, from
``export_hunspell.py``) is built from Ancient and Medieval Greek sources, so
it has almost no Modern Greek vocabulary: τώρα, ἀκόμη, σπίτι, the weak
possessive pronouns του, της, μας and the conjunction κι are all missing,
some of them deliberately. A keyboard that ships only the grc dictionary
therefore rewrites polytonic Modern Greek (Modern Greek written with
breathings and accents, as in print before 1982, in church texts, and by some
writers today) into Ancient Greek neighbors: τώρα becomes τἄρα, ἀκόμη becomes
ἀκμὴ.

This script writes a separate list that such a keyboard can merge into the
grc dictionary, leaving the grc dictionary itself untouched::

    build/hunspell/grc_mg_polytonic.dic       one spelling per line + fr:
    build/hunspell/grc_mg_polytonic.aff       minimal, no affix rules
    build/hunspell/grc_mg_polytonic.version   version sidecar

Every entry is a spelling the grc dictionary does not have, so the whole list
is Modern-only by construction.

Source
------
The polytonic Modern Greek slice of Wikisource that the next-word language
model also trains on (``extract_polytonic_mg.iter_polytonic_mg_documents``):
documents by authors with a post-1800 year whose words are at least 40%
polytonic. Only the language model's TRAINING sentences are read
(``train_lm.sentence_goes_to_dev`` is false), so the language model's dev
sentences stay held out from the list as well.

Selection
---------
A spelling is listed when all of the following hold. Each rule is a
constant below, with its reason.

1. It is read from a polytonic source (``exclude monotonic sources``).
   The 40% register filter admits documents that are partly or largely
   monotonic. A document is dropped when more than
   ``DOC_MONOTONIC_MAX`` of its lowercase words carry a monotonic
   signal (``monotonic_signal``): a vowel-initial word without a
   breathing, or one of the monosyllables polytonic writing always
   accents (και, να, δεν, ...). Within the remaining documents, a
   sentence with any such word is skipped too.
2. It is attested often and widely enough: at least ``MIN_TOKENS``
   tokens from at least ``MIN_AUTHORS`` different authors. One author
   can repeat a misspelling or an idiolect form many times; a second
   author makes it a spelling of the language.
3. It is well-formed polytonic Modern Greek (``mg_orthography_reason``):
   the grc structural rules (breathing on an initial vowel or rho, one
   accent within the last three syllables, grave only on the ultima, no
   breathing inside the word, ...), except that the closed list of words
   Modern Greek writes without an accent (``MG_UNACCENTED_WORDS``: the
   weak possessive pronouns and κι) is accepted, and an elided word may
   end in an unaccented ι (κι᾽, γι᾽).
4. It is not a weak mark-only respelling of a much commoner spelling of
   the same letters in the same corpus (``DOMINATED_SHARE``): the
   monotonic μπορεί beside μπορεῖ, a dropped breathing beside the right
   one.
5. Its case is the case the texts give it: a lowercase entry when the
   word occurs in lowercase at least as often as capitalized inside a
   sentence, else the capitalized spelling (a proper name). All-capital
   headings are not read.
6. The grc dictionary does not already have it, and a recorded review did
   not reject it (``data/hunspell_grc_spelling_review.json``, and the grc
   explicit rejects such as the acute θά/γιά/στό/στά, apart from the
   weak pronoun του).

Each listed oxytone also gets its contextual twin: the acute of a grave
(στὴν -> στήν) and the grave of a final acute (τοπικό -> τοπικὸ), as long
as the twin is well-formed, not rejected and not already in grc. A keyboard
needs both to write the grave inside a sentence and the acute before
punctuation.

The ``fr:`` field uses the grc bucket edges (``export_hunspell.freq_bucket``:
C >= 1000, M >= 100, R >= 1) on the spelling's own token count in the
Modern Greek slice, the acute and grave twins counted together.

Evaluation variants
-------------------
The language model's dev split is per sentence, so a dev sentence shares
its document with training sentences. ``--holdout-dev-documents`` builds the
list from documents without any dev sentence, and ``--holdout-author-fold
K/N`` from authors outside fold K of N (authors hashed into N folds), so a
held-out dev sentence never contributes to the list it is scored against.
``eval/eval_mg_polytonic.py`` measures coverage with these.

Usage
-----

    python export_mg_polytonic.py                      # shipping list
    python export_mg_polytonic.py --out-dir DIR \\
        --holdout-author-fold 0/5                       # evaluation list
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, NamedTuple

from export_hunspell import (
    GRC_REJECT_FORMS,
    OUT,
    SPACING_DIACRITICS,
    _greek_bases,
    _syllable_base_indexes,
    contextual_acute,
    freq_bucket,
    get_git_commit,
    grc_orthography_reason,
    load_grc_spelling_review,
    new_form_structural_reason,
    read_version_file,
)

ROOT = Path(__file__).parent
GRC_DIC = OUT / "grc_polytonic.dic"
LOOKUP_DB = ROOT / "data" / "lookup.db"
DIC_NAME = "grc_mg_polytonic"

KORONIS = "\u1FBD"
# Marks the Wikisource texts write for elision (κι', ἀπ’) and aphaeresis
# ('ς, ’ναι): ASCII apostrophe, U+2019, U+02BC, the koronis, and the spacing
# psili and dasia (νά ῾βρω, 238 times). All are stored as the koronis, the
# grc dictionary's glyph. U+2018 is left out: it opens quotations, and the
# slice never writes it.
ELISION_MARKS = "'\u2019\u02BC\u1FBD\u1FBF\u1FFE"
# A spacing psili or dasia in front of a vowel is that vowel's breathing,
# set before a capital the old way (῾Η, ᾿Αφ'), not a mark.
_SPACING_BREATHINGS = {"\u1FBF": "\u0313", "\u1FFE": "\u0314"}

# Greek letters and combining marks. The spacing koronis, psili, dasia and
# the other spacing accents of the Greek Extended block are left out, so a
# mark is never part of a word's letters; so are the numeral signs and the
# Greek question mark and ano teleia of the Greek and Coptic block.
_LETTERS = (
    "\u0300-\u036F\u0386\u0388-\u03FF"
    "\u1F00-\u1FBC\u1FBE\u1FC2-\u1FCC\u1FD0-\u1FDB\u1FE0-\u1FEC\u1FF2-\u1FFC"
)
TOKEN_RE = re.compile(
    rf"(?P<lead>[{ELISION_MARKS}])?(?P<word>[{_LETTERS}]+)"
    rf"(?P<trail>[{ELISION_MARKS}])?"
)
# A token touching one of these is part of something that is not a plain
# word: a hyphenated or line-broken word (ἀπο-φασίζω), a numeral (αʹ, Β΄),
# or letters run into digits or Latin script.
_JOINERS = frozenset(
    "-\u2010\u2011\u00AD\u0374\u0384\u00B4\u02B9\u2032\u2033\u1FFD\u1FEF")
# Letters some editions print with a compatibility character: the micro sign
# for mu (OCR of the mu in μακαρίσωμεν), and the symbol forms of beta,
# theta, phi, kappa, rho and pi. NFC keeps them apart from the letters;
# NFKC would fold them, but it would also take the koronis apart.
_LETTER_VARIANTS = str.maketrans({
    "\u00B5": "\u03BC", "\u03D0": "\u03B2", "\u03D1": "\u03B8",
    "\u03D5": "\u03C6", "\u03F0": "\u03BA", "\u03F1": "\u03C1",
    "\u03D6": "\u03C0",
})

# --------------------------------------------------------------------------
# Selection constants
# --------------------------------------------------------------------------

# A spelling must be attested by at least MIN_TOKENS training tokens from at
# least MIN_AUTHORS authors. Measured on the language model's dev sentences
# with each author's own texts held out (eval/eval_mg_polytonic.py), the
# second author is what makes a spelling carry over to other writers:
# accepting one-author spellings adds 3,200 entries, which cover 0.8% more
# of the dev words when their authors' texts are read and only 0.2% more
# when they are held out. A floor of 2 tokens would add 9,100 entries, of
# which 60% have a monotonic spelling Dilemma's lexicon knows, against 68%
# at 3 or 4 tokens and 41% at 1: more of them are slips and misprints.
MIN_TOKENS = 3
MIN_AUTHORS = 2

# A document is a monotonic or partly monotonic source when more than this
# share of its lowercase words carries a monotonic signal. A polytonic
# document has a few per thousand at most (a breathing OCR read as an
# accent, a stray quotation); monotonic text has several in every sentence
# (6 of the 16 lowercase words of the license note Wikisource adds to its
# pages), so 1% flags a document a few percent of whose text is monotonic.
# It flags 57 of the slice's 2,213 documents, 11% of its words. A short poem
# with one typing slip is not a monotonic source, so a document also needs
# DOC_MONOTONIC_MIN_SIGNALS signal words; its stray sentence is skipped by
# the sentence rule anyway.
DOC_MONOTONIC_MAX = 0.01
DOC_MONOTONIC_MIN_SIGNALS = 3

# Monosyllables that polytonic Modern Greek always writes with an accent and
# monotonic writes without one: an unaccented one is a monotonic signal.
# Weak pronouns (με, σε, το, τα, ...) are not in the list: polytonic writes
# them unaccented when they are enclitic.
MG_ALWAYS_ACCENTED_MONOSYLLABLES = frozenset({
    "και", "να", "θα", "δεν", "δε", "για", "μια", "πια",
    "στο", "στα", "στη", "στην", "στον", "στις", "στους",
})

# Words polytonic Modern Greek writes without an accent although they are
# not Ancient Greek enclitics: the weak genitive (possessive) pronouns, which
# lean on the word before them (ὁ πατέρας του, τὰ σπίτια μας), and κι, the
# form of καὶ before a vowel. μου and σου are Ancient Greek enclitics as
# well and the grc dictionary already has them. The weak accusatives
# το, τα, τον, την are left out on purpose: a bare-typed το is almost
# always the article τὸ, and accepting the unaccented spelling would stop
# a keyboard from restoring the article's accent.
MG_UNACCENTED_WORDS = frozenset({
    "μου", "σου", "του", "της", "μας", "σας", "τους", "των", "κι",
})

# An explicit grc reject that is a correct Modern Greek spelling: the weak
# pronoun του. The other grc rejects (θά, γιά, στό, στά, τού, ...) are wrong
# in Modern Greek too.
MG_REJECT_EXEMPT = frozenset({"του"})

# A spelling whose count is below this share of the commonest spelling with
# the same letters (marks aside, grave folded into acute) is a mark-level
# misspelling of it (εἴχε beside εἶχε, ὄτι beside ὅτι), unless it is a
# different word: its monotonic spelling differs from the commoner one's
# and Dilemma's lexicon (lookup.db) knows it (χρονιά beside χρόνια, ἔμενα
# beside ἐμένα, ὅποια beside ὁποῖα). DOMINATED_MIN is the count the commoner
# spelling needs for the rule to apply, since beside a rare word the rarer
# spelling is as often a real one.
DOMINATED_SHARE = 0.05
DOMINATED_MIN = 100

# Elided spellings may end in an unaccented iota: κι᾽, γι᾽, μι᾽, where the ι
# is the glide left when γιὰ or μιὰ loses its α.
_ELIDED_GLIDE_FINALS = frozenset("ιΙ")

GREEK_VOWELS = frozenset("αεηιουωΑΕΗΙΟΥΩ")
SMOOTH, ROUGH = "\u0313", "\u0314"
TONAL = frozenset("\u0300\u0301\u0342")


# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------

class Token(NamedTuple):
    """One word of a sentence.

    ``form`` is the NFC spelling with any elision or aphaeresis mark written
    as the koronis. ``initial`` is true at the start of a sentence or a line,
    where a capital says nothing about the word. ``plain`` is false for a
    word touching a hyphen, numeral sign, digit or Latin letter.
    """

    form: str
    lead: bool
    trail: bool
    initial: bool
    plain: bool


def _is_vowel(char: str) -> bool:
    return unicodedata.normalize("NFD", char)[:1] in GREEK_VOWELS


def tokenize(sentence: str) -> list[Token]:
    """Split ``sentence`` into words, keeping elision and aphaeresis marks.

    A mark after a final sigma closes a quotation rather than eliding a
    vowel, since elision leaves the medial σ (λέγουσ᾽), and a mark before a
    vowel opens one, since aphaeresis takes the initial vowel away ('ναι,
    'ς); neither is kept. A spacing breathing before a vowel is put on it.
    """
    text = unicodedata.normalize("NFC", sentence.translate(_LETTER_VARIANTS))
    tokens: list[Token] = []
    previous_end = 0
    for match in TOKEN_RE.finditer(text):
        word = match.group("word")
        start, end = match.span()
        before = text[start - 1] if start else ""
        after = text[end] if end < len(text) else ""
        gap = text[previous_end:start]
        initial = not tokens or "\n" in gap
        previous_end = end
        plain = not (
            before in _JOINERS or after in _JOINERS
            or (before and before.isalnum()) or (after and after.isalnum())
        )
        mark = match.group("lead")
        if mark in _SPACING_BREATHINGS and _is_vowel(word[0]):
            nfd = unicodedata.normalize("NFD", word)
            word = unicodedata.normalize(
                "NFC", nfd[0] + _SPACING_BREATHINGS[mark] + nfd[1:])
        lead = bool(mark) and not _is_vowel(word[0])
        trail = bool(match.group("trail")) and word[-1] not in "ςΣ"
        form = (KORONIS if lead else "") + word + (KORONIS if trail else "")
        tokens.append(Token(unicodedata.normalize("NFC", form), lead, trail,
                            initial, plain))
    return tokens


# Letters a Greek word can end in; a word ending in any other consonant
# right before a period is an abbreviation (τόμ., σελ., Θεόδ., ἅγ.).
_WORD_FINAL_CONSONANTS = frozenset("νςξψρΝΣΞΨΡ")


def is_abbreviation(token: Token) -> bool:
    """Whether ``token``, followed directly by a period, is an abbreviation:
    it ends in a consonant no complete Greek word ends in."""
    if token.trail:
        return False
    base = _bases(token.form)[-1][0]
    return base not in GREEK_VOWELS and base not in _WORD_FINAL_CONSONANTS


def _bases(word: str) -> list[tuple[str, str]]:
    """[(base letter, its combining marks)] of a word in NFD."""
    out: list[list[str]] = []
    for char in unicodedata.normalize("NFD", word):
        if unicodedata.combining(char) and out:
            out[-1][1] += char
        elif not unicodedata.combining(char):
            out.append([char, ""])
    return [(base, marks) for base, marks in out]


def monotonic_signal(token: Token) -> str | None:
    """Why a lowercase word reads as monotonic spelling, or None.

    Only lowercase words are judged: capitals in polytonic print often go
    without breathings and accents (headings, initials). An elided or
    aphaeresized word is not judged either, since its mark may have taken
    the accent or the breathing with it ('ναι, ἀπ᾽), nor a word that is
    not plain (a numeral such as αʹ, letters run into digits).
    """
    if token.lead or token.trail or not token.plain:
        return None
    word = token.form
    if not word.islower():
        return None
    letters = _bases(word)
    if not letters:
        return None
    first, first_marks = letters[0]
    if first in GREEK_VOWELS:
        marks = first_marks
        if len(letters) > 1 and letters[1][0] in "ιυ" and not (
                set(first_marks) & (TONAL | {"\u0308"})):
            marks += letters[1][1]
        if SMOOTH not in marks and ROUGH not in marks:
            return "missing_breathing"
    if word in MG_ALWAYS_ACCENTED_MONOSYLLABLES:
        return "unaccented_monosyllable"
    return None


# --------------------------------------------------------------------------
# Corpus pass
# --------------------------------------------------------------------------

@dataclass
class DocumentInfo:
    key: str
    author: str
    title: str
    words: int = 0              # lowercase words, the signal share's base
    signals: int = 0
    sentences: int = 0
    dev_sentences: int = 0

    @property
    def signal_share(self) -> float:
        return self.signals / self.words if self.words else 0.0

    @property
    def monotonic(self) -> bool:
        """Whether the document is a monotonic or partly monotonic source."""
        return (self.signals >= DOC_MONOTONIC_MIN_SIGNALS
                and self.signal_share > DOC_MONOTONIC_MAX)


@dataclass
class CorpusCounts:
    """Token counts of the Modern Greek slice, kept per document so that any
    subset of documents (a held-out split) can be selected afterwards.

    ``forms`` maps (spelling, position) to {document index: tokens}, where
    position is ``lower`` (a lowercase word), ``cap`` (a capitalized word
    inside a sentence) or ``initial`` (a capitalized word opening a sentence
    or line). ``before_vowel`` counts, per spelling, the tokens followed in
    their sentence by a vowel-initial word, for the elision table.
    ``dev`` holds the dev sentences' tokens, per document.
    """

    documents: list[DocumentInfo] = field(default_factory=list)
    forms: dict[tuple[str, str], Counter] = field(
        default_factory=lambda: defaultdict(Counter))
    before_vowel: dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter))
    dev: dict[int, list[list[Token]]] = field(
        default_factory=lambda: defaultdict(list))
    skipped_sentences: int = 0


def _position(token: Token) -> str | None:
    """``lower``, ``cap``, ``initial``, or None for an all-capital word."""
    letters = [c for c in token.form if c.isalpha()]
    if not letters:
        return None
    if all(not c.isupper() for c in letters):
        return "lower"
    if len(letters) > 1 and all(c.isupper() for c in letters):
        return None
    if letters[0].isupper() and all(not c.isupper() for c in letters[1:]):
        return "initial" if token.initial else "cap"
    return None


def count_corpus(
    parquet_path: Path | None = None,
    max_docs: int | None = None,
) -> CorpusCounts:
    """Read the slice once: monotonic signals per document, token counts of
    the training sentences, and the dev sentences.

    Short sentences (fewer than three words) are skipped, as the language
    model's loader skips them. A sentence with a monotonic signal is counted
    towards its document's share but contributes no tokens.
    """
    from extract_polytonic_mg import (
        iter_polytonic_mg_documents,
        sentence_id,
        split_sentences_with_closers,
    )
    from train_lm import sentence_goes_to_dev

    counts = CorpusCounts()
    for doc in iter_polytonic_mg_documents(parquet_path, max_docs=max_docs):
        index = len(counts.documents)
        info = DocumentInfo(doc.key, doc.author, doc.title)
        counts.documents.append(info)
        for i, (sentence, closer) in enumerate(
                split_sentences_with_closers(doc.text)):
            tokens = tokenize(sentence)
            if len(tokens) < 3:
                continue
            if closer.startswith(".") and is_abbreviation(tokens[-1]):
                tokens[-1] = tokens[-1]._replace(plain=False)
            info.sentences += 1
            signals = 0
            for token in tokens:
                if _position(token) == "lower":
                    info.words += 1
                    if monotonic_signal(token):
                        signals += 1
            info.signals += signals
            if sentence_goes_to_dev(sentence_id(doc.key, i)):
                info.dev_sentences += 1
                counts.dev[index].append(tokens)
                continue
            if signals:
                counts.skipped_sentences += 1
                continue
            for k, token in enumerate(tokens):
                if not token.plain:
                    continue
                position = _position(token)
                if position is None:
                    continue
                counts.forms[(token.form, position)][index] += 1
                following = tokens[k + 1] if k + 1 < len(tokens) else None
                if (following is not None and not following.lead
                        and _is_vowel(following.form[0])):
                    counts.before_vowel[token.form][index] += 1
    return counts


# --------------------------------------------------------------------------
# Document selection (shipping list and held-out evaluation lists)
# --------------------------------------------------------------------------

def author_fold(author: str, folds: int) -> int:
    """Deterministic fold of an author, for author-held-out evaluation."""
    digest = hashlib.blake2b(author.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest[:4], "little") % folds


def source_documents(
    counts: CorpusCounts,
    *,
    holdout_dev_documents: bool = False,
    holdout_author_fold: tuple[int, int] | None = None,
) -> set[int]:
    """The documents the list may read: polytonic ones, minus a held-out set.

    ``holdout_dev_documents`` drops every document holding a dev sentence;
    ``holdout_author_fold`` (k, n) drops every author in fold k of n.
    """
    keep: set[int] = set()
    for index, info in enumerate(counts.documents):
        if info.monotonic:
            continue
        if holdout_dev_documents and info.dev_sentences:
            continue
        if (holdout_author_fold is not None
                and author_fold(info.author, holdout_author_fold[1])
                == holdout_author_fold[0]):
            continue
        keep.add(index)
    return keep


# --------------------------------------------------------------------------
# Orthography
# --------------------------------------------------------------------------

# The accent rules grc_orthography_reason reports on the syllable count.
_ACCENT_WINDOW_REASONS = frozenset({
    "accent_before_antepenult", "circumflex_before_penult",
    "misplaced_second_accent",
})
_TONAL_CODES = frozenset({0x0300, 0x0301, 0x0342})


def _mg_nuclei(bases: list[tuple[str, frozenset[int]]]) -> list[list[int]]:
    """Vowel nuclei of a Modern Greek word, counting what Modern Greek
    pronounces as one syllable as one.

    Two things merge, on top of the Ancient Greek diphthongs:

    * synizesis: an unaccented ι, ει, οι or υ before another vowel is a
      glide, not a syllable (πλά-για-σε, τέ-λειω-σε, μετά-νοιω-σα);
    * a falling diphthong: an accented vowel followed by ι or υ, with or
      without a diaeresis, is one syllable (γάϊ-δα-ρος, βόι-δια).
    """
    merged: list[list[int]] = []
    for members in _syllable_base_indexes(bases):
        if merged and members[0] == merged[-1][-1] + 1:
            previous = merged[-1]
            prev_tonal = any(bases[i][1] & _TONAL_CODES for i in previous)
            last_base, last_marks = bases[previous[-1]]
            first_base, first_marks = bases[members[0]]
            glide = (not prev_tonal and last_base.lower() in "ιυ"
                     and 0x0308 not in last_marks)
            falling = (prev_tonal and len(members) == 1
                       and first_base.lower() in "ιυ"
                       and not first_marks & _TONAL_CODES)
            if glide or falling:
                previous.extend(members)
                continue
        merged.append(list(members))
    return merged


def mg_accent_window_ok(form: str) -> bool:
    """Whether ``form``'s accents sit where Modern Greek allows, syllables
    counted by :func:`_mg_nuclei`: one accent within the last three
    syllables (a circumflex within the last two), or a proparoxytone or
    properispomenon with an enclitic's second acute on its ultima."""
    bases = _greek_bases(form)
    nuclei = _mg_nuclei(bases)
    nucleus_of = {i: n for n, members in enumerate(nuclei) for i in members}
    accents = [(nucleus_of.get(i), marks & _TONAL_CODES)
               for i, (_base, marks) in enumerate(bases)
               if marks & _TONAL_CODES]
    if not accents or any(n is None for n, _ in accents):
        return False
    elided = ord(form[-1]) in SPACING_DIACRITICS
    last = len(nuclei) - 1 + (1 if elided else 0)
    if len(accents) == 1:
        (n, mark), = accents
        after = last - n
        return after <= (1 if mark == {0x0342} else 2)
    if len(accents) == 2 and not elided:
        (first, first_mark), (second, second_mark) = accents
        return (second == len(nuclei) - 1 and second_mark == {0x0301}
                and ((first_mark == {0x0301} and second - first == 2)
                     or (first_mark == {0x0342} and second - first == 1)))
    return False


def truncates_word(form: str, lexicon: set[str] | frozenset[str]) -> bool:
    """Whether ``form`` is a word of ``lexicon`` that lost its final vowel
    without an elision mark (τόμ for τόμο, τώρ for τώρα): the same letters
    and marks plus one vowel spell a word."""
    return any(form + vowel in lexicon for vowel in "αεηιουω")


def mg_orthography_reason(
    form: str, lexicon: set[str] | frozenset[str] | None = None,
) -> str | None:
    """Return why ``form`` is not a well-formed polytonic Modern Greek word.

    The grc structural rules hold for polytonic Modern Greek, which keeps
    the historical breathings and accents: :func:`grc_orthography_reason`
    and :func:`new_form_structural_reason`. Four things differ:

    * the closed list ``MG_UNACCENTED_WORDS`` (του, της, μας, σας, τους,
      των, κι) is correct without an accent, where grc rejects an
      unaccented monosyllable;
    * του, an explicit grc reject as a misspelled article, is the Modern
      Greek weak pronoun;
    * syllables are counted as Modern Greek pronounces them
      (:func:`mg_accent_window_ok`), so τέλειωσε, πλάγιασε and γάϊδαρος
      keep their accent within the last three syllables;
    * a lowercase word may end in a consonant other than ν, ρ, ς, ξ, ψ:
      Modern Greek has interjections and loanwords so spelled (ἄχ,
      κονιάκ, χανούμ). It must not be a word of ``lexicon`` with its final
      vowel lost (:func:`truncates_word`); without a lexicon the grc rule
      stands.

    An elided word may also end in an unaccented ι, the glide of κι᾽,
    γι᾽, μι᾽, where Ancient Greek elision leaves a consonant.
    """
    if form in MG_UNACCENTED_WORDS:
        return None
    if form in GRC_REJECT_FORMS and form not in MG_REJECT_EXEMPT:
        return "explicit_reject"
    reason = grc_orthography_reason(form)
    if reason in _ACCENT_WINDOW_REASONS and mg_accent_window_ok(form):
        reason = None
    if reason is not None:
        return reason
    reason = new_form_structural_reason(form)
    if reason == "truncated_fragment" and lexicon is not None:
        reason = "truncated_word" if truncates_word(form, lexicon) else None
    if reason is not None:
        return reason
    if form.endswith(KORONIS) and len(form) > 1:
        base, marks = _bases(form[:-1])[-1]
        if base in GREEK_VOWELS and (
                base not in _ELIDED_GLIDE_FINALS or set(marks) & TONAL):
            return "vowel_before_elision"
    if form.startswith(KORONIS):
        rest = _bases(form[1:])
        if not rest or rest[0][0] in GREEK_VOWELS:
            return "vowel_after_aphaeresis"
    return None


_MONOTONIC_DROP = frozenset("\u0313\u0314\u0345\u0304\u0306")
_MONOTONIC_ACUTE = {"\u0300": "\u0301", "\u0342": "\u0301"}


def monotonic_spelling(form: str) -> str:
    """``form`` as monotonic Greek writes it: no breathings, iota subscript,
    vowel-length marks or elision mark, every accent an acute, and no
    diaeresis after an accented vowel (σόϊ -> σόι). Monosyllables keep
    their accent here; :func:`load_known_words` also looks them up bare."""
    out: list[str] = []
    after_accent = False
    for char in unicodedata.normalize("NFD", form.replace(KORONIS, "")):
        if not unicodedata.combining(char):
            out.append(char)
            after_accent = after_accent and char in GREEK_VOWELS
            continue
        if char in _MONOTONIC_DROP:
            continue
        char = _MONOTONIC_ACUTE.get(char, char)
        if char == "\u0301":
            after_accent = True
        elif char == "\u0308" and after_accent:
            continue
        out.append(char)
    return unicodedata.normalize("NFC", "".join(out))


def _without_accent(form: str) -> str:
    nfd = unicodedata.normalize("NFD", form)
    return unicodedata.normalize("NFC", nfd.replace("\u0301", ""))


def load_known_words(path: Path = LOOKUP_DB):
    """A test of whether Dilemma's lexicon (``lookup.db``, Wiktionary and
    the treebanks, Modern and Ancient Greek) has a monotonic spelling, or
    None when the database is not on disk. A monosyllable is also looked
    up without its accent, which monotonic writing leaves off (μια)."""
    if not path.exists():
        return None
    import sqlite3

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    def known(form: str) -> bool:
        spellings = {form, form.lower()}
        if len(_mg_nuclei(_greek_bases(form))) <= 1:
            spellings |= {_without_accent(s) for s in spellings}
        return any(
            conn.execute("SELECT 1 FROM lookup WHERE form = ? LIMIT 1",
                         (spelling,)).fetchone()
            for spelling in spellings)

    return known


def distinct_word(form: str, dominant: str, known_word) -> bool:
    """Whether ``form``, a rare spelling of ``dominant``'s letters, is a
    different word rather than a misspelling of it: its monotonic spelling
    is not ``dominant``'s and the lexicon knows it. An elided spelling is
    judged by the words it can elide (κάθ᾽ from κάθε, beside καθ᾽ from
    κατά)."""
    mono = monotonic_spelling(form)
    if mono == monotonic_spelling(dominant):
        return False
    if form.endswith(KORONIS):
        return any(known_word(mono + vowel) for vowel in "αεηιουω")
    return known_word(mono)


def spelling_skeleton(form: str) -> str:
    """The letters of ``form`` without accents, breathings, diaeresis or
    iota subscript, in its own case: the key under which mark-only
    respellings compete (rule 4). Case is kept so that a name (Δία) does
    not compete with a common word (διά)."""
    nfd = unicodedata.normalize("NFD", contextual_acute(form))
    return "".join(c for c in nfd if not unicodedata.combining(c))


def oxytone_twins(form: str) -> list[str]:
    """The contextual twin of an oxytone spelling: the acute of a final
    grave (στὴν -> στήν), or the grave of a final acute (τοπικό ->
    τοπικὸ). A word with two accents (an enclitic's second acute) and an
    elided word have none."""
    if any(ord(c) in SPACING_DIACRITICS for c in form):
        return []
    nfd = unicodedata.normalize("NFD", form)
    tonal = [i for i, c in enumerate(nfd) if c in TONAL]
    if len(tonal) != 1:
        return []
    i = tonal[0]
    if nfd[i] == "\u0300":
        return [contextual_acute(form)]
    if nfd[i] != "\u0301":
        return []
    # The acute must sit on the last vowel: nothing but consonants and
    # marks may follow it.
    if any(c in GREEK_VOWELS for c in nfd[i + 1:]):
        return []
    return [unicodedata.normalize("NFC", nfd[:i] + "\u0300" + nfd[i + 1:])]


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def read_grc_words(path: Path = GRC_DIC) -> set[str]:
    """Every spelling the grc dictionary accepts as written."""
    from export_lm import read_hunspell_words

    if not path.exists():
        raise SystemExit(
            f"{path} not found: run `python export_hunspell.py` first; the "
            "Modern Greek list holds only spellings it lacks.")
    return read_hunspell_words(path)


def _capitalized(form: str) -> str:
    """``form`` with its first letter capitalized (a lead koronis aside)."""
    offset = 1 if form.startswith(KORONIS) else 0
    head = form[offset:offset + 1]
    return unicodedata.normalize(
        "NFC", form[:offset] + head.upper() + form[offset + 1:])


def _lowercase(form: str) -> str:
    return unicodedata.normalize("NFC", form.lower())


class Candidate(NamedTuple):
    form: str
    tokens: int
    authors: int
    documents: int


def gather_candidates(
    counts: CorpusCounts, sources: set[int],
) -> dict[str, Candidate]:
    """Fold the per-position counts of the source documents into one
    candidate per word, in the case the texts give it (rule 5)."""
    lower: Counter = Counter()
    cap: Counter = Counter()
    initial: Counter = Counter()
    docs: dict[str, set[int]] = defaultdict(set)
    capital_spelling: dict[str, Counter] = defaultdict(Counter)
    for (form, position), per_doc in counts.forms.items():
        n = sum(c for d, c in per_doc.items() if d in sources)
        if not n:
            continue
        key = _lowercase(form)
        {"lower": lower, "cap": cap, "initial": initial}[position][key] += n
        if position != "lower":
            capital_spelling[key][form] += n
        docs[key].update(d for d in per_doc if d in sources)
    out: dict[str, Candidate] = {}
    for key, documents in docs.items():
        total = lower[key] + cap[key] + initial[key]
        if lower[key] and lower[key] >= cap[key]:
            form = key
        elif cap[key] > lower[key]:
            form = capital_spelling[key].most_common(1)[0][0]
        else:
            continue        # only ever seen opening a sentence or line
        authors = {counts.documents[d].author for d in documents}
        out[form] = Candidate(form, total, len(authors), len(documents))
    return out


class Selection(NamedTuple):
    entries: dict[str, int]         # spelling -> token count for fr:
    report: dict[str, int]
    rejected: dict[str, list[str]]  # reason -> sample spellings


def select_forms(
    candidates: dict[str, Candidate],
    grc_words: set[str],
    reviewed_rejects: frozenset[str] = frozenset(),
    known_word=None,
) -> Selection:
    """Apply rules 2, 3, 4 and 6 and add the oxytone twins.

    ``known_word`` (:func:`load_known_words`) lets a rare spelling that is a
    different word survive rule 4; without it every such spelling goes.
    """
    report: Counter = Counter()
    rejected: dict[str, list[str]] = defaultdict(list)

    def reject(reason: str, form: str) -> None:
        report[reason] += 1
        if len(rejected[reason]) < 50:
            rejected[reason].append(form)

    reviewed = {contextual_acute(f) for f in reviewed_rejects}
    # What a consonant-final word must not be the truncation of: the
    # spellings the slice attests MIN_TOKENS times. Not grc, whose paradigm
    # tables extend almost any stem by a vowel (ἄχε, ἄχι, ἄχω).
    lexicon = {f for f, c in candidates.items() if c.tokens >= MIN_TOKENS}

    # Rule 4 needs the commonest well-formed spelling of each letter
    # sequence, with the twins counted together.
    by_key: Counter = Counter()
    for form, cand in candidates.items():
        by_key[contextual_acute(form)] += cand.tokens
    dominant: dict[str, tuple[int, str]] = {}
    for key, n in by_key.items():
        if mg_orthography_reason(key, lexicon) is None:
            skeleton = spelling_skeleton(key)
            dominant[skeleton] = max(dominant.get(skeleton, (0, "")), (n, key))

    selected: dict[str, int] = {}
    for form, cand in sorted(candidates.items()):
        if cand.tokens < MIN_TOKENS:
            reject("too_few_tokens", form)
            continue
        if cand.authors < MIN_AUTHORS:
            reject("single_author", form)
            continue
        reason = mg_orthography_reason(form, lexicon)
        if reason is not None:
            reject(f"orthography:{reason}", form)
            continue
        key = contextual_acute(form)
        top, top_key = dominant.get(spelling_skeleton(key), (0, ""))
        if (top >= DOMINATED_MIN and by_key[key] < top
                and by_key[key] < DOMINATED_SHARE * top):
            if known_word is None or not distinct_word(form, top_key, known_word):
                reject("dominated_respelling", form)
                continue
            report["distinct_rare_spelling"] += 1
        if key in reviewed:
            reject("reviewed_reject", form)
            continue
        selected[form] = by_key[key]

    entries: dict[str, int] = {}
    for form, n in selected.items():
        for spelling in [form, *oxytone_twins(form)]:
            if spelling in entries:
                continue
            if spelling != form:
                if (mg_orthography_reason(spelling, lexicon) is not None
                        or contextual_acute(spelling) in reviewed):
                    report["twin_rejected"] += 1
                    continue
                report["twins"] += 1
            if spelling in grc_words:
                report["in_grc"] += 1
                continue
            entries[spelling] = n
    report["entries"] = len(entries)
    return Selection(entries, dict(report), dict(rejected))


# --------------------------------------------------------------------------
# Modern Greek elisions, for the morphology table
# --------------------------------------------------------------------------

# An elided spelling enters the Modern Greek elision table when the slice
# writes it at least this often, by this many authors: the measured
# keyboard change behind the table used these thresholds, and one author's
# habit is not the language's.
MG_ELISION_MIN_TOKENS = 10
MG_ELISION_MIN_AUTHORS = 2

# What Modern Greek elision removes from the end of a word: one vowel, or
# the digraphs αι, ει, οι, ου that spell one (εἶναι -> εἶν᾽, ποὺ -> π᾽,
# μοῦ -> μ᾽).
MG_ELIDED_ENDINGS = frozenset({
    "α", "ε", "η", "ι", "ο", "υ", "ω", "αι", "ει", "οι", "ου",
})

# κι᾽ is no elided word but κι itself, the form of καὶ before a vowel,
# written with the mark as most older polytonic print does (κι᾽ ἐγώ). It
# has no longer full form, so the rule below cannot find it.
MG_EXTRA_ELISIONS = {"κι": "κι\u1FBD"}


# The aspirated consonants elision before a rough breathing leaves (ἀφ᾽
# ὅτου for ἀπ᾽, καθ᾽ ἑκάστην for κατ᾽), and the plain ones they stand for.
_DEASPIRATED = {"φ": "π", "θ": "τ", "χ": "κ"}


class ElisionCandidate(NamedTuple):
    full: str
    elided: str
    elided_tokens: int
    elided_before_vowel: int
    full_before_vowel: int


def _elided_ending(full: str, stem: str) -> str | None:
    """The marked letters ``full`` has after ``stem``, when ``full`` is the
    stem's letters and marks plus one of ``MG_ELIDED_ENDINGS``; else None."""
    full_nfd = unicodedata.normalize("NFD", full)
    stem_nfd = unicodedata.normalize("NFD", stem)
    if not full_nfd.startswith(stem_nfd):
        return None
    rest = full_nfd[len(stem_nfd):]
    if not rest or unicodedata.combining(rest[0]):
        return None
    letters = "".join(c for c in rest if not unicodedata.combining(c))
    return rest if letters in MG_ELIDED_ENDINGS else None


def modern_greek_elisions(
    counts: CorpusCounts, sources: set[int],
) -> tuple[dict[str, str], list[ElisionCandidate]]:
    """The Modern Greek elision table: ``{full: elided}``.

    An elided spelling E qualifies when the slice writes it at least
    ``MG_ELISION_MIN_TOKENS`` times, by ``MG_ELISION_MIN_AUTHORS`` authors,
    and it is well-formed (:func:`mg_orthography_reason`). Its full form is
    the commonest well-formed word attested ``MIN_TOKENS`` times by
    ``MIN_AUTHORS`` authors that spells E's letters, with the same marks,
    plus one ending of ``MG_ELIDED_ENDINGS``: an unaccented ending when E
    keeps an accent (τώρα -> τώρ᾽, ὅλα -> ὅλ᾽), an accented one when E has
    none, since an oxytone loses its accent (γιὰ -> γι᾽, στὸ -> στ᾽,
    ποὺ -> π᾽). The full form must be at least as common as the elided
    spelling: an elided spelling that outnumbers every word it could
    shorten is mostly another word's (καθ᾽ is κατά before a rough
    breathing, not the rare καθό). The full form's contextual twin (στό
    beside στὸ) is keyed too when it is well-formed. An unaccented
    preposition's aspirated spelling (ἀφ᾽, καθ᾽, μεθ᾽) is left out when the
    plain one is attested: it is the plain elision before a rough
    breathing, which a keyboard derives itself. ``MG_EXTRA_ELISIONS`` adds
    κι -> κι᾽.

    Also returns, for each pair, how often the slice writes the elided and
    the full spelling right before a vowel-initial word: elision in Modern
    Greek is optional, and a keyboard deciding whether to rewrite a word
    before every vowel needs to know how often the texts do.
    """
    tokens: Counter = Counter()
    authors: dict[str, set[str]] = defaultdict(set)
    for (form, position), per_doc in counts.forms.items():
        if position != "lower":
            continue
        for d, n in per_doc.items():
            if d in sources:
                tokens[form] += n
                authors[form].add(counts.documents[d].author)
    before_vowel: Counter = Counter()
    for form, per_doc in counts.before_vowel.items():
        before_vowel[form] = sum(n for d, n in per_doc.items() if d in sources)

    lexicon = {f for f, n in tokens.items() if n >= MIN_TOKENS}
    words: dict[str, list[str]] = defaultdict(list)
    for form, n in tokens.items():
        if (n >= MIN_TOKENS and len(authors[form]) >= MIN_AUTHORS
                and KORONIS not in form
                and mg_orthography_reason(form, lexicon) is None):
            words[spelling_skeleton(form)[:-1]].append(form)
            words[spelling_skeleton(form)[:-2]].append(form)

    pairs: dict[str, str] = {}
    found: list[ElisionCandidate] = []
    for elided, n in sorted(tokens.items()):
        if (not elided.endswith(KORONIS) or elided.startswith(KORONIS)
                or n < MG_ELISION_MIN_TOKENS
                or len(authors[elided]) < MG_ELISION_MIN_AUTHORS
                or mg_orthography_reason(elided, lexicon) is not None):
            continue
        stem = elided[:-1]
        accented = any(c in TONAL for c in unicodedata.normalize("NFD", stem))
        plain = stem[:-1] + _DEASPIRATED.get(stem[-1], "") + KORONIS
        if (not accented and len(stem) > 1 and stem[-1] in _DEASPIRATED
                and tokens[plain] >= MG_ELISION_MIN_TOKENS):
            continue
        fulls = []
        for full in set(words.get(spelling_skeleton(stem), ())):
            ending = _elided_ending(full, stem)
            if ending is None or tokens[full] < n:
                continue
            if accented == any(c in TONAL for c in ending):
                continue
            fulls.append((tokens[full], full))
        if not fulls:
            continue
        _, full = max(fulls)
        for key in [full, *oxytone_twins(full)]:
            if key in pairs or (key != full and (
                    mg_orthography_reason(key, lexicon) is not None)):
                continue
            pairs[key] = elided
            found.append(ElisionCandidate(
                key, elided, n, before_vowel[elided], before_vowel[key]))
    for full, elided in MG_EXTRA_ELISIONS.items():
        pairs.setdefault(full, elided)
        found.append(ElisionCandidate(
            full, elided, tokens[elided], before_vowel[elided],
            before_vowel[full]))
    return dict(sorted(pairs.items())), found


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def corpus_identity(parquet_path: Path | None = None) -> str:
    """The Wikisource snapshot the list was read from."""
    from extract_polytonic_mg import DEFAULT_PARQUET

    path = Path(parquet_path or DEFAULT_PARQUET)
    return f"glossAPI/Wikisource_Greek_texts snapshot {path.parent.name}"


def write_list(
    entries: dict[str, int],
    out_dir: Path,
    *,
    variant: str,
    source: str,
    version: str | None = None,
    commit: str | None = None,
) -> dict:
    """Write ``<DIC_NAME>.dic``, ``.aff`` and ``.version`` to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    version = version or read_version_file()
    commit = commit or get_git_commit()
    lines = [f"{form}\tfr:{freq_bucket(n)}" for form, n in sorted(entries.items())]
    dic_path = out_dir / f"{DIC_NAME}.dic"
    dic_path.write_text(
        f"{len(lines)}\n" + "".join(line + "\n" for line in lines),
        encoding="utf-8")
    aff_path = out_dir / f"{DIC_NAME}.aff"
    aff_path.write_text(
        "# Dilemma polytonic Modern Greek word list\n"
        f"# Version {version} (built from commit {commit})\n"
        "# Generated by export_mg_polytonic.py. The .dic has no affix\n"
        "# flags, so this file only makes the pair loadable on its own.\n"
        "SET UTF-8\nLANG grc\nFLAG num\n",
        encoding="utf-8")
    buckets = Counter(freq_bucket(n) for n in entries.values())
    ver_path = out_dir / f"{DIC_NAME}.version"
    ver_path.write_text(
        f"version: {version}\n"
        f"commit: {commit}\n"
        f"variant: {variant}\n"
        f"entries: {len(lines)}\n"
        f"aff_rules: 0\n"
        f"source: {source}\n"
        f"buckets: C={buckets['C']} M={buckets['M']} R={buckets['R']}\n",
        encoding="utf-8")
    return {"entries": len(lines), "dic_path": str(dic_path),
            "buckets": dict(buckets)}


def build(
    out_dir: Path = OUT,
    *,
    parquet_path: Path | None = None,
    holdout_dev_documents: bool = False,
    holdout_author_fold: tuple[int, int] | None = None,
    counts: CorpusCounts | None = None,
    grc_words: set[str] | None = None,
) -> tuple[Selection, dict]:
    """Read the slice, select the spellings and write the list."""
    counts = counts or count_corpus(parquet_path)
    grc_words = grc_words if grc_words is not None else read_grc_words()
    sources = source_documents(
        counts, holdout_dev_documents=holdout_dev_documents,
        holdout_author_fold=holdout_author_fold)
    candidates = gather_candidates(counts, sources)
    known_word = load_known_words()
    if known_word is None:
        print(f"  NOTE: {LOOKUP_DB} not found; every rare respelling of a "
              "commoner spelling is dropped", file=sys.stderr)
    selection = select_forms(candidates, grc_words, load_grc_spelling_review(),
                             known_word)
    if holdout_dev_documents:
        variant = "grc-mg (evaluation: documents with a dev sentence held out)"
    elif holdout_author_fold is not None:
        k, n = holdout_author_fold
        variant = f"grc-mg (evaluation: author fold {k} of {n} held out)"
    else:
        variant = "grc-mg"
    stats = write_list(
        selection.entries, out_dir, variant=variant,
        source=f"{corpus_identity(parquet_path)}, language-model training "
               f"split, {len(sources)} of {len(counts.documents)} documents")
    return selection, stats


def _fold(text: str) -> tuple[int, int]:
    k, _, n = text.partition("/")
    k_i, n_i = int(k), int(n)
    if not 0 <= k_i < n_i:
        raise argparse.ArgumentTypeError(f"fold {text!r} is not K/N with 0 <= K < N")
    return k_i, n_i


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=OUT,
                    help=f"output directory (default {OUT})")
    ap.add_argument("--parquet", type=Path, default=None,
                    help="override the Wikisource parquet path")
    held = ap.add_mutually_exclusive_group()
    held.add_argument("--holdout-dev-documents", action="store_true",
                      help="evaluation list: drop every document that holds "
                           "a language-model dev sentence")
    held.add_argument("--holdout-author-fold", type=_fold, metavar="K/N",
                      help="evaluation list: drop the authors in fold K of N")
    args = ap.parse_args(list(argv) if argv is not None else None)

    print("Reading the polytonic Modern Greek slice...")
    counts = count_corpus(args.parquet)
    docs = counts.documents
    monotonic = [d for d in docs if d.monotonic]
    print(f"  {len(docs):,} documents, {len(monotonic):,} monotonic or partly "
          f"monotonic (> {DOC_MONOTONIC_MAX:.0%} signal words) not read")
    print(f"  {counts.skipped_sentences:,} training sentences skipped for a "
          "monotonic signal")
    selection, stats = build(
        args.out_dir, parquet_path=args.parquet, counts=counts,
        holdout_dev_documents=args.holdout_dev_documents,
        holdout_author_fold=args.holdout_author_fold)
    for key, value in sorted(selection.report.items()):
        print(f"  {key}: {value:,}")
    print(f"  wrote {stats['dic_path']} ({stats['entries']:,} entries, "
          f"buckets {stats['buckets']})")


if __name__ == "__main__":
    sys.exit(main())
