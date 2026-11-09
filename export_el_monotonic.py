#!/usr/bin/env python3
"""Export the monotonic Modern Greek word list for a monotonic keyboard.

Writes, to ``build/hunspell/`` by default::

    el_monotonic.dic                 the word list
    el_monotonic.aff                 SET UTF-8, LANG el, FLAG num, TRY
    el_monotonic.version             provenance, counts, input sha256
    el_monotonic_abbreviations.txt   abbreviations, with their periods
    el_monotonic_acronyms.txt        all-capital acronyms (ΗΠΑ, ΟΤΕ)
    el_monotonic.audit.json          what each route admitted and rejected
    el_monotonic.evidence.tsv.gz     per entry: route, rating, counts

The ``.dic`` has the entry count on its first line, then one line per
entry sorted by UTF-8 bytes: ``form<TAB>fr:C|M|R|X``, with `` mg:generated``
after ``fr:X`` for a paradigm form no frequency source has. Forms are NFC,
use the monotonic letters of the Greek block only, keep lexical
capitalization (names keep their capital, sentence capitals of common words
are not listed), end in ς, write elisions with U+2019, and carry no affix
flags. Double-accented enclitic forms (όνομά) are never entries: the
keyboard adds the second accent by rule.

Routes
------
A spelling must pass ``el_lexicon.orthography_reason`` on every route, must
not be an elided word written without its mark (απ for απ’), and must not be
under ``rejects`` in ``data/el_monotonic_review.json``.

1. ``paradigm``: every form of a Wiktionary lemma that has a form attested
   at least ``ATTESTED_LEMMA_MIN_TOKENS`` times in one frequency source.
   Lemmas and forms come from ``data/mg_pairs.json`` (with its
   ``el_participles.json`` companion, read by
   ``mg_polytonic_paradigms.load_paradigms``, which drops Katharevousa and
   dative cells, table fragments and undeclined foreign names), the
   lemmas' own headwords, and the ``el`` side of ``data/lookup.db`` for
   forms mg_pairs does not have at all (εμείς, συνιστά), as long as a form
   starts like its lemma and is not the unmarked letters of a Wiktionary
   form with a tonos or a dialytika (πλαι for πλάι, σοι for σόι). A form no frequency source has is rated X and
   marked ``mg:generated``. The capitalized forms of a lemma (names, months)
   need their own evidence: one of them written with its capital inside a
   sentence ``NAME_LEMMA_MIN_TOKENS`` times in ``NAME_LEMMA_MIN_DOCS``
   documents of the open corpus. A capitalized form that is no Wiktionary
   name, whose lowercase is a word, and that text writes with its capital
   in less than ``NAME_TWIN_MIN_CAPITAL_SHARE`` of its mid-sentence uses
   (Μέσα, Αυτού, Νέοι) is left out.
2. ``function_word``: the closed list in
   ``data/el_monotonic_function_words.json`` (και, ο, ή, ...).
3. ``corpus_word``: a lowercase word the lexicon lacks (ρωτήσω), with at
   least ``CORPUS_WORD_MIN_SUBTITLE_TOKENS`` tokens in the FrequencyWords
   counts and ``CORPUS_WORD_MIN_OPEN_TOKENS`` lowercase tokens in
   ``CORPUS_WORD_MIN_OPEN_DOCS`` documents of the open corpus, or, for the
   formal words subtitles rarely have (επικροτεί, θεσπιστεί), at least
   ``CORPUS_WORD_FORMAL_MIN_SUBTITLE_TOKENS`` FrequencyWords tokens and
   ``CORPUS_WORD_FORMAL_MIN_OPEN_TOKENS`` open tokens in
   ``CORPUS_WORD_FORMAL_MIN_OPEN_DOCS`` documents. It must
   have exactly one tonos (the monosyllables are a closed class) and be
   written in lowercase in most of its mid-sentence uses. Unless
   Wiktionary has the spelling or ``corpus_accepts`` in the review file
   lists it, no known lowercase word with the same letters
   (``el_lexicon.bare_key``) or the same sound and stressed syllable
   (``el_lexicon.sound_key``) may be far commoner in both the open corpus
   and the FrequencyWords counts (``VARIANT_MIN_TWIN_RATIOS``): those are
   a missing dialytika (προυπόθεση), misplaced accents (έιναι, ποίος) and
   old or wrong spellings (κάθησε, νοιώθεις, βαθειά). A Katharevousa
   ending that sounds like a Modern Greek one (``LEARNED_ENDINGS``: εν
   γνώσει, εν καιρώ, νομικώς) is no misspelling. Nor may a known word one
   typing slip away (a letter left out, added, replaced, or swapped with
   the next one; ``slip_keys``) be ``SLIP_MIN_TWIN_RATIO`` times as common
   in both: those are typing errors (είνα, είανι, πρέπε, φία for για,
   συγκεριμένα) and Katharevousa endings of common words (χώραν, έχειν),
   which a keyboard must not accept as words, since a slip that lands on
   one is then never corrected. Nor is one of ``RUN_TOGETHER_HEADS`` (θα,
   να, από, είναι, ...) written with no space before a common (C) entry or
   a function word of at least ``RUN_TOGETHER_MIN_PART_LETTERS`` letters
   (είναιτο, θαήθελα, απότο).
4. ``name``: a capitalized spelling the lexicon lacks or holds without its
   own evidence, written with its capital inside a sentence at least
   ``NAME_CORPUS_MIN_TOKENS`` times in ``NAME_CORPUS_MIN_DOCS`` documents,
   with a capitalized share of its mid-sentence uses whose 95% Wilson lower
   bound is at least ``NAME_MIN_CAPITAL_SHARE``. When its lowercase is a
   word, the lower bound must be at least
   ``NAME_HOMOGRAPH_MIN_CAPITAL_SHARE`` over the whole corpus, and no more
   of the pools with ``NAME_POOL_MIN_MID_TOKENS`` mid-sentence uses may
   write it without its capital than with it (a pool votes for a spelling
   whose share has a lower bound of ``NAME_HOMOGRAPH_MIN_CAPITAL_SHARE``);
   a spelling that is no form of a Wiktionary name page needs more votes
   for its capital than against, and at least
   ``NAME_MIN_CAPITALIZED_POOLS`` of them (Ρώμη beside ρώμη, Ένωση of
   Ευρωπαϊκή Ένωση beside ένωση). A title or an institution one source
   capitalizes (Υπουργός, Βουλευτή and Αίθουσα in the Parliament minutes)
   is no name. A name is also left out when a
   capitalized spelling of its letters with another accent is
   ``NAME_ACCENT_TWIN_RATIO`` times as frequent inside sentences, or one
   that differs in its dialytika only is ``NAME_DIALYTIKA_TWIN_RATIO``
   times as frequent (Βρετάνια beside Βρετανία, Γουϊνέα beside Γουινέα).
5. ``elision``: a spelling with U+2019 (σ’, απ’, γι’, ’ναι) seen at least
   ``ELISION_MIN_TOKENS`` times in ``ELISION_MIN_DOCS`` documents of the
   open corpus, whose full word is an attested entry (the letters plus a
   final vowel, or a first vowel plus the letters), and which stands
   before a vowel (an elision) or after one (an aphaeresis) in most of its
   uses; see ``elision_reason``. An elision with no tonos is left out when
   an elision of the same letters with a tonos is
   ``ELISION_ACCENT_TWIN_RATIO`` times as common (ολ’ beside όλ’); a pair
   whose unaccented spelling is the common one stays (αλλ’ beside άλλ’).
6. ``reviewed``: the indeclinable loanwords of
   ``data/hunspell_mg_reviewed_loanwords.json`` with at least 50
   FrequencyWords tokens.

Last, an entry no frequency source has (X) is dropped when a C or M entry
has its letters with another accent (αερόπλανα beside αεροπλάνα): a
mistyped accent would otherwise pass as a word nobody writes. And on every
route, a lowercase spelling with no tonos that synizesis makes a
monosyllable (τρια, πλοιο, the accent-stripped aliases of lookup.db) must
be Wiktionary's own (μια, δυο, ποιος, γεια).

Two more passes run over the finished list. A rare entry is dropped when an
entry with its letters and another accent or dialytika is much commoner: a
lowercase entry rated R or X, in fewer than ``ACCENT_TWIN_MAX_DOCS`` open
documents, whose twin is ``ACCENT_TWIN_RATIO`` times as common in the open
corpus and in the FrequencyWords counts (καλήμερα beside καλημέρα; πότε
beside ποτέ stays), and a capitalized entry whose capitalized twin is
``ACCENT_TWIN_CAPITAL_RATIO`` times as frequent inside sentences (Σαββάτο
beside Σάββατο). Function words, elisions and reviewed loanwords are never
dropped, nor a form of the twin's own Wiktionary lemma (ακούσαν beside
άκουσαν, Αύγουστου beside Αυγούστου) or of a lemma with a form rated C or
M (τροπών beside τρόπων). And a paradigm form of at most ``SHORT_FORM_MAX_LETTERS`` letters
rated R or X, in fewer than ``SHORT_FORM_MAX_DOCS`` open documents, is
dropped when a word whose letters it spells with the same sound or with one
key replaced by its neighbor is ``SLIP_MIN_TWIN_RATIO`` times as common in
both the open corpus and the FrequencyWords counts (μάι beside με, ξάι
beside και, άε beside σε); see ``short_slips_to_drop``.

Frequency ratings
-----------------
The rating of a form is the highest of its rates per million words in the
frequency sources (``el_lexicon.bucket_for_rate``: C >= 37, M >= 3.7, R
below, X none). The frequency sources are the FrequencyWords counts
(``data/mg_freq.txt``, OpenSubtitles2018, CC BY-SA 4.0, counts only) and
each source of the open corpus (``build/el_corpus``, the training split
only, so dev text stays held out) with at least ``MIN_RATING_SOURCE_WORDS``
words; the smaller sources are pooled into one. FrequencyWords lowercases
and strips apostrophes, so it rates lowercase words only, and a name only
when its lowercase is no lowercase word. In the open corpus, a capital at
the start of a sentence is shared between a word and its capitalized twin
in proportion to their mid-sentence uses. An open-corpus rate counts for C
only when the spelling is in ``RATING_C_MIN_DOCS`` of the pool's documents,
and for M only in ``RATING_M_MIN_DOCS``; otherwise it is held just under
that edge. A pool of fewer than ``RATING_C_CONFIRM_MIN_WORDS`` words (the
Wikisource and EELLAK texts, the pooled small sources) rates a form C only
when a larger source does too; otherwise its rate is held just under the C
edge, so older literary forms (ψυχήν, όλην) are not rated C on a few
hundred tokens.

Companion lists
---------------
Abbreviations (π.χ., κ.λπ., δισ., σελ.) are the tokens the corpus
tokenizer kept with their period inside sentences, seen in enough
documents, and written that way rather than without the period; plus the
words the tokenizer did not know that a lowercase word follows after a
period at least ``ABBREVIATION_MIN_SHARE`` of the time. A spelling that
differs from a listed one in case is listed too when it has evidence of
its own (π.Χ., before Christ, beside π.χ., for example). Acronyms are
all-capital Greek tokens of two or more letters that open text writes in
capitals (95% Wilson lower bound of the share at least
``ACRONYM_MIN_SHARE``) and whose letters are no entry of the word list,
unless the acronym is ``ACRONYM_OVER_WORD_RATIO`` times as frequent as the
entry's letters written otherwise (ΣΥΡΙΖΑ beside σύριζα).
Two words run together in capitals (ΚΕΙΜΕΝΟΣΕΛΙΔΑ, ΔΙΑΤΑΞΗΤΗΣ) are no
acronym: a token of at least ``ACRONYM_RUN_TOGETHER_MIN_LETTERS`` letters
whose letters split into two rated (C or M) entries, the first of at least
``ACRONYM_RUN_TOGETHER_MIN_HEAD`` letters, is left out.

Usage
-----
    python export_el_monotonic.py                    # the shipping list
    python export_el_monotonic.py --out-dir DIR      # anywhere else
    python export_el_monotonic.py --max-lemmas 5000 --out-dir DIR
                                                     # a quick sanity run

The corpus must have been extracted first (``extract_el_monotonic_lm.py``);
its type counts are counted on first use into
``build/el_monotonic/type_counts`` and reused until the corpus changes.
A run with ``--max-lemmas`` is refused in ``build/hunspell``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sqlite3
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from dilemma_build import el_lexicon as L
from dilemma_build import el_type_counts as TC
from dilemma_build import el_text as T

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "build" / "hunspell"
DIC_NAME = "el_monotonic"
VARIANT = "el"

MG_PAIRS = DATA / "mg_pairs.json"
LOOKUP_DB = DATA / "lookup.db"
MG_FREQ = DATA / "mg_freq.txt"
FUNCTION_WORDS = DATA / "el_monotonic_function_words.json"
REVIEW = DATA / "el_monotonic_review.json"
REVIEWED_LOANWORDS = DATA / "hunspell_mg_reviewed_loanwords.json"
CORPUS_MANIFEST = DATA / "el_lm_corpora.json"
HF_MANIFEST = DATA / "hf_manifest.json"
DEFAULT_CORPUS = ROOT / "build" / "el_corpus"

ATTESTED_LEMMA_MIN_TOKENS = 3
MIN_RATING_SOURCE_WORDS = 1_000_000
# A rate from an open-corpus pool counts for C only when the spelling is in
# this many of the pool's documents, and for M only in this many; otherwise
# it is held just under that edge. One novel that repeats a character's
# name (Κατίγκω, Δυσσέας) or one article that repeats a term is no evidence
# that a word is common.
RATING_C_MIN_DOCS = 10
RATING_M_MIN_DOCS = 3
# A pool of fewer words than this (monotonic Wikisource, the EELLAK
# articles, the small sources pooled) rates a spelling at most M by itself:
# its rate counts for C only when FrequencyWords or a larger pool also rates
# the spelling C. About 80 tokens in 10 documents of the 2.2M words of
# Wikisource would otherwise make an older or poetic form common (ψυχήν,
# όλην, κάθουνταν).
RATING_C_CONFIRM_MIN_WORDS = 10_000_000
SMALL_POOL = "open_small"
SUBTITLES = "subtitles"
CORPUS_WORD_MIN_SUBTITLE_TOKENS = 50
CORPUS_WORD_MIN_OPEN_TOKENS = 5
CORPUS_WORD_MIN_OPEN_DOCS = 2
# A word that is rarer in FrequencyWords, as formal words are (επικροτεί,
# θεσπιστεί), qualifies with wider spread in the open corpus.
CORPUS_WORD_FORMAL_MIN_SUBTITLE_TOKENS = 3
CORPUS_WORD_FORMAL_MIN_OPEN_TOKENS = 20
CORPUS_WORD_FORMAL_MIN_OPEN_DOCS = 10
# Subtitles cannot attest every technical or formal word. Without their
# evidence, require substantial spread in at least two independent sources.
CORPUS_WORD_OPEN_MIN_TOKENS = 100
CORPUS_WORD_OPEN_MIN_DOCS = 20
CORPUS_WORD_OPEN_MIN_SOURCES = 2
CORPUS_WORD_OPEN_MIN_SOURCE_TOKENS = 5
CORPUS_WORD_OPEN_MIN_SOURCE_DOCS = 5
NAME_LEMMA_MIN_TOKENS = 3
NAME_LEMMA_MIN_DOCS = 2
NAME_CORPUS_MIN_TOKENS = 20
NAME_CORPUS_MIN_DOCS = 3
NAME_MIN_CAPITAL_SHARE = 0.9
# A capitalized spelling whose lowercase is a word (Ρώμη, ρώμη; Σοφία,
# σοφία) is a name when the 95% Wilson lower bound of its capitalized share
# is at least this (and see NAME_POOL_MIN_MID_TOKENS).
NAME_HOMOGRAPH_MIN_CAPITAL_SHARE = 0.75
# A capitalized form of a lemma that is no Wiktionary name, whose lowercase
# is a word, and that is written with its capital in fewer than this share
# of at least NAME_TWIN_MIN_MID_TOKENS mid-sentence uses (Μέσα, Αυτού,
# Νέοι) is left out: Wiktionary has the capitalized sense, but text writes
# the lowercase word.
NAME_TWIN_MIN_CAPITAL_SHARE = 0.25
NAME_TWIN_MIN_MID_TOKENS = 20
# A capitalized spelling whose lowercase is a word is a name only when no
# more pools write it without its capital than with it, a pool with at
# least this many mid-sentence uses voting for a spelling whose share has a
# Wilson lower bound of NAME_HOMOGRAPH_MIN_CAPITAL_SHARE; a spelling that is
# no form of a Wiktionary name page needs more votes for the capital than
# against it, and at least NAME_MIN_CAPITALIZED_POOLS of them. Parliament minutes capitalize titles (Βουλευτή,
# Αίθουσα) that other text writes in lowercase, while Ένωση (Ευρωπαϊκή
# Ένωση) and Εθνών (Ηνωμένων Εθνών) are capitalized nearly everywhere.
NAME_POOL_MIN_MID_TOKENS = 20
NAME_MIN_CAPITALIZED_POOLS = 2
# A corpus name whose letters a capitalized spelling with another accent
# has this many times as often inside sentences is a misspelling of it
# (Προέδρε, Βρετάνια, Ιτάλια); with only another dialytika, this many
# (Γουϊνέα, Αχαΐα written Αχαία). Transcriptions of foreign names with the
# stress elsewhere (Μάρια, Ίσαακ, Άντρεα) are seldom 100 times rarer.
NAME_ACCENT_TWIN_RATIO = 100.0
NAME_DIALYTIKA_TWIN_RATIO = 4.0
# A rare entry whose letters a much commoner spelling has with another
# accent (καλήμερα beside καλημέρα) is dropped when its own evidence is
# weak. Lowercase, rated R or X: the twin is this many times as common in
# lowercase in the open corpus and in FrequencyWords, and the entry is in
# fewer than ACCENT_TWIN_MAX_DOCS documents. Capitalized: the twin is
# ACCENT_TWIN_CAPITAL_RATIO times as common written with its capital inside
# sentences (Σαββάτο beside Σάββατο, Ιούλιου beside Ιουλίου). Genuine pairs
# have spread in open text (ποτέ and πότε, νόμου and νομού, έκτος and
# εκτός, Αθηνά and Αθήνα) and stay.
ACCENT_TWIN_RATIO = 50.0
ACCENT_TWIN_MAX_DOCS = 10
ACCENT_TWIN_CAPITAL_RATIO = 40.0
# A corpus word that is one of these words run together with a common
# entry or a function word of at least RUN_TOGETHER_MIN_PART_LETTERS
# letters (είναιτο, θαήθελα, απότο, πρέπεινα). Only these heads: other
# splits are mostly learned compounds and endings (διαφόρους, ανέργου,
# θεωρείτο).
RUN_TOGETHER_HEADS = ("θα", "να", "δεν", "που", "από", "είναι", "πρέπει",
                      "μπορεί", "αυτό", "κατά", "για")
RUN_TOGETHER_MIN_PART_LETTERS = 2
# Rare short paradigm forms (μάι, ξάι, άε), in fewer than
# SHORT_FORM_MAX_DOCS open documents, one typing slip from a word
# SLIP_MIN_TWIN_RATIO times as common catch slips of it (με, και, σε).
SHORT_FORM_MAX_LETTERS = 3
SHORT_FORM_MAX_DOCS = 10
ELISION_MIN_TOKENS = 10
ELISION_MIN_DOCS = 3
# An elision without a tonos (ολ’) whose letters an admitted elision has
# with one (όλ’) at least this many times as often is the accented one
# written without its tonos; the genuine pairs have the unaccented one far
# commoner (κατ’ and κάτ’, παρ’ and πάρ’, καθ’ and κάθ’).
ELISION_ACCENT_TWIN_RATIO = 2.0
ELISION_MIN_NEXT_VOWEL_SHARE = 0.5
APHAERESIS_MIN_PREV_VOWEL_SHARE = 0.7
# A corpus word that is a variant spelling of a known word is rejected when
# the known word is this many times as common in the open corpus and in the
# FrequencyWords counts: (open corpus, FrequencyWords) per kind of variant.
VARIANT_MIN_TWIN_RATIOS = {
    "dialytika_variant_of_a_known_word": (4.0, 4.0),
    "accent_variant_of_a_known_word": (20.0, 4.0),
    "spelling_variant_of_a_known_word": (4.0, 4.0),
}
# Katharevousa endings that sound like a Modern Greek ending of the same
# stem (εν γνώσει, εν καιρώ, νομικώς): (corpus word ending, known word
# ending). A spelling variant that differs only there is not rejected.
LEARNED_ENDINGS = (("ει", "η"), ("ώ", "ό"), ("ω", "ο"), ("ώς", "ός"))
# A corpus word one typing slip from a known word this many times as common
# in the open corpus and in the FrequencyWords counts is a slip of it.
# Measured on the 2026-11 export: at 300 the rule rejects 227 corpus words,
# nearly all misspellings or Katharevousa endings; the real Modern Greek
# words it caught are listed under corpus_accepts in the review file.
SLIP_MIN_TWIN_RATIO = 300.0
SLIP_LETTERS = "αβγδεζηθικλμνξοπρστυφχψω"
REVIEWED_LOANWORD_MIN_TOKENS = 50
ABBREVIATION_MIN_TOKENS = 10
ABBREVIATION_MIN_DOCS = 3
ABBREVIATION_MIN_SHARE = 0.5
ABBREVIATION_CANDIDATE_MIN_TOKENS = 20
ACRONYM_MIN_TOKENS = 20
ACRONYM_MIN_DOCS = 5
ACRONYM_MIN_SHARE = 0.9
ACRONYM_RUN_TOGETHER_MIN_LETTERS = 9
ACRONYM_RUN_TOGETHER_MIN_HEAD = 5
# An acronym whose letters are an entry stays when its capitals are at
# least this many times as frequent (per million words) as the entry's
# letters written otherwise, in the open corpus and in the subtitles
# (ΣΥΡΙΖΑ beside σύριζα, ΝΑΤΟ beside Νάτο); otherwise the entry wins (ΠΕ
# beside πε), so that a word typed in capitals still gets its accent.
ACRONYM_OVER_WORD_RATIO = 10.0
ACRONYM_MIN_CAPITALIZED_SOURCES = 2
# Bytes of the keyboard's compiled dictionary (TONOSDIC v5): a 56-byte
# header, three u32 tables, the raw and stripped string pools and one
# frequency byte per entry.
COMPILED_HEADER_BYTES = 56
AUDIT_LIST_LIMIT = 300

ROUTES = ("paradigm", "function_word", "corpus_word", "name", "elision",
          "reviewed")


class ExportError(RuntimeError):
    """A missing or mismatched input."""


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_hf_pins(paths: dict[str, Path], digests: dict[str, str],
                  manifest_path: Path = HF_MANIFEST) -> list[str]:
    """Inputs whose sha256 differs from the pin in ``data/hf_manifest.json``
    (the version CI downloads). Inputs the manifest does not track pass."""
    if not manifest_path.exists():
        return []
    files = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", {})
    out = []
    for name, path in paths.items():
        try:
            rel = str(Path(path).resolve().relative_to(ROOT.resolve()))
        except ValueError:
            rel = str(path)
        rel = rel if rel in files else f"data/{Path(path).name}"
        pin = files.get(rel)
        if pin and digests.get(name) and digests[name] != pin["sha256"]:
            out.append(f"{rel}: sha256 {digests[name][:12]} is not the pinned "
                       f"{pin['sha256'][:12]}")
    return out


def load_subtitle_counts(path: Path) -> dict[str, int]:
    """FrequencyWords counts: lowercase form -> tokens."""
    out: dict[str, int] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rsplit(" ", 1)
            if len(parts) == 2 and parts[1].strip().isdigit():
                form = unicodedata.normalize("NFC", parts[0])
                out[form] = out.get(form, 0) + int(parts[1])
    return out


def load_function_words(path: Path) -> dict[str, str]:
    """``{word: category}`` of the closed list."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ExportError(f"{path}: unsupported schema")
    out: dict[str, str] = {}
    for category, words in data["categories"].items():
        for word in words:
            out.setdefault(unicodedata.normalize("NFC", word), category)
    return out


def load_review(path: Path) -> dict[str, dict]:
    """``{"rejects": {form: reason}, "corpus_accepts": ...,
    "elision_accepts": ..., "abbreviation_rejects": ...,
    "acronym_rejects": ..., "acronym_accepts": ..., "preferred_spellings": ...}``.
    Preferred spellings retain their usage and source records.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ExportError(f"{path}: unsupported schema")
    out = {}
    for key in ("rejects", "corpus_accepts", "elision_accepts",
                "abbreviation_rejects", "acronym_rejects", "acronym_accepts"):
        out[key] = {unicodedata.normalize("NFC", e["form"]): e["reason"]
                    for e in data.get(key, [])}
    out["preferred_spellings"] = {}
    for entry in data.get("preferred_spellings", []):
        form = unicodedata.normalize("NFC", entry["form"])
        preferred = unicodedata.normalize("NFC", entry["preferred"])
        if (not form or form == preferred or not entry.get("reason")
                or not entry.get("sources")
                or entry.get("usage") not in {"dialectal", "rare", "archaic_spelling"}
                or L.orthography_reason(preferred) is not None):
            raise ExportError(f"{path}: invalid preferred spelling for {form!r}")
        out["preferred_spellings"][form] = {
            **entry, "form": form, "preferred": preferred}
    return out


def load_reviewed_loanwords(path: Path) -> list[str]:
    if not Path(path).exists():
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [unicodedata.normalize("NFC", e["form"]) for e in data.get("entries", [])
            if e.get("language") == "el"]


@dataclass
class Lexicon:
    """The Wiktionary lexicon, grouped by lemma."""

    groups: dict[str, dict[str, str]]       # lemma key -> {form: origin}
    raw_forms: set[str]                     # every mg_pairs form, any cell
    origins: Counter = field(default_factory=Counter)
    # capitalized forms of Wiktionary's name pages (pos "name")
    name_forms: set[str] = field(default_factory=set)

    def forms(self) -> set[str]:
        return {f for forms in self.groups.values() for f in forms}


def _has_marks(form: str) -> bool:
    """True when ``form`` has a tonos or a dialytika."""
    return L.bare_key(form) != L.lower_el(form).replace("ς", "σ")


def _initials(form: str) -> set[str]:
    """The first letter of a form, and the letter after a past tense's
    augment (ε, η, ει before a consonant)."""
    b = L.bare_key(form)
    out = {b[:1]}
    for augment in ("ει", "ε", "η"):
        if (b.startswith(augment) and len(b) > len(augment) + 1
                and b[len(augment)] not in L.VOWELS):
            out.add(b[len(augment)])
    return out


def load_lexicon(mg_pairs: Path = MG_PAIRS, lookup_db: Path | None = LOOKUP_DB,
                 *, max_lemmas: int | None = None, log=print) -> Lexicon:
    """Read the Wiktionary paradigms, their headwords and the lookup.db
    forms mg_pairs lacks (see the module docstring, route 1)."""
    import mg_polytonic_paradigms as M

    t0 = time.time()
    with open(mg_pairs, encoding="utf-8") as fh:
        rows = json.load(fh)
    raw_forms: set[str] = set()
    name_forms: set[str] = set()
    headwords: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        form = unicodedata.normalize("NFC", row.get("form") or "")
        if form:
            raw_forms.add(form)
            if row.get("pos") == "name" and L.case_shape(form) == "title":
                name_forms.add(form)
        lemma = unicodedata.normalize("NFC", row.get("lemma") or "")
        if lemma and (row.get("pos") or "") not in M.EXCLUDED_POS:
            headwords[lemma.lower()].add(lemma)
    del rows
    paradigms = M.load_paradigms(Path(mg_pairs))
    keys = sorted(paradigms)
    if max_lemmas is not None:
        keys = keys[:max_lemmas]
    groups: dict[str, dict[str, str]] = {}
    origins: Counter = Counter()
    for key in keys:
        forms = {f: "wiktionary" for f in paradigms[key]}
        has_title = any(L.case_shape(f) == "title" for f in forms)
        for head in sorted(headwords.get(key, ())):
            shape = L.case_shape(head)
            if shape == "lower" or (shape == "title" and not has_title):
                forms.setdefault(head.lower(), "headword")
            elif shape == "title":
                forms.setdefault(head, "headword")
        groups[key] = forms
    if lookup_db is not None and Path(lookup_db).exists():
        wanted = set(keys) if max_lemmas is not None else None
        # The letters of Wiktionary forms with a tonos or a dialytika: an
        # unmarked lookup.db form with the same letters is the stripped
        # alias of one, even when the stripping makes a monosyllable that
        # passes the spelling check (πλαι for πλάι, σοι for σόι, φαι for
        # φαΐ).
        marked_keys = {L.bare_key(f) for f in raw_forms if _has_marks(f)}
        con = sqlite3.connect(f"file:{Path(lookup_db).resolve()}?mode=ro", uri=True)
        try:
            query = ("SELECT l.form, m.text FROM lookup l JOIN lemmas m "
                     "ON m.id = l.lemma_id WHERE l.src = 'el'")
            for form, lemma in con.execute(query):
                form = unicodedata.normalize("NFC", form)
                lemma = unicodedata.normalize("NFC", lemma)
                if form in raw_forms or L.case_shape(lemma) != "lower":
                    continue
                if L.case_shape(form) != "lower":
                    continue        # a capitalized alias of a common word
                if not _has_marks(form) and L.bare_key(form) in marked_keys:
                    continue        # a stripped alias (πλαι for πλάι)
                key = lemma
                if wanted is not None and key not in wanted:
                    continue
                if L.bare_key(lemma)[:1] not in _initials(form):
                    continue        # a stray row (ια under φαρδύς)
                group = groups.setdefault(key, {})
                if form not in group:
                    group[form] = "lookup"
        finally:
            con.close()
    for forms in groups.values():
        origins.update(forms.values())
    if log:
        log(f"  lexicon: {len(groups):,} lemmas, "
            f"{sum(len(g) for g in groups.values()):,} forms "
            f"({', '.join(f'{k} {v:,}' for k, v in sorted(origins.items()))}), "
            f"{time.time() - t0:.0f} s")
    return Lexicon(groups, raw_forms, origins, name_forms)


# --------------------------------------------------------------------------
# Frequency evidence
# --------------------------------------------------------------------------

def titlecase(form: str) -> str:
    """The form with its first letter capitalized (’ναι -> ’Ναι)."""
    for k, c in enumerate(form):
        if c in L.LETTERS:
            return form[:k] + c.upper() + form[k + 1:]
    return form


@dataclass
class Pool:
    """One open frequency source: one corpus source, or the small ones
    together."""

    name: str
    members: list[str]
    words: int
    count: Counter
    initial: Counter
    docs: Counter
    context: dict = field(default_factory=dict)


def doc_capped_rate(rate: float, docs: int) -> float:
    """``rate`` held just under the C edge when the spelling is in fewer
    than ``RATING_C_MIN_DOCS`` documents of the pool, and just under the M
    edge when it is in fewer than ``RATING_M_MIN_DOCS``."""
    if rate >= L.C_PER_MILLION and docs < RATING_C_MIN_DOCS:
        rate = math.nextafter(L.C_PER_MILLION, 0.0)
    if rate >= L.M_PER_MILLION and docs < RATING_M_MIN_DOCS:
        rate = math.nextafter(L.M_PER_MILLION, 0.0)
    return rate


class Evidence:
    """Counts of every spelling in every frequency source."""

    def __init__(self, subtitles: dict[str, int], tables: list[TC.TypeCounts],
                 min_source_words: int = MIN_RATING_SOURCE_WORDS,
                 c_confirm_min_words: int = RATING_C_CONFIRM_MIN_WORDS):
        self.subtitles = subtitles
        self.c_confirm_min_words = c_confirm_min_words
        self.subtitle_words = sum(subtitles.values())
        self.tables = tables
        big = [t for t in tables if t.words >= min_source_words]
        small = [t for t in tables if t.words < min_source_words]
        self.pools: list[Pool] = [
            Pool(t.source, [t.source], t.words, t.count, t.initial, t.docs,
                 t.context)
            for t in big]
        if small:
            merged = TC.TypeCounts(SMALL_POOL, "train")
            for t in small:
                merged.merge(t)
            self.pools.append(Pool(SMALL_POOL, [t.source for t in small],
                                   merged.words, merged.count, merged.initial,
                                   merged.docs, merged.context))
        self.open_words = sum(t.words for t in tables)

    # -- open corpus, per pool ------------------------------------------

    def _split(self, pool: Pool, low: str) -> tuple[float, float, int, int]:
        """(lowercase folded count, capitalized folded count, lowercase
        docs, capitalized docs) of a lowercase spelling and its twin with a
        capital initial, in one pool."""
        title = titlecase(low)
        n_low = pool.count.get(low, 0)
        n_title = pool.count.get(title, 0) if title != low else 0
        if not n_low and not n_title:
            return 0.0, 0.0, 0, 0
        ini_low = pool.initial.get(low, 0)
        ini_title = pool.initial.get(title, 0) if title != low else 0
        mid_low = n_low - ini_low
        mid_title = n_title - ini_title
        share = (mid_low + 1) / (mid_low + mid_title + 2)
        return (n_low + ini_title * share, mid_title + ini_title * (1 - share),
                pool.docs.get(low, 0), pool.docs.get(title, 0) if title != low else 0)

    def lower_counts(self, low: str) -> dict[str, float]:
        return {p.name: self._split(p, low)[0] for p in self.pools}

    def title_counts(self, title: str) -> dict[str, float]:
        low = L.lower_el(title)
        return {p.name: self._split(p, low)[1] for p in self.pools}

    def exact(self, token: str) -> tuple[int, int, int]:
        """(tokens, mid-sentence tokens, documents) of an exact spelling,
        summed over the open corpus."""
        n = sum(p.count.get(token, 0) for p in self.pools)
        ini = sum(p.initial.get(token, 0) for p in self.pools)
        docs = sum(p.docs.get(token, 0) for p in self.pools)
        return n, n - ini, docs

    def neighbors(self, form: str) -> list[int]:
        """[next_word, next_vowel, next_clitic, prev_word, prev_vowel] of a
        spelling with U+2019 (``el_type_counts``), lowercase and capitalized
        together."""
        out = [0, 0, 0, 0, 0]
        spellings = {form, titlecase(form)}
        for pool in self.pools:
            for spelling in spellings:
                for k, v in enumerate(pool.context.get(spelling, ())):
                    out[k] += v
        return out

    def capital_share(self, title: str) -> tuple[int, int]:
        """(mid-sentence capitalized tokens, mid-sentence tokens in either
        case) of a capitalized spelling."""
        low = L.lower_el(title)
        cap = sum(p.count.get(title, 0) - p.initial.get(title, 0) for p in self.pools)
        lower = sum(p.count.get(low, 0) - p.initial.get(low, 0) for p in self.pools)
        return cap, cap + lower

    # -- ratings ----------------------------------------------------------

    def rates(self, form: str, *, subtitles: bool = True) -> dict[str, float]:
        """Rates per million words of ``form`` in each frequency source,
        each open-corpus rate held under the C or M edge when the spelling
        is in too few of the pool's documents (``doc_capped_rate``), and
        the rate of a pool smaller than ``RATING_C_CONFIRM_MIN_WORDS`` held
        under the C edge unless a larger source also reaches it."""
        title = L.case_shape(form.replace(L.APOSTROPHE, "")) == "title"
        low = L.lower_el(form) if title else form
        out = {}
        for p in self.pools:
            n_low, n_title, d_low, d_title = self._split(p, low)
            n, docs = (n_title, d_title) if title else (n_low, d_low)
            out[p.name] = doc_capped_rate(L.rate_per_million(n, p.words), docs)
        if subtitles and L.APOSTROPHE not in form:
            out[SUBTITLES] = L.rate_per_million(self.subtitles.get(low, 0),
                                                self.subtitle_words)
        small = [p.name for p in self.pools if p.words < self.c_confirm_min_words]
        confirmed = any(rate >= L.C_PER_MILLION for name, rate in out.items()
                        if name not in small)
        if not confirmed:
            for name in small:
                if out[name] >= L.C_PER_MILLION:
                    out[name] = math.nextafter(L.C_PER_MILLION, 0.0)
        # FrequencyWords folds names to lowercase, while sentence-example
        # collections repeatedly use a few names. Neither alone establishes
        # that a capitalized name is C in general text.
        if title and not any(out[p.name] >= L.C_PER_MILLION
                             for p in self.pools
                             if p.words >= self.c_confirm_min_words):
            for name in out:
                if out[name] >= L.C_PER_MILLION:
                    out[name] = math.nextafter(L.C_PER_MILLION, 0.0)
        return out

    def raw_counts(self, form: str, *, subtitles: bool = True) -> dict[str, float]:
        if L.case_shape(form.replace(L.APOSTROPHE, "")) == "title":
            counts = self.title_counts(form)
            low = L.lower_el(form)
        else:
            counts = self.lower_counts(form)
            low = form
        if subtitles and L.APOSTROPHE not in form:
            counts[SUBTITLES] = self.subtitles.get(low, 0)
        return counts

    def _words(self, name: str) -> int:
        for p in self.pools:
            if p.name == name:
                return p.words
        return self.subtitle_words

    def rating(self, form: str, *, subtitles: bool = True) -> tuple[str, str, float]:
        """(bucket, winning source, best rate)."""
        rates = self.rates(form, subtitles=subtitles)
        best = max(sorted(rates), key=lambda k: rates[k]) if rates else ""
        rate = rates.get(best, 0.0)
        return L.bucket_for_rate(rate), (best if rate > 0 else ""), rate

    def frequency_sources(self) -> dict[str, dict]:
        out = {SUBTITLES: {"words": self.subtitle_words,
                           "members": ["FrequencyWords el 2018 (OpenSubtitles2018)"]}}
        for p in self.pools:
            out[p.name] = {"words": p.words, "members": p.members}
        return out


def is_elision_fragment(form: str, evidence: Evidence) -> bool:
    """True for the letters of an elided word written without its mark (απ
    for απ’, χει for ’χει), which FrequencyWords counts for the elided word
    because it strips apostrophes: the open corpus writes the form with
    the mark at least as often as without."""
    if L.APOSTROPHE in form or L.case_shape(form) != "lower":
        return False
    plain = evidence.exact(form)[0] + evidence.exact(titlecase(form))[0]
    marked = 0
    for spelling in (form + L.APOSTROPHE, L.APOSTROPHE + form):
        marked += evidence.exact(spelling)[0]
        marked += evidence.exact(titlecase(spelling))[0]
    return marked > 0 and marked >= plain


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

@dataclass
class Entry:
    form: str
    route: str
    bucket: str
    source: str
    generated: bool = False
    lemma: str = ""
    accent_alternatives: tuple[str, ...] = ()
    spelling_alternatives: tuple[str, ...] = ()
    usage: str = ""


@dataclass
class Selection:
    entries: dict[str, Entry] = field(default_factory=dict)
    rejected: Counter = field(default_factory=Counter)
    rejected_examples: dict[str, list] = field(default_factory=lambda: defaultdict(list))
    audit: dict = field(default_factory=dict)

    def add(self, form: str, route: str, evidence: Evidence, *, lemma: str = "",
            subtitles: bool = True, paradigm: bool = False) -> Entry:
        if form in self.entries:
            return self.entries[form]
        bucket, source, _ = evidence.rating(form, subtitles=subtitles)
        entry = Entry(form, route, bucket, source,
                      generated=paradigm and bucket == "X", lemma=lemma)
        self.entries[form] = entry
        return entry

    def reject(self, form: str, reason: str, weight: float = 0) -> None:
        self.rejected[reason] += 1
        examples = self.rejected_examples[reason]
        examples.append((weight, form))


def slip_keys(key: str) -> set[str]:
    """Every accent-free key (``el_lexicon.bare_key``) one typing slip
    from ``key``: one letter left out, added or replaced, or two adjacent
    letters swapped."""
    out = set()
    n = len(key)
    for i in range(n):
        out.add(key[:i] + key[i + 1:])
        for c in SLIP_LETTERS:
            if c != key[i]:
                out.add(key[:i] + c + key[i + 1:])
    for i in range(n + 1):
        for c in SLIP_LETTERS:
            out.add(key[:i] + c + key[i:])
    for i in range(n - 1):
        if key[i] != key[i + 1]:
            out.add(key[:i] + key[i + 1] + key[i] + key[i + 2:])
    out.discard(key)
    out.discard("")
    return out


def _spelling_ok(form: str, wiktionary_forms: set[str]) -> str | None:
    """``orthography_reason`` with two exceptions for Wiktionary's own
    spellings:

    * a spelling that synizesis makes a monosyllable keeps its accent when
      Wiktionary has no unaccented spelling of it (ιός, σκιά, τριών; not
      ποιός, μιά, γειά);
    * loanwords and foreign names may carry the accent before the third
      syllable from the end (κάμεραμαν, Λίχτενσταϊν), which the corpus
      routes reject because their misspellings (έυκολο) look the same.
    """
    reason = L.orthography_reason(form)
    if reason == "accented_synizesis":
        twin = unicodedata.normalize(
            "NFC", "".join(c for c in unicodedata.normalize("NFD", form)
                           if c != "\u0301"))
        if twin not in wiktionary_forms:
            return None
    if reason == "accent_too_early":
        return None
    return reason


# The final vowels Modern Greek elides: α, ε, ι, ο, υ (μεταξ’), αι (είν’)
# and ου (όπ’). Not η or ω, so that έπ’ (for επ’) does not pass as έπη and
# εδ’ (a numbering mark) as εδώ.
ELIDED_VOWELS = tuple("αειουάέίόύ") + ("αι", "ου", "ού")
# Before a word that had a rough breathing, an elided τ, π or κ is written
# θ, φ or χ: μεθ’ ημών from μετά, αφ’ ότου from από, εφ’ όσον from επί.
ASPIRATED = {"θ": "τ", "φ": "π", "χ": "κ"}
APHAERESIS_PREFIXES = tuple("αεηιουωάέήίόύώ") + (
    "αι", "ει", "οι", "ου", "αί", "εί", "οί", "ού")


_NO_DIALYTIKA = str.maketrans({"ϊ": "ι", "ϋ": "υ", "ΐ": "ί", "ΰ": "ύ",
                               "Ϊ": "Ι", "Ϋ": "Υ"})


def without_dialytika(form: str) -> str:
    return form.translate(_NO_DIALYTIKA)


def aphaeresis_key(core: str) -> str:
    """How an aphaeresis is matched to what is left of its full word: a
    monosyllable by its bare letters (’κει from εκεί, ’χει from έχει),
    anything longer letter for letter, accent included."""
    return L.bare_key(core) if L.syllable_count(core) <= 1 else core


def elision_index(words: set[str]) -> tuple[set[str], dict[str, str]]:
    """The two lookups :func:`elision_reason` needs from the attested
    lowercase entries: the entries themselves, and the aphaeresis key of
    what is left of each after its first vowel or vowel digraph -> the
    entry. The rest keeps the accent of the word (εκείνο -> κείνο), and has
    none when the dropped vowel had it (ήθελα -> θελα, άδηλος -> δηλος, so
    ’δήλος is no aphaeresis of it)."""
    after_vowels: dict[str, str] = {}
    for word in sorted(words):
        for prefix in APHAERESIS_PREFIXES:
            if word.startswith(prefix) and len(word) > len(prefix):
                rest = word[len(prefix):]
                if L.bare_key(rest[0]) not in L.VOWELS:
                    after_vowels.setdefault(aphaeresis_key(rest), word)
    return set(words), after_vowels


def elision_reason(form: str, attested: set[str], after_vowels: dict[str, str],
                   admitted: set[str], accepts: set[str],
                   neighbors: list[int]) -> str | None:
    """Why an elided spelling may not be listed, or None.

    An elision (σ’, απ’, είν’, μεταξ’) must be an attested word cut before
    its last vowel or vowel digraph (``ELIDED_VOWELS``), letter for letter,
    accent included: είν’ from είναι, απ’ from από, γι’ from για. Its last
    letter may be the aspirated form of the word's consonant (``ASPIRATED``:
    μεθ’ from μετά, αφ’ from από). An
    unaccented elision of two or more syllables must come from a word
    accented on the vowel that was dropped. It may not be a word by itself
    (εξ’, εν’, εκ’: εκ, εξ and εν do not elide), unless ``accepts`` lists
    it, nor a word in ς with a quotation mark after it (τους’, χρονιάς’).
    And it must stand where elision happens: before a word that starts
    with a vowel, or, in colloquial speech, before an article or a weak
    pronoun (πάρ’ το, δώσ’ μου, απ’ τη, ``el_type_counts.CLITICS``). The 95%
    Wilson lower bound of that share of its uses before a word is at least
    ``ELISION_MIN_NEXT_VOWEL_SHARE``. Numbering marks (αβ’, ββ’: the Greek
    numeral sign typed as an apostrophe) stand before anything.

    An aphaeresis (’ναι, ’χει, ’ρθει, ’κει) must be an attested word
    without its first vowel or vowel digraph, and must follow a word that
    ends with a vowel (να ’χει, που ’ναι): the Wilson lower bound of that
    share is at least ``APHAERESIS_MIN_PREV_VOWEL_SHARE``. Opening quotation
    marks folded into apostrophes (‘το σπίτι’, ’και) follow anything.
    ``neighbors`` is ``Evidence.neighbors`` of the form."""
    core = form.strip(L.APOSTROPHE)
    next_word, next_vowel, next_clitic, prev_word, prev_vowel = neighbors
    if form.endswith(L.APOSTROPHE) and not form.startswith(L.APOSTROPHE):
        if form not in accepts and core in admitted:
            return "elision_of_a_word"
        if core.endswith("σ") and len(core) > 1 and core[:-1] + "ς" in admitted:
            return "quotation_mark_after_a_word"
        unaccented = L.tonos_count(core) == 0 and L.syllable_count(core) >= 2
        stems = [core]
        if core[-1] in ASPIRATED:
            stems.append(core[:-1] + ASPIRATED[core[-1]])
        if not any(stem + v in attested and (not unaccented or L.tonos_count(v))
                   for stem in stems for v in ELIDED_VOWELS):
            return "elision_without_full_word"
        if (L.wilson_lower(next_vowel + next_clitic, next_word)
                < ELISION_MIN_NEXT_VOWEL_SHARE):
            return "elision_not_before_a_vowel"
        return None
    if form.startswith(L.APOSTROPHE) and not form.endswith(L.APOSTROPHE):
        if aphaeresis_key(core) not in after_vowels:
            return "elision_without_full_word"
        if L.wilson_lower(prev_vowel, prev_word) < APHAERESIS_MIN_PREV_VOWEL_SHARE:
            return "aphaeresis_not_after_a_vowel"
        return None
    return "elision_without_full_word"


def select(lexicon: Lexicon, evidence: Evidence, *,
           function_words: dict[str, str], review: dict[str, dict],
           reviewed_loanwords: list[str], log=print) -> Selection:
    """Run the routes of the module docstring."""
    sel = Selection()
    rejects = review["rejects"]
    wiktionary_forms = ({f for forms in lexicon.groups.values()
                         for f, origin in forms.items() if origin != "lookup"}
                        | set(function_words) | lexicon.raw_forms)
    audit_routes: dict[str, dict] = {r: {} for r in ROUTES}

    def gate(form: str, lexicon_form: bool = False) -> str | None:
        if form in rejects:
            return "reviewed_reject"
        reason = (_spelling_ok(form, wiktionary_forms) if lexicon_form
                  else L.orthography_reason(form))
        if reason:
            return reason
        if (L.case_shape(form) == "lower" and L.tonos_count(form) == 0
                and L.merged_by_synizesis(form) and form not in wiktionary_forms):
            # A lowercase monosyllable only by synizesis and with no tonos
            # (τρια, πλοιο, γρια): the accent-stripped alias of a word,
            # unless Wiktionary spells it so (μια, δυο, ποιος, γεια). Names
            # transliterated from other languages are often such
            # monosyllables (Πιερ, Γιορκ) and keep theirs.
            return "unaccented_synizesis"
        if is_elision_fragment(form, evidence):
            return "elision_fragment"
        return None

    def weight(form: str) -> float:
        low = L.lower_el(form)
        return evidence.subtitles.get(low, 0) + evidence.exact(form)[0]

    lexicon_words = lexicon.forms() | lexicon.raw_forms

    def capitalized_common_word(form: str) -> bool:
        """A capitalized lexicon form that is no Wiktionary name, whose
        lowercase is a word that text writes far more often (Μέσα, Αυτού,
        Νέοι; see ``NAME_TWIN_MIN_CAPITAL_SHARE``)."""
        if (L.case_shape(form) != "title" or form in lexicon.name_forms
                or L.lower_el(form) not in lexicon_words):
            return False
        cap, total = evidence.capital_share(form)
        return (total >= NAME_TWIN_MIN_MID_TOKENS
                and cap < NAME_TWIN_MIN_CAPITAL_SHARE * total)

    # 1. paradigms of attested lemmas
    t0 = time.time()
    lemmas_attested = names_attested = 0
    for key in sorted(lexicon.groups):
        forms = lexicon.groups[key]
        ok_lower, ok_title = [], []
        for form in sorted(forms):
            reason = gate(form, lexicon_form=True)
            if reason is None and capitalized_common_word(form):
                reason = "capitalized_common_word"
            if reason:
                sel.reject(form, reason, weight(form))
                continue
            (ok_title if L.case_shape(form.replace(L.APOSTROPHE, "")) == "title"
             else ok_lower).append(form)
        if ok_lower and any(
                max(evidence.raw_counts(f).values(), default=0)
                >= ATTESTED_LEMMA_MIN_TOKENS for f in ok_lower):
            lemmas_attested += 1
            for form in ok_lower:
                sel.add(form, "paradigm", evidence, lemma=key, paradigm=True)
        elif ok_lower:
            for form in ok_lower:
                sel.rejected["lemma_not_attested"] += 1
        if ok_title:
            cap = docs = 0
            for form in ok_title:
                n, mid, d = evidence.exact(form)
                cap += mid
                docs += d
            if cap >= NAME_LEMMA_MIN_TOKENS and docs >= NAME_LEMMA_MIN_DOCS:
                names_attested += 1
                for form in ok_title:
                    sel.add(form, "paradigm", evidence, lemma=key, paradigm=True,
                            subtitles=L.lower_el(form) not in lexicon_words)
            else:
                sel.rejected["name_not_attested"] += len(ok_title)
    audit_routes["paradigm"] = {"lemmas_with_lowercase_evidence": lemmas_attested,
                                "lemmas_with_capital_evidence": names_attested}
    if log:
        log(f"  paradigms: {lemmas_attested:,} lemmas attested, "
            f"{names_attested:,} capitalized lemmas, {len(sel.entries):,} forms, "
            f"{time.time() - t0:.0f} s")

    # 2. function words
    missing = []
    for word in sorted(function_words):
        reason = L.orthography_reason(word)
        if reason:
            raise ExportError(f"function word {word!r} fails the spelling check: {reason}")
        if word not in sel.entries:
            missing.append(word)
            sel.add(word, "function_word", evidence)
    audit_routes["function_word"] = {"added": missing}

    # 3. corpus words
    corpus_accepts = set(review.get("corpus_accepts", {}))
    known_bare: dict[str, list[str]] = defaultdict(list)
    known_sound: dict[tuple[str, int | None], list[str]] = defaultdict(list)
    for f in sorted(lexicon_words | set(sel.entries)):
        if L.case_shape(f) == "lower" and L.APOSTROPHE not in f:
            known_bare[L.bare_key(f)].append(f)
            known_sound[(L.sound_key(f), L.accent_position(f))].append(f)

    def variant_reason(form: str) -> str | None:
        """Why ``form`` is a misspelling of a far commoner known twin, if
        it is one (``VARIANT_MIN_TWIN_RATIOS``)."""
        n_open = max(evidence.exact(form)[0], 1)
        n_sub = max(evidence.subtitles.get(form, 0), 1)
        bare = L.bare_key(form)
        plain = without_dialytika(form)
        candidates = [(t, "dialytika_variant_of_a_known_word"
                       if without_dialytika(t) == plain
                       else "accent_variant_of_a_known_word")
                      for t in known_bare.get(bare, [])]
        for t in known_sound.get((L.sound_key(form), L.accent_position(form)), []):
            if L.bare_key(t) == bare:
                continue
            if any(form.endswith(a) and t.endswith(b)
                   and form[:-len(a)] == t[:-len(b)] for a, b in LEARNED_ENDINGS):
                continue
            candidates.append((t, "spelling_variant_of_a_known_word"))
        for twin, reason in candidates:
            if twin == form:
                continue
            open_ratio, sub_ratio = VARIANT_MIN_TWIN_RATIOS[reason]
            if (evidence.exact(twin)[0] >= open_ratio * n_open
                    and evidence.subtitles.get(twin, 0) >= sub_ratio * n_sub):
                return reason
        return None

    def slip_reason(form: str) -> str | None:
        """``slip_of_a_known_word`` when a known word one typing slip
        away is ``SLIP_MIN_TWIN_RATIO`` times as common in both sources."""
        n_open = max(evidence.exact(form)[0], 1)
        n_sub = max(evidence.subtitles.get(form, 0), 1)
        for key in sorted(slip_keys(L.bare_key(form))):
            for twin in known_bare.get(key, ()):
                if (evidence.exact(twin)[0] >= SLIP_MIN_TWIN_RATIO * n_open
                        and evidence.subtitles.get(twin, 0)
                        >= SLIP_MIN_TWIN_RATIO * n_sub):
                    return "slip_of_a_known_word"
        return None

    tails = {f for f, e in sel.entries.items()
             if (e.bucket == "C" or e.route == "function_word")
             and L.case_shape(f) == "lower"
             and len(f) >= RUN_TOGETHER_MIN_PART_LETTERS}

    def run_together_reason(form: str) -> str | None:
        """``run_together_words`` for one of ``RUN_TOGETHER_HEADS`` written
        with no space before a common entry or a function word (είναιτο:
        είναι and το)."""
        for head in RUN_TOGETHER_HEADS:
            if form.startswith(head) and form[len(head):] in tails:
                return "run_together_words"
        return None

    corpus_words = []
    open_candidates = open_corpus_candidates(evidence)
    for form in sorted(set(evidence.subtitles) | open_candidates):
        n = evidence.subtitles.get(form, 0)
        open_only = n < CORPUS_WORD_FORMAL_MIN_SUBTITLE_TOKENS
        if (open_only and form not in open_candidates) or form in sel.entries:
            continue
        if L.case_shape(form) != "lower":
            continue
        tokens, _, docs = evidence.exact(form)
        if open_only:
            pass       # open_corpus_candidates checked counts and source spread
        elif n >= CORPUS_WORD_MIN_SUBTITLE_TOKENS:
            if tokens < CORPUS_WORD_MIN_OPEN_TOKENS or docs < CORPUS_WORD_MIN_OPEN_DOCS:
                continue
        elif (tokens < CORPUS_WORD_FORMAL_MIN_OPEN_TOKENS
              or docs < CORPUS_WORD_FORMAL_MIN_OPEN_DOCS):
            continue
        reason = gate(form)
        if reason is None and open_only and open_corpus_archaic_form(form, known_bare):
            reason = "open_corpus_katharevousa"
        if reason is None and L.tonos_count(form) != 1:
            # Monosyllables are a closed class the lexicon and the function
            # words cover; the corpus ones are fragments (λι, ντι, εντ).
            reason = "corpus_monosyllable"
        if reason is None:
            cap, total = evidence.capital_share(titlecase(form))
            if total and cap * 2 >= total:
                # FrequencyWords lowercases names (τζακ, αμερική).
                reason = "capitalized_in_open_text"
        if (reason is None and form not in lexicon.raw_forms
                and form not in corpus_accepts):
            # Wiktionary's own forms (εν μέρει, εν περιπτώσει) are words even
            # when the paradigm reader dropped them; other spellings are
            # misspellings when a known twin is far commoner: the letters
            # without the dialytika (προυπόθεση) or with the accent elsewhere
            # (έιναι, ποίος), or the sound with the same stressed syllable
            # (κάθησε, νοιώθεις, βαθειά), or one typing slip away (είνα,
            # πρέπε, έχειν).
            reason = (variant_reason(form) or slip_reason(form)
                      or run_together_reason(form))
        if reason:
            sel.reject(form, reason, n + tokens)
            continue
        sel.add(form, "corpus_word", evidence)
        corpus_words.append((n + tokens, form, n, tokens, docs, form in lexicon.raw_forms))
    corpus_words.sort(key=lambda r: (-r[0], r[1]))
    audit_routes["corpus_word"] = {
        "count": len(corpus_words),
        "without_subtitle_evidence": sum(r[2] < CORPUS_WORD_FORMAL_MIN_SUBTITLE_TOKENS
                                         for r in corpus_words),
        "without_subtitle_evidence_examples": [
            {"form": r[1], "subtitles": r[2], "open": r[3], "docs": r[4]}
            for r in corpus_words if r[2] < CORPUS_WORD_FORMAL_MIN_SUBTITLE_TOKENS
        ][:AUDIT_LIST_LIMIT],
        "in_mg_pairs_but_dropped_by_load_paradigms": sum(1 for r in corpus_words if r[5]),
        "top": [{"form": r[1], "subtitles": r[2], "open": r[3], "docs": r[4]}
                for r in corpus_words[:AUDIT_LIST_LIMIT]]}

    # 4. names from the corpus
    names = []
    seen_titles: set[str] = set()
    for pool in evidence.pools:
        for token in pool.count:
            if token in seen_titles:
                continue
            if (L.case_shape(token) != "title" or L.APOSTROPHE in token
                    or token.endswith(".")):
                continue
            seen_titles.add(token)
    titles_by_bare: dict[str, list[str]] = defaultdict(list)
    for token in sorted(seen_titles):
        titles_by_bare[L.bare_key(token)].append(token)

    def capitalized_across_sources(token: str, name_page: bool) -> bool:
        """For a form of a Wiktionary name page, True unless more pools
        write ``token`` without its capital than with it; for any other
        spelling, True when more pools write it with its capital than
        without, and at least ``NAME_MIN_CAPITALIZED_POOLS`` do. A pool with at least
        ``NAME_POOL_MIN_MID_TOKENS`` mid-sentence uses of the two spellings
        votes for the one whose share has a 95% Wilson lower bound of at
        least ``NAME_HOMOGRAPH_MIN_CAPITAL_SHARE``, and abstains when
        neither has."""
        low = L.lower_el(token)
        capital = lowercase = 0
        for pool in evidence.pools:
            cap = pool.count.get(token, 0) - pool.initial.get(token, 0)
            lower = pool.count.get(low, 0) - pool.initial.get(low, 0)
            if cap + lower < NAME_POOL_MIN_MID_TOKENS:
                continue
            if L.wilson_lower(cap, cap + lower) >= NAME_HOMOGRAPH_MIN_CAPITAL_SHARE:
                capital += 1
            elif L.wilson_lower(lower, cap + lower) >= NAME_HOMOGRAPH_MIN_CAPITAL_SHARE:
                lowercase += 1
        if name_page:
            return capital >= lowercase
        return capital >= NAME_MIN_CAPITALIZED_POOLS and capital > lowercase

    def name_variant_reason(token: str) -> str | None:
        """Why a corpus name is a misspelling of a capitalized spelling of
        the same letters that text uses far more often inside sentences
        (``NAME_ACCENT_TWIN_RATIO``, ``NAME_DIALYTIKA_TWIN_RATIO``)."""
        mid = max(evidence.exact(token)[1], 1)
        for twin in titles_by_bare.get(L.bare_key(token), ()):
            if L.lower_el(twin) == L.lower_el(token):
                continue
            dialytika_only = without_dialytika(twin) == without_dialytika(token)
            ratio = NAME_DIALYTIKA_TWIN_RATIO if dialytika_only else NAME_ACCENT_TWIN_RATIO
            if evidence.exact(twin)[1] >= ratio * mid:
                return ("dialytika_variant_of_a_commoner_name" if dialytika_only
                        else "accent_variant_of_a_commoner_name")
        return None

    for token in sorted(seen_titles):
        if token in sel.entries:
            continue
        n, mid, docs = evidence.exact(token)
        if mid < NAME_CORPUS_MIN_TOKENS or docs < NAME_CORPUS_MIN_DOCS:
            continue
        cap, total = evidence.capital_share(token)
        share = L.wilson_lower(cap, total)
        low = L.lower_el(token)
        if low in lexicon_words or low in sel.entries:
            # A common word one source capitalizes as a title or an
            # institution (Υπουργός, Πρόεδρε, Βουλευτή and Αίθουσα in the
            # Parliament minutes, which Wiktionary files as names too) is
            # no name; a spelling that text writes with its capital
            # whatever the source is one (Ρώμη beside ρώμη, Ένωση of
            # Ευρωπαϊκή Ένωση beside ένωση). A spelling of no Wiktionary
            # name page needs that in NAME_MIN_CAPITALIZED_POOLS pools.
            if (share < NAME_HOMOGRAPH_MIN_CAPITAL_SHARE
                    or not capitalized_across_sources(
                        token, token in lexicon.name_forms)):
                if share >= NAME_MIN_CAPITAL_SHARE:
                    sel.reject(token, "capitalized_common_word", mid)
                continue
        elif share < NAME_MIN_CAPITAL_SHARE:
            continue
        reason = gate(token) or name_variant_reason(token)
        if reason:
            sel.reject(token, reason, mid)
            continue
        sel.add(token, "name", evidence,
                subtitles=L.lower_el(token) not in lexicon_words)
        names.append((mid, token, docs, cap, total))
    names.sort(key=lambda r: (-r[0], r[1]))
    audit_routes["name"] = {
        "count": len(names),
        "top": [{"form": r[1], "mid_sentence": r[0], "docs": r[2],
                 "capital_share": round(r[3] / r[4], 4)} for r in names[:AUDIT_LIST_LIMIT]]}

    # 5. elisions
    elided: Counter = Counter()
    elided_docs: Counter = Counter()
    for pool in evidence.pools:
        for token, n in pool.count.items():
            if L.APOSTROPHE not in token or token.endswith("."):
                continue
            low = L.lower_el(token)
            elided[low] += n
            elided_docs[low] += pool.docs.get(token, 0)
    attested_lower = {f for f, e in sel.entries.items()
                      if e.bucket != "X" and L.case_shape(f) == "lower"}
    attested, after_vowels = elision_index(attested_lower)
    admitted = set(sel.entries)
    accepts = set(review.get("elision_accepts", {}))
    elisions = []
    for form in sorted(elided):
        n, docs = elided[form], elided_docs[form]
        if n < ELISION_MIN_TOKENS or docs < ELISION_MIN_DOCS or form in sel.entries:
            continue
        if form in rejects:
            sel.reject(form, "reviewed_reject", n)
            continue
        reason = L.orthography_reason(form)
        if reason == "unaccented_elision":
            reason = None          # elision_reason checks the full word
        if reason is None:
            reason = elision_reason(form, attested, after_vowels, admitted, accepts,
                                    evidence.neighbors(form))
        if reason:
            sel.reject(form, reason, n)
            continue
        sel.add(form, "elision", evidence, subtitles=False)
        elisions.append((n, form, docs))
    # An elision written without the tonos its accented twin carries far
    # more often is a typing slip that would block the accented one (ολ’
    # beside όλ’); a pair whose unaccented spelling is the common one stays
    # (αλλ’ beside άλλ’, καθ’ beside κάθ’).
    elided_entries = [f for f in sel.entries if L.APOSTROPHE in f]
    for n, form, docs in list(elisions):
        core = form.strip(L.APOSTROPHE)
        if (not form.endswith(L.APOSTROPHE) or form.startswith(L.APOSTROPHE)
                or L.tonos_count(core) != 0):
            continue
        if any(L.bare_key(other) == L.bare_key(form) and L.tonos_count(other)
               and elided.get(other, 0) >= ELISION_ACCENT_TWIN_RATIO * n
               for other in elided_entries):
            del sel.entries[form]
            sel.reject(form, "elision_without_the_tonos_of_a_commoner_twin", n)
            elisions.remove((n, form, docs))
    elisions.sort(key=lambda r: (-r[0], r[1]))
    audit_routes["elision"] = {
        "count": len(elisions),
        "all": [{"form": r[1], "open": r[0], "docs": r[2]} for r in elisions]}

    # 6. reviewed loanwords
    added = []
    for form in reviewed_loanwords:
        if form in sel.entries:
            continue
        if evidence.subtitles.get(form, 0) < REVIEWED_LOANWORD_MIN_TOKENS:
            continue
        if gate(form, lexicon_form=True) is None:
            sel.add(form, "reviewed", evidence)
            added.append(form)
    audit_routes["reviewed"] = {"added": added}

    # 7. an unattested form whose letters a common entry has with another
    # accent (αερόπλανα beside αεροπλάνα, αδίκει beside αδικεί) is left out:
    # a mistyped accent would pass as it, and no source has it.
    common_bare: dict[str, str] = {}
    for form, entry in sel.entries.items():
        if entry.bucket in ("C", "M"):
            common_bare.setdefault(L.bare_key(form), form)
    pruned = []
    for form in sorted(sel.entries):
        entry = sel.entries[form]
        twin = common_bare.get(L.bare_key(form))
        if entry.bucket == "X" and twin is not None and twin != form:
            del sel.entries[form]
            sel.reject(form, "unattested_accent_twin_of_a_common_entry")
            pruned.append([form, twin])
    audit_routes["paradigm"]["unattested_accent_twins_removed"] = len(pruned)
    audit_routes["paradigm"]["unattested_accent_twins_examples"] = pruned[:AUDIT_LIST_LIMIT]

    # 8. a rare entry beside a much commoner spelling of its letters with
    # another accent, whose own evidence is weak (καλήμερα beside καλημέρα,
    # Σαββάτο beside Σάββατο); see ACCENT_TWIN_RATIO.
    dropped = accent_twins_to_drop(sel, evidence, set(function_words),
                                   titles_by_bare, lexicon)
    for form, twin in dropped:
        del sel.entries[form]
        sel.reject(form, "rare_accent_twin_of_a_commoner_spelling", weight(form))
    audit_routes["paradigm"]["rare_accent_twins_removed"] = len(dropped)
    audit_routes["paradigm"]["rare_accent_twins_examples"] = [
        list(d) for d in dropped[:AUDIT_LIST_LIMIT]]

    # 9. rare short forms one typing slip from a far commoner word (μάι
    # for με, ξάι for και, άε for σε).
    short = short_slips_to_drop(sel, evidence)
    for form, twin in short:
        del sel.entries[form]
        sel.reject(form, "rare_short_slip_of_a_common_word", weight(form))
    audit_routes["paradigm"]["rare_short_slips_removed"] = len(short)
    audit_routes["paradigm"]["rare_short_slips_examples"] = [
        list(d) for d in short[:AUDIT_LIST_LIMIT]]

    alternatives = accent_alternatives_to_mark(sel, evidence, lexicon)
    for form, twins in alternatives.items():
        sel.entries[form].accent_alternatives = twins
    preferred_audit = apply_preferred_spellings(sel, review.get("preferred_spellings", {}))
    alternatives = {form: entry.accent_alternatives
                    for form, entry in sorted(sel.entries.items())
                    if entry.accent_alternatives}
    audit_routes["paradigm"]["accent_alternatives_marked"] = len(alternatives)
    audit_routes["paradigm"]["accent_alternatives_examples"] = [
        [form, list(twins)] for form, twins in alternatives.items()][:AUDIT_LIST_LIMIT]
    audit_routes["paradigm"]["reviewed_preferred_spellings"] = preferred_audit

    sel.audit["route_details"] = audit_routes
    return sel


def apply_preferred_spellings(sel: Selection, reviewed: dict[str, dict]) -> list[dict]:
    """Annotate real dialect, rare or older spellings without deleting them.

    The target must itself have C/M evidence. Counts and lexical acceptance
    stay unchanged; proposal generators use the preference and usage fields.
    """
    audit = []
    for form, decision in sorted(reviewed.items()):
        if form not in sel.entries:
            continue
        preferred = decision["preferred"]
        row = dict(decision)
        target = sel.entries.get(preferred)
        if target is None or target.bucket not in ("C", "M"):
            row["not_applied"] = "preferred spelling lacks C/M evidence"
        else:
            entry = sel.entries[form]
            entry.usage = decision["usage"]
            field = ("accent_alternatives" if L.bare_key(form) == L.bare_key(preferred)
                     else "spelling_alternatives")
            values = set(getattr(entry, field)) | {preferred}
            setattr(entry, field, tuple(sorted(values, key=lambda s: s.encode("utf-8"))))
        audit.append(row)
    return audit


def open_corpus_archaic_form(form: str, known_bare: dict) -> bool:
    """Restrict the new subtitle-free route without changing corpus splits.

    Formal participles remain useful Modern Greek. Obsolete infinitives and
    personal endings do not qualify on formal-corpus counts alone. Older
    case endings are excluded only when the corresponding Modern spelling
    is already known; this leaves words such as παρόν and προϊόν intact.
    """
    bare = L.bare_key(form)
    if T.is_katharevousa_marker(form) or bare.endswith(
            ("ειν", "εσθαι", "ουσι", "ουσιν", "ωσι", "ωσιν",
             "ομεν", "ομεθα", "ουμεθα", "εσθε")):
        return True
    # A blanket final-ν comparison also matches different verb cells:
    # ρυπαίνων / ρυπαίνω, απώλεσαν / απώλεσα, απάρτιζαν / απάρτιζα.
    if (bare.endswith(("ον", "ην")) or form.endswith("ίαν")) and bare[:-1] in known_bare:
        return True
    for old, modern in (("εως", "ης"), ("οτητος", "οτητας"),
                        ("υτητος", "υτητας"), ("ιδος", "ιδας"),
                        ("αδος", "αδας"), ("σσης", "σσας"), ("σιν", "ση")):
        old, modern = L.bare_key(old), L.bare_key(modern)
        if bare.endswith(old) and bare[:-len(old)] + modern in known_bare:
            return True
    return False


def open_corpus_candidates(evidence: Evidence) -> set[str]:
    """Formal words corroborated independently, without subtitle counts."""
    sources: dict[str, set[str]] = defaultdict(set)
    for table in evidence.tables:
        for form, count in table.count.items():
            if (count >= CORPUS_WORD_OPEN_MIN_SOURCE_TOKENS
                    and table.docs.get(form, 0) >= CORPUS_WORD_OPEN_MIN_SOURCE_DOCS
                    and L.case_shape(form) == "lower"):
                sources[form].add(table.source)
    return {form for form, votes in sources.items()
            if len(votes) >= CORPUS_WORD_OPEN_MIN_SOURCES
            and evidence.exact(form)[0] >= CORPUS_WORD_OPEN_MIN_TOKENS
            and evidence.exact(form)[2] >= CORPUS_WORD_OPEN_MIN_DOCS}


def accent_alternatives_to_mark(sel: Selection, evidence: Evidence,
                                lexicon: Lexicon) -> dict[str, tuple[str, ...]]:
    """Keep valid rare words, but expose much commoner accent alternatives.

    A dictionary head is not proof that the rare reading is a typing error:
    πινάκες is also a plural of πινάκα, and κάνονες of κάνονας. Mark rather
    than delete these entries. Shared paradigm cells (πινάκων) are not
    independent evidence for the rare reading. Genuine grammatical variants
    of one lemma, and distinct lemmas with their own C/M form, stay unmarked.
    """
    lemmas_of: dict[str, set[str]] = defaultdict(set)
    for lemma, forms in lexicon.groups.items():
        for form in forms:
            lemmas_of[form].add(lemma)
    common: dict[str, list[str]] = defaultdict(list)
    for form, entry in sorted(sel.entries.items()):
        if (entry.bucket in ("C", "M") and L.case_shape(form) == "lower"
                and L.APOSTROPHE not in form):
            common[L.bare_key(form)].append(form)
    out = {}
    for form, entry in sorted(sel.entries.items()):
        if (entry.bucket not in ("R", "X") or L.case_shape(form) != "lower"
                or L.APOSTROPHE in form or entry.route in ("function_word", "reviewed")):
            continue
        twins = []
        for twin in common.get(L.bare_key(form), ()):
            if (sum(evidence.lower_counts(twin).values()) + 1
                    < ACCENT_TWIN_RATIO * (sum(evidence.lower_counts(form).values()) + 1)
                    or evidence.subtitles.get(twin, 0) + 1
                    < ACCENT_TWIN_RATIO * (evidence.subtitles.get(form, 0) + 1)):
                continue
            lemmas, twin_lemmas = lemmas_of[form], lemmas_of[twin]
            if lemmas & twin_lemmas:
                continue
            twin_forms = {f for lemma in twin_lemmas for f in lexicon.groups[lemma]}
            if any(other != form and other not in twin_forms and other in sel.entries
                   and sel.entries[other].bucket in ("C", "M")
                   for lemma in lemmas for other in lexicon.groups[lemma]):
                continue
            twins.append(twin)
        if twins:
            out[form] = tuple(sorted(twins, key=lambda s: s.encode("utf-8")))
    return out


def short_slips_to_drop(sel: Selection, evidence: Evidence) -> list[tuple[str, str]]:
    """``(entry, word)`` for every entry rule 9 of :func:`select` drops: a
    paradigm form of at most ``SHORT_FORM_MAX_LETTERS`` letters rated R or
    X, in fewer than ``SHORT_FORM_MAX_DOCS`` open documents, whose bare
    letters are a slip of the bare letters of a word ``SLIP_MIN_TWIN_RATIO``
    times as common in the open corpus and in the FrequencyWords counts.
    The slips are those of the restoration evaluation: the letters of the
    same sound (``el_lexicon.sound_key``: μαι for με) and one key replaced
    by the key beside it on its row (ξαι for και, αε for σε). Dropping or
    adding a letter is no slip here: a short word is one letter from many
    others (ξι beside έξι, ρέω beside ρω)."""
    by_bare: dict[str, list[str]] = defaultdict(list)
    by_sound: dict[str, list[str]] = defaultdict(list)
    for f in sorted(sel.entries):
        if L.case_shape(f) == "lower" and L.APOSTROPHE not in f:
            by_bare[L.bare_key(f)].append(f)
            by_sound[L.sound_key(L.bare_key(f))].append(f)
    out = []
    for form in sorted(sel.entries):
        entry = sel.entries[form]
        if (entry.route != "paradigm" or entry.bucket not in ("R", "X")
                or L.case_shape(form) != "lower" or L.APOSTROPHE in form
                or sum(c in L.LETTERS for c in form) > SHORT_FORM_MAX_LETTERS):
            continue
        n_open, _, docs = evidence.exact(form)
        if docs >= SHORT_FORM_MAX_DOCS:
            continue
        key = L.bare_key(form)
        words = [t for t in by_sound.get(L.sound_key(key), ()) if L.bare_key(t) != key]
        for i, c in enumerate(key):
            for d in sorted(L.NEIGHBOR_KEYS.get(c, ())):
                words += by_bare.get(key[:i] + d + key[i + 1:], [])
        n_open = max(n_open, 1)
        n_sub = max(evidence.subtitles.get(form, 0), 1)
        word = next((t for t in sorted(set(words))
                     if evidence.exact(t)[0] >= SLIP_MIN_TWIN_RATIO * n_open
                     and evidence.subtitles.get(t, 0) >= SLIP_MIN_TWIN_RATIO * n_sub),
                    None)
        if word is not None:
            out.append((form, word))
    return out


def accent_twins_to_drop(sel: Selection, evidence: Evidence, keep: set[str],
                         titles_by_bare: dict[str, list[str]],
                         lexicon: Lexicon) -> list[tuple[str, str]]:
    """``(entry, twin)`` for every entry rule 8 of :func:`select` drops: a
    rare entry whose letters a much commoner spelling has with another
    accent or dialytika, its own evidence weak (``ACCENT_TWIN_RATIO``,
    ``ACCENT_TWIN_MAX_DOCS``, ``ACCENT_TWIN_CAPITAL_RATIO``). Function
    words, elisions and reviewed loanwords are never dropped. Nor is a form
    of the twin's own Wiktionary lemma (the colloquial ακούσαν beside
    άκουσαν, the imperative αναζήτα beside αναζητά, Αύγουστου beside
    Αυγούστου), or a form of a lemma that has a form rated C or M (τροπών,
    of τροπή, beside τρόπων): those are words, not misplaced accents."""
    lower_by_bare: dict[str, list[str]] = defaultdict(list)
    for f in sorted(sel.entries):
        if L.case_shape(f) == "lower" and L.APOSTROPHE not in f:
            lower_by_bare[L.bare_key(f)].append(f)
    lemmas_of: dict[str, set[str]] = defaultdict(set)
    for lemma, forms in lexicon.groups.items():
        for f in forms:
            lemmas_of[f].add(lemma)

    def open_lower(f: str) -> float:
        return sum(evidence.lower_counts(f).values())

    def a_word_of_its_own(form: str, twin: str) -> bool:
        lemmas = lemmas_of.get(form, set())
        if lemmas & lemmas_of.get(twin, set()):
            return True
        return any(other != form and other in sel.entries
                   and sel.entries[other].bucket in ("C", "M")
                   for lemma in lemmas for other in lexicon.groups[lemma])

    out = []
    for form in sorted(sel.entries):
        entry = sel.entries[form]
        if form in keep or L.APOSTROPHE in form or entry.route in ("function_word", "reviewed"):
            continue
        key = L.bare_key(form)
        shape = L.case_shape(form)
        if shape == "lower":
            if entry.bucket not in ("R", "X"):
                continue
            docs = sum(p.docs.get(form, 0) for p in evidence.pools)
            if docs >= ACCENT_TWIN_MAX_DOCS:
                continue
            n_open, n_sub = open_lower(form), evidence.subtitles.get(form, 0)
            for twin in lower_by_bare.get(key, ()):
                if twin == form:
                    continue
                if (open_lower(twin) + 1 >= ACCENT_TWIN_RATIO * (n_open + 1)
                        and evidence.subtitles.get(twin, 0) + 1
                        >= ACCENT_TWIN_RATIO * (n_sub + 1)
                        and not a_word_of_its_own(form, twin)):
                    out.append((form, twin))
                    break
        elif shape == "title":
            mid = max(evidence.exact(form)[1], 1)
            for twin in titles_by_bare.get(key, ()):
                if L.lower_el(twin) == L.lower_el(form):
                    continue
                if (evidence.exact(twin)[1] >= ACCENT_TWIN_CAPITAL_RATIO * mid
                        and not a_word_of_its_own(form, twin)):
                    out.append((form, twin))
                    break
    return out


# --------------------------------------------------------------------------
# Companion lists
# --------------------------------------------------------------------------

def _read_candidates(corpus_dir: Path, sources: list[str]) -> Counter:
    """Words a lowercase word follows after a period, summed over the
    training split of ``sources`` (``abbreviation_candidates.tsv.gz``)."""
    out: Counter = Counter()
    for source in sources:
        path = Path(corpus_dir) / source / "abbreviation_candidates.tsv.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                word, n = line.rstrip("\n").split("\t")
                out[word] += int(n)
    return out


def _has_lowercase(text: str) -> bool:
    return any(c in L.LOWER_LETTERS for c in text)


def abbreviation_shape_reason(spelling: str, words: set[str]) -> str | None:
    """Why a dotted token is no abbreviation by its shape, or None: a
    single letter with a tonos or a final sigma (ά., ς.), or a part of
    three or more letters that is a word (του.Β., λόγο.Απών.: two
    sentences with no space between them)."""
    parts = [part for part in spelling.split(".") if part]
    if len(parts) == 1 and len(parts[0]) == 1:
        if parts[0] in "ς" or L.tonos_count(parts[0]):
            return "a single letter with a mark"
    if len(parts) > 1 and any(len(part) >= 3 and part in words for part in parts):
        return "a word with a period inside"
    return None


def select_abbreviations(evidence: Evidence, candidates: Counter,
                         lexicon_words: set[str],
                         rejects: dict[str, str]) -> tuple[list[str], list[dict]]:
    """Abbreviations with their periods (see the module docstring).

    Abbreviations with a lowercase letter (π.χ., κ.λπ., δισ., Δρ., μ.Χ.)
    must not end the sentence. All-capital spellings and single capital
    initials are left out. When the commonest spelling is uppercase (Ε.Ε.),
    a lowercase variant still qualifies with its own token and document
    evidence (ε.ε.). No lowercase variants are inferred from uppercase use.
    The shapes :func:`abbreviation_shape_reason` refuses are excluded too.

    A one-period abbreviation of two or more letters must be written with
    its period more often than without (95% Wilson lower bound of the share
    at least ``ABBREVIATION_MIN_SHARE``); the dotted ones (π.χ.) and single
    letters (κ., σ.) cannot be words, so their counts suffice. A word the
    corpus tokenizer did not know as an abbreviation qualifies when a
    lowercase word follows it after a period, by the same share, at least
    ``ABBREVIATION_CANDIDATE_MIN_TOKENS`` times.

    Spellings that differ only in case are counted together, and the
    commonest is listed. Another spelling with a lowercase letter and its
    own ``ABBREVIATION_MIN_TOKENS`` tokens in ``ABBREVIATION_MIN_DOCS``
    documents is listed as well (π.Χ. beside π.χ., υπ. beside Υπ.), unless
    it is a listed spelling with its first letter capitalized (Παρ., a
    sentence start), which a keyboard lowercases before it looks."""
    by_key: dict[str, Counter] = defaultdict(Counter)
    docs: Counter = Counter()
    spelling_docs: Counter = Counter()
    for pool in evidence.pools:
        for token, n in pool.count.items():
            if not token.endswith(".") or len(token) < 2:
                continue
            letters = token.replace(".", "")
            if not letters or any(c not in L.LETTERS for c in letters):
                continue
            key = L.lower_el(token).replace("ς", "σ")  # σ before the period
            by_key[key][token] += n
            docs[key] += pool.docs.get(token, 0)
            spelling_docs[token] += pool.docs.get(token, 0)
    rows = []
    for key, spellings in by_key.items():
        n = sum(spellings.values())
        spelling = max(sorted(spellings), key=lambda s: spellings[s])
        if not _has_lowercase(spelling):
            # Do not discard a genuinely attested lowercase abbreviation
            # merely because its uppercase counterpart occurs more often.
            lower = [s for s in spellings if _has_lowercase(s)
                     and spellings[s] >= ABBREVIATION_MIN_TOKENS
                     and spelling_docs[s] >= ABBREVIATION_MIN_DOCS]
            if not lower:
                continue
            spelling = max(sorted(lower), key=lambda s: spellings[s])
        letters = spelling.replace(".", "")
        if n < ABBREVIATION_MIN_TOKENS or docs[key] < ABBREVIATION_MIN_DOCS:
            continue
        plain = sum(evidence.exact(v)[0] for v in {letters, L.lower_el(letters),
                                                   titlecase(L.lower_el(letters))})
        share = n / (n + plain)
        if (spelling.count(".") == 1 and len(letters) > 1
                and L.wilson_lower(n, n + plain) < ABBREVIATION_MIN_SHARE):
            continue
        rows.append({"abbreviation": spelling, "tokens": n, "docs": docs[key],
                     "without_period": plain, "share": round(share, 4),
                     "route": "tokenizer"})
        variants = [s for s in sorted(spellings, key=lambda s: (-spellings[s], s))
                    if s != spelling and _has_lowercase(s)
                    and spellings[s] >= ABBREVIATION_MIN_TOKENS
                    and spelling_docs[s] >= ABBREVIATION_MIN_DOCS]
        listed = {spelling} | set(variants)
        for variant in variants:
            if variant[0] != L.lower_el(variant[0]) and \
                    L.lower_el(variant[0]) + variant[1:] in listed:
                continue
            rows.append({"abbreviation": variant, "tokens": spellings[variant],
                         "docs": spelling_docs[variant], "without_period": plain,
                         "share": round(share, 4), "route": "case_variant"})
    for word, n in candidates.items():
        abbr = word + "."
        if (L.lower_el(abbr).replace("ς", "σ") in by_key
                or n < ABBREVIATION_CANDIDATE_MIN_TOKENS
                or any(c not in L.LETTERS for c in word)
                or L.case_shape(word) not in ("lower", "title")):
            continue
        plain = evidence.exact(word)[0]
        if L.wilson_lower(n, n + plain) < ABBREVIATION_MIN_SHARE:
            continue
        if word in lexicon_words and plain >= n:
            continue
        rows.append({"abbreviation": abbr, "tokens": n, "docs": None,
                     "without_period": plain, "share": round(n / (n + plain), 4),
                     "route": "followed_by_lowercase"})
    kept, audit = [], []
    for row in sorted(rows, key=lambda r: (-r["tokens"], r["abbreviation"])):
        shape = abbreviation_shape_reason(row["abbreviation"], lexicon_words)
        if shape:
            row["rejected"] = shape
        elif row["abbreviation"] in rejects:
            row["rejected"] = rejects[row["abbreviation"]]
        else:
            kept.append(row["abbreviation"])
        audit.append(row)
    return sorted(set(kept), key=lambda s: s.encode("utf-8")), audit


def run_together_split(key: str, common_keys: set[str]) -> tuple[str, str] | None:
    """The two rated entries whose letters (``el_lexicon.bare_key``) make
    up ``key``, for a token of capitals that is two words run together
    (κειμενοσελιδα: κειμενο + σελιδα), or None."""
    if len(key) < ACRONYM_RUN_TOGETHER_MIN_LETTERS:
        return None
    for i in range(ACRONYM_RUN_TOGETHER_MIN_HEAD, len(key)):
        if key[:i] in common_keys and key[i:] in common_keys:
            return key[:i], key[i:]
    return None


def select_acronyms(evidence: Evidence, entries: set[str],
                    rejects: dict[str, str],
                    common_entries: set[str] = frozenset(), *,
                    accepts: dict[str, str] | None = None
                    ) -> tuple[list[str], list[dict]]:
    """All-capital acronyms (see the module docstring). The letters of an
    acronym must not be those of an entry of the word list (ΠΕ, πε), so
    that a word typed in capitals is still resolved to its accented
    spelling, unless the acronym is far more frequent than the entry
    (``ACRONYM_OVER_WORD_RATIO``: ΣΥΡΙΖΑ beside σύριζα). A name the lexicon has but the list does not, because text
    writes it in capitals (Σύριζα), leaves its acronym (ΣΥΡΙΖΑ) in.
    ``common_entries`` are the entries rated C or M, whose letters make up
    two words run together (``run_together_split``). Reviewed ``accepts``
    allow source-identified dual readings such as ΕΚ/εκ, still requiring
    20 capital tokens in five documents of each of two independent sources.
    """
    others: Counter = Counter()          # bare key -> tokens not in capitals
    caps: Counter = Counter()
    caps_docs: Counter = Counter()
    independent: dict[str, set[str]] = defaultdict(set)
    attested: dict[str, set[str]] = defaultdict(set)
    for table in evidence.tables:
        source_caps: Counter = Counter()
        source_others: Counter = Counter()
        for token, n in table.count.items():
            if token.endswith(".") or L.APOSTROPHE in token:
                continue
            if any(c not in L.LETTERS for c in token):
                continue
            shape = L.case_shape(token)
            if shape == "upper" and len(token) >= 2:
                caps[token] += n
                caps_docs[token] += table.docs.get(token, 0)
                source_caps[token] += n
            elif shape in ("lower", "title"):
                others[L.bare_key(token)] += n
                source_others[L.bare_key(token)] += n
        for token, n in source_caps.items():
            other = source_others[L.bare_key(token)]
            if n >= ACRONYM_MIN_TOKENS and table.docs.get(token, 0) >= ACRONYM_MIN_DOCS:
                attested[token].add(table.source)
                if n >= ACRONYM_OVER_WORD_RATIO * other:
                    independent[token].add(table.source)
    bare_entries: Counter = Counter()    # bare key -> subtitle tokens
    for w in entries:
        if L.APOSTROPHE not in w:
            bare_entries[L.bare_key(w)] += evidence.subtitles.get(L.lower_el(w), 0)
    common_keys = {L.bare_key(w) for w in common_entries if L.APOSTROPHE not in w}
    kept, audit = [], []
    for token in sorted(caps):
        n = caps[token]
        if n < ACRONYM_MIN_TOKENS or caps_docs[token] < ACRONYM_MIN_DOCS:
            continue
        if any(c in "ΆΈΉΊΌΎΏ" for c in token):
            continue        # a tonos on a capital: a word, not an acronym
        key = L.bare_key(token)
        other = others.get(key, 0)
        reviewed = token in (accepts or {})
        confirmed_review = reviewed and len(attested[token]) >= ACRONYM_MIN_CAPITALIZED_SOURCES
        if not reviewed and L.wilson_lower(n, n + other) < ACRONYM_MIN_SHARE:
            continue
        row = {"acronym": token, "tokens": n, "docs": caps_docs[token],
               "not_in_capitals": other,
               "capitalized_sources": sorted(independent[token])}
        if reviewed:
            row.update(reviewed_reason=accepts[token], attested_sources=sorted(attested[token]))
        if token in rejects:
            row["rejected"] = rejects[token]
        elif reviewed and not confirmed_review:
            row["rejected"] = "reviewed acronym lacks independent corpus evidence"
        elif not confirmed_review and key in bare_entries and not (
                n >= ACRONYM_OVER_WORD_RATIO * other
                and (n * evidence.subtitle_words
                     >= ACRONYM_OVER_WORD_RATIO * bare_entries[key] * evidence.open_words
                     or len(independent[token]) >= ACRONYM_MIN_CAPITALIZED_SOURCES)):
            row["rejected"] = "its letters are an entry of the word list"
            row["subtitle_tokens_of_the_entry"] = bare_entries[key]
        elif (split := run_together_split(key, common_keys)) is not None:
            row["rejected"] = f"two words run together: {split[0]} + {split[1]}"
        else:
            kept.append(token)
        audit.append(row)
    audit.sort(key=lambda r: (-r["tokens"], r["acronym"]))
    return sorted(kept, key=lambda s: s.encode("utf-8")), audit


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def try_letters(evidence: Evidence) -> str:
    """The lowercase monotonic letters by their frequency in the open
    corpus's lowercase words, commonest first."""
    letters: Counter = Counter()
    for pool in evidence.pools:
        for token, n in pool.count.items():
            if L.case_shape(token.replace(L.APOSTROPHE, "")) != "lower":
                continue
            for c in token:
                if c in L.LOWER_LETTERS:
                    letters[c] += n
    for c in L.LOWER_LETTERS:
        letters.setdefault(c, 0)
    return "".join(sorted(letters, key=lambda c: (-letters[c], c)))


def compiled_size(forms: list[str], try_line: str) -> int:
    """Bytes of the keyboard's compiled dictionary (TONOSDIC v5) for these
    forms: header, raw offsets, raw pool, TRY, stripped index, stripped
    offsets, stripped pool, frequency bytes."""
    n = len(forms)
    raw = sum(len(f.encode("utf-8")) for f in forms)
    stripped = sum(len(L.bare_key(f).encode("utf-8")) for f in forms)
    return (COMPILED_HEADER_BYTES + 4 * (n + 1) + raw + len(try_line.encode("utf-8"))
            + 4 * n + 4 * (n + 1) + stripped + n)


def git_commit() -> str:
    """HEAD, marked ``-dirty`` when tracked files differ from it."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode()
    except Exception:
        return commit
    return f"{commit}-dirty" if status.strip() else commit


def read_version() -> str:
    path = ROOT / "VERSION"
    return path.read_text().strip() if path.exists() else "0.0.0"


def is_shipping_dir(out_dir: Path) -> bool:
    out_dir = Path(out_dir)
    try:
        return out_dir.samefile(OUT)
    except OSError:
        return str(out_dir.resolve()).casefold() == str(OUT.resolve()).casefold()


LM_VERSION = ROOT / "build" / "lm" / "el_ngram.version"


def stale_model_note(dic_path: Path, version_path: Path | None = None) -> str | None:
    """A note when the next-word model at ``version_path`` (default
    ``LM_VERSION``) was built from another word list than ``dic_path``: its
    vocabulary and its out-of-vocabulary table come from the list, so it
    must be counted and exported again. None when they match or either
    file is missing."""
    from export_lm import dictionary_drift

    return dictionary_drift(Path(version_path or LM_VERSION), Path(dic_path),
                            lang="el")


def dic_lines(entries: dict[str, Entry]) -> list[str]:
    lines = [f"{e.form}\tfr:{e.bucket}" + (" mg:generated" if e.generated else "")
             + "".join(f" ac:{twin}" for twin in e.accent_alternatives)
             + "".join(f" sp:{twin}" for twin in e.spelling_alternatives)
             + (f" us:{e.usage}" if e.usage else "")
             for e in entries.values()]
    return sorted(lines, key=lambda s: s.encode("utf-8"))


def write_text(path: Path, text: str) -> str:
    path.write_bytes(text.encode("utf-8"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_outputs(sel: Selection, out_dir: Path, *, evidence: Evidence,
                  abbreviations: list[str], acronyms: list[str],
                  inputs: dict[str, str], try_line: str, version: str,
                  commit: str, extra_audit: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = dic_lines(sel.entries)
    dic_sha = write_text(out_dir / f"{DIC_NAME}.dic",
                         f"{len(lines)}\n" + "".join(l + "\n" for l in lines))
    aff_sha = write_text(
        out_dir / f"{DIC_NAME}.aff",
        "# Dilemma monotonic Modern Greek word list\n"
        f"# Version {version} (built from commit {commit})\n"
        "# Generated by export_el_monotonic.py. The .dic has no affix flags;\n"
        "# TRY lists the lowercase letters by their frequency in open text.\n"
        "SET UTF-8\nLANG el\nFLAG num\n"
        f"TRY {try_line}\n")
    abbr_sha = write_text(out_dir / f"{DIC_NAME}_abbreviations.txt",
                          "".join(a + "\n" for a in abbreviations))
    acr_sha = write_text(out_dir / f"{DIC_NAME}_acronyms.txt",
                         "".join(a + "\n" for a in acronyms))
    entries = list(sel.entries.values())
    buckets = Counter(e.bucket for e in entries)
    routes = Counter(e.route for e in entries)
    generated = sum(1 for e in entries if e.generated)
    size = compiled_size([e.form for e in entries], try_line)
    sources = evidence.frequency_sources()
    version_text = (
        f"version: {version}\n"
        f"commit: {commit}\n"
        f"variant: {VARIANT}\n"
        f"entries: {len(entries)}\n"
        f"attested: {len(entries) - buckets['X']}\n"
        f"generated: {generated}\n"
        f"buckets: C={buckets['C']} M={buckets['M']} R={buckets['R']} X={buckets['X']}\n"
        f"bucket_edges_per_million: C={L.C_PER_MILLION:g} M={L.M_PER_MILLION:g}\n"
        "routes: " + " ".join(f"{r}={routes[r]}" for r in ROUTES) + "\n"
        "frequency_sources: " + " ".join(f"{k}={v['words']}" for k, v in sources.items())
        + "\n"
        f"abbreviations: {len(abbreviations)}\n"
        f"acronyms: {len(acronyms)}\n"
        f"accent_alternatives: {sum(bool(e.accent_alternatives) for e in entries)}\n"
        f"spelling_alternatives: {sum(bool(e.spelling_alternatives) for e in entries)}\n"
        f"compiled_size_estimate_bytes: {size}\n"
        f"aff_rules: 0\n"
        f"dic_sha256: {dic_sha}\n"
        f"aff_sha256: {aff_sha}\n"
        f"abbreviations_sha256: {abbr_sha}\n"
        f"acronyms_sha256: {acr_sha}\n"
        + "".join(f"{name}_sha256: {digest}\n" for name, digest in inputs.items()))
    write_text(out_dir / f"{DIC_NAME}.version", version_text)
    # evidence table
    with open(out_dir / f"{DIC_NAME}.evidence.tsv.gz", "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as fh:
            header = ["form", "route", "bucket", "generated", "best_source", "lemma",
                      SUBTITLES] + [p.name for p in evidence.pools]
            fh.write(("\t".join(header) + "\n").encode("utf-8"))
            rows = []
            for e in sorted(entries, key=lambda e: e.form.encode("utf-8")):
                counts = evidence.raw_counts(e.form)
                rows.append("\t".join(
                    [e.form, e.route, e.bucket, "1" if e.generated else "0", e.source,
                     e.lemma, str(int(counts.get(SUBTITLES, 0)))]
                    + [f"{counts.get(p.name, 0):.1f}".rstrip("0").rstrip(".")
                       for p in evidence.pools]) + "\n")
                if len(rows) >= 50_000:
                    fh.write("".join(rows).encode("utf-8"))
                    rows.clear()
            fh.write("".join(rows).encode("utf-8"))
    route_buckets = {r: dict(sorted(Counter(e.bucket for e in entries
                                            if e.route == r).items()))
                     for r in ROUTES}
    rejected_top = {}
    for reason, examples in sorted(sel.rejected_examples.items()):
        top = sorted(examples, key=lambda x: (-x[0], x[1]))[:50]
        rejected_top[reason] = [[f, int(w)] for w, f in top]
    audit = {
        "schema_version": 1,
        "variant": VARIANT,
        "version": version,
        "commit": commit,
        "entries": len(entries),
        "buckets": dict(sorted(buckets.items())),
        "generated": generated,
        "routes": dict(sorted(routes.items())),
        "route_buckets": route_buckets,
        "frequency_sources": sources,
        "thresholds": {k: v for k, v in sorted(globals().items())
                       if k.isupper() and isinstance(v, (int, float))
                       and not isinstance(v, bool)},
        "variant_min_twin_ratios": VARIANT_MIN_TWIN_RATIOS,
        "slip_min_twin_ratio": SLIP_MIN_TWIN_RATIO,
        "learned_endings": [list(pair) for pair in LEARNED_ENDINGS],
        "compiled_size_estimate_bytes": size,
        "bytes_per_entry": round(size / max(1, len(entries)), 2),
        "rejected": dict(sorted(sel.rejected.items())),
        "rejected_top": rejected_top,
        "inputs_sha256": inputs,
        **sel.audit,
        **extra_audit,
    }
    write_text(out_dir / f"{DIC_NAME}.audit.json",
               json.dumps(audit, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    return {"entries": len(entries), "buckets": dict(buckets), "routes": dict(routes),
            "generated": generated, "compiled_size_estimate_bytes": size,
            "abbreviations": len(abbreviations), "acronyms": len(acronyms)}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def type_count_digest(paths: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for source in sorted(paths):
        digest.update(f"{source}:{sha256_file(paths[source])}\n".encode())
    return digest.hexdigest()


def run(args) -> dict:
    t0 = time.time()
    out_dir = Path(args.out_dir)
    if args.max_lemmas is not None and is_shipping_dir(out_dir):
        raise ExportError(f"refusing to write a partial list (--max-lemmas) to {OUT}; "
                          "pass another --out-dir")
    if (getattr(args, "c_confirm_min_words", RATING_C_CONFIRM_MIN_WORDS)
            != RATING_C_CONFIRM_MIN_WORDS and is_shipping_dir(out_dir)):
        raise ExportError(f"refusing to write a test rating rule to {OUT}; "
                          "pass another --out-dir")
    corpus_dir = Path(args.corpus)
    if not (corpus_dir / "manifest.json").exists():
        raise ExportError(f"{corpus_dir}/manifest.json is missing; run "
                          "extract_el_monotonic_lm.py first")
    for path in (args.mg_pairs, args.mg_freq, args.function_words, args.review):
        if not Path(path).exists():
            raise ExportError(f"{path} is missing")
    print("Counting or reading the corpus type counts ...", flush=True)
    tables = TC.ensure_tables(corpus_dir, Path(args.type_counts), split="train",
                              workers=args.workers)
    counts = [TC.read_table(tables[s]) for s in sorted(tables)]
    print("Reading the lexicon ...", flush=True)
    lexicon = load_lexicon(Path(args.mg_pairs),
                           None if args.no_lookup else Path(args.lookup_db),
                           max_lemmas=args.max_lemmas)
    evidence = Evidence(load_subtitle_counts(Path(args.mg_freq)), counts,
                        c_confirm_min_words=getattr(args, "c_confirm_min_words",
                                                    RATING_C_CONFIRM_MIN_WORDS))
    function_words = load_function_words(Path(args.function_words))
    review = load_review(Path(args.review))
    loanwords = load_reviewed_loanwords(Path(args.reviewed_loanwords))
    print("Selecting ...", flush=True)
    sel = select(lexicon, evidence, function_words=function_words, review=review,
                 reviewed_loanwords=loanwords)
    lexicon_words = lexicon.forms() | lexicon.raw_forms
    candidates = _read_candidates(corpus_dir, [t.source for t in counts])
    abbreviations, abbr_audit = select_abbreviations(
        evidence, candidates, lexicon_words | set(sel.entries),
        review["abbreviation_rejects"])
    acronyms, acr_audit = select_acronyms(
        evidence, set(sel.entries), review["acronym_rejects"],
        {f for f, e in sel.entries.items() if e.bucket in ("C", "M")},
        accepts=review["acronym_accepts"])
    paths = {"mg_pairs": Path(args.mg_pairs), "mg_freq": Path(args.mg_freq),
             "function_words": Path(args.function_words), "review": Path(args.review),
             "corpus_manifest": corpus_dir / "manifest.json"}
    companion = Path(args.mg_pairs).with_name("el_participles.json")
    if companion.exists():
        paths["el_participles"] = companion
    if not args.no_lookup and Path(args.lookup_db).exists():
        paths["lookup_db"] = Path(args.lookup_db)
    if Path(args.reviewed_loanwords).exists():
        paths["reviewed_loanwords"] = Path(args.reviewed_loanwords)
    if CORPUS_MANIFEST.exists():
        paths["el_lm_corpora"] = CORPUS_MANIFEST
    inputs = {name: sha256_file(p) for name, p in sorted(paths.items())}
    inputs["type_counts"] = type_count_digest(tables)
    mismatched = check_hf_pins({k: v for k, v in paths.items()
                                if k in ("mg_pairs", "el_participles", "lookup_db")},
                               inputs)
    if mismatched and not args.allow_unpinned:
        raise ExportError("inputs differ from data/hf_manifest.json (download the "
                          "pinned files or pass --allow-unpinned): "
                          + "; ".join(mismatched))
    try_line = try_letters(evidence)
    result = write_outputs(
        sel, out_dir, evidence=evidence, abbreviations=abbreviations,
        acronyms=acronyms, inputs=inputs, try_line=try_line,
        version=args.version or read_version(), commit=args.commit or git_commit(),
        extra_audit={"abbreviation_review": abbr_audit,
                     "acronym_review": acr_audit[:AUDIT_LIST_LIMIT * 2],
                     "lexicon_origins": dict(sorted(lexicon.origins.items())),
                     "lexicon_lemmas": len(lexicon.groups),
                     "max_lemmas": args.max_lemmas,
                     "unpinned_inputs": mismatched})
    print(f"Wrote {out_dir}/{DIC_NAME}.dic: {result['entries']:,} entries "
          f"({', '.join(f'{k}={v:,}' for k, v in sorted(result['buckets'].items()))}; "
          f"{result['generated']:,} generated), {result['abbreviations']} abbreviations, "
          f"{result['acronyms']} acronyms, compiled about "
          f"{result['compiled_size_estimate_bytes'] / 1e6:.1f} MB "
          f"({time.time() - t0:.0f} s)")
    if is_shipping_dir(out_dir):
        note = stale_model_note(out_dir / f"{DIC_NAME}.dic")
        if note:
            print(f"NOTE: {note}")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--type-counts", type=Path, default=TC.DEFAULT_OUT_DIR)
    ap.add_argument("--mg-pairs", type=Path, default=MG_PAIRS)
    ap.add_argument("--lookup-db", type=Path, default=LOOKUP_DB)
    ap.add_argument("--no-lookup", action="store_true",
                    help="do not read the el side of lookup.db")
    ap.add_argument("--mg-freq", type=Path, default=MG_FREQ)
    ap.add_argument("--function-words", type=Path, default=FUNCTION_WORDS)
    ap.add_argument("--review", type=Path, default=REVIEW)
    ap.add_argument("--reviewed-loanwords", type=Path, default=REVIEWED_LOANWORDS)
    ap.add_argument("--max-lemmas", type=int, default=None,
                    help="read only the first N lemmas (a quick sanity run)")
    ap.add_argument("--allow-unpinned", action="store_true",
                    help="accept HF inputs whose sha256 differs from data/hf_manifest.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--version", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--commit", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--c-confirm-min-words", type=int,
                    default=RATING_C_CONFIRM_MIN_WORDS, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    try:
        run(args)
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
