#!/usr/bin/env python3
"""Export Ancient Greek boundary-rewrite morphology tables for Tonos.

Produces a single compact JSON at ``build/hunspell/grc_morph.json`` that
the Tonos iOS keyboard reads at install time. The file carries two
boundary-sensitive rewrite tables, and a Modern Greek overlay for the
second:

* ``nu`` - surface forms that take movable nu (``ἐστί`` -> ``ἐστίν``)
  when the next word is vowel-initial. Includes:

  - Verb 3sg active indicative past (imperfect / aorist / pluperfect /
    perfect) ending in ``-ε``;
  - Verb 3sg active indicative present ending in ``-σι`` / ``-τι``
    (``ἐστί``, ``δίδωσι``, ``τίθησι``);
  - Verb 3pl active indicative (present / future / perfect) ending in
    ``-σι`` (``λέγουσι``, ``γράφουσι``);
  - Noun / pronoun / adjective dative plural ending in ``-σι`` /
    ``-ξι`` / ``-ψι`` (``ἀνδράσι``, ``πᾶσι``);
  - Adverbs in ``-σι`` and adverbs of place in ``-θε`` that the texts
    write with the ν (``παντάπασι``, ``πρόσθε``, ``ὄπισθε``);
  - A small closed list of numerals that historically take movable nu
    (``εἴκοσι``).

  - Verb 3sg aorist optative in ``-ειε`` (``δόξειε``, ``ποιήσειε``).

  Other moods and non-finite forms are excluded, apart from what they
  share with the endings above: the subjunctive's third person in
  ``-σι``, a participle's dative plural, and the optative in ``-ειε``.
  So is an unaugmented past in ``-ε`` that is spelled like the
  imperative and that the texts never write with the ν (``κατένεγκε``).

* ``el`` - full-form -> elided-form pairs harvested from GLAUx,
  Diorisis, and the canonical dilemma lookup table. The elided form
  is stored in NFC with its final elision glyph canonicalised to
  U+1FBD GREEK KORONIS (same convention as
  ``HunspellCompiler.normalizeEntry`` and ``GreekStyle``'s existing
  hardcoded particle table). The Tonos output layer rewrites the
  koronis to the user's chosen elision glyph via
  ``GreekStyle.applyingElisionMark``.

* ``el_modern`` - the Modern Greek elisions (``τώρα`` -> ``τώρ᾽``,
  ``γιὰ`` -> ``γι᾽``, ``στὸ`` -> ``στ᾽``, ``ποὺ`` -> ``π᾽``), in the same
  ``{full: elided}`` shape, derived from the polytonic Modern Greek slice
  by ``export_mg_polytonic.modern_greek_elisions``. Only the pairs the
  Ancient Greek ``el`` table lacks, or spells otherwise, are listed, so a
  keyboard writing Modern Greek reads ``el`` with ``el_modern`` laid over
  it, and one writing Ancient Greek reads ``el`` alone, as before. They are
  kept out of ``el`` because a keyboard that elides automatically would
  otherwise apply them to Ancient Greek text, where the article ``τὸ`` or
  the particle ``ὅτι`` never elides. The key is optional: a reader that
  knows only ``nu`` and ``el`` ignores it.

* ``el_modern_share`` - for each ``el_modern`` key, how often the slice
  elides it: of the times it writes a word the elided spelling stands for
  right before a vowel-initial word, the share it writes it elided
  (``γιὰ`` 0.6, ``τώρα`` 0.06). Modern Greek elides optionally, and most
  of these elisions are the minority spelling, so a keyboard can elide a
  word automatically only where the texts mostly do.

The derivation pulls from:

* ``data/glaux_pairs.json`` + ``data/diorisis_pairs.json``: morpho-
  tagged token stream with ``pos`` and grammatical feature tags. This
  is where the nu-eligibility tags come from.
* ``data/lookup.db`` (grc src only): full form-to-lemma table, used to
  augment the elision pair list with forms that appear only in the
  Wiktionary paradigm expansion.

When the corpus offers multiple elided variants for a single full form
(e.g. ``ἀπ᾽`` vs ``ἀφ᾽`` for ``ἀπό``; different editions use different
glyphs), we score candidates and pick the one that best matches the
full form's casing and breathing profile. Canonical ten particles are
additionally pinned to a fixed mapping so ``ἀλλά`` always elides to
``ἀλλ᾽`` regardless of whatever minority spellings the corpus carries.

Output is a single JSON file; Tonos compresses it with LZMA as part of
``scripts/prebuild_archives.sh``. Total compressed size is under
300 KB as of this writing, against the app's 10 MB size budget for the
combined morphology feature.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from dilemma.form_sanitize import has_editorial_sigla
from export_hunspell import exact_form_key, grc_orthography_reason

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "build" / "hunspell"

GLAUX_PAIRS = DATA / "glaux_pairs.json"
DIORISIS_PAIRS = DATA / "diorisis_pairs.json"
LOOKUP_DB = DATA / "lookup.db"

# Elision glyphs we treat as equivalent on the trailing position.
# U+0313 COMBINING COMMA ABOVE and U+1FBD GREEK KORONIS are the two
# canonical Greek editing conventions; U+02BC MODIFIER LETTER APOSTROPHE
# and U+2019 RIGHT SINGLE QUOTATION MARK appear in modern typeset
# editions (Oxford, Teubner) that reach for a generic apostrophe. All
# four are folded to U+1FBD for storage.
ELISION_GLYPHS = frozenset("\u0313\u1FBD\u02BC\u2019")
KORONIS = "\u1FBD"

# Unicode combining categories we strip when normalising a form for
# accent-blind prefix comparison.
_COMBINING = "Mn"

# The ten canonical particle / preposition mappings. These are pinned so
# the iconic entries (``ἀλλ᾽``, ``κατ᾽`` and friends) use the textbook
# spelling regardless of what variant wins the corpus vote. Listed as
# {full: elided} where both strings are NFC; the elided form carries
# U+1FBD GREEK KORONIS as its final scalar, matching the canonical
# storage glyph used by ``HunspellCompiler.normalizeEntry`` and by
# ``GreekStyle.elidableParticles``.
CANONICAL_ELISION_OVERRIDES: dict[str, str] = {
    "ἀλλά": "ἀλλ" + KORONIS,
    "διά":  "δι" + KORONIS,
    "ἐπί":  "ἐπ" + KORONIS,
    "κατά": "κατ" + KORONIS,
    "μετά": "μετ" + KORONIS,
    "παρά": "παρ" + KORONIS,
    "ὑπό":  "ὑπ" + KORONIS,
    "ἀπό":  "ἀπ" + KORONIS,
    "ἀντί": "ἀντ" + KORONIS,
    "δέ":   "δ" + KORONIS,
}

# Extra forms that classical grammar flags as movable-nu-eligible but
# that corpus tagging misses. ``εἴκοσι`` is the only one that survives
# cleanly, so we keep the list short.
EXTRA_NU_FORMS: frozenset[str] = frozenset([
    "εἴκοσι",
    # The prodelided ἐστι (ὁ ’στιν) as a keyboard reads it after an
    # apostrophe, which ends the word before it. It takes nu before a vowel
    # in every GLAUx instance; the orthography rules reject it bare.
    "στι",
])


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _strip_lower(s: str) -> str:
    """Return a lowercased, diacritic-stripped copy of ``s``.

    Used for accent-blind prefix matching during elision pair
    derivation. The goal is to let ``κατά`` (full) match ``κατ᾽``
    (elided, stripped stem ``κατ``) whether or not either carries a
    breathing or final accent.
    """
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != _COMBINING).lower()


def _last_base_vowel(s: str) -> str | None:
    """Return the lowercased base vowel at the end of ``s`` once all
    trailing combining marks are skipped, or None if the last base
    character isn't one of the seven Greek vowels.
    """
    nfd = unicodedata.normalize("NFD", s)
    for c in reversed(nfd):
        if unicodedata.category(c) != _COMBINING:
            lower = c.lower()
            return lower if lower in "αεηιουω" else None
    return None


def _has_accent(s: str) -> bool:
    nfd = unicodedata.normalize("NFD", s)
    return any(ord(c) in (0x0300, 0x0301, 0x0342) for c in nfd)


def _canon_elided(s: str) -> str:
    """Normalise the trailing elision glyph to U+1FBD."""
    if not s:
        return s
    if s[-1] in ELISION_GLYPHS:
        return s[:-1] + KORONIS
    return s


_GREEK_VOWELS = frozenset("αεηιουωΑΕΗΙΟΥΩ")

# Smyth 174: an oxytone that loses its final vowel throws the accent back
# onto the penult as an acute (ἀνδρί -> ἄνδρ᾽, αὐτή -> αὔτ᾽). Prepositions
# and conjunctions lose it outright instead (ἀλλά -> ἀλλ᾽). The ten in
# CANONICAL_ELISION_OVERRIDES are pinned anyway; these are the rest of the
# class, which nothing else would keep bare.
ELISION_KEEPS_NO_ACCENT: frozenset[str] = frozenset([
    # prepositions
    "ανα", "αμφι", "αντι", "απο", "δια", "επι", "κατα", "μετα", "παρα",
    "υπο",
    # conjunctions of the same class
    "αλλα", "δε", "ουδε", "μηδε",
    # enclitics, which have no accent of their own to throw back
    # (Smyth 183). The orthotone εἰμί and φημί are not among them:
    # εἰμί elides to εἴμ᾽ and ἐμέ, the emphatic pronoun, to ἔμ᾽.
    "τε", "γε", "τοι", "ποτε", "που", "πως", "πη", "νυν", "ρα",
    # Epic and Doric members of the same classes. τοτέ, "at times", is an
    # accented adverb rather than an enclitic, so it is not among them.
    "ποτι", "ηδε", "ηε",
    "με", "σε", "μοι", "σοι", "τινα", "τινι", "τινε", "τινος",
    # Epic, Ionic, Doric and Aeolic prepositions and particles, the dual
    # enclitic σφωε, prepositions and conjunctions in crasis (κἀπί, τἀπό,
    # μἀλλά), and ἰδέ, which elides bare as the epic conjunction, most of its
    # elided tokens, rather than as the imperative.
    "προτι", "κοτε", "ποκα", "πεδα", "υπα", "σφωε",
    "καπι", "καπο", "ταπι", "ταπο", "καντι", "μαλλα", "κουδε", "ιδε",
])

# Matched with the breathing kept: the epic preposition ἐνί elides bare,
# and the numeral ἑνί retracts like any oxytone.
ELISION_KEEPS_NO_ACCENT_SMOOTH: frozenset[str] = frozenset(["ενι", "εινι"])


def _elides_bare(full: str) -> bool:
    """True for an oxytone that loses its accent in elision instead of
    throwing it back: a preposition, a conjunction or an enclitic."""
    stripped = _strip_lower(full)
    if stripped in ELISION_KEEPS_NO_ACCENT:
        return True
    return (stripped in ELISION_KEEPS_NO_ACCENT_SMOOTH
            and "\u0314" not in unicodedata.normalize("NFD", full))


# Elision removes a short final vowel and nothing else. η and ω are always
# long, υ does not elide, a circumflex or an iota subscript marks a long
# vowel, and the second element of a diphthong goes with the first.
_ELIDABLE_VOWELS = frozenset("αειο")

# Words whose final vowel is short but which do not elide. Attic never
# elides ὅτι, περί, πρό, ἄχρι or μέχρι (Smyth 72), and ὅτ᾽ reads as ὅτε, δι᾽
# as διά. διό and καθά already contain an elision (δι᾽ ὅ, καθ᾽ ἅ). The
# emphatic and deictic -ί is long (οὐχί, τουτί, ὁδί). A keyboard rewrites
# the word before any vowel, so these would change the text's meaning
# rather than its spelling.
NEVER_ELIDED: frozenset[str] = frozenset([
    "οτι", "περι", "προ", "αχρι", "μεχρι",
    "διο", "καθα",
    "ουχι", "ναιχι", "νυνι", "τουτι", "ταυτι", "ωδι", "οδι", "ηδι", "τοδι",
    "ταδι", "ενθαδι", "ουτοσι", "αυτηι", "τονδι", "τηνδι", "τασδι",
    "τουσδι", "οιδι", "τοιοσδι", "τοιαδι", "τοιονδι", "τοσονδι", "τουδι",
    "τηδι", "τωδι", "ταυτηι", "τουτωι", "τουτουι", "ουτωσι", "εκεινοσι",
    "εκεινηι", "τοιουτοσι", "τοσουτοσι", "τοιαυτι", "τοσαυτι", "δευρι",
    "ενταυθι",
    # contracted neuter plurals of nouns in -ας, whose α is long
    "κρεα", "κερα", "γερα",
])

# A monosyllable does not elide unless it ends in ε (Smyth 72): δέ, τε, σε.
# These particles are the exceptions the corpora attest.
ELIDING_MONOSYLLABLES: frozenset[str] = frozenset(["ρα", "κα", "γα"])

_DIPHTHONGS = frozenset(["αι", "ει", "οι", "υι", "αυ", "ευ", "ηυ", "ου"])


def _final_vowel(form: str):
    """Return (index, base, marks) of the form's last vowel in NFD."""
    nfd = unicodedata.normalize("NFD", form)
    last = -1
    for i, ch in enumerate(nfd):
        if ch in _GREEK_VOWELS:
            last = i
    if last < 0:
        return None
    return last, nfd[last].lower(), nfd[last + 1:]


def _nuclei(nfd: str) -> list[tuple[int, int]]:
    """Spans of an NFD string's vowel nuclei, a diphthong counting as one.

    An accent on the ε or ο of a pair marks a hiatus, not a diphthong, whose
    accent would sit on the second vowel: βασιλέι is βασιλέϊ.
    """
    spans: list[list[int]] = []
    single = None       # (base, marks) when the last span is a lone vowel
    i = 0
    while i < len(nfd):
        j = i + 1
        while j < len(nfd) and unicodedata.combining(nfd[j]):
            j += 1
        base, marks = nfd[i].lower(), nfd[i + 1:j]
        if base in "αεηιουω":
            hiatus = single is not None and single[0] in "εο" and any(
                m in single[1] for m in "\u0300\u0301")
            if (single is not None and spans[-1][1] == i
                    and single[0] + base in _DIPHTHONGS
                    and "\u0308" not in marks and not hiatus):
                spans[-1][1] = j
                single = None
            else:
                spans.append([i, j])
                single = (base, marks)
        else:
            single = None
        i = j
    return [(a, b) for a, b in spans]


def _is_oxytone(form: str) -> bool:
    """True when an acute or a grave sits on the form's final vowel.

    The grave is only the contextual spelling of an oxytone, and it is the
    spelling a word carries mid-sentence, which is exactly where elision
    happens. ``αὐτὸ`` retracts to ``αὔτ᾽`` for the same reason ``αὐτή``
    retracts to ``αὔτ᾽``.
    """
    found = _final_vowel(form)
    if not found:
        return False
    index, _base, marks = found
    if "\u0301" not in marks and "\u0300" not in marks:
        return False
    # A second accent on the last syllable is an enclitic's, thrown onto the
    # word's own acute or onto a circumflex on the penult (σῶμά τι, χεῖρά
    # τε). It leaves with the elided vowel, and the word keeps its own
    # accent where the user put it: χεῖρά elides to χεῖρ᾽, not χείρ᾽, and
    # the grave rule's τοῦτὸ to τοῦτ᾽. A circumflex further back cannot host
    # one, so ὦγαθέ, a crasis, is an oxytone (ὦγάθ᾽); a grave earlier in the
    # word is only a malformed spelling (μὲτὰ).
    nfd = unicodedata.normalize("NFD", form)
    spans = _nuclei(nfd)
    penult = nfd[spans[-2][0]:spans[-2][1]] if len(spans) >= 2 else ""
    return "\u0301" not in nfd[:index] and "\u0342" not in penult


def _can_elide(form: str) -> bool:
    """True when the form's final vowel is one elision can remove."""
    stripped = _strip_lower(form)
    if stripped in NEVER_ELIDED:
        return False
    found = _final_vowel(form)
    if not found:
        return False
    _index, base, marks = found
    if base not in _ELIDABLE_VOWELS:
        return False
    # A circumflex, an iota subscript or a macron marks a long vowel.
    if "\u0342" in marks or "\u0345" in marks or "\u0304" in marks:
        return False
    nfd = unicodedata.normalize("NFD", form)
    spans = _nuclei(nfd)
    # An iota written after a long η or ω is the subscript written out
    # (λόγωι, τῆι), not a vowel of its own.
    before = [c for c in nfd[:spans[-1][0]] if not unicodedata.combining(c)]
    if (base == "ι" and "\u0308" not in marks
            and before and before[-1].lower() in "ηω"):
        return False
    last = nfd[spans[-1][0]:spans[-1][1]]
    if sum(c.lower() in "αεηιουω" for c in last) > 1:
        return False            # the second element of a diphthong
    if len(spans) == 1 and base != "ε":
        return stripped in ELIDING_MONOSYLLABLES
    return True


def _rule_elision(full: str) -> str:
    """The spelling elision gives ``full``, by rule rather than by corpus.

    The final vowel goes with its own marks. An oxytone throws its accent
    back onto the new last vowel as an acute (ἀνδρί -> ἄνδρ᾽, αὐτό -> αὔτ᾽),
    unless it is one of the words that elide bare (ἀλλά -> ἀλλ᾽), and a
    monosyllable has no vowel left to carry one (σέ -> σ᾽). Everything else
    keeps every mark where it was.
    """
    index, _base, _marks = _final_vowel(full)
    stem = unicodedata.normalize("NFD", full)[:index]
    if _is_oxytone(full) and not _elides_bare(full):
        last = max((i for i, ch in enumerate(stem) if ch in _GREEK_VOWELS),
                   default=None)
        if last is not None:
            end = last + 1
            while end < len(stem) and unicodedata.combining(stem[end]):
                end += 1
            stem = stem[:end] + "\u0301" + stem[end:]
    return _nfc(stem) + KORONIS


def _match_initial_case(elided: str, full: str) -> str:
    """Give ``elided`` the case of ``full``'s first letter.

    Case is not part of the word: a sentence-initial ``Κατ᾽`` is the same
    elision as ``κατ᾽``. The corpora pair a lowercase full form with a
    capitalized elided one wherever the only elided token they saw opened
    a sentence (``ἑλλάδα`` with ``Ἑλλάδ᾽``), and a keyboard writes the
    value into the user's text as it stands.
    """
    head = elided[:1]
    if not head.isalpha():
        return elided
    if full[:1].isupper() and head.islower():
        head = head.upper()
    elif full[:1].islower() and head.isupper():
        head = head.lower()
    else:
        return elided
    return _nfc(head + elided[1:])


# Tags considered disqualifying for movable nu: the moods and non-finite
# forms, apart from the endings ``_nu_reading`` lets through first (a
# subjunctive's -σι, a participle's dative plural, the optative's -ειε).
_NU_DISQUALIFIERS = frozenset([
    "subjunctive", "optative", "imperative", "infinitive", "participle",
])

def _nu_reading(stripped: str, pos: str, tags: set[str]) -> str | None:
    """How one analysis of a spelling stands on movable nu (Smyth 134).

    ``stripped`` is the spelling without marks, lowercased, and without the
    ν when the spelling is the one that already carries it. Returns
    ``"takes"`` for the forms that take it:

      1. Verb 3sg active indicative past (imperfect / aorist /
         pluperfect / perfect) ending in ``-ε``.
      2. Verb 3sg active indicative present ending in ``-σι``
         (``δίδωσι``, ``τίθησι``), or in ``-στι``: ``ἐστί``, its compounds
         and its crasis and prodelided spellings (``πάρεστι``, ``κἀστί``,
         ``’στι``). The Doric ``-τι`` (``ἐντί``, ``δίδωτι``) takes none.
      3. Verb 3pl active indicative (present / future / perfect)
         ending in ``-σι``.
      4. Dative plural ending in ``-σι``, ``-ξι``, or ``-ψι``, of any
         word: the article's and the numerals' as much as a noun's
         (``τοῖσιν``, ``τρισίν``), and a participle's. Diorisis files some
         epic datives as adverbs (``ἀκλινέεσσι``); the case decides.
      5. The subjunctive's third person in ``-σι``, the plural (ὦσιν,
         λύωσιν) and the epic singular (ἐθέλῃσιν).
      6. The aorist optative's third person singular in ``-ειε``
         (δόξειεν, ποιήσειεν), a third person singular in -ε like the
         past's (Smyth 134).

    ``"never"`` for an analysis that cannot take it: another mood or a
    non-finite form, a first or second person, a case other than the
    dative, a dative singular or dual, an indeclinable word in ``-ε``
    (the conjunction ``ηὖτε``, the interjection ``φέρε``). ``None`` when
    the tags do not say,
    as for an untagged adverb (Ἀθήνησι) or a verb form given a voice it
    cannot have (GLAUx files half of ἔσκε as middle).
    """
    last = stripped[-1:]
    # The aorist optative in -ειε is a third person singular in -ε too, and
    # takes the ν like the rest (δόξειεν ἄν; Smyth 134).
    if ("optative" in tags and "third-person" in tags and "singular" in tags
            and stripped.endswith("ειε")):
        return "takes"
    if ("participle" in tags and "dative" in tags and "plural" in tags
            and stripped.endswith(("σι", "ξι", "ψι"))):
        return "takes"
    if ("subjunctive" in tags and "third-person" in tags
            and stripped.endswith("σι")):
        return "takes"
    if tags & _NU_DISQUALIFIERS:
        return "never"
    if (pos == "verb"
            and "third-person" in tags and "singular" in tags
            and "active" in tags and "indicative" in tags
            and (tags & {"imperfect", "aorist", "pluperfect", "perfect"})
            and last == "ε"):
        return "takes"
    if (pos == "verb"
            and "third-person" in tags and "singular" in tags
            and "active" in tags and "indicative" in tags
            and "present" in tags
            and stripped.endswith(("σι", "στι"))):
        return "takes"
    if (pos == "verb"
            and "third-person" in tags and "plural" in tags
            and "active" in tags and "indicative" in tags
            and (tags & {"present", "future", "perfect"})
            and stripped.endswith("σι")):
        return "takes"
    if (pos != "verb" and "dative" in tags and "plural" in tags
            and stripped.endswith(("σι", "ξι", "ψι"))):
        return "takes"
    if tags & {"first-person", "second-person"}:
        return "never"
    if tags & {"nominative", "genitive", "accusative", "vocative"}:
        return "never"
    if "dative" in tags and tags & {"singular", "dual"}:
        return "never"
    # An adverb of place in -θε (or the same word governing a genitive, which
    # GLAUx files as a preposition) alternates with -θεν (Smyth 134 D:
    # πρόσθε(ν)); whether it takes the ν is left to the texts, as for an
    # adverb in -σι (``_alternating_adverb``).
    if _alternating_adverb(stripped, pos):
        return None
    if pos in ("adv", "conj", "particle", "prep", "intj") and last == "ε":
        return "never"
    return None


def _alternating_adverb(stripped: str, pos: str) -> bool:
    """Whether an analysis is an adverb whose final ν the texts add or leave
    off: the local adverbs in -σι (Ἀθήνησι(ν), παντάπασι(ν); Smyth 134) and
    the adverbs of place in -θε (πρόσθε(ν), ὄπισθε(ν); Smyth 134 D), which
    GLAUx files as prepositions when they govern a genitive. ``stripped`` is
    the spelling without the ν."""
    if pos == "adv" and stripped.endswith("σι"):
        return True
    return pos in ("adv", "prep") and stripped.endswith("θε")


_VOWEL_LETTERS = frozenset("αεηιουω")
_CONSONANT_LETTERS = frozenset("βγδζθκλμνξπρστφχψ")
_LABIALS = frozenset("πβφψμ")
_VELARS = frozenset("κγχξ")

# The preverbs a compound verb can begin with, each with its spellings and
# the letters each spelling stands before (None: any). A preverb ending in a
# vowel elides it before a vowel, and before a rough breathing aspirates
# (ἀφ-, καθ-), except περί and πρό and, often, ἀμφί (ἀμφιέννυμι); so
# ἐπιάχω is ἐπ- and ἰάχω, and διαιρέω δι- and αἱρέω. ἐκ becomes ἐξ before
# a vowel; ἐν and σύν assimilate to the consonant that follows, and Attic
# writes ξύν for σύν.
_PREVERB_SPELLINGS: dict[str, tuple[tuple[str, frozenset | None], ...]] = {
    "αμφι": (("αμφι", None), ("αμφ", _VOWEL_LETTERS)),
    "ανα": (("ανα", _CONSONANT_LETTERS), ("αν", _VOWEL_LETTERS)),
    "αντι": (("αντι", _CONSONANT_LETTERS), ("αντ", _VOWEL_LETTERS),
             ("ανθ", _VOWEL_LETTERS)),
    "απο": (("απο", _CONSONANT_LETTERS), ("απ", _VOWEL_LETTERS),
            ("αφ", _VOWEL_LETTERS)),
    "δια": (("δια", _CONSONANT_LETTERS), ("δι", _VOWEL_LETTERS)),
    "εισ": (("εισ", None), ("εσ", None)),
    "εκ": (("εκ", _CONSONANT_LETTERS), ("εξ", None)),
    "εν": (("εν", None), ("εμ", _LABIALS), ("εγ", _VELARS),
           ("ελ", frozenset("λ")), ("ερ", frozenset("ρ"))),
    "επι": (("επι", _CONSONANT_LETTERS), ("επ", _VOWEL_LETTERS),
            ("εφ", _VOWEL_LETTERS)),
    "κατα": (("κατα", _CONSONANT_LETTERS), ("κατ", _VOWEL_LETTERS),
             ("καθ", _VOWEL_LETTERS)),
    "μετα": (("μετα", _CONSONANT_LETTERS), ("μετ", _VOWEL_LETTERS),
             ("μεθ", _VOWEL_LETTERS)),
    "παρα": (("παρα", _CONSONANT_LETTERS), ("παρ", _VOWEL_LETTERS)),
    "περι": (("περι", None),),
    "προσ": (("προσ", None),),
    "προ": (("προ", None), ("πρου", _VOWEL_LETTERS)),
    "συν": (("συν", None), ("συμ", _LABIALS), ("συγ", _VELARS),
            ("συλ", frozenset("λ")), ("συρ", frozenset("ρ")),
            ("συσ", frozenset("σ")), ("συ", frozenset("σζ")),
            ("ξυν", None), ("ξυμ", _LABIALS), ("ξυγ", _VELARS),
            ("ξυλ", frozenset("λ")), ("ξυ", frozenset("σζ"))),
    "υπερ": (("υπερ", None),),
    "υπο": (("υπο", _CONSONANT_LETTERS), ("υπ", _VOWEL_LETTERS),
            ("υφ", _VOWEL_LETTERS)),
}
_PREVERBS_LONGEST_FIRST = sorted(
    ((spelling, before, preverb)
     for preverb, spellings in _PREVERB_SPELLINGS.items()
     for spelling, before in spellings),
    key=lambda row: -len(row[0]))


# The endings a verb is cited with, longest first.
_CITATION_ENDINGS = ("ουμαι", "ωμαι", "ομαι", "αμαι", "εμαι", "υμαι", "μαι",
                     "μι", "ω")


def _verb_stem(verb: str) -> str:
    """A stripped verb without its citation ending (``βαλλω`` -> ``βαλλ``)."""
    for ending in _CITATION_ENDINGS:
        if verb.endswith(ending):
            return verb[: len(verb) - len(ending)]
    return verb


def _split_lemma_preverbs(lemma: str) -> tuple[list[str], str]:
    """The preverbs of a stripped lemma and the simple verb after them
    (``καταφερω`` -> ``(["κατα"], "φερω")``). The simple verb must keep a
    stem of at least two letters, so ἄγω is not ἀ- plus a verb, and ἄντομαι
    not ἀντ- plus -ομαι."""
    chain: list[str] = []
    rest = lemma
    while True:
        for spelling, before, preverb in _PREVERBS_LONGEST_FIRST:
            after = rest[len(spelling):]
            if (rest.startswith(spelling) and len(after) >= 3
                    and len(_verb_stem(after)) >= 2
                    and (before is None or after[0] in before)):
                chain.append(preverb)
                rest = rest[len(spelling):]
                break
        else:
            return chain, rest


def _split_form_preverbs(form: str, chain: list[str]) -> tuple[str, str] | None:
    """A stripped form split after the lemma's preverbs, in whatever spelling
    the form gives each one, or None when the form does not begin with them."""
    prefix, rest = "", form
    for preverb in chain:
        for spelling, _ in sorted(_PREVERB_SPELLINGS[preverb],
                                  key=lambda row: -len(row[0])):
            if rest.startswith(spelling) and len(rest) > len(spelling):
                prefix += spelling
                rest = rest[len(spelling):]
                break
        else:
            return None
    return prefix, rest


def _augmented_starts(rest: str) -> list[str]:
    """The augmented spellings of a verb stem that has none (Smyth 429-437):
    ε- before a consonant (ἐρρ- before ρ), and the vowel lengthened, α and ε
    to η, ο to ω, αι to ῃ, οι to ῳ, αυ and ευ to ηυ; ε also to ει (εἶχε).
    Empty for a stem whose augment does not show (ι, υ, η, ω, ει)."""
    if not rest:
        return []
    first, two = rest[0], rest[:2]
    if first == "ρ":
        return ["ερρ" + rest[1:], "ερ" + rest[1:]]
    if first not in _VOWEL_LETTERS:
        return ["ε" + rest]
    if two == "ει":
        return []
    if two == "αι":
        return ["η" + rest[1:], "η" + rest[2:]]
    if two == "οι":
        return ["ω" + rest[1:], "ω" + rest[2:]]
    if two in ("αυ", "ευ"):
        return ["ηυ" + rest[2:]]
    if first == "α":
        return ["η" + rest[1:]]
    if first == "ε":
        return ["η" + rest[1:], "ει" + rest[1:]]
    if first == "ο":
        return ["ω" + rest[1:]]
    return []


def _common_prefix_length(a: str, b: str) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def _accent_held_by_augment(form: str, rest: str) -> bool:
    """Whether the accent of a past spelling in -ε stops short of where a
    verb's recessive accent goes, which only the augment does: the accent
    of a compound cannot go back past it, and a long vowel the augment
    leaves as it is takes the accent (ὑπεῖκε beside the imperative ὕπεικε,
    συνεξεῦρε, Doric συνᾶγε; Smyth 426). That is a word of three or more
    syllables with its main accent on the penult, where the penult belongs
    to the verb after its preverbs (``rest``), since a compound imperative
    of a one-syllable verb keeps the accent on the preverb (ἐπίσχες)."""
    nfd = unicodedata.normalize("NFD", form)
    spans = _nuclei(nfd)
    if len(spans) < 3 or len(_nuclei(rest)) < 2:
        return False
    accents = (_COMBINING_ACUTE, _COMBINING_CIRCUMFLEX)
    for index, (start, end) in enumerate(spans):
        if any(a in nfd[start:end] for a in accents):
            return index == len(spans) - 2
    return False


# Crasis joins καί, the article or a preposition to the verb and hides where
# the augment would go (κἄκλαε, χὑπέμεινε, θἀτέρου).
_CRASIS_OPENINGS = ("κα", "χα", "χυ", "κυ", "θα", "χη", "κη")


def _unaugmented(form: str, lemma: str,
                 lemmas_of: dict[str, set[str]]) -> bool:
    """Whether a 3sg past spelling in -ε of ``lemma`` lacks its augment, which
    makes it the same spelling as the 2sg imperative of the same stem
    (κατένεγκε, the aorist of καταφέρω beside κατήνεγκε). A compound
    augments after its last preverb (Smyth 450). A consonant-initial stem
    shows the missing ε-, and an α-, ο-, αι-, αυ- or οι-initial stem its
    unlengthened vowel. An ε-initial or suppletive stem (ἐνεγκ- of φέρω)
    cannot show it, so the augmented spelling has to be filed under the
    same lemma, or under the simple verb (ἤνεγκε under φέρω).
    ``lemmas_of`` maps a stripped spelling to the stripped lemmas the
    corpora file it under. False whenever the augment cannot be seen:
    an ι-, υ-, η- or ω-initial stem, a crasis, or a reduplicated stem."""
    marks = "".join(ELISION_GLYPHS) + "'"
    spelled = _strip_lower(form).lstrip(marks)
    prodelided = form[:1] in marks
    lemma_s = _strip_lower(lemma)
    if spelled.startswith(_CRASIS_OPENINGS) and not lemma_s.startswith(spelled[:2]):
        return False
    chain, stem = _split_lemma_preverbs(lemma_s)
    if prodelided and spelled[:1] not in _VOWEL_LETTERS and not (
            chain and chain[0].startswith("ε")):
        return False  # ’βάδιζε: what the mark stands for is the augment
    split = _split_form_preverbs(spelled, chain)
    if split is None and spelled[:1] not in _VOWEL_LETTERS:
        # Prodelision took the ε of ἐξ- (’ξένεγκε, Smyth 76).
        split = _split_form_preverbs("ε" + spelled, chain)
        if split is not None:
            spelled = "ε" + spelled
    if split is None or not split[1] or not stem:
        return False
    prefix, rest = split
    if chain and chain[-1] == "προ" and prefix.endswith("πρου"):
        return False  # προὔβαινε: the augment is in the crasis
    if _accent_held_by_augment(form, rest):
        return False
    # The spelling has to be the lemma's own stem before its first letter
    # says anything: ἐπεστελλε or ἀνεωγε filed under a lemma it only begins
    # like is left to the augmented-partner test.
    own_stem = _common_prefix_length(rest, _verb_stem(stem)) >= len(_verb_stem(stem)) - 1
    initial = stem[0]
    if initial not in _VOWEL_LETTERS:
        if rest.startswith("ε" + initial) or (initial == "ρ" and rest.startswith("ερρ")):
            return False
        if rest[0] == initial and own_stem:
            reduplicated = len(rest) > 2 and rest[1] == "ε" and rest[2] == initial
            return not reduplicated
    if stem[:2] in ("αι", "οι", "αυ") or initial in "αο":
        head = stem[:2] if stem[:2] in ("αι", "οι", "αυ") else initial
        if rest.startswith(head) and own_stem:
            return True
        if rest[0] in "ηω":
            return False
    if initial == "ε" and stem[1:2] not in ("ι", "υ"):
        if rest.startswith("ει") or rest[0] == "η":
            return False
    # The augmented partner, after the same preverbs in any of their
    # spellings (ἐξένεγκε beside ἐξήνεγκε), filed under the same lemma.
    heads = [""]
    if chain:
        last = chain[-1]
        before_last = prefix
        for spelling, _ in sorted(_PREVERB_SPELLINGS[last], key=lambda row: -len(row[0])):
            if prefix.endswith(spelling):
                before_last = prefix[: len(prefix) - len(spelling)]
                break
        heads = [before_last + spelling for spelling, _ in _PREVERB_SPELLINGS[last]]
    for augmented in _augmented_starts(rest):
        for head in heads:
            partner = head + augmented
            if partner == spelled:
                continue
            if any(lemma_s in lemmas_of.get(p, ()) for p in (partner, partner + "ν")):
                return True
    if chain:
        for augmented in _augmented_starts(rest):
            if any(stem in lemmas_of.get(p, ()) for p in (augmented, augmented + "ν")):
                return True
    return False


# Aorists whose imperative is spelled otherwise (λῦσε beside λῦσον, θῆκε
# beside θές) and the iteratives, which have none.
_NOT_IMPERATIVE_SHAPED = ("σε", "ξε", "ψε", "σκε", "ηκε", "ωκε")


# The vowel a liquid verb's first aorist lengthens its stem vowel to
# (Smyth 544): α to η, or to ᾱ in some verbs in -αίνω (ἐγλύκᾱνα), ε to ει;
# ι and υ lengthen without a change of spelling.
_LIQUID_AORIST_VOWELS = {"αι": ("η", "α"), "α": ("η",), "ε": ("ει",),
                         "ει": ("ει",), "ι": ("ι",), "υ": ("υ",)}


def _liquid_first_aorist(stripped: str, lemma: str) -> bool:
    """Whether an aorist spelling in -ε is a liquid verb's first aorist
    spelled otherwise than the present stem (σήμηνε of σημαίνω, ἔχθηρε of
    ἐχθαίρω, πῆλε of πάλλω, στεῖλε of στέλλω, μεῖνε of μένω): its imperative
    is in -ον, so no imperative shares the spelling. When the lengthening
    does not show (κτεῖνε, κρῖνε), the spelling is the present imperative's,
    and a thematic second aorist (βάλε, τάμε) keeps its short vowel."""
    present = _verb_stem(_strip_lower(lemma))
    if present.endswith("λλ"):
        liquid, liquid_aorist = "λλ", "λ"
    elif present[-1:] in ("λ", "μ", "ν", "ρ"):
        liquid = liquid_aorist = present[-1]
    else:
        return False
    before = present[: len(present) - len(liquid)]
    vowel = before[-2:] if before[-2:] in _LIQUID_AORIST_VOWELS else before[-1:]
    for lengthened in _LIQUID_AORIST_VOWELS.get(vowel, ()):
        aorist = lengthened + liquid_aorist
        if aorist != vowel + liquid and stripped[:-1].endswith(aorist):
            return True
    return False


def _imperative_shaped(stripped: str, lemma: str, tense: str) -> bool:
    """Whether a past spelling could be an imperative by its ending. An
    imperfect is the present stem plus ε, which is the present imperative
    whatever the stem ends in (δίδασκε, πρᾶσσε beside διδάσκω, πράσσω);
    otherwise a σ- or κ-aorist, a liquid first aorist whose stem vowel shows
    its lengthening, or an iterative in -σκε is not."""
    lemma_s = _strip_lower(lemma)
    if (tense == "imperfect" and lemma_s.endswith("ω") and len(lemma_s) > 2
            and stripped[:-1].endswith(lemma_s[-3:-1])):
        return True
    if tense == "aorist" and _liquid_first_aorist(stripped, lemma):
        return False
    return not stripped.endswith(_NOT_IMPERATIVE_SHAPED)


def _past_tense(tags: set[str]) -> str:
    """The tense of an indicative analysis; empty for any other mood (the
    optative in -ειε has no augment to lose)."""
    if "indicative" not in tags:
        return ""
    for tense in ("imperfect", "aorist", "pluperfect", "perfect"):
        if tense in tags:
            return tense
    return ""


def _attestation_key(form: str) -> str:
    """``form`` without the marks that do not make another word: an
    enclitic's second accent, a diaeresis and an iota subscript, so that
    ἀπάλλαττέ counts as written with the ν when ἀπάλλαττεν is."""
    out: list[str] = []
    accented = False
    for c in unicodedata.normalize("NFD", form):
        if c in (_COMBINING_DIAERESIS, _COMBINING_YPOGEGRAMMENI):
            continue
        if c in (_COMBINING_ACUTE, _COMBINING_GRAVE, _COMBINING_CIRCUMFLEX):
            if accented:
                continue
            accented = True
        out.append(c)
    return _nfc("".join(out))


def _imperative_homograph(form: str, readings, lemmas_of: dict[str, set[str]],
                          written_with_nu: set[str]) -> bool:
    """Whether a spelling in -ε that some analysis reads as a 3sg past is
    also the 2sg imperative, and the texts never write the past with its ν.

    Movable nu goes on the past indicative and never on the imperative
    (Smyth 134). The augment tells them apart, and epic, lyric and Ionic
    leave it off (Smyth 438), so an unaugmented imperfect or thematic
    aorist is the imperative's spelling (φέρε, κατένεγκε). The taggers do
    not separate them: Morpheus lists no imperative for the ἐνεγκ-
    compounds, and GLAUx's tagger files the imperatives of Lucian's
    κατένεγκε and Galen's recipes (προσέμβαλλε, ἐπέμβαλλε) as indicatives.
    GLAUx writes the ν on 99.6% of the 51,949 3sg past indicatives in -ε
    that stand before a vowel (only Diorisis's Herodotus, at 0.1%, prints
    none), so a past whose spelling with ν is attested nowhere hardly ever
    stood before a vowel as an indicative, and what stood there bare was
    the imperative. Such a spelling is dropped when every reading that
    takes nu is an unaugmented imperfect or aorist (``_unaugmented``); the
    token vote still decides the ones written with ν somewhere.
    ``written_with_nu`` holds the ``_attestation_key`` of every spelling the
    corpora write with the ν, without it."""
    stripped = _strip_lower(form)
    if not readings:
        return False
    if not all(tense in ("imperfect", "aorist") and lemma
               and _imperative_shaped(stripped, lemma, tense)
               and _unaugmented(form, lemma, lemmas_of)
               for lemma, tense in readings):
        return False
    pooled = {form, form[:1].lower() + form[1:]}
    return not any(_attestation_key(spelling) in written_with_nu
                   for spelling in pooled)


def _entry_analyses(entry: dict) -> list[tuple[str, set[str], int]]:
    """The token-counted analyses of a pairs entry: every analysis with its
    count when its tokens carry several, else its one analysis with its
    count. Empty for a pairs file without counts."""
    if "analyses" in entry:
        return [(pos or "", set(tags), n) for pos, tags, n in entry["analyses"]]
    if "count" in entry:
        return [(entry.get("pos", "") or "", set(entry.get("tags", [])),
                 entry["count"])]
    return []


def _derive_nu_forms(pairs_files: list[Path]) -> set[str]:
    """Union set of NFC surface forms eligible for movable nu.

    A spelling is eligible when some analysis of it takes movable nu
    (``_nu_reading``) and the texts do not mostly mean another one. Most
    spellings mean one thing, but φέρε is an imperative 956 times in GLAUx
    and the unaugmented imperfect 45, and the imperative takes no nu; a
    pairs file keeps the first analysis it meets for each form and lemma,
    so presence alone put φέρε, ἄκουε, ἴδε and some 130 other imperatives
    on the list and the keyboard wrote φέρεν εἰπέ. So the token-counted
    analyses vote: those that take nu, with the tokens of the spelling that
    already carries it, against those that cannot, and a spelling the
    ``never`` readings outnumber is dropped. μέλλε is an imperative 14
    times and an imperfect 7, but the imperfect is spelled μέλλεν 17 more
    times before a vowel, so it stays. Votes pool a word's capitalized and
    lowercase spellings for a capitalized one, since a keyboard reads a
    capital through the lowercase entry, but a lowercase word counts only
    its lowercase spellings, so the vocative of a name (Κέλσε) does not
    veto the verb (κέλσε). GLAUx is the only source tagged token by token;
    Diorisis lists every candidate analysis of a spelling on each token, so
    its analyses make a spelling eligible but cannot vote.

    A spelling in -σι, -ξι or -ψι is a dative plural or a third person,
    both of which take nu, so an analysis that says otherwise is a tagging
    slip (GLAUx files one ζηλῶσί as a first person) and abstains. An
    adverb in -σι, or an adverb of place in -θε, takes nu when the texts
    write it with one in at least a tenth of its tokens (παντάπασιν 70%,
    Ἀθήνησιν 42%; Smyth 134, and 134 D for πρόσθε(ν)); the deictic
    οὑτωσί is written οὑτωσίν twice in 451, a slip. The -θε adverbs that
    qualify are written with the ν in at least a fifth of their tokens and
    the rest never, so the threshold does not choose among them.

    A spelling the orthography rules reject is never eligible. There is no
    fallback to raw frequency counts, because corpus co-occurrence alone
    produces too many false positives on neuter nominative participles
    (e.g. ``γραφέν``) that share the ``-ε`` surface with a 3sg past.
    """
    def key(bare: str) -> str:
        return bare.lower()

    entries_by_file = []
    for p in pairs_files:
        with open(p, encoding="utf-8") as f:
            entries_by_file.append(json.load(f))

    # [takes, never] over every spelling of a word, and over its
    # lowercase spellings alone.
    votes: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    lowercase_votes: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    # Adverbs in -σι: tokens without the ν and with it.
    adverb_tokens: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    candidates: dict[str, bool] = {}  # form -> some analysis takes nu
    # The lemma and tense of every reading of a spelling in -ε that takes
    # nu, every spelling the corpora write, and the lemmas each stripped
    # spelling is filed under: what the imperative test below reads.
    past_readings: dict[str, set[tuple[str, str]]] = defaultdict(set)
    spellings: set[str] = set()
    lemmas_of: dict[str, set[str]] = defaultdict(set)
    for entries in entries_by_file:
        for entry in entries:
            form = _nfc(entry.get("form", "").strip())
            if not form or len(form) < 2 or has_editorial_sigla(form):
                continue
            lemma = _nfc(entry.get("lemma", "").strip())
            spellings.add(form)
            if lemma:
                lemmas_of[_strip_lower(form)].add(_strip_lower(lemma))
            carries_nu = form.endswith("ν") or form.endswith("Ν")
            bare = form[:-1] if carries_nu else form
            stripped = _strip_lower(bare)
            plural_ending = stripped.endswith(("σι", "ξι", "ψι"))
            analyses = _entry_analyses(entry)
            first = (entry.get("pos", "") or "", set(entry.get("tags", [])))
            takes = False
            lowercase = not bare[:1].isupper()
            for pos, tags, n in analyses:
                reading = _nu_reading(stripped, pos, tags)
                slot = None
                if reading == "takes":
                    slot = 0
                    takes = True
                    if not carries_nu and stripped.endswith("ε"):
                        past_readings[form].add((lemma, _past_tense(tags)))
                elif (reading == "never" and not carries_nu
                        and not plural_ending):
                    slot = 1
                if slot is not None:
                    votes[key(bare)][slot] += n
                    if lowercase:
                        lowercase_votes[key(bare)][slot] += n
                if _alternating_adverb(stripped, pos):
                    adverb_tokens[key(bare)][carries_nu] += n
            if carries_nu:
                continue
            if _nu_reading(stripped, *first) == "takes":
                takes = True
                if stripped.endswith("ε"):
                    past_readings[form].add((lemma, _past_tense(first[1])))
            if _alternating_adverb(stripped, first[0]):
                candidates.setdefault(form, False)
            if takes:
                candidates[form] = True

    written_with_nu = {_attestation_key(s[:-1]) for s in spellings
                       if s.endswith(("ν", "Ν"))}

    def adverb_takes_nu(form: str) -> bool:
        without, with_nu = adverb_tokens[key(form)]
        return with_nu > 0 and 10 * with_nu >= without

    nu_eligible: set[str] = set()
    for form, takes in candidates.items():
        if not takes and not adverb_takes_nu(form):
            continue
        # A malformed spelling gets nothing: a ν appended to ἒστι or
        # λέγουσὶ is written into the user's text as it stands. A
        # prodelided form (’στι for ἐστι) is checked with the mark in
        # the dictionary's own glyph.
        probe = KORONIS + form[1:] if form[0] in ELISION_GLYPHS else form
        if grc_orthography_reason(probe) is not None:
            continue
        pool = votes if form[:1].isupper() else lowercase_votes
        takes_votes, never_votes = pool[key(form)]
        if never_votes > takes_votes:
            continue
        if _imperative_homograph(form, past_readings.get(form, ()), lemmas_of,
                                 written_with_nu):
            continue
        nu_eligible.add(form)

    # Closed-list numerals. Only ``εἴκοσι`` survives the morphological
    # audit; other -κοντα cardinals end in α, not ι / ε, so movable nu
    # doesn't apply.
    for w in EXTRA_NU_FORMS:
        nu_eligible.add(_nfc(w))

    return nu_eligible


def _require_token_counts(glaux_pairs: Path) -> None:
    """Stop when the GLAUx pairs file predates its token counts: without
    them nothing votes on movable nu, and the imperatives come back
    without a word of warning."""
    with open(glaux_pairs, encoding="utf-8") as f:
        head = f.read(1 << 16)
    if '"count":' not in head:
        raise SystemExit(
            f"{glaux_pairs} has no token counts. Rebuild it with "
            "build/build_glaux_pairs.py, or download the current file."
        )


def _dative_key(form: str) -> str:
    """``form`` with a grave read as an acute and its case kept: the key the
    dative test looks spellings up by."""
    return _nfc(unicodedata.normalize("NFD", form).replace("\u0300", "\u0301"))


_COMBINING_DIAERESIS = unicodedata.lookup("COMBINING DIAERESIS")
_COMBINING_ACUTE = unicodedata.lookup("COMBINING ACUTE ACCENT")
_COMBINING_GRAVE = unicodedata.lookup("COMBINING GRAVE ACCENT")
_COMBINING_CIRCUMFLEX = unicodedata.lookup("COMBINING GREEK PERISPOMENI")
_COMBINING_YPOGEGRAMMENI = unicodedata.lookup("COMBINING GREEK YPOGEGRAMMENI")


def _uncontracted_ei(form: str) -> bool:
    """Whether ``form`` ends in ε and a ι written apart with a diaeresis
    (κήδεϊ, ἔγχεϊ): the uncontracted dative singular of an s-, u- or
    eu-stem."""
    nfd = unicodedata.normalize("NFD", form)
    return (_strip_lower(form).endswith("ει")
            and nfd.rstrip(_COMBINING_ACUTE + _COMBINING_GRAVE)
                   .endswith(_COMBINING_DIAERESIS))


def _derive_dative_keys(pairs_files: list[Path]) -> dict[str, bool]:
    """For each spelling the tagged corpora analyze (``_dative_key``),
    whether every analysis of it is a dative. ``_dative_analysis`` reads it.

    The dative -ι and -σι elide only in epic (Smyth 72), and rarely there:
    GLAUx elides them in about 2% of epic tokens before a vowel and 0.2% of
    the rest, and the elided spelling it does attest is nearly always
    another case's (ἄνδρ᾽ is ἄνδρα, πάντ᾽ is πάντα). The spelling does not
    say which forms are datives, so the tagged corpora do. The first file
    that has a form decides, because the treebanks disagree on some:
    Diorisis files λέγουσι only as a dative participle, GLAUx only as the
    verb, and ἀνδρί once as a vocative of ἀνδρίς. The pair files keep the
    first analysis they met for each form and lemma, so a form that is
    both a verb and a dative participle (θέλουσι) follows that one; at
    worst that costs a poetic elision of the verb, which movable nu, the
    commoner spelling before a vowel, would take anyway.

    Diorisis files the uncontracted datives in -εϊ (κήδεϊ, κράτεϊ) as
    nominative duals, the analysis of γένει, which is contracted from
    γένεε. The diaeresis says the ε and ι were not contracted, so a dual
    analysis of such a spelling is set aside, and a spelling left with no
    other analysis counts as unanalyzed.
    """
    analyses: dict[str, list[set[str]]] = {}
    for p in pairs_files:
        seen: dict[str, list[set[str]]] = defaultdict(list)
        with open(p, encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            form = _nfc(entry.get("form", "").strip())
            if not form or has_editorial_sigla(form):
                continue
            tags = set(entry.get("tags", []))
            if "dual" in tags and _uncontracted_ei(form):
                continue
            seen[_dative_key(form)].append(tags)
        for key, tag_sets in seen.items():
            analyses.setdefault(key, tag_sets)
    return {key: all("dative" in tags for tags in tag_sets)
            for key, tag_sets in analyses.items()}


def _dative_analysis(full: str, dative_keys: dict[str, bool], *,
                     fold: bool) -> bool | None:
    """Whether ``full`` is a dative, by the analyses of the spelling nearest
    to it, or None when the tagged corpora analyze no spelling of it. Case
    is kept apart because a capitalized homograph carries its own analyses,
    some of them junk: GLAUx tags a Χερσὶ with no case and a Φανέντι as a
    nominative noun, and either would veto the lowercase dative. So a
    capitalized word is read through its lowercase spelling first, and a
    lowercase one through its own, then its capitalized one (πλάτωνι is
    analyzed only as Πλάτωνι). With ``fold``, a spelling with a diaeresis
    is also read without it, which the caller does only after every other
    test, since ἔγχεϊ is a dative and ἔγχει an imperative."""
    own = _dative_key(full)
    lower = _dative_key(full.lower())
    capital = _dative_key(full[:1].title() + full[1:])
    order = [lower, own] if own != lower else [own, capital]
    for key in order:
        if key in dative_keys:
            return dative_keys[key]
    if fold:
        for key in order:
            folded = _nfc(unicodedata.normalize("NFD", key)
                          .replace(_COMBINING_DIAERESIS, ""))
            if folded != key and folded in dative_keys:
                return dative_keys[folded]
    return None


# A lemma cited in one of these is a verb.
_VERB_CITATION_ENDINGS = ("ω", "μι", "μαι")


def _mark_groups(form: str) -> list[tuple[str, str]]:
    """Each base letter of ``form``, lowercased, with the marks on it."""
    groups: list[list[str]] = []
    for c in unicodedata.normalize("NFD", form):
        if unicodedata.combining(c) and groups:
            groups[-1][1] += c
        else:
            groups.append([c.lower(), ""])
    return [(letter, marks) for letter, marks in groups]


def _third_declension_dative_plurals(stem: str) -> dict[str, tuple[str, str]]:
    """The dative plural spellings a third-declension stem gives, each with
    the kind of stem and, for a -ντ- stem, what precedes its vowel. Before
    the σ of -σι a dental or ν drops, ντ drops and lengthens the vowel
    before it (ο to ου, ε to ει, α to ᾱ), a labial gives ψ and a velar ξ
    (Smyth 100); epic also writes -σσι and -εσσι."""
    out: dict[str, tuple[str, str]] = {}
    if stem.endswith("ντ"):
        before = stem[:-2]
        if before.endswith(("ου", "ω", "υ")):
            out[before + "σι"] = ("contract", "")
        elif before.endswith("ο"):
            out[before[:-1] + "ουσι"] = ("o", before[:-1])
        elif before.endswith("ε"):
            out[before[:-1] + "εισι"] = ("athematic", before[:-1])
        elif before.endswith("α"):
            out[before[:-1] + "ασι"] = ("athematic", before[:-1])
        return out
    last = stem[-1:]
    if last in "τδθν":
        out[stem[:-1] + "σι"] = ("nominal", "")
        if last in "τδ":
            out[stem[:-1] + "σσι"] = ("nominal", "")
    elif last in "κγχ":
        out[stem[:-1] + "ξι"] = ("nominal", "")
    elif last in "πβφ":
        out[stem[:-1] + "ψι"] = ("nominal", "")
    elif last == "ρ":
        out[stem + "σι"] = ("nominal", "")
    elif last in _VOWEL_LETTERS:
        out[stem + "σι"] = ("nominal", "")
        out[stem + "σσι"] = ("nominal", "")
    out.setdefault(stem + "εσσι", ("nominal", ""))
    return out


def _lemma_reads_dative(full: str, lemma: str, forms: set[str]) -> bool:
    """Whether the forms of ``lemma`` make ``full``, a spelling in -ι, a
    dative.

    A third-declension dative singular is the genitive's stem plus ι
    (δμητῆρι beside δμητῆρος), so it is one when the lemma has that
    genitive as a form other than its citation form (which keeps out an
    adverb in -ί filed under its adjective, ἀμισθί under ἄμισθος). A
    spelling in -εϊ is the uncontracted dative of an s-, u- or eu-stem
    (μήδεϊ of μῆδος, ταχέϊ of ταχύς). A dative plural is the stem plus -σι
    as ``_third_declension_dative_plurals`` spells it, from a stem that
    shows the genitive in -ος and one more third-declension case (-ι, -α or
    -ες), so that ἑσταός alone does not make a stem.

    The dative plural of a present or future participle is also the verb's
    third person plural (λύουσι, τιμῶσι, ποιοῦσι, βαλοῦσι, ἱστᾶσι, τιθεῖσι,
    διδοῦσι); an aorist or perfect participle's is not. So for a verb
    lemma a -σι reading of a contract -ντ- stem, of an -αντ- or -εντ- stem
    whose verb is in -ημι, of an -οντ- stem of a verb in -ωμι, or of any
    other -οντ- stem is not a dative, unless the form is accented like a
    second aorist, on the ending in both the genitive (λαβόντος) and the
    dative (λαβοῦσι), and the lemma has no liquid future in -οῦμεν."""
    spelled, lemma_s = _strip_lower(full), _strip_lower(lemma)
    stripped_forms = {_strip_lower(f) for f in forms} | {lemma_s}
    if _uncontracted_ei(full):
        if (lemma_s in {spelled[:-2] + e for e in ("ος", "ης", "υς", "ευς")}
                or {spelled[:-1] + "ος", spelled[:-1] + "ως",
                    spelled[:-2] + "ους"} & stripped_forms):
            return True
    elif (spelled[:-1] + "ος" != lemma_s
            and spelled[:-1] + "ος" in stripped_forms):
        return True
    if not spelled.endswith(("σι", "ξι", "ψι")):
        return False
    verb = lemma_s.endswith(_VERB_CITATION_ENDINGS)
    found = False
    for genitive in stripped_forms:
        if not genitive.endswith("ος") or len(genitive) < 4:
            continue
        stem = genitive[:-2]
        if not {stem + "ι", stem + "α", stem + "ες"} & stripped_forms:
            continue
        kind = _third_declension_dative_plurals(stem).get(spelled)
        if not kind:
            continue
        found = True
        if not verb:
            continue
        stem_kind, before = kind
        if stem_kind == "contract":
            return False
        if stem_kind == "athematic" and before + "ημι" in stripped_forms:
            return False
        if stem_kind == "o":
            if before + "ωμι" == lemma_s:
                return False
            marks = _mark_groups(full)
            ending_accent = (len(marks) >= 4 and marks[-4][0] == "ο"
                             and marks[-3][0] == "υ"
                             and _COMBINING_CIRCUMFLEX in marks[-3][1])
            genitive_on_ending = any(
                _strip_lower(f) == genitive and len(g := _mark_groups(f)) >= 5
                and g[-5][0] == "ο"
                and (_COMBINING_ACUTE in g[-5][1] or _COMBINING_GRAVE in g[-5][1])
                for f in forms)
            if (not (ending_accent and genitive_on_ending)
                    or before + "ουμεν" in stripped_forms):
                return False
    return found


def _paradigm_dative(full: str, lemmas: set[str],
                     lemma_to_forms: dict[str, set[str]]) -> bool:
    """Whether every lemma ``full`` is filed under reads it as a dative
    (``_lemma_reads_dative``). A lemma that does not leaves room for
    another reading, which keeps the elision."""
    return bool(lemmas) and all(
        _lemma_reads_dative(full, lemma, lemma_to_forms[lemma])
        for lemma in lemmas)


def _load_lemma_forms(
    pairs_files: list[Path], lookup_db: Path | None,
) -> dict[str, set[str]]:
    """Collect {lemma -> set of NFC surface forms} from all sources.

    Merging morpho-tagged corpora and the Wiktionary-expanded lookup
    table gives us the widest possible net for elision-pair discovery.
    The lookup table contributes forms that corpus tagging misses,
    typically rarer inflections that a Wiktionary Lua paradigm
    generated. Malformed tokens (OCR artifacts with brackets or
    newlines) are dropped.
    """
    lemma_to_forms: dict[str, set[str]] = defaultdict(set)

    for p in pairs_files:
        with open(p, encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            form = _nfc(entry.get("form", "").strip())
            lemma = _nfc(entry.get("lemma", "").strip())
            if (not form or not lemma
                    or has_editorial_sigla(form)
                    or has_editorial_sigla(lemma)
                    or any(c in form or c in lemma for c in "{}<>\n")):
                continue
            lemma_to_forms[lemma].add(form)

    if lookup_db is not None and lookup_db.exists():
        conn = sqlite3.connect(str(lookup_db))
        cur = conn.cursor()
        rows = cur.execute(
            "SELECT k.form, l.text FROM lookup k "
            "JOIN lemmas l ON k.lemma_id = l.id WHERE k.src='grc'"
        )
        for form, lemma in rows:
            nfc = _nfc(form)
            lemma_nfc = _nfc(lemma)
            if (not nfc or not lemma_nfc
                    or has_editorial_sigla(nfc)
                    or has_editorial_sigla(lemma_nfc)
                    or any(c in nfc or c in lemma_nfc for c in "{}<>\n")):
                continue
            lemma_to_forms[lemma_nfc].add(nfc)
        conn.close()

    return lemma_to_forms


def _derive_elision_pairs(
    lemma_to_forms: dict[str, set[str]],
    dative_keys: dict[str, bool] | None = None,
) -> dict[str, str]:
    """Return {full_form_nfc -> elided_form_nfc_with_koronis}.

    For each lemma we partition its attested forms into *fulls* (end
    without an elision glyph) and *elideds* (end with one of the four
    elision glyphs; we canonicalise all four to U+1FBD on storage). A
    full form F matches an elided form E when:

      - ``strip(F)`` starts with ``strip(E_stem)`` where ``E_stem`` is
        E minus its final koronis;
      - ``strip(F)`` is exactly one character longer than
        ``strip(E_stem)``, i.e. F has exactly one extra base
        character relative to the elided stem;
      - that extra character is one of the seven Greek vowels, so F's
        final vowel is what elision dropped.

    A full form can belong to several lemmas, so its candidates are pooled
    across all of them, each first taking the full form's case
    (``_match_initial_case``). The rules of elision then say what the
    elided spelling must be (``_rule_elision``), and the pair is emitted
    only when the corpora attest exactly that spelling: the evidence says
    whether the word elides, the rules say how. Canonical overrides for
    the ten iconic particles are applied last so the textbook forms always
    win regardless of corpus noise.
    """
    candidates_by_full: dict[str, set[str]] = defaultdict(set)
    # The Attic accusative of a noun in -εύς ends in a long ᾱ (βασιλέᾱ,
    # Smyth 276), which cannot elide; the spelling does not show it, the
    # lemma does.
    long_final: set[str] = set()

    for lemma, forms in lemma_to_forms.items():
        if has_editorial_sigla(lemma):
            continue
        eus = _strip_lower(lemma).endswith("ευς")
        elideds = set()
        fulls: list[str] = []
        for f in forms:
            if not f or has_editorial_sigla(f):
                continue
            if f[-1] in ELISION_GLYPHS:
                elideds.add(_canon_elided(f))
            else:
                fulls.append(f)
        if not elideds:
            continue
        for full in fulls:
            # A key has to be something a keyboard reads as one word, which
            # a prodelided ᾽στί is not.
            if len(full) < 2 or not full[0].isalpha():
                continue
            # The keyboard rewrites the user's text with this table, so a
            # word elision cannot touch has no business carrying an entry,
            # whatever spelling the corpus happens to pair with it.
            if not _can_elide(full):
                continue
            stripped_full = _strip_lower(full)
            if eus and stripped_full.endswith("εα"):
                long_final.add(full)
            # Find elided candidates whose stripped stem is exactly
            # one vowel shorter than the full form.
            for e in elideds:
                stem_e = e[:-1]  # drop koronis
                if not stem_e:
                    continue
                stripped_stem = _strip_lower(stem_e)
                if (stripped_full.startswith(stripped_stem)
                        and len(stripped_full) - len(stripped_stem) == 1):
                    if stripped_full[-1] in "αεηιουω":
                        candidates_by_full[full].add(
                            _match_initial_case(e, full))

    # Decided only once every lemma has contributed. Assigning per lemma
    # let the last lemma to claim a form win: the corpora carry a
    # capitalized lemma Κατά whose one elided token opened a sentence, and
    # it overwrote κατὰ's κατ᾽ with Κατ᾽.
    #
    # The corpus candidates were ranked here until the rules proved they
    # only ever broke ties among spellings that were often all wrong: a
    # long final α took the elided neuter plural (αἰτία -> αἴτι᾽), and an
    # oxytone the circumflex of its accusative (γυναικί -> γυναῖκ᾽). A
    # keyboard writes the value into the user's text, and offering nothing
    # is better than offering another word.
    # Only the dative's own -ι is tested: τῷδε and ἔμοιγε are datives too,
    # but what they lose is the ε of δε and γε, which elides like any other.
    # A spelling in -ι that no tagged corpus analyzes (the lookup table's
    # paradigm expansions: δμητῆρι, λυθεῖσι) is read through the forms of
    # the lemmas it is filed under.
    iota_final = [full for full in candidates_by_full
                  if _final_vowel(full)[1] == "ι"]
    unanalyzed = {full for full in iota_final
                  if not dative_keys
                  or _dative_analysis(full, dative_keys, fold=False) is None}
    wanted = unanalyzed | {full.lower() for full in unanalyzed}
    lemmas_of: dict[str, set[str]] = defaultdict(set)
    for lemma, forms in lemma_to_forms.items():
        for form in wanted.intersection(forms):
            lemmas_of[form].add(lemma)

    def dative(full: str) -> bool:
        if full not in unanalyzed:
            return bool(_dative_analysis(full, dative_keys, fold=False))
        if _paradigm_dative(full, lemmas_of[full] | lemmas_of[full.lower()],
                            lemma_to_forms):
            return True
        return bool(dative_keys
                    and _dative_analysis(full, dative_keys, fold=True))

    pairs: dict[str, str] = {}
    for full, candidates in candidates_by_full.items():
        if full in long_final:
            continue
        if _final_vowel(full)[1] == "ι" and dative(full):
            continue
        elided = _rule_elision(full)
        if elided in candidates and grc_orthography_reason(elided) is None:
            pairs[full] = elided

    # Pin canonical overrides last. NFC everything for safety.
    for full, elided in CANONICAL_ELISION_OVERRIDES.items():
        pairs[_nfc(full)] = _nfc(elided)

    return pairs


def modern_only_elisions(
    ancient: dict[str, str], modern: dict[str, str],
) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    """The Modern Greek pairs ``ancient`` lacks or spells otherwise, and
    those conflicts: ``{full: (ancient elided, modern elided)}``."""
    only = {full: elided for full, elided in modern.items()
            if ancient.get(full) != elided}
    conflicts = {full: (ancient[full], elided) for full, elided in only.items()
                 if full in ancient}
    return dict(sorted(only.items())), conflicts


def derive_modern_greek_elisions(parquet_path: Path | None = None):
    """Modern Greek elision pairs from the polytonic Modern Greek slice,
    read as ``export_mg_polytonic`` reads it for the word list, and each
    pair's elided share before a vowel."""
    from export_mg_polytonic import (
        count_corpus,
        modern_greek_elision_shares,
        modern_greek_elisions,
        source_documents,
    )
    counts = count_corpus(parquet_path)
    pairs, found = modern_greek_elisions(counts, source_documents(counts))
    return pairs, modern_greek_elision_shares(found)


def build(out_dir: Path, modern_greek: bool = True,
          mg_parquet: Path | None = None) -> dict:
    """Drive the full morphology export end to end and write
    ``<out_dir>/grc_morph.json``. Returns a stats dict.

    ``modern_greek`` adds the ``el_modern`` table, which needs the
    Wikisource parquet ``extract_polytonic_mg`` reads.
    """
    if not GLAUX_PAIRS.exists():
        print(
            f"ERROR: {GLAUX_PAIRS} not found. Download with "
            f"`huggingface-cli download ciscoriordan/dilemma --local-dir . "
            f"--include 'data/*'`",
            file=sys.stderr,
        )
        sys.exit(1)
    _require_token_counts(GLAUX_PAIRS)

    pairs_files = [GLAUX_PAIRS, DIORISIS_PAIRS]
    pairs_files = [p for p in pairs_files if p.exists()]
    print(
        f"Reading morphology tags from {len(pairs_files)} file(s): "
        f"{', '.join(p.name for p in pairs_files)}"
    )

    nu_forms = _derive_nu_forms(pairs_files)
    print(f"  nu-eligible surface forms: {len(nu_forms):,}")

    lemma_to_forms = _load_lemma_forms(
        pairs_files, LOOKUP_DB if LOOKUP_DB.exists() else None
    )
    print(f"  lemmas with attested forms: {len(lemma_to_forms):,}")

    dative_keys = _derive_dative_keys(pairs_files)
    print(f"  forms tagged only as datives: {sum(dative_keys.values()):,}")
    elision_pairs = _derive_elision_pairs(lemma_to_forms, dative_keys)
    print(f"  elision full -> elided pairs: {len(elision_pairs):,}")

    el_modern: dict[str, str] = {}
    if modern_greek:
        mg_pairs, mg_shares = derive_modern_greek_elisions(mg_parquet)
        el_modern, conflicts = modern_only_elisions(elision_pairs, mg_pairs)
        print(f"  Modern Greek elision pairs the Ancient table lacks or "
              f"spells otherwise: {len(el_modern):,}")
        for full, (ancient, modern) in sorted(conflicts.items()):
            print(f"    differs from el: {full} -> {ancient} (Ancient), "
                  f"{modern} (Modern)")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grc_morph.json"
    payload = {
        "version": 1,
        "nu": sorted(nu_forms),
        "el": dict(sorted(elision_pairs.items())),
    }
    if modern_greek:
        payload["el_modern"] = el_modern
        payload["el_modern_share"] = {
            full: mg_shares[full] for full in el_modern}
    # Compact JSON: no extra whitespace, but pretty-ish for diffing.
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            payload, f, ensure_ascii=False, separators=(",", ":"),
            sort_keys=False,
        )
    size = out_path.stat().st_size
    print(f"  wrote {out_path} ({size:,} bytes)")

    return {
        "nu_count": len(nu_forms),
        "elision_count": len(elision_pairs),
        "modern_elision_count": len(el_modern),
        "bytes": size,
        "path": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir", default=str(OUT),
        help=f"Output directory (default: {OUT})",
    )
    ap.add_argument(
        "--no-modern-greek", action="store_true",
        help="leave out the el_modern table (it needs the Wikisource "
             "parquet of the polytonic Modern Greek slice)",
    )
    ap.add_argument(
        "--mg-parquet", type=Path, default=None,
        help="override the Wikisource parquet path",
    )
    args = ap.parse_args()
    build(Path(args.out_dir), modern_greek=not args.no_modern_greek,
          mg_parquet=args.mg_parquet)


if __name__ == "__main__":
    main()
