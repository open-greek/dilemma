#!/usr/bin/env python3
"""Export Ancient Greek boundary-rewrite morphology tables for Tonos.

Produces a single compact JSON at ``build/hunspell/grc_morph.json`` that
the Tonos iOS keyboard reads at install time. The file carries two
boundary-sensitive rewrite tables:

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
  - A small closed list of numerals that historically take movable nu
    (``εἴκοσι``).

  Subjunctive, optative, imperative, infinitive, and participle forms
  are explicitly excluded because Ancient Greek movable nu never
  attaches to them.

* ``el`` - full-form -> elided-form pairs harvested from GLAUx,
  Diorisis, and the canonical dilemma lookup table. The elided form
  is stored in NFC with its final elision glyph canonicalised to
  U+1FBD GREEK KORONIS (same convention as
  ``HunspellCompiler.normalizeEntry`` and ``GreekStyle``'s existing
  hardcoded particle table). The Tonos output layer rewrites the
  koronis to the user's chosen elision glyph via
  ``GreekStyle.applyingElisionMark``.

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


# Tags considered disqualifying for movable nu. Movable nu never
# attaches to subjunctive, optative, imperative, infinitive, or
# participle forms. Indicative past / present (where the letter
# conditions allow) is the carrier.
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

    ``"never"`` for an analysis that cannot take it: another mood or a
    non-finite form, a first or second person, a case other than the
    dative, a dative singular or dual, an indeclinable word in ``-ε``
    (the conjunction ``ηὖτε``, the interjection ``φέρε``). ``None`` when
    the tags do not say,
    as for an untagged adverb (Ἀθήνησι) or a verb form given a voice it
    cannot have (GLAUx files half of ἔσκε as middle).
    """
    last = stripped[-1:]
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
    if pos in ("adv", "conj", "particle", "prep", "intj") and last == "ε":
        return "never"
    return None


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
    adverb in -σι takes nu when the texts write it with one in at least a
    tenth of its tokens (παντάπασιν 70%, Ἀθήνησιν 42%; Smyth 134); the
    deictic οὑτωσί is written οὑτωσίν twice in 451, a slip.

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
    for entries in entries_by_file:
        for entry in entries:
            form = _nfc(entry.get("form", "").strip())
            if not form or len(form) < 2 or has_editorial_sigla(form):
                continue
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
                elif (reading == "never" and not carries_nu
                        and not plural_ending):
                    slot = 1
                if slot is not None:
                    votes[key(bare)][slot] += n
                    if lowercase:
                        lowercase_votes[key(bare)][slot] += n
                if pos == "adv" and stripped.endswith("σι"):
                    adverb_tokens[key(bare)][carries_nu] += n
            if carries_nu:
                continue
            if _nu_reading(stripped, *first) == "takes":
                takes = True
            if first[0] == "adv" and stripped.endswith("σι"):
                candidates.setdefault(form, False)
            if takes:
                candidates[form] = True

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


def _derive_dative_keys(pairs_files: list[Path]) -> dict[str, bool]:
    """For each spelling the tagged corpora analyze (``_dative_key``),
    whether every analysis of it is a dative. ``_is_dative`` reads it.

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
    """
    analyses: dict[str, list[set[str]]] = {}
    for p in pairs_files:
        seen: dict[str, list[set[str]]] = defaultdict(list)
        with open(p, encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            form = _nfc(entry.get("form", "").strip())
            if form and not has_editorial_sigla(form):
                seen[_dative_key(form)].append(set(entry.get("tags", [])))
        for key, tag_sets in seen.items():
            analyses.setdefault(key, tag_sets)
    return {key: all("dative" in tags for tags in tag_sets)
            for key, tag_sets in analyses.items()}


def _is_dative(full: str, dative_keys: dict[str, bool]) -> bool:
    """Whether ``full`` is a dative, by the analyses of the spelling nearest
    to it. Case is kept apart because a capitalized homograph carries its
    own analyses, some of them junk: GLAUx tags a Χερσὶ with no case and a
    Φανέντι as a nominative noun, and either would veto the lowercase
    dative. So a capitalized word is read through its lowercase spelling
    first, and a lowercase one through its own, then its capitalized one
    (πλάτωνι is analyzed only as Πλάτωνι). A diaeresis is dropped only when
    no spelling with it is analyzed, since ἔγχεϊ is a dative and ἔγχει an
    imperative."""
    own = _dative_key(full)
    lower = _dative_key(full.lower())
    capital = _dative_key(full[:1].title() + full[1:])
    order = [lower, own] if own != lower else [own, capital]
    for key in order:
        if key in dative_keys:
            return dative_keys[key]
    for key in order:
        folded = _nfc(unicodedata.normalize("NFD", key).replace("\u0308", ""))
        if folded != key and folded in dative_keys:
            return dative_keys[folded]
    return False


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
    pairs: dict[str, str] = {}
    for full, candidates in candidates_by_full.items():
        if full in long_final:
            continue
        # Only the dative's own -ι: τῷδε and ἔμοιγε are datives too, but
        # what they lose is the ε of δε and γε, which elides like any other.
        if (dative_keys and _is_dative(full, dative_keys)
                and _final_vowel(full)[1] == "ι"):
            continue
        elided = _rule_elision(full)
        if elided in candidates and grc_orthography_reason(elided) is None:
            pairs[full] = elided

    # Pin canonical overrides last. NFC everything for safety.
    for full, elided in CANONICAL_ELISION_OVERRIDES.items():
        pairs[_nfc(full)] = _nfc(elided)

    return pairs


def build(out_dir: Path) -> dict:
    """Drive the full morphology export end to end and write
    ``<out_dir>/grc_morph.json``. Returns a stats dict.
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

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grc_morph.json"
    payload = {
        "version": 1,
        "nu": sorted(nu_forms),
        "el": dict(sorted(elision_pairs.items())),
    }
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
        "bytes": size,
        "path": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir", default=str(OUT),
        help=f"Output directory (default: {OUT})",
    )
    args = ap.parse_args()
    build(Path(args.out_dir))


if __name__ == "__main__":
    main()
