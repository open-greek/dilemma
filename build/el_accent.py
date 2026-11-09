"""Accent restoration the way a monotonic Modern Greek keyboard does it, for
evaluating the word list and the next-word model together
(``eval/eval_el_accent_restoration.py``).

Nothing here reads files.

What the keyboard has
---------------------
A word typed without accents is resolved with two artifacts only:

* the word list ``el_monotonic.dic``, looked up by the accent-free key of
  :func:`el_lexicon.bare_key` (lowercase, no tonos, no dialytika, ς as σ),
  which is how the keyboard's compiled dictionary is indexed; every entry
  with that key is a candidate, with its frequency rating (C, M, R, X);
* the next-word model ``el_ngram.bin``, which scores a candidate after the
  two words before it (stupid backoff, ``LMReader.score``). Only the left
  context exists when the word is committed.

The decision (:func:`decide`)
-----------------------------
1. The candidates are grouped by their lowercase spelling, because the
   restored word keeps the case that was typed (:func:`transfer_marks`
   copies the candidate's marks onto the typed letters). When the typed word
   is lowercase and some candidate is lowercase, capitalized candidates
   (names) are left out. With no lowercase candidate, only names rated C
   or M may supply accents; rare names rated R or X stay out. A capitalized
   typed word can still match a name of any rating.
2. No candidate: the word stays as typed.
3. One candidate: its marks are applied. When the typed letters are that
   candidate (a monosyllable such as ``και``), nothing changes.
4. Several candidates (``πότε``/``ποτέ``, ``που``/``πού``, ``η``/``ή``):
   each is scored by the model in left context; the best one is applied only
   when it leads the runner-up by at least ``margin`` (natural log units).
   When the typed letters are a word themselves (``που``, ``η``), the lead
   must reach ``word_margin`` instead, which can be set higher (or to
   infinity) so that one valid word is replaced by another only on strong
   evidence. Below the lead the word stays as typed (policy ``keep``, which
   keeps ``που`` and ``η`` and leaves ``ποτε`` bare). Policy ``top``
   applies the best candidate anyway when the typed letters are not a word
   (a bare polysyllable is always misspelled), and still keeps a typed
   word such as ``που``.

The enclitic second accent (:func:`enclitic_accent`)
-----------------------------------------------------
The word list holds no double accents; the next-word model keeps them as
tokens of their own (:func:`enclitic_twin`), and the keyboard compares the
two readings by the model's scores (:func:`enclitic_evidence`,
:func:`keyboard_enclitic_decision`) before it falls back to the rules below. A word stressed on the third
syllable from the end, counted with synizesis (:func:`el_lexicon.
is_proparoxytone`: ``άνθρωπος`` yes, ``σπίτια`` no), takes a second tonos
on its last syllable before an enclitic pronoun (``ο άνθρωπός μου``,
``άκουσέ με``). ``μου``, ``σου``, ``μας`` and ``σας`` are always pronouns
after a noun or verb. ``του``, ``της``, ``τους``, ``το``, ``τα``, ``τον``,
``την``, ``τη`` and ``τις`` are also articles (``ο δάσκαλος του
σχολείου``), and ``με`` and ``σε`` are also prepositions, so
:func:`takes_enclitic_accent` adds the accent before them only when they
end the sentence, the one case the left context and the next token settle
without a part-of-speech tagger.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Callable, Iterable

from dilemma_build import el_lexicon

TONOS = "́"
DIALYTIKA = "̈"
MARKS = frozenset({TONOS, DIALYTIKA})
ALWAYS_ENCLITICS = frozenset({"μου", "σου", "μας", "σας"})
ARTICLE_ENCLITICS = frozenset({"του", "της", "τους", "το", "τα", "τον",
                               "την", "τη", "τις"})
PREPOSITION_ENCLITICS = frozenset({"με", "σε"})
ENCLITICS = ALWAYS_ENCLITICS | ARTICLE_ENCLITICS | PREPOSITION_ENCLITICS
POLICIES = ("keep", "top")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def strip_marks(word: str) -> str:
    """``word`` with every tonos and dialytika removed, case and final ς
    kept: what a person types who never presses the accent key."""
    nfd = unicodedata.normalize("NFD", word)
    if not any(c in MARKS for c in nfd):
        return word
    return nfc("".join(c for c in nfd if c not in MARKS))


def marks_of(letter: str) -> str:
    """The tonos and dialytika of one letter, in canonical order."""
    return "".join(c for c in unicodedata.normalize("NFD", letter) if c in MARKS)


def transfer_marks(typed: str, spelling: str) -> str:
    """The typed letters with the marks of ``spelling`` and no others. The
    two must share their :func:`el_lexicon.bare_key`, so they have the same
    length in NFC (every marked letter of the Greek block is one code
    point). Case and final sigma follow ``typed``. All-capital words lose
    tonos but retain dialytika, including hiatus previously shown by the
    first vowel's tonos (τσάι -> ΤΣΑΪ). The one-letter ή keeps its tonos.
    """
    if len(typed) != len(spelling):
        raise ValueError(f"{typed!r} and {spelling!r} differ in length")
    out = []
    capitals = typed.isupper() and sum(c.isalpha() for c in typed) > 1
    for i, (t, s) in enumerate(zip(typed, spelling)):
        base = strip_marks(t)
        marks = marks_of(s)
        if capitals:
            marks = marks.replace(TONOS, "")
            if (i and TONOS in marks_of(spelling[i - 1])
                    and strip_marks(spelling[i - 1:i + 1]).lower() in el_lexicon.DIGRAPHS
                    and strip_marks(s).lower() in {"ι", "υ"}
                    and DIALYTIKA not in marks):
                marks += DIALYTIKA
        out.append(nfc(base + marks) if marks else base)
    return "".join(out)


def has_marks(word: str) -> bool:
    return any(c in MARKS for c in unicodedata.normalize("NFD", word))


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Option:
    """One lowercase spelling with a key: ``spellings`` are the list
    entries that lowercase to it (``Νίκη`` and ``νίκη``), ``rank`` the best
    rating among them (C=3, M=2, R=1, X=0)."""
    lower: str
    spellings: tuple[str, ...]
    rank: int


class AccentIndex:
    """The word list by accent-free key."""

    def __init__(self, entries: Iterable[tuple[str, str]]):
        """``entries``: ``(spelling, rating)`` pairs, rating one of C M R X."""
        by_key: dict[str, list[tuple[str, int]]] = {}
        for spelling, rating in entries:
            by_key.setdefault(el_lexicon.bare_key(spelling), []).append(
                (spelling, el_lexicon.BUCKET_RANK.get(rating, 0)))
        self.by_key = by_key
        self._cache: dict[tuple[str, bool], tuple[Option, ...]] = {}

    def __len__(self) -> int:
        return len(self.by_key)

    def options(self, typed: str) -> tuple[Option, ...]:
        """The candidates for a typed word (see the module docstring, step
        1), most common rating first, then by spelling."""
        key = el_lexicon.bare_key(typed)
        lower_typed = el_lexicon.case_shape(typed) == "lower"
        cached = self._cache.get((key, lower_typed))
        if cached is not None:
            return cached
        found = self.by_key.get(key, ())
        if lower_typed:
            has_lower = any(el_lexicon.case_shape(s) == "lower" for s, _ in found)
            found = [(s, r) for s, r in found
                     if el_lexicon.case_shape(s) == "lower"
                     or (not has_lower and r >= el_lexicon.BUCKET_RANK["M"])]
        groups: dict[str, list[tuple[str, int]]] = {}
        for s, r in found:
            groups.setdefault(el_lexicon.lower_el(s), []).append((s, r))
        opts = tuple(sorted(
            (Option(low, tuple(sorted(s for s, _ in g)), max(r for _, r in g))
             for low, g in groups.items()),
            key=lambda o: (-o.rank, o.lower)))
        self._cache[(key, lower_typed)] = opts
        return opts


def proposal_options(typed: str, options: Iterable[Option],
                     preferred: dict[str, tuple[str, ...]], *,
                     contextual_score: Callable[[Option], float] | None = None,
                     context_margin: float = 3.0) -> tuple[Option, ...]:
    """Filter default proposals without changing dictionary acceptance.

    ``preferred`` combines the dictionary's ac/sp targets, keyed by lowercase
    spelling. A marked rare or older variant is not a routine alternative
    when its preferred form is available. Keep it when the person explicitly
    typed its marks, or context favors it over every available preferred
    reading by ``context_margin`` natural log units. Only pass a score based
    on actual preceding text, never a context-free unigram score.

    Keep the typed-as-is choice separately. This filter does not reject a
    deliberately typed dialect word, and unmarked genuine ambiguities are
    unaffected. Candidate generation and ordering belong to the caller.
    """
    options = tuple(options)
    by_lower = {option.lower: option for option in options}
    typed_lower = el_lexicon.lower_el(typed)
    kept = []
    for option in options:
        targets = [by_lower[target] for target in preferred.get(option.lower, ())
                   if target in by_lower]
        explicit = has_marks(typed) and typed_lower == option.lower
        if not targets or explicit:
            kept.append(option)
            continue
        if contextual_score is not None:
            score = contextual_score(option)
            target_score = max(contextual_score(target) for target in targets)
            if score > -math.inf and score - target_score >= context_margin:
                kept.append(option)
    return tuple(kept)


# --------------------------------------------------------------------------
# The decision
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    """``kind`` is one of:

    * ``none``: no candidate, left as typed;
    * ``same``: one candidate, and it is the typed letters;
    * ``unique``: one candidate, its marks applied;
    * ``model``: several candidates, the model's choice applied (it may be
      the typed letters, as ``που``);
    * ``kept``: several candidates, no clear lead, left as typed;
    * ``top``: several candidates, no clear lead, the best one applied
      (policy ``top`` only).
    """
    output: str
    kind: str
    options: tuple[str, ...] = ()
    gap: float | None = None


def gap_of(scores: list[float]) -> float:
    """The lead of the best score over the runner-up, in natural log units:
    infinite when only the best one is known, 0 when neither is."""
    ordered = sorted(scores, reverse=True)
    best, second = ordered[0], ordered[1]
    if best == -math.inf:
        return 0.0
    if second == -math.inf:
        return math.inf
    return best - second


def decide(typed: str, options: tuple[Option, ...],
           score: Callable[[Option], float] | None, *, margin: float,
           word_margin: float | None = None, policy: str = "keep") -> Decision:
    """Restore the accents of ``typed`` (see the module docstring).

    ``margin`` is the lead the best candidate needs when the typed letters
    are not a word (``ποτε``); ``word_margin`` (default: ``margin``) the
    lead it needs to replace typed letters that are a word themselves
    (``η`` to ``ή``, ``που`` to ``πού``). An infinite ``word_margin`` never
    replaces a typed word."""
    if not options:
        return Decision(typed, "none")
    lowers = tuple(o.lower for o in options)
    if len(options) == 1:
        out = transfer_marks(typed, options[0].spellings[0])
        return Decision(out, "same" if out == typed else "unique", lowers)
    scores = [score(o) for o in options]
    gap = gap_of(scores)
    # Ties (equal scores, both unknown) go to the more common rating, then
    # to the spelling, so the choice never depends on dictionary order.
    best = max(range(len(options)),
               key=lambda k: (scores[k], options[k].rank, -k))
    typed_is_word = el_lexicon.lower_el(typed) in lowers
    need = (margin if word_margin is None else word_margin) if typed_is_word else margin
    if gap >= need and gap > 0:
        return Decision(transfer_marks(typed, options[best].spellings[0]),
                        "model", lowers, gap)
    if policy == "top" and not typed_is_word:
        return Decision(transfer_marks(typed, options[best].spellings[0]),
                        "top", lowers, gap)
    return Decision(typed, "kept", lowers, gap)


# --------------------------------------------------------------------------
# The enclitic second accent
# --------------------------------------------------------------------------

def _last_vowel_slot(word: str) -> int | None:
    """Index of the letter that takes the tonos of the last syllable: the
    single vowel, or the second letter of a digraph (``ου``, ``αι``, ``ευ``
    ...), as monotonic spelling places it."""
    low = el_lexicon.lower_el(word)
    letters = [el_lexicon._MARKS.get(c, (c, False, False)) for c in low]
    slot = None
    i = 0
    n = len(letters)
    while i < n:
        base, tonos, dial = letters[i]
        if base not in el_lexicon.VOWELS:
            i += 1
            continue
        if (i + 1 < n and base + letters[i + 1][0] in el_lexicon.DIGRAPHS
                and not letters[i + 1][2] and not tonos and not dial):
            slot = i + 1
            i += 2
            continue
        slot = i
        i += 1
    return slot


def enclitic_accent(word: str) -> str | None:
    """``word`` with the second tonos an enclitic gives it (``όνομα`` to
    ``όνομά``, ``άκουσε`` to ``άκουσέ``), or None when the word does not take
    one: not exactly one tonos, not stressed on the third syllable from the
    end (synizesis counted), or elided."""
    if el_lexicon.APOSTROPHE in word or el_lexicon.tonos_count(word) != 1:
        return None
    if not el_lexicon.is_proparoxytone(word):
        return None
    slot = _last_vowel_slot(word)
    if slot is None or marks_of(word[slot]).count(TONOS):
        return None
    return word[:slot] + nfc(word[slot] + TONOS) + word[slot + 1:]


def enclitic_twin(host: str) -> str | None:
    """The double-accented form of ``host`` the keyboard can write before an
    enclitic (``EncliticAccent.twin(of:)``): Greek letters only, in
    lowercase or with a capital first letter, exactly one tonos, stressed on
    the third syllable from the end counted with or without synizesis
    (``όνομα`` to ``όνομά``, ``δωμάτιο`` to ``δωμάτιό``, ``σπίτια`` to
    ``σπίτιά``). None otherwise. The next-word model keeps these forms as
    tokens of their own, so the keyboard can compare the two readings."""
    if not host or any(c not in el_lexicon.LETTERS for c in host):
        return None
    if el_lexicon.case_shape(host) not in ("lower", "title"):
        return None
    if el_lexicon.tonos_count(host) != 1:
        return None
    if not (el_lexicon.is_proparoxytone(host) or needs_synizesis_call(host)):
        return None
    slot = _last_vowel_slot(host)
    if slot is None or marks_of(host[slot]).count(TONOS):
        return None
    return host[:slot] + nfc(host[slot] + TONOS) + host[slot + 1:]


# The keyboard's enclitics (``EncliticAccent.pronouns`` and
# ``articleHomographs``) and the words after which an article homograph is
# read as the enclitic (``possessiveSignals``).
PRONOUN_ENCLITICS = frozenset({"μου", "σου", "μας", "σας"})
ARTICLE_HOMOGRAPHS = frozenset({"του", "της", "τους", "το", "τα", "τον", "την",
                                "τις"})
KEYBOARD_ENCLITICS = PRONOUN_ENCLITICS | ARTICLE_HOMOGRAPHS
POSSESSIVE_SIGNALS = frozenset({
    "και", "κι", "αλλά", "ή", "ενώ", "όμως", "μα", "ούτε", "είτε", "που",
    "όταν", "αν", "γιατί", "επειδή", "ότι", "πως", "ώστε", "αφού", "μόλις",
    "καθώς", "θα", "δεν", "δε", "να", "μη", "μην", "ας", "σε", "σ’", "στο",
    "στη", "στην", "στον", "στα", "στους", "στις", "στου", "στης", "από",
    "απ’", "με", "για", "χωρίς", "μέχρι", "ως", "προς", "κατά", "μετά",
    "παρά", "είναι", "ήταν"})
ENCLITIC_MARGIN = 1.0


def enclitic_evidence(lm, host: str, twin: str, enclitic: str,
                      next_word: str | None, prev: str | None,
                      curr: str | None, margin: float = ENCLITIC_MARGIN) -> bool | None:
    """``EncliticAccent.evidence``: True when the model prefers the
    double-accented reading ``twin enclitic`` over ``host enclitic`` by
    ``margin`` (natural log units), False when it prefers the plain one,
    None when it has no say (a form it has never seen, or a smaller
    difference). Each reading scores the host form after ``prev curr``, the
    enclitic after ``curr`` and the form, and, when ``next_word`` is given
    and both readings score it, that word after the form and the enclitic.
    ``lm`` answers ``knows(word)`` and ``score(word, prev, curr)`` the way
    the keyboard's reader does (``gnlm_reader.GNLMReader``)."""
    if not (lm.knows(twin) and lm.knows(host)):
        return None

    def path(form: str) -> float:
        return lm.score(form, prev, curr) + lm.score(enclitic, curr, form)

    twin_score = path(twin)
    host_score = path(host)
    if next_word is not None:
        after_twin = lm.score(next_word, twin, enclitic)
        after_host = lm.score(next_word, host, enclitic)
        if math.isfinite(after_twin) and math.isfinite(after_host):
            twin_score += after_twin
            host_score += after_host
    if not (math.isfinite(twin_score) or math.isfinite(host_score)):
        return None
    if not math.isfinite(host_score) or twin_score - host_score >= margin:
        return True
    if not math.isfinite(twin_score) or host_score - twin_score >= margin:
        return False
    return None


def keyboard_enclitic_decision(lm, host: str, enclitic: str, next_word: str | None,
                               prev: str | None, curr: str | None,
                               margin: float = ENCLITIC_MARGIN) -> bool:
    """Whether the keyboard writes ``host`` with the second accent before
    ``enclitic`` (lowercase) once ``next_word`` is known (None: the
    sentence ends after the enclitic), as ``EncliticAccent.decide`` does
    over the two commits. A pronoun is decided on the model's evidence with
    the next word, else without it, else by the syllable rule; an article
    homograph at the end of the sentence on the evidence without a next
    word, else by the syllable rule, and before a word on the evidence with
    it, else only before a ``POSSESSIVE_SIGNALS`` word and by the syllable
    rule. ``lm`` None decides without evidence."""
    twin = enclitic_twin(host)
    if twin is None or enclitic not in KEYBOARD_ENCLITICS:
        return False
    rule = el_lexicon.is_proparoxytone(host)

    def ev(nxt):
        if lm is None:
            return None
        return enclitic_evidence(lm, host, twin, enclitic, nxt, prev, curr, margin)

    if enclitic in PRONOUN_ENCLITICS:
        if next_word is None:
            first = ev(None)
            return rule if first is None else first
        with_next = ev(next_word)
        if with_next is not None:
            return with_next
        first = ev(None)
        return rule if first is None else first
    if next_word is None:
        first = ev(None)
        return rule if first is None else first
    with_next = ev(next_word)
    if with_next is not None:
        return with_next
    return el_lexicon.lower_el(next_word) in POSSESSIVE_SIGNALS and rule


def takes_enclitic_accent(enclitic: str, next_token: str | None) -> bool:
    """Whether the keyboard adds the second accent before ``enclitic``
    (lowercased) followed by ``next_token`` (None at the end of the
    sentence). See the module docstring."""
    if enclitic in ALWAYS_ENCLITICS:
        return True
    if enclitic in ARTICLE_ENCLITICS or enclitic in PREPOSITION_ENCLITICS:
        return next_token is None
    return False


def needs_synizesis_call(word: str) -> bool:
    """True for a word that is proparoxytone only when its glide is counted
    as a syllable (``δωμάτιο``: δω-μά-τιο with synizesis, δω-μά-τι-ο
    without). Spelling alone cannot say whether such a word takes the
    enclitic accent: usage writes both ``το δωμάτιό μου`` and ``τα σπίτια
    μου``."""
    if el_lexicon.tonos_count(word) != 1:
        return False
    nuclei = el_lexicon.nuclei(word)
    units = el_lexicon.vowel_units(el_lexicon.lower_el(word).replace(
        el_lexicon.APOSTROPHE, ""))
    if len(units) == len(nuclei):
        return False
    marks = [k for k, (_, accented, _) in enumerate(units) if accented]
    return len(marks) == 1 and len(units) - marks[0] == 3 \
        and not el_lexicon.is_proparoxytone(word)
