"""Restore the initial breathings the Sophocles OCR lost.

The digitized Sophocles lexicon (``data/sophocles/chunks``) prints no
breathing on most headwords of its ο chunk (0800.xml: οφφίκιον) and on the
initial rho of its ρ chunk (0950.xml: ρῆσις), and a few elsewhere. Such a head
is no Greek spelling. ``review_heads`` decides each one from evidence and
records the rule:

* ``initial_rho``: an initial ρ before a vowel takes the rough breathing.
* ``same_head``: another head source (Wiktionary, LSJ, DGE, LBG, ...) has
  the same letters, accents and case with exactly one breathing.
* ``word_family``: the other sources' heads that share the longest initial
  stretch of letters with it (at least ``FAMILY_MIN`` letters, accents and
  breathings aside) all have the same breathing (οφφίκιον: ὀφφικιάλιος).

Anything else (both breathings attested, no evidence, an initial ρ before a
consonant, two accents or an initial diaeresis from the OCR) is withheld with its reason and candidates: no breathing is
invented, and the unbreathed spelling is not a head either.
build/build_grc_lexical_heads.py stores the result in grc_lexical_heads.json
(``sophocles_breathing``); build/expand_sophocles.py reads it from there.
"""
from __future__ import annotations

from collections import defaultdict
import unicodedata

SMOOTH, ROUGH = "̓", "̔"
BREATHINGS = {SMOOTH: "smooth", ROUGH: "rough"}
MARKS = {name: mark for mark, name in BREATHINGS.items()}
VOWELS = frozenset("αεηιουω")
DIPHTHONGS = frozenset({"αι", "ει", "οι", "υι", "αυ", "ευ", "ου", "ηυ"})
DIAERESIS = "̈"
FAMILY_MIN = 4
SCHEMA = 1


def _letters(word: str) -> list[tuple[str, str]]:
    """(base letter, its combining marks) per letter of NFD ``word``."""
    out: list[tuple[str, str]] = []
    for char in unicodedata.normalize("NFD", word):
        if unicodedata.combining(char) and out:
            out[-1] = (out[-1][0], out[-1][1] + char)
        else:
            out.append((char, ""))
    return out


def _carrier(letters: list[tuple[str, str]]) -> int | None:
    """Index of the letter an initial breathing sits on, or None when the
    word does not begin with a vowel or rho."""
    if not letters:
        return None
    first = letters[0][0].lower()
    if first == "ρ":
        return 0
    if first not in VOWELS:
        return None
    if (len(letters) > 1 and first + letters[1][0].lower() in DIPHTHONGS
            and DIAERESIS not in letters[1][1]
            and not set(letters[0][1]) & {"́", "̀", "͂"}):
        return 1
    return 0


def breathing_of(word: str) -> str | None:
    """'smooth', 'rough', or None for a word with no initial breathing."""
    letters = _letters(word)
    for _, marks in letters[:2]:
        for mark, name in BREATHINGS.items():
            if mark in marks:
                return name
    return None


def needs_breathing(word: str) -> bool:
    """A head beginning with a vowel or rho that carries no breathing."""
    return _carrier(_letters(word)) is not None and breathing_of(word) is None


def add_breathing(word: str, kind: str) -> str:
    """``word`` with a ``kind`` ('smooth'/'rough') breathing on its initial
    vowel, diphthong or rho, placed before any accent."""
    letters = _letters(word)
    at = _carrier(letters)
    if at is None:
        raise ValueError(f"no initial vowel or rho: {word}")
    base, marks = letters[at]
    letters[at] = (base, MARKS[kind] + marks.replace(SMOOTH, "").replace(ROUGH, ""))
    return unicodedata.normalize("NFC", "".join(b + m for b, m in letters))


def strip_breathing(word: str) -> str:
    return unicodedata.normalize("NFC", "".join(
        c for c in unicodedata.normalize("NFD", word) if c not in BREATHINGS))


def _skeleton(word: str) -> str:
    """Lowercase base letters only (final sigma folded)."""
    return "".join(b.lower() for b, _ in _letters(word)).replace("ς", "σ")


class _Evidence:
    def __init__(self, heads):
        self.exact = defaultdict(set)
        self.by_skeleton = {}
        for head in heads:
            kind = breathing_of(head)
            if kind is None:
                continue
            self.exact[strip_breathing(head)].add((head, kind))
            self.by_skeleton.setdefault(_skeleton(head), set()).add((head, kind))
        self.skeletons = sorted(self.by_skeleton)

    def family(self, word: str):
        """The heads sharing the longest initial stretch (>= FAMILY_MIN
        letters) of ``word``'s skeleton, and that length."""
        import bisect
        skeleton = _skeleton(word)
        for length in range(len(skeleton), FAMILY_MIN - 1, -1):
            prefix = skeleton[:length]
            at = bisect.bisect_left(self.skeletons, prefix)
            members = set()
            while at < len(self.skeletons) and self.skeletons[at].startswith(prefix):
                members |= self.by_skeleton[self.skeletons[at]]
                at += 1
            if members:
                return members, length
        return set(), 0


ACCENTS = frozenset("\u0301\u0300\u0342")


def _unusable(head: str) -> str | None:
    """Why a head cannot take a breathing at all: an OCR diaeresis on the
    initial vowel (ϊκανᾶτος), or two accents (άφθαρσία, where the OCR read
    the breathing as an acute; the source's spelling is unknown)."""
    letters = _letters(head)
    at = _carrier(letters)
    if at is not None and DIAERESIS in letters[at][1]:
        return "diaeresis_on_initial"
    if sum(mark in ACCENTS for _, marks in letters for mark in marks) > 1:
        return "two_accents"
    return None


def review_heads(chunk_heads: dict[str, set[str]], other_heads) -> dict:
    """Decide every unbreathed chunk head. ``chunk_heads`` maps a head to the
    chunk sources that print it; ``other_heads`` are the heads of every other
    source. Deterministic: the output depends only on the two inputs."""
    evidence = _Evidence(other_heads)
    corrected, withheld = {}, {}
    for head in sorted(chunk_heads):
        if not needs_breathing(head):
            continue
        sources = sorted(chunk_heads[head])
        letters = _letters(head)
        if (reason := _unusable(head)):
            withheld[head] = {"reason": reason, "sources": sources}
            continue
        if letters[0][0].lower() == "ρ":
            if len(letters) > 1 and letters[1][0].lower() in VOWELS:
                corrected[head] = {"head": add_breathing(head, "rough"),
                                   "rule": "initial_rho", "sources": sources}
            else:
                withheld[head] = {"reason": "rho_before_consonant", "sources": sources}
            continue
        same = {(h, k) for h, k in evidence.exact.get(head, ())}
        kinds = {k for _, k in same}
        if len(kinds) == 1:
            kind = kinds.pop()
            corrected[head] = {"head": add_breathing(head, kind), "rule": "same_head",
                               "evidence": sorted(h for h, _ in same), "sources": sources}
            continue
        if len(kinds) > 1:
            withheld[head] = {"reason": "both_breathings_attested",
                              "candidates": sorted(h for h, _ in same), "sources": sources}
            continue
        members, length = evidence.family(head)
        kinds = {k for _, k in members}
        if len(kinds) == 1:
            kind = kinds.pop()
            corrected[head] = {"head": add_breathing(head, kind), "rule": "word_family",
                               "shared_letters": length,
                               "evidence": sorted(h for h, _ in members)[:5],
                               "evidence_count": len(members), "sources": sources}
        elif kinds:
            withheld[head] = {"reason": "family_breathings_differ",
                              "shared_letters": length,
                              "candidates": sorted(h for h, _ in members)[:10],
                              "sources": sources}
        else:
            withheld[head] = {"reason": "no_evidence", "sources": sources}
    return {"schema_version": SCHEMA, "family_min_letters": FAMILY_MIN,
            "corrected": corrected, "withheld": withheld,
            "counts": {"corrected": len(corrected), "withheld": len(withheld)}}


def correct(head: str, review: dict | None) -> str | None:
    """The head to use for a chunk head: corrected, unchanged, or None when
    the review withheld it."""
    if not review:
        return head
    if head in review["withheld"]:
        return None
    row = review["corrected"].get(head)
    return row["head"] if row else head
