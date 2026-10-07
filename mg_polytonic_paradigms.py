#!/usr/bin/env python3
"""Polytonic spellings for the inflected forms of frequent Modern Greek verbs.

The polytonic Modern Greek word list (``export_mg_polytonic.py``) lists the
spellings its source texts attest. Those texts are mostly older prose, so the
second person and the spoken forms of common verbs are rare in them: the list
has ἔρθει and ἔρθουν but not ἔρθεις, and no μιλήσεις or πιεῖς. This module
writes those forms from the monotonic paradigms Dilemma already has
(``data/mg_pairs.json``, Wiktionary's tagged Modern Greek inflection tables),
for the forms frequent in monotonic text (``data/mg_freq.txt``).

A monotonic spelling has the letters, the accented vowel and any diaeresis of
the polytonic one. Three things are missing, and each is decided as follows.

* The breathing of an initial vowel or ρ. The attested polytonic forms of the
  same verb with the same first letters decide it (ἔρθω, ἔρθει -> ἔρθεις);
  without them, the Ancient Greek words of the grc dictionary that begin with
  the same letters (ἀναρωτιέμαι from ἀνα-), and failing those the smooth
  breathing (the rough one on υ). On a diphthong it sits on the second vowel.
* The type of the accent, by the Ancient rule that polytonic Modern Greek
  keeps: an accented antepenult takes the acute; an accented penult the
  circumflex when its vowel is long and the last syllable short (δῶσε,
  ποῦμε, ἦρθα), else the acute; an accented last syllable the circumflex
  when its vowel is long (μπορεῖς, δοθεῖ, πιῶ), the acute when short (πές).
  η, ω and the diphthongs are long and ε, ο short; a final αι or οι counts
  short. α, ι and υ can be either, and the verb's attested forms decide
  them (ἐλᾶτε, πάρε); without them a contracted ending is long (μιλᾶτε,
  μιλᾶς) and any other vowel short (κάνε, πίνε). Monotonic writing leaves a
  monosyllable unaccented (πιω, πεις); its accent goes on its last vowel.
* The iota subscript, which a verb's attested forms carry over to the same
  vowel, and which the contracted endings take as the texts mostly write
  them (ἀγαπᾷς).

The texts write the subjunctive of the second and third person singular both
as the indicative (νὰ πάρεις, νὰ πάρει) and the traditional way (νὰ πάρῃς,
νὰ πάρῃ), so both are written, for the dependent forms and for the present
forms that serve as their imperfective subjunctive.

Generated spellings are written with the field ``mg:generated``, so that a
reader can tell them from the attested spellings of the list.
``eval/eval_mg_paradigms.py`` measures how often the generation reproduces an
attested spelling held out from its evidence.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, NamedTuple

ROOT = Path(__file__).parent
DATA = ROOT / "data"
# Wiktionary's Modern Greek inflection tables, monotonic, with tags.
VERB_PARADIGMS = DATA / "mg_pairs.json"
# Monotonic Modern Greek word frequencies (FrequencyWords, OpenSubtitles).
MG_FORM_FREQ = DATA / "mg_freq.txt"

# A verb form is generated when monotonic text has it at least this many
# times (OpenSubtitles, 263M tokens). See export_mg_polytonic.py.
GENERATED_MIN_TOKENS = 50
PERSON_TAGS = frozenset({"first-person", "second-person", "third-person"})
# A synthetic tag: one row of the form is a finite verb cell.
FINITE = "finite"
# A cell Wiktionary tags rare, archaic, dialectal, Katharevousa, obsolete or
# dated, or whose table it could not parse (error-unrecognized-form), is
# generated like any other when monotonic text has it GENERATED_MIN_TOKENS
# times (ὑπακούουν, ἀπεχθάνεσαι, συντριβεῖ), unless its letters are also a
# form of another part of speech: then the count is mostly that word's
# (πιάνου, the piano's; ἐλέγχου, the check's; τέλει).
RARE_CELL_TAGS = frozenset({"rare", "archaic", "error-unrecognized-form",
                            "dialectal", "Katharevousa", "obsolete", "dated"})
# An attested spelling below the list's own floor (3 tokens, as in
# export_mg_polytonic.MIN_TOKENS) does not decide a vowel's length or its
# iota subscript: one token of a misspelling is no evidence (γελᾶστε,
# ἡσυχᾶστε). It still votes for a breathing, which a wrong vote rarely
# flips and a missing one often does (ἁλυσοδεμένος, ἁπλοποιήσουμε).
SIBLING_MIN_TOKENS = 3
# Prefixes of grc words, up to this many letters, decide a breathing the
# verb's own forms do not (``Evidence.breathing``).
PREFIX_LETTERS = 7

ACUTE, GRAVE, CIRCUMFLEX = "́", "̀", "͂"
SMOOTH, ROUGH = "̓", "̔"
DIAERESIS, SUBSCRIPT = "̈", "ͅ"
TONAL = frozenset((ACUTE, GRAVE, CIRCUMFLEX))
VOWELS = frozenset("αεηιουω")
LONG_VOWELS = frozenset("ηω")
SHORT_VOWELS = frozenset("εο")
DICHRONA = frozenset("αιυ")
DIPHTHONGS = frozenset({"αι", "ει", "οι", "υι", "αυ", "ευ", "ηυ", "ου"})
# A final αι or οι counts short for the accent (εἶσαι, ἔρχεται).
SHORT_FINAL_DIPHTHONGS = frozenset({"αι", "οι"})


# --------------------------------------------------------------------------
# Letters and syllables
# --------------------------------------------------------------------------

class Letter(NamedTuple):
    base: str            # lowercase base letter
    marks: frozenset     # combining marks as characters


class Sibling(NamedTuple):
    """An attested spelling of one of a verb's forms."""
    spelling: str
    tokens: int
    tags: frozenset = frozenset()   # the tags of the forms it spells


def letters(spelling: str) -> list[Letter]:
    """The Greek letters of ``spelling`` with their combining marks."""
    out: list[Letter] = []
    for char in unicodedata.normalize("NFD", spelling):
        if unicodedata.combining(char):
            if out:
                out[-1] = Letter(out[-1].base, out[-1].marks | {char})
            continue
        out.append(Letter(char.lower(), frozenset()))
    return out


def bare(spelling: str) -> str:
    """The letters a bare keyboard types for ``spelling``: no marks,
    lowercase, final sigma as σ."""
    return "".join(l.base for l in letters(spelling)).replace("ς", "σ")


def nuclei(word: list[Letter], *, glides: bool = False) -> list[list[int]]:
    """The vowel nuclei of a word, as lists of letter indexes: the Ancient
    diphthongs (unless the second vowel has a diaeresis or the first an
    accent) and single vowels. Polytonic Modern Greek places its accents by
    these syllables (ἤ-πι-ε, ἤ-πι-αν), and only with ``glides`` is an
    unaccented ι or υ before another vowel joined to it, as Modern Greek
    pronounces it (πιῶ, πιεῖς are one syllable)."""
    groups: list[list[int]] = []
    i = 0
    while i < len(word):
        if word[i].base not in VOWELS:
            i += 1
            continue
        members = [i]
        if (i + 1 < len(word) and word[i + 1].base in VOWELS
                and word[i].base + word[i + 1].base in DIPHTHONGS
                and DIAERESIS not in word[i + 1].marks
                and not word[i].marks & TONAL):
            members.append(i + 1)
            i += 1
        groups.append(members)
        i += 1
    if not glides:
        return groups
    merged: list[list[int]] = []
    for members in groups:
        if merged and members[0] == merged[-1][-1] + 1:
            prev = merged[-1]
            last = word[prev[-1]]
            glide = (len(prev) == 1 and last.base in "ιυ"
                     and not last.marks & TONAL
                     and DIAERESIS not in last.marks)
            if glide:
                prev.extend(members)
                continue
        merged.append(list(members))
    return merged


def main_part(word: list[Letter], nucleus: list[int]) -> list[int]:
    """The vowel letters of a nucleus without a leading glide."""
    if (len(nucleus) > 1 and word[nucleus[0]].base in "ιυ"
            and word[nucleus[0]].base + word[nucleus[1]].base
            not in DIPHTHONGS):
        return nucleus[1:]
    return nucleus


def vowel_length(word: list[Letter], nucleus: list[int], *,
                 final: bool) -> str:
    """``long``, ``short`` or ``either`` (α, ι, υ) for a nucleus."""
    part = main_part(word, nucleus)
    vowels = "".join(word[i].base for i in part)
    if len(part) >= 2 and vowels[-2:] in DIPHTHONGS:
        if final and vowels[-2:] in SHORT_FINAL_DIPHTHONGS \
                and part[-1] == len(word) - 1:
            return "short"
        return "long"
    if any(SUBSCRIPT in word[i].marks for i in part):
        return "long"
    v = vowels[-1]
    if v in LONG_VOWELS:
        return "long"
    if v in SHORT_VOWELS:
        return "short"
    return "either"


def accent_carrier(word: list[Letter], nucleus: list[int]) -> int:
    """The letter of a nucleus that carries its accent: the second vowel of
    a diphthong, else the vowel after a glide, else the vowel."""
    part = main_part(word, nucleus)
    return part[-1] if len(part) >= 2 else part[0]


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------

def monotonic_key(spelling: str) -> str:
    """``spelling`` as monotonic writing spells it: no breathing, subscript
    or length mark, every accent an acute, and no accent at all on a word
    of one syllable (πιῶ -> πιω, πές -> πες)."""
    word = letters(spelling)
    out = []
    for l in word:
        marks = {m for m in l.marks if m in (ACUTE, DIAERESIS)}
        marks |= {ACUTE for m in l.marks if m in (GRAVE, CIRCUMFLEX)}
        out.append(Letter(l.base, frozenset(marks)))
    if len(nuclei(out, glides=True)) == 1:
        out = [Letter(l.base, l.marks - {ACUTE}) for l in out]
    return compose(out)


def compose(word: list[Letter]) -> str:
    """NFC spelling of letters and marks, marks in canonical order."""
    order = {SMOOTH: 0, ROUGH: 0, DIAERESIS: 1, ACUTE: 2, GRAVE: 2,
             CIRCUMFLEX: 2, SUBSCRIPT: 3}
    chars = []
    for l in word:
        chars.append(l.base)
        chars.extend(sorted(l.marks, key=lambda m: order.get(m, 9)))
    text = unicodedata.normalize("NFC", "".join(chars))
    if text.endswith("σ"):
        text = text[:-1] + "ς"
    return text


def breathing_of(spelling: str) -> str | None:
    """The breathing on the first letters of ``spelling``, if any."""
    for l in letters(spelling)[:2]:
        if SMOOTH in l.marks:
            return SMOOTH
        if ROUGH in l.marks:
            return ROUGH
    return None


class Evidence:
    """What decides the marks of a generated spelling.

    ``attested`` maps a monotonic key (:func:`monotonic_key`) to the
    attested polytonic spellings with that key and their token counts;
    ``prefix_breathing`` maps the first two to ``PREFIX_LETTERS`` bare
    letters of the grc dictionary's lowercase vowel-initial words to the
    count of each breathing.
    """

    def __init__(self, attested: dict[str, Counter],
                 grc_words: Iterable[str] = (),
                 paradigms: dict[str, dict[str, frozenset]] | None = None):
        self.attested = attested
        self.prefix_breathing: dict[str, Counter] = defaultdict(Counter)
        for w in grc_words:
            b = breathing_of(w)
            if b is None or not w[:1].islower():
                continue
            key = bare(w)
            for k in range(2, PREFIX_LETTERS + 1):
                if len(key) > k:
                    self.prefix_breathing[key[:k]][b] += 1
        self.subscript_share = self._contract_subscripts(paradigms or {})

    def _contract_subscripts(self, paradigms) -> dict[str, float]:
        """For the contracted present endings -ᾶς and -ᾶ (μιλᾶς, ζητᾶ), the
        share of the attested tokens of such forms written with the iota
        subscript (-ᾷς, -ᾷ)."""
        votes = {"ᾶς": Counter(), "ᾶ": Counter()}
        for forms in paradigms.values():
            for form, tags in forms.items():
                if not ({"present", "singular"} <= tags
                        and {"second-person", "third-person"} & tags):
                    continue
                ending = "ᾶς" if form.endswith("άς") else \
                    "ᾶ" if form.endswith("ά") else None
                if ending is None:
                    continue
                for spelling, n in self.attested.get(
                        monotonic_key(form), {}).items():
                    sub = ending.replace("ᾶ", "ᾷ")
                    if spelling.endswith(sub):
                        votes[ending][True] += n
                    elif spelling.endswith(ending):
                        votes[ending][False] += n
        return {e: v[True] / (v[True] + v[False]) if v else 0.0
                for e, v in votes.items()}

    def lemma_spellings(self, forms: Iterable[str] | dict[str, frozenset],
                        exclude: frozenset[str] = frozenset(),
                        ) -> list[Sibling]:
        """The attested spellings of a verb's ``forms`` with their token
        counts and the tags of the forms they spell, the most frequent
        first (then by spelling, so that ties do not depend on set order),
        leaving out the monotonic keys in ``exclude``."""
        tags_of: dict[str, frozenset] = defaultdict(frozenset)
        for form in forms:
            tags = forms[form] if isinstance(forms, dict) else frozenset()
            tags_of[monotonic_key(form)] |= tags
        found: Counter = Counter()
        found_tags: dict[str, frozenset] = defaultdict(frozenset)
        for key in sorted(set(tags_of) - exclude):
            for s, n in self.attested.get(key, {}).items():
                found[s] += n
                found_tags[s] |= tags_of[key]
        return [Sibling(s, n, found_tags[s])
                for s, n in sorted(found.items(), key=lambda x: (-x[1], x[0]))]

    def breathing(self, key: str, siblings: list[Sibling]) -> str:
        """The breathing for a word whose bare letters are ``key``: the one
        the verb's attested forms with the same first two letters write
        most, else the grc words' with the longest prefix of ``key`` that
        at least 5 of them share. That prefix decides, if 90% of its words
        agree; if they do not, a shorter one cannot, since its words are
        further from ``key``: εστι- is split by ἐστί, so ἑστιάζω follows
        εστια-, ἑστία, and never εσ-, smooth almost throughout. Without
        any such prefix the breathing is smooth, rough on υ."""
        votes: Counter = Counter()
        for sib in siblings:
            if bare(sib[0])[:2] == key[:2]:
                b = breathing_of(sib[0])
                if b:
                    votes[b] += sib[1]
        if votes:
            return max(votes.items(), key=lambda x: (x[1], x[0]))[0]
        default = ROUGH if key[:1] == "υ" else SMOOTH
        for k in range(min(PREFIX_LETTERS, len(key)), 1, -1):
            counts = self.prefix_breathing.get(key[:k])
            if counts and sum(counts.values()) >= 5:
                b, n = max(counts.items(), key=lambda x: (x[1], x[0]))
                return b if n >= 0.9 * sum(counts.values()) else default
        return default


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def verb_class(lemma: str, forms: dict[str, frozenset]) -> str:
    """``A`` (γράφω), ``B1`` (μιλάω, -άς), ``B2`` (μπορώ, -είς),
    ``passive`` (-μαι) or ``other``."""
    if lemma.endswith("μαι"):
        return "passive"
    if lemma.endswith("άω"):
        return "B1"
    second = [f for f, tags in forms.items()
              if {"present", "second-person", "singular"} <= tags
              and "passive" not in tags]
    if lemma.endswith(("ώ", "ῶ")):
        if any(f.endswith("άς") for f in second):
            return "B1"
        if any(f.endswith("είς") for f in second):
            return "B2"
    if lemma.endswith("ω") and not lemma.endswith("ώ"):
        return "A"
    return "other"


def cell(tags: frozenset) -> str:
    """The paradigm cell group of a form, for reporting."""
    if "participle" in tags or {"masculine", "feminine", "neuter"} & tags:
        return "participle"
    if "imperative" in tags:
        return "imperative"
    if "dependent" in tags:
        return "dependent"
    if "imperfect" in tags:
        return "imperfect"
    if "past" in tags or "aorist" in tags:
        return "past"
    if "present" in tags:
        return "present"
    return "other"


def _stem_length(word: list[Letter], index: int,
                 siblings: list[Sibling],
                 tags: frozenset = frozenset()) -> str | None:
    """``long`` or ``short`` for the vowel at ``index`` (α, ι or υ), by the
    tokens of the verb's attested forms with the same letters up to it that
    accent it where the type of the accent shows its length: a circumflex
    (long), or an acute on a penult before a short last syllable (short).
    A spelling with fewer than ``SIBLING_MIN_TOKENS`` tokens does not vote.

    The vowel before a consonant of a perfective stem is the stem's
    (γελάστε, κοιτάξτε, κοιτάχτε), and the long α of an imperfective
    contraction (γελᾶς, γελᾶτε, κοιτᾶς) says nothing of it, so there the
    forms must have the same letters through the next one as well, unless
    what follows is a contracted ending itself (φᾶμε beside φᾶς). The
    other way round, the acute of a perfective form (σκάσε) says nothing of
    an imperfective form's contracted α (σκᾶς, σκᾶνε): there a perfective
    form votes only with a circumflex."""
    perfective = "perfective" in tags
    rest = "".join(l.base for l in word[index + 1:])
    strict = (perfective and rest[:1] not in VOWELS | {""}
              and rest not in CONTRACTED_ENDINGS)
    key = "".join(l.base for l in word[:index + 1 + strict])
    votes: Counter = Counter()
    for sib in siblings:
        s, n = sib[0], sib[1]
        stags = sib[2] if len(sib) > 2 else frozenset()
        if n < SIBLING_MIN_TOKENS:
            continue
        sw = letters(s)
        if "".join(l.base for l in sw[:len(key)]) != key:
            continue
        marks = sw[index].marks
        if CIRCUMFLEX in marks:
            votes["long"] += n
        elif ACUTE in marks or GRAVE in marks:
            if (not perfective and "perfective" in stags
                    and "imperfective" not in stags):
                continue
            groups = nuclei(sw)
            pos = next((len(groups) - 1 - k for k, g in enumerate(groups)
                        if index in g), None)
            if pos == 1 and vowel_length(sw, groups[-1], final=True) \
                    == "short":
                votes["short"] += n
    if not votes:
        return None
    return max(votes.items(), key=lambda x: (x[1], x[0]))[0]


def _stem_subscript(word: list[Letter], index: int,
                    siblings: list[Sibling]) -> bool:
    """Whether most tokens of the verb's attested forms with the same
    letters up to the letter after ``index``, and an accent at ``index``,
    write an iota subscript under that vowel. The next letter must match:
    the subscript of a contraction (ἀγαπᾷ) is no part of the uncontracted
    ending (ἀγαπάει). A spelling with fewer than ``SIBLING_MIN_TOKENS``
    tokens does not vote."""
    key = "".join(l.base for l in word[:index + 2])
    votes: Counter = Counter()
    for sib in siblings:
        s, n = sib[0], sib[1]
        if n < SIBLING_MIN_TOKENS:
            continue
        sw = letters(s)
        if ("".join(l.base for l in sw[:index + 2]) == key
                and len(sw) > index and sw[index].marks & TONAL):
            votes[SUBSCRIPT in sw[index].marks] += n
    return bool(votes) and votes[True] > votes[False]


def _is_participle(tags: frozenset) -> bool:
    return FINITE not in tags and ("participle" in tags or bool(
        {"masculine", "feminine", "neuter"} & tags))


def _aorist_passive_participle(tags: frozenset, word: list[Letter]) -> bool:
    """The nominative of an aorist passive participle in -είς (γραφείς,
    σταλείς, δοθείς), whose long last syllable takes the acute; any other
    participle accented there follows its vowel (ἐπιζῶν, τηρῶν,
    πυροδοτοῦν), as the -θείς of a finite dependent form does (δοθεῖς)."""
    return (_is_participle(tags)
            and "".join(l.base for l in word[-3:]) in ("εις", "εισ"))


def polytonic(mono: str, tags: frozenset, vclass: str,
              siblings: list[Sibling],
              evidence: Evidence, lemma: str = "") -> str | None:
    """The polytonic spelling of the monotonic verb form ``mono`` of
    ``lemma``, or None when it cannot be decided (an unaccented word of
    several syllables)."""
    word = letters(unicodedata.normalize("NFC", mono.lower()))
    groups = nuclei(word)
    if not groups:
        return None
    accented = [n for n, g in enumerate(groups)
                if any(ACUTE in word[i].marks for i in g)]
    if len(accented) > 1:
        return None
    if accented:
        n = accented[0]
        carrier = next(i for i in groups[n] if ACUTE in word[i].marks)
    elif len(nuclei(word, glides=True)) == 1:
        # Monotonic writing leaves a word of one syllable unaccented.
        n = len(groups) - 1
        carrier = accent_carrier(word, groups[n])
    else:
        return None
    position = len(groups) - 1 - n
    # The vowel an accented last syllable carries is long or short as
    # written; only for an earlier accent does a final αι or οι count short.
    length = vowel_length(word, groups[n], final=False)
    if length == "either":
        length = _stem_length(word, carrier, siblings, tags) or (
            "long" if _contracted(vclass, tags, word, groups, n, lemma)
            else "short")
    if position >= 2:
        accent = ACUTE
    elif position == 1:
        last = vowel_length(word, groups[-1], final=True)
        if last != "long" and (_long_final(tags, word, vclass)
                               or (GERUND_ACUTE and _gerund(word))):
            last = "long"
        accent = CIRCUMFLEX if length == "long" and last != "long" else ACUTE
    else:
        accent = CIRCUMFLEX if length == "long" \
            and not _aorist_passive_participle(tags, word) else ACUTE
    out = [Letter(l.base, l.marks - {ACUTE}) for l in word]
    out[carrier] = Letter(out[carrier].base, out[carrier].marks | {accent})
    if out[carrier].base in "αηω" and (
            _stem_subscript(word, carrier, siblings)
            or (accent == CIRCUMFLEX and position == 0
                and out[carrier].base == "α"
                and cell(tags) in ("present", "other")
                and _contract_subscript(word, carrier, siblings, evidence))):
        out[carrier] = Letter(out[carrier].base,
                              out[carrier].marks | {SUBSCRIPT})
    key = "".join(l.base for l in word)
    if word[0].base in VOWELS or word[0].base == "ρ":
        if word[0].base == "ρ":
            breathing = ROUGH
        elif _augment(tags, key, lemma):
            breathing = SMOOTH
        else:
            breathing = evidence.breathing(key, siblings)
        target = 0
        if (len(word) > 1 and word[0].base + word[1].base in DIPHTHONGS
                and DIAERESIS not in word[1].marks):
            target = 1
        out[target] = Letter(out[target].base, out[target].marks | {breathing})
    return compose(out)


def _long_final(tags: frozenset, word: list[Letter], vclass: str) -> bool:
    """A final syllable the texts treat as long where its vowel would not
    say so: the -α of an imperative (φεύγα, and ζήτα, βοήθα, contracted
    from -αε in the -άω verbs, which Wiktionary may tag only as a form).
    The gerund's -ώντας and -ῶντας are written about equally often (283
    tokens to 241); the vowel's rule gives the circumflex, and the acute
    of the majority is kept by ``GERUND_ACUTE``."""
    if word[-1].base == "α" and ("imperative" in tags or (
            vclass == "B1" and FINITE not in tags
            and not {"past", "imperfect", "aorist"} & tags)):
        return True
    return False


# The gerund in -ώντας: the texts write the acute 283 times and the
# circumflex 241 in the paradigms' gerunds (μιλώντας, ζητῶντας); the acute
# is generated.
GERUND_ACUTE = True


def _gerund(word: list[Letter]) -> bool:
    return "".join(l.base for l in word[-5:]) in ("ωντασ", "ωντας")


def _augment(tags: frozenset, key: str, lemma: str) -> bool:
    """Whether the form's initial vowel is the augment of a past tense
    (ἔγραψα, ἦρθα, ἦβρα), which takes the smooth breathing: a past form
    whose verb begins with another letter."""
    if cell(tags) not in ("past", "imperfect") or not lemma:
        return False
    if len(key) < 2 or key[1] in VOWELS:
        return False      # ηὗρα keeps the breathing of its stem (εὑρ-)
    return bare(lemma)[:1] != key[:1]


def _contracted(vclass: str, tags: frozenset, word: list[Letter],
                groups: list[list[int]], n: int, lemma: str = "") -> bool:
    """Whether the accented α, ι or υ is long by contraction: in the
    present and imperative endings of the -άω verbs and of those Wiktionary
    files under a polytonic -ῶ (μιλᾶμε, μιλᾶτε, μιλᾶς, μιλᾶνε), in the
    present of the passive -άμαι and -ώμαι (θυμᾶμαι, φοβᾶται, and
    αὐταπατᾶσαι of αυταπατώμαι), in the
    imperative plural -ᾶτε of the others (ἐλᾶτε, τρεχᾶτε), and in a word
    of one syllable (πᾶς, φᾶς)."""
    if len(nuclei(word, glides=True)) == 1:
        return True
    carrier = groups[n][-1]
    if word[carrier].base != "α":
        return False
    rest = "".join(l.base for l in word[carrier + 1:])
    if ((vclass == "B1" or lemma.endswith("ῶ"))
            and rest in CONTRACTED_ENDINGS
            and ("present" in tags or "imperative" in tags
                 or FINITE not in tags)
            and "perfective" not in tags):
        return True
    if (vclass == "passive"
            and lemma.endswith(("άμαι", "ᾶμαι", "ώμαι", "ῶμαι"))
            and rest in CONTRACTED_PASSIVE_ENDINGS
            and "perfective" not in tags):
        return True
    return ("imperative" in tags and "plural" in tags
            and "".join(l.base for l in word[-3:]) == "ατε")


# What follows the contracted α of the -άω verbs' present (μιλᾶ, μιλᾶς,
# μιλᾶμε, μιλᾶτε, μιλᾶνε, μιλᾶν), and of the passive -άμαι's (θυμᾶμαι,
# θυμᾶσαι, θυμᾶται, θυμᾶστε).
CONTRACTED_ENDINGS = frozenset({"", "ς", "σ", "με", "τε", "νε", "ν"})
CONTRACTED_PASSIVE_ENDINGS = frozenset({"μαι", "σαι", "ται", "στε"})


def _contract_subscript(word: list[Letter], index: int,
                        siblings: list[Sibling],
                        evidence: Evidence) -> bool:
    """Whether the contracted -ᾶς or -ᾶ takes the iota subscript (ἀγαπᾷς),
    as most attested tokens of such forms of all verbs do (the same verb's
    texts often write both). A word of one syllable (πᾶς, φᾶς) does not."""
    ending = "ᾶς" if index == len(word) - 2 and word[-1].base in "σς" \
        else "ᾶ" if index == len(word) - 1 else None
    if ending is None:
        return False
    if len(nuclei(word, glides=True)) == 1:
        return False
    return evidence.subscript_share.get(ending, 0) > 0.5


def subjunctive_variant(spelling: str) -> str | None:
    """The traditional subjunctive spelling of a second or third person
    singular in -εις or -ει: the ει written ῃ (πάρεις -> πάρῃς, δοθεῖ ->
    δοθῇ). None for other spellings."""
    word = letters(spelling)
    tail = 2 if word and word[-1].base in "σς" else 1
    if len(word) < tail + 2:
        return None
    e, i = word[-tail - 1], word[-tail]
    if e.base != "ε" or i.base != "ι" or e.marks or DIAERESIS in i.marks:
        return None
    accent = i.marks & TONAL
    if accent:
        accent = {CIRCUMFLEX}
    eta = Letter("η", frozenset(accent | {SUBSCRIPT}))
    return compose(word[:-tail - 1] + [eta] + word[len(word) - tail + 1:])


SUBJUNCTIVE_CELLS = frozenset({"dependent", "present"})
# Tags that say a form is no second or third person singular of either.
NOT_SUBJUNCTIVE_TAGS = frozenset({
    "past", "imperfect", "aorist", "imperative", "plural", "first-person",
    "participle", "masculine", "feminine", "neuter", "infinitive"})


def is_subjunctive_cell(tags: frozenset, form: str = "") -> bool:
    """A second or third person singular of the dependent forms or of the
    present (which serves as the imperfective subjunctive). Wiktionary tags
    many such forms only as a form of their verb (ξαναδεῖ, ξεφύγεις,
    ὁδηγεῖς); a finite verb form in -ει or -εις is one of these cells
    whenever no tag says otherwise."""
    if (cell(tags) in SUBJUNCTIVE_CELLS and "singular" in tags
            and ({"second-person", "third-person"} & tags)):
        return True
    if not form or tags & NOT_SUBJUNCTIVE_TAGS:
        return False
    letters_ = bare(form)
    return letters_.endswith(("ει", "εισ")) and len(letters_) > 3


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_verb_paradigms(path: Path = VERB_PARADIGMS
                        ) -> dict[str, dict[str, frozenset]]:
    """lemma -> {monotonic form: union of its tags} for Wiktionary's Modern
    Greek verbs.

    A lemma in capitals is an abbreviation or a heading Wiktionary files
    verb forms under (ΔΕΣ for δες, ΠΒ), and is left out with its forms. A
    capitalized one is a page title in the sentence case of a heading
    (Ερωτώ, Ντυμένος) and is read in lowercase, so its forms (ρώτησα,
    ντυμένη) join their verb; a name among them (Μιλάνος, with μιλάν) is
    no verb, and a form it shares with one is the verb's
    (``_home_lemma``). The fragments of tables Wiktionary could not split
    are left out too (``drop_table_fragments``), and so are the rows tagged
    ``RARE_CELL_TAGS`` whose letters are also another part of speech's."""
    with open(path, encoding="utf-8") as fh:
        pairs = json.load(fh)
    out: dict[str, dict[str, frozenset]] = defaultdict(dict)
    rare: dict[str, dict[str, frozenset]] = defaultdict(dict)
    other_words: set[str] = set()
    for p in pairs:
        form = unicodedata.normalize("NFC", p.get("form") or "")
        if p.get("pos") != "verb":
            other_words.add(form.lower())
            continue
        tags = frozenset(p.get("tags") or ())
        lemma = unicodedata.normalize("NFC", p["lemma"])
        if not form or " " in form or not form.islower():
            continue
        if not lemma or (len(lemma) > 1 and lemma.isupper()):
            continue
        lemma = lemma.lower()
        if tags & PERSON_TAGS:
            # A form that is also a finite cell is accented as one, whatever
            # participle tags another of its rows carries.
            tags |= {FINITE}
        target = rare if tags & RARE_CELL_TAGS else out
        target[lemma][form] = target[lemma].get(form, frozenset()) | tags
    for lemma, forms in rare.items():
        for form, tags in forms.items():
            if form in out.get(lemma, {}):
                out[lemma][form] |= tags
            elif form not in other_words:
                out[lemma][form] = tags
    return drop_table_fragments(dict(out))


# The tags that place a form in its paradigm, for comparing a form with the
# endings of other verbs' forms (``drop_table_fragments``).
CELL_TAGS = frozenset({
    "present", "past", "imperfect", "dependent", "imperative", "aorist",
    "first-person", "second-person", "third-person", "singular", "plural",
    "passive", "active", "participle"})
# A form is a fragment when this share of all the verbs end a form of the
# same cell in its letters (``drop_table_fragments``).
FRAGMENT_SHARE = 0.05


def _initials(form: str) -> set[str]:
    """The first letter of a form, and after a past tense's augment (ε, η,
    ει before a consonant) the letter that follows it."""
    b = bare(form)
    out = {b[:1]}
    for augment in ("ει", "ε", "η"):
        if b.startswith(augment) and len(b) > len(augment) + 1 \
                and b[len(augment)] not in VOWELS:
            out.add(b[len(augment)])
    return out


def drop_table_fragments(paradigms: dict[str, dict[str, frozenset]]
                         ) -> dict[str, dict[str, frozenset]]:
    """The paradigms without the pieces of tables Wiktionary could not
    split into stem and ending: σαλπάρω's table holds ούμε, όμαστε, όσουν,
    όμουν and όταν, endings without their stem. Such a piece starts with
    another letter than its verb, even after an augment, and is the very
    ending of the same cell in at least ``FRAGMENT_SHARE`` of all the verbs
    (όμαστε ends the present passive first person plural of hundreds). A
    suppletive form also starts otherwise (φάγατε, είπα, δώσε, ήρθα), but
    ends no other verbs' forms but its own compounds'."""
    suspects: dict[str, set[frozenset]] = defaultdict(set)
    for lemma, forms in paradigms.items():
        for form, tags in forms.items():
            if bare(lemma)[:1] not in _initials(form):
                suspects[bare(form)].add(tags & CELL_TAGS)
    # The verbs with a longer form ending in a suspect's letters, in a cell
    # its tags allow (all of them, where Wiktionary gave it no others).
    users: dict[tuple[str, frozenset], set[str]] = defaultdict(set)
    for lemma, forms in paradigms.items():
        for form, tags in forms.items():
            letters_ = bare(form)
            cell_tags = tags & CELL_TAGS
            for k in range(1, len(letters_)):
                for suspect_tags in suspects.get(letters_[-k:], ()):
                    if suspect_tags <= cell_tags:
                        users[(letters_[-k:], suspect_tags)].add(lemma)
    floor = FRAGMENT_SHARE * len(paradigms)
    out: dict[str, dict[str, frozenset]] = {}
    for lemma, forms in paradigms.items():
        kept = {form: tags for form, tags in forms.items()
                if bare(lemma)[:1] in _initials(form)
                or len(users.get((bare(form), tags & CELL_TAGS), set())
                       - {lemma}) < floor}
        if kept:
            out[lemma] = kept
    return out


def load_form_frequencies(path: Path = MG_FORM_FREQ) -> dict[str, int]:
    """Monotonic form -> token count."""
    freq: dict[str, int] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rsplit(" ", 1)
            if len(parts) == 2 and parts[1].strip().isdigit():
                form = unicodedata.normalize("NFC", parts[0])
                freq[form] = freq.get(form, 0) + int(parts[1])
    return freq


class Generated(NamedTuple):
    spelling: str
    mono: str          # the monotonic form it spells (letters of the variant
    lemma: str         # differ for a subjunctive variant)
    tokens: int        # the monotonic form's count
    kind: str          # "indicative" or "subjunctive"
    vclass: str
    cell: str


def _home_lemma(form: str, lemmas: list[str],
                paradigms: dict[str, dict[str, frozenset]],
                totals: dict[str, int]) -> str:
    """The verb a form shared by several is generated as: a verb rather
    than a page of another word (μιλάν is μιλάω's, not the name Μιλάνος's),
    then the one whose letters it shares the longest beginning with
    (πουλάμε is πουλάω's, not πωλώ's), then the one monotonic text uses
    most, then the first in alphabetical order."""
    letters_ = bare(form)

    def verb(lemma: str) -> bool:
        return lemma.endswith(VERB_LEMMA_ENDINGS)

    def common(lemma: str) -> int:
        other = bare(lemma)
        n = 0
        while n < min(len(letters_), len(other)) and letters_[n] == other[n]:
            n += 1
        return n
    return min(lemmas, key=lambda l: (not verb(l), -common(l),
                                      -totals.get(l, 0), l))


# The endings of a verb's dictionary form (γράφω, μιλώ, μιλῶ, έρχομαι,
# πρέπει, βρέχει, πρόκειται).
VERB_LEMMA_ENDINGS = ("ω", "ώ", "ῶ", "μαι", "ει", "εί", "ται")


def generate(paradigms: dict[str, dict[str, frozenset]],
             frequencies: dict[str, int],
             evidence: Evidence,
             min_tokens: int = GENERATED_MIN_TOKENS,
             ) -> list[Generated]:
    """Polytonic spellings for every verb form monotonic text has at least
    ``min_tokens`` times, and the subjunctive spellings of its second and
    third person singular, whether or not the texts attest them; the
    caller drops what the dictionaries already have.

    A form several verbs share (επόμενες, of επόμενος and έπομαι; πουλάμε,
    of πουλάω and πωλώ) gets one spelling: it reads the attested spellings
    of all of their forms, and takes its tags and class from the verb it
    shares the most letters with (``_home_lemma``). The candidates come out
    most frequent first, so that a caller keeping the first of a set of
    spellings with the same letters keeps the commonest."""
    lemmas_of: dict[str, list[str]] = defaultdict(list)
    for lemma, forms in paradigms.items():
        for form in forms:
            if frequencies.get(form, 0) >= min_tokens:
                lemmas_of[form].append(lemma)
    if not lemmas_of:
        return []
    totals = {lemma: sum(frequencies.get(f, 0) for f in forms)
              for lemma, forms in paradigms.items()}
    own = {lemma: evidence.lemma_spellings(paradigms[lemma])
           for lemma in {l for ls in lemmas_of.values() for l in ls}}
    pooled: dict[tuple[str, ...], list[Sibling]] = {}
    out: list[Generated] = []
    for form in sorted(lemmas_of, key=lambda f: (-frequencies[f], f)):
        lemmas = sorted(lemmas_of[form])
        lemma = _home_lemma(form, lemmas, paradigms, totals)
        key = tuple(lemmas)
        if key not in pooled:
            pooled[key] = own[lemma] if len(lemmas) == 1 else _pool(
                [own[l] for l in lemmas])
        tags = paradigms[lemma][form]
        vclass = verb_class(lemma, paradigms[lemma])
        spelling = polytonic(form, tags, vclass, pooled[key], evidence, lemma)
        if spelling is None:
            continue
        n = frequencies[form]
        out.append(Generated(spelling, form, lemma, n, "indicative",
                             vclass, cell(tags)))
        if is_subjunctive_cell(tags, form):
            variant = subjunctive_variant(spelling)
            # Monotonic writing spells the subjunctive -ει(ς), and writes it
            # -η(ς) only by mistake, a few times in a hundred (σταματήσης,
            # 80, beside σταματήσεις). Where it has the variant's letters
            # and accent GENERATED_MIN_TOKENS times and at least a tenth as
            # often as the form, they are another word's: the nouns
            # υπολογιστή (12,664) beside υπολογιστεί, θέσης beside θέσεις.
            written = frequencies.get(monotonic_key(variant), 0) \
                if variant else 0
            if variant and (written < min_tokens or written * 10 < n):
                out.append(Generated(variant, form, lemma, n,
                                     "subjunctive", vclass, cell(tags)))
    return out


def _pool(lists: list[list[Sibling]]) -> list[Sibling]:
    """Several verbs' attested spellings as one list, in the same order."""
    tokens: dict[str, int] = {}
    tags: dict[str, frozenset] = defaultdict(frozenset)
    for siblings in lists:
        for sib in siblings:
            tokens[sib.spelling] = max(tokens.get(sib.spelling, 0), sib.tokens)
            tags[sib.spelling] |= sib.tags
    return [Sibling(s, n, tags[s])
            for s, n in sorted(tokens.items(), key=lambda x: (-x[1], x[0]))]
