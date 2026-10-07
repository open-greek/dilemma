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
reader can weight them below the attested spellings of the list.
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
# Paradigm cells that are not the spellings of everyday text.
SKIPPED_TAGS = frozenset({"rare", "archaic", "error-unrecognized-form",
                          "dialectal", "Katharevousa", "obsolete", "dated"})

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
    ``prefix_breathing`` maps the first two to four bare letters of the
    grc dictionary's vowel-initial words to the count of each breathing.
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
            for k in (2, 3, 4):
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

    def lemma_spellings(self, forms: Iterable[str],
                        exclude: frozenset[str] = frozenset(),
                        ) -> list[tuple[str, int]]:
        """The attested spellings of a verb's ``forms`` with their token
        counts, the most frequent first, leaving out the monotonic keys in
        ``exclude``."""
        found: Counter = Counter()
        for key in {monotonic_key(form) for form in forms} - exclude:
            for s, n in self.attested.get(key, {}).items():
                found[s] += n
        return found.most_common()

    def breathing(self, key: str, siblings: list[tuple[str, int]]) -> str:
        """The breathing for a word whose bare letters are ``key``: the one
        the verb's attested forms with the same first two letters write
        most, else the grc words' with the same first letters."""
        votes: Counter = Counter()
        for s, n in siblings:
            if bare(s)[:2] == key[:2]:
                b = breathing_of(s)
                if b:
                    votes[b] += n
        if votes:
            return votes.most_common(1)[0][0]
        for k in (4, 3, 2):
            counts = self.prefix_breathing.get(key[:k])
            if counts and sum(counts.values()) >= 5:
                b, n = counts.most_common(1)[0]
                if n >= 0.9 * sum(counts.values()):
                    return b
        return ROUGH if key[:1] == "υ" else SMOOTH


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
                 siblings: list[tuple[str, int]],
                 tags: frozenset = frozenset()) -> str | None:
    """``long`` or ``short`` for the vowel at ``index`` (α, ι or υ), by the
    tokens of the verb's attested forms with the same letters up to it that
    accent it where the type of the accent shows its length: a circumflex
    (long), or an acute on a penult before a short last syllable (short).

    The vowel before the σ, ξ or ψ of a perfective stem is the stem's
    (γελάστε, κοιτάξτε), and the long α of an imperfective contraction
    (γελᾶς, γελᾶτε, κοιτᾶς) says nothing of it, so there the forms must
    have the same letters through the next one as well."""
    strict = ("perfective" in tags and index + 1 < len(word)
              and word[index + 1].base in "σξψ")
    key = "".join(l.base for l in word[:index + 1 + strict])
    votes: Counter = Counter()
    for s, n in siblings:
        sw = letters(s)
        if "".join(l.base for l in sw[:len(key)]) != key:
            continue
        marks = sw[index].marks
        if CIRCUMFLEX in marks:
            votes["long"] += n
        elif ACUTE in marks or GRAVE in marks:
            groups = nuclei(sw)
            pos = next((len(groups) - 1 - k for k, g in enumerate(groups)
                        if index in g), None)
            if pos == 1 and vowel_length(sw, groups[-1], final=True) \
                    == "short":
                votes["short"] += n
    return votes.most_common(1)[0][0] if votes else None


def _stem_subscript(word: list[Letter], index: int,
                    siblings: list[tuple[str, int]]) -> bool:
    """Whether most tokens of the verb's attested forms with the same
    letters up to the letter after ``index``, and an accent at ``index``,
    write an iota subscript under that vowel. The next letter must match:
    the subscript of a contraction (ἀγαπᾷ) is no part of the uncontracted
    ending (ἀγαπάει)."""
    key = "".join(l.base for l in word[:index + 2])
    votes: Counter = Counter()
    for s, n in siblings:
        sw = letters(s)
        if ("".join(l.base for l in sw[:index + 2]) == key
                and len(sw) > index and sw[index].marks & TONAL):
            votes[SUBSCRIPT in sw[index].marks] += n
    return bool(votes) and votes[True] > votes[False]


def _is_participle(tags: frozenset) -> bool:
    return FINITE not in tags and ("participle" in tags or bool(
        {"masculine", "feminine", "neuter"} & tags))


def polytonic(mono: str, tags: frozenset, vclass: str,
              siblings: list[tuple[str, int]],
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
            "long" if _contracted(vclass, tags, word, groups, n) else "short")
    if position >= 2:
        accent = ACUTE
    elif position == 1:
        last = vowel_length(word, groups[-1], final=True)
        if last != "long" and (_long_final(tags, word, vclass)
                               or (GERUND_ACUTE and _gerund(word))):
            last = "long"
        accent = CIRCUMFLEX if length == "long" and last != "long" else ACUTE
    else:
        nominative_participle = _is_participle(tags) and not (
            {"genitive", "plural"} <= tags)
        accent = CIRCUMFLEX if length == "long" and not nominative_participle \
            else ACUTE
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
    The gerund's -ώντας and -ῶντας are written about equally often (303
    tokens to 267); the vowel's rule gives the circumflex, and the acute
    of the majority is kept by ``GERUND_ACUTE``."""
    if word[-1].base == "α" and ("imperative" in tags or (
            vclass == "B1" and FINITE not in tags
            and not {"past", "imperfect", "aorist"} & tags)):
        return True
    return False


# The gerund in -ώντας: the texts write the acute 303 times and the
# circumflex 267 (μιλώντας, ζητῶντας); the acute is generated.
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
                groups: list[list[int]], n: int) -> bool:
    """Whether the accented α, ι or υ is long by contraction: in the
    present and imperative endings of the -άω verbs (μιλᾶμε, μιλᾶτε,
    μιλᾶς), in the imperative plural -ᾶτε of the others (ἐλᾶτε, τρεχᾶτε),
    and in a word of one syllable (πᾶς, φᾶς)."""
    if len(nuclei(word, glides=True)) == 1:
        return True
    carrier = groups[n][-1]
    if word[carrier].base != "α":
        return False
    rest = "".join(l.base for l in word[carrier + 1:])
    if (vclass == "B1" and rest in CONTRACTED_ENDINGS
            and ("present" in tags or "imperative" in tags
                 or FINITE not in tags)):
        return True
    return ("imperative" in tags and "plural" in tags
            and "".join(l.base for l in word[-3:]) == "ατε")


# What follows the contracted α of the -άω verbs' present (μιλᾶ, μιλᾶς,
# μιλᾶμε, μιλᾶτε, μιλᾶνε, μιλᾶν).
CONTRACTED_ENDINGS = frozenset({"", "ς", "σ", "με", "τε", "νε", "ν"})


def _contract_subscript(word: list[Letter], index: int,
                        siblings: list[tuple[str, int]],
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


def is_subjunctive_cell(tags: frozenset) -> bool:
    """A second or third person singular of the dependent forms or of the
    present (which serves as the imperfective subjunctive)."""
    return (cell(tags) in SUBJUNCTIVE_CELLS and "singular" in tags
            and ({"second-person", "third-person"} & tags))


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_verb_paradigms(path: Path = VERB_PARADIGMS
                        ) -> dict[str, dict[str, frozenset]]:
    """lemma -> {monotonic form: union of its tags} for Wiktionary's Modern
    Greek verbs, leaving out cells tagged ``SKIPPED_TAGS``."""
    with open(path, encoding="utf-8") as fh:
        pairs = json.load(fh)
    out: dict[str, dict[str, frozenset]] = defaultdict(dict)
    for p in pairs:
        if p.get("pos") != "verb":
            continue
        tags = frozenset(p.get("tags") or ())
        if tags & SKIPPED_TAGS:
            continue
        form = unicodedata.normalize("NFC", p["form"])
        lemma = unicodedata.normalize("NFC", p["lemma"])
        if not form or " " in form or not form.islower():
            continue
        if tags & PERSON_TAGS:
            # A form that is also a finite cell is accented as one, whatever
            # participle tags another of its rows carries.
            tags |= {FINITE}
        out[lemma][form] = out[lemma].get(form, frozenset()) | tags
    return dict(out)


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


def generate(paradigms: dict[str, dict[str, frozenset]],
             frequencies: dict[str, int],
             evidence: Evidence,
             min_tokens: int = GENERATED_MIN_TOKENS,
             ) -> list[Generated]:
    """Polytonic spellings for every verb form monotonic text has at least
    ``min_tokens`` times, and the subjunctive spellings of its second and
    third person singular, whether or not the texts attest them; the
    caller drops what the dictionaries already have."""
    out: list[Generated] = []
    for lemma, forms in paradigms.items():
        frequent = {f: t for f, t in forms.items()
                    if frequencies.get(f, 0) >= min_tokens}
        if not frequent:
            continue
        vclass = verb_class(lemma, forms)
        siblings = evidence.lemma_spellings(forms)
        for form, tags in frequent.items():
            spelling = polytonic(form, tags, vclass, siblings, evidence,
                                 lemma)
            if spelling is None:
                continue
            n = frequencies.get(form, 0)
            out.append(Generated(spelling, form, lemma, n, "indicative",
                                 vclass, cell(tags)))
            if is_subjunctive_cell(tags):
                variant = subjunctive_variant(spelling)
                if variant:
                    out.append(Generated(variant, form, lemma, n,
                                         "subjunctive", vclass, cell(tags)))
    return out
