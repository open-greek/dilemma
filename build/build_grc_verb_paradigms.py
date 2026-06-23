#!/usr/bin/env python3
"""Build per-lemma Ancient Greek verb paradigms from dilemma's tagged pairs.

Aggregates tagged form->lemma pairs (kaikki + GLAUx + LSJ expansion) into
per-lemma paradigm dicts keyed by `<voice>_<tense>_<mood>_<person><number>`,
with infinitives keyed `<voice>_<tense>_infinitive` and participle nominative
forms keyed `<voice>_<tense>_participle_nom_<gender>_sg`. Output schema
matches `jtauber_ag_paradigms.json` so a downstream canonical-paradigm
builder can consume the file as a drop-in alternative or supplement.

Inputs (from data/):
  - ag_pairs.json     Wiktionary kaikki form-lemma pairs (now tense-tagged)
  - glaux_pairs.json  GLAUx corpus, fully morph-tagged
  - verb_extra_pairs.json (optional) pairs from LSJ expansion

Output:
  - data/ag_verb_paradigms.json    {lemma: {forms: {key: form},
                                            form_count: N,
                                            source: "dilemma"}}

Dialect handling: the default Attic forms (no dialect tag, or explicit
"Attic") populate the top-level paradigm. Forms tagged with one of
{Epic, Ionic, Doric, Aeolic, Koine, ...} are emitted under
`forms.<dialect_lower>` as a parallel paradigm slice. A downstream consumer
currently only reads the Attic slice; we keep the dialect data so it can
be picked up later without rebuilding.

Usage:
  python build/build_grc_verb_paradigms.py
  python build/build_grc_verb_paradigms.py --sanity   # 5-lemma smoke test
  python build/build_grc_verb_paradigms.py --only γράφω,τίθημι
"""

import argparse
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"

# Make sibling build/ modules importable when this file is run as a
# script (`python build/build_grc_verb_paradigms.py`).
_BUILD_DIR = Path(__file__).resolve().parent
if str(_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_DIR))

# Make the dilemma package importable for the augment helpers in
# dilemma.morph_diff (used by `is_augmented_past_indicative`).
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dilemma.morph_diff import (  # noqa: E402
    _detect_syllabic_augment,
    _detect_temporal_augment,
    _strip_diacritics,
)

AG_PAIRS = DATA_DIR / "ag_pairs.json"
GLAUX_PAIRS = DATA_DIR / "glaux_pairs.json"
VERB_EXTRA_PAIRS = DATA_DIR / "verb_extra_pairs.json"
LSJ_VERB_PAIRS = DATA_DIR / "ag_lsj_verb_pairs.json"  # produced by expand_lsj
LSJ_HEADWORDS_PATH = DATA_DIR / "lsj_headwords.json"
LSJ9_GLOSSES = Path.home() / "Documents" / "lsj9" / "lsj9_glosses.jsonl"
OUT_PATH = DATA_DIR / "ag_verb_paradigms.json"


def strip_accents(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(c for c in nfd if not unicodedata.combining(c)))

# Tag vocabulary - matches dilemma/Wiktionary tag strings
TENSE_TAGS = {
    "present", "imperfect", "future", "aorist",
    "perfect", "pluperfect", "future-perfect",
}
VOICE_TAGS = {"active", "middle", "passive", "mediopassive"}
MOOD_TAGS = {
    "indicative", "subjunctive", "optative", "imperative",
    "infinitive", "participle",
}
PERSON_TAGS = {"first-person", "second-person", "third-person"}
NUMBER_TAGS = {"singular", "plural", "dual"}
CASE_TAGS = {"nominative", "genitive", "dative", "accusative", "vocative"}
GENDER_TAGS = {"masculine", "feminine", "neuter"}
DIALECT_TAGS = {
    "Attic", "Epic", "Ionic", "Doric", "Koine", "Aeolic",
    "Homeric", "Laconian", "Boeotian", "Arcadocypriot",
}

NUMBER_SHORT = {"singular": "sg", "plural": "pl", "dual": "du"}
PERSON_SHORT = {"first-person": "1", "second-person": "2", "third-person": "3"}
CASE_SHORT = {
    "nominative": "nom", "genitive": "gen", "dative": "dat",
    "accusative": "acc", "vocative": "voc",
}
GENDER_SHORT = {"masculine": "m", "feminine": "f", "neuter": "n"}


def grave_to_acute(s: str) -> str:
    """Convert combining grave (U+0300) to combining acute (U+0301).

    For citation forms the acute is canonical; the grave only appears
    mid-sentence to avoid stacking acutes.
    """
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", nfd.replace("̀", "́"))


def has_polytonic(s: str) -> bool:
    nfd = unicodedata.normalize("NFD", s)
    return any(c in nfd for c in ("̓", "̔", "͂"))


def is_stripped(s: str) -> bool:
    nfd = unicodedata.normalize("NFD", s)
    return not any(unicodedata.combining(c) for c in nfd)


def is_elided(s: str) -> bool:
    return bool(s) and s[-1] in ("'", "’", "ʼ", "᾽", "ʹ")


_GREEK_RANGES = [(0x0370, 0x03FF), (0x1F00, 0x1FFF)]
_GREEK_DIACRITICS_RANGE = (0x0300, 0x036F)


def is_pure_greek(s: str) -> bool:
    """True if every character is a Greek letter / diacritic.

    Forms / lemmas that contain non-Greek punctuation (hyphens between
    LSJ compound prefixes, modifier apostrophes from old OCR scans,
    digits, Latin letters, etc.) are filtered out. dilemma's downstream
    consumers expect bare Greek tokens.
    """
    if not s:
        return False
    for c in s:
        cp = ord(c)
        if not any(lo <= cp <= hi for lo, hi in _GREEK_RANGES):
            return False
    return True


def has_internal_capital(s: str) -> bool:
    """True when the lemma has a capital letter past position 0.

    Verbs are essentially never proper nouns, and capitals after the
    first character are a strong signal of a corpus annotation glitch
    (ΒΕἔστημι), prefixed table cells leaking into a lemma slot, or
    accidentally-spliced strings. We drop these entirely rather than
    try to canonicalize.
    """
    nfd = unicodedata.normalize("NFD", s)
    chars = [c for c in nfd if not unicodedata.combining(c)]
    if not chars:
        return False
    for c in chars[1:]:
        if c.isupper():
            return True
    return False


def lowercase_initial(s: str) -> str:
    """Lowercase only the first letter; preserve diacritics on the
    underlying letter via NFD/NFC roundtrip. Used to normalize verb
    lemmas like Τίκτω -> τίκτω where the corpus capitalized a sentence-
    initial token."""
    if not s:
        return s
    nfd = unicodedata.normalize("NFD", s)
    out = []
    seen_letter = False
    for c in nfd:
        if not seen_letter and not unicodedata.combining(c) and c.isalpha():
            out.append(c.lower())
            seen_letter = True
        else:
            out.append(c)
    return unicodedata.normalize("NFC", "".join(out))


_PAST_INDICATIVE_TENSES = ("aorist", "imperfect", "pluperfect")


def is_past_indicative_key(key: str) -> bool:
    """True when ``key`` names a past-tense indicative cell.

    Augment is mandatory in indicative aorist / imperfect / pluperfect.
    Non-indicative moods (subjunctive / optative / imperative / infinitive
    / participle) never carry augment, so this check is gated on
    ``_indicative`` appearing alongside one of the past-tense markers.
    """
    if not key:
        return False
    if "_indicative_" not in key and not key.endswith("_indicative"):
        return False
    return any(t in key for t in _PAST_INDICATIVE_TENSES)


def has_augment(form: str, lemma: str) -> bool:
    """True when ``form`` carries a syllabic or temporal augment vs ``lemma``.

    Wraps the augment detectors in :mod:`dilemma.morph_diff`. Used to
    rank past-indicative variants in :func:`pick_best_form` so we never
    emit an unaugmented 3sg / 3pl when an augmented variant is also
    attested in the corpus.
    """
    if not form or not lemma:
        return False
    fb = _strip_diacritics(form)
    lb = _strip_diacritics(lemma)
    if _detect_syllabic_augment(fb, lb):
        return True
    if _detect_temporal_augment(fb, lb):
        return True
    return False


def pick_best_form(forms, key: str | None = None, lemma: str | None = None):
    """Pick the canonical surface form from a list of variants.

    For past-indicative cells (aorist / imperfect / pluperfect) where
    ``lemma`` is provided we prefer augment-bearing variants. Multiple
    Wiktionary / GLAUx entries for the same cell often include both
    Homeric un-augmented forms (``λῦσε``, ``λῦσαν``) and Attic augmented
    forms (``ἔλυσε``, ``ἔλυσαν``); without the augment preference, the
    `-len(f)` tie-breaker silently picks the shorter, un-augmented
    variant for the canonical Attic slice.
    """
    if not forms:
        return None
    if isinstance(forms, set):
        forms = list(forms)
    polyt = [f for f in forms if has_polytonic(f)]
    diacr = [f for f in forms if not is_stripped(f)]
    pool = polyt or diacr or list(forms)
    no_elide = [f for f in pool if not is_elided(f)]
    if no_elide:
        pool = no_elide
    counts = Counter(forms)
    prefer_augment = bool(lemma) and is_past_indicative_key(key or "")
    return max(pool, key=lambda f: (
        counts[f],          # most attested wins
        # Past-indicative cells: augment-bearing forms win over un-
        # augmented variants regardless of count, so we never emit
        # λῦσε / λῦσαν over ἔλυσε / ἔλυσαν for the canonical 3sg / 3pl.
        has_augment(f, lemma) if prefer_augment else False,
        has_polytonic(f),   # break ties by polytonic richness
        -len(f),            # shorter wins (ἐστί over ἐστίν)
        f,                  # alphabetical for determinism
    ))


def verb_key_from_tags(tags):
    """Convert a tag set to a paradigm key.

    Returns None if the tags are not a fully-specified finite verb /
    infinitive / participle cell.
    """
    tags = set(tags)
    voice = next(iter(tags & VOICE_TAGS), None)
    tense = next(iter(tags & TENSE_TAGS), None)
    mood = next(iter(tags & MOOD_TAGS), None)
    if not voice or not tense or not mood:
        return None
    if "mediopassive" in tags:
        voice = "middle"

    if mood == "infinitive":
        return f"{voice}_{tense}_infinitive"

    if mood == "participle":
        case = next(iter(tags & CASE_TAGS), None)
        gender = next(iter(tags & GENDER_TAGS), None)
        number = next(iter(tags & NUMBER_TAGS), None)
        # Wiktionary tables usually give only the nom-sg of each gender;
        # corpus data (GLAUx) carries full case+number on participle forms.
        # Default to nom-sg when only gender is present (Wiktionary case).
        if gender and not case:
            case = "nominative"
        if gender and not number:
            number = "singular"
        if not (case and gender and number):
            return None
        return (
            f"{voice}_{tense}_participle_"
            f"{CASE_SHORT[case]}_{GENDER_SHORT[gender]}_{NUMBER_SHORT[number]}"
        )

    person = next(iter(tags & PERSON_TAGS), None)
    number = next(iter(tags & NUMBER_TAGS), None)
    if not person or not number:
        return None
    # Imperative has no first person. kaikki sometimes mistags Koine
    # alternatives like λυσάτωσαν as 1sg (Wiktionary's 'type-a/type-b'
    # appendix rows leak into the 1sg slot). Drop these to avoid
    # corrupting the paradigm.
    if mood == "imperative" and person == "first-person":
        return None
    return (
        f"{voice}_{tense}_{mood}_"
        f"{PERSON_SHORT[person]}{NUMBER_SHORT[number]}"
    )


def extract_dialect(tags):
    """Return the (single) dialect this form belongs to, or '' for Attic/default."""
    tags = set(tags)
    for d in DIALECT_TAGS:
        if d in tags:
            if d in ("Attic", "Homeric"):
                # Treat Attic as the default slice. Homeric is alias for Epic
                # and we fold it in.
                return "" if d == "Attic" else "epic"
            return d.lower()
    return ""


# ---------------------------------------------------------------------------
# Non-Attic / sandhi form detectors
#
# GLAUx provides AGDT 9-position morph tags but no dialect axis: a Homeric
# unaugmented imperfect / aorist is tagged identically to its Attic
# counterpart, and a sandhi crasis form like κἄβλεψας gets the same
# active-aorist-2sg tag as a regular ἔβλεψας would. The detectors below
# recover dialect / sandhi status from the surface form itself, so the
# canonical Attic paradigm slice doesn't get polluted with Homeric or
# textual-artifact entries that glaux mis-tags. Forms detected as Epic
# get routed to the ``epic`` dialect slice (still kept in the paradigm,
# just out of the Attic default); crasis forms are dropped entirely.
# ---------------------------------------------------------------------------


_BREATHING_MARKS = ("̓", "̔")  # smooth, rough


def _has_breathing(form: str, position: int) -> bool:
    """True when the letter at NFD index ``position`` carries a breathing
    mark. ``position`` indexes into the base-letter sequence after NFD
    decomposition, not the raw NFC string.
    """
    nfd = unicodedata.normalize("NFD", form)
    base_idx = -1
    i = 0
    while i < len(nfd):
        c = nfd[i]
        if not unicodedata.combining(c):
            base_idx += 1
            if base_idx == position:
                # Look ahead at combining marks attached to this base
                j = i + 1
                while j < len(nfd) and unicodedata.combining(nfd[j]):
                    if nfd[j] in _BREATHING_MARKS:
                        return True
                    j += 1
                return False
        i += 1
    return False


def _nfd_base_chars(form: str) -> list:
    """Return the list of base (non-combining) characters in ``form``."""
    nfd = unicodedata.normalize("NFD", form)
    return [c for c in nfd if not unicodedata.combining(c)]


_GREEK_CONSONANTS = set("βγδζθκλμνξπρστφχψ")


_DIPHTHONG_SECOND = set("ιυ")  # second vowel of a Greek diphthong


def is_crasis_form(form: str) -> bool:
    """True when ``form`` is a crasis / sandhi artifact rather than a
    canonical paradigm cell.

    Crasis (the contraction of two adjacent words into one) produces
    surface forms whose first base letter is a CONSONANT and whose
    second OR third base letter carries a smooth or rough breathing
    mark - the coronis. Examples:

    - κἄβλεψας (καί + ἔβλεψας): breathing on position 1 (α)
    - κἀγώ (καί + ἐγώ): breathing on position 1 (α)
    - χἠμεῖς (καί + ἡμεῖς): breathing on position 1 (η)
    - τοὔνομα (τό + ὄνομα): breathing on position 2 (υ of ου-diphthong)
    - τἀνδρός (τοῦ + ἀνδρός): breathing on position 1 (α)

    No native Greek verb form starts with consonant + breathing-marked
    internal vowel: breathing only attaches to word-initial vowels,
    and the presence of a breathing mark on a non-initial base letter
    always indicates crasis or a typesetting artifact. We drop these
    entirely so they don't leak into the canonical paradigm.

    Diphthongs at word start (αὐ-, εὐ-, οὐ-, ηὐ-) put the breathing on
    the first vowel of the diphthong (position 0), so first-base-char
    is a vowel and the consonant gate doesn't fire.
    """
    if not form:
        return False
    base_chars = _nfd_base_chars(form)
    if len(base_chars) < 2:
        return False
    first = base_chars[0].lower()
    if first not in _GREEK_CONSONANTS:
        return False
    # Direct case: breathing on position 1 (κἀγώ, τἀνδρός).
    if _has_breathing(form, 1):
        return True
    # Diphthong case: breathing on position 2 when position-1 + position-2
    # is a Greek diphthong (τοὔνομα = τ + ου + breathing). Without the
    # diphthong gate we'd flag any consonant-vowel-vowel-breathing form,
    # which is too permissive.
    if len(base_chars) >= 3:
        c1 = base_chars[1].lower()
        c2 = base_chars[2].lower()
        if c1 in "αεοηω" and c2 in _DIPHTHONG_SECOND and _has_breathing(form, 2):
            return True
    return False


# ---------------------------------------------------------------------------
# Enclitic-context (extra acute on ultima) detector
#
# AGDT / treebank corpora preserve every accent that appears on the surface
# token in the running text, including the "extra" acute that a following
# enclitic (τις, μοι, σε, με, γε, ...) projects back onto its host word.
# The standard rules (Smyth #183-187): an enclitic immediately following a
# host whose primary accent is on the antepenult, OR a perispomenon penult,
# OR a properispomenon, "echoes" an acute on the ultima of the host. So
# ``ἤκουόν τι`` (= ἤκουον + τι, "I heard something") shows up in the
# treebank as a single host token ``ἤκουόν`` carrying both the original
# acute on η and the enclitic-derived acute on ο.
#
# Greek verbs are recessive: a finite verb form in isolation has exactly
# one accent. Any verb token tagged as a single-word paradigm cell that
# carries a primary accent + an extra acute on the ultima vowel is
# guaranteed to be one of these enclitic-context surface forms, NOT a
# canonical citation form. We detect them and drop them at the ingestion
# stage so the canonical paradigm slice gets the clean form (from
# Wiktionary / kaikki, or from synthesised principal-parts templating).
#
# This pattern matters because glaux is the only source for some past-
# indicative cells (kaikki Wiktionary tables don't expose a 1sg imperfect
# tag for many verbs, so synthesis is the fallback). Without this filter
# the bad enclitic-context form wins ``pick_best_form`` as the only
# attested variant, and the synth pass declines to overwrite it.
# ---------------------------------------------------------------------------


_GREEK_VOWELS = set("αεηιουωΑΕΗΙΟΥΩ")
_COMBINING_ACUTE = "́"
_COMBINING_CIRCUMFLEX = "͂"  # GREEK PERISPOMENI
_COMBINING_GRAVE = "̀"


def _vowel_marks_per_base(form: str):
    """Return list of (base_char, has_acute, has_circumflex, has_grave) for
    each base (non-combining) NFD character in ``form``. Marks are
    accumulated onto the immediately preceding base character.
    """
    nfd = unicodedata.normalize("NFD", form)
    out = []
    cur = None
    cur_acute = cur_circ = cur_grave = False
    for c in nfd:
        if not unicodedata.combining(c):
            if cur is not None:
                out.append((cur, cur_acute, cur_circ, cur_grave))
            cur = c
            cur_acute = cur_circ = cur_grave = False
        else:
            if c == _COMBINING_ACUTE:
                cur_acute = True
            elif c == _COMBINING_CIRCUMFLEX:
                cur_circ = True
            elif c == _COMBINING_GRAVE:
                cur_grave = True
    if cur is not None:
        out.append((cur, cur_acute, cur_circ, cur_grave))
    return out


def _ultima_vowel_index(marks):
    """Return the index of the rightmost vowel in ``marks``, or -1 if no
    vowel is present."""
    for i in range(len(marks) - 1, -1, -1):
        if marks[i][0] in _GREEK_VOWELS:
            return i
    return -1


def is_enclitic_context_form(form: str) -> bool:
    """True when ``form`` carries a primary accent plus an extra acute /
    grave on the ultima, matching the surface pattern of a host word
    followed by an enclitic.

    A single-token Greek verb form has exactly one lexical accent. An
    extra acute appearing on the ultima (the rightmost vowel of the
    word) when there is already a primary accent earlier is the
    canonical signature of enclitic accent-projection: e.g. ``ἤκουόν``
    is the surface of ``ἤκουον + τι`` ("I heard something"), with τι's
    accent transferred onto the host. The treebank annotates this as a
    single token in the cell where a clean ``ἤκουον`` belongs.

    The same projection also surfaces with a grave on the ultima
    (``ἐποίησὲ``) when the editorial convention wrote a "no-following-
    enclitic" oxytone — both forms originate as a host plus a clitic;
    only the editor's accent normalisation differs. We treat both shapes
    identically: an acute *or* grave on the ultima alongside one earlier
    lexical accent flags the cell as enclitic-context noise.

    Returns True only on the conservative two-mark shape: exactly one
    accent earlier in the word AND an acute or grave (not circumflex)
    on the ultima vowel. Forms with three or more accents, or with
    circumflex on the ultima, or with the only accent on the ultima,
    are NOT flagged - they fall through to the regular pipeline.

    Why this is safe: Greek verbs are recessive. No clean canonical
    verb form in any tense / mood / voice carries two accent marks.
    The 17K-form glaux corpus survey shows >99% of double-accent verb
    tokens match this exact shape; the remaining <1% are typo /
    untokenised-multi-word artifacts that we'd want to drop anyway.
    """
    if not form:
        return False
    marks = _vowel_marks_per_base(form)
    if len(marks) < 2:
        return False
    n_acute = sum(1 for m in marks if m[1])
    n_circ = sum(1 for m in marks if m[2])
    n_grave = sum(1 for m in marks if m[3])
    if n_acute + n_circ + n_grave != 2:
        return False
    ult = _ultima_vowel_index(marks)
    if ult < 0:
        return False
    # Ultima vowel must carry an acute or grave (the enclitic-derived
    # mark, possibly normalised to grave by the editor) and NOT a
    # circumflex (a perispomenon ultima is the verb's own accent, not
    # enclitic-derived).
    if marks[ult][2]:
        return False
    if not (marks[ult][1] or marks[ult][3]):
        return False
    # Exactly one OTHER accent earlier in the word: that's the verb's
    # original lexical accent. If the only accent in the word is on the
    # ultima we already returned via the n=2 check (n_acute+n_circ+
    # n_grave==2 would never be hit with a single mark), but guard
    # explicitly.
    earlier_marks = sum(1 for i, m in enumerate(marks)
                        if i < ult and (m[1] or m[2] or m[3]))
    return earlier_marks == 1


# ---------------------------------------------------------------------------
# Iota-dropped contract-stem detector
#
# Hellenistic / Ionic spelling routinely drops the iota in the contract
# stem of an ``-ιέω`` verb: ποιέω -> ποέω, ἐποίησα -> ἐπόησα, ἐποίει ->
# ἐπόει. Wiktionary (and therefore kaikki) lists ``ποέω`` as an
# alternative to ``ποιέω`` and emits all of its inflectional forms under
# the ``ποιέω`` lemma. Glaux likewise lemmatises iota-less Hellenistic
# tokens to the canonical Attic ``ποιέω``. Once both are pooled, the
# ``pick_best_form`` ``-len(f)`` tiebreaker picks the iota-less variant
# over the canonical Attic spelling (``ἐπόησα`` < ``ἐποίησα``), and
# every past-indicative cell ends up reporting the wrong canonical form.
#
# The fix is to drop iota-less forms whose lemma is a -ιέω / -ιέομαι
# contract verb, BEFORE ``pick_best_form`` runs. The synth fallback
# (``synth_verb_moods.synthesize_past_indicatives``) and the canonical
# Wiktionary ``ποιέω`` table will fill the slot with the correct
# iota-bearing surface form.
# ---------------------------------------------------------------------------


import re as _re_iota  # local alias to avoid colliding with module-level imports

_GREEK_CONS_FOR_IOTA = "βγδζθκλμνξπρστφχψ"
_GREEK_VOWELS_LOWER = "αεηιοωυ"


def is_iota_dropped_contract_form(form: str, lemma: str) -> bool:
    """True when ``form`` is the iota-less Hellenistic / Ionic spelling
    of an ``-ιέω`` (or ``-ιέομαι``) contract verb whose canonical Attic
    spelling carries the contract-stem iota.

    The lemma must end in ``-ιέω`` or ``-ιέομαι`` after diacritics are
    stripped: ποιέω, ἐμποιέω, μεταποιέω, etc. The canonical contract
    junction is ``CV + ι + (vowel)`` — e.g. ``ποι + η`` in the aor stem
    ``ποιησ-``. The iota-less spelling collapses ``ποι`` to ``πο``,
    yielding ``πο + η`` (``ποη-``), ``πο + ε`` (``ποε-``), ``πο + ει``
    (``ποει-``), or ``πο + ου`` (``ποου-`` / ``ποουν``).

    The detector tests whether the form's diacritic-stripped surface
    contains the lemma's pre-iota CV pair followed directly by a
    contract-fused vowel WITHOUT the canonical iota in between. Forms
    that contain the canonical CV+ι sequence (i.e. the iota survives)
    are NOT flagged. This keeps proper Attic forms safe and only drops
    the orthographic variants.

    Returns False for non-contract lemmas, athematic lemmas, lemmas
    where the pre-iota slot isn't a CV pair, and forms whose
    diacritic-stripped surface doesn't contain the canonical ``-CV``
    region at all (suppletive variants, prefix-only fragments, etc.).
    """
    if not form or not lemma:
        return False
    lb = strip_accents(lemma).lower()
    fb = strip_accents(form).lower().rstrip("'’ʼ᾽ʹ")
    if not fb:
        return False
    if lb.endswith("ιεω"):
        i_idx = len(lb) - 3
    elif lb.endswith("ιεομαι"):
        i_idx = len(lb) - 6
    else:
        return False
    # Need at least 2 chars before the iota: CV.
    if i_idx < 2:
        return False
    cv = lb[i_idx - 2:i_idx]
    if cv[0] not in _GREEK_CONS_FOR_IOTA:
        return False
    if cv[1] not in _GREEK_VOWELS_LOWER:
        return False
    canonical_pat = _re_iota.escape(cv) + r"ι[" + _GREEK_VOWELS_LOWER + r"]"
    dropped_pat = _re_iota.escape(cv) + r"(η|ει|ου|ω|ε|α)"
    if _re_iota.search(canonical_pat, fb):
        return False
    return bool(_re_iota.search(dropped_pat, fb))


# Iterative -σκ- infix endings. The Homeric iterative imperfect inserts
# -σκ- between the present stem and a thematic personal ending. Active
# endings: -ον / -ες / -ε(ν) / -ομεν / -ετε / -ον. Middle/passive
# endings: -όμην / -εο/-ευ/-ου / -ετο / -όμεθα / -εσθε / -οντο. The
# infix attaches with one of the thematic vowels (ε, α, ο), giving the
# accent-stripped suffixes below. Forms ending here AND tagged as
# imperfect indicative are reclassified as Epic.
_ITERATIVE_SUFFIXES = (
    # Active
    "εσκον", "εσκες", "εσκε", "εσκεν", "εσκομεν", "εσκετε",
    "εσκετον", "εσκετην",
    "ασκον", "ασκες", "ασκε", "ασκεν", "ασκομεν", "ασκετε",
    "ασκετον", "ασκετην",
    "οσκον", "οσκες", "οσκε", "οσκεν", "οσκομεν", "οσκετε",
    "οσκετον", "οσκετην",
    # Middle / passive
    "εσκομην", "εσκετο", "εσκοντο", "εσκομεθα", "εσκεσθε",
    "εσκεσθον", "εσκεσθην", "εσκεο", "εσκευ", "εσκου",
    "ασκομην", "ασκετο", "ασκοντο", "ασκομεθα", "ασκεσθε",
    "ασκεο", "ασκευ",
    "οσκομην", "οσκετο", "οσκοντο", "οσκομεθα", "οσκεσθε",
)


def is_homeric_iterative_imperfect(
    form: str, lemma: str, key: str | None
) -> bool:
    """True when ``form`` is a Homeric iterative imperfect.

    The iterative infix ``-σκ-`` between the present stem and a
    thematic personal ending is a Homeric / Ionic dialect feature. It
    only attests in imperfect indicative cells. Verbs whose lemma
    natively contains the inceptive ``-σκ-`` (διδάσκω, γιγνώσκω,
    εὑρίσκω, μιμνήσκω, βιβρώσκω, βόσκω, ...) are NOT iterative when
    they show ``-σκ-`` in their imperfect; we filter them out here by
    requiring the lemma's stripped form to NOT end in ``σκω`` /
    ``σκομαι``.
    """
    if not form or not lemma or not key:
        return False
    if "_imperfect_indicative" not in key:
        return False
    fb = strip_accents(form).lower().rstrip("’'ʼ᾽ʹ")
    lb = strip_accents(lemma).lower()
    # Lemma natively has an inceptive σκ in its present stem -> not iterative.
    if lb.endswith("σκω") or lb.endswith("σκομαι") or lb.endswith("σκον"):
        return False
    return any(fb.endswith(suffix) for suffix in _ITERATIVE_SUFFIXES)


def is_homeric_unaugmented_past_indicative(
    form: str, lemma: str, key: str | None
) -> bool:
    """True when ``form`` is an unaugmented past-indicative form.

    Augment is mandatory in Attic indicative aorist / imperfect /
    pluperfect; an unaugmented surface form in those slots is a
    Homeric / Epic variant (Wiktionary explicitly tags the same
    citations as Epic, but glaux's AGDT tagger has no dialect axis).
    Routing such forms to the ``epic`` dialect slice keeps the data
    available without polluting the canonical Attic paradigm.

    The augment check uses :func:`has_augment` which wraps the
    syllabic / temporal augment detectors in :mod:`dilemma.morph_diff`.
    Only fires when the lemma starts with a consonant (syllabic
    augment) or a short vowel (temporal augment). For lemmas where
    augment is morphologically blocked (already long-vowel initial,
    diphthong-initial in some classes), this function returns False.
    """
    if not form or not lemma or not key:
        return False
    if not is_past_indicative_key(key):
        return False
    if has_augment(form, lemma):
        return False
    # Augment must be morphologically observable for this lemma; if
    # the lemma already starts with a long vowel (η / ω) the temporal
    # augment is invisible and we can't distinguish Homeric from Attic.
    lb = _strip_diacritics(lemma)
    if not lb:
        return False
    first = lb[0]
    # Syllabic augment: consonant-initial lemmas always show ἐ-.
    # Temporal augment: short-vowel-initial lemmas (α/ε/ο/ι/υ) lengthen.
    # Long-vowel / diphthong starts are ambiguous and skipped here.
    if first in ("η", "ω"):
        return False
    # Prefixed compound verbs (ἐκ-, ἐν-, ἐπι-, ἐξ-, etc.) hide their
    # augment between prefix and root: ἐκ-μολεῖν -> ἐξ-έ-μολεν, with
    # the augment inside the form rather than at position 0. The
    # augment detector :func:`_detect_syllabic_augment` only spots
    # initial augments and reports such forms as unaugmented, so we
    # skip them here. We use a conservative shape test: lemma starts
    # with ε-, AND form starts with the same character. Real Homeric
    # unaugmented variants of vowel-initial verbs (ἤκουον vs ἀκούω)
    # start with a different initial vowel from the lemma so they
    # still pass.
    if first == "ε":
        fb = _strip_diacritics(form)
        if fb and fb[0] == "ε":
            return False
    return True


# Middle-voice personal endings. The Homeric root-aorist used these on
# the bare verb root in passive function (ἐλύμην, ἔλυντο, λύτο), even
# though Attic distinguishes middle (uses these endings) from passive
# (uses -θη- + active endings, or -η- + active endings for aor-2). When
# glaux tags such forms as ``aorist passive indicative``, the middle-
# voice ending shape is the giveaway that we're looking at a Homeric
# root-aorist mis-classified as 1st-aorist-passive.
_ROOT_AORIST_MIDDLE_ENDINGS = (
    "μην",        # 1sg
    "σο",         # 2sg
    "το",         # 3sg (also 3sg dual)
    "μεθα",       # 1pl
    "σθε",        # 2pl
    "ντο",        # 3pl
    "σθον",       # 2du
    "σθην",       # 3du
)


def is_homeric_root_aorist_passive(
    form: str, lemma: str, key: str | None
) -> bool:
    """True when ``form`` is a Homeric root-aorist passive.

    The Homeric / Epic root-aorist used middle-voice personal endings
    (``-μην / -σο / -το / -μεθα / -σθε / -ντο``) on the bare verb root
    in passive function: ``ἐλύμην``, ``ἔλυντο``, ``λύμην``, ``λύτο``,
    ``λύντο``, ``λῦτο``. Glaux tags these as ``aorist passive indicative
    <person>-<number>``, putting them in the same cell as the Attic
    1st-aorist-passive (``ἐλύθην`` / ``ἐλύθη``) and confusing readers
    expecting a Classical paradigm.

    Distinguishing the Homeric root-aorist passive from Attic forms is
    a question of ending shape, not of the ``-θ-`` marker: the Attic
    2nd / strong aorist passive (``ἐγράφην`` for γράφω) also lacks θ
    but uses active endings (``-ν``, ``-ς``, ``-η``, ...). What
    uniquely flags the Homeric root-aorist passive is the use of the
    middle-voice personal endings on a slot tagged passive.
    """
    if not form or not lemma or not key:
        return False
    if not key.startswith("passive_aorist_indicative_"):
        return False
    fb = strip_accents(form).lower().rstrip("’'ʼ᾽ʹ")
    return any(fb.endswith(ending) for ending in _ROOT_AORIST_MIDDLE_ENDINGS)


def load_pairs(path: Path):
    if not path.exists():
        print(f"  skipping {path.name} (not present)", flush=True)
        return []
    print(f"  loading {path.name} ...", flush=True)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_lsj_head_texts() -> dict:
    """Load the leading paragraph of every LSJ entry (the gloss without
    `level`/`number` is the entry head, which carries the principal-
    parts header before the English definition starts).

    Returns a dict ``{headword: head_text}``. Empty dict if the LSJ9
    glosses file is unavailable.
    """
    heads: dict[str, str] = {}
    if not LSJ9_GLOSSES.exists():
        print(f"  lsj9 glosses not found at {LSJ9_GLOSSES}; "
              f"principal-parts synthesis disabled")
        return heads
    print(f"  loading lsj9 head texts from {LSJ9_GLOSSES.name} ...",
          flush=True)
    with open(LSJ9_GLOSSES, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            hw = e.get("headword")
            if not hw:
                continue
            if "level" in e or "number" in e:
                continue
            if hw not in heads:
                heads[hw] = e.get("text", "")
    print(f"  lsj9 head texts: {len(heads):,}")
    return heads


def synthesize_missing_moods(results: dict) -> tuple[int, int]:
    """Fill in missing finite-mood cells via principal-parts templating.

    For each verb in ``results``, parse its LSJ head text into
    principal parts and run them through
    ``synth_verb_moods.synthesize_active_moods`` and
    ``synth_verb_moods.synthesize_mp_moods`` to produce templated
    subjunctive / optative / imperative / aorist-infinitive forms in
    active, middle, and passive voices. Only writes into slots that are
    currently empty; real corpus / Wiktionary cells are never overwritten.

    Returns ``(verbs_touched, cells_added)``.
    """
    try:
        from synth_verb_moods import (
            synthesize_active_moods,
            synthesize_mp_moods,
            synthesize_aor2_moods,
            synthesize_contract_moods,
            synthesize_past_indicatives,
        )
        from lsj_principal_parts import parse_principal_parts
    except ImportError as e:
        print(f"  synthesis skipped (import failure: {e})")
        return 0, 0

    head_texts = load_lsj_head_texts()
    verbs_touched = 0
    cells_added = 0
    mp_cells_added = 0
    aor2_cells_added = 0
    contract_cells_added = 0
    past_cells_added = 0
    cells_skipped_overlap = 0
    for lemma, paradigm in results.items():
        head_text = head_texts.get(lemma, "")
        try:
            parts = parse_principal_parts(head_text, lemma) if head_text else {}
        except Exception:
            parts = {}
        try:
            templated = synthesize_active_moods(lemma, parts)
        except Exception:
            templated = {}
        try:
            templated_mp = synthesize_mp_moods(lemma, parts)
        except Exception:
            templated_mp = {}
        try:
            templated_aor2 = synthesize_aor2_moods(lemma, parts)
        except Exception:
            templated_aor2 = {}
        try:
            templated_contract = synthesize_contract_moods(lemma, parts)
        except Exception:
            templated_contract = {}
        try:
            templated_past = synthesize_past_indicatives(lemma, parts)
        except Exception:
            templated_past = {}
        # Track per-voice to report stats; merge for the actual write.
        if not (templated or templated_mp or templated_aor2
                or templated_contract or templated_past):
            continue
        forms = paradigm.setdefault("forms", {})
        added = 0
        added_mp = 0
        added_aor2 = 0
        added_contract = 0
        added_past = 0
        for key, val in templated.items():
            if key in forms:
                cells_skipped_overlap += 1
                continue
            forms[key] = val
            added += 1
        for key, val in templated_mp.items():
            if key in forms:
                cells_skipped_overlap += 1
                continue
            forms[key] = val
            added_mp += 1
        for key, val in templated_aor2.items():
            if key in forms:
                cells_skipped_overlap += 1
                continue
            forms[key] = val
            added_aor2 += 1
        for key, val in templated_contract.items():
            if key in forms:
                cells_skipped_overlap += 1
                continue
            forms[key] = val
            added_contract += 1
        for key, val in templated_past.items():
            if key in forms:
                cells_skipped_overlap += 1
                continue
            forms[key] = val
            added_past += 1
        if (added or added_mp or added_aor2 or added_contract
                or added_past):
            verbs_touched += 1
            cells_added += added
            mp_cells_added += added_mp
            aor2_cells_added += added_aor2
            contract_cells_added += added_contract
            past_cells_added += added_past
            paradigm["form_count"] = len(forms)
    print(f"  synthesised active cells: {cells_added:,} across "
          f"{verbs_touched:,} verbs")
    print(f"  synthesised mp/passive cells: {mp_cells_added:,}")
    print(f"  synthesised aor-2 cells: {aor2_cells_added:,}")
    print(f"  synthesised contract cells: {contract_cells_added:,}")
    print(f"  synthesised past-indicative 1sg cells: {past_cells_added:,}")
    print(f"  cells skipped (already present): {cells_skipped_overlap:,}")
    return verbs_touched, (
        cells_added + mp_cells_added + aor2_cells_added
        + contract_cells_added + past_cells_added
    )


def synthesize_missing_participles(results: dict) -> tuple[int, int]:
    """Fill in missing participle cells via principal-parts templating.

    For each verb in ``results``, parse its LSJ head text into principal
    parts and run them through
    ``synth_verb_participles.synthesize_participles`` to produce the
    full case×gender×number declension for present-active /
    present-mp / future-active / future-middle / future-passive /
    aorist-active / aorist-middle / aorist-passive / perfect-active /
    perfect-mp participles. Only writes into slots that are currently
    empty; real corpus / Wiktionary cells are never overwritten.

    Returns ``(verbs_touched, cells_added)``.
    """
    try:
        from synth_verb_participles import (
            synthesize_participles,
            synthesize_aor2_participles,
            synthesize_contract_participles,
        )
        from lsj_principal_parts import parse_principal_parts
    except ImportError as e:
        print(f"  participle synthesis skipped (import failure: {e})")
        return 0, 0

    head_texts = load_lsj_head_texts()
    verbs_touched = 0
    cells_added = 0
    aor2_cells_added = 0
    contract_cells_added = 0
    cells_skipped_overlap = 0
    for lemma, paradigm in results.items():
        head_text = head_texts.get(lemma, "")
        try:
            parts = parse_principal_parts(head_text, lemma) if head_text else {}
        except Exception:
            parts = {}
        try:
            templated = synthesize_participles(lemma, parts)
        except Exception:
            templated = {}
        try:
            templated_aor2 = synthesize_aor2_participles(lemma, parts)
        except Exception:
            templated_aor2 = {}
        try:
            templated_contract = synthesize_contract_participles(lemma, parts)
        except Exception:
            templated_contract = {}
        if not (templated or templated_aor2 or templated_contract):
            continue
        forms = paradigm.setdefault("forms", {})
        added = 0
        added_aor2 = 0
        added_contract = 0
        for key, val in templated.items():
            if key in forms:
                cells_skipped_overlap += 1
                continue
            forms[key] = val
            added += 1
        for key, val in templated_aor2.items():
            if key in forms:
                cells_skipped_overlap += 1
                continue
            forms[key] = val
            added_aor2 += 1
        for key, val in templated_contract.items():
            if key in forms:
                cells_skipped_overlap += 1
                continue
            forms[key] = val
            added_contract += 1
        if added or added_aor2 or added_contract:
            verbs_touched += 1
            cells_added += added
            aor2_cells_added += added_aor2
            contract_cells_added += added_contract
            paradigm["form_count"] = len(forms)
    print(f"  synthesised participle cells: {cells_added:,} across "
          f"{verbs_touched:,} verbs")
    print(f"  synthesised aor-2 participle cells: {aor2_cells_added:,}")
    print(f"  synthesised contract participle cells: {contract_cells_added:,}")
    print(f"  participle cells skipped (already present): "
          f"{cells_skipped_overlap:,}")
    return verbs_touched, cells_added + aor2_cells_added + contract_cells_added


def build_paradigms(only_lemmas=None):
    """Aggregate verb pairs from all sources into per-lemma paradigms."""
    print("Building Ancient Greek verb paradigms ...")
    sources = []
    for path, name in [
        (AG_PAIRS, "ag_pairs.json"),
        (GLAUX_PAIRS, "glaux_pairs.json"),
    ]:
        pairs = load_pairs(path)
        if isinstance(pairs, list):
            verb_pairs = [p for p in pairs if isinstance(p, dict)
                          and p.get("pos") == "verb"]
        else:
            verb_pairs = []
        print(f"  {name}: {len(verb_pairs)} verb pairs")
        sources.append((name, verb_pairs))
    # ag_lsj_verb_pairs.json is already merged into ag_pairs.json by
    # build_data.py. verb_extra_pairs.json is a flat form->lemma dict
    # (no tags) so it can't contribute paradigm entries.

    # Load LSJ headwords for canonical-spelling lookup. Used to
    # normalize variants where kaikki / corpus data emit a slightly
    # off form (wrong breathing, missing iota subscript). The LSJ
    # headword is treated as authoritative when both the input lemma
    # and a single accent-stripped LSJ candidate exist.
    lsj_headwords: set[str] = set()
    lsj_by_stripped: dict[str, list[str]] = defaultdict(list)
    if LSJ_HEADWORDS_PATH.exists():
        with open(LSJ_HEADWORDS_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        for hw in raw:
            if not isinstance(hw, str) or not hw:
                continue
            # Strip the LSJ "Α α" header rows and multi-word phrases.
            if " " in hw:
                continue
            if not is_pure_greek(hw):
                continue
            lsj_headwords.add(hw)
            stripped = strip_accents(hw.lower())
            if stripped:
                lsj_by_stripped[stripped].append(hw)
    print(f"  LSJ headwords loaded: {len(lsj_headwords):,}")

    def canonicalize_lemma(lemma: str) -> str | None:
        """Apply lemma normalization. Returns None if the lemma should
        be dropped entirely (e.g. mojibake-like ΒΕἔστημι)."""
        if not lemma or not is_pure_greek(lemma):
            return None
        if has_internal_capital(lemma):
            return None  # mojibake / corpus glitch, drop
        # LSJ canonicalization first: pick up the LSJ-canonical spelling
        # for breathing / iota-subscript variants (ἀλίσκω -> ἁλίσκω,
        # ἀθωόω -> ἀθῳόω) when kaikki used a different convention than
        # LSJ for the same lemma.
        if lsj_headwords and lemma not in lsj_headwords:
            stripped = strip_accents(lemma.lower())
            candidates = lsj_by_stripped.get(stripped, [])
            if len(candidates) == 1:
                lemma = candidates[0]
        # Then force lowercase: verbs aren't proper nouns, and LSJ /
        # kaikki occasionally preserve a capital from a denominal
        # (Αἰγυπτιάζω, Βακχεύω, Δημοσθενίζω - "to act like X" verbs
        # derived from proper-noun X). Citation form should be
        # lowercase regardless.
        if lemma and lemma[0].isupper():
            lemma = lowercase_initial(lemma)
        return lemma

    # Group by lemma, then by dialect, then by paradigm key
    by_lemma_dialect_key = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    dropped_internal_capital = 0
    dropped_crasis = 0
    dropped_enclitic = 0
    dropped_iota_dropped = 0
    rerouted_iterative = 0
    rerouted_unaugmented = 0
    rerouted_root_aorist_passive = 0
    canonicalized = 0
    lowercased = 0
    for src_name, verb_pairs in sources:
        for p in verb_pairs:
            lemma_raw = p.get("lemma")
            form = (p.get("form") or "").strip()
            tags = p.get("tags", [])
            if not lemma_raw or not form:
                continue
            if not is_pure_greek(lemma_raw) or not is_pure_greek(form):
                continue
            if has_internal_capital(lemma_raw):
                dropped_internal_capital += 1
                continue
            # Glaux preserves sentence-initial capitalisation on tokens
            # (``Ἐποίησα`` mid-sentence, ``Ἤκουσα`` at quote start). The
            # canonical paradigm cell is always lowercase; an uppercase
            # leading char would otherwise cause ``pick_best_form``'s
            # alphabetical tiebreaker to prefer ``Ἐποίησα`` over
            # ``ἐποίησα`` because capital ``Ε`` (U+0395) sorts before
            # lowercase ``ε`` (U+03B5). Lowercase the leading char on
            # the form too.
            if form and form[0].isupper():
                form = lowercase_initial(form)
            lemma = canonicalize_lemma(lemma_raw)
            if lemma is None:
                continue
            if lemma != lemma_raw:
                if lemma_raw[0].isupper() and lemma == lowercase_initial(lemma_raw):
                    lowercased += 1
                else:
                    canonicalized += 1
            if only_lemmas is not None and lemma not in only_lemmas:
                continue
            if "form-of" in tags or "alt-of" in tags or "alternative" in tags:
                continue
            key = verb_key_from_tags(tags)
            if not key:
                continue
            # Crasis / sandhi forms (κἄβλεψας from καί + ἔβλεψας) are
            # textual artifacts, not canonical paradigm cells. Drop
            # entirely so they leak into neither the Attic slice nor a
            # dialect slice.
            if is_crasis_form(form):
                dropped_crasis += 1
                continue
            # Enclitic-context forms (ἤκουόν from ἤκουον + τι) carry an
            # extra acute on the ultima projected from a following
            # enclitic. They are surface artifacts of the running text,
            # not canonical paradigm cells. Drop entirely so the
            # canonical slot stays open for the clean Wiktionary form
            # or, failing that, the synthesised form. This is
            # particularly important for past-indicative cells like
            # ἀκούω 1sg imperfect where glaux is the only source and
            # the synth pass only writes into empty cells.
            if is_enclitic_context_form(form):
                dropped_enclitic += 1
                continue
            # Iota-dropped Hellenistic / Ionic spellings (ποιέω -> ἐπόησα
            # for ἐποίησα) come in under the canonical Attic lemma
            # because Wiktionary lists ``ποέω`` as an alt for ``ποιέω``
            # and pools both inflectional tables. The iota-less spelling
            # would otherwise win ``pick_best_form`` on shorter length
            # over the canonical Attic form. Drop these so the synth
            # fallback (or a canonical Wiktionary cell) supplies the
            # correct iota-bearing surface form.
            if is_iota_dropped_contract_form(form, lemma):
                dropped_iota_dropped += 1
                continue
            dialect = extract_dialect(tags)
            # Glaux has no dialect axis, so Homeric / Epic forms come
            # in tagged identically to their Attic counterparts. The
            # detectors below recover dialect status from the surface
            # form itself and route detected non-Attic variants to the
            # ``epic`` dialect slice rather than the canonical Attic
            # paradigm. Attic slots stay clean; the Homeric / Epic
            # forms are still preserved for downstream consumers that
            # want them.
            if not dialect:
                if is_homeric_iterative_imperfect(form, lemma, key):
                    dialect = "epic"
                    rerouted_iterative += 1
                elif is_homeric_root_aorist_passive(form, lemma, key):
                    dialect = "epic"
                    rerouted_root_aorist_passive += 1
                elif is_homeric_unaugmented_past_indicative(form, lemma, key):
                    dialect = "epic"
                    rerouted_unaugmented += 1
            by_lemma_dialect_key[lemma][dialect][key].append(form)
    if dropped_internal_capital:
        print(f"  dropped (internal capitals / mojibake): "
              f"{dropped_internal_capital:,}")
    if dropped_crasis:
        print(f"  dropped (crasis / sandhi): {dropped_crasis:,}")
    if dropped_enclitic:
        print(f"  dropped (enclitic-context double accent): "
              f"{dropped_enclitic:,}")
    if dropped_iota_dropped:
        print(f"  dropped (iota-dropped contract spelling): "
              f"{dropped_iota_dropped:,}")
    if rerouted_iterative:
        print(f"  rerouted to epic (Homeric iterative imperfect): "
              f"{rerouted_iterative:,}")
    if rerouted_root_aorist_passive:
        print(f"  rerouted to epic (Homeric root-aorist passive): "
              f"{rerouted_root_aorist_passive:,}")
    if rerouted_unaugmented:
        print(f"  rerouted to epic (unaugmented past indicative): "
              f"{rerouted_unaugmented:,}")
    if lowercased:
        print(f"  lowercased sentence-initial verb lemmas: {lowercased:,}")
    if canonicalized:
        print(f"  LSJ-canonicalized lemmas: {canonicalized:,}")

    print(f"  candidate lemmas: {len(by_lemma_dialect_key)}")

    results = {}
    for lemma in sorted(by_lemma_dialect_key.keys()):
        by_dialect = by_lemma_dialect_key[lemma]
        attic_forms_raw = by_dialect.get("", {})
        # Pick the best form for each key in the Attic slice
        attic_forms = {}
        for key, variants in attic_forms_raw.items():
            best = pick_best_form(variants, key=key, lemma=lemma)
            if best:
                attic_forms[key] = grave_to_acute(best)

        # Default convention: the lemma is the active present indicative 1sg
        # if it ends in -ω, or middle present indicative 1sg if it ends in
        # -μαι (deponent). For -μι verbs we leave the slot blank since the
        # paradigm itself should attest the 1sg form.
        if attic_forms or any(by_dialect.values()):
            if lemma.endswith("ω"):  # ω
                attic_forms.setdefault(
                    "active_present_indicative_1sg", lemma)
            elif lemma.endswith("μαι"):  # μαι
                attic_forms.setdefault(
                    "middle_present_indicative_1sg", lemma)

        if not attic_forms:
            continue

        paradigm = {
            "forms": attic_forms,
            "form_count": len(attic_forms),
            "source": "dilemma",
        }

        # Add per-dialect paradigm slices for non-Attic dialects.
        # We do NOT pass `lemma` for the augment-preference rule here:
        # Epic / Homeric / Doric variants regularly omit the augment
        # (e.g. λῦσε in Homer is the canonical Epic 3sg), so forcing
        # augment-preference would override the dialect's own usage.
        for dialect, kv in by_dialect.items():
            if not dialect:
                continue
            picked = {}
            for key, variants in kv.items():
                best = pick_best_form(variants, key=key)
                if best:
                    picked[key] = grave_to_acute(best)
            if picked:
                paradigm.setdefault("dialects", {})[dialect] = picked

        results[lemma] = paradigm

    print(f"  built paradigms for {len(results)} lemmas")
    if results:
        counts = sorted(v["form_count"] for v in results.values())
        n = len(counts)
        print(f"  forms per lemma: min={counts[0]} median={counts[n//2]} "
              f"max={counts[-1]} avg={sum(counts)/n:.1f}")

    # Procedural synthesis pass: fill missing finite-mood cells
    # (subjunctive / optative / imperative / aorist infinitive) for
    # thematic -ω verbs from LSJ-extracted principal parts. Only writes
    # into empty slots, never overwrites corpus-derived forms.
    print("  synthesising missing moods from principal parts ...")
    synthesize_missing_moods(results)
    if results:
        counts = sorted(v["form_count"] for v in results.values())
        n = len(counts)
        print(f"  forms per lemma (post-mood-synth): "
              f"min={counts[0]} median={counts[n//2]} "
              f"max={counts[-1]} avg={sum(counts)/n:.1f}")

    # Second synthesis pass: full case×gender×number participle
    # declension. Like the mood pass, only fills empty cells.
    print("  synthesising missing participles from principal parts ...")
    synthesize_missing_participles(results)
    if results:
        counts = sorted(v["form_count"] for v in results.values())
        n = len(counts)
        print(f"  forms per lemma (post-participle-synth): "
              f"min={counts[0]} median={counts[n//2]} "
              f"max={counts[-1]} avg={sum(counts)/n:.1f}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sanity", action="store_true",
                    help="6-lemma smoke test (writes to *.sanity.json)")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated lemma list (debug)")
    ap.add_argument("--out", type=str, default=None,
                    help="output path override")
    args = ap.parse_args()

    only = None
    if args.sanity:
        only = ["γράφω",       # γράφω
                "τίθημι", # τίθημι
                "αἰρέω",       # αἰρέω - normal contract
                "δίδωμι", # δίδωμι
                "λύω",                   # λύω
                "εἰμί",             # εἰμί
                "ἀκούω",       # ἀκούω
                "φιλέω",       # φιλέω - epsilon contract
                "τιμάω",       # τιμάω - alpha contract
                "δηλόω",       # δηλόω - omicron contract
                ]
    elif args.only:
        only = [s.strip() for s in args.only.split(",") if s.strip()]
    only_set = set(only) if only else None

    paradigms = build_paradigms(only_lemmas=only_set)

    if args.sanity:
        out = OUT_PATH.with_suffix(".sanity.json")
    elif args.out:
        out = Path(args.out)
    else:
        out = OUT_PATH

    out.write_text(json.dumps(paradigms, ensure_ascii=False))
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"  -> {out} ({size_mb:.1f} MB)")

    if only:
        # Print sanity output for inspection
        for lemma in only:
            if lemma in paradigms:
                p = paradigms[lemma]
                print(f"\n{lemma}: {p['form_count']} forms")
                for k in sorted(p["forms"].keys())[:8]:
                    print(f"  {k} = {p['forms'][k]}")
                if len(p["forms"]) > 8:
                    print(f"  ... ({len(p['forms']) - 8} more)")
            else:
                print(f"\n{lemma}: NOT FOUND")


if __name__ == "__main__":
    main()
