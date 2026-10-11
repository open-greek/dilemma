#!/usr/bin/env python3
"""Procedural synthesis of finite verb mood forms from principal parts.

Given an Ancient Greek verb lemma and the principal-parts dict produced
by ``build/lsj_principal_parts.parse_principal_parts``, this module
generates the missing finite-mood paradigm cells (subjunctive, optative,
imperative, aorist infinitive) by stem-templating with regular endings.

Strategy (v3): The contract / aor-2 extensions use **option (a)** -- per-
class ending tables that already encode the contracted vowel quality and
accent placement. We considered option (b), a generic ending-applier
followed by a contract-rule post-processor, but jtauber's actual cells
use idiosyncratic accent/length combinations (τιμᾷς, τιμᾱ́σθω, ποιοίημεν
vs. ποιοῖμεν, etc.) that are easier to ship correctly with explicit
tables than to derive procedurally. The α-/ε-/ο-class tables are kept in
parallel; conflict-free expansion is checked by comparing the dilemma
output against jtauber's verbatim cells in the test suite.

Scope (v3):
  - Active, middle, and passive voices.
  - Plain thematic -ω verbs (v1+v2).
  - Aor-2 (strong-aorist) thematic verbs whose aor 1sg ends in -ον
    (ἔλιπον, ἔλαβον, ἔπεσον, εὗρον, ...). Active-only synthesis; middle
    aor-2 piggy-backs on the present-style middle endings.
  - Contract verbs (-άω / -έω / -όω) for the present system only.
    Future / aorist of contracts already work through the regular
    sigmatic synthesis on the lengthened stem from ``parts['fut']``.
  - Tense / mood combinations covered:
      * present subjunctive    (active / middle; mp shared in present)
      * present optative       (active / middle)
      * present imperative     (active / middle; 2sg, 3sg, 2pl, 3pl, duals
                                in active)
      * aorist subjunctive     (active / middle / passive)
      * aorist optative        (active / middle / passive; passive carries
                                duals to mirror jtauber)
      * aorist imperative      (active / middle / passive; sigmatic only)
      * aorist infinitive      (active / middle / passive)
      * present infinitive     (active / middle)
  - The function only produces forms; the caller decides which slots
    to write (and whether to overwrite or only fill empty cells).
  - Accent: the synthesis preserves the lemma / fut-stem accent on stem
    syllables, but does NOT compute fresh accent placement on
    enclitic-final cells (3sg/3pl imperative). Those endings carry an
    inherent acute, but the stem-accent in the synthesised form remains
    where the input has it. Result: 3sg/3pl imperatives may be
    accent-imperfect on multi-syllable stems, but the segmentation is
    always correct so the caller's training pipeline still benefits.
  - jtauber emits one ``middle_present_*`` column for the present
    mediopassive (middle and passive share the same form in the present
    system). We mirror that: this module emits ``middle_present_*`` keys
    only -- no ``passive_present_*``. In the aorist system middle and
    passive split into distinct stems (sigmatic σ-stem vs θη-stem), so
    ``middle_aorist_*`` and ``passive_aorist_*`` get separate cells.

What this module does NOT do (deferred to v3+):
  - Aor-2 (thematic ``ἔλιπον``-style) imperative: ending pattern is
    different (2sg ``λίπε`` vs sigmatic ``λῦσον``).
  - Contract verbs: contract-future / contract-aorist forms have
    distinct vowel-stem rules.
  - Perfect-system synthesis (perfect subj/opt is rare and almost
    always periphrastic).
  - Future-system finite moods (no future subjunctive in Greek; future
    optative exists but is mostly indirect-discourse).
  - Participles: full case×gender×number declension lives in
    ``build/synth_verb_participles.py``.

Pure-Python; no I/O, no DB, no external deps. Mirrors the shape of
``dilemma.morph_diff`` and ``build.lsj_principal_parts``.
"""

from __future__ import annotations

import unicodedata
from typing import Dict, Optional


__all__ = [
    "synthesize_active_moods",
    "synthesize_mp_moods",
    "synthesize_aor2_moods",
    "synthesize_contract_moods",
    "synthesize_past_indicatives",
    "is_thematic_omega",
    "is_contract",
    "contract_class",
    "extract_aor2_stem",
]


# ---------------------------------------------------------------------------
# Lemma classification
# ---------------------------------------------------------------------------


_GREEK_VOWELS = set("αεηιουωᾰᾱ")


def _strip_accents_lower(s: str) -> str:
    """Diacritic-stripped, lowercased version of a Greek string."""
    nfd = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in nfd if not unicodedata.combining(c)).lower()


def is_thematic_omega(lemma: str) -> bool:
    """True iff ``lemma`` is a plain thematic -ω verb suitable for
    template-based mood synthesis.

    Filters out contract (-άω/-έω/-όω), athematic (-μι/-μαι), and any
    lemma that doesn't end in plain ω. Mirrors
    ``dilemma.paradigm._is_regular_thematic_omega`` but accepts both
    bare -ω and accented -ώ (ποιῶ as an alternate citation form, etc.).
    """
    if not lemma:
        return False
    base = _strip_accents_lower(lemma)
    if not base.endswith("ω"):
        return False
    if base.endswith(("αω", "εω", "οω")):
        return False
    if base.endswith(("μι", "μαι")):
        return False
    if len(base) < 2:
        return False
    return True


def contract_class(lemma: str) -> Optional[str]:
    """Return the contract class of a verb: 'alpha', 'epsilon', 'omicron',
    or None if not a contract.

    The classifier looks at the diacritic-stripped lemma's tail. Lemmas
    ending in ``-άω`` / ``-αω`` are alpha-contract; ``-έω`` / ``-εω`` are
    epsilon-contract; ``-όω`` / ``-οω`` are omicron-contract.

    Athematic / mediopassive lemmas (-μι, -μαι, ώμαι etc.) are NOT
    contracts and return None even when their diacritic-stripped tail
    looks similar (``ἵσταμαι`` -> -μαι, not contract).
    """
    if not lemma:
        return None
    base = _strip_accents_lower(lemma)
    if base.endswith(("μι", "μαι")):
        return None
    if not base.endswith("ω"):
        return None
    if len(base) < 3:
        return None
    if base.endswith("αω"):
        return "alpha"
    if base.endswith("εω"):
        return "epsilon"
    if base.endswith("οω"):
        return "omicron"
    return None


def is_contract(lemma: str) -> bool:
    """True iff ``lemma`` is a contract verb (-άω/-έω/-όω)."""
    return contract_class(lemma) is not None


# Ending patterns that mark an aor-2 (strong-aorist) active 1sg form.
# Augmented + ``-ον`` (ἔλιπον / ἔπεσον / ἔλαβον / ἤγαγον) for active.
def _is_aor_2_active_form(aor_form: str) -> bool:
    if not aor_form:
        return False
    plain = _strip_accents_lower(aor_form)
    # Reject sigmatic patterns -- handled elsewhere.
    if plain.endswith(("σα", "ψα", "ξα")):
        return False
    if plain.endswith("κα"):  # κ-aorists like ἔδωκα, ἔθηκα: athematic
        return False
    if plain.endswith("ον") and len(plain) >= 3:
        return True
    return False


# These mixed-alpha forms retain the thematic second-aorist mood system.
# An alpha ending alone is not evidence: liquid first aorists and sigmatic
# variants also end in alpha even when the present stem looks different.
_MIXED_ALPHA_AORISTS = frozenset({
    ("πιπτω", "επεσα"),
    ("λεγω", "ειπα"),
    ("ευρισκω", "ευρα"),
})


def _is_aor_2_alpha_form(aor_form: str, lemma: Optional[str] = None) -> bool:
    """Recognize supported mixed-alpha second aorists by lemma and form.

    Unrecognized alpha forms supply no second-aorist templates. Source
    cells remain available independently, including dialectal forms and
    compounds whose mood formation has not been established here.
    """
    return (_strip_accents_lower(lemma), _strip_accents_lower(aor_form)) in \
        _MIXED_ALPHA_AORISTS


def _strip_simple_augment(form: str) -> Optional[str]:
    """Strip a syllabic ε-augment from ``form``. Returns the unaugmented
    form, or ``None`` if no augment was found.

    This is a lightweight version of ``_strip_augment`` (which handles
    breathing marks) used by the mixed-α detection heuristic. It
    handles syllabic augment ε + consonant and the common temporal
    augments η -> α, ω -> ο.
    """
    if not form:
        return None
    nfd = unicodedata.normalize("NFD", form)
    chars = list(nfd)
    if not chars:
        return None
    if chars[0] == "ε":
        i = 1
        while i < len(chars) and unicodedata.combining(chars[i]):
            i += 1
        if i < len(chars) and chars[i] not in _GREEK_VOWELS:
            rest = "".join(chars[i:])
            return unicodedata.normalize("NFC", rest)
    if chars[0] == "η":
        chars[0] = "α"
        return unicodedata.normalize("NFC", "".join(chars))
    if chars[0] == "ω":
        chars[0] = "ο"
        return unicodedata.normalize("NFC", "".join(chars))
    return unicodedata.normalize("NFC", form)


def extract_aor2_stem(
    aor_form: str, lemma: Optional[str] = None
) -> Optional[str]:
    """Extract the (unaugmented) aor-2 stem from a 1sg active form.

    ``ἔλιπον`` -> ``λιπ``  (drop ε-augment + ``ον``)
    ``ἔλαβον`` -> ``λαβ``
    ``ἔπεσον`` -> ``πεσ``
    ``εὗρον``  -> ``εὑρ``  (no augment if root starts with vowel + breath)
    ``ἤγαγον`` -> ``ἀγαγ``  (temporal augment η -> α)

    Also accepts mixed-α aor-2 forms (-α suffix instead of -ον):
    ``ἔπεσα`` -> ``πεσ``, ``εἶπα`` -> ``εἰπ``, ``εὗρα`` -> ``εὑρ``.

    Returns ``None`` if the form doesn't look like an aor-2.

    The stem retains its accent. For accent-recovery on cells where the
    ending carries an inherent accent, the caller strips the stem accent.
    """
    if not aor_form:
        return None
    nfc = unicodedata.normalize("NFC", aor_form)
    # Ordinary aor-2 in -ον.
    if _is_aor_2_active_form(aor_form):
        if not nfc.endswith("ον"):
            return None
        body = nfc[:-2]
        return _aor2_strip_augment(body)
    # Mixed-α aor-2 in -α.
    if _is_aor_2_alpha_form(aor_form, lemma):
        # Drop trailing α (with combining marks if present).
        nfd = unicodedata.normalize("NFD", nfc)
        chars = list(nfd)
        # Walk back to the last α base char.
        j = len(chars) - 1
        while j >= 0:
            if chars[j] in ("α", "ᾰ"):
                # Drop this char + any following combining marks.
                rest = "".join(chars[:j])
                body = unicodedata.normalize("NFC", rest)
                if body:
                    return _aor2_strip_augment(body)
                return None
            j -= 1
        return None
    return None


def _aor2_strip_augment(body: str) -> Optional[str]:
    """Helper: strip syllabic / temporal augment from an aor-2 body
    (the form with the inflectional ending already stripped)."""
    if not body:
        return None
    nfc = unicodedata.normalize("NFC", body)
    # Strip the augment if present.
    nfd = unicodedata.normalize("NFD", body)
    chars = list(nfd)
    if not chars:
        return None
    # Syllabic augment: leading ε (with optional breathing/accent on it)
    # followed by a consonant -> drop the ε.
    if chars[0] == "ε":
        # Look at the next base letter.
        i = 1
        while i < len(chars) and unicodedata.combining(chars[i]):
            i += 1
        if i < len(chars):
            nxt = chars[i]
            if nxt not in _GREEK_VOWELS:
                # Syllabic augment, drop the ε + its combining marks.
                rest = "".join(chars[i:])
                stem = unicodedata.normalize("NFC", rest)
                return stem if stem else None
            # ε + vowel (= diphthong): leave alone; e.g. εὗρον is its own
            # augment.
    # Temporal augment: η -> α, ω -> ο, ηυ -> αυ.
    if chars[0] == "η":
        # η could be temporal augment of α.
        if len(chars) > 1 and (chars[1] == "υ" or unicodedata.combining(chars[1])):
            # ηυ from αυ: replace with αυ
            if len(chars) > 1 and chars[1] == "υ":
                chars[0] = "α"
                rest = "".join(chars)
                return unicodedata.normalize("NFC", rest)
        chars[0] = "α"
        rest = "".join(chars)
        return unicodedata.normalize("NFC", rest)
    if chars[0] == "ω":
        chars[0] = "ο"
        rest = "".join(chars)
        return unicodedata.normalize("NFC", rest)
    # No augment recognised -- return body as-is (covers εὗρον,
    # ηὗρον -> body εὑρ / ηὑρ; we accept either since accent-position
    # is recovered by the caller stripping tonal accents on demand).
    return unicodedata.normalize("NFC", body)


# ---------------------------------------------------------------------------
# Stem extraction
# ---------------------------------------------------------------------------


def _strip_final_omega(form: str) -> Optional[str]:
    """Drop trailing ω from a form, preserving accents on the stem.

    Returns None if the final character is not ω (or accented ώ).
    """
    if not form:
        return None
    nfc = unicodedata.normalize("NFC", form)
    if nfc.endswith("ω") or nfc.endswith("ώ"):
        return nfc[:-1]
    return None


def _aor_terminal_cluster(aor_form: str) -> Optional[str]:
    """Identify the sigmatic terminal cluster (σ / ψ / ξ) of an aor 1sg
    active form. Returns ``None`` for non-sigmatic / second-aorist
    patterns.

    ``ἔλυσα``     -> ``σ``
    ``ἔγραψα``    -> ``ψ``
    ``ἤγαγον``    -> None (aor-2)
    ``ἐπαίδευσα`` -> ``σ``
    """
    if not aor_form:
        return None
    plain = _strip_accents_lower(aor_form)
    if not plain or len(plain) < 2:
        return None
    if not (plain.endswith("α") or plain.endswith("ᾰ")):
        return None
    cluster = plain[-2]
    if cluster in ("σ", "ψ", "ξ"):
        return cluster
    return None


def _aorist_stem_from_lemma_and_aor(
    lemma: str, aor_form: str
) -> Optional[str]:
    """Build a sigmatic aorist stem by grafting the aor form's
    terminal consonant cluster onto the lemma's present stem.

    Suppletive aorists (where the aor stem differs from the present
    stem in a way the rules don't predict) return None: e.g. πίπτω
    has aor ἔπεσον / ἔπεσα which is a different ROOT (πεσ-, not πιπτ-),
    so we can't graft the σ from ἔπεσα onto πιπτ to get *πίπσω. The
    detection compares the aor body (ε-augment + σ/ψ/ξ + α stripped)
    to the lemma stem with its predicted-cluster source consonant
    stripped; mismatch = suppletive = bail.

    The present stem keeps the lemma's accent. The aor form's σ/ψ/ξ
    replaces the present stem's final consonant when the swap is
    consistent with the standard future-stem composition rule:

        π/β/φ + σ -> ψ
        κ/γ/χ + σ -> ξ
        τ/δ/θ + σ -> σ  (dental drops before σ)
        ν + σ     -> σ
        vowel + σ -> σ

    For verbs whose aorist stem differs from the present stem in a
    way these rules don't predict (suppletion, ablaut, second-aorist
    formed off a different root), this returns ``None`` and the caller
    skips synthesis.
    """
    if not lemma or not aor_form:
        return None
    cluster = _aor_terminal_cluster(aor_form)
    if cluster is None:
        return None
    pres_stem = _present_stem(lemma)
    if not pres_stem:
        return None
    plain_pres = _strip_accents_lower(pres_stem)
    if not plain_pres:
        return None
    last = plain_pres[-1]
    expected_cluster: Optional[str] = None
    if last in ("π", "β", "φ"):
        expected_cluster = "ψ"
    elif last in ("κ", "γ", "χ"):
        expected_cluster = "ξ"
    elif last in ("τ", "δ", "θ", "ν", "ζ"):
        expected_cluster = "σ"
    elif last in _GREEK_VOWELS:
        expected_cluster = "σ"
    if expected_cluster is None or expected_cluster != cluster:
        return None
    # Suppletion guard: verify that the aor form's body (after stripping
    # the augment + cluster + α) matches the lemma stem with its
    # cluster-source consonant trimmed. Mismatch = suppletion = bail
    # (e.g. πίπτω + ἔπεσα: lemma stem πιπτ, expected cluster σ via τ
    # absorption, but body πε mismatches lemma_root πιπ → suppletive).
    body_no_aug = _strip_simple_augment(aor_form)
    if body_no_aug is not None:
        body_plain = _strip_accents_lower(body_no_aug)
        if body_plain.endswith(cluster + "α") or \
                body_plain.endswith(cluster + "ᾰ"):
            body_root = body_plain[:-2]
            lemma_root = (
                plain_pres[:-1] if last not in _GREEK_VOWELS
                else plain_pres
            )
            if body_root and lemma_root and body_root != lemma_root:
                return None
    # Splice the cluster onto the lemma stem, replacing the final
    # consonant unless the stem ended in a vowel (then we just append
    # the σ).
    if last in _GREEK_VOWELS:
        return pres_stem + cluster
    # Drop the final NFC code point and append the cluster. Diacritics
    # ride on the previous letter (vowel), so this preserves the accent.
    return pres_stem[:-1] + cluster


def _aorist_stem_from_fut(fut_form: str) -> Optional[str]:
    """Extract the sigmatic aorist stem from a fut 1sg active form.

    ``λύσω`` -> ``λύσ`` -- preserves accent.
    Returns None if the form doesn't end in ω.

    Validity check: the future stem ends in σ or ψ or ξ for sigmatic
    futures. Verbs with deponent / liquid futures (-εῖ, -οῦμαι) get
    skipped here so we don't templatise non-sigmatic patterns.
    """
    stem = _strip_final_omega(fut_form)
    if stem is None:
        return None
    plain = _strip_accents_lower(stem)
    # Sigmatic future stems end in σ/ψ/ξ (also -ησ for some, -ασ).
    # Reject contract / liquid futures (φανῶ, μενῶ) and middle deponents.
    if not plain or plain[-1] not in ("σ", "ψ", "ξ"):
        return None
    return stem


def _present_stem(lemma: str) -> Optional[str]:
    """Strip the trailing ω/ώ from a thematic lemma."""
    return _strip_final_omega(lemma)


# Tonal accents we strip for ending-accented cells (3sg/3pl imperative).
# Macron / breve quantitative marks (U+0304 / U+0306) are preserved.
_TONAL_ACCENTS = {0x0301, 0x0300, 0x0342}  # acute, grave, circumflex


def _strip_tonal_accents(form: str) -> str:
    """Remove acute/grave/circumflex while keeping macron, breve, and
    breathing marks. Used to neutralise the stem accent on cells whose
    ending carries an inherent acute (3sg / 3pl imperative).
    """
    nfd = unicodedata.normalize("NFD", form or "")
    return unicodedata.normalize(
        "NFC",
        "".join(c for c in nfd if ord(c) not in _TONAL_ACCENTS),
    )


# ---------------------------------------------------------------------------
# Ending tables (active voice)
# ---------------------------------------------------------------------------


# Present-active thematic endings. Used for present subj / opt / imp
# when the cell is empty.
_PRES_SUBJ: Dict[tuple, str] = {
    ("1", "sg"): "ω",
    ("2", "sg"): "ῃς",
    ("3", "sg"): "ῃ",
    ("1", "pl"): "ωμεν",
    ("2", "pl"): "ητε",
    ("3", "pl"): "ωσι(ν)",
    ("2", "du"): "ητον",
    ("3", "du"): "ητον",
}

_PRES_OPT: Dict[tuple, str] = {
    ("1", "sg"): "οιμι",
    ("2", "sg"): "οις",
    ("3", "sg"): "οι",
    ("1", "pl"): "οιμεν",
    ("2", "pl"): "οιτε",
    ("3", "pl"): "οιεν",
    ("2", "du"): "οιτον",
    ("3", "du"): "οίτην",
}

_PRES_IMP: Dict[tuple, str] = {
    ("2", "sg"): "ε",
    ("3", "sg"): "έτω",
    ("2", "pl"): "ετε",
    ("3", "pl"): "όντων",
    ("2", "du"): "ετον",
    ("3", "du"): "έτων",
}


# Aorist-active sigmatic endings. The stem already carries the σ
# (e.g. ``λύσ-`` from ``λύσω``), so the endings are vowel-initial.
_AOR_SUBJ: Dict[tuple, str] = {
    ("1", "sg"): "ω",
    ("2", "sg"): "ῃς",
    ("3", "sg"): "ῃ",
    ("1", "pl"): "ωμεν",
    ("2", "pl"): "ητε",
    ("3", "pl"): "ωσι(ν)",
}

_AOR_OPT: Dict[tuple, str] = {
    ("1", "sg"): "αιμι",
    ("2", "sg"): "αις",
    ("3", "sg"): "αι",
    ("1", "pl"): "αιμεν",
    ("2", "pl"): "αιτε",
    ("3", "pl"): "αιεν",
}

_AOR_IMP: Dict[tuple, str] = {
    ("2", "sg"): "ον",
    ("3", "sg"): "άτω",
    ("2", "pl"): "ατε",
    ("3", "pl"): "άντων",
}

_AOR_INF = "αι"


# ---------------------------------------------------------------------------
# Middle / passive stem helpers
# ---------------------------------------------------------------------------


def _strip_augment(form: str) -> Optional[str]:
    """Strip a syllabic augment ε- (with breathing) from a 1sg form.

    ``ἐλύθην`` -> ``λύθην``.  Returns ``None`` if the form doesn't start
    with an augment we can safely strip (e.g. lengthened-vowel temporal
    augments like ``ἤγαγον`` -- those signal aor-2 anyway).

    Mirrors ``synth_verb_participles._strip_augment``.
    """
    if not form:
        return None
    nfd = unicodedata.normalize("NFD", form)
    chars = list(nfd)
    if not chars:
        return None
    if chars[0] != "ε":
        return None
    i = 1
    while i < len(chars) and unicodedata.combining(chars[i]):
        i += 1
    rest = "".join(chars[i:])
    return unicodedata.normalize("NFC", rest)


def _aor_passive_stem(aor_p_form: str) -> Optional[str]:
    """Extract the aorist passive stem from a 1sg form like ``ἐλύθην``.

    Strips the augment, then strips the trailing ``ην``. Validates that
    the resulting stem ends in θ (sigmatic θη-aorist). Stem-accents are
    preserved -- the caller decides whether to strip them when splicing
    accent-bearing endings.

    Mirrors ``synth_verb_participles._aor_passive_stem`` but does NOT
    pre-strip stem accents: callers here splice both stem-accented
    (athematic-style ``ώ``, ``ῶ``) and ending-accented (``θητι``)
    endings, so the strip happens at emit time.
    """
    if not aor_p_form:
        return None
    body = _strip_augment(aor_p_form)
    if body is None:
        return None
    plain_body = _strip_accents_lower(body)
    if not plain_body.endswith("ην"):
        return None
    nfc = unicodedata.normalize("NFC", body)
    if len(nfc) < 2:
        return None
    stem = nfc[:-2]
    plain_stem = _strip_accents_lower(stem)
    if not plain_stem or plain_stem[-1] != "θ":
        return None
    return stem


# ---------------------------------------------------------------------------
# Ending tables (middle voice)
# ---------------------------------------------------------------------------


# Present-middle thematic endings. jtauber labels these as ``middle_*``
# even though middle and passive share these forms in the present.
_PRES_MID_SUBJ: Dict[tuple, str] = {
    ("1", "sg"): "ωμαι",
    ("2", "sg"): "ῃ",
    ("3", "sg"): "ηται",
    ("1", "pl"): "ώμεθα",
    ("2", "pl"): "ησθε",
    ("3", "pl"): "ωνται",
}


_PRES_MID_OPT: Dict[tuple, str] = {
    ("1", "sg"): "οίμην",
    ("2", "sg"): "οιο",
    ("3", "sg"): "οιτο",
    ("1", "pl"): "οίμεθα",
    ("2", "pl"): "οισθε",
    ("3", "pl"): "οιντο",
}


# Present-middle imperative.
#  2sg ου comes from -εο contracted, lemma-stem accent rides through.
#  3sg/2pl/3pl carry their own accent on the ending.
_PRES_MID_IMP: Dict[tuple, str] = {
    ("2", "sg"): "ου",
    ("3", "sg"): "έσθω",
    ("2", "pl"): "εσθε",
    ("3", "pl"): "έσθων",
}


_PRES_MID_INF = "εσθαι"


# Aorist-middle endings (sigmatic σ-stem). Stem already carries σ.
_AOR_MID_SUBJ: Dict[tuple, str] = {
    ("1", "sg"): "ωμαι",
    ("2", "sg"): "ῃ",
    ("3", "sg"): "ηται",
    ("1", "pl"): "ώμεθα",
    ("2", "pl"): "ησθε",
    ("3", "pl"): "ωνται",
}


_AOR_MID_OPT: Dict[tuple, str] = {
    ("1", "sg"): "αίμην",
    ("2", "sg"): "αιο",
    ("3", "sg"): "αιτο",
    ("1", "pl"): "αίμεθα",
    ("2", "pl"): "αισθε",
    ("3", "pl"): "αιντο",
}


# Aorist-middle imperative. The 2sg is recessive (κέλευσαι, not the
# infinitive κελεῦσαι); final -αι counts as short. The other endings
# retain their existing stem or ending accent.
_AOR_MID_IMP: Dict[tuple, str] = {
    ("2", "sg"): "αι",
    ("3", "sg"): "άσθω",
    ("2", "pl"): "ασθε",
    ("3", "pl"): "άσθων",
}


_AOR_MID_INF = "ασθαι"


# ---------------------------------------------------------------------------
# Ending tables (passive aorist)
# ---------------------------------------------------------------------------


# Aorist-passive endings (athematic θη-aorist). Note: subj/opt take
# active-style endings on a passive stem -- this is correct: the aor
# passive is morphologically active.  The contraction with the θη/θε-
# linker creates circumflex/acute on the joined vowel.
#
# subj 1sg: λυθῶ <- λυθή-ω with ε+ω contraction. Endings as listed are
# what gets appended after the θ stem.
_AOR_PASS_SUBJ: Dict[tuple, str] = {
    ("1", "sg"): "ῶ",
    ("2", "sg"): "ῇς",
    ("3", "sg"): "ῇ",
    ("1", "pl"): "ῶμεν",
    ("2", "pl"): "ῆτε",
    ("3", "pl"): "ῶσι(ν)",
}


# opt 1sg: λυθείην <- θη + ιη + ν. Athematic optative ι-η-marker fuses
# with the η stem-vowel to ει. jtauber emits both pl forms for 1pl and
# the duals (2du / 3du).
_AOR_PASS_OPT: Dict[tuple, str] = {
    ("1", "sg"): "είην",
    ("2", "sg"): "είης",
    ("3", "sg"): "είη",
    ("1", "pl"): "εῖμεν",
    ("2", "pl"): "εῖτε",
    ("3", "pl"): "εῖεν",
    ("2", "du"): "εῖτον",
    ("3", "du"): "είτην",
}


# imp 2sg: -ητι (or -ηθι after stems where dissimilation gives -θητι ->
#                 -θηθι, but for regular θη-aorists -θητι is the form).
# imp 3sg: -ήτω -- ending-accented.
# imp 2pl: -ητε -- stem-accented.
# imp 3pl: -έντων -- 3rd-decl participle form, ending-accented; the η
#                    of the stem drops before the participle linker -ε-.
_AOR_PASS_IMP: Dict[tuple, str] = {
    ("2", "sg"): "ητι",
    ("3", "sg"): "ήτω",
    ("2", "pl"): "ητε",
    ("3", "pl"): "έντων",
}


# Set of (person, number) keys whose passive-aorist imperative ending
# carries its own accent and so requires the stem accent to be stripped
# first, AND additionally drops the stem-final η before splicing.
_AOR_PASS_IMP_3PL_DROPS_ETA = {("3", "pl")}


_AOR_PASS_INF = "ῆναι"


# ---------------------------------------------------------------------------
# Aor-2 (strong-aorist) ending tables
# ---------------------------------------------------------------------------
#
# Aor-2 verbs use present-style endings on a separate (non-sigmatic) stem.
# The unaugmented stem (``λιπ-``, ``λαβ-``, ``πεσ-``) takes
# subjunctive/optative/imperative/infinitive endings; the augmented stem
# (``ἔλιπ-``, ``ἔλαβ-``, ``ἔπεσ-``) takes indicative endings.
#
# Schema mirrors jtauber (πίπτω, λαμβάνω, βαίνω). The 2sg imperative
# accent is recessive on most aor-2 verbs (λίπε / πέσε) but enclitic on
# four classical "irregular" aor-2 verbs (λαβέ / εἰπέ / ἐλθέ / εὑρέ).
# We synthesise the recessive form; the classical-irregular accents come
# from corpus / Wiktionary attestations and won't be overwritten.

# Aor-2 active indicative -- thematic-vowel endings on augmented stem.
# Used for "pure" aor-2 verbs whose 1sg ends in -ον (λείπω → ἔλιπον,
# λαμβάνω → ἔλαβον, εὑρίσκω → εὗρον).
_AOR2_ACT_IND: Dict[tuple, str] = {
    ("1", "sg"): "ον",
    ("2", "sg"): "ες",
    ("3", "sg"): "ε",
    ("1", "pl"): "ομεν",
    ("2", "pl"): "ετε",
    ("3", "pl"): "ον",
}

# Aor-2 middle indicative -- on augmented stem.
_AOR2_MID_IND: Dict[tuple, str] = {
    ("1", "sg"): "όμην",
    ("2", "sg"): "ου",
    ("3", "sg"): "ετο",
    ("1", "pl"): "όμεθα",
    ("2", "pl"): "εσθε",
    ("3", "pl"): "οντο",
}

# Note: a handful of classical aor-2 verbs (πίπτω, εἶπον/εἶπα,
# ἤνεγκον/ἤνεγκα) take α-style endings instead in Attic. We synthesise
# them with the ο-thematic pattern by default; corpus / Wiktionary cells
# carrying the α-form for those specific verbs will not be overwritten,
# and we leave the discrepancy in the cells where corpus is silent.


# Mixed α-aor-2 active indicative endings.
# Used for verbs whose aor 1sg ends in -α not -ον (πίπτω → ἔπεσα,
# λέγω → εἶπα, εὑρίσκω → εὗρα). The 3sg cell is left BLANK to mirror
# jtauber, which doesn't emit a 3sg for these forms (the κ-aorist 3sg
# is -ε, but its accent placement is irregular and varies by verb).
_AOR2_ALPHA_ACT_IND: Dict[tuple, str] = {
    ("1", "sg"): "α",
    ("2", "sg"): "ας",
    ("1", "pl"): "αμεν",
    ("2", "pl"): "ατε",
    ("3", "pl"): "αν",
}

# Mixed α-aor-2 middle indicative endings (πίπτω → ἐπεσάμην, etc.).
# 2sg is omitted from synthesis: jtauber emits -ου for vowel-stems
# (εἴπου / εὕρου) and -ω for consonant-stems (ἐπέσω) and we can't
# reliably predict which without inspecting the unaugmented stem's
# final character. Leave the 2sg cell empty so corpus / Wiktionary
# supplies it (mirroring jtauber's blank 3sg active for these verbs).
_AOR2_ALPHA_MID_IND: Dict[tuple, str] = {
    ("1", "sg"): "άμην",
    ("3", "sg"): "ατο",
    ("1", "pl"): "άμεθα",
    ("2", "pl"): "ασθε",
    ("3", "pl"): "αντο",
}

# Aor-2 active subjunctive -- same shape as present subj.
_AOR2_ACT_SUBJ: Dict[tuple, str] = {
    ("1", "sg"): "ω",
    ("2", "sg"): "ῃς",
    ("3", "sg"): "ῃ",
    ("1", "pl"): "ωμεν",
    ("2", "pl"): "ητε",
    ("3", "pl"): "ωσι(ν)",
}

# Aor-2 middle subjunctive.
_AOR2_MID_SUBJ: Dict[tuple, str] = {
    ("1", "sg"): "ωμαι",
    ("2", "sg"): "ῃ",
    ("3", "sg"): "ηται",
    ("1", "pl"): "ώμεθα",
    ("2", "pl"): "ησθε",
    ("3", "pl"): "ωνται",
}

# Aor-2 active optative -- same shape as present opt.
_AOR2_ACT_OPT: Dict[tuple, str] = {
    ("1", "sg"): "οιμι",
    ("2", "sg"): "οις",
    ("3", "sg"): "οι",
    ("1", "pl"): "οιμεν",
    ("2", "pl"): "οιτε",
    ("3", "pl"): "οιεν",
}

# Aor-2 middle optative.
_AOR2_MID_OPT: Dict[tuple, str] = {
    ("1", "sg"): "οίμην",
    ("2", "sg"): "οιο",
    ("3", "sg"): "οιτο",
    ("1", "pl"): "οίμεθα",
    ("2", "pl"): "οισθε",
    ("3", "pl"): "οιντο",
}

# Aor-2 active imperative.
# 2sg recessive ``-ε`` (λίπε / πέσε).  3sg/3pl carry their own accent.
_AOR2_ACT_IMP: Dict[tuple, str] = {
    ("2", "sg"): "ε",
    ("3", "sg"): "έτω",
    ("2", "pl"): "ετε",
    ("3", "pl"): "όντων",
}

# Aor-2 middle imperative.
# 2sg ``-οῦ`` (λαβοῦ / πεσοῦ -- circumflex from ε+ο contraction on the
# stem-final). 3sg/2pl/3pl carry their own accent.
_AOR2_MID_IMP: Dict[tuple, str] = {
    ("2", "sg"): "οῦ",
    ("3", "sg"): "έσθω",
    ("2", "pl"): "εσθε",
    ("3", "pl"): "έσθων",
}

_AOR2_ACT_INF = "εῖν"
_AOR2_MID_INF = "έσθαι"


# Aor-2 cells whose ending is inherently accented; the stem accent
# is dropped before splicing.
_AOR2_END_ACCENTED_KEYS = {
    "active_aorist_imperative_3sg",   # έτω
    "active_aorist_imperative_3pl",   # όντων
    "active_aorist_infinitive",       # εῖν (end-accented)
    "active_aorist_indicative_1sg",   # augmented stem already has accent
    "middle_aorist_subjunctive_1pl",  # ώμεθα
    "middle_aorist_optative_1sg",     # οίμην
    "middle_aorist_optative_1pl",     # οίμεθα
    "middle_aorist_imperative_2sg",   # οῦ
    "middle_aorist_imperative_3sg",   # έσθω
    "middle_aorist_imperative_3pl",   # έσθων
    "middle_aorist_infinitive",       # έσθαι
}


# ---------------------------------------------------------------------------
# Contract-verb ending tables (-άω / -έω / -όω, present system)
# ---------------------------------------------------------------------------
#
# These tables encode the *contracted* surface form for every cell. The
# stem passed in is the BARE stem (no thematic vowel): ``τιμα-``,
# ``ποιε-``, ``δηλο-`` (or accent-stripped ``τιμ-`` / ``ποι-`` / ``δηλ-``
# for end-accented cells).
#
# The full ending here includes the contract vowel + thematic vowel
# (already fused with the personal ending). Splice as ``stem + ending``
# with the bare stem (no contract vowel).
#
# Stem-accent placement: most cells have the accent fall on the contract
# vowel (the now-fused syllable). We pre-place that accent in the table
# entry and strip the original lemma accent before splicing.

# Alpha contract: stem α. The bare stem is what's left after dropping
# the final α from the lemma stem (e.g. τιμα -> τιμ).
_CONTRACT_ALPHA_ACT_IND: Dict[tuple, str] = {
    ("1", "sg"): "ῶ",
    ("2", "sg"): "ᾷς",
    ("3", "sg"): "ᾷ",
    ("1", "pl"): "ῶμεν",
    ("2", "pl"): "ᾶτε",
    ("3", "pl"): "ῶσι(ν)",
    ("2", "du"): "ᾶτον",
    ("3", "du"): "ᾶτον",
}

_CONTRACT_ALPHA_ACT_SUBJ: Dict[tuple, str] = {
    ("1", "sg"): "ῶ",
    ("2", "sg"): "ᾷς",
    ("3", "sg"): "ᾷ",
    ("1", "pl"): "ῶμεν",
    ("2", "pl"): "ᾶτε",
    ("3", "pl"): "ῶσι(ν)",
}

# alpha-contract optative: -ῴην / -ῴης / -ῴη / -ῷμεν or -ῴημεν / -ῷτε / -ῷεν.
# jtauber emits the longer -ῴημεν / -ῴητε / -ῴησαν forms on τιμάω.
_CONTRACT_ALPHA_ACT_OPT: Dict[tuple, str] = {
    ("1", "sg"): "ῴην",
    ("2", "sg"): "ῴης",
    ("3", "sg"): "ῴη",
    ("1", "pl"): "ῴημεν",
    ("2", "pl"): "ῴητε",
    ("3", "pl"): "ῴησαν",
    ("3", "du"): "ῴτην",
}

# alpha-contract imperative: 2sg short -α; 3sg -ᾱ́τω; 2pl -ᾶτε; 3pl -ώντων.
_CONTRACT_ALPHA_ACT_IMP: Dict[tuple, str] = {
    ("2", "sg"): "α",
    ("3", "sg"): "άτω",
    ("2", "pl"): "ᾶτε",
    ("3", "pl"): "ώντων",
    ("3", "du"): "άτων",
}

_CONTRACT_ALPHA_ACT_INF = "ᾶν"

_CONTRACT_ALPHA_MID_IND: Dict[tuple, str] = {
    ("1", "sg"): "ῶμαι",
    ("2", "sg"): "ᾷ",
    ("3", "sg"): "ᾶται",
    ("1", "pl"): "ώμεθα",
    ("2", "pl"): "ᾶσθε",
    ("3", "pl"): "ῶνται",
}

_CONTRACT_ALPHA_MID_SUBJ: Dict[tuple, str] = {
    ("1", "sg"): "ῶμαι",
    ("2", "sg"): "ᾷ",
    ("3", "sg"): "ᾶται",
    ("1", "pl"): "ώμεθα",
    ("2", "pl"): "ᾶσθε",
    ("3", "pl"): "ῶνται",
}

_CONTRACT_ALPHA_MID_OPT: Dict[tuple, str] = {
    ("1", "sg"): "ῴμην",
    ("2", "sg"): "ῷο",
    ("3", "sg"): "ῷτο",
    ("1", "pl"): "ῴμεθα",
    ("2", "pl"): "ῷσθε",
    ("3", "pl"): "ῷντο",
}

_CONTRACT_ALPHA_MID_IMP: Dict[tuple, str] = {
    ("2", "sg"): "ῶ",
    ("3", "sg"): "ᾱ́σθω",
    ("2", "pl"): "ᾶσθε",
    ("3", "pl"): "ᾱ́σθων",
}

_CONTRACT_ALPHA_MID_INF = "ᾶσθαι"


# Epsilon contract: stem ε. Bare stem is e.g. ποι- (from ποιε-).
_CONTRACT_EPSILON_ACT_IND: Dict[tuple, str] = {
    ("1", "sg"): "ῶ",
    ("2", "sg"): "εῖς",
    ("3", "sg"): "εῖ",
    ("1", "pl"): "οῦμεν",
    ("2", "pl"): "εῖτε",
    ("3", "pl"): "οῦσι(ν)",
    # Note: jtauber emits the *uncontracted* dual forms ποιέετον /
    # φιλέετον for epsilon-contracts (an Ionic / Epic preservation that
    # bleeds into Attic dual paradigms in the Wiktionary tables jtauber
    # mirrors). We don't synthesise the dual here -- if we emit the
    # contracted ποιεῖτον / φιλεῖτον, it will conflict with jtauber.
    # Leave the dual unfilled.
}

_CONTRACT_EPSILON_ACT_SUBJ: Dict[tuple, str] = {
    ("1", "sg"): "ῶ",
    ("2", "sg"): "ῇς",
    ("3", "sg"): "ῇ",
    ("1", "pl"): "ῶμεν",
    ("2", "pl"): "ῆτε",
    ("3", "pl"): "ῶσι(ν)",
}

# Epsilon-contract optative active: -οίην / -οίης / -οίη / -οίημεν or
# the shorter -οῖμεν. jtauber's ποιέω uses the longer -οίη(...) forms.
_CONTRACT_EPSILON_ACT_OPT: Dict[tuple, str] = {
    ("1", "sg"): "οίην",
    ("2", "sg"): "οίης",
    ("3", "sg"): "οίη",
    ("1", "pl"): "οίημεν",
    ("2", "pl"): "οίητε",
    ("3", "pl"): "οίησαν",
}

# Epsilon-contract imperative.
_CONTRACT_EPSILON_ACT_IMP: Dict[tuple, str] = {
    ("2", "sg"): "ει",
    ("3", "sg"): "είτω",
    ("2", "pl"): "εῖτε",
    ("3", "pl"): "ούντων",
    ("3", "du"): "είτων",
}

_CONTRACT_EPSILON_ACT_INF = "εῖν"

_CONTRACT_EPSILON_MID_IND: Dict[tuple, str] = {
    ("1", "sg"): "οῦμαι",
    ("2", "sg"): "εῖ",
    ("3", "sg"): "εῖται",
    ("1", "pl"): "ούμεθα",
    ("2", "pl"): "εῖσθε",
    ("3", "pl"): "οῦνται",
}

_CONTRACT_EPSILON_MID_SUBJ: Dict[tuple, str] = {
    ("1", "sg"): "ῶμαι",
    ("2", "sg"): "ῇ",
    ("3", "sg"): "ῆται",
    ("1", "pl"): "ώμεθα",
    ("2", "pl"): "ῆσθε",
    ("3", "pl"): "ῶνται",
}

_CONTRACT_EPSILON_MID_OPT: Dict[tuple, str] = {
    ("1", "sg"): "οίμην",
    ("2", "sg"): "οῖο",
    ("3", "sg"): "οῖτο",
    ("1", "pl"): "οίμεθα",
    ("2", "pl"): "οῖσθε",
    ("3", "pl"): "οῖντο",
}

_CONTRACT_EPSILON_MID_IMP: Dict[tuple, str] = {
    ("2", "sg"): "οῦ",
    ("3", "sg"): "είσθω",
    ("2", "pl"): "εῖσθε",
    ("3", "pl"): "είσθων",
}

_CONTRACT_EPSILON_MID_INF = "εῖσθαι"


# Omicron contract: stem ο. Bare stem is e.g. δηλ- (from δηλο-).
_CONTRACT_OMICRON_ACT_IND: Dict[tuple, str] = {
    ("1", "sg"): "ῶ",
    ("2", "sg"): "οῖς",
    ("3", "sg"): "οῖ",
    ("1", "pl"): "οῦμεν",
    ("2", "pl"): "οῦτε",
    ("3", "pl"): "οῦσι(ν)",
    ("2", "du"): "οῦτον",
    ("3", "du"): "οῦτον",
}

_CONTRACT_OMICRON_ACT_SUBJ: Dict[tuple, str] = {
    ("1", "sg"): "ῶ",
    ("2", "sg"): "οῖς",
    ("3", "sg"): "οῖ",
    ("1", "pl"): "ῶμεν",
    ("2", "pl"): "ῶτε",
    ("3", "pl"): "ῶσι(ν)",
    ("2", "du"): "ῶτον",
    ("3", "du"): "ῶτον",
}

_CONTRACT_OMICRON_ACT_OPT: Dict[tuple, str] = {
    ("1", "sg"): "οίην",
    ("2", "sg"): "οίης",
    ("3", "sg"): "οίη",
    ("1", "pl"): "οίημεν",
    ("2", "pl"): "οίητε",
    ("3", "pl"): "οίησαν",
}

_CONTRACT_OMICRON_ACT_IMP: Dict[tuple, str] = {
    ("2", "sg"): "ου",
    ("3", "sg"): "ούτω",
    ("2", "pl"): "οῦτε",
    ("3", "pl"): "ούντων",
    ("3", "du"): "οέτων",
}

_CONTRACT_OMICRON_ACT_INF = "οῦν"

_CONTRACT_OMICRON_MID_IND: Dict[tuple, str] = {
    ("1", "sg"): "οῦμαι",
    ("2", "sg"): "οῖ",
    ("3", "sg"): "οῦται",
    ("1", "pl"): "ούμεθα",
    ("2", "pl"): "οῦσθε",
    ("3", "pl"): "οῦνται",
}

_CONTRACT_OMICRON_MID_SUBJ: Dict[tuple, str] = {
    ("1", "sg"): "ῶμαι",
    ("2", "sg"): "οῖ",
    ("3", "sg"): "ῶται",
    ("1", "pl"): "ώμεθα",
    ("2", "pl"): "ῶσθε",
    ("3", "pl"): "ῶνται",
}

_CONTRACT_OMICRON_MID_OPT: Dict[tuple, str] = {
    ("1", "sg"): "οίμην",
    ("2", "sg"): "οῖο",
    ("3", "sg"): "οῖτο",
    ("1", "pl"): "οίμεθα",
    ("2", "pl"): "οῖσθε",
    ("3", "pl"): "οῖντο",
}

_CONTRACT_OMICRON_MID_IMP: Dict[tuple, str] = {
    ("2", "sg"): "οῦ",
    ("3", "sg"): "ούσθω",
    ("2", "pl"): "οῦσθε",
    ("3", "pl"): "ούσθων",
}

_CONTRACT_OMICRON_MID_INF = "οῦσθαι"


# Contract endings are ALL ending-accented (the contraction places an
# inherent accent on the new fused syllable). The stem accent is dropped
# before splicing on every contract cell. This sentinel is the only
# pattern needed; we always strip on contracts.


# Set of mp/passive cells whose ending carries an inherent accent.
# Stem-accents on these get neutralised before splicing so we don't
# produce double-accent forms.
_MP_END_ACCENTED_KEYS = {
    "middle_present_optative_1sg",   # οίμην
    "middle_present_optative_1pl",   # οίμεθα
    "middle_present_subjunctive_1pl",  # ώμεθα
    "middle_present_imperative_3sg",   # έσθω
    "middle_present_imperative_3pl",   # έσθων
    "middle_aorist_optative_1sg",   # αίμην
    "middle_aorist_optative_1pl",   # αίμεθα
    "middle_aorist_subjunctive_1pl",  # ώμεθα
    "middle_aorist_imperative_3sg",   # άσθω
    "middle_aorist_imperative_3pl",   # άσθων
    # Passive aorist subjunctive: ALL cells are ending-accented (the
    # θη + ω contraction produces ω/ῇς/ῇ etc.). Stem accent is dropped.
    "passive_aorist_subjunctive_1sg",
    "passive_aorist_subjunctive_2sg",
    "passive_aorist_subjunctive_3sg",
    "passive_aorist_subjunctive_1pl",
    "passive_aorist_subjunctive_2pl",
    "passive_aorist_subjunctive_3pl",
    # Passive aorist optative: same -- ει contraction is ending-accented.
    "passive_aorist_optative_1sg",
    "passive_aorist_optative_2sg",
    "passive_aorist_optative_3sg",
    "passive_aorist_optative_1pl",
    "passive_aorist_optative_2pl",
    "passive_aorist_optative_3pl",
    "passive_aorist_optative_2du",
    "passive_aorist_optative_3du",
    # Passive aorist imperative 3sg / 3pl carry their own accent.
    "passive_aorist_imperative_3sg",
    "passive_aorist_imperative_3pl",
    # Passive aorist infinitive: ending-accented (-ῆναι).
    "passive_aorist_infinitive",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def synthesize_active_moods(
    lemma: str,
    principal_parts: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Synthesise active subjunctive / optative / imperative / aorist
    infinitive forms for a thematic -ω verb.

    Returns a dict ``{paradigm_key: form}`` with whatever cells the
    templating succeeded for. Returns an empty dict when the lemma is
    not a plain thematic -ω verb.

    The caller is responsible for merging this into the existing
    paradigm and *only* writing into empty slots; this function never
    decides whether a real cell already exists.

    Parameters
    ----------
    lemma : str
        The verb's citation form. Must be NFC.
    principal_parts : dict, optional
        Output of ``parse_principal_parts``. When None or empty, only
        the present-tense moods (which need only the lemma) are
        synthesised.
    """
    if not lemma or not is_thematic_omega(lemma):
        return {}
    parts = principal_parts or {}

    out: Dict[str, str] = {}

    # Cells whose ending carries an inherent acute (έτω / όντων / έτων /
    # άτω / άντων / οίτην). Stem accent is dropped before appending so we
    # don't produce double-accented forms like ``γράψάτω``.
    end_accented_keys = {
        "active_present_imperative_3sg",
        "active_present_imperative_3pl",
        "active_present_imperative_3du",
        "active_present_optative_3du",
        "active_aorist_imperative_3sg",
        "active_aorist_imperative_3pl",
    }

    def _emit(key: str, stem: str, ending: str) -> None:
        if key in end_accented_keys:
            out[key] = _strip_tonal_accents(stem) + ending
        else:
            out[key] = stem + ending

    # ---- Present-system moods (stem from lemma) ----
    pres_stem = _present_stem(lemma)
    if pres_stem:
        for (p, n), end in _PRES_SUBJ.items():
            _emit(f"active_present_subjunctive_{p}{n}", pres_stem, end)
        for (p, n), end in _PRES_OPT.items():
            _emit(f"active_present_optative_{p}{n}", pres_stem, end)
        for (p, n), end in _PRES_IMP.items():
            _emit(f"active_present_imperative_{p}{n}", pres_stem, end)
        # Present infinitive is also derivable but already covered by the
        # paradigm.py template fallback elsewhere; we add it for
        # completeness so a fresh paradigm without that fallback gets it.
        out["active_present_infinitive"] = pres_stem + "ειν"

    # ---- Aorist-system moods (sigmatic only) ----
    # Two paths to the sigmatic aorist stem:
    #   1. Future principal part: strip ω, validate σ/ψ/ξ ending.
    #      Carries the original accent without an augment.
    #   2. Aorist principal part + lemma: identify the σ/ψ/ξ cluster on
    #      the aor form, then graft it onto the lemma's present stem
    #      (which carries the accent). Skips when the present-final
    #      consonant doesn't predict the aorist cluster (suppletion,
    #      ablaut, irregular).
    aor_stem: Optional[str] = None
    if "fut" in parts:
        aor_stem = _aorist_stem_from_fut(parts["fut"])
    if aor_stem is None and "aor" in parts:
        aor_stem = _aorist_stem_from_lemma_and_aor(lemma, parts["aor"])
    # Final sanity: stem must look sigmatic.
    if aor_stem is not None:
        plain = _strip_accents_lower(aor_stem)
        if not plain or plain[-1] not in ("σ", "ψ", "ξ"):
            aor_stem = None

    if aor_stem:
        for (p, n), end in _AOR_SUBJ.items():
            _emit(f"active_aorist_subjunctive_{p}{n}", aor_stem, end)
        for (p, n), end in _AOR_OPT.items():
            _emit(f"active_aorist_optative_{p}{n}", aor_stem, end)
        for (p, n), end in _AOR_IMP.items():
            _emit(f"active_aorist_imperative_{p}{n}", aor_stem, end)
        out["active_aorist_infinitive"] = _long_penult_circumflex(aor_stem, _AOR_INF)

    return out


def _resolve_aor_stem(
    parts: Dict[str, str], lemma: str
) -> Optional[str]:
    """Resolve the sigmatic σ/ψ/ξ-stem from principal parts.

    Tries the future principal part first (carries the original accent
    without an augment), then derives from aor + lemma. Mirrors the
    in-line resolution at the top of ``synthesize_active_moods``.
    """
    aor_stem: Optional[str] = None
    if "fut" in parts:
        aor_stem = _aorist_stem_from_fut(parts["fut"])
    if aor_stem is None and "aor" in parts:
        aor_stem = _aorist_stem_from_lemma_and_aor(lemma, parts["aor"])
    if aor_stem is not None:
        plain = _strip_accents_lower(aor_stem)
        if not plain or plain[-1] not in ("σ", "ψ", "ξ"):
            return None
    return aor_stem


def synthesize_mp_moods(
    lemma: str,
    principal_parts: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Synthesise middle / passive subj / opt / imp / inf forms for a
    thematic -ω verb.

    Returns a dict ``{paradigm_key: form}`` keyed in the jtauber-style
    flat shape:
      ``middle_present_subjunctive_<persnum>``
      ``middle_present_optative_<persnum>``
      ``middle_present_imperative_<persnum>``
      ``middle_present_infinitive``
      ``middle_aorist_subjunctive_<persnum>``
      ``middle_aorist_optative_<persnum>``
      ``middle_aorist_imperative_<persnum>``
      ``middle_aorist_infinitive``
      ``passive_aorist_subjunctive_<persnum>``
      ``passive_aorist_optative_<persnum>`` (incl. 2du / 3du)
      ``passive_aorist_imperative_<persnum>``
      ``passive_aorist_infinitive``

    Important schema notes (matched to jtauber):
      - Present middle and passive share form, so we emit only
        ``middle_present_*`` keys -- no ``passive_present_*``.
      - Aorist middle uses the σ/ψ/ξ-stem (sigmatic).
      - Aorist passive uses the θη-stem (athematic-style endings).
      - When the σ-stem is not derivable (aor-2, liquid future) the
        aorist-middle cells are simply skipped.
      - When the aor passive principal part is missing (no ``aor_p`` in
        the dict, e.g. γράφω which has 2nd-aor passive ἐγράφην that
        doesn't fit the θη pattern) the passive-aorist cells are
        skipped.

    Returns an empty dict when the lemma is not a plain thematic -ω
    verb. The caller is responsible for merging this into the existing
    paradigm and *only* writing into empty slots.
    """
    if not lemma or not is_thematic_omega(lemma):
        return {}
    parts = principal_parts or {}

    out: Dict[str, str] = {}

    def _emit_mp(key: str, stem: str, ending: str) -> None:
        if key in _MP_END_ACCENTED_KEYS:
            out[key] = _strip_tonal_accents(stem) + ending
        else:
            out[key] = stem + ending

    # ---- Present middle (= present mediopassive in jtauber) ----
    pres_stem = _present_stem(lemma)
    if pres_stem:
        for (p, n), end in _PRES_MID_SUBJ.items():
            _emit_mp(f"middle_present_subjunctive_{p}{n}", pres_stem, end)
        for (p, n), end in _PRES_MID_OPT.items():
            _emit_mp(f"middle_present_optative_{p}{n}", pres_stem, end)
        for (p, n), end in _PRES_MID_IMP.items():
            _emit_mp(f"middle_present_imperative_{p}{n}", pres_stem, end)
        out["middle_present_infinitive"] = pres_stem + _PRES_MID_INF

    # ---- Aorist middle (sigmatic σ-stem) ----
    aor_stem = _resolve_aor_stem(parts, lemma)
    if aor_stem:
        for (p, n), end in _AOR_MID_SUBJ.items():
            _emit_mp(f"middle_aorist_subjunctive_{p}{n}", aor_stem, end)
        for (p, n), end in _AOR_MID_OPT.items():
            _emit_mp(f"middle_aorist_optative_{p}{n}", aor_stem, end)
        for (p, n), end in _AOR_MID_IMP.items():
            key = f"middle_aorist_imperative_{p}{n}"
            if (p, n) == ("2", "sg"):
                out[key] = _recessive_full_form(aor_stem, end)
            else:
                _emit_mp(key, aor_stem, end)
        out["middle_aorist_infinitive"] = aor_stem + _AOR_MID_INF

    # ---- Aorist passive (θη-stem) ----
    pass_stem: Optional[str] = None
    if "aor_p" in parts:
        pass_stem = _aor_passive_stem(parts["aor_p"])
    if pass_stem:
        # All passive-aorist subjunctive / optative cells are
        # ending-accented (the θη-vowel contracts with the ending),
        # so the stem accent is dropped on the entire subj/opt grid.
        # The passive aor imperative needs special handling for the
        # 3pl form (-έντων), which drops the stem-final η before
        # splicing the participle-style ending.
        for (p, n), end in _AOR_PASS_SUBJ.items():
            _emit_mp(f"passive_aorist_subjunctive_{p}{n}", pass_stem, end)
        for (p, n), end in _AOR_PASS_OPT.items():
            _emit_mp(f"passive_aorist_optative_{p}{n}", pass_stem, end)
        for (p, n), end in _AOR_PASS_IMP.items():
            key = f"passive_aorist_imperative_{p}{n}"
            if (p, n) in _AOR_PASS_IMP_3PL_DROPS_ETA:
                # 3pl ``λυθέντων``: drop the stem accent AND splice the
                # ending after the bare θ (no η). That means the stem
                # we pass is already θ-only, not θη-... but our pass_stem
                # ends in θ already (we cut ``ην`` off), so we just need
                # to strip the stem accent and use the -έντων ending.
                out[key] = _strip_tonal_accents(pass_stem) + end
            else:
                _emit_mp(key, pass_stem, end)
        out["passive_aorist_infinitive"] = (
            _strip_tonal_accents(pass_stem) + _AOR_PASS_INF
        )

    return out


def _add_recessive_accent(stem: str) -> str:
    """Place an acute accent on the stem's vowel for recessive cells.

    Returns ``stem`` unchanged when it already carries a tonal accent
    (so existing accent isn't doubled). Otherwise finds the stem's
    final-syllable vowel and adds an acute there.

    Used on aor-2 cells like 2sg ``λίπε`` / ``πέσε`` where the bare
    stem (``λιπ`` / ``πεσ``) needs a recessive penult accent before the
    vocalic ending ``-ε`` joins it.
    """
    if not stem:
        return stem
    nfd = unicodedata.normalize("NFD", stem)
    # If a tonal accent is already present, keep the stem as-is.
    if any(ord(c) in _TONAL_ACCENTS for c in nfd):
        return stem
    chars = list(nfd)
    # Walk back to find the last vowel base.
    j = len(chars) - 1
    while j >= 0 and (unicodedata.combining(chars[j]) or chars[j] not in _GREEK_VOWELS):
        j -= 1
    if j < 0:
        return stem
    # Insert combining acute (U+0301) right after the vowel base, but
    # AFTER any combining marks that ride on it (breathing, iota
    # subscript). NFD orders breathing before iota subscript before
    # accent typically, but we just append after all combining marks
    # following the vowel.
    k = j + 1
    while k < len(chars) and unicodedata.combining(chars[k]):
        k += 1
    chars.insert(k, "́")
    return unicodedata.normalize("NFC", "".join(chars))


# Vowels that count as long for accent purposes in final syllables.
# η, ω, ου, ει, αι (when not nominative pl), οι (when not nom pl) etc.
# Simplified: η ω = always long; ει ου ηυ = long diphthong.
_LONG_VOWELS_BASE = set("ηω")


def _ending_is_long(ending: str) -> bool:
    """True iff the final syllable of ``ending`` counts as long for
    Greek accent purposes.

    Heuristic: looks at the last vowel cluster in the ending. η, ω,
    diphthongs ει/ου/ηυ are long; α, ε, ο, αι, οι (final) are short.
    """
    if not ending:
        return False
    plain = _strip_accents_lower(ending)
    if not plain:
        return False
    # Strip a trailing consonant (ν, σ, etc.) so we look at the vowel.
    while plain and plain[-1] not in _GREEK_VOWELS:
        plain = plain[:-1]
    if not plain:
        return False
    last = plain[-1]
    if last in ("η", "ω"):
        return True
    if len(plain) >= 2:
        diph = plain[-2:]
        if diph in ("ει", "ου", "ηυ", "ευ", "αυ"):
            return True
        # αι and οι are short in nom-pl but long elsewhere.
        # In endings used for finite verbs, -αι (active aor inf) is short;
        # -οι final is short. -ηι etc are not common.
        if diph == "αι" or diph == "οι":
            return False
    return False


def _recessive_full_form(prefix: str, ending: str) -> str:
    """Compose ``prefix + ending`` with a freshly-placed recessive accent.

    The accent goes:
      - on the antepenult if the final syllable's vowel is short, OR
      - on the penult if the final syllable's vowel is long.

    Both ``prefix`` and ``ending`` are stripped of pre-existing tonal
    accents before combining; the result has exactly one acute on the
    correct syllable.

    For 2-syllable forms (e.g. λίπε from λιπ + ε), the accent is on
    the only valid recessive position (penult).
    """
    return _accent_full_form(prefix, ending)


def _long_penult_circumflex(prefix: str, ending: str) -> str:
    """Apply the long-penult rule without moving a persistent stem accent.

    First-aorist active infinitives and neuter participles retain their
    accent position, unlike finite recessive forms (Smyth 425, 427).
    Only change an acute to a circumflex when quantity establishes it.
    """
    original = unicodedata.normalize("NFD", prefix + ending)
    candidate = unicodedata.normalize(
        "NFD", _accent_full_form(prefix, ending, penult=True)
    )
    if candidate.replace("͂", "́") == original:
        return unicodedata.normalize("NFC", candidate)
    return unicodedata.normalize("NFC", original)


def _accent_full_form(prefix: str, ending: str, *, penult: bool = False) -> str:
    """Place a recessive or fixed-penult accent using explicit quantity."""
    if not prefix:
        return ending
    pre = _strip_tonal_accents(prefix)
    end = _strip_tonal_accents(ending)
    full_nfd = unicodedata.normalize("NFD", pre + end)
    chars = list(full_nfd)

    def is_diphthong(first: int, second: int) -> bool:
        if (chars[first] + chars[second]).lower() not in (
                "αι", "ει", "οι", "υι", "αυ", "ευ", "ηυ", "ου"):
            return False
        # A diaeresis explicitly separates the second vowel: ἀΐξαι
        # has three syllables, not two. Preserve the mark in the result.
        after = second + 1
        while after < len(chars) and unicodedata.combining(chars[after]):
            if chars[after] == "̈":
                return False
            after += 1
        return True

    # Identify vowel-syllable positions (one per maximal vowel run).
    # We collect (position-of-base, is_long-flag) so we can pick the
    # antepenult / penult.
    syllables: list[int] = []  # index of base vowel char in `chars`
    i = 0
    while i < len(chars):
        c = chars[i]
        if unicodedata.combining(c):
            i += 1
            continue
        if c in _GREEK_VOWELS:
            # Start of a vowel-syllable. Scan to the end of the cluster
            # (consecutive base vowels merge into a diphthong = one syl).
            start = i
            i += 1
            # Skip combining marks attached to this vowel.
            while i < len(chars) and unicodedata.combining(chars[i]):
                i += 1
            # Greedily merge consecutive vowel bases as a diphthong.
            while i < len(chars) and chars[i] in _GREEK_VOWELS:
                # only treat as same syllable for known diphthongs
                # (αι ει οι υι αυ ευ ηυ ου).
                if is_diphthong(start, i):
                    i += 1
                    while i < len(chars) and unicodedata.combining(chars[i]):
                        i += 1
                else:
                    break
            syllables.append(start)
        else:
            i += 1
    if not syllables:
        return unicodedata.normalize("NFC", pre + end)
    # Determine if the final syllable is long.
    final_long = _ending_is_long(end)
    # Pick target syllable index.
    n = len(syllables)
    if n == 1:
        target = 0
    elif penult or n == 2 or final_long:
        target = n - 2  # penult
    else:
        target = n - 3  # antepenult
    if target < 0:
        target = 0
    base_idx = syllables[target]
    # Determine if the target syllable's vowel is long. Long base
    # vowels: η, ω. Long diphthongs: αι, ει, οι, υι, αυ, ευ, ηυ, ου
    # (when followed by another vowel base in the same syllable).
    target_vowel = chars[base_idx]
    target_is_diph = False
    nxt = base_idx + 1
    while nxt < len(chars) and unicodedata.combining(chars[nxt]):
        nxt += 1
    if nxt < len(chars) and chars[nxt] in _GREEK_VOWELS:
        if is_diphthong(base_idx, nxt):
            target_is_diph = True
    target_long = (
        target_vowel.lower() in ("η", "ω") or target_is_diph
    )
    # A supplied macron or circumflex also establishes vowel quantity.
    # Repositioning an accent must not turn a known long α/ι/υ into a
    # guessed short one. Match by base-letter position because stripping
    # the old accent changes the combining-character offsets.
    original_letters = []
    for char in unicodedata.normalize("NFD", prefix + ending):
        if unicodedata.combining(char):
            if original_letters:
                original_letters[-1].append(char)
        else:
            original_letters.append([char])
    base_number = sum(not unicodedata.combining(c) for c in chars[:base_idx])
    target_marks = original_letters[base_number][1:]
    target_long = target_long or any(mark in target_marks for mark in ("̄", "͂"))
    # Greek accent rule: circumflex (̃) goes on the penult when the
    # penult vowel is LONG and the ultimate is SHORT; otherwise we
    # place an acute (́). The rule applies only to penult/ult, not to
    # antepenult cells.
    use_circumflex = (
        target == n - 2 and target_long and not final_long
    )
    # For diphthongs, the accent conventionally sits on the SECOND
    # vowel of the diphthong. Walk forward past combining marks AND
    # any second vowel that forms a diphthong with this base.
    k = base_idx + 1
    while k < len(chars) and unicodedata.combining(chars[k]):
        k += 1
    if k < len(chars) and chars[k] in _GREEK_VOWELS:
        if is_diphthong(base_idx, k):
            k += 1
            while k < len(chars) and unicodedata.combining(chars[k]):
                k += 1
    accent_mark = "͂" if use_circumflex else "́"
    chars.insert(k, accent_mark)
    return unicodedata.normalize("NFC", "".join(chars))


# Aor-2 cells whose stem vowel takes a recessive acute (when the stem
# itself has no accent, e.g. after augment-stripping). 2sg ``λίπε``
# (penult acute on stem vowel ι) and 2pl ``λίπετε`` (antepenult acute
# on stem vowel for trisyllable). Subjunctive/optative cells inherit
# the recessive pattern via the long-vowel ending so don't need extra
# attention here.
_AOR2_NEEDS_RECESSIVE_ACCENT = {
    "active_aorist_imperative_2sg",   # λίπε / πέσε
    "active_aorist_imperative_2pl",   # λίπετε / πέσετε
    "active_aorist_subjunctive_1sg",  # λίπω
    "active_aorist_subjunctive_1pl",  # λίπωμεν
    "active_aorist_subjunctive_2pl",  # λίπητε
    "active_aorist_subjunctive_2sg",  # λίπῃς
    "active_aorist_subjunctive_3sg",  # λίπῃ
    "active_aorist_subjunctive_3pl",  # λίπωσι
    "active_aorist_optative_1sg",     # λίποιμι
    "active_aorist_optative_2sg",     # λίποις
    "active_aorist_optative_3sg",     # λίποι
    "active_aorist_optative_1pl",     # λίποιμεν
    "active_aorist_optative_2pl",     # λίποιτε
    "active_aorist_optative_3pl",     # λίποιεν
    "middle_aorist_subjunctive_1sg",  # λίπωμαι
    "middle_aorist_subjunctive_2sg",  # λίπῃ
    "middle_aorist_subjunctive_3sg",  # λίπηται
    "middle_aorist_subjunctive_2pl",  # λίπησθε
    "middle_aorist_subjunctive_3pl",  # λίπωνται
    "middle_aorist_optative_2sg",     # λίποιο
    "middle_aorist_optative_3sg",     # λίποιτο
    "middle_aorist_optative_2pl",     # λίποισθε
    "middle_aorist_optative_3pl",     # λίποιντο
    "middle_aorist_imperative_2pl",   # λίπεσθε
}


def _aor2_emit(
    out: Dict[str, str], key: str, stem: str, ending: str
) -> None:
    """Splice ``stem + ending`` for an aor-2 cell, dropping stem accent
    if the cell is end-accented, or computing a recessive accent on the
    full form (stem + ending) when the cell wants recessive accent."""
    if key in _AOR2_END_ACCENTED_KEYS:
        out[key] = _strip_tonal_accents(stem) + ending
    elif key in _AOR2_NEEDS_RECESSIVE_ACCENT:
        out[key] = _recessive_full_form(stem, ending)
    else:
        out[key] = stem + ending


def synthesize_aor2_moods(
    lemma: str,
    principal_parts: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Synthesise aor-2 (strong-aorist) cells for verbs with a non-
    sigmatic aorist principal part.

    Returns ``{paradigm_key: form}`` covering active and middle aorist
    indicative / subjunctive / optative / imperative / infinitive when
    a parseable aor-2 stem is available. Empty dict when:
      - The lemma is not a thematic-style -ω verb (athematic / contract).
      - ``parts['aor']`` is missing or doesn't look like an aor-2.
      - The lemma is a proper aor-1 verb (sigmatic) — those go through
        ``synthesize_active_moods`` / ``synthesize_mp_moods`` instead.

    Caller is responsible for merging into the existing paradigm and
    *only* writing into empty slots; aor-2 stems often coexist with
    sigmatic aor-1 attestations, and we don't want to overwrite the
    corpus's choice between them.
    """
    if not lemma:
        return {}
    parts = principal_parts or {}
    # Keep the selected principal part if it fits this active template.
    # Otherwise try the separately identified second aorist; that field
    # can also contain a middle form, which must not mask an active form.
    aor_form = parts.get("aor")
    aor_stem = extract_aor2_stem(aor_form, lemma)
    if not aor_stem:
        aor_form = parts.get("aor2")
        aor_stem = extract_aor2_stem(aor_form, lemma)
    if not aor_stem:
        return {}

    # Detect mixed-α aor-2: aor 1sg ends in -α (not -ον), but the form
    # isn't a clean sigmatic σ-aorist. πίπτω/ἔπεσα, λέγω/εἶπα,
    # εὑρίσκω/εὗρα. Active and middle indicative cells use α-style
    # endings on the augmented stem; everything else (subj/opt/imp/inf)
    # uses regular aor-2 ο-thematic endings on the unaugmented stem.
    is_alpha = _is_aor_2_alpha_form(aor_form, lemma)

    # The augmented stem prefixes the indicative cells. We rebuild it
    # from the bare stem by re-prepending the augment when possible.
    # The aor_form already has the augmented stem, so cut its trailing
    # ``ον`` (or ``α`` for mixed-α) to get the augmented stem.
    nfc_aor = unicodedata.normalize("NFC", aor_form)
    aug_stem: Optional[str] = None
    if nfc_aor.endswith("ον"):
        aug_stem = nfc_aor[:-2]
    elif is_alpha:
        # Drop the trailing α (with combining marks if present).
        nfd_aor = unicodedata.normalize("NFD", nfc_aor)
        chars_aor = list(nfd_aor)
        j = len(chars_aor) - 1
        while j >= 0:
            if chars_aor[j] in ("α", "ᾰ"):
                aug_stem = unicodedata.normalize(
                    "NFC", "".join(chars_aor[:j])
                )
                break
            j -= 1
        if not aug_stem:
            aug_stem = None

    # Pick ending tables based on α-pattern detection.
    act_ind_table = _AOR2_ALPHA_ACT_IND if is_alpha else _AOR2_ACT_IND
    mid_ind_table = _AOR2_ALPHA_MID_IND if is_alpha else _AOR2_MID_IND

    out: Dict[str, str] = {}

    # ---- Active aorist indicative (augmented stem) ----
    # Indicative is recessive; we compute the accent fresh from the
    # full augmented form so multi-syllable forms like ``ἐλίπομεν``
    # land the accent on the antepenult rather than the augment ε.
    if aug_stem:
        for (p, n), end in act_ind_table.items():
            key = f"active_aorist_indicative_{p}{n}"
            out[key] = _recessive_full_form(aug_stem, end)

    # ---- Active aorist subjunctive / optative / imperative / infinitive
    # (unaugmented stem) ----
    for (p, n), end in _AOR2_ACT_SUBJ.items():
        _aor2_emit(out, f"active_aorist_subjunctive_{p}{n}", aor_stem, end)
    for (p, n), end in _AOR2_ACT_OPT.items():
        _aor2_emit(out, f"active_aorist_optative_{p}{n}", aor_stem, end)
    for (p, n), end in _AOR2_ACT_IMP.items():
        _aor2_emit(out, f"active_aorist_imperative_{p}{n}", aor_stem, end)
    _aor2_emit(out, "active_aorist_infinitive", aor_stem, _AOR2_ACT_INF)

    # ---- Middle aorist (unaugmented stem for non-indic, augmented for
    # indic) ----
    # Middle aor-2 indicative is recessive too; compute fresh.
    if aug_stem:
        for (p, n), end in mid_ind_table.items():
            key = f"middle_aorist_indicative_{p}{n}"
            out[key] = _recessive_full_form(aug_stem, end)
    for (p, n), end in _AOR2_MID_SUBJ.items():
        _aor2_emit(out, f"middle_aorist_subjunctive_{p}{n}", aor_stem, end)
    for (p, n), end in _AOR2_MID_OPT.items():
        _aor2_emit(out, f"middle_aorist_optative_{p}{n}", aor_stem, end)
    for (p, n), end in _AOR2_MID_IMP.items():
        _aor2_emit(out, f"middle_aorist_imperative_{p}{n}", aor_stem, end)
    _aor2_emit(out, "middle_aorist_infinitive", aor_stem, _AOR2_MID_INF)

    return out


# ---------------------------------------------------------------------------
# Contract verbs (present-system mood synthesis)
# ---------------------------------------------------------------------------


def _contract_bare_stem(lemma: str) -> Optional[str]:
    """Strip the trailing contract-vowel + ω from a contract lemma.

    ``τιμάω`` -> ``τιμ`` (drop ``άω``)
    ``ποιέω`` -> ``ποι`` (drop ``έω``)
    ``δηλόω`` -> ``δηλ`` (drop ``όω``)

    Accent on the stem (if any beyond the contract syllable) rides
    through. Lemmas with the accent ON the contract syllable
    (``τιμᾷ`` as alt citation form) yield bare stems with no accent.

    Returns None if the lemma isn't a contract.
    """
    if not lemma:
        return None
    cls = contract_class(lemma)
    if cls is None:
        return None
    nfc = unicodedata.normalize("NFC", lemma)
    # Drop the final character (ω/ώ).
    if nfc.endswith("ω") or nfc.endswith("ώ"):
        nfc = nfc[:-1]
    # Drop the contract-vowel base char. We work in NFD so we can step
    # past any combining marks that ride on the contract vowel.
    nfd = unicodedata.normalize("NFD", nfc)
    chars = list(nfd)
    # Walk from the end: collect combining marks, then drop one base char.
    j = len(chars) - 1
    while j >= 0 and unicodedata.combining(chars[j]):
        j -= 1
    if j < 0:
        return None
    # chars[j] is the contract vowel base (α / ε / ο). Drop it AND its
    # combining marks (which are at chars[j+1..]).
    rest = chars[:j]
    out = unicodedata.normalize("NFC", "".join(rest))
    return out if out else None


def synthesize_contract_moods(
    lemma: str,
    principal_parts: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Synthesise the present-system mood cells for a contract verb.

    Returns ``{paradigm_key: form}`` covering active + middle present
    indicative / subjunctive / optative / imperative / infinitive for
    -άω / -έω / -όω verbs. Future and aorist of contracts behave like
    regular sigmatic thematic verbs (the contract vowel lengthens before
    σ), so they are NOT synthesised here -- the regular
    ``synthesize_active_moods`` / ``synthesize_mp_moods`` paths can fill
    those when given a future principal part.

    Returns an empty dict when the lemma is not a contract.

    Caller is responsible for merging into the existing paradigm and
    *only* writing into empty slots.
    """
    if not lemma:
        return {}
    cls = contract_class(lemma)
    if cls is None:
        return {}
    bare = _contract_bare_stem(lemma)
    if bare is None:
        return {}

    # All contract endings are inherently end-accented; the bare stem's
    # original accent (if any) gets dropped to avoid double-accent. The
    # only stem accent that should survive is on multi-syllable stems
    # where the lemma has accent BEFORE the contract syllable, but
    # jtauber's table never preserves it on the bare stem there either
    # (e.g. lemma ποίημι -> stem ποι- accent dropped). We always strip.
    bare_no_accent = _strip_tonal_accents(bare)

    out: Dict[str, str] = {}

    if cls == "alpha":
        ind_act = _CONTRACT_ALPHA_ACT_IND
        subj_act = _CONTRACT_ALPHA_ACT_SUBJ
        opt_act = _CONTRACT_ALPHA_ACT_OPT
        imp_act = _CONTRACT_ALPHA_ACT_IMP
        inf_act = _CONTRACT_ALPHA_ACT_INF
        ind_mid = _CONTRACT_ALPHA_MID_IND
        subj_mid = _CONTRACT_ALPHA_MID_SUBJ
        opt_mid = _CONTRACT_ALPHA_MID_OPT
        imp_mid = _CONTRACT_ALPHA_MID_IMP
        inf_mid = _CONTRACT_ALPHA_MID_INF
    elif cls == "epsilon":
        ind_act = _CONTRACT_EPSILON_ACT_IND
        subj_act = _CONTRACT_EPSILON_ACT_SUBJ
        opt_act = _CONTRACT_EPSILON_ACT_OPT
        imp_act = _CONTRACT_EPSILON_ACT_IMP
        inf_act = _CONTRACT_EPSILON_ACT_INF
        ind_mid = _CONTRACT_EPSILON_MID_IND
        subj_mid = _CONTRACT_EPSILON_MID_SUBJ
        opt_mid = _CONTRACT_EPSILON_MID_OPT
        imp_mid = _CONTRACT_EPSILON_MID_IMP
        inf_mid = _CONTRACT_EPSILON_MID_INF
    elif cls == "omicron":
        ind_act = _CONTRACT_OMICRON_ACT_IND
        subj_act = _CONTRACT_OMICRON_ACT_SUBJ
        opt_act = _CONTRACT_OMICRON_ACT_OPT
        imp_act = _CONTRACT_OMICRON_ACT_IMP
        inf_act = _CONTRACT_OMICRON_ACT_INF
        ind_mid = _CONTRACT_OMICRON_MID_IND
        subj_mid = _CONTRACT_OMICRON_MID_SUBJ
        opt_mid = _CONTRACT_OMICRON_MID_OPT
        imp_mid = _CONTRACT_OMICRON_MID_IMP
        inf_mid = _CONTRACT_OMICRON_MID_INF
    else:
        return {}

    # Active present
    for (p, n), end in ind_act.items():
        out[f"active_present_indicative_{p}{n}"] = bare_no_accent + end
    for (p, n), end in subj_act.items():
        out[f"active_present_subjunctive_{p}{n}"] = bare_no_accent + end
    for (p, n), end in opt_act.items():
        out[f"active_present_optative_{p}{n}"] = bare_no_accent + end
    for (p, n), end in imp_act.items():
        # 2sg imperative for contracts is recessive: τίμα / ποίει / δήλου
        # carry their accent on the lemma stem (penult of the full form),
        # not the contract vowel. Compute the recessive accent fresh on
        # the joined form.
        if (p, n) == ("2", "sg") and cls in ("alpha", "epsilon", "omicron"):
            out[f"active_present_imperative_{p}{n}"] = (
                _recessive_full_form(bare_no_accent, end)
            )
        else:
            out[f"active_present_imperative_{p}{n}"] = bare_no_accent + end
    out["active_present_infinitive"] = bare_no_accent + inf_act

    # Middle present
    for (p, n), end in ind_mid.items():
        out[f"middle_present_indicative_{p}{n}"] = bare_no_accent + end
    for (p, n), end in subj_mid.items():
        out[f"middle_present_subjunctive_{p}{n}"] = bare_no_accent + end
    for (p, n), end in opt_mid.items():
        out[f"middle_present_optative_{p}{n}"] = bare_no_accent + end
    for (p, n), end in imp_mid.items():
        out[f"middle_present_imperative_{p}{n}"] = bare_no_accent + end
    out["middle_present_infinitive"] = bare_no_accent + inf_mid

    return out


# ---------------------------------------------------------------------------
# Past-indicative 1sg synthesis (imperfect / aorist active / middle / passive)
# ---------------------------------------------------------------------------
#
# kaikki occasionally drops tense tags on Wiktionary past-indicative cells,
# so high-frequency verbs end up with the only attested 1sg-imperfect or
# 1sg-aorist forms tagged as Homeric (which `build_grc_verb_paradigms` now
# correctly filters out into ``dialects.epic``). The result: empty 1sg
# slots in the canonical Attic slice.
#
# This module fills those gaps by templating the 1sg from principal parts
# (when the parts dict is populated by :func:`lsj_principal_parts.parse_principal_parts`)
# plus the lemma's present stem. Aorist 1sg cells come straight from the
# principal-parts dict (the canonical principal part IS the 1sg active /
# middle / passive aorist). Imperfect 1sg cells are built procedurally from
# the present stem + augment + thematic ending.
#
# Conservatism rules:
#   * Athematic -μι / -μαι verbs are skipped (covered by dilemma's own
#     athematic synthesis where it exists).
#   * Contract verbs (-άω / -έω / -όω) get separate ending tables for the
#     imperfect (the contract vowel + thematic vowel fuse into ω / ουν).
#   * Verbs whose lemma starts with η-, ω-, or a long-vowel diphthong are
#     ambiguous on temporal augment (the augmented form looks identical to
#     the unaugmented one), and we bail.
#   * Prefixed compounds (lemma starts with ε- and the principal part
#     also starts with ε-) get the augment internally between prefix and
#     root; we don't try to insert it.
#   * Aorist cells whose principal-part 1sg ending shape doesn't match the
#     expected category (e.g. -μην for an aor_med slot) are skipped to
#     avoid templating wrong-voice forms.


# Vowel-lengthening map for temporal augment. Mirrors
# ``dilemma.morph_diff._AUGMENT_VOWEL_LENGTHENINGS`` but lives here so this
# module stays import-light.
_TEMPORAL_AUGMENT: Dict[str, str] = {
    "α": "η",
    "ε": "η",
    "ο": "ω",
    "ι": "ι",  # length unchanged in Attic, just marked long; we keep ι
    "υ": "υ",  # length unchanged; we keep υ
    "αι": "ῃ",
    "ει": "ῃ",
    "οι": "ῳ",
    "αυ": "ηυ",
    "ευ": "ηυ",
}


# Imperfect endings. Active and middle/passive have separate tables.
# 1sg / 3pl active are homophonous (-ον). 1sg middle is ``-όμην``;
# accent on the augment for short-stem verbs (ἐλυόμην, ἐγραφόμην) but
# the synthesis here just appends without re-accenting -- the stem we
# pass already carries its accent, and ``-όμην`` ends-accented gets
# stripped onto the stem accent only when the cell is end-accented.
_IMPF_ACT_END: Dict[tuple, str] = {
    ("1", "sg"): "ον",
    ("2", "sg"): "ες",
    ("3", "sg"): "ε",
    ("1", "pl"): "ομεν",
    ("2", "pl"): "ετε",
    ("3", "pl"): "ον",
}


_IMPF_MP_END: Dict[tuple, str] = {
    ("1", "sg"): "όμην",
    ("2", "sg"): "ου",
    ("3", "sg"): "ετο",
    ("1", "pl"): "όμεθα",
    ("2", "pl"): "εσθε",
    ("3", "pl"): "οντο",
}


# Imperfect endings for contract verbs. The contract vowel + thematic
# vowel fuse with the personal ending. Tables encode the fused result
# without an accent; the caller computes recessive accent on the
# augmented + ending sequence.
#
# Note on quantity: the fused vowels (ω, ου, α) are LONG even when not
# circumflexed. ``_ending_is_long`` reads this off the surface vowel,
# so ω / ου trigger penult accent (no antepenult). For the α-contract
# impf 3sg ``ἐτίμα`` the surface α is long (ᾱ from ε+α+ε), but we don't
# mark macron here — the accent placement is computed off the ending
# shape ``-α`` which heuristically reads as short. We pre-mark the
# alpha-contract 3sg ending with a macron so the heuristic correctly
# skips the antepenult on 2-syllable forms like ``ἐτίμα`` (penult ι is
# the only valid recessive position on the augmented stem).
_IMPF_CONTRACT_ALPHA_ACT: Dict[tuple, str] = {
    ("1", "sg"): "ων",      # ε + α + ον -> ων
    ("2", "sg"): "ας",      # ε + α + ες -> ας
    ("3", "sg"): "α",       # ε + α + ε  -> α (ᾱ surface)
    ("1", "pl"): "ωμεν",    # ε + α + ομεν -> ωμεν
    ("2", "pl"): "ατε",     # ε + α + ετε -> ατε
    ("3", "pl"): "ων",      # ε + α + ον  -> ων
}


_IMPF_CONTRACT_ALPHA_MP: Dict[tuple, str] = {
    ("1", "sg"): "ωμην",    # ε + α + ομην -> ωμην
    ("2", "sg"): "ω",       # ε + α + ου  -> ω
    ("3", "sg"): "ατο",     # ε + α + ετο -> ατο
    ("1", "pl"): "ωμεθα",
    ("2", "pl"): "ασθε",
    ("3", "pl"): "ωντο",
}


_IMPF_CONTRACT_EPSILON_ACT: Dict[tuple, str] = {
    ("1", "sg"): "ουν",     # ε + ε + ον -> ουν
    ("2", "sg"): "εις",     # ε + ε + ες -> εις
    ("3", "sg"): "ει",      # ε + ε + ε  -> ει
    ("1", "pl"): "ουμεν",   # ε + ε + ομεν -> ουμεν
    ("2", "pl"): "ειτε",
    ("3", "pl"): "ουν",
}


_IMPF_CONTRACT_EPSILON_MP: Dict[tuple, str] = {
    ("1", "sg"): "ουμην",   # ε + ε + ομην -> ουμην
    ("2", "sg"): "ου",
    ("3", "sg"): "ειτο",
    ("1", "pl"): "ουμεθα",
    ("2", "pl"): "εισθε",
    ("3", "pl"): "ουντο",
}


_IMPF_CONTRACT_OMICRON_ACT: Dict[tuple, str] = {
    ("1", "sg"): "ουν",     # ε + ο + ον -> ουν
    ("2", "sg"): "ους",
    ("3", "sg"): "ου",
    ("1", "pl"): "ουμεν",
    ("2", "pl"): "ουτε",
    ("3", "pl"): "ουν",
}


_IMPF_CONTRACT_OMICRON_MP: Dict[tuple, str] = {
    ("1", "sg"): "ουμην",   # ε + ο + ομην -> ουμην
    ("2", "sg"): "ου",
    ("3", "sg"): "ουτο",
    ("1", "pl"): "ουμεθα",
    ("2", "pl"): "ουσθε",
    ("3", "pl"): "ουντο",
}


def _is_thematic_deponent(lemma: str) -> bool:
    """True iff ``lemma`` is a plain thematic -ομαι deponent.

    Mirrors :func:`is_thematic_omega` but for middle/passive citation
    forms ending in -ομαι (γίγνομαι, βούλομαι, ἀγωνίζομαι). Excludes
    athematic -μαι (κεῖμαι, δύναμαι, ἵσταμαι, ἐπίσταμαι, ...) and
    contract -άομαι / -έομαι / -όομαι (which use the contract pattern
    on a deponent stem).
    """
    if not lemma:
        return False
    base = _strip_accents_lower(lemma)
    if not base.endswith("ομαι"):
        return False
    if base.endswith(("αομαι", "εομαι", "οομαι")):
        return False
    # Athematic -μαι forms end in stem-vowel (α / υ) + μαι (κεῖμαι, δύναμαι).
    # Thematic deponents always have a consonant or short vowel BEFORE the
    # -ομαι. Since we already required ``-ομαι`` and ruled out the
    # contract suffixes, what's left is the thematic deponent class.
    if len(base) < 5:
        return False
    return True


def _add_augment(stem: str) -> Optional[str]:
    """Prepend a syllabic ε- (with smooth breathing) to a consonant-initial
    stem, or lengthen a leading vowel (temporal augment).

    Returns the augmented stem with all tonal accents stripped (the
    caller re-applies a recessive accent on the full inflected form).
    Returns ``None`` when the augment is morphologically ambiguous
    (long-vowel-initial stems where the augmented form is
    indistinguishable from the unaugmented one).
    """
    if not stem:
        return None
    nfc = unicodedata.normalize("NFC", stem)
    plain = _strip_accents_lower(nfc)
    if not plain:
        return None
    first = plain[0]
    # Long-vowel-initial: augment is invisible. Bail.
    if first in ("η", "ω"):
        return None
    # Diphthong / long-vowel cases that we don't lengthen: ει-, ευ-, ου-
    # (already long), ἀ- with macron (ambiguous). Be conservative.
    if len(plain) >= 2 and plain[:2] in ("ει", "ου"):
        return None

    # Strip any existing tonal accents from the stem; the caller
    # re-applies a recessive accent on the full augmented form.
    stripped_stem = _strip_tonal_accents(nfc)

    # Consonant-initial: prepend syllabic ε with smooth breathing (ἐ-).
    if first not in _GREEK_VOWELS:
        # Augmented form: ἐ + stem.  Smooth breathing on the ε.
        # We use the precomposed ἐ (U+1F10) for the smooth-breathing
        # form, which is what jtauber and Wiktionary use.
        return "ἐ" + stripped_stem

    # Vowel-initial: temporal augment.  Lengthen the leading vowel.
    # First check 2-character diphthongs (αι/ει/οι/αυ/ευ) before single
    # characters.
    two = plain[:2] if len(plain) >= 2 else ""
    if two in _TEMPORAL_AUGMENT:
        return _replace_initial_vowels(stripped_stem, 2, _TEMPORAL_AUGMENT[two])
    if first in _TEMPORAL_AUGMENT:
        target = _TEMPORAL_AUGMENT[first]
        # Skip when target == first (η/ω/ι/υ already long) -- that's
        # covered by the η/ω bail above for η/ω, and ι/υ are
        # quantitatively long but not orthographically distinct.
        if target == first:
            return None
        return _replace_initial_vowels(stripped_stem, 1, target)
    return None


def _replace_initial_vowels(form: str, count: int, replacement: str) -> str:
    """Replace the first ``count`` base characters of ``form`` with
    ``replacement``, preserving any breathing mark that was attached
    to the first character (rough / smooth) and dropping tonal accents
    on the replaced segment. The replacement string is used as-is.
    """
    nfd = unicodedata.normalize("NFD", form)
    chars = list(nfd)
    base_seen = 0
    breathing: Optional[str] = None
    keep_marks: list[str] = []
    i = 0
    while i < len(chars) and base_seen < count:
        c = chars[i]
        if not unicodedata.combining(c):
            base_seen += 1
        else:
            # Collect breathing mark from the FIRST base char's combining
            # marks; drop tonal accents from all replaced segments.
            if c in ("̓", "̔") and breathing is None:
                breathing = c
            # Macron / breve quantity marks: drop, replacement implies length.
        i += 1
    rest = "".join(chars[i:])
    new = replacement
    if breathing is not None:
        # Insert breathing after the FIRST base char of the replacement.
        # The replacement is plain Greek (no diacritics), so we splice
        # breathing in at NFD index 1 of the first character.
        nfd_repl = unicodedata.normalize("NFD", replacement)
        repl_chars = list(nfd_repl)
        if repl_chars:
            # Find index after the first base char.
            j = 1
            while j < len(repl_chars) and unicodedata.combining(repl_chars[j]):
                j += 1
            new = "".join(repl_chars[:j]) + breathing + "".join(repl_chars[j:])
    return unicodedata.normalize("NFC", new + rest)


def _imperfect_active_stem(lemma: str) -> Optional[str]:
    """Return the augmented imperfect-active stem for a thematic -ω
    lemma, or None when synthesis isn't safe.

    The returned stem carries the augment but NOT the imperfect ending;
    the caller appends the ending. For ``λύω`` -> ``ἐλυ``; ``ἀκούω`` ->
    ``ἤκου``; ``γράφω`` -> ``ἔγραφ``.

    Bails when the lemma starts with η/ω (augment invisible), starts
    with ει- / ου- (already long), or is a contract / athematic / aor-2
    pattern that doesn't fit the simple thematic mould.
    """
    if not lemma:
        return None
    pres_stem = _present_stem(lemma)
    if not pres_stem:
        return None
    # Check for prefixed-compound shape: stem starts with ε- AND the
    # next character is a vowel (covers ἐπι-, ἐξ-, ἐν-, ἐκ-). We bail
    # because the augment goes between the prefix and the root, and we
    # can't reliably split the prefix without a morphological dictionary.
    plain = _strip_accents_lower(pres_stem)
    if plain.startswith("ε") and len(plain) >= 2:
        # ε followed by another vowel (diphthong / hiatus) is genuine
        # initial-ε; no compound.  ε followed by a consonant could be
        # the root or a compound prefix.  We bail conservatively here:
        # plain ε-initial verbs whose augmented imperfect is εἰ- (ἐθέλω
        # -> ἤθελον but also ἔχω -> εἶχον) are tricky; let other
        # sources fill these.
        if plain[1] not in _GREEK_VOWELS:
            return None
    return _add_augment(pres_stem)


def _imperfect_mp_stem(lemma: str) -> Optional[str]:
    """Return the augmented imperfect-mp stem for a thematic -ω OR
    -ομαι lemma. Mirrors :func:`_imperfect_active_stem` but accepts
    deponents whose citation form ends in -ομαι.
    """
    if not lemma:
        return None
    base = _strip_accents_lower(lemma)
    if base.endswith("ομαι"):
        # Strip the -ομαι to get the stem, then augment.  We work in NFC
        # and trim 4 characters (ο, μ, α, ι are all base codepoints in
        # standard NFC for these forms).
        nfc = unicodedata.normalize("NFC", lemma)
        if len(nfc) < 4:
            return None
        stem = nfc[:-4]
        if not stem:
            return None
        plain = _strip_accents_lower(stem)
        if plain.startswith("ε") and len(plain) >= 2 and plain[1] not in _GREEK_VOWELS:
            return None
        return _add_augment(stem)
    return _imperfect_active_stem(lemma)


def _looks_like_active_aor_1sg(form: str) -> bool:
    """Sanity check: the form should end in -α / -κα / -ον / -ην /
    -ξα / -ψα for an active aorist 1sg.  Rejects -μην (middle 1sg)
    so deponent middle-aorist principal parts don't leak into active
    cells.

    -ην is included for κ-aor 3rd-person-style forms and for athematic
    aor-2 (ἔστην, ἔβην, ἔγνων, ἔδραν) where the 1sg active genuinely
    ends in -ην or -ων.  -ν alone (no preceding η) is rejected to keep
    middle 1sg out.
    """
    if not form:
        return False
    plain = _strip_accents_lower(form)
    if not plain:
        return False
    # Middle 1sg ends in -μην; reject.
    if plain.endswith("μην"):
        return False
    # Accept the conventional active-aorist 1sg endings.
    if plain.endswith(("α", "ᾰ")):
        return True
    if plain.endswith("ον"):
        return True
    if plain.endswith(("ην", "ων")):
        return True
    return False


def _looks_like_middle_aor_1sg(form: str) -> bool:
    """Sanity check: middle aorist 1sg ends in -μην."""
    if not form:
        return False
    plain = _strip_accents_lower(form)
    return plain.endswith("μην")


def _looks_like_passive_aor_1sg(form: str) -> bool:
    """Sanity check: passive aorist 1sg ends in -ην (-θην, -ην for
    aor-2-passive ἐγράφην-style)."""
    if not form:
        return False
    plain = _strip_accents_lower(form)
    return plain.endswith("ην")


def synthesize_past_indicatives(
    lemma: str,
    principal_parts: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Synthesise missing past-indicative 1sg cells for a verb.

    Returns a dict ``{paradigm_key: form}`` covering up to five cells:
        active_imperfect_indicative_1sg
        middle_imperfect_indicative_1sg
        active_aorist_indicative_1sg
        middle_aorist_indicative_1sg
        passive_aorist_indicative_1sg

    Aorist 1sg cells come straight from the principal-parts dict (the
    canonical principal part IS the 1sg form). Imperfect 1sg cells are
    built from the lemma's present stem + augment + thematic ending.

    Returns an empty dict when the lemma is athematic (-μι), suppletive
    in a way that makes the augment ambiguous (η-/ω-initial), or a
    prefixed compound where the augment's position can't be safely
    determined.

    Caller decides whether to merge into an existing paradigm (only
    write empty cells) or overwrite.
    """
    if not lemma:
        return {}
    parts = principal_parts or {}

    out: Dict[str, str] = {}

    # ---- Aorist principal-part copies ----
    # Active aorist 1sg comes from parts['aor'] (the canonical aor 1sg).
    # The principal part is already augmented, so direct copy.
    if "aor" in parts and _looks_like_active_aor_1sg(parts["aor"]):
        out["active_aorist_indicative_1sg"] = parts["aor"]

    # Middle aorist 1sg: parts['aor_med'] when LSJ explicitly attests
    # it (separate :--Med. section). Many verbs have only a sigmatic
    # active aorist, so we don't synthesise the middle from the active
    # here -- if LSJ doesn't carry a separate aor_med the verb may not
    # have an Attic-attested middle aorist 1sg.
    if "aor_med" in parts and _looks_like_middle_aor_1sg(parts["aor_med"]):
        out["middle_aorist_indicative_1sg"] = parts["aor_med"]

    # Passive aorist 1sg: parts['aor_p'] (the canonical aor passive 1sg).
    if "aor_p" in parts and _looks_like_passive_aor_1sg(parts["aor_p"]):
        out["passive_aorist_indicative_1sg"] = parts["aor_p"]

    # ---- Imperfect synthesis ----
    # The imperfect 1sg is templated from the lemma's present stem +
    # augment + ending. We dispatch on lemma shape (plain thematic ω,
    # contract -άω/-έω/-όω, deponent -ομαι) and bail on athematic forms.
    # Imperfect indicative is recessive-accented across the board, so we
    # compose the stem and ending and let _recessive_full_form place the
    # accent.
    cls = contract_class(lemma)
    if cls is not None:
        # Contract verbs: bare stem + augment + fused ending.  We splice
        # the augmented bare stem with the unaccented fused ending and
        # let the recessive-accent helper place a single accent on the
        # correct syllable. The fused vowels (-ων / -ουν) are long, so
        # 3-syllable forms get penult accent (ἐποίουν), 2-syllable forms
        # get penult (ἐτίμα), and 4+-syllable forms still get antepenult
        # only when the ultima is short (impf endings ending in -ε / -α
        # only).
        bare = _contract_bare_stem(lemma)
        if bare is not None:
            aug_bare = _add_augment(bare)
            if aug_bare is not None:
                if cls == "alpha":
                    act_end = _IMPF_CONTRACT_ALPHA_ACT
                    mp_end = _IMPF_CONTRACT_ALPHA_MP
                elif cls == "epsilon":
                    act_end = _IMPF_CONTRACT_EPSILON_ACT
                    mp_end = _IMPF_CONTRACT_EPSILON_MP
                else:  # omicron
                    act_end = _IMPF_CONTRACT_OMICRON_ACT
                    mp_end = _IMPF_CONTRACT_OMICRON_MP
                end_act = act_end.get(("1", "sg"))
                end_mp = mp_end.get(("1", "sg"))
                if end_act:
                    out["active_imperfect_indicative_1sg"] = (
                        _recessive_full_form(aug_bare, end_act)
                    )
                if end_mp:
                    out["middle_imperfect_indicative_1sg"] = (
                        _recessive_full_form(aug_bare, end_mp)
                    )
    elif is_thematic_omega(lemma):
        aug_stem = _imperfect_active_stem(lemma)
        if aug_stem is not None:
            out["active_imperfect_indicative_1sg"] = _recessive_full_form(
                aug_stem, _IMPF_ACT_END[("1", "sg")]
            )
            out["middle_imperfect_indicative_1sg"] = _recessive_full_form(
                aug_stem, _IMPF_MP_END[("1", "sg")]
            )
    elif _is_thematic_deponent(lemma):
        # Deponent -ομαι: only middle/passive imperfect; no active.
        aug_stem = _imperfect_mp_stem(lemma)
        if aug_stem is not None:
            out["middle_imperfect_indicative_1sg"] = _recessive_full_form(
                aug_stem, _IMPF_MP_END[("1", "sg")]
            )

    return out


if __name__ == "__main__":
    # Smoke test for a few canonical verbs.
    samples = [
        ("λύω", {"fut": "λύσω", "aor": "ἔλυσα", "aor_p": "ἐλύθην"}),
        ("παιδεύω", {"aor": "ἐπαίδευσα", "aor_p": "ἐπαιδεύθην"}),
        ("γράφω", {"aor": "ἔγραψα", "fut_med": "γράψομαι"}),  # no aor_p
        ("πείθω", {"fut": "πείσω", "aor": "ἔπεισα", "aor_p": "ἐπείσθην"}),
        ("τιμάω", {"fut": "τιμήσω"}),  # contract; should be skipped
        ("τίθημι", {}),                # athematic; should be skipped
        ("μένω", {"fut": "μενῶ"}),     # liquid future; aorist skipped
        ("λείπω", {"aor": "ἔλιπον"}),  # aor-2
        ("πίπτω", {"aor": "ἔπεσον"}),  # aor-2
    ]
    for lemma, parts in samples:
        out_act = synthesize_active_moods(lemma, parts)
        out_mp = synthesize_mp_moods(lemma, parts)
        out_aor2 = synthesize_aor2_moods(lemma, parts)
        out_contract = synthesize_contract_moods(lemma, parts)
        print(f"=== {lemma} (parts={parts}) ===")
        print(f"  active: {len(out_act)} cells, mp: {len(out_mp)} cells, "
              f"aor2: {len(out_aor2)}, contract: {len(out_contract)}")
        for k in sorted(out_aor2)[:6]:
            print(f"  {k} = {out_aor2[k]}")
        print()
