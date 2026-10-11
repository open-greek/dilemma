#!/usr/bin/env python3
"""Procedural synthesis of Greek participle paradigms from principal parts.

Sister module to ``build/synth_verb_moods.py``. Where that module fills
finite-mood cells (subjunctive / optative / imperative / aorist
infinitive), this one synthesises the full case×gender×number declension
of the participles for each tense×voice combo a thematic -ω verb
supports.

Scope (v1):
  - Plain thematic -ω verbs only. Contracts (-άω/-έω/-όω), athematic
    (-μι/-μαι), deponents (lemma -μαι), and aor-2 verbs are skipped.
  - Tense × voice combos covered:
      * present_active     (3rd-decl, -ων / -ουσα / -ον)
      * present_mp         (1st/2nd-decl, -όμενος / -ομένη / -όμενον)
      * future_active      (3rd-decl, -σων / -σουσα / -σον)
      * future_middle      (1st/2nd-decl, -σόμενος / -σομένη / -σόμενον)
      * future_passive     (1st/2nd-decl, -θησόμενος / -η / -ον)
      * aorist_active      (3rd-decl, -σας / -σασα / -σαν)
      * aorist_middle      (1st/2nd-decl, -σάμενος / -η / -ον)
      * aorist_passive     (3rd-decl, -θείς / -θεῖσα / -θέν)
      * perfect_active     (3rd-decl, -κώς / -κυῖα / -κός)
      * perfect_mp         (1st/2nd-decl, -μένος / -η / -ον)

Key shape mirrors jtauber_ag_paradigms.json:

    {voice}_{tense}_participle_{case}_{gender}_{number}

with ``case`` in (nom, gen, dat, acc, voc), ``gender`` in (m, f, n),
``number`` in (sg, pl). Duals are NOT synthesised in v1: jtauber emits
duals only for present_active and perfect_active, and getting the dual
forms right requires extra accent rules; we punt for now.

Accent: synthesis preserves the lemma's stem accent for stem-accented
participles (present_active, perfect_mp, present_mp). For ending-
accented participles (perfect_active, aorist_passive) the stem accent is
stripped before splicing the ending. We do NOT attempt to recompute
recessive accent placement on multi-syllable stems for cells where the
ending forces an accent shift past the antepenult: the produced cells
may have the accent one syllable off in those cases, but the segmentation
is always correct so the cell still helps downstream training pipelines.

Pure-Python; no I/O, no DB, no external deps. Mirrors the shape of
``build.synth_verb_moods``.
"""

from __future__ import annotations

import unicodedata
from typing import Dict, Optional


__all__ = [
    "synthesize_participles",
    "synthesize_aor2_participles",
    "synthesize_contract_participles",
    "is_thematic_omega",
]


# ---------------------------------------------------------------------------
# Lemma classification (mirrors synth_verb_moods.is_thematic_omega)
# ---------------------------------------------------------------------------


_GREEK_VOWELS = set("αεηιουωᾰᾱ")


def _strip_accents_lower(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in nfd if not unicodedata.combining(c)).lower()


def is_thematic_omega(lemma: str) -> bool:
    """True iff ``lemma`` is a plain thematic -ω verb suitable for
    template-based participle synthesis."""
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


# ---------------------------------------------------------------------------
# Stem extraction helpers
# ---------------------------------------------------------------------------


# Tonal accents (acute, grave, circumflex). Stripped on the stem before
# splicing endings that carry their own accent.
_TONAL_ACCENTS = {0x0301, 0x0300, 0x0342}


def _strip_tonal_accents(form: str) -> str:
    """Remove acute/grave/circumflex while keeping macron, breve, and
    breathing marks."""
    nfd = unicodedata.normalize("NFD", form or "")
    return unicodedata.normalize(
        "NFC",
        "".join(c for c in nfd if ord(c) not in _TONAL_ACCENTS),
    )


def _strip_final_omega(form: str) -> Optional[str]:
    """Drop trailing ω from a form (preserves accents on the stem)."""
    if not form:
        return None
    nfc = unicodedata.normalize("NFC", form)
    if nfc.endswith("ω") or nfc.endswith("ώ"):
        return nfc[:-1]
    return None


def _present_stem(lemma: str) -> Optional[str]:
    """Strip the trailing ω/ώ from a thematic lemma."""
    return _strip_final_omega(lemma)


def _aor_terminal_cluster(aor_form: str) -> Optional[str]:
    """Identify the sigmatic terminal cluster (σ / ψ / ξ) of an aor 1sg
    active form. Returns ``None`` for non-sigmatic / second-aorist
    patterns."""
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
    """Build a sigmatic aorist stem by grafting the aor form's terminal
    consonant cluster onto the lemma's present stem.

    Mirrors ``synth_verb_moods._aorist_stem_from_lemma_and_aor``.
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
    if last in _GREEK_VOWELS:
        return pres_stem + cluster
    return pres_stem[:-1] + cluster


def _aorist_stem_from_fut(fut_form: str) -> Optional[str]:
    """Extract the sigmatic aorist stem from a fut 1sg active form."""
    stem = _strip_final_omega(fut_form)
    if stem is None:
        return None
    plain = _strip_accents_lower(stem)
    if not plain or plain[-1] not in ("σ", "ψ", "ξ"):
        return None
    return stem


def _aor2_stem(aor_form: str) -> Optional[str]:
    """Extract the unaugmented aor-2 stem from a 1sg form like ``ἔλιπον``.

    Returns the stem with stem-accents preserved (the caller strips them
    when splicing end-accented endings). Returns None if the form isn't
    a thematic aor-2 (-ον ending).

    Mirrors ``synth_verb_moods.extract_aor2_stem``.
    """
    if not aor_form:
        return None
    plain = _strip_accents_lower(aor_form)
    if not plain.endswith("ον") or len(plain) < 3:
        return None
    nfc = unicodedata.normalize("NFC", aor_form)
    body = nfc[:-2]  # drop final ον
    nfd = unicodedata.normalize("NFD", body)
    chars = list(nfd)
    if not chars:
        return None
    # Strip ε-augment.
    if chars[0] == "ε":
        i = 1
        while i < len(chars) and unicodedata.combining(chars[i]):
            i += 1
        if i < len(chars) and chars[i] not in _GREEK_VOWELS:
            rest = "".join(chars[i:])
            stem = unicodedata.normalize("NFC", rest)
            return stem if stem else None
    # Temporal augments (η -> α, ω -> ο).
    if chars[0] == "η":
        if len(chars) > 1 and chars[1] == "υ":
            chars[0] = "α"
            return unicodedata.normalize("NFC", "".join(chars))
        chars[0] = "α"
        return unicodedata.normalize("NFC", "".join(chars))
    if chars[0] == "ω":
        chars[0] = "ο"
        return unicodedata.normalize("NFC", "".join(chars))
    # No augment recognised; return body as-is.
    return unicodedata.normalize("NFC", body)


def _strip_augment(form: str) -> Optional[str]:
    """Strip a syllabic augment ε- (with breathing) from a 1sg form.

    ``ἐλύθην`` -> ``λύθην``
    ``ἐπαύθην`` -> ``παύθην``
    ``ἔλυσα`` -> ``λυσα``  (loses the accent that was on the augment)

    Returns ``None`` if the form doesn't start with an augment we can
    safely strip (e.g. lengthened-vowel temporal augments like ``ἤγαγον``
    are skipped — those signal aor-2 verbs anyway).
    """
    if not form:
        return None
    nfd = unicodedata.normalize("NFD", form)
    chars = list(nfd)
    if not chars:
        return None
    # First base letter must be ε
    if chars[0] != "ε":
        return None
    # Skip the ε and any combining marks attached to it; keep the rest.
    i = 1
    while i < len(chars) and unicodedata.combining(chars[i]):
        i += 1
    rest = "".join(chars[i:])
    return unicodedata.normalize("NFC", rest)


def _aor_passive_stem(aor_p_form: str) -> Optional[str]:
    """Extract the aorist passive stem from a 1sg form like ``ἐλύθην``.

    Strips the augment, then strips the trailing ``ην``. The remaining
    stem (e.g. ``λυθ`` for ``ἐλύθην``) gets the participle endings
    spliced on with their own accent.
    """
    if not aor_p_form:
        return None
    body = _strip_augment(aor_p_form)
    if body is None:
        return None
    plain_body = _strip_accents_lower(body)
    if not plain_body.endswith("ην"):
        return None
    # Cut the last 2 NFC code points (η + ν). Diacritics on the η ride
    # with that letter so they're naturally dropped.
    nfc = unicodedata.normalize("NFC", body)
    if len(nfc) < 2:
        return None
    stem = nfc[:-2]
    # Aorist-passive stems should end in θ (sigmatic θη-aorist) for the
    # regular pattern. Verbs whose θ has been assimilated (πέμπω →
    # ἐπέμφθην, πεμφθ-) are also valid: stem ends in φθ / σθ / χθ /
    # bare θ.
    plain_stem = _strip_accents_lower(stem)
    if not plain_stem or plain_stem[-1] != "θ":
        return None
    # Strip stem accent: aor-pass participles are ending-accented.
    return _strip_tonal_accents(stem)


def _perfect_active_stem(pf_form: str) -> Optional[str]:
    """Extract the perfect active stem from a 1sg form like ``λέλυκα``.

    Strips the trailing ``α`` (or ``ᾰ``). Then strips the stem accent
    because perfect-active participles are ending-accented (-ώς).
    Skips strong/2-perfect forms that don't end in -κα (like ``πέποιθα``
    for πείθω): those would need the πεποιθώς pattern, which we leave
    untemplated since corpus / Wiktionary cells already supply it
    where attested.
    """
    if not pf_form:
        return None
    plain = _strip_accents_lower(pf_form)
    if len(plain) < 2:
        return None
    # Require -κα ending: weak/1-perfect only. Strong/2-perfect (no κ)
    # would mis-template (γέγραφα is strong: 'γεγραφώς' is fine, but the
    # template would synthesize 'γεγραφώς' from 'γέγραφα' the same way
    # as the κ-perfect, so allow either). Actually accept -α generally
    # but require the stem to be a single consonant cluster (not a
    # vowel-stem like ἀκήκοα → ἀκηκοώς, which is irregular reduplication).
    if not (plain.endswith("α") or plain.endswith("ᾰ")):
        return None
    nfc = unicodedata.normalize("NFC", pf_form)
    stem = nfc[:-1]
    plain_stem = _strip_accents_lower(stem)
    if not plain_stem:
        return None
    last = plain_stem[-1]
    # Reject if the stem ends in a vowel — those are usually irregular
    # (ἀκήκοα/ἀκηκοώς has a hiatus between κο and ώς, which our endings
    # would stack incorrectly). Accept κ, γ, χ, π, β, φ, τ, δ, θ, σ, ν.
    if last in _GREEK_VOWELS:
        return None
    return _strip_tonal_accents(stem)


def _perfect_mp_stem(pf_mp_form: str) -> Optional[str]:
    """Extract the perfect mediopassive stem from a 1sg form like
    ``λέλυμαι``. Strips the trailing ``μαι``. Keeps the stem accent
    because perfect-mp participles are stem-accented (λελυμένος).
    """
    if not pf_mp_form:
        return None
    plain = _strip_accents_lower(pf_mp_form)
    if not plain.endswith("μαι"):
        return None
    nfc = unicodedata.normalize("NFC", pf_mp_form)
    if len(nfc) < 3:
        return None
    return nfc[:-3]


# ---------------------------------------------------------------------------
# Ending tables
# ---------------------------------------------------------------------------


# 3rd-declension participle endings, present-active pattern (-ων/-ουσα/-ον).
# Keys are (case, gender, number) -> ending appended to the present stem.
# These cells are STEM-ACCENTED: the lemma's accent on the stem rides through.
#
# Macron / breve quantity marks on feminine endings follow jtauber's
# convention (matched against λύω, παύω, γράφω, παιδεύω, τιμάω, ποιέω,
# δηλόω, ὁράω):
#   * nom_f_sg / voc_f_sg / acc_f_sg: final α takes BREVE (-σᾰ / -σᾰν)
#   * nom_f_du / acc_f_du / voc_f_du: final α takes MACRON (-σᾱ)
#   * acc_f_pl: final α takes MACRON (-σᾱς)
#   * Other feminine cells (-σαι, -σης, -σῃ, -σαις, -σων, -σῶν): unmarked
_PRES_ACTIVE_3RD: Dict[tuple, str] = {
    # masculine
    ("nom", "m", "sg"): "ων",
    ("gen", "m", "sg"): "οντος",
    ("dat", "m", "sg"): "οντι",
    ("acc", "m", "sg"): "οντα",
    ("voc", "m", "sg"): "ων",
    ("nom", "m", "pl"): "οντες",
    ("gen", "m", "pl"): "όντων",   # gen pl is ending-accented in 3rd-decl
    ("dat", "m", "pl"): "ουσι(ν)",
    ("acc", "m", "pl"): "οντας",
    ("voc", "m", "pl"): "οντες",
    # neuter
    ("nom", "n", "sg"): "ον",
    ("gen", "n", "sg"): "οντος",
    ("dat", "n", "sg"): "οντι",
    ("acc", "n", "sg"): "ον",
    ("voc", "n", "sg"): "ον",
    ("nom", "n", "pl"): "οντα",
    ("gen", "n", "pl"): "όντων",
    ("dat", "n", "pl"): "ουσι(ν)",
    ("acc", "n", "pl"): "οντα",
    ("voc", "n", "pl"): "οντα",
    # feminine (1st-decl-α-impure: -ουσα/-ούσης/-ούσῃ/-ουσαν/-ουσα)
    ("nom", "f", "sg"): "ουσᾰ",
    ("gen", "f", "sg"): "ούσης",
    ("dat", "f", "sg"): "ούσῃ",
    ("acc", "f", "sg"): "ουσᾰν",
    ("voc", "f", "sg"): "ουσᾰ",
    ("nom", "f", "pl"): "ουσαι",
    ("gen", "f", "pl"): "ουσῶν",
    ("dat", "f", "pl"): "ούσαις",
    ("acc", "f", "pl"): "ούσᾱς",
    ("voc", "f", "pl"): "ουσαι",
}


# Ending-accented cells in the 3rd-decl present-active pattern. The stem
# accent is stripped before splicing these endings (since the ending
# carries its own acute/circumflex).
_PRES_ACTIVE_3RD_END_ACCENTED = {
    ("gen", "m", "pl"),
    ("gen", "n", "pl"),
    ("gen", "f", "pl"),
    ("gen", "f", "sg"),
    ("dat", "f", "sg"),
    ("dat", "f", "pl"),
    ("acc", "f", "pl"),
}


# 3rd-declension participle endings, future-active pattern (-σων/-σουσα).
# Same shape as present active (-ων/-ουσα) but jtauber does NOT emit
# the breve on the present-style nom/voc/acc f sg (compare γράψουσα with
# no marks vs. γράφουσᾰ with breve). Only the acc_f_pl carries a macron.
# This split matches jtauber's data exactly.
_FUT_ACTIVE_3RD: Dict[tuple, str] = {
    # masculine
    ("nom", "m", "sg"): "ων",
    ("gen", "m", "sg"): "οντος",
    ("dat", "m", "sg"): "οντι",
    ("acc", "m", "sg"): "οντα",
    ("voc", "m", "sg"): "ων",
    ("nom", "m", "pl"): "οντες",
    ("gen", "m", "pl"): "όντων",
    ("dat", "m", "pl"): "ουσι(ν)",
    ("acc", "m", "pl"): "οντας",
    ("voc", "m", "pl"): "οντες",
    # neuter
    ("nom", "n", "sg"): "ον",
    ("gen", "n", "sg"): "οντος",
    ("dat", "n", "sg"): "οντι",
    ("acc", "n", "sg"): "ον",
    ("voc", "n", "sg"): "ον",
    ("nom", "n", "pl"): "οντα",
    ("gen", "n", "pl"): "όντων",
    ("dat", "n", "pl"): "ουσι(ν)",
    ("acc", "n", "pl"): "οντα",
    ("voc", "n", "pl"): "οντα",
    # feminine: jtauber leaves final α unmarked except acc_f_pl macron.
    ("nom", "f", "sg"): "ουσα",
    ("gen", "f", "sg"): "ούσης",
    ("dat", "f", "sg"): "ούσῃ",
    ("acc", "f", "sg"): "ουσαν",
    ("voc", "f", "sg"): "ουσα",
    ("nom", "f", "pl"): "ουσαι",
    ("gen", "f", "pl"): "ουσῶν",
    ("dat", "f", "pl"): "ούσαις",
    ("acc", "f", "pl"): "ούσᾱς",
    ("voc", "f", "pl"): "ουσαι",
}


# 3rd-declension participle endings, aorist-active pattern (-σας/-σασα/-σαν).
# The σ is already on the stem (e.g. λυσ-, γραψ-).
#
# Macron / breve marks on feminine endings (jtauber convention, matched
# against λύω, γράφω, παύω, παιδεύω, τιμάω, ποιέω, δηλόω, φέρω):
#   * The σ-stem α (the FIRST α between the two σ's) takes MACRON
#     UNIVERSALLY across every feminine cell. The σ-stem α is the
#     "linking" α of the aorist participle, always long.
#   * The FINAL α additionally takes MACRON in acc_f_pl (-σᾱ́σᾱς).
#   * Other final-α cells (nom/voc/acc sg, nom/voc pl, dat sg/pl): the
#     final α is unmarked.
#   * gen_f_pl / gen_f_sg: gen pl is end-accented (-σᾱσῶν), gen sg also
#     end-accented (-σᾱ́σης) — both keep the σ-stem macron.
_AOR_ACTIVE_3RD: Dict[tuple, str] = {
    # masculine
    ("nom", "m", "sg"): "ας",
    ("gen", "m", "sg"): "αντος",
    ("dat", "m", "sg"): "αντι",
    ("acc", "m", "sg"): "αντα",
    ("voc", "m", "sg"): "ας",
    ("nom", "m", "pl"): "αντες",
    ("gen", "m", "pl"): "άντων",
    ("dat", "m", "pl"): "ασι(ν)",
    ("acc", "m", "pl"): "αντας",
    ("voc", "m", "pl"): "αντες",
    # neuter
    ("nom", "n", "sg"): "αν",
    ("gen", "n", "sg"): "αντος",
    ("dat", "n", "sg"): "αντι",
    ("acc", "n", "sg"): "αν",
    ("voc", "n", "sg"): "αν",
    ("nom", "n", "pl"): "αντα",
    ("gen", "n", "pl"): "άντων",
    ("dat", "n", "pl"): "ασι(ν)",
    ("acc", "n", "pl"): "αντα",
    ("voc", "n", "pl"): "αντα",
    # feminine: jtauber marks σ-stem α with macron throughout.
    ("nom", "f", "sg"): "ᾱσα",
    ("gen", "f", "sg"): "ᾱ́σης",
    ("dat", "f", "sg"): "ᾱ́σῃ",
    ("acc", "f", "sg"): "ᾱσαν",
    ("voc", "f", "sg"): "ᾱσα",
    ("nom", "f", "pl"): "ᾱσαι",
    ("gen", "f", "pl"): "ᾱσῶν",
    ("dat", "f", "pl"): "ᾱ́σαις",
    ("acc", "f", "pl"): "ᾱ́σᾱς",
    ("voc", "f", "pl"): "ᾱσαι",
}


_AOR_ACTIVE_3RD_END_ACCENTED = {
    ("gen", "m", "pl"),
    ("gen", "n", "pl"),
    ("gen", "f", "pl"),
    ("gen", "f", "sg"),
    ("dat", "f", "sg"),
    ("dat", "f", "pl"),
    ("acc", "f", "pl"),
}


# Aor-2 participle endings (3rd-decl, -ών/-οῦσα/-όν). End-accented
# throughout: the stem accent is always stripped, and the ending carries
# its own acute/circumflex. Used for verbs like λαβών / πεσών / λιπών.
#
# Macron rule: jtauber leaves the final α unmarked except acc_f_pl
# (-οῦσᾱς). The aor-2 fem table differs from the aor-1 table because
# the perispomeni ου̃ already encodes a long stem-vowel, so jtauber does
# not add an additional macron to the final α elsewhere.
_AOR2_ACTIVE_3RD: Dict[tuple, str] = {
    # masculine
    ("nom", "m", "sg"): "ών",
    ("gen", "m", "sg"): "όντος",
    ("dat", "m", "sg"): "όντι",
    ("acc", "m", "sg"): "όντα",
    ("voc", "m", "sg"): "ών",
    ("nom", "m", "pl"): "όντες",
    ("gen", "m", "pl"): "όντων",
    ("dat", "m", "pl"): "οῦσι(ν)",
    ("acc", "m", "pl"): "όντας",
    ("voc", "m", "pl"): "όντες",
    # neuter
    ("nom", "n", "sg"): "όν",
    ("gen", "n", "sg"): "όντος",
    ("dat", "n", "sg"): "όντι",
    ("acc", "n", "sg"): "όν",
    ("voc", "n", "sg"): "όν",
    ("nom", "n", "pl"): "όντα",
    ("gen", "n", "pl"): "όντων",
    ("dat", "n", "pl"): "οῦσι(ν)",
    ("acc", "n", "pl"): "όντα",
    ("voc", "n", "pl"): "όντα",
    # feminine (-οῦσα / -ούσης / -ούσῃ / -οῦσαν / -οῦσα)
    ("nom", "f", "sg"): "οῦσα",
    ("gen", "f", "sg"): "ούσης",
    ("dat", "f", "sg"): "ούσῃ",
    ("acc", "f", "sg"): "οῦσαν",
    ("voc", "f", "sg"): "οῦσα",
    ("nom", "f", "pl"): "οῦσαι",
    ("gen", "f", "pl"): "ουσῶν",
    ("dat", "f", "pl"): "ούσαις",
    ("acc", "f", "pl"): "ούσᾱς",  # jtauber emits explicit macron on α
    ("voc", "f", "pl"): "οῦσαι",
}


# Aorist passive endings (3rd-decl, -θείς/-θεῖσα/-θέν). The stem already
# ends in θ (e.g. λυθ-). All cells are ending-accented.
#
# Macron: jtauber emits the final α as MACRON only in acc_f_pl (-είσᾱς).
# All other final-α cells are unmarked. The σ-stem α is unmarked here
# because the perispomeni on the εῖ already encodes the long syllable.
_AOR_PASSIVE_3RD: Dict[tuple, str] = {
    # masculine
    ("nom", "m", "sg"): "είς",
    ("gen", "m", "sg"): "έντος",
    ("dat", "m", "sg"): "έντι",
    ("acc", "m", "sg"): "έντα",
    ("voc", "m", "sg"): "είς",
    ("nom", "m", "pl"): "έντες",
    ("gen", "m", "pl"): "έντων",
    ("dat", "m", "pl"): "εῖσι(ν)",
    ("acc", "m", "pl"): "έντας",
    ("voc", "m", "pl"): "έντες",
    # neuter
    ("nom", "n", "sg"): "έν",
    ("gen", "n", "sg"): "έντος",
    ("dat", "n", "sg"): "έντι",
    ("acc", "n", "sg"): "έν",
    ("voc", "n", "sg"): "έν",
    ("nom", "n", "pl"): "έντα",
    ("gen", "n", "pl"): "έντων",
    ("dat", "n", "pl"): "εῖσι(ν)",
    ("acc", "n", "pl"): "έντα",
    ("voc", "n", "pl"): "έντα",
    # feminine (-εῖσα/-είσης/-είσῃ/-εῖσαν/-εῖσα)
    ("nom", "f", "sg"): "εῖσα",
    ("gen", "f", "sg"): "είσης",
    ("dat", "f", "sg"): "είσῃ",
    ("acc", "f", "sg"): "εῖσαν",
    ("voc", "f", "sg"): "εῖσα",
    ("nom", "f", "pl"): "εῖσαι",
    ("gen", "f", "pl"): "εισῶν",
    ("dat", "f", "pl"): "είσαις",
    ("acc", "f", "pl"): "είσᾱς",  # jtauber: macron on acc_f_pl α
    ("voc", "f", "pl"): "εῖσαι",
}


# Perfect active endings (3rd-decl, -ώς/-υῖα/-ός). Stem must already
# have its accent stripped. All cells ending-accented.
#
# Macron: jtauber marks the final α with MACRON in BOTH acc_f_pl
# (-υίᾱς) AND gen_f_sg (-υίᾱς). Other feminine final-α cells unmarked.
_PF_ACTIVE_3RD: Dict[tuple, str] = {
    # masculine
    ("nom", "m", "sg"): "ώς",
    ("gen", "m", "sg"): "ότος",
    ("dat", "m", "sg"): "ότι",
    ("acc", "m", "sg"): "ότα",
    ("voc", "m", "sg"): "ώς",
    ("nom", "m", "pl"): "ότες",
    ("gen", "m", "pl"): "ότων",
    ("dat", "m", "pl"): "όσι(ν)",
    ("acc", "m", "pl"): "ότας",
    ("voc", "m", "pl"): "ότες",
    # neuter
    ("nom", "n", "sg"): "ός",
    ("gen", "n", "sg"): "ότος",
    ("dat", "n", "sg"): "ότι",
    ("acc", "n", "sg"): "ός",
    ("voc", "n", "sg"): "ός",
    ("nom", "n", "pl"): "ότα",
    ("gen", "n", "pl"): "ότων",
    ("dat", "n", "pl"): "όσι(ν)",
    ("acc", "n", "pl"): "ότα",
    ("voc", "n", "pl"): "ότα",
    # feminine (-υῖα/-υίας/-υίᾳ/-υῖαν/-υῖα)
    ("nom", "f", "sg"): "υῖα",
    ("gen", "f", "sg"): "υίᾱς",   # jtauber: macron on gen_f_sg α
    ("dat", "f", "sg"): "υίᾳ",
    ("acc", "f", "sg"): "υῖαν",
    ("voc", "f", "sg"): "υῖα",
    ("nom", "f", "pl"): "υῖαι",
    ("gen", "f", "pl"): "υιῶν",
    ("dat", "f", "pl"): "υίαις",
    ("acc", "f", "pl"): "υίᾱς",   # jtauber: macron on acc_f_pl α
    ("voc", "f", "pl"): "υῖαι",
}


# 1st/2nd-declension endings for -μενος/-μενη/-μενον participles (mp,
# middle, future-passive). Each ending is paired with a flag indicating
# whether the FINAL SYLLABLE of the resulting form is "long" for accent
# purposes. Long final → accent retreats to penult (the -μέν- ε); short
# final → accent stays on antepenult (the link vowel).
#
# Greek accent rule for the penult/antepenult choice:
#   - Final syllables containing η, ω, ου, ῃ, ῳ, or a vowel + ν that
#     scans long are LONG.
#   - Final αι and οι in nominative plural are SHORT for accent (the
#     "αι/οι short" rule).
#   - All other -ος / -ε / -ον / -α endings are SHORT.
#
# The stem passed in for present mp / aorist mp must end in the link
# vowel-bearing syllable (e.g. ``λυομεν``, ``λυσαμεν``, ``λελυμεν``).
# The synthesiser inserts the accent at the right position based on the
# pattern (-ομεν / -αμεν / -μεν) and the final-syllable length.
_MENOS_ENDINGS: Dict[tuple, tuple[str, bool]] = {
    # (case, gender, number) -> (ending, final_is_long)
    # masculine
    ("nom", "m", "sg"): ("ος", False),
    ("gen", "m", "sg"): ("ου", True),
    ("dat", "m", "sg"): ("ῳ", True),
    ("acc", "m", "sg"): ("ον", False),
    ("voc", "m", "sg"): ("ε", False),
    ("nom", "m", "pl"): ("οι", False),       # -οι short for accent
    ("gen", "m", "pl"): ("ων", True),
    ("dat", "m", "pl"): ("οις", True),
    ("acc", "m", "pl"): ("ους", True),
    ("voc", "m", "pl"): ("οι", False),
    # neuter
    ("nom", "n", "sg"): ("ον", False),
    ("gen", "n", "sg"): ("ου", True),
    ("dat", "n", "sg"): ("ῳ", True),
    ("acc", "n", "sg"): ("ον", False),
    ("voc", "n", "sg"): ("ον", False),
    ("nom", "n", "pl"): ("α", False),
    ("gen", "n", "pl"): ("ων", True),
    ("dat", "n", "pl"): ("οις", True),
    ("acc", "n", "pl"): ("α", False),
    ("voc", "n", "pl"): ("α", False),
    # feminine (1st-decl, -η/-ης/-ῃ/-ην/-η pattern). jtauber marks the
    # acc_f_pl α with MACRON (-νᾱς); other feminine cells unmarked.
    ("nom", "f", "sg"): ("η", True),
    ("gen", "f", "sg"): ("ης", True),
    ("dat", "f", "sg"): ("ῃ", True),
    ("acc", "f", "sg"): ("ην", True),
    ("voc", "f", "sg"): ("η", True),
    ("nom", "f", "pl"): ("αι", False),       # -αι short for accent
    ("gen", "f", "pl"): ("ων", True),
    ("dat", "f", "pl"): ("αις", True),
    ("acc", "f", "pl"): ("ᾱς", True),         # macron on jtauber's α
    ("voc", "f", "pl"): ("αι", False),
}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def decline_3rd_decl_participle(
    stem: str,
    pattern: str,
) -> Dict[tuple, str]:
    """Return the full case×gender×number declension for a 3rd-decl
    participle off ``stem``.

    ``pattern`` selects the ending table:

      - ``"present_active"`` → -ων/-ουσα/-ον  (stem-accented except gen pl)
      - ``"aorist_active"``  → -ας/-ασα/-αν   (stem-accented except gen pl)
      - ``"future_active"``  → same as present (just on a σ-stem)
      - ``"aorist_passive"`` → -είς/-εῖσα/-έν (always ending-accented)
      - ``"perfect_active"`` → -ώς/-υῖα/-ός   (always ending-accented)

    Returns a dict keyed ``(case, gender, number)``.
    """
    if not stem:
        return {}
    out: Dict[tuple, str] = {}
    if pattern == "present_active":
        table = _PRES_ACTIVE_3RD
        end_accented = _PRES_ACTIVE_3RD_END_ACCENTED
        always_strip = False
    elif pattern == "future_active":
        table = _FUT_ACTIVE_3RD
        end_accented = _PRES_ACTIVE_3RD_END_ACCENTED
        always_strip = False
    elif pattern == "aorist_active":
        table = _AOR_ACTIVE_3RD
        end_accented = _AOR_ACTIVE_3RD_END_ACCENTED
        always_strip = False
    elif pattern == "aor2_active":
        # Aor-2 active participle (-ών/-οῦσα/-όν): always stem-stripped,
        # endings carry their own accents.
        table = _AOR2_ACTIVE_3RD
        end_accented = set(table.keys())
        always_strip = True
    elif pattern == "aorist_passive":
        table = _AOR_PASSIVE_3RD
        end_accented = set(table.keys())
        always_strip = True
    elif pattern == "perfect_active":
        table = _PF_ACTIVE_3RD
        end_accented = set(table.keys())
        always_strip = True
    else:
        return {}
    stem_no_accent = _strip_tonal_accents(stem) if not always_strip else stem
    for cgn, ending in table.items():
        if always_strip or cgn in end_accented:
            out[cgn] = stem_no_accent + ending
        else:
            out[cgn] = stem + ending
    return out


def decline_1st2nd_decl_participle(
    bare_stem: str,
    pattern: str,
) -> Dict[tuple, str]:
    """Return the full case×gender×number declension for a -μενος/-η/-ον
    participle.

    Parameters
    ----------
    bare_stem : str
        The verb root WITHOUT the -μεν- linker, accent-stripped. For
        present mp this is the present stem (e.g. ``λυ`` for λύω). For
        aorist mp this is the σ-stem (``λυσ``). For perfect mp this is
        the perfect stem (``λελυ``). For future passive this is the
        aor-pass stem with no -θη- removed (``λυθ``).

    pattern : str
        One of:
          - ``"present_mp"`` / ``"future_middle"``  — link ``-ομεν-``,
            recessive accent (antepenult on -ό- when final short, else
            penult on -μέ-).
          - ``"aorist_middle"`` — link ``-αμεν-``, recessive accent
            (antepenult on -ά-, penult on -μέ-).
          - ``"perfect_mp"`` — link ``-μεν-``, persistent accent on
            -μέν- always.
          - ``"future_passive"`` — link ``-θησομεν-``, recessive accent
            same as future_middle.

    Returns
    -------
    dict[(case, gender, number), str]
    """
    if not bare_stem:
        return {}
    # Decide linker, antepenult-vowel character, and persistent flag.
    # antepenult_vowel is the vowel that takes the accent for
    # short-final cells (None means "no antepenult shift; always penult").
    if pattern == "present_mp" or pattern == "future_middle":
        link_short = "όμεν"   # antepenult-accented form
        link_long = "ομέν"    # penult-accented form
        persistent = False
    elif pattern == "aorist_middle":
        link_short = "άμεν"
        link_long = "αμέν"
        persistent = False
    elif pattern == "perfect_mp":
        link_short = "μέν"
        link_long = "μέν"
        persistent = True
    elif pattern == "future_passive":
        # -ησομεν-: the aor-pass stem already ends in θ (e.g. λυθ-).
        # Append -ησομεν- with recessive accent: λυθησόμενος / λυθησομένης.
        link_short = "ησόμεν"
        link_long = "ησομέν"
        persistent = False
    else:
        return {}
    out: Dict[tuple, str] = {}
    for cgn, (ending, final_long) in _MENOS_ENDINGS.items():
        if persistent or final_long:
            stem = bare_stem + link_long
        else:
            stem = bare_stem + link_short
        out[cgn] = stem + ending
    return out


# ---------------------------------------------------------------------------
# Per-tense/voice synthesis
# ---------------------------------------------------------------------------


def _emit(
    out: Dict[str, str],
    voice: str,
    tense: str,
    cells: Dict[tuple, str],
) -> None:
    """Convert ``{(case, gender, number): form}`` into the jtauber-style
    flat key shape and emit into ``out``."""
    for (case, gender, number), form in cells.items():
        key = (
            f"{voice}_{tense}_participle_"
            f"{case}_{gender}_{number}"
        )
        out[key] = form


def _present_active(lemma: str, out: Dict[str, str]) -> None:
    pres_stem = _present_stem(lemma)
    if not pres_stem:
        return
    cells = decline_3rd_decl_participle(pres_stem, "present_active")
    _emit(out, "active", "present", cells)


def _present_mp(lemma: str, out: Dict[str, str]) -> None:
    pres_stem = _present_stem(lemma)
    if not pres_stem:
        return
    bare_stem = _strip_tonal_accents(pres_stem)
    cells = decline_1st2nd_decl_participle(bare_stem, "present_mp")
    _emit(out, "middle", "present", cells)


def _future_active(parts: Dict[str, str], lemma: str, out: Dict[str, str]) -> None:
    aor_stem = _resolve_aor_stem(parts, lemma)
    if not aor_stem:
        return
    cells = decline_3rd_decl_participle(aor_stem, "future_active")
    _emit(out, "active", "future", cells)


def _future_middle(parts: Dict[str, str], lemma: str, out: Dict[str, str]) -> None:
    aor_stem = _resolve_aor_stem(parts, lemma)
    if not aor_stem:
        return
    bare_stem = _strip_tonal_accents(aor_stem)
    cells = decline_1st2nd_decl_participle(bare_stem, "future_middle")
    _emit(out, "middle", "future", cells)


def _future_passive(parts: Dict[str, str], out: Dict[str, str]) -> None:
    if "aor_p" not in parts:
        return
    pass_stem = _aor_passive_stem(parts["aor_p"])
    if not pass_stem:
        return
    # Stem already accent-stripped by _aor_passive_stem.
    cells = decline_1st2nd_decl_participle(pass_stem, "future_passive")
    _emit(out, "passive", "future", cells)


def _aorist_active(parts: Dict[str, str], lemma: str, out: Dict[str, str]) -> None:
    aor_stem = _resolve_aor_stem(parts, lemma)
    if not aor_stem:
        return
    cells = decline_3rd_decl_participle(aor_stem, "aorist_active")
    _emit(out, "active", "aorist", cells)


def _aorist_middle(parts: Dict[str, str], lemma: str, out: Dict[str, str]) -> None:
    aor_stem = _resolve_aor_stem(parts, lemma)
    if not aor_stem:
        return
    bare_stem = _strip_tonal_accents(aor_stem)
    cells = decline_1st2nd_decl_participle(bare_stem, "aorist_middle")
    _emit(out, "middle", "aorist", cells)


def _aorist_passive(parts: Dict[str, str], out: Dict[str, str]) -> None:
    if "aor_p" not in parts:
        return
    pass_stem = _aor_passive_stem(parts["aor_p"])
    if not pass_stem:
        return
    cells = decline_3rd_decl_participle(pass_stem, "aorist_passive")
    _emit(out, "passive", "aorist", cells)


def _perfect_active(parts: Dict[str, str], out: Dict[str, str]) -> None:
    if "pf" not in parts:
        return
    pf_stem = _perfect_active_stem(parts["pf"])
    if not pf_stem:
        return
    cells = decline_3rd_decl_participle(pf_stem, "perfect_active")
    _emit(out, "active", "perfect", cells)


def _perfect_mp(parts: Dict[str, str], out: Dict[str, str]) -> None:
    if "pf_mp" not in parts:
        return
    pf_mp_stem = _perfect_mp_stem(parts["pf_mp"])
    if not pf_mp_stem:
        return
    # Perfect-mp accent is persistent on the -μέν- regardless of final
    # syllable length: λελυμένος, λελυμένης both have accent on μέ.
    bare_stem = _strip_tonal_accents(pf_mp_stem)
    cells = decline_1st2nd_decl_participle(bare_stem, "perfect_mp")
    _emit(out, "middle", "perfect", cells)


def _resolve_aor_stem(parts: Dict[str, str], lemma: str) -> Optional[str]:
    """Resolve the sigmatic σ-stem from principal parts. Tries fut first
    (carries the original accent without an augment), then derives from
    aor + lemma. Same logic as ``synth_verb_moods``."""
    aor_stem: Optional[str] = None
    if "fut" in parts:
        aor_stem = _aorist_stem_from_fut(parts["fut"])
    if aor_stem is None and "aor" in parts:
        aor_stem = _aorist_stem_from_lemma_and_aor(lemma, parts["aor"])
    if aor_stem is None:
        return None
    plain = _strip_accents_lower(aor_stem)
    if not plain or plain[-1] not in ("σ", "ψ", "ξ"):
        return None
    return aor_stem


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def synthesize_participles(
    lemma: str,
    principal_parts: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Synthesise participle paradigm cells for a thematic -ω verb.

    Returns a dict ``{paradigm_key: form}`` keyed in the jtauber-
    compatible flat shape ``{voice}_{tense}_participle_{case}_{gender}_{number}``.
    The caller is responsible for merging into existing paradigms;
    this function never decides whether a real cell already exists.

    Synthesis is conservative: tense×voice combos for which the
    required principal part is missing or doesn't fit the regular
    pattern are simply skipped, leaving an empty dict for that combo.
    """
    if not lemma or not is_thematic_omega(lemma):
        return {}
    parts = principal_parts or {}
    out: Dict[str, str] = {}

    # ---- Present-system (uses lemma alone) ----
    _present_active(lemma, out)
    _present_mp(lemma, out)

    # ---- Future system (needs σ-stem from fut or derivable from aor) ----
    _future_active(parts, lemma, out)
    _future_middle(parts, lemma, out)
    _future_passive(parts, out)

    # ---- Aorist system (sigmatic only) ----
    _aorist_active(parts, lemma, out)
    _aorist_middle(parts, lemma, out)
    _aorist_passive(parts, out)

    # ---- Perfect system ----
    _perfect_active(parts, out)
    _perfect_mp(parts, out)

    return out


def synthesize_aor2_participles(
    lemma: str,
    principal_parts: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Synthesise active aor-2 participle declension cells for a verb
    whose ``parts['aor']`` ends in -ον.

    Returns a dict ``{paradigm_key: form}`` covering all
    case×gender×number combinations (m/f/n × sg/pl) for the active
    aor-2 participle, plus the middle aor-2 participle (1st/2nd-decl
    -ομενος pattern on the aor-2 stem). Returns empty when no aor-2
    can be parsed.
    """
    if not lemma:
        return {}
    parts = principal_parts or {}
    # A generic first aorist must not mask a separately supplied active
    # second aorist; a middle-only second aorist is not this template.
    aor_form = parts.get("aor")
    stem = _aor2_stem(aor_form)
    if not stem:
        aor_form = parts.get("aor2")
        stem = _aor2_stem(aor_form)
    if not stem:
        return {}

    out: Dict[str, str] = {}

    # Active aor-2 participle (3rd-decl, -ών/-οῦσα/-όν, end-accented).
    cells = decline_3rd_decl_participle(stem, "aor2_active")
    _emit(out, "active", "aorist", cells)

    # Middle aor-2 participle (1st/2nd-decl, -όμενος/-ομένη/-όμενον,
    # recessive accent like present mp). Uses the same -ομεν- linker.
    bare_stem = _strip_tonal_accents(stem)
    cells_mid = decline_1st2nd_decl_participle(bare_stem, "present_mp")
    _emit(out, "middle", "aorist", cells_mid)

    return out


# Contract verb participles (present-system only).
# The active present participle uses -ῶν/-οῦσα/-οῦν (alpha-contract:
# -ῶν/-ῶσα/-ῶν; ε-contract: -ῶν/-οῦσα/-οῦν; ο-contract: -ῶν/-οῦσα/-οῦν).
# All are end-accented on the contract syllable.

# Alpha-contract active present participle (-ῶν/-ῶσα/-ῶν).
# Macron / breve quantity marks follow jtauber (matched against τιμάω,
# ὁράω): the final α takes BREVE in nom/voc/acc_f_sg (-ῶσᾰ / -ῶσᾰν),
# MACRON in acc_f_pl (-ώσᾱς), and is unmarked elsewhere.
_CONTRACT_ALPHA_ACTIVE_3RD: Dict[tuple, str] = {
    ("nom", "m", "sg"): "ῶν",
    ("gen", "m", "sg"): "ῶντος",
    ("dat", "m", "sg"): "ῶντι",
    ("acc", "m", "sg"): "ῶντα",
    ("voc", "m", "sg"): "ῶν",
    ("nom", "m", "pl"): "ῶντες",
    ("gen", "m", "pl"): "ώντων",
    ("dat", "m", "pl"): "ῶσι(ν)",
    ("acc", "m", "pl"): "ῶντας",
    ("voc", "m", "pl"): "ῶντες",
    ("nom", "n", "sg"): "ῶν",
    ("gen", "n", "sg"): "ῶντος",
    ("dat", "n", "sg"): "ῶντι",
    ("acc", "n", "sg"): "ῶν",
    ("voc", "n", "sg"): "ῶν",
    ("nom", "n", "pl"): "ῶντα",
    ("gen", "n", "pl"): "ώντων",
    ("dat", "n", "pl"): "ῶσι(ν)",
    ("acc", "n", "pl"): "ῶντα",
    ("voc", "n", "pl"): "ῶντα",
    ("nom", "f", "sg"): "ῶσᾰ",
    ("gen", "f", "sg"): "ώσης",
    ("dat", "f", "sg"): "ώσῃ",
    ("acc", "f", "sg"): "ῶσᾰν",
    ("voc", "f", "sg"): "ῶσᾰ",
    ("nom", "f", "pl"): "ῶσαι",
    ("gen", "f", "pl"): "ωσῶν",
    ("dat", "f", "pl"): "ώσαις",
    ("acc", "f", "pl"): "ώσᾱς",
    ("voc", "f", "pl"): "ῶσαι",
}


# Epsilon-contract active present participle (-ῶν/-οῦσα/-οῦν).
# Macron / breve rule (jtauber, matched against ποιέω, φιλέω):
# nom_f_sg / voc_f_sg / acc_f_sg take BREVE on final α (-οῦσᾰ /
# -οῦσᾰν), acc_f_pl takes MACRON (-ούσᾱς). Other cells unmarked.
_CONTRACT_EPSILON_ACTIVE_3RD: Dict[tuple, str] = {
    ("nom", "m", "sg"): "ῶν",
    ("gen", "m", "sg"): "οῦντος",
    ("dat", "m", "sg"): "οῦντι",
    ("acc", "m", "sg"): "οῦντα",
    ("voc", "m", "sg"): "ῶν",
    ("nom", "m", "pl"): "οῦντες",
    ("gen", "m", "pl"): "ούντων",
    ("dat", "m", "pl"): "οῦσι(ν)",
    ("acc", "m", "pl"): "οῦντας",
    ("voc", "m", "pl"): "οῦντες",
    ("nom", "n", "sg"): "οῦν",
    ("gen", "n", "sg"): "οῦντος",
    ("dat", "n", "sg"): "οῦντι",
    ("acc", "n", "sg"): "οῦν",
    ("voc", "n", "sg"): "οῦν",
    ("nom", "n", "pl"): "οῦντα",
    ("gen", "n", "pl"): "ούντων",
    ("dat", "n", "pl"): "οῦσι(ν)",
    ("acc", "n", "pl"): "οῦντα",
    ("voc", "n", "pl"): "οῦντα",
    ("nom", "f", "sg"): "οῦσᾰ",
    ("gen", "f", "sg"): "ούσης",
    ("dat", "f", "sg"): "ούσῃ",
    ("acc", "f", "sg"): "οῦσᾰν",
    ("voc", "f", "sg"): "οῦσᾰ",
    ("nom", "f", "pl"): "οῦσαι",
    ("gen", "f", "pl"): "ουσῶν",
    ("dat", "f", "pl"): "ούσαις",
    ("acc", "f", "pl"): "ούσᾱς",
    ("voc", "f", "pl"): "οῦσαι",
}


# Omicron-contract active present participle (-ῶν/-οῦσα/-οῦν).
# Same shape as ε-contract but masculine forms differ in the dat-pl
# circumflex distribution. We share the table since jtauber's δηλόω
# matches the epsilon pattern.
_CONTRACT_OMICRON_ACTIVE_3RD: Dict[tuple, str] = dict(
    _CONTRACT_EPSILON_ACTIVE_3RD
)


def _decline_contract_active_participle(
    bare_stem: str, contract_class_: str
) -> Dict[tuple, str]:
    """Decline the active present participle of a contract verb."""
    if not bare_stem:
        return {}
    if contract_class_ == "alpha":
        table = _CONTRACT_ALPHA_ACTIVE_3RD
    elif contract_class_ == "epsilon":
        table = _CONTRACT_EPSILON_ACTIVE_3RD
    elif contract_class_ == "omicron":
        table = _CONTRACT_OMICRON_ACTIVE_3RD
    else:
        return {}
    out: Dict[tuple, str] = {}
    for cgn, ending in table.items():
        out[cgn] = bare_stem + ending
    return out


def synthesize_contract_participles(
    lemma: str,
    principal_parts: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Synthesise the present-system participle declension for a contract
    verb (-άω / -έω / -όω).

    Returns a dict ``{paradigm_key: form}`` covering active + middle
    present participles for every case×gender×number combination.

    Aorist / future / perfect participles of contracts can be filled by
    the regular sigmatic synthesis since the contract vowel lengthens
    before the σ; we do not duplicate that here. Returns empty when the
    lemma is not a contract.
    """
    if not lemma:
        return {}
    # Local lazy import to avoid a circular dep with synth_verb_moods.
    # We add the build/ dir to sys.path here so this works both when
    # called from build_grc_verb_paradigms (which prepends build/ to
    # sys.path) and when called from tests that load via importlib.
    import sys
    from pathlib import Path
    _build_dir = str(Path(__file__).resolve().parent)
    if _build_dir not in sys.path:
        sys.path.insert(0, _build_dir)
    from synth_verb_moods import contract_class as _contract_class
    from synth_verb_moods import _contract_bare_stem as _contract_bare
    cls = _contract_class(lemma)
    if cls is None:
        return {}
    bare = _contract_bare(lemma)
    if bare is None:
        return {}
    bare_no_accent = _strip_tonal_accents(bare)

    out: Dict[str, str] = {}

    # Active present participle.
    cells_act = _decline_contract_active_participle(bare_no_accent, cls)
    _emit(out, "active", "present", cells_act)

    # Middle present participle: jtauber uses a mix of contracted
    # (δηλούμενος, ποιούμενος, τιμώμενος) and uncontracted
    # (δηλοόμενος, ποιεόμενος, τιμαόμενος) forms across cells. We follow
    # jtauber's dominant pattern for each contract class:
    #   * Alpha: contracted everywhere (-ώμεν / -ωμέν link).
    #   * Epsilon: contracted everywhere except a small set of cells
    #     jtauber consistently leaves uncontracted (voc_m_sg, acc_f_pl,
    #     dat_f_pl). We skip those cells so corpus / Wiktionary fills
    #     them with the uncontracted form.
    #   * Omicron: jtauber's pattern is too irregular cell-by-cell to
    #     synthesise safely; we emit only the cells where contraction
    #     is reliable (gen/dat sg + nom/acc sg + nom/voc m pl) and leave
    #     the rest empty.
    if cls == "alpha":
        link_short = "ώμεν"   # antepenult-accented
        link_long = "ωμέν"    # penult-accented
        # jtauber's τιμάω uses UNCONTRACTED forms in most feminine
        # cells (-αομέν- / -αόμεν-) but contracted forms elsewhere.
        # Skip the cells where jtauber consistently uses uncontracted
        # forms so the corpus / Wiktionary cells fill them in.
        skip_cells_alpha = {
            ("voc", "m", "sg"),    # τιμαόμενε
            ("gen", "f", "sg"),    # τιμαομένης
            ("dat", "f", "sg"),    # τιμαομένῃ
            ("acc", "f", "sg"),    # τιμαομένην
            ("nom", "f", "pl"),    # τιμαόμεναι
            ("voc", "f", "pl"),    # τιμαόμεναι
            ("dat", "f", "pl"),    # τιμαομέναις
            ("acc", "f", "pl"),    # τιμαομένᾱς
        }
        cells_mid: Dict[tuple, str] = {}
        for cgn, (ending, final_long) in _MENOS_ENDINGS.items():
            if cgn in skip_cells_alpha:
                continue
            link = link_long if final_long else link_short
            cells_mid[cgn] = bare_no_accent + link + ending
        _emit(out, "middle", "present", cells_mid)
    elif cls == "epsilon":
        # ε-contract: link is -ούμεν- contracted (from -εομεν-).
        # Antepenult acute on ού (short final) → -ούμεν- pattern.
        # Penult acute on έ (long final) → -ουμέν- pattern.
        link_short = "ούμεν"  # antepenult-accented
        link_long = "ουμέν"   # penult-accented
        # Cells jtauber consistently leaves UNCONTRACTED for ε-contracts
        # (vs the dominant contracted pattern). We skip them so the
        # corpus / Wiktionary cells fill in with -εο- forms intact.
        skip_cells = {
            ("voc", "m", "sg"),   # ποιεόμενε
            ("acc", "f", "pl"),   # ποιεομένᾱς
            ("dat", "f", "pl"),   # ποιεομέναις
        }
        cells_mid_e: Dict[tuple, str] = {}
        for cgn, (ending, final_long) in _MENOS_ENDINGS.items():
            if cgn in skip_cells:
                continue
            link = link_long if final_long else link_short
            cells_mid_e[cgn] = bare_no_accent + link + ending
        _emit(out, "middle", "present", cells_mid_e)
    # ο-contract middle participle is intentionally not synthesised.

    return out


if __name__ == "__main__":
    # Smoke test for a few canonical verbs.
    samples = [
        ("λύω", {"fut": "λύσω", "aor": "ἔλυσα", "pf": "λέλυκα",
                  "pf_mp": "λέλυμαι", "aor_p": "ἐλύθην"}),
        ("παιδεύω", {"fut": "παιδεύσω", "aor": "ἐπαίδευσα"}),
        ("γράφω", {"aor": "ἔγραψα", "pf": "γέγραφα"}),
        ("τιμάω", {"fut": "τιμήσω"}),     # contract; should be skipped
        ("τίθημι", {}),                     # athematic; skipped
        ("λείπω", {"aor": "ἔλιπον"}),       # aor-2; aorist branch skipped
    ]
    for lemma, parts in samples:
        out = synthesize_participles(lemma, parts)
        print(f"=== {lemma} ({len(out)} cells, parts={parts}) ===")
        for k in sorted(out)[:6]:
            print(f"  {k} = {out[k]}")
        if len(out) > 6:
            print(f"  ... and {len(out) - 6} more")
        print()
