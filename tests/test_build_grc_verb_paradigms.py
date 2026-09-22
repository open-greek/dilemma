"""Tests for ``build/build_grc_verb_paradigms.py``.

Focus: the augment-preference rule in :func:`pick_best_form` for past-
indicative cells (aorist / imperfect / pluperfect indicative). Multiple
corpora regularly attest both Homeric un-augmented forms (``λῦσε``,
``λῦσαν``) and Attic augmented forms (``ἔλυσε``, ``ἔλυσαν``) for the same
cell. Without the augment preference, the ``-len(f)`` tie-breaker silently
picks the shorter, un-augmented variant for the canonical Attic slice -
which is wrong for every classical / koine / modern reader expecting
augmented past indicatives.

Run with:

    python -m pytest tests/test_build_grc_verb_paradigms.py -x -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "build" / "build_grc_verb_paradigms.py"


@pytest.fixture(scope="module")
def b():
    """Load build/build_grc_verb_paradigms.py as a module without
    package install (mirrors the pattern in test_synth_verb_moods.py)."""
    spec = importlib.util.spec_from_file_location(
        "build_grc_verb_paradigms", MODULE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# is_past_indicative_key
# ---------------------------------------------------------------------------


class TestIsPastIndicativeKey:
    """The augment-preference rule only fires on past-tense indicative
    cells. Non-indicative moods and non-past tenses must be left alone."""

    @pytest.mark.parametrize("key", [
        "active_aorist_indicative_1sg",
        "active_aorist_indicative_2sg",
        "active_aorist_indicative_3sg",
        "active_aorist_indicative_1pl",
        "active_aorist_indicative_2pl",
        "active_aorist_indicative_3pl",
        "middle_aorist_indicative_3sg",
        "passive_aorist_indicative_1pl",
        "active_imperfect_indicative_3sg",
        "active_imperfect_indicative_3pl",
        "middle_imperfect_indicative_2sg",
        "active_pluperfect_indicative_1sg",
        "middle_pluperfect_indicative_3pl",
    ])
    def test_past_indicative_keys_match(self, b, key):
        assert b.is_past_indicative_key(key)

    @pytest.mark.parametrize("key", [
        "active_present_indicative_3sg",
        "active_present_indicative_3pl",
        "middle_perfect_indicative_3sg",
        "active_future_indicative_1sg",
        "middle_future_indicative_3pl",
        "active_aorist_subjunctive_3sg",
        "active_aorist_optative_3pl",
        "active_aorist_imperative_2sg",
        "active_aorist_infinitive",
        "active_aorist_participle_nom_m_sg",
        "middle_imperfect_subjunctive_1sg",  # nonsense, but mood test
        "",
    ])
    def test_non_past_or_non_indicative_skipped(self, b, key):
        assert not b.is_past_indicative_key(key)


# ---------------------------------------------------------------------------
# has_augment
# ---------------------------------------------------------------------------


class TestHasAugment:
    """Augment detection wraps the dilemma.morph_diff helpers."""

    @pytest.mark.parametrize("form,lemma", [
        ("ἔλυσα", "λύω"),
        ("ἔλυσας", "λύω"),
        ("ἔλυσε", "λύω"),
        ("ἐλύσαμεν", "λύω"),
        ("ἐλύσατε", "λύω"),
        ("ἔλυσαν", "λύω"),
        ("ἔπαυσε", "παύω"),
        ("ἔπαυσαν", "παύω"),
        ("ἔγραψε", "γράφω"),
        ("ἔπεισε", "πείθω"),
        ("ἔλυε", "λύω"),
        ("ἔλυον", "λύω"),
        # Temporal augment for vowel-initial stems
        ("ἤκουσα", "ἀκούω"),
        ("ἤθελον", "ἐθέλω"),  # ε -> η
    ])
    def test_augmented_forms_detected(self, b, form, lemma):
        assert b.has_augment(form, lemma), f"expected augment in {form} <- {lemma}"

    @pytest.mark.parametrize("form,lemma", [
        # Un-augmented past-indicative variants (Homeric / poetic)
        ("λῦσε", "λύω"),
        ("λῦσαν", "λύω"),
        ("λῦε", "λύω"),
        ("λύον", "λύω"),
        ("παῦσε", "παύω"),
        ("παῦσαν", "παύω"),
        # Present forms (no augment by definition)
        ("λύω", "λύω"),
        ("λύει", "λύω"),
        ("ἀκούω", "ἀκούω"),
        ("ἐθέλω", "ἐθέλω"),
    ])
    def test_unaugmented_forms_detected(self, b, form, lemma):
        assert not b.has_augment(form, lemma), \
            f"unexpected augment in {form} <- {lemma}"

    def test_empty_inputs_safe(self, b):
        assert not b.has_augment("", "λύω")
        assert not b.has_augment("ἔλυσε", "")
        assert not b.has_augment("", "")


# ---------------------------------------------------------------------------
# pick_best_form: augment preference for past-indicative cells
# ---------------------------------------------------------------------------


class TestPickBestFormAugmentPreference:
    """The original bug: glaux corpus emits both ``λῦσε`` and ``ἔλυσε``
    once each for λύω aor-act-ind 3sg, and the un-augmented ``λῦσε``
    won via the ``-len(f)`` tie-breaker. The fix: when ``key`` names a
    past-indicative cell and ``lemma`` is provided, augment-bearing
    variants outrank un-augmented variants regardless of length."""

    # --- λύω aor-act-ind: full bug regression suite ------------------------

    def test_lyo_aor_act_ind_1sg(self, b):
        forms = ["ἔλυσα", "ἔλυσά"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_1sg", lemma="λύω")
        assert got == "ἔλυσα"

    def test_lyo_aor_act_ind_2sg(self, b):
        forms = ["ἔλυσας", "ἔλυσάς"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_2sg", lemma="λύω")
        assert got in {"ἔλυσας", "ἔλυσάς"}

    def test_lyo_aor_act_ind_3sg_augmented_wins(self, b):
        # The reported bug: un-augmented λῦσε used to win here.
        forms = ["ἔλυσ’", "λῦσεν", "ἔλυσεν", "λῦσε", "ἔλυσε",
                 "ἔλυσέ", "ἔλυσέν"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3sg", lemma="λύω")
        assert b.has_augment(got, "λύω"), \
            f"3sg pick {got} lacks augment"
        # Specifically must not be λῦσε / λῦσεν.
        assert got not in {"λῦσε", "λῦσεν"}

    def test_lyo_aor_act_ind_3sg_no_augment_in_pool_ok(self, b):
        # If only un-augmented variants are attested, we still pick
        # something rather than returning None.
        forms = ["λῦσε", "λῦσεν"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3sg", lemma="λύω")
        assert got in {"λῦσε", "λῦσεν"}

    def test_lyo_aor_act_ind_1pl(self, b):
        forms = ["ἐλύσαμεν"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_1pl", lemma="λύω")
        assert got == "ἐλύσαμεν"

    def test_lyo_aor_act_ind_2pl(self, b):
        forms = ["ἐλύσατε"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_2pl", lemma="λύω")
        assert got == "ἐλύσατε"

    def test_lyo_aor_act_ind_3pl_augmented_wins(self, b):
        # The reported bug: λῦσαν used to win here.
        forms = ["ἔλυσαν", "λῦσαν", "ἔλυσάν", "λύσαν"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3pl", lemma="λύω")
        assert b.has_augment(got, "λύω"), \
            f"3pl pick {got} lacks augment"
        assert got not in {"λῦσαν", "λύσαν"}

    # --- λύω aor-mid-ind: should already be augmented ---------------------

    def test_lyo_aor_mid_ind_3sg(self, b):
        forms = ["ἐλύσατο", "λύσατο"]
        got = b.pick_best_form(
            forms, key="middle_aorist_indicative_3sg", lemma="λύω")
        assert b.has_augment(got, "λύω")

    def test_lyo_aor_mid_ind_3pl(self, b):
        forms = ["ἐλύσαντο", "λύσαντο"]
        got = b.pick_best_form(
            forms, key="middle_aorist_indicative_3pl", lemma="λύω")
        assert b.has_augment(got, "λύω")

    # --- λύω imperfect-ind: same bug class --------------------------------

    def test_lyo_imperfect_act_ind_3sg(self, b):
        # λῦε / λύε / λύεν all un-augmented; ἔλυεν / ἔλυέν are right.
        forms = ["λύεσκε", "λῦε", "ἔλυεν", "λύε", "ἔλυέν", "λύεν"]
        got = b.pick_best_form(
            forms, key="active_imperfect_indicative_3sg", lemma="λύω")
        assert b.has_augment(got, "λύω")
        assert got not in {"λῦε", "λύε", "λύεν"}

    def test_lyo_imperfect_act_ind_3pl(self, b):
        forms = ["λύον", "ἔλυον"]
        got = b.pick_best_form(
            forms, key="active_imperfect_indicative_3pl", lemma="λύω")
        assert got == "ἔλυον"

    # --- παύω aor-act-ind: same bug class ---------------------------------

    def test_pauo_aor_act_ind_3sg_augmented_wins(self, b):
        forms = ["παῦσε", "ἔπαυσε"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3sg", lemma="παύω")
        assert got == "ἔπαυσε"

    def test_pauo_aor_act_ind_3pl_augmented_wins(self, b):
        forms = ["παῦσαν", "ἔπαυσαν"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3pl", lemma="παύω")
        assert got == "ἔπαυσαν"

    # --- πείθω aor-act-ind: bug-adjacent (3sg sometimes correct) ----------

    def test_peitho_aor_act_ind_3sg_augmented_wins(self, b):
        forms = ["πεῖσε", "ἔπεισε"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3sg", lemma="πείθω")
        assert got == "ἔπεισε"

    # --- γράφω aor-act-ind: bug-adjacent ---------------------------------

    def test_grapho_aor_act_ind_3sg_augmented_wins(self, b):
        forms = ["γράψε", "ἔγραψε"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3sg", lemma="γράφω")
        assert got == "ἔγραψε"

    def test_grapho_aor_act_ind_3pl_augmented_wins(self, b):
        forms = ["γράψαν", "ἔγραψαν"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3pl", lemma="γράφω")
        assert got == "ἔγραψαν"

    # --- λούω aor-act-ind: same bug class ---------------------------------

    def test_louo_aor_act_ind_3sg_augmented_wins(self, b):
        forms = ["λοῦσε", "ἔλουσε"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3sg", lemma="λούω")
        assert got == "ἔλουσε"

    def test_louo_aor_act_ind_3pl_augmented_wins(self, b):
        forms = ["λοῦσαν", "ἔλουσαν"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3pl", lemma="λούω")
        assert got == "ἔλουσαν"

    # --- Vowel-initial stem: temporal augment ----------------------------

    def test_akouo_aor_act_ind_3sg_temporal_augment(self, b):
        # ἀκούω -> ἤκουσε (temporal augment α -> η).
        forms = ["ἄκουσε", "ἤκουσε"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3sg", lemma="ἀκούω")
        assert got == "ἤκουσε"


# ---------------------------------------------------------------------------
# pick_best_form: behaviour outside past-indicative is unchanged
# ---------------------------------------------------------------------------


class TestPickBestFormPreservesNonPast:
    """The augment preference must NOT activate for non-past or non-
    indicative cells. Subjunctive / optative / imperative / infinitive /
    participle cells never carry augment, so we keep the original
    polytonic / length / alphabetical tie-breaker chain."""

    def test_present_indicative_unchanged(self, b):
        # No augment-preference: shortest variant still wins on tie.
        forms = ["λύει", "λύεις"]
        got = b.pick_best_form(
            forms, key="active_present_indicative_3sg", lemma="λύω")
        # Before the fix this would pick by polytonic / length /
        # alphabetical. After the fix we did not change this branch.
        assert got in forms

    def test_aorist_subjunctive_unchanged(self, b):
        # Subjunctive never has augment.
        forms = ["λύσῃ", "λύσῃς"]
        got = b.pick_best_form(
            forms, key="active_aorist_subjunctive_3sg", lemma="λύω")
        assert got in forms

    def test_aorist_imperative_unchanged(self, b):
        forms = ["λῦσον"]
        got = b.pick_best_form(
            forms, key="active_aorist_imperative_2sg", lemma="λύω")
        assert got == "λῦσον"

    def test_aorist_infinitive_unchanged(self, b):
        forms = ["λῦσαι"]
        got = b.pick_best_form(
            forms, key="active_aorist_infinitive", lemma="λύω")
        assert got == "λῦσαι"

    def test_aorist_participle_unchanged(self, b):
        forms = ["λύσας"]
        got = b.pick_best_form(
            forms, key="active_aorist_participle_nom_m_sg", lemma="λύω")
        assert got == "λύσας"

    def test_no_lemma_falls_back_to_length(self, b):
        # Without a lemma (a dialect slice) the augment-preference rule is
        # disabled, and corpus frequency decides: ἔλυσε is the spelling the
        # corpora attest. Length only breaks a tie, as below.
        forms = ["λῦσε", "ἔλυσε"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3sg")
        assert got == "ἔλυσε"
        # Forms no corpus attests still fall back to the shorter spelling.
        assert b.pick_best_form(
            ["ζζυσεν", "ζζυσε"], key="active_aorist_indicative_3sg"
        ) == "ζζυσε"

    def test_no_key_falls_back_to_length(self, b):
        # No key either: corpus frequency still ranks the attested spelling
        # first, and length decides only among forms no corpus attests.
        assert b.pick_best_form(["λῦσε", "ἔλυσε"]) == "ἔλυσε"
        assert b.pick_best_form(["ζζυσεν", "ζζυσε"]) == "ζζυσε"

    def test_count_still_dominates(self, b):
        # If one variant is attested twice and the other once, the
        # higher-count variant wins regardless of augment status. Real
        # corpus quality > our augment heuristic.
        forms = ["λῦσε", "λῦσε", "ἔλυσε"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3sg", lemma="λύω")
        assert got == "λῦσε"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """The old call signature ``pick_best_form(forms)`` must still work
    so any in-repo / out-of-repo callers that haven't migrated still
    behave identically to the pre-fix code path."""

    def test_no_kwargs_old_signature(self, b):
        forms = ["λύει", "λύεις"]
        got = b.pick_best_form(forms)
        assert got in forms

    def test_set_input(self, b):
        forms = {"ἔλυσα", "ἔλυσά"}
        got = b.pick_best_form(forms, key="active_aorist_indicative_1sg",
                               lemma="λύω")
        assert got in forms

    def test_empty_returns_none(self, b):
        assert b.pick_best_form([]) is None
        assert b.pick_best_form(set()) is None


# ---------------------------------------------------------------------------
# is_crasis_form
# ---------------------------------------------------------------------------


class TestIsCrasisForm:
    """Crasis forms (καί + verb -> κἄβλεψας, etc.) leak into glaux's
    indicative-active-aorist-2sg slot for βλέπω because glaux has no
    sandhi axis. Detection: a consonant-initial word with a breathing
    mark on its second base letter is crasis. Native verbs never put
    breathings on internal vowels.
    """

    @pytest.mark.parametrize("form", [
        "κἄβλεψας",   # καί + ἔβλεψας: the reported bug for βλέπω
        "κἀγώ",       # καί + ἐγώ
        "κἀκείνων",   # καί + ἐκείνων
        "χἠμεῖς",     # καί + ἡμεῖς (κ aspirated to χ before rough breathing)
        "τοὔνομα",    # τό + ὄνομα: breathing on second vowel of ου-diphthong
        "τἀνδρός",    # τοῦ + ἀνδρός
        "τἆλλα",      # τὰ + ἄλλα: breathing + circumflex
    ])
    def test_crasis_forms_detected(self, b, form):
        assert b.is_crasis_form(form), f"crasis not detected: {form}"

    @pytest.mark.parametrize("form", [
        # Native consonant-initial verbs: no breathing on second letter
        "κάμνω",      # κ + α + acute
        "κρίνω",      # κ + ρ (consonant cluster, no breathing)
        "γράφω", "γράψω", "ἔγραψα",
        "λέγω", "ἔλεξα", "λύω", "λύει",
        "παύω", "ἔπαυσε", "παύσομαι",
        "βλέπω", "βλέψω", "ἔβλεψα",
        "τρέχω", "πέμπω", "δίδωμι",
        "φέρω", "πείθω", "γίγνομαι",
        # Vowel-initial native verbs: breathing on first letter is fine
        "ἀκούω", "ἤκουσα", "ἐθέλω", "ἤθελον",
        "εἰμί", "εἶδον", "ἦν",
        # Diphthong-initial: breathing on second vowel of diphthong is
        # at base position 0/1 but the FIRST base char is a vowel.
        "αὐτός", "εὐχή", "οὐρανός",
        # Verbs with augment on diphthong (ηὐ-, ηὐχόμην, ηὐλόγει)
        "ηὐχόμην", "ηὐλόγει",
    ])
    def test_native_forms_pass(self, b, form):
        assert not b.is_crasis_form(form), \
            f"native form falsely flagged as crasis: {form}"

    def test_empty_form_not_crasis(self, b):
        assert not b.is_crasis_form("")

    def test_single_letter_not_crasis(self, b):
        # Too short to have crasis structure.
        assert not b.is_crasis_form("ἁ")
        assert not b.is_crasis_form("κ")


# ---------------------------------------------------------------------------
# is_homeric_iterative_imperfect
# ---------------------------------------------------------------------------


class TestIsHomericIterativeImperfect:
    """Homeric iterative imperfect inserts -σκ- between the present
    stem and a thematic personal ending (παύεσκον, ἔσκε, ἀμφιέπεσκεν).
    Verbs whose lemma natively ends in -σκω (διδάσκω, γιγνώσκω,
    εὑρίσκω) carry σκ in their present stem and must NOT be flagged.
    """

    @pytest.mark.parametrize("form,lemma,key", [
        # The reported bug: παύεσκον leaking into παύω 1sg active impf.
        ("παύεσκον", "παύω", "active_imperfect_indicative_1sg"),
        ("παύεσκες", "παύω", "active_imperfect_indicative_2sg"),
        ("παύεσκε", "παύω", "active_imperfect_indicative_3sg"),
        ("παύεσκεν", "παύω", "active_imperfect_indicative_3sg"),
        # Other Homeric iteratives in the corpus
        ("κλαίεσκεν", "κλαίω", "active_imperfect_indicative_3sg"),
        ("πέρθεσκον", "πέρθω", "active_imperfect_indicative_3pl"),
        ("ἄγεσκον", "ἄγω", "active_imperfect_indicative_3pl"),
        ("τέμνεσκεν", "τέμνω", "active_imperfect_indicative_3sg"),
        ("ἔσκε", "εἰμί", "active_imperfect_indicative_3sg"),
        ("ἔσκεν", "εἰμί", "active_imperfect_indicative_3sg"),
        # Middle / passive iteratives
        ("παυέσκετο", "παύω", "middle_imperfect_indicative_3sg"),
        ("παυεσκόμην", "παύω", "middle_imperfect_indicative_1sg"),
        ("παυέσκοντο", "παύω", "middle_imperfect_indicative_3pl"),
        # Long-vowel iterative variants -ασκον / -οσκον
        ("γοάασκεν", "γοάω", "active_imperfect_indicative_3sg"),
        ("βοάασκεν", "βοάω", "active_imperfect_indicative_3sg"),
    ])
    def test_iterative_detected(self, b, form, lemma, key):
        assert b.is_homeric_iterative_imperfect(form, lemma, key), \
            f"iterative not detected: {form} ({lemma}, {key})"

    @pytest.mark.parametrize("form,lemma,key", [
        # Native -σκω lemmas: σκ is part of the present stem, not iterative
        ("ἐδίδασκον", "διδάσκω", "active_imperfect_indicative_1sg"),
        ("ἐγίγνωσκον", "γιγνώσκω", "active_imperfect_indicative_1sg"),
        ("ηὕρισκον", "εὑρίσκω", "active_imperfect_indicative_1sg"),
        ("ἔβοσκον", "βόσκω", "active_imperfect_indicative_1sg"),
        ("ἔπασχον", "πάσχω", "active_imperfect_indicative_1sg"),
        # Non-imperfect cells must never be flagged, even with -σκ-
        ("παύεσκον", "παύω", "active_aorist_indicative_1sg"),
        ("παύεσκον", "παύω", "active_present_indicative_1sg"),
        ("διδασκόμενον", "διδάσκω",
         "middle_present_participle_acc_m_sg"),
        # Forms without iterative shape
        ("ἔπαυον", "παύω", "active_imperfect_indicative_1sg"),
        ("ἔλυον", "λύω", "active_imperfect_indicative_1sg"),
        ("ἔγραφον", "γράφω", "active_imperfect_indicative_1sg"),
    ])
    def test_non_iterative_pass(self, b, form, lemma, key):
        assert not b.is_homeric_iterative_imperfect(form, lemma, key), \
            f"non-iterative falsely flagged: {form} ({lemma}, {key})"

    def test_empty_inputs_safe(self, b):
        assert not b.is_homeric_iterative_imperfect(
            "", "παύω", "active_imperfect_indicative_1sg")
        assert not b.is_homeric_iterative_imperfect(
            "παύεσκον", "", "active_imperfect_indicative_1sg")
        assert not b.is_homeric_iterative_imperfect(
            "παύεσκον", "παύω", "")
        assert not b.is_homeric_iterative_imperfect(
            "παύεσκον", "παύω", None)


# ---------------------------------------------------------------------------
# is_homeric_unaugmented_past_indicative
# ---------------------------------------------------------------------------


class TestIsHomericUnaugmentedPastIndicative:
    """Past-indicative cells require the augment in Attic. Unaugmented
    surface forms are Homeric / Epic variants and should be routed to
    the dialect slice rather than the canonical paradigm.
    """

    @pytest.mark.parametrize("form,lemma,key", [
        # The reported bug: λυόμην leaking into λύω middle impf 1sg
        ("λυόμην", "λύω", "middle_imperfect_indicative_1sg"),
        ("λύετο", "λύω", "middle_imperfect_indicative_3sg"),
        ("λύοντο", "λύω", "middle_imperfect_indicative_3pl"),
        ("λύον", "λύω", "active_imperfect_indicative_3pl"),
        ("λῦσε", "λύω", "active_aorist_indicative_3sg"),
        ("λῦσαν", "λύω", "active_aorist_indicative_3pl"),
        # γράφω, παύω equivalents
        ("γράψε", "γράφω", "active_aorist_indicative_3sg"),
        ("παῦσε", "παύω", "active_aorist_indicative_3sg"),
        ("παύοντο", "παύω", "middle_imperfect_indicative_3pl"),
    ])
    def test_unaugmented_detected(self, b, form, lemma, key):
        assert b.is_homeric_unaugmented_past_indicative(
            form, lemma, key), \
            f"unaugmented not detected: {form} ({lemma}, {key})"

    @pytest.mark.parametrize("form,lemma,key", [
        # Augmented forms must never be flagged
        ("ἐλυόμην", "λύω", "middle_imperfect_indicative_1sg"),
        ("ἔλυσε", "λύω", "active_aorist_indicative_3sg"),
        ("ἔλυσαν", "λύω", "active_aorist_indicative_3pl"),
        ("ἔλυον", "λύω", "active_imperfect_indicative_3pl"),
        ("ἤκουσε", "ἀκούω", "active_aorist_indicative_3sg"),
        ("ἤθελον", "ἐθέλω", "active_imperfect_indicative_1sg"),
        # Non-past-indicative cells: detector must be off
        ("λύσῃ", "λύω", "active_aorist_subjunctive_3sg"),
        ("λῦσον", "λύω", "active_aorist_imperative_2sg"),
        ("λῦσαι", "λύω", "active_aorist_infinitive"),
        ("λύω", "λύω", "active_present_indicative_1sg"),
        # Long-vowel-initial lemmas (η, ω): augment is invisible, skip
        ("ηὕρισκον", "εὑρίσκω", "active_imperfect_indicative_1sg"),
        # Prefixed verbs: lemma starts with ε- (ἐκ-, ἐν-, ἐπι-, ἐξ-,
        # etc.), augment is internal between prefix and root. The
        # syllabic-augment detector only spots word-initial augments,
        # so it sees these as unaugmented; we explicitly skip ε-
        # prefixed lemmas with ε-prefixed forms to avoid clobbering
        # legitimate compound-verb past indicatives.
        ("ἐξέμολεν", "ἐκμολεῖν", "active_aorist_indicative_3sg"),
        ("ἐξεμόλομεν", "ἐκμολεῖν", "active_aorist_indicative_1pl"),
        ("ἐνεκάλεσε", "ἐγκαλέω", "active_aorist_indicative_3sg"),
        ("ἐπέγραψε", "ἐπιγράφω", "active_aorist_indicative_3sg"),
    ])
    def test_augmented_or_non_past_pass(self, b, form, lemma, key):
        assert not b.is_homeric_unaugmented_past_indicative(
            form, lemma, key), \
            f"falsely flagged: {form} ({lemma}, {key})"

    def test_empty_inputs_safe(self, b):
        assert not b.is_homeric_unaugmented_past_indicative(
            "", "λύω", "active_aorist_indicative_3sg")
        assert not b.is_homeric_unaugmented_past_indicative(
            "λῦσε", "", "active_aorist_indicative_3sg")
        assert not b.is_homeric_unaugmented_past_indicative(
            "λῦσε", "λύω", None)


# ---------------------------------------------------------------------------
# is_homeric_root_aorist_passive
# ---------------------------------------------------------------------------


class TestIsHomericRootAoristPassive:
    """The Homeric / Epic root-aorist used middle-voice personal endings
    (-μην / -σο / -το / -μεθα / -σθε / -ντο) on the bare verb root in
    passive function. Glaux tags these identically to the Attic
    1st-aorist-passive cells, so without filtering we ship ἐλύμην /
    ἔλυντο in the slot where ἐλύθην / ἐλύθησαν belongs.
    """

    @pytest.mark.parametrize("form,lemma,key", [
        # The reported bug for λύω passive aorist
        ("ἐλύμην", "λύω", "passive_aorist_indicative_1sg"),
        ("ἔλυντο", "λύω", "passive_aorist_indicative_3pl"),
        ("λύμην", "λύω", "passive_aorist_indicative_1sg"),
        ("λύτο", "λύω", "passive_aorist_indicative_3sg"),
        ("λύντο", "λύω", "passive_aorist_indicative_3pl"),
        ("λῦτο", "λύω", "passive_aorist_indicative_3sg"),
    ])
    def test_root_aorist_detected(self, b, form, lemma, key):
        assert b.is_homeric_root_aorist_passive(form, lemma, key), \
            f"root-aorist not detected: {form} ({lemma}, {key})"

    @pytest.mark.parametrize("form,lemma,key", [
        # Attic 1st-aorist-passive: -θη- formant + active endings
        ("ἐλύθην", "λύω", "passive_aorist_indicative_1sg"),
        ("ἐλύθης", "λύω", "passive_aorist_indicative_2sg"),
        ("ἐλύθη", "λύω", "passive_aorist_indicative_3sg"),
        ("ἐλύθημεν", "λύω", "passive_aorist_indicative_1pl"),
        ("ἐλύθητε", "λύω", "passive_aorist_indicative_2pl"),
        ("ἐλύθησαν", "λύω", "passive_aorist_indicative_3pl"),
        # Attic 2nd-aorist-passive (γράφω: ἐγράφην, no θ but active endings)
        ("ἐγράφην", "γράφω", "passive_aorist_indicative_1sg"),
        ("ἐγράφη", "γράφω", "passive_aorist_indicative_3sg"),
        ("ἐγράφησαν", "γράφω", "passive_aorist_indicative_3pl"),
        # Other θη-aorists
        ("ἐπείσθη", "πείθω", "passive_aorist_indicative_3sg"),
        ("ἐτύφθη", "τύπτω", "passive_aorist_indicative_3sg"),
        # Detector must be inactive on non-passive-aorist-indicative keys
        ("ἐλύμην", "λύω", "middle_aorist_indicative_1sg"),
        ("ἐλυσάμην", "λύω", "middle_aorist_indicative_1sg"),
        ("λύομαι", "λύω", "middle_present_indicative_1sg"),
    ])
    def test_attic_or_other_keys_pass(self, b, form, lemma, key):
        assert not b.is_homeric_root_aorist_passive(form, lemma, key), \
            f"falsely flagged: {form} ({lemma}, {key})"

    def test_empty_inputs_safe(self, b):
        assert not b.is_homeric_root_aorist_passive(
            "", "λύω", "passive_aorist_indicative_1sg")
        assert not b.is_homeric_root_aorist_passive(
            "ἐλύμην", "", "passive_aorist_indicative_1sg")
        assert not b.is_homeric_root_aorist_passive(
            "ἐλύμην", "λύω", None)


# ---------------------------------------------------------------------------
# is_enclitic_context_form
# ---------------------------------------------------------------------------


class TestIsEncliticContextForm:
    """Treebank corpora preserve the surface accent that a following
    enclitic projects onto its host word, so glaux ships forms like
    ἤκουόν (= ἤκουον + τι) where the canonical 1sg-imperfect cell
    expects a clean ἤκουον. Greek finite verbs are recessive and have
    exactly one accent, so a primary accent + an extra acute on the
    ultima vowel is the diagnostic signature of these enclitic-context
    surface forms. Detection: NFD-decompose, count vowel marks; flag
    when there are exactly two marks AND the rightmost vowel carries
    an acute (not circumflex) AND there is exactly one other accent
    earlier in the word.
    """

    @pytest.mark.parametrize("form", [
        # The reported bug for ἀκούω 1sg imperfect: ἤκουόν.
        "ἤκουόν",
        # Same shape on other thematic verbs (3sg / 3pl aorist + enclitic).
        "ἔπεμψέ",      # ἔπεμψε + (τι) -> ἔπεμψέ
        "ἔπεμψέν",     # ἔπεμψεν + (τι)
        "ἔγραφέ",      # γράφω 3sg impf + enclitic
        "ἔγραψέν",
        "ἤκουσέ",      # ἀκούω 3sg aor + enclitic
        # Aorist participles with enclitic-following accent
        "δράσαντές",   # δράσαντες + (τι)
        "πέμψαντές",
        # Perispomenon penult + ultima acute (a perispomenon's penult
        # already echoes accent onto the ultima per Smyth #186).
        "δῶκέ",        # δῶκε + enclitic
        "δεῖράν",
        # Infinitive endings: -σθαι / -σαι take an extra acute too.
        "στῆσαί",      # στῆσαι + (τι)
        # Optative / future / various tenses
        "βουλεύοιέν",
        "φώνησέν",     # φώνησεν + enclitic
        # The 17K-form glaux survey: every one of these has the same
        # shape regardless of mood / tense / voice.
        "ἐπικλείοιτέ",
        "ψεύδοιό",
        "δεύοιτό",
        "θαρσύνετόν",  # synthetic; 2du middle imperfect + enclitic
        "βαρύθεσκέ",   # iterative + enclitic
        "ὦρσέν",       # ὦρσεν + (τι)
        "ἔτεκέν",
    ])
    def test_enclitic_context_detected(self, b, form):
        assert b.is_enclitic_context_form(form), \
            f"enclitic-context not detected: {form!r}"

    @pytest.mark.parametrize("form", [
        # Clean canonical citation forms: exactly one accent each.
        "ἤκουον",       # 1sg / 3pl impf - the form ἤκουόν *should* collapse to.
        "ἤκουσε",
        "ἤκουσεν",
        "ἔπεμψε",
        "ἔπεμψεν",
        "ἔγραφον",
        "ἔγραψε",
        "ἐποίει",
        "ἐποίουν",
        # Recessive accent on antepenult: still ONE accent.
        "ἐποιήσαμεν",
        "ἐδιδάσκετε",
        # Circumflex-bearing forms: still one accent.
        "λῦσε",
        "δῶκε",
        "βλέπω",
        "ἐδίδασκον",
        # Single oxytone accent on the ultima (e.g. πᾶς, αὐτός): one
        # accent only, must NOT trigger.
        "αὐτός",
        "πᾶς",
        "πατήρ",
        # Forms with NO accent (corpus glitch / stripped form): not
        # flagged as enclitic-context.
        "ακουω",
        "παυω",
        # Empty / single-char.
        "",
        "α",
    ])
    def test_clean_forms_pass(self, b, form):
        assert not b.is_enclitic_context_form(form), \
            f"clean form falsely flagged as enclitic-context: {form!r}"

    def test_circumflex_on_ultima_not_flagged(self, b):
        # The ultima MUST carry an acute (the enclitic-derived mark).
        # A perispomenon ultima (circumflex) is the verb's own accent
        # and must not trigger, even if there's another accent earlier
        # somehow (defensive: such forms are rare but possible in OCR).
        assert not b.is_enclitic_context_form("λῦε")
        assert not b.is_enclitic_context_form("δοκεῖ")

    def test_three_or_more_accents_not_flagged(self, b):
        # We only handle the conservative two-mark shape. Forms with
        # 3+ accent marks are corpus glitches we don't try to repair
        # heuristically; let them fall through.
        # (Construct an artificial 3-acute form to verify gating.)
        # ή́ + κ + ο + ύ + ο + ν́ -> 3 acutes
        weird = "η" + "́" + "κο" + "υ" + "́" + "ο" + "ν" + "́"
        # NFC-compose
        import unicodedata
        weird = unicodedata.normalize("NFC", weird)
        assert not b.is_enclitic_context_form(weird)


# ---------------------------------------------------------------------------
# Integration: enclitic-context filter must not break past-indicative cells
# ---------------------------------------------------------------------------


class TestEncliticContextDoesNotShadowGoodForms:
    """End-to-end check: with the filter active, the canonical 1sg /
    3sg / 3pl imperfect cells for ἀκούω, γράφω, διδάσκω, πέμπω, ποιέω
    must come out single-accented. The filter drops the corpus's
    enclitic-context form ἤκουόν, leaving the cell empty (or filled
    by a clean variant); the synth pass then writes ἤκουον into the
    1sg slot.

    These tests exercise the end-to-end pipeline but are scoped to a
    single small `--only` build run so they don't slow the suite.
    """

    @pytest.fixture(scope="class")
    def paradigms(self):
        """Build paradigms for a small lemma set, reusing the real data
        files. Skips if data files are not present."""
        import importlib.util
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[1]
        data_dir = repo_root / "data"
        if not (data_dir / "ag_pairs.json").exists():
            pytest.skip("data/ag_pairs.json not present; skipping integration")
        if not (data_dir / "glaux_pairs.json").exists():
            pytest.skip("data/glaux_pairs.json not present; skipping")
        spec = importlib.util.spec_from_file_location(
            "build_grc_verb_paradigms",
            repo_root / "build" / "build_grc_verb_paradigms.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        only = {"ἀκούω"}
        return mod.build_paradigms(only_lemmas=only)

    def test_akouo_imperfect_1sg_clean(self, paradigms):
        # The reported bug: previously emitted ἤκουόν (double accent).
        # After the fix: clean ἤκουον.
        entry = paradigms.get("ἀκούω")
        assert entry is not None, "no paradigm built for ἀκούω"
        forms = entry["forms"]
        got = forms.get("active_imperfect_indicative_1sg")
        assert got == "ἤκουον", \
            f"active_imperfect_indicative_1sg = {got!r} (expected ἤκουον)"

    @pytest.mark.parametrize("key", [
        "active_imperfect_indicative_1sg",
        "active_imperfect_indicative_2sg",
        "active_imperfect_indicative_3sg",
        "active_imperfect_indicative_1pl",
        "active_imperfect_indicative_2pl",
        "active_imperfect_indicative_3pl",
        "middle_imperfect_indicative_1sg",
        "middle_imperfect_indicative_3sg",
        "middle_imperfect_indicative_3pl",
    ])
    def test_akouo_imperfect_cells_single_accent(self, paradigms, key):
        """Every imperfect-indicative cell in ἀκούω's paradigm must
        have exactly one accent mark. The full set covers 1sg/2sg/3sg/
        1pl/2pl/3pl active and 1sg/3sg/3pl middle - the cells that the
        glaux corpus contributed enclitic-context variants for in the
        original bug report.
        """
        import unicodedata
        entry = paradigms.get("ἀκούω")
        assert entry is not None
        forms = entry["forms"]
        f = forms.get(key)
        if f is None:
            pytest.skip(f"cell {key!r} not present (no source attests it)")
        nfd = unicodedata.normalize("NFD", f)
        n_accents = sum(1 for c in nfd if c in ("́", "͂"))
        assert n_accents == 1, \
            f"{key} = {f!r} has {n_accents} accents (expected 1)"


# ---------------------------------------------------------------------------
# Enclitic-context: grave-on-ultima coverage
# ---------------------------------------------------------------------------


class TestEncliticContextGraveOnUltima:
    """An oxytone host word followed by an enclitic in running text often
    surfaces with grave (not acute) on the ultima after editorial
    accent-normalisation: ``ἐποίησὲ`` is the same enclitic-context shape
    as ``ἐποίησέ`` (= ἐποίησε + τι), just rendered with the running-text
    grave that an oxytone gets when more text follows. Both shapes must
    be filtered, otherwise ``pick_best_form``'s polytonic-richness
    tiebreaker preferred ``ἐποίησὲ`` over the canonical ``ἐποίησε``,
    and the subsequent grave-to-acute pass turned the wrong winner into
    the bogus ``ἐποίησέ``.
    """

    @pytest.mark.parametrize("form", [
        # Aorist + grave-on-ultima from running text after enclitic.
        "ἐποίησὲ",       # ποιέω 3sg aor + (τι) with grave normalisation
        "ἤκουσὲ",        # ἀκούω 3sg aor variant
        "ἔπεμψὲ",        # πέμπω 3sg aor variant
        # Synthetic but morphologically sound: imperfect 1sg with grave.
        "ἤκουὸν",
    ])
    def test_grave_on_ultima_detected(self, b, form):
        assert b.is_enclitic_context_form(form), \
            f"grave-on-ultima enclitic-context not flagged: {form!r}"

    @pytest.mark.parametrize("form", [
        # Oxytone words whose ultima carries grave because more text
        # follows in running prose are GENUINE Greek words (αὐτός in
        # mid-sentence -> αὐτὸς). They must NOT be flagged as enclitic-
        # context: there is no other accent earlier in the word.
        "αὐτὸς",
        "πατὴρ",
    ])
    def test_grave_oxytone_alone_not_flagged(self, b, form):
        assert not b.is_enclitic_context_form(form), \
            f"single-grave oxytone falsely flagged: {form!r}"


# ---------------------------------------------------------------------------
# Iota-dropped contract-stem detector
# ---------------------------------------------------------------------------


class TestIsIotaDroppedContractForm:
    """Hellenistic / Ionic spelling drops the iota in the contract stem
    of an -ιέω verb: ποιέω -> ποέω, ἐποίησα -> ἐπόησα, ἐποίει -> ἐπόει.
    Wiktionary lists ποέω as an alternative form of ποιέω and emits
    its iota-less inflectional table under the canonical ποιέω lemma;
    glaux likewise lemmatises iota-less Hellenistic tokens to ποιέω.
    Once both sources pool, the ``-len(f)`` tiebreaker in
    ``pick_best_form`` picks ἐπόησα over ἐποίησα and every past-
    indicative cell ends up reporting the wrong canonical form.
    """

    @pytest.mark.parametrize("form,lemma", [
        # Bug-report verbs.
        ("ἐπόησα", "ποιέω"),
        ("ἐπόησε", "ποιέω"),
        ("ἐπόησέ", "ποιέω"),
        ("ἐπόησαν", "ποιέω"),
        ("ἐπόει", "ποιέω"),
        ("ἐπόουν", "ποιέω"),
        ("ποήσει", "ποιέω"),
        ("ποήσεις", "ποιέω"),
        # Bare alt headword (the ποέω lemma cited under ποιέω).
        ("ποέω", "ποιέω"),
        ("ποέει", "ποιέω"),
        ("ποέεις", "ποιέω"),
        # Prefixed compounds inherit the same orthographic pattern.
        ("μετεπόησα", "μεταποιέω"),
        ("ἐνεπόησα", "ἐμποιέω"),
        ("περιεπόησα", "περιποιέω"),
    ])
    def test_iota_dropped_detected(self, b, form, lemma):
        assert b.is_iota_dropped_contract_form(form, lemma), \
            f"iota-dropped not detected: {form!r} for {lemma!r}"

    @pytest.mark.parametrize("form,lemma", [
        # Canonical Attic forms with the iota intact: must not flag.
        ("ἐποίησα", "ποιέω"),
        ("ἐποίησε", "ποιέω"),
        ("ἐποίησεν", "ποιέω"),
        ("ποίησα", "ποιέω"),
        ("ἐποίει", "ποιέω"),
        ("ἐποίουν", "ποιέω"),
        ("ἐποίησάμην", "ποιέω"),
        # Compound prefix with the iota intact.
        ("μετεποίησα", "μεταποιέω"),
        ("ἐνεποίησα", "ἐμποιέω"),
        # Non-contract, non-ιέω lemmas: filter must always return False.
        ("ἔγραψα", "γράφω"),
        ("ἤκουσα", "ἀκούω"),
        ("ἐπέμπον", "πέμπω"),
        ("ἐδίδασκον", "διδάσκω"),
        # ε-contract WITHOUT iota in the stem: filter doesn't fire.
        ("ἐκίνησα", "κινέω"),
        ("ἐδόκησα", "δοκέω"),
        # α-contract: filter never fires (lemma doesn't end in -ιεω).
        ("ἐτίμησα", "τιμάω"),
        # Empty / degenerate.
        ("", "ποιέω"),
        ("ἐποίησα", ""),
    ])
    def test_clean_forms_pass(self, b, form, lemma):
        assert not b.is_iota_dropped_contract_form(form, lemma), \
            f"clean form falsely flagged: {form!r} for {lemma!r}"


# ---------------------------------------------------------------------------
# Integration: ποιέω canonical past-indicative cells survive synth + filter
# ---------------------------------------------------------------------------


class TestPoieoCanonicalPastIndicatives:
    """End-to-end check on ποιέω: with the iota-dropped filter active
    AND the grave-on-ultima extension to the enclitic detector AND the
    sentence-initial form lowercase pass, the canonical past-indicative
    cells must be the iota-bearing Attic forms with a single accent
    each. Replicates the bug report.
    """

    @pytest.fixture(scope="class")
    def paradigms(self):
        import importlib.util
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[1]
        data_dir = repo_root / "data"
        if not (data_dir / "ag_pairs.json").exists():
            pytest.skip("data/ag_pairs.json not present")
        if not (data_dir / "glaux_pairs.json").exists():
            pytest.skip("data/glaux_pairs.json not present")
        spec = importlib.util.spec_from_file_location(
            "build_grc_verb_paradigms",
            repo_root / "build" / "build_grc_verb_paradigms.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.build_paradigms(only_lemmas={"ποιέω"})

    @pytest.mark.parametrize("key,expected", [
        ("active_aorist_indicative_1sg", "ἐποίησα"),
        ("active_aorist_indicative_3sg", "ἐποίησε"),
        ("active_aorist_indicative_3pl", "ἐποίησαν"),
        ("active_imperfect_indicative_1sg", "ἐποίουν"),
        ("active_imperfect_indicative_3sg", "ἐποίει"),
        ("active_imperfect_indicative_3pl", "ἐποίουν"),
    ])
    def test_canonical_attic_form(self, paradigms, key, expected):
        entry = paradigms.get("ποιέω")
        assert entry is not None, "no paradigm for ποιέω"
        got = entry["forms"].get(key)
        assert got == expected, \
            f"{key} = {got!r} (expected {expected!r})"


# ---------------------------------------------------------------------------
# Integration: γράφω / διδάσκω / πέμπω past-indicative cells single-accented
# ---------------------------------------------------------------------------


class TestThematicVerbsPastIndicativeAccents:
    """End-to-end check that γράφω, διδάσκω, πέμπω each emit the
    canonical 1sg / 3sg past-indicative forms reported in the bug
    report after the enclitic + iota-dropped filter pass and synth
    fallback finishes.
    """

    @pytest.fixture(scope="class")
    def paradigms(self):
        import importlib.util
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[1]
        data_dir = repo_root / "data"
        if not (data_dir / "ag_pairs.json").exists():
            pytest.skip("data/ag_pairs.json not present")
        if not (data_dir / "glaux_pairs.json").exists():
            pytest.skip("data/glaux_pairs.json not present")
        spec = importlib.util.spec_from_file_location(
            "build_grc_verb_paradigms",
            repo_root / "build" / "build_grc_verb_paradigms.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.build_paradigms(only_lemmas={"γράφω", "διδάσκω", "πέμπω"})

    @pytest.mark.parametrize("lemma,key,expected", [
        # γράφω: bug report had ἔγραφόν (double accent); expect ἔγραφον.
        ("γράφω", "active_imperfect_indicative_1sg", "ἔγραφον"),
        ("γράφω", "active_imperfect_indicative_3pl", "ἔγραφον"),
        ("γράφω", "active_aorist_indicative_1sg", "ἔγραψα"),
        ("γράφω", "active_aorist_indicative_3sg", "ἔγραψε"),
        # διδάσκω: bug report had ἐδίδασκόν; expect ἐδίδασκον.
        ("διδάσκω", "active_imperfect_indicative_1sg", "ἐδίδασκον"),
        ("διδάσκω", "active_imperfect_indicative_3pl", "ἐδίδασκον"),
        ("διδάσκω", "active_aorist_indicative_1sg", "ἐδίδαξα"),
        ("διδάσκω", "active_aorist_indicative_3sg", "ἐδίδαξε"),
        # πέμπω: bug report had Ἔπεμψέ (capital + double accent);
        # expect ἔπεμψε with single accent and lowercase.
        ("πέμπω", "active_aorist_indicative_1sg", "ἔπεμψα"),
        ("πέμπω", "active_aorist_indicative_3sg", "ἔπεμψε"),
        ("πέμπω", "active_imperfect_indicative_1sg", "ἔπεμπον"),
        ("πέμπω", "active_imperfect_indicative_3sg", "ἔπεμπε"),
    ])
    def test_canonical_form(self, paradigms, lemma, key, expected):
        entry = paradigms.get(lemma)
        assert entry is not None, f"no paradigm for {lemma!r}"
        got = entry["forms"].get(key)
        assert got == expected, \
            f"{lemma} {key} = {got!r} (expected {expected!r})"


class TestAccentTypeSkeleton:
    """The key that decides when Wiktionary's spelling may correct a cell."""

    def test_acute_and_circumflex_on_the_same_vowel_share_a_key(self, b):
        assert (b.accent_type_skeleton("λῦσαν")
                == b.accent_type_skeleton("λύσαν"))
        # Vowel-length marks are notation, not spelling.
        assert (b.accent_type_skeleton("λῦσᾰν")
                == b.accent_type_skeleton("λῦσαν"))

    def test_other_differences_keep_words_apart(self, b):
        # Iota subscript: τιμᾷς is the verb, τιμάς the noun's accusative.
        assert (b.accent_type_skeleton("τιμᾷς")
                != b.accent_type_skeleton("τιμάς"))
        # Accent position: λυσόμενα is Attic, λυσομένα Doric.
        assert (b.accent_type_skeleton("λυσόμενα")
                != b.accent_type_skeleton("λυσομένα"))
        # Breathing.
        assert (b.accent_type_skeleton("ἑστός")
                != b.accent_type_skeleton("ἐστός"))


class TestWiktionarySpellings:
    def test_quantity_marked_cell_decides_the_accent(self, b):
        pairs = [
            {"form": "λῦσᾰν", "lemma": "λύω", "pos": "verb", "tags": []},
            {"form": "λύσαν", "lemma": "λύω", "pos": "verb", "tags": []},
        ]
        spellings, letters = b.wiktionary_spellings(pairs)
        assert spellings[("λύω", b.accent_type_skeleton("λύσαν"))] == "λῦσαν"
        assert ("λύω", b.accent_type_skeleton("λῦσαν")) in letters

    def test_unmarked_disagreement_is_left_alone(self, b):
        # Neither spelling carries quantity and the corpus attests neither:
        # the accent turns on a vowel length nothing records.
        pairs = [
            {"form": "ζζῦσαν", "lemma": "ζζύω", "pos": "verb", "tags": []},
            {"form": "ζζύσαν", "lemma": "ζζύω", "pos": "verb", "tags": []},
        ]
        spellings, _letters = b.wiktionary_spellings(pairs)
        assert ("ζζύω", b.accent_type_skeleton("ζζύσαν")) not in spellings


class TestAuxiliaryForms:
    def test_eimi_forms_are_the_auxiliary_of_a_periphrastic_cell(self, b):
        # "τετιμηκότες εἶτε" leaves εἶτε tagged as a cell of τιμάω.
        assert "εἶτε" in b.EIMI_AUXILIARY_FORMS
        assert "εἴην" in b.EIMI_AUXILIARY_FORMS
        assert "ὦ" in b.EIMI_AUXILIARY_FORMS


class TestPolytonicPoolNoLongerDropsAtticForms:
    def test_circumflex_does_not_eliminate_an_acute_rival(self, b):
        # δίδως (Attic, 171 corpus tokens) against Ionic διδοῖς (16): the
        # circumflex used to remove δίδως from the pool before ranking.
        assert b.pick_best_form(["δίδως", "διδοῖς"],
                                key="active_present_indicative_2sg",
                                lemma="δίδωμι") == "δίδως"

    def test_stripped_spellings_still_lose_to_accented_ones(self, b):
        assert b.pick_best_form(["λυω", "λύω"]) == "λύω"


class TestSubjunctiveShape:
    def test_a_short_vowel_ending_cannot_be_a_subjunctive(self, b):
        for form, key in [
            ("κολακεύσεις", "active_aorist_subjunctive_2sg"),
            ("σῴσει", "active_aorist_subjunctive_3sg"),
            ("δράσομεν", "active_aorist_subjunctive_1pl"),
            ("λύουσιν", "active_present_subjunctive_3pl"),
            ("αἰδέσεται", "middle_aorist_subjunctive_3sg"),
            ("λυόμεθα", "middle_present_subjunctive_1pl"),
        ]:
            assert not b.has_subjunctive_shape(form, key), form

    def test_real_subjunctives_pass(self, b):
        for form, key in [
            ("κολακεύσῃς", "active_aorist_subjunctive_2sg"),
            ("δῷς", "active_aorist_subjunctive_2sg"),
            ("δηλοῖς", "active_present_subjunctive_2sg"),
            ("λύσωμεν", "active_aorist_subjunctive_1pl"),
            ("λύωσι", "active_present_subjunctive_3pl"),
            ("αἰδέσηται", "middle_aorist_subjunctive_3sg"),
            ("ᾖ", "active_present_subjunctive_3sg"),
        ]:
            assert b.has_subjunctive_shape(form, key), form

    def test_endings_the_two_moods_share_are_left_alone(self, b):
        # The alpha contracts spell the indicative and the subjunctive the
        # same, so the rule must not touch them.
        for form, key in [
            ("τιμᾶτε", "active_present_subjunctive_2pl"),
            ("τιμᾶται", "middle_present_subjunctive_3sg"),
            ("τιμᾶσθε", "middle_present_subjunctive_2pl"),
            ("τιμᾷς", "active_present_subjunctive_2sg"),
        ]:
            assert b.has_subjunctive_shape(form, key), form

    def test_every_other_cell_is_untouched(self, b):
        assert b.has_subjunctive_shape("κολακεύσεις",
                                       "active_future_indicative_2sg")
        assert b.has_subjunctive_shape("λύει", None)

    def test_it_only_breaks_a_tie(self, b):
        # GLAUx tags both as the aorist subjunctive 2sg; without the shape
        # rule the commoner future indicative takes the cell.
        assert b.pick_best_form(
            ["κολακεύσεις", "κολακεύσῃς"],
            key="active_aorist_subjunctive_2sg", lemma="κολακεύω",
        ) == "κολακεύσῃς"

    def test_a_cell_with_no_well_formed_candidate_still_fills(self, b):
        assert b.pick_best_form(
            ["κολακεύσεις"], key="active_aorist_subjunctive_2sg",
            lemma="κολακεύω",
        ) == "κολακεύσεις"


class TestStripTonalAccents:
    def test_the_iota_subscript_and_breathing_survive(self, b):
        assert b.strip_tonal_accents("κολακεύσῃς") == "κολακευσῃς"
        assert b.strip_tonal_accents("ᾧ") == "ᾡ"
        assert b.strip_tonal_accents("λῦσαι") == "λυσαι"
        # The other strip_accents drops the iota with everything else.
        assert b.strip_accents("ᾧ") == "ω"


class TestKeyCellTags:
    def test_finite_participle_and_infinitive_keys(self, b):
        assert b.key_cell_tags("active_aorist_optative_3sg") == frozenset(
            {"active", "optative", "third-person", "singular"})
        assert b.key_cell_tags("middle_aorist_imperative_2sg") == frozenset(
            {"middle", "imperative", "second-person", "singular"})
        assert b.key_cell_tags("active_present_participle_nom_n_sg") == (
            frozenset({"active", "participle", "nominative", "neuter",
                       "singular"}))
        assert b.key_cell_tags("active_aorist_infinitive") == frozenset(
            {"active", "infinitive"})


class TestWiktionaryCells:
    # λύω's tables give the two accentuations of these letters to two
    # different cells; βαίνω's give both spellings of the neuter participle
    # to the same one.
    PAIRS = [
        {"form": "λύσαι", "lemma": "λύω", "pos": "verb",
         "tags": ["active", "optative", "singular", "third-person"]},
        {"form": "λῦσαι", "lemma": "λύω", "pos": "verb",
         "tags": ["imperative", "middle", "second-person", "singular"]},
        {"form": "βαῖνον", "lemma": "βαίνω", "pos": "verb",
         "tags": ["active", "neuter", "participle"]},
        {"form": "βαίνον", "lemma": "βαίνω", "pos": "verb",
         "tags": ["active", "neuter", "participle"]},
        {"form": "βαίνον", "lemma": "βαίνω", "pos": "verb", "tags": []},
    ]

    def test_spellings_are_grouped_by_cell_and_by_letters(self, b):
        cells, by_letters = b.wiktionary_cells(self.PAIRS)
        assert cells[("λύω", "λύσαι")] == {
            frozenset({"active", "optative", "singular", "third-person"})}
        assert cells[("βαίνω", "βαίνον")] == cells[("βαίνω", "βαῖνον")]
        assert by_letters[("λύω", b.accent_type_skeleton("λύσαι"))] == {
            "λύσαι", "λῦσαι"}

    def test_a_tagless_entry_contributes_no_cell(self, b):
        cells, _ = b.wiktionary_cells(
            [{"form": "λύσαι", "lemma": "λύω", "pos": "verb", "tags": []}])
        assert cells == {}

    def test_mediopassive_is_read_as_middle(self, b):
        cells, _ = b.wiktionary_cells([
            {"form": "λύεται", "lemma": "λύω", "pos": "verb",
             "tags": ["mediopassive", "indicative", "third-person",
                      "singular"]}])
        cell = next(iter(cells[("λύω", "λύεται")]))
        assert "middle" in cell and "mediopassive" not in cell

    def test_quantity_marks_are_dropped_from_the_key(self, b):
        cells, by_letters = b.wiktionary_cells([
            {"form": "λῦσᾰν", "lemma": "λύω", "pos": "verb",
             "tags": ["active", "neuter", "participle"]}])
        assert ("λύω", "λῦσαν") in cells


class TestDropCollapsedCells:
    # kaikki delivers τιμάω's whole present active indicative row tagged
    # first-person singular, so one cell holds six different forms.
    COLLAPSED = [
        {"form": f, "lemma": "τιμάω", "pos": "verb",
         "tags": ["active", "present", "indicative", "first-person",
                  "singular"]}
        for f in ("τιμῶ", "τιμᾷς", "τιμᾷ", "τιμῶμεν", "τιμᾶτε", "τιμῶσι")
    ]
    GOOD = [
        {"form": "λύω", "lemma": "λύω", "pos": "verb",
         "tags": ["active", "present", "indicative", "first-person",
                  "singular"]},
        {"form": "λύεις", "lemma": "λύω", "pos": "verb",
         "tags": ["active", "present", "indicative", "second-person",
                  "singular"]},
    ]

    def test_a_cell_holding_a_whole_row_is_dropped(self, b):
        kept, cells, pairs = b.drop_collapsed_cells(self.COLLAPSED)
        assert kept == []
        assert (cells, pairs) == (1, 6)

    def test_well_tagged_cells_survive(self, b):
        kept, cells, pairs = b.drop_collapsed_cells(self.GOOD)
        assert kept == self.GOOD
        assert (cells, pairs) == (0, 0)

    def test_only_the_collapsed_cell_goes(self, b):
        kept, cells, _ = b.drop_collapsed_cells(self.COLLAPSED + self.GOOD)
        assert [p["form"] for p in kept] == ["λύω", "λύεις"]
        assert cells == 1

    def test_spelling_variants_of_one_form_are_not_a_collapsed_cell(self, b):
        # Movable nu and Wiktionary's quantity marks are the same form.
        pairs = [
            {"form": f, "lemma": "λύω", "pos": "verb",
             "tags": ["active", "present", "indicative", "third-person",
                      "plural"]}
            for f in ("λύουσι", "λύουσιν", "λύουσῐ", "λύουσῐν")
        ]
        kept, cells, _ = b.drop_collapsed_cells(pairs)
        assert cells == 0 and len(kept) == 4

    def test_entries_with_no_cell_are_left_alone(self, b):
        pairs = [{"form": "τιμῶ", "lemma": "τιμάω", "pos": "verb",
                  "tags": ["active", "indicative"]}]
        assert b.drop_collapsed_cells(pairs)[0] == pairs
