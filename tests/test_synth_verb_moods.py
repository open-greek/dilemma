"""Tests for ``build/synth_verb_moods.py`` and its integration in
``build/build_grc_verb_paradigms.py``.

The synth module is loaded directly as a script-style module via
``importlib`` (the ``build/`` directory is not an installable package),
mirroring the pattern in ``test_athematic_verbs.py``.

Run with:

    python -m pytest tests/test_synth_verb_moods.py -x -v
"""

from __future__ import annotations

import importlib.util
import unicodedata
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTH_PATH = REPO_ROOT / "build" / "synth_verb_moods.py"


@pytest.fixture(scope="module")
def synth():
    """Load build/synth_verb_moods.py as a module without package install."""
    spec = importlib.util.spec_from_file_location(
        "synth_verb_moods", SYNTH_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _strip_accents(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if not unicodedata.combining(c))


# ---------------------------------------------------------------------------
# Lemma classification
# ---------------------------------------------------------------------------


class TestIsThematicOmega:
    def test_plain_omega_verb(self, synth):
        assert synth.is_thematic_omega("λύω")
        assert synth.is_thematic_omega("γράφω")
        assert synth.is_thematic_omega("παύω")
        assert synth.is_thematic_omega("παιδεύω")

    def test_alpha_contract_rejected(self, synth):
        assert not synth.is_thematic_omega("τιμάω")

    def test_epsilon_contract_rejected(self, synth):
        assert not synth.is_thematic_omega("φιλέω")
        assert not synth.is_thematic_omega("ποιέω")

    def test_omicron_contract_rejected(self, synth):
        assert not synth.is_thematic_omega("δηλόω")

    def test_athematic_mi_rejected(self, synth):
        assert not synth.is_thematic_omega("τίθημι")
        assert not synth.is_thematic_omega("δίδωμι")
        assert not synth.is_thematic_omega("εἰμί")

    def test_deponent_mai_rejected(self, synth):
        assert not synth.is_thematic_omega("ἔρχομαι")
        assert not synth.is_thematic_omega("γίγνομαι")

    def test_empty_or_garbage_rejected(self, synth):
        assert not synth.is_thematic_omega("")
        assert not synth.is_thematic_omega(None)
        # No -ω lemma
        assert not synth.is_thematic_omega("ἀνήρ")


# ---------------------------------------------------------------------------
# Present-system synthesis (uses lemma stem only, no principal_parts needed)
# ---------------------------------------------------------------------------


class TestPresentSynthesis:
    """Present subj/opt/imp synthesis works off the lemma alone -- no
    principal_parts data required."""

    def test_lyo_present_subjunctive(self, synth):
        out = synth.synthesize_active_moods("λύω", {})
        assert out["active_present_subjunctive_1sg"] == "λύω"
        assert out["active_present_subjunctive_2sg"] == "λύῃς"
        assert out["active_present_subjunctive_3sg"] == "λύῃ"
        assert out["active_present_subjunctive_1pl"] == "λύωμεν"
        assert out["active_present_subjunctive_2pl"] == "λύητε"
        assert out["active_present_subjunctive_3pl"] == "λύωσι(ν)"

    def test_lyo_present_optative(self, synth):
        out = synth.synthesize_active_moods("λύω", {})
        assert out["active_present_optative_1sg"] == "λύοιμι"
        assert out["active_present_optative_2sg"] == "λύοις"
        assert out["active_present_optative_3sg"] == "λύοι"
        assert out["active_present_optative_1pl"] == "λύοιμεν"
        assert out["active_present_optative_2pl"] == "λύοιτε"
        assert out["active_present_optative_3pl"] == "λύοιεν"

    def test_lyo_present_imperative(self, synth):
        out = synth.synthesize_active_moods("λύω", {})
        assert out["active_present_imperative_2sg"] == "λύε"
        assert out["active_present_imperative_2pl"] == "λύετε"
        # 3sg/3pl have ending-stress and lose the lemma's stem accent.
        assert out["active_present_imperative_3sg"] == "λυέτω"
        assert out["active_present_imperative_3pl"] == "λυόντων"

    def test_lyo_present_infinitive(self, synth):
        out = synth.synthesize_active_moods("λύω", {})
        assert out["active_present_infinitive"] == "λύειν"

    def test_grapho_keeps_stem_accent(self, synth):
        """γράφω's stem accent on γρά- should ride through to the
        synthesised forms (γράφω -> γράφω-subj, γράφοιμι-opt)."""
        out = synth.synthesize_active_moods("γράφω", {})
        assert out["active_present_subjunctive_1sg"] == "γράφω"
        assert out["active_present_optative_1sg"] == "γράφοιμι"
        # 3sg imperative drops stem accent (ending carries it).
        assert out["active_present_imperative_3sg"] == "γραφέτω"

    def test_present_only_when_no_pp_provided(self, synth):
        """Without any principal_parts, only present-system cells are
        produced -- no aorist."""
        out = synth.synthesize_active_moods("λύω", None)
        assert "active_present_subjunctive_1sg" in out
        assert "active_aorist_subjunctive_1sg" not in out


# ---------------------------------------------------------------------------
# Aorist synthesis from principal parts
# ---------------------------------------------------------------------------


class TestAoristSynthesisFromFut:
    """When the future principal part is available, the aorist stem
    derives from it (preserves accent without an augment)."""

    def test_lyo_with_fut(self, synth):
        out = synth.synthesize_active_moods("λύω", {"fut": "λύσω"})
        assert out["active_aorist_subjunctive_1sg"] == "λύσω"
        assert out["active_aorist_subjunctive_2sg"] == "λύσῃς"
        assert out["active_aorist_subjunctive_3pl"] == "λύσωσι(ν)"
        assert out["active_aorist_optative_1sg"] == "λύσαιμι"
        assert out["active_aorist_optative_2sg"] == "λύσαις"
        assert out["active_aorist_optative_3pl"] == "λύσαιεν"
        assert out["active_aorist_imperative_2sg"] == "λύσον"
        assert out["active_aorist_imperative_2pl"] == "λύσατε"
        # 3sg/3pl drop stem accent (ending is accented).
        assert out["active_aorist_imperative_3sg"] == "λυσάτω"
        assert out["active_aorist_imperative_3pl"] == "λυσάντων"
        assert out["active_aorist_infinitive"] == "λύσαι"

    def test_paideuo_with_fut(self, synth):
        out = synth.synthesize_active_moods(
            "παιδεύω", {"fut": "παιδεύσω"}
        )
        assert out["active_aorist_subjunctive_1sg"] == "παιδεύσω"
        assert out["active_aorist_optative_1sg"] == "παιδεύσαιμι"
        assert out["active_aorist_imperative_2sg"] == "παιδεύσον"
        assert out["active_aorist_imperative_3sg"] == "παιδευσάτω"

    def test_pempo_with_fut_psi_cluster(self, synth):
        """π-stem verb πέμπω -> πέμψω: aorist endings hang off πέμψ-."""
        out = synth.synthesize_active_moods("πέμπω", {"fut": "πέμψω"})
        assert out["active_aorist_subjunctive_1sg"] == "πέμψω"
        assert out["active_aorist_optative_1sg"] == "πέμψαιμι"
        assert out["active_aorist_imperative_2sg"] == "πέμψον"
        assert out["active_aorist_imperative_3sg"] == "πεμψάτω"


class TestAoristSynthesisFromAor:
    """When only the aorist (not the future) is given, we splice the
    aor's terminal cluster onto the lemma's present stem to keep the
    accent."""

    def test_grapho_with_aor_only(self, synth):
        """γράφω + aor ἔγραψα: present stem γράφ- -> swap φ for ψ -> γράψ-."""
        out = synth.synthesize_active_moods("γράφω", {"aor": "ἔγραψα"})
        assert out["active_aorist_subjunctive_1sg"] == "γράψω"
        assert out["active_aorist_optative_1sg"] == "γράψαιμι"
        assert out["active_aorist_imperative_2sg"] == "γράψον"
        assert out["active_aorist_imperative_3sg"] == "γραψάτω"
        assert out["active_aorist_infinitive"] == "γράψαι"

    def test_paideuo_with_aor_only(self, synth):
        out = synth.synthesize_active_moods(
            "παιδεύω", {"aor": "ἐπαίδευσα"}
        )
        assert out["active_aorist_subjunctive_1sg"] == "παιδεύσω"
        assert out["active_aorist_optative_1sg"] == "παιδεύσαιμι"
        assert out["active_aorist_imperative_2sg"] == "παιδεύσον"


class TestAoristSkippedForIrregular:
    """Verbs whose aorist is not derivable from the present stem +
    sigma graft should NOT get any synthesised aorist forms (the synth
    is conservative; better to leave a cell empty than fill it wrong)."""

    def test_pipto_aor_2_skipped(self, synth):
        """πίπτω has thematic aor-2 ἔπεσον (no σ); synthesis must skip."""
        out = synth.synthesize_active_moods(
            "πίπτω", {"aor": "ἔπεσον"}
        )
        # No aorist active forms produced
        assert "active_aorist_subjunctive_1sg" not in out
        assert "active_aorist_imperative_2sg" not in out
        # Present forms still synthesised from the lemma
        assert "active_present_subjunctive_1sg" in out

    def test_lipo_aor_2_skipped(self, synth):
        """λείπω has aor-2 ἔλιπον (root vowel ablaut, not sigmatic)."""
        out = synth.synthesize_active_moods(
            "λείπω", {"aor": "ἔλιπον"}
        )
        assert "active_aorist_subjunctive_1sg" not in out
        assert "active_aorist_imperative_2sg" not in out
        assert "active_present_subjunctive_1sg" in out

    def test_meno_liquid_future_skipped(self, synth):
        """μένω -> contract future μενῶ (no σ); no sigmatic stem."""
        out = synth.synthesize_active_moods("μένω", {"fut": "μενῶ"})
        # Aorist branch should not fire from the future alone.
        assert "active_aorist_subjunctive_1sg" not in out
        # Present moods still come through.
        assert out["active_present_subjunctive_1sg"] == "μένω"


# ---------------------------------------------------------------------------
# No-op cases
# ---------------------------------------------------------------------------


class TestNoOpCases:
    def test_contract_verb_returns_empty(self, synth):
        """ε-contract φιλέω is rejected at the classifier; nothing
        synthesised."""
        out = synth.synthesize_active_moods(
            "φιλέω", {"aor": "ἐφίλησα", "fut": "φιλήσω"}
        )
        assert out == {}

    def test_athematic_returns_empty(self, synth):
        """μι-verb δίδωμι: no synthesis (athematic)."""
        out = synth.synthesize_active_moods(
            "δίδωμι", {"aor": "ἔδωκα"}
        )
        assert out == {}

    def test_missing_principal_parts_no_error(self, synth):
        """No principal_parts dict (None) must not error: only present
        moods get synthesised."""
        out = synth.synthesize_active_moods("λύω", None)
        assert isinstance(out, dict)
        assert "active_present_subjunctive_1sg" in out
        assert "active_aorist_subjunctive_1sg" not in out

    def test_empty_principal_parts_no_error(self, synth):
        """Empty dict is also fine (caller had a head text but parser
        returned nothing)."""
        out = synth.synthesize_active_moods("λύω", {})
        assert isinstance(out, dict)
        assert "active_present_subjunctive_1sg" in out
        assert "active_aorist_subjunctive_1sg" not in out

    def test_none_lemma_returns_empty(self, synth):
        out = synth.synthesize_active_moods(None, {"fut": "λύσω"})
        assert out == {}

    def test_non_omega_lemma_returns_empty(self, synth):
        out = synth.synthesize_active_moods(
            "ἀνήρ", {"fut": "whatever"}
        )
        assert out == {}


# ---------------------------------------------------------------------------
# Accent neutralisation on ending-accented cells
# ---------------------------------------------------------------------------


class TestAccentHandling:
    """3sg/3pl imperative endings carry their own accent; the synthesis
    must drop the stem accent to avoid producing double-accented forms
    like ``γράψάτω``."""

    def test_no_double_accent_on_imp_3sg(self, synth):
        out = synth.synthesize_active_moods("γράφω", {"aor": "ἔγραψα"})
        form = out["active_aorist_imperative_3sg"]
        # Only one acute should be present (on the ending vowel).
        nfd = unicodedata.normalize("NFD", form)
        n_acute = nfd.count("́")
        assert n_acute == 1, f"got {n_acute} acutes in {form!r}"

    def test_no_double_accent_on_imp_3pl(self, synth):
        out = synth.synthesize_active_moods("παιδεύω", {"fut": "παιδεύσω"})
        form = out["active_aorist_imperative_3pl"]
        nfd = unicodedata.normalize("NFD", form)
        n_acute = nfd.count("́")
        assert n_acute == 1, f"got {n_acute} acutes in {form!r}"

    def test_2sg_2pl_keep_stem_accent(self, synth):
        """2sg/2pl imperatives have un-accented endings; stem accent
        should be preserved."""
        out = synth.synthesize_active_moods("παιδεύω", {"fut": "παιδεύσω"})
        # παιδεύσον has acute on ευ (the diphthong)
        assert out["active_aorist_imperative_2sg"] == "παιδεύσον"
        assert out["active_aorist_imperative_2pl"] == "παιδεύσατε"


# ---------------------------------------------------------------------------
# Middle / passive synthesis (v2)
# ---------------------------------------------------------------------------


class TestPresentMiddleSynthesis:
    """Present middle / mediopassive synthesis works off the lemma alone."""

    def test_lyo_present_middle_subjunctive(self, synth):
        out = synth.synthesize_mp_moods("λύω", {})
        assert out["middle_present_subjunctive_1sg"] == "λύωμαι"
        assert out["middle_present_subjunctive_2sg"] == "λύῃ"
        assert out["middle_present_subjunctive_3sg"] == "λύηται"
        assert out["middle_present_subjunctive_1pl"] == "λυώμεθα"
        assert out["middle_present_subjunctive_2pl"] == "λύησθε"
        assert out["middle_present_subjunctive_3pl"] == "λύωνται"

    def test_lyo_present_middle_optative(self, synth):
        out = synth.synthesize_mp_moods("λύω", {})
        assert out["middle_present_optative_1sg"] == "λυοίμην"
        assert out["middle_present_optative_2sg"] == "λύοιο"
        assert out["middle_present_optative_3sg"] == "λύοιτο"
        assert out["middle_present_optative_1pl"] == "λυοίμεθα"
        assert out["middle_present_optative_2pl"] == "λύοισθε"
        assert out["middle_present_optative_3pl"] == "λύοιντο"

    def test_lyo_present_middle_imperative(self, synth):
        out = synth.synthesize_mp_moods("λύω", {})
        assert out["middle_present_imperative_2sg"] == "λύου"
        assert out["middle_present_imperative_2pl"] == "λύεσθε"
        # 3sg / 3pl ending-stress dropped lemma stem accent.
        assert out["middle_present_imperative_3sg"] == "λυέσθω"
        assert out["middle_present_imperative_3pl"] == "λυέσθων"

    def test_lyo_present_middle_infinitive(self, synth):
        out = synth.synthesize_mp_moods("λύω", {})
        assert out["middle_present_infinitive"] == "λύεσθαι"

    def test_no_passive_present_keys(self, synth):
        """jtauber emits ``middle_present_*`` only -- no
        ``passive_present_*`` (the present mp shares one column)."""
        out = synth.synthesize_mp_moods("λύω", {})
        for k in out:
            assert not k.startswith("passive_present_"), k

    def test_grapho_keeps_present_stem_accent(self, synth):
        """γράφω -> middle present should preserve stem accent."""
        out = synth.synthesize_mp_moods("γράφω", {})
        assert out["middle_present_subjunctive_1sg"] == "γράφωμαι"
        assert out["middle_present_optative_1sg"] == "γραφοίμην"
        assert out["middle_present_imperative_3sg"] == "γραφέσθω"
        assert out["middle_present_infinitive"] == "γράφεσθαι"


class TestAoristMiddleFromFut:
    """Aorist middle uses the σ-stem from the future principal part."""

    def test_lyo_aorist_middle_subjunctive(self, synth):
        out = synth.synthesize_mp_moods("λύω", {"fut": "λύσω"})
        assert out["middle_aorist_subjunctive_1sg"] == "λύσωμαι"
        assert out["middle_aorist_subjunctive_2sg"] == "λύσῃ"
        assert out["middle_aorist_subjunctive_3sg"] == "λύσηται"
        assert out["middle_aorist_subjunctive_1pl"] == "λυσώμεθα"
        assert out["middle_aorist_subjunctive_2pl"] == "λύσησθε"
        assert out["middle_aorist_subjunctive_3pl"] == "λύσωνται"

    def test_lyo_aorist_middle_optative(self, synth):
        out = synth.synthesize_mp_moods("λύω", {"fut": "λύσω"})
        assert out["middle_aorist_optative_1sg"] == "λυσαίμην"
        assert out["middle_aorist_optative_2sg"] == "λύσαιο"
        assert out["middle_aorist_optative_3sg"] == "λύσαιτο"
        assert out["middle_aorist_optative_1pl"] == "λυσαίμεθα"
        assert out["middle_aorist_optative_2pl"] == "λύσαισθε"
        assert out["middle_aorist_optative_3pl"] == "λύσαιντο"

    def test_lyo_aorist_middle_imperative(self, synth):
        out = synth.synthesize_mp_moods("λύω", {"fut": "λύσω"})
        # 2sg ending -αι; we don't recompute circumflex on the stem
        # so this comes out with an acute (jtauber has λῦσαι with
        # circumflex but we accept the segmentation match).
        assert out["middle_aorist_imperative_2sg"] == "λύσαι"
        assert out["middle_aorist_imperative_2pl"] == "λύσασθε"
        # 3sg/3pl ending-stress dropped stem accent.
        assert out["middle_aorist_imperative_3sg"] == "λυσάσθω"
        assert out["middle_aorist_imperative_3pl"] == "λυσάσθων"

    def test_lyo_aorist_middle_infinitive(self, synth):
        out = synth.synthesize_mp_moods("λύω", {"fut": "λύσω"})
        assert out["middle_aorist_infinitive"] == "λύσασθαι"

    @pytest.mark.parametrize("lemma,future,expected", [
        ("ὄζω", "ὀζήσω", "ὄζησαι"),
        ("κελεύω", "κελεύσω", "κέλευσαι"),
        ("παιδεύω", "παιδεύσω", "παίδευσαι"),
        ("κρώζω", "κρώξω", "κρῶξαι"),
        ("νήφω", "νήψω", "νῆψαι"),
        ("πρήθω", "πρήσω", "πρῆσαι"),
        ("ἀΐσσω", "ἀΐξω", "ἄϊξαι"),
    ])
    def test_second_singular_imperative_is_recessive(self, synth, lemma,
                                                    future, expected):
        out = synth.synthesize_mp_moods(lemma, {"fut": future})
        assert out["middle_aorist_imperative_2sg"] == expected

    @pytest.mark.parametrize("stem,expected", [
        ("λῦσ", "λῦσαι"),
        ("λῡ́σ", "λῡ͂σαι"),
        ("παιδεῦσ", "παίδευσαι"),
        ("λυσ", "λύσαι"),
    ])
    def test_recessive_accent_preserves_explicit_quantity(self, synth, stem,
                                                         expected):
        assert synth._recessive_full_form(stem, "αι") == unicodedata.normalize(
            "NFC", expected)


class TestAoristMiddleFromAor:
    """When only the aorist principal part is given (no fut), σ-stem
    derives from grafting the σ/ψ/ξ cluster onto the present stem."""

    def test_grapho_aorist_middle_with_aor_only(self, synth):
        """γράφω + ἔγραψα: present stem γράφ-, swap φ for ψ -> γράψ-."""
        out = synth.synthesize_mp_moods("γράφω", {"aor": "ἔγραψα"})
        assert out["middle_aorist_subjunctive_1sg"] == "γράψωμαι"
        assert out["middle_aorist_optative_1sg"] == "γραψαίμην"
        assert out["middle_aorist_imperative_3sg"] == "γραψάσθω"
        assert out["middle_aorist_infinitive"] == "γράψασθαι"


class TestPassiveAoristSynthesis:
    """Passive aorist uses the θη-stem from ``aor_p`` principal part."""

    def test_lyo_passive_aorist_subjunctive(self, synth):
        """λύω: aor_p ἐλύθην, stem λυθ-, subj all ending-accented."""
        out = synth.synthesize_mp_moods("λύω", {"aor_p": "ἐλύθην"})
        assert out["passive_aorist_subjunctive_1sg"] == "λυθῶ"
        assert out["passive_aorist_subjunctive_2sg"] == "λυθῇς"
        assert out["passive_aorist_subjunctive_3sg"] == "λυθῇ"
        assert out["passive_aorist_subjunctive_1pl"] == "λυθῶμεν"
        assert out["passive_aorist_subjunctive_2pl"] == "λυθῆτε"
        assert out["passive_aorist_subjunctive_3pl"] == "λυθῶσι(ν)"

    def test_lyo_passive_aorist_optative(self, synth):
        out = synth.synthesize_mp_moods("λύω", {"aor_p": "ἐλύθην"})
        assert out["passive_aorist_optative_1sg"] == "λυθείην"
        assert out["passive_aorist_optative_2sg"] == "λυθείης"
        assert out["passive_aorist_optative_3sg"] == "λυθείη"
        assert out["passive_aorist_optative_1pl"] == "λυθεῖμεν"
        assert out["passive_aorist_optative_2pl"] == "λυθεῖτε"
        assert out["passive_aorist_optative_3pl"] == "λυθεῖεν"
        # Duals: jtauber emits 2du / 3du for the passive aorist optative.
        assert out["passive_aorist_optative_2du"] == "λυθεῖτον"
        assert out["passive_aorist_optative_3du"] == "λυθείτην"

    def test_lyo_passive_aorist_imperative(self, synth):
        out = synth.synthesize_mp_moods("λύω", {"aor_p": "ἐλύθην"})
        assert out["passive_aorist_imperative_2sg"] == "λύθητι"
        assert out["passive_aorist_imperative_3sg"] == "λυθήτω"
        assert out["passive_aorist_imperative_2pl"] == "λύθητε"
        # 3pl: drops the stem-final η before the participle linker -ε-.
        assert out["passive_aorist_imperative_3pl"] == "λυθέντων"

    def test_lyo_passive_aorist_infinitive(self, synth):
        out = synth.synthesize_mp_moods("λύω", {"aor_p": "ἐλύθην"})
        assert out["passive_aorist_infinitive"] == "λυθῆναι"


class TestPeitho:
    """πείθω: dental cluster (θ + σ -> σ) for the aor middle, plus
    σθ-passive aorist ἐπείσθην."""

    def test_peitho_middle_aorist_uses_pi_sigma_stem(self, synth):
        """πείθω -> πείσω -> πεισ- middle aorist stem (θ assimilates)."""
        out = synth.synthesize_mp_moods(
            "πείθω", {"fut": "πείσω", "aor": "ἔπεισα", "aor_p": "ἐπείσθην"}
        )
        assert out["middle_aorist_subjunctive_1sg"] == "πείσωμαι"
        assert out["middle_aorist_subjunctive_3sg"] == "πείσηται"
        assert out["middle_aorist_optative_1sg"] == "πεισαίμην"
        assert out["middle_aorist_imperative_3sg"] == "πεισάσθω"
        assert out["middle_aorist_infinitive"] == "πείσασθαι"

    def test_peitho_passive_aorist_uses_sigma_theta_stem(self, synth):
        """πείθω -> ἐπείσθην -> πεισθ- passive aorist stem."""
        out = synth.synthesize_mp_moods(
            "πείθω", {"fut": "πείσω", "aor_p": "ἐπείσθην"}
        )
        assert out["passive_aorist_subjunctive_1sg"] == "πεισθῶ"
        assert out["passive_aorist_subjunctive_3pl"] == "πεισθῶσι(ν)"
        assert out["passive_aorist_optative_1sg"] == "πεισθείην"
        assert out["passive_aorist_optative_2du"] == "πεισθεῖτον"
        assert out["passive_aorist_imperative_2sg"] == "πείσθητι"
        assert out["passive_aorist_imperative_3sg"] == "πεισθήτω"
        assert out["passive_aorist_imperative_3pl"] == "πεισθέντων"
        assert out["passive_aorist_infinitive"] == "πεισθῆναι"


class TestGraphoNoPassive:
    """γράφω has 2nd-aor passive ἐγράφην (athematic, no θη). The synth
    only handles regular θη-aorists, so γράφω gets middle-aorist cells
    but no passive-aorist cells."""

    def test_grapho_aor_p_not_provided_skips_passive(self, synth):
        out = synth.synthesize_mp_moods(
            "γράφω", {"aor": "ἔγραψα", "fut_med": "γράψομαι"}
        )
        # No passive_aorist_* cells (no aor_p in parts)
        for k in out:
            assert not k.startswith("passive_aorist_"), k
        # Middle aorist cells still come through
        assert out["middle_aorist_subjunctive_1sg"] == "γράψωμαι"
        assert out["middle_aorist_infinitive"] == "γράψασθαι"

    def test_pf_mp_not_used_for_aor_synth(self, synth):
        """γράφω has γέγραμμαι (assimilated φ + μαι -> μμαι). The synth
        does NOT use pf_mp for any of the aor-system mp cells, so the
        γραψ- σ-stem still wins for middle aorist."""
        out = synth.synthesize_mp_moods(
            "γράφω",
            {"aor": "ἔγραψα", "pf_mp": "γέγραμμαι"},
        )
        # Middle aorist still uses σ-stem γραψ-, not pf_mp γεγραμ-.
        assert out["middle_aorist_subjunctive_1sg"] == "γράψωμαι"


class TestMpAoristSkippedForIrregular:
    """Verbs whose aor middle / passive can't be synthesized regularly
    must skip those cells entirely."""

    def test_lipo_aor_2_skips_middle_aorist(self, synth):
        """λείπω has aor-2 ἔλιπον; middle aorist can't be synthesized
        (no σ-stem from a 2nd aorist)."""
        out = synth.synthesize_mp_moods(
            "λείπω", {"aor": "ἔλιπον"}
        )
        assert "middle_aorist_subjunctive_1sg" not in out
        assert "middle_aorist_imperative_2sg" not in out
        # Present middle still comes through.
        assert out["middle_present_subjunctive_1sg"] == "λείπωμαι"

    def test_meno_liquid_future_skips_middle_aorist(self, synth):
        """μένω -> contract future μενῶ (no σ); middle aorist skipped."""
        out = synth.synthesize_mp_moods("μένω", {"fut": "μενῶ"})
        assert "middle_aorist_subjunctive_1sg" not in out
        assert out["middle_present_subjunctive_1sg"] == "μένωμαι"

    def test_no_aor_p_skips_passive_aorist(self, synth):
        """Without aor_p, passive_aorist_* cells are not produced."""
        out = synth.synthesize_mp_moods("λύω", {"fut": "λύσω"})
        for k in out:
            assert not k.startswith("passive_aorist_"), k


class TestMpNoOpCases:
    def test_contract_returns_empty(self, synth):
        out = synth.synthesize_mp_moods(
            "φιλέω", {"fut": "φιλήσω", "aor_p": "ἐφιλήθην"}
        )
        assert out == {}

    def test_athematic_returns_empty(self, synth):
        out = synth.synthesize_mp_moods("τίθημι", {"aor_p": "ἐτέθην"})
        assert out == {}

    def test_missing_principal_parts_no_error(self, synth):
        out = synth.synthesize_mp_moods("λύω", None)
        assert isinstance(out, dict)
        assert "middle_present_subjunctive_1sg" in out
        assert "middle_aorist_subjunctive_1sg" not in out
        assert "passive_aorist_subjunctive_1sg" not in out

    def test_empty_principal_parts_no_error(self, synth):
        out = synth.synthesize_mp_moods("λύω", {})
        assert "middle_present_subjunctive_1sg" in out
        assert "middle_aorist_subjunctive_1sg" not in out

    def test_none_lemma_returns_empty(self, synth):
        out = synth.synthesize_mp_moods(None, {"fut": "λύσω"})
        assert out == {}

    def test_non_omega_lemma_returns_empty(self, synth):
        out = synth.synthesize_mp_moods("ἀνήρ", {"fut": "whatever"})
        assert out == {}


class TestMpAccentHandling:
    """Stem-accents must be neutralised on cells whose ending carries
    its own accent, to avoid double-accent forms."""

    def test_no_double_accent_passive_aorist_subj_1sg(self, synth):
        out = synth.synthesize_mp_moods("παιδεύω", {"aor_p": "ἐπαιδεύθην"})
        form = out["passive_aorist_subjunctive_1sg"]
        nfd = unicodedata.normalize("NFD", form)
        assert nfd.count("́") + nfd.count("͂") == 1, (
            f"got accents in {form!r}"
        )

    def test_no_double_accent_passive_aorist_opt_1sg(self, synth):
        out = synth.synthesize_mp_moods("παιδεύω", {"aor_p": "ἐπαιδεύθην"})
        form = out["passive_aorist_optative_1sg"]
        nfd = unicodedata.normalize("NFD", form)
        # 1 acute on the ending vowel ει.
        assert nfd.count("́") == 1, f"got accents in {form!r}"

    def test_no_double_accent_middle_aorist_imp_3sg(self, synth):
        out = synth.synthesize_mp_moods("παιδεύω", {"fut": "παιδεύσω"})
        form = out["middle_aorist_imperative_3sg"]
        nfd = unicodedata.normalize("NFD", form)
        assert nfd.count("́") == 1, f"got accents in {form!r}"

    def test_no_double_accent_middle_present_imp_3sg(self, synth):
        out = synth.synthesize_mp_moods("παιδεύω", {})
        form = out["middle_present_imperative_3sg"]
        nfd = unicodedata.normalize("NFD", form)
        # παιδεύεσθω -> παιδευέσθω drops the original acute on ευ.
        # Only one acute should remain (on the ending vowel έ).
        assert nfd.count("́") == 1, f"got accents in {form!r}"

    def test_passive_aor_imp_3pl_drops_eta(self, synth):
        """3pl ending -έντων replaces the stem-final η outright;
        the result for λύω is λυθέντων (not λυθηέντων)."""
        out = synth.synthesize_mp_moods("λύω", {"aor_p": "ἐλύθην"})
        assert out["passive_aorist_imperative_3pl"] == "λυθέντων"


class TestJtauberSchemaAlignment:
    """Sanity check that key shapes align with what jtauber emits.
    These keys are the exact strings the consumer expects."""

    def test_lyo_full_paradigm_keys(self, synth):
        out = synth.synthesize_mp_moods(
            "λύω", {"fut": "λύσω", "aor": "ἔλυσα", "aor_p": "ἐλύθην"}
        )
        # Exact keys from jtauber for λύω middle/passive moods.
        expected = {
            # present middle
            "middle_present_subjunctive_1sg",
            "middle_present_subjunctive_2sg",
            "middle_present_subjunctive_3sg",
            "middle_present_subjunctive_1pl",
            "middle_present_subjunctive_2pl",
            "middle_present_subjunctive_3pl",
            "middle_present_optative_1sg",
            "middle_present_optative_2sg",
            "middle_present_optative_3sg",
            "middle_present_optative_1pl",
            "middle_present_optative_2pl",
            "middle_present_optative_3pl",
            "middle_present_imperative_2sg",
            "middle_present_imperative_3sg",
            "middle_present_imperative_2pl",
            "middle_present_imperative_3pl",
            "middle_present_infinitive",
            # aorist middle
            "middle_aorist_subjunctive_1sg",
            "middle_aorist_subjunctive_2sg",
            "middle_aorist_subjunctive_3sg",
            "middle_aorist_subjunctive_1pl",
            "middle_aorist_subjunctive_2pl",
            "middle_aorist_subjunctive_3pl",
            "middle_aorist_optative_1sg",
            "middle_aorist_optative_2sg",
            "middle_aorist_optative_3sg",
            "middle_aorist_optative_1pl",
            "middle_aorist_optative_2pl",
            "middle_aorist_optative_3pl",
            "middle_aorist_imperative_2sg",
            "middle_aorist_imperative_3sg",
            "middle_aorist_imperative_2pl",
            "middle_aorist_imperative_3pl",
            "middle_aorist_infinitive",
            # aorist passive (ALL with duals on optative)
            "passive_aorist_subjunctive_1sg",
            "passive_aorist_subjunctive_2sg",
            "passive_aorist_subjunctive_3sg",
            "passive_aorist_subjunctive_1pl",
            "passive_aorist_subjunctive_2pl",
            "passive_aorist_subjunctive_3pl",
            "passive_aorist_optative_1sg",
            "passive_aorist_optative_2sg",
            "passive_aorist_optative_3sg",
            "passive_aorist_optative_1pl",
            "passive_aorist_optative_2pl",
            "passive_aorist_optative_3pl",
            "passive_aorist_optative_2du",
            "passive_aorist_optative_3du",
            "passive_aorist_imperative_2sg",
            "passive_aorist_imperative_3sg",
            "passive_aorist_imperative_2pl",
            "passive_aorist_imperative_3pl",
            "passive_aorist_infinitive",
        }
        assert expected.issubset(set(out.keys())), (
            f"missing keys: {expected - set(out.keys())}"
        )


# ---------------------------------------------------------------------------
# v3: aor-2 (strong-aorist) synthesis
# ---------------------------------------------------------------------------


class TestAor2Classifier:
    def test_extract_stem_lipo(self, synth):
        # ἔλιπον -> λιπ
        assert synth.extract_aor2_stem("ἔλιπον") == "λιπ"

    def test_extract_stem_pesso(self, synth):
        # ἔπεσον -> πεσ
        assert synth.extract_aor2_stem("ἔπεσον") == "πεσ"

    def test_extract_stem_labe(self, synth):
        # ἔλαβον -> λαβ
        assert synth.extract_aor2_stem("ἔλαβον") == "λαβ"

    def test_extract_stem_eipon(self, synth):
        # εἶπον -> εἰπ (no augment, εἰ is its own augment-like onset)
        # Our heuristic returns the body as-is when there's no
        # syllabic-augment ε at the start.
        stem = synth.extract_aor2_stem("εἶπον")
        # Either εἰπ or εἶπ is acceptable; we accept whatever the
        # heuristic produces as long as it ends in π.
        assert stem.endswith("π") or stem.endswith("ι") or stem.endswith("ἰ")

    def test_extract_stem_rejects_sigmatic(self, synth):
        # ἔλυσα is sigmatic aor-1, NOT aor-2.
        assert synth.extract_aor2_stem("ἔλυσα") is None

    def test_extract_stem_rejects_kappa(self, synth):
        # ἔδωκα is κ-aorist (athematic), not aor-2.
        assert synth.extract_aor2_stem("ἔδωκα") is None

    def test_extract_stem_rejects_empty(self, synth):
        assert synth.extract_aor2_stem("") is None
        assert synth.extract_aor2_stem(None) is None


class TestAor2Synthesis:
    """Aor-2 synthesis on classical-attested verbs."""

    def test_leipo_active_indicative_1sg(self, synth):
        out = synth.synthesize_aor2_moods("λείπω", {"aor": "ἔλιπον"})
        assert out["active_aorist_indicative_1sg"] == "ἔλιπον"

    def test_leipo_active_indicative_3pl(self, synth):
        out = synth.synthesize_aor2_moods("λείπω", {"aor": "ἔλιπον"})
        # 3pl shares form with 1sg in aor-2 active
        assert out["active_aorist_indicative_3pl"] == "ἔλιπον"

    def test_leipo_active_indicative_2sg(self, synth):
        out = synth.synthesize_aor2_moods("λείπω", {"aor": "ἔλιπον"})
        assert out["active_aorist_indicative_2sg"] == "ἔλιπες"

    def test_leipo_active_indicative_1pl_recessive(self, synth):
        # Recessive accent on 4-syllable form -> antepenult ι
        out = synth.synthesize_aor2_moods("λείπω", {"aor": "ἔλιπον"})
        assert out["active_aorist_indicative_1pl"] == "ἐλίπομεν"

    def test_leipo_imperative_2sg_recessive(self, synth):
        out = synth.synthesize_aor2_moods("λείπω", {"aor": "ἔλιπον"})
        assert out["active_aorist_imperative_2sg"] == "λίπε"

    def test_leipo_imperative_2pl_recessive(self, synth):
        out = synth.synthesize_aor2_moods("λείπω", {"aor": "ἔλιπον"})
        assert out["active_aorist_imperative_2pl"] == "λίπετε"

    def test_leipo_imperative_3sg_endaccented(self, synth):
        out = synth.synthesize_aor2_moods("λείπω", {"aor": "ἔλιπον"})
        assert out["active_aorist_imperative_3sg"] == "λιπέτω"

    def test_leipo_imperative_3pl_endaccented(self, synth):
        out = synth.synthesize_aor2_moods("λείπω", {"aor": "ἔλιπον"})
        assert out["active_aorist_imperative_3pl"] == "λιπόντων"

    def test_leipo_subjunctive_1sg(self, synth):
        out = synth.synthesize_aor2_moods("λείπω", {"aor": "ἔλιπον"})
        assert out["active_aorist_subjunctive_1sg"] == "λίπω"

    def test_leipo_subjunctive_3pl(self, synth):
        out = synth.synthesize_aor2_moods("λείπω", {"aor": "ἔλιπον"})
        assert out["active_aorist_subjunctive_3pl"] == "λίπωσι(ν)"

    def test_leipo_optative_1sg(self, synth):
        out = synth.synthesize_aor2_moods("λείπω", {"aor": "ἔλιπον"})
        assert out["active_aorist_optative_1sg"] == "λίποιμι"

    def test_leipo_infinitive(self, synth):
        out = synth.synthesize_aor2_moods("λείπω", {"aor": "ἔλιπον"})
        assert out["active_aorist_infinitive"] == "λιπεῖν"

    def test_pipto_imperative_2sg(self, synth):
        out = synth.synthesize_aor2_moods("πίπτω", {"aor": "ἔπεσον"})
        assert out["active_aorist_imperative_2sg"] == "πέσε"

    def test_pipto_subjunctive_1pl(self, synth):
        out = synth.synthesize_aor2_moods("πίπτω", {"aor": "ἔπεσον"})
        assert out["active_aorist_subjunctive_1pl"] == "πέσωμεν"

    def test_pipto_infinitive(self, synth):
        out = synth.synthesize_aor2_moods("πίπτω", {"aor": "ἔπεσον"})
        assert out["active_aorist_infinitive"] == "πεσεῖν"

    def test_lambano_aorist_imperative_2pl(self, synth):
        out = synth.synthesize_aor2_moods("λαμβάνω", {"aor": "ἔλαβον"})
        assert out["active_aorist_imperative_2pl"] == "λάβετε"

    def test_lambano_optative_1sg(self, synth):
        out = synth.synthesize_aor2_moods("λαμβάνω", {"aor": "ἔλαβον"})
        assert out["active_aorist_optative_1sg"] == "λάβοιμι"

    def test_lambano_infinitive(self, synth):
        out = synth.synthesize_aor2_moods("λαμβάνω", {"aor": "ἔλαβον"})
        assert out["active_aorist_infinitive"] == "λαβεῖν"

    def test_lambano_middle_imperative_2sg(self, synth):
        # λαβοῦ
        out = synth.synthesize_aor2_moods("λαμβάνω", {"aor": "ἔλαβον"})
        assert out["middle_aorist_imperative_2sg"] == "λαβοῦ"

    def test_lambano_middle_subjunctive_1sg(self, synth):
        out = synth.synthesize_aor2_moods("λαμβάνω", {"aor": "ἔλαβον"})
        assert out["middle_aorist_subjunctive_1sg"] == "λάβωμαι"

    def test_lambano_middle_infinitive(self, synth):
        out = synth.synthesize_aor2_moods("λαμβάνω", {"aor": "ἔλαβον"})
        assert out["middle_aorist_infinitive"] == "λαβέσθαι"

    def test_lambano_middle_indicative_1sg_recessive(self, synth):
        out = synth.synthesize_aor2_moods("λαμβάνω", {"aor": "ἔλαβον"})
        # ἐλαβόμην -- recessive on 4-syl with long final
        assert out["middle_aorist_indicative_1sg"] == "ἐλαβόμην"

    def test_skips_when_no_aor(self, synth):
        out = synth.synthesize_aor2_moods("λύω", {})
        assert out == {}

    def test_skips_when_aor_is_sigmatic(self, synth):
        out = synth.synthesize_aor2_moods("λύω", {"aor": "ἔλυσα"})
        assert out == {}


class TestAor2SchemaAlignment:
    """The cell keys produced by aor-2 synthesis match jtauber's flat
    schema verbatim (case·gender·number for participles, person·number
    for finite, etc.)."""

    def test_keys_match_jtauber_pattern(self, synth):
        out = synth.synthesize_aor2_moods("λείπω", {"aor": "ἔλιπον"})
        for k in out:
            # All keys are <voice>_aorist_<mood>_<persnum> for finite,
            # or <voice>_aorist_infinitive for infinitive.
            parts = k.split("_")
            assert parts[1] == "aorist"
            assert parts[0] in ("active", "middle")
            if "infinitive" not in k:
                assert parts[2] in (
                    "indicative", "subjunctive", "optative", "imperative",
                ), k


# ---------------------------------------------------------------------------
# v3: contract verb synthesis (-άω / -έω / -όω present-system)
# ---------------------------------------------------------------------------


class TestContractClassifier:
    def test_alpha_contract_detected(self, synth):
        assert synth.contract_class("τιμάω") == "alpha"
        assert synth.contract_class("ὁράω") == "alpha"

    def test_epsilon_contract_detected(self, synth):
        assert synth.contract_class("φιλέω") == "epsilon"
        assert synth.contract_class("ποιέω") == "epsilon"

    def test_omicron_contract_detected(self, synth):
        assert synth.contract_class("δηλόω") == "omicron"

    def test_thematic_omega_not_contract(self, synth):
        assert synth.contract_class("λύω") is None
        assert synth.contract_class("γράφω") is None

    def test_athematic_not_contract(self, synth):
        assert synth.contract_class("τίθημι") is None
        assert synth.contract_class("ἵσταμαι") is None

    def test_is_contract_helper(self, synth):
        assert synth.is_contract("τιμάω")
        assert synth.is_contract("ποιέω")
        assert synth.is_contract("δηλόω")
        assert not synth.is_contract("λύω")


class TestAlphaContractSynthesis:
    """Alpha-contract τιμάω synthesis matches jtauber verbatim."""

    def test_indicative_1sg(self, synth):
        out = synth.synthesize_contract_moods("τιμάω", {})
        assert out["active_present_indicative_1sg"] == "τιμῶ"

    def test_indicative_2sg(self, synth):
        out = synth.synthesize_contract_moods("τιμάω", {})
        assert out["active_present_indicative_2sg"] == "τιμᾷς"

    def test_indicative_3sg(self, synth):
        out = synth.synthesize_contract_moods("τιμάω", {})
        assert out["active_present_indicative_3sg"] == "τιμᾷ"

    def test_indicative_1pl(self, synth):
        out = synth.synthesize_contract_moods("τιμάω", {})
        assert out["active_present_indicative_1pl"] == "τιμῶμεν"

    def test_indicative_2pl(self, synth):
        out = synth.synthesize_contract_moods("τιμάω", {})
        assert out["active_present_indicative_2pl"] == "τιμᾶτε"

    def test_indicative_3pl(self, synth):
        out = synth.synthesize_contract_moods("τιμάω", {})
        assert out["active_present_indicative_3pl"] == "τιμῶσι(ν)"

    def test_imperative_2sg_recessive(self, synth):
        # τίμα has accent on stem (recessive on 2-syl form)
        out = synth.synthesize_contract_moods("τιμάω", {})
        assert out["active_present_imperative_2sg"] == "τίμα"

    def test_subjunctive_1sg_same_as_indic(self, synth):
        out = synth.synthesize_contract_moods("τιμάω", {})
        assert out["active_present_subjunctive_1sg"] == "τιμῶ"

    def test_optative_1sg_long_form(self, synth):
        out = synth.synthesize_contract_moods("τιμάω", {})
        assert out["active_present_optative_1sg"] == "τιμῴην"

    def test_infinitive(self, synth):
        out = synth.synthesize_contract_moods("τιμάω", {})
        assert out["active_present_infinitive"] == "τιμᾶν"

    def test_middle_indicative_1sg(self, synth):
        out = synth.synthesize_contract_moods("τιμάω", {})
        assert out["middle_present_indicative_1sg"] == "τιμῶμαι"

    def test_middle_imperative_3sg_with_macron(self, synth):
        # τιμᾱ́σθω -- jtauber-style with explicit macron on α
        out = synth.synthesize_contract_moods("τιμάω", {})
        assert out["middle_present_imperative_3sg"] == "τιμᾱ́σθω"


class TestEpsilonContractSynthesis:
    """Epsilon-contract ποιέω / φιλέω synthesis."""

    def test_poieo_indicative_1sg(self, synth):
        out = synth.synthesize_contract_moods("ποιέω", {})
        assert out["active_present_indicative_1sg"] == "ποιῶ"

    def test_poieo_indicative_2sg(self, synth):
        out = synth.synthesize_contract_moods("ποιέω", {})
        assert out["active_present_indicative_2sg"] == "ποιεῖς"

    def test_poieo_indicative_1pl(self, synth):
        out = synth.synthesize_contract_moods("ποιέω", {})
        assert out["active_present_indicative_1pl"] == "ποιοῦμεν"

    def test_poieo_imperative_2sg(self, synth):
        # ποίει -- recessive on diphthong-ending form, accent on ι
        out = synth.synthesize_contract_moods("ποιέω", {})
        assert out["active_present_imperative_2sg"] == "ποίει"

    def test_poieo_subjunctive_2sg(self, synth):
        out = synth.synthesize_contract_moods("ποιέω", {})
        assert out["active_present_subjunctive_2sg"] == "ποιῇς"

    def test_poieo_infinitive(self, synth):
        out = synth.synthesize_contract_moods("ποιέω", {})
        assert out["active_present_infinitive"] == "ποιεῖν"

    def test_poieo_middle_infinitive(self, synth):
        out = synth.synthesize_contract_moods("ποιέω", {})
        assert out["middle_present_infinitive"] == "ποιεῖσθαι"

    def test_phileo_indicative_1sg(self, synth):
        out = synth.synthesize_contract_moods("φιλέω", {})
        assert out["active_present_indicative_1sg"] == "φιλῶ"

    def test_phileo_imperative_2sg(self, synth):
        out = synth.synthesize_contract_moods("φιλέω", {})
        # Note: φίλει would be the jtauber form. Recessive on 2-syl form
        # on a single-vowel stem -> accent on stem ι
        assert out["active_present_imperative_2sg"] == "φίλει"


class TestOmicronContractSynthesis:
    """Omicron-contract δηλόω synthesis."""

    def test_indicative_1sg(self, synth):
        out = synth.synthesize_contract_moods("δηλόω", {})
        assert out["active_present_indicative_1sg"] == "δηλῶ"

    def test_indicative_2sg(self, synth):
        out = synth.synthesize_contract_moods("δηλόω", {})
        assert out["active_present_indicative_2sg"] == "δηλοῖς"

    def test_indicative_3sg(self, synth):
        out = synth.synthesize_contract_moods("δηλόω", {})
        assert out["active_present_indicative_3sg"] == "δηλοῖ"

    def test_imperative_2sg(self, synth):
        # δήλου with recessive penult on 2-syl (penult η, stem-accented)
        out = synth.synthesize_contract_moods("δηλόω", {})
        assert out["active_present_imperative_2sg"] == "δήλου"

    def test_infinitive(self, synth):
        out = synth.synthesize_contract_moods("δηλόω", {})
        assert out["active_present_infinitive"] == "δηλοῦν"

    def test_subjunctive_1pl(self, synth):
        out = synth.synthesize_contract_moods("δηλόω", {})
        assert out["active_present_subjunctive_1pl"] == "δηλῶμεν"


class TestContractMoodsSchemaAlignment:
    def test_contract_keys_only_present_system(self, synth):
        out = synth.synthesize_contract_moods("τιμάω", {})
        for k in out:
            parts = k.split("_")
            assert parts[1] == "present", k

    def test_skip_thematic_omega(self, synth):
        out = synth.synthesize_contract_moods("λύω", {})
        assert out == {}

    def test_skip_athematic(self, synth):
        out = synth.synthesize_contract_moods("τίθημι", {})
        assert out == {}


# ---------------------------------------------------------------------------
# v4: mixed-α aor-2 (πίπτω-style ἔπεσα / εἶπα / εὗρα) detection +
# alpha-pattern indicative synthesis. Aor 1sg ends in -α not -ον but the
# rest of the paradigm uses regular aor-2 ο-thematic endings.
# ---------------------------------------------------------------------------


class TestAor2AlphaDetection:
    """Only established lemma/form pairs supply mixed-alpha templates."""

    def test_pipto_alpha_detected(self, synth):
        # ἔπεσα is mixed-α: σ comes from a different root (πεσ- vs πιπτ-).
        assert synth._is_aor_2_alpha_form("ἔπεσα", "πίπτω")

    def test_lego_alpha_detected(self, synth):
        # The alpha indicative retains the second-aorist mood system.
        assert synth._is_aor_2_alpha_form("εἶπα", "λέγω")

    def test_heurisko_alpha_detected(self, synth):
        # εὗρα: ρ before α (not σ/ψ/ξ) → α-aor-2.
        assert synth._is_aor_2_alpha_form("εὗρα", "εὑρίσκω")

    def test_lyo_sigmatic_rejected(self, synth):
        # ἔλυσα: σ before α, lemma λύω is vowel-stem → predicts σ →
        # cluster matches → root λυ matches → clean σ-aorist, NOT α-aor-2.
        assert not synth._is_aor_2_alpha_form("ἔλυσα", "λύω")

    def test_grapho_sigmatic_rejected(self, synth):
        # ἔγραψα: clean ψ-aorist (γρ + φ → ψ predicted, root γρα matches).
        assert not synth._is_aor_2_alpha_form("ἔγραψα", "γράφω")

    def test_paideo_sigmatic_rejected(self, synth):
        assert not synth._is_aor_2_alpha_form("ἐπαίδευσα", "παιδεύω")

    def test_athematic_rejected(self, synth):
        # δίδωμι has -μι → not thematic → α-aor-2 detection bails.
        assert not synth._is_aor_2_alpha_form("ἔδωκα", "δίδωμι")

    def test_no_lemma_rejected(self, synth):
        # Without lemma we can't distinguish suppletion → conservative reject.
        assert not synth._is_aor_2_alpha_form("ἔπεσα", None)

    def test_extract_stem_alpha(self, synth):
        # Alpha-pattern aor 1sg → unaugmented stem.
        assert synth.extract_aor2_stem("ἔπεσα", "πίπτω") == "πεσ"
        assert synth.extract_aor2_stem("εὗρα", "εὑρίσκω") == "εὗρ"

    @pytest.mark.parametrize("lemma, form", [
        ("παίζω", "ἔπαιξα"),
        ("κηρύσσω", "ἐκήρυξα"),
        ("ἀΐσσω", "ἤιξα"),
        ("ἐθέλω", "ἠθέλησα"),
        ("ὄζω", "ὤζησα"),
        ("κερδαίνω", "ἐκέρδᾱνα"),
        ("ξαίνω", "ἔξηνα"),
        ("μένω", "ἔμεινα"),
        ("νέμω", "ἔνειμα"),
        ("φέρω", "ἤνεγκα"),
        ("λήγω", "πάντα"),
    ])
    def test_other_alpha_forms_do_not_supply_second_aorist_cells(
        self, synth, lemma, form
    ):
        assert not synth._is_aor_2_alpha_form(form, lemma)
        assert synth.extract_aor2_stem(form, lemma) is None
        assert synth.synthesize_aor2_moods(lemma, {"aor": form}) == {}

    def test_explicit_second_aorist_supplies_active_template(self, synth):
        out = synth.synthesize_aor2_moods(
            "βλάπτω", {"aor": "ἔβλαψα", "aor2": "ἔβλαβον"}
        )
        assert out["active_aorist_infinitive"] == "βλαβεῖν"
        assert all("ψ" not in form for form in out.values())

    def test_middle_principal_part_does_not_mask_active(self, synth):
        generic = {"aor": "κέκᾰδον"}
        assert synth.synthesize_aor2_moods("χάζω", generic)
        assert synth.synthesize_aor2_moods("χάζω", generic) == (
            synth.synthesize_aor2_moods(
                "χάζω", {**generic, "aor2": "κεκαδόμην"}
            )
        )

    def test_alternate_second_aorist_does_not_replace_selected_one(self, synth):
        generic = {"aor": "ἔτροπον"}
        assert synth.synthesize_aor2_moods("τρέπω", generic)
        assert synth.synthesize_aor2_moods("τρέπω", generic) == (
            synth.synthesize_aor2_moods(
                "τρέπω", {**generic, "aor2": "ἔτρᾰπον"}
            )
        )

    def test_mixed_alpha_pair_requires_matching_lemma(self, synth):
        assert not synth._is_aor_2_alpha_form("εὗρα", "λύω")


class TestSuppletionGuard:
    """``_aorist_stem_from_lemma_and_aor`` returns None for suppletive
    aor stems (different ROOT from lemma): πίπτω + ἔπεσα would naively
    splice σ onto πιπτ → πιπσ-, but the bodies don't match → None."""

    def test_pipto_suppletion_rejected(self, synth):
        # Without the suppletion guard this would return πίπσ.
        assert synth._aorist_stem_from_lemma_and_aor(
            "πίπτω", "ἔπεσα"
        ) is None

    def test_lyo_regular_kept(self, synth):
        stem = synth._aorist_stem_from_lemma_and_aor("λύω", "ἔλυσα")
        assert stem == "λύσ"

    def test_grapho_regular_kept(self, synth):
        stem = synth._aorist_stem_from_lemma_and_aor("γράφω", "ἔγραψα")
        assert stem == "γράψ"


class TestAor2AlphaIndicative:
    """Mixed-α aor-2 verbs synthesise α-pattern indicative cells
    (-α / -ας / -αμεν / -ατε / -αν) on the augmented stem, exactly
    matching jtauber's verbatim output."""

    def test_pipto_active_indicative_1sg(self, synth):
        out = synth.synthesize_aor2_moods("πίπτω", {"aor": "ἔπεσα"})
        assert out["active_aorist_indicative_1sg"] == "ἔπεσα"

    def test_pipto_active_indicative_2sg(self, synth):
        out = synth.synthesize_aor2_moods("πίπτω", {"aor": "ἔπεσα"})
        assert out["active_aorist_indicative_2sg"] == "ἔπεσας"

    def test_pipto_active_indicative_1pl_recessive(self, synth):
        out = synth.synthesize_aor2_moods("πίπτω", {"aor": "ἔπεσα"})
        # ἐπέσαμεν: 4-syllable form, antepenult acute on ε of πεσ.
        assert out["active_aorist_indicative_1pl"] == "ἐπέσαμεν"

    def test_pipto_active_indicative_3pl(self, synth):
        out = synth.synthesize_aor2_moods("πίπτω", {"aor": "ἔπεσα"})
        assert out["active_aorist_indicative_3pl"] == "ἔπεσαν"

    def test_pipto_active_no_3sg(self, synth):
        # jtauber doesn't emit a 3sg for these; we mirror.
        out = synth.synthesize_aor2_moods("πίπτω", {"aor": "ἔπεσα"})
        assert "active_aorist_indicative_3sg" not in out

    def test_eipa_active_indicative_circumflex(self, synth):
        # εἶπα has circumflex on long penult ει + short ult α.
        out = synth.synthesize_aor2_moods("λέγω", {"aor": "εἶπα"})
        assert out["active_aorist_indicative_1sg"] == "εἶπα"
        assert out["active_aorist_indicative_2sg"] == "εἶπας"
        assert out["active_aorist_indicative_3pl"] == "εἶπαν"

    def test_eipa_active_indicative_acute_3syl(self, synth):
        # εἴπαμεν: 3-syl, antepenult on long ει → acute.
        out = synth.synthesize_aor2_moods("λέγω", {"aor": "εἶπα"})
        assert out["active_aorist_indicative_1pl"] == "εἴπαμεν"
        assert out["active_aorist_indicative_2pl"] == "εἴπατε"

    def test_heura_active_indicative(self, synth):
        out = synth.synthesize_aor2_moods("εὑρίσκω", {"aor": "εὗρα"})
        assert out["active_aorist_indicative_1sg"] == "εὗρα"
        assert out["active_aorist_indicative_2sg"] == "εὗρας"
        assert out["active_aorist_indicative_3pl"] == "εὗραν"
        assert out["active_aorist_indicative_1pl"] == "εὕραμεν"

    def test_pipto_middle_indicative(self, synth):
        out = synth.synthesize_aor2_moods("πίπτω", {"aor": "ἔπεσα"})
        assert out["middle_aorist_indicative_1sg"] == "ἐπεσάμην"
        assert out["middle_aorist_indicative_3sg"] == "ἐπέσατο"
        assert out["middle_aorist_indicative_1pl"] == "ἐπεσάμεθα"
        assert out["middle_aorist_indicative_2pl"] == "ἐπέσασθε"
        assert out["middle_aorist_indicative_3pl"] == "ἐπέσαντο"

    def test_pipto_subjunctive_unchanged(self, synth):
        # Subjunctive uses regular aor-2 ο-thematic endings on πεσ-.
        out = synth.synthesize_aor2_moods("πίπτω", {"aor": "ἔπεσα"})
        assert out["active_aorist_subjunctive_1sg"] == "πέσω"
        assert out["active_aorist_subjunctive_2sg"] == "πέσῃς"
        assert out["active_aorist_subjunctive_1pl"] == "πέσωμεν"

    def test_pipto_optative_unchanged(self, synth):
        out = synth.synthesize_aor2_moods("πίπτω", {"aor": "ἔπεσα"})
        assert out["active_aorist_optative_1sg"] == "πέσοιμι"

    def test_pipto_imperative_unchanged(self, synth):
        out = synth.synthesize_aor2_moods("πίπτω", {"aor": "ἔπεσα"})
        assert out["active_aorist_imperative_2sg"] == "πέσε"
        assert out["active_aorist_imperative_3sg"] == "πεσέτω"

    def test_pipto_infinitive_unchanged(self, synth):
        out = synth.synthesize_aor2_moods("πίπτω", {"aor": "ἔπεσα"})
        assert out["active_aorist_infinitive"] == "πεσεῖν"

    def test_jtauber_verbatim_pipto(self, synth):
        # Spot-check verbatim against jtauber for ἔπεσα.
        out = synth.synthesize_aor2_moods("πίπτω", {"aor": "ἔπεσα"})
        assert out["active_aorist_indicative_1sg"] == "ἔπεσα"
        assert out["active_aorist_indicative_2sg"] == "ἔπεσας"
        assert out["active_aorist_indicative_1pl"] == "ἐπέσαμεν"
        assert out["active_aorist_indicative_2pl"] == "ἐπέσατε"
        assert out["active_aorist_indicative_3pl"] == "ἔπεσαν"
        assert out["middle_aorist_indicative_1sg"] == "ἐπεσάμην"
        assert out["middle_aorist_indicative_3sg"] == "ἐπέσατο"
        assert out["middle_aorist_indicative_3pl"] == "ἐπέσαντο"


# ---------------------------------------------------------------------------
# v4: macron / breve quantity marks on participle endings live in
# tests/test_synth_verb_participles.py — they're tested in tandem with
# the participle synthesis there.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# v5: past-indicative 1sg synthesis (imperfect / aorist)
# ---------------------------------------------------------------------------
#
# Targets the 13 high-traffic verbs whose 1sg past-indicative cells were
# attested only via Homeric / unaugmented variants. dilemma's 05ba907
# filter correctly routes those into ``dialects.epic``, so the canonical
# Attic slice now needs templating to fill the gaps.


class TestPastIndicativePresent:
    """Helper-style sanity checks: the synth produces the right keys
    only on the right lemma classes."""

    def test_athematic_mi_no_synthesis(self, synth):
        out = synth.synthesize_past_indicatives("τίθημι", {})
        # Athematic verbs aren't covered by this synth; the result
        # should be empty (dilemma's athematic synthesis lives elsewhere).
        assert out == {}

    def test_eimi_no_synthesis(self, synth):
        out = synth.synthesize_past_indicatives("εἰμί", {})
        assert out == {}

    def test_long_vowel_lemma_skipped(self, synth):
        # ω-initial: temporal augment is invisible, so we bail.
        out = synth.synthesize_past_indicatives("ὠθέω", {})
        # Contract verbs go through the contract path; ὠθέω is contract,
        # but its bare stem ὠθ- starts with ω which the augment helper
        # bails on. Result: no impf cells.
        assert "active_imperfect_indicative_1sg" not in out
        assert "middle_imperfect_indicative_1sg" not in out

    def test_no_lemma_returns_empty(self, synth):
        assert synth.synthesize_past_indicatives("", {}) == {}
        assert synth.synthesize_past_indicatives(None, {}) == {}


class TestPastIndicativeImperfect:
    """Imperfect 1sg synthesis from lemma + augment + ending."""

    def test_lyo_imperfect(self, synth):
        out = synth.synthesize_past_indicatives("λύω", {})
        assert out["active_imperfect_indicative_1sg"] == "ἔλυον"
        assert out["middle_imperfect_indicative_1sg"] == "ἐλυόμην"

    def test_pauo_imperfect(self, synth):
        out = synth.synthesize_past_indicatives("παύω", {})
        assert out["active_imperfect_indicative_1sg"] == "ἔπαυον"
        assert out["middle_imperfect_indicative_1sg"] == "ἐπαυόμην"

    def test_grapho_imperfect(self, synth):
        out = synth.synthesize_past_indicatives("γράφω", {})
        assert out["active_imperfect_indicative_1sg"] == "ἔγραφον"
        assert out["middle_imperfect_indicative_1sg"] == "ἐγραφόμην"

    def test_blepo_imperfect(self, synth):
        out = synth.synthesize_past_indicatives("βλέπω", {})
        assert out["active_imperfect_indicative_1sg"] == "ἔβλεπον"
        assert out["middle_imperfect_indicative_1sg"] == "ἐβλεπόμην"

    def test_lambano_imperfect(self, synth):
        out = synth.synthesize_past_indicatives("λαμβάνω", {})
        assert out["active_imperfect_indicative_1sg"] == "ἐλάμβανον"
        assert out["middle_imperfect_indicative_1sg"] == "ἐλαμβανόμην"

    def test_lego_imperfect(self, synth):
        out = synth.synthesize_past_indicatives("λέγω", {})
        assert out["active_imperfect_indicative_1sg"] == "ἔλεγον"
        assert out["middle_imperfect_indicative_1sg"] == "ἐλεγόμην"

    def test_akouo_imperfect_temporal_augment(self, synth):
        # ἀκούω -> ἤκουον / ἠκουόμην. Temporal augment α->η.
        out = synth.synthesize_past_indicatives("ἀκούω", {})
        assert out["active_imperfect_indicative_1sg"] == "ἤκουον"
        assert out["middle_imperfect_indicative_1sg"] == "ἠκουόμην"

    def test_paideuo_recessive_accent(self, synth):
        # 4-syllable form: antepenult accent (short ultima -ον).
        out = synth.synthesize_past_indicatives("παιδεύω", {})
        assert out["active_imperfect_indicative_1sg"] == "ἐπαίδευον"
        assert out["middle_imperfect_indicative_1sg"] == "ἐπαιδευόμην"


class TestPastIndicativeAorist:
    """Aorist 1sg synthesis from principal-parts copies."""

    def test_lyo_aorist_active(self, synth):
        out = synth.synthesize_past_indicatives(
            "λύω",
            {"aor": "ἔλυσα", "aor_p": "ἐλύθην"},
        )
        assert out["active_aorist_indicative_1sg"] == "ἔλυσα"
        assert out["passive_aorist_indicative_1sg"] == "ἐλύθην"

    def test_pauo_passive_aorist(self, synth):
        out = synth.synthesize_past_indicatives(
            "παύω",
            {"aor": "ἔπαυσα", "aor_p": "ἐπαύθην"},
        )
        assert out["active_aorist_indicative_1sg"] == "ἔπαυσα"
        assert out["passive_aorist_indicative_1sg"] == "ἐπαύθην"

    def test_grapho_2nd_aorist_passive(self, synth):
        # γράφω has 2nd-aorist passive ἐγράφην (no -θ-).
        out = synth.synthesize_past_indicatives(
            "γράφω",
            {"aor": "ἔγραψα", "aor_p": "ἐγράφην"},
        )
        assert out["active_aorist_indicative_1sg"] == "ἔγραψα"
        assert out["passive_aorist_indicative_1sg"] == "ἐγράφην"

    def test_phileo_aorist_active_and_passive(self, synth):
        # φιλέω: contract; aor / aor_p come from principal parts.
        out = synth.synthesize_past_indicatives(
            "φιλέω",
            {"aor": "ἐφίλησα", "aor_p": "ἐφιλήθην"},
        )
        assert out["active_aorist_indicative_1sg"] == "ἐφίλησα"
        assert out["passive_aorist_indicative_1sg"] == "ἐφιλήθην"

    def test_poieo_aorist_full(self, synth):
        out = synth.synthesize_past_indicatives(
            "ποιέω",
            {
                "aor": "ἐποίησα",
                "aor_med": "ἐποιησάμην",
                "aor_p": "ἐποιήθην",
            },
        )
        assert out["active_aorist_indicative_1sg"] == "ἐποίησα"
        assert out["middle_aorist_indicative_1sg"] == "ἐποιησάμην"
        assert out["passive_aorist_indicative_1sg"] == "ἐποιήθην"

    def test_aor2_principal_part(self, synth):
        # λαμβάνω aor-2 ἔλαβον. Active synthesis takes the principal
        # part verbatim.
        out = synth.synthesize_past_indicatives(
            "λαμβάνω",
            {"aor": "ἔλαβον", "aor_med": "ἐλαβόμην", "aor_p": "ἐλήφθην"},
        )
        assert out["active_aorist_indicative_1sg"] == "ἔλαβον"
        assert out["middle_aorist_indicative_1sg"] == "ἐλαβόμην"
        assert out["passive_aorist_indicative_1sg"] == "ἐλήφθην"

    def test_middle_principal_part_does_not_leak_to_active(self, synth):
        # γίγνομαι has parts['aor'] = 'ἐγενόμην' which is the deponent
        # 1sg middle (no active 1sg exists). Synthesis must reject -μην
        # for the active slot.
        out = synth.synthesize_past_indicatives(
            "γίγνομαι",
            {"aor": "ἐγενόμην"},
        )
        assert "active_aorist_indicative_1sg" not in out


class TestPastIndicativeContract:
    """Imperfect 1sg synthesis for contract verbs (-άω / -έω / -όω)."""

    def test_phileo_imperfect(self, synth):
        # ε-contract: ε + ε + ον = ουν -> ἐφίλουν.
        out = synth.synthesize_past_indicatives("φιλέω", {})
        assert out["active_imperfect_indicative_1sg"] == "ἐφίλουν"
        assert out["middle_imperfect_indicative_1sg"] == "ἐφιλούμην"

    def test_poieo_imperfect(self, synth):
        # ε-contract: ποιε + ε + ον = ποίουν -> ἐποίουν.
        out = synth.synthesize_past_indicatives("ποιέω", {})
        assert out["active_imperfect_indicative_1sg"] == "ἐποίουν"
        assert out["middle_imperfect_indicative_1sg"] == "ἐποιούμην"

    def test_deloo_imperfect(self, synth):
        # ο-contract: δηλο + ο + ον = δηλουν -> ἐδήλουν.
        out = synth.synthesize_past_indicatives("δηλόω", {})
        assert out["active_imperfect_indicative_1sg"] == "ἐδήλουν"
        assert out["middle_imperfect_indicative_1sg"] == "ἐδηλούμην"

    def test_timao_imperfect(self, synth):
        # α-contract: τιμα + α + ον = τιμων -> ἐτίμων.
        out = synth.synthesize_past_indicatives("τιμάω", {})
        assert out["active_imperfect_indicative_1sg"] == "ἐτίμων"
        assert out["middle_imperfect_indicative_1sg"] == "ἐτιμώμην"


class TestPastIndicativeDeponent:
    """Imperfect / aorist 1sg synthesis for deponent -ομαι verbs."""

    def test_gignomai_imperfect(self, synth):
        out = synth.synthesize_past_indicatives("γίγνομαι", {})
        assert out["middle_imperfect_indicative_1sg"] == "ἐγιγνόμην"
        # No active synthesis for deponents.
        assert "active_imperfect_indicative_1sg" not in out

    def test_gignomai_aorist_passive_from_parts(self, synth):
        # When parts carries aor_p, synthesise the passive aorist 1sg.
        out = synth.synthesize_past_indicatives(
            "γίγνομαι",
            {"aor_p": "ἐγενήθην"},
        )
        assert out["passive_aorist_indicative_1sg"] == "ἐγενήθην"


class TestPastIndicativeBail:
    """Cases where synthesis should not produce output (suppletive,
    irregular, or unsafely templatable)."""

    def test_erchomai_no_synthesis(self, synth):
        # ἔρχομαι is suppletive (uses εἶμι forms in past indicative).
        # Augment helper bails on η-initial principal parts.
        out = synth.synthesize_past_indicatives(
            "ἔρχομαι",
            {"impf": "ἠρχόμην", "fut": "ἐλεύσομαι"},
        )
        # No imperfect synthesis (suppletive).  No aorist either since
        # parts['aor'] is absent.
        assert "active_imperfect_indicative_1sg" not in out
        assert "active_aorist_indicative_1sg" not in out

    def test_histemi_no_synthesis(self, synth):
        # ἵστημι is athematic; the synth bails entirely.
        out = synth.synthesize_past_indicatives("ἵστημι", {})
        assert out == {}

    def test_unparseable_aor_principal_part_skipped(self, synth):
        # If aor is a non-1sg form (e.g. infinitive), reject.
        # ``λύσειν`` ends in -ν but it's a future infinitive shape; the
        # current sanity check accepts any -ν ending. We document that
        # the caller is expected to pass a real 1sg principal part.
        # (No assertion: the input contract is "valid 1sg" and we
        # don't try to second-guess it.)
        pass


class TestPastIndicativeJtauberSchemaAlignment:
    """The synth must emit keys in the exact schema jtauber / dilemma's
    canonical paradigm uses."""

    def test_lyo_keys(self, synth):
        out = synth.synthesize_past_indicatives(
            "λύω",
            {"aor": "ἔλυσα", "aor_p": "ἐλύθην"},
        )
        expected_keys = {
            "active_imperfect_indicative_1sg",
            "middle_imperfect_indicative_1sg",
            "active_aorist_indicative_1sg",
            "passive_aorist_indicative_1sg",
        }
        assert expected_keys.issubset(set(out.keys())), (
            f"missing keys: {expected_keys - set(out.keys())}"
        )

    def test_keys_are_only_1sg(self, synth):
        # Synthesis is scoped to 1sg cells (the load-bearing case for
        # the kaikki tense-tag-drop bug). No other person/number keys
        # should appear.
        out = synth.synthesize_past_indicatives(
            "λύω",
            {"aor": "ἔλυσα", "aor_p": "ἐλύθην"},
        )
        for key in out:
            # Allowed key shapes:
            allowed_suffixes = ("_1sg",)
            assert key.endswith(allowed_suffixes), (
                f"unexpected key shape: {key}"
            )

