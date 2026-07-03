"""Tests for ``build/synth_verb_participles.py`` and its integration in
``build/build_grc_verb_paradigms.py``.

The synth module is loaded directly as a script-style module via
``importlib`` (the ``build/`` directory is not an installable package),
mirroring the pattern in ``test_synth_verb_moods.py``.

Run with:

    python -m pytest tests/test_synth_verb_participles.py -x -v
"""

from __future__ import annotations

import importlib.util
import os
import unicodedata
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTH_PATH = REPO_ROOT / "build" / "synth_verb_participles.py"


@pytest.fixture(scope="module")
def synth():
    """Load build/synth_verb_participles.py as a module without package install."""
    spec = importlib.util.spec_from_file_location(
        "synth_verb_participles", SYNTH_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _strip_quantity(s):
    """Strip macron (U+0304) and breve (U+0306) — quantity marks that
    jtauber adds prosodically but we don't synthesise. Comparison
    against jtauber forms ignores these."""
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize(
        "NFC",
        "".join(c for c in nfd if ord(c) not in (0x0304, 0x0306)),
    )


# ---------------------------------------------------------------------------
# Lemma classification
# ---------------------------------------------------------------------------


class TestIsThematicOmega:
    def test_plain_omega_verb(self, synth):
        assert synth.is_thematic_omega("λύω")
        assert synth.is_thematic_omega("γράφω")
        assert synth.is_thematic_omega("παύω")
        assert synth.is_thematic_omega("παιδεύω")
        assert synth.is_thematic_omega("πείθω")

    def test_contract_rejected(self, synth):
        assert not synth.is_thematic_omega("τιμάω")
        assert not synth.is_thematic_omega("φιλέω")
        assert not synth.is_thematic_omega("δηλόω")

    def test_athematic_rejected(self, synth):
        assert not synth.is_thematic_omega("τίθημι")
        assert not synth.is_thematic_omega("δίδωμι")

    def test_deponent_rejected(self, synth):
        assert not synth.is_thematic_omega("ἔρχομαι")

    def test_empty_or_garbage_rejected(self, synth):
        assert not synth.is_thematic_omega("")
        assert not synth.is_thematic_omega(None)
        assert not synth.is_thematic_omega("ἀνήρ")


# ---------------------------------------------------------------------------
# Present-active synthesis (lemma alone is enough)
# ---------------------------------------------------------------------------


class TestPresentActive:
    """λύω → λύων / λύουσα / λῦον. We don't synthesise the circumflex
    on neuter sg cells, so ignore quantity marks for those compares."""

    def test_lyo_present_active_masculine(self, synth):
        out = synth.synthesize_participles("λύω", {})
        assert out["active_present_participle_nom_m_sg"] == "λύων"
        assert out["active_present_participle_gen_m_sg"] == "λύοντος"
        assert out["active_present_participle_dat_m_sg"] == "λύοντι"
        assert out["active_present_participle_acc_m_sg"] == "λύοντα"
        assert out["active_present_participle_nom_m_pl"] == "λύοντες"
        assert out["active_present_participle_gen_m_pl"] == "λυόντων"
        assert out["active_present_participle_dat_m_pl"] == "λύουσι(ν)"
        assert out["active_present_participle_acc_m_pl"] == "λύοντας"

    def test_lyo_present_active_feminine(self, synth):
        out = synth.synthesize_participles("λύω", {})
        # jtauber convention: -σᾰ / -σᾰν get breve, -σᾱς gets macron.
        assert out["active_present_participle_nom_f_sg"] == "λύουσᾰ"
        assert out["active_present_participle_gen_f_sg"] == "λυούσης"
        assert out["active_present_participle_dat_f_sg"] == "λυούσῃ"
        assert out["active_present_participle_acc_f_sg"] == "λύουσᾰν"
        assert out["active_present_participle_nom_f_pl"] == "λύουσαι"
        assert out["active_present_participle_dat_f_pl"] == "λυούσαις"
        assert out["active_present_participle_acc_f_pl"] == "λυούσᾱς"

    def test_lyo_present_active_neuter(self, synth):
        out = synth.synthesize_participles("λύω", {})
        # Neuter pl uses same forms as masc -οντα.
        assert out["active_present_participle_nom_n_pl"] == "λύοντα"
        assert out["active_present_participle_gen_n_pl"] == "λυόντων"

    def test_grapho_present_active(self, synth):
        out = synth.synthesize_participles("γράφω", {})
        assert out["active_present_participle_nom_m_sg"] == "γράφων"
        assert out["active_present_participle_gen_m_pl"] == "γραφόντων"
        assert out["active_present_participle_nom_f_sg"] == "γράφουσᾰ"
        assert out["active_present_participle_dat_f_sg"] == "γραφούσῃ"


# ---------------------------------------------------------------------------
# Present middle/passive synthesis (lemma alone)
# ---------------------------------------------------------------------------


class TestPresentMP:
    """λύω → λυόμενος / λυομένη / λυόμενον (middle present participle).
    Recessive accent: antepenult on -ό- when final short, penult on
    -μέ- when final long."""

    def test_lyo_present_mp_short_final(self, synth):
        out = synth.synthesize_participles("λύω", {})
        # Final -ος, -ον, -ε, -οι, -α, -αι are short → antepenult acute.
        assert out["middle_present_participle_nom_m_sg"] == "λυόμενος"
        assert out["middle_present_participle_acc_m_sg"] == "λυόμενον"
        assert out["middle_present_participle_voc_m_sg"] == "λυόμενε"
        assert out["middle_present_participle_nom_m_pl"] == "λυόμενοι"
        assert out["middle_present_participle_nom_n_pl"] == "λυόμενα"
        assert out["middle_present_participle_acc_n_pl"] == "λυόμενα"

    def test_lyo_present_mp_long_final(self, synth):
        out = synth.synthesize_participles("λύω", {})
        # Final -η, -ης, -ῃ, -ην, -ων, -ου, -ῳ, -ους, -οις, -αις are long
        # → penult acute.
        assert out["middle_present_participle_nom_f_sg"] == "λυομένη"
        assert out["middle_present_participle_gen_f_sg"] == "λυομένης"
        assert out["middle_present_participle_dat_f_sg"] == "λυομένῃ"
        assert out["middle_present_participle_acc_f_sg"] == "λυομένην"
        assert out["middle_present_participle_gen_m_sg"] == "λυομένου"
        assert out["middle_present_participle_dat_m_sg"] == "λυομένῳ"
        assert out["middle_present_participle_gen_m_pl"] == "λυομένων"
        assert out["middle_present_participle_acc_m_pl"] == "λυομένους"

    def test_grapho_present_mp(self, synth):
        out = synth.synthesize_participles("γράφω", {})
        assert out["middle_present_participle_nom_m_sg"] == "γραφόμενος"
        assert out["middle_present_participle_gen_m_sg"] == "γραφομένου"
        assert out["middle_present_participle_nom_f_sg"] == "γραφομένη"


# ---------------------------------------------------------------------------
# Aorist active synthesis (3rd-decl, sigmatic)
# ---------------------------------------------------------------------------


class TestAoristActive:
    """λύω → λύσας / λύσασα / λῦσαν (aor act participle). σ-stem comes
    either from fut (λύσω → λύσ-) or from aor (ἔλυσα → λύσ-)."""

    def test_lyo_aorist_active_from_fut(self, synth):
        out = synth.synthesize_participles("λύω", {"fut": "λύσω"})
        assert out["active_aorist_participle_nom_m_sg"] == "λύσας"
        assert out["active_aorist_participle_gen_m_sg"] == "λύσαντος"
        assert out["active_aorist_participle_dat_m_sg"] == "λύσαντι"
        assert out["active_aorist_participle_acc_m_sg"] == "λύσαντα"
        assert out["active_aorist_participle_nom_m_pl"] == "λύσαντες"
        assert out["active_aorist_participle_gen_m_pl"] == "λυσάντων"
        assert out["active_aorist_participle_dat_m_pl"] == "λύσασι(ν)"

    def test_lyo_aorist_active_feminine(self, synth):
        out = synth.synthesize_participles("λύω", {"fut": "λύσω"})
        # jtauber: σ-stem α gets MACRON throughout the fem decl
        # (λύσᾱσα / λυσᾱ́σης / λύσᾱσαι / λυσᾱ́σᾱς).
        assert out["active_aorist_participle_nom_f_sg"] == "λύσᾱσα"
        assert out["active_aorist_participle_gen_f_sg"] == "λυσᾱ́σης"
        assert out["active_aorist_participle_dat_f_sg"] == "λυσᾱ́σῃ"
        assert out["active_aorist_participle_acc_f_sg"] == "λύσᾱσαν"
        assert out["active_aorist_participle_nom_f_pl"] == "λύσᾱσαι"
        assert out["active_aorist_participle_acc_f_pl"] == "λυσᾱ́σᾱς"

    def test_grapho_aorist_active_from_aor(self, synth):
        """γράφω + aor ἔγραψα: graft ψ-cluster onto present stem."""
        out = synth.synthesize_participles("γράφω", {"aor": "ἔγραψα"})
        assert out["active_aorist_participle_nom_m_sg"] == "γράψας"
        assert out["active_aorist_participle_gen_m_pl"] == "γραψάντων"
        assert out["active_aorist_participle_nom_f_sg"] == "γράψᾱσα"
        assert out["active_aorist_participle_dat_f_pl"] == "γραψᾱ́σαις"

    def test_pempo_aorist_active(self, synth):
        out = synth.synthesize_participles("πέμπω", {"fut": "πέμψω"})
        assert out["active_aorist_participle_nom_m_sg"] == "πέμψας"
        assert out["active_aorist_participle_acc_m_sg"] == "πέμψαντα"
        assert out["active_aorist_participle_nom_f_sg"] == "πέμψᾱσα"


# ---------------------------------------------------------------------------
# Aorist middle synthesis
# ---------------------------------------------------------------------------


class TestAoristMiddle:
    """λύω → λυσάμενος / λυσαμένη / λυσάμενον."""

    def test_lyo_aorist_middle_short_final(self, synth):
        out = synth.synthesize_participles("λύω", {"fut": "λύσω"})
        assert out["middle_aorist_participle_nom_m_sg"] == "λυσάμενος"
        assert out["middle_aorist_participle_acc_m_sg"] == "λυσάμενον"
        assert out["middle_aorist_participle_voc_m_sg"] == "λυσάμενε"
        assert out["middle_aorist_participle_nom_m_pl"] == "λυσάμενοι"
        assert out["middle_aorist_participle_nom_n_pl"] == "λυσάμενα"
        assert out["middle_aorist_participle_acc_n_pl"] == "λυσάμενα"

    def test_lyo_aorist_middle_long_final(self, synth):
        out = synth.synthesize_participles("λύω", {"fut": "λύσω"})
        assert out["middle_aorist_participle_nom_f_sg"] == "λυσαμένη"
        assert out["middle_aorist_participle_gen_f_sg"] == "λυσαμένης"
        assert out["middle_aorist_participle_dat_m_sg"] == "λυσαμένῳ"
        assert out["middle_aorist_participle_gen_m_pl"] == "λυσαμένων"


# ---------------------------------------------------------------------------
# Aorist passive synthesis (3rd-decl, ending-accented -θείς)
# ---------------------------------------------------------------------------


class TestAoristPassive:
    """λύω + aor.p ἐλύθην → λυθείς / λυθεῖσα / λυθέν."""

    def test_lyo_aorist_passive_masculine(self, synth):
        out = synth.synthesize_participles(
            "λύω", {"aor_p": "ἐλύθην"}
        )
        assert out["passive_aorist_participle_nom_m_sg"] == "λυθείς"
        assert out["passive_aorist_participle_gen_m_sg"] == "λυθέντος"
        assert out["passive_aorist_participle_dat_m_sg"] == "λυθέντι"
        assert out["passive_aorist_participle_acc_m_sg"] == "λυθέντα"
        assert out["passive_aorist_participle_nom_m_pl"] == "λυθέντες"
        assert out["passive_aorist_participle_gen_m_pl"] == "λυθέντων"
        assert out["passive_aorist_participle_dat_m_pl"] == "λυθεῖσι(ν)"

    def test_lyo_aorist_passive_feminine(self, synth):
        out = synth.synthesize_participles(
            "λύω", {"aor_p": "ἐλύθην"}
        )
        assert out["passive_aorist_participle_nom_f_sg"] == "λυθεῖσα"
        assert out["passive_aorist_participle_gen_f_sg"] == "λυθείσης"
        assert out["passive_aorist_participle_dat_f_sg"] == "λυθείσῃ"
        assert out["passive_aorist_participle_acc_f_sg"] == "λυθεῖσαν"
        assert out["passive_aorist_participle_nom_f_pl"] == "λυθεῖσαι"
        assert out["passive_aorist_participle_gen_f_pl"] == "λυθεισῶν"

    def test_lyo_aorist_passive_neuter(self, synth):
        out = synth.synthesize_participles(
            "λύω", {"aor_p": "ἐλύθην"}
        )
        assert out["passive_aorist_participle_nom_n_sg"] == "λυθέν"
        assert out["passive_aorist_participle_acc_n_sg"] == "λυθέν"
        assert out["passive_aorist_participle_gen_n_sg"] == "λυθέντος"
        assert out["passive_aorist_participle_nom_n_pl"] == "λυθέντα"

    def test_paideuo_aorist_passive(self, synth):
        out = synth.synthesize_participles(
            "παιδεύω", {"aor_p": "ἐπαιδεύθην"}
        )
        assert out["passive_aorist_participle_nom_m_sg"] == "παιδευθείς"
        assert out["passive_aorist_participle_gen_m_pl"] == "παιδευθέντων"


# ---------------------------------------------------------------------------
# Perfect active synthesis (3rd-decl, -ώς/-υῖα/-ός)
# ---------------------------------------------------------------------------


class TestPerfectActive:
    """λύω + pf λέλυκα → λελυκώς / λελυκυῖα / λελυκός."""

    def test_lyo_perfect_active_masculine(self, synth):
        out = synth.synthesize_participles(
            "λύω", {"pf": "λέλυκα"}
        )
        assert out["active_perfect_participle_nom_m_sg"] == "λελυκώς"
        assert out["active_perfect_participle_gen_m_sg"] == "λελυκότος"
        assert out["active_perfect_participle_dat_m_sg"] == "λελυκότι"
        assert out["active_perfect_participle_acc_m_sg"] == "λελυκότα"
        assert out["active_perfect_participle_nom_m_pl"] == "λελυκότες"
        assert out["active_perfect_participle_gen_m_pl"] == "λελυκότων"
        assert out["active_perfect_participle_dat_m_pl"] == "λελυκόσι(ν)"

    def test_lyo_perfect_active_feminine(self, synth):
        out = synth.synthesize_participles("λύω", {"pf": "λέλυκα"})
        # jtauber: macron on gen_f_sg α (-υίᾱς) and acc_f_pl α (-υίᾱς).
        assert out["active_perfect_participle_nom_f_sg"] == "λελυκυῖα"
        assert out["active_perfect_participle_gen_f_sg"] == "λελυκυίᾱς"
        assert out["active_perfect_participle_dat_f_sg"] == "λελυκυίᾳ"
        assert out["active_perfect_participle_acc_f_sg"] == "λελυκυῖαν"
        assert out["active_perfect_participle_acc_f_pl"] == "λελυκυίᾱς"

    def test_lyo_perfect_active_neuter(self, synth):
        out = synth.synthesize_participles("λύω", {"pf": "λέλυκα"})
        assert out["active_perfect_participle_nom_n_sg"] == "λελυκός"
        assert out["active_perfect_participle_acc_n_sg"] == "λελυκός"
        assert out["active_perfect_participle_nom_n_pl"] == "λελυκότα"


# ---------------------------------------------------------------------------
# Perfect middle/passive synthesis (1st/2nd-decl, persistent -μέν- accent)
# ---------------------------------------------------------------------------


class TestPerfectMP:
    """λύω + pf_mp λέλυμαι → λελυμένος / λελυμένη / λελυμένον.
    Persistent accent: always on -μέν- (penult), never antepenult."""

    def test_lyo_perfect_mp_persistent_accent(self, synth):
        out = synth.synthesize_participles(
            "λύω", {"pf_mp": "λέλυμαι"}
        )
        # Even with short final (-ος), accent is on penult -μέ-.
        assert out["middle_perfect_participle_nom_m_sg"] == "λελυμένος"
        assert out["middle_perfect_participle_acc_m_sg"] == "λελυμένον"
        assert out["middle_perfect_participle_nom_m_pl"] == "λελυμένοι"
        assert out["middle_perfect_participle_nom_f_sg"] == "λελυμένη"
        assert out["middle_perfect_participle_gen_m_sg"] == "λελυμένου"
        assert out["middle_perfect_participle_dat_m_sg"] == "λελυμένῳ"

    def test_peitho_perfect_mp(self, synth):
        """πέπεισμαι (assimilated -σμαι from -θμαι) → πεπεισμένος."""
        out = synth.synthesize_participles(
            "πείθω", {"pf_mp": "πέπεισμαι"}
        )
        assert out["middle_perfect_participle_nom_m_sg"] == "πεπεισμένος"
        assert out["middle_perfect_participle_acc_f_sg"] == "πεπεισμένην"
        assert out["middle_perfect_participle_gen_m_pl"] == "πεπεισμένων"

    def test_pauo_perfect_mp(self, synth):
        out = synth.synthesize_participles(
            "παύω", {"pf_mp": "πέπαυμαι"}
        )
        assert out["middle_perfect_participle_nom_m_sg"] == "πεπαυμένος"
        assert out["middle_perfect_participle_nom_f_sg"] == "πεπαυμένη"


# ---------------------------------------------------------------------------
# Future synthesis
# ---------------------------------------------------------------------------


class TestFutureActive:
    """λύσω → λύσων / λύσουσα / λῦσον (future active participle).
    Same shape as present active but on the σ-stem."""

    def test_lyo_future_active(self, synth):
        out = synth.synthesize_participles("λύω", {"fut": "λύσω"})
        assert out["active_future_participle_nom_m_sg"] == "λύσων"
        assert out["active_future_participle_gen_m_sg"] == "λύσοντος"
        assert out["active_future_participle_acc_m_sg"] == "λύσοντα"
        # Future-pattern fem: NO breve on final α (jtauber writes
        # γράψουσα plain, not γράψουσᾰ).
        assert out["active_future_participle_nom_f_sg"] == "λύσουσα"
        assert out["active_future_participle_dat_f_sg"] == "λυσούσῃ"
        assert out["active_future_participle_acc_f_pl"] == "λυσούσᾱς"

    def test_grapho_future_active(self, synth):
        """γράφω: derive σ-stem from aor since no fut given."""
        out = synth.synthesize_participles("γράφω", {"aor": "ἔγραψα"})
        assert out["active_future_participle_nom_m_sg"] == "γράψων"
        assert out["active_future_participle_gen_m_pl"] == "γραψόντων"


class TestFutureMiddle:
    """λύσομαι → λυσόμενος / λυσομένη / λυσόμενον."""

    def test_lyo_future_middle(self, synth):
        out = synth.synthesize_participles("λύω", {"fut": "λύσω"})
        assert out["middle_future_participle_nom_m_sg"] == "λυσόμενος"
        assert out["middle_future_participle_acc_m_sg"] == "λυσόμενον"
        assert out["middle_future_participle_nom_f_sg"] == "λυσομένη"
        assert out["middle_future_participle_gen_m_sg"] == "λυσομένου"


class TestFuturePassive:
    """λυθήσομαι → λυθησόμενος / λυθησομένη / λυθησόμενον."""

    def test_lyo_future_passive(self, synth):
        out = synth.synthesize_participles(
            "λύω", {"aor_p": "ἐλύθην"}
        )
        assert out["passive_future_participle_nom_m_sg"] == "λυθησόμενος"
        assert out["passive_future_participle_acc_m_sg"] == "λυθησόμενον"
        assert out["passive_future_participle_nom_f_sg"] == "λυθησομένη"
        assert out["passive_future_participle_gen_m_sg"] == "λυθησομένου"
        # No double-θ bug
        assert "λυθθ" not in out["passive_future_participle_nom_m_sg"]


# ---------------------------------------------------------------------------
# Spot-check against jtauber_ag_paradigms.json shape (cross-source sanity)
# ---------------------------------------------------------------------------


class TestJtauberCompat:
    """Spot checks against jtauber's known good values for shared verbs.
    Ignores macron/breve quantity marks (we don't synthesise those)."""

    @pytest.fixture(scope="class")
    def jtauber(self):
        import json
        # resolve like test_paradigm.py: $JTAUBER_PARADIGMS (file), then the
        # documented $DILEMMA_PARADIGM_DATA (dir), then a home-dir fallback
        candidates = []
        if os.environ.get("JTAUBER_PARADIGMS"):
            candidates.append(Path(os.environ["JTAUBER_PARADIGMS"]))
        if os.environ.get("DILEMMA_PARADIGM_DATA"):
            candidates.append(Path(os.environ["DILEMMA_PARADIGM_DATA"])
                              / "jtauber_ag_paradigms.json")
        candidates.append(
            Path.home() / "Documents" / "jtauber_ag_paradigms.json")
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            pytest.skip(f"jtauber paradigms not found (tried {candidates})")
        with open(path) as f:
            return json.load(f)

    def test_lyo_full_paradigm_match_ratio(self, synth, jtauber):
        """Synthesise a complete λύω paradigm; >95% of cells common with
        jtauber should match (mod quantity marks)."""
        out = synth.synthesize_participles(
            "λύω", {
                "fut": "λύσω", "aor": "ἔλυσα", "pf": "λέλυκα",
                "pf_mp": "λέλυμαι", "aor_p": "ἐλύθην",
            },
        )
        jtf = jtauber["λύω"]["forms"]
        common = set(out) & set(jtf)
        assert len(common) >= 200, f"too few common keys: {len(common)}"
        match = sum(1 for k in common if _strip_quantity(out[k]) ==
                    _strip_quantity(jtf[k]))
        ratio = match / len(common)
        assert ratio >= 0.95, (
            f"only {match}/{len(common)} cells match jtauber for λύω "
            f"(ratio {ratio:.3f})"
        )

    def test_grapho_full_paradigm_match_ratio(self, synth, jtauber):
        out = synth.synthesize_participles(
            "γράφω", {"aor": "ἔγραψα", "pf": "γέγραφα"},
        )
        jtf = jtauber["γράφω"]["forms"]
        common = set(out) & set(jtf)
        assert len(common) >= 100
        match = sum(1 for k in common if _strip_quantity(out[k]) ==
                    _strip_quantity(jtf[k]))
        ratio = match / len(common)
        assert ratio >= 0.95, (
            f"γράφω: {match}/{len(common)} match (ratio {ratio:.3f})"
        )

    def test_pauo_full_paradigm_match_ratio(self, synth, jtauber):
        out = synth.synthesize_participles(
            "παύω", {
                "fut": "παύσω", "aor": "ἔπαυσα", "pf": "πέπαυκα",
                "pf_mp": "πέπαυμαι", "aor_p": "ἐπαύθην",
            },
        )
        jtf = jtauber["παύω"]["forms"]
        common = set(out) & set(jtf)
        assert len(common) >= 100
        match = sum(1 for k in common if _strip_quantity(out[k]) ==
                    _strip_quantity(jtf[k]))
        ratio = match / len(common)
        # παύω has the circumflex-on-monosyll-stem issue on a handful of
        # neuter sg cells; allow slightly lower threshold.
        assert ratio >= 0.92, (
            f"παύω: {match}/{len(common)} match (ratio {ratio:.3f})"
        )


# ---------------------------------------------------------------------------
# No-op cases
# ---------------------------------------------------------------------------


class TestNoOpCases:
    def test_contract_returns_empty(self, synth):
        out = synth.synthesize_participles(
            "φιλέω", {"fut": "φιλήσω", "aor": "ἐφίλησα"}
        )
        assert out == {}

    def test_athematic_returns_empty(self, synth):
        out = synth.synthesize_participles(
            "δίδωμι", {"aor": "ἔδωκα"}
        )
        assert out == {}

    def test_deponent_returns_empty(self, synth):
        out = synth.synthesize_participles(
            "ἔρχομαι", {"aor": "ἦλθον"}
        )
        assert out == {}

    def test_no_principal_parts_only_present(self, synth):
        """Without principal parts, only present-system participles
        (active and mp) are synthesised."""
        out = synth.synthesize_participles("λύω", None)
        assert "active_present_participle_nom_m_sg" in out
        assert "middle_present_participle_nom_m_sg" in out
        # No aor/fut/pf cells
        assert "active_aorist_participle_nom_m_sg" not in out
        assert "active_future_participle_nom_m_sg" not in out
        assert "active_perfect_participle_nom_m_sg" not in out

    def test_aor2_lemma_skips_aorist(self, synth):
        """λείπω + aor-2 ἔλιπον: no σ-stem available; aorist skipped."""
        out = synth.synthesize_participles("λείπω", {"aor": "ἔλιπον"})
        assert "active_aorist_participle_nom_m_sg" not in out
        assert "middle_aorist_participle_nom_m_sg" not in out
        # Present forms still synthesised.
        assert "active_present_participle_nom_m_sg" in out

    def test_none_lemma_returns_empty(self, synth):
        out = synth.synthesize_participles(None, {"fut": "λύσω"})
        assert out == {}

    def test_non_omega_lemma_returns_empty(self, synth):
        out = synth.synthesize_participles(
            "ἀνήρ", {"fut": "whatever"}
        )
        assert out == {}

    def test_missing_pf_mp_skips_perfect_mp(self, synth):
        """No pf_mp → no middle_perfect_participle cells, but other
        cells still produced."""
        out = synth.synthesize_participles(
            "λύω", {"pf": "λέλυκα", "fut": "λύσω"}
        )
        assert "active_perfect_participle_nom_m_sg" in out
        assert "middle_perfect_participle_nom_m_sg" not in out

    def test_missing_aor_p_skips_passive(self, synth):
        """No aor_p → no aor-pass / fut-pass cells."""
        out = synth.synthesize_participles(
            "λύω", {"fut": "λύσω", "aor": "ἔλυσα"}
        )
        assert "passive_aorist_participle_nom_m_sg" not in out
        assert "passive_future_participle_nom_m_sg" not in out


# ---------------------------------------------------------------------------
# Build-pipeline integration: never overwrite existing cells
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    """Mirrors the build_grc_verb_paradigms.py merge logic: synth must
    only fill empty cells; existing corpus / Wiktionary cells must
    survive."""

    def test_skip_collision(self, synth):
        """The synth function returns ALL synthesizable cells; the
        caller is responsible for skipping collisions. Verify via a
        manual merge that no synthesised value overwrites an existing
        cell."""
        existing = {
            "active_present_participle_nom_m_sg": "λύων",   # existing, stays
            "active_present_participle_nom_f_sg": "λύουσα",
        }
        templated = synth.synthesize_participles("λύω", {"fut": "λύσω"})
        # Apply the same merge rule build_grc_verb_paradigms uses:
        merged = dict(existing)
        for k, v in templated.items():
            if k not in merged:
                merged[k] = v
        # Existing cells survived
        assert merged["active_present_participle_nom_m_sg"] == "λύων"
        assert merged["active_present_participle_nom_f_sg"] == "λύουσα"
        # New cells added
        assert "active_aorist_participle_nom_m_sg" in merged
        assert merged["active_aorist_participle_nom_m_sg"] == "λύσας"

    def test_jtauber_key_format(self, synth):
        """All emitted keys must follow the jtauber-compatible pattern
        ``{voice}_{tense}_participle_{case}_{gender}_{number}``."""
        out = synth.synthesize_participles(
            "λύω", {
                "fut": "λύσω", "aor": "ἔλυσα", "pf": "λέλυκα",
                "pf_mp": "λέλυμαι", "aor_p": "ἐλύθην",
            },
        )
        valid_voices = {"active", "middle", "passive"}
        valid_tenses = {"present", "future", "aorist", "perfect"}
        valid_cases = {"nom", "gen", "dat", "acc", "voc"}
        valid_genders = {"m", "f", "n"}
        valid_numbers = {"sg", "pl"}
        for k in out:
            parts = k.split("_")
            assert len(parts) == 6, f"unexpected key shape: {k!r}"
            voice, tense, ppl_word, case, gender, number = parts
            assert voice in valid_voices, k
            assert tense in valid_tenses, k
            assert ppl_word == "participle", k
            assert case in valid_cases, k
            assert gender in valid_genders, k
            assert number in valid_numbers, k


# ---------------------------------------------------------------------------
# v3: aor-2 participle synthesis
# ---------------------------------------------------------------------------


class TestAor2ParticipleStem:
    def test_extract_lipo(self, synth):
        assert synth._aor2_stem("ἔλιπον") == "λιπ"

    def test_extract_pesso(self, synth):
        assert synth._aor2_stem("ἔπεσον") == "πεσ"

    def test_extract_labe(self, synth):
        assert synth._aor2_stem("ἔλαβον") == "λαβ"

    def test_rejects_sigmatic(self, synth):
        assert synth._aor2_stem("ἔλυσα") is None

    def test_rejects_kappa(self, synth):
        assert synth._aor2_stem("ἔδωκα") is None


class TestAor2ParticipleSynthesis:
    """Aor-2 participle synthesis matches jtauber on canonical verbs."""

    def test_lambano_active_nom_m_sg(self, synth):
        out = synth.synthesize_aor2_participles("λαμβάνω", {"aor": "ἔλαβον"})
        assert out["active_aorist_participle_nom_m_sg"] == "λαβών"

    def test_lambano_active_nom_f_sg(self, synth):
        out = synth.synthesize_aor2_participles("λαμβάνω", {"aor": "ἔλαβον"})
        assert out["active_aorist_participle_nom_f_sg"] == "λαβοῦσα"

    def test_lambano_active_nom_n_sg(self, synth):
        out = synth.synthesize_aor2_participles("λαμβάνω", {"aor": "ἔλαβον"})
        assert out["active_aorist_participle_nom_n_sg"] == "λαβόν"

    def test_lambano_active_gen_m_sg(self, synth):
        out = synth.synthesize_aor2_participles("λαμβάνω", {"aor": "ἔλαβον"})
        assert out["active_aorist_participle_gen_m_sg"] == "λαβόντος"

    def test_lambano_active_dat_m_pl(self, synth):
        out = synth.synthesize_aor2_participles("λαμβάνω", {"aor": "ἔλαβον"})
        assert out["active_aorist_participle_dat_m_pl"] == "λαβοῦσι(ν)"

    def test_lambano_active_gen_m_pl(self, synth):
        out = synth.synthesize_aor2_participles("λαμβάνω", {"aor": "ἔλαβον"})
        assert out["active_aorist_participle_gen_m_pl"] == "λαβόντων"

    def test_pipto_active_nom_m_sg(self, synth):
        out = synth.synthesize_aor2_participles("πίπτω", {"aor": "ἔπεσον"})
        assert out["active_aorist_participle_nom_m_sg"] == "πεσών"

    def test_pipto_active_acc_n_sg(self, synth):
        out = synth.synthesize_aor2_participles("πίπτω", {"aor": "ἔπεσον"})
        assert out["active_aorist_participle_acc_n_sg"] == "πεσόν"

    def test_leipo_active_nom_m_sg(self, synth):
        out = synth.synthesize_aor2_participles("λείπω", {"aor": "ἔλιπον"})
        assert out["active_aorist_participle_nom_m_sg"] == "λιπών"

    def test_leipo_middle_nom_m_sg(self, synth):
        # λιπόμενος (middle aor-2 participle)
        out = synth.synthesize_aor2_participles("λείπω", {"aor": "ἔλιπον"})
        assert out["middle_aorist_participle_nom_m_sg"] == "λιπόμενος"

    def test_emits_full_grid(self, synth):
        # Every (case × gender × number) for active should be present.
        out = synth.synthesize_aor2_participles("λαμβάνω", {"aor": "ἔλαβον"})
        for case in ("nom", "gen", "dat", "acc", "voc"):
            for gender in ("m", "f", "n"):
                for number in ("sg", "pl"):
                    key = (
                        f"active_aorist_participle_{case}_{gender}_{number}"
                    )
                    assert key in out, f"missing {key}"

    def test_skips_when_no_aor(self, synth):
        out = synth.synthesize_aor2_participles("λύω", {})
        assert out == {}

    def test_skips_sigmatic_aor(self, synth):
        out = synth.synthesize_aor2_participles("λύω", {"aor": "ἔλυσα"})
        assert out == {}


class TestAor2ParticipleSchemaAlignment:
    def test_keys_match_jtauber_pattern(self, synth):
        out = synth.synthesize_aor2_participles("λαμβάνω", {"aor": "ἔλαβον"})
        for k in out:
            parts = k.split("_")
            assert parts[1] == "aorist", k
            assert parts[2] == "participle", k
            assert parts[0] in ("active", "middle"), k


# ---------------------------------------------------------------------------
# v3: contract participle synthesis
# ---------------------------------------------------------------------------


class TestContractParticipleSynthesis:
    """Contract present participle synthesis matches jtauber's
    active-voice forms on τιμάω / ποιέω / δηλόω."""

    def test_timao_nom_m_sg(self, synth):
        out = synth.synthesize_contract_participles("τιμάω", {})
        assert out["active_present_participle_nom_m_sg"] == "τιμῶν"

    def test_timao_gen_m_sg(self, synth):
        out = synth.synthesize_contract_participles("τιμάω", {})
        assert out["active_present_participle_gen_m_sg"] == "τιμῶντος"

    def test_timao_nom_n_sg(self, synth):
        out = synth.synthesize_contract_participles("τιμάω", {})
        assert out["active_present_participle_nom_n_sg"] == "τιμῶν"

    def test_timao_dat_m_pl(self, synth):
        out = synth.synthesize_contract_participles("τιμάω", {})
        assert out["active_present_participle_dat_m_pl"] == "τιμῶσι(ν)"

    def test_timao_gen_m_pl(self, synth):
        out = synth.synthesize_contract_participles("τιμάω", {})
        assert out["active_present_participle_gen_m_pl"] == "τιμώντων"

    def test_poieo_nom_m_sg(self, synth):
        out = synth.synthesize_contract_participles("ποιέω", {})
        assert out["active_present_participle_nom_m_sg"] == "ποιῶν"

    def test_poieo_gen_m_sg(self, synth):
        out = synth.synthesize_contract_participles("ποιέω", {})
        assert out["active_present_participle_gen_m_sg"] == "ποιοῦντος"

    def test_poieo_nom_n_sg(self, synth):
        out = synth.synthesize_contract_participles("ποιέω", {})
        assert out["active_present_participle_nom_n_sg"] == "ποιοῦν"

    def test_poieo_dat_m_pl(self, synth):
        out = synth.synthesize_contract_participles("ποιέω", {})
        assert out["active_present_participle_dat_m_pl"] == "ποιοῦσι(ν)"

    def test_dêlôô_nom_m_sg(self, synth):
        out = synth.synthesize_contract_participles("δηλόω", {})
        assert out["active_present_participle_nom_m_sg"] == "δηλῶν"

    def test_dêlôô_gen_m_sg(self, synth):
        out = synth.synthesize_contract_participles("δηλόω", {})
        assert out["active_present_participle_gen_m_sg"] == "δηλοῦντος"

    def test_dêlôô_nom_n_sg(self, synth):
        out = synth.synthesize_contract_participles("δηλόω", {})
        assert out["active_present_participle_nom_n_sg"] == "δηλοῦν"

    def test_phileo_nom_m_sg(self, synth):
        out = synth.synthesize_contract_participles("φιλέω", {})
        assert out["active_present_participle_nom_m_sg"] == "φιλῶν"

    def test_skips_thematic_omega(self, synth):
        out = synth.synthesize_contract_participles("λύω", {})
        assert out == {}

    def test_skips_athematic(self, synth):
        out = synth.synthesize_contract_participles("τίθημι", {})
        assert out == {}


class TestContractParticipleSchemaAlignment:
    def test_keys_only_present_tense(self, synth):
        out = synth.synthesize_contract_participles("τιμάω", {})
        for k in out:
            parts = k.split("_")
            assert parts[1] == "present", k
            assert parts[2] == "participle", k
            assert parts[0] in ("active", "middle"), k


# ---------------------------------------------------------------------------
# v4: macron / breve quantity marks on feminine participle endings.
# jtauber's convention is per-tense / per-pattern, NOT per-lemma:
#   * Present active 3rd-decl -ουσᾰ family: BREVE on final α in
#     nom/voc/acc f sg, MACRON in nom/voc/acc f du and acc f pl.
#   * Aorist active 3rd-decl -σᾱσα family: MACRON on σ-stem α (the
#     first α between σ's) UNIVERSALLY across every f cell, plus
#     MACRON on final α in acc f pl.
#   * Aor-2 / future / aorist-passive 3rd-decl -οῦσα / -σουσα / -εῖσα:
#     no breve, MACRON only on acc f pl.
#   * Perfect active -υῖα: MACRON on gen f sg AND acc f pl final α.
#   * 1st/2nd-decl -μενος/-η/-ον: MACRON on acc f pl (-μενᾱς).
#   * Alpha-contract present active -ῶσᾰ: BREVE on final α in
#     nom/voc/acc f sg (matching the present-active pattern).
# ---------------------------------------------------------------------------


class TestParticipleQuantityMarks:
    """Spot-checks of the macron / breve marks on feminine participle
    endings against jtauber's verbatim output."""

    def test_present_active_lyo_breve_on_nom_acc(self, synth):
        out = synth.synthesize_participles("λύω", {})
        assert out["active_present_participle_nom_f_sg"] == "λύουσᾰ"
        assert out["active_present_participle_voc_f_sg"] == "λύουσᾰ"
        assert out["active_present_participle_acc_f_sg"] == "λύουσᾰν"

    def test_present_active_lyo_macron_on_acc_pl(self, synth):
        out = synth.synthesize_participles("λύω", {})
        assert out["active_present_participle_acc_f_pl"] == "λυούσᾱς"

    def test_present_active_lyo_unmarked_dat_gen(self, synth):
        out = synth.synthesize_participles("λύω", {})
        # No quantity mark on dat / gen feminine cells.
        assert out["active_present_participle_gen_f_sg"] == "λυούσης"
        assert out["active_present_participle_dat_f_sg"] == "λυούσῃ"
        assert out["active_present_participle_gen_f_pl"] == "λυουσῶν"
        assert out["active_present_participle_dat_f_pl"] == "λυούσαις"

    def test_aorist_active_lyo_macron_throughout_fem(self, synth):
        out = synth.synthesize_participles("λύω", {"fut": "λύσω"})
        # σ-stem α gets MACRON across all f cells; final α also gets
        # MACRON in acc f pl (-σᾱ́σᾱς).
        assert out["active_aorist_participle_nom_f_sg"] == "λύσᾱσα"
        assert out["active_aorist_participle_acc_f_sg"] == "λύσᾱσαν"
        assert out["active_aorist_participle_gen_f_sg"] == "λυσᾱ́σης"
        assert out["active_aorist_participle_dat_f_sg"] == "λυσᾱ́σῃ"
        assert out["active_aorist_participle_gen_f_pl"] == "λυσᾱσῶν"
        assert out["active_aorist_participle_dat_f_pl"] == "λυσᾱ́σαις"
        assert out["active_aorist_participle_acc_f_pl"] == "λυσᾱ́σᾱς"
        assert out["active_aorist_participle_nom_f_pl"] == "λύσᾱσαι"

    def test_aorist_active_grapho(self, synth):
        # γράψᾱσα / γραψᾱ́σᾱς / γραψᾱσῶν against jtauber.
        out = synth.synthesize_participles("γράφω", {"aor": "ἔγραψα"})
        assert out["active_aorist_participle_nom_f_sg"] == "γράψᾱσα"
        assert out["active_aorist_participle_acc_f_pl"] == "γραψᾱ́σᾱς"
        assert out["active_aorist_participle_gen_f_pl"] == "γραψᾱσῶν"

    def test_future_active_lyo_no_breve(self, synth):
        # Future active uses present-style 3rd-decl endings BUT without
        # the breve on f sg. jtauber writes γράψουσα plain (no marks).
        out = synth.synthesize_participles("λύω", {"fut": "λύσω"})
        assert out["active_future_participle_nom_f_sg"] == "λύσουσα"
        assert out["active_future_participle_acc_f_sg"] == "λύσουσαν"
        assert out["active_future_participle_acc_f_pl"] == "λυσούσᾱς"

    def test_aorist_passive_lyo_macron_only_acc_pl(self, synth):
        out = synth.synthesize_participles("λύω", {"aor_p": "ἐλύθην"})
        assert out["passive_aorist_participle_nom_f_sg"] == "λυθεῖσα"
        assert out["passive_aorist_participle_acc_f_sg"] == "λυθεῖσαν"
        # Only acc_f_pl gets a macron.
        assert out["passive_aorist_participle_acc_f_pl"] == "λυθείσᾱς"

    def test_perfect_active_lyo_macron_gen_acc_pl(self, synth):
        out = synth.synthesize_participles("λύω", {"pf": "λέλυκα"})
        # gen_f_sg AND acc_f_pl both get macron.
        assert out["active_perfect_participle_gen_f_sg"] == "λελυκυίᾱς"
        assert out["active_perfect_participle_acc_f_pl"] == "λελυκυίᾱς"
        # Other cells unmarked.
        assert out["active_perfect_participle_nom_f_sg"] == "λελυκυῖα"
        assert out["active_perfect_participle_acc_f_sg"] == "λελυκυῖαν"

    def test_middle_present_lyo_macron_only_acc_pl(self, synth):
        out = synth.synthesize_participles("λύω", {})
        # Only acc_f_pl in -μενᾱς gets a macron.
        assert out["middle_present_participle_acc_f_pl"] == "λυομένᾱς"
        assert out["middle_present_participle_nom_f_sg"] == "λυομένη"
        assert out["middle_present_participle_acc_f_sg"] == "λυομένην"

    def test_middle_perfect_lyo_macron_only_acc_pl(self, synth):
        out = synth.synthesize_participles(
            "λύω", {"pf_mp": "λέλυμαι"}
        )
        assert out["middle_perfect_participle_acc_f_pl"] == "λελυμένᾱς"
        assert out["middle_perfect_participle_nom_f_sg"] == "λελυμένη"

    def test_alpha_contract_breve_on_nom_acc(self, synth):
        out = synth.synthesize_contract_participles("τιμάω", {})
        # τιμῶσᾰ (breve), τιμῶσᾰν (breve), τιμώσᾱς (macron).
        assert out["active_present_participle_nom_f_sg"] == "τιμῶσᾰ"
        assert out["active_present_participle_voc_f_sg"] == "τιμῶσᾰ"
        assert out["active_present_participle_acc_f_sg"] == "τιμῶσᾰν"
        assert out["active_present_participle_acc_f_pl"] == "τιμώσᾱς"

    def test_epsilon_contract_breve_on_nom_acc(self, synth):
        # ε-contract: BREVE on nom/voc/acc f sg, MACRON on acc f pl.
        # Matches jtauber's ποιοῦσᾰ / ποιοῦσᾰν / ποιούσᾱς.
        out = synth.synthesize_contract_participles("ποιέω", {})
        assert out["active_present_participle_nom_f_sg"] == "ποιοῦσᾰ"
        assert out["active_present_participle_voc_f_sg"] == "ποιοῦσᾰ"
        assert out["active_present_participle_acc_f_sg"] == "ποιοῦσᾰν"
        assert out["active_present_participle_acc_f_pl"] == "ποιούσᾱς"

    def test_aor2_pesa_macron_only_acc_pl(self, synth):
        out = synth.synthesize_aor2_participles(
            "πίπτω", {"aor": "ἔπεσον"}
        )
        # πεσοῦσα has no breve on final α; πεσούσᾱς has macron.
        assert out["active_aorist_participle_nom_f_sg"] == "πεσοῦσα"
        assert out["active_aorist_participle_acc_f_sg"] == "πεσοῦσαν"
        assert out["active_aorist_participle_acc_f_pl"] == "πεσούσᾱς"
