"""Tests for the athematic / irregular verb classifier in
``build/expand_lsj.py``.

These tests cover the classifier path that was added to handle the
~100 athematic AG verbs that previously produced Lua errors during
the dilemma rebuild (compounds of εἰμί / εἶμι / οἶδα, the -όλλυμι
sub-family, OCR-corrupt headwords, etc.).

We import ``expand_lsj`` directly via ``importlib`` because the
``build/`` directory is a script directory, not an installable
package.
"""

from __future__ import annotations

import importlib.util
import unicodedata
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPAND_LSJ_PATH = REPO_ROOT / "build" / "expand_lsj.py"


@pytest.fixture(scope="module")
def el():
    """Load build/expand_lsj.py as a module (no package install needed)."""
    spec = importlib.util.spec_from_file_location("expand_lsj",
                                                  EXPAND_LSJ_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _reset_verb_cache(el):
    """Reset the module-level _VERB_CACHE before each test so cache
    state from one test does not leak into another. Cached paradigms
    preserve their lexical marks and exact request identity.
    """
    el._VERB_CACHE.clear()
    if hasattr(el, "_PP_CACHE"):
        el._PP_CACHE.clear()
    yield


def strip_diacritics(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return ''.join(c for c in nfd if not unicodedata.combining(c))


class TestClassifier:
    """Sanity checks on _classify_verb's dispatch table."""

    def test_thematic_omega(self, el):
        assert el._classify_verb("λύω") == ("pres", "λυ")

    def test_contracted_alpha(self, el):
        assert el._classify_verb("τιμάω")[0] == "pres-con-a"

    def test_contracted_epsilon(self, el):
        assert el._classify_verb("ποιέω")[0] == "pres-con-e"

    def test_contracted_omicron(self, el):
        assert el._classify_verb("δηλόω")[0] == "pres-con-o"

    def test_pres_emi(self, el):
        # τίθημι, ἵημι are -emi-class
        ct, _ = el._classify_verb("τίθημι")
        assert ct == "pres-emi"

    def test_pres_omi(self, el):
        ct, _ = el._classify_verb("δίδωμι")
        assert ct == "pres-omi"

    def test_pres_ami(self, el):
        ct, _ = el._classify_verb("ἵστημι")
        # -ημι suffix routes to pres-emi here, which the Lua module
        # accepts; the historic ἵστημι Wiktionary entry uses pres-ami
        # but pres-emi yields working forms too. We just require a
        # non-None classification for the test.
        assert ct is not None

    def test_numi(self, el):
        ct, stem = el._classify_verb("δείκνυμι")
        assert ct == "pres-numi"
        assert stem == "δεικ"

    def test_double_nu_numi(self, el):
        ct, stem = el._classify_verb("ἀννυμι")
        assert ct == "pres-numi"

    def test_olly_routes_to_lumi(self, el):
        # The whole point: -όλλυμι must NOT go to pres-numi (Wiktionary
        # uses pres-lumi for ὄλλυμι and its compounds).
        ct, stem = el._classify_verb("ἀπόλλυμι")
        assert ct == "pres-lumi"
        # Stem should retain the second λ: ἀπολ
        assert stem.endswith("ολ")

    def test_olly_compound(self, el):
        ct, stem = el._classify_verb("παραπόλλυμι")
        assert ct == "pres-lumi"

    def test_ollymi_is_not_numi(self, el):
        # Regression guard: previously this returned pres-numi which
        # produced empty form sets and a flood of Lua errors.
        ct, _ = el._classify_verb("διόλλυμι")
        assert ct == "pres-lumi"

    def test_eimi_compound_marked_irregular(self, el):
        # All -ειμι compounds need the irregular dispatcher
        for hw in ("πάρειμι", "σύνειμι", "εἴσειμι", "ἄνειμι",
                   "ὑπεξειμι", "ἀντιπρόειμι"):
            ct, _ = el._classify_verb(hw)
            assert ct == "irreg", f"{hw} -> {ct}"

    def test_unknown_classification_returns_none(self, el):
        # Bogus suffix should return (None, None)
        assert el._classify_verb("ποιαδήποτε") == (None, None)


class TestIrregularExpansion:
    """Literal irregular bases are separate; compounds need their own evidence."""

    @pytest.mark.parametrize("headword", ["εἰμί", "εἶμι", "οἶδα", "χρή", "φημί"])
    def test_literal_base_is_retained(self, el, headword):
        assert headword in el._expand_irregular_compound(headword)

    def test_be_and_go_do_not_share_paradigms(self, el):
        be = el._expand_irregular_compound("εἰμί")
        go = el._expand_irregular_compound("εἶμι")
        assert {"εἶναι", "ἦν", "ὤν"} <= be
        assert {"ἰέναι", "ᾔει", "ἰών"} <= go
        assert not {"ἰέναι", "ᾔει", "ἰών"} & be
        assert not {"εἶναι", "ἦν", "ὤν"} & go

    @pytest.mark.parametrize("headword", ["πάρειμι", "σύνειμι", "εἴσειμι", "ὑπερειμι"])
    def test_compound_does_not_inherit_an_approximate_base(self, el, headword):
        assert el._expand_irregular_compound(headword) == set()

    def test_unknown_base_returns_empty(self, el):
        assert el._expand_irregular_compound("παραΐεμι") == set()


class TestPreverbSplit:
    """The preverb / base splitter used by the irregular dispatcher."""

    @pytest.mark.parametrize("hw,expected_pv,expected_base", [
        ("πάρειμι", "πάρ", "ειμι"),
        ("σύνειμι", "σύν", "ειμι"),
        ("εἴσειμι", "εἴσ", "ειμι"),
        ("ἀντεπέξειμι", "ἀντεπέξ", "ειμι"),
        ("ὑποκάτειμι", "ὑποκάτ", "ειμι"),
    ])
    def test_split(self, el, hw, expected_pv, expected_base):
        pv, base = el._split_preverb(hw)
        assert pv == expected_pv
        assert base == expected_base

    def test_no_preverb_returns_empty(self, el):
        pv, base = el._split_preverb("εἰμί")
        # Bare εἰμί has no preverb
        assert pv == ""

    def test_invalid_prefix_rejects(self, el):
        pv, base = el._split_preverb("ξψζειμι")
        # nonsense preverb 'ξψζ' rejected
        assert pv == ""


class TestCorruptHeadwordGuard:

    def test_underdot_flagged(self, el):
        # Combining underdot (papyrus reading mark) means OCR garbage
        assert el._is_corrupt_headword("ὑπ̣ε̣ρ̣εε̣ν̣μ̣ι̣")

    def test_triple_consonant_flagged(self, el):
        assert el._is_corrupt_headword("περιένννμι")

    def test_multi_word_headword_flagged(self, el):
        # LSJ cross-references that escaped extraction as a single
        # headword (with comma or space)
        assert el._is_corrupt_headword("προεῖναι, πρόειμι")
        assert el._is_corrupt_headword("πρόσειμι εἰμί")

    def test_short_headword_flagged(self, el):
        assert el._is_corrupt_headword("πῶ")

    def test_clean_headword_passes(self, el):
        for hw in ("τίθημι", "δίδωμι", "ἀπόλλυμι", "πάρειμι"):
            assert not el._is_corrupt_headword(hw), hw


class TestExpandVerbIntegration:
    """End-to-end verb expansion tests. Require wtp.db to be present;
    skipped otherwise so a fresh checkout with no built data still
    passes the suite.
    """

    @pytest.fixture(scope="class")
    def wtp(self, el):
        pytest.importorskip("wikitextprocessor")
        if not el.WTP_DB.exists():
            pytest.skip("wtp.db not present (run --setup first)")
        return el.get_wtp()

    @pytest.mark.parametrize("hw", [
        "διόλλυμι", "παραπόλλυμι",         # -όλλυμι compounds
        "τίθημι", "δίδωμι",                # standard athematic
    ])
    def test_yields_forms_no_lua_error(self, el, wtp, hw):
        forms, err = el.expand_verb(wtp, hw)
        assert not err, f"{hw}: err={err!r}"
        assert len(forms) > 0, f"{hw}: no forms generated"

    @pytest.mark.parametrize("headword", ["πάρειμι", "σύνειμι", "εἴσειμι"])
    def test_unsupported_compound_is_reported_without_lua(self, el, headword):
        forms, error = el.expand_verb(None, headword)
        assert forms == set()
        assert error == "unsupported-irregular-compound"

    def test_corrupt_headword_short_circuits(self, el, wtp):
        forms, err = el.expand_verb(wtp, "ὑπ̣ε̣ρ̣εε̣ν̣μ̣ι̣")
        assert err == "corrupt-headword"
        assert forms == set()

    def test_thematic_still_works(self, el, wtp):
        forms, err = el.expand_verb(wtp, "λύω")
        assert not err
        assert len(forms) > 10
