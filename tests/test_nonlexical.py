#!/usr/bin/env python3
"""Tests for the NON-LEXICAL token classifier (dilemma.nonlexical).

The classifier recognizes editorial/typographic residue that leaks into OCR'd
Greek corpora - variant marks, Greek numerals, bracket refs, abbreviations,
sigla, vowel-less fragments - so they stop counting as lemmatization failures.
Two things are asserted throughout:

  1. every non-lexical class is caught, and
  2. real words (including ones that superficially resemble an abbreviation or
     a numeral, and elided monosyllables) are NEVER misclassified.

Run with:
    cd dilemma && python -m pytest tests/test_nonlexical.py -x -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dilemma import (
    Dilemma,
    LemmaCandidate,
    classify_nonlexical,
    is_lexical,
    NONLEXICAL_CLASSES,
    NONLEXICAL_POS,
)
from dilemma import nonlexical as nl


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def d():
    """Combined-lookup Dilemma. The non-lexical short-circuit fires before the
    transformer, and the real words tested resolve via lookup, so no model
    weights are needed (these tests are not marked slow)."""
    return Dilemma(lang="all", resolve_articles=True)


# ---------------------------------------------------------------------------
# Each non-lexical class is recognized
# ---------------------------------------------------------------------------

class TestVariantMark:
    def test_bare(self):
        assert classify_nonlexical("γρ") == nl.VARIANT_MARK

    def test_with_period(self):
        assert classify_nonlexical("γρ.") == nl.VARIANT_MARK

    def test_with_apostrophe(self):
        assert classify_nonlexical("γρ'") == nl.VARIANT_MARK


class TestNumeral:
    @pytest.mark.parametrize("form", [
        "κζ'",       # 27 (apostrophe-marked)
        "ρʹ",        # 100 (keraia-marked, single letter)
        "χβ'",       # 602
        "ργ'",       # 103
        "ρνα'",      # 151
        ",αφ'",      # 1500 (thousands mark)
        "͵αφ'",       # 1500 (lower-keraia thousands mark)
    ])
    def test_marked_numerals(self, form):
        assert classify_nonlexical(form) == nl.NUMERAL

    def test_bare_all_consonant_numeral_is_nonlexical(self):
        # κζ / λϛ have no vowel, so they are caught as consonant-cluster even
        # without a numeral sign - still non-lexical, which is what matters.
        assert classify_nonlexical("κζ") is not None
        assert classify_nonlexical("λϛ") is not None

    def test_single_letter_bare_apostrophe_not_claimed(self):
        # ρ' with a plain apostrophe is ambiguous with an elision; only a real
        # keraia makes a single-letter numeral. Left to the resolver.
        assert classify_nonlexical("ρ'") is None


class TestBracketRef:
    @pytest.mark.parametrize("form", ["[76]", "[49—59]", "[49-59]", "(3)", "<12>"])
    def test_bracket_refs(self, form):
        assert classify_nonlexical(form) == nl.BRACKET_REF

    def test_bracketed_greek_is_not_a_ref(self):
        # A bracketed Greek word is not a numeric reference.
        assert classify_nonlexical("[λόγος]") != nl.BRACKET_REF


class TestAbbreviation:
    @pytest.mark.parametrize("form", ["fr.", "p.", "cf.", "Herod.", "Menand.", "ibid."])
    def test_latin_abbreviations(self, form):
        assert classify_nonlexical(form) == nl.ABBREVIATION


class TestSymbol:
    @pytest.mark.parametrize("form", ["†", "—", "·", "‖", "123", "§"])
    def test_symbols_and_bare_numbers(self, form):
        assert classify_nonlexical(form) == nl.SYMBOL


class TestConsonantCluster:
    @pytest.mark.parametrize("form", ["πλ", "ντ", "τρ", "γγ", "μς"])
    def test_vowelless_greek_runs(self, form):
        assert classify_nonlexical(form) == nl.CONSONANT_CLUSTER


# ---------------------------------------------------------------------------
# Real words are NEVER misclassified
# ---------------------------------------------------------------------------

# Ordinary vocabulary spanning eras/scripts, plus the transformer-tail lookup
# gaps the census surfaced (κουβουκλεῖον etc.) - all must stay lexical.
REAL_WORDS = [
    "λόγος", "ἄνθρωπος", "θεός", "μῆνιν", "ἄειδε", "θεά", "φημί", "γράφω",
    "καί", "δέ", "τε", "μή", "εἰ", "οὐ", "ὁ", "ἡ", "τό", "γάρ", "γε", "ἀλλά",
    "περί", "αὐτό", "εἶναι", "σπήλαιο", "σπήλαιον", "οὐδείς", "πρότερον",
    "ἐστιν", "μᾶλλον", "Σωκράτης", "ἐγώ", "Ἔρις", "κουβουκλεῖον",
    "δεσμωτήριον", "ἐπανασύνταξις", "Πανσανίας", "μάγιστρον", "πανάγιον",
    "ῥώς", "ὥς", "ᾧ", "ὦ",
]


@pytest.mark.parametrize("word", REAL_WORDS)
def test_real_words_are_lexical(word):
    assert classify_nonlexical(word) is None
    assert is_lexical(word) is True


# Words that superficially resemble an abbreviation/numeral but are real:
# elided monosyllables end in an apostrophe/breathing yet are genuine words.
@pytest.mark.parametrize("form", [
    "δ᾿", "μ᾿", "τ᾿", "γ᾿", "δ'", "ἀλλ᾿", "κατ᾿", "οὐδ᾿", "θεός’", "μᾶλλον’",
])
def test_elided_and_apostrophe_words_not_misclassified(form):
    assert classify_nonlexical(form) is None


def test_te_looks_like_a_numeral_but_is_a_word():
    # τε reads as τ(300)+ε(5), a descending numeral, but bare (no numeral sign)
    # it must not be claimed as a numeral.
    assert classify_nonlexical("τε") is None


# ---------------------------------------------------------------------------
# is_lexical / class invariants
# ---------------------------------------------------------------------------

def test_is_lexical_complement():
    assert is_lexical("λόγος") is True
    assert is_lexical("γρ") is False
    assert is_lexical("") is False
    assert is_lexical("   ") is False


def test_class_labels_are_registered():
    for form in ["γρ", "κζ'", "[76]", "fr.", "†", "πλ"]:
        assert classify_nonlexical(form) in NONLEXICAL_CLASSES


# ---------------------------------------------------------------------------
# Integration with the Dilemma resolver
# ---------------------------------------------------------------------------

class TestResolverIntegration:
    def test_static_predicates(self, d):
        assert d.classify_nonlexical("γρ") == nl.VARIANT_MARK
        assert d.is_lexical("γρ") is False
        assert d.is_lexical("λόγος") is True

    def test_lemmatize_returns_identity_for_nonlexical(self, d):
        # Non-words return unchanged (like the non-Greek passthrough) instead
        # of a manufactured model lemma.
        for form in ["γρ", "κζ'", "πλ"]:
            assert d.lemmatize(form) == form

    def test_lemmatize_real_words_unchanged(self, d):
        # These resolve via lookup, unaffected by the non-lexical branch.
        assert d.lemmatize("λόγος") == "λόγος"
        assert d.lemmatize("καί") == "καί"

    def test_verbose_tags_nonlexical(self, d):
        cands = d.lemmatize_verbose("γρ")
        assert len(cands) == 1
        c = cands[0]
        assert isinstance(c, LemmaCandidate)
        assert c.source == "nonlexical"
        assert c.via == nl.VARIANT_MARK
        assert c.tag == NONLEXICAL_POS == "X"
        assert c.is_lexical is False

    def test_verbose_real_word_is_lexical(self, d):
        cands = d.lemmatize_verbose("ἔριδι")
        assert cands
        assert all(c.source != "nonlexical" for c in cands)
        assert all(c.is_lexical for c in cands)

    def test_batch_mixes_lexical_and_nonlexical(self, d):
        out = d.lemmatize_batch(["λόγος", "γρ", "κζ'", "πλ"])
        assert out[0] == "λόγος"
        assert out[1] == "γρ"     # non-lexical, unchanged
        assert out[2] == "κζ'"    # non-lexical, unchanged
        assert out[3] == "πλ"     # non-lexical, unchanged

    def test_census_style_failure_filter(self, d):
        # A census consumer separates "not a word" from "failed real word".
        tokens = ["λόγος", "γρ", "κζ'", "[76]", "fr.", "θεός"]
        lexical = [t for t in tokens if d.is_lexical(t)]
        assert lexical == ["λόγος", "θεός"]
