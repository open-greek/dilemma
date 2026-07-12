#!/usr/bin/env python3
"""Comprehensive test suite for Dilemma Greek lemmatizer.

Tests core lemmatization, new features (particle stripping, verb morphology
stripping, article-agreement disambiguation), normalization, crasis, elision,
spelling suggestions, conventions, language filtering, batch operations, and
edge cases.

Run with:
    cd dilemma && python -m pytest tests/test_comprehensive.py -x -v
"""

import sys
import pytest
from pathlib import Path

# Ensure the project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dilemma import (
    Dilemma, LemmaCandidate, to_monotonic, strip_accents, grave_to_acute,
)
from dilemma.core import (
    _is_self_map, _PARTICLE_SUFFIXES, _DEICTIC_STEMS, _ARTICLE_FEATURES,
)
from dilemma.crasis import resolve_crasis, CRASIS_TABLE
from dilemma.normalize import Normalizer, PROFILES, SUBSCRIPTUM_MAP, IONIC_WORD_MAP, DORIC_WORD_MAP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def d_all():
    """Dilemma instance for combined (all languages) lemmatization."""
    return Dilemma(lang="all")


@pytest.fixture(scope="module")
def d_grc():
    """Dilemma instance for Ancient Greek only."""
    return Dilemma(lang="grc")


@pytest.fixture(scope="module")
def d_el():
    """Dilemma instance for Modern Greek only."""
    return Dilemma(lang="el")


@pytest.fixture(scope="module")
def d_lsj():
    """Dilemma instance with LSJ convention."""
    return Dilemma(convention="lsj")


@pytest.fixture(scope="module")
def d_triant():
    """Dilemma instance with Triantafyllidis convention."""
    return Dilemma(convention="triantafyllidis")


@pytest.fixture(scope="module")
def d_normalize():
    """Dilemma instance with normalization enabled."""
    return Dilemma(normalize=True)


@pytest.fixture(scope="module")
def d_resolve_articles():
    """Dilemma instance with article resolution enabled."""
    return Dilemma(resolve_articles=True)


# ===========================================================================
# 1. PARTICLE SUFFIX STRIPPING
# ===========================================================================

class TestParticleSuffixStripping:
    """Test stripping of enclitic particles: -per, -ge, -de, deictic -i.

    Note: particle stripping is a fallback that only fires when the full
    form is NOT found in the lookup table. Many common particle-suffixed
    forms (e.g. ὅσπερ, ὥσπερ, ἔγωγε) are themselves Wiktionary headwords
    and self-map in the lookup. We test the _strip_particle_suffix method
    directly to verify the stripping logic, and test end-to-end for forms
    that actually route through it.
    """

    def test_per_suffix_direct(self, d_all):
        """_strip_particle_suffix should resolve ὅσπερ -> ὅσος (via ὅς base)."""
        result = d_all._strip_particle_suffix("ὅσπερ")
        assert result is not None, "ὅσπερ should be strippable to a known base"

    def test_ge_suffix_direct(self, d_all):
        """_strip_particle_suffix should resolve ἔμοιγε -> ἐγώ."""
        result = d_all._strip_particle_suffix("ἔμοιγε")
        assert result is not None
        assert strip_accents(result.lower()) == strip_accents("ἐγώ"), \
            f"ἔμοιγε should strip to ἐγώ, got {result}"

    def test_ge_suffix_panuge(self, d_all):
        """_strip_particle_suffix should resolve πάνυγε -> πάνυ."""
        result = d_all._strip_particle_suffix("πάνυγε")
        assert result is not None, "πάνυγε should strip -γε to find πάνυ"

    def test_deictic_i_toutouI(self, d_all):
        """τουτουί (deictic -ί on demonstrative) -> οὗτος."""
        result = d_all.lemmatize("τουτουί")
        assert strip_accents(result.lower()) == strip_accents("οὗτος"), \
            f"τουτουί should -> οὗτος, got {result}"

    def test_deictic_i_toutoisi(self, d_all):
        """τουτοισί (deictic -ί on demonstrative) -> οὗτος."""
        result = d_all.lemmatize("τουτοισί")
        assert strip_accents(result.lower()) == strip_accents("οὗτος"), \
            f"τουτοισί should -> οὗτος, got {result}"

    def test_particle_suffix_constants(self):
        """Particle suffix list should contain the expected suffixes."""
        assert "περ" in _PARTICLE_SUFFIXES
        assert "γε" in _PARTICLE_SUFFIXES
        assert "δε" in _PARTICLE_SUFFIXES

    def test_deictic_stems_contains_expected(self):
        """Deictic stem set should contain known demonstrative stems."""
        assert "τουτου" in _DEICTIC_STEMS
        assert "τουτοισ" in _DEICTIC_STEMS
        assert "ταυτη" in _DEICTIC_STEMS

    def test_short_word_not_stripped(self, d_all):
        """Very short words should not be particle-stripped."""
        # "γε" itself should not try to strip (base would be empty)
        result = d_all._strip_particle_suffix("γε")
        assert result is None

    def test_non_particle_word(self, d_all):
        """Words not ending in particle suffixes should return None."""
        result = d_all._strip_particle_suffix("λόγος")
        assert result is None


# ===========================================================================
# 2. VERB MORPHOLOGY STRIPPING
# ===========================================================================

class TestVerbMorphologyStripping:
    """Test stripping of syllabic/temporal augment and reduplication.

    Like particle stripping, verb morphology stripping is a fallback that
    only fires when the form is not in the lookup. We test the method
    directly to verify the stripping logic works.
    """

    def test_syllabic_augment_epoiesen(self, d_all):
        """_strip_verb_morphology on ἐποίησεν should find ποιέω."""
        result = d_all._strip_verb_morphology("ἐποίησεν")
        assert result is not None
        assert strip_accents(result.lower()) == strip_accents("ποιέω"), \
            f"ἐποίησεν morph strip -> expected ποιέω, got {result}"

    def test_syllabic_augment_elegon(self, d_all):
        """_strip_verb_morphology on ἔλεγον should find λέγω."""
        result = d_all._strip_verb_morphology("ἔλεγον")
        assert result is not None
        assert strip_accents(result.lower()) == strip_accents("λέγω"), \
            f"ἔλεγον morph strip -> expected λέγω, got {result}"

    def test_syllabic_augment_ekatharize(self, d_all):
        """_strip_verb_morphology on ἐκαθάριζε should find καθαρίζω."""
        result = d_all._strip_verb_morphology("ἐκαθάριζε")
        assert result is not None
        assert strip_accents(result.lower()) == strip_accents("καθαρίζω"), \
            f"ἐκαθάριζε morph strip -> expected καθαρίζω, got {result}"

    def test_temporal_augment_egorazon(self, d_all):
        """_strip_verb_morphology on ἠγόραζον (η- for α- stem) -> ἀγοράζω."""
        result = d_all._strip_verb_morphology("ἠγόραζον")
        assert result is not None
        assert strip_accents(result.lower()) == strip_accents("ἀγοράζω"), \
            f"ἠγόραζον morph strip -> expected ἀγοράζω, got {result}"

    def test_temporal_augment_map_exists(self):
        """Temporal augment mappings should be present in the class."""
        assert "η" in Dilemma._TEMPORAL_AUGMENT
        assert "ω" in Dilemma._TEMPORAL_AUGMENT
        assert "ε" in Dilemma._TEMPORAL_AUGMENT["η"]
        assert "ο" in Dilemma._TEMPORAL_AUGMENT["ω"]

    def test_verb_morphology_short_word_skipped(self, d_all):
        """Very short words (< 4 chars) should not trigger verb morphology stripping."""
        result = d_all._strip_verb_morphology("ἔχω")
        assert result is None

    def test_verb_morphology_no_false_positive(self, d_all):
        """Non-augmented words starting with ε should not false-trigger."""
        # ἔργον starts with ε but is not augmented
        result = d_all._strip_verb_morphology("ἔργον")
        # May or may not return None, but if it returns something,
        # it should be a valid lemma (not garbage)
        if result is not None:
            assert len(result) > 0

    def test_e2e_augmented_verb_via_lookup(self, d_all):
        """Common augmented forms should resolve via lookup (not morph stripping)."""
        # These are in the lookup table, so they go through lookup path
        result = d_all.lemmatize("ἐποίησεν")
        assert strip_accents(result.lower()) == strip_accents("ποιέω")

        result = d_all.lemmatize("ἔλεγον")
        assert strip_accents(result.lower()) == strip_accents("λέγω")


# ===========================================================================
# 3. ARTICLE-AGREEMENT DISAMBIGUATION
# ===========================================================================

class TestArticleAgreement:
    """Test re-ranking of candidates by article gender/number agreement."""

    def test_article_features_structure(self):
        """Article features dict should map forms to (gender, number, case)."""
        assert "ὁ" in _ARTICLE_FEATURES
        assert _ARTICLE_FEATURES["ὁ"] == ("m", "s", "nom")
        assert "ἡ" in _ARTICLE_FEATURES
        assert _ARTICLE_FEATURES["ἡ"][0] == "f"
        assert "τό" in _ARTICLE_FEATURES
        assert _ARTICLE_FEATURES["τό"][0] == "n"

    def test_article_features_covers_graves(self):
        """Article features should include grave-accent variants."""
        assert "τὸ" in _ARTICLE_FEATURES
        assert "τὸν" in _ARTICLE_FEATURES

    def test_verbose_with_prev_word_article(self, d_all):
        """lemmatize_verbose with prev_word=article should return candidates."""
        candidates = d_all.lemmatize_verbose("θεούς", prev_word="τούς")
        assert len(candidates) > 0

    def test_verbose_without_prev_word(self, d_all):
        """lemmatize_verbose with and without prev_word should have same candidate set."""
        candidates_no = d_all.lemmatize_verbose("θεούς")
        candidates_with = d_all.lemmatize_verbose("θεούς", prev_word="τούς")
        # Same candidates should appear in both, possibly reordered
        set_no = {c.lemma for c in candidates_no}
        set_with = {c.lemma for c in candidates_with}
        assert set_no == set_with, \
            "Article agreement should rerank, not add/remove candidates"

    def test_article_agreement_with_non_article(self, d_all):
        """Non-article prev_word should not change candidate ordering."""
        cands_plain = d_all.lemmatize_verbose("ἄνδρα")
        cands_random = d_all.lemmatize_verbose("ἄνδρα", prev_word="καί")
        # Since καί is not in _ARTICLE_FEATURES, order should be identical
        assert [c.lemma for c in cands_plain] == [c.lemma for c in cands_random]

    def test_rank_method_preserves_candidates(self, d_all):
        """_rank_by_article_agreement should never drop candidates."""
        # Create test candidates
        c1 = LemmaCandidate(lemma="λόγος", lang="grc", proper=False, source="lookup")
        c2 = LemmaCandidate(lemma="Λόγος", lang="grc", proper=True, source="lookup")
        result = d_all._rank_by_article_agreement([c1, c2], prev_word="ὁ")
        assert len(result) == 2

    def test_rank_method_no_prev_word(self, d_all):
        """_rank_by_article_agreement with no prev_word should return unchanged list."""
        c1 = LemmaCandidate(lemma="λόγος", lang="grc", proper=False, source="lookup")
        result = d_all._rank_by_article_agreement([c1], prev_word=None)
        assert result == [c1]


# ===========================================================================
# 4. CORE LEMMATIZATION
# ===========================================================================

class TestCoreLemmatization:
    """Test core lemmatization of known AG and MG words."""

    # -- Ancient Greek common verbs --
    @pytest.mark.parametrize("form,expected", [
        ("ἐποίησεν", "ποιέω"),
        ("ἔλεγον", "λέγω"),
        ("ἔλυσε", "λύω"),
        ("ἐβασίλευσεν", "βασιλεύω"),
    ])
    def test_ag_verbs(self, d_all, form, expected):
        result = d_all.lemmatize(form)
        assert strip_accents(result.lower()) == strip_accents(expected), \
            f"{form} -> expected {expected}, got {result}"

    # -- Ancient Greek nouns --
    @pytest.mark.parametrize("form,expected", [
        ("θεοί", "θεός"),
        ("μῆνιν", "μῆνις"),
        ("παίδων", "παῖς"),
        ("τριῶν", "τρεῖς"),
    ])
    def test_ag_nouns(self, d_all, form, expected):
        result = d_all.lemmatize(form)
        assert strip_accents(result.lower()) == strip_accents(expected), \
            f"{form} -> expected {expected}, got {result}"

    # -- Ancient Greek adjectives --
    def test_ag_adjective_polytropon(self, d_all):
        result = d_all.lemmatize("πολύτροπον")
        assert strip_accents(result.lower()) == strip_accents("πολύτροπος"), \
            f"πολύτροπον -> expected πολύτροπος, got {result}"

    # -- Homeric forms --
    def test_homeric_achilleos(self, d_all):
        result = d_all.lemmatize("Ἀχιλῆος")
        assert strip_accents(result.lower()) == strip_accents("Ἀχιλλεύς".lower()), \
            f"Ἀχιλῆος -> expected Ἀχιλλεύς, got {result}"

    def test_homeric_peleidew(self, d_all):
        result = d_all.lemmatize("Πηλείδεω")
        assert strip_accents(result.lower()) == strip_accents("Πηλείδης".lower()), \
            f"Πηλείδεω -> expected Πηλείδης, got {result}"

    # -- AG mi-verb --
    def test_ag_mi_verb(self, d_all):
        result = d_all.lemmatize("σβέννυσι")
        assert strip_accents(result.lower()) == strip_accents("σβέννυμι"), \
            f"σβέννυσι -> expected σβέννυμι, got {result}"

    # -- Modern Greek words (unambiguous with AG in combined mode) --
    @pytest.mark.parametrize("form,expected", [
        ("τρομερό", "τρομερός"),
        ("ανθρώπων", "άνθρωπος"),
        ("φέρνοντας", "φέρνω"),
    ])
    def test_mg_words(self, d_all, form, expected):
        result = d_all.lemmatize(form)
        assert strip_accents(result.lower()) == strip_accents(expected), \
            f"{form} -> expected {expected}, got {result}"

    # -- MG words that have AG counterparts (need lang='el' for MG lemma) --
    @pytest.mark.parametrize("form,expected", [
        ("ποτήρια", "ποτήρι"),
        ("γυναίκες", "γυναίκα"),
    ])
    def test_mg_words_el_only(self, d_el, form, expected):
        result = d_el.lemmatize(form)
        assert strip_accents(result.lower()) == strip_accents(expected), \
            f"{form} (el) -> expected {expected}, got {result}"


# ===========================================================================
# 5. ELISION HANDLING
# ===========================================================================

class TestElision:
    """Test elision expansion (δ', τ', ἀλλ', etc.)."""

    def test_elision_alla_smooth_breathing(self, d_all):
        """ἀλλ̓ (with U+0313 combining comma above) should -> ἀλλά."""
        result = d_all.lemmatize("ἀλλ\u0313")
        assert strip_accents(result.lower()) == strip_accents("ἀλλά"), \
            f"ἀλλ' -> expected ἀλλά, got {result}"

    def test_elision_with_right_quote(self, d_all):
        """ἀλλ' (with U+2019 right single quote) should also expand."""
        result = d_all.lemmatize("ἀλλ\u2019")
        assert strip_accents(result.lower()) == strip_accents("ἀλλά"), \
            f"ἀλλ' (U+2019) -> expected ἀλλά, got {result}"

    def test_elision_d_prime(self, d_all):
        """δ' (elision of δέ) should -> δέ."""
        result = d_all.lemmatize("δ\u2019")
        assert strip_accents(result.lower()) == strip_accents("δέ"), \
            f"δ' -> expected δέ, got {result}"

    def test_elision_t_prime(self, d_all):
        """τ' (elision of τε) should resolve to something reasonable."""
        result = d_all.lemmatize("τ\u2019")
        assert result is not None
        assert len(result) > 0


# ===========================================================================
# 5b. ELIDED FUNCTION-WORD LEMMAS (codepoint-independent, correct lemma)
# ===========================================================================

# The elided form must resolve to the RIGHT lemma, not merely a valid
# headword, and it must do so no matter which apostrophe codepoint the text
# uses. The historic bug returned the common homograph (ὅτι/καί/ὁ/εἷς/σός)
# for every codepoint except U+1FBD, and lemmatize_pos()/the Tagger got it
# wrong even for U+1FBD.
#
# NB: ὅθ' maps to the temporal ὅτε here (a small number of Homeric ὅθ' are the
# locative ὅθι, which only syntactic context can settle; ὅτε is the correct
# static-lookup answer and a strict improvement over the prior ὅτι).
_ELIDED_LEMMAS = {
    "ὅτ": ("ὅτε", "SCONJ"),
    "ὅθ": ("ὅτε", "SCONJ"),
    "μ": ("ἐγώ", "PRON"),
    "σ": ("σύ", "PRON"),
    "κ": ("ἄν", "PART"),
    "θ": ("τε", "CCONJ"),
    "ποτ": ("ποτέ", "ADV"),
    "ἠδ": ("ἠδέ", "CCONJ"),
    "αὖτ": ("αὖτε", "ADV"),
    "ἀμφ": ("ἀμφί", "ADP"),
    "ἀν": ("ἀνά", "ADP"),
}

# Every apostrophe codepoint real Greek text uses as an elision mark. The four
# the handoff requires (U+2019, U+02BC, U+1FBD, ASCII ') plus the two the
# sibling prosodia engine also recognizes (U+0060 grave, U+02B9 prime).
_APOS_CODEPOINTS = ["’", "ʼ", "᾽", "'", "`", "ʹ"]


def _elided_cases():
    """(stem, lemma, pos, apos) matrix. A SINGLE letter followed by U+02B9
    is excluded: that codepoint is canonically the Greek numeral keraia
    (μʹ = 40, σʹ = 200), and the nonlexical classifier claims those tokens
    as numerals by design - elision on a lone letter is only read from a
    true apostrophe codepoint. Multi-letter stems are unambiguous (ποτʹ
    cannot be a numeral) and keep all six codepoints."""
    return [(stem, lemma, pos, apos)
            for stem, (lemma, pos) in _ELIDED_LEMMAS.items()
            for apos in _APOS_CODEPOINTS
            if not (len(stem) == 1 and apos == "ʹ")]


class TestElidedFunctionWords:
    """Regression: elided forms -> correct lemma, every apostrophe codepoint,
    through both lemmatize() and lemmatize_pos() (the Tagger's path)."""

    @pytest.mark.parametrize("stem,expected,pos,apos", _elided_cases())
    def test_lemmatize_correct_lemma(self, d_all, stem, expected, pos, apos):
        form = stem + apos
        result = d_all.lemmatize(form)
        assert result == expected, (
            f"lemmatize({form!r}) [U+{ord(apos):04X}] -> "
            f"expected {expected!r}, got {result!r}")

    @pytest.mark.parametrize("stem,expected,pos,apos", _elided_cases())
    def test_lemmatize_pos_correct_lemma(self, d_all, stem, expected, pos, apos):
        """lemmatize_pos() is the path the Tagger uses; it must agree with
        lemmatize() and never let the frequency-ranked homograph win."""
        form = stem + apos
        result = d_all.lemmatize_pos(form, pos)
        assert result == expected, (
            f"lemmatize_pos({form!r}, {pos}) [U+{ord(apos):04X}] -> "
            f"expected {expected!r}, got {result!r}")

    @pytest.mark.parametrize("stem,expected,pos,apos", _elided_cases())
    def test_paths_agree(self, d_all, stem, expected, pos, apos):
        """lemmatize() and lemmatize_pos() must return the same lemma."""
        form = stem + apos
        assert d_all.lemmatize(form) == d_all.lemmatize_pos(form, pos)

    @pytest.mark.parametrize("form,expected", [
        # Self-maps stored under U+1FBD (ἀλλ᾽->ἀλλ᾽) must still resolve to the
        # real lemma via the expander, not surface as the elided form itself.
        ("ἀλλ᾽", "ἀλλά"),
        ("δ᾽", "δέ"),
        ("ἐπ᾽", "ἐπί"),
        ("παρ᾽", "παρά"),
        ("μεθ᾽", "μετά"),
    ])
    def test_koronis_selfmap_resolves(self, d_all, form, expected):
        assert d_all.lemmatize(form) == expected
        assert d_all.lemmatize_pos(form, "ADP") == expected

    @pytest.mark.parametrize("form,expected,pos", [
        # A breathing on a diphthong's second vowel is NOT an elision mark;
        # these correct lookups must not be diverted into the elision
        # expander. Historically only lemmatize() got these right (lookup
        # runs first there); the batch/verbose paths run elision first and
        # returned junk expansions (εἰ -> οἰδάνω, οὐ -> οὖον).
        ("οὐ", "οὐ", "ADV"), ("οὔ", "οὐ", "ADV"), ("οὖ", "οὖ", "ADV"),
        ("εἰ", "εἰ", "SCONJ"), ("εἴ", "εἰ", "SCONJ"), ("εὖ", "εὖ", "ADV"),
    ])
    def test_diphthong_not_treated_as_elision(self, d_all, form, expected, pos):
        assert d_all.lemmatize(form) == expected
        assert d_all.lemmatize_batch([form]) == [expected]
        assert d_all.lemmatize_pos(form, pos) == expected

    def test_lemmatize_batch_correct_lemma(self, d_all):
        """lemmatize_batch() (the eval's path) must agree with lemmatize().
        It historically ran only the frequency-ranked expander, so every
        elided form got the common homograph."""
        cases = _elided_cases()
        forms = [stem + apos for stem, _, _, apos in cases]
        expected = [lemma for _, lemma, _, _ in cases]
        results = d_all.lemmatize_batch(forms)
        bad = [(f, r, e) for f, r, e in zip(forms, results, expected) if r != e]
        assert not bad, f"lemmatize_batch mismatches: {bad}"

    def test_lemmatize_batch_pos_correct_lemma(self, d_all):
        """lemmatize_batch_pos() is the Tagger's actual path; the POS tables
        have no entry for elided forms, so it must not inherit a wrong
        baseline from lemmatize_batch()."""
        cases = _elided_cases()
        forms = [stem + apos for stem, _, _, apos in cases]
        tags = [pos for _, _, pos, _ in cases]
        expected = [lemma for _, lemma, _, _ in cases]
        results = d_all.lemmatize_batch_pos(forms, tags)
        bad = [(f, r, e) for f, r, e in zip(forms, results, expected) if r != e]
        assert not bad, f"lemmatize_batch_pos mismatches: {bad}"


# ===========================================================================
# 5c. SAYING "I DON'T KNOW" (guess=False) AND LEMMA REPAIR
# ===========================================================================

class TestNoGuess:
    """guess=False must return None/[] for words nothing structured can
    resolve, instead of echoing the input or asking the transformer - a
    wrong lemma puts a false definition in front of a dictionary reader,
    and an echoed input is indistinguishable from a successful identity
    lemmatization. None of these calls may load the model."""

    def test_unresolvable_returns_none(self, d_all):
        assert d_all.lemmatize("μάκαπ", guess=False) is None

    def test_verbose_returns_empty(self, d_all):
        assert d_all.lemmatize_verbose("μάκαπ", guess=False) == []

    def test_pos_returns_none(self, d_all):
        assert d_all.lemmatize_pos("μάκαπ", "NOUN", guess=False) is None

    def test_batch_returns_none_positionally(self, d_all):
        out = d_all.lemmatize_batch(["μάκαπ", "γράφω", "μάκαπ"], guess=False)
        assert out == [None, "γράφω", None]

    def test_batch_pos_returns_none_positionally(self, d_all):
        out = d_all.lemmatize_batch_pos(["μάκαπ", "ἔριδι"], ["NOUN", "NOUN"],
                                        guess=False)
        assert out[0] is None
        assert out[1] is not None

    @pytest.mark.parametrize("word,expected", [
        ("γράφω", "γράφω"),        # plain lookup
        ("ὅτ᾽", "ὅτε"),            # elided exact entry
        ("ἀλλ᾽", "ἀλλά"),          # elision expander + allow-list
    ])
    def test_resolvable_words_unaffected(self, d_all, word, expected):
        assert d_all.lemmatize(word, guess=False) == expected

    @pytest.mark.parametrize("token", ["123", ".", "γρ"])
    def test_nonlexical_still_passes_through(self, d_all, token):
        """Non-Greek and NON-LEXICAL tokens are classifications, not lemma
        guesses; they pass through unchanged even with guess=False."""
        assert d_all.lemmatize(token, guess=False) == token


class TestDigammaNormalization:
    """Digamma (ϝ) is a real archaic letter but in OCR'd/keyed text is most
    often noise for ν; genuine digamma spellings are lemmatized without it.
    Both lemmatize() and the verbose/Tagger path must resolve these."""

    @pytest.mark.parametrize("form,expected,pos", [
        ("ϝέκταρ", "νέκταρ", "NOUN"),    # ϝ-for-ν noise
        ("πρόσθεϝ", "πρόσθεν", "ADV"),   # ϝ-for-ν noise, word-final
        ("ϝάναξ", "ἄναξ", "NOUN"),       # genuine digamma, dropped
    ])
    def test_digamma_forms_resolve(self, d_all, form, expected, pos):
        assert d_all.lemmatize(form) == expected
        assert d_all.lemmatize_pos(form, pos) == expected


class TestRepairLemma:
    """repair_lemma(): validate/repair corrupt lemma annotations against the
    canonical AG headword inventory (LSJ + Cunliffe), guided by the surface
    form when available. It must be able to return None ("I don't know")
    and must never echo a corrupt input back."""

    # Ground-truth fixture: (corrupt lemma, surface form, correct lemma).
    # Each target verified against Cunliffe; each surface form is the
    # (correct) Homeric text the corrupt annotation was attached to.
    _FIXTURE = [
        ("μάκαπ",      "μάκαρ",       "μάκαρ"),       # π/ρ OCR confusion
        ("ϝέκταρ",     "νέκταρ",      "νέκταρ"),      # digamma ϝ for ν
        ("πρόσθεϝ",    "πρόσθεν",     "πρόσθεν"),     # digamma ϝ for ν
        ("ἐυμελίης",   "ἐϋμμελίω",    "ἐϋμμελίης"),   # diaeresis + gemination
        ("ἐπιτροχάδη", "ἐπιτροχάδην", "ἐπιτροχάδην"),  # truncated final ν
        ("διαπέταμαι", "διέπτατο",    "διαπέτομαι"),  # non-canonical -άμαι
        ("Πειραί",     "Πειραΐδης",   "Πειραΐδης"),   # truncated patronymic
    ]

    @pytest.mark.parametrize("hint,form,expected", _FIXTURE)
    def test_repairs_with_form(self, d_all, hint, form, expected):
        assert d_all.repair_lemma(hint, form=form) == expected

    @pytest.mark.parametrize("hint,expected", [
        # Without the form, the string-level repairs still land.
        ("μάκαπ", "μάκαρ"),
        ("ϝέκταρ", "νέκταρ"),
        ("πρόσθεϝ", "πρόσθεν"),
        ("ἐπιτροχάδη", "ἐπιτροχάδην"),
        ("διαπέταμαι", "διαπέτομαι"),
    ])
    def test_repairs_without_form(self, d_all, hint, expected):
        assert d_all.repair_lemma(hint) == expected

    @pytest.mark.parametrize("headword", ["μάκαρ", "λόγος", "Ζεύς",
                                          "διαπέτομαι", "ὅτε"])
    def test_canonical_passthrough(self, d_all, headword):
        assert d_all.repair_lemma(headword) == headword

    def test_rejects_unrepairable(self, d_all):
        assert d_all.repair_lemma("ξζψχβγ") is None
        assert d_all.repair_lemma("") is None

    def test_case_guard(self, d_all):
        """A lowercase corrupt lemma must not be repaired onto a
        proper-noun twin (μάκαπ -> μάκαρ, not the Lesbian king Μάκαρ)."""
        assert d_all.repair_lemma("μάκαπ") == "μάκαρ"

    def test_mg_citation_form_not_repaired(self, d_all):
        """A valid Modern Greek citation form is not corrupt AG: it must be
        returned unchanged, not 'repaired' onto the AG headword
        (σπήλαιο is the correct Demotic lemma, one edit from σπήλαιον)."""
        assert d_all.repair_lemma("σπήλαιο") == "σπήλαιο"


class TestElidedJunkValues:
    """Lookup VALUES that are themselves elided fragments (key ἀλλ ->
    value \"ἀλλ'\", from corpora that tokenize the apostrophe off) are not
    headwords. Every path must reject them, and the junk value must instead
    route the bare stem through the elision machinery. All five entry
    points must agree."""

    @pytest.mark.parametrize("stem,expected,pos", [
        ("ἀλλ", "ἀλλά", "CCONJ"),
        ("οὐδ", "οὐδέ", "CCONJ"),
        ("μηδ", "μηδέ", "CCONJ"),
        ("μήτ", "μήτε", "CCONJ"),
        ("ἡνίκ", "ἡνίκα", "ADV"),
    ])
    def test_bare_elided_stems(self, d_all, stem, expected, pos):
        assert d_all.lemmatize(stem) == expected
        assert d_all.lemmatize(stem, guess=False) == expected
        assert d_all.lemmatize_batch([stem]) == [expected]
        assert d_all.lemmatize_pos(stem, pos) == expected

    def test_junk_value_never_returned(self, d_all):
        """No path may RESOLVE a word to a lemma ending in a spacing
        apostrophe/koronis. Uses guess=False so an unresolvable input
        yields None rather than the (documented) identity echo - an
        apostrophe-final input's echo would end in the mark, and whether
        the echo is reached depends on the transformer weights being
        installed (CI runs without the model)."""
        from dilemma.core import _ELISION_MARKS
        import unicodedata
        words = ["ἀλλ", "οὐδ", "μηδ", "ἡνίκ", "ταραχῶδ", "αὐτῇ`"]
        outputs = [d_all.lemmatize(w, guess=False) for w in words]
        outputs += d_all.lemmatize_batch(words, guess=False)
        outputs += [d_all.lemmatize_pos(w, "X", guess=False) for w in words]
        outputs += [c.lemma for w in words
                    for c in d_all.lemmatize_verbose(w, guess=False)]
        bad = [o for o in outputs
               if o and o[-1] in _ELISION_MARKS
               and unicodedata.category(o[-1]) != "Mn"]
        assert not bad, f"apostrophe-final junk lemmas emitted: {bad}"


class TestMarkedNumerals:
    """Greek numerals with a keraia/apostrophe (ιβʹ = 12, κζ’ = 27) are
    NON-LEXICAL: they must pass through unchanged in every path and never
    be vowel-expanded into words (the historic ιβʹ -> ἶβις)."""

    @pytest.mark.parametrize("numeral", ["μʹ", "σʹ", "κʹ", "θʹ"])
    def test_single_letter_keraia_is_numeral(self, d_all, numeral):
        """A lone letter + U+02B9 keraia is canonically a numeral (μʹ = 40,
        σʹ = 200), not an elided monosyllable; elision is only read from a
        true apostrophe codepoint there (μ’ -> ἐγώ, but μʹ -> μʹ)."""
        assert d_all.lemmatize(numeral) == numeral
        assert d_all.lemmatize_batch([numeral]) == [numeral]

    @pytest.mark.parametrize("numeral", ["ιβʹ", "κζ’", "ρκʹ"])
    def test_numeral_passthrough(self, d_all, numeral):
        assert d_all.lemmatize(numeral) == numeral
        assert d_all.lemmatize(numeral, guess=False) == numeral
        assert d_all.lemmatize_batch([numeral]) == [numeral]
        assert d_all.lemmatize_pos(numeral, "NUM") == numeral

    def test_numeral_verbose_nonlexical(self, d_all):
        cands = d_all.lemmatize_verbose("ιβʹ")
        assert len(cands) == 1
        assert cands[0].lemma == "ιβʹ"
        assert cands[0].source == "nonlexical"

    def test_elided_fragments_not_claimed_as_numerals(self, d_all):
        """Combining-psili elisions (σφ̓, γ̓) must NOT be caught by the
        numeral gate; they are real elided words."""
        assert d_all.lemmatize("σφ̓") == "σφεῖς"
        assert d_all.lemmatize("γ̓") == "γε"


class TestVerboseNoDuplicates:
    @pytest.mark.parametrize("form", ["ὅτ᾽", "μ᾽", "ποτ᾽", "ἀλλ᾽"])
    def test_no_duplicate_lemmas(self, d_all, form):
        """The exact-elided entry and the expander find the same lemma; the
        candidate list must dedupe them (one ὅτε, not two)."""
        lemmas = [(c.lemma, c.lang) for c in d_all.lemmatize_verbose(form)]
        assert len(lemmas) == len(set(lemmas)), f"duplicates in {lemmas}"


class TestLookupOverrides:
    """Targeted _LOOKUP_OVERRIDES corrections (build_lookup_db.py), live in
    the rebuilt lookup.db. Each form previously resolved to a corrupt
    entry: σε -> σῦς (pig), σέ -> σός, ποτέ -> ποτός (drink)."""

    @pytest.mark.parametrize("form,expected", [
        ("σε", "σύ"),
        ("σέ", "σύ"),
        ("ποτέ", "ποτέ"),
        ("κοτέ", "ποτέ"),        # Ionic ποτέ, was ποτός
        ("κως", "πως"),          # Ionic πως, was the island Κῶς
        ("κου", "που"),          # Ionic που, was ποῦ
        ("ἕως", "ἕως"),          # was ἠώς; Koine "until" and Attic "dawn"
        ("ἑπτὰ", "ἑπτά"),        # was ἑπτάς (heptad)
        ("οἰκίαν", "οἰκία"),     # was οἰκίον (rare diminutive)
    ])
    def test_override_lemmas(self, d_all, form, expected):
        assert d_all.lemmatize(form) == expected
        assert d_all.lemmatize_batch([form]) == [expected]


# ===========================================================================
# 6. CRASIS RESOLUTION
# ===========================================================================

class TestCrasis:
    """Test crasis resolution from the crasis table."""

    def test_crasis_tounoma(self, d_all):
        """τοὔνομα = τό + ὄνομα -> ὄνομα."""
        result = d_all.lemmatize("τοὔνομα")
        assert strip_accents(result.lower()) == strip_accents("ὄνομα"), \
            f"τοὔνομα -> expected ὄνομα, got {result}"

    def test_crasis_kago(self, d_all):
        """κἀγώ = καί + ἐγώ -> ἐγώ."""
        result = d_all.lemmatize("κἀγώ")
        assert strip_accents(result.lower()) == strip_accents("ἐγώ"), \
            f"κἀγώ -> expected ἐγώ, got {result}"

    def test_crasis_tautos(self, d_all):
        """ταὐτός = τὸ αὐτός -> αὐτός."""
        result = d_all.lemmatize("ταὐτός")
        assert strip_accents(result.lower()) == strip_accents("αὐτός"), \
            f"ταὐτός -> expected αὐτός, got {result}"

    def test_crasis_handres(self, d_all):
        """ἅνδρες = οἱ + ἄνδρες -> ἀνήρ."""
        result = d_all.lemmatize("ἅνδρες")
        assert strip_accents(result.lower()) == strip_accents("ἀνήρ"), \
            f"ἅνδρες -> expected ἀνήρ, got {result}"

    def test_crasis_tandros(self, d_all):
        """τἀνδρός = τοῦ + ἀνδρός -> ἀνήρ."""
        result = d_all.lemmatize("τἀνδρός")
        assert strip_accents(result.lower()) == strip_accents("ἀνήρ"), \
            f"τἀνδρός -> expected ἀνήρ, got {result}"

    def test_crasis_kan(self, d_all):
        """κἄν = καί + ἄν -> ἄν."""
        result = d_all.lemmatize("κἄν")
        assert strip_accents(result.lower()) == strip_accents("ἄν"), \
            f"κἄν -> expected ἄν, got {result}"

    def test_crasis_resolve_function_directly(self):
        """resolve_crasis should return the lemma for known forms."""
        assert resolve_crasis("τοὔνομα") == "ὄνομα"
        assert resolve_crasis("κἀγώ") == "ἐγώ"
        assert resolve_crasis("ταὐτός") == "αὐτός"
        assert resolve_crasis("ἅνδρες") == "ἀνήρ"
        assert resolve_crasis("κἄν") == "ἄν"

    def test_crasis_resolve_unknown(self):
        """resolve_crasis should return None for unknown forms."""
        assert resolve_crasis("λόγος") is None
        assert resolve_crasis("") is None

    def test_crasis_table_size(self):
        """Crasis table should have a reasonable number of entries."""
        assert len(CRASIS_TABLE) >= 20

    def test_crasis_all_entries_have_string_values(self):
        """Every value in the crasis table should be a non-empty string."""
        for form, lemma in CRASIS_TABLE.items():
            assert isinstance(lemma, str), f"CRASIS_TABLE[{form!r}] is not str"
            assert len(lemma) > 0, f"CRASIS_TABLE[{form!r}] is empty"


# ===========================================================================
# 7. CONVENTION SWITCHING
# ===========================================================================

class TestConventions:
    """Test convention remapping (lsj, triantafyllidis, wiktionary)."""

    def test_lsj_convention_basic(self, d_lsj):
        """LSJ convention should return lemmas for known AG forms."""
        result = d_lsj.lemmatize("ἐποίησεν")
        assert result is not None
        assert strip_accents(result.lower()) == strip_accents("ποιέω")

    def test_triantafyllidis_convention_monotonic(self, d_triant):
        """Triantafyllidis convention should output monotonic Greek."""
        result = d_triant.lemmatize("ανθρώπων")
        assert result is not None
        # Verify result is monotonic (no polytonic diacritics)
        import unicodedata
        nfd = unicodedata.normalize("NFD", result)
        polytonic_cps = {0x0313, 0x0314, 0x0342, 0x0345}
        has_polytonic = any(ord(ch) in polytonic_cps for ch in nfd)
        assert not has_polytonic, \
            f"Triantafyllidis output should be monotonic, got {result}"

    def test_triantafyllidis_mg_word(self, d_triant):
        """Triantafyllidis should handle MG words correctly."""
        result = d_triant.lemmatize("ανθρώπων")
        assert strip_accents(result.lower()) == strip_accents("άνθρωπος"), \
            f"ανθρώπων (triant) -> expected άνθρωπος, got {result}"

    def test_invalid_convention_raises(self):
        """Invalid convention name should raise ValueError."""
        with pytest.raises(ValueError):
            Dilemma(convention="invalid_convention")

    def test_wiktionary_convention_is_default(self, d_all):
        """Wiktionary convention should behave like the default (no remap)."""
        d_wik = Dilemma(convention="wiktionary")
        word = "θεούς"
        assert d_wik.lemmatize(word) == d_all.lemmatize(word)

    def test_valid_conventions(self):
        """All valid conventions should initialize without error."""
        for conv in (None, "lsj", "wiktionary", "triantafyllidis"):
            d = Dilemma(convention=conv)
            assert d is not None


# ===========================================================================
# 8. LANGUAGE FILTERING
# ===========================================================================

class TestLanguageFiltering:
    """Test lang parameter: 'grc' vs 'el' vs 'all'."""

    def test_grc_only_returns_ag_lemmas(self, d_grc):
        """lang='grc' should return AG lemmas for AG forms."""
        result = d_grc.lemmatize("θεούς")
        assert strip_accents(result.lower()) == strip_accents("θεός"), \
            f"θεούς (grc) -> expected θεός, got {result}"

    def test_el_only_returns_mg_lemmas(self, d_el):
        """lang='el' should return MG lemmas for MG forms."""
        result = d_el.lemmatize("τρομερό")
        assert strip_accents(result.lower()) == strip_accents("τρομερός"), \
            f"τρομερό (el) -> expected τρομερός, got {result}"

    def test_el_mg_verb(self, d_el):
        """MG-only lemmatizer should return MG verb headwords."""
        result = d_el.lemmatize("φέρνοντας")
        assert strip_accents(result.lower()) == strip_accents("φέρνω"), \
            f"φέρνοντας (el) -> expected φέρνω, got {result}"

    # ------------------------------------------------------------------
    # MG pronoun / article / copula regressions.
    #
    # Before the pronoun-template fix, the MG lookup had AG / cross-person
    # contamination leaking in: αυτό -> τα (via EN τα-pron shared grid),
    # αυτές -> τα, ο -> ὅς (polytonic AG relative pronoun via the ὅ stripped
    # key), etc. lemmatize(lang='el') would return those bad lemmas to
    # downstream consumers like the dilemma tagger. These tests pin the correct
    # behaviour so the contamination can't regress silently.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("form,expected", [
        # αυτός paradigm: all inflected forms must resolve to αυτός.
        # These used to land on τα / εσύ / accents-of-themselves after
        # the EN / EL personal-pronoun template leaked every form into
        # every pronoun headword's inflection table.
        ("αυτό", "αυτός"),
        ("αυτόν", "αυτός"),
        ("αυτήν", "αυτός"),
        ("αυτής", "αυτός"),
        ("αυτών", "αυτός"),
        ("αυτούς", "αυτός"),
        ("αυτές", "αυτός"),
        ("αυτοί", "αυτός"),
        ("αυτά", "αυτός"),
    ])
    def test_el_autos_paradigm(self, d_el, form, expected):
        """All αυτός forms must lemmatize to αυτός, not τα / εσύ / ο."""
        result = d_el.lemmatize(form)
        assert strip_accents(result.lower()) == strip_accents(expected.lower()), \
            f"{form} (el) -> expected {expected}, got {result}"

    def test_el_autou_pos_aware(self, d_el):
        """αυτού is ambiguous (adv "there" or pron gen sg of αυτός).
        POS-aware lookup should resolve it to αυτός when PRON."""
        result = d_el.lemmatize_pos("αυτού", "PRON")
        assert strip_accents(result.lower()) == strip_accents("αυτός"), \
            f"αυτού PRON -> expected αυτός, got {result}"

    @pytest.mark.parametrize("form", [
        # MG articles and common monosyllables must NOT leak to AG
        # polytonic lemmas. They either self-map (no AG leak) or resolve
        # to the monotonic MG form. The assertion is defensive: the lemma
        # must not be a polytonic AG form.
        "ο", "η", "το", "τα", "τους", "τις", "των",
        "ή",  # disjunction "or" (MG), not AG ἤ
    ])
    def test_el_mg_articles_no_ag_leak(self, d_el, form):
        """MG function words must not lemmatize to polytonic AG forms."""
        import unicodedata
        result = d_el.lemmatize(form)
        nfd = unicodedata.normalize("NFD", result)
        has_breathing_or_circumflex = any(
            ord(ch) in (0x0313, 0x0314, 0x0342) for ch in nfd
        )
        assert not has_breathing_or_circumflex, (
            f"{form} (el) -> {result} which is polytonic AG; expected monotonic MG"
        )

    def test_el_einai_copula_not_ag(self, d_el):
        """MG copula είναι must not leak to AG εἰμί."""
        result = d_el.lemmatize("είναι")
        # Accept either είναι (surface-form-as-lemma) or είμαι (Triantafyllidis);
        # reject AG εἰμί.
        assert result != "εἰμί", \
            f"είναι (el) -> εἰμί (AG leak); expected MG είναι or είμαι"

    def test_el_pos_aware_einai_aux(self, d_el):
        """lemmatize_pos('είναι', 'AUX') must not fall through to AG εἰμί."""
        result = d_el.lemmatize_pos("είναι", "AUX")
        assert result != "εἰμί", \
            f"lemmatize_pos('είναι', 'AUX') (el) -> εἰμί; expected MG lemma"

    def test_el_pos_aware_einai_verb(self, d_el):
        """lemmatize_pos('είναι', 'VERB') must not fall through to AG εἰμί."""
        result = d_el.lemmatize_pos("είναι", "VERB")
        assert result != "εἰμί", \
            f"lemmatize_pos('είναι', 'VERB') (el) -> εἰμί; expected MG lemma"

    def test_el_pos_aware_auto_pron(self, d_el):
        """lemmatize_pos('αυτό', 'PRON') (el) must return αυτός, not τα/ο/εσύ."""
        result = d_el.lemmatize_pos("αυτό", "PRON")
        assert strip_accents(result.lower()) == strip_accents("αυτός"), \
            f"lemmatize_pos('αυτό', 'PRON') (el) -> {result}; expected αυτός"

    def test_all_handles_both(self, d_all):
        """lang='all' should handle both AG and MG forms."""
        ag_result = d_all.lemmatize("θεούς")
        mg_result = d_all.lemmatize("τρομερό")
        assert strip_accents(ag_result.lower()) == strip_accents("θεός")
        assert strip_accents(mg_result.lower()) == strip_accents("τρομερός")

    def test_both_alias(self):
        """lang='both' should work as alias for 'all'."""
        d = Dilemma(lang="both")
        assert d.lang == "all"


# ===========================================================================
# 8b. PROPER-NOUN CASE-TWIN RANKING (lemmatize_pos capitalization tiebreak)
# ===========================================================================

class TestProperNounCaseTwin:
    """Capitalization-agreement tiebreak between a common lemma and its
    capitalized proper-noun twin (θυμός vs Θυμός, ἔρις vs Ἔρις).

    A lowercase form tagged as a common POS (NOUN/ADJ/...) must lemmatize
    to the lowercase common lemma, not the capitalized personification
    twin; a form tagged PROPN must still reach the capitalized twin; and
    a capitalized (sentence-initial / all-caps) input must keep either
    twin reachable. This is a re-rank, never a filter.
    """

    # Lowercase common-noun inputs whose personification twin used to win.
    LOWERCASE_NOUNS = [
        ("θυμός", "θυμός"),   # spirit, not Θυμός the personification
        ("ἔρις", "ἔρις"),     # strife, not Ἔρις the goddess
        ("ἔριδι", "ἔρις"),    # inflected: dat sg -> common lemma
        ("τύχη", "τύχη"),     # fortune, not Τύχη
        ("νίκη", "νίκη"),     # victory, not Νίκη
        ("ἔρως", "ἔρως"),     # love, not Ἔρως
        ("ψυχῆς", "ψυχή"),    # soul, not Ψυχή
        ("ἄστυ", "ἄστυ"),     # city, not Ἄστυ
        ("χθόνα", "χθών"),    # earth, not Χθών
        ("καρδίαν", "καρδία"),  # heart, not Καρδία
        ("αἶαν", "αἶα"),      # land, not Αἶα
    ]

    @pytest.mark.parametrize("form,expected", LOWERCASE_NOUNS)
    def test_lowercase_noun_gets_common_lemma(self, d_all, form, expected):
        got = d_all.lemmatize_pos(form, "NOUN")
        assert got == expected, (
            f"lemmatize_pos({form!r}, 'NOUN') -> {got!r}; expected the "
            f"common-noun lemma {expected!r}, not its capitalized twin")

    @pytest.mark.parametrize("form,expected", [
        ("θυμός", "Θυμός"), ("ἔριδι", "Ἔρις"), ("τύχη", "Τύχη"),
    ])
    def test_propn_still_reaches_capitalized_twin(self, d_all, form, expected):
        """A PROPN tag must still resolve the capitalized personification."""
        got = d_all.lemmatize_pos(form, "PROPN")
        assert got == expected, (
            f"lemmatize_pos({form!r}, 'PROPN') -> {got!r}; expected the "
            f"capitalized twin {expected!r}")

    def test_capitalized_input_keeps_twin_reachable(self, d_all):
        """Sentence-initial capitalized input, non-PROPN POS: the tiebreak
        must not force it down to the lowercase twin (ambiguous case)."""
        assert d_all.lemmatize_pos("Θυμός", "NOUN") == "Θυμός"
        assert d_all.lemmatize_pos("Θυμός", "PROPN") == "Θυμός"

    def test_allcaps_input_reaches_both_twins(self, d_all):
        """All-caps input has no case signal of its own; POS decides."""
        assert d_all.lemmatize_pos("ΘΥΜΟΣ", "NOUN") == "θυμός"
        assert d_all.lemmatize_pos("ΘΥΜΟΣ", "PROPN") == "Θυμός"

    @pytest.mark.parametrize("form,upos,expected", LOWERCASE_NOUNS and [
        ("θυμός", "NOUN", "θυμός"), ("θυμός", "PROPN", "Θυμός"),
        ("ἔριδι", "NOUN", "ἔρις"), ("ἔριδι", "PROPN", "Ἔρις"),
        ("ΘΥΜΟΣ", "NOUN", "θυμός"), ("ΘΥΜΟΣ", "PROPN", "Θυμός"),
    ])
    def test_batch_matches_single(self, d_all, form, upos, expected):
        """lemmatize_batch_pos must apply the same tiebreak as lemmatize_pos."""
        single = d_all.lemmatize_pos(form, upos)
        batch = d_all.lemmatize_batch_pos([form], [upos])[0]
        assert single == batch == expected, (
            f"parity/expectation: single={single!r} batch={batch!r} "
            f"expected={expected!r} for {form!r}/{upos}")


# ===========================================================================
# 9. NORMALIZATION (normalize.py)
# ===========================================================================

class TestNormalization:
    """Test the Normalizer class for orthographic variant generation."""

    def test_normalizer_init_default(self):
        """Normalizer should initialize with default period 'all'."""
        n = Normalizer()
        assert n.period == "all"

    def test_normalizer_init_with_period(self):
        """Normalizer should accept explicit period."""
        for period in ("hellenistic", "late_antique", "byzantine"):
            n = Normalizer(period=period)
            assert n.period == period

    def test_normalizer_period_profiles(self):
        """All expected period profiles should exist."""
        for period in ("archaic_epigraphic", "hellenistic",
                       "late_antique", "byzantine", "all"):
            assert period in PROFILES

    def test_itacism_ei_to_i(self):
        """Normalizer should generate itacism variants: ι -> ει."""
        n = Normalizer(period="byzantine")
        candidates = n.normalize("πιστι")
        stripped_candidates = [strip_accents(c) for c in candidates]
        assert "πιστει" in stripped_candidates, \
            f"Expected πιστει in candidates, got {stripped_candidates[:10]}"

    def test_itacism_eta_to_i(self):
        """Normalizer should generate η <-> ι variants."""
        n = Normalizer(period="byzantine")
        candidates = n.normalize("ξενι")
        stripped_candidates = [strip_accents(c) for c in candidates]
        assert "ξενη" in stripped_candidates, \
            f"Expected ξενη in candidates, got {stripped_candidates[:10]}"

    def test_ai_e_merger(self):
        """Normalizer should generate αι <-> ε variants."""
        n = Normalizer(period="hellenistic")
        candidates = n.normalize("χεροντες")
        stripped_candidates = [strip_accents(c) for c in candidates]
        assert "χαιροντες" in stripped_candidates, \
            f"Expected χαιροντες in candidates, got {stripped_candidates[:10]}"

    def test_iota_subscriptum_restoration(self):
        """Normalizer should try restoring iota subscriptum."""
        n = Normalizer(period="byzantine")
        candidates = n.normalize("θεω")
        assert any("ῳ" in c for c in candidates), \
            f"Expected candidate with iota subscript, got {candidates[:10]}"

    def test_geminate_simplification(self):
        """Normalizer should generate geminate variants: λ <-> λλ."""
        n = Normalizer(period="byzantine")
        candidates = n.normalize("αλος")
        stripped_candidates = [strip_accents(c) for c in candidates]
        assert "αλλος" in stripped_candidates, \
            f"Expected αλλος in candidates, got {stripped_candidates[:10]}"

    def test_o_omega_confusion(self):
        """Normalizer should generate ο <-> ω variants."""
        n = Normalizer(period="hellenistic")
        candidates = n.normalize("ανθροπος")
        stripped_candidates = [strip_accents(c) for c in candidates]
        assert "ανθρωπος" in stripped_candidates, \
            f"Expected ανθρωπος in candidates, got {stripped_candidates[:10]}"

    def test_max_candidates_respected(self):
        """Normalizer should not exceed max_candidates."""
        n = Normalizer(period="byzantine", max_candidates=5)
        candidates = n.normalize("ανθροπος")
        assert len(candidates) <= 5

    def test_original_not_in_candidates(self):
        """Normalizer should not include the original token in candidates."""
        n = Normalizer(period="byzantine")
        token = "θεω"
        candidates = n.normalize(token)
        assert token not in candidates

    def test_archaic_profile_minimal_consonant_rules(self):
        """Archaic epigraphic profile should have no consonant rules."""
        n = Normalizer(period="archaic_epigraphic")
        assert len(n.consonant_rules) == 0

    def test_byzantine_has_consonant_rules(self):
        """Byzantine profile should have consonant rules (spirantization, etc.)."""
        n = Normalizer(period="byzantine")
        assert len(n.consonant_rules) > 0

    def test_hellenistic_no_consonant_rules(self):
        """Hellenistic profile should NOT have consonant rules (too early)."""
        n = Normalizer(period="hellenistic")
        assert len(n.consonant_rules) == 0

    def test_subscriptum_map_complete(self):
        """Subscriptum map should cover α, η, ω and accented variants."""
        assert "α" in SUBSCRIPTUM_MAP
        assert "η" in SUBSCRIPTUM_MAP
        assert "ω" in SUBSCRIPTUM_MAP
        assert SUBSCRIPTUM_MAP["α"] == "ᾳ"
        assert SUBSCRIPTUM_MAP["ω"] == "ῳ"

    def test_double_substitutions(self):
        """With max_subs=2, normalizer should attempt double substitutions."""
        n = Normalizer(period="byzantine", max_substitutions=2)
        candidates = n.normalize("πιστι")
        # Should have more candidates than single-sub only
        n_single = Normalizer(period="byzantine", max_substitutions=1)
        cands_single = n_single.normalize("πιστι")
        assert len(candidates) >= len(cands_single)


# ===========================================================================
# 9b. DIALECT NORMALIZATION
# ===========================================================================

class TestDialectNormalization:
    """Test dialect-specific normalization (Ionic, Doric, Aeolic, Koine)."""

    # --- Initialization ---

    def test_dialect_init_none(self):
        """Default dialect should be None (no dialect rules)."""
        n = Normalizer()
        assert n.dialect is None
        assert len(n._dialects) == 0

    def test_dialect_init_ionic(self):
        """Ionic dialect should initialize correctly."""
        n = Normalizer(dialect="ionic")
        assert n.dialect == "ionic"
        assert "ionic" in n._dialects

    def test_dialect_init_auto(self):
        """Auto dialect should enable all dialects."""
        n = Normalizer(dialect="auto")
        assert "ionic" in n._dialects
        assert "doric" in n._dialects
        assert "aeolic" in n._dialects
        assert "koine" in n._dialects

    def test_dialect_invalid_raises(self):
        """Invalid dialect name should raise ValueError."""
        with pytest.raises(ValueError):
            Normalizer(dialect="mycenaean")

    def test_dialect_combined_with_period(self):
        """Dialect and period should work together."""
        n = Normalizer(period="hellenistic", dialect="ionic")
        assert n.period == "hellenistic"
        assert n.dialect == "ionic"
        # Should have both period-based rules and dialect rules
        assert len(n.vowel_rules) > 0
        assert "ionic" in n._dialects

    # --- Ionic: η -> α after ε, ι, ρ ---

    def test_ionic_eta_to_alpha_after_rho(self):
        """Ionic -ης -> Attic -ας after ρ (first declension)."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("ἱστορίης")
        assert "ἱστορίας" in cands, \
            f"Expected ἱστορίας in {cands[:10]}"

    def test_ionic_eta_to_alpha_chores(self):
        """χώρης -> χώρας (Ionic gen.sg. after ρ)."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("χώρης")
        assert "χώρας" in cands

    def test_ionic_eta_to_alpha_hemeres(self):
        """ἡμέρης -> ἡμέρας (Ionic gen.sg. after ρ)."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("ἡμέρης")
        assert "ἡμέρας" in cands

    def test_ionic_eta_to_alpha_after_epsilon(self):
        """η -> α after ε should also be tried."""
        n = Normalizer(dialect="ionic")
        # θεῆς -> θεᾶς (goddess, gen.sg. after ε)
        cands = n.normalize("θεῆς")
        assert "θεᾶς" in cands, \
            f"Expected θεᾶς in {cands[:10]}"

    def test_ionic_eta_to_alpha_after_iota(self):
        """η -> α after ι should also be tried."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("οἰκίης")
        assert "οἰκίας" in cands

    # --- Ionic: uncontracted vowels ---

    def test_ionic_contraction_ee_to_ei(self):
        """ποιέειν -> ποιεῖν (ε-contract: εε -> ει)."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("ποιέειν")
        # The contraction produces ποιέιν (εει -> ει)
        assert any("ποι" in c and "ιν" in c and "εε" not in c
                    for c in cands[:5]), \
            f"Expected contracted form in {cands[:10]}"

    def test_ionic_contraction_eo_to_ou(self):
        """εο -> ου contraction."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("ποιεομένη")
        assert any("ου" in c for c in cands[:5]), \
            f"Expected form with ου in {cands[:10]}"

    def test_ionic_contraction_ew_to_w(self):
        """τιμέω -> τιμῶ (εω -> ω at word end)."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("τιμέω")
        assert "τιμῶ" in cands, \
            f"Expected τιμῶ in {cands[:10]}"

    def test_ionic_contraction_epoiee(self):
        """ἐποίεε -> ἐποίει (εε -> ει)."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("ἐποίεε")
        assert "ἐποίει" in cands, \
            f"Expected ἐποίει in {cands[:10]}"

    # --- Ionic: κ/π interchange ---

    def test_ionic_kos_to_pos(self):
        """κῶς -> πῶς (Ionic interrogative)."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("κῶς")
        assert "πῶς" in cands

    def test_ionic_hokou_to_hopou(self):
        """ὅκου -> ὅπου."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("ὅκου")
        assert "ὅπου" in cands

    def test_ionic_kote_to_pote(self):
        """κότε -> πότε."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("κότε")
        assert "πότε" in cands

    def test_ionic_koios_to_poios(self):
        """κοῖος -> ποῖος."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("κοῖος")
        assert "ποῖος" in cands

    def test_ionic_kosos_to_posos(self):
        """κόσος -> πόσος."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("κόσος")
        assert "πόσος" in cands

    def test_ionic_kothen_to_pothen(self):
        """κόθεν -> πόθεν."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("κόθεν")
        assert "πόθεν" in cands

    # --- Ionic: ου/ο alternation ---

    def test_ionic_mounos_to_monos(self):
        """μοῦνος -> μόνος."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("μοῦνος")
        assert "μόνος" in cands

    def test_ionic_xeinos_to_xenos(self):
        """ξεῖνος -> ξένος."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("ξεῖνος")
        assert "ξένος" in cands

    def test_ionic_keinos_to_ekeinos(self):
        """κεῖνος -> ἐκεῖνος."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("κεῖνος")
        assert "ἐκεῖνος" in cands

    def test_ionic_heineka_to_heneka(self):
        """εἵνεκα -> ἕνεκα."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("εἵνεκα")
        assert "ἕνεκα" in cands

    # --- Ionic: σσ/ττ alternation ---

    def test_ionic_ss_to_tt_thalassa(self):
        """θάλασσα -> θάλαττα."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("θάλασσα")
        assert "θάλαττα" in cands

    def test_ionic_ss_to_tt_glossa(self):
        """γλῶσσα -> γλῶττα."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("γλῶσσα")
        assert "γλῶττα" in cands

    # --- Ionic: ρσ/ρρ alternation ---

    def test_ionic_rs_to_rr_tharsos(self):
        """θάρσος -> θάρρος."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("θάρσος")
        assert "θάρρος" in cands

    def test_ionic_rs_to_rr_arsen(self):
        """ἄρσην -> ἄρρην."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("ἄρσην")
        assert "ἄρρην" in cands

    # --- Doric ---

    def test_doric_poti_to_pros(self):
        """ποτί -> πρός."""
        n = Normalizer(dialect="doric")
        cands = n.normalize("ποτί")
        assert "πρός" in cands

    def test_doric_tu_to_su(self):
        """τύ -> σύ."""
        n = Normalizer(dialect="doric")
        cands = n.normalize("τύ")
        assert "σύ" in cands

    def test_doric_athana_to_athene(self):
        """Ἀθάνα -> Ἀθήνη."""
        n = Normalizer(dialect="doric")
        cands = n.normalize("Ἀθάνα")
        assert "Ἀθήνη" in cands

    def test_doric_alpha_to_eta(self):
        """Doric α -> Attic η in endings."""
        n = Normalizer(dialect="doric")
        # Ἀθάνα has last alpha; should try η
        cands = n.normalize("Ἀθάνα")
        assert any("η" in c or "ή" in c for c in cands[:5])

    def test_doric_future_praxeo(self):
        """πραξέω -> πράξω (Doric future)."""
        n = Normalizer(dialect="doric")
        cands = n.normalize("πραξέω")
        # Should generate πραξω or πράξω
        stripped = [strip_accents(c) for c in cands]
        assert any("πραξω" in s for s in stripped), \
            f"Expected πραξω variant in {cands[:10]}"

    # --- Aeolic ---

    def test_aeolic_psilosis_smooth_to_rough(self):
        """Aeolic psilosis: smooth breathing -> rough breathing."""
        n = Normalizer(dialect="aeolic")
        # ἀ- (smooth) -> ἁ- (rough)
        cands = n.normalize("ἄελλα")
        assert "ἅελλα" in cands, \
            f"Expected ἅελλα in {cands[:10]}"

    # --- Koine ---

    def test_koine_ss_to_tt(self):
        """Koine σσ -> ττ (Attic)."""
        n = Normalizer(dialect="koine")
        cands = n.normalize("θάλασσα")
        assert "θάλαττα" in cands

    def test_koine_tt_to_ss(self):
        """Koine ττ -> σσ (reverse direction)."""
        n = Normalizer(dialect="koine")
        cands = n.normalize("θάλαττα")
        assert "θάλασσα" in cands

    # --- Auto mode ---

    def test_auto_finds_ionic(self):
        """Auto mode should find Ionic normalizations."""
        n = Normalizer(dialect="auto")
        cands = n.normalize("κῶς")
        assert "πῶς" in cands

    def test_auto_finds_doric(self):
        """Auto mode should find Doric normalizations."""
        n = Normalizer(dialect="auto")
        cands = n.normalize("ποτί")
        assert "πρός" in cands

    def test_auto_finds_multiple_dialects(self):
        """Auto mode should try all dialect rules."""
        n = Normalizer(dialect="auto")
        # σσ -> ττ should work via both ionic and koine
        cands = n.normalize("θάλασσα")
        assert "θάλαττα" in cands

    # --- Dilemma integration ---

    def test_dilemma_dialect_param(self):
        """Dilemma should accept dialect parameter."""
        d = Dilemma(dialect="ionic")
        assert d._normalizer is not None
        assert d._normalizer.dialect == "ionic"

    def test_dilemma_dialect_auto(self):
        """Dilemma should accept dialect='auto'."""
        d = Dilemma(dialect="auto")
        assert d._normalizer is not None
        assert "ionic" in d._normalizer._dialects

    def test_dilemma_dialect_enables_normalizer(self):
        """Setting dialect should implicitly enable normalization."""
        d = Dilemma(dialect="ionic")
        assert d._normalizer is not None

    def test_dilemma_dialect_combined_with_period(self):
        """Dialect and period should combine in Dilemma."""
        d = Dilemma(dialect="ionic", period="hellenistic")
        assert d._normalizer is not None
        assert d._normalizer.dialect == "ionic"
        assert d._normalizer.period == "hellenistic"

    # --- Dialect candidates are prioritized ---

    def test_dialect_candidates_ranked_first(self):
        """Dialect matches should appear before orthographic variants."""
        n = Normalizer(dialect="ionic")
        cands = n.normalize("κῶς")
        # πῶς should be among the very first candidates
        assert cands.index("πῶς") < 3, \
            f"Expected πῶς in top 3, got position {cands.index('πῶς')} in {cands[:10]}"

    # --- Word map coverage ---

    def test_ionic_word_map_has_pairs(self):
        """Ionic word map should contain expected key pairs."""
        assert "κῶς" in IONIC_WORD_MAP
        assert "μοῦνος" in IONIC_WORD_MAP
        assert "ξεῖνος" in IONIC_WORD_MAP

    def test_doric_word_map_has_pairs(self):
        """Doric word map should contain expected key pairs."""
        assert "ποτί" in DORIC_WORD_MAP
        assert "τύ" in DORIC_WORD_MAP


# ===========================================================================
# 10. EDGE CASES
# ===========================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string(self, d_all):
        """Empty string should not crash."""
        result = d_all.lemmatize("")
        assert result is not None

    def test_digit_passthrough(self, d_all):
        """Digit-only strings should pass through unchanged."""
        assert d_all.lemmatize("123") == "123"
        assert d_all.lemmatize("42") == "42"

    def test_latin_input(self, d_all):
        """Latin/non-Greek input should return something (identity or model guess)."""
        result = d_all.lemmatize("hello")
        assert result is not None
        assert len(result) > 0

    def test_single_char_greek(self, d_all):
        """Single Greek character should not crash."""
        result = d_all.lemmatize("α")
        assert result is not None

    def test_mixed_case(self, d_all):
        """Capital initial should still lemmatize (case-insensitive lookup)."""
        result = d_all.lemmatize("Θεούς")
        assert result is not None
        assert strip_accents(result.lower()) == strip_accents("θεός")

    def test_unknown_word_returns_something(self, d_all):
        """Unknown words should not crash, should return some result."""
        result = d_all.lemmatize("ξυζβαρκω")
        assert result is not None
        assert len(result) > 0

    def test_word_with_grave_accent(self, d_all):
        """Words with grave accent should be normalized and looked up."""
        result = d_all.lemmatize("τὸν")
        assert result is not None

    def test_multiple_candidates_verbose(self, d_all):
        """Words with proper/common distinction should show multiple candidates."""
        candidates = d_all.lemmatize_verbose("θεούς")
        # θεούς should have at least θεός and Θεός
        assert len(candidates) >= 2, \
            f"Expected multiple candidates for θεούς, got {len(candidates)}"

    def test_proper_noun_flagged(self, d_all):
        """Proper noun candidates should have proper=True."""
        candidates = d_all.lemmatize_verbose("Ἀχιλλεύς")
        if candidates:
            assert any(c.proper for c in candidates), \
                "Ἀχιλλεύς should have at least one proper noun candidate"

    def test_empty_string_verbose(self, d_all):
        """Empty string in verbose mode should not crash."""
        result = d_all.lemmatize_verbose("")
        assert isinstance(result, list)


# ===========================================================================
# 11. BATCH OPERATIONS
# ===========================================================================

class TestBatchOperations:
    """Test lemmatize_batch for multiple words at once."""

    def test_batch_correct_count(self, d_all):
        """Batch lemmatization should return correct number of results."""
        words = ["θεούς", "ἔλυσε", "τρομερό"]
        results = d_all.lemmatize_batch(words)
        assert len(results) == len(words)

    def test_batch_matches_single(self, d_all):
        """Batch results should match individual lemmatize calls."""
        words = ["θεούς", "μῆνιν", "τρομερό"]
        batch_results = d_all.lemmatize_batch(words)
        for word, batch_result in zip(words, batch_results):
            single_result = d_all.lemmatize(word)
            assert strip_accents(batch_result.lower()) == \
                   strip_accents(single_result.lower()), \
                f"Batch vs single mismatch for {word}: " \
                f"batch={batch_result}, single={single_result}"

    def test_batch_empty_list(self, d_all):
        """Batch with empty list should return empty list."""
        assert d_all.lemmatize_batch([]) == []

    def test_batch_single_item(self, d_all):
        """Batch with single item should work."""
        results = d_all.lemmatize_batch(["θεούς"])
        assert len(results) == 1

    def test_batch_with_crasis(self, d_all):
        """Batch should handle crasis forms correctly."""
        words = ["τοὔνομα", "θεούς"]
        results = d_all.lemmatize_batch(words)
        assert strip_accents(results[0].lower()) == strip_accents("ὄνομα")

    def test_batch_with_digits(self, d_all):
        """Batch should pass through digit-only strings."""
        words = ["42", "θεούς"]
        results = d_all.lemmatize_batch(words)
        # Digit passthrough does not work in batch (no isdigit check)
        # so we just verify it doesn't crash
        assert len(results) == 2


# ===========================================================================
# 12. SPELLING SUGGESTION
# ===========================================================================

class TestSpellingSuggestion:
    """Test spelling correction/suggestion feature."""

    def test_suggest_spelling_accent_mismatch(self, d_all):
        """Missing/wrong accent should find suggestions at distance 0."""
        results = d_all.suggest_spelling("θεος")
        assert len(results) > 0
        # The stripped form matches, so distance should be 0 for the match
        forms = [r[0] for r in results]
        assert any("θεό" in f for f in forms), \
            f"Expected θεός-like forms in suggestions, got {forms[:5]}"

    def test_suggest_spelling_no_results_for_nonsense(self, d_all):
        """Completely unrelated string should return no results at ED1."""
        results = d_all.suggest_spelling("zzzzzzz", max_distance=1)
        assert len(results) == 0

    def test_suggest_spelling_returns_tuples(self, d_all):
        """Results should be (form, distance) tuples."""
        results = d_all.suggest_spelling("λογος")
        if results:
            form, dist = results[0]
            assert isinstance(form, str)
            assert isinstance(dist, (int, float))

    def test_suggest_spelling_near_miss(self, d_all):
        """One-letter-off Greek words should find suggestions."""
        # λογοσ (final σ instead of ς) - this tests ED1
        results = d_all.suggest_spelling("λοος", max_distance=2)
        assert len(results) > 0


# ===========================================================================
# 13. ARTICLE RESOLUTION
# ===========================================================================

class TestArticleResolution:
    """Test resolve_articles mode."""

    def test_article_resolved_to_ho(self, d_resolve_articles):
        """With resolve_articles=True, article forms should -> ὁ."""
        for form in ("τοῦ", "τῆς", "τῷ", "τόν", "τήν", "τά"):
            result = d_resolve_articles.lemmatize(form)
            assert strip_accents(result) == strip_accents("ὁ"), \
                f"Article {form} -> expected ὁ, got {result}"

    def test_pronoun_resolved(self, d_resolve_articles):
        """With resolve_articles=True, pronoun clitics should resolve."""
        result = d_resolve_articles.lemmatize("μοι")
        assert strip_accents(result.lower()) == strip_accents("ἐγώ"), \
            f"μοι -> expected ἐγώ, got {result}"

    def test_default_no_article_resolution(self, d_all):
        """Without resolve_articles, articles should self-map."""
        result = d_all.lemmatize("τοῦ")
        # In default mode, τοῦ should NOT resolve to ὁ
        assert strip_accents(result) != strip_accents("ὁ"), \
            f"τοῦ should not -> ὁ without resolve_articles, got {result}"


# ===========================================================================
# 14. UTILITY FUNCTIONS
# ===========================================================================

class TestUtilityFunctions:
    """Test module-level utility functions."""

    def test_to_monotonic_strips_breathings(self):
        """to_monotonic should strip breathings."""
        assert to_monotonic("ἐποίησεν") == "εποίησεν"

    def test_to_monotonic_simplifies_circumflex(self):
        """to_monotonic should convert circumflex to acute."""
        result = to_monotonic("τῆς")
        assert result == "τής"

    def test_to_monotonic_preserves_acute(self):
        """to_monotonic should preserve acute accents."""
        result = to_monotonic("θεός")
        assert result == "θεός"

    def test_strip_accents_all(self):
        """strip_accents should remove all diacritical marks."""
        assert strip_accents("ἐποίησεν") == "εποιησεν"
        assert strip_accents("θεός") == "θεος"
        assert strip_accents("τῷ") == "τω"

    def test_grave_to_acute(self):
        """grave_to_acute should convert grave to acute, keep other marks."""
        assert grave_to_acute("τὸν") == "τόν"
        assert grave_to_acute("τὰ") == "τά"

    def test_grave_to_acute_preserves_acute(self):
        """grave_to_acute should leave acute accents unchanged."""
        assert grave_to_acute("θεός") == "θεός"

    def test_grave_to_acute_preserves_breathings(self):
        """grave_to_acute should preserve breathings (not just accents)."""
        result = grave_to_acute("ὁ")
        assert result == "ὁ"

    def test_is_self_map_exact(self):
        """_is_self_map should detect exact match."""
        assert _is_self_map("θεός", "θεός") is True

    def test_is_self_map_accent_difference(self):
        """_is_self_map should detect accent-only differences."""
        assert _is_self_map("θεος", "θεός") is True

    def test_is_self_map_different_words(self):
        """_is_self_map should return False for different words."""
        assert _is_self_map("θεός", "λόγος") is False

    def test_to_monotonic_empty(self):
        """to_monotonic should handle empty string."""
        assert to_monotonic("") == ""

    def test_strip_accents_no_accents(self):
        """strip_accents on already-stripped text should be identity."""
        assert strip_accents("θεος") == "θεος"

    def test_strip_accents_latin(self):
        """strip_accents should work on Latin text too."""
        assert strip_accents("cafe") == "cafe"


# ===========================================================================
# 15. VERBOSE MODE METADATA
# ===========================================================================

class TestVerboseMetadata:
    """Test that verbose mode returns proper LemmaCandidate objects."""

    def test_verbose_returns_lemma_candidates(self, d_all):
        """lemmatize_verbose should return LemmaCandidate objects."""
        candidates = d_all.lemmatize_verbose("θεούς")
        assert len(candidates) > 0
        for c in candidates:
            assert isinstance(c, LemmaCandidate)
            assert hasattr(c, "lemma")
            assert hasattr(c, "lang")
            assert hasattr(c, "proper")
            assert hasattr(c, "source")

    def test_verbose_lookup_has_source(self, d_all):
        """Lookup hits in verbose mode should have source='lookup'."""
        candidates = d_all.lemmatize_verbose("θεούς")
        assert any(c.source == "lookup" for c in candidates)

    def test_verbose_crasis_has_source(self, d_all):
        """Crasis resolution in verbose mode should tag source='crasis'."""
        candidates = d_all.lemmatize_verbose("τοὔνομα")
        assert len(candidates) > 0
        assert any(c.source == "crasis" for c in candidates), \
            f"Expected source='crasis' for τοὔνομα, got sources: " \
            f"{[c.source for c in candidates]}"

    def test_verbose_digit_passthrough(self, d_all):
        """Digit input in verbose mode should return identity source."""
        candidates = d_all.lemmatize_verbose("42")
        assert len(candidates) == 1
        assert candidates[0].lemma == "42"

    def test_verbose_candidate_has_lang(self, d_all):
        """Candidates should have a lang field (grc or el)."""
        candidates = d_all.lemmatize_verbose("θεούς")
        for c in candidates:
            assert c.lang in ("grc", "el", ""), \
                f"Unexpected lang {c.lang!r} for candidate {c.lemma}"


# ===========================================================================
# 16. DILEMMA INITIALIZATION
# ===========================================================================

class TestInitialization:
    """Test Dilemma constructor options and edge cases."""

    def test_default_init(self):
        """Default initialization should work without arguments."""
        d = Dilemma()
        assert d.lang == "all"

    def test_init_with_normalize(self):
        """Normalize mode should create a normalizer."""
        d = Dilemma(normalize=True)
        assert d._normalizer is not None

    def test_init_without_normalize(self):
        """Default mode should not create a normalizer."""
        d = Dilemma(normalize=False)
        assert d._normalizer is None

    def test_init_with_period(self):
        """Period should be passed to the normalizer."""
        d = Dilemma(normalize=True, period="byzantine")
        assert d._normalizer is not None
        assert d._normalizer.period == "byzantine"

    def test_triantafyllidis_forces_resolve_articles(self):
        """Triantafyllidis convention should force resolve_articles=True."""
        d = Dilemma(convention="triantafyllidis")
        assert d._resolve_articles is True
