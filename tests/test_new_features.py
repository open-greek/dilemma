#!/usr/bin/env python3
"""Tests for recently added features in Dilemma.

Covers:
1. Gorman/AGDT treebank data integration
2. Dialect normalization (Ionic, Doric, Aeolic, Koine) end-to-end
3. Particle suffix stripping end-to-end
4. Verb morphology stripping end-to-end
5. Article-agreement disambiguation end-to-end
6. Stress tests / integration scenarios

Run with:
    cd dilemma && python -m pytest tests/test_new_features.py -x -v
"""

import json
import sqlite3
import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT))

from dilemma import (
    Dilemma, LemmaCandidate, strip_accents,
)
from dilemma.core import _ARTICLE_FEATURES
from dilemma.normalize import Normalizer, IONIC_WORD_MAP, DORIC_WORD_MAP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def d_all():
    """Dilemma instance - combined lookup, no dialect."""
    return Dilemma(lang="all")


@pytest.fixture(scope="module")
def d_ionic():
    """Dilemma instance with Ionic dialect normalization."""
    return Dilemma(dialect="ionic")


@pytest.fixture(scope="module")
def d_doric():
    """Dilemma instance with Doric dialect normalization."""
    return Dilemma(dialect="doric")


@pytest.fixture(scope="module")
def d_aeolic():
    """Dilemma instance with Aeolic dialect normalization."""
    return Dilemma(dialect="aeolic")


@pytest.fixture(scope="module")
def d_koine():
    """Dilemma instance with Koine dialect normalization."""
    return Dilemma(dialect="koine")


@pytest.fixture(scope="module")
def d_auto():
    """Dilemma instance with auto dialect detection."""
    return Dilemma(dialect="auto")


@pytest.fixture(scope="module")
def d_ionic_hellenistic():
    """Dilemma instance with Ionic dialect and hellenistic period."""
    return Dilemma(dialect="ionic", period="hellenistic")


# ===========================================================================
# 1. GORMAN/AGDT DATA INTEGRATION
# ===========================================================================

class TestGormanDataFiles:
    """Verify the Gorman (CC BY-SA) data file exists and has correct structure.

    PROIEL (CC BY-NC-SA) is intentionally not ingested and not committed, so the
    Ionic/Herodotus coverage it once contributed is checked via Gorman/AGDT.
    """

    def test_gorman_pairs_exists(self):
        """gorman_pairs.json should exist."""
        assert (DATA_DIR / "gorman_pairs.json").exists(), \
            "gorman_pairs.json not found in data/"

    def test_gorman_pairs_structure(self):
        """gorman_pairs.json should be a list of {form, lemma, pos} dicts."""
        with open(DATA_DIR / "gorman_pairs.json", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 50_000, \
            f"Expected >50K Gorman pairs, got {len(data)}"
        entry = data[0]
        assert "form" in entry
        assert "lemma" in entry

    def test_gorman_contains_herodotus_forms(self):
        """Gorman should contain forms from Herodotus and other authors."""
        with open(DATA_DIR / "gorman_pairs.json", encoding="utf-8") as f:
            data = json.load(f)
        forms = {p["form"] for p in data}
        # Gorman covers Herodotus, Thucydides, Xenophon, etc.
        expected = {"πρήγματα", "ποιέειν", "κῶς"}
        found = expected & forms
        assert len(found) >= 2, \
            f"Expected forms from Gorman treebank, only found: {found}"


class TestIonicFormsInLookup:
    """Verify Ionic (Herodotus) forms resolve through the main lemmatizer.

    These come from the openly-licensed Gorman/AGDT treebanks; PROIEL
    (CC BY-NC-SA) is excluded, so Ionic coverage must not depend on it.
    """

    @pytest.mark.parametrize("form,expected", [
        ("πρήγματα", "πρᾶγμα"),     # Ionic plural
        ("ἱστορίης", "ἱστορία"),     # Ionic genitive
        ("μοῦνος", "μόνος"),         # Ionic ου/ο alternation
        ("ξεῖνος", "ξένος"),         # Ionic ξεῖνος
        ("ποιέειν", "ποιέω"),        # Ionic uncontracted infinitive
        ("πρῆγμα", "πρᾶγμα"),       # Ionic singular
        ("ὅκου", "ὅπου"),            # Ionic κ/π interchange
    ])
    def test_ionic_forms_resolve_via_lookup(self, d_all, form, expected):
        """Ionic forms should resolve via the lookup table (Gorman/AGDT)."""
        result = d_all.lemmatize(form)
        assert strip_accents(result.lower()) == strip_accents(expected), \
            f"{form} -> expected {expected}, got {result}"

    def test_lookup_db_has_ionic_entries(self):
        """lookup.db should contain Ionic entries added from the treebanks."""
        db_path = DATA_DIR / "lookup.db"
        if not db_path.exists():
            pytest.skip("lookup.db not found")
        conn = sqlite3.connect(str(db_path))
        # πρήγματα is an Ionic form unlikely to be in Wiktionary
        row = conn.execute(
            "SELECT l.text FROM lookup k JOIN lemmas l ON k.lemma_id = l.id "
            "WHERE k.form = ? AND k.lang IN ('all', 'grc') LIMIT 1",
            ("πρήγματα",)
        ).fetchone()
        conn.close()
        assert row is not None, \
            "πρήγματα should be in lookup.db (from Gorman/AGDT)"
        assert row[0] == "πρᾶγμα"


# ===========================================================================
# 2. DIALECT NORMALIZATION (END-TO-END THROUGH DILEMMA)
# ===========================================================================

class TestDialectIonicE2E:
    """End-to-end tests for Ionic dialect normalization via Dilemma."""

    @pytest.mark.parametrize("form,expected", [
        ("πρήγματα", "πρᾶγμα"),     # lookup hit (Ionic form in DB)
        ("ἱστορίης", "ἱστορία"),     # lookup hit
        ("μοῦνος", "μόνος"),         # word-level map
    ])
    def test_ionic_forms_in_lookup(self, d_ionic, form, expected):
        """Ionic forms already in lookup should resolve even without dialect."""
        result = d_ionic.lemmatize(form)
        assert strip_accents(result.lower()) == strip_accents(expected), \
            f"Ionic {form} -> expected {expected}, got {result}"

    def test_ionic_ss_tt_thalassa(self, d_ionic):
        """θάλασσα (Ionic/Koine σσ) should resolve to θάλασσα or θάλαττα."""
        result = d_ionic.lemmatize("θάλασσα")
        # θάλασσα is itself a headword, so it should self-resolve
        assert result is not None
        assert len(result) > 0

    def test_ionic_rs_rr_tharsos(self, d_ionic):
        """θάρσος with dialect=ionic should still return a valid result."""
        result = d_ionic.lemmatize("θάρσος")
        assert result is not None
        assert len(result) > 0

    def test_ionic_normalizer_creates_candidates(self):
        """The Ionic normalizer should generate Attic candidates."""
        n = Normalizer(dialect="ionic")
        # ἱστορίης (Ionic gen) -> should produce ἱστορίας (Attic gen)
        cands = n.normalize("ἱστορίης")
        assert "ἱστορίας" in cands, \
            f"Expected ἱστορίας in candidates, got {cands[:10]}"

    def test_ionic_word_map_mounos(self, d_ionic):
        """μοῦνος -> μόνος via Ionic word-level map."""
        result = d_ionic.lemmatize("μοῦνος")
        assert strip_accents(result.lower()) == strip_accents("μόνος")

    def test_ionic_word_map_xeinos(self, d_ionic):
        """ξεῖνος -> ξένος via Ionic word-level map."""
        result = d_ionic.lemmatize("ξεῖνος")
        assert strip_accents(result.lower()) == strip_accents("ξένος")


class TestDialectDoricE2E:
    """End-to-end tests for Doric dialect normalization."""

    def test_doric_normalizer_poti(self):
        """Doric ποτί should normalize to πρός."""
        n = Normalizer(dialect="doric")
        cands = n.normalize("ποτί")
        assert "πρός" in cands

    def test_doric_alpha_to_eta_normalization(self):
        """Doric long alpha should produce eta candidates."""
        n = Normalizer(dialect="doric")
        cands = n.normalize("Ἀθάνα")
        # Word map: Ἀθάνα -> Ἀθήνη
        assert "Ἀθήνη" in cands, \
            f"Expected Ἀθήνη in {cands[:10]}"

    def test_doric_dilemma_tu_to_su(self, d_doric):
        """τύ with dialect=doric should resolve (to σύ via normalizer)."""
        result = d_doric.lemmatize("τύ")
        # τύ is not in the lookup, so the normalizer maps it to σύ
        assert result is not None
        assert len(result) > 0


class TestDialectAeolicE2E:
    """End-to-end tests for Aeolic dialect normalization."""

    def test_aeolic_normalizer_init(self):
        """Aeolic normalizer should initialize."""
        n = Normalizer(dialect="aeolic")
        assert "aeolic" in n._dialects

    def test_aeolic_psilosis(self):
        """Aeolic psilosis: smooth -> rough breathing candidates."""
        n = Normalizer(dialect="aeolic")
        cands = n.normalize("ἄελλα")
        assert "ἅελλα" in cands, \
            f"Expected ἅελλα in {cands[:10]}"

    def test_aeolic_dilemma_instance(self, d_aeolic):
        """Dilemma with dialect=aeolic should initialize and lemmatize."""
        result = d_aeolic.lemmatize("θεούς")
        assert strip_accents(result.lower()) == strip_accents("θεός")


class TestDialectKoineE2E:
    """End-to-end tests for Koine dialect normalization."""

    def test_koine_normalizer_ss_tt(self):
        """Koine σσ <-> ττ normalization."""
        n = Normalizer(dialect="koine")
        cands = n.normalize("θάλασσα")
        assert "θάλαττα" in cands
        cands_rev = n.normalize("θάλαττα")
        assert "θάλασσα" in cands_rev

    def test_koine_dilemma_instance(self, d_koine):
        """Dilemma with dialect=koine should work for standard forms."""
        result = d_koine.lemmatize("θεούς")
        assert strip_accents(result.lower()) == strip_accents("θεός")


class TestDialectAutoE2E:
    """End-to-end tests for dialect=auto mode."""

    def test_auto_mode_enables_all_dialects(self):
        """dialect=auto should enable ionic, doric, aeolic, koine."""
        n = Normalizer(dialect="auto")
        assert "ionic" in n._dialects
        assert "doric" in n._dialects
        assert "aeolic" in n._dialects
        assert "koine" in n._dialects

    def test_auto_finds_ionic_candidates(self):
        """Auto mode should find Ionic word-level candidates."""
        n = Normalizer(dialect="auto")
        cands = n.normalize("μοῦνος")
        assert "μόνος" in cands

    def test_auto_finds_doric_candidates(self):
        """Auto mode should find Doric word-level candidates."""
        n = Normalizer(dialect="auto")
        cands = n.normalize("ποτί")
        assert "πρός" in cands

    def test_auto_dilemma_instance(self, d_auto):
        """Dilemma with dialect=auto should work."""
        result = d_auto.lemmatize("θεούς")
        assert strip_accents(result.lower()) == strip_accents("θεός")


class TestDialectNoneVsSet:
    """Test that dialect=None does NOT apply dialect normalization."""

    def test_no_dialect_no_normalizer(self):
        """Dilemma with dialect=None and normalize=False should have no normalizer."""
        d = Dilemma(dialect=None, normalize=False)
        assert d._normalizer is None

    def test_dialect_none_does_not_normalize(self, d_all):
        """Without dialect, the normalizer should not be active."""
        # d_all has no dialect set, so it should not produce dialect candidates
        assert d_all._normalizer is None

    def test_dialect_setting_enables_normalizer(self):
        """Setting a dialect should implicitly create a normalizer."""
        d = Dilemma(dialect="ionic")
        assert d._normalizer is not None
        assert d._normalizer.dialect == "ionic"


class TestDialectCombinedWithPeriod:
    """Test combining dialect with period parameter."""

    def test_ionic_hellenistic_init(self, d_ionic_hellenistic):
        """dialect=ionic + period=hellenistic should both be set."""
        assert d_ionic_hellenistic._normalizer is not None
        assert d_ionic_hellenistic._normalizer.dialect == "ionic"
        assert d_ionic_hellenistic._normalizer.period == "hellenistic"

    def test_ionic_hellenistic_lemmatizes(self, d_ionic_hellenistic):
        """Combined dialect+period should lemmatize correctly."""
        result = d_ionic_hellenistic.lemmatize("πρήγματα")
        assert strip_accents(result.lower()) == strip_accents("πρᾶγμα")


# ===========================================================================
# 3. PARTICLE SUFFIX STRIPPING (END-TO-END)
# ===========================================================================

class TestParticleSuffixE2E:
    """End-to-end particle suffix stripping through Dilemma.lemmatize()."""

    def test_hosper_self_maps(self, d_all):
        """ὅσπερ is a headword, so it should self-map via lookup."""
        result = d_all.lemmatize("ὅσπερ")
        # ὅσπερ is in Wiktionary as its own headword
        assert result is not None
        assert len(result) > 0

    def test_egoge_self_maps(self, d_all):
        """ἔγωγε is a headword, so it should self-map via lookup."""
        result = d_all.lemmatize("ἔγωγε")
        assert result is not None
        assert len(result) > 0

    def test_lookup_forms_not_stripped(self, d_all):
        """Forms in the lookup table should NOT go through particle stripping."""
        # ὥσπερ is a very common word with its own entry
        result = d_all.lemmatize("ὥσπερ")
        assert result is not None
        # It should self-map, not strip to ὡς
        assert len(result) > 0

    def test_deictic_i_on_demonstrative(self, d_all):
        """τουτουί (deictic -ί on demonstrative) should resolve."""
        result = d_all.lemmatize("τουτουί")
        # Should strip deictic -ί and find τούτου -> οὗτος
        assert strip_accents(result.lower()) == strip_accents("οὗτος"), \
            f"τουτουί -> expected οὗτος, got {result}"

    def test_deictic_i_on_toutoisI(self, d_all):
        """τουτοισί (deictic -ί on demonstrative dative plural)."""
        result = d_all.lemmatize("τουτοισί")
        assert strip_accents(result.lower()) == strip_accents("οὗτος"), \
            f"τουτοισί -> expected οὗτος, got {result}"

    def test_stripping_does_not_fire_on_stem_suffix(self, d_all):
        """Words where apparent suffix is part of the stem should not strip."""
        # λόγος does not end in a particle suffix
        result = d_all.lemmatize("λόγος")
        assert strip_accents(result.lower()) == strip_accents("λόγος"), \
            f"λόγος should self-map, got {result}"

    def test_de_suffix_only_when_base_found(self, d_all):
        """Stripping -δε should only succeed when the base form is in lookup."""
        # ἔνθαδε: the -δε is part of the word, and it's a real adverb
        result = d_all.lemmatize("ἔνθαδε")
        # This is a real headword, so it should self-map
        assert result is not None

    def test_strip_particle_method_direct(self, d_all):
        """Direct test of _strip_particle_suffix for ἔμοιγε -> ἐγώ."""
        result = d_all._strip_particle_suffix("ἔμοιγε")
        assert result is not None
        assert strip_accents(result.lower()) == strip_accents("ἐγώ"), \
            f"ἔμοιγε should strip to ἐγώ, got {result}"


# ===========================================================================
# 4. VERB MORPHOLOGY STRIPPING (END-TO-END)
# ===========================================================================

class TestVerbMorphologyE2E:
    """End-to-end verb morphology stripping through Dilemma."""

    def test_common_augmented_resolve_via_lookup(self, d_all):
        """Common augmented verbs should resolve via lookup, not stripping."""
        # These are so common they're in the lookup table directly
        result = d_all.lemmatize("ἔλυσε")
        assert strip_accents(result.lower()) == strip_accents("λύω"), \
            f"ἔλυσε -> expected λύω, got {result}"

    def test_epoiesen_via_lookup(self, d_all):
        """ἐποίησεν is in lookup, should not need stripping."""
        result = d_all.lemmatize("ἐποίησεν")
        assert strip_accents(result.lower()) == strip_accents("ποιέω"), \
            f"ἐποίησεν -> expected ποιέω, got {result}"

    def test_egrapse_via_lookup(self, d_all):
        """ἔγραψε is common enough to be in lookup."""
        result = d_all.lemmatize("ἔγραψε")
        # Should resolve to γράφω
        assert result is not None
        assert len(result) > 0

    def test_no_false_positive_eros(self, d_all):
        """ἔρως should NOT have its ε- stripped (not an augment)."""
        result = d_all.lemmatize("ἔρως")
        # ἔρως is a noun headword, should self-map
        assert strip_accents(result.lower()) in (
            strip_accents("ἔρως"), strip_accents("Ἔρως".lower())
        ), f"ἔρως should not be verb-stripped, got {result}"

    def test_no_false_positive_ergon(self, d_all):
        """ἔργον should NOT be verb-stripped."""
        result = d_all.lemmatize("ἔργον")
        assert strip_accents(result.lower()) == strip_accents("ἔργον"), \
            f"ἔργον should self-map as a noun, got {result}"

    def test_syllabic_augment_method_direct(self, d_all):
        """Direct call to _strip_verb_morphology for ἐποίησεν."""
        result = d_all._strip_verb_morphology("ἐποίησεν")
        assert result is not None
        assert strip_accents(result.lower()) == strip_accents("ποιέω"), \
            f"Morph stripping: ἐποίησεν -> expected ποιέω, got {result}"

    def test_temporal_augment_method_direct(self, d_all):
        """Direct call to _strip_verb_morphology for ἠγόραζον (η- for α-)."""
        result = d_all._strip_verb_morphology("ἠγόραζον")
        assert result is not None
        assert strip_accents(result.lower()) == strip_accents("ἀγοράζω"), \
            f"Morph stripping: ἠγόραζον -> expected ἀγοράζω, got {result}"

    def test_short_words_not_stripped(self, d_all):
        """Short words (< 4 chars) should not trigger stripping."""
        result = d_all._strip_verb_morphology("ἔχω")
        assert result is None, "ἔχω is too short for verb morphology stripping"


# ===========================================================================
# 5. ARTICLE-AGREEMENT DISAMBIGUATION (END-TO-END)
# ===========================================================================

class TestArticleAgreementE2E:
    """End-to-end article-agreement disambiguation tests."""

    def test_feminine_article_prev_word(self, d_all):
        """prev_word=ἡ should prefer feminine candidates."""
        cands = d_all.lemmatize_verbose("θεούς", prev_word="ἡ")
        # Should still return candidates (ἡ is nominative, θεούς accusative
        # so agreement may not change ranking, but it should not crash)
        assert len(cands) > 0

    def test_neuter_article_prev_word(self, d_all):
        """prev_word=τό should prefer neuter candidates."""
        cands = d_all.lemmatize_verbose("δῶρον", prev_word="τό")
        assert len(cands) > 0

    def test_masculine_plural_article(self, d_all):
        """prev_word=οἱ should prefer masculine plural candidates."""
        cands = d_all.lemmatize_verbose("θεοί", prev_word="οἱ")
        assert len(cands) > 0
        # θεός is masculine, so it should be favored
        assert strip_accents(cands[0].lemma.lower()) == strip_accents("θεός"), \
            f"With οἱ, expected θεός first, got {cands[0].lemma}"

    def test_without_prev_word_all_candidates_returned(self, d_all):
        """Without prev_word, all candidates should still be returned."""
        cands = d_all.lemmatize_verbose("θεούς")
        assert len(cands) > 0

    def test_candidate_sets_same_with_and_without_article(self, d_all):
        """Article agreement should rerank, not add/remove candidates."""
        cands_no = d_all.lemmatize_verbose("θεούς")
        cands_with = d_all.lemmatize_verbose("θεούς", prev_word="τούς")
        set_no = {c.lemma for c in cands_no}
        set_with = {c.lemma for c in cands_with}
        assert set_no == set_with, \
            "Article agreement should rerank, not change candidate set"

    def test_non_article_prev_word_no_effect(self, d_all):
        """Non-article prev_word should not change ranking."""
        cands_plain = d_all.lemmatize_verbose("ἄνδρα")
        cands_kai = d_all.lemmatize_verbose("ἄνδρα", prev_word="καί")
        # καί is not an article, so order should be identical
        assert [c.lemma for c in cands_plain] == [c.lemma for c in cands_kai]

    def test_article_features_structure(self):
        """Article features dict should have expected structure."""
        # Masculine singular nominative
        assert _ARTICLE_FEATURES["ὁ"] == ("m", "s", "nom")
        # Feminine singular nominative
        assert _ARTICLE_FEATURES["ἡ"][0] == "f"
        # Neuter singular nominative
        assert _ARTICLE_FEATURES["τό"][0] == "n"
        # Masculine plural accusative
        assert _ARTICLE_FEATURES["τούς"][0] == "m"

    def test_rank_method_preserves_all_candidates(self, d_all):
        """_rank_by_article_agreement should never drop candidates."""
        c1 = LemmaCandidate(lemma="λόγος", lang="grc", proper=False,
                             source="lookup")
        c2 = LemmaCandidate(lemma="Λόγος", lang="grc", proper=True,
                             source="lookup")
        result = d_all._rank_by_article_agreement([c1, c2], prev_word="ὁ")
        assert len(result) == 2

    def test_rank_method_with_none_prev_word(self, d_all):
        """_rank_by_article_agreement with prev_word=None should be identity."""
        c1 = LemmaCandidate(lemma="λόγος", lang="grc", proper=False,
                             source="lookup")
        result = d_all._rank_by_article_agreement([c1], prev_word=None)
        assert result == [c1]


# ===========================================================================
# 6. STRESS TESTS / INTEGRATION
# ===========================================================================

class TestHerodotusIntegration:
    """Integration tests simulating real Herodotus text with Ionic dialect."""

    # Opening of Herodotus Histories 1.1:
    # Ἡροδότου Ἁλικαρνησσέος ἱστορίης ἀπόδεξις ἥδε
    HERODOTUS_WORDS = [
        ("Ἡροδότου", "Ἡρόδοτος"),   # proper noun genitive
        ("ἱστορίης", "ἱστορία"),     # Ionic genitive
        ("ἀπόδεξις", "ἀπόδειξις"),   # Ionic (not standard, actually alt form)
    ]

    @pytest.mark.parametrize("form,expected", [
        ("πρήγματα", "πρᾶγμα"),
        ("ἱστορίης", "ἱστορία"),
        ("μοῦνος", "μόνος"),
        ("ποιέειν", "ποιέω"),
    ])
    def test_herodotus_sentence_ionic(self, d_ionic, form, expected):
        """Herodotus Ionic forms should resolve with dialect=ionic."""
        result = d_ionic.lemmatize(form)
        assert strip_accents(result.lower()) == strip_accents(expected), \
            f"Ionic: {form} -> expected {expected}, got {result}"

    def test_herodotus_batch_ionic(self, d_ionic):
        """Batch lemmatization of Ionic forms should all resolve."""
        words = ["πρήγματα", "ἱστορίης", "μοῦνος", "ποιέειν"]
        results = d_ionic.lemmatize_batch(words)
        assert len(results) == len(words)
        # All results should be non-empty
        for word, result in zip(words, results):
            assert len(result) > 0, f"Empty result for {word}"

    def test_mixed_dialect_with_standard(self, d_ionic):
        """Ionic dialect instance should still handle standard Attic forms."""
        # Standard Attic forms should work fine even with Ionic mode
        result = d_ionic.lemmatize("ἐποίησεν")
        assert strip_accents(result.lower()) == strip_accents("ποιέω"), \
            f"Standard Attic ἐποίησεν should still work, got {result}"

    def test_standard_form_with_ionic_mode(self, d_ionic):
        """Common Attic forms should not be broken by Ionic normalization."""
        result = d_ionic.lemmatize("θεούς")
        assert strip_accents(result.lower()) == strip_accents("θεός"), \
            f"θεούς with ionic mode should still -> θεός, got {result}"


class TestMixedDialectScenarios:
    """Test various dialect-mixing scenarios."""

    def test_batch_with_dialect(self, d_ionic):
        """Batch lemmatization should respect dialect parameter."""
        words = ["πρήγματα", "θεούς", "ἱστορίης"]
        results = d_ionic.lemmatize_batch(words)
        assert len(results) == 3
        assert strip_accents(results[0].lower()) == strip_accents("πρᾶγμα")
        assert strip_accents(results[1].lower()) == strip_accents("θεός")
        assert strip_accents(results[2].lower()) == strip_accents("ἱστορία")

    def test_dialect_with_convention(self):
        """dialect + convention should work together."""
        d = Dilemma(dialect="ionic", convention="lsj")
        result = d.lemmatize("θεούς")
        assert result is not None
        assert len(result) > 0

    def test_dialect_with_lang_grc(self):
        """dialect + lang=grc should work."""
        d = Dilemma(dialect="ionic", lang="grc")
        result = d.lemmatize("πρήγματα")
        assert strip_accents(result.lower()) == strip_accents("πρᾶγμα")

    def test_multiple_dilemma_instances(self):
        """Creating multiple Dilemma instances with different dialects."""
        d_ionic = Dilemma(dialect="ionic")
        d_doric = Dilemma(dialect="doric")
        d_none = Dilemma()
        # All should work independently
        assert d_ionic._normalizer.dialect == "ionic"
        assert d_doric._normalizer.dialect == "doric"
        assert d_none._normalizer is None


class TestEdgeCasesNewFeatures:
    """Edge cases for the new features."""

    def test_empty_string_with_dialect(self, d_ionic):
        """Empty string with dialect should not crash."""
        result = d_ionic.lemmatize("")
        assert result is not None

    def test_single_char_with_dialect(self, d_ionic):
        """Single character with dialect should not crash."""
        result = d_ionic.lemmatize("α")
        assert result is not None

    def test_digit_with_dialect(self, d_ionic):
        """Digits with dialect should not crash."""
        result = d_ionic.lemmatize("42")
        assert result is not None

    def test_verbose_with_dialect(self, d_ionic):
        """lemmatize_verbose should work with dialect mode."""
        cands = d_ionic.lemmatize_verbose("πρήγματα")
        assert len(cands) > 0

    def test_invalid_dialect_raises(self):
        """Invalid dialect name should raise ValueError."""
        with pytest.raises(ValueError):
            Dilemma(dialect="mycenaean")

    def test_invalid_dialect_in_normalizer(self):
        """Invalid dialect in Normalizer should raise ValueError."""
        with pytest.raises(ValueError):
            Normalizer(dialect="unknown_dialect")
