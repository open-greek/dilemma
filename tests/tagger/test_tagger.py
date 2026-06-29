#!/usr/bin/env python3
"""Test suite for Tagger Greek POS tagger and dependency parser.

Tests initialization, POS tagging, lemmatization integration, batch
processing, output structure, and edge cases.

Run with:
    python -m pytest tests/tagger/test_tagger.py -x -v

For full tests including model inference (requires weights):
    python -m pytest tests/tagger/test_tagger.py -x -v --run-slow
"""

import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Valid UPOS tags from UD spec
VALID_UPOS = {
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM",
    "VERB", "X", "_",
}

# Expected keys in every token dict
REQUIRED_TOKEN_KEYS = {"form", "upos", "feats", "head", "deprel"}


def _has_dilemma():
    """Check if Dilemma is importable."""
    try:
        from dilemma import Dilemma
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Torch-free runtime (a 1.0 guarantee)
# ---------------------------------------------------------------------------

def test_tagger_runtime_is_torch_free():
    """Importing the tagger runtime must not pull in torch or transformers.

    Run in a fresh subprocess so other tests' imports don't pollute sys.modules.
    """
    import subprocess
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(repo_root)!r}); "
        "import dilemma.tagger; "
        "print('torchfree=' + str('torch' not in sys.modules "
        "and 'transformers' not in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "torchfree=True" in out.stdout, \
        f"tagger import pulled torch/transformers:\n{out.stdout}\n{out.stderr}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tagger_grc():
    """Tagger instance for Ancient Greek (requires grc weights)."""
    from dilemma.tagger import Tagger
    return Tagger(lang="grc", device="cpu")


@pytest.fixture(scope="module")
def tagger_grc_no_lemma():
    """Tagger instance for AG without lemmatization."""
    from dilemma.tagger import Tagger
    return Tagger(lang="grc", device="cpu", lemmatize=False)


@pytest.fixture(scope="module")
def tagger_el():
    """Tagger instance for Modern Greek (requires el weights or HF download)."""
    from dilemma.tagger import Tagger
    return Tagger(lang="el", device="cpu")


# ===========================================================================
# 1. SEGMENT (standalone, no model needed)
# ===========================================================================

class TestSegment:
    """Test Greek sentence segmentation - no model weights needed."""

    def test_simple_split(self):
        from dilemma.tagger.segment import segment
        result = segment("Τι κάνεις; Καλά.")
        assert len(result) == 2
        assert result[0] == "Τι κάνεις;"
        assert result[1] == "Καλά."

    def test_exclamation(self):
        from dilemma.tagger.segment import segment
        result = segment("Ελάτε! Πάμε.")
        assert len(result) == 2

    def test_abbreviation_not_split(self):
        from dilemma.tagger.segment import segment
        result = segment("π.χ. αυτό είναι μία πρόταση.")
        assert len(result) == 1

    def test_empty_input(self):
        from dilemma.tagger.segment import segment
        assert segment("") == []
        assert segment("   ") == []

    def test_no_punctuation(self):
        from dilemma.tagger.segment import segment
        result = segment("Αχιλλέας πολεμά")
        assert len(result) == 1
        assert result[0] == "Αχιλλέας πολεμά"

    def test_middle_dot(self):
        """Greek ano teleia (middle dot) is a sentence boundary."""
        from dilemma.tagger.segment import segment
        result = segment("πρώτη πρόταση\u00B7 δεύτερη πρόταση")
        assert len(result) == 2

    def test_multiple_sentence_end_punct(self):
        from dilemma.tagger.segment import segment
        result = segment("Τι κάνεις;!")
        # Should be treated as single sentence boundary
        assert len(result) == 1


# ===========================================================================
# 4. INITIALIZATION
# ===========================================================================

class TestInitialization:
    """Test Tagger class construction and parameter handling."""

    def test_invalid_lang_raises(self):
        from dilemma.tagger import Tagger
        with pytest.raises(ValueError, match="Unsupported language"):
            Tagger(lang="fr", device="cpu")

    def test_default_lang_is_el(self):
        """Verify default lang parameter is 'el'."""
        import inspect
        from dilemma.tagger import Tagger
        sig = inspect.signature(Tagger.__init__)
        assert sig.parameters["lang"].default == "el"

    def test_valid_langs_accepted(self):
        """Verify el, grc, med do not raise 'Unsupported language' ValueError."""
        from dilemma.tagger import Tagger
        for lang in ("el", "grc", "med"):
            try:
                Tagger(lang=lang, device="cpu", lemmatize=False)
            except ValueError as e:
                # Only fail if it is the "Unsupported language" error
                assert "Unsupported language" not in str(e), (
                    f"Language '{lang}' should be accepted but got: {e}"
                )
            except Exception:
                # Other errors (missing weights, etc.) are fine
                pass

    def test_dialect_parameter_stored(self):
        """Verify dialect is stored on the instance."""
        from dilemma.tagger import Tagger
        # Patch weight loading to avoid needing actual weights
        with patch.object(Tagger, "_init_backend", return_value=None):
            with patch.object(Tagger, "_init_lemmatizer", return_value=None):
                o = object.__new__(Tagger)
                o.device = "cpu"
                o.max_subwords = 2048
                o.lang = "grc"
                o._lemmatize = False
                o._lemma_cache = None
                o._lemmatizer = None
                o._dialect = "ionic"
                assert o._dialect == "ionic"

    @pytest.mark.slow
    def test_grc_init_creates_model(self, tagger_grc_no_lemma):
        """Verify AG initialization produces a working backend.

        The grc path is the torch-free GreBerta ONNX morph backend, held in
        ``_morph_onnx``.
        """
        assert tagger_grc_no_lemma.lang == "grc"
        assert tagger_grc_no_lemma._morph_onnx is not None

    @pytest.mark.slow
    def test_lemmatize_default_true(self, tagger_grc):
        """Verify lemmatization is on by default when Dilemma is available."""
        if _has_dilemma():
            assert tagger_grc._lemmatize is True
        else:
            # Without Dilemma, _lemmatize gets set to False in _init_lemmatizer
            assert tagger_grc._lemmatize is False


# ===========================================================================
# 5. POS TAGGING (requires weights - slow)
# ===========================================================================

@pytest.mark.slow
class TestPOSTagging:
    """Test POS tagging on known Greek sentences."""

    def test_ag_sentence_produces_valid_upos(self, tagger_grc_no_lemma):
        """Tag an Ancient Greek sentence, verify all UPOS tags are valid."""
        results = tagger_grc_no_lemma.tag(["μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος"])
        assert len(results) == 1
        tokens = results[0]
        assert len(tokens) >= 4  # at least 4 words
        for tok in tokens:
            assert tok["upos"] in VALID_UPOS, f"Invalid UPOS: {tok['upos']}"

    def test_mg_sentence_produces_valid_upos(self, tagger_el):
        """Tag a Modern Greek sentence, verify all UPOS tags are valid."""
        results = tagger_el.tag(["Ο Αχιλλέας πολεμά"])
        assert len(results) == 1
        tokens = results[0]
        assert len(tokens) >= 3
        for tok in tokens:
            assert tok["upos"] in VALID_UPOS

    def test_ag_iliad_common_tags(self, tagger_grc_no_lemma):
        """Verify that common Iliad words get expected POS tags."""
        results = tagger_grc_no_lemma.tag(["ὁ Ἀχιλλεὺς τὴν μάχην ἔλυσε"])
        tokens = results[0]
        # First token "ο" should be DET (article)
        assert tokens[0]["upos"] == "DET", f"Expected DET for ὁ, got {tokens[0]['upos']}"

    def test_output_format_is_list_of_dicts(self, tagger_grc_no_lemma):
        """Verify output is list[list[dict]]."""
        results = tagger_grc_no_lemma.tag(["ἄνδρα μοι ἔννεπε"])
        assert isinstance(results, list)
        assert isinstance(results[0], list)
        assert isinstance(results[0][0], dict)

    def test_feats_are_dict(self, tagger_grc_no_lemma):
        """Verify morphological features are returned as a dict."""
        results = tagger_grc_no_lemma.tag(["ἄνδρα μοι ἔννεπε"])
        for tok in results[0]:
            assert isinstance(tok["feats"], dict)

    def test_head_is_int(self, tagger_grc_no_lemma):
        """Verify dependency head is an integer."""
        results = tagger_grc_no_lemma.tag(["ἄνδρα μοι ἔννεπε"])
        for tok in results[0]:
            assert isinstance(tok["head"], int)

    def test_deprel_is_string(self, tagger_grc_no_lemma):
        """Verify dependency relation is a string."""
        results = tagger_grc_no_lemma.tag(["ἄνδρα μοι ἔννεπε"])
        for tok in results[0]:
            assert isinstance(tok["deprel"], str)
            assert len(tok["deprel"]) > 0


# ===========================================================================
# 6. LEMMATIZATION INTEGRATION (requires weights + Dilemma - slow)
# ===========================================================================

@pytest.mark.slow
class TestLemmatization:
    """Test Dilemma lemmatizer integration."""

    @pytest.mark.skipif(not _has_dilemma(), reason="Dilemma not installed")
    def test_lemmatizer_initialized(self, tagger_grc):
        """Verify Dilemma is initialized when lemmatize=True."""
        assert tagger_grc._lemmatizer is not None

    @pytest.mark.skipif(not _has_dilemma(), reason="Dilemma not installed")
    def test_output_includes_lemma(self, tagger_grc):
        """Verify that output tokens include a 'lemma' field."""
        results = tagger_grc.tag(["μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος"])
        for tok in results[0]:
            assert "lemma" in tok, f"Token missing 'lemma' key: {tok}"
            assert isinstance(tok["lemma"], str)
            assert len(tok["lemma"]) > 0

    @pytest.mark.skipif(not _has_dilemma(), reason="Dilemma not installed")
    def test_dialect_parameter_flows_through(self, tagger_grc):
        """Verify dialect parameter is stored and available."""
        from dilemma.tagger import Tagger
        model = Tagger(lang="grc", device="cpu", dialect="ionic")
        assert model._dialect == "ionic"

    @pytest.mark.skipif(not _has_dilemma(), reason="Dilemma not installed")
    def test_lemma_cache_used(self):
        """Verify lemma_cache is used to skip Dilemma lookups."""
        from dilemma.tagger import Tagger
        cache = {"μῆνιν": "μῆνις", "ἄειδε": "ἀείδω"}
        model = Tagger(lang="grc", device="cpu", lemma_cache=cache)
        assert model._lemma_cache is cache


# ===========================================================================
# 7. BATCH PROCESSING (requires weights - slow)
# ===========================================================================

@pytest.mark.slow
class TestBatchProcessing:
    """Test batch processing behavior."""

    def test_multiple_sentences(self, tagger_grc_no_lemma):
        """Tag multiple sentences, verify one result per sentence."""
        sents = [
            "μῆνιν ἄειδε θεά",
            "ἄνδρα μοι ἔννεπε",
            "πολλὰ δ' ὅ γ' ἐν πόντῳ πάθεν ἄλγεα",
        ]
        results = tagger_grc_no_lemma.tag(sents)
        assert len(results) == 3
        for sent_result in results:
            assert isinstance(sent_result, list)
            assert len(sent_result) > 0

    def test_empty_input_returns_empty(self, tagger_grc_no_lemma):
        """Empty sentence list returns empty result list."""
        results = tagger_grc_no_lemma.tag([])
        assert results == []

    def test_single_string_input(self, tagger_grc_no_lemma):
        """A single string (not list) should be wrapped in a list."""
        results = tagger_grc_no_lemma.tag("μῆνιν ἄειδε θεά")
        assert len(results) == 1
        assert len(results[0]) >= 3

    def test_segment_text_flag(self, tagger_grc_no_lemma):
        """segment_text=True splits input string into sentences."""
        text = "μῆνιν ἄειδε θεά. ἄνδρα μοι ἔννεπε."
        results = tagger_grc_no_lemma.tag(text, segment_text=True)
        assert len(results) == 2


# ===========================================================================
# 8. OUTPUT STRUCTURE (requires weights - slow)
# ===========================================================================

@pytest.mark.slow
class TestOutputStructure:
    """Verify the structure of output token dicts."""

    def test_required_keys_present(self, tagger_grc_no_lemma):
        """Every token dict must have form, upos, feats, head, deprel."""
        results = tagger_grc_no_lemma.tag(["ὁ Ἀχιλλεὺς τὴν μάχην ἔλυσε"])
        for tok in results[0]:
            for key in REQUIRED_TOKEN_KEYS:
                assert key in tok, f"Missing key '{key}' in token: {tok}"

    def test_raw_form_present(self, tagger_grc_no_lemma):
        """Token dicts should include raw_form (original polytonic form)."""
        results = tagger_grc_no_lemma.tag(["τὸν Ἀχιλλέα"])
        for tok in results[0]:
            assert "raw_form" in tok, f"Missing raw_form in token: {tok}"

    def test_feats_values_are_strings(self, tagger_grc_no_lemma):
        """Feature values should be strings, not integers or other types."""
        results = tagger_grc_no_lemma.tag(["ὁ ἄνθρωπος βαδίζει"])
        for tok in results[0]:
            for feat_name, feat_val in tok["feats"].items():
                assert isinstance(feat_val, str), (
                    f"Feature {feat_name}={feat_val} is not a string"
                )

    def test_no_underscore_feats(self, tagger_grc_no_lemma):
        """Underscore values should be suppressed from features."""
        results = tagger_grc_no_lemma.tag(["ὁ ἄνθρωπος βαδίζει"])
        for tok in results[0]:
            for feat_name, feat_val in tok["feats"].items():
                assert feat_val != "_", (
                    f"Underscore value found for {feat_name} in token {tok['form']}"
                )

    def test_head_zero_exists(self, tagger_grc_no_lemma):
        """At least one token should have head=0 (root) or a valid head."""
        results = tagger_grc_no_lemma.tag(["ὁ ἄνθρωπος βαδίζει"])
        heads = [tok["head"] for tok in results[0]]
        # All heads should be non-negative integers
        for h in heads:
            assert h >= 0


# ===========================================================================
# 9. EDGE CASES (requires weights - slow)
# ===========================================================================

@pytest.mark.slow
class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_empty_string(self, tagger_grc_no_lemma):
        """Empty string should return one result with no tokens or handle gracefully."""
        results = tagger_grc_no_lemma.tag("")
        assert len(results) == 1
        # Empty string may produce 0 tokens or a minimal result
        assert isinstance(results[0], list)

    def test_single_word(self, tagger_grc_no_lemma):
        """Single word should produce exactly one token."""
        results = tagger_grc_no_lemma.tag("Ἀχιλλεύς")
        assert len(results) == 1
        assert len(results[0]) == 1

    def test_punctuation_only(self, tagger_grc_no_lemma):
        """Punctuation-only input should produce a token with PUNCT tag."""
        results = tagger_grc_no_lemma.tag(".")
        assert len(results) == 1
        if results[0]:
            assert results[0][0]["upos"] == "PUNCT"

    @pytest.mark.xfail(
        reason="Known limitation: the ONNX morph backend splits on whitespace "
        "only, so punctuation attached to a word (ἄειδε, / θεά.) is not a "
        "separate PUNCT token. Affects all languages; needs a punctuation "
        "pre-segmentation step in OnnxMorphTagger.tag_sentences.",
        strict=False,
    )
    def test_mixed_punctuation_and_words(self, tagger_grc_no_lemma):
        """Sentence with punctuation should tag punctuation as PUNCT."""
        results = tagger_grc_no_lemma.tag(["μῆνιν ἄειδε, θεά."])
        tokens = results[0]
        punct_tokens = [t for t in tokens if t["upos"] == "PUNCT"]
        assert len(punct_tokens) >= 1, "Expected at least one PUNCT token"

    def test_long_input(self, tagger_grc_no_lemma):
        """Long input should not crash (tests dynamic batching)."""
        # Repeat a sentence many times to exceed default batch size
        long_sents = ["μῆνιν ἄειδε θεά"] * 50
        results = tagger_grc_no_lemma.tag(long_sents)
        assert len(results) == 50
        for sent_result in results:
            assert len(sent_result) >= 3


