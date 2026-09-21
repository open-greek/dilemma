"""Elision-safe normalization shared by the corpus frequency builders."""

from __future__ import annotations

import sys
from pathlib import Path


BUILD_DIR = Path(__file__).resolve().parents[1] / "build"
sys.path.insert(0, str(BUILD_DIR))

from corpus_freq_key import (  # noqa: E402
    beta_trailing_paren_is_elision,
    corpus_freq_key,
)
from tei_tokenize import GREEK_RUN, normalize_key  # noqa: E402
from build_cog_public_freq import _git_commit, _sha256  # noqa: E402
from merge_corpus_freq import load_freq  # noqa: E402


def test_corpus_key_preserves_and_canonicalizes_elision_marks():
    variants = ["δ'", "δ`", "δʼ", "δ᾽", "δ᾿", "δ’", "δ\u0313"]
    assert {corpus_freq_key(form) for form in variants} == {"δ’"}


def test_corpus_key_strips_accents_but_not_numeral_prime():
    assert corpus_freq_key("ἄνθρωπος") == "ανθρωπος"
    assert corpus_freq_key("αʹ") == "αʹ"


def test_spacing_psili_distinguishes_breathing_from_aphaeresis():
    assert corpus_freq_key("᾿ο") == "ο"
    assert corpus_freq_key("᾿ς") == "’ς"


def test_diorisis_beta_paren_distinguishes_breathing_from_elision():
    assert not beta_trailing_paren_is_elision("ou)")
    assert not beta_trailing_paren_is_elision("*ei)")
    assert beta_trailing_paren_is_elision("d)")
    assert beta_trailing_paren_is_elision("di)")
    assert beta_trailing_paren_is_elision("par)")
    assert beta_trailing_paren_is_elision("a)ll)")


def test_tei_tokenizer_keeps_decomposed_elision_mark():
    source = "δ\u0313 ἀλλ᾽ λόγος"
    assert GREEK_RUN.findall(source) == ["δ\u0313", "ἀλλ᾽", "λόγος"]
    assert normalize_key("δ\u0313") == "δ’"


def test_cog_provenance_helpers_pin_content_and_commit(tmp_path):
    lexicon = tmp_path / "public_lexicon.tsv"
    lexicon.write_text("δ’\t3\n", encoding="utf-8")
    assert _sha256(lexicon) == (
        "38cf8036c24086b0a6ad04165d24a07d4f70b656317730094aa5a67f875a69d9"
    )
    assert _git_commit(lexicon) == ""


def test_merge_loader_preserves_artifact_source_provenance(tmp_path):
    artifact = tmp_path / "freq.json"
    artifact.write_text(
        '{"_total_tokens":3,"_sources":["OGC commit=abc; sha256=def"],'
        '"forms":{"δ’":[3]}}',
        encoding="utf-8",
    )

    total, forms, sources = load_freq(artifact)

    assert total == 3
    assert forms == {"δ’": [3]}
    assert sources == ["OGC commit=abc; sha256=def"]
