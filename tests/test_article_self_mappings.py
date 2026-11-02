"""Recognizing an Ancient article is separate from canonicalizing it to ὁ."""
import hashlib
import json
import unicodedata

import pytest

from dilemma import core


@pytest.fixture
def article_engine(tmp_path, monkeypatch):
    # Deliberately conflicting lookup/POS readings reproduce both missing
    # article entries and entries incorrectly assigned to another headword.
    lookup = {"ὁ": "ὅς", "ἡ": "ή", "τό": "τά", "τοῖς": "ο",
              "τοῦ": "ὅς", "ἤ": "ἤ", "ἦ": "ἦ", "ἥ": "ὅς", "ὅ": "ὅς"}

    def load(self, **kwargs):
        self._lookup = dict(lookup)
        self._ag_lookup = self._lookup
        self._pos_ag_lookup = {form: {"DET": "ὅς"} for form in lookup}

    monkeypatch.setattr(core.Dilemma, "_load_lookups", load)
    heads = tmp_path / "heads.json"
    heads.write_text(json.dumps({"schema_version": 2, "heads": {
        "ὁ": ["fixture"]}}), encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "schema_version": 2, "language": "grc", "rejected_corpus_targets": {},
        "inputs": {"grc_lexical_heads": {"size": heads.stat().st_size,
            "sha256": hashlib.sha256(heads.read_bytes()).hexdigest()}}}), encoding="utf-8")
    monkeypatch.setattr(core, "GRC_LEXICAL_HEADS_PATH", heads)
    monkeypatch.setattr(core, "TREEBANK_POS_AUDIT_PATH", audit)

    def make(**kwargs):
        engine = core.Dilemma(**kwargs)
        engine._trusted_grave_targets = {"ὁ"}
        return engine

    return make


@pytest.mark.parametrize("lang", ["grc", "all"])
@pytest.mark.parametrize("resolve", [False, True])
@pytest.mark.parametrize("form,retained", [
    ("ὁ", "ὁ"), ("ἡ", "ἡ"), ("τό", "τό"), ("τὸ", "τό"),
    ("τοῖς", "τοῖς"), ("τοῦ", "τοῦ"), ("τῆς", "τῆς"),
    ("οἱ", "οἱ"), ("αἱ", "αἱ"), ("τῷ", "τῷ"), ("τῇ", "τῇ"),
    ("τὸν", "τόν"), ("τὴν", "τήν"), ("τοὺς", "τούς"),
    ("τὰς", "τάς"), ("τά", "τά"), ("τοῖν", "τοῖν"),
    ("ταῖν", "ταῖν"), ("τώ", "τώ"), ("Ὁ", "ὁ"), ("Ἡ", "ἡ"),
    (unicodedata.normalize("NFD", "ἡ"), "ἡ"),
])
def test_article_recognition_across_public_apis(article_engine, lang, resolve, form, retained):
    engine = article_engine(lang=lang, resolve_articles=resolve)
    expected = "ὁ" if resolve else retained
    assert engine.lemmatize(form, guess=False) == expected
    assert engine.lemmatize_batch([form], guess=False) == [expected]
    candidates = engine.lemmatize_verbose(form, guess=False)
    assert [candidate.lemma for candidate in candidates] == [expected]
    assert candidates[0].source == "article"
    if not resolve:
        assert candidates[0].lang == "grc"
        assert candidates[0].tag == "DET"
    assert engine.lemmatize_pos(form, "DET", guess=False) == expected
    assert engine.lemmatize_batch_pos([form], ["DET"], guess=False) == [expected]


@pytest.mark.parametrize("word,lemma", [("ἤ", "ἤ"), ("ἦ", "ἦ"), ("ἥ", "ὅς"), ("ὅ", "ὅς")])
def test_accented_particles_and_relative_pronouns_are_not_articles(article_engine, word, lemma):
    engine = article_engine(lang="grc")
    assert engine.lemmatize(word, guess=False) == lemma
    assert engine.lemmatize_batch([word], guess=False) == [lemma]
    assert engine.lemmatize_verbose(word, guess=False)[0].lemma == lemma


def test_modern_mode_keeps_lookup_behavior(article_engine):
    engine = article_engine(lang="el")
    assert engine.lemmatize("ἡ", guess=False) == "ή"


def test_article_recognition_respects_attestation_gate(article_engine, monkeypatch):
    engine = article_engine(lang="grc")
    monkeypatch.setattr(engine, "_resolve_attested_form", lambda word: None)
    assert engine.lemmatize("ἡ", guess=False, attested_only=True) is None
    assert engine.lemmatize_batch(["ἡ"], guess=False, attested_only=True) == [None]
    assert engine.lemmatize_verbose("ἡ", guess=False, attested_only=True) == []
    assert engine.lemmatize_pos("ἡ", "DET", guess=False, attested_only=True) is None
    assert engine.lemmatize_batch_pos(["ἡ"], ["DET"], guess=False, attested_only=True) == [None]


def test_strict_citation_policy_still_requires_headword_evidence(article_engine):
    engine = article_engine(lang="grc", citation_policy="strict_ag")
    assert engine.lemmatize("τοῖς", guess=False) is None
    assert engine.lemmatize_verbose("τοῖς", guess=False) == []
    assert engine.citation_status("ξζψχβγ", lang="grc")["ok"] is False
