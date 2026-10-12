"""Greek capitals with prosgegrammeni retain their lexical case identity."""
from __future__ import annotations

import sqlite3
import unicodedata
from types import SimpleNamespace
import json

import pytest

from build import build_treebank_pos_lookup as build
from build import build_glaux_pairs, build_glaux_pos_lookup
from dilemma.core import Dilemma
from dilemma.core import strip_accents as runtime_strip_accents
from dilemma import core
from dilemma.form_sanitize import is_capitalized
from export_hunspell import (
    FormProfileEvidence,
    cased_form_key,
    corpus_lexical_evidence_is_clean,
    exact_form_key,
    filter_case_artifacts,
    include_attested_name_ownership,
    include_reviewed_name_ownership,
    lexical_head_key,
    load_case_ownership,
    treebank_spelling_reason,
)


@pytest.mark.parametrize("form", ["ᾍδης", "ᾨδή", "ᾜειν", "ᾼσμα", "Ἅιδης", "ΑΘΗΝΑ"])
@pytest.mark.parametrize("normalization", ["NFC", "NFD"])
def test_capitals_include_titlecase_without_rewriting(form, normalization):
    original = unicodedata.normalize(normalization, form)
    assert is_capitalized(original)


@pytest.mark.parametrize("form", ["ᾅδης", "ᾠδή", "ᾔειν", "ᾳσμα", "λόγος", "", "1ᾍδης"])
def test_lowercase_subscripts_and_nonletter_initials_are_not_capitals(form):
    assert not is_capitalized(form)
    assert not is_capitalized(unicodedata.normalize("NFD", form))


def test_titlecase_lexical_owner_keeps_name_and_rejects_unattested_lowercase():
    connection = sqlite3.connect(":memory:")
    connection.executescript("""
        CREATE TABLE lemmas (id INTEGER, text TEXT);
        CREATE TABLE lookup (form TEXT, lemma_id INTEGER, src TEXT, lang TEXT);
        INSERT INTO lemmas VALUES (1, 'ᾍδης'), (2, 'οἴομαι');
        INSERT INTO lookup VALUES ('Ἀΐδα', 1, 'grc', 'all'),
            ('ᾍδῃ', 1, 'grc', 'all'), ('ᾬμην', 2, 'grc', 'all');
    """)
    ownership = load_case_ownership(connection)
    connection.close()
    assert ownership[exact_form_key("Ἀΐδα")] == 1
    assert ownership[exact_form_key("ᾍδῃ")] == 1
    assert ownership[exact_form_key("ᾬμην")] == 2
    pairs = [("Ἀΐδα", "ᾍδης"), ("ᾍδῃ", "ᾍδης"),
             ("ᾅδῃ", "ᾍδης"), ("ᾬμην", "οἴομαι")]
    evidence = FormProfileEvidence({}, {}, {}, {}, cased={})
    assert filter_case_artifacts(pairs, evidence, ownership) == (pairs[:2], pairs[2:])


@pytest.mark.parametrize("form,lemma", [("Ἀΐδα", "ᾍδης"), ("ᾍδῃ", "ᾍδης")])
@pytest.mark.parametrize("normalization", ["NFC", "NFD"])
def test_complete_owner_audit_and_export_preserve_titlecase_readings(form, lemma, normalization):
    form, lemma = (unicodedata.normalize(normalization, value) for value in (form, lemma))
    votes = {}
    build._record_capital_owner(votes, form, lemma, "NOUN", "glaux", 2,
                                {build._lexical_target_key(lemma)})
    readings = build._capital_owner_audit(votes)
    assert readings == {unicodedata.normalize("NFC", form): [
        {"upos": "NOUN", "lemma": unicodedata.normalize("NFC", lemma),
         "source_votes": {"glaux": 2}}]}
    ownership = {exact_form_key(form): 2}
    evidence = FormProfileEvidence({}, {}, {}, {}, cased={cased_form_key(form): 3})
    retained = include_attested_name_ownership(
        ownership, evidence, readings=readings,
        lexical_heads=frozenset({lexical_head_key(lemma)}))
    assert retained[0]["lemmas"] == [unicodedata.normalize("NFC", lemma)]
    assert ownership[exact_form_key(form)] == 3
    assert filter_case_artifacts([(form, lemma)], evidence, ownership)[1] == []


def test_lowercase_corpus_owner_stays_ineligible_and_reviewed_titlecase_owner_is_kept():
    votes = {}
    build._record_capital_owner(votes, "ᾍδῃ", "ᾅδης", "NOUN", "glaux", 2,
                                {build._lexical_target_key("ᾅδης")})
    assert votes == {}
    ownership = {exact_form_key("Ἀΐδα"): 2}
    include_reviewed_name_ownership(ownership, [{"form": "Ἀΐδα", "lemma": "ᾍδης"}])
    assert ownership[exact_form_key("Ἀΐδα")] == 3


def test_internal_titlecase_is_rejected_like_other_internal_capitals():
    assert not corpus_lexical_evidence_is_clean("λόγοςᾍδης")
    assert treebank_spelling_reason("λόγοςᾍδης") == "inner_capital"
    assert corpus_lexical_evidence_is_clean("ᾍδης")


def test_runtime_proper_classifier_and_case_twins_recognize_titlecase():
    model = Dilemma.__new__(Dilemma)
    assert model._is_proper("ᾍδης")
    assert not model._is_proper("ᾅδης")
    candidates = [SimpleNamespace(lemma="ᾅδης"), SimpleNamespace(lemma="ᾍδης")]
    assert model._prefer_case_twin("ᾍδῃ", "PROPN", "ᾅδης", candidates) == "ᾍδης"
    assert model._prefer_case_twin("ᾅδῃ", "NOUN", "ᾍδης", candidates) == "ᾅδης"
    assert model._prefer_case_twin("ᾍδῃ", "NOUN", "ᾍδης", candidates) == "ᾍδης"


def test_runtime_headword_repair_prefers_same_titlecase_identity():
    model = Dilemma.__new__(Dilemma)
    heads = {"ᾍδης", "ᾅδης"}
    key = runtime_strip_accents("ᾍδης".lower())
    model._canonical_ag_headwords = lambda: (heads, {key: heads}, heads)
    model.attestation = lambda lemma: None
    assert model._canonicalize_headword("ᾍδη̄ς") == "ᾍδης"


@pytest.mark.parametrize("word,lower,capital", [
    ("ᾅδῃ", "ᾅδης", "ᾍδης"),
    ("ἔριδι", "ἔρις", "Ἔρις"),
])
def test_actual_verbose_lookup_retains_lowercase_and_titlecase_twins(
    word, lower, capital, tmp_path, monkeypatch,
):
    monkeypatch.setattr(Dilemma, "_load_lookups", lambda *args, **kwargs: None)
    monkeypatch.setattr(Dilemma, "_check_backend", lambda *args: None)
    for name in ("AG_POS_LOOKUP_PATH", "GLAUX_POS_LOOKUP_PATH",
                 "TREEBANK_POS_LOOKUP_PATH", "MG_POS_LOOKUP_PATH"):
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(core, name, path)
    model = Dilemma(lang="grc", skip_pos=True)
    model._ag_lookup = {word: lower, lower: lower, capital: capital}
    model._lookup = model._ag_lookup
    model.attestation = lambda lemma: None
    candidates = model.lemmatize_verbose(word, guess=False)
    assert {(row.lemma, row.proper) for row in candidates} == {
        (lower, False), (capital, True)}
    assert next(row for row in candidates if row.lemma == capital).via == "exact+case_alt"


@pytest.mark.parametrize("normalization", ["NFC", "NFD"])
def test_glaux_capital_shape_preserves_internal_and_all_capital_rejection(normalization):
    normalize = lambda value: unicodedata.normalize(normalization, value)
    assert build_glaux_pairs._capital_greek(normalize("ᾍδης"))
    assert not build_glaux_pairs._capital_greek(normalize("ᾅδης"))
    assert not build_glaux_pairs._capital_greek(normalize("Αᾍδης"))
    assert not build_glaux_pairs._capital_greek("ΑΔΗΣ")


def test_glaux_pos_table_keeps_titlecase_proper_reading(tmp_path, monkeypatch):
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps([
        {"form": "Ἀΐδα", "lemma": "ᾍδης", "pos": "noun"},
        {"form": "Ἀΐδα", "lemma": "ᾄδω", "pos": "verb"},
    ]), encoding="utf-8")
    monkeypatch.setattr(build_glaux_pos_lookup, "GLAUX_PAIRS", path)
    monkeypatch.setattr(build_glaux_pos_lookup, "OUTPUT_PATH", tmp_path / "pos.json")
    result = build_glaux_pos_lookup.build_lookup()
    assert result["Ἀΐδα"] == {"NOUN": "ᾍδης", "PROPN": "ᾍδης", "VERB": "ᾄδω"}
