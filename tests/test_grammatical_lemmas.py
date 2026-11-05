"""Grammatical inflections stay separate from related dictionary headwords."""
from dataclasses import asdict
import hashlib
import json
import sqlite3
import unicodedata

import pytest

from dilemma import core
from dilemma.grammatical_lemmas import (
    APEIMI_IMPERFECT, correct_grammatical_lemma, reviewed_verb_grammar,
)


@pytest.fixture(params=["json", "sqlite"])
def engine_factory(request, tmp_path, monkeypatch):
    lookup = dict.fromkeys(APEIMI_IMPERFECT, "ἀπέρχομαι")
    lookup.update({"ἄπειμι": "ἄπειμι", "ἀπῆμεν": "ἄπειμι",
                   "ἀπέλθομεν": "ἀπέρχομαι", "λόγοι": "λόγος"})
    database = tmp_path / "lookup.db"
    with sqlite3.connect(database) as db:
        db.executescript("CREATE TABLE lemmas (id INTEGER PRIMARY KEY, text TEXT);"
                         "CREATE TABLE lookup (form TEXT, lemma_id INTEGER, lang TEXT);")
        for index, (form, lemma) in enumerate(lookup.items()):
            db.execute("INSERT INTO lemmas VALUES (?, ?)", (index, lemma))
            db.execute("INSERT INTO lookup VALUES (?, ?, 'all')", (form, index))

    def load(self, **kwargs):
        self._lookup = (dict(lookup) if request.param == "json"
                        else core.LookupDB(database))
        self._ag_lookup = self._lookup
        self._mg_lookup = self._lookup
        self._pos_lookup = {form: {"VERB": "ἀπέρχομαι"} for form in APEIMI_IMPERFECT}
        self._pos_ag_lookup = self._pos_lookup

    monkeypatch.setattr(core.Dilemma, "_load_lookups", load)
    heads = tmp_path / "heads.json"
    heads.write_text(json.dumps({"schema_version": 2, "heads": {
        "ἄπειμι": ["lsj"], "ἀπέρχομαι": ["lsj"]}}), encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"schema_version": 2, "language": "grc",
        "rejected_corpus_targets": {}, "inputs": {"grc_lexical_heads": {
            "size": heads.stat().st_size,
            "sha256": hashlib.sha256(heads.read_bytes()).hexdigest()}}}), encoding="utf-8")
    monkeypatch.setattr(core, "GRC_LEXICAL_HEADS_PATH", heads)
    monkeypatch.setattr(core, "TREEBANK_POS_AUDIT_PATH", audit)

    def make(**kwargs):
        engine = core.Dilemma(skip_pos=True, **kwargs)
        engine._trusted_grave_targets = {"ἄπειμι", "ἀπέρχομαι"}
        return engine
    return make


@pytest.mark.parametrize("lang", ["grc", "all"])
@pytest.mark.parametrize("form", ["ἀπῇμεν", "Ἀπῇμεν", "ἀπῄμεν",
                                  unicodedata.normalize("NFD", "ἀπῇμεν")])
def test_reported_form_across_public_apis(engine_factory, lang, form):
    engine = engine_factory(lang=lang)
    assert engine.lemmatize(form, guess=False) == "ἄπειμι"
    assert engine.lemmatize_batch([form], guess=False) == ["ἄπειμι"]
    assert engine.lemmatize_pos(form, "VERB", guess=False) == "ἄπειμι"
    assert engine.lemmatize_batch_pos([form], ["VERB"], guess=False) == ["ἄπειμι"]
    candidate, = engine.lemmatize_verbose(form, guess=False)
    assert candidate.lemma == "ἄπειμι" and candidate.tag == "VERB"
    grammar = asdict(candidate)["grammar"]
    assert grammar["lemma"] == "ἄπειμι" and grammar["base_lemma"] == "εἶμι"
    analysis, = grammar["analyses"]
    assert (analysis["tense"], analysis["voice"], analysis["mood"],
            analysis["person"], analysis["number"]) == (
                "imperfect", "active", "indicative", "1", "pl")
    related, = grammar["related_headwords"]
    assert related["lemma"] == "ἀπέρχομαι"
    assert related["relation"] == "present_used_as_future"
    assert related["applies_to_input"] is False
    assert len(grammar["evidence"]) == 2
    json.dumps(asdict(candidate))
    assert engine._model is None


@pytest.mark.parametrize("form", APEIMI_IMPERFECT)
def test_other_reviewed_imperfects_have_their_own_person_and_number(engine_factory, form):
    engine = engine_factory(lang="grc")
    candidate, = engine.lemmatize_verbose(form, guess=False)
    assert candidate.lemma == "ἄπειμι"
    assert {(a["person"], a["number"]) for a in candidate.grammar["analyses"]} == set(
        APEIMI_IMPERFECT[form])
    assert engine.lemmatize_pos(form, "VERB", guess=False) == "ἄπειμι"


def test_homonym_and_other_tenses_do_not_receive_this_analysis(engine_factory):
    engine = engine_factory(lang="grc")
    for form, lemma in [("ἀπῆμεν", "ἄπειμι"), ("ἄπειμι", "ἄπειμι"),
                        ("ἀπέλθομεν", "ἀπέρχομαι"), ("λόγοι", "λόγος")]:
        candidate, = engine.lemmatize_verbose(form, guess=False)
        assert candidate.lemma == lemma and candidate.grammar is None
    assert reviewed_verb_grammar("απημεν") is None
    assert correct_grammatical_lemma("ἀπῇμεν", "ἄλλος") == "ἄλλος"


def test_modern_language_scope_keeps_existing_lookup(engine_factory):
    engine = engine_factory(lang="el")
    candidate, = engine.lemmatize_verbose("ἀπῇμεν", guess=False)
    assert candidate.lemma == "ἀπέρχομαι" and candidate.grammar is None
    assert engine.lemmatize("ἀπῇμεν", guess=False) == "ἀπέρχομαι"


def test_grammar_survives_requested_headword_convention(engine_factory):
    engine = engine_factory(lang="grc")
    engine._convention_map = {"ἄπειμι": "ἀπέρχομαι"}
    candidate, = engine.lemmatize_verbose("ἀπῇμεν", guess=False)
    assert candidate.lemma == "ἀπέρχομαι"
    assert candidate.grammar["lemma"] == "ἄπειμι"


def test_metadata_is_not_shared_between_calls(engine_factory):
    engine = engine_factory(lang="grc")
    first, = engine.lemmatize_verbose("ἀπῇμεν", guess=False)
    first.grammar["analyses"][0]["tense"] = "future"
    second, = engine.lemmatize_verbose("ἀπῇμεν", guess=False)
    assert second.grammar["analyses"][0]["tense"] == "imperfect"


def test_attestation_and_strict_citation_gates_still_apply(engine_factory, monkeypatch):
    engine = engine_factory(lang="grc", citation_policy="strict_ag")
    assert engine.lemmatize("ἀπῇμεν", guess=False) == "ἄπειμι"
    monkeypatch.setattr(engine, "_resolve_attested_form", lambda word: None)
    assert engine.lemmatize("ἀπῇμεν", guess=False, attested_only=True) is None
    assert engine.lemmatize_batch(["ἀπῇμεν"], guess=False, attested_only=True) == [None]
    assert engine.lemmatize_verbose("ἀπῇμεν", guess=False, attested_only=True) == []
    assert engine.lemmatize_pos("ἀπῇμεν", "VERB", guess=False, attested_only=True) is None
    assert engine.lemmatize_batch_pos(["ἀπῇμεν"], ["VERB"], guess=False,
                                      attested_only=True) == [None]
    engine._trusted_grave_targets = set()
    assert engine.lemmatize("ἀπῇμεν", guess=False) is None
    assert engine.lemmatize_verbose("ἀπῇμεν", guess=False) == []
