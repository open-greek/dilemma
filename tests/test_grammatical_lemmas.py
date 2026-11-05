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
    reviewed_verb_lemmas, reviewed_verb_readings,
)

ADDITIONAL_READINGS = [
    ("ᾖμεν", "εἶμι", "ἔρχομαι"),
    ("ἐξῇμεν", "ἔξειμι", "ἐξέρχομαι"),
    ("εἰσῇμεν", "εἴσειμι", "εἰσέρχομαι"),
    ("προσῇμεν", "πρόσειμι", "προσέρχομαι"),
    ("παρῇμεν", "πάρειμι", "παρέρχομαι"),
    ("προῇμεν", "πρόειμι", "προέρχομαι"),
]


@pytest.fixture(params=["json", "sqlite"])
def engine_factory(request, tmp_path, monkeypatch):
    lookup = dict.fromkeys(APEIMI_IMPERFECT, "ἀπέρχομαι")
    lookup.update({form: headword for form, _, headword in ADDITIONAL_READINGS})
    lookup.update({"ἄπειμι": "ἄπειμι", "ἀπῆμεν": "ἄπειμι",
                   "ἀπέλθομεν": "ἀπέρχομαι", "λόγοι": "λόγος",
                   "ᾖσαν": "ἀείδω", "ἴσθι": "εἰμί", "ἦν": "εἰμί",
                   "ἦ": "ἦ", "ξύνισαν": "σύνοιδα", "ἔπεσαν": "ἔπειμι",
                   "ἐπήϊεν": "ἐπαΐω", "ἐπήισαν": "ἔπειμι",
                   "ᾔειν": "ἔρχομαι", "κάτῃμεν": "κατέρχομαι",
                   "ἐξῄειν": "ἔξειμι", "ᾔδειν": "οἶδα"})
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
        self._pos_lookup = {form: {"VERB": lemma} for form, lemma in lookup.items()}
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
@pytest.mark.parametrize("form,lemma,headword", ADDITIONAL_READINGS)
def test_other_reviewed_suppletive_families(engine_factory, lang, form, lemma, headword):
    engine = engine_factory(lang=lang)
    assert engine.lemmatize(form, guess=False) == lemma
    assert engine.lemmatize_batch([form], guess=False) == [lemma]
    assert engine.lemmatize_pos(form, "VERB", guess=False) == lemma
    assert engine.lemmatize_batch_pos([form], ["VERB"], guess=False) == [lemma]
    candidate, = engine.lemmatize_verbose(form, guess=False)
    assert candidate.lemma == candidate.grammar["lemma"] == lemma
    assert candidate.grammar["related_headwords"][0]["lemma"] == headword
    analysis, = candidate.grammar["analyses"]
    assert (analysis["person"], analysis["number"], analysis["tense"]) == ("1", "pl", "imperfect")


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


@pytest.mark.parametrize("form,lemma,tense,people", [
    ("ᾔειν", "εἶμι", "imperfect", {("1", "sg"), ("3", "sg")}),
    ("ᾔεισθα", "εἶμι", "imperfect", {("2", "sg")}),
    ("ἐξῄειν", "ἔξειμι", "imperfect", {("1", "sg"), ("3", "sg")}),
    ("πρόσῃτε", "πρόσειμι", "imperfect", {("2", "pl")}),
    ("κάτῃμεν", "κάτειμι", "imperfect", {("1", "pl")}),
    ("περιῄτην", "περίειμι", "imperfect", {("3", "du")}),
    ("διῄεισθα", "δίειμι", "imperfect", {("2", "sg")}),
    ("ᾔδειν", "οἶδα", "pluperfect", {("1", "sg"), ("3", "sg")}),
])
def test_expanded_readings_have_independent_person_and_tense_expectations(
        engine_factory, form, lemma, tense, people):
    engine = engine_factory(lang="grc")
    assert engine.lemmatize(form, guess=False) == lemma
    assert engine.lemmatize_batch([form], guess=False) == [lemma]
    assert engine.lemmatize_pos(form, "VERB", guess=False) == lemma
    assert engine.lemmatize_batch_pos([form], ["VERB"], guess=False) == [lemma]
    candidate, = engine.lemmatize_verbose(form, guess=False)
    assert candidate.grammar["lemma"] == lemma
    assert {a["tense"] for a in candidate.grammar["analyses"]} == {tense}
    assert {(a["person"], a["number"]) for a in candidate.grammar["analyses"]} == people


@pytest.mark.parametrize("form,preferred,expected", [
    ("ᾖσαν", "ἀείδω", {"εἶμι": "imperfect", "οἶδα": "pluperfect", "ἀείδω": "aorist"}),
    ("ἴσθι", "εἰμί", {"εἰμί": "present", "οἶδα": "perfect"}),
    ("ἦν", "εἰμί", {"εἰμί": "imperfect", "ἠμί": "imperfect"}),
])
def test_lexical_ambiguity_keeps_every_reading_and_existing_default(
        engine_factory, form, preferred, expected):
    engine = engine_factory(lang="grc")
    candidates = engine.lemmatize_verbose(form, guess=False)
    assert candidates[0].lemma == engine.lemmatize(form, guess=False) == preferred
    assert {c.grammar["lemma"]: c.grammar["analyses"][0]["tense"]
            for c in candidates} == expected
    assert engine.lemmatize_batch([form], guess=False) == [preferred]
    assert reviewed_verb_grammar(form) is None
    assert form not in reviewed_verb_lemmas()


def test_pos_can_choose_an_alternative_without_erasing_other_readings(engine_factory):
    engine = engine_factory(lang="grc")
    engine._pos_lookup["ᾖσαν"] = {"VERB": "οἶδα"}
    assert engine.lemmatize_pos("ᾖσαν", "VERB", guess=False) == "οἶδα"
    assert engine.lemmatize_batch_pos(["ᾖσαν"], ["VERB"], guess=False) == ["οἶδα"]
    assert engine.lemmatize("ᾖσαν", guess=False) == "ἀείδω"
    assert len(engine.lemmatize_verbose("ᾖσαν", guess=False)) == 3


def test_convention_collisions_do_not_discard_grammatical_identities(engine_factory):
    engine = engine_factory(lang="grc")
    engine._convention_map = {"εἶμι": "ἄπειμι", "οἶδα": "ἄπειμι", "ἀείδω": "ἄπειμι"}
    candidates = engine.lemmatize_verbose("ᾖσαν", guess=False)
    assert [c.lemma for c in candidates] == ["ἄπειμι"] * 3
    assert {c.grammar["lemma"] for c in candidates} == {"εἶμι", "οἶδα", "ἀείδω"}


def test_ambiguous_readings_normalize_unicode_and_are_not_shared(engine_factory):
    engine = engine_factory(lang="grc")
    for form in ("ᾖσαν", "ᾖσαν".title(), unicodedata.normalize("NFD", "ᾖσαν")):
        assert len(engine.lemmatize_verbose(form, guess=False)) == 3
    readings = reviewed_verb_readings("ᾖσαν")
    readings[0]["analyses"].clear()
    assert reviewed_verb_readings("ᾖσαν")[0]["analyses"]


@pytest.mark.parametrize("form,preferred,readings", [
    ("ξύνισαν", "σύνοιδα", {("σύνοιδα", "pluperfect"), ("σύνειμι", "imperfect")}),
    ("ἐπήϊεν", "ἐπαΐω", {("ἐπαΐω", "imperfect"), ("ἔπειμι", "imperfect")}),
    ("ἐπήισαν", "ἔπειμι", {("ἐπαΐω", "aorist"), ("ἔπειμι", "imperfect")}),
    ("ἔπεσαν", "ἔπειμι", {("πίπτω", "aorist"), ("ἔπειμι", "imperfect")}),
])
def test_compound_homographs_retain_independent_verbs(engine_factory, form, preferred, readings):
    engine = engine_factory(lang="grc")
    assert engine.lemmatize(form, guess=False) == preferred
    candidates = engine.lemmatize_verbose(form, guess=False)
    assert {(c.grammar["lemma"], c.grammar["analyses"][0]["tense"])
            for c in candidates} == readings


def test_pos_headword_correction_keeps_a_different_scalar_preference(engine_factory):
    engine = engine_factory(lang="grc")
    engine._pos_lookup["ἐπήϊεν"] = {"VERB": "ἐπέρχομαι"}
    assert engine.lemmatize("ἐπήϊεν", guess=False) == "ἐπαΐω"
    assert engine.lemmatize_pos("ἐπήϊεν", "VERB", guess=False) == "ἔπειμι"
    assert engine.lemmatize_batch_pos(["ἐπήϊεν"], ["VERB"], guess=False) == ["ἔπειμι"]
    assert {c.lemma for c in engine.lemmatize_verbose("ἐπήϊεν", guess=False)} == {"ἐπαΐω", "ἔπειμι"}


def test_nonverbal_owner_survives_beside_reviewed_verbs(engine_factory):
    engine = engine_factory(lang="grc")
    candidates = engine.lemmatize_verbose("ἦ", guess=False)
    assert candidates[0].lemma == engine.lemmatize("ἦ", guess=False) == "ἦ"
    assert candidates[0].grammar is None
    assert {c.grammar["lemma"] for c in candidates if c.grammar} == {"ἠμί", "εἰμί"}


def test_copula_retains_both_tenses_and_scopes_doric_plural(engine_factory):
    engine = engine_factory(lang="grc")
    candidate, = engine.lemmatize_verbose("ἦτε", guess=False)
    assert engine.lemmatize("ἦτε", guess=False) == "εἰμί"
    assert {(a["tense"], a["mood"]) for a in candidate.grammar["analyses"]} == {
        ("imperfect", "indicative"), ("present", "subjunctive")}
    copula = next(c.grammar for c in engine.lemmatize_verbose("ἦν", guess=False)
                  if c.grammar["lemma"] == "εἰμί")
    plural, = [a for a in copula["analyses"] if a["number"] == "pl"]
    assert plural["person"] == "3" and plural["dialect"] == "doric"


def test_ambiguous_candidates_each_pass_the_citation_and_attestation_gates(
        engine_factory, monkeypatch):
    engine = engine_factory(lang="grc", citation_policy="strict_ag")
    engine._trusted_grave_targets = {"οἶδα"}
    candidates = engine.lemmatize_verbose("ᾖσαν", guess=False)
    assert [c.lemma for c in candidates] == ["οἶδα"]
    assert candidates[0].grammar["lemma"] == "οἶδα"
    monkeypatch.setattr(engine, "_resolve_attested_form", lambda word: None)
    assert engine.lemmatize_verbose("ᾖσαν", guess=False, attested_only=True) == []
