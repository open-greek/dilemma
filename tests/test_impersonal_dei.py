"""The impersonal δεῖ keeps its own headword (#155).

GLAUx files δεῖ and δεῖν under δέω, which is ambiguous between 'bind' and
'lack', and its POS cells answered a tagged lookup with δέω; the lookup row
of δεῖν read δέω and that of ἔδει the epic ἔδω under ἐσθίω. The reviewed
readings of δεῖ, δεῖν, ἔδει and δέοι answer δεῖ, untagged and under VERB
and AUX, as AGDT and the New Testament pairs read them. The forms that also
spell a personal form of δέω or δέομαι keep their rows.
"""
import hashlib
import json
import sqlite3
import unicodedata

import pytest

from dilemma import core
from dilemma.grammatical_lemmas import (
    reviewed_verb_grammar, reviewed_verb_lemmas, reviewed_verb_readings,
)

LOOKUP = {"δεῖ": "δεῖ", "δεῖν": "δέω", "ἔδει": "ἐσθίω", "δέοι": "δέω",
          "δέον": "δέον", "δέῃ": "δέω", "δεήσει": "δέω", "δεῖται": "δέω",
          "δέομαι": "δέω", "ἔδησε": "δέω"}
POS = {"δεῖ": {"VERB": "δέω", "NOUN": "Δεύς"}, "δεῖν": {"VERB": "δέω"},
       "ἔδει": {"VERB": "δέω"}, "δέοι": {"VERB": "δέω"},
       "δέον": {"VERB": "δέω"}, "δέῃ": {"VERB": "δέω"},
       "δεήσει": {"VERB": "δέω", "NOUN": "δέησις"},
       "δεῖται": {"VERB": "δέω"}, "δέομαι": {"VERB": "δέομαι"},
       "ἔδησε": {"VERB": "δέω"}}
IMPERSONAL = ["δεῖ", "δεῖν", "ἔδει", "δέοι"]
PERSONAL = [("δέον", "δέον", "δέω"), ("δέῃ", "δέω", "δέω"),
            ("δεήσει", "δέω", "δέω"), ("δεῖται", "δέω", "δέω"),
            ("δέομαι", "δέω", "δέομαι"), ("ἔδησε", "δέω", "δέω")]


@pytest.fixture(params=["json", "sqlite"])
def engine_factory(request, tmp_path, monkeypatch):
    database = tmp_path / "lookup.db"
    with sqlite3.connect(database) as db:
        db.executescript("CREATE TABLE lemmas (id INTEGER PRIMARY KEY, text TEXT);"
                         "CREATE TABLE lookup (form TEXT, lemma_id INTEGER, lang TEXT);")
        for index, (form, lemma) in enumerate(LOOKUP.items()):
            db.execute("INSERT INTO lemmas VALUES (?, ?)", (index, lemma))
            db.execute("INSERT INTO lookup VALUES (?, ?, 'all')", (form, index))

    def load(self, **kwargs):
        self._lookup = (dict(LOOKUP) if request.param == "json"
                        else core.LookupDB(database))
        self._ag_lookup = self._lookup
        self._mg_lookup = self._lookup
        self._pos_lookup = {form: dict(cells) for form, cells in POS.items()}
        self._pos_ag_lookup = self._pos_lookup

    monkeypatch.setattr(core.Dilemma, "_load_lookups", load)
    heads = tmp_path / "heads.json"
    heads.write_text(json.dumps({"schema_version": 2, "heads": {
        "δεῖ": ["lsj"], "δέω": ["lsj"]}}), encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"schema_version": 2, "language": "grc",
        "rejected_corpus_targets": {}, "inputs": {"grc_lexical_heads": {
            "size": heads.stat().st_size,
            "sha256": hashlib.sha256(heads.read_bytes()).hexdigest()}}}),
        encoding="utf-8")
    monkeypatch.setattr(core, "GRC_LEXICAL_HEADS_PATH", heads)
    monkeypatch.setattr(core, "TREEBANK_POS_AUDIT_PATH", audit)

    def make(**kwargs):
        return core.Dilemma(skip_pos=True, **kwargs)
    return make


@pytest.mark.parametrize("lang", ["grc", "all"])
@pytest.mark.parametrize("form", IMPERSONAL)
def test_impersonal_forms_read_dei(engine_factory, lang, form):
    engine = engine_factory(lang=lang)
    assert engine.lemmatize(form, guess=False) == "δεῖ"
    assert engine.lemmatize_batch([form], guess=False) == ["δεῖ"]
    for upos in ("VERB", "AUX"):
        assert engine.lemmatize_pos(form, upos, guess=False) == "δεῖ"
    assert engine.lemmatize_batch_pos(
        [form, form], ["VERB", "AUX"], guess=False) == ["δεῖ", "δεῖ"]
    candidate, = engine.lemmatize_verbose(form, guess=False)
    assert candidate.lemma == "δεῖ" and candidate.grammar["lemma"] == "δεῖ"


def test_the_issue_batch(engine_factory):
    engine = engine_factory(lang="all", resolve_articles=True)
    assert engine.lemmatize_batch_pos(
        ["δεῖν", "δεῖν", "δεῖ", "δεῖ"], ["VERB", "AUX", "VERB", "AUX"],
        guess=False) == ["δεῖ"] * 4


@pytest.mark.parametrize("form", ["Δεῖ", "Δεῖν", unicodedata.normalize("NFD", "δεῖν")])
def test_capital_and_decomposed_spellings(engine_factory, form):
    engine = engine_factory(lang="grc")
    assert engine.lemmatize(form, guess=False) == "δεῖ"
    assert engine.lemmatize_pos(form, "VERB", guess=False) == "δεῖ"


@pytest.mark.parametrize("form,untagged,verb", PERSONAL)
def test_personal_and_ambiguous_forms_keep_their_rows(engine_factory, form,
                                                      untagged, verb):
    engine = engine_factory(lang="grc")
    assert engine.lemmatize(form, guess=False) == untagged
    assert engine.lemmatize_pos(form, "VERB", guess=False) == verb
    assert reviewed_verb_grammar(form) is None


def test_modern_scope_keeps_its_lookup(engine_factory):
    engine = engine_factory(lang="el")
    assert engine.lemmatize("δεῖν", guess=False) == "δέω"


def test_readings_record_the_cells():
    infinitive, = reviewed_verb_readings("δεῖν")
    assert infinitive["analyses"] == [{
        **infinitive["analyses"][0], "mood": "infinitive", "tense": "present"}]
    for form, tense, mood in (("δεῖ", "present", "indicative"),
                              ("ἔδει", "imperfect", "indicative"),
                              ("δέοι", "present", "optative")):
        grammar = reviewed_verb_grammar(form)
        analysis, = grammar["analyses"]
        assert (grammar["lemma"], analysis["tense"], analysis["mood"],
                analysis["person"], analysis["number"]) == (
                    "δεῖ", tense, mood, "3", "sg")
        assert grammar["related_headwords"][0]["lemma"] == "δέω"
    assert {form: reviewed_verb_lemmas()[form] for form in IMPERSONAL} == \
        dict.fromkeys(IMPERSONAL, "δεῖ")
