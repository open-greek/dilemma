"""The order of the optional normalizer's respellings (#132) and the untagged
reading of the σε + article contractions (#148).
"""
import sqlite3
import unicodedata

import pytest

from dilemma import core

# (form, lemma, lang, src)
ROWS = [
    ("ανοιξη", "ἀνοίγω", "all", "grc"), ("ανοιξη", "άνοιξη", "el", "el"),
    ("άνοιξη", "άνοιξη", "all", "el"),
    ("αλλιώς", "αλλιώς", "all", "el"), ("αλλιως", "αλλιώς", "all", "el"),
    ("ανάβει", "ανάβω", "all", "el"), ("αναβει", "ανάβω", "all", "el"),
    ("εξης", "ἑξῆς", "all", "grc"), ("έξης", "έξη", "all", "el"),
    ("ἑξῆς", "ἑξῆς", "all", "grc"),
    ("ἁλιεῦσι", "ἁλιεύς", "all", "grc"),
    ("λογος", "λόγος", "all", "grc"), ("λόγος", "λόγος", "all", "grc"),
    ("στην", "ἵστημι", "all", "grc"), ("στῆν", "ἵστημι", "all", "grc"),
    ("στον", "στον", "all", "el"), ("στου", "ἕκαστος", "all", "grc"),
    ("στην", "σε", "el", "el"), ("στον", "σε", "el", "el"),
    ("στου", "σε", "el", "el"),
    # respellings of the normalizer test
    ("κρινοσ", "κρίνω", "all", "grc"), ("κρίνος", "κρίνον", "all", "grc"),
]


@pytest.fixture
def make(tmp_path, monkeypatch):
    database = tmp_path / "lookup.db"
    with sqlite3.connect(database) as db:
        db.executescript(
            "CREATE TABLE lemmas (id INTEGER PRIMARY KEY, text TEXT);"
            "CREATE TABLE lookup (form TEXT, lemma_id INTEGER, src TEXT,"
            " lang TEXT, source_id INTEGER);")
        ids = {}
        for form, lemma, lang, src in ROWS:
            if lemma not in ids:
                ids[lemma] = len(ids)
                db.execute("INSERT INTO lemmas VALUES (?, ?)", (ids[lemma], lemma))
            db.execute("INSERT INTO lookup VALUES (?, ?, ?, ?, 1)",
                       (form, ids[lemma], src, lang))

    def load(self, **kwargs):
        if self.lang == "grc":
            self._lookup = core.LookupDB(database, lang="grc")
            self._ag_lookup = self._lookup
        else:
            self._lookup = core.LookupDB(database, lang="all")
            self._ag_lookup = core.LookupDB(database, lang="grc")
            if self.lang == "el":
                self._mg_lookup = core.LookupDB(database, lang="el")
        self._using_db = True
        self._pos_lookup, self._pos_ag_lookup = {}, {}

    monkeypatch.setattr(core.Dilemma, "_load_lookups", load)
    def breathing(text):
        return any(mark in unicodedata.normalize("NFD", text)
                   for mark in "\u0313\u0314")
    # A loose key may not drop the breathing the form is written with.
    monkeypatch.setattr(core.Dilemma, "_restore_breathing",
                        lambda self, word, lemma: None
                        if breathing(word) and not breathing(lemma) else lemma)
    monkeypatch.setattr(core.Dilemma, "_strip_particle_suffix",
                        lambda self, word: None)
    monkeypatch.setattr(core.Dilemma, "_participle_nominative_lemma",
                        lambda self, word: None)
    monkeypatch.setattr(core.Dilemma, "_aorist_participle_lemma",
                        lambda self, word: None)
    monkeypatch.setattr(core.Dilemma, "_strip_verb_morphology",
                        lambda self, word: None)
    monkeypatch.setattr(core.Dilemma, "_decompose_compound",
                        lambda self, word, **kwargs: None)
    monkeypatch.setattr(core.Dilemma, "_prefix_strip_lookup",
                        lambda self, word: None)
    monkeypatch.setattr(core.Dilemma, "_expand_elision", lambda self, word: None)
    monkeypatch.setattr(core.Dilemma, "_lookup_elided", lambda self, word: None)

    def factory(**kwargs):
        model = core.Dilemma(skip_pos=True, **kwargs)
        model._apply_convention = lambda lemma, **kw: lemma
        return model
    return factory


class FakeNormalizer:
    def __init__(self, respellings):
        self.respellings = respellings

    def normalize(self, word):
        return list(self.respellings.get(word, ()))


def test_a_misaccented_ancient_word_still_reads_its_stripped_key(make):
    model = make(lang="grc")
    assert model.lemmatize("ἕξης", guess=False) == "ἑξῆς"
    assert model.lemmatize_batch(["ἕξης"], guess=False) == ["ἑξῆς"]


def test_the_normalizer_reads_rows_as_written_before_looser_keys(make):
    # The first respelling reaches a row only through its stripped key, the
    # second has a row as it is written: the second answers.
    model = make(lang="grc")
    model._normalizer = FakeNormalizer({"κρινος": ["κρινόσ", "κρίνος"]})
    assert model._normalized_reading("κρινος") == "κρίνον"
    model._normalizer = FakeNormalizer({"κρινος": ["κρινόσ"]})
    assert model._normalized_reading("κρινος") == "κρίνω"


@pytest.mark.parametrize("lang", ["all", "el"])
@pytest.mark.parametrize("word", ["στην", "στὴν", "στήν", "στον", "στου", "ΣΤΗΝ"])
def test_contractions_read_se_untagged(make, lang, word):
    model = make(lang=lang)
    assert model.lemmatize(word, guess=False) == "σε"
    assert model.lemmatize_batch([word], guess=False) == ["σε"]
    assert model.lemmatize_verbose(word, guess=False)[0].lemma == "σε"


@pytest.mark.parametrize("lang,word,expected", [
    ("grc", "στην", "ἵστημι"),   # the Ancient view reads Ancient Greek only
    ("all", "στῆν", "ἵστημι"),   # a circumflex spells the Ancient verb form
])
def test_ancient_spellings_keep_their_rows(make, lang, word, expected):
    assert make(lang=lang).lemmatize(word, guess=False) == expected


def test_other_tags_keep_the_row_of_a_contraction_spelling(make):
    model = make(lang="all")
    assert model.lemmatize_pos("στου", "ADJ", guess=False) == "ἕκαστος"
    assert model.lemmatize_batch_pos(["στου"], ["ADJ"], guess=False) == ["ἕκαστος"]
    assert model.lemmatize_pos("στου", "ADP", guess=False) == "σε"
    # The untagged reading is back once the tagged call returns.
    assert model.lemmatize("στου", guess=False) == "σε"
