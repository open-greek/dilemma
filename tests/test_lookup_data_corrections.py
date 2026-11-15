"""Shipped answers for the 1.3.25 lookup and table corrections.

#103 (unaccented dictionary-expansion lemmas), #108 (reviewed non-lemmas and
capitalised pronoun twins), #110 (reviewed lookup overrides) and #122 (Modern
Greek table fragments). Most of these answers come from data/lookup.db and
data/mg_pos_lookup.json as rebuilt for 1.3.25.

test_unaccented_expansion_artefact_is_gone stays xfail: the rebuild drops the
expansion reading λαβῃ -> λαβλυμι, but GLAUx carries the same unaccented
lemma for the form and refills the key (GLAUX_REFILL below).
"""

import pytest

from dilemma import core
from dilemma.grc_rejected_lemmas import REJECTED_LEMMAS

GLAUX_REFILL = pytest.mark.xfail(
    strict=False,
    reason="GLAUx refills λαβῃ with the same unaccented lemma λαβλυμι; "
           "the #103 filter judges expansion readings only")

_MODELS = {}


def _model(lang):
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    if lang not in _MODELS:
        _MODELS.clear()
        _MODELS[lang] = core.Dilemma(lang=lang, resolve_articles=True)
    return _MODELS[lang]


# #108 ---------------------------------------------------------------------

@pytest.mark.parametrize("lang", ["all", "grc"])
def test_sentence_initial_pronouns_take_the_pronoun_lemma(lang):
    model = _model(lang)
    forms, tags = ["Αὕτη", "Τούτου", "Πάντα"], ["PRON", "PRON", "ADJ"]
    expected = ["οὗτος", "οὗτος", "πᾶς"]
    assert model.lemmatize_batch_pos(forms, tags, guess=False) == expected
    assert [model.lemmatize_pos(f, t, guess=False)
            for f, t in zip(forms, tags)] == expected


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_sentence_initial_pronouns_untagged(lang):
    model = _model(lang)
    forms = ["Ταύτης", "Πάντες", "Αὕτη", "Τούτου", "Πάντα"]
    expected = ["οὗτος", "πᾶς", "οὗτος", "οὗτος", "πᾶς"]
    assert model.lemmatize_batch(forms, guess=False) == expected
    assert [model.lemmatize(f, guess=False) for f in forms] == expected


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_pos_cells_of_reviewed_non_lemmas_are_ignored(lang):
    model = _model(lang)
    forms = ["Ταύτην", "ταύτην", "Τοῦτον", "Ἡμῶν"]
    tags = ["PROPN", "NOUN", "PROPN", "PROPN"]
    expected = ["οὗτος", "οὗτος", "οὗτος", "ἐγώ"]
    assert model.lemmatize_batch_pos(forms, tags, guess=False) == expected
    assert [model.lemmatize_pos(f, t, guess=False)
            for f, t in zip(forms, tags)] == expected
    for table in (model._pos_lookup, model._pos_ag_lookup):
        assert not {lemma for cells in table.values()
                    for lemma in cells.values()} & REJECTED_LEMMAS


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_no_answer_is_a_reviewed_non_lemma(lang):
    model = _model(lang)
    forms = ["Ταύτη", "Ταύτῃ", "Ταύτης", "Τοῦτος", "Ἡμων", "ἡμων", "Τοῦτο"]
    for form in forms:
        assert model.lemmatize(form, guess=False) not in REJECTED_LEMMAS


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_pronoun_controls_and_other_heads_keep_their_readings(lang):
    model = _model(lang)
    assert model.lemmatize_batch_pos(
        ["Ταύτην", "Οὗτος", "Ταῦτα", "Πάντες"], ["PRON", "PRON", "PRON", "ADJ"],
        guess=False) == ["οὗτος", "οὗτος", "οὗτος", "πᾶς"]
    assert model.lemmatize("ἄτη", guess=False) == "ἄτη"
    assert model.lemmatize("τοῦτος", guess=False) == "τοῦτος"


# #103 ---------------------------------------------------------------------

@GLAUX_REFILL
@pytest.mark.parametrize("lang", ["all", "grc", "el"])
def test_unaccented_expansion_artefact_is_gone(lang):
    model = _model(lang)
    assert model.lemmatize_batch_pos(["λαβηι"], ["NOUN"], guess=False) != ["λαβλυμι"]
    assert model.lemmatize("λαβηι", guess=False) != "λαβλυμι"
    if lang == "all":
        assert model.lemmatize("λαβηι", guess=False) == "λαβή"


def test_unaccented_expansion_doublet_is_repaired():
    assert _model("all").lemmatize("σύνοχα", guess=False) == "σύνοχος"


# #122 ---------------------------------------------------------------------

def test_ancient_participles_do_not_take_a_modern_fragment():
    model = _model("all")
    forms = ["οὐσῶν", "οὐσῶν", "οὖσης", "ὅντων", "ὄντων"]
    tags = ["ADJ", "VERB", "ADJ", "ADJ", "ADJ"]
    assert model.lemmatize_batch_pos(forms, tags, guess=False) == ["εἰμί"] * 5
    for table in (model._pos_lookup,):
        for key in ("ούσης", "ουσης", "ουσών", "ουσων", "όντων", "οντων"):
            assert table.get(key, {}).get("ADJ") != "αρχαΐζων"


def test_modern_participle_fragment_in_el():
    model = _model("el")
    assert model.lemmatize_batch_pos(["ὄντων", "οὐσῶν"], ["ADJ", "ADJ"],
                                     guess=False) == ["εἰμί", "εἰμί"]


# #110 ---------------------------------------------------------------------

@pytest.mark.parametrize("lang", ["all", "grc"])
def test_merged_layer_reassignments_are_corrected_tagged(lang):
    model = _model(lang)
    forms = ["γραφῆς", "γραφήν", "γραφή", "γραφαί", "σεῖο", "σέο"]
    tags = ["NOUN"] * 4 + ["PRON"] * 2
    expected = ["γραφή"] * 4 + ["σύ"] * 2
    assert model.lemmatize_batch_pos(forms, tags, guess=False) == expected
    assert [model.lemmatize_pos(f, t, guess=False)
            for f, t in zip(forms, tags)] == expected


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_merged_layer_reassignments_are_corrected_untagged(lang):
    model = _model(lang)
    forms = ["σύ", "σοῦ", "σέθεν", "γραφή"]
    expected = ["σύ", "σύ", "σύ", "γραφή"]
    assert model.lemmatize_batch(forms, guess=False) == expected
    assert [model.lemmatize(f, guess=False) for f in forms] == expected


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_single_valued_lookup_corrections(lang):
    model = _model(lang)
    forms = ["ἱερέων", "ἄριστα", "ἀρίστοις", "ἀρὰν"]
    tags = ["NOUN", "ADJ", "ADJ", "NOUN"]
    assert model.lemmatize_batch_pos(forms, tags, guess=False) == [
        "ἱερεύς", "ἀγαθός", "ἀγαθός", "ἀρά"]


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_lookup_correction_controls(lang):
    model = _model(lang)
    assert model.lemmatize_batch_pos(["γραφὴν", "ἀράν", "ἱερά"],
                                     ["NOUN", "NOUN", "ADJ"], guess=False) == [
        "γραφή", "ἀρά", "ἱερός"]
    assert model.lemmatize_pos("Λάμπε", "VERB", guess=False) == "λάμπω"


# Shipped lookup.db invariants ------------------------------------------------

def _lookup_rows(sql, params=()):
    import sqlite3
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    conn = sqlite3.connect(f"file:{core.LOOKUP_DB_PATH}?mode=ro", uri=True)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def test_no_unaccented_unlisted_expansion_lemma_ships():
    import build_lookup_db as builder
    heads, by_letters = builder.load_expansion_lemma_heads(core.GRC_LEXICAL_HEADS_PATH)
    rows = _lookup_rows(
        "SELECT DISTINCT m.text FROM lookup l JOIN lemmas m ON m.id = l.lemma_id "
        "JOIN sources s ON s.id = l.source_id WHERE s.code = 'grc_lexicon_expansion'")
    assert not [lemma for (lemma,) in rows
                if builder.unaccented_expansion_lemma(lemma, heads, by_letters)]


def test_no_reviewed_non_lemma_ships():
    marks = ",".join("?" * len(REJECTED_LEMMAS))
    assert _lookup_rows(
        "SELECT l.form, m.text FROM lookup l JOIN lemmas m ON m.id = l.lemma_id "
        f"WHERE m.text IN ({marks})", sorted(REJECTED_LEMMAS)) == []
