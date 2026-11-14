"""Shipped answers for the 1.3.25 lookup and table corrections.

#103 (unaccented dictionary-expansion lemmas), #108 (reviewed non-lemmas and
capitalised pronoun twins), #110 (reviewed lookup overrides) and #122 (Modern
Greek table fragments). Most of these answers come from data/lookup.db and
data/mg_pos_lookup.json, so they change only when the rebuilt artifacts are
published. Those tests are listed in SHIPPED_AFTER_REBUILD and marked
xfail(strict=False) with the reason "needs the 1.3.25 lookup rebuild"; drop
the marks once the rebuilt data is pinned in data/hf_manifest.json.
"""

import pytest

from dilemma import core
from dilemma.grc_rejected_lemmas import REJECTED_LEMMAS

REBUILD = pytest.mark.xfail(strict=False,
                            reason="needs the 1.3.25 lookup rebuild")

SHIPPED_AFTER_REBUILD = [
    "test_sentence_initial_pronouns_take_the_pronoun_lemma",
    "test_sentence_initial_pronouns_untagged",
    "test_no_answer_is_a_reviewed_non_lemma",
    "test_unaccented_expansion_artefact_is_gone",
    "test_unaccented_expansion_doublet_is_repaired",
    "test_ancient_participles_do_not_take_a_modern_fragment",
    "test_modern_participle_fragment_in_el",
]

_MODELS = {}


def _model(lang):
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    if lang not in _MODELS:
        _MODELS.clear()
        _MODELS[lang] = core.Dilemma(lang=lang, resolve_articles=True)
    return _MODELS[lang]


# #108 ---------------------------------------------------------------------

@REBUILD
@pytest.mark.parametrize("lang", ["all", "grc"])
def test_sentence_initial_pronouns_take_the_pronoun_lemma(lang):
    model = _model(lang)
    forms, tags = ["Αὕτη", "Τούτου", "Πάντα"], ["PRON", "PRON", "ADJ"]
    expected = ["οὗτος", "οὗτος", "πᾶς"]
    assert model.lemmatize_batch_pos(forms, tags, guess=False) == expected
    assert [model.lemmatize_pos(f, t, guess=False)
            for f, t in zip(forms, tags)] == expected


@REBUILD
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


@REBUILD
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

@REBUILD
@pytest.mark.parametrize("lang", ["all", "grc", "el"])
def test_unaccented_expansion_artefact_is_gone(lang):
    model = _model(lang)
    assert model.lemmatize_batch_pos(["λαβηι"], ["NOUN"], guess=False) != ["λαβλυμι"]
    assert model.lemmatize("λαβηι", guess=False) != "λαβλυμι"
    if lang == "all":
        assert model.lemmatize("λαβηι", guess=False) == "λαβή"


@REBUILD
def test_unaccented_expansion_doublet_is_repaired():
    assert _model("all").lemmatize("σύνοχα", guess=False) == "σύνοχος"


# #122 ---------------------------------------------------------------------

@REBUILD
def test_ancient_participles_do_not_take_a_modern_fragment():
    model = _model("all")
    forms = ["οὐσῶν", "οὐσῶν", "οὖσης", "ὅντων", "ὄντων"]
    tags = ["ADJ", "VERB", "ADJ", "ADJ", "ADJ"]
    assert model.lemmatize_batch_pos(forms, tags, guess=False) == ["εἰμί"] * 5
    for table in (model._pos_lookup,):
        for key in ("ούσης", "ουσης", "ουσών", "ουσων", "όντων", "οντων"):
            assert table.get(key, {}).get("ADJ") != "αρχαΐζων"


@REBUILD
def test_modern_participle_fragment_in_el():
    model = _model("el")
    assert model.lemmatize_batch_pos(["ὄντων", "οὐσῶν"], ["ADJ", "ADJ"],
                                     guess=False) == ["εἰμί", "εἰμί"]
