"""A smooth-breathing αι/οι form is never the article (issue #112)."""

import unicodedata

import pytest

from dilemma import core
from dilemma.core import Dilemma


def resolver(*, enabled=True, lang="grc"):
    instance = object.__new__(Dilemma)
    instance.lang = lang
    instance._resolve_articles = enabled
    instance._convention_name = None
    return instance


@pytest.mark.parametrize("form", ["αἴ", "αἲ", "αἶ", "οἴ", "οἲ", "οἶ", "αἰ", "οἰ",
                                  "Αἴ", "ΑἼ", "Οἴ"])
def test_smooth_breathing_is_not_the_article(form):
    assert not Dilemma._is_ancient_article_form(form)
    assert resolver()._resolve_closed_class(form) is None


@pytest.mark.parametrize("form", ["ὀ", "τἠς"])
def test_scribal_smooth_breathing_on_other_articles_is_kept(form):
    # DBBE writes ὀ and τἠς for the article; only αι/οι have real
    # smooth-breathing homographs.
    assert Dilemma._is_ancient_article_form(form)
    assert resolver()._resolve_closed_class(form) == "ὁ"


def test_decomposed_smooth_breathing_is_not_the_article():
    form = unicodedata.normalize("NFD", "αἴ")
    assert not Dilemma._is_ancient_article_form(form)


@pytest.mark.parametrize("form", ["αἵ", "οἵ", "αἳ", "οἳ", "αἱ", "οἱ", "Αἱ", "ΟἹ",
                                  "αί", "οί", "αι", "οι", "ΑΙ", "ΟΙ", "ἡ", "ὁ"])
def test_rough_breathing_and_unbreathed_articles_still_resolve(form):
    assert Dilemma._is_ancient_article_form(form)
    assert resolver()._resolve_closed_class(form) == "ὁ"


@pytest.fixture(scope="module", params=["all", "grc"])
def shipped_article_model(request):
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    return Dilemma(lang=request.param, resolve_articles=True)


def test_shipped_smooth_breathing_interjections_keep_their_readings(
        shipped_article_model):
    model = shipped_article_model
    forms = ["αἴ", "αἴ", "αἲ", "οἴ"]
    tags = ["SCONJ", "PART", "PART", "INTJ"]
    tagged = model.lemmatize_batch_pos(forms, tags, guess=False)
    assert "ὁ" not in tagged
    assert tagged[1:] == ["αἴ", "αἴ", "οἴ"]
    assert [model.lemmatize_pos(f, t, guess=False) for f, t in zip(forms, tags)] == tagged
    assert model.lemmatize_batch(["αἴ", "οἴ"], guess=False) == ["αἴ", "οἴ"]
    assert model.lemmatize("οἴ", guess=False) == "οἴ"
    assert all(c.source != "article"
               for c in model.lemmatize_verbose("αἴ", guess=False))
    # Controls from the issue.
    assert model.lemmatize_pos("αἴ", "INTJ", guess=False) == "αἴ"
    assert model.lemmatize_pos("αἴ", "CCONJ", guess=False) == "εἰ"
    assert model.lemmatize_pos("αἱ", "DET", guess=False) == "ὁ"
    assert model.lemmatize_pos("αἵ", "PRON", guess=False) == "ὅς"
    assert model.lemmatize_pos("οἴμοι", "INTJ", guess=False) == "οἴμοι"
    assert model.lemmatize_batch(["αἵ", "οἵ", "αί", "Αἱ"], guess=False) == ["ὁ"] * 4


@pytest.mark.parametrize("form,elided", [
    ("αἶ̓", True), ("οἷ̓", True), ("βαί̓", True), ("ὁποῖ̓", True),
    ("αἰ", False), ("ἐ", False), ("οὐ", False), ("εἰ", False), ("ἀλλ̓", False),
])
def test_psili_after_the_vowels_own_marks_is_an_elision(form, elided):
    assert core._is_marked_vowel_psili_elision(form) is elided


@pytest.mark.parametrize("form", ["αἶ̓", "οἷ̓"])
def test_elided_form_with_combining_psili_is_not_the_article(form):
    # The trailing U+0313 is the apostrophe of αἶ(α) and οἷ(ος), not a
    # breathing; the elided word is no article.
    assert not Dilemma._is_ancient_article_form(form)
    assert resolver()._resolve_closed_class(form) is None


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_shipped_combining_psili_elision_agrees_across_calls(lang):
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    model = Dilemma(lang=lang, resolve_articles=True)
    forms = ["αἶ̓", "οἷ̓", "βαί̓", "ὁποῖ̓", "αἶ’", "οἷ’"]
    expected = ["αἶα", "οἷος", "βαιός", "ὁποῖος", "αἶα", "οἷος"]
    assert [model.lemmatize(f, guess=False) for f in forms] == expected
    assert model.lemmatize_batch(forms, guess=False) == expected
