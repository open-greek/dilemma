"""A NOUN tag on a dative with an adverbial self-map returns the noun (issue #111)."""

import pytest

from dilemma import core


def pos_model(lemmas, pos_lemma, lang="grc"):
    model = core.Dilemma.__new__(core.Dilemma)
    model.lang = lang
    model._convention_name = None
    model._pos_rejected_targets = frozenset()  # This fixture supplies no vetoed targets.
    candidates = [core.LemmaCandidate(lemma=value, source="lookup",
                  proper=value[:1].isupper()) for value in lemmas]
    model._lemmatize_verbose_impl = lambda word, guess: candidates
    model._pos_table_lookup = lambda word, upos: pos_lemma
    model._article_pos_lemma = lambda word, upos: None
    model._apply_convention = lambda lemma: lemma
    model.lemmatize_batch = lambda words, **kwargs: [lemmas[0]] * len(words)
    model.lemmatize = lambda word, **kwargs: lemmas[0]
    return model


@pytest.mark.parametrize("lang", ["grc", "all"])
@pytest.mark.parametrize("word,noun", [
    ("σιγῇ", "σιγή"), ("βίᾳ", "βία"), ("ἡμέρᾳ", "ἡμέρα"),
])
def test_subscript_selfmap_defers_to_a_nominal_target(lang, word, noun):
    model = pos_model([word], noun, lang=lang)
    for upos in ("NOUN", "PROPN"):
        assert model._prefer_exact_pos_citation(word, upos, word, noun) == noun
    assert model.lemmatize_pos(word, "NOUN", guess=False) == noun
    assert model.lemmatize_batch_pos([word], ["NOUN"], guess=False) == [noun]


@pytest.mark.parametrize("upos", ["ADV", "ADJ", "VERB"])
def test_subscript_selfmap_stays_under_other_tags(upos):
    model = pos_model(["σιγῇ"], "σιγή")
    assert model._prefer_exact_pos_citation("σιγῇ", upos, "σιγῇ", "σιγή") == "σιγῇ"


def test_independent_word_still_keeps_its_accent():
    # The οὖ beside οὐ safeguard is untouched: no iota subscript is involved.
    model = pos_model(["οὖ"], "οὐ")
    assert model._prefer_exact_pos_citation("οὖ", "ADV", "οὖ", "οὐ") == "οὖ"
    assert model._prefer_exact_pos_citation("οὖ", "NOUN", "οὖ", "οὐ") == "οὖ"


def test_subscript_in_the_stem_is_not_the_dative_mark():
    # ᾠδή has its own subscript; the dative adds one more.
    model = pos_model(["ᾠδῇ"], "ᾠδή")
    assert model._prefer_exact_pos_citation("ᾠδῇ", "NOUN", "ᾠδῇ", "ᾠδή") == "ᾠδή"
    # Equal subscripts: the ordinary exact-citation rule keeps the candidate.
    assert model._prefer_exact_pos_citation("ᾠδή", "NOUN", "ᾠδή", "ᾠδη") == "ᾠδή"


def test_modern_greek_is_untouched():
    model = pos_model(["σιγῇ"], "σιγή", lang="el")
    assert model._prefer_exact_pos_citation("σιγῇ", "NOUN", "σιγῇ", "σιγή") == "σιγῇ"


@pytest.fixture(scope="module", params=["all", "grc"])
def shipped_model(request):
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    return core.Dilemma(lang=request.param, resolve_articles=True)


_ISSUE_ROWS = [("βίᾳ", "βία"), ("ἡμέρᾳ", "ἡμέρα"), ("σιγῇ", "σιγή"),
               ("σιωπῇ", "σιωπή"), ("παρρησίᾳ", "παρρησία"), ("κομιδῇ", "κομιδή")]


def test_shipped_noun_datives_return_the_noun(shipped_model):
    forms = [form for form, _ in _ISSUE_ROWS]
    nouns = [noun for _, noun in _ISSUE_ROWS]
    assert shipped_model.lemmatize_batch_pos(forms, ["NOUN"] * len(forms),
                                             guess=False) == nouns
    for form, noun in _ISSUE_ROWS:
        assert shipped_model.lemmatize_pos(form, "NOUN", guess=False) == noun


def test_shipped_adverb_readings_and_controls_are_kept(shipped_model):
    model = shipped_model
    assert model.lemmatize_pos("σιγῇ", "ADV", guess=False) == "σιγῇ"
    assert model.lemmatize_batch_pos(["σιγῇ"], ["ADV"], guess=False) == ["σιγῇ"]
    assert model.lemmatize("βίᾳ", guess=False) == "βίᾳ"
    assert model.lemmatize_pos("οὖ", "ADV", guess=False) == "οὖ"
    assert model.lemmatize_batch_pos(["χρόνῳ", "ὁδῷ", "σπουδῇ"], ["NOUN"] * 3,
                                     guess=False) == ["χρόνος", "ὁδός", "σπουδή"]
