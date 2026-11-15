"""Polytonic vernacular Greek under POS tags (#119, #120, #121).

* #119: with convention="triantafyllidis" the Modern and Ancient lookups
  read one row, so a Modern POS entry the polytonic rule let stand meets
  the same lemma from both sides; that is one competing reading.
* #120: the Modern Greek entries carry no SCONJ tag; an SCONJ request reads
  the Modern CCONJ entry when the Ancient table has no CCONJ entry.
* #121: the contractions of σε with the article give ο under DET and σε
  under ADP.
"""
import unicodedata

import pytest

from dilemma import core


@pytest.fixture
def table_model(tmp_path, monkeypatch):
    """A lang="all" model with given candidates and Ancient/Modern POS tables."""
    monkeypatch.setattr(core.Dilemma, "_load_lookups", lambda *a, **k: None)
    monkeypatch.setattr(core.Dilemma, "_check_backend", lambda *a: None)
    for name in ("AG_POS_LOOKUP_PATH", "GLAUX_POS_LOOKUP_PATH",
                 "TREEBANK_POS_LOOKUP_PATH", "MG_POS_LOOKUP_PATH"):
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(core, name, path)
    monkeypatch.setattr(core, "POS_CELL_CORRECTIONS_PATH", tmp_path / "none.json")

    def make(candidates, ancient, modern, *, applies=True, lang="all"):
        model = core.Dilemma(lang="grc", skip_pos=True)
        model.lang = lang
        model._pos_ag_lookup = {f: dict(p) for f, p in ancient.items()}
        combined = {f: dict(p) for f, p in ancient.items()}
        for form, by_pos in modern.items():
            for upos, lemma in by_pos.items():
                combined.setdefault(form, {}).setdefault(upos, lemma)
        model._pos_lookup = combined
        model._pos_rejected_targets = frozenset()
        made = [core.LemmaCandidate(lemma=lemma, lang=src_lang,
                                    source="lookup", via="exact")
                for lemma, src_lang in candidates]
        model._lemmatize_verbose_impl = lambda word, **kwargs: list(made)
        model._apply_convention = lambda lemma: lemma
        model._modern_pos_reading_applies = lambda word, upos, modern: applies
        model._lowercase_ancient_twin = lambda word, modern: None
        first = made[0].lemma if made else None
        model.lemmatize = lambda word, **kwargs: first
        model.lemmatize_batch = lambda words, **kwargs: [first] * len(words)
        return model

    return make


def both(model, word, upos):
    scalar = model.lemmatize_pos(word, upos, guess=False)
    batch = model.lemmatize_batch_pos([word], [upos], guess=False)
    assert batch == [scalar], (word, upos, scalar, batch)
    return scalar


# ---- #119 ----------------------------------------------------------------

def test_one_lemma_from_both_sides_yields_to_a_standing_modern_entry(table_model):
    model = table_model([("σέω", "el"), ("σέω", "grc")], {},
                        {"σεις": {"PRON": "εσείς"}})
    assert both(model, "σεῖς", "PRON") == "εσείς"


def test_two_lemmas_still_decide_against_the_modern_entry(table_model):
    model = table_model([("σέω", "el"), ("σεύω", "grc")], {},
                        {"σεις": {"PRON": "εσείς"}})
    assert both(model, "σεῖς", "PRON") == "σέω"


def test_an_ancient_only_entry_is_not_a_modern_one(table_model):
    # The Ancient table answers the tag itself: the existing rules apply
    # and two candidates are not collapsed.
    model = table_model([("κυρόω", "el"), ("κυρόω", "grc")],
                        {"κυροῦ": {"NOUN": "κῦρος"}}, {})
    assert not model._one_reading_against_modern_entry(
        "κυροῦ", "NOUN", model._lemmatize_verbose_impl("κυροῦ"))
    assert both(model, "κυροῦ", "NOUN") == "κυρόω"


@pytest.mark.parametrize("word,lang,upos", [
    ("σεις", "all", "PRON"),   # monotonic input never reaches the Modern branch
    ("σεῖς", "grc", "PRON"),   # lang="grc" has no Modern branch
    ("σεῖς", "all", "VERB"),   # VERB keeps its case-twin rule
])
def test_collapse_only_for_polytonic_lang_all_non_verb(table_model, word, lang, upos):
    model = table_model([("σέω", "el"), ("σέω", "grc")], {}, {}, lang=lang)
    assert not model._one_reading_against_modern_entry(
        word, upos, model._lemmatize_verbose_impl(word))


# ---- #120 ----------------------------------------------------------------

@pytest.mark.parametrize("word,key,lemma", [("νὰ", "να", "να"), ("σὰν", "σαν", "σαν")])
def test_sconj_reads_the_modern_cconj_entry(table_model, word, key, lemma):
    model = table_model([], {}, {key: {"CCONJ": lemma}})
    assert model._pos_table_lookup(word, "SCONJ") == lemma
    assert model._pos_table_lookup(word, "CCONJ") == lemma


def test_sconj_does_not_read_cconj_when_the_ancient_table_has_one(table_model):
    model = table_model([], {"νά": {"CCONJ": "νή"}}, {"να": {"CCONJ": "να"}})
    assert model._pos_table_lookup("νὰ", "SCONJ") is None


def test_sconj_entry_comes_before_cconj(table_model):
    model = table_model([], {}, {"να": {"CCONJ": "να", "SCONJ": "ναι"}})
    assert model._pos_table_lookup("νὰ", "SCONJ") == "ναι"


def test_sconj_cconj_entry_still_needs_the_rule(table_model):
    model = table_model([], {}, {"να": {"CCONJ": "να"}}, applies=False)
    assert model._pos_table_lookup("νὰ", "SCONJ") is None


def test_monotonic_sconj_is_unchanged(table_model):
    # The combined table answers monotonic input by the requested tag only.
    model = table_model([], {}, {"να": {"CCONJ": "να"}})
    assert model._pos_table_lookup("να", "SCONJ") is None


# ---- #121 ----------------------------------------------------------------

CONTRACTIONS = ["στὸ", "στὸν", "στὴ", "στὴν", "στὰ", "στοῦ", "στῆς", "στῶν",
                "στοὺς", "στὶς", "Στὸ", "ΣΤῸ"]


@pytest.mark.parametrize("word", CONTRACTIONS)
def test_contraction_gives_article_and_preposition(table_model, word):
    model = table_model([], {}, {})
    assert model._vernacular_contraction_pos_lemma(word, "DET") == "ο"
    assert model._vernacular_contraction_pos_lemma(word, "ADP") == "σε"
    decomposed = unicodedata.normalize("NFD", word)
    assert model._vernacular_contraction_pos_lemma(decomposed, "DET") == "ο"


@pytest.mark.parametrize("word,upos,lang", [
    ("στὸν", "ADJ", "all"),    # Ancient readings stand under other tags
    ("στὴν", "VERB", "all"),
    ("στὸν", "NOUN", "all"),
    ("στὸ", "DET", "grc"),     # lang="grc" reads Ancient Greek only
    ("στο", "DET", "all"),     # monotonic: the regular closed-class path
    ("στόν", "DET", "all"),    # acute only: no polytonic evidence
    ("στῷ", "DET", "all"),     # iota subscript: no contraction
    ("ἀστὸν", "DET", "all"),
    ("στὸλ", "DET", "all"),
    ("τὸν", "DET", "all"),
])
def test_contraction_rule_scope(table_model, word, upos, lang):
    model = table_model([], {}, {}, lang=lang)
    assert model._vernacular_contraction_pos_lemma(word, upos) is None


def test_contraction_is_answered_before_the_candidates(table_model):
    model = table_model([("ἀστός", "grc")], {"στόν": {"DET": "ἀστός"}}, {})
    model._resolve_articles = True
    assert both(model, "στὸν", "DET") == "ο"
    assert both(model, "στὸν", "ADP") == "σε"


# ---- shipped answers -------------------------------------------------------

@pytest.fixture(scope="module")
def shipped_models():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    return {conv: core.Dilemma(lang="all", convention=conv, resolve_articles=True)
            for conv in ("wiktionary", "triantafyllidis")}


@pytest.mark.parametrize("conv,forms,tags,expected", [
    # #119 (the issue's table): the default convention already gave these.
    ("triantafyllidis", ["σεῖς", "κυροῦ", "σὰν", "νὰ"],
     ["PRON", "NOUN", "CCONJ", "CCONJ"], ["εσείς", "κυρός", "σαν", "νά"]),
    ("wiktionary", ["σεῖς", "κυροῦ", "σὰν", "νὰ"],
     ["PRON", "NOUN", "CCONJ", "CCONJ"], ["εσείς", "κυρός", "σαν", "να"]),
    # #120
    ("wiktionary", ["νὰ", "νὰ", "νὰ", "σὰν", "σὰν", "σὰ", "ὄντας"],
     ["SCONJ", "PART", "CCONJ", "SCONJ", "CCONJ", "SCONJ", "SCONJ"],
     ["να", "να", "να", "σαν", "σαν", "σα", "είμαι"]),
    # Ancient conjunctions keep their Ancient entries.
    ("wiktionary", ["ἐπειδὴ", "ὅτι"], ["SCONJ", "SCONJ"], ["ἐπεί", "ὅτι"]),
    # #121
    ("wiktionary", ["στὸ", "στὸν", "στὴν", "στὰ", "στοὺς", "στῆς"],
     ["DET"] * 6, ["ο"] * 6),
    ("wiktionary", ["στὸ", "στὸν", "στὴν", "στὰ", "στοὺς", "στῆς"],
     ["ADP"] * 6, ["σε"] * 6),
    ("triantafyllidis", ["στὸ", "στὸν", "στὴν", "στὰ", "στοὺς", "στῆς"],
     ["DET"] * 6, ["ο"] * 6),
    ("triantafyllidis", ["στὸ", "στὸν", "στοὺς"], ["ADP"] * 3, ["σε"] * 3),
    # Their Ancient readings stand under the other tags.
    ("wiktionary", ["στὸν", "στὴν"], ["ADJ", "VERB"], ["ἀστός", "ἵστημι"]),
])
def test_shipped_vernacular_pos_answers(shipped_models, conv, forms, tags, expected):
    model = shipped_models[conv]
    assert model.lemmatize_batch_pos(forms, tags, guess=False) == expected
    assert [model.lemmatize_pos(f, t, guess=False)
            for f, t in zip(forms, tags)] == expected
