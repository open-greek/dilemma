"""Oblique -μενος participles resolve through their nominative (issue #113)."""

import json

import pytest

from dilemma import core


@pytest.fixture
def fallback_model(tmp_path, monkeypatch):
    """Use small source tables without loading the shipped lookup or model."""
    monkeypatch.setattr(core.Dilemma, "_load_lookups", lambda *a, **k: None)
    monkeypatch.setattr(core.Dilemma, "_check_backend", lambda *a: None)
    for name in ("AG_POS_LOOKUP_PATH", "GLAUX_POS_LOOKUP_PATH",
                 "TREEBANK_POS_LOOKUP_PATH", "MG_POS_LOOKUP_PATH"):
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(core, name, path)

    def make(lookup, readings):
        core.AG_POS_LOOKUP_PATH.write_text(
            json.dumps(readings, ensure_ascii=False), encoding="utf-8")
        model = core.Dilemma(lang="grc", skip_pos=True)
        model._lookup = lookup
        model._ag_lookup = lookup
        return model

    return make


_ERXOMAI = {"ἐλευσόμενος": "ἔρχομαι", "ἔρχομαι": "ἔρχομαι",
            # λευσομένῳ is a genuine future participle of λεύσσω.
            "λευσομένῳ": "λεύσσω", "λευσομενω": "λεύσσω", "λεύσσω": "λεύσσω"}
_ERXOMAI_READINGS = {"ἐλευσόμενος": {"VERB": "ἔρχομαι"},
                     "λευσομένῳ": {"VERB": "λεύσσω"}}


@pytest.mark.parametrize("word", [
    "ἐλευσομένῳ", "ἐλευσομένου", "ἐλευσομένοις", "ἐλευσόμενοι",
    "ἐλευσομένας", "ἐλευσόμενα", "ἐλευσόμεναι", "ἐλευσομέναις",
])
def test_oblique_participle_takes_the_nominative_lemma(fallback_model, word):
    model = fallback_model(dict(_ERXOMAI), _ERXOMAI_READINGS)
    assert model._participle_nominative_lemma(word) == "ἔρχομαι"
    assert model.lemmatize(word, guess=False) == "ἔρχομαι"
    assert model.lemmatize_batch([word], guess=False) == ["ἔρχομαι"]
    candidates = model.lemmatize_verbose(word, guess=False)
    assert [(c.lemma, c.via) for c in candidates] == [
        ("ἔρχομαι", "participle_nominative")]


def test_participle_carries_no_augment(fallback_model):
    # Without the nominative row, ἐλευσομένῳ must not become λευσομένῳ.
    lookup = dict(_ERXOMAI)
    del lookup["ἐλευσόμενος"]
    model = fallback_model(lookup, _ERXOMAI_READINGS)
    assert model._participle_nominative_lemma("ἐλευσομένῳ") is None
    assert model._strip_verb_morphology("ἐλευσομένῳ") is None
    assert model.lemmatize("ἐλευσομένῳ", guess=False) is None


def test_perfect_participle_uses_the_oxytone_nominative(fallback_model):
    model = fallback_model({"ἐσταλμένος": "στέλλω", "στέλλω": "στέλλω"},
                           {"ἐσταλμένος": {"VERB": "στέλλω"}})
    assert model._participle_nominative_lemma("ἐσταλμένοις") == "στέλλω"


def test_nominative_must_be_a_verb_reading(fallback_model):
    # A self-mapped adjectival participle is not evidence for a verb.
    model = fallback_model({"ἡγούμενος": "ἡγούμενος"},
                           {"ἡγούμενος": {"NOUN": "ἡγούμενος"}})
    assert model._participle_nominative_lemma("ἡγουμένοις") is None


@pytest.mark.parametrize("word", ["ἐλευσ", "μένου", "λυόμενος", "λόγου"])
def test_non_oblique_or_short_forms_are_ignored(fallback_model, word):
    model = fallback_model(dict(_ERXOMAI), _ERXOMAI_READINGS)
    assert model._participle_nominative_lemma(word) is None


def test_temporal_augment_lemma_must_begin_with_the_restored_vowel(fallback_model):
    # ει -> ε gives ἐκόνισεν, itself the augmented aorist of κονίω.
    model = fallback_model({"εκονισεν": "κονίω", "κονίω": "κονίω",
                            "εχον": "ἔχω", "ἔχω": "ἔχω"},
                           {"ἐκόνισεν": {"VERB": "κονίω"},
                            "ἔχον": {"VERB": "ἔχω"}})
    assert model._strip_verb_morphology("εἰκόνισεν") is None
    assert model._strip_verb_morphology("εἶχον") == "ἔχω"


@pytest.fixture(scope="module", params=["all", "grc"])
def shipped_model(request):
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    return core.Dilemma(lang=request.param, resolve_articles=True)


_ISSUE_FORMS = ["ἐλευσομένῳ", "ἐλευσομένου", "ἐλευσομένοις", "ἐλευσόμενοι",
                "ἐλευσομένας", "ἐλευσόμενα", "ἐλευσόμεναι", "ἐλευσομέναις"]


def test_shipped_issue_participles(shipped_model):
    model = shipped_model
    assert model.lemmatize_batch_pos(_ISSUE_FORMS, ["VERB"] * len(_ISSUE_FORMS),
                                     guess=False) == ["ἔρχομαι"] * len(_ISSUE_FORMS)
    assert model.lemmatize_batch(_ISSUE_FORMS, guess=False) == ["ἔρχομαι"] * len(_ISSUE_FORMS)
    assert [model.lemmatize(f, guess=False) for f in _ISSUE_FORMS] == [
        "ἔρχομαι"] * len(_ISSUE_FORMS)
    assert model.lemmatize_verbose("ἐλευσομένῳ", guess=False)[0].via == "participle_nominative"
    assert model.lemmatize_pos("ἐλευσόμενος", "VERB", guess=False) == "ἔρχομαι"
    assert model.lemmatize_pos("ἐλεύσεται", "VERB", guess=False) == "ἔρχομαι"


def test_shipped_augment_strip_is_narrowed(shipped_model):
    model = shipped_model
    assert model._strip_verb_morphology("εἰκόνισεν") != "κονίω"
    assert model.lemmatize("εἰκόνισεν", guess=False) != "κονίω"
    for word, lemma in [("ἐποίησεν", "ποιέω"), ("ἔλεγον", "λέγω"),
                        ("ἠγόραζον", "ἀγοράζω"), ("εἶχον", "ἔχω")]:
        assert model._strip_verb_morphology(word) == lemma
    for word, lemma in [("ἐσταλμένοις", "στέλλω"), ("ἐζητημένου", "ζητέω"),
                        ("πεπλησμένην", "πίμπλημι"), ("ὠρθωμένοις", "ὀρθόω")]:
        assert model.lemmatize(word, guess=False) == lemma
