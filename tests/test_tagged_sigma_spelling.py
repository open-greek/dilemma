"""Tagged lookups read the POS tables under the spelling lemmatize() reads (#154).

lemmatize() turns a lunate sigma into σ/ς and normalizes to NFC before it
reads the lookup; lemmatize_pos() and lemmatize_batch_pos() read the POS
tables under the same spelling, so Δῖόϲ PROPN finds the cell of Δῖός.
"""
import unicodedata

import pytest

from dilemma import core


@pytest.fixture
def table_model(tmp_path, monkeypatch):
    """A model with given candidates and POS tables, and the real selection."""
    monkeypatch.setattr(core.Dilemma, "_load_lookups", lambda *a, **k: None)
    monkeypatch.setattr(core.Dilemma, "_check_backend", lambda *a: None)
    for name in ("AG_POS_LOOKUP_PATH", "GLAUX_POS_LOOKUP_PATH",
                 "TREEBANK_POS_LOOKUP_PATH", "MG_POS_LOOKUP_PATH"):
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(core, name, path)
    monkeypatch.setattr(core, "POS_CELL_CORRECTIONS_PATH", tmp_path / "none.json")

    def make(candidates, tables, *, lang="grc"):
        model = core.Dilemma(lang=lang, skip_pos=True)
        model._pos_ag_lookup = {form: dict(by_pos) for form, by_pos in tables.items()}
        model._pos_lookup = {form: dict(by_pos) for form, by_pos in tables.items()}
        model._pos_rejected_targets = frozenset()
        made = [core.LemmaCandidate(lemma=lemma, source=source, via=via)
                for lemma, source, via in candidates]
        asked = []

        def verbose(word, **kwargs):
            asked.append(word)
            return list(made)
        model._lemmatize_verbose_impl = verbose
        model._article_pos_lemma = lambda word, upos: None
        model._apply_convention = lambda lemma: lemma
        first = made[0].lemma if made else None
        model.lemmatize = lambda word, **kwargs: first
        model.lemmatize_batch = lambda words, **kwargs: [first] * len(words)
        model.asked = asked
        return model

    return make


def both(model, word, upos):
    scalar = model.lemmatize_pos(word, upos, guess=False)
    batch = model.lemmatize_batch_pos([word], [upos], guess=False)
    assert batch == [scalar], (word, upos, scalar, batch)
    return scalar


DIOS = [("δῖος", "lookup", "exact"), ("Δῖος", "lookup", "exact+case_alt")]
TABLES = {"Δῖός": {"PROPN": "Ζεύς"}}


@pytest.mark.parametrize("lang", ["grc", "all"])
@pytest.mark.parametrize("word", [
    "Δῖός",                                  # standard sigma
    "Δῖόϲ",                                  # lunate sigma
    "Δῖόσ",                                  # medial sigma at the end
    unicodedata.normalize("NFD", "Δῖόϲ"),    # decomposed, lunate
])
def test_lunate_and_decomposed_spellings_find_the_pos_cell(table_model, lang, word):
    model = table_model(DIOS, TABLES, lang=lang)
    assert both(model, word, "PROPN") == "Ζεύς"


def test_capital_lunate_sigma(table_model):
    model = table_model([("Σωκράτης", "lookup", "exact")],
                        {"Σωκράτης": {"PROPN": "Σωκράτης"},
                         "Σωκράτους": {"PROPN": "Σωκράτης"}})
    assert both(model, "Ϲωκράτουϲ", "PROPN") == "Σωκράτης"


def test_candidates_are_built_from_the_standard_spelling(table_model):
    model = table_model(DIOS, TABLES)
    model.lemmatize_pos("Δῖόϲ", "PROPN", guess=False)
    assert model.asked and all(word == "Δῖός" for word in model.asked)


@pytest.mark.parametrize("token", ["...", "12", "abc", "e\u0301"])
def test_tokens_without_greek_letters_pass_unchanged(token):
    assert core.Dilemma._tagged_spelling(token) == token
