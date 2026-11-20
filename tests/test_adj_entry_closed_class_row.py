"""An ADJ entry against a row that reads a conjunction (#148)."""
import json

import pytest

from dilemma import core


@pytest.fixture
def table_model(tmp_path, monkeypatch):
    monkeypatch.setattr(core.Dilemma, "_load_lookups", lambda *a, **k: None)
    monkeypatch.setattr(core.Dilemma, "_check_backend", lambda *a: None)
    for name in ("AG_POS_LOOKUP_PATH", "GLAUX_POS_LOOKUP_PATH",
                 "TREEBANK_POS_LOOKUP_PATH", "MG_POS_LOOKUP_PATH"):
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(core, name, path)
    monkeypatch.setattr(core, "POS_CELL_CORRECTIONS_PATH", tmp_path / "none.json")
    indeclinables = tmp_path / "indeclinables.json"
    indeclinables.write_text(json.dumps({"ἀλλά": "conjunction", "ἐπεί": "conjunction",
                                         "αὖθις": "adverb"}), encoding="utf-8")
    monkeypatch.setattr(core, "LSJ9_INDECLINABLES_PATH", indeclinables)
    monkeypatch.setattr(core.Dilemma, "_is_ancient_headword", lambda self, lemma: True)

    def make(candidates, tables, *, lang="grc"):
        model = core.Dilemma(lang=lang, skip_pos=True)
        model._pos_ag_lookup = {form: dict(by_pos) for form, by_pos in tables.items()}
        model._pos_lookup = {form: dict(by_pos) for form, by_pos in tables.items()}
        model._pos_rejected_targets = frozenset()
        made = [core.LemmaCandidate(lemma=lemma, source="lookup", via="exact")
                for lemma in candidates]
        model._lemmatize_verbose_impl = lambda word, **kwargs: list(made)
        model._article_pos_lemma = lambda word, upos: None
        model._apply_convention = lambda lemma: lemma
        first = made[0].lemma if made else None
        model.lemmatize = lambda word, **kwargs: first
        model.lemmatize_batch = lambda words, **kwargs: [first] * len(words)
        return model
    return make


def both(model, word, upos):
    scalar = model.lemmatize_pos(word, upos, guess=False)
    assert model.lemmatize_batch_pos([word], [upos], guess=False) == [scalar]
    return scalar


def test_adj_entry_stands_against_a_conjunction_row(table_model):
    model = table_model(["ἀλλά"], {"αλλη": {"ADJ": "ἄλλος"}})
    assert both(model, "ἀλλή", "ADJ") == "ἄλλος"


def test_adj_entry_still_yields_to_an_adjective_row(table_model):
    # The #130 rule: a stripped-key entry does not replace a listed head.
    model = table_model(["οὗτος"], {"ταυτην": {"ADJ": "ταὐτός"}})
    assert both(model, "ταύτην", "ADJ") == "οὗτος"


def test_an_adverb_row_is_not_a_closed_class_row(table_model):
    model = table_model(["αὖθις"], {"αυθι": {"ADJ": "αὐθιγενής"}})
    assert both(model, "αὖθι", "ADJ") == "αὖθις"
