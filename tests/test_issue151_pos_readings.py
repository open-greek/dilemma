"""Readings an explicit POS tag or a sibling cell recovers (#151)."""

import pytest

import build_lookup_db as builder
from dilemma import core


@pytest.fixture
def model_from(tmp_path, monkeypatch):
    """A model with small lookup and POS tables, without shipped data."""
    monkeypatch.setattr(core.Dilemma, "_load_lookups", lambda *a, **k: None)
    monkeypatch.setattr(core.Dilemma, "_check_backend", lambda *a: None)
    for name in ("AG_POS_LOOKUP_PATH", "GLAUX_POS_LOOKUP_PATH",
                 "TREEBANK_POS_LOOKUP_PATH", "MG_POS_LOOKUP_PATH"):
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(core, name, path)
    monkeypatch.setattr(core, "POS_CELL_CORRECTIONS_PATH", tmp_path / "none.json")

    def make(lookup, *, lang="grc", pos=None, headwords=(), vernacular=()):
        model = core.Dilemma(lang="grc", skip_pos=True)
        model.lang = lang
        model._lookup = dict(lookup)
        model._ag_lookup = dict(lookup)
        model._mg_lookup = {}
        model._pos_ag_lookup = dict(pos or {})
        model._pos_lookup = dict(pos or {})
        model._pos_rejected_targets = frozenset()
        model._headword_sets["lsj"] = set(headwords)
        model._ancient_lexical_heads = set(headwords)
        model._vernacular_lexical_heads = set(vernacular)
        return model

    return make


def tagged(model, word, upos):
    scalar = model.lemmatize_pos(word, upos, guess=False)
    assert model.lemmatize_batch_pos([word], [upos], guess=False) == [scalar]
    return scalar


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_tag_chooses_between_the_readings_the_tables_split(model_from, lang):
    model = model_from(
        {"ἀξίαν": "ἄξιος", "Ἄξιος": "Ἄξιος", "ἱερὰ": "ἱερόν", "Ἱερόν": "Ἱερόν"},
        lang=lang,
        pos={"ἀξίαν": {"ADJ": "ἄξιος", "NOUN": "ἀξία"},
             "ἱερὰ": {"ADJ": "ἱερός", "NOUN": "ἱερόν"}})
    assert tagged(model, "ἀξίαν", "NOUN") == "ἀξία"
    assert tagged(model, "ἀξίαν", "ADJ") == "ἄξιος"
    assert tagged(model, "ἱερὰ", "ADJ") == "ἱερός"
    assert tagged(model, "ἱερὰ", "NOUN") == "ἱερόν"


def test_entry_that_only_disagrees_with_the_row_does_not_decide(model_from):
    model = model_from({"ἐλάττονος": "μικρός", "Μικρός": "Μικρός"},
                       pos={"ἐλάττονος": {"ADJ": "ἐλαχύς"}})
    assert tagged(model, "ἐλάττονος", "ADJ") == "μικρός"


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_modern_contract_head_spells_the_ancient_verb(model_from, lang):
    model = model_from({"αστοχήσει": "αστοχώ"}, lang=lang,
                       headwords={"ἀστοχέω"}, vernacular={"αστοχώ"})
    assert model._lookup_word("ἀστοχήσει") == "ἀστοχέω"


def test_two_ancient_contract_verbs_are_not_chosen_between(model_from):
    model = model_from({"αστοχήσει": "αστοχώ"}, lang="all",
                       headwords={"ἀστοχέω", "ἀστοχάω"}, vernacular={"αστοχώ"})
    assert model._lookup_word("ἀστοχήσει") == "αστοχώ"


def participle_model(model_from, cells):
    model = model_from({"ἱερώμενος": "ἱεράομαι", "ἱερωμένος": "ἱερόω"})
    model._fallback_readings = {}
    model._fallback_exact_readings = {
        form: {"VERB": {lemma}} for form, lemma in cells.items()}
    model._fallback_verb_lemmas = {"ἱεράομαι", "ἱερόω"}
    return model


def test_competing_participles_go_to_the_verb_with_more_cells(model_from):
    model = participle_model(model_from, {
        "ἱερωμένη": "ἱερόω", "ἱερωμένοι": "ἱερόω", "ἱερωμένου": "ἱεράομαι"})
    assert model._participle_nominative_lemma("ἱερωμέναις") == "ἱερόω"


def test_competing_participles_tied_keep_the_first_nominative(model_from):
    model = participle_model(model_from, {
        "ἱερωμένη": "ἱερόω", "ἱερωμένου": "ἱεράομαι"})
    assert model._participle_nominative_lemma("ἱερωμέναις") == "ἱεράομαι"


def test_aorist_participle_reads_the_verb_of_its_feminine(model_from):
    model = model_from({"εὐτονήσασα": "εὐτονέω", "κεράσασα": "κέρας"})
    assert tagged(model, "εὐτονήσας", "VERB") == "εὐτονέω"
    assert model.lemmatize("εὐτονήσας", guess=False) == "εὐτονέω"
    assert model._aorist_participle_lemma("κεράσας") is None


def test_dative_without_its_iota_reads_the_nominal(model_from):
    model = model_from({"μοναχῶ": "μοναχόω", "μοναχῷ": "μοναχός"})
    assert tagged(model, "μοναχῶ", "ADJ") == "μοναχός"
    assert tagged(model, "μοναχῶ", "NOUN") == "μοναχός"
    assert tagged(model, "μοναχῶ", "VERB") == "μοναχόω"


def test_oxytone_dative_plural_needs_a_sibling_cell(model_from):
    model = model_from({"ποδηγοῖς": "ποδηγέω", "ποδηγῷ": "ποδηγός"},
                       headwords={"ποδηγός"})
    assert tagged(model, "ποδηγοῖς", "NOUN") == "ποδηγός"
    assert tagged(model, "ποδηγοῖς", "VERB") == "ποδηγέω"
    alone = model_from({"ποδηγοῖς": "ποδηγέω"}, headwords={"ποδηγός"})
    assert tagged(alone, "ποδηγοῖς", "NOUN") == "ποδηγέω"


def test_lowercase_name_reads_its_capitalized_row(model_from):
    model = model_from({"Θεοφάνην": "Θεοφάνης"})
    assert tagged(model, "θεοφάνην", "NOUN") == "Θεοφάνης"
    assert tagged(model, "θεοφάνην", "VERB") is None


def test_builder_gives_the_ancient_view_the_verb_a_modern_row_hides():
    combined = {"βάλει": "βάζω", "βαλει": "βάλλω", "σπίτια": "σπίτι",
                "αγαπώ": "αγαπώ", "Βάλει": "Βάλης", "λύει": "λύνω"}
    ag = {"βαλει": "βάλλω", "λύει": "λύω", "λυει": "λύω", "σπιτια": "σπιτίον"}
    src_map = {form: "el" for form in combined}
    src_map["βαλει"] = "grc"
    assert builder.ancient_verb_cells_under_modern_rows(
        combined, ag, src_map) == {"βάλει": "βαλει"}
