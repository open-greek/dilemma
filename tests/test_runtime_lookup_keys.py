"""Runtime lookup keys: marked double rho.

#141, #134: a double rho printed with breathings (ῤῥ) is read as plain ρρ
(ἀπεῤῥάγησαν, Βοῤῥᾶ PROPN).
"""

import unicodedata

import pytest

from dilemma import core
from dilemma.core import to_plain_double_rho


@pytest.fixture
def model_from(tmp_path, monkeypatch):
    """A model with a small lookup table, without shipped data."""
    monkeypatch.setattr(core.Dilemma, "_load_lookups", lambda *a, **k: None)
    monkeypatch.setattr(core.Dilemma, "_check_backend", lambda *a: None)
    for name in ("AG_POS_LOOKUP_PATH", "GLAUX_POS_LOOKUP_PATH",
                 "TREEBANK_POS_LOOKUP_PATH", "MG_POS_LOOKUP_PATH"):
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(core, name, path)
    monkeypatch.setattr(core, "POS_CELL_CORRECTIONS_PATH", tmp_path / "none.json")

    def make(lookup, *, lang="grc", headwords=(), vernacular=()):
        model = core.Dilemma(lang="grc", skip_pos=True)
        model.lang = lang
        model._lookup = dict(lookup)
        model._ag_lookup = {form: lemma for form, lemma in lookup.items()
                            if lemma not in vernacular}
        model._mg_lookup = {}
        model._pos_ag_lookup = {}
        model._pos_lookup = {}
        model._pos_rejected_targets = frozenset()
        model._headword_sets["lsj"] = set(headwords)
        model._ancient_lexical_heads = set()
        model._vernacular_lexical_heads = set(vernacular)
        return model

    return make


@pytest.fixture(scope="module")
def shipped_models():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    return {lang: core.Dilemma(lang=lang, resolve_articles=True)
            for lang in ("all", "grc")}


# ------------------------------------------------- marked double rho

@pytest.mark.parametrize("printed, plain", [
    ("ἀπεῤῥάγησαν", "ἀπερράγησαν"),
    ("Βοῤῥᾶ", "Βορρᾶ"),
    ("παῤῥησίας", "παρρησίας"),
    ("ἄῤῥητος", "ἄρρητος"),
    # Half-marked pairs, as the same editions and their OCR have them.
    ("παρῥησία", "παρρησία"),
    ("παῤρησία", "παρρησία"),
    # Two pairs in one word.
    ("διαῤῥήξας καταῤῥεῖ", "διαρρήξας καταρρεῖ"),
])
def test_marked_double_rho_becomes_plain(printed, plain):
    assert to_plain_double_rho(printed) == plain
    decomposed = unicodedata.normalize("NFD", printed)
    assert unicodedata.normalize(
        "NFC", to_plain_double_rho(decomposed)) == plain


@pytest.mark.parametrize("word", [
    "ῥήτωρ", "Ῥόδος", "ἄρρητος", "πυρρός", "γάῤ", "ῥαῤ", "ἀρετή", "ἔρως",
    "ὁ", "ἡ", "abc", ""])
def test_other_rhos_are_left_alone(word):
    assert to_plain_double_rho(word) == word


def test_marked_double_rho_reads_the_plain_row(model_from):
    model = model_from({"ἀπερράγησαν": "ἀπορρήγνυμι", "παρρησίας": "παρρησία"})
    assert model._lookup_word("ἀπεῤῥάγησαν") is None
    assert model._lookup_word(
        to_plain_double_rho("ἀπεῤῥάγησαν")) == "ἀπορρήγνυμι"
    readings = model._lemmatize_verbose_impl("παῤῥησίας", guess=False)
    assert [(r.lemma, r.via) for r in readings] == [("παρρησία", "exact")]


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_shipped_marked_double_rho(shipped_models, lang):
    model = shipped_models[lang]
    forms = ["ἀπεῤῥάγησαν", "Βοῤῥᾶ", "παῤῥησίας", "ἀῤῥήτου"]
    expected = ["ἀπορρήγνυμι", "Βορέας", "παρρησία", "ἄρρητος"]
    plain = [to_plain_double_rho(form) for form in forms]
    assert model.lemmatize_batch(plain, guess=False) == expected
    assert model.lemmatize_batch(forms, guess=False) == expected
    assert [model.lemmatize(f, guess=False) for f in forms] == expected
    tags = ["VERB", "PROPN", "NOUN", "ADJ"]
    assert model.lemmatize_batch_pos(forms, tags, guess=False) == expected
    assert [model.lemmatize_pos(f, t, guess=False)
            for f, t in zip(forms, tags)] == expected
    assert [model.lemmatize_verbose(f, guess=False)[0].lemma
            for f in forms] == expected
    # #134: the tag no longer reads the Modern Greek table.
    assert model.lemmatize_pos("Βοῤῥᾶ", "PROPN", guess=False) == (
        model.lemmatize_pos("Βορρᾶ", "PROPN", guess=False))


def test_shipped_marked_double_rho_matches_the_plain_spelling_in_modern_mode():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    model = core.Dilemma(lang="el")
    for printed, tag in [("Βοῤῥᾶ", "PROPN"), ("παῤῥησίας", "NOUN")]:
        plain = to_plain_double_rho(printed)
        assert model.lemmatize(printed, guess=False) == model.lemmatize(
            plain, guess=False)
        assert model.lemmatize_pos(printed, tag, guess=False) == (
            model.lemmatize_pos(plain, tag, guess=False))
