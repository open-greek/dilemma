"""Breathings on loose-key lemmas and the polytonic Modern POS rule.

#107: an Ancient POS-table target whose initial rho lacks its breathing
(ρῆσις) is a corpus spelling and does not replace the lemma.
#109: a lemma found only through the lookup's monotonic or accent-stripped
key may not drop the breathing the form has (εὐσεβεστάτῃ -> εὐσεβής).
#123: the polytonic Modern POS rule counts syllables with synizesis
(προῖὸν is no monosyllable), returns the lowercase Ancient headword a
capitalized Ancient name stands for (εὐήνεμός ADJ), and keeps a lowercase
Ancient reading for a capitalized PROPN form (Ὀρᾷς PROPN -> ὁράω).
"""

import pytest

from dilemma import core
from dilemma.core import (_has_bare_initial_rho, _initial_breathing,
                          _with_initial_breathing)


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

    def make(lookup, *, lang="grc", ancient_pos=None, modern_pos=None,
             headwords=(), heads=()):
        core.AG_POS_LOOKUP_PATH.write_text("{}", encoding="utf-8")
        model = core.Dilemma(lang="grc", skip_pos=True)
        model.lang = lang
        model._lookup = dict(lookup)
        model._ag_lookup = dict(lookup)
        ancient_pos = ancient_pos or {}
        model._pos_ag_lookup = ancient_pos
        combined = {form: dict(by_pos) for form, by_pos in ancient_pos.items()}
        for form, by_pos in (modern_pos or {}).items():
            for upos, lemma in by_pos.items():
                combined.setdefault(form, {}).setdefault(upos, lemma)
        model._pos_lookup = combined
        model._pos_rejected_targets = frozenset()
        model._headword_sets["lsj"] = set(headwords)
        model._ancient_lexical_heads = set(heads)
        return model

    return make


@pytest.fixture(scope="module")
def shipped_models():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    return {lang: core.Dilemma(lang=lang, resolve_articles=True)
            for lang in ("all", "grc")}


# ---------------------------------------------------------------- #107

def test_bare_initial_rho():
    assert _has_bare_initial_rho("ρῆσις")
    assert _has_bare_initial_rho("Ραχίλ")
    assert not _has_bare_initial_rho("ῥῆσις")
    assert not _has_bare_initial_rho("βίος")
    assert not _has_bare_initial_rho(None)


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_ancient_pos_target_without_rho_breathing_is_skipped(model_from, lang):
    model = model_from({}, lang=lang, ancient_pos={
        "ρήσεις": {"NOUN": "ρῆσις", "VERB": "ρέω"},
        "ῥῆσιν": {"NOUN": "ῥῆσις"}})
    assert model._pos_table_lookup("ῥήσεις", "NOUN") is None
    assert model._pos_table_lookup("ῥήσεις", "VERB") is None
    assert model._pos_table_lookup("ῥῆσιν", "NOUN") == "ῥῆσις"


def test_modern_pos_entry_with_bare_rho_still_counts(model_from):
    model = model_from({}, lang="all", modern_pos={"ρα": {"INTJ": "ρε"}})
    model._lemmatize_verbose_impl = lambda word, **kwargs: []
    assert model._pos_table_lookup("ῥα", "INTJ") == "ρε"


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_shipped_rho_targets_keep_the_breathing(shipped_models, lang):
    model = shipped_models[lang]
    forms = ["ῥήσεις", "ῥήσεις", "ῥυήσονται", "ῥοίζῳ", "ῥομφαῖαν"]
    tags = ["NOUN", "VERB", "VERB", "ADJ", "NOUN"]
    expected = ["ῥῆσις", "ῥῆσις", "ῥέω", "ῥοῖζος", "ῥομφαία"]
    assert model.lemmatize_batch_pos(forms, tags, guess=False) == expected
    assert [model.lemmatize_pos(f, t, guess=False)
            for f, t in zip(forms, tags)] == expected
    # Controls.
    assert model.lemmatize_batch_pos(
        ["ῥῆσιν", "ῥομφαίαν", "ῥεῖ", "ῥοίζῳ"], ["NOUN", "NOUN", "VERB", "NOUN"],
        guess=False) == ["ῥῆσις", "ῥομφαία", "ῥέω", "ῥοῖζος"]


# ---------------------------------------------------------------- #109

def test_initial_breathing():
    assert _initial_breathing("εὐσεβής") == "\u0313"
    assert _initial_breathing("Ἑλλήνων") == "\u0314"
    assert _initial_breathing("ῥῆσις") == "\u0314"
    assert _initial_breathing("ευσεβής") == ""
    assert _initial_breathing("ρῆσις") == ""
    assert _initial_breathing("βίος") is None


def test_breathing_goes_on_the_diphthong_second_vowel():
    assert _with_initial_breathing("ευσύνοπτος", "\u0313") == "εὐσύνοπτος"
    assert _with_initial_breathing("αϊδής", "\u0314") == "ἁϊδής"
    assert _with_initial_breathing("Ραχίλ", "\u0314") == "Ῥαχίλ"
    assert _with_initial_breathing("οφφίκιον", "\u0313") == "ὀφφίκιον"


def test_restore_breathing_finds_the_ancient_headword(model_from):
    model = model_from({}, headwords={
        "ἀνεξάντλητος", "εὐσύνοπτος", "ἑδραῖος", "εὐσεβής"})
    restore = model._restore_breathing
    assert restore("ἀνεξάντλητον", "ανεξάντλητος") == "ἀνεξάντλητος"
    assert restore("εὐσύνοπτον", "ευσύνοπτος") == "εὐσύνοπτος"
    # The other breathing, with the acute turned circumflex.
    assert restore("ἐδραία", "εδραίος") == "ἑδραῖος"
    # No Ancient lemma with those letters.
    assert restore("ἀστοχήσει", "αστοχώ") is None
    # Lemmas that keep a breathing and forms without one are untouched.
    assert restore("εὐσεβεστάτῃ", "εὐσεβής") == "εὐσεβής"
    assert restore("ευσεβεστατη", "ευσεβής") == "ευσεβής"
    assert restore("βίου", "βίος") == "βίος"


def test_restore_breathing_accepts_an_ancient_self_map(model_from):
    model = model_from({"Ἀββακούμ": "Ἀββακούμ"})
    assert model._restore_breathing("ἀββακοὺμ", "Αββακούμ") == "Ἀββακούμ"


def test_restore_breathing_repairs_a_breathingless_ancient_head(model_from):
    # Sophocles' lexicon writes its omicron heads without breathings.
    model = model_from({}, heads={"οφφίκιον"})
    assert model._restore_breathing("ὀφφικίου", "οφφίκιον") == "ὀφφίκιον"


def test_restore_breathing_keeps_a_modern_name(model_from):
    model = model_from({})
    assert model._restore_breathing("Ῥαχὶλ", "Ραχίλ") == "Ραχίλ"


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_loose_lookup_key_keeps_the_breathing(model_from, lang):
    model = model_from({"ευσεβεστατη": "ευσεβής", "αστοχήσει": "αστοχώ"},
                       lang=lang, headwords={"εὐσεβής"})
    model._mg_lookup = {}
    assert model._lookup_word("εὐσεβεστάτῃ") == "εὐσεβής"
    assert model._lookup_word("ἀστοχήσει") is None
    # Monotonic input reads the Modern rows as before.
    assert model._lookup_word("αστοχήσει") == "αστοχώ"


def test_modern_mode_keeps_the_modern_row(model_from):
    model = model_from({"αστοχήσει": "αστοχώ"}, lang="el")
    model._mg_lookup = {}
    assert model._lookup_word("ἀστοχήσει") == "αστοχώ"


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_shipped_loose_keys_keep_the_breathing(shipped_models, lang):
    model = shipped_models[lang]
    forms = ["εὐσεβεστάτῃ", "ἀστοχήσει", "ἀνεξάντλητον", "ἀνεξαντλήτου", "ὀφφικίου"]
    assert model.lemmatize_batch(forms, guess=False) == [
        "εὐσεβής", None, "ἀνεξάντλητος", "ἀνεξάντλητος", "ὀφφίκιον"]
    assert [model.lemmatize(f, guess=False) for f in forms] == [
        "εὐσεβής", None, "ἀνεξάντλητος", "ἀνεξάντλητος", "ὀφφίκιον"]
    tagged = model.lemmatize_batch_pos(forms[:4], ["ADJ", "VERB", "ADJ", "ADJ"],
                                       guess=False)
    assert tagged == ["εὐσεβής", None, "ἀνεξάντλητος", "ἀνεξάντλητος"]
    assert model.lemmatize_batch(
        ["ἀνευροῦς", "ἐδραία", "εὐσεβεστάτη"], guess=False) == [
        "ἄνευρος", "ἑδραῖος", "εὐσεβής"]
    assert model.lemmatize_pos("ἀστοχήσας", "VERB", guess=False) == "ἀστοχέω"
    # A name only Modern Greek Wiktionary lists keeps its lemma.
    assert model.lemmatize("Ῥαχὶλ", guess=False) == "Ραχίλ"


def test_shipped_modern_mode_is_unchanged():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    assert core.Dilemma(lang="el").lemmatize("ἀστοχήσει") == "αστοχώ"
