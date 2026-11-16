"""Runtime lookup keys: marked double rho, capitalized loose keys, respellings.

#141, #134: a double rho printed with breathings (ῤῥ) is read as plain ρρ
(ἀπεῤῥάγησαν, Βοῤῥᾶ PROPN).
#141: a capitalized word also tries its monotonic and accent-stripped keys
with the capital, where the generated paradigms put them, for a capitalized
Ancient head (Σαμαρείτις).
The Byzantine respelling fallback prefers a respelling that reads a row as
it is written over one that also needs its accents changed, whatever the
rule order (αναγνώστει).
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


# ---------------------------------------------- capitalized loose keys

def test_loose_keys_of_a_capitalized_word():
    lowercase = core.Dilemma._loose_keys
    capitalized = core.Dilemma._capitalized_loose_keys
    assert lowercase("Σαμαρείτις") == ["σαμαρείτις", "σαμαρειτις"]
    assert capitalized("Σαμαρείτις") == ["Σαμαρείτις", "Σαμαρειτις"]
    assert capitalized("σαμαρείτις") == []
    assert lowercase("Ἀθηνῶν", stripped=False) == ["αθηνών"]
    assert capitalized("Ἀθηνῶν", stripped=False) == ["Αθηνών"]
    assert capitalized("Ἀθηνῶν", monotonic=False) == ["Αθηνων"]


@pytest.mark.parametrize("lemma, expected", [
    ("Σαμαρεῖτις", True), ("Ἀρειομανίτης", True), ("Καισάριον", True),
    ("Ῥωμαΐς", True), ("Γώγ", True),
    # A Modern Greek name, a corpus lemma without accents, a letter name,
    # a lowercase head.
    ("Ολυμπιακούς", False), ("Ραφιδίν", False), ("Εζεκιος", False),
    ("Γ", False), ("σαμαρεῖτις", False),
])
def test_capitalized_ancient_head(lemma, expected):
    assert core.Dilemma._is_capitalized_ancient_head(lemma) is expected


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_capitalized_word_reads_its_capitalized_stripped_key(model_from, lang):
    model = model_from({"Σαμαρειτις": "Σαμαρεῖτις"}, lang=lang)
    assert model._lookup_word("Σαμαρείτις") == "Σαμαρεῖτις"
    assert model._lookup_word("Σαμαρεῖτις") == "Σαμαρεῖτις"
    # A lowercase word does not read a capitalized key.
    assert model._lookup_word("σαμαρείτις") is None
    readings = model._lemmatize_verbose_impl("Σαμαρείτις", guess=False)
    assert [(r.lemma, r.via) for r in readings] == [("Σαμαρεῖτις", "stripped")]


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_capitalized_key_reads_an_ancient_head_only(model_from, lang):
    model = model_from({"Εζεκιου": "Εζεκιος", "Ολυμπιακούς": "Ολυμπιακούς",
                        "Γο": "Γ", "Γωγ": "Γώγ"}, lang=lang)
    assert model._lookup_word("Ἑζεκίου") is None
    assert model._lookup_word("Ὀλυμπιακοὺς") is None
    assert model._lookup_word("Γὸ") is None
    assert model._lookup_word("Γῶγ") == "Γώγ"
    assert model._lemmatize_verbose_impl("Ἑζεκίου", guess=False) == []


def test_own_row_of_a_capitalized_word_is_not_filtered(model_from):
    # Λευκήν is its own monotonic key: the row is the word's own, whatever
    # its lemma looks like.
    model = model_from({"Λευκήν": "λευκός", "Καλήν": "Καλήν"})
    assert model._lookup_word("Λευκήν") == "λευκός"
    readings = model._lemmatize_verbose_impl("Λευκήν", guess=False)
    assert [(r.lemma, r.via) for r in readings] == [("λευκός", "exact")]


def test_modern_table_is_read_under_lowercase_keys_only(model_from):
    model = model_from({}, lang="el")
    model._mg_lookup = {"Σαμαρειτις": "Σαμαρεῖτις", "σαμαρείτισσα": "Σαμαρείτισσα"}
    assert model._lookup_word("Σαμαρείτις") is None
    assert model._lookup_word("Σαμαρείτισσα") == "Σαμαρείτισσα"
    model._lookup = {"Σαμαρειτις": "Σαμαρεῖτις"}
    assert model._lookup_word("Σαμαρείτις") == "Σαμαρεῖτις"


def test_lowercase_loose_keys_come_first(model_from):
    model = model_from({"σαμαρειτις": "σαμαρεῖτις", "Σαμαρειτις": "Σαμαρεῖτις",
                        "Νέαν": "Νέα", "νεαν": "νέος", "Νεαν": "Νέα"})
    assert model._lookup_word("Σαμαρείτις") == "σαμαρεῖτις"
    assert model._lookup_word("Νεάν") == "νέος"
    # The word's own rows still come before any loose key.
    assert model._lookup_word("Νέαν") == "Νέα"


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_shipped_capitalized_stripped_key(shipped_models, lang):
    model = shipped_models[lang]
    assert model.lemmatize("Σαμαρείτις", guess=False) == "Σαμαρεῖτις"
    assert model.lemmatize_batch(
        ["Σαμαρείτις", "Σαμαρεῖτις"], guess=False) == ["Σαμαρεῖτις"] * 2
    assert model.lemmatize_verbose(
        "Σαμαρείτις", guess=False)[0].lemma == "Σαμαρεῖτις"
    if lang == "grc":
        # Rows that are not capitalized Ancient heads stay unread.
        assert model.lemmatize_batch(
            ["Ὀλυμπιακοὺς", "Ἑξαπλᾶ"], guess=False) == [
            "Ὀλυμπιακός", "ἑξαπλόος"]


# ------------------------------------------------- respelling order

def test_lookup_steps_run_alone(model_from):
    model = model_from({"λόγου": "λόγος", "λογου": "λογάς", "ἀγαθῇ": "ἀγαθός",
                        "αγαθή": "αγαθός"})
    own, mono, stripped = (model._OWN_ROWS, model._MONOTONIC_KEY,
                           model._STRIPPED_KEY)
    assert model._lookup_word("λόγου", step=own) == "λόγος"
    assert model._lookup_word("λόγου", step=stripped) == "λογάς"
    assert model._lookup_word("λογοῦ", step=own) is None
    assert model._lookup_word("λογοῦ", step=mono) is None
    assert model._lookup_word("λογοῦ", step=stripped) == "λογάς"
    assert model._lookup_word("λογοῦ") == "λογάς"
    assert model._lookup_word("αγαθῆ", step=own) is None
    assert model._lookup_word("αγαθῆ", step=mono) == "αγαθός"
    assert model._lookup_word("αγαθῆ") == "αγαθός"


def test_respelling_with_its_own_row_beats_an_earlier_loose_one(model_from):
    from dilemma.normalize import Normalizer

    respellings = list(Normalizer(period="byzantine").normalize("αναγνώστει"))
    # The rule weights put ι -> οι before ει -> η.
    assert respellings.index("αναγνώστεοι") < respellings.index("αναγνώστη")

    loose = {"αναγνωστεοι": "ἀναγνωστέος"}
    model = model_from(loose)
    assert model._normalize_byzantine("αναγνώστει") == "ἀναγνωστέος"
    model = model_from({**loose, "αναγνώστη": "αναγνώστης"})
    assert model._normalize_byzantine("αναγνώστει") == "αναγνώστης"
    # A monotonic key ranks between the two.
    model = model_from({**loose, "αναγνώστη": "ἀναγνώστης"})
    assert model._lookup_word("ἀναγνώστη", step=model._OWN_ROWS) is None
    assert model._normalize_byzantine("ἀναγνώστει") == "ἀναγνώστης"
    model = model_from({**loose, "αναγνώστη": "ἀναγνώστης",
                        "ἀναγνώστι": "ἀνάγνωστις"})
    assert model._normalize_byzantine("ἀναγνώστει") == "ἀνάγνωστις"


def test_identity_reading_still_settles_a_respelling(model_from):
    # The first row a respelling reaches decides for it: an identity hit
    # (not a lemma for the word) is not replaced by a looser key's row.
    model = model_from({"αναγνώστη": "αναγνώστη", "αναγνωστη": "ἀναγιγνώσκω"})
    assert model._lookup_word("αναγνώστη") == "αναγνώστη"
    assert model._normalize_byzantine("αναγνώστει") is None
