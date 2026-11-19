"""Numeral letters with a flattened ordinal ending (#152).

An ordinal printed as a letter with a superscript ending, Η with ον for
ὄγδοον, reaches the lemmatizer as Ηον. With skip_numeral_ordinals=True such a
token gets no lemma from any entry point. Without it nothing changes.
"""
import pytest

from dilemma import core

ROWS = {"ηον": "εἶμι", "γον": "γόνος", "βον": "Βον", "τον": "ὁ", "σον": "σός",
        "ιον": "ἴον"}
ORDINALS = ["Αον", "Βον", "Γον", "Δον", "Εον", "Ϛον", "ϛον", "Ζον", "Ηον", "Θον"]
WORDS = ["Τον", "Σον", "Ιον"]


@pytest.fixture
def model(tmp_path, monkeypatch):
    """A model that reads ROWS, with every ordinal also answered by its row."""
    monkeypatch.setattr(core.Dilemma, "_load_lookups", lambda *a, **k: None)
    monkeypatch.setattr(core.Dilemma, "_check_backend", lambda *a: None)
    for name in ("AG_POS_LOOKUP_PATH", "GLAUX_POS_LOOKUP_PATH",
                 "TREEBANK_POS_LOOKUP_PATH", "MG_POS_LOOKUP_PATH"):
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(core, name, path)
    monkeypatch.setattr(core, "POS_CELL_CORRECTIONS_PATH", tmp_path / "none.json")

    def make(flag):
        made = core.Dilemma(lang="grc", skip_pos=True,
                            skip_numeral_ordinals=flag)
        rows = dict(ROWS)
        rows.update((word.lower(), "λῆμμα") for word in ORDINALS
                    if word.lower() not in rows)
        made._pos_ag_lookup = {}
        made._pos_lookup = {}
        made._pos_rejected_targets = frozenset()
        made._lookup_word = lambda word, **kwargs: rows.get(word.lower())
        made._lookup_elided = lambda word: None
        made._article_pos_lemma = lambda word, upos: None
        made._apply_convention = lambda lemma, **kwargs: lemma
        made._checked_pos_result = lambda word, upos, lemma: lemma
        made._lemmatize_verbose_impl = lambda word, *args, **kwargs: [
            core.LemmaCandidate(lemma=rows[word.lower()], source="lookup",
                                via="exact")]
        return made

    return make


ENTRY_POINTS = {
    "lemmatize": lambda m, w: m.lemmatize(w, guess=False),
    "lemmatize_pos": lambda m, w: m.lemmatize_pos(w, "NOUN", guess=False),
    "lemmatize_batch": lambda m, w: m.lemmatize_batch([w], guess=False)[0],
    "lemmatize_batch_pos": lambda m, w: m.lemmatize_batch_pos(
        [w], ["NOUN"], guess=False)[0],
    "lemmatize_verbose": lambda m, w: next(
        (c.lemma for c in m.lemmatize_verbose(w, guess=False)), None),
}


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("word", ORDINALS)
def test_the_flag_leaves_an_ordinal_without_a_lemma(model, entry, word):
    assert ENTRY_POINTS[entry](model(True), word) is None


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("word", ["Ηον", "Γον", "Βον"])
def test_without_the_flag_the_row_still_answers(model, entry, word):
    assert ENTRY_POINTS[entry](model(False), word) == ROWS[word.lower()]


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("word", WORDS)
def test_letters_that_spell_common_words_are_not_ordinals(
        model, entry, word, flag):
    assert ENTRY_POINTS[entry](model(flag), word) == ROWS[word.lower()]


def test_a_pos_correction_cannot_bring_the_lemma_back(model):
    made = model(True)
    made._checked_pos_table_lookup = lambda word, upos: "ὄγδοος"
    assert made.lemmatize_batch_pos(["Ηον", "Τον"], ["ADJ", "DET"],
                                    guess=False)[0] is None


@pytest.mark.parametrize("word,expected", [
    ("Ηον", True), ("ϛον", True), ("Ϛον", True), ("γον", False),
    ("ηον", False), ("ΗΟΝ", False), ("Ηονα", False), ("Ἡον", False),
    ("Τον", False), ("Σον", False), ("Ιον", False), ("", False), (None, False),
])
def test_what_counts_as_an_ordinal(word, expected):
    assert core.is_numeral_ordinal(word) is expected
