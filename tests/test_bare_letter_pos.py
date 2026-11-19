"""A bare Greek letter tagged NOUN or ADJ lemmatizes to itself (#116).

One-letter keys in the POS tables come from dictionary abbreviations and
stray corpus tokens (α ADJ -> Λητογενής, ι NOUN -> ζευγίτης, δ NOUN ->
δεσποινίς). A single unmarked letter under NOUN or ADJ is a numeral, siglum
or list label, so both POS entry points return the letter. Other tags and
accented or breathed letters keep their existing answers.
"""
import pytest

from dilemma import core


@pytest.mark.parametrize("word", [
    "α", "Η", "ω", "ς", "ϝ", "Ϝ", "ϛ", "ϙ", "ϡ", "ϲ", "Ϲ", "Ω",
])
@pytest.mark.parametrize("upos", ["NOUN", "ADJ"])
def test_bare_letter_nominal_accepts_one_unmarked_letter(word, upos):
    assert core._bare_letter_nominal(word, upos) == word


@pytest.mark.parametrize("word", [
    "ά", "ἡ", "ὁ", "ᾳ", "ὰ", "αβ", "1", "a", "", "α'", "ʹ", ";",
])
def test_bare_letter_nominal_rejects_marked_or_non_letters(word):
    assert core._bare_letter_nominal(word, "NOUN") is None


@pytest.mark.parametrize("upos", [
    "DET", "PRON", "X", "NUM", "PROPN", "VERB", "ADV", None,
])
def test_bare_letter_nominal_leaves_other_tags(upos):
    assert core._bare_letter_nominal("α", upos) is None


@pytest.fixture
def junk_model(tmp_path, monkeypatch):
    """A model whose POS table and candidates hold the issue's junk cells."""
    monkeypatch.setattr(core.Dilemma, "_load_lookups", lambda *a, **k: None)
    monkeypatch.setattr(core.Dilemma, "_check_backend", lambda *a: None)
    for name in ("AG_POS_LOOKUP_PATH", "GLAUX_POS_LOOKUP_PATH",
                 "TREEBANK_POS_LOOKUP_PATH", "MG_POS_LOOKUP_PATH"):
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(core, name, path)
    monkeypatch.setattr(core, "POS_CELL_CORRECTIONS_PATH",
                        tmp_path / "none.json")
    model = core.Dilemma(lang="grc", skip_pos=True)
    model.lang = "all"
    table = {
        "α": {"ADJ": "Λητογενής", "NOUN": "άρθρο", "VERB": "ἐπέλπομαι"},
        "ι": {"NOUN": "ζευγίτης"},
        "δ": {"NOUN": "δεσποινίς"},
        "ε": {"ADJ": "αισώπειος"},
    }
    model._pos_ag_lookup = {f: dict(p) for f, p in table.items()}
    model._pos_lookup = {f: dict(p) for f, p in table.items()}
    model._pos_rejected_targets = frozenset()
    rows = {"ε": "ἔ", "α": "ὁ"}

    def verbose(word, **kwargs):
        lemma = rows.get(word)
        return ([core.LemmaCandidate(lemma=lemma, lang="grc",
                                     source="lookup", via="exact")]
                if lemma else [])

    model._lemmatize_verbose_impl = verbose
    model._apply_convention = lambda lemma: lemma
    model.lemmatize = lambda word, **kwargs: rows.get(word)
    model.lemmatize_batch = lambda words, **kwargs: [rows.get(w) for w in words]
    return model


@pytest.mark.parametrize("word,upos", [
    ("α", "ADJ"), ("α", "NOUN"), ("ι", "NOUN"), ("δ", "NOUN"),
    ("ε", "ADJ"), ("ε", "NOUN"),
])
def test_junk_cells_and_rows_give_way_to_the_letter(junk_model, word, upos):
    assert junk_model.lemmatize_pos(word, upos, guess=False) == word
    assert junk_model.lemmatize_batch_pos([word], [upos], guess=False) == [word]


def test_other_tags_still_read_the_table(junk_model):
    assert junk_model.lemmatize_pos("α", "VERB", guess=False) == "ἐπέλπομαι"
    assert junk_model.lemmatize_batch_pos(
        ["α", "α"], ["VERB", "ADJ"], guess=False) == ["ἐπέλπομαι", "α"]


# ---- shipped answers -------------------------------------------------------

ISSUE_FORMS = ["α", "ι", "υ", "ω", "ο", "ε", "ε", "α", "δ", "η", "γ"]
ISSUE_TAGS = ["ADJ", "NOUN", "NOUN", "NOUN", "ADJ", "NOUN", "ADJ", "NOUN",
              "NOUN", "NOUN", "NOUN"]


@pytest.fixture(scope="module", params=["all", "grc", "el"])
def shipped_model(request):
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    return core.Dilemma(lang=request.param, resolve_articles=True)


def test_shipped_issue_letters_return_themselves(shipped_model):
    expected = list(ISSUE_FORMS)
    assert shipped_model.lemmatize_batch_pos(
        ISSUE_FORMS, ISSUE_TAGS, guess=False) == expected
    assert [shipped_model.lemmatize_pos(form, tag, guess=False)
            for form, tag in zip(ISSUE_FORMS, ISSUE_TAGS)] == expected


def test_shipped_other_tags_and_untagged_are_unchanged(shipped_model):
    model = shipped_model
    if model.lang == "all":
        assert model.lemmatize_pos("η", "DET") == "ὁ"
        assert model.lemmatize_batch_pos(["η"], ["DET"]) == ["ὁ"]
    if model.lang == "grc":
        # The accent-stripped row of ἔ still serves untagged input.
        assert model.lemmatize("ε", guess=False) == "ἔ"
    # Accented letters are not bare: ἡ keeps the article logic.
    assert model.lemmatize_pos("ἡ", "NOUN", guess=False) == "ὁ"
    # X keeps the untagged answer: the Ancient lookup's Κ under "grc";
    # under "all" the untagged κ, not the single candidate's capital Κ,
    # whose case disagrees with the form (#128, as γ VERB -> γ).
    expected = "Κ" if model.lang == "grc" else "κ"
    assert model.lemmatize("κ", guess=False) == expected
    assert model.lemmatize_pos("κ", "X", guess=False) == expected
    assert model.lemmatize_batch_pos(["κ"], ["X"], guess=False) == [expected]
