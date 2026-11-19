"""Readings an explicit POS tag or a sibling cell recovers (#151)."""

import sqlite3

import pytest

from dilemma import core
from dilemma.grc_view_rows import GRC_VIEW_ROWS


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


def strip_key(word):
    return core.strip_accents(word.lower())


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
             "ἱερὰ": {"ADJ": "ἱερός", "NOUN": "ἱερόν"}},
        headwords={"ἀξία", "ἄξιος", "ἱερός", "ἱερόν"})
    assert tagged(model, "ἀξίαν", "NOUN") == "ἀξία"
    assert tagged(model, "ἀξίαν", "ADJ") == "ἄξιος"
    assert tagged(model, "ἱερὰ", "ADJ") == "ἱερός"
    assert tagged(model, "ἱερὰ", "NOUN") == "ἱερόν"


def test_entry_that_only_disagrees_with_the_row_does_not_decide(model_from):
    model = model_from({"ἐλάττονος": "μικρός", "Μικρός": "Μικρός"},
                       pos={"ἐλάττονος": {"ADJ": "ἐλαχύς"}})
    assert tagged(model, "ἐλάττονος", "ADJ") == "μικρός"


@pytest.mark.parametrize("lang", ["all", "grc"])
@pytest.mark.parametrize("word, upos, row, twin, cells, headwords", [
    # A capitalized cell is not the other half of a noun/adjective split.
    ("πάντα", "NOUN", "πᾶς", "Πᾶς", {"ADJ": "πᾶς", "NOUN": "Πάν"}, {"πᾶς"}),
    ("πολλὰ", "NOUN", "πολύς", "Πολύς",
     {"ADJ": "πολύς", "NOUN": "Πολλά"}, {"πολύς"}),
    # A verb row is not under ADJ or NOUN.
    ("κεῖται", "NOUN", "κεῖμαι", "Κεῖμαι",
     {"VERB": "κεῖμαι", "NOUN": "Κίτης"}, {"κεῖμαι"}),
    # A lowercase headword the form is no regular cell of.
    ("πάντα", "NOUN", "πᾶς", "Πᾶς", {"ADJ": "πᾶς", "NOUN": "πᾶν"}, {"πᾶς", "πᾶν"}),
    # A real noun/adjective split that is not a reviewed pair: a mistagged
    # common noun must not become the rare adjective.
    ("ζωήν", "ADJ", "ζωή", "Ζωή", {"NOUN": "ζωή", "ADJ": "ζωός"}, {"ζωή", "ζωός"}),
    ("βασιλείαν", "ADJ", "βασιλεία", "Βασιλεία",
     {"NOUN": "βασιλεία", "ADJ": "βασίλειος"}, {"βασιλεία", "βασίλειος"}),
])
def test_other_cells_do_not_override_a_row_with_a_capitalized_twin(
        model_from, lang, word, upos, row, twin, cells, headwords):
    model = model_from({word: row, twin: twin}, lang=lang,
                       pos={word: cells}, headwords=headwords)
    assert tagged(model, word, upos) == row


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_modern_contract_head_spells_the_ancient_verb(model_from, lang):
    model = model_from({"αστοχήσει": "αστοχώ"}, lang=lang,
                       headwords={"ἀστοχέω"}, vernacular={"αστοχώ"})
    assert model._lookup_word("ἀστοχήσει") == "ἀστοχέω"


def test_two_ancient_contract_verbs_are_not_chosen_between(model_from):
    model = model_from({"αστοχήσει": "αστοχώ"}, lang="all",
                       headwords={"ἀστοχέω", "ἀστοχάω"}, vernacular={"αστοχώ"})
    assert model._lookup_word("ἀστοχήσει") == "αστοχώ"


@pytest.mark.parametrize("word, row, head", [
    ("ἠχοῦς", "ηχώ", "ἠχέω"),      # the noun echo
    ("ἑνῶ", "ενώ", "ἑνόω"),        # the conjunction
])
def test_a_modern_head_in_o_is_a_verb_only_with_a_verb_form(
        model_from, word, row, head):
    model = model_from({strip_key(word): row}, lang="all", headwords={head},
                       vernacular={row})
    assert model._lookup_word(word) == row


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


@pytest.mark.parametrize("rows, word, verb", [
    ({"βαλῶ": "βάλλω", "βαλῷ": "βαλός"}, "βαλῶ", "βάλλω"),
    ({"γραικῶ": "γραικόω", "γραικῷ": "Γραικός"}, "γραικῶ", "γραικόω"),
    ({"βαλοῖς": "βάλλω", "βαλῷ": "βαλός"}, "βαλοῖς", "βάλλω"),
])
def test_only_a_verb_derived_from_the_nominal_gives_way(
        model_from, rows, word, verb):
    model = model_from(rows, headwords={"βαλός"})
    assert tagged(model, word, "NOUN") == verb


@pytest.mark.parametrize("word, dative, verb, nominal", [
    ("ποιῶ", "ποιῷ", "ποιέω", "ποιός"),
    ("δοκῶ", "δοκῷ", "δοκέω", "δοκός"),
    ("καλῶ", "καλῷ", "καλέω", "καλός"),
])
@pytest.mark.parametrize("upos", ["NOUN", "ADJ"])
def test_a_verb_the_tables_know_keeps_its_first_person(
        model_from, word, dative, verb, nominal, upos):
    model = with_name_cells(model_from({word: verb, dative: nominal}), set())
    model._fallback_verb_lemmas = {verb}
    assert tagged(model, word, upos) == verb


@pytest.mark.parametrize("rows, word, row", [
    ({"φειδῶ": "φειδώ", "φειδῷ": "φειδός"}, "φειδῶ", "φειδώ"),
    ({"πειθοῖς": "πειθώ", "πειθῷ": "πειθός"}, "πειθοῖς", "πειθώ"),
])
def test_a_noun_in_o_is_not_a_verb_giving_way(model_from, rows, word, row):
    model = model_from(rows, headwords={"πειθός", "φειδός"})
    assert tagged(model, word, "NOUN") == row


def test_oxytone_dative_plural_needs_a_sibling_cell(model_from):
    model = model_from({"ποδηγοῖς": "ποδηγέω", "ποδηγῷ": "ποδηγός"},
                       headwords={"ποδηγός"})
    assert tagged(model, "ποδηγοῖς", "NOUN") == "ποδηγός"
    assert tagged(model, "ποδηγοῖς", "VERB") == "ποδηγέω"
    alone = model_from({"ποδηγοῖς": "ποδηγέω"}, headwords={"ποδηγός"})
    assert tagged(alone, "ποδηγοῖς", "NOUN") == "ποδηγέω"


def with_name_cells(model, names):
    model._fallback_readings = {}
    model._fallback_exact_readings = {}
    model._fallback_verb_lemmas = set()
    model._fallback_name_lemmas = frozenset(names)
    return model


def test_lowercase_name_reads_its_capitalized_row(model_from):
    model = with_name_cells(model_from({"Θεοφάνην": "Θεοφάνης"}), {"Θεοφάνης"})
    assert tagged(model, "θεοφάνην", "NOUN") == "Θεοφάνης"
    assert tagged(model, "θεοφάνην", "VERB") is None


@pytest.mark.parametrize("word, row, lemma", [
    ("κανδιδάτους", "Κανδιδάτους", "Κανδιδάτος"),
    ("μορίους", "Μορίους", "Μόριος"),
])
def test_capitalized_row_without_name_evidence_is_not_taken(
        model_from, word, row, lemma):
    model = with_name_cells(model_from({row: lemma}), set())
    assert tagged(model, word, "NOUN") is None


MODERN_ROWS = {"εδώ": "εδώ", "κάνει": "κάνω", "άκου": "ακούω", "ευρώ": "ευρώ",
               "βάλει": "βάζω"}
STRIPPED_ANCIENT_ROWS = {"εδω": "ἐσθίω", "κανει": "καίνω", "ακου": "ἀκέομαι",
                         "ευρω": "εὑρίσκω", "βαλει": "βάλλω"}


@pytest.fixture
def small_lookup_db(tmp_path):
    """A lookup.db with the Modern rows and the Ancient stripped keys."""
    path = tmp_path / "lookup.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE lemmas (id INTEGER PRIMARY KEY, text TEXT)")
    conn.execute("CREATE TABLE lookup (form TEXT, lemma_id INTEGER, lang TEXT)")
    rows = {**MODERN_ROWS, **STRIPPED_ANCIENT_ROWS}
    ids = {lemma: i for i, lemma in enumerate(sorted(set(rows.values())))}
    conn.executemany("INSERT INTO lemmas VALUES (?, ?)",
                     [(i, lemma) for lemma, i in ids.items()])
    conn.executemany("INSERT INTO lookup VALUES (?, ?, 'all')",
                     [(form, ids[lemma]) for form, lemma in rows.items()])
    conn.commit()
    conn.close()
    return path


def db_model(model_from, path, lang):
    model = model_from({}, lang=lang)
    if lang == "grc":
        model._lookup = core.LookupDB(path, lang="grc",
                                      reviewed_rows=GRC_VIEW_ROWS)
        model._ag_lookup = model._lookup
    else:
        model._lookup = core.LookupDB(path, lang="all")
        model._ag_lookup = core.LookupDB(path, lang="grc")
    return model


def test_the_reviewed_ancient_row_is_only_the_ancient_view(
        model_from, small_lookup_db):
    ancient = db_model(model_from, small_lookup_db, "grc")
    assert ancient.lemmatize("βάλει", guess=False) == "βάλλω"
    assert tagged(ancient, "βάλει", "VERB") == "βάλλω"
    ancient._lookup.bulk_load()
    assert ancient._lookup.get("βάλει") == "βάλλω"
    combined = db_model(model_from, small_lookup_db, "all")
    assert combined.lemmatize("βάλει", guess=False) == "βάζω"


@pytest.mark.parametrize("word", ["εδώ", "κάνει", "άκου", "ευρώ"])
def test_modern_words_do_not_read_the_ancient_verb_of_their_letters(
        model_from, small_lookup_db, word):
    assert word not in GRC_VIEW_ROWS
    for lang in ("grc", "all"):
        model = db_model(model_from, small_lookup_db, lang)
        assert model.lemmatize(word, guess=False) == MODERN_ROWS[word]


@pytest.mark.parametrize("word", sorted(MODERN_ROWS))
def test_lemmatize_and_verbose_agree_with_lang_all(
        model_from, small_lookup_db, word):
    model = db_model(model_from, small_lookup_db, "all")
    top = model.lemmatize_verbose(word, guess=False)[0].lemma
    assert top == model.lemmatize(word, guess=False) == MODERN_ROWS[word]


def test_shipped_lookup_keeps_modern_words_out_of_the_ancient_view():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("lookup.db not downloaded")
    ancient = core.LookupDB(core.LOOKUP_DB_PATH, lang="grc",
                            reviewed_rows=GRC_VIEW_ROWS)
    assert {word: ancient.get(word) for word in MODERN_ROWS} == {
        **MODERN_ROWS, "βάλει": "βάλλω"}
