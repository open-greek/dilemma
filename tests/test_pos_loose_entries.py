"""POS-table entries found only under a shared key (#105, #130).

An entry reached only through the accent-stripped key (ὄρους finds the
ὀρός of ὀροῦς under ορους) may not respell or replace the reading of the
form's own spelling. Under ADJ and PRON that holds for rows whose lemma an
Ancient dictionary lists (#130); under PRON the row must also be a reading
the tables record under PRON or DET.
"""
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
    batch = model.lemmatize_batch_pos([word], [upos], guess=False)
    assert batch == [scalar], (word, upos, scalar, batch)
    return scalar


EXACT = ("lookup", "exact")


@pytest.mark.parametrize("via,expected", [("exact", "ὄρος"), ("stripped", "ὀρός")])
def test_stripped_key_entry_does_not_respell_an_exact_row(table_model, via, expected):
    model = table_model([("ὄρος", "lookup", via)], {"ορους": {"NOUN": "ὀρός"}})
    assert both(model, "ὄρους", "NOUN") == expected


@pytest.mark.parametrize("key", ["ὄρους", "όρους"])
def test_own_and_monotonic_key_entries_still_respell(table_model, key):
    # The monotonic key keeps the accent's place: a breathing or accent
    # type misspelt in the source still finds its word there.
    model = table_model([("ὄρος", *EXACT)], {key: {"NOUN": "ὀρός"}})
    assert both(model, "ὄρους", "NOUN") == "ὀρός"


def test_reaccented_echo_row_does_not_own_the_spelling(table_model):
    model = table_model([("ὤρα", *EXACT)], {"ωρα": {"NOUN": "ὥρα"}})
    assert both(model, "ὦρα", "NOUN") == "ὥρα"


@pytest.mark.parametrize("word,upos,exact,entry", [
    ("Ἄργους", "PROPN", "Ἄργος", "Ἀργώ"),
    ("μᾶζαν", "PROPN", "μᾶζα", "Μαζάν"),
    ("ὅλα", "NOUN", "ὅλος", "ὀλή"),
])
def test_stripped_key_entry_does_not_replace_an_exact_nominal_row(
        table_model, word, upos, exact, entry):
    from dilemma.core import strip_accents
    model = table_model([(exact, *EXACT)],
                        {strip_accents(word.lower()): {upos: entry}})
    assert both(model, word, upos) == exact


def test_nominal_tag_still_replaces_an_exact_verb_row(table_model):
    # Byzantine μοχθῶ (= μόχθῳ) is held by the lookup as a form of μοχθέω.
    model = table_model([("μοχθέω", *EXACT)], {"μοχθω": {"NOUN": "μόχθος"}})
    assert both(model, "μοχθῶ", "NOUN") == "μόχθος"


def test_other_tags_keep_the_stripped_key_override(table_model):
    model = table_model([("τῇδε", *EXACT)], {"τηδε": {"PRON": "ὅδε"}})
    assert both(model, "τῆδε", "PRON") == "ὅδε"


# --- #130: ADJ and PRON ------------------------------------------------------

def listed_model(table_model, candidates, tables, listed, *, lang="grc",
                 rows=None, recorded=None):
    """A model whose Ancient dictionaries list *listed*, whose lookup holds
    *rows* (for the lowercase spelling of a capitalized word) and whose POS
    tables record the readings *recorded* ({bare key: {tag: lemmas}})."""
    model = table_model(candidates, tables, lang=lang)
    model._is_ancient_headword = lambda lemma: lemma in listed
    model._ag_lookup = dict(rows or {})
    model._mg_lookup = {}
    model._med_lookup = {}
    model._fallback_readings = {
        key: {tag: set(lemmas) for tag, lemmas in by_tag.items()}
        for key, by_tag in (recorded or {}).items()}
    model._fallback_exact_readings = {}
    model._fallback_verb_lemmas = set()
    return model


@pytest.mark.parametrize("word,upos,exact,entry,recorded", [
    ("ταύτην", "ADJ", "οὗτος", "ταὐτός", None),
    ("ἀθληταί", "ADJ", "ἀθλητής", "ἄθλητος", None),
    # Under PRON the tables record the row's lemma as a pronoun's or a
    # determiner's reading of these letters (οὗ PRON ἕ, ταύτων DET οὗτος).
    ("οὕ", "PRON", "ἕ", "ὅς", {"ου": {"PRON": ["ἕ", "ὅς"]}}),
    ("ταύτων", "PRON", "οὗτος", "αὐτός",
     {"ταυτων": {"PRON": ["αὐτός"], "DET": ["οὗτος"]}}),
])
@pytest.mark.parametrize("lang", ["grc", "all"])
def test_stripped_key_entry_does_not_replace_a_listed_row_under_adj_or_pron(
        table_model, word, upos, exact, entry, recorded, lang):
    from dilemma.core import strip_accents
    model = listed_model(table_model, [(exact, *EXACT)],
                         {strip_accents(word.lower()): {upos: entry}},
                         {exact, entry}, lang=lang, recorded=recorded)
    assert both(model, word, upos) == exact


@pytest.mark.parametrize("word,exact,entry,recorded", [
    # The row reads a noun or an adjective the tables record for these
    # letters: a misaccented ὅδε is not the vocative of ὁδός, nor κεινὸς
    # PRON the epic κενός.
    ("ὁδὲ", "ὁδός", "ὅδε", {"οδε": {"NOUN": ["ὁδός"], "PRON": ["ὅδε"]}}),
    ("κεινὸς", "κενός", "ἐκεῖνος",
     {"κεινος": {"ADJ": ["κενός", "ἐκεῖνος"], "PRON": ["ἐκεῖνος"]}}),
    # No table records the row's lemma for these letters: ἑκάτερος with a
    # smooth breathing, the accented Modern clitic μάς beside the month μής.
    ("ἐκάτερος", "ἐκάς", "ἑκάτερος", {"εκατερος": {"PRON": ["ἑκάτερος"]}}),
    ("μάς", "μής", "μας", {"μας": {"PRON": ["μας"]}}),
    ("οὕ", "ἕ", "ὅς", {}),
])
@pytest.mark.parametrize("lang", ["grc", "all"])
def test_row_no_table_records_as_a_pronoun_yields_under_pron(
        table_model, word, exact, entry, recorded, lang):
    from dilemma.core import strip_accents
    model = listed_model(table_model, [(exact, *EXACT)],
                         {strip_accents(word.lower()): {"PRON": entry}},
                         {exact, entry}, lang=lang, recorded=recorded)
    assert both(model, word, "PRON") == entry
    # ADJ asks for no such record: an adjective's row is often a noun's.
    model = listed_model(table_model, [(exact, *EXACT)],
                         {strip_accents(word.lower()): {"ADJ": entry}},
                         {exact, entry}, lang=lang, recorded=recorded)
    assert both(model, word, "ADJ") == exact


def test_row_with_an_unlisted_lemma_still_yields_under_adj(table_model):
    # μεγός is the corpus lemma of a misaccented token, no headword.
    model = listed_model(table_model, [("μεγός", *EXACT)],
                         {"μεγα": {"ADJ": "μέγας"}}, {"μέγας"})
    assert both(model, "μεγά", "ADJ") == "μέγας"


@pytest.mark.parametrize("word,upos,exact,entry", [
    ("τοῖϊν", "DET", "ὅς", "ὁ"),
    ("ιέ", "NUM", "ἰός", "ιε"),
    ("μᾶλα", "ADV", "μῆλον", "μάλα"),
])
def test_det_num_and_other_tags_keep_the_stripped_key_override(
        table_model, word, upos, exact, entry):
    from dilemma.core import strip_accents
    model = listed_model(table_model, [(exact, *EXACT)],
                         {strip_accents(word.lower()): {upos: entry}},
                         {exact, entry})
    assert both(model, word, upos) == entry


def test_modern_settings_keep_the_stripped_key_override_under_adj(table_model):
    tables = {"τεταρτου": {"ADJ": "τέταρτος"}}
    model = listed_model(table_model, [("τέταρτο", *EXACT)], tables,
                         {"τέταρτο", "τέταρτος"}, lang="el")
    assert both(model, "τετάρτου", "ADJ") == "τέταρτος"
    model = listed_model(table_model, [("τέταρτο", *EXACT)], tables,
                         {"τέταρτο", "τέταρτος"}, lang="all")
    model._convention_name = "triantafyllidis"
    assert both(model, "τετάρτου", "ADJ") == "τέταρτος"


def test_exact_verb_row_still_yields_under_adj(table_model):
    model = listed_model(table_model, [("πιστόω", *EXACT)],
                         {"πιστω": {"ADJ": "πιστός"}}, {"πιστόω", "πιστός"})
    assert both(model, "πιστῶ", "ADJ") == "πιστός"


@pytest.mark.parametrize("rows,expected", [
    ({"ἀντίαν": "ἀντίος"}, "ἀντίος"),   # the lowercase spelling's row agrees
    ({"ἀντίαν": "ἀντία"}, "ἀντία"),
    ({}, "ἀντία"),
])
def test_capitalized_row_yields_to_an_entry_its_lowercase_row_reads(
        table_model, rows, expected):
    model = listed_model(table_model, [("ἀντία", *EXACT)],
                         {"αντιαν": {"ADJ": "ἀντίος"}}, {"ἀντία", "ἀντίος"},
                         rows=rows)
    assert both(model, "Ἀντίαν", "ADJ") == expected


@pytest.mark.parametrize("upos", ["NOUN", "ADJ"])
def test_stripped_key_entry_does_not_replace_an_article_reading(table_model, upos):
    model = table_model([("αὐτός", "article", "")], {"αυτη": {upos: "ἀϋτή"}})
    assert both(model, "αὐτή", upos) == "αὐτός"


def test_unknown_entry_provenance_keeps_its_authority(table_model):
    model = table_model([("αὐτός", "article", "")], {})
    model._pos_table_lookup = lambda word, upos: "ἀϋτή"
    assert both(model, "αὐτή", "NOUN") == "ἀϋτή"


@pytest.mark.parametrize("word,key,family,verb,expected", [
    ("πόθῳ", "ποθω", ("πόθος", "Πόθος"), "πίνω", "πόθος"),
    ("ἰδῶν", "ιδών", ("ἴδη", "Ἴδη"), "ὁράω", "ὁράω"),
])
def test_verb_case_twin_rule_needs_more_than_the_stripped_key(
        table_model, word, key, family, verb, expected):
    model = table_model([(family[0], "lookup", "exact"),
                         (family[1], "lookup", "exact+case_alt")],
                        {key: {"VERB": verb}})
    assert both(model, word, "VERB") == expected


# --- shipped data ---------------------------------------------------------

@pytest.fixture(scope="module", params=["grc", "all"])
def shipped_model(request):
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    return core.Dilemma(lang=request.param, resolve_articles=True)


SHIPPED_105 = [
    ("ὄρους", "NOUN", "ὄρος"), ("ὅρους", "NOUN", "ὅρος"),
    ("ὅλων", "NOUN", "ὅλος"), ("ὅλα", "NOUN", "ὅλος"),
    ("ἐκτός", "ADV", "ἐκτός"), ("ἕκτος", "ADV", "ἕκτος"),
    ("Ἄργους", "PROPN", "Ἄργος"), ("αὐτή", "NOUN", "αὐτός"),
    ("μᾶζαν", "PROPN", "μᾶζα"), ("πόθῳ", "VERB", "πόθος"),
    ("αὐτόν", "ADJ", "αὐτός"),
    # controls the issue names, and the Byzantine misspellings that rely on
    # a looser entry
    ("ὄρει", "NOUN", "ὄρος"), ("ὅρον", "NOUN", "ὅρος"), ("Ἄργει", "PROPN", "Ἄργος"),
    ("μοχθῶ", "NOUN", "μόχθος"), ("πιστῶ", "ADJ", "πιστός"),
    ("σοφῶ", "ADJ", "σοφός"), ("καμοῦ", "PRON", "ἐγώ"),
    ("τῆδε", "PRON", "ὅδε"), ("ἤδει", "VERB", "οἶδα"),
    ("Ὦρα", "NOUN", "ὥρα"), ("Λάμπε", "VERB", "λάμπω"),
    # #130: ADJ and PRON keep an exact row with a listed lemma
    ("ταύτην", "ADJ", "οὗτος"), ("ταῦτα", "ADJ", "οὗτος"),
    ("ἀθληταί", "ADJ", "ἀθλητής"), ("ἑνός", "ADJ", "εἷς"),
    ("βάθη", "ADJ", "βάθος"), ("οὕ", "PRON", "ἕ"),
    # and its controls: an unlisted row lemma, DET, a capitalized word whose
    # lowercase row agrees with the entry
    ("μεγά", "ADJ", "μέγας"), ("τοῖϊν", "DET", "ὁ"),
    ("Ἀντίαν", "ADJ", "ἀντίος"), ("ταύτην", "PRON", "οὗτος"),
    # PRON: a row the tables record as a pronoun's stands (ταύτὴν, αὕται);
    # the row of another kind of word yields to the pronoun, as in 1.3.26
    ("ταύτὴν", "PRON", "οὗτος"), ("αὕται", "PRON", "οὗτος"),
    ("ὁδὲ", "PRON", "ὅδε"), ("ἐκάτερος", "PRON", "ἑκάτερος"),
    ("οὔδεσι", "PRON", "οὐδείς"), ("σφησί", "PRON", "σφός"),
    ("κεινὸς", "PRON", "ἐκεῖνος"), ("παντὰ", "PRON", "πᾶς"),
]



@pytest.mark.parametrize("word,upos,lemma", SHIPPED_105)
def test_shipped_pos_answers(shipped_model, word, upos, lemma):
    assert both(shipped_model, word, upos) == lemma


def test_shipped_modern_clitic_keeps_its_pronoun_entry(shipped_model):
    # #130: with lang="all" the row of the accented clitic μάς reads the
    # Doric month μής, a listed noun; the pronoun entry still decides.
    if shipped_model.lang != "all":
        pytest.skip("the Modern Greek POS table is loaded with lang='all'")
    assert both(shipped_model, "μάς", "PRON") == "μας"


def test_shipped_batch_matches_scalar_in_context(shipped_model):
    rows = SHIPPED_105
    batch = shipped_model.lemmatize_batch_pos(
        [row[0] for row in rows], [row[1] for row in rows], guess=False)
    assert batch == [row[2] for row in rows]


# --- #151: the form's own reading, unaccented pronouns, name twins ----------

@pytest.mark.parametrize("word,upos,row,entry", [
    # A capitalized self row is often only a name of the onomastic lists.
    ("Γαίων", "NOUN", "Γαίων", "γαῖα"),
    # An adjective's feminine is spelled like the noun.
    ("ἐπιδημία", "ADJ", "ἐπιδημία", "ἐπιδήμιος"),
    # A listed spelling that repeats the form is no noun of its own: ὅρη is
    # a misbreathed ὄρη, and οἴκοι NOUN is the noun, not the adverb.
    ("ὅρη", "NOUN", "ὅρη", "ὄρος"),
    ("οἴκοι", "NOUN", "οἴκοι", "οἶκος"),
])
def test_stripped_key_entry_replaces_other_self_rows(
        table_model, word, upos, row, entry):
    from dilemma.core import strip_accents
    model = listed_model(table_model, [(row, *EXACT)],
                         {strip_accents(word.lower()): {upos: entry}}, {row, entry})
    assert both(model, word, upos) == entry


@pytest.mark.parametrize("word,row,entry,recorded", [
    # The table stores these cells under the unaccented key, which is
    # also the word as written; the row is a pronoun reading of the letters.
    ("ταυτης", "οὗτος", "αὐτός", {"ταυτης": {"PRON": ["αὐτός", "οὗτος"]}}),
    ("συ", "σύ", "ἐγώ", {"συ": {"PRON": ["ἐγώ", "σύ"]}}),
])
@pytest.mark.parametrize("lang", ["grc", "all"])
def test_unaccented_pronoun_keeps_its_recorded_row(
        table_model, word, row, entry, recorded, lang):
    model = listed_model(table_model, [(row, *EXACT)], {word: {"PRON": entry}},
                         {row, entry}, lang=lang, recorded=recorded)
    assert both(model, word, "PRON") == row


@pytest.mark.parametrize("word,row,entry,recorded", [
    # The row reads another kind of word: οἵδε is ὅδε, not a form of οἶδα;
    # σής the moth is no pronoun, so unaccented σης stays σός.
    ("οιδε", "οἶδα", "ὅδε", {"οιδε": {"VERB": ["οἶδα"], "PRON": ["ὅδε"]}}),
    ("σης", "σής", "σός", {"σης": {"NOUN": ["σής"], "PRON": ["σός"]}}),
    # The article is recorded for α only under DET: α PRON is the relative ἅ.
    ("α", "ὁ", "ὅς", {"α": {"DET": ["ὁ"], "PRON": ["ὅς"]}}),
    # An entry with the row's letters: no accent here to choose between them.
    ("οιαν", "οἶος", "οἷος", {"οιαν": {"PRON": ["οἷος", "οἶος"]}}),
])
def test_unaccented_pronoun_entry_beats_a_row_of_another_kind(
        table_model, word, row, entry, recorded):
    model = listed_model(table_model, [(row, *EXACT)], {word: {"PRON": entry}},
                         {row, entry}, recorded=recorded)
    assert both(model, word, "PRON") == entry


def test_unaccented_noun_keeps_the_entry_of_its_key(table_model):
    # Under NOUN the cell is often the better reading of bare letters: the
    # row here is an adjective and the entry the noun.
    model = listed_model(table_model, [("ἐνύπνιος", *EXACT)],
                         {"ενυπνιοις": {"NOUN": "ἐνύπνιον"}},
                         {"ἐνύπνιος", "ἐνύπνιον"})
    assert both(model, "ενυπνιοις", "NOUN") == "ἐνύπνιον"


TWINS = [("ἡδύς", "lookup", "exact"), ("Ἡδύς", "lookup", "exact+case_alt")]


@pytest.mark.parametrize("lang", ["grc", "all"])
def test_name_entry_with_other_letters_leaves_a_name_twin(table_model, lang):
    model = listed_model(table_model, TWINS, {"ἡδύς": {"PROPN": "Ἡδεῖα"}},
                         {"ἡδύς"}, lang=lang)
    assert both(model, "Ἡδὺς", "PROPN") == "Ἡδύς"


@pytest.mark.parametrize("word,rows,entry", [
    # An unaccented capitalized word: the bare letters' rows are no twin of it.
    ("Διος", [("δῖος", "lookup", "exact"), ("Δῖος", "lookup", "exact+case_alt")], "Ζεύς"),
    ("Αρη", [("ἀρή", "lookup", "exact"), ("Ἀρή", "lookup", "exact+case_alt")], "Ἄρης"),
    # The genitive of the name the entry gives.
    ("Ἀράχνης", [("ἀράχνης", "lookup", "exact"), ("Ἀράχνης", "lookup", "exact+case_alt")],
     "Ἀράχνη"),
])
@pytest.mark.parametrize("lang", ["grc", "all"])
def test_name_entry_stands_beside_rows_that_are_no_twin_of_the_form(
        table_model, word, rows, entry, lang):
    from dilemma.core import strip_accents
    model = listed_model(table_model, rows,
                         {strip_accents(word.lower()): {"PROPN": entry}},
                         {entry}, lang=lang)
    assert both(model, word, "PROPN") == entry


def test_name_entry_still_replaces_a_lone_modern_self_row(table_model):
    # The capitalized row of a genitive is no name twin of a common word.
    model = listed_model(table_model, [("Παύλου", *EXACT)],
                         {"Παύλου": {"PROPN": "Παῦλος"}}, {"Παῦλος"})
    assert both(model, "Παύλου", "PROPN") == "Παῦλος"


@pytest.mark.parametrize("letter", ["α", "θ"])
@pytest.mark.parametrize("upos", ["ADJ", "NOUN"])
def test_bare_letter_outranks_the_rejected_corpus_target(table_model, letter, upos):
    model = table_model([("ὁ", *EXACT)], {letter: {"ADJ": "Λητογενής"}})
    model._pos_rejected_targets = frozenset({letter})
    assert both(model, letter, upos) == letter


def test_rejected_letter_is_still_rejected_under_other_tags(table_model):
    model = table_model([("α", *EXACT)], {})
    model._pos_rejected_targets = frozenset({"α"})
    assert model.lemmatize_batch_pos(["α"], ["VERB"], guess=False) == [None]


def test_lowercase_cyrus_genitive_reads_the_name_under_noun(table_model):
    from pathlib import Path
    record = Path(__file__).resolve().parents[1] / "data" / "pos_cell_corrections.json"
    table = core._apply_pos_cell_corrections(
        "ag", {"κύρου": {"PROPN": "Κῦρος", "VERB": "κυρόω"}}, record)
    assert table["κύρου"]["NOUN"] == "Κῦρος"
    model = listed_model(table_model, [("κυρόω", *EXACT)], table, {"Κῦρος"})
    assert both(model, "κύρου", "NOUN") == "Κῦρος"
    assert both(model, "κύρου", "VERB") == "κυρόω"


@pytest.mark.parametrize("lang", ["grc", "all"])
def test_lowercase_servant_noun_keeps_its_reviewed_cell(table_model, lang):
    from pathlib import Path
    record = Path(__file__).resolve().parents[1] / "data" / "pos_cell_corrections.json"
    table = core._apply_pos_cell_corrections(
        "ag", {"λατρης": {"NOUN": "λατρεύς"}}, record)
    assert table["λάτρης"]["NOUN"] == "λάτρης"
    model = listed_model(table_model, [("λάτρης", *EXACT)], table,
                         {"λάτρης", "λατρεύς"}, lang=lang)
    assert both(model, "λάτρης", "NOUN") == "λάτρης"
