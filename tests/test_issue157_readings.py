"""Readings reported on #157 and the #147 leftovers fixed in 1.3.30."""

import json
import sqlite3

import pytest

from dilemma import core
from dilemma.core import _is_one_letter_acute_elision, to_standard_elision
from dilemma.grc_os_gaps import select_os_lemmas
from dilemma.grc_reviewed_respellings import (
    REVIEWED_LEMMA_RESPELLINGS, respelled_lemma)


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

    def make(lookup, *, lang="grc", pos=None, headwords=(), rejected=()):
        model = core.Dilemma(lang="grc", skip_pos=True)
        model.lang = lang
        model._lookup = dict(lookup)
        model._ag_lookup = dict(lookup)
        model._mg_lookup = {}
        model._pos_ag_lookup = dict(pos or {})
        model._pos_lookup = dict(pos or {})
        model._pos_rejected_targets = frozenset(rejected)
        model._headword_sets["lsj"] = set(headwords)
        model._ancient_lexical_heads = set(headwords)
        model._vernacular_lexical_heads = set()
        return model

    return make


def tagged(model, word, upos):
    scalar = model.lemmatize_pos(word, upos, guess=False)
    assert model.lemmatize_batch_pos([word], [upos], guess=False) == [scalar]
    return scalar


def plain(model, word):
    scalar = model.lemmatize(word, guess=False)
    assert model.lemmatize_batch([word], guess=False) == [scalar]
    return scalar


# The spacing acute printed for the elision apostrophe ----------------------

@pytest.mark.parametrize("word, written", [
    ("δι´", "δι’"), ("Δι´", "Δι’"), ("ἀλλ´", "ἀλλ’"), ("κατ´", "κατ’"),
    ("καθ´", "καθ’"), ("τοῦτ´", "τοῦτ’"), ("ἵν´", "ἵν’"),
    # δ´ and τ´ are δέ and τε far more often than 4 and 300.
    ("δ´", "δ’"), ("τ´", "τ’"),
    # After an accented final vowel the acute is a stray mark.
    ("ἀλλὰ´", "ἀλλὰ"), ("καὶ´", "καὶ"), ("ὃ´", "ὃ"),
    # The tonos and the prime after a word that is no numeral.
    ("δι΄", "δι’"), ("Δι΄", "Δι’"), ("παρ΄", "παρ’"), ("μετ΄", "μετ’"),
    ("ἀλλ΄", "ἀλλ’"), ("ὑπ΄", "ὑπ’"), ("δι′", "δι’"), ("έπ΄", "έπ’"),
    # The elided words written with no breathing and no accent.
    ("αλλ´", "αλλ’"), ("μηδ´", "μηδ’"), ("ποτ´", "ποτ’"), ("Καθ´", "Καθ’"),
])
def test_a_final_spacing_acute_is_the_elision_mark(word, written):
    assert to_standard_elision(word) == written


@pytest.mark.parametrize("word", [
    # Numerals: one letter other than δ and τ, a capital, a numeral of
    # several letters (ιστ´ and ις΄ with στ and ς for ϛ), and one letter or
    # a numeral with the tonos or the prime.
    "α´", "β´", "ε´", "γ´", "μ´", "Γ´", "Δ´", "ϛ´", "ιβ´", "ιε´", "ιστ´",
    "ις´", "δ΄", "τ΄", "β΄", "ιβ΄", "κα΄", "μη΄", "ις΄", "ιστ΄", "ΙΓ΄",
    "Ἀ΄", "δ′", "ιβ′",
    # Other unmarked letters with the mark: numerals in any order of their
    # letters, as tables print them.
    "δε´", "ζα´", "παο´", "δφ´", "βφ΄", "γχ΄", "ατ΄", "δω΄", "ῑβ΄", "ΡΑΓ΄",
    # Not a mark after a Greek letter.
    "´", "a´", "1´", "δ´´",
    # Already standard.
    "δι’", "δι᾽", "διά",
])
def test_numerals_and_other_tokens_keep_their_spelling(word):
    assert to_standard_elision(word) == word


def test_one_letter_elisions_are_recognized_for_the_num_tag():
    assert _is_one_letter_acute_elision("δ´")
    assert _is_one_letter_acute_elision("τ´")
    assert not _is_one_letter_acute_elision("β´")
    assert not _is_one_letter_acute_elision("δι´")
    assert not _is_one_letter_acute_elision(None)


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_elided_preposition_with_an_acute_reads_its_elided_key(model_from, lang):
    # The POS table keys the elided cells under ’, the spelling the tagged
    # paths read δι´ under.
    model = model_from({"δι᾽": "διά", "Δι᾽": "Ζεύς", "δ᾽": "δέ"}, lang=lang,
                       pos={"Δι’": {"ADP": "διά", "NOUN": "Ζεύς"}})
    assert plain(model, "δι´") == "διά"
    assert tagged(model, "δι´", "ADP") == "διά"
    assert tagged(model, "Δι´", "ADP") == "διά"
    assert plain(model, "Δι´") == "Ζεύς"
    assert plain(model, "δ´") == "δέ"
    # Tagged NUM, δ´ is the numeral 4: the token is its own answer.
    assert model.lemmatize_pos("δ´", "NUM", guess=False) is None
    assert model.lemmatize_pos("δ´", "NUM") == "δ´"
    assert model.lemmatize_batch_pos(["δ´", "δ´"], ["NUM", "CCONJ"]) == [
        "δ´", "δέ"]


# The lookup ----------------------------------------------------------------

def test_a_second_accent_from_an_enclitic_finds_the_own_row(model_from):
    model = model_from({"ἐφαπτόμενοι": "ἐφάπτω", "εφαπτομενοι": "εφάπτομαι"},
                       lang="all", headwords={"ἐφάπτω"})
    assert plain(model, "ἐφαπτόμενοί") == "ἐφάπτω"


@pytest.mark.parametrize("word", ["συμπτύσσοντες", "Συμπτύσσοντες",
                                  "συμπτύσσοντας", "συμπτυσσόντων"])
def test_an_oblique_active_participle_reads_the_verb_of_its_nominative(
        model_from, word):
    model = model_from({"συμπτύσσων": "συμπτύσσω", "συμπτύσσω": "συμπτύσσω",
                        "συμπτύσσουσα": "συμπτύσσω"},
                       headwords={"συμπτύσσω"})
    model._fallback_readings = {}
    model._fallback_exact_readings = {}
    model._fallback_verb_lemmas = {"συμπτύσσω"}
    assert plain(model, word) == "συμπτύσσω"


def test_an_active_participle_needs_a_verb_nominative(model_from):
    # γέρων reads a noun, so γέροντες is not a participle of it.
    model = model_from({"γέρων": "γέρων"}, headwords={"γέρων"})
    assert model._active_participle_lemma("γέροντες") is None


def _participle_model(model_from, rows, verbs):
    model = model_from(rows, headwords=set(rows.values()))
    model._fallback_readings = {}
    model._fallback_exact_readings = {}
    model._fallback_verb_lemmas = set(verbs)
    return model


def test_a_contract_participle_reads_its_contract_nominative(model_from):
    model = _participle_model(
        model_from, {"παρῶν": "παρόω", "παρόω": "παρόω"}, {"παρόω"})
    assert model._active_participle_lemma("παροῦντα") == "παρόω"


def test_a_misaccented_participle_with_two_nominatives_has_no_answer(
        model_from):
    # παροῦντα may be παρόντα misaccented: παρών, of πάρειμι, vetoes παρόω.
    model = _participle_model(
        model_from, {"παρῶν": "παρόω", "παρόω": "παρόω", "παρών": "παρών"},
        {"παρόω"})
    assert model._active_participle_lemma("παροῦντα") is None
    # An -οντ- stem never declines from -ῶν: ἄλλοντα is not of ἀναλόω.
    model = _participle_model(
        model_from, {"ἀλλῶν": "ἀναλόω", "ἀναλόω": "ἀναλόω",
                     "ἄλλων": "ἄλλος"}, {"ἀναλόω"})
    assert model._active_participle_lemma("ἄλλοντα") is None


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_a_rejected_pos_target_reads_the_headword_it_spells(model_from, lang):
    model = model_from({"αἱμοπτυικοὺς": "αἱμοπτυικός", "ἔχόντων": "ἔχοω",
                        "ἴσταται": "ἴστημι", "ὄρους": "ὀρος", "Ὅρα": "Ὅρας"},
                       lang=lang,
                       headwords={"αἱμοπτυϊκός", "ἵστημι", "ὄρος", "ὅρος", "Ορας"},
                       rejected={"αἱμοπτυικός", "ἔχοω", "ἴστημι", "ὀρος", "Ὅρας"})
    model._dictionary_heads = frozenset({"αἱμοπτυϊκός", "ἵστημι", "ὄρος", "ὅρος"})
    # The untagged answer is a dictionary headword written with other marks.
    assert tagged(model, "αἱμοπτυικοὺς", "ADJ") == "αἱμοπτυϊκός"
    assert tagged(model, "ἴσταται", "VERB") == "ἵστημι"
    # An artefact spells no headword, two headwords are no answer, and a
    # head only a name list gives does not count.
    assert tagged(model, "ἔχόντων", "VERB") is None
    assert tagged(model, "ὄρους", "NOUN") is None
    assert tagged(model, "Ὅρα", "PROPN") is None
    # The untagged lookup is unchanged.
    assert plain(model, "ἴσταται") == "ἴστημι"


def test_dictionary_heads_are_the_edited_dictionaries(tmp_path):
    path = tmp_path / "heads.json"
    path.write_text(json.dumps({"heads": {
        "ἁρπάζω": ["lsj9_headwords", "pd_headwords"], "Ορας": ["lgpn_names"],
        "Ἱστρικός": ["sophocles/0600.xml"], "ἐπιλεπτολογέω": ["dge_headwords"],
    }}, ensure_ascii=False), encoding="utf-8")
    stat = path.stat()
    lists = core._lexical_head_lists(str(path), stat.st_size, stat.st_mtime_ns)
    assert lists["grc_dictionary"] == {"ἁρπάζω", "ἐπιλεπτολογέω"}
    assert lists["grc"] == {"ἁρπάζω", "Ορας", "Ἱστρικός", "ἐπιλεπτολογέω"}


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_the_dative_of_an_action_noun_under_noun(model_from, lang):
    model = model_from({"συμπτύξει": "συμπτύσσω", "πράξει": "πράσσω"},
                       lang=lang,
                       headwords={"συμπτύσσω", "σύμπτυξις", "πράσσω", "πρᾶξις"})
    assert tagged(model, "συμπτύξει", "NOUN") == "σύμπτυξις"
    assert tagged(model, "πράξει", "NOUN") == "πρᾶξις"
    assert tagged(model, "συμπτύξει", "VERB") == "συμπτύσσω"


def test_an_action_noun_needs_one_headword_with_the_verb_stem(model_from):
    model = model_from({"λύσει": "λύω"}, headwords={"λύω"})
    assert model._action_noun_of_future("λύσει", "λύω") is None
    model = model_from({"γράψει": "γράφω"}, headwords={"γράφω", "γρᾶψις"})
    # A headword whose stem is not the verb's.
    assert model._action_noun_of_future("γράψει", "τρέφω") is None


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_a_capitalized_ordinal_tagged_propn_keeps_the_ordinal(model_from, lang):
    model = model_from(
        {"Ἕβδομον": "ἕβδομος", "Ἕβδομος": "Ἕβδομος", "Θυμόν": "θυμός",
         "Θυμός": "Θυμός"}, lang=lang,
        pos={"Ἕβδομον": {"PROPN": "Ἕβδομος"}, "Θυμόν": {"PROPN": "Θυμός"}},
        headwords={"ἕβδομος", "Ἕβδομος", "θυμός", "Θυμός"})
    assert tagged(model, "Ἕβδομον", "PROPN") == "ἕβδομος"
    # Any other case twin keeps the name under PROPN.
    assert tagged(model, "Θυμόν", "PROPN") == "Θυμός"


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_noun_and_adjective_split_of_meilichia(model_from, lang):
    model = model_from({"μειλιχίῃ": "μειλίχιος", "Μειλίχιος": "Μειλίχιος"},
                       lang=lang,
                       pos={"μειλιχίῃ": {"ADJ": "μειλίχιος", "NOUN": "μειλιχία"}},
                       headwords={"μειλιχία", "μειλίχιος"})
    assert tagged(model, "μειλιχίῃ", "NOUN") == "μειλιχία"
    assert tagged(model, "μειλιχίῃ", "ADJ") == "μειλίχιος"


# The builds ----------------------------------------------------------------

@pytest.mark.parametrize("form, given, lemma", [
    ("λημφθεῖσα", "λήμπω", "λαμβάνω"),
    ("ἀναλημφθεῖσα", "ἀναλήμπω", "ἀναλαμβάνω"),
    ("ἠθμοῦ", "ἡθμός", "ἠθμός"),
    ("ὀρθοπνοϊκοῖς", "ὀρθοπνοικός", "ὀρθοπνοϊκός"),
    ("αἱμοπτυικοὺς", "αἱμοπτυικός", "αἱμοπτυϊκός"),
    ("ἐπιλημψίας", "ἐπιλημψία", "ἐπιληψία"),
    ("ἐναπολήμψεις", "ἐναπόλημψις", "ἐναπόληψις"),
    ("πλαττόμεναι", "πλήσσω", "πλάσσω"),
    ("ἐπλάττετο", "πλήσσω", "πλάσσω"),
    ("διηρημένον", "διερέω", "διαιρέω"),
    ("διήρηται", "διερέω", "διαιρέω"),
    # Only the reviewed letters move: πλήττω's forms keep πλήσσω.
    ("πλήττεται", "πλήσσω", "πλήσσω"),
    ("διερέω", "διερέω", "διερέω"),
    ("λόγου", "λόγος", "λόγος"),
])
def test_reviewed_respellings(form, given, lemma):
    assert respelled_lemma(form, given) == lemma


def test_respellings_name_heads_that_are_not_respelled_again():
    for old, new in REVIEWED_LEMMA_RESPELLINGS.items():
        assert new not in REVIEWED_LEMMA_RESPELLINGS, old


def test_build_respells_each_table_on_its_own():
    from build_lookup_db import apply_reviewed_respellings
    combined = {"ηθμός": "ηθμός", "ἡθμοῦ": "ἡθμός", "πλάττοντα": "πλήσσω"}
    sources = {}
    assert apply_reviewed_respellings(combined, sources) == 2
    # The Modern row of the combined table keeps its own lemma.
    assert combined == {"ηθμός": "ηθμός", "ἡθμοῦ": "ἠθμός",
                        "πλάττοντα": "πλάσσω"}
    assert set(sources) == {"ἡθμοῦ", "πλάττοντα"}


def test_reviewed_readings_are_in_the_build():
    import build_lookup_db as build
    readings = build.REVIEWED_READINGS_1_3_30
    assert readings["Χάους"] == "χάος"
    assert readings["ἐπειδὴ"] == "ἐπειδή"
    assert readings["Τεύκριον"] == "τεύκριον"
    assert readings["οἱουτινοσοῦν"] == "οἱοστισοῦν"
    assert readings["ἀποικέω"] == "ἀποικέω"
    assert readings["δινεύω"] == "δινεύω" and readings["ἑλκέω"] == "ἑλκέω"
    absent = build.REVIEWED_ABSENT_KEY_READINGS
    for form in ("ἑνιαία", "ἑνιαίᾳ", "ἑνιαῖαι", "ἑνιαίαις"):
        assert absent[form] == "ἑνιαῖος"
    assert absent["ἀποκεκραμέναις"] == absent["ἀποκεκραμένου"] == "ἀποκεράννυμι"
    assert absent["ἐπιλεπτολογοῦσι"] == "ἐπιλεπτολογέω"
    assert absent["ἐπέτραπται"] == "ἐπιτρέπω"
    assert absent["τρεπτῆς"] == "τρεπτός"
    assert readings["συζήσας"] == "συζάω"
    assert readings["δηῦτε"] == "δηὖτε"
    assert readings["Ἀπίωνα"] == "Ἀπίων"


def test_a_genitive_without_the_plural_is_no_declining_table():
    # A Wiktionary page restored beside the head gave the singular only
    # (#147): the plural cells still come from the generator.
    lookup = {"ἀγχίαλος": "ἀγχίαλος", "ἀγχιάλου": "ἀγχίαλος",
              "ἀγχιάλῳ": "ἀγχίαλος",
              "ποικίλος": "ποικίλος", "ποικίλου": "ποικίλος",
              "ποικίλοι": "ποικίλος"}
    sources = {"ἀγχιάλου": "wiktionary_grc", "ποικίλου": "wiktionary_grc"}
    listed = {"ἀγχίαλος": "adjective", "ποικίλος": "adjective_eta"}
    assert select_os_lemmas(lookup, listed, sources.get) == {
        "ἀγχίαλος": "adjective"}


# Shipped answers -------------------------------------------------------------

_MODELS = {}


def _model(lang):
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    key = (lang,)
    if key not in _MODELS:
        _MODELS.clear()
        _MODELS[key] = (core.Dilemma(lang="all", resolve_articles=True)
                        if lang == "all" else
                        core.Dilemma(lang="grc", skip_numeral_ordinals=True))
    return _MODELS[key]


# (form, UPOS, lemma): the reports on #157 and the held Damascius and
# Dioscorides readings, as the annotations and the corpus lemma step ask.
_SHIPPED = [
    ("ἑνιαία", "ADJ", "ἑνιαῖος"),
    ("Χάους", "NOUN", "χάος"),
    ("ἀποκεκραμέναις", "VERB", "ἀποκεράννυμι"),
    ("δι´", "ADP", "διά"),
    ("Δι´", "ADP", "διά"),
    ("δι΄", "ADP", "διά"),
    ("ἐφαπτόμενοί", "VERB", "ἐφάπτω"),
    ("Συμπτύσσοντες", "VERB", "συμπτύσσω"),
    ("ἐπιλεπτολογοῦσι", "VERB", "ἐπιλεπτολογέω"),
    ("πλαττόμεναι", "VERB", "πλάσσω"),
    ("πλάττεται", "VERB", "πλάσσω"),
    ("διηρημένον", "VERB", "διαιρέω"),
    ("λίνῳ", "NOUN", "λίνον"),
    ("συμπτύξει", "NOUN", "σύμπτυξις"),
    ("ἐπειδὴ", "SCONJ", "ἐπειδή"),
    ("Ἕβδομον", "PROPN", "ἕβδομος"),
    ("Δέκατον", "PROPN", "δέκατος"),
    ("ἠθμοῦ", "NOUN", "ἠθμός"),
    ("Τεύκριον", "PROPN", "τεύκριον"),
    ("αἱμοπτυικοὺς", "ADJ", "αἱμοπτυϊκός"),
    ("ὀρθοπνοϊκοῖς", "ADJ", "ὀρθοπνοϊκός"),
    ("ὀρθοπνοικοὺς", "ADJ", "ὀρθοπνοϊκός"),
    ("λημφθεῖσα", "VERB", "λαμβάνω"),
    ("ἀναλημφθεῖσα", "VERB", "ἀναλαμβάνω"),
    ("οἱουτινοσοῦν", "PRON", "οἱοστισοῦν"),
    ("ἐπιλημψίας", "NOUN", "ἐπιληψία"),
    ("ἐναπολήμψεις", "NOUN", "ἐναπόληψις"),
    ("ἐπέτραπται", "VERB", "ἐπιτρέπω"),
    ("μειλιχίῃ", "NOUN", "μειλιχία"),
]

# The untagged answer, where it differs from the tagged one.
_UNTAGGED = {"Δι´": "Ζεύς", "λίνῳ": "λίνον", "συμπτύξει": "συμπτύσσω",
             "μειλιχίῃ": "μειλίχιος"}


def test_shipped_tagged_answers():
    model = _model("all")
    words = [form for form, _, _ in _SHIPPED]
    tags = [upos for _, upos, _ in _SHIPPED]
    want = [lemma for _, _, lemma in _SHIPPED]
    assert model.lemmatize_batch_pos(words, tags) == want
    assert [model.lemmatize_pos(w, t) for w, t in zip(words, tags)] == want


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_shipped_untagged_answers(lang):
    words = [form for form, _, _ in _SHIPPED]
    want = [_UNTAGGED.get(form, lemma) for form, _, lemma in _SHIPPED]
    assert _model(lang).lemmatize_batch(words, guess=False) == want


# Untagged answers the vocabulary comparison checked: the participle
# resolver, the enclitic-accent step and the Byzantine normalizer had read
# these through rows that 1.3.30 corrects, or found a second verb.
_SHIPPED_CORPUS = [
    ("δινεύω", "δινεύω"), ("ἑλκέω", "ἑλκέω"), ("ἀποικέω", "ἀποικέω"),
    ("διεῖπον", "διέπω"),
    ("τρεπτῆς", "τρεπτός"), ("τρεπτήν", "τρεπτός"),
    ("συζήσαντος", "συζάω"), ("δηῦτέ", "δηὖτε"), ("Ἀππίωνά", "Ἀπίων"),
    ("φρονούντες", "φρονέω"), ("ἄδοντες", "ἀείδω"),
    ("παροῦντα", None), ("ἄλλοντα", None), ("σιγαλόντα", None),
]


def test_shipped_corpus_answers():
    words = [form for form, _ in _SHIPPED_CORPUS]
    want = [lemma for _, lemma in _SHIPPED_CORPUS]
    assert _model("grc").lemmatize_batch(words, guess=False) == want
    assert [_model("grc").lemmatize(w, guess=False) for w in words] == want


def test_shipped_cells_of_restored_heads():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    with sqlite3.connect(f"file:{core.LOOKUP_DB_PATH}?mode=ro", uri=True) as conn:
        rows = dict(conn.execute(
            "SELECT l.form, m.text FROM lookup l JOIN lemmas m "
            "ON m.id = l.lemma_id WHERE l.lang = 'all' AND l.form IN "
            "('ἀγχίαλα', 'ἀγχίαλοι', 'ἀγχιάλοις', 'ἀξύλοις', 'ἠθμοῦ', "
            "'ηθμός')"))
    assert rows == {"ἀγχίαλα": "ἀγχίαλος", "ἀγχίαλοι": "ἀγχίαλος",
                    "ἀγχιάλοις": "ἀγχίαλος", "ἀξύλοις": "ἄξυλος",
                    "ἠθμοῦ": "ἠθμός", "ηθμός": "ηθμός"}
