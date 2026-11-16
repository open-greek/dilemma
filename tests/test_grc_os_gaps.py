"""Ancient second-declension -ος words that no table declines gain their
regular cells (#131).

The rule, selection and finder tests need no data. The shipped-answer tests
read data/lookup.db, whose generated rows arrive with the 1.3.27 rebuild.
"""
import json
from pathlib import Path
import sqlite3
import sys

import pytest

from dilemma import core
from dilemma.grc_noun_gaps import fill_gap_cells
from dilemma.grc_os_gaps import (
    ADJECTIVE_SOURCE, CLASSES, KINDS, MIN_STEM_LETTERS, NOUN_SOURCE,
    accent_type, genitive_singular, has_only_initial_breathing, os_cells,
    os_paradigms, read_classes, select_os_lemmas,
)

ROOT = Path(__file__).resolve().parents[1]


def _forms(cells):
    return [form for form, _ in cells]


def _cell(cells, name):
    return [form for form, cell in cells if name in cell.split("; ")]


# Rules ---------------------------------------------------------------------

def test_accent_type_reads_diphthongs_and_hiatus():
    assert accent_type("ἀγαθός") == "oxytone"
    assert accent_type("ποικίλος") == "paroxytone"
    assert accent_type("σπουδαῖος") == "properispomenon"
    assert accent_type("ἔμφλογος") == "proparoxytone"
    # αι is one syllable, ά-ι two: the accent sits on the first vowel.
    assert accent_type("δίκαιος") == "proparoxytone"
    assert accent_type("λάινος") == "proparoxytone"
    # One syllable, two accents or an accent before the antepenult: no type.
    assert accent_type("ὅς") is None
    assert accent_type("διάκοίλος") is None
    assert accent_type("ἄνθρωπινος") is None


def test_noun_cells_keep_the_accent_where_the_ending_allows():
    assert _forms(os_cells("ἄνθρωπος", "noun")) == [
        "ἀνθρώπου", "ἀνθρώπῳ", "ἄνθρωπον", "ἄνθρωποι", "ἀνθρώπων",
        "ἀνθρώποις", "ἀνθρώπους"]
    # An oxytone takes the circumflex in the genitives and datives.
    assert _forms(os_cells("ποταμός", "noun")) == [
        "ποταμοῦ", "ποταμῷ", "ποταμόν", "ποταμοί", "ποταμῶν", "ποταμοῖς",
        "ποταμούς"]
    # A properispomenon takes the acute before a long ultima.
    assert _forms(os_cells("δῆμος", "noun"))[:4] == [
        "δήμου", "δήμῳ", "δῆμον", "δῆμοι"]
    assert _forms(os_cells("παρθένος", "noun"))[:3] == [
        "παρθένου", "παρθένῳ", "παρθένον"]
    assert genitive_singular("ἔμφλογος") == "ἐμφλόγου"


def test_no_dual_and_no_vocative_singular():
    for declension in CLASSES:
        cells = os_cells("ἔμφλογος", declension)
        assert not any("dual" in cell for _, cell in cells)
        assert "ἔμφλογε" not in _forms(cells) and "ἐμφλόγω" not in _forms(cells)


def test_adjective_of_two_endings_has_the_neuter_and_no_feminine():
    cells = os_cells("ἔμφλογος", "adjective")
    assert _forms(cells) == [
        "ἐμφλόγου", "ἐμφλόγῳ", "ἔμφλογον", "ἔμφλογοι", "ἐμφλόγων",
        "ἐμφλόγοις", "ἐμφλόγους", "ἔμφλογα"]
    assert dict(cells)["ἔμφλογον"] == (
        "accusative singular; neuter nominative/accusative/vocative singular")
    assert not any("feminine" in cell for _, cell in cells)


def test_three_endings_in_eta_and_alpha():
    eta = os_cells("ἀγαθός", "adjective_eta")
    assert _cell(eta, "feminine nominative/vocative singular") == ["ἀγαθή"]
    assert _cell(eta, "feminine genitive singular") == ["ἀγαθῆς"]
    assert _cell(eta, "feminine dative singular") == ["ἀγαθῇ"]
    assert _cell(eta, "feminine accusative singular") == ["ἀγαθήν"]
    assert _cell(eta, "feminine nominative/vocative plural") == ["ἀγαθαί"]
    assert _cell(eta, "feminine dative plural") == ["ἀγαθαῖς"]
    assert _cell(eta, "feminine accusative plural") == ["ἀγαθάς"]
    alpha = os_cells("δίκαιος", "adjective_alpha")
    # The feminine singular has a long ultima, the nominative plural a
    # short one, and the genitive plural is the masculine's.
    assert _cell(alpha, "feminine nominative/vocative singular") == ["δικαία"]
    assert _cell(alpha, "feminine genitive singular") == ["δικαίας"]
    assert _cell(alpha, "feminine nominative/vocative plural") == ["δίκαιαι"]
    assert _cell(alpha, "neuter nominative/accusative/vocative plural") == ["δίκαια"]
    assert _cell(alpha, "genitive plural") == ["δικαίων"]
    # δικαίας is the genitive singular and the accusative plural.
    assert dict(alpha)["δικαίας"] == (
        "feminine genitive singular; feminine accusative plural")
    assert len(_forms(alpha)) == len(set(_forms(alpha)))
    oxytone = os_cells("λαμπρός", "adjective_alpha")
    assert _cell(oxytone, "feminine genitive singular") == ["λαμπρᾶς"]
    assert _cell(oxytone, "feminine accusative plural") == ["λαμπράς"]
    assert _cell(os_cells("ποιητέος", "adjective_alpha"),
                 "feminine dative singular") == ["ποιητέᾳ"]


def test_an_accent_that_lands_on_iota_after_a_vowel_takes_the_diaeresis():
    assert _forms(os_cells("λάινος", "adjective"))[:3] == [
        "λαΐνου", "λαΐνῳ", "λάινον"]
    assert genitive_singular("γήινος") == "γηΐνου"
    assert genitive_singular("πρώιος") == "πρωΐου"
    # A diphthong keeps its accent on the second letter.
    assert genitive_singular("ἀρχαῖος") == "ἀρχαίου"
    assert genitive_singular("ἄνοικος") == "ἀνοίκου"


def test_words_the_rule_does_not_decline():
    assert os_cells("εὔνοος", "adjective") == []      # stem in -ο contracts
    assert os_cells("ἔμφλογος", "neuter") == []       # not a class
    assert os_cells("σαφής", "adjective") == []
    assert os_cells("ὅς", "noun") == []
    assert os_cells("εμφλογος", "adjective") == []    # no accent
    assert genitive_singular("διάκοίλος") is None


def test_breathing_inside_a_word_is_an_artefact():
    assert has_only_initial_breathing("ἔμφλογος")
    assert has_only_initial_breathing("αὐτοσύστατος")
    assert has_only_initial_breathing("ῥόδινος")
    assert has_only_initial_breathing("κολοβόρινος")
    assert not has_only_initial_breathing("ἀνοἰκοδομητέος")


# Selection -----------------------------------------------------------------

def test_selection_takes_listed_heads_that_no_table_declines():
    lookup = {
        "ἔμφλογος": "ἔμφλογος",                       # headword only
        "ποικίλος": "ποικίλος", "ποικίλου": "ποικίλος",  # declined already
        "ἄτοπος": "ἄλλος",                             # not its own head
        "Ῥωμαϊκός": "Ῥωμαϊκός",                        # a name
        "ομοιόσχημος": "ομοιόσχημος",                  # lacks its breathing
        "ἀνοἰκοδομητέος": "ἀνοἰκοδομητέος",            # breathing inside
        "ζῆνος": "ζῆνος",                              # stem too short
        "κολοβόρινος": "κολοβόρινος",
    }
    listed = {lemma: "adjective" for lemma in lookup} | {"ἄγραφος": "adjective"}
    assert select_os_lemmas(lookup, listed) == {
        "ἔμφλογος": "adjective", "κολοβόρινος": "adjective"}
    assert MIN_STEM_LETTERS == 4


def test_a_genitive_from_a_corpus_or_a_modern_row_is_no_table():
    lookup = {"ψευδηγόρος": "ψευδηγόρος", "ψευδηγόρου": "ψευδηγόρος",
              "νοερός": "νοερός", "νοεροῦ": "νοερός",
              "ποικίλος": "ποικίλος", "ποικίλου": "ποικίλος",
              "λουβός": "λουβός", "λουβοῦ": "λουβός"}
    sources = {"ψευδηγόρου": "diorisis", "νοεροῦ": "wiktionary_el",
               "ποικίλου": "wiktionary_grc", "λουβοῦ": "lbg_generated"}
    listed = {"ψευδηγόρος": "adjective", "νοερός": "adjective_alpha",
              "ποικίλος": "adjective_eta", "λουβός": "noun"}
    assert select_os_lemmas(lookup, listed, sources.get) == {
        "ψευδηγόρος": "adjective", "νοερός": "adjective_alpha"}
    # Without sources every genitive row counts as a table's.
    assert select_os_lemmas(lookup, listed) == {}


def test_cells_fill_absent_keys_and_keep_other_readings():
    lookup = {"ἔμφλογος": "ἔμφλογος", "ἐμφλόγῳ": "ἐμφλογόω",
              "δικός": "δικός", "δίκης": "δίκη", "δικης": "δίκη",
              "νοερός": "νοερός", "νοεροῦ": "νοερός"}
    selected = {"ἔμφλογος": "adjective", "δικός": "adjective_eta",
                "νοερός": "adjective_alpha"}
    added = fill_gap_cells(lookup, os_paradigms(selected), strict=KINDS)
    # accented and accentless keys
    assert lookup["ἐμφλόγου"] == lookup["εμφλογου"] == "ἔμφλογος"
    assert lookup["ἔμφλογα"] == lookup["εμφλογα"] == "ἔμφλογος"
    # an existing reading is kept, and its cell is added under no key
    assert lookup["ἐμφλόγῳ"] == "ἐμφλογόω" and "εμφλογω" not in lookup
    # δικῆς differs from δίκης in its accent only: in a text it is that word
    assert "δικῆς" not in lookup and lookup["δικης"] == "δίκη"
    assert lookup["δικοῦ"] == "δικός"
    # a partly declined word gains the cells it lacks
    assert lookup["νοεράν"] == lookup["νοεράς"] == "νοερός"
    assert "νοεροῦ" not in {row["form"] for row in added}
    assert {row["kind"] for row in added} == {
        "os_adjective", "os_adjective_eta", "os_adjective_alpha"}
    assert all(row["cell"] for row in added)


def test_without_the_strict_gate_the_accentless_key_is_not_weighed():
    lookup = {"δικός": "δικός", "δίκης": "δίκη", "δικης": "δίκη"}
    fill_gap_cells(lookup, os_paradigms({"δικός": "adjective_eta"}))
    assert lookup["δικῆς"] == "δικός" and lookup["δικης"] == "δίκη"


def test_accent_doublets_leave_their_shared_cells_to_neither():
    lookup = {"εὐπρόσοδος": "εὐπρόσοδος", "εὐπροσόδος": "εὐπροσόδος"}
    selected = {"εὐπρόσοδος": "adjective", "εὐπροσόδος": "adjective"}
    fill_gap_cells(lookup, os_paradigms(selected), strict=KINDS)
    assert "εὐπροσόδων" not in lookup and "ευπροσοδων" not in lookup
    # Where the accent tells them apart each keeps its own key.
    assert lookup["εὐπρόσοδον"] == "εὐπρόσοδος"
    assert lookup["εὐπροσόδον"] == "εὐπροσόδος"
    assert "ευπροσοδον" not in lookup


def test_fill_does_not_depend_on_input_order():
    def run(lemmas):
        lookup = {lemma: lemma for lemma in lemmas}
        selected = select_os_lemmas(lookup, {lemma: "adjective" for lemma in lemmas})
        return fill_gap_cells(lookup, os_paradigms(selected), strict=KINDS)
    lemmas = ["ἔμφλογος", "κολοβόρινος", "αὐτοσύστατος"]
    assert run(lemmas) == run(list(reversed(lemmas)))


def test_sources_by_class():
    assert NOUN_SOURCE == "grc_noun_gaps"
    assert ADJECTIVE_SOURCE == "grc_adjective_gaps"
    assert KINDS == {"os_noun", "os_adjective", "os_adjective_eta",
                     "os_adjective_alpha"}


# The list (build/find_grc_second_declension.py) -----------------------------

@pytest.fixture(scope="module")
def finder():
    sys.path.insert(0, str(ROOT / "build"))
    import find_grc_second_declension
    return find_grc_second_declension


def test_head_text_gives_the_class(finder):
    read = finder.head_text_class
    assert read("ἀγαθός", "ή, όν, good") == "adjective_eta"
    assert read("καλλίσφυρος", "ον, beautiful-ankled") == "adjective"
    assert read("δίκαιος", "α, ον, also ος, ον") == "adjective_alpha"
    assert read("ἄξιος", "ία, ιον, weighing as much") == "adjective_alpha"
    assert read("λιθίδιος", "[ῐδ], α, ον, of stone") == "adjective_alpha"
    assert read("λόγος", "ὁ, verbal noun of λέγω") == "noun"
    assert read("ὁδός", "ἡ, way") == "noun"
    assert read("σκότος", "εος, τό, darkness") == finder.NEUTER
    assert read("τεῖχος", "τό, wall") == finder.NEUTER
    assert read("ἀγαθός", "v. sub ἀγαθός") is None
    # The endings must be the headword's own.
    assert read("ἀγαθός", "η, ἡ, = foreg.") is None


def test_suffix_settles_the_feminine_only_where_it_is_a_suffix(finder):
    assert finder.suffix_feminine("τακτικός") == "adjective_eta"
    assert finder.suffix_feminine("ἀρχαϊκός") == "adjective_eta"
    assert finder.suffix_feminine("ἡλιακός") == "adjective_eta"
    assert finder.suffix_feminine("ποιητέος") == "adjective_alpha"
    for lemma in ("δικός", "ἀεικός", "δαρεικός", "νικός", "τέος", "ἔμφλογος"):
        assert finder.suffix_feminine(lemma) is None


def test_settle_keeps_the_class_with_the_fewer_cells(finder):
    settle = finder.settle
    assert settle({"adjective_eta"}, "ἀγαθός") == ("adjective_eta", False)
    assert settle({"adjective", "adjective_alpha"}, "δίκαιος") == (
        "adjective_alpha", False)
    assert settle({"adjective_eta", "adjective_alpha"}, "ἄγριος") == (
        "adjective", False)
    assert settle({"noun", "adjective"}, "φίλος") == ("adjective", False)
    assert settle({"adjective"}, "τακτικός") == ("adjective_eta", True)
    assert settle({"noun"}, "λόγος") == ("noun", False)
    assert settle({"noun", finder.NEUTER}, "σκότος") == (None, False)
    assert settle(set(), "ἔμφλογος") == (None, False)


def test_finder_reads_the_lexica_and_writes_a_stable_list(finder, tmp_path):
    lsj9 = tmp_path / "lsj9"
    lsj9.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    (lsj9 / "lsj9_headwords.json").write_text(json.dumps([
        {"headword": "ἀπόβλητος", "grammar": "ον"},
        {"headword": "ποταμός", "grammar": "ὁ"},
        {"headword": "σκότος", "grammar": "ὁ"},
        {"headword": "σκότος", "grammar": "τό"},
        {"headword": "εὔνοος", "grammar": "ον"},
        {"headword": "Ἀθηναῖος", "grammar": "ον"},
    ], ensure_ascii=False), encoding="utf-8")
    glosses = [
        {"headword": "ἀγαθός", "text": "ή, όν, good"},
        {"headword": "ἀγαθός", "text": "ignored sense", "level": 1},
        {"headword": "ἀδράνεος", "text": "η, ον, = ἀδρανής"},
        {"headword": "ἀπόβλητος", "text": "to be thrown away"},
        {"headword": "ὑλικός", "text": "of matter"},
    ]
    (lsj9 / "lsj9_glosses.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in glosses),
        encoding="utf-8")
    (lsj9 / "lsj9_headword_pos.json").write_text(json.dumps({
        "αὐτο-σύστατος": "ADJ", "ὑλικός": "ADJ", "ποταμός": "NOUN"},
        ensure_ascii=False), encoding="utf-8")
    (data / "lbg_headwords.json").write_text(json.dumps([
        {"lemma": "ἔμφλογος"}, {"lemma": "φιλοσοφικός"},
        {"lemma": "ἀββᾶς"}, {"lemma": "σχηματισμός"},
        {"lemma": "λουβός", "gender": "ὁ"}, {"lemma": "πλάτος", "gender": "τό"},
        {"lemma": "ἀγαθός"},
    ], ensure_ascii=False), encoding="utf-8")
    record = finder.find(lsj9, data)
    assert record["rule"] == finder.RULE and record["schema_version"] == 1
    assert set(record["inputs"]) == {
        "lsj9_headwords.json", "lsj9_glosses.jsonl", "lsj9_headword_pos.json",
        "lbg_headwords.json"}
    assert all(len(digest) == 64 for digest in record["inputs"].values())
    classes = read_classes(record)
    assert classes == {
        "ἀγαθός": "adjective_eta",       # LSJ's endings beat the Byzantine blank
        "ἀδράνεος": "adjective_eta",
        "ἀπόβλητος": "adjective",        # grammar 'ον'
        "αὐτοσύστατος": "adjective",     # sub-entry, POS table only
        "ὑλικός": "adjective_eta",       # POS table, feminine by suffix
        "ἔμφλογος": "adjective",         # Byzantine headword without article
        "φιλοσοφικός": "adjective_eta",
        "λουβός": "noun",
        "ποταμός": "noun",
    }
    # σκότος (a source calls it a neuter), εὔνοος (stem in -ο), the name,
    # the Byzantine -μός without an article and the neuter are left out.
    assert record["classes"]["adjective_eta"]["lbg+suffix"] == ["φιλοσοφικός"]
    assert record["classes"]["adjective_eta"]["lsj9+lbg"] == ["ἀγαθός"]
    assert record["counts"] == {
        "noun": 2, "adjective": 3, "adjective_eta": 4, "adjective_alpha": 0}
    assert finder.find(lsj9, data) == record


def test_committed_list_matches_its_rule_and_declines(finder):
    record = json.loads(
        (ROOT / "data" / "grc_second_declension.json").read_text(encoding="utf-8"))
    assert record["rule"] == finder.RULE
    classes = read_classes(record)
    assert sum(record["counts"].values()) == len(classes) > 40_000
    assert set(record["classes"]) == set(CLASSES)
    for lemma, declension in classes.items():
        assert finder.is_candidate(lemma), lemma
        assert os_cells(lemma, declension), lemma
    for groups in record["classes"].values():
        for lemmas in groups.values():
            assert lemmas == sorted(lemmas)
    assert classes["ἔμφλογος"] == classes["κολοβόρινος"] == "adjective"
    assert classes["αὐτοσύστατος"] == "adjective"
    assert classes["ἀδράνεος"] == "adjective_eta"
    assert classes["ποιητέος"] == "adjective_alpha"
    # No suffix settles δικός, and neuters and stems in -ο are left out.
    assert classes["δικός"] == "adjective"
    for lemma in ("τεῖχος", "πλάτος", "εὔνοος"):
        assert lemma not in classes, lemma
    # LSJ's σκότος is ὁ first (σκότου) and the s-stem neuter second.
    assert classes["σκότος"] == "noun"


# Shipped answers (need the 1.3.27 lookup rebuild) -------------------------

_MODELS = {}


def _model(lang):
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    if lang not in _MODELS:
        _MODELS.clear()
        _MODELS[lang] = core.Dilemma(lang=lang)
    return _MODELS[lang]


def _lookup_rows():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    with sqlite3.connect(f"file:{core.LOOKUP_DB_PATH}?mode=ro", uri=True) as conn:
        return dict(conn.execute(
            "SELECT l.form, m.text || '|' || s.code FROM lookup l "
            "JOIN lemmas m ON m.id = l.lemma_id "
            "JOIN sources s ON s.id = l.source_id WHERE l.lang = 'all' "
            "AND s.code IN ('grc_noun_gaps', 'grc_adjective_gaps')"))


_SHIPPED = [
    ("ἐμφλόγου", "ἔμφλογος"), ("ἔμφλογον", "ἔμφλογος"),
    ("κολοβόρινον", "κολοβόρινος"), ("κολοβορίνων", "κολοβόρινος"),
    ("αὐτοσύστατοι", "αὐτοσύστατος"), ("αὐτοσυστάτου", "αὐτοσύστατος"),
    ("ἀδρανέους", "ἀδράνεος"), ("ἀδράνεα", "ἀδράνεος"),
    ("νοεράν", "νοερός"), ("δυσπενθοῦς", "δυσπενθής"), ("ἀκαμπῆ", "ἀκαμπής"),
]


def test_shipped_cells_name_their_source():
    rows = _lookup_rows()
    for form, lemma in _SHIPPED:
        assert rows.get(form) == f"{lemma}|{ADJECTIVE_SOURCE}", form
    assert sum(row.endswith(f"|{NOUN_SOURCE}") for row in rows.values()) > 39_000


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_shipped_cells_read_their_headword(lang):
    forms = [form for form, _ in _SHIPPED]
    assert _model(lang).lemmatize_batch(forms, guess=False) == [
        lemma for _, lemma in _SHIPPED]


def test_shipped_cells_keep_other_words_readings():
    rows = _lookup_rows()
    # No dual, no vocative, no cell of a word with a short stem, and no cell
    # whose accentless key is another word's: δικῆς beside δίκης, and
    # ἀδράνεον, which the verb ἀδρανέω has as an unaugmented imperfect.
    for form in ("ἐμφλόγω", "ἔμφλογε", "ζῆναι", "ὕπου", "δικῆς", "ἀδράνεον"):
        assert form not in rows, form
    model = _model("grc")
    assert model.lemmatize("δίκης", guess=False) == "δίκη"
    assert model.lemmatize("λόγου", guess=False) == "λόγος"
    assert model.lemmatize("ἀδράνεον", guess=False) == "ἀδρανέω"
    # The cells added in 1.3.25 are untouched.
    assert rows["στέφους"] == "στέφος|grc_noun_gaps"
    assert rows["σκιφους"] == "σκίφος|grc_noun_gaps"
