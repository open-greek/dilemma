"""Reviewed corrections to POS-table cells (data/pos_cell_corrections.json).

#106 (and the numeral cells of #105): wrong treebank, GLAUx and Wiktionary
cells are dropped or corrected when the tables load, without a rebuild.
#128: a Modern cell that only a dropped paradigm fragment kept is added
back, and an lsj9 accent-stripped cell is dropped.
#125, #126: cells the rebuilt tables no longer carry are added back, new
cells that displace a 1.3.25 answer are corrected or dropped, and names
beside a restored lowercase head keep their capitalized lemma.
"""
import json
from pathlib import Path

import pytest

from dilemma import core

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "data" / "pos_cell_corrections.json"
TABLE_PATHS = {"treebank": ROOT / "data" / "treebank_pos_lookup.json",
               "glaux": ROOT / "data" / "glaux_pos_lookup.json",
               "ag": ROOT / "data" / "ag_pos_lookup.json",
               "mg": ROOT / "data" / "mg_pos_lookup.json",
               "lsj9": ROOT / "data" / "lsj9_pos_lookup.json"}


def _record(tmp_path, rows):
    path = tmp_path / "pos_cell_corrections.json"
    path.write_text(json.dumps({"schema_version": 1, "corrections": rows},
                               ensure_ascii=False), encoding="utf-8")
    return path


def _row(**fields):
    row = {"table": "treebank", "form": "μάκαρ", "upos": "ADJ",
           "target": "μάκαπ", "action": "drop", "reason": "typo"}
    row.update(fields)
    return row


def test_correction_record_drops_and_corrects_matching_cells(tmp_path):
    path = _record(tmp_path, [
        _row(),
        _row(form="μία", upos="NUM", target="εἶς", action="correct", corrected="εἷς"),
        _row(form="καί", upos="INTJ", target="αἴ"),
    ])
    table = {"μάκαρ": {"ADJ": "μάκαπ", "NOUN": "μάκαρ"}, "μία": {"NUM": "εἶς"},
             "καί": {"INTJ": "καί"}}
    core._apply_pos_cell_corrections("treebank", table, path)
    assert table == {"μάκαρ": {"NOUN": "μάκαρ"}, "μία": {"NUM": "εἷς"},
                     "καί": {"INTJ": "καί"}}  # a cell already fixed is left alone
    other = {"μάκαρ": {"ADJ": "μάκαπ"}}
    core._apply_pos_cell_corrections("glaux", other, path)
    assert other == {"μάκαρ": {"ADJ": "μάκαπ"}}


def test_add_row_supplies_a_missing_cell_only(tmp_path):
    path = _record(tmp_path, [
        {"table": "mg", "form": "ας", "upos": "PART", "action": "add",
         "corrected": "ας", "reason": "particle"},
        {"table": "mg", "form": "να", "upos": "PART", "action": "add",
         "corrected": "να", "reason": "particle"},
    ])
    table = {"να": {"PART": "νά"}}
    core._apply_pos_cell_corrections("mg", table, path)
    # A table that already holds a cell for the tag keeps it.
    assert table == {"ας": {"PART": "ας"}, "να": {"PART": "νά"}}
    other = {}
    core._apply_pos_cell_corrections("ag", other, path)
    assert other == {}


@pytest.mark.parametrize("bad", [
    {"action": "rename"}, {"table": "lexicon"}, {"reason": ""},
    {"action": "correct"}, {"corrected": "μάκαρ"},
    {"action": "add", "corrected": "μάκαρ"},  # an add row names no target
    {"action": "add", "target": None},
    {"target": None},
])
def test_malformed_correction_rows_fail_clearly(tmp_path, bad):
    path = _record(tmp_path, [_row(**bad)])
    with pytest.raises(ValueError, match="malformed POS cell correction"):
        core._apply_pos_cell_corrections("treebank", {}, path)


def test_dropped_treebank_cell_lets_the_lower_table_answer(tmp_path):
    (tmp_path / "treebank_pos_lookup.json").write_text(json.dumps(
        {"μάκαρ": {"ADJ": "μάκαπ"}}, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "ag_pos_lookup.json").write_text(json.dumps(
        {"μάκαρ": {"ADJ": "μάκαρ"}}, ensure_ascii=False), encoding="utf-8")
    model = core.Dilemma.__new__(core.Dilemma)
    model.lang = "grc"
    model._pos_lookup, model._pos_ag_lookup = {}, {}
    model._load_pos_lookups(data_dir=tmp_path)
    assert model._pos_ag_lookup["μάκαρ"]["ADJ"] == "μάκαπ"
    _record(tmp_path, [_row()])
    model._pos_lookup, model._pos_ag_lookup = {}, {}
    model._load_pos_lookups(data_dir=tmp_path)
    assert model._pos_ag_lookup["μάκαρ"]["ADJ"] == "μάκαρ"


def test_shipped_correction_record_is_reviewed_and_current():
    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    tables = {name: json.loads(path.read_text(encoding="utf-8"))
              for name, path in TABLE_PATHS.items() if path.exists()}
    seen = set()
    for row in record["corrections"]:
        assert row["reason"] and row["evidence"] and row["issue"] in (
            105, 106, 125, 126, 128)
        key = (row["table"], row["form"], row["upos"])
        assert key not in seen
        seen.add(key)
        if row["table"] not in tables:
            continue  # lsj9 is a Hub download; the base CI set may lack it
        # A rebuilt table that already fixed a cell makes its row a no-op;
        # remove such rows rather than keeping dead corrections. An add row
        # is dead once the table supplies the cell itself.
        assert tables[row["table"]].get(row["form"], {}).get(row["upos"]) \
            == row.get("target"), row
    rows = core._pos_cell_correction_rows(
        str(RECORD_PATH), RECORD_PATH.stat().st_size, RECORD_PATH.stat().st_mtime_ns)
    assert sum(len(cells) for cells in rows.values()) == len(record["corrections"])


# --- shipped data ---------------------------------------------------------

@pytest.fixture(scope="module", params=["grc", "all"])
def shipped_model(request):
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    return core.Dilemma(lang=request.param, resolve_articles=True)


SHIPPED_106 = [
    ("μάκαρ", "ADJ", "μάκαρ"), ("οὗπερ", "PRON", "ὅσπερ"),
    ("καί", "INTJ", "καί"), ("καί", "VERB", "καί"), ("και", "ADV", "καί"),
    ("τούς", "ADJ", "ὁ"), ("Μία", "NUM", "εἷς"), ("μία", "NUM", "εἷς"),
    ("κλήσει", "NOUN", "κλῆσις"), ("κλήσεις", "NOUN", "κλῆσις"),
    ("κλῄσει", "NOUN", "κλῇσις"),
    ("μάκαρες", "ADJ", "μάκαρ"), ("μίαν", "NUM", "εἷς"), ("κλήσεως", "NOUN", "κλῆσις"),
    ("γλυκερός", "ADJ", "γλυκερός"), ("γλυκερόν", "ADJ", "γλυκερός"),
]


@pytest.mark.parametrize("word,upos,lemma", SHIPPED_106)
def test_shipped_corrected_cells(shipped_model, word, upos, lemma):
    scalar = shipped_model.lemmatize_pos(word, upos, guess=False)
    assert scalar == lemma
    assert shipped_model.lemmatize_batch_pos([word], [upos], guess=False) == [lemma]


def test_shipped_corrected_cells_in_one_batch(shipped_model):
    batch = shipped_model.lemmatize_batch_pos(
        [row[0] for row in SHIPPED_106], [row[1] for row in SHIPPED_106], guess=False)
    assert batch == [row[2] for row in SHIPPED_106]


@pytest.mark.parametrize("word", ["ἕκτος", "ἐκτός", "ἑπτά", "ἑξήκοντα"])
def test_shipped_numerals_have_no_multiplicative_adverb(shipped_model, word):
    assert not shipped_model.lemmatize_pos(word, "ADV", guess=False).endswith("άκις")


# #128: tagged answers that only Modern paradigm fragments used to reach.

SHIPPED_128 = [
    ("ας", "PART", "ας"),       # the Modern particle, not ὅς
    ("ονων", "NOUN", "ὄνος"),   # not the rare ὀνή of the stripped lsj9 key
    ("όντων", "NOUN", "ὄντα"),  # the Ancient noun; ων is a participle
    ("γ", "VERB", "γ"),         # no verb reading: the untagged answer
]


@pytest.fixture(scope="module")
def shipped_all_model():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    return core.Dilemma(lang="all")


@pytest.mark.parametrize("word,upos,lemma", SHIPPED_128)
def test_shipped_128_tagged_readings(shipped_all_model, word, upos, lemma):
    assert shipped_all_model.lemmatize_pos(word, upos, guess=False) == lemma
    assert shipped_all_model.lemmatize_batch_pos([word], [upos], guess=False) == [lemma]


def test_shipped_128_in_one_batch(shipped_all_model):
    assert shipped_all_model.lemmatize_batch_pos(
        [row[0] for row in SHIPPED_128], [row[1] for row in SHIPPED_128],
        guess=False) == [row[2] for row in SHIPPED_128]


def test_shipped_128_untagged_answers_unchanged(shipped_all_model):
    assert shipped_all_model.lemmatize_batch(
        ["ας", "ονων", "όντων", "γ"], guess=False) == ["ὅς", "ὄνος", "εἰμί", "γ"]
    assert shipped_all_model.lemmatize_pos("ας", "NOUN", guess=False) != "μονοτονία"
    assert shipped_all_model.lemmatize_pos("ὀνῶν", "NOUN", guess=False) == "ὀνή"


# #125, #126: tagged answers of 1.3.25 that the rebuilt tables displaced.

SHIPPED_125 = [
    # Cells the rebuilt treebank table dropped; a lower table answered.
    ("ουδέν", "ADJ", "οὐδείς"), ("δήτα", "ADV", "δῆτα"),
    ("ζώδιον", "NOUN", "ζῴδιον"), ("πλοιάριον", "NOUN", "πλοιάριον"),
    ("γραφών", "NOUN", "γραφή"), ("κίρρα", "NOUN", "Κίρρα"),
    ("Κίρρᾳ", "NOUN", "Κίρρα"), ("Ἅβραμος", "NOUN", "Ἄβραμος"),
    ("Θεσπρωτοὶ", "NOUN", "Θεσπρωτοί"), ("Κορκυραῖος", "ADJ", "Κορκυραῖος"),
    # New cells that replaced a 1.3.25 answer.
    ("εισήεσαν", "VERB", "εἴσειμι"), ("εἰσήεσαν", "VERB", "εἴσειμι"),
    ("συνήεσαν", "VERB", "σύνειμι"), ("ήμεν", "VERB", "εἰμί"),
    ("ἤμεν", "VERB", "εἰμί"), ("λόγιον", "ADJ", "λόγιος"),
    ("λόγιον", "NOUN", "λόγιον"),
    # Junk GLAUx lemmas the treebank table used to mask.
    ("καμέ", "PRON", "ἐγώ"), ("Δημοτέλη", "NOUN", "Δημοτέλης"),
    # No table has an SCONJ cell: ἤν "if" under its monotonic keys, and
    # the relative under the rough-breathing spellings.
    ("ην", "SCONJ", "ἐάν"), ("ήν", "SCONJ", "ἐάν"),
    ("ἣν", "SCONJ", "ὅς"), ("ἥν", "SCONJ", "ὅς"), ("ἣν", "PRON", "ὅς"),
    # A mis-accented treebank lemma.
    ("Λευκαδία", "ADJ", "Λευκάδιος"), ("λευκαδία", "ADJ", "Λευκάδιος"),
    # Names beside a restored lowercase head; the lowercase key keeps the
    # common word.
    ("Ἔρως", "NOUN", "Ἔρως"), ("ἔρως", "NOUN", "ἔρως"),
    ("Πρόξενος", "NOUN", "Πρόξενος"), ("πρόξενος", "NOUN", "πρόξενος"),
    ("Προξένου", "NOUN", "Πρόξενος"), ("προξένου", "NOUN", "πρόξενος"),
    ("Πρόξενον", "NOUN", "Πρόξενος"), ("Δρύας", "NOUN", "Δρύας"),
    ("Φιλάδελφος", "NOUN", "Φιλάδελφος"), ("φιλάδελφος", "ADJ", "φιλάδελφος"),
    ("Μέροπος", "NOUN", "Μέροψ"),
]


@pytest.mark.parametrize("word,upos,lemma", SHIPPED_125)
def test_shipped_125_tagged_readings(shipped_model, word, upos, lemma):
    assert shipped_model.lemmatize_pos(word, upos, guess=False) == lemma
    assert shipped_model.lemmatize_batch_pos([word], [upos], guess=False) == [lemma]


def test_shipped_125_in_one_batch(shipped_model):
    assert shipped_model.lemmatize_batch_pos(
        [row[0] for row in SHIPPED_125], [row[1] for row in SHIPPED_125],
        guess=False) == [row[2] for row in SHIPPED_125]


def test_shipped_125_untagged_answers_unchanged(shipped_model):
    # The rows answer a tag; the lookup reading stays what the build chose.
    assert shipped_model.lemmatize_batch(
        ["ην", "ήν", "λόγιον"], guess=False) == ["εἰμί", "εἰμί", "λόγιον"]


# The Modern cells answer under lang="el" and, for monotonic keys, "all".

SHIPPED_MODERN_126 = [
    ("θεα", "NOUN", "θεά"), ("θεά", "NOUN", "θεά"), ("θεὰ", "NOUN", "θεά"),
    ("παρέξω", "VERB", "παρέχω"), ("Παρέξω", "VERB", "παρέχω"),
    ("παρέξω", "ADV", "παρέξω"),  # the adverb page keeps its own tag
]


@pytest.fixture(scope="module", params=[None, "triantafyllidis"])
def shipped_el_model(request):
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    if request.param is None:
        return core.Dilemma(lang="el")
    return core.Dilemma(lang="el", convention=request.param)


@pytest.mark.parametrize("word,upos,lemma", SHIPPED_MODERN_126)
def test_shipped_modern_cells(shipped_el_model, word, upos, lemma):
    assert shipped_el_model.lemmatize_pos(word, upos, guess=False) == lemma
    assert shipped_el_model.lemmatize_batch_pos(
        [word], [upos], guess=False) == [lemma]


def test_shipped_modern_cell_answers_the_unaccented_key_under_all(
        shipped_all_model):
    assert shipped_all_model.lemmatize_pos("θεα", "NOUN", guess=False) == "θεά"
    assert shipped_all_model.lemmatize_batch_pos(
        ["θεα", "θεά"], ["NOUN", "NOUN"], guess=False) == ["θεά", "θεά"]
