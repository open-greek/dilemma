"""Nominal readings that the LSJ verb cells leave alone (#145)."""
import json
import sqlite3
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))
import expand_lsj  # noqa: E402


def nfc(text):
    return unicodedata.normalize("NFC", text)


def test_first_declension_cells_have_the_ionic_eta():
    cells = expand_lsj.first_declension_cells("πρασιά")
    assert nfc("πρασιῇ") in cells
    assert nfc("πρασιᾷ") in cells
    assert nfc("πρασιῶν") in cells
    # Barytone and unaccented words are not declined here.
    assert expand_lsj.first_declension_cells("μοῦσα") == []
    assert expand_lsj.first_declension_cells("πρασια") == []


def test_nominal_cells_of_lsj_heads(tmp_path):
    second = tmp_path / "second.json"
    second.write_text(json.dumps({"classes": {"adjective": {
        "lsj9": ["ἐπιπόθητος", "ἀναίσιμος"]}}}, ensure_ascii=False),
        encoding="utf-8")
    entries = {"ἐπικλινής": {"gender": "", "itype": "ές"},
               "πρασιά": {"gender": "f", "itype": ""},
               "ἐπικλίνω": {"gender": "", "itype": ""}}
    cells = expand_lsj.nominal_cells(entries, second)
    assert cells[nfc("ἐπικλινεῖς")] == "ἐπικλινής"
    assert cells[nfc("ἐπικλινεῖ")] == "ἐπικλινής"
    assert cells[nfc("πρασιῇ")] == "πρασιά"
    assert cells[nfc("ἐπιπόθητον")] == "ἐπιπόθητος"
    assert cells[nfc("ἀναισίμους")] == "ἀναίσιμος"
    # The accent tells the participle περιχαρείς from περιχαρεῖς, and the
    # subjunctive ὑποφύγῃ from ὑποφυγῇ.
    assert nfc("ἐπικλινείς") not in cells


def test_nominal_lemma():
    heads = frozenset({"ἀναίσιμος", "Αἰγιεύς", "κατεῖπον", "ἐπικλίνω",
                       "πρότυπον"})
    entries = {"κατεῖπον": {"gender": "", "itype": ""},
               "ἀναίσιμος": {"gender": "", "itype": "ον"}}
    assert expand_lsj.nominal_lemma("ἀναίσιμος", heads, entries)
    assert expand_lsj.nominal_lemma("Αἰγιεύς", heads, entries)
    assert expand_lsj.nominal_lemma("πρότυπον", heads, entries)
    # A verb, an aorist head of LSJ and a word no dictionary heads are not.
    assert not expand_lsj.nominal_lemma("ἐπικλίνω", heads, entries)
    assert not expand_lsj.nominal_lemma("κατεῖπον", heads, entries)
    assert not expand_lsj.nominal_lemma("κατεσίγνυμυ", heads, entries)
    assert not expand_lsj.nominal_lemma(None, heads, entries)


def test_treebank_names_of_capitalised_spellings(tmp_path):
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps([
        {"form": "Αἰγιεῖς", "lemma": "Αἰγιεύς", "pos": "noun"},
        {"form": "Ἐφέσω", "lemma": "Ἔφεσος", "pos": "noun"},
        {"form": "Ἐκόσμησε", "lemma": "κοσμέω", "pos": "verb"}],
        ensure_ascii=False), encoding="utf-8")
    names = {}
    readings = expand_lsj.load_treebank_readings(
        [path], {"αἰγιεῖς", "ἐφέσω", "ἐκόσμησε"}, names)
    key = expand_lsj.reading_key
    assert names == {key("αἰγιεῖς").casefold(): "Αἰγιεύς",
                     key("ἐφέσω").casefold(): "Ἔφεσος"}
    assert readings == {key("ἐκόσμησε"): [("κοσμέω", "verb")]}


def test_lowercase_keys(tmp_path):
    path = tmp_path / "form_profile.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE forms (form_id INTEGER PRIMARY KEY, form TEXT, "
        "form_norm TEXT)")
    connection.executemany(
        "INSERT INTO forms (form, form_norm) VALUES (?, ?)",
        [("Αἰγιεῖς", "αἰγιεῖς"), ("ἐκόσμησε", "ἐκόσμησε"),
         ("Ἐκόσμησε", "ἐκόσμησε")])
    connection.commit()
    connection.close()
    assert expand_lsj.load_lowercase_keys(path) == frozenset({"ἐκόσμησε"})
