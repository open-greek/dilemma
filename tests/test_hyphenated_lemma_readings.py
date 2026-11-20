"""Reviewed readings for the plain spellings left without a row when the
lookup stopped taking hyphenated corpus lemmas (#137, #149)."""

import json
import sqlite3
import unicodedata
from pathlib import Path

import pytest

from dilemma.citation import hyphenated_lemma

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "grc_hyphenated_lemma_readings.json"
HEADS = ROOT / "data" / "grc_lexical_heads.json"
LOOKUP = ROOT / "data" / "lookup.db"


@pytest.fixture(scope="module")
def readings():
    return json.loads(PATH.read_text(encoding="utf-8"))["readings"]


def test_records_are_complete_and_unique(readings):
    assert len(readings) == 11
    forms = [r["form"] for r in readings]
    assert len(forms) == len(set(forms))
    for r in readings:
        assert set(r) == {"form", "lemma", "source_lemma", "reason"}
        for key in ("form", "lemma"):
            assert r[key] == unicodedata.normalize("NFC", r[key])
            assert not hyphenated_lemma(r[key]) and "-" not in r[key]
        assert hyphenated_lemma(r["source_lemma"])
        assert len(r["reason"]) > 30


def test_every_lemma_is_a_listed_head(readings):
    if not HEADS.exists():
        pytest.skip("grc_lexical_heads.json not present")
    heads = json.loads(HEADS.read_text(encoding="utf-8"))["heads"]
    assert [r["lemma"] for r in readings if r["lemma"] not in heads] == []


def test_the_lookup_reads_each_form(readings):
    """Holds once build_lookup_db.py reads the file (central rebuild)."""
    if not LOOKUP.exists():
        pytest.skip("lookup.db not present")
    with sqlite3.connect(f"file:{LOOKUP}?mode=ro", uri=True) as db:
        got = {r["form"]: [row[0] for row in db.execute(
            "SELECT l.text FROM lookup k JOIN lemmas l ON k.lemma_id = l.id "
            "WHERE k.form = ? AND k.lang = 'all'", (r["form"],))]
            for r in readings}
    assert {form: lemmas for form, lemmas in got.items()
            if lemmas != [next(r["lemma"] for r in readings if r["form"] == form)]} == {}


def test_build_adds_a_reading_only_where_no_row_exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_lookup_db", ROOT / "build_lookup_db.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    combined = {"κατεῖχον": "κατέχω"}
    sources = {"κατεῖχον": "wiktionary_grc"}
    added = module.apply_hyphenated_lemma_readings(
        [{"form": "κατεῖχον", "lemma": "ἄλλο"},
         {"form": "ὑπεικαθέων", "lemma": "ὑπείκω"}],
        combined=combined, combined_sources=sources)
    assert added == ["ὑπεικαθέων"]
    assert combined == {"κατεῖχον": "κατέχω", "ὑπεικαθέων": "ὑπείκω"}
    assert sources["ὑπεικαθέων"] == module.LOOKUP_OVERRIDES_INPUT
