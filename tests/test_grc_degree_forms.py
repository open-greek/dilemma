"""Regular comparatives and superlatives of Ancient adjectives, kept where
the texts attest them (#152)."""
import json
from pathlib import Path
import sqlite3
import sys

import pytest

from dilemma.grc_degree_forms import (
    KIND, LONG, S_STEM, SHORT, SOURCE, degree_cells, degree_paradigms,
    rule_formations, statement_formations)
from dilemma.grc_noun_gaps import fill_gap_cells

ROOT = Path(__file__).resolve().parent.parent


def _forms(lemma, formation):
    return [form for form, _ in degree_cells(lemma, formation)]


def test_comparative_and_superlative_cells():
    cells = dict(degree_cells("ὁλικός", SHORT))
    assert cells["ὁλικώτερος"] == "comparative masculine nominative singular"
    assert cells["ὁλικωτάτων"].startswith("superlative genitive plural")
    for form in ("ὁλικωτέρου", "ὁλικωτέρῳ", "ὁλικώτερον", "ὁλικώτεροι",
                 "ὁλικωτέροις", "ὁλικωτέρους", "ὁλικωτέρα", "ὁλικωτέρας",
                 "ὁλικωτέρᾳ", "ὁλικωτέραν", "ὁλικώτεραι", "ὁλικωτέραις",
                 "ὁλικώτερα", "ὁλικώτατος", "ὁλικωτάτου", "ὁλικωτάτῳ",
                 "ὁλικώτατον", "ὁλικωτάτη", "ὁλικωτάτης", "ὁλικωτάτῃ",
                 "ὁλικωτάτην", "ὁλικώταται", "ὁλικωτάταις", "ὁλικωτάτας",
                 "ὁλικώτατα"):
        assert form in cells, form
    # No dual and no vocative singular, as in the other generated paradigms.
    assert "ὁλικωτέρω" not in cells and "ὁλικώτερε" not in cells
    assert _forms("σοφός", LONG)[0] == "σοφότερος"
    assert "ἀληθέστερον" in _forms("ἀληθής", S_STEM)
    assert "συνηθεστάτη" in _forms("συνήθης", S_STEM)
    assert "δικαιότατα" in _forms("δίκαιος", LONG)


@pytest.mark.parametrize("lemma, expected", [
    ("σοφός", (SHORT,)),            # ο before one consonant
    ("νέος", (SHORT,)),             # ε before the ending
    ("ἄξιος", (SHORT,)),            # the -ι- of -ιος is short
    ("ὁλικός", (SHORT,)),           # the suffix -ικός
    ("δίκαιος", (LONG,)),           # a diphthong
    ("κοῦφος", (LONG,)),
    ("δῆλος", (LONG,)),             # η
    ("λεπτός", (LONG,)),            # two consonants
    ("ἄκομψος", (LONG,)),           # a double consonant
    ("ἀγᾰθός", (SHORT,)),           # LSJ's breve
    ("ἀνιᾱρός", (LONG,)),           # LSJ's macron
    ("ἰσχυρός", (LONG, SHORT)),     # υ unmarked: the texts decide
    ("πικρός", (LONG, SHORT)),      # a stop and a liquid after a short vowel
    ("ἀληθής", (S_STEM,)),
    ("εὐώδης", (S_STEM,)),
])
def test_the_stem_rule(lemma, expected):
    assert rule_formations(lemma) == expected


def test_lsj_statements_settle_the_formation():
    text = ("ή, όν, universal ... cj. in Gal.19.194: Comp.\n"
            "   -ώτερον Phlp.in APo.121.6.")
    assert statement_formations("ὁλικός", [text]) == (SHORT,)
    assert statement_formations("ἀγαθός", ["Comp. ἀμείνων ... Comp. "
                                          "ἀγαθώτερος D.S.16.85"]) == (SHORT,)
    assert statement_formations("ἀκίνδῡνος", ["Comp. ἀκινδυνότερον"]) == (LONG,)
    assert statement_formations("ἄκρατος", ["Comp.ἀκρατέστερος"]) == (S_STEM,)
    # Another stem, and the comparative head that names its positive.
    assert statement_formations("ἀγαθός", ["Comp. ἀμείνων"]) == ()
    assert statement_formations("ἀγχότερος", ["Comp. of ἀγχοῦ"]) == ()


def _record(lemmas):
    return {"lemmas": {
        lemma: {"formations": formations,
                "cells": [[form, "", 1, {}] for form in forms]}
        for lemma, (formations, forms) in lemmas.items()}}


def test_paradigms_keep_the_attested_cells_of_a_head():
    record = _record({
        "ὁλικός": ([SHORT], ["ὁλικωτάτων", "ὁλικωτέρα"]),
        "ἀγνωστικός": ([SHORT], ["ἀγνωστικώτερον"]),      # no head in the table
    })
    lookup = {"ὁλικός": "ὁλικός"}
    rows = list(degree_paradigms(lookup, record))
    assert rows == [(KIND, "ὁλικός", [
        ("ὁλικωτέρα", "comparative feminine nominative/vocative singular"),
        ("ὁλικωτάτων", "superlative genitive plural")])]
    added = fill_gap_cells(lookup, rows, strict={KIND})
    assert lookup["ὁλικωτάτων"] == lookup["ὁλικωτέρα"] == "ὁλικός"
    # Accented keys only, as for the other rare-headword cells.
    assert "ολικωτατων" not in lookup
    assert {row["kind"] for row in added} == {KIND}


def test_a_degree_word_another_lemma_reads_is_left_out():
    record = _record({"κάτος": ([SHORT], ["κατώτερον", "κατωτέρᾳ",
                                          "κατωτάτῳ"])})
    lookup = {"κάτος": "κάτος", "κατώτερος": "κατώτερος",
              "κατώτατος": "κατώτατος"}
    dropped = []
    assert list(degree_paradigms(lookup, record, dropped=dropped)) == []
    assert {row[1] for row in dropped} == {"κατώτερος", "κατώτατος"}


def test_a_name_and_a_modern_reading_of_the_same_adjective_are_no_rival():
    record = _record({"δίκαιος": ([LONG], ["δικαιότερον", "δικαιοτάτην"])})
    lookup = {"δίκαιος": "δίκαιος", "δικαιότερος": "Δίκαιος",
              "δικαιοτάτη": "δικαιος"}
    rows = list(degree_paradigms(
        lookup, record, lambda reading, lemma: reading in (lemma, "δικαιος")))
    assert [form for form, _ in rows[0][2]] == ["δικαιότερον", "δικαιοτάτην"]


def test_reviewed_not_declined_lemmas_get_no_degree_forms():
    record = _record({"μεγάλος": ([SHORT], ["μεγαλώτερον"])})
    assert list(degree_paradigms({"μεγάλος": "μεγάλος"}, record)) == []


def _write_profile(path, forms):
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE forms (form_id INTEGER PRIMARY KEY, form TEXT, form_norm TEXT);"
        "CREATE TABLE form_profile (form_id INTEGER PRIMARY KEY, total_count INTEGER,"
        " source_counts_json TEXT);"
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);"
        "INSERT INTO meta VALUES ('content_hash', 'fixture');")
    from dilemma._attest_db import norm_key
    for i, (form, count) in enumerate(forms.items(), 1):
        connection.execute("INSERT INTO forms VALUES (?, ?, ?)",
                           (i, form, norm_key(form)))
        connection.execute("INSERT INTO form_profile VALUES (?, ?, ?)",
                           (i, count, json.dumps({"glaux": count})))
    connection.commit()
    connection.close()


def test_builder_keeps_attested_unambiguous_cells(tmp_path):
    sys.path.insert(0, str(ROOT / "build"))
    import build_grc_degree_forms as B
    lsj9 = tmp_path / "lsj9"
    lsj9.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    (lsj9 / "lsj9_headwords.json").write_text(json.dumps([
        {"headword": "ὁλικός"}, {"headword": "ἀγᾰθός"}], ensure_ascii=False),
        encoding="utf-8")
    (lsj9 / "lsj9_glosses.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in [
            {"headword": "ὁλικός", "text": "ή, όν, universal: Comp. -ώτερον"},
            {"headword": "ἀγᾰθός", "text": "ή, όν, good"}]), encoding="utf-8")
    (data / "grc_second_declension.json").write_text(json.dumps({"classes": {
        "adjective_eta": {"lsj9": ["ἀγαθός", "ὁλικός"]},
        "adjective": {"lbg": ["εὐπρόσοδος", "εὐπροσόδος", "μεγάλος"]}}},
        ensure_ascii=False), encoding="utf-8")
    (data / "grc_ikos_adjectives.json").write_text(json.dumps(
        {"lemmas": []}), encoding="utf-8")
    (data / "grc_es_adjectives.json").write_text(json.dumps(
        {"lemmas": {"lsj9": ["ἀληθής"]}}, ensure_ascii=False), encoding="utf-8")
    _write_profile(data / "form_profile.db", {
        "ὁλικωτάτων": 1, "Ὁλικωτέρα": 2, "ὁλικότερον": 3,   # wrong formation
        "ἀγαθώτερος": 4, "ἀληθέστερον": 5, "εὐπροσοδώτερον": 6,
        "μεγαλώτερον": 7})
    record = B.build(lsj9, data, B.Attestation(data / "form_profile.db"))
    cells = B.read_cells(record)
    assert sorted(form for form, _ in cells["ὁλικός"]) == sorted([
        "ὁλικωτέρα", "ὁλικωτάτων"])
    assert record["lemmas"]["ὁλικός"]["formation_evidence"] == "lsj9_statement"
    assert record["lemmas"]["ἀγαθός"]["formation_evidence"] == "stem_rule_lsj9_marks"
    assert [form for form, _ in cells["ἀληθής"]] == ["ἀληθέστερον"]
    assert record["ambiguous"] == {"εὐπροσοδώτερον": ["εὐπροσόδος", "εὐπρόσοδος"]}
    assert "μεγάλος" not in cells
    attested = {row[0]: row[2:] for row in record["lemmas"]["ὁλικός"]["cells"]}
    assert attested["ὁλικωτέρα"] == [2, {"glaux": 2}]
    assert record["inputs"]["form_profile.db"] == {"content_hash": "fixture"}


def test_committed_record_and_source():
    record = json.loads((ROOT / "data" / "grc_degree_forms.json").read_text(
        encoding="utf-8"))
    sys.path.insert(0, str(ROOT / "build"))
    import build_grc_degree_forms as B
    assert record["rule"] == B.RULE
    cells = B.read_cells(record)
    assert ("ὁλικωτάτων", "superlative genitive plural") in cells["ὁλικός"]
    assert record["counts"]["cells"] == sum(map(len, cells.values()))
    for lemma, entry in record["lemmas"].items():
        generated = {form for formation in entry["formations"]
                     for form, _ in degree_cells(lemma, formation)}
        assert {row[0] for row in entry["cells"]} <= generated, lemma
    sources = json.loads((ROOT / "data" / "lookup_sources.json").read_text(
        encoding="utf-8"))["sources"]
    assert SOURCE in {source["code"] for source in sources}


# Shipped answers (need the lookup rebuild) ---------------------------------

def test_shipped_degree_forms_read_the_positive():
    from dilemma import core
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    with sqlite3.connect(f"file:{core.LOOKUP_DB_PATH}?mode=ro", uri=True) as conn:
        rows = dict(conn.execute(
            "SELECT l.form, m.text || '|' || s.code FROM lookup l "
            "JOIN lemmas m ON m.id = l.lemma_id JOIN sources s "
            "ON s.id = l.source_id WHERE l.lang = 'all' AND l.form IN "
            "('ὁλικωτάτων', 'ὁλικωτάτῳ', 'ὁλικώτατα', 'ὁλικωτέρα')"))
    assert rows == {form: f"ὁλικός|{SOURCE}" for form in
                    ("ὁλικωτάτων", "ὁλικωτάτῳ", "ὁλικώτατα", "ὁλικωτέρα")}
