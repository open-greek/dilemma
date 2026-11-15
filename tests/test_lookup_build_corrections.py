"""Lookup build corrections: expansion artefacts (#103), reviewed non-lemmas
and capitalised corpus twins (#108), and reviewed lookup overrides (#110).

The builder runs on a small fixture in a temporary data directory; no
production data is read or written.
"""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_lookup_db as B  # noqa: E402


# Pure helper ---------------------------------------------------------------

HEADS = frozenset({"τε", "σύνοχος", "μός", "κλεῖδα", "μόνος", "μονός"})
BY_LETTERS = {}
for _head in HEADS:
    BY_LETTERS.setdefault(B._bare_letters(_head), set()).add(_head)


@pytest.mark.parametrize("lemma,verdict", [
    ("λαβλυμι", ("unaccented_expansion_lemma_unlisted", None)),
    ("αελιυς", ("unaccented_expansion_lemma_unlisted", None)),
    ("σερvιλια", ("unaccented_expansion_lemma_unlisted", None)),
    ("προ-ανακηρυσσω", ("unaccented_expansion_lemma_unlisted", None)),
    ("λαβ’μι", ("unaccented_expansion_lemma_unlisted", None)),
    ("συνοχος", ("unaccented_expansion_lemma_repaired", "σύνοχος")),
    ("μος", ("unaccented_expansion_lemma_repaired", "μός")),
    # Two heads with these letters: no repair.
    ("μονος", ("unaccented_expansion_lemma_unlisted", None)),
    ("τε", None),
    ("λαβή", None),
    ("Ῥώμη", None),
])
def test_unaccented_expansion_lemma(lemma, verdict):
    assert B.unaccented_expansion_lemma(lemma, HEADS, BY_LETTERS) == verdict


def test_repair_target_must_be_a_citation_head():
    # κλεῖδα is a head-list cross-reference entry (an accusative of κλείς),
    # not a lemma the lookup maps to itself: the reading is dropped.
    citations = {"σύνοχος"}
    judge = lambda lemma: B.unaccented_expansion_lemma(
        lemma, HEADS, BY_LETTERS, citations.__contains__)
    assert judge("κλειδα") == ("unaccented_expansion_lemma_head_not_citation", None)
    assert judge("συνοχος") == ("unaccented_expansion_lemma_repaired", "σύνοχος")
    assert judge("λαβλυμι") == ("unaccented_expansion_lemma_unlisted", None)
    assert judge("τε") is None


def test_expansion_heads_index_keeps_case_and_ancient_heads_only(tmp_path):
    path = tmp_path / "heads.json"
    path.write_text(json.dumps({"heads": ["Βάβυρσα", "σύνοχος"],
                                "vernacular_heads": ["μπαλκόνι"]},
                               ensure_ascii=False), encoding="utf-8")
    heads, by_letters = B.load_expansion_lemma_heads(path)
    assert heads == {"Βάβυρσα", "σύνοχος", "μπαλκόνι"}
    assert by_letters == {"Βαβυρσα": {"Βάβυρσα"}, "συνοχος": {"σύνοχος"}}


# Builder on a fixture ------------------------------------------------------

SCRIPT = """
from functools import partial
from pathlib import Path
import sys
import build_lookup_db as builder

data = Path(sys.argv[1])
original_data = builder.DATA_DIR
def relocate(value):
    if isinstance(value, Path) and value.parent == original_data:
        return data / value.name
    if isinstance(value, (tuple, list)):
        return type(value)(relocate(item) for item in value)
    return value
for name, value in list(vars(builder).items()):
    setattr(builder, name, relocate(value))
builder.DATA_DIR = data
builder._write_citation_rejection_report = partial(
    builder._write_citation_rejection_report,
    path=builder.CITATION_REJECTIONS_PATH,
)
builder.build()
"""

# Readings of the dictionary expansion: keys of the reference that the
# Wiktionary base lacks.
EXPANSION = {
    "λαβῃ": "λαβλυμι",          # OCR garble, in no head list: dropped
    "κληϊδα": "κλειδα",          # the one same-letter head is no citation head
    "Αἴλιοι": "αελιυς",          # Latin read as Beta Code: dropped
    "σύνοχα": "συνοχος",         # unaccented doublet: repaired
    "γε": "γε",                  # a listed unaccented head: kept
    "μοναχικοῦ": "μοναχικός",    # accented: kept
}
WIKTIONARY = {"λόγος": "λόγος", "σύνοχος": "σύνοχος", "κλεῖδα": "κλείς"}


def _fixture(tmp_path, extra=None):
    data = tmp_path / "data"
    data.mkdir()
    ag = {**WIKTIONARY, **EXPANSION, **(extra or {}).pop("ag", {})}
    sources = {
        "ag_lookup.json": ag,
        "mg_lookup.json": {"λόγος": "λόγος"},
        "med_lookup.json": {},
        "grc_lexical_heads.json": {"heads": ["σύνοχος", "γε", "λόγος",
                                             "μοναχικός", "λαβή", "κλεῖδα"],
                                   "vernacular_heads": []},
        **(extra or {}),
    }
    for name, value in sources.items():
        (data / name).write_text(json.dumps(value, ensure_ascii=False),
                                 encoding="utf-8")
    (data / "lookup_sources.json").write_bytes(
        (ROOT / "data" / "lookup_sources.json").read_bytes())
    base = data / "ag_lookup_wikt.json"
    base.write_text(json.dumps(WIKTIONARY, ensure_ascii=False), encoding="utf-8")
    reference = tmp_path / "expansion_reference.json"
    reference.write_text(json.dumps({**WIKTIONARY, **EXPANSION},
                                    ensure_ascii=False), encoding="utf-8")
    pin = lambda path: {"size": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (data / "ag_expansion_reference.json").write_text(json.dumps({
        "schema_version": 1, "reference": pin(reference),
        "reference_base": pin(base)}))
    (data / "changes").mkdir()
    for name in ("el_present_completion.json", "el_imperfect_completion.json"):
        (data / "changes" / name).write_text(json.dumps({"lookup_additions": []}))
    env = {**os.environ, "DILEMMA_AG_EXPANSION_REFERENCE": str(reference),
           "PYTHONHASHSEED": "0"}
    return data, env


def _build(data, env):
    result = subprocess.run([sys.executable, "-c", SCRIPT, str(data)], cwd=ROOT,
                            env=env, capture_output=True, text=True, timeout=60)
    return result


def _rows(data, lang="all"):
    with sqlite3.connect(data / "lookup.db") as conn:
        return dict(conn.execute(
            "SELECT form, lemmas.text FROM lookup JOIN lemmas "
            "ON lookup.lemma_id = lemmas.id WHERE lang = ?", (lang,)))


def _rejections(data):
    lines = (data / "citation_hygiene_rejections.tsv").read_text(
        encoding="utf-8").splitlines()[1:]
    return {tuple(line.split("\t")) for line in lines}


def test_unaccented_expansion_lemmas_are_dropped_or_repaired(tmp_path):
    data, env = _fixture(tmp_path)
    result = _build(data, env)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = _rows(data)
    assert "λαβῃ" not in rows and "Αἴλιοι" not in rows
    assert rows["σύνοχα"] == "σύνοχος"
    assert rows["γε"] == "γε" and rows["μοναχικοῦ"] == "μοναχικός"
    assert "λαβλυμι" not in set(rows.values()) | set(_rows(data, "grc").values())
    rejected = _rejections(data)
    layer = B.AG_EXPANSION_LAYER
    assert ("AG", layer, "unaccented_expansion_lemma_unlisted", "λαβῃ",
            "λαβλυμι") in rejected
    assert ("AG", layer, "unaccented_expansion_lemma_repaired_to_σύνοχος",
            "σύνοχα", "συνοχος") in rejected
    assert "κληϊδα" not in rows
    assert ("AG", layer, "unaccented_expansion_lemma_head_not_citation",
            "κληϊδα", "κλειδα") in rejected


def test_corpus_pairs_cannot_bring_back_a_dropped_expansion_lemma(tmp_path):
    # GLAUx annotates λαβῃ with the expansion's OCR lemma λαβλυμι, and
    # ag_headwords.json lists it, so headword validation alone let the pair
    # refill the key the #103 check emptied.
    glaux = [{"form": "λαβῃ", "lemma": "λαβλυμι", "pos": "verb"},
             {"form": "συνόχου", "lemma": "συνοχος", "pos": "adj"}]
    diorisis = [{"form": "λαβῇ", "lemma": "λαβλυμι", "pos": "verb"}]
    data, env = _fixture(tmp_path, {
        "ag_headwords.json": ["λαβλυμι", "συνοχος", "σύνοχος", "λόγος"],
        "glaux_pairs.json": glaux,
        "diorisis_pairs.json": diorisis,
    })
    result = _build(data, env)
    assert result.returncode == 0, result.stdout + result.stderr
    for lang in ("all", "grc", "el"):
        rows = _rows(data, lang)
        assert "λαβλυμι" not in set(rows.values())
        assert "λαβῃ" not in rows and "λαβῇ" not in rows
    assert _rows(data)["συνόχου"] == "σύνοχος"
    rejected = _rejections(data)
    assert ("AG", B.GLAUX_PAIRS_PATH.name, "unaccented_expansion_lemma_unlisted",
            "λαβῃ", "λαβλυμι") in rejected
    assert ("AG", B.DIORISIS_PAIRS_PATH.name, "unaccented_expansion_lemma_unlisted",
            "λαβῇ", "λαβλυμι") in rejected


def test_missing_lexical_heads_stop_a_build_with_expansion_readings(tmp_path):
    data, env = _fixture(tmp_path)
    (data / "grc_lexical_heads.json").unlink()
    result = _build(data, env)
    assert result.returncode != 0
    assert "grc_lexical_heads.json is missing" in result.stderr


# #108: reviewed non-lemmas and capitalised pronoun twins -------------------

def test_capitalized_pronoun_twin_rule():
    ag = {"αὕτη": "οὗτος", "οἰκία": "οἰκία", "πάντα": "πᾶς", "τούτου": "οὗτος"}
    corpus = {"τούτου"}
    tables = ((ag, lambda f: f in corpus),)
    twin = B.capitalized_pronoun_twin
    assert twin("Αὕτη", "ἄτη", tables) == "οὗτος"
    assert twin("Πάντα", "πάν", tables) == "πᾶς"
    # Same lemma, a lowercase form, a name lemma: nothing to skip.
    assert twin("Αὕτη", "οὗτος", tables) is None
    assert twin("αὕτη", "ἄτη", tables) is None
    assert twin("Αὕτη", "Ἄτη", tables) is None
    # The twin is no pronoun, or its reading came from a corpus.
    assert twin("Οἰκία", "οἶκος", tables) is None
    assert twin("Τούτου", "τοῦτος", tables) is None


def test_reviewed_record_lists_reason_and_evidence():
    from dilemma.grc_rejected_lemmas import (REJECTED_LEMMA_TARGETS,
                                             REJECTED_LEMMAS)
    assert {"Ταύτη", "Τοῦτος", "Ἡμων", "Ταύτης"} <= REJECTED_LEMMAS
    assert not {"τοῦτος", "ἄτη", "πάν", "οὗτος"} & REJECTED_LEMMAS
    for entry in REJECTED_LEMMA_TARGETS.values():
        assert entry["reason"] and entry["evidence"]


def test_reviewed_non_lemmas_and_pronoun_twins_leave_the_lookup(tmp_path):
    glaux = [
        {"form": "Αὕτη", "lemma": "Ἄτη", "pos": "noun"},
        {"form": "Οἰκία", "lemma": "οἰκία", "pos": "noun"},
        {"form": "Ταύτῃ", "lemma": "Ταύτη", "pos": "noun"},
        # Reviewed, though a head list has its lowercase ταύτης.
        {"form": "Ταύτῃ", "lemma": "Ταύτης", "pos": "noun"},
        {"form": "Ταύτῃ", "lemma": "οὗτος", "pos": "pron"},
    ]
    diorisis = [{"form": "Ἡμῶν", "lemma": "ἡμός", "pos": "pron"}]
    data, env = _fixture(tmp_path, {
        "ag": {"αὕτη": "οὗτος", "οἰκία": "οἰκία", "Ταύτη": "Ταύτη",
               "ἡμῶν": "ἐγώ", "ἄτη": "ἄτη"},
        "ag_headwords.json": ["ἄτη", "οὗτος", "οἰκία", "ἡμός", "ἐγώ", "λόγος",
                              "ταύτης"],
        "glaux_pairs.json": glaux,
        "diorisis_pairs.json": diorisis,
    })
    result = _build(data, env)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = _rows(data)
    assert "Αὕτη" not in rows and rows["αὕτη"] == "οὗτος"
    assert "Ἡμῶν" not in rows and rows["ἄτη"] == "ἄτη"
    assert rows["Οἰκία"] == "οἰκία"
    assert rows["Ταύτῃ"] == "οὗτος"
    assert "Ταύτη" not in rows
    assert ("AG", "ag_lookup.json", "reviewed_rejected_target", "Ταύτη",
            "Ταύτη") in _rejections(data)
    assert "1 capitalised pronoun twins skipped" in result.stdout


def test_glaux_pos_builder_skips_reviewed_non_lemmas(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_glaux_pos_lookup", ROOT / "build" / "build_glaux_pos_lookup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pairs = [
        {"form": "ταύτην", "lemma": "Ταύτη", "pos": "noun"},
        {"form": "ταύτην", "lemma": "οὗτος", "pos": "pron"},
        {"form": "λόγον", "lemma": "λόγος", "pos": "noun"},
        {"form": "λόγον", "lemma": "λέγω", "pos": "verb"},
    ]
    module.GLAUX_PAIRS = tmp_path / "pairs.json"
    module.GLAUX_PAIRS.write_text(json.dumps(pairs, ensure_ascii=False),
                                  encoding="utf-8")
    module.OUTPUT_PATH = tmp_path / "out.json"
    lookup = module.build_lookup()
    assert "ταύτην" not in lookup
    assert lookup["λόγον"] == {"NOUN": "λόγος", "VERB": "λέγω"}


# #110: reviewed lookup overrides --------------------------------------------

def test_merged_layer_reassignments_are_overridden(tmp_path):
    data, env = _fixture(tmp_path, {
        "ag": {"γραφῆς": "γραφεύς", "γραφή": "γραφεύς", "γραφέα": "γραφεύς",
               "σεῖο": "σῦς", "σύ": "σῦς", "συός": "σῦς",
               "ἱερέων": "ἱερός", "ἄριστα": "ἄριστον", "ἀρὰν": "ἀρή"},
    })
    result = _build(data, env)
    assert result.returncode == 0, result.stdout + result.stderr
    combined = _rows(data)
    for lang in ("all", "grc"):
        # A grc row is written only where it differs from the combined one.
        rows = {**combined, **_rows(data, lang)}
        assert rows["γραφῆς"] == rows["γραφή"] == "γραφή"
        assert rows["σεῖο"] == rows["σύ"] == "σύ"
        assert rows["ἱερέων"] == "ἱερεύς" and rows["ἄριστα"] == "ἀγαθός"
        assert rows["ἀρὰν"] == "ἀρά"
        # The other words' own forms are untouched.
        assert rows["γραφέα"] == "γραφεύς" and rows["συός"] == "σῦς"


# Reviewed readings against el-only corpus rows and absent keys --------------

def test_reviewed_readings_displace_el_only_corpus_rows(tmp_path):
    # The Ancient table agrees with a corpus pair (Μάγων, εἴσομαι, ᾔει) or
    # lacks the form (ἀπεικός), so the gap-fill copies the pair into the el
    # table before the overrides correct the combined and Ancient tables.
    glaux = [{"form": "Μάγων", "lemma": "Μάγων", "pos": "noun"},
             {"form": "ἀπεικός", "lemma": "ἀφίημι", "pos": "verb"},
             {"form": "ᾔει", "lemma": "ἔρχομαι", "pos": "verb"}]
    diorisis = [{"form": "εἴσομαι", "lemma": "εἶμι", "pos": "verb"}]
    data, env = _fixture(tmp_path, {
        "ag": {"Μάγων": "Μάγων", "εἴσομαι": "εἶμι", "ᾔει": "ἔρχομαι",
               "μεγάλα": "μέγα"},
        "mg_lookup.json": {"λόγος": "λόγος", "μεγάλα": "μεγάλος"},
        "ag_headwords.json": ["Μάγων", "μάγος", "ἀφίημι", "ἔρχομαι", "εἶμι",
                              "οἶδα", "λόγος"],
        "glaux_pairs.json": glaux,
        "diorisis_pairs.json": diorisis,
    })
    result = _build(data, env)
    assert result.returncode == 0, result.stdout + result.stderr
    combined, el = _rows(data), _rows(data, "el")
    assert combined["Μάγων"] == "μάγος" and combined["εἴσομαι"] == "οἶδα"
    assert combined["ἀπεικός"] == "ἀπεικός" and combined["μεγάλα"] == "μέγας"
    # lang="el" reads its own row first, then the combined one.
    answers = {**combined, **el}
    assert answers["Μάγων"] == "μάγος" and answers["εἴσομαι"] == "οἶδα"
    assert answers["ἀπεικός"] == "ἀπεικός"
    assert not {"Μάγων", "εἴσομαι", "ἀπεικός"} & el.keys()
    # A Modern Wiktionary reading keeps its el row, and so does the corpus
    # row of an override the published tables already ship (ᾔει -> εἶμι).
    assert el["μεγάλα"] == "μεγάλος"
    assert combined["ᾔει"] == "εἶμι" and el["ᾔει"] == "ἔρχομαι"
    assert "3 el-only corpus readings removed" in result.stdout


def test_reviewed_absent_key_reading_is_added(tmp_path):
    data, env = _fixture(tmp_path)
    result = _build(data, env)
    assert result.returncode == 0, result.stdout + result.stderr
    # No source of the fixture has the bare elided stem.
    assert _rows(data)["μήτ"] == "μήτε"
    assert "μήτ" not in _rows(data, "grc") and "μήτ" not in _rows(data, "el")
    with sqlite3.connect(data / "lookup.db") as conn:
        assert conn.execute(
            "SELECT s.code FROM lookup l JOIN sources s ON s.id = l.source_id "
            "WHERE l.form = 'μήτ'").fetchall() == [("dilemma_corrections",)]


def test_modern_spellings_keep_their_modern_reading_under_all(tmp_path):
    # The Ancient table gained monotonic alias keys of Ancient lemmas, and
    # the combined table is Ancient-first.
    data, env = _fixture(tmp_path, {
        "ag": {"αυτοκινήτου": "αὐτοκίνητος", "αὐτοκίνητος": "αὐτοκίνητος",
               "Βιέννης": "Βίεννα"},
        "mg_lookup.json": {"λόγος": "λόγος", "αυτοκίνητο": "αυτοκίνητο",
                           "αυτοκινήτου": "αυτοκίνητο"},
    })
    result = _build(data, env)
    assert result.returncode == 0, result.stdout + result.stderr
    combined, grc, el = _rows(data), _rows(data, "grc"), _rows(data, "el")
    assert combined["αυτοκινήτου"] == "αυτοκίνητο"
    assert combined["Βιέννης"] == "Βιέννη"
    # lang="grc" still reads the Ancient lemma; lang="el" needs no row of
    # its own, because the combined row is the Modern one.
    assert grc["αυτοκινήτου"] == "αὐτοκίνητος" and grc["Βιέννης"] == "Βίεννα"
    assert "αυτοκινήτου" not in el and "Βιέννης" not in el
    assert combined["αὐτοκίνητος"] == "αὐτοκίνητος"
    with sqlite3.connect(data / "lookup.db") as conn:
        rows = dict(conn.execute(
            "SELECT l.form, l.src || ' ' || s.code FROM lookup l "
            "JOIN sources s ON s.id = l.source_id WHERE l.lang = 'all'"))
    # The row is the Modern table's own where that table has the reading.
    assert rows["αυτοκινήτου"] == "el wiktionary_el"
    assert rows["Βιέννης"] == "el dilemma_corrections"
    assert "Modern readings kept under lang=all: 2" in result.stdout


# #126: lexical heads displaced by another lemma's homograph entry ----------

def test_lexical_heads_by_kind_separates_wiktionary_entries(tmp_path):
    path = tmp_path / "heads.json"
    path.write_text(json.dumps({"heads": {
        "κλαίω": ["lsj9_headwords", "wiktionary_en_grc"],
        "ἕως": ["wiktionary_el_grc"],
        "σπαθίον": ["lsj9_headwords", "pd_headwords"],
        "ἄπειρος": ["wiktionary_el_medieval"]}}, ensure_ascii=False),
        encoding="utf-8")
    wiktionary, listed = B.lexical_heads_by_kind(path)
    assert wiktionary == {"κλαίω", "ἕως"}
    assert listed == {"κλαίω", "ἕως", "σπαθίον", "ἄπειρος"}
    # The list form of older fixtures names no source: no head qualifies.
    path.write_text(json.dumps({"heads": ["κλαίω"]}), encoding="utf-8")
    assert B.lexical_heads_by_kind(path) == (frozenset(), frozenset())


def test_corpus_form_evidence_sums_counts_of_listed_forms():
    pairs = [{"form": "ἕως", "lemma": "ἕως", "count": 7},
             {"form": "ἕως", "lemma": "ἠώς", "count": 1},
             {"form": "ἕως", "lemma": "ἕως", "count": 2},
             {"form": "λόγος", "lemma": "λόγος", "count": 9},
             {"form": "θρόνον", "lemma": "θρόνος"}]
    assert B.corpus_form_evidence(pairs, {"ἕως", "θρόνον"}) == {
        "ἕως": {"ἕως": 9, "ἠώς": 1}, "θρόνον": {"θρόνος": 1}}


def test_restore_displaced_lexical_heads_follows_corpus_then_base():
    wiktionary = {"κλαίω", "ἕως", "θρόνον", "ἀγγελθείης", "ἄπειρος", "ἄπειρον",
                  "μήν", "μῆν"}
    listed = {"σπαθίον", "ποδίσκος", "ἠρίσταμεν", "νηός"}
    ag = {"κλαίω": "κλείω", "κλαιω": "κλείω",       # base self-map, no corpus
          "ἕως": "ἠώς", "εως": "ἠώς", "έως": "ἠώς",  # corpus prefers the head
          "θρόνον": "θρόνος",                         # corpus prefers the lemma
          "ἀγγελθείης": "ἀγγέλλω",                    # neither supports the head
          "ἄπειρος": "ἤπειρος", "ἄπειρον": "ἤπειρος",
          "απειρος": "ἤπειρος",
          "μήν": "μείς", "μῆν": "μείς", "μην": "μείς",  # two heads, one alias
          "σπαθίον": "σπάθη",      # dictionary head, base and corpus agree
          "ποδίσκος": "πούς",      # dictionary head, base only: stays
          "ἠρίσταμεν": "ἀριστάω",  # an inflected form a list holds: stays
          "νηός": "ναός",          # corpus prefers the head, base does not
          "λόγος": "λόγος"}
    base = {"κλαίω": "κλαίω", "θρόνον": "θρόνον", "ἄπειρος": "ἄπειρος",
            "ἄπειρον": "ἄπειρον", "σπαθίον": "σπαθίον", "ποδίσκος": "ποδίσκος",
            "ἠρίσταμεν": "ἠρίσταμεν"}
    evidence = {"ἕως": {"ἕως": 5399, "ἠώς": 1}, "θρόνον": {"θρόνος": 201},
                "μήν": {"μήν": 587, "μείς": 9}, "μῆν": {"μῆν": 3},
                "σπαθίον": {"σπαθίον": 2}, "ἠρίσταμεν": {"ἀριστάω": 1},
                "νηός": {"νηός": 5}}
    restored = B.restore_displaced_lexical_heads(
        ag, wiktionary_heads=wiktionary, listed_heads=listed, base=base,
        evidence=evidence)
    assert ag["κλαίω"] == ag["κλαιω"] == "κλαίω"
    assert ag["ἕως"] == ag["εως"] == ag["έως"] == "ἕως"
    assert ag["θρόνον"] == "θρόνος" and ag["ἀγγελθείης"] == "ἀγγέλλω"
    assert ag["ἄπειρος"] == "ἄπειρος" and ag["ἄπειρον"] == "ἄπειρον"
    # Both restored heads claim the unaccented key: it keeps its reading.
    assert ag["μήν"] == "μήν" and ag["μῆν"] == "μῆν" and ag["μην"] == "μείς"
    assert ag["απειρος"] == "ἄπειρος"
    assert ag["σπαθίον"] == "σπαθίον"
    assert ag["ποδίσκος"] == "πούς" and ag["ἠρίσταμεν"] == "ἀριστάω"
    assert ag["νηός"] == "ναός"
    by_form = {record["form"]: record for record in restored}
    assert by_form["ἕως"] == {"form": "ἕως", "lemma": "ἠώς", "head": "ἕως",
                              "head_count": 5399, "lemma_count": 1}
    assert by_form["εως"]["head"] == "ἕως" and by_form["εως"]["head_count"] is None
    assert "θρόνον" not in by_form and "μην" not in by_form
    assert [record["form"] for record in restored] == sorted(by_form)


def test_restored_head_aliases_need_the_corpus_reading_of_the_whole_key():
    ag = {"γυναικών": "γυνή", "γυναικων": "γυνή",
          "ῥᾶ": "ῥεῖα", "ρα": "ῥεῖα",
          "ἕως": "ἠώς", "εως": "ἠώς"}
    base = {"γυναικών": "γυναικών"}
    evidence = {"ῥᾶ": {"ῥᾶ": 8}, "ἕως": {"ἕως": 5399, "ἠώς": 1}}
    # γυναικῶν folds to γυναικων too; ῥα folds to ρα and the corpus reads
    # it as ἄρα; εως has only ἕως behind it.
    key_evidence = {"γυναικων": {"γυνή": 1785},
                    "ρα": {"ἄρα": 752, "ῥᾶ": 8},
                    "εως": {"ἕως": 5400, "ἠώς": 1}}
    restored = B.restore_displaced_lexical_heads(
        ag, wiktionary_heads={"γυναικών", "ῥᾶ", "ἕως"}, base=base,
        evidence=evidence, key_evidence=key_evidence)
    assert ag["γυναικών"] == "γυναικών" and ag["γυναικων"] == "γυνή"
    assert ag["ῥᾶ"] == "ῥᾶ" and ag["ρα"] == "ῥεῖα"
    assert ag["ἕως"] == ag["εως"] == "ἕως"
    assert {r["form"] for r in restored} == {"γυναικών", "ῥᾶ", "ἕως", "εως"}


def test_case_twins_are_restored_only_toward_the_commoner_lemma():
    ag = {"Υἱός": "υἱός", "υιος": "υἱός",           # capital head: never
          "Ϡ": "ϡ", "ϡ": "ϡ",
          "ψυχή": "Ψυχή", "ψυχη": "Ψυχή",           # lowercase lemma commoner
          "φίλιππος": "Φίλιππος", "φιλιππος": "Φίλιππος"}  # the name commoner
    heads = {"Υἱός", "Ϡ", "ψυχή", "φίλιππος"}
    evidence = {"Υἱός": {"Υἱός": 8}, "Ϡ": {"Ϡ": 12},
                "ψυχή": {"ψυχή": 760}, "φίλιππος": {"φίλιππος": 8}}
    totals = {"υἱός": 587, "Υἱός": 8, "ψυχή": 15029, "Ψυχή": 9,
              "φίλιππος": 37, "Φίλιππος": 2970}
    restored = B.restore_displaced_lexical_heads(
        ag, wiktionary_heads=heads, base={}, evidence=evidence,
        lemma_totals=totals, key_evidence={"ψυχη": {"ψυχή": 15029}})
    assert ag["Υἱός"] == ag["υιος"] == "υἱός"
    assert ag["Ϡ"] == ag["ϡ"] == "ϡ"
    assert ag["ψυχή"] == ag["ψυχη"] == "ψυχή"
    assert ag["φίλιππος"] == ag["φιλιππος"] == "Φίλιππος"
    assert {r["form"] for r in restored} == {"ψυχή", "ψυχη"}


def test_listed_heads_need_two_tokens_and_a_reading_the_build_keeps():
    ag = {"ἐγώγε": "ἐγώ", "ὄ": "ὅς", "ό": "ὅς", "σπαθίον": "σπάθη"}
    base = {"ἐγώγε": "ἐγώγε", "ὄ": "ὄ", "σπαθίον": "σπαθίον"}
    evidence = {"ἐγώγε": {"ἐγώγε": 1}, "ὄ": {"ὄ": 3},
                "σπαθίον": {"σπαθίον": 2}}
    restored = B.restore_displaced_lexical_heads(
        ag, wiktionary_heads=set(), listed_heads={"ἐγώγε", "ὄ", "σπαθίον"},
        base=base, evidence=evidence,
        # the build's article filter drops ὄ -> ὄ, which would leave the
        # key without a row
        rejected=lambda form, head: head == "ὄ")
    assert ag["ἐγώγε"] == "ἐγώ"
    assert ag["ὄ"] == ag["ό"] == "ὅς"
    assert ag["σπαθίον"] == "σπαθίον"
    assert [r["form"] for r in restored] == ["σπαθίον"]


def test_corpus_key_evidence_folds_forms_to_their_loose_keys():
    pairs = [{"form": "γυναικῶν", "lemma": "γυνή", "count": 1785},
             {"form": "Γυναικῶν", "lemma": "γυνή", "count": 2},
             {"form": "ψυχή", "lemma": "ψυχή", "count": 5}]
    assert B.corpus_key_evidence(pairs, {"γυναικων", "ψυχη"}) == {
        "γυναικων": {"γυνή": 1787}, "ψυχη": {"ψυχή": 5}}
    assert B.corpus_lemma_totals(pairs) == {"γυνή": 1787, "ψυχή": 5}


def test_restore_displaced_lexical_heads_respects_eligibility():
    ag = {"κλαίω": "κλείω"}
    restored = B.restore_displaced_lexical_heads(
        ag, wiktionary_heads={"κλαίω"}, base={"κλαίω": "κλαίω"}, evidence={},
        eligible=lambda form: False)
    assert restored == [] and ag == {"κλαίω": "κλείω"}


def _repin_base(data, base):
    """Replace the fixture's Wiktionary-only base and its pin."""
    path = data / "ag_lookup_wikt.json"
    path.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")
    manifest_path = data / "ag_expansion_reference.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["reference_base"] = {
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest_path.write_text(json.dumps(manifest))


FIXTURE_HEADS = {"σύνοχος": ["lsj9_headwords"], "γε": ["wiktionary_en_grc"],
                 "λόγος": ["wiktionary_en_grc"], "μοναχικός": ["lsj9_headwords"],
                 "λαβή": ["wiktionary_en_grc"], "κλεῖδα": ["wiktionary_en_grc"]}


def test_displaced_lexical_heads_take_back_their_reading(tmp_path):
    data, env = _fixture(tmp_path, {
        "ag": {"κλαίω": "κλείω", "κλαιω": "κλείω", "κλείω": "κλείω",
               "μήν": "μείς", "μείς": "μείς",
               "θρόνον": "θρόνος", "θρόνος": "θρόνος",
               "ἀγγελθείης": "ἀγγέλλω", "σπαθίον": "σπάθη", "σπάθη": "σπάθη",
               "ποδίσκος": "πούς"},
        "grc_lexical_heads.json": {"heads": {
            **FIXTURE_HEADS,
            "κλαίω": ["wiktionary_en_grc"], "μήν": ["wiktionary_el_grc"],
            "θρόνον": ["wiktionary_en_grc"], "ἀγγελθείης": ["wiktionary_el_grc"],
            "σπαθίον": ["lsj9_headwords"], "ποδίσκος": ["lsj9_headwords"]},
            "vernacular_heads": []},
        "glaux_pairs.json": [
            {"form": "μήν", "lemma": "μήν", "pos": "part", "count": 50},
            {"form": "μήν", "lemma": "μείς", "pos": "noun", "count": 1},
            {"form": "θρόνον", "lemma": "θρόνος", "pos": "noun", "count": 20},
            {"form": "σπαθίον", "lemma": "σπαθίον", "pos": "noun", "count": 2}],
    })
    _repin_base(data, {**WIKTIONARY, "κλαίω": "κλαίω", "θρόνον": "θρόνον",
                       "σπαθίον": "σπαθίον", "ποδίσκος": "ποδίσκος"})
    result = _build(data, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Displaced lexical heads restored: 3 heads, 4 keys" in result.stdout
    combined = _rows(data)
    for lang in ("all", "grc"):
        rows = {**combined, **_rows(data, lang)}
        assert rows["κλαίω"] == rows["κλαιω"] == "κλαίω"
        assert rows["μήν"] == "μήν" and rows["σπαθίον"] == "σπαθίον"
        assert rows["θρόνον"] == "θρόνος" and rows["ποδίσκος"] == "πούς"
        assert rows["ἀγγελθείης"] == "ἀγγέλλω"
        assert rows["κλείω"] == "κλείω" and rows["μείς"] == "μείς"
    with sqlite3.connect(data / "lookup.db") as conn:
        origin = dict(conn.execute(
            "SELECT form, s.code FROM lookup l JOIN sources s "
            "ON s.id = l.source_id WHERE lang = 'all'"))
    assert origin["κλαίω"] == origin["μήν"] == origin["κλαιω"] == "wiktionary_grc"
    rejected = _rejections(data)
    assert ("AG", "ag_lookup.json", "displaced_lexical_head_restored_to_κλαίω",
            "κλαίω", "κλείω") in rejected
    assert ("AG", "ag_lookup.json", "displaced_lexical_head_restored_to_μήν",
            "μήν", "μείς") in rejected


def test_displaced_heads_without_corpus_pairs_follow_the_base(tmp_path):
    data, env = _fixture(tmp_path, {
        "ag": {"κλαίω": "κλείω", "μήν": "μείς", "σπαθίον": "σπάθη"},
        "grc_lexical_heads.json": {"heads": {
            **FIXTURE_HEADS, "κλαίω": ["wiktionary_en_grc"],
            "μήν": ["wiktionary_el_grc"], "σπαθίον": ["lsj9_headwords"]},
            "vernacular_heads": []},
    })
    _repin_base(data, {**WIKTIONARY, "κλαίω": "κλαίω", "σπαθίον": "σπαθίον"})
    result = _build(data, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "glaux_pairs.json not found; displaced heads" in result.stdout
    rows = {**_rows(data), **_rows(data, "grc")}
    assert rows["κλαίω"] == "κλαίω" and rows["μήν"] == "μείς"
    assert rows["σπαθίον"] == "σπάθη"
