"""Exact corpus POS targets require independent merged lexical evidence."""

import json
import sqlite3

import pytest

from build import build_treebank_pos_lookup as build


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "DATA_DIR", tmp_path)
    monkeypatch.setattr(build, "TREEBANK_DIRS", [])
    monkeypatch.setattr(build, "AGDT_DIR", tmp_path / "missing_agdt")
    monkeypatch.setattr(build, "KAIKKI_DIR", None)
    monkeypatch.setattr(build, "LSJ9_HEADWORDS", None)
    monkeypatch.setattr(build, "OUTPUT_PATH", tmp_path / "result.json")
    monkeypatch.setattr(build, "AUDIT_PATH", None)
    monkeypatch.setattr(build, "LOOKUP_DB_PATH", tmp_path / "lookup.db")
    for source in ("GLAUX", "DIORISIS", "NT", "PTNK", "PEDALION"):
        monkeypatch.setattr(build, source + "_PAIRS_PATH",
                            tmp_path / (source.lower() + ".json"))

    def write(sources, heads):
        for source, rows in sources.items():
            path = getattr(build, source.upper() + "_PAIRS_PATH")
            path.write_text(json.dumps(rows, ensure_ascii=False))
        (tmp_path / "grc_lexical_heads.json").write_text(json.dumps({
            "schema_version": 2,
            "heads": heads,
        }, ensure_ascii=False))
        # Deliberately include unsupported targets in the SQLite lemma table:
        # generated rows must not be mistaken for independent definitions.
        lemmas = sorted({row["lemma"] for rows in sources.values() for row in rows})
        with sqlite3.connect(build.LOOKUP_DB_PATH) as db:
            db.execute("CREATE TABLE lemmas (id INTEGER PRIMARY KEY, text TEXT)")
            db.execute("CREATE TABLE lookup (form TEXT, lemma_id INTEGER, lang TEXT)")
            db.executemany("INSERT INTO lemmas VALUES (?, ?)", enumerate(lemmas, 1))

    return write


@pytest.mark.parametrize("source", ["glaux", "diorisis", "nt", "ptnk", "pedalion"])
def test_generated_lookup_target_is_rejected_from_every_corpus(inputs, source):
    rows = [
        {"form": "πεῖραι", "lemma": "πεῖρα", "pos": "noun"},
        {"form": "πεῖραι", "lemma": "ἀπολάω", "pos": "verb"},
        {"form": "ἀπλάνητος", "lemma": "ἀπλάνητος", "pos": "adj"},
        {"form": "ἀπλάνητος", "lemma": "ἀπολάω", "pos": "verb"},
        {"form": "ἀπόλλων", "lemma": "Ἀπόλλων", "pos": "noun"},
        {"form": "ἀπόλλων", "lemma": "ἀπολάω", "pos": "verb"},
    ]
    inputs({source: rows}, {
        "πεῖρα": ["wiktionary_en_grc", "wiktionary_el_grc"],
        "ἀπλάνητος": ["lsj9_headwords"],
        "Ἀπόλλων": ["cunliffe_headwords", "lsj9_headwords"],
    })
    result = build.build_lookup()
    assert not any("ἀπολάω" in values.values() for values in result.values())
    audit = json.loads(build.OUTPUT_PATH.with_suffix(".audit.json").read_text())
    assert audit["rejected_corpus_targets"] == {"ἀπολάω": {source: 3}}


def test_rejection_preserves_a_supported_competing_verb(inputs):
    inputs({
        "glaux": [
            {"form": "πεῖραι", "lemma": "πεῖρα", "pos": "noun"},
            {"form": "πεῖραι", "lemma": "ἀπολάω", "pos": "verb"},
        ],
        "diorisis": [{"form": "πεῖραι", "lemma": "πείρω", "pos": "verb"}],
    }, {"πεῖρα": ["wiktionary_el_grc"], "πείρω": ["lsj9_headwords"]})
    assert build.build_lookup()["πεῖραι"] == {"NOUN": "πεῖρα", "VERB": "πείρω"}


@pytest.mark.parametrize(("head", "target", "accepted"), [
    ("στορέννυμι", "στόρεννυμι", True),
    ("ἵημι", "ῑ̔́ημι", True),
    ("ἄγω", "ἅγω", False),
    ("λόγος", "Λόγος", False),
    ("Μωϋσῆς", "Μωυσῆς", False),
    ("ᾄδω", "ἄδω", False),
])
def test_target_matching_retains_lexical_marks(inputs, head, target, accepted):
    inputs({"glaux": [
        {"form": "δοκιμή", "lemma": "λόγος", "pos": "noun"},
        {"form": "δοκιμή", "lemma": target, "pos": "verb"},
    ]}, {"λόγος": ["lsj9_headwords"], head: ["wiktionary_en_grc"]})
    values = build.build_lookup().get("δοκιμή", {})
    assert (values.get("VERB") == target) is accepted


@pytest.mark.parametrize("reverse", [False, True])
def test_tied_owners_are_audited_without_removing_other_readings(inputs, reverse):
    rows = [
        {"form": "δοκιμή", "lemma": "λόγος", "pos": "noun"},
        {"form": "δοκιμή", "lemma": "καλός", "pos": "adj"},
        {"form": "δοκιμή", "lemma": "ἄγω", "pos": "verb"},
        {"form": "δοκιμή", "lemma": "ἔχω", "pos": "verb"},
    ]
    if reverse:
        rows.reverse()
    inputs({"glaux": rows}, {row["lemma"]: ["lsj9_headwords"] for row in rows})
    assert build.build_lookup()["δοκιμή"] == {"ADJ": "καλός", "NOUN": "λόγος"}
    path = build.OUTPUT_PATH.with_suffix(".audit.json")
    audit = json.loads(path.read_text())
    assert audit["tied_owners"]["δοκιμή"] == {
        "VERB": {"source": "corpus", "votes": {"ἄγω": 2, "ἔχω": 2}}}
    original = path.read_bytes()
    build.build_lookup()
    assert path.read_bytes() == original


@pytest.mark.parametrize("malformed", [False, True])
def test_missing_or_malformed_heads_leave_existing_outputs_intact(inputs, malformed):
    inputs({}, {})
    heads = build.DATA_DIR / "grc_lexical_heads.json"
    if malformed:
        heads.write_text('{"schema_version": 1, "heads": {}}')
    else:
        heads.unlink()
    build.OUTPUT_PATH.write_text("existing runtime table")
    audit = build.OUTPUT_PATH.with_suffix(".audit.json")
    audit.write_text("existing audit")
    with pytest.raises((FileNotFoundError, ValueError)):
        build.build_lookup()
    assert build.OUTPUT_PATH.read_text() == "existing runtime table"
    assert audit.read_text() == "existing audit"


def test_heads_without_positive_sources_cannot_authorize_a_target(inputs):
    inputs({"glaux": [
        {"form": "δοκιμή", "lemma": "λόγος", "pos": "noun"},
        {"form": "δοκιμή", "lemma": "ἄγω", "pos": "verb"},
    ]}, {"λόγος": ["lsj9_headwords"], "ἄγω": []})
    assert build.build_lookup() == {}
