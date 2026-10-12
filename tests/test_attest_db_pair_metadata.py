"""A citation form_id is valid only within its matching profile build."""

import json
from pathlib import Path
import shutil
import sqlite3

import pytest

from build import build_form_attestation as builder
from dilemma import _attest_db
from dilemma._attest_db import AttestDB


def build_pair(directory, forms=("λόγος", "μῆνιν")):
    directory.mkdir()
    profiles = {}
    sink_path = directory / "citations.tmp"
    sink = builder.CiteSink(sink_path)
    for form_id, form in enumerate(forms):
        profile = builder.FormProfile()
        profile.observe("glaux", "noun")
        profile.add_deduped("oratory", -4, "Attic")
        profiles[form_id] = profile
        sink.add([(form_id, "0001-001", "glaux", form + ":1", "line", 1, -4)])
    works = {"0001-001": {
        "work_id": "0001-001", "id_scheme": "tlg", "source": "glaux",
        "author": "Fixture", "title": "Fixture work", "genre": "oratory",
        "dialect": "Attic", "century": -4, "start_year": -400, "end_year": -301,
    }}
    content_hash = builder.write_dbs(
        list(forms), profiles, works, sink, sink_path, 200,
        {"glaux_xml": "b" * 64}, directory / "form_profile.db",
        directory / "form_citations.db",
    )
    return content_hash


@pytest.fixture
def pair(tmp_path):
    path = tmp_path / "current"
    build_pair(path)
    return path


def update_meta(path, key, value):
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE meta SET value=? WHERE key=?", (value, key))


def assert_profile_only(record):
    assert record["attested"] is True
    assert record["form"] == "λόγος"
    assert record["total_count"] == 1
    assert record["source_counts"] == {"glaux": 1}
    assert record["by_century"] == {-4: 1}
    assert record["citations"] == []
    note = record["citations_note"]
    assert "form_profile.db" in note and "form_citations.db" in note
    assert "redownload both together" in note and "--with-citations" in note


def test_matching_real_builder_pair_returns_correct_loci(pair):
    db = AttestDB(pair)
    record = db.attestation("λόγος")
    assert record["citations"][0]["locus"] == "λόγος:1"
    assert record["citations"][0]["title"] == "Fixture work"
    assert "citations_note" not in record
    assert db.attestation_by_norm("Λόγος")["citations"] == record["citations"]
    db.close()


@pytest.mark.parametrize("normalized", [False, True])
def test_reordered_form_ids_never_return_another_words_locus(pair, tmp_path, normalized):
    stale = tmp_path / "stale"
    stale_hash = build_pair(stale, ("μῆνιν", "λόγος"))
    with sqlite3.connect(pair / "form_profile.db") as conn:
        current_hash = conn.execute("SELECT value FROM meta WHERE key='content_hash'").fetchone()[0]
        form_id = conn.execute("SELECT form_id FROM forms WHERE form='λόγος'").fetchone()[0]
    assert stale_hash != current_hash
    with sqlite3.connect(stale / "form_citations.db") as conn:
        assert conn.execute("SELECT locus FROM citations WHERE form_id=?", (form_id,)).fetchone()[0] == "μῆνιν:1"
    shutil.copyfile(stale / "form_citations.db", pair / "form_citations.db")
    db = AttestDB(pair)
    record = db.attestation_by_norm("Λόγος") if normalized else db.attestation("λόγος")
    assert_profile_only(record)
    db.close()


@pytest.mark.parametrize("name", ["form_profile.db", "form_citations.db"])
@pytest.mark.parametrize("key,value", [
    ("schema_version", "2"), ("schema_version", ""),
    ("content_hash", "c" * 64), ("content_hash", "not-a-hash"),
    ("content_hash", None),
    ("source_sha", json.dumps({"glaux_xml": "c" * 64})),
    ("source_sha", "{}"), ("source_sha", "[]"),
    ("source_sha", "not-json"), ("source_sha", None),
    ("source_sha", json.dumps({"glaux_xml": "bad"})),
])
def test_invalid_or_mismatched_metadata_preserves_profile(pair, name, key, value):
    update_meta(pair / name, key, value)
    db = AttestDB(pair)
    assert_profile_only(db.attestation("λόγος"))
    db.close()


@pytest.mark.parametrize("name", ["form_profile.db", "form_citations.db"])
def test_unversioned_pair_preserves_profile(pair, name):
    with sqlite3.connect(pair / name) as conn:
        conn.execute("DROP TABLE meta")
    db = AttestDB(pair)
    assert_profile_only(db.attestation("λόγος"))
    db.close()


@pytest.mark.parametrize("key", ["schema_version", "content_hash", "source_sha"])
def test_missing_citation_identity_field_preserves_profile(pair, key):
    with sqlite3.connect(pair / "form_citations.db") as conn:
        conn.execute("DELETE FROM meta WHERE key=?", (key,))
    db = AttestDB(pair)
    assert_profile_only(db.attestation("λόγος"))
    db.close()


def test_unreadable_citation_database_preserves_profile(pair):
    (pair / "form_citations.db").write_bytes(b"not a SQLite database")
    db = AttestDB(pair)
    assert_profile_only(db.attestation("λόγος"))
    db.close()


@pytest.mark.parametrize("valid", [False, True])
def test_pair_metadata_is_checked_once_per_open_instance(pair, monkeypatch, valid):
    if not valid:
        update_meta(pair / "form_citations.db", "content_hash", "c" * 64)
    queries = []
    original_connect = AttestDB._connect_ro

    def connect(path):
        conn = original_connect(path)
        conn.set_trace_callback(queries.append)
        return conn

    monkeypatch.setattr(AttestDB, "_connect_ro", staticmethod(connect))
    db = AttestDB(pair)
    for form in ("λόγος", "μῆνιν", "λόγος"):
        record = db.attestation(form)
        assert bool(record["citations"]) is valid
    assert sum("FROM meta" in query for query in queries) == 2
    db.close()


def test_profile_only_request_never_opens_or_checks_citations(pair, monkeypatch):
    (pair / "form_citations.db").write_bytes(b"unreadable citations")
    with sqlite3.connect(pair / "form_profile.db") as conn:
        conn.execute("DROP TABLE meta")
    db = AttestDB(pair)

    def unexpected():
        pytest.fail("max_citations=0 must not touch citation validation")

    monkeypatch.setattr(db, "_citations", unexpected)
    record = db.attestation("λόγος", max_citations=0)
    assert record["total_count"] == 1 and record["citations"] == []
    assert "citations_note" not in record
    db.close()


def test_close_rechecks_metadata_after_reopening(pair):
    db = AttestDB(pair)
    assert db.attestation("λόγος")["citations"]
    db.close()
    update_meta(pair / "form_citations.db", "content_hash", "c" * 64)
    assert_profile_only(db.attestation("λόγος"))
    db.close()
