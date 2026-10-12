"""The compatibility gate must reject fragments and retain complete words."""
import sqlite3

import pytest

import export_hunspell as exporter
from scripts import audit_hunspell_frequency as audit


@pytest.fixture
def compatibility_gate(tmp_path, monkeypatch):
    database = tmp_path / "lookup.db"
    with sqlite3.connect(database) as conn:
        conn.executescript("""
            CREATE TABLE lemmas(id INTEGER PRIMARY KEY, text TEXT);
            CREATE TABLE lookup(form TEXT, lemma_id INTEGER, src TEXT, lang TEXT);
        """)
        for index, (form, lemma) in enumerate([
            ("Γλαῦκ᾽", "Γλαῦκος"),
            ("Βηθλέεμ᾽", "Βηθλέεμα"),
            ("Βηθλέεμ", "Βηθλέεμ"),
            ("Ἰακώβ᾽", "Ἰακώβ"),
        ]):
            conn.execute("INSERT INTO lemmas VALUES (?,?)", (index, lemma))
            conn.execute("INSERT INTO lookup VALUES (?,?,'grc','all')", (form, index))
    required = {"Γλαῦκ", "Γλαῦκ᾽", "Βηθλέεμ", "Ἰακώβ", "λόγος"}
    monkeypatch.setattr(audit, "LOOKUP_DB", database)
    monkeypatch.setattr(audit, "load_gzip_fixture", lambda path: {"forms": required})
    monkeypatch.setattr(audit, "load_grc_spelling_review", lambda: {})
    monkeypatch.setattr(audit, "load_form_profile_freq", lambda: None)
    monkeypatch.setattr(audit, "load_case_ownership", lambda conn: {})
    monkeypatch.setattr(audit, "filter_case_artifacts", lambda pairs, *args: (pairs, []))
    monkeypatch.setattr(audit, "load_top_lsj9_lemmas", lambda count: set())
    monkeypatch.setattr(audit, "weak_respellings", lambda *args: [])
    monkeypatch.setattr(audit, "load_unsupported_bare_elisions", lambda conn:
                        exporter.load_unsupported_bare_elisions(
                            conn, {exporter.exact_form_key("Ἰακώβ")}), raising=False)

    def check(forms):
        monkeypatch.setattr(audit, "expanded_export_forms", lambda path: (forms, []))
        return audit.audit_whole_artifact(tmp_path / "dictionary")

    return required, check


def test_compatibility_excludes_only_unsupported_bare_fragment(compatibility_gate):
    required, check = compatibility_gate
    result = check(required - {"Γλαῦκ"})
    assert result["compatibility_missing"] == []
    assert result["invalid_forms"] == []


def test_compatibility_reintroduction_of_fragment_fails(compatibility_gate):
    required, check = compatibility_gate
    result = check(required)
    assert result["invalid_forms"] == ["Γλαῦκ"]


@pytest.mark.parametrize("missing", ["Γλαῦκ᾽", "Βηθλέεμ", "Ἰακώβ", "λόγος"])
def test_compatibility_still_requires_marked_and_complete_words(compatibility_gate, missing):
    required, check = compatibility_gate
    result = check(required - {"Γλαῦκ", missing})
    assert result["compatibility_missing"] == [missing]
