"""Regression tests for the revision-pinned GlossAPI corpus audit."""

import json
import sqlite3

import pytest

from eval.eval_glossapi_variety import chunk_words, summarize
from scripts.audit_glossapi_coverage import (
    AuditError,
    LookupConflictResolver,
    RawSample,
    audit_corpus,
    build_frequency_experiment,
    extract_row_text,
    iter_greek_tokens,
    iter_parquet_rows,
    load_manifest,
    measure_coverage,
    mg_frequency_counts,
    token_defects,
    unknown_category,
    validate_manifest,
)


def test_manifest_pins_exactly_nine_corpora_and_quarantines_ogc_candidates():
    manifest = load_manifest()
    assert len(manifest["corpora"]) == 9
    assert len({item["key"] for item in manifest["corpora"]}) == 9
    for corpus in manifest["corpora"]:
        assert len(corpus["revision"]) == 40
        assert corpus["revision"] not in {"main", "master"}
        for artifact in corpus["files"]:
            assert len(artifact["sha256"]) == 64
            assert artifact["size"] > 0
        if corpus["role"] == "ogc_candidate":
            assert corpus["frequency_token_cap"] == 0
            assert corpus["ogc_decision"]


def test_manifest_rejects_floating_revision():
    manifest = load_manifest()
    manifest = json.loads(json.dumps(manifest))
    manifest["corpora"][0]["revision"] = "main"
    with pytest.raises(AuditError, match="full commit SHA"):
        validate_manifest(manifest)


def test_audit_rejects_a_silent_empty_sample(monkeypatch):
    corpus = {
        "key": "empty",
        "audit_token_cap": 100,
        "lookup_lang": "el",
    }
    monkeypatch.setattr(
        "scripts.audit_glossapi_coverage.collect_corpus_sample",
        lambda *args, **kwargs: RawSample(token_cap=100),
    )
    with pytest.raises(AuditError, match="produced no Greek-bearing tokens"):
        audit_corpus(
            corpus,
            source="cache",
            max_tokens=None,
            max_documents=1,
        )


def test_nested_and_json_extraction_uses_values_not_json_keys():
    row = {
        "title": "Δοκιμή",
        "articles": [{
            "body_text": "Κείμενο άρθρου",
            "comments": [{"content": "Σχόλιο"}],
        }],
        "payload": json.dumps({
            "not_counted_key": "Ελληνική τιμή",
            "url": "https://example.test/ελληνικά",
        }),
    }
    reader = {
        "text_fields": [
            {"path": "title", "format": "plain"},
            {"path": "articles[].body_text", "format": "plain"},
            {"path": "articles[].comments[].content", "format": "plain"},
            {"path": "payload", "format": "json_values"},
        ]
    }
    text = extract_row_text(row, reader)
    assert "Δοκιμή" in text
    assert "Κείμενο άρθρου" in text
    assert "Σχόλιο" in text
    assert "Ελληνική τιμή" in text
    assert "not_counted_key" not in text
    assert "example.test" not in text


def test_parliament_counts_passage_once_without_serialized_json_metadata():
    reader = next(c["reader"] for c in load_manifest()["corpora"]
                  if c["key"] == "european_parliament")
    assert extract_row_text({"section": "Τροπολογία\nΑριθμός", "header": "Τροπολογία",
                             "json_section": '[{"text":"Τροπολογία\\nΑριθμός"}]',
                             "predicted_section": "κ"}, reader) == "Τροπολογία\nΑριθμός"


def test_parquet_sampling_round_robins_across_row_groups(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "sample.parquet"
    parquet.write_table(
        pyarrow.table({"text": [f"row-{index}" for index in range(8)]}),
        path,
        row_group_size=2,
    )
    reader = {
        "text_fields": [{"path": "text", "format": "plain"}],
    }
    with path.open("rb") as handle:
        rows = list(iter_parquet_rows(handle, reader, max_documents=4))
    assert [row["text"] for row in rows] == ["row-0", "row-2", "row-4", "row-6"]


def test_tokenizer_preserves_unicode_defects_for_measurement():
    tokens = list(iter_greek_tokens("κόσμος κóσμος α1 \u0301α"))
    assert tokens == ["κόσμος", "κóσμος", "α1", "\u0301α"]
    assert token_defects("κόσμος") == ()
    assert "mixed_greek_latin" in token_defects("κóσμος")
    assert "mixed_greek_digit" in token_defects("α1")
    assert "leading_combining_mark" in token_defects("\u0301α")
    assert "orphan_combining_mark" in token_defects("\u0301α")


def _write_lookup(path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE lemmas (id INTEGER PRIMARY KEY, text TEXT NOT NULL);
        CREATE TABLE lookup (
            form TEXT NOT NULL,
            lemma_id INTEGER NOT NULL,
            src TEXT NOT NULL,
            lang TEXT NOT NULL
        );
        INSERT INTO lemmas VALUES (1, 'κοινός');
        INSERT INTO lemmas VALUES (2, 'MG lemma');
        INSERT INTO lemmas VALUES (3, 'AG lemma');
        INSERT INTO lookup VALUES ('κοινή', 1, 'el', 'all');
        INSERT INTO lookup VALUES ('διπλή', 2, 'el', 'el');
        INSERT INTO lookup VALUES ('διπλή', 3, 'grc', 'grc');
        """
    )
    connection.commit()
    connection.close()


def test_lookup_conflicts_compare_effective_language_rows(tmp_path):
    database = tmp_path / "lookup.db"
    _write_lookup(database)
    report = LookupConflictResolver(database).conflicts(
        {"κοινή": 7, "διπλή": 3}
    )
    assert report["types"] == 1
    assert report["tokens"] == 3
    assert report["examples"][0]["form"] == "διπλή"
    assert report["examples"][0]["el"]["lemma"] == "MG lemma"
    assert report["examples"][0]["grc"]["lemma"] == "AG lemma"


def test_frequency_experiment_balances_caps_and_excludes_historical(tmp_path):
    existing = tmp_path / "mg_freq.txt"
    existing.write_text("και 100\nλέξη 50\n", encoding="utf-8")
    modern = RawSample(token_cap=10)
    modern.tokens.update({"και": 8, "λέξη": 2})
    legal = RawSample(token_cap=4)
    legal.tokens.update({"νόμος": 4})
    historical = RawSample(token_cap=100)
    historical.tokens.update({"ἀνήρ": 100})
    corpora = [
        {"key": "modern", "role": "mg_core", "frequency_token_cap": 5},
        {"key": "legal", "role": "mg_legal", "frequency_token_cap": 2},
        {"key": "old", "role": "ogc_candidate", "frequency_token_cap": 0},
    ]
    report = build_frequency_experiment(
        corpora,
        {"modern": modern, "legal": legal, "old": historical},
        existing_frequency=existing,
    )
    assert report["replaces_mg_freq"] is False
    assert report["combined_tokens"] == 7
    assert set(report["sources"]) == {"modern", "legal"}
    assert "ἀνήρ" not in dict(report["combined_top_forms"])
    assert dict(report["combined_top_forms"])["και"] == 4
    assert dict(report["combined_top_forms"])["λέξη"] == 1
    assert dict(report["combined_top_forms"])["νόμος"] == 2


def test_tokenizer_keeps_encoding_damage_visible():
    assert list(iter_greek_tokens("λέ\ufffdξη λέ\ue001ξη")) == ["λέ\ufffdξη", "λέ\ue001ξη"]
    assert "replacement_character" in token_defects("λέ\ufffdξη")
    assert "private_use_character" in token_defects("λέ\ue001ξη")
    assert "possible_joined_boundary" in token_defects("ΘεοτοκίονὉ")


def test_unknown_categories_do_not_call_names_or_identifiers_lexical_gaps():
    assert unknown_category("31η") == "number_or_identifier"
    assert unknown_category("ΕΕΛΛΑΚ") == "acronym_or_numeral_candidate"
    assert unknown_category("Βριλησσίων") == "name_or_sentence_initial_candidate"
    assert unknown_category("nΤροπολογία") == "unicode_or_extraction_candidate"
    assert unknown_category("τριτεκνική") == "unresolved_lexical_candidate"


def test_conflicts_try_grave_to_acute_before_losing_breathings(tmp_path):
    database = tmp_path / "lookup.db"
    _write_lookup(database)
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO lookup VALUES ('ὅς', 2, 'el', 'el')")
        connection.execute("INSERT INTO lookup VALUES ('ὅς', 3, 'grc', 'grc')")
    report = LookupConflictResolver(database).conflicts({"ὃς": 3})
    assert report["types"] == 1
    assert report["examples"][0]["grc"]["variant"] == "ὅς"


def test_odd_multi_artifact_caps_keep_defect_counts_consistent(monkeypatch):
    budgets = []
    def collect(corpus, artifact, *, token_cap, **kwargs):
        budgets.append(token_cap)
        sample = RawSample(token_cap=token_cap)
        sample.add_document(" ".join(["α1"] * token_cap))
        return sample
    monkeypatch.setattr("scripts.audit_glossapi_coverage.collect_artifact_sample", collect)
    from scripts.audit_glossapi_coverage import collect_corpus_sample
    result = collect_corpus_sample({"files": [{}, {}]}, source="cache", token_cap=5)
    assert budgets == [3, 2]
    assert result.token_count == result.defect_tokens["mixed_greek_digit"] == 5


def test_parquet_collection_gives_late_groups_a_budget(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from contextlib import contextmanager
    from scripts.audit_glossapi_coverage import collect_artifact_sample
    path = tmp_path / "sample.parquet"
    pq.write_table(pa.table({"text": ["άλφα " * 100, "βήτα " * 100]}), path,
                   row_group_size=1)
    @contextmanager
    def opened(*args):
        with path.open("rb") as handle:
            yield handle
    monkeypatch.setattr("scripts.audit_glossapi_coverage.open_artifact", opened)
    corpus = {"key": "test", "reader": {"kind": "parquet", "text_fields": [
        {"path": "text", "format": "plain"}]}}
    result = collect_artifact_sample(corpus, {"path": "sample.parquet"},
                                     source="cache", token_cap=5, max_documents=2)
    assert result.tokens == {"άλφα": 3, "βήτα": 2}
    assert result.documents == 2


def test_frequency_cleaning_preserves_nfc_and_filters_malformed_tokens():
    sample = RawSample(token_cap=20)
    sample.add_document("λέξη λέξη λέ\ufffdξη nΤροπολογία ΘεοτοκίονὉ α1")
    assert mg_frequency_counts(sample, 20) == {"λέξη": 2}


def test_coverage_excludes_nonlexical_and_never_requests_guesses(monkeypatch):
    from types import SimpleNamespace
    options = []
    class Lookup:
        def __init__(self, **kwargs):
            options.append(kwargs)
        def preload(self):
            pass
        def classify_nonlexical(self, form):
            return "numeral" if form == "α1" else None
        def lemmatize_verbose(self, form, *, guess):
            assert guess is False
            return [SimpleNamespace(source="lookup")] if form == "λέξη" else []
    monkeypatch.setattr("scripts.audit_glossapi_coverage.Dilemma", Lookup)
    monkeypatch.setattr(LookupConflictResolver, "conflicts", lambda self, forms: {})
    sample = RawSample(token_cap=10)
    sample.tokens.update({"λέξη": 3, "άγνωστο": 1, "α1": 2})
    result = measure_coverage(sample, lookup_lang="el")
    assert result["lexical_tokens"] == 4
    assert result["lexical_types"] == 2
    assert result["token_coverage_percent"] == 75
    assert result["type_coverage_percent"] == 50
    assert options[0]["convention"] == "triantafyllidis"
    assert result["excluded_nonlexical"]["numeral"]["tokens"] == 2


def test_variety_chunking_merges_a_short_tail():
    chunks = chunk_words(" ".join(f"w{i}" for i in range(205)))
    assert [len(chunk.split()) for chunk in chunks] == [80, 80, 45]


def test_variety_summary_scores_only_in_scope_examples():
    predictions = [
        {
            "fixture": "ag", "expected_label": 0, "predicted_label": 0,
            "predicted_name": "ancient", "correct": True,
        },
        {
            "fixture": "mg", "expected_label": 1, "predicted_label": 2,
            "predicted_name": "demotic", "correct": False,
        },
        {
            "fixture": "medieval", "expected_label": None,
            "predicted_label": 0, "predicted_name": "ancient", "correct": None,
        },
    ]
    report = summarize(predictions, {0: "ancient", 1: "modern", 2: "demotic"})
    assert report["scored_chunks"] == 2
    assert report["accuracy"] == 0.5
    assert report["by_fixture"]["medieval"]["accuracy"] is None
