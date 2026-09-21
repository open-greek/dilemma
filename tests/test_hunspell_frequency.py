"""Corpus-head coverage regressions for the Ancient Greek Hunspell export."""
from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path

import pytest

from export_lm import write_binary
from scripts.audit_hunspell_frequency import (
    audit_dictionary,
    audit_export_orthography,
    fixture_payload_from_binary,
    fixture_payload_from_json,
    load_exclusions,
    lookup_form,
)
from export_hunspell import (
    AG_EXPORT_OVERRIDES,
    AG_FUNCTION_WORDS,
    BARE_ELISION_STEMS,
    GRC_FORM_FREQ,
    LOOKUP_DB,
    exact_form_key,
    filter_by_lemma_freq,
    filter_grc_orthography,
    load_canonical_ag_sets,
    load_grc_form_freq,
    sanitize_export_pairs,
    select_forms,
    strip_accents,
    write_variant,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURE_DIR / "hunspell_lm_top1000.json"
EXCLUSIONS = FIXTURE_DIR / "hunspell_lm_top1000_exclusions.json"

# These were independently observed as regressions from Tonos's April 0.4.1
# dictionary. Keep them explicit even when their LM rank falls below the
# frequency-head fixture.
REPORTED_REGRESSIONS = {
    "γε", "καλῶς", "ἕως", "κακῶς", "λέγω", "περ", "ὁμοίως",
    "τριάκοντα", "ὀρθῶς", "ῥᾳδίως", "τάχα", "πεντήκοντα", "πατήρ",
    "εἰκότως", "πόθεν", "ὅλως", "λέων", "εἰκός", "βασιλεύς", "μηκέτι",
    "παρά", "δώδεκα", "φέρω", "χώρα", "δύναμις", "τε", "τις", "ποτε",
    "πάλιν", "σήμερον", "χάριν", "χάρις", "χάριτος", "χάριτι",
}


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_frequency_fixture_is_well_formed():
    fixture = _fixture()
    rows = fixture["forms"]
    assert fixture["top_n"] == 1000 == len(rows)
    assert all(row["count"] > 0 for row in rows)
    assert [row["count"] for row in rows] == sorted(
        (row["count"] for row in rows), reverse=True
    )
    assert fixture["training"]["sanity"] is False
    assert fixture["training"]["total_tokens"] == 30_933_396
    assert next(row for row in rows if row["form"] == "γε")["count"] == 43_741
    assert sum(row["count"] for row in rows) == 16_543_630


def test_frequency_exclusions_are_explicit_and_within_fixture():
    fixture_forms = {row["form"] for row in _fixture()["forms"]}
    exclusions = load_exclusions(EXCLUSIONS)

    assert len(exclusions) == 29
    assert set(exclusions) <= fixture_forms
    assert {"του", "ἀλλ", "ἀπ", "αʹ", "βʹ", "τώρα"} <= set(exclusions)


def test_exact_attestation_map_is_pinned_and_preserves_polytonic_marks():
    with gzip.open(GRC_FORM_FREQ, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    metadata = payload["_meta"]
    forms = payload["forms"]

    assert metadata["training"]["sanity"] is False
    assert metadata["training"]["total_tokens"] == 30_933_396
    assert metadata["sources"]["grc_ngram.bin"]["sha256"] == (
        "862a104d5fde6795259b11852fa0fafc084534e9bce6fad9377b1c8e8dfd70e7"
    )
    assert len(forms) == 76_153
    assert forms["λέγω"] == 7_977
    assert "λεγω" not in forms
    assert forms["αὐτός"] == 31_486
    assert forms["αυτός"] == 23
    assert forms["μηδ᾽"] == 2_699
    assert exact_form_key("τῳ") in forms
    assert exact_form_key("τωι") not in forms


def test_json_fixture_rejects_sanity_lm_run(tmp_path):
    vocab = tmp_path / "vocab.json"
    unigrams = tmp_path / "unigrams.json"
    stats = tmp_path / "stats.json"
    vocab.write_text('["<PAD>", "λόγος"]', encoding="utf-8")
    unigrams.write_text('{"1": 10}', encoding="utf-8")
    stats.write_text('{"sanity": true, "n_train_tokens": 10}', encoding="utf-8")

    with pytest.raises(ValueError, match="requires a full LM run"):
        fixture_payload_from_json(vocab, unigrams, stats, top_n=1)


def test_binary_fixture_reads_embedded_vocabulary_counts(tmp_path):
    path = tmp_path / "grc_ngram.bin"
    vocab = ["</s>", "<PAD>", "<UNK>", "<s>", "γε", "λόγος"]
    write_binary(
        path,
        id2tok=vocab,
        vocab_counts=[0, 0, 0, 0, 12, 20],
        unigram_topk=[],
        bigram_ctx=[],
        trigram_ctx=[],
        total_tokens=32,
        reserved_ids={
            "</s>": 0,
            "<PAD>": 1,
            "<UNK>": 2,
            "<s>": 3,
        },
    )

    fixture = fixture_payload_from_binary(path, None, top_n=2)

    assert fixture["training"]["total_tokens"] == 32
    assert fixture["forms"] == [
        {"form": "λόγος", "count": 20},
        {"form": "γε", "count": 12},
    ]


@pytest.mark.skipif(not LOOKUP_DB.exists(), reason="lookup.db not downloaded")
def test_expanded_export_accepts_frequency_head_and_reported_regressions(tmp_path):
    Dictionary = pytest.importorskip("spylls.hunspell").Dictionary
    fixture = _fixture()

    # Lemma admission still uses the aggregate, accent-stripped corpus map;
    # per-form acute attestation uses the pinned, accent-preserving full LM.
    freq = {
        strip_accents(row["form"]): row["count"]
        for row in fixture["forms"]
    }
    exact_freq = load_grc_form_freq()
    for form in REPORTED_REGRESSIONS:
        freq[strip_accents(form)] = max(freq.get(strip_accents(form), 0), 3)

    conn = sqlite3.connect(f"file:{LOOKUP_DB}?mode=ro", uri=True)
    _, canonical_lemmas = load_canonical_ag_sets()
    pairs = select_forms(
        conn,
        "grc",
        keep_lemmas=canonical_lemmas,
        attestation_freq=exact_freq,
    )
    conn.close()
    pairs = filter_by_lemma_freq(
        pairs,
        freq,
        min_lemma_count=3,
        strict_acute_min=1,
        strict_form_freq_map=exact_freq,
    )
    existing = {form for form, _lemma in pairs}
    pairs.extend(
        (form, lemma)
        for form, lemma in (AG_FUNCTION_WORDS | AG_EXPORT_OVERRIDES).items()
        if form not in existing
    )
    pairs, _changed, _dropped = sanitize_export_pairs(pairs)
    pairs, invalid_initial = filter_grc_orthography(pairs)
    assert invalid_initial
    write_variant(
        variant="grc",
        form_lemma=pairs,
        freq_map=freq,
        out_dir=tmp_path,
        dic_name="coverage",
        lang_tag="grc",
        version="test",
        commit="test",
    )

    dictionary = Dictionary.from_files(str(tmp_path / "coverage"))
    exclusions = load_exclusions(EXCLUSIONS)
    expected = {
        lookup_form(row["form"])
        for row in fixture["forms"]
        if row["form"] not in exclusions
    }
    expected.update(REPORTED_REGRESSIONS)
    missing = sorted(form for form in expected if not dictionary.lookup(form))
    assert missing == []
    assert not any(dictionary.lookup(lookup_form(form)) for form in exclusions)
    assert not any(dictionary.lookup(stem) for stem in BARE_ELISION_STEMS)

    missing, accepted_exclusions = audit_dictionary(
        tmp_path / "coverage", fixture, exclusions
    )
    assert missing == []
    assert accepted_exclusions == []
    invalid_initial, synthetic_flags = audit_export_orthography(
        tmp_path / "coverage"
    )
    assert invalid_initial == []
    assert synthetic_flags == []
