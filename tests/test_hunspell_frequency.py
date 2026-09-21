"""Corpus-head coverage regressions for the Ancient Greek Hunspell export."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from export_lm import write_binary
from scripts.audit_hunspell_frequency import (
    fixture_payload_from_binary,
    fixture_payload_from_json,
)
from export_hunspell import (
    AG_FUNCTION_WORDS,
    BARE_ELISION_STEMS,
    LOOKUP_DB,
    filter_by_lemma_freq,
    load_canonical_ag_sets,
    sanitize_export_pairs,
    select_forms,
    strip_accents,
    write_variant,
)

FIXTURE = Path(__file__).parent / "fixtures" / "hunspell_lm_top100.json"

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
    assert fixture["top_n"] == 100 == len(rows)
    assert all(row["count"] > 0 for row in rows)
    assert [row["count"] for row in rows] == sorted(
        (row["count"] for row in rows), reverse=True
    )
    assert fixture["training"]["sanity"] is False
    assert fixture["training"]["total_tokens"] == 30_933_396
    assert next(row for row in rows if row["form"] == "γε")["count"] == 43_741


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

    # The production frequency map is accent-stripped. Give every explicit
    # regression enough evidence to cross the real lemma threshold as well.
    freq = {
        strip_accents(row["form"]): row["count"]
        for row in fixture["forms"]
    }
    for form in REPORTED_REGRESSIONS:
        freq[strip_accents(form)] = max(freq.get(strip_accents(form), 0), 3)

    conn = sqlite3.connect(f"file:{LOOKUP_DB}?mode=ro", uri=True)
    _, canonical_lemmas = load_canonical_ag_sets()
    pairs = select_forms(
        conn,
        "grc",
        keep_lemmas=canonical_lemmas,
        attestation_freq=freq,
    )
    conn.close()
    existing = {form for form, _lemma in pairs}
    pairs.extend(
        (form, lemma) for form, lemma in AG_FUNCTION_WORDS.items()
        if form not in existing
    )
    pairs = filter_by_lemma_freq(
        pairs, freq, min_lemma_count=3, strict_acute_min=1
    )
    pairs, _changed, _dropped = sanitize_export_pairs(pairs)
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
    expected = {row["form"] for row in fixture["forms"]}
    expected.update(REPORTED_REGRESSIONS)
    missing = sorted(form for form in expected if not dictionary.lookup(form))
    assert missing == []
    assert not any(dictionary.lookup(stem) for stem in BARE_ELISION_STEMS)
