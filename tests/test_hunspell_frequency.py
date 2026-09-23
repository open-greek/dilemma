"""Corpus-head coverage regressions for the Ancient Greek Hunspell export."""
from __future__ import annotations

import json
import sqlite3
import unicodedata
from pathlib import Path

import pytest

from export_lm import write_binary
from scripts.audit_hunspell_frequency import (
    DEFAULT_COMPATIBILITY,
    DEFAULT_HELDOUT,
    audit_dictionary,
    audit_export_orthography,
    audit_textbook_paradigms,
    audit_whole_artifact,
    heldout_regressions,
    measure_heldout_corpora,
    fixture_payload_from_binary,
    fixture_payload_from_json,
    load_gzip_fixture,
    load_exclusions,
    lookup_form,
)
from scripts.build_hunspell_heldout_fixture import exact_tokens
from export_hunspell import (
    BARE_ELISION_STEMS,
    FORM_PROFILE_DB,
    GRC_CLOSED_LIST_FORMS,
    GRC_COMPLETE_PARADIGM_LEMMAS,
    HOMERIC_SHORT_PREPOSITIONS,
    LOOKUP_DB,
    add_grc_reviewed_forms,
    exact_form_key,
    filter_by_lemma_freq,
    finalize_grc_pairs,
    grc_orthography_reason,
    grc_pinned_forms,
    load_canonical_ag_sets,
    load_form_profile_freq,
    load_grc_compatibility_forms,
    load_grc_textbook_forms,
    load_lm_head_required_forms,
    load_top_lsj9_lemmas,
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

REPORTED_BARE_ELISION_FALLBACKS = {
    "ἀφ", "ὑφ", "ἵν", "τοῦτ", "ταῦτ", "ὅτ", "ἀνθ", "οὔτ", "μήτ",
    "ἔπειτ", "εἶτ", "ἀντ", "μετ", "γ", "μ", "σ", "θ",
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


@pytest.mark.skipif(
    not FORM_PROFILE_DB.exists(), reason="form_profile.db not downloaded"
)
def test_full_form_profile_is_pinned_and_preserves_polytonic_marks():
    forms, dominant, treebank, metadata, confirmed = load_form_profile_freq()

    assert metadata["content_hash"] == (
        "d9bfe0176b2b17b948961ea7ad9e47ae0527bb27f79e57210df7b55a15d8e086"
    )
    assert len(forms) == 1_773_559
    # The larger of the work-deduplicated total and any single source's own
    # count: a spelling found only in a lower-priority copy of a work still
    # counts as attested.
    assert forms["λέγω"] == 16_564
    assert forms["λεγω"] == 55
    assert forms["αὐτός"] == 62_362
    assert forms["αυτός"] == 32
    assert forms["μηδ᾽"] == 3_328
    assert exact_form_key("τῳ") in forms
    assert forms[exact_form_key("τῳ")] != forms[exact_form_key("τωι")]
    # Dominance keys drop every combining mark but keep case-folded letters.
    assert dominant["αυτος"] == 62_362
    assert dominant["εγω"] == 42_497
    assert dominant["ταις"] == 70_631
    # Treebank support separates Doric τᾷ from OCR respellings of common words.
    assert treebank[exact_form_key("τᾷ")] == 731
    assert treebank[exact_form_key("ἑγώ")] == 1
    assert exact_form_key("ταΐς") not in treebank
    # Both treebanks annotate the dual πρώτω and Doric γλώσσᾳ, which is what
    # lets them past the floor their raw counts fail; the treebanks' stray
    # taggings of ὅτι and ἐγώ are not that.
    assert {exact_form_key(f) for f in ("πρώτω", "γλώσσᾳ", "πειρᾷς")} <= confirmed
    assert exact_form_key("ταΐς") not in confirmed
    assert exact_form_key("ὄσα") not in confirmed


def test_whole_artifact_compatibility_fixture_is_reviewed_and_pinned():
    fixture = load_gzip_fixture(DEFAULT_COMPATIBILITY)

    # The baseline is whatever Tonos ships now, not the April export it
    # started from: it moves forward with each swap-in, and the contract is
    # that the next export does not lose what the keyboard already has.
    assert fixture["baseline"]["version"] == "1.3.6"
    assert fixture["baseline"]["commit"] == (
        "2266ce119046ee1e4c5ad3948e745352c3e6358d"
    )
    assert fixture["baseline"]["compiled_entries"] == 1_366_673
    assert len(fixture["forms"]) == 1_363_274
    # The structural classes the gate strips before pinning. They are far
    # smaller than the April baseline's because that export's junk is gone
    # rather than grandfathered: truncated stems 8,971 -> 18, missing
    # breathings 166,226 -> 0.
    rejected = fixture["policy"]["rejected_by_class"]
    assert rejected["truncated-stem"] == 18
    assert rejected["common_word_respelling"] == 2_069
    assert "missing-initial-breathing" not in rejected


def test_heldout_fixture_pins_all_five_corpus_samples():
    fixture = load_gzip_fixture(DEFAULT_HELDOUT)

    assert set(fixture["corpora"]) == {
        "new_testament", "septuagint", "iliad_book_1",
        "herodotus_book_1", "katharevousa",
    }
    assert fixture["baseline"]["commit"] == (
        "1ac4f62ed6c3c5722f35ce518697220a6d5f41a5"
    )
    assert fixture["corpora"]["new_testament"]["total_tokens"] == 137_434
    assert fixture["corpora"]["septuagint"]["total_tokens"] == 583_774
    assert all(
        source["files"] and len(source["manifest_sha256"]) == 64
        for source in fixture["sources"].values()
    )


def test_textbook_paradigm_fixture_is_pinned_and_complete():
    forms, fixture = load_grc_textbook_forms()

    assert fixture["source"]["sha256"] == (
        "4b09d9736060335672db4b25475557a9d5ce9ad3c185a4139bfdf6a931b72fa2"
    )
    assert fixture["review"]["sha256"] == (
        "21dc3490f174a3a684abeb6d6764323986a05dce3af11f2f6fc8eae1b173e41b"
    )
    assert set(fixture["paradigms"]) == GRC_COMPLETE_PARADIGM_LEMMAS
    assert len(forms) == 2_154
    assert {"λύω", "λύοιμι", "παιδεύω", "τίθημι", "δίδωμι",
            "ἵστημι", "τιμάω", "ποιέω", "δηλόω"} <= forms
    # Reviewed corrections: generator garbage is gone, standard cells the
    # generator lacked are present, and no cell keeps vowel-length marks.
    assert not {"λύ", "λύσαν", "λύε", "ἵστω", "εἶτε", "ἔδων"} & forms
    assert {"λύεις", "λῦσαν", "λῦε", "δίδωσι", "τίθησι", "ἔδωκα",
            "ποιεῖσθαι"} <= forms
    # The generator keeps the accent that tells λύω's aorist optative 3sg
    # from its infinitive, and no longer reads a collapsed kaikki table row
    # as six spellings of τιμάω's first-person singular.
    assert {"λύσαι", "λῦσαι", "τιμῶ", "τιμᾷς"} <= forms
    assert "τιμάς" not in forms
    assert not any(
        ord(char) in (0x0304, 0x0306)
        for form in forms for char in unicodedata.normalize("NFD", form)
    )


def test_heldout_requirements_follow_the_current_orthography_policy():
    fixture = load_gzip_fixture(DEFAULT_HELDOUT)
    stale = sorted(
        form
        for corpus in fixture["corpora"].values()
        for form, _count in corpus["forms"]
        if grc_orthography_reason(form) is not None
        or form in BARE_ELISION_STEMS
    )
    # A rule change must be followed by regenerating the fixture.
    assert stale == []


def test_heldout_regressions_compare_both_ceilings():
    metrics = {
        "a": {"rejected_types": 3, "baseline_rejected_types": 3,
              "rejected_tokens": 9, "baseline_rejected_tokens": 10},
        "b": {"rejected_types": 4, "baseline_rejected_types": 3,
              "rejected_tokens": 9, "baseline_rejected_tokens": 10},
        "c": {"rejected_types": 2, "baseline_rejected_types": 3,
              "rejected_tokens": 11, "baseline_rejected_tokens": 10},
    }
    assert set(heldout_regressions(metrics)) == {"b", "c"}


def test_heldout_tokenizer_preserves_accents_and_splits_punctuation():
    assert exact_tokens("πειρασμὸς, δ’ἐγώ ηὕρισκον·") == [
        "πειρασμός", "δ᾽", "ἐγώ", "ηὕρισκον",
    ]


def test_top_lsj9_gate_normalizes_dictionary_display_notation():
    lemmas = load_top_lsj9_lemmas(2000)

    assert len(lemmas) == 2000
    assert "νικάω" in lemmas
    assert "νῑκάω" not in lemmas
    assert not any("-" in lemma for lemma in lemmas)


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
@pytest.mark.skipif(
    not FORM_PROFILE_DB.exists(), reason="form_profile.db not downloaded"
)
def test_expanded_export_accepts_frequency_head_and_reported_regressions(tmp_path):
    Dictionary = pytest.importorskip("spylls.hunspell").Dictionary
    fixture = _fixture()

    # Lemma admission still uses the aggregate, accent-stripped corpus map;
    # sparse acute forms use the complete accent-preserving corpus profile.
    freq = {
        strip_accents(row["form"]): row["count"]
        for row in fixture["forms"]
    }
    profile = load_form_profile_freq()
    for form in REPORTED_REGRESSIONS:
        freq[strip_accents(form)] = max(freq.get(strip_accents(form), 0), 3)

    conn = sqlite3.connect(f"file:{LOOKUP_DB}?mode=ro", uri=True)
    canonical_forms, canonical_lemmas = load_canonical_ag_sets()
    top_lemmas = load_top_lsj9_lemmas()
    pairs = select_forms(
        conn,
        "grc",
        keep_lemmas=(
            canonical_lemmas | top_lemmas | GRC_COMPLETE_PARADIGM_LEMMAS
        ),
        attestation_freq=profile.exact,
    )
    conn.close()
    textbook_forms, textbook_meta = load_grc_textbook_forms()
    pairs = filter_by_lemma_freq(
        pairs,
        freq,
        min_lemma_count=3,
        strict_acute_min=1,
        strict_form_freq_map=profile.exact,
        keep_forms=top_lemmas | textbook_forms,
    )
    compatibility_forms, _compatibility_meta = load_grc_compatibility_forms()
    # The same helpers, in the same order, as export_hunspell.run_export.
    pairs, _added = add_grc_reviewed_forms(
        pairs,
        GRC_CLOSED_LIST_FORMS,
        textbook_meta["paradigms"],
        compatibility_forms,
    )
    pairs, _changed, _dropped = sanitize_export_pairs(pairs)
    pairs, report = finalize_grc_pairs(
        pairs,
        evidence=profile,
        compatibility_forms=compatibility_forms,
        textbook_forms=textbook_forms,
        export_overrides=GRC_CLOSED_LIST_FORMS,
        protected_forms=grc_pinned_forms(
            canonical_forms, top_lemmas, textbook_forms,
            load_lm_head_required_forms(),
        ),
    )
    assert report["invalid"]
    assert report["dominated"]
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
    assert not any(
        dictionary.lookup(stem) for stem in REPORTED_BARE_ELISION_FALLBACKS
    )
    assert dictionary.lookup("ἄν")
    assert all(dictionary.lookup(form) for form in HOMERIC_SHORT_PREPOSITIONS)

    missing, accepted_exclusions = audit_dictionary(
        tmp_path / "coverage", fixture, exclusions
    )
    assert missing == []
    assert accepted_exclusions == []
    invalid_initial, synthetic_flags = audit_export_orthography(
        tmp_path / "coverage",
        reviewed=compatibility_forms | set(GRC_CLOSED_LIST_FORMS),
    )
    assert invalid_initial == []
    assert synthetic_flags == []
    # The release contract: every reviewed shipped form, citation headword,
    # textbook cell, and acute twin, no new weak respelling, and no
    # regression-corpus rejection count above the held-out baseline's.
    whole = audit_whole_artifact(tmp_path / "coverage")
    assert {key: rows for key, rows in whole.items() if rows} == {}
    assert audit_textbook_paradigms(tmp_path / "coverage") == []
    assert heldout_regressions(
        measure_heldout_corpora(tmp_path / "coverage")
    ) == {}
