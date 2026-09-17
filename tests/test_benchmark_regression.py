import unicodedata

import pytest

from scripts.check_benchmark_regressions import (
    BENCHMARK_DIR,
    SPECS,
    equivalent,
    load_gold_fixture,
    score_predictions,
)


def test_committed_gold_fixtures_are_valid_utf8_nfc_tsv():
    for spec in SPECS:
        pairs = load_gold_fixture(BENCHMARK_DIR / spec.fixture)
        assert pairs
        assert all(unicodedata.is_normalized("NFC", value)
                   for pair in pairs for value in pair)


def test_fixture_validation_rejects_wrong_column_count(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("form\tlemma\textra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly two TSV columns"):
        load_gold_fixture(path)


def test_scores_equivalence_without_hiding_strict_change():
    pairs = [("form", "λέγω")]
    equivalences = {"λέγω": {"λέγω", "φημί"}, "φημί": {"λέγω", "φημί"}}
    assert equivalent("φημί", "λέγω", equivalences)
    assert not equivalent("γράφω", "λέγω", equivalences)
    assert score_predictions(pairs, ["φημί"], equivalences) == {
        "tokens": 1,
        "strict_correct": 0,
        "equiv_correct": 1,
    }
