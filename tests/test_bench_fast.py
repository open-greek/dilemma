"""Missing predictions count as benchmark errors without crashing scoring."""
from bench_fast import are_equivalent


def test_missing_prediction_is_incorrect_even_for_identity_gold():
    assert are_equivalent(None, "λόγος") is False


def test_existing_accent_insensitive_scoring_is_preserved():
    assert are_equivalent("λόγος", "λογος") is True
    assert are_equivalent("λόγος", "γράφω") is False
