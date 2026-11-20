"""A word typed with the micro sign (U+00B5) reads as its Greek spelling with
mu (U+03BC) (#144)."""
import pytest

from dilemma import core
from dilemma.core import MICRO_SIGN, to_greek_mu, to_standard_sigma

TYPED = "γνω" + MICRO_SIGN + "οδότου"


def test_micro_sign_is_written_as_mu():
    assert MICRO_SIGN == "µ" and "µ" != "μ"
    assert to_greek_mu(TYPED) == "γνωμοδότου"
    assert to_greek_mu("λόγος") == "λόγος"
    # The early normalization of every lemmatize method goes through it.
    assert to_standard_sigma(MICRO_SIGN + "έροϲ") == "μέρος"


_MODEL = []


def _model():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    if not _MODEL:
        _MODEL.append(core.Dilemma(lang="all"))
    return _MODEL[0]


def test_every_method_reads_the_micro_sign_as_mu():
    model = _model()
    expected = model.lemmatize("γνωμοδότου", guess=False)
    assert expected == "γνωμοδότης"
    assert model.lemmatize(TYPED, guess=False) == expected
    assert model.lemmatize_batch([TYPED], guess=False) == [expected]
    assert model.lemmatize_pos(TYPED, "NOUN", guess=False) == expected
    assert model.lemmatize_batch_pos([TYPED], ["NOUN"], guess=False) == [expected]
    assert model.lemmatize_verbose(TYPED, guess=False)[0].lemma == expected
