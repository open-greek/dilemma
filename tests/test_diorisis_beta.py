"""Beta Code conversion of Diorisis tokens (extract_diorisis_lm.beta_to_nfc)."""
from __future__ import annotations

import pytest

pytest.importorskip("betacode")

from extract_diorisis_lm import _run_sanity_checks, beta_to_nfc  # noqa: E402


@pytest.mark.parametrize("beta,expected", [
    # A trailing apostrophe is elision, including after an unaccented vowel
    # that the betacode package would otherwise mark with a macron.
    ("di'", "δι’"),
    ("*di'", "Δι’"),
    ("a'", "α’"),
    ("tou'", "του’"),
    ("dikai'", "δικαι’"),
    # Unchanged cases: consonant-final and accented elisions.
    ("par'", "παρ’"),
    ("di/'", "δί’"),
    ("a)ll'", "ἀλλ’"),
    ("tw=nd'", "τῶνδ’"),
    # The paren written for elision after a consonant.
    ("par)", "παρ’"),
    # No elision.
    ("a)/ndra", "ἄνδρα"),
])
def test_beta_to_nfc_elision(beta, expected):
    assert beta_to_nfc(beta) == expected


def test_no_macron_from_an_elision_mark():
    for beta in ("di'", "a'", "tou'", "po/tni'", "e)sti'"):
        assert "̄" not in __import__("unicodedata").normalize(
            "NFD", beta_to_nfc(beta)
        )


def test_sanity_checks_pass():
    _run_sanity_checks()


@pytest.mark.parametrize("beta,expected", [
    ("di)", "δι"),      # Diorisis's elision paren: not a breathing on ι
    ("par)", "παρ"),
    ("a)ll)", "ἀλλ"),
    ("ou)", "οὐ"),      # a real breathing on a short vowel word
])
def test_pairs_builder_drops_the_elision_paren(beta, expected):
    pytest.importorskip("beta_code")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))
    import build_diorisis_pairs

    assert build_diorisis_pairs.betacode_to_unicode(beta) == expected
