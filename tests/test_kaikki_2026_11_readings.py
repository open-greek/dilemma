"""Reviewed readings that the 2026-11 kaikki dumps changed for the worse."""
import inspect
import json
from pathlib import Path

import pytest

import build_lookup_db as B

ROOT = Path(__file__).resolve().parents[1]


def test_overrides_restore_the_treebank_readings():
    o = B.KAIKKI_2026_11_READING_OVERRIDES
    assert o["ην"] == o["ήν"] == "εἰμί"
    assert "ἤν" not in o          # accented ἤν is ἐάν, as the dumps have it
    assert o["μεγάλα"] == o["μεγαλα"] == "μέγας"
    assert o["εἴσομαι"] == o["είσομαι"] == o["εισομαι"] == "οἶδα"
    assert o["ἑψόμενα"] == "ἕψω"
    assert o["τοῖ"] == "σύ"
    assert o["κἆτα"] == "εἶτα"


def test_overrides_are_applied_with_the_reviewed_lookup_overrides():
    source = inspect.getsource(B.main) if hasattr(B, "main") else inspect.getsource(B)
    assert "**KAIKKI_2026_11_READING_OVERRIDES" in source


def test_override_targets_are_lexical_heads():
    path = ROOT / "data" / "grc_lexical_heads.json"
    if not path.exists():
        pytest.skip("grc_lexical_heads.json is not downloaded")
    heads = json.loads(path.read_text(encoding="utf-8"))["heads"]
    for lemma in set(B.KAIKKI_2026_11_READING_OVERRIDES.values()):
        assert lemma in heads, lemma
