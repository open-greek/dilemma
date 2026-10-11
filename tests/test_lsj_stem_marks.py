"""Lexical marks remain available to the Greek conjugation module."""
import importlib.util
from pathlib import Path
import sys
import unicodedata

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))
import expand_lsj


@pytest.mark.parametrize("headword,kind,stem", [
    ("ὑγροποιέω", "pres-con-e", "ὑγροποι"),
    ("ὁράω", "pres-con-a", "ὁρ"),
    ("δηλόω", "pres-con-o", "δηλ"),
    ("ἀπόλλυμι", "pres-lumi", "ἀπολ"),
    ("ῥήγνυμι", "pres-numi", "ῥηγ"),
    ("ὑπεκδίδωμι", "pres-omi", "ὑπεκδιδ"),
    ("προΐστημι", "pres-emi", "προϊστ"),
    ("ἀΐσσω", "pres", "ἀϊσσ"),
    ("ζῳοποιέω", "pres-con-e", "ζῳοποι"),
    ("κρῑ́νω", "pres", "κρῑν"),
    ("δῐώκω", "pres", "δῐωκ"),
    ("ἥδομαι", "pres", "ἡδο"),
])
def test_classification_keeps_lexical_marks_in_nfc_and_nfd(headword, kind, stem):
    expected = (kind, unicodedata.normalize("NFC", stem))
    assert expand_lsj._classify_verb(headword) == expected
    assert expand_lsj._classify_verb(unicodedata.normalize("NFD", headword)) == expected


def test_stem_normalization_removes_only_tone_accents():
    assert expand_lsj._unaccented_verb_stem("ᾄ\u0304\u0306") == unicodedata.normalize(
        "NFC", "α\u0313\u0345\u0304\u0306")
    assert expand_lsj._unaccented_verb_stem("ΐῥ") == "ϊῥ"


@pytest.mark.skipif(
    not expand_lsj.WTP_DB.exists() or importlib.util.find_spec("wikitextprocessor") is None,
    reason="requires the optional pinned Wiktionary Lua cache and runtime",
)
@pytest.mark.parametrize("headword", [
    "ὑγροποιέω", "ἁρπάζω", "ἀΐσσω", "ῥίπτω", "ζῳοποιέω",
    "ὑπεκδίδωμι", "ἀνατίθημι", "ὄλλυμι", "ῥήγνυμι", "ἵστημι",
])
def test_real_lua_preserves_marked_head_form(headword):
    expand_lsj._VERB_CACHE.clear()
    forms, error = expand_lsj.expand_verb(expand_lsj.get_wtp(), headword)
    assert not error and headword in forms


@pytest.mark.skipif(
    not expand_lsj.WTP_DB.exists() or importlib.util.find_spec("wikitextprocessor") is None,
    reason="requires the optional pinned Wiktionary Lua cache and runtime",
)
def test_quantity_marks_reach_lua_without_changing_unmarked_control():
    wtp = expand_lsj.get_wtp()
    expand_lsj._VERB_CACHE.clear()
    forms, error = expand_lsj.expand_verb(wtp, "λύω")
    assert not error
    for word, expected in [("κρῑ́νω", "κρίνω"), ("δῐώκω", "διώκω"), ("λῡ́ω", "λύω")]:
        marked, error = expand_lsj.expand_verb(wtp, word)
        assert not error and expected in marked
    # The existing unmarked request retains Lua's accent placement.
    assert {"λύω", "λύεις", "λῦε"} <= forms
