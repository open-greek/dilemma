"""Noun memoization must preserve each lexeme's marked paradigm."""
import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))
import expand_lsj


@pytest.fixture(autouse=True)
def clear_noun_cache():
    expand_lsj._NOUN_CACHE.clear()
    yield
    expand_lsj._NOUN_CACHE.clear()


class FakeWtp:
    def __init__(self):
        self.calls = []

    def start_page(self, word):
        self.word = word

    def expand(self, template):
        self.calls.append((self.word, template))
        return self.word


def test_same_declension_words_are_expanded_independently(monkeypatch):
    expected = {
        "ἄνθρωπος": {"ἄνθρωπος", "ἀνθρώπου"},
        "ἄγγελος": {"ἄγγελος", "ἀγγέλου"},
    }
    monkeypatch.setattr(expand_lsj, "parse_html_forms",
                        lambda html, word: expected[word].copy())
    wtp = FakeWtp()
    assert expand_lsj.expand_noun(wtp, "ἄνθρωπος", "m", "ἀνθρώπου") == (expected["ἄνθρωπος"], "")
    assert expand_lsj.expand_noun(wtp, "ἄγγελος", "m", "ἀγγέλου") == (expected["ἄγγελος"], "")
    assert [word for word, _ in wtp.calls] == ["ἄνθρωπος", "ἄγγελος"]


def test_identical_resolved_request_preserves_marks_and_cache(monkeypatch):
    expected = {"ἄνθρωπος", "ἀνθρώπου"}
    monkeypatch.setattr(expand_lsj, "parse_html_forms", lambda html, word: expected.copy())
    monkeypatch.setattr(expand_lsj, "infer_genitive", lambda *args: "ἀνθρώπου")
    wtp = FakeWtp()
    cold, error = expand_lsj.expand_noun(wtp, "ἄνθρωπος", "m")
    assert not error and cold == expected
    warm, error = expand_lsj.expand_noun(wtp, "ἄνθρωπος", "m", "ἀνθρώπου")
    assert not error and warm == cold
    warm.clear()
    assert expand_lsj.expand_noun(wtp, "ἄνθρωπος", "m", "ἀνθρώπου")[0] == cold
    assert len(wtp.calls) == 1


@pytest.mark.skipif(
    not expand_lsj.WTP_DB.exists() or importlib.util.find_spec("wikitextprocessor") is None,
    reason="requires the optional pinned Wiktionary Lua cache and runtime",
)
def test_real_marked_forms_match_with_a_warm_cache():
    wtp = expand_lsj.get_wtp()
    cold, error = expand_lsj.expand_noun(wtp, "λόγος", "m", "λόγου")
    assert not error and "λόγου" in cold
    expand_lsj._NOUN_CACHE.clear()
    other, error = expand_lsj.expand_noun(wtp, "ἄνθρωπος", "m", "ἀνθρώπου")
    assert not error and other
    warm, error = expand_lsj.expand_noun(wtp, "λόγος", "m", "λόγου")
    assert not error and warm == cold
    assert expand_lsj.expand_noun(wtp, "λόγος", "m", "λόγου")[0] == cold
