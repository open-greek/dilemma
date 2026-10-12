"""Present-system memoization must not transplant another verb's suffixes."""
import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))
import expand_lsj


@pytest.fixture(autouse=True)
def clear_verb_cache():
    expand_lsj._VERB_CACHE.clear()
    yield
    expand_lsj._VERB_CACHE.clear()


class FakeWtp:
    def __init__(self):
        self.calls = []

    def start_page(self, word):
        self.word = word

    def expand(self, template):
        self.calls.append((self.word, template))
        return self.word


def test_same_conjugation_verbs_preserve_their_own_stems_and_marks(monkeypatch):
    expected = {
        "ἀκούω": {"ἀκούω", "ἀκούεις", "ἄκουε"},
        "λύω": {"λύω", "λύεις", "λῦε"},
    }
    monkeypatch.setattr(expand_lsj, "parse_html_forms",
                        lambda html, word: expected[word].copy())
    wtp = FakeWtp()
    assert expand_lsj.expand_verb(wtp, "ἀκούω") == (expected["ἀκούω"], "")
    assert expand_lsj.expand_verb(wtp, "λύω") == (expected["λύω"], "")
    assert [word for word, _ in wtp.calls] == ["ἀκούω", "λύω"]
    cached, error = expand_lsj.expand_verb(wtp, "ἀκούω")
    assert not error and cached == expected["ἀκούω"]
    cached.clear()
    assert expand_lsj.expand_verb(wtp, "ἀκούω")[0] == expected["ἀκούω"]
    assert len(wtp.calls) == 2


def test_present_cache_does_not_cache_other_head_text_principal_parts(monkeypatch):
    monkeypatch.setattr(expand_lsj, "parse_html_forms", lambda html, word: {"λύω", "λύεις"})
    monkeypatch.setattr(expand_lsj, "expand_principal_parts",
                        lambda wtp, word, text: {text})
    wtp = FakeWtp()
    assert expand_lsj.expand_verb(wtp, "λύω", "ἔλυσα")[0] == {"λύω", "λύεις", "ἔλυσα"}
    assert expand_lsj.expand_verb(wtp, "λύω", "λέλυκα")[0] == {"λύω", "λύεις", "λέλυκα"}
    assert len(wtp.calls) == 1


@pytest.mark.skipif(
    not expand_lsj.WTP_DB.exists() or importlib.util.find_spec("wikitextprocessor") is None,
    reason="requires the optional pinned Wiktionary Lua cache and runtime",
)
def test_real_present_forms_match_with_a_warm_cache():
    wtp = expand_lsj.get_wtp()
    cold, error = expand_lsj.expand_verb(wtp, "λύω")
    assert not error and "λύεις" in cold
    expand_lsj._VERB_CACHE.clear()
    other, error = expand_lsj.expand_verb(wtp, "ἀκούω")
    assert not error and other
    warm, error = expand_lsj.expand_verb(wtp, "λύω")
    assert not error and warm == cold
    assert expand_lsj.expand_verb(wtp, "λύω")[0] == cold
