"""A present-system template must match the lexical stem and voice."""
import importlib.util
from pathlib import Path
import sys
import unicodedata

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))
import expand_lsj


@pytest.fixture(autouse=True)
def clear_cache():
    expand_lsj._VERB_CACHE.clear()
    yield
    expand_lsj._VERB_CACHE.clear()


@pytest.mark.parametrize("word,kind,stem", [
    ("ἥδομαι", "pres", "ἡδ"),
    ("ῥυΐσκομαι", "pres", "ῥυϊσκ"),
    ("ἐράομαι", "pres-con-a", "ἐρ"),
    ("ἐνθυμέομαι", "pres-con-e", "ἐνθυμ"),
    ("ὑπισχόομαι", "pres-con-o", "ὑπισχ"),
    ("ἀμφιέννυμι", "pres-numi", "ἀμφιεν"),
    ("ἔννυμι", "pres-numi", "ἐν"),
])
def test_supported_stems_in_both_unicode_forms(word, kind, stem):
    expected = (kind, stem)
    assert expand_lsj._classify_verb(word) == expected
    assert expand_lsj._classify_verb(unicodedata.normalize("NFD", word)) == expected


@pytest.mark.parametrize("word", [
    "τιμῶμαι", "ποιοῦμαι", "δίδωμαι", "κάθημαι", "ἐπίσταμαι", "ἀμφάγνυμαι",
])
def test_unproven_middle_types_fail_without_calling_lua(word):
    assert expand_lsj.expand_verb(None, word) == (set(), "unsupported-middle-ending")


def test_middle_template_requests_only_middle_passive(monkeypatch):
    class Wtp:
        def start_page(self, word):
            assert word == "ἥδομαι"

        def expand(self, template):
            assert template == "{{grc-conj|pres|ἡδ|form=mp}}"
            return "ἥδομαι"

    monkeypatch.setattr(expand_lsj, "parse_html_forms", lambda html, _: {html})
    assert expand_lsj.expand_verb(Wtp(), "ἥδομαι") == ({"ἥδομαι"}, "")


@pytest.mark.parametrize("word", ["ὑγροποιέω", "δίδωμι"])
def test_lua_failure_is_reported_without_changing_conjugation(word):
    class Wtp:
        calls = 0

        def start_page(self, _word):
            pass

        def expand(self, _template):
            self.calls += 1
            raise RuntimeError("fixture Lua failure")

    wtp = Wtp()
    assert expand_lsj.expand_verb(wtp, word) == (set(), "fixture Lua failure")
    assert wtp.calls == 1
    assert not expand_lsj._VERB_CACHE


def test_only_exact_phemi_uses_the_reviewed_explicit_paradigm():
    forms, error = expand_lsj.expand_verb(None, "φημί")
    assert not error
    assert {"φημί", "φῄς", "φησίν"} <= forms
    assert not {"φῆμι", "φέμαι"} & forms
    assert expand_lsj.expand_verb(None, "φημι") == (set(), "unsupported-phemi-family")


@pytest.mark.parametrize("word", ["σύμφημι", "πρόφημι", "διφήμι", "παρακαρπίφημι"])
def test_phemi_suffix_does_not_license_a_compound_paradigm(word):
    assert expand_lsj.expand_verb(None, word) == (set(), "unsupported-phemi-family")
    assert not expand_lsj._expand_irregular_compound(word)


@pytest.mark.skipif(
    not expand_lsj.WTP_DB.exists() or importlib.util.find_spec("wikitextprocessor") is None,
    reason="requires the optional pinned Wiktionary Lua cache and runtime",
)
@pytest.mark.parametrize("word,good,bad", [
    ("ἥδομαι", "ἡδόμεθα", "ἥδω"),
    ("ἀβακίζομαι", "ἀβακιζόμεθα", "ἀβακιζόομαι"),
    ("ῥυΐσκομαι", "ῥυϊσκόμεθα", "ῥυΐσκω"),
    ("ἐράομαι", "ἐρώμεθα", "ἐρᾶτε"),
    ("ἐνθυμέομαι", "ἐνθυμούμεθα", "ἐνθυμῶ"),
    ("ὑπισχόομαι", "ὑπισχούμεθα", "ὑπισχῶ"),
    ("ἀμφιέννυμι", "ἀμφιέννυμεν", "ἀμφιένυμι"),
    ("ἔννυμι", "ἔννυμεν", "ἔνυμι"),
])
def test_real_templates_preserve_head_and_correct_voice(word, good, bad):
    forms, error = expand_lsj.expand_verb(expand_lsj.get_wtp(), word)
    assert not error
    assert word in forms
    assert good in forms
    assert bad not in forms
