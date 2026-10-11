"""Do not export the pinned Lua module's malformed periphrastic cells."""
import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))
import expand_lsj


@pytest.fixture(autouse=True)
def clear_cache():
    expand_lsj._PP_CACHE.clear()
    yield
    expand_lsj._PP_CACHE.clear()


def test_only_perfect_middle_periphrastic_cell_auxiliaries_are_omitted():
    class Wtp:
        calls = []

        def start_page(self, _word):
            pass

        def expand(self, template):
            self.calls.append(template)
            return "<span>κἀγώ</span>"

    wtp = Wtp()
    original = ["perf", "λελυκ", "λελυ"]
    assert expand_lsj._expand_principal_part(wtp, "λύω", original) == {"κἀγώ"}
    assert original == ["perf", "λελυκ", "λελυ"]
    assert wtp.calls[0] == "{{grc-conj|perf|λελυκ|λελυ}}"
    expand_lsj._expand_principal_part(wtp, "λύω", ["plup", "λελυκ", "λελυ"])
    assert wtp.calls[1] == "{{grc-conj|plup|λελυκ|λελυ}}"
    expand_lsj._expand_principal_part(wtp, "λύω", ["fut", "λυσ"])
    assert wtp.calls[2] == "{{grc-conj|fut|λυσ}}"


def test_periphrastic_boundary_retains_the_supplied_participle_token():
    table = '''! rowspan="4" | active
! subjunctive
| class="grc-conj-finite-form" | λελύκω
! rowspan="4" | middle/<wbr>passive
! indicative
| class="grc-conj-finite-form" | λέλυμαι
! subjunctive
| class="grc-conj-finite-form" | λελυμένοι λελυὦμεν
! optative
| class="grc-conj-finite-form" | λελυμένω λελυεἰήτην,<br>εἴτην
! imperative
| class="grc-conj-finite-form" | λέλυσο
κἀγώ
'''
    fixed = expand_lsj._perfect_participle_cells(table)
    assert "λελυὦμεν" not in fixed and "λελυεἰήτην" not in fixed and "εἴτην" not in fixed
    assert {"λελύκω", "λέλυμαι", "λελυμένοι", "λελυμένω", "λέλυσο", "κἀγώ"} <= expand_lsj.parse_html_forms(fixed, "λύω")


@pytest.mark.skipif(
    not expand_lsj.WTP_DB.exists() or importlib.util.find_spec("wikitextprocessor") is None,
    reason="requires the optional pinned Wiktionary Lua cache and runtime",
)
def test_real_perfect_keeps_synthetic_cells_and_participles_without_joined_auxiliaries():
    wtp = expand_lsj.get_wtp()
    forms = expand_lsj._expand_principal_part(wtp, "λύω", ["perf", "λελυκ", "λελυ"])
    assert {"λέλυκα", "λέλυμαι", "λελυκώς", "λελυμένος", "λελυμένοι", "λελυμένω"} <= forms
    assert not {"λελυὦ", "λελυὦμεν", "λελυεἰήτην", "λελυεἴημεν", "εἶμεν"} & forms
    # The correction is a cell override, not a ban on medial breathings.
    assert expand_lsj.parse_html_forms("<span>κἀγώ</span>", "ἐγώ") == {"κἀγώ"}
