"""Different positively identified words retain their own frequency evidence."""
from collections import Counter

import pytest

import export_mg_polytonic as mg


@pytest.mark.parametrize("evidence,heads,expected", [
    ({}, {"στυλό", "στύλος"}, "R"),
    ({"στυλό": frozenset({("στυλό", "noun nominative singular")}),
      "στῦλο": frozenset({("στύλος", "noun accusative singular")})}, {"στυλό"}, "R"),
    ({"στυλό": frozenset({("στυλό", "noun nominative singular")}),
      "στῦλο": frozenset({("στύλος", "noun accusative singular")})}, {"στυλό", "στύλος"}, "M"),
    ({"στυλό": frozenset({("στύλος", "noun nominative singular")}),
      "στῦλο": frozenset({("στύλος", "noun accusative singular")})}, {"στύλος"}, "R"),
])
def test_separate_source_words_do_not_share_a_frequency_ceiling(monkeypatch, evidence, heads, expected):
    monkeypatch.setattr(mg, "_verb_form_keys", lambda: set())
    frequencies = {"στυλό": 2000, "στύλο": 100, "άλλο": 99_997_900}
    result = mg.list_buckets({"στύλο": 3}, {"στυλό": 2000}, frequencies, Counter(),
                            realizations=evidence, lexical_heads=frozenset(heads))
    assert result["στυλό"] == expected


def test_different_forms_of_the_same_verb_keep_the_ceiling(monkeypatch):
    monkeypatch.setattr(mg, "_verb_form_keys", lambda: set())
    result = mg.list_buckets({"μιλᾶ": 3}, {"μίλα": 9000},
        {"μιλά": 1000, "μίλα": 9000, "άλλο": 99_990_000}, Counter(),
        realizations={"μιλᾶ": frozenset({("μιλάω", "verb present singular third-person")}),
                      "μίλα": frozenset({("μιλάω", "verb imperative singular second-person")})},
        lexical_heads=frozenset({"μιλάω"}))
    assert result["μίλα"] == result["μιλᾶ"] == "M"
