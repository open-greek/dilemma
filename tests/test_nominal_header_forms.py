"""A header suffix cannot supply a complete Ancient inflection."""

import json

import pytest

from build_data import extract_pairs
from dilemma.nominal_header_forms import is_abbreviated_nominal_header


def _entry(word="αὐλή", pos="noun", form="ῆς", tags=None, source="header"):
    return {"word": word, "pos": pos, "senses": [{"glosses": ["courtyard"]}],
            "forms": [{"form": form, "tags": tags or ["feminine"], "source": source}]}


@pytest.mark.parametrize("word,form", [
    ("αὐλή", "ῆς"), ("τάφος", "ου"), ("βλωμός", "οῦ"),
    ("Ζεύς", "ός"), ("Ὀδυσσεύς", "ος"), ("Κυδίππη", "ης"),
    ("ἐτυμολογία", "ας"), ("σμῶδιξ", "γος"),
])
def test_unqualified_header_suffix_is_not_a_word(word, form):
    entry = _entry(word=word, form=form)
    assert is_abbreviated_nominal_header(entry, entry["forms"][0])


@pytest.mark.parametrize("word,form,tags,source", [
    ("αὐλή", "αὐλῆς", ["genitive", "singular"], "header"),
    ("Ἥρα", "Ἥρη", ["feminine"], "header"),
    ("ῥοή", "ῥοά", ["Doric"], "header"),
    ("ἔαρ", "ἔᾰρ", ["neuter"], "header"),
    ("ὅς", "οὗ", ["genitive"], "header"),
    ("οὗ", "οὗ", ["masculine"], "header"),
    ("ἕ", "ὅς", ["masculine"], "header"),
    ("αὐλή", "ῆς", ["feminine", "alternative"], "header"),
    ("αὐλή", "ῆς", ["genitive", "singular"], "declension"),
])
def test_complete_marked_or_independent_readings_are_retained(word, form, tags, source):
    entry = _entry(word=word, form=form, tags=tags, source=source)
    assert not is_abbreviated_nominal_header(entry, entry["forms"][0])


@pytest.mark.parametrize("lang", ["grc-en", "grc-el"])
def test_header_guard_precedes_all_extracted_products(tmp_path, lang):
    entry = _entry()
    entry["forms"].append({"form": "αὐλῆς", "tags": ["genitive", "singular"]})
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps(entry, ensure_ascii=False) + "\n")
    pairs, lookup, _, typed, *_ = extract_pairs(path, lang)
    assert not any(p["form"] in {"ῆς", "ής"} for p in pairs)
    assert not ({"ῆς", "ής", "ης"} & lookup.keys())
    assert not ({"ῆς", "ής", "ης"} & typed.keys())
    assert lookup["αὐλῆς"][0] == "αὐλή"


@pytest.mark.parametrize("lang,pos", [("el-el", "noun"), ("med-en", "noun"),
                                     ("grc-el", "pron"), ("grc-en", "adj")])
def test_other_languages_and_parts_of_speech_are_unchanged(tmp_path, lang, pos):
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps(_entry(pos=pos), ensure_ascii=False) + "\n")
    pairs, *_ = extract_pairs(path, lang)
    assert any(p["form"] == "ῆς" for p in pairs)
