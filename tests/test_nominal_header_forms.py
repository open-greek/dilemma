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


# A Modern Greek header suffix is dropped by its own rule
# (is_modern_nominal_fragment, tested below).
@pytest.mark.parametrize("lang,pos", [("med-en", "noun"),
                                     ("grc-el", "pron"), ("grc-en", "adj")])
def test_other_languages_and_parts_of_speech_are_unchanged(tmp_path, lang, pos):
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps(_entry(pos=pos), ensure_ascii=False) + "\n")
    pairs, *_ = extract_pairs(path, lang)
    assert any(p["form"] == "ῆς" for p in pairs)


# Modern Greek table fragments (#122) -----------------------------------

from dilemma.nominal_header_forms import is_modern_nominal_fragment  # noqa: E402


def _modern(word, pos, form, tags, source="declension"):
    entry = {"word": word, "pos": pos, "lang_code": "el",
             "senses": [{"glosses": ["x"]}],
             "forms": [{"form": form, "tags": tags, "source": source}]}
    return entry, entry["forms"][0]


@pytest.mark.parametrize("word,pos,form,tags,source", [
    ("αρχαΐζων", "adj", "ούσης", ["feminine", "genitive", "singular"], "declension"),
    ("αρχαΐζων", "adj", "όντων", ["genitive", "masculine", "plural"], "declension"),
    ("αρχαΐζων", "adj", "ουσών", ["feminine", "genitive", "plural"], "declension"),
    ("αισώπειος", "adj", "ος", ["masculine", "nominative", "singular"], "declension"),
    ("αισώπειος", "adj", "ε", ["masculine", "vocative", "singular"], "declension"),
    ("κρεατομηχανή", "noun", "μηχανή", ["nominative", "singular"], "declension"),
    ("μουσικομανής", "adj", "ές", [], "header"),
    ("βρόντημα", "noun", "γ", ["neuter"], "header"),
])
def test_modern_stemless_cells_are_fragments(word, pos, form, tags, source):
    assert is_modern_nominal_fragment(*_modern(word, pos, form, tags, source))


@pytest.mark.parametrize("word,pos,form,tags,source", [
    ("αρχαΐζων", "adj", "αρχαΐζοντος", ["genitive", "masculine", "singular"], "declension"),
    ("κακός", "adj", "χειρότερος", ["comparative", "masculine", "nominative"], "declension"),
    ("Ζευς", "name", "Δία", ["accusative", "singular"], "declension"),
    ("ναργιλές", "noun", "αργιλές", ["masculine"], "header"),
    ("αρχαΐζων", "adj", "αρχαΐζων", ["masculine", "nominative", "singular"], "declension"),
    ("αρχαΐζων", "verb", "ούσης", ["feminine", "genitive", "singular"], "declension"),
    ("αισώπειος", "adj", "ος", ["alternative"], "header"),
])
def test_complete_modern_cells_are_kept(word, pos, form, tags, source):
    assert not is_modern_nominal_fragment(*_modern(word, pos, form, tags, source))


def test_modern_fragment_guard_precedes_all_extracted_products(tmp_path):
    entry, _ = _modern("αρχαΐζων", "adj", "αρχαΐζοντος",
                       ["genitive", "masculine", "singular"])
    entry["forms"] += [
        {"form": "ούσης", "tags": ["feminine", "genitive", "singular"],
         "source": "declension"},
        {"form": "όντων", "tags": ["genitive", "masculine", "plural"],
         "source": "declension"},
    ]
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps(entry, ensure_ascii=False) + "\n")
    pairs, lookup, _, typed, *_ = extract_pairs(path, "el-en")
    fragments = {"ούσης", "ουσης", "όντων", "οντων"}
    assert not any(p["form"] in fragments for p in pairs)
    assert not (fragments & lookup.keys())
    assert not (fragments & typed.keys())
    assert lookup["αρχαΐζοντος"][0] == "αρχαΐζων"


def test_ancient_tables_ignore_the_modern_fragment_rule(tmp_path):
    entry = _entry(word="αὐλή", form="αὐλῶν", tags=["genitive", "plural"],
                   source="declension")
    entry["forms"].append({"form": "ῶν", "tags": ["genitive", "plural"],
                           "source": "declension"})
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps(entry, ensure_ascii=False) + "\n")
    pairs, *_ = extract_pairs(path, "grc-en")
    assert any(p["form"] == "ῶν" for p in pairs)
