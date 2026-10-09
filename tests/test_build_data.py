"""Focused tests for Kaikki build-time morphology extraction."""

from __future__ import annotations

import json

import pytest

from build_data import _parse_verb_tense, extract_pairs, resolve_cross_source


@pytest.mark.parametrize(("header", "expected"), [
    ("present", "present"),
    ("Epic aorist", "aorist"),
    ("Attic contracted future", "future"),
    ("future perfect", "future-perfect"),
    ("Attic declension-3", ""),
    ("Epic", ""),
])
def test_parse_verb_tense(header, expected):
    assert _parse_verb_tense(header) == expected


def test_extract_pairs_propagates_table_tense(tmp_path):
    dump = tmp_path / "kaikki.jsonl"
    entry = {
        "word": "λύω",
        "pos": "verb",
        "senses": [{"glosses": ["loosen"]}],
        "forms": [
            {"form": "Epic aorist", "tags": ["table-tags"]},
            {
                "form": "ἔλυσα",
                "tags": [
                    "active", "indicative", "first-person", "singular",
                ],
            },
            {"form": "Attic declension-3", "tags": ["table-tags"]},
            {
                "form": "λύων",
                "tags": ["active", "participle", "nominative"],
            },
        ],
    }
    dump.write_text(json.dumps(entry, ensure_ascii=False) + "\n",
                    encoding="utf-8")

    pairs, *_ = extract_pairs(dump, "grc-en")
    by_form = {pair["form"]: pair for pair in pairs}

    assert "aorist" in by_form["ἔλυσα"]["tags"]
    assert "Epic" in by_form["ἔλυσα"]["tags"]
    assert not ({"present", "imperfect", "future", "aorist", "perfect",
                 "pluperfect", "future-perfect"}
                & set(by_form["λύων"]["tags"]))


def _write(tmp_path, entries):
    dump = tmp_path / "kaikki.jsonl"
    dump.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n"
                            for e in entries), encoding="utf-8")
    return dump


def test_a_capitalized_modern_page_does_not_take_a_lowercase_word(tmp_path):
    # The female name Ερωτώ and the abbreviation ΔΕΣ must not become the
    # lemma of the verb ερωτώ or of δες, whose own pages are form-of pages
    # (the lowest confidence).
    dump = _write(tmp_path, [
        {"word": "Ερωτώ", "pos": "name",
         "senses": [{"glosses": ["γυναικείο όνομα"]}]},
        {"word": "ΔΕΣ", "pos": "verb",
         "senses": [{"glosses": ["συντομογραφία"]}]},
        {"word": "ερωτώ", "pos": "verb",
         "senses": [{"glosses": ["λόγια μορφή του ρωτώ"],
                     "form_of": [{"word": "ρωτώ"}]}]},
        {"word": "δες", "pos": "verb",
         "senses": [{"glosses": ["β' ενικό προστακτικής του βλέπω"],
                     "form_of": [{"word": "βλέπω"}]}]},
    ])
    _, lookup, *_ = extract_pairs(dump, "el-el")
    assert lookup["ερωτώ"][0] == "ρωτώ"
    assert lookup["δες"][0] == "βλέπω"
    # The capitalized pages keep their own keys.
    assert lookup["Ερωτώ"][0] == "Ερωτώ"
    # An Ancient Greek dump is read as before.
    _, lookup, *_ = extract_pairs(dump, "grc-el")
    assert lookup["ερωτώ"][0] == "Ερωτώ"


def test_cross_source_resolution_makes_no_cycle():
    # EL's ρωτώ table lists ρωτάω; EN files ρωτώ as a form of ρωτάω. The
    # EN reference would point ρωτώ back at ρωτάω, a cycle that deletes
    # both: ρωτώ stays the lemma.
    lookup = {"ρωτώ": ("ρωτώ", 3), "ρωτάω": ("ρωτώ", 1),
              "αυτούς": ("αυτούς", 1), "αυτός": ("αυτός", 3)}
    moved = resolve_cross_source(lookup, {"ρωτώ": "ρωτάω",
                                          "αυτούς": "αυτός"}, {}, set())
    assert moved == 1
    assert lookup["ρωτώ"] == ("ρωτώ", 3) and lookup["ρωτάω"][0] == "ρωτώ"
    assert lookup["αυτούς"][0] == "αυτός"


def test_a_pronoun_grid_of_several_persons_is_not_read(tmp_path):
    # EN's τα page carries the whole personal-pronoun grid under a generic
    # template; its first- and second-person forms are no forms of τα.
    grid = [{"form": "inflection-table-top", "tags": ["inflection-template"]},
            {"form": "εγώ", "tags": ["first-person", "nominative",
                                     "singular", "strong"]},
            {"form": "εσύ", "tags": ["second-person", "nominative",
                                     "singular", "strong"]},
            {"form": "αυτού", "tags": ["third-person", "genitive",
                                       "singular", "strong"]}]
    dump = _write(tmp_path, [
        {"word": "τα", "pos": "pron", "forms": grid,
         "senses": [{"glosses": ["they"]}]},
        {"word": "αυτός", "pos": "pron",
         "forms": [{"form": "αυτού", "tags": ["genitive", "singular"]}],
         "senses": [{"glosses": ["he"]}]},
    ])
    pairs, lookup, _, pos_map, _ = extract_pairs(dump, "el-en")
    assert not [p for p in pairs if p["lemma"] == "τα" and p["form"] != "τα"]
    assert pos_map["αυτού"]["PRON"] == "αυτός"
