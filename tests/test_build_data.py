"""Focused tests for Kaikki build-time morphology extraction."""

from __future__ import annotations

import json

import pytest

from build_data import _parse_verb_tense, extract_pairs


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
