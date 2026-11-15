"""Another word's page carrying the personal-pronoun grid lends it no forms.

The Greek Wiktionary's Ancient Greek page for the determiner ὅς (2026-11
dump) shows the grc-προσωπική αντωνυμία grid, whose rows are forms of ἐγώ,
σύ and οὗ, with the persons named in header cells (α΄ πρόσωπο) rather than
tags. Before the fix, ἡμῖν, ἡμῶν, ὑμῖν, ὑμᾶς and σύ became DET forms of ὅς.
"""
from __future__ import annotations

import json

import pytest

from build_data import extract_pairs, personal_pronoun_table, _person_header_cells


def _write(tmp_path, entries):
    dump = tmp_path / "kaikki.jsonl"
    dump.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n"
                            for e in entries), encoding="utf-8")
    return dump


GRID = [
    {"form": "grc-προσωπική αντωνυμία", "tags": ["inflection-template"],
     "source": "conjugation"},
    {"form": "α΄ πρόσωπο", "tags": ["plural"], "source": "conjugation"},
    {"form": "β΄ πρόσωπο", "tags": ["plural"], "source": "conjugation"},
    {"form": "σύ", "tags": ["singular", "nominative"], "source": "conjugation"},
    {"form": "ἡμῶν", "tags": ["genitive", "plural"], "source": "conjugation"},
    {"form": "ἡμῖν", "tags": ["dative", "plural"], "source": "conjugation"},
    {"form": "ὑμῖν", "tags": ["dative", "plural"], "source": "conjugation"},
    {"form": "ὑμᾶς", "tags": ["accusative", "plural"], "source": "conjugation"},
    {"form": "ἡμέτερος", "tags": ["adjective", "plural"], "source": "conjugation"},
]
HEADER = [{"form": "ὅς", "source": "header"}, {"form": "ἥ", "source": "header"},
          {"form": "ὅν", "source": "header"}]


def determiner(forms):
    return {"word": "ὅς", "pos": "det", "lang_code": "grc", "forms": forms,
            "senses": [{"glosses": ["γʹ προσδιοριστής αντί ἑός· δικός του"]}]}


def test_header_cells_count_as_persons():
    assert _person_header_cells(GRID) == {"first-person", "second-person"}
    assert personal_pronoun_table(GRID)
    untemplated = [row for row in GRID if "inflection-template" not in row["tags"]]
    assert personal_pronoun_table(untemplated)
    one_person = [{"form": "α΄ πρόσωπο", "tags": ["plural"]},
                  {"form": "ἡμῖν", "tags": ["dative", "plural"]}]
    assert not personal_pronoun_table(one_person)


@pytest.mark.parametrize("template", ["grc-decl", "grc-αναφορική αντωνυμία 'ὅς'"])
def test_other_tables_are_not_the_grid(template):
    """grc-decl is the English edition's ordinary declension template."""
    forms = [{"form": template, "tags": ["inflection-template"]},
             {"form": "ὅς", "tags": ["singular", "nominative"]},
             {"form": "οὗ", "tags": ["genitive", "singular"]}]
    assert not personal_pronoun_table(forms)


@pytest.mark.parametrize("templated", [True, False])
def test_determiner_page_keeps_only_its_headword_line(tmp_path, templated):
    grid = GRID if templated else GRID[1:]
    pairs, lookup, *_ = extract_pairs(_write(tmp_path, [determiner(HEADER + grid)]), "grc-el")
    forms = {p["form"] for p in pairs if p["lemma"] == "ὅς"}
    assert not forms & {"σύ", "ἡμῶν", "ἡμῖν", "ὑμῖν", "ὑμᾶς", "ἡμέτερος"}
    for form in ("ἡμῖν", "ὑμᾶς", "σύ"):
        assert lookup.get(form) != "ὅς"


def test_determiner_own_table_is_kept(tmp_path):
    own = [{"form": "ὅς", "tags": ["singular", "nominative", "masculine"]},
           {"form": "ἥν", "tags": ["singular", "accusative", "feminine"]}]
    pairs, *_ = extract_pairs(_write(tmp_path, [determiner(HEADER + own)]), "grc-el")
    assert "ἥν" in {p["form"] for p in pairs if p["lemma"] == "ὅς"}


def test_the_pronouns_own_grid_is_still_skipped(tmp_path):
    entry = {"word": "ἐγώ", "pos": "pron", "lang_code": "grc",
             "senses": [{"glosses": ["εγώ"]}],
             "forms": [{"form": "ἐγώ", "source": "header"}] + GRID[1:]}
    pairs, *_ = extract_pairs(_write(tmp_path, [entry]), "grc-el")
    assert not {p["form"] for p in pairs if p["lemma"] == "ἐγώ"} & {"σύ", "ὑμᾶς"}
