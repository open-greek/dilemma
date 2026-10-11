import json

import pytest

from build.build_grc_lexical_heads import (
    EDITED_INPUTS, KAIKKI_INPUTS, build_heads, has_definition, normalize_headword,
)


@pytest.mark.parametrize("gloss", [
    "third-person singular present indicative of γράφω",
    "genitive plural of λόγος", "feminine singular of ἀγαθός",
    "alternative form of λύω", "obsolete spelling of θεός",
    "γενική ενικού του λόγος", "πολυτονική γραφή του λύνω",
])
def test_grammatical_reference_is_not_a_definition(gloss):
    assert not has_definition({"pos": "verb", "senses": [{"glosses": [gloss]}]})


@pytest.mark.parametrize("gloss", [
    "to be in charge of the household", "a kind of fish", "the form of a thing",
    "part of a building", "αιτία ενός γεγονότος",
])
def test_lexical_definition_can_include_ordinary_of(gloss):
    assert has_definition({"pos": "noun", "senses": [{"glosses": [gloss]}]})


def test_form_reference_does_not_hide_an_independent_sense():
    senses = [{"glosses": ["plural of α"], "form_of": [{"word": "α"}]},
              {"glosses": ["a dog"]}]
    assert has_definition({"pos": "noun", "senses": senses})
    assert not has_definition({"pos": "noun", "tags": ["form-of"], "senses": senses})
    # Greek Wiktionary puts etymology references at entry level.
    entry = {"pos": "noun", "form_of": [{"word": "κύων"}],
             "senses": [{"glosses": ["a dog"]}]}
    assert has_definition(entry, "el")
    assert not has_definition(entry, "en")


def test_head_normalization_preserves_lexical_marks_and_case():
    assert normalize_headword("ἀνᾱ-ψῡχή") == "ἀναψυχή"
    assert normalize_headword("Σωκράτης") == "Σωκράτης"
    for word in ["ΛΓτῇ", "ΧΙΙπροθυμία", "ἀπόΑΕ", "συγγράμμμα", "σύν,ἐν-κλείω"]:
        assert normalize_headword(word) is None
    assert normalize_headword("γία") == "γία"


def test_builder_merges_independent_sources_and_pins_portable_inputs(tmp_path):
    data, kaikki = tmp_path / "data", tmp_path / "kaikki"
    data.mkdir(); kaikki.mkdir()
    for name in EDITED_INPUTS:
        (data / name).write_text(json.dumps(["λόγος"] if name == "dge_headwords.json" else []))
    for source, relative, language in KAIKKI_INPUTS:
        path = kaikki / relative; path.parent.mkdir(parents=True, exist_ok=True)
        word = "ἄνθρωπος" if source == "wiktionary_en_grc" else "σπίτι"
        entries = [{"word": word, "lang_code": language, "pos": "noun",
                    "senses": [{"glosses": ["a person or house"]}]},
                   {"word": "βουλήντοιας", "lang_code": language, "pos": "noun",
                    "senses": [{"glosses": ["genitive singular of βουλήντοιάνδε"]}]}]
        path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    lsj = tmp_path / "lsj.json"
    lsj.write_text(json.dumps([{"headword": "γία"}]))
    chunks = data / "sophocles" / "chunks"; chunks.mkdir(parents=True)
    (chunks / "01.xml").write_text("<root><entry><orth>τιθήνημα</orth>child</entry></root>")
    first = build_heads(kaikki, data, lsj)
    second = build_heads(kaikki, data, lsj)
    assert first == second
    assert set(first["heads"]) == {"λόγος", "ἄνθρωπος", "σπίτι", "γία", "τιθήνημα"}
    assert first["heads"]["σπίτι"] == ["wiktionary_el_grc", "wiktionary_el_medieval"]
    assert all(len(row["sha256"]) == 64 for row in first["inputs"].values())
    assert str(tmp_path) not in json.dumps(first)


def test_builder_fails_closed_without_an_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_heads(tmp_path, tmp_path, tmp_path / "lsj.json")
