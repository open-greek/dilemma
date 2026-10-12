import json

import pytest

from build.build_grc_lexical_heads import (
    EDITED_INPUTS, KAIKKI_INPUTS, VERNACULAR_INPUTS, build_heads, has_definition,
    normalize_headword,
    principal_quantity_rows,
)


@pytest.mark.parametrize("gloss", [
    "third-person singular present indicative of γράφω",
    "genitive plural of λόγος", "feminine singular of ἀγαθός",
    "alternative form of λύω", "obsolete spelling of θεός",
    "γενική ενικού του λόγος", "πολυτονική γραφή του λύνω",
    "ιωνικός & επικός τύπος του παρθενία", "συνηρημένη μορφή του τιμάω",
    "Delphic form of θωάζω", "Doric and Pamphylian form of σημαίνω",
    "Tragic form of ἀμφιέπω",
    "μορφή του στενάχω", "μεσοπαθητική φωνή του ρήματος αἱρέω",
])
def test_grammatical_reference_is_not_a_definition(gloss):
    assert not has_definition({"pos": "verb", "senses": [{"glosses": [gloss]}]})


@pytest.mark.parametrize("gloss", [
    "to be in charge of the household", "a kind of fish", "the form of a thing",
    "part of a building", "αιτία ενός γεγονότος",
])
def test_lexical_definition_can_include_ordinary_of(gloss):
    assert has_definition({"pos": "noun", "senses": [{"glosses": [gloss]}]})


@pytest.mark.parametrize("pos,gloss", [
    ("adj", "άλλη μορφή του φαίδιμος : λαμπερός, λαμπρός, ακτινοβόλος"),
    ("noun", "συνηρημένη μορφή του ἀμυγδαλέα: η αμυγδαλιά"),
    ("verb", "άλλη μορφή του χειμάζω (διαχειμάζω, περνάω το χειμώνα)"),
    ("adj", "μεταγενέστερος τύπος του καθάρειος: καθαρός"),
])
def test_reference_prefix_preserves_inline_lexical_meaning(pos, gloss):
    assert has_definition({"pos": pos, "senses": [{"glosses": [gloss]}]}, "el")


@pytest.mark.parametrize("gloss", [
    "άλλη μορφή του φαίδιμος", "παλαιότερη μορφή του φαίδιμος",
    "πολυτονική μορφή του φαίδιμος",
])
def test_complete_form_reference_is_not_a_definition(gloss):
    assert not has_definition({"pos": "adj", "senses": [{"glosses": [gloss]}]}, "el")


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
    for source, relative, language in KAIKKI_INPUTS + VERNACULAR_INPUTS:
        path = kaikki / relative; path.parent.mkdir(parents=True, exist_ok=True)
        word = ("παλάτι" if language == "el" else
                "ἄνθρωπος" if source == "wiktionary_en_grc" else "σπίτι")
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
    assert first["schema_version"] == 2
    assert first["vernacular_heads"] == {
        "παλάτι": ["wiktionary_el_el", "wiktionary_en_el"]}
    assert "παλάτι" not in first["heads"]
    assert all(len(row["sha256"]) == 64 for row in first["inputs"].values())
    assert str(tmp_path) not in json.dumps(first)


def test_builder_fails_closed_without_an_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_heads(tmp_path, tmp_path, tmp_path / "lsj.json")


def test_principal_quantities_keep_source_marks_and_exclude_other_cells():
    entry = {"word": "θύνω", "pos": "verb",
             "senses": [{"glosses": ["to rush"]}],
             "forms": [
                 {"form": "θῡ́νω", "tags": ["canonical"]},
                 {"form": "θῠ́νω", "tags": ["indicative", "first-person", "singular"]},
                 {"form": "θῡ́νει", "tags": ["indicative", "third-person", "singular"]},
                 {"form": "θῡ́νω", "tags": ["indicative", "first-person", "third-person", "singular"]},
                 {"form": "θύνω", "tags": ["canonical"]},
             ]}
    rows = principal_quantity_rows(entry, "wiktionary_en_grc")
    assert [row["form"] for row in rows] == ["θῡ́νω", "θῠ́νω"]
    assert all(row["source"] == "wiktionary_en_grc" for row in rows)
    entry["senses"] = [{"glosses": ["alternative form of θύω"]}]
    assert principal_quantity_rows(entry, "wiktionary_en_grc") == []
    entry["pos"] = "noun"
    assert principal_quantity_rows(entry, "wiktionary_en_grc") == []


@pytest.mark.parametrize("header", ["grc-noun", "grc-adj", "head"])
def test_quantity_provenance_rejects_nominal_header_mislabeled_as_verb(header):
    entry = {"word": "κινδύνευμα", "pos": "verb",
             "head_templates": [{"name": header, "args": {"1": "grc", "2": "noun"}}],
             "senses": [{"glosses": ["a risk, venture"]}],
             "forms": [{"form": "κινδῡ́νευμᾰ", "tags": ["canonical"]}]}
    assert principal_quantity_rows(entry, "wiktionary_en_grc") == []


@pytest.mark.parametrize("extra_tag", ["optative", "subjunctive", "imperative",
                                        "infinitive", "participle", "neuter"])
def test_quantity_provenance_rejects_conflicting_finite_cell_tags(extra_tag):
    entry = {"word": "συμφύω", "pos": "verb",
             "senses": [{"glosses": ["to grow together"]}],
             "forms": [{"form": "σῠμφῠείην", "tags": [
                 "indicative", "first-person", "singular", extra_tag]}]}
    assert principal_quantity_rows(entry, "wiktionary_en_grc") == []


def test_quantity_provenance_rejects_nominal_citation_tags():
    entry = {"word": "ἐνηρόσιον", "pos": "verb",
             "senses": [{"glosses": ["rent for corn-land"]}],
             "forms": [{"form": "ἐνηρόσῐον", "tags": ["canonical", "neuter"]}]}
    assert principal_quantity_rows(entry, "wiktionary_en_grc") == []


def test_canonical_tag_does_not_override_contradictory_mood_tags():
    entry = {"word": "θύνω", "pos": "verb", "senses": [{"glosses": ["to rush"]}],
             "forms": [{"form": "θῡ́νω", "tags": ["canonical", "indicative", "optative"]}]}
    assert principal_quantity_rows(entry, "wiktionary_en_grc") == []


@pytest.mark.parametrize("header", [
    {"name": "grc-verb form"},
    {"name": "head", "args": {"1": "grc", "2": "verb form"}},
])
@pytest.mark.parametrize("gloss", [
    "late third-person singular aorist middle optative of δύνω",
    "uncontracted first-person singular present mediopassive indicative of κρατέω",
    "/(Attic) first-person singular aorist active subjunctive of ἀγγέλλω",
    "future active infinitive uncontracted of βάλλω",
    "second/third-person dual aorist active subjunctive of ἀγγέλλω",
    "prevocalic/clause-final third-person singular aorist active indicative of ἀνατίθημι",
])
def test_declared_verb_form_page_cannot_supply_independent_quantities(header, gloss):
    entry = {"word": "μέμαα", "pos": "verb",
             "head_templates": [header],
             "senses": [{"glosses": [gloss]}],
             "forms": [{"form": "μέμᾰᾰ", "tags": ["canonical"]}]}
    assert principal_quantity_rows(entry, "wiktionary_en_grc") == []
    entry["senses"].append({"glosses": ["to wish eagerly, strive, yearn, desire"]})
    assert principal_quantity_rows(entry, "wiktionary_en_grc")


@pytest.mark.parametrize("gloss", ["to pull down", "wake up, rouse", "raise or build instead"])
def test_declared_form_page_retains_a_separate_lexical_definition(gloss):
    entry = {"word": "μέμαα", "pos": "verb",
             "head_templates": [{"name": "grc-verb form"}],
             "senses": [{"glosses": [gloss]}],
             "forms": [{"form": "μέμᾰᾰ", "tags": ["canonical"]}]}
    assert principal_quantity_rows(entry, "wiktionary_en_grc")


def test_loader_requires_separate_vernacular_evidence(tmp_path):
    from export_hunspell import load_grc_lexical_heads, lexical_head_key
    path = tmp_path / "heads.json"
    path.write_text(json.dumps({"schema_version": 1, "heads": {"λόγος": ["test"]}}))
    with pytest.raises(ValueError):
        load_grc_lexical_heads(path)
    path.write_text(json.dumps({"schema_version": 2, "heads": {"λόγος": ["test"]},
                               "vernacular_heads": {"παλάτι": ["test"]}}))
    assert load_grc_lexical_heads(path) == frozenset({lexical_head_key("λόγος")})
    assert load_grc_lexical_heads(path, vernacular=True) == frozenset({lexical_head_key("παλάτι")})
