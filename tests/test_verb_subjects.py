"""Verb-subject evidence must retain sense scope and avoid pronoun guesses."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from dilemma import ParadigmForm, ParadigmSlot, generate, generate_paradigm, verb_subjects
from dilemma.paradigm import fill_canonical_dict, reset_cache
from dilemma.verb_subjects import annotate_verb_entry
from dilemma_build.build_verb_subjects import build_inventory, extract_sense, main

ROOT = Path(__file__).resolve().parents[1]


def sense(*glosses, **extra):
    return {"id": "example-sense", "raw_glosses": list(glosses), **extra}


def entry(word, *senses, lang="grc"):
    return {"word": word, "lang_code": lang, "pos": "verb", "senses": list(senses)}


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    en = tmp_path / "en.jsonl"
    el = tmp_path / "el.jsonl"
    # Minimal source labels, with distinct senses and disagreeing editions.
    entries = [
        entry("ἀπαιτέω",
              sense("(passive voice)", "(of persons, transitive) to have something demanded of one", id="person"),
              sense("(passive voice)", "(of things, intransitive) to be demanded", id="thing")),
        entry("πολεμέω", sense("(transitive, with a person as object) to make war on",
                              "(passive voice, of a person) to have war made on one")),
        entry("φαίνω", sense("(passive voice) to appear", "(Koine, impersonal) to seem")),
        entry("συνάγω", sense("(of things) gather together", "(passive voice) to be carried along with")),
        entry("γράφω", sense("(perfect passive) be written down")),
        entry("ἄγω", sense("(perfect passive, of persons) to have been led")),
        entry("δοκέω", sense("(passive voice, of persons) example")),
    ]
    en.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in entries) + "\n")
    el.write_text(json.dumps(entry("δοκέω", {"id": "other-edition", "tags": ["passive"],
                                          "raw_tags": ["για πράγματα"]}), ensure_ascii=False) + "\n")
    paths = {("grc", "en"): en, ("grc", "el"): el}
    data = build_inventory(paths)
    (tmp_path / "verb_subjects.json").write_text(json.dumps(data, ensure_ascii=False))
    monkeypatch.setenv("DILEMMA_DATA_DIR", str(tmp_path))
    yield data, paths
    reset_cache()


def test_personal_passive_is_source_backed(inventory):
    result = verb_subjects("πολεμέω")
    assert result["subject_type"] == "person"
    assert result["coverage"] == "documented_senses"
    assert result["constructions"] == ["personal"]
    reading, = result["readings"]
    assert reading["evidence"] == ["(passive voice, of a person)"]
    assert reading["source"]["sha256"] == inventory[0]["_meta"]["sources"]["wiktionary-en-grc"]["sha256"]
    assert reading["reference"].startswith("https://en.wiktionary.org/wiki/")


def test_person_and_thing_senses_remain_separate(inventory):
    result = verb_subjects("ἀπαιτέω")
    assert result["subject_type"] == "either"
    assert {r["sense_id"]: r["subject_type"] for r in result["readings"]} == {
        "person": "person", "thing": "thing"}
    assert verb_subjects("ἀπαιτέω", sense_id="thing")["subject_type"] == "thing"
    assert verb_subjects("ἀπαιτέω", sense_id="unrecorded") is None


def test_conflicting_editions_are_not_resolved_by_vote(inventory):
    result = verb_subjects("δοκέω")
    assert result["subject_type"] == "either"
    assert {r["source_id"] for r in result["readings"]} == {
        "wiktionary-el-grc", "wiktionary-en-grc"}
    assert {r["subject_type"] for r in result["readings"]} == {"person", "thing"}


def test_impersonal_is_not_a_thing(inventory):
    result = verb_subjects("φαίνω")
    assert result["subject_type"] is None
    assert result["constructions"] == ["impersonal"]
    assert result["readings"][0]["construction"] == "impersonal"


@pytest.mark.parametrize("word", ["ἀποκτείνω", "unknown"])
def test_unmarked_lemmas_remain_unknown(inventory, word):
    assert verb_subjects(word) is None


@pytest.mark.parametrize("word", ["γράφω", "συνάγω"])
def test_unmarked_and_object_only_passives_remain_unknown(inventory, word):
    result = verb_subjects(word)
    assert result["subject_type"] is None
    assert result["subject_types"] == []
    assert result["has_unknown_readings"] is True
    assert result["constructions"] == []


def test_wrong_voice_and_language_do_not_inherit_subjects(inventory):
    assert verb_subjects("πολεμέω", voice="active") is None
    assert verb_subjects("πολεμέω", voice="middle") is None
    assert verb_subjects("πολεμέω", lang="el") is None


@pytest.mark.parametrize("kwargs", [{"lang": "all"}, {"lang": "ag"}, {"voice": "unknown"}])
def test_invalid_query_is_rejected(inventory, kwargs):
    with pytest.raises(ValueError):
        verb_subjects("πολεμέω", **kwargs)


@pytest.mark.parametrize("label", [
    "(with accusative of person) to persuade", "(with dative of person) to help",
    "(with genitive of thing) to desire", "(personal) to fit",
    "(of things) to gather",
    "to talk about the subject (of a person)",
])
def test_non_subject_and_negative_labels_are_not_evidence(label):
    assert extract_sense(sense(label)) == []


@pytest.mark.parametrize("label", ["(passive voice) to be stripped of things",
                                    "(passive voice, with a person) to be accompanied"])
def test_passive_complements_do_not_establish_subject_type(label):
    result = extract_sense(sense(label))
    assert result[0]["subject_type"] is None


def test_negative_impersonal_qualifier_does_not_assert_impersonal():
    result = extract_sense(sense("(middle voice, never impersonal) to need"))
    assert result[0]["construction"] == "personal"
    assert result[0]["subject_type"] is None


def test_unknown_alternative_prevents_an_exclusive_summary():
    from dilemma.verb_subjects import summarize
    result = summarize([
        {"subject_type": "person", "construction": "personal"},
        {"subject_type": None, "construction": "personal"},
    ])
    assert result["subject_type"] is None
    assert result["subject_types"] == ["person"]
    assert result["has_unknown_readings"] is True


def test_nested_child_subject_replaces_ancestor_subject():
    rows = extract_sense(sense("(passive voice, of persons) to have a thing begged of one",
                              "(of things) to be asked for"))
    assert rows[0]["subject_type"] == "thing"
    assert rows[0]["evidence"][-1] == "(of things)"


def test_unknown_child_restriction_does_not_inherit_parent_type():
    rows = extract_sense(sense("(passive voice, of persons) example",
                              "(of animals) example"))
    assert rows[0]["subject_type"] is None


def test_shared_active_passive_label_does_not_type_active_subjects():
    rows = extract_sense(sense("(active voice or passive voice, of persons) example"))
    assert {r["voice"]: r["subject_type"] for r in rows} == {"active": None, "passive": "person"}


def test_explicit_subject_label_is_independent_of_voice():
    rows = extract_sense(sense("(active voice, with a person as the subject) example"))
    assert rows[0]["subject_type"] == "person"


def test_explicit_subject_can_be_either():
    rows = extract_sense(sense("(passive voice, of people and things) to be moved"))
    assert rows[0]["subject_type"] == "either"


@pytest.mark.parametrize("extra", [{"form_of": [{"word": "γράφω"}]},
                                    {"tags": ["form-of"]},
                                    {"alt_of": [{"word": "γράφω"}]}])
def test_inflected_and_alternative_entries_cannot_claim_lemma_scope(extra):
    assert extract_sense(sense("(passive voice, of persons) example", **extra)) == []


def test_unspecified_voice_does_not_imply_passive(tmp_path, monkeypatch):
    src = tmp_path / "el.jsonl"
    src.write_text(json.dumps(entry("χρή", {"tags": ["impersonal"]})))
    data = build_inventory({("grc", "el"): src})
    (tmp_path / "verb_subjects.json").write_text(json.dumps(data))
    monkeypatch.setenv("DILEMMA_DATA_DIR", str(tmp_path))
    assert verb_subjects("χρή") is None
    assert verb_subjects("χρή", voice="unspecified")["constructions"] == ["impersonal"]


def test_greek_object_qualifier_without_voice_is_unknown():
    assert extract_sense({"raw_tags": ["για πρόσωπα"]}) == []


def test_missing_source_ids_are_stable():
    definition = {"tags": ["impersonal"]}
    assert extract_sense(definition) == extract_sense(copy.deepcopy(definition))
    assert extract_sense(definition)[0]["sense_id"].startswith("sha256:")


def test_deterministic_merge_and_source_changes(inventory):
    data, paths = inventory
    assert build_inventory(dict(reversed(list(paths.items())))) == data
    old_hash = data["_meta"]["sources"]["wiktionary-en-grc"]["sha256"]
    with paths[("grc", "en")].open("a") as f:
        f.write(json.dumps(entry("δέω", sense("unmarked"))) + "\n")
    rebuilt = build_inventory(paths)
    assert rebuilt["grc"] == data["grc"]
    assert rebuilt["_meta"]["sources"]["wiktionary-en-grc"]["sha256"] != old_hash


def test_cli_requires_both_editions(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--kaikki", str(tmp_path), "--lang", "grc", "--out", str(tmp_path / "out.json")])
    assert error.value.code == 2
    assert not (tmp_path / "out.json").exists()


def test_missing_table_does_not_fall_back_to_another_directory(inventory, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert verb_subjects("πολεμέω", data_dir=empty) is None


def test_unicode_length_normalization_is_not_accent_folding(inventory):
    assert verb_subjects(unicodedata.normalize("NFD", "ἀπαιτέω"))["subject_type"] == "either"
    assert verb_subjects("ᾰ̓παιτέω")["subject_type"] == "either"
    assert verb_subjects("απαιτεω") is None


def test_returned_evidence_does_not_mutate_cached_sources(inventory):
    first = verb_subjects("πολεμέω")
    first["readings"][0]["source"]["sha256"] = "bad"
    first["readings"].clear()
    assert verb_subjects("πολεμέω")["readings"][0]["source"]["sha256"] != "bad"


def test_rebuilt_table_invalidates_cache(inventory, tmp_path):
    assert verb_subjects("πολεμέω")
    table = tmp_path / "verb_subjects.json"
    data = json.loads(table.read_text())
    data["grc"].pop("πολεμέω")
    table.write_text(json.dumps(data))
    assert verb_subjects("πολεμέω") is None


def test_schema_mismatch_is_not_silently_unknown(inventory, tmp_path):
    table = tmp_path / "verb_subjects.json"
    data = json.loads(table.read_text())
    data["_meta"]["schema_version"] = 999
    table.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema"):
        verb_subjects("πολεμέω")


@pytest.mark.parametrize("field,value", [
    ("subject_type", "guessed-human"), ("source_id", "missing-source"),
    ("construction", "maybe"), ("evidence", []),
    ("conditions", {"unknown-condition": "value"}),
])
def test_invalid_claims_are_rejected(inventory, tmp_path, field, value):
    data = copy.deepcopy(inventory[0])
    data["grc"]["πολεμέω"][0][field] = value
    (tmp_path / "verb_subjects.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="reading"):
        verb_subjects("πολεμέω")


def _slot(person="3", tense="present"):
    return ParadigmSlot.verb_finite(voice="passive", tense=tense, mood="indicative",
                                    person=person, number="sg")


def test_impersonal_and_tense_conditions_respect_generated_cell(inventory):
    assert verb_subjects("φαίνω", slot=_slot())
    assert verb_subjects("φαίνω", slot=_slot(person="1")) is None
    assert verb_subjects("ἄγω", slot=_slot(tense="perfect"))
    assert verb_subjects("ἄγω", slot=_slot()) is None


def test_opt_in_generation_preserves_spelling_and_equality(inventory, tmp_path, monkeypatch):
    (tmp_path / "dilemma_ag_verb_paradigms.json").write_text(json.dumps({
        "πολεμέω": {"forms": {"passive_present_indicative_3sg": "πολεμεῖται"}}}))
    monkeypatch.setenv("DILEMMA_PARADIGM_DATA", str(tmp_path))
    reset_cache()
    plain = generate("πολεμέω", _slot(), allow_template=False)
    enriched = generate("πολεμέω", _slot(), allow_template=False, with_subjects=True)
    assert plain == enriched == ParadigmForm("πολεμεῖται", "dilemma_corpus")
    assert plain.subject_evidence is None
    assert enriched.subject_evidence["subject_type"] == "person"
    paradigm = generate_paradigm("πολεμέω", "verb", allow_template=False, with_subjects=True)
    assert paradigm[_slot().key] == enriched
    assert paradigm[_slot().key].subject_evidence == enriched.subject_evidence


def test_entry_annotation_keeps_existing_fields_and_readings(inventory):
    original = {"forms": {"passive_present_indicative_3sg": "ἀπαιτεῖται"},
                "form_count": 1, "source": "dilemma", "notes": "keep"}
    out = copy.deepcopy(original)
    annotate_verb_entry("ἀπαιτέω", out)
    assert {k: out[k] for k in original} == original
    assert out["subject_evidence"]["passive"]["subject_type"] == "either"
    assert len(out["subject_evidence"]["passive"]["readings"]) == 2
    original_evidence = {"passive": {"source": "caller-owned"}}
    out["subject_evidence"] = copy.deepcopy(original_evidence)
    annotate_verb_entry("ἀπαιτέω", out)
    assert out["subject_evidence"] == original_evidence


@pytest.mark.parametrize("existing", [None, "caller-owned"])
def test_existing_non_object_evidence_is_not_overwritten(inventory, existing):
    out = {"subject_evidence": existing}
    annotate_verb_entry("ἀπαιτέω", out)
    assert out == {"subject_evidence": existing}


def test_raw_paradigm_builder_opt_in_preserves_forms(inventory, tmp_path, monkeypatch):
    from dilemma_build import build_grc_verb_paradigms as builder
    row = {"lemma": "ἀπαιτέω", "form": "ἀπαιτεῖται", "pos": "verb",
           "tags": ["passive", "present", "indicative", "third-person", "singular"]}
    lexical = tmp_path / "heads.json"
    lexical.write_text(json.dumps({"heads": {}}))
    monkeypatch.setattr(builder, "LEXICAL_HEADS_PATH", lexical)
    monkeypatch.setattr(builder, "LSJ_HEADWORDS_PATH", tmp_path / "missing-heads.json")
    monkeypatch.setattr(builder, "load_pairs", lambda path: [row] if path == builder.GLAUX_PAIRS else [])
    monkeypatch.setattr(builder, "load_lsj_head_texts", lambda: {})
    monkeypatch.setattr(builder, "corpus_counts", lambda: {})
    plain = builder.build_paradigms()
    enriched = builder.build_paradigms(with_subjects=True)
    subject_record = enriched["ἀπαιτέω"].pop("subject_evidence")
    assert subject_record["passive"]["subject_type"] == "either"
    assert enriched == plain


def test_canonical_cli_emits_source_evidence(inventory, tmp_path, monkeypatch):
    source = tmp_path / "input.json"
    out = tmp_path / "output.json"
    source.write_text(json.dumps({"v": {"lemma": "ἀπαιτέω", "pos": "verb", "keep": 12}}))
    monkeypatch.delenv("DILEMMA_PARADIGM_DATA", raising=False)
    subprocess.run([sys.executable, "-m", "dilemma", "paradigm", "fill",
                    "--in", str(source), "--out", str(out), "--with-subjects"],
                   cwd=ROOT, check=True, capture_output=True)
    record = json.loads(out.read_text())["v"]
    assert record["keep"] == 12
    assert record["subject_evidence"]["passive"]["subject_type"] == "either"


def test_canonical_export_annotation_does_not_need_new_cells(inventory, monkeypatch):
    monkeypatch.delenv("DILEMMA_PARADIGM_DATA", raising=False)
    reset_cache()
    original = {"v": {"lemma": "ἀπαιτέω", "pos": "verb", "notes": "keep"},
                "n": {"lemma": "ἀπαιτέω", "pos": "noun"}}
    plain, _ = fill_canonical_dict(copy.deepcopy(original), allow_template=False)
    annotated, stats = fill_canonical_dict(copy.deepcopy(original), allow_template=False, with_subjects=True)
    assert plain == original
    assert stats["lemmas_touched"] == 0
    assert annotated["v"]["subject_evidence"]["passive"]["subject_type"] == "either"
    assert annotated["n"] == original["n"]


def test_queries_in_fresh_process_do_not_import_native_backends(inventory, tmp_path):
    script = """
import sys
class Guard:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'numpy', 'torch', 'onnxruntime', 'transformers'}:
            raise AssertionError('native backend imported: ' + fullname)
sys.meta_path.insert(0, Guard())
from dilemma import verb_subjects
assert verb_subjects('ἀπαιτέω')['subject_type'] == 'either'
"""
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True,
                   env={**os.environ, "DILEMMA_DATA_DIR": str(tmp_path)})


def test_published_source_table_has_reviewed_passive_examples():
    path = ROOT / "data" / "verb_subjects.json"
    if not path.exists():
        pytest.skip("verb_subjects.json has not been downloaded")
    data = json.loads(path.read_text())
    assert set(data["_meta"]["sources"]) == {
        "wiktionary-en-grc", "wiktionary-el-grc", "wiktionary-en-el", "wiktionary-el-el"}
    assert verb_subjects("ἀπαιτέω", data_dir=path.parent)["subject_type"] == "either"
    assert "impersonal" in verb_subjects("φαίνω", data_dir=path.parent)["constructions"]
    assert "person" in verb_subjects("αὐλέω", data_dir=path.parent)["subject_types"]
    assert verb_subjects("γράφω", data_dir=path.parent)["subject_type"] is None
