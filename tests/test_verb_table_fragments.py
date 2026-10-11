"""Source-table fragments must not become lookup or training words."""

import json

import pytest

from build_data import extract_pairs
from dilemma.verb_table_fragments import (
    retain_complete_passive_aorist_readings, stemless_passive_aorist_dialects,
    retain_complete_aorist_nonpast_readings, stemless_aorist_nonpast_voices,
)


@pytest.mark.parametrize("tags", [
    ["passive", "aorist", "infinitive"],
    ["passive", "aorist", "infinitive", "first-person", "plural"],
])
def test_only_identified_passive_infinitive_is_withheld(tags):
    assert retain_complete_passive_aorist_readings("ῆναι", tags) is None


@pytest.mark.parametrize("tags", [
    ["passive", "infinitive"],
    ["passive", "aorist", "present", "infinitive"],
    ["passive", "aorist", "infinitive", "participle"],
    ["passive", "aorist", "optative", "plural"],
    ["passive", "aorist", "optative", "first-person", "second-person", "plural"],
    ["passive", "aorist", "optative", "first-person", "singular", "plural"],
    ["passive", "aorist", "participle", "masculine", "singular"],
    ["passive", "aorist", "participle", "masculine", "nominative"],
])
def test_unidentified_or_conflicting_axes_are_preserved(tags):
    assert retain_complete_passive_aorist_readings("εῖμεν", tags) == tags


@pytest.mark.parametrize("voice", ["active", "middle", "mediopassive"])
def test_independent_voice_is_retained_without_mutating_source_tags(voice):
    tags = ["passive", voice, "aorist", "infinitive", "Ionic"]
    result = retain_complete_passive_aorist_readings("ῆναι", tags)
    assert result == [voice, "aorist", "infinitive", "Ionic"]
    assert tags[0] == "passive"


def test_only_explicit_participle_cell_is_withheld():
    tags = ["passive", "aorist", "participle", "masculine", "nominative", "singular"]
    assert retain_complete_passive_aorist_readings("είς", tags) is None


def test_missing_participle_axes_need_actual_stemless_template_evidence():
    tags = ["passive", "aorist", "participle", "masculine"]
    assert retain_complete_passive_aorist_readings("είς", tags) == tags
    assert retain_complete_passive_aorist_readings("είς", tags, stemless_table=True) is None
    assert retain_complete_passive_aorist_readings("τυπείς", tags, stemless_table=True) == tags


def _template(**changes):
    return {"name": "grc-conj", "args": {
        "1": "aor-1", "form": "ap", "2": "ἐτυψ", "3": "τυψ", "dial": "ion", **changes}}


@pytest.mark.parametrize("changes", [{"4": "ἐτυπ"}, {"5": "τυπ"}, {"form": "a"},
                                     {"2": ""}, {"3": ""}, {"1": "pres"},
                                     {"dial": "unidentified"}])
def test_template_witness_requires_explicit_missing_passive_stems(changes):
    assert not stemless_passive_aorist_dialects({"inflection_templates": [_template(**changes)]})


def test_competing_complete_template_blocks_ambiguous_table_attribution():
    entry = {"inflection_templates": [_template(), _template(**{"4": "ἐτυπ", "5": "τυπ"})]}
    assert not stemless_passive_aorist_dialects(entry)
    entry["inflection_templates"][1]["args"]["dial"] = "att"
    assert stemless_passive_aorist_dialects(entry) == frozenset({"Ionic"})


@pytest.mark.parametrize("witness", [False, True])
def test_extraction_participle_head_needs_its_own_table_witness(tmp_path, witness):
    entry = _entry()
    if witness:
        entry["inflection_templates"] = [_template()]
    entry["forms"].insert(2, {"form": "είς", "tags": ["passive", "participle", "masculine"]})
    pairs, lookup, _, typed, *_ = _extract(tmp_path, [entry])
    assert any(p["form"] == "είς" for p in pairs) == (not witness)
    assert ("είς" in lookup) == (not witness)
    assert ("είς" in typed) == (not witness)


@pytest.mark.parametrize("form,tags", [
    ("τυπῆναι", ["passive", "aorist", "infinitive"]),
    ("ἐτύπην", ["passive", "aorist", "indicative", "first-person", "singular"]),
    ("εἷναι", ["active", "aorist", "infinitive"]),
    ("εἷτε", ["active", "aorist", "optative", "second-person", "plural"]),
    ("ἦν", ["active", "imperfect", "indicative", "first-person", "singular"]),
    ("είς", ["active", "aorist", "participle", "masculine"]),
])
def test_complete_words_and_other_grammatical_homographs_survive(form, tags):
    assert retain_complete_passive_aorist_readings(form, tags) == tags


def _extract(tmp_path, entries, lang="grc-en"):
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(entry, ensure_ascii=False) + "\n"
                              for entry in entries))
    return extract_pairs(source, lang)


def _entry(pos="verb", mixed=False):
    return {"word": "τύπτω", "pos": pos, "senses": [{"glosses": ["strike"]}],
            "forms": [
                {"form": "Ionic aorist", "tags": ["table-tags"]},
                {"form": "ῆναι", "tags": ["passive", "infinitive"]
                 + (["active"] if mixed else [])},
                {"form": "τυψάμην", "tags": ["middle", "indicative", "first-person", "singular"]},
                {"form": "Attic aorist", "tags": ["table-tags"]},
                {"form": "τυπῆναι", "tags": ["passive", "infinitive"]},
                {"form": "Epic aorist", "tags": ["table-tags"]},
                {"form": "τυπῆναι", "tags": ["passive", "infinitive"]},
            ]}


@pytest.mark.parametrize("lang", ["grc-en", "grc-el"])
def test_extraction_filters_after_inherited_tense_before_all_products(tmp_path, lang):
    pairs, lookup, _, typed, *_ = _extract(tmp_path, [_entry()], lang)
    assert not any(p["form"] in {"ῆναι", "ήναι"} for p in pairs)
    assert not any(form in lookup or form in typed for form in ("ῆναι", "ήναι", "ηναι"))
    assert any(p["form"] == "τυψάμην" and "Ionic" in p["tags"] for p in pairs)
    assert {p["tags"][-2] for p in pairs if p["form"] == "τυπῆναι"} == {"Attic", "Epic"}
    assert lookup["τυπῆναι"][0] == "τύπτω"


def test_extraction_retains_independent_active_reading(tmp_path):
    pairs, lookup, _, typed, *_ = _extract(tmp_path, [_entry(mixed=True)])
    bad_shape = [p for p in pairs if p["form"] in {"ῆναι", "ήναι"}]
    assert len(bad_shape) == 2
    assert all("active" in p["tags"] and "passive" not in p["tags"] for p in bad_shape)
    assert lookup["ῆναι"][0] == "τύπτω" and typed["ῆναι"]["VERB"] == "τύπτω"


@pytest.mark.parametrize("lang,pos", [("el-en", "verb"), ("el-el", "verb"),
                                      ("med-en", "verb"), ("grc-en", "noun"),
                                      ("grc-el", "adj")])
def test_other_languages_and_nonverb_readings_are_unchanged(tmp_path, lang, pos):
    pairs, *_ = _extract(tmp_path, [_entry(pos)], lang)
    assert any(p["form"] == "ῆναι" for p in pairs)


def _missing_nonpast_template(**changes):
    return {"name": "grc-conj", "args": {"1": "aor-1", "2": "ᾠδην", **changes}}


@pytest.mark.parametrize("changes", [
    {"3": "οἰδην"}, {"2": ""}, {"indonly": "1"}, {"indonly": ""},
    {"form": "pass"}, {"form": "unidentified"}, {"dial": "unidentified"},
    {"1": "aor-2"},
])
def test_missing_nonpast_template_requires_positive_scope(changes):
    assert not stemless_aorist_nonpast_voices(
        {"inflection_templates": [_missing_nonpast_template(**changes)]})


def test_complete_competing_nonpast_template_protects_each_voice_and_dialect():
    entry = {"inflection_templates": [
        _missing_nonpast_template(),
        _missing_nonpast_template(**{"3": "οἰδην", "form": "mid"})]}
    assert stemless_aorist_nonpast_voices(entry) == {"": frozenset({"active"})}
    entry["inflection_templates"][1]["args"]["dial"] = "epi"
    assert stemless_aorist_nonpast_voices(entry) == {"": frozenset({"active", "middle"})}
    entry["inflection_templates"][1]["args"]["dial"] = ""
    entry["inflection_templates"][1]["args"]["form"] = ""
    assert not stemless_aorist_nonpast_voices(entry)
    entry["inflection_templates"][1]["args"]["1"] = "aor-2"
    assert not stemless_aorist_nonpast_voices(entry)
    entry["inflection_templates"][1]["args"]["1"] = "aor-irreg"
    assert not stemless_aorist_nonpast_voices(entry)


@pytest.mark.parametrize("form,tags", [
    ("ῶ", ["active", "subjunctive", "first-person", "singular"]),
    ("εῖεν", ["active", "optative", "third-person", "singular"]),
    ("ᾰ́των", ["active", "imperative", "third-person", "dual"]),
    ("ῶμαι", ["middle", "subjunctive", "first-person", "singular"]),
    ("αίμεσθᾰ", ["middle", "optative", "first-person", "plural"]),
    ("αῖ", ["middle", "imperative", "second-person", "singular"]),
    ("αῖ", ["active", "infinitive"]),
    ("ᾰ́σθαι", ["middle", "infinitive", "second-person", "third-person", "dual"]),
    ("ᾰ́μενος", ["middle", "participle", "masculine", "dual", "third-person"]),
])
def test_only_witnessed_nonpast_ending_readings_are_withheld(form, tags):
    tags = tags + ["aorist"]
    assert retain_complete_aorist_nonpast_readings(form, tags) == tags
    assert retain_complete_aorist_nonpast_readings(
        form, tags, stemless_voices=frozenset({"active", "middle"})) is None


@pytest.mark.parametrize("form,tags", [
    ("ᾤδηνα", ["active", "aorist", "indicative", "first-person", "singular"]),
    ("οἰδήνω", ["active", "aorist", "subjunctive", "first-person", "singular"]),
    ("οἰδῆναι", ["passive", "aorist", "infinitive"]),
    ("αῖ", ["active", "future", "infinitive"]),
    ("ῶ", ["active", "aorist", "present", "subjunctive", "first-person", "singular"]),
    ("ῶ", ["active", "aorist", "subjunctive", "optative", "first-person", "singular"]),
    ("ῶ", ["active", "aorist", "subjunctive", "singular"]),
    ("ῶ", ["active", "aorist", "subjunctive", "first-person", "second-person", "singular"]),
    ("ῶ", ["active", "aorist", "subjunctive", "first-person", "singular", "plural"]),
    ("ᾶς", ["active", "aorist", "participle", "masculine", "neuter"]),
])
def test_nonpast_guard_retains_complete_and_uncertain_readings(form, tags):
    assert retain_complete_aorist_nonpast_readings(
        form, tags, stemless_voices=frozenset({"active", "middle"})) == tags


def test_nonpast_guard_preserves_valid_competing_voice_without_mutation():
    tags = ["active", "passive", "aorist", "subjunctive", "first-person", "singular"]
    assert retain_complete_aorist_nonpast_readings(
        "ῶ", tags, stemless_voices=frozenset({"active"})) == tags[1:]
    assert tags[0] == "active"


@pytest.mark.parametrize("lang,pos,witness", [
    ("grc-en", "verb", True), ("grc-el", "verb", True),
    ("el-en", "verb", True), ("grc-en", "noun", True),
    ("grc-en", "verb", False),
])
def test_nonpast_fragments_cannot_license_lookup_pos_or_training(tmp_path, lang, pos, witness):
    entry = {"word": "οἰδάνω", "pos": pos,
             "senses": [{"glosses": ["to swell"]}],
             "forms": [
                 {"form": "aorist", "tags": ["table-tags"]},
                 {"form": "ῶ", "tags": ["active", "subjunctive", "first-person", "singular"]},
                 {"form": "ᾤδηνα", "tags": ["active", "indicative", "first-person", "singular"]},
                 {"form": "οἰδήνω", "tags": ["active", "subjunctive", "first-person", "singular"]}]}
    if witness:
        entry["inflection_templates"] = [_missing_nonpast_template()]
    pairs, lookup, _, typed, *_ = _extract(tmp_path, [entry], lang)
    reject = lang.startswith("grc-") and pos == "verb" and witness
    assert any(p["form"] == "ῶ" for p in pairs) == (not reject)
    assert ("ῶ" in lookup) == (not reject)
    assert ("ῶ" in typed) == (not reject)
    assert any(p["form"] == "ᾤδηνα" for p in pairs)
    assert any(p["form"] == "οἰδήνω" for p in pairs)


def test_complete_competing_template_preserves_existing_nonpast_source_rows(tmp_path):
    entry = {"word": "οἰδάνω", "pos": "verb", "senses": [{"glosses": ["to swell"]}],
             "inflection_templates": [_missing_nonpast_template(),
                                      _missing_nonpast_template(**{"3": "οἰδην"})],
             "forms": [{"form": "aorist", "tags": ["table-tags"]},
                       {"form": "ῶ", "tags": ["active", "subjunctive", "first-person", "singular"]}]}
    assert any(p["form"] == "ῶ" for p in _extract(tmp_path, [entry])[0])
