"""Source-owned participles remain separate from finite verb paradigms."""
import json

import pytest

import export_mg_polytonic as M
import mg_polytonic_paradigms as P


def cell(form, lemma="περιπλανώμενος", *, kind="inflection", tags=()):
    return {"form": form, "lemma": lemma, "pos": "verb", "kind": kind,
            "tags": sorted({"participle", *tags}), "sources": ["el"],
            "source_form": form, "source_lemma": lemma}


def dataset(tmp_path, cells):
    pairs = tmp_path / "mg_pairs.json"
    pairs.write_text(json.dumps([{
        "form": "γράφει", "lemma": "γράφω", "pos": "verb",
        "tags": ["present", "third-person", "singular"]}], ensure_ascii=False))
    companion = tmp_path / "el_participles.json"
    companion.write_text(json.dumps({"schema_version": 1, "language": "el",
                                      "cells": cells}, ensure_ascii=False))
    return pairs, companion


def test_standalone_cells_generate_without_a_finite_parent(tmp_path):
    pairs, _ = dataset(tmp_path, [
        cell("περιπλανώμενος", kind="headword", tags={"present"}),
        cell("περιπλανώμενη", tags={"nominative", "feminine", "singular"}),
        cell("περιπλανώμενη", tags={"accusative", "feminine", "singular"}),
        cell("περιπλανώμενης", tags={"genitive", "feminine", "singular"}),
        cell("περιπλανωμένη", tags={"dative", "feminine", "singular"}),
    ])
    paradigms = P.load_paradigms(pairs)
    assert set(paradigms) == {"γράφω", "περιπλανώμενος"}
    assert P.verb_class("περιπλανώμενος", paradigms["περιπλανώμενος"]) == "participle"
    assert all(not P.is_verb(tags) for tags in paradigms["περιπλανώμενος"].values())
    assert "περιπλανωμένη" not in paradigms["περιπλανώμενος"]
    rows = P.form_tag_rows(paradigms, "περιπλανώμενος", "περιπλανώμενη")
    assert {P.cell(tags) for tags in rows} == {
        "participle nominative singular", "participle accusative singular"}
    frequencies = {f: 200 for f in paradigms["περιπλανώμενος"]}
    generated = P.generate(paradigms, frequencies, P.Evidence({}, (), paradigms))
    assert {g.spelling for g in generated} == {
        "περιπλανώμενος", "περιπλανώμενη", "περιπλανώμενης"}
    assert all(g.kind != "subjunctive" and g.vclass == "participle" for g in generated)
    assert set(P.load_verb_paradigms(pairs)) == {"γράφω"}


def test_headword_only_preserves_only_that_exact_form(tmp_path):
    pairs, _ = dataset(tmp_path, [cell("ξεθεωμένος", "ξεθεωμένος", kind="headword")])
    paradigms = P.load_paradigms(pairs)
    generated = P.generate(paradigms, {"ξεθεωμένος": 200, "ξεθεωμένη": 300},
                           P.Evidence({}, (), paradigms))
    assert [(g.spelling, g.vclass) for g in generated] == [("ξεθεωμένος", "participle")]


def test_default_inputs_require_companion_and_explicit_legacy_omits_it(tmp_path, monkeypatch):
    pairs, companion = dataset(tmp_path, [])
    companion.unlink()
    monkeypatch.setattr(P, "VERB_PARADIGMS", pairs)
    with pytest.raises(FileNotFoundError, match="standalone participle coverage"):
        P.load_paradigms(pairs)
    assert set(P.load_paradigms(pairs, legacy_without_participles=True)) == {"γράφω"}
    freq = tmp_path / "mg_freq.txt"
    freq.write_text("γράφει 200\n")
    monkeypatch.setattr(P, "MG_FORM_FREQ", freq)
    monkeypatch.setattr(M, "_VERB_INPUTS", {})
    with pytest.raises(FileNotFoundError, match="standalone participle coverage"):
        M._verb_inputs()


def test_explicit_legacy_does_not_read_available_companion(tmp_path):
    pairs, _ = dataset(tmp_path, [cell("ξεθεωμένος", "ξεθεωμένος", kind="headword")])
    assert "ξεθεωμένος" not in P.load_paradigms(pairs, legacy_without_participles=True)


@pytest.mark.parametrize("tags", [
    {"nominative", "singular", "first-person"},
    {"nominative", "singular", "indicative"},
    {"nominative", "singular", "subjunctive"},
    {"nominative", "singular", "imperative"},
    {"nominative", "singular", "infinitive"},
    {"nominative", "singular", "dependent"},
    {"nominative", "singular", "pos:verb"},
    {"feminine"},
])
def test_inventory_rejects_finite_or_unplaced_inflections(tmp_path, tags):
    _, companion = dataset(tmp_path, [cell("περιπλανώμενη", tags=tags)])
    with pytest.raises(ValueError, match="Invalid standalone participle cell"):
        P.load_participle_cells(companion)


@pytest.mark.parametrize("change", [
    {"schema_version": 2}, {"language": "grc"}, {"cells": {}},
])
def test_inventory_rejects_wrong_envelope(tmp_path, change):
    _, companion = dataset(tmp_path, [])
    data = json.loads(companion.read_text()) | change
    companion.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid standalone participle inventory"):
        P.load_participle_cells(companion)
