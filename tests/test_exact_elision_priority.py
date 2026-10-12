"""An exact elided name keeps priority without removing other readings."""
import json

import pytest
from dilemma import core



@pytest.mark.parametrize(('lang', 'ancient', 'modern'), [
    ('all', True, True), ('grc', True, False), ('el', False, True),
])
def test_positive_inventory_respects_language(tmp_path, monkeypatch, lang, ancient, modern):
    monkeypatch.setattr(core.Dilemma, '_load_lookups', lambda *a, **k: None)
    monkeypatch.setattr(core.Dilemma, '_check_backend', lambda *a: None)
    path = tmp_path/'heads.json'
    path.write_text(json.dumps({'heads': {'Σικυών': ['edited']},
                               'vernacular_heads': {'γιος': ['definition']}}))
    monkeypatch.setattr(core, 'GRC_LEXICAL_HEADS_PATH', path)
    result = core.Dilemma(lang=lang)
    assert result._positive_elision_target('Σικυών') is ancient
    assert result._positive_elision_target('γιος') is modern
    assert not result._positive_elision_target('γίγνηταμαι')


@pytest.mark.parametrize('target', ['Ταύς', 'Ταῦς', 'τάυς'])
def test_tone_and_case_do_not_certify_another_target(tmp_path, monkeypatch, target):
    monkeypatch.setattr(core.Dilemma, '_load_lookups', lambda *a, **k: None)
    monkeypatch.setattr(core.Dilemma, '_check_backend', lambda *a: None)
    path = tmp_path/'heads.json'
    path.write_text(json.dumps({'heads': {'ταῦς': ['edited']}}))
    monkeypatch.setattr(core, 'GRC_LEXICAL_HEADS_PATH', path)
    result = core.Dilemma(lang='grc')
    assert not result._positive_elision_target(target)
    assert result._positive_elision_target('ταῦς')


def test_quantity_and_lexicon_separators_are_not_citation_distinctions(tmp_path, monkeypatch):
    monkeypatch.setattr(core.Dilemma, '_load_lookups', lambda *a, **k: None)
    monkeypatch.setattr(core.Dilemma, '_check_backend', lambda *a: None)
    path = tmp_path/'heads.json'
    path.write_text(json.dumps({'heads': {'Σικυών': ['edited']}}))
    monkeypatch.setattr(core, 'GRC_LEXICAL_HEADS_PATH', path)
    result = core.Dilemma(lang='grc')
    assert result._positive_elision_target('Σῐ-κυών')


def test_missing_evidence_does_not_authorize_new_priority(tmp_path, monkeypatch):
    monkeypatch.setattr(core.Dilemma, '_load_lookups', lambda *a, **k: None)
    monkeypatch.setattr(core.Dilemma, '_check_backend', lambda *a: None)
    monkeypatch.setattr(core, 'GRC_LEXICAL_HEADS_PATH', tmp_path/'missing.json')
    assert not core.Dilemma(lang='grc')._positive_elision_target('Σικυών')


@pytest.fixture(params=["all", "grc"])
def model(request, monkeypatch):
    monkeypatch.setattr(core.Dilemma, "_load_lookups", lambda *a, **k: None)
    monkeypatch.setattr(core.Dilemma, "_check_backend", lambda *a: None)
    result = core.Dilemma(lang=request.param)
    monkeypatch.setattr(result, "_lang_of", lambda word: "grc")
    monkeypatch.setattr(result, "attestation", lambda word: None)
    monkeypatch.setattr(result, "_positive_elision_target", lambda word: True)
    monkeypatch.setattr(result, "_lookup_elided", lambda word: "Σικυών")
    monkeypatch.setattr(result, "_expand_elision_all", lambda word: [
        ("Σικυῶνω", "σικύωνος", "ω"),
        ("Σικυῶνε", "σικυών", "ε"),
        ("Σικυῶνα", "Σικυών", "α"),
        ("Σικυῶνη", "Σικυώνη", "η"),
    ])
    return result


def test_exact_name_stays_first_and_all_readings_survive(model):
    candidates = model.lemmatize_verbose("Σικυῶν̓", guess=False)
    assert candidates[0].lemma == "Σικυών"
    assert candidates[0].via == "elision-exact"
    assert {row.lemma for row in candidates} == {
        "Σικυών", "σικύωνος", "σικυών", "Σικυώνη"}
    assert model.lemmatize("Σικυῶν̓", guess=False) == "Σικυών"
    assert model.lemmatize_batch(["Σικυῶν̓"], guess=False) == ["Σικυών"]


@pytest.mark.parametrize("upos", ["NOUN", "VERB", "PROPN"])
def test_unmatched_pos_does_not_replace_exact_name(model, upos):
    model._pos_lookup = {"σικυών": {upos: "ἄλλος"}}
    assert model.lemmatize_pos("Σικυῶν̓", upos, guess=False) == "Σικυών"
    assert model.lemmatize_batch_pos(["Σικυῶν̓"], [upos], guess=False) == ["Σικυών"]


def test_matching_pos_can_still_select_another_reading(model):
    model._pos_lookup = {"σικυών": {"NOUN": "σικύωνος"}}
    assert model.lemmatize_pos("Σικυῶν̓", "NOUN", guess=False) == "σικύωνος"
    assert model.lemmatize_batch_pos(["Σικυῶν̓"], ["NOUN"], guess=False) == ["σικύωνος"]
    assert model.lemmatize_verbose("Σικυῶν̓", guess=False)[0].lemma == "Σικυών"


def test_without_exact_reading_common_expansion_still_has_priority(model, monkeypatch):
    monkeypatch.setattr(model, "_lookup_elided", lambda word: None)
    candidates = model.lemmatize_verbose("Σικυῶν̓", guess=False)
    assert candidates[0].lemma == "σικύωνος"
    assert {row.lemma for row in candidates} == {
        "Σικυών", "σικύωνος", "σικυών", "Σικυώνη"}


def test_unsupported_exact_target_keeps_prior_order_and_all_candidates(model, monkeypatch):
    monkeypatch.setattr(model, '_positive_elision_target', lambda word: False)
    rows = model.lemmatize_verbose('Σικυῶν̓', guess=False)
    assert rows[0].lemma == 'σικύωνος'
    assert {r.lemma for r in rows} == {'Σικυών', 'σικύωνος', 'σικυών', 'Σικυώνη'}
    assert model.lemmatize_pos('Σικυῶν̓', 'NOUN', guess=False) == 'σικύωνος'


def test_raw_support_survives_monotonic_convention(model, tmp_path, monkeypatch):
    path = tmp_path/'heads.json'
    path.write_text(json.dumps({'heads': {'Ὅμηρος': ['edited']}}))
    monkeypatch.setattr(core, 'GRC_LEXICAL_HEADS_PATH', path)
    monkeypatch.setattr(model, '_positive_elision_target',
                        core.Dilemma._positive_elision_target.__get__(model))
    monkeypatch.setattr(model, '_lookup_elided', lambda word: 'Ὅμηρος')
    monkeypatch.setattr(model, '_expand_elision_all',
                        lambda word: [('Ὅμηρε', 'ὁμηρίζω', 'ε')])
    model._convention_monotonic = True
    model._convention_name = 'triantafyllidis'
    rows = model.lemmatize_verbose('Ὅμηῤ', guess=False)
    assert rows[0].lemma == 'Όμηρος' and rows[0].via == 'elision-exact'


def test_bare_stem_bridge_keeps_its_exact_reading_first(model, monkeypatch):
    expansions = model._expand_elision_all("Σικυῶν᾽")
    monkeypatch.setattr(model, "_lookup_elided", lambda word:
                        "Σικυών" if word.endswith("᾽") else None)
    monkeypatch.setattr(model, "_expand_elision_all", lambda word:
                        expansions if word.endswith("᾽") else [])
    model._ag_lookup = {"Σικυῶν": "Σικυῶν᾽"}
    model._lookup = model._ag_lookup
    candidates = model.lemmatize_verbose("Σικυῶν", guess=False)
    assert candidates[0].lemma == "Σικυών"
    assert candidates[0].via == "elision-exact"
    assert {row.lemma for row in candidates} == {
        "Σικυών", "σικύωνος", "σικυών", "Σικυώνη"}


@pytest.mark.parametrize("mark", sorted(core._ELISION_MARKS))
def test_every_recognized_mark_uses_the_same_exact_entry(model, monkeypatch, mark):
    model._lookup = {"Σικυῶν᾽": "Σικυών"}
    monkeypatch.setattr(model, "_lookup_elided",
                        core.Dilemma._lookup_elided.__get__(model))
    word = "Σικυῶν" + mark
    assert model._lookup_elided(word) == "Σικυών"
    candidates = model.lemmatize_verbose(word, guess=False)
    assert candidates[0].lemma == "Σικυών"
    assert candidates[0].via == "elision-exact"
