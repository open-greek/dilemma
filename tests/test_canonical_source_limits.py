"""Source citation marks and defective tense/voice limits remain explicit."""

import pytest

from build import build_grc_verb_paradigms as B


def pair(lemma="οικειόομαι", form="οἰκειόομαι", tags=None):
    return {"lemma": lemma, "form": form, "pos": "verb",
            "tags": tags or ["present", "indicative", "middle", "first-person", "singular"]}


def test_unique_breathing_recovery_uses_own_present_citation():
    assert B.recover_citation_breathings([pair()]) == {"οικειόομαι": "οἰκειόομαι"}


@pytest.mark.parametrize("row", [
    pair(lemma="οἰκειόομαι", form="οἱκειόομαι"),
    pair(form="οἰκείομαι"),
    pair(form="οἰκειοόμαι"),
    pair(form="οἰκειόομαί"),
    pair(form="οἰκειόομαι", tags=["present", "indicative", "middle", "second-person", "singular"]),
])
def test_marked_head_accents_letters_and_cell_are_not_replaced(row):
    assert B.recover_citation_breathings([row]) == {}


def test_two_supported_breathings_remain_unresolved():
    assert B.recover_citation_breathings([pair(), pair(form="οἱκειόομαι")]) == {}


def test_family_limit_keeps_only_supported_voice_and_tense():
    forms = {"active_aorist_infinitive": "ἄεσαι",
             "active_aorist_optative_1sg": "ἀέσαιμι",
             "active_present_infinitive": "ἀέσκειν",
             "middle_aorist_infinitive": "ἀέσασθαι"}
    assert B.restrict_synthesis_family(forms, {("active", "aorist")}) == {
        "active_aorist_infinitive": "ἄεσαι", "active_aorist_optative_1sg": "ἀέσαιμι"}
    assert B.restrict_synthesis_family(forms, None) is forms
    assert B.restrict_synthesis_family(forms, set()) == {}


def test_both_synthesis_passes_respect_the_original_family(monkeypatch):
    monkeypatch.setattr(B, "load_lsj_head_texts", lambda: {
        "ἀέσκω": "aor. 1 ἄεσα, inf. ἄεσαι :—sleep."})
    forms = {"active_aorist_infinitive": "ἄεσαι"}
    result = {"ἀέσκω": {"forms": dict(forms), "form_count": 1}}
    limits = {"ἀέσκω": {("active", "aorist")}}
    B.synthesize_missing_moods(result, limits)
    B.synthesize_missing_participles(result, limits)
    assert result["ἀέσκω"]["forms"]["active_aorist_infinitive"] == "ἄεσαι"
    assert len(result["ἀέσκω"]["forms"]) > 1
    assert {tuple(key.split("_")[:2]) for key in result["ἀέσκω"]["forms"]} == {("active", "aorist")}


def test_family_evidence_excludes_inserted_citation_and_other_head_tags(monkeypatch, tmp_path):
    rows = [
        {"lemma": "ἀέσκω", "form": "ἀέσκω", "pos": "verb", "tags": ["verb", "defective"]},
        {"lemma": "ἀέσκω", "form": "ἄεσαι", "pos": "verb", "tags": ["active", "aorist", "infinitive"]},
        {"lemma": "ἀφραδέω", "form": "ἀφραδέω", "pos": "verb", "tags": ["verb", "defective"]},
        {"lemma": "ἀφραδέω", "form": "ἀφραδέουσι", "pos": "verb", "tags": ["active", "present", "indicative", "third-person", "plural", "Epic"]},
        {"lemma": "λέγω", "form": "ἠμί", "pos": "verb", "tags": ["verb", "defective"]},
        {"lemma": "λέγω", "form": "λέγει", "pos": "verb", "tags": ["active", "present", "indicative", "third-person", "singular"]},
    ]
    monkeypatch.setattr(B, "LSJ_HEADWORDS_PATH", tmp_path / "absent")
    monkeypatch.setattr(B, "load_pairs", lambda path: rows if path == B.AG_PAIRS else [])
    monkeypatch.setattr(B, "pick_best_form", lambda forms, **kw: forms[0])
    calls = []
    monkeypatch.setattr(B, "synthesize_missing_moods", lambda result, families, quantities: calls.append(families))
    monkeypatch.setattr(B, "synthesize_missing_participles", lambda result, families, quantities: calls.append(families))
    result = B.build_paradigms()
    assert result["ἀέσκω"]["forms"]["active_present_indicative_1sg"] == "ἀέσκω"
    assert calls == [{"ἀέσκω": {("active", "aorist")}, "ἀφραδέω": {("active", "present")}}] * 2


@pytest.mark.parametrize("lemma,header,observed,future", [
    ("ἁνδάνω", "fut. ἁδήσω: aor. 1 ἧσα.", "ἅδοι", "ἁδήσων"),
    ("ἐνίπτω", "fut. ἐνίψω: aor. ἠνίπᾰπε, also ἐνένιπε.", "ἐνένιπε", "ἐνίψων"),
])
def test_irregular_aorist_does_not_inherit_future_stem(
    monkeypatch, lemma, header, observed, future
):
    monkeypatch.setattr(B, "load_lsj_head_texts", lambda: {lemma: header})
    observed_key = "active_aorist_optative_3sg" if lemma == "ἁνδάνω" else "active_aorist_indicative_3sg"
    original = {observed_key: observed}
    result = {lemma: {"forms": dict(original), "form_count": 1}}
    B.synthesize_missing_moods(result)
    B.synthesize_missing_participles(result)
    forms = result[lemma]["forms"]
    assert forms[observed_key] == observed
    assert "active_present_imperative_2sg" in forms
    assert forms["active_future_participle_nom_m_sg"] == future
    assert "active_aorist_imperative_2sg" not in forms
    assert not any("_aorist_participle_" in key for key in forms)
    assert not any(key.startswith("middle_aorist_") for key in forms)
    if lemma == "ἁνδάνω":
        assert forms["active_aorist_indicative_1sg"] == "ἧσα"


def test_suppletive_aorist_alone_does_not_license_regular_participles(monkeypatch):
    monkeypatch.setattr(B, "load_lsj_head_texts", lambda: {
        "ἁνδάνω": "aor. 1 ἧσα."})
    result = {"ἁνδάνω": {"forms": {
        "active_aorist_participle_nom_m_sg": "ἁδών"}, "form_count": 1}}
    B.synthesize_missing_participles(result)
    forms = result["ἁνδάνω"]["forms"]
    assert {key: value for key, value in forms.items()
            if "_aorist_" in key} == {"active_aorist_participle_nom_m_sg": "ἁδών"}


def test_aorist_source_limit_keeps_original_parts_and_ordinary_inference():
    parts = {"fut": "ἁδήσω", "aor": "ἧσα", "pf": "ἅδηκα"}
    assert B.parts_for_aorist_synthesis("ἁνδάνω", parts) == {
        "aor": "ἧσα", "pf": "ἅδηκα"}
    assert parts["fut"] == "ἁδήσω"
    assert B.parts_for_aorist_synthesis("λύω", parts) is parts


def test_source_quantity_does_not_copy_accent_or_choose_a_conflicting_length():
    evidence = [{"form": "θῡ́νω"}]
    assert B.restore_source_quantity("θύνω", evidence) == "θῡ́νω"
    assert B.restore_source_quantity("θύνω", [{"form": "θῡ̀νω"}]) == "θύνω"
    assert B.restore_source_quantity("θύνω", evidence + [{"form": "θῠ́νω"}]) == "θύνω"
    assert B.restore_source_quantity("θύω", evidence) == "θύω"
    assert B.restore_source_quantity("θῠ́νω", evidence) == "θῠ́νω"
    assert B.restore_source_quantity("θῦνω", [{"form": "θῠ́νω"}]) == "θῦνω"


def test_explicit_principal_quantities_survive_synthesis_without_notation(monkeypatch):
    monkeypatch.setattr(B, "load_lsj_head_texts", lambda: {
        "θύνω": "[ῡ], only pres. and impf., rush.",
        "πράσσω": "fut. πράξω: aor. ἔπραξα."})
    result = {head: {"forms": {}, "form_count": 0} for head in ("θύνω", "πράσσω")}
    quantities = {"θύνω": [{"form": "θῡ́νω"}],
                  "πράσσω": [{"form": "πρᾱ́σσω"}, {"form": "πρᾱ́ξω"},
                             {"form": "ἔπρᾱξα"}]}
    B.synthesize_missing_moods(result, principal_quantities=quantities)
    B.synthesize_missing_participles(result, principal_quantities=quantities)
    assert result["θύνω"]["forms"]["active_present_imperative_2sg"] == "θῦνε"
    assert result["πράσσω"]["forms"]["middle_aorist_imperative_2sg"] == "πρᾶξαι"
    assert result["πράσσω"]["forms"]["active_aorist_optative_3sg"] == "πράξαι"
    assert all(B.strip_vowel_length(form) == form
               for row in result.values() for form in row["forms"].values())
