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
    monkeypatch.setattr(B, "synthesize_missing_moods", lambda result, families: calls.append(families))
    monkeypatch.setattr(B, "synthesize_missing_participles", lambda result, families: calls.append(families))
    result = B.build_paradigms()
    assert result["ἀέσκω"]["forms"]["active_present_indicative_1sg"] == "ἀέσκω"
    assert calls == [{"ἀέσκω": {("active", "aorist")}, "ἀφραδέω": {("active", "present")}}] * 2
