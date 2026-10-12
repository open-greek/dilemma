"""Defined, exactly attested names survive flat lowercase lookup owners."""
from __future__ import annotations

import pytest

from export_hunspell import (
    FormProfileEvidence,
    cased_form_key,
    exact_form_key,
    filter_case_artifacts,
    finalize_grc_pairs,
    include_attested_name_ownership,
    lexical_head_key,
)


def _evidence(counts):
    return FormProfileEvidence(
        {}, {}, {}, {}, cased={cased_form_key(k): v for k, v in counts.items()}
    )


def test_attested_name_retains_competing_common_owner():
    ownership = {exact_form_key("Ἱπποκράτην"): 2}
    evidence = _evidence({"Ἱπποκράτην": 271})
    rows = include_attested_name_ownership(
        ownership, evidence,
        readings={"Ἱπποκράτην": {"NOUN": "Ἱπποκράτης"}},
        lexical_heads=frozenset({lexical_head_key("Ἱπποκράτης")}),
    )
    assert ownership[exact_form_key("Ἱπποκράτην")] == 3
    pairs = [("Ἱπποκράτην", "ἱπποκράτης"), ("ἱπποκράτην", "ἱπποκράτης")]
    assert filter_case_artifacts(pairs, evidence, ownership) == (pairs, [])
    assert rows == [{
        "form": "Ἱπποκράτην", "lemmas": ["Ἱπποκράτης"],
        "cased_count": 271, "source": "treebank_pos_lookup.audit.json",
        "previous_owner_bits": 2, "owner_bits": 3,
    }]


@pytest.mark.parametrize("counts,readings,heads", [
    ({"ἱπποκράτην": 271}, {"NOUN": "Ἱπποκράτης"}, {"Ἱπποκράτης"}),
    ({"Ἱπποκράτην": 0}, {"NOUN": "Ἱπποκράτης"}, {"Ἱπποκράτης"}),
    ({"Ἱπποκράτην": 271}, {"VERB": "Ἱπποκράτης"}, {"Ἱπποκράτης"}),
    ({"Ἱπποκράτην": 271}, {"NOUN": "ἱπποκράτης"}, {"Ἱπποκράτης"}),
    ({"Ἱπποκράτην": 271}, {"NOUN": "Ἱπποκράτης"}, set()),
])
def test_projected_capitals_and_undefined_owners_do_not_supply_evidence(
    counts, readings, heads
):
    ownership = {exact_form_key("Ἱπποκράτην"): 2}
    assert include_attested_name_ownership(
        ownership, _evidence(counts), readings={"Ἱπποκράτην": readings},
        lexical_heads=frozenset(map(lexical_head_key, heads)),
    ) == []
    assert ownership == {exact_form_key("Ἱπποκράτην"): 2}


def test_missing_profile_cannot_restore_projected_capitals():
    ownership = {exact_form_key("Ἱπποκράτην"): 2}
    assert include_attested_name_ownership(
        ownership, FormProfileEvidence({}, {}, {}, {}),
        readings={"Ἱπποκράτην": {"NOUN": "Ἱπποκράτης"}},
        lexical_heads=frozenset({lexical_head_key("Ἱπποκράτης")}),
    ) == []


def test_name_evidence_preserves_distinct_marked_lexical_readings():
    ownership = {exact_form_key("Πομπηίου"): 2}
    assert include_attested_name_ownership(
        ownership, _evidence({"Πομπηίου": 384}),
        readings={"Πομπηίου": {"NOUN": "Πομπήιος"}},
        lexical_heads=frozenset({lexical_head_key("Πομπήϊος")}),
    ) == []
    assert ownership[exact_form_key("Πομπηίου")] == 2


def test_all_supported_name_readings_remain_in_audit():
    ownership = {exact_form_key("Ἱπποκράτην"): 2}
    rows = include_attested_name_ownership(
        ownership, _evidence({"Ἱπποκράτην": 271}),
        readings={"Ἱπποκράτην": {"NOUN": "Ἱπποκράτης", "PROPN": "Ἱπποκράτη"}},
        lexical_heads=frozenset(map(lexical_head_key, {"Ἱπποκράτης", "Ἱπποκράτη"})),
    )
    assert rows[0]["lemmas"] == ["Ἱπποκράτη", "Ἱπποκράτης"]


def test_attested_grave_counts_support_acute_name_without_lowercase_leak():
    ownership = {exact_form_key("Ἁβραάμ"): 2, exact_form_key("Γερμανία"): 1}
    evidence = _evidence({"Ἁβραὰμ": 540, "Γερμανία": 5})
    include_attested_name_ownership(
        ownership, evidence,
        readings={"Ἁβραάμ": {"NOUN": "Ἀβραάμ"},
                  "Γερμανία": {"PROPN": "Γερμανία"}},
        lexical_heads=frozenset(map(lexical_head_key, {"Ἀβραάμ", "Γερμανία"})),
    )
    pairs = [("Ἁβραὰμ", "ἀβραάμ"), ("Γερμανία", "Γερμανία"),
             ("γερμανία", "Γερμανία")]
    assert filter_case_artifacts(pairs, evidence, ownership) == (pairs[:2], pairs[2:])


def test_attested_ethnic_adjective_keeps_independent_capital_owner():
    ownership = {exact_form_key(form): 2 for form in ("Ἰταλικῶν", "Σαφής")}
    rows = include_attested_name_ownership(
        ownership, _evidence({"Ἰταλικῶν": 96, "Σαφής": 20}),
        readings={"Ἰταλικῶν": {"ADJ": "Ἰταλικός"}, "Σαφής": {"ADJ": "σαφής"}},
        lexical_heads=frozenset(map(lexical_head_key, {"Ἰταλικός", "σαφής"})),
    )
    assert [row["form"] for row in rows] == ["Ἰταλικῶν"]
    pairs = [("Ἰταλικῶν", "ἰταλικός"), ("Σαφής", "σαφής")]
    assert filter_case_artifacts(pairs, _evidence({}), ownership) == (pairs[:1], pairs[1:])


def test_corpus_typo_cannot_bypass_existing_orthography_guard():
    ownership = {exact_form_key("Δίά"): 2}
    assert include_attested_name_ownership(
        ownership, _evidence({"Δίά": 2}), readings={"Δίά": {"PROPN": "Ζεύς"}},
        lexical_heads=frozenset({lexical_head_key("Ζεύς")}),
    ) == []


def test_attested_compatibility_spelling_is_retained_without_inventing_an_owner():
    form = "Θεσμοφοριαζούσαις"
    ownership = {exact_form_key(form): 2}
    retained = []
    pairs = [(form, "θεσμοφοριάζω")]
    assert filter_case_artifacts(pairs, _evidence({form: 48}), ownership,
                                 {form}, retained) == (pairs, [])
    assert ownership == {exact_form_key(form): 2}
    assert retained == [{
        "form": form, "cased_count": 48, "owner_bits": 2,
        "reason": "attested_inherited_spelling; lexical_ownership_unresolved",
    }]


@pytest.mark.parametrize("counts,compatibility", [
    ({"Θεσμοφοριαζούσαις": 0}, {"Θεσμοφοριαζούσαις"}),
    ({"θεσμοφοριαζούσαις": 48}, {"Θεσμοφοριαζούσαις"}),
    ({"Θεσμοφοριαζούσαις": 48}, set()),
    ({"Θεσμοφοριαζούσαις": 48}, {"θεσμοφοριαζούσαις"}),
])
def test_case_retention_requires_positive_cased_count_and_exact_compatibility(counts, compatibility):
    form = "Θεσμοφοριαζούσαις"
    pairs = [(form, "θεσμοφοριάζω")]
    retained = []
    assert filter_case_artifacts(pairs, _evidence(counts),
        {exact_form_key(form): 2}, compatibility, retained) == ([], pairs)
    assert retained == []


def test_absent_cased_profile_does_not_preserve_inherited_capital():
    form = "Θεσμοφοριαζούσαις"
    pairs = [(form, "θεσμοφοριάζω")]
    assert filter_case_artifacts(pairs, FormProfileEvidence({}, {}, {}, {}),
        {exact_form_key(form): 2}, {form}) == ([], pairs)


@pytest.mark.parametrize("veto", ["reviewed", "bare"])
def test_explicit_rejection_precedes_inherited_case_retention(veto):
    form = "Θεσμοφοριαζούσαις"
    pairs, report = finalize_grc_pairs([(form, "θεσμοφοριάζω")],
        evidence=_evidence({form: 48}), compatibility_forms={form},
        textbook_forms=set(), export_overrides={}, protected_forms=set(),
        case_ownership={exact_form_key(form): 2},
        reviewed_rejects=frozenset({form}) if veto == "reviewed" else frozenset(),
        bare_elision_keys={cased_form_key(form)} if veto == "bare" else set())
    assert pairs == []
    assert report["attested_compatibility_cases"] == []


def test_zero_count_lowercase_name_alias_is_not_protected_by_compatibility():
    pairs = [("ἑλλάδος", "Ἑλλάς")]
    assert filter_case_artifacts(pairs, _evidence({"Ἑλλάδος": 300}),
        {exact_form_key("ἑλλάδος"): 1}, {"ἑλλάδος"}) == ([], pairs)
