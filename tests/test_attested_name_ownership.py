"""Defined, exactly attested names survive flat lowercase lookup owners."""
from __future__ import annotations

import pytest

from export_hunspell import (
    FormProfileEvidence,
    cased_form_key,
    exact_form_key,
    filter_case_artifacts,
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
