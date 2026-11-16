"""Regression coverage for the pinned Ancient Greek expansion layer."""

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from overlay_lsj import (
    overlay_expansion,
    resolve_target_editorial_entries,
    verify_file,
    verify_sentinels,
    load_noun_generation_audit,
    restore_attested_compound_aliases,
    load_alias_lexical_heads,
    load_alias_source_pairs,
    alias_spelling_exclusions,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "ag_expansion_reference.json"
LOOKUP_DB = ROOT / "data" / "lookup.db"


def test_attested_alias_recovery_uses_source_cells_and_corrected_owners():
    reference = {"ὕπην": "ὕπειμι", "υπην": "ὕπειμι",
                 "ἄνοντος": "ἄνειμι", "ανοντος": "ἄνειμι",
                 "παρέξοντος": "παρέξειμι", "παρεξοντος": "παρέξειμι"}
    target = {}
    sources = [("glaux_pairs.json", [
        {"form": "ὑπῆν", "lemma": "ὕπειμι"},
        {"form": "ἄνοντος", "lemma": "ἄνω"},
        {"form": "παρέξοντος", "lemma": "παρέχω"},
    ])]
    audit = restore_attested_compound_aliases(
        reference, {}, target, sources, {"ὕπειμι", "ἄνω", "παρέχω"})
    assert target["υπην"] == "ὕπειμι"
    assert target["ανοντος"] == "ἄνω"
    assert target["παρεξοντος"] == "παρέχω"
    assert "ὕπην" not in target
    assert audit["recovered_count"] == len(target)


def test_source_alias_scope_includes_old_base_cells_and_defined_compound_heads():
    reference = {"ανοντος": "ἄνω", "παρεξοντος": "παρέχω"}
    base = {"ανοντος": "ἄνω", "παρεξοντος": "παρέχω",
            "διέξειμι": "διέξειμι"}
    target = {}
    sources = [("glaux_pairs.json", [{"form": "ἄνοντος", "lemma": "ἄνω"},
                                     {"form": "παρέξοντος", "lemma": "παρέχω"}])]
    restore_attested_compound_aliases(reference, base, target, sources,
                                     {"ἄνω", "παρέχω", "ἄνειμι", "παρέξειμι", "διέξειμι"})
    assert target["ανοντος"] == "ἄνω"
    assert target["παρεξοντος"] == "παρέχω"
    assert target["διέξειμι"] == target["διεξειμι"] == "διέξειμι"


def test_compound_cell_shape_never_generates_or_licenses_a_spelling():
    reference = {"ανοντος": "ἄνω", "παρεξοντος": "παρέχω"}
    target = {}
    audit = restore_attested_compound_aliases(reference, {}, target, [],
                                             {"ἄνειμι", "παρέξειμι"})
    assert audit["pattern_scoped_aliases"] == 2
    assert not target


def test_existing_historical_key_can_use_a_corrected_compound_source_owner():
    reference = {"αντεπησαν": "ἀντεπαείδω"}
    target = {}
    audit = restore_attested_compound_aliases(reference, {}, target,
        [("glaux_pairs.json", [{"form": "ἀντεπῇσαν", "lemma": "ἀντέπειμι"},
                               {"form": "ἀντεπῇς", "lemma": "ἀντέπειμι"}])],
        {"ἀντέπειμι"})
    assert target["αντεπησαν"] == "ἀντέπειμι"
    assert "ἀντεπῇς" not in target
    assert audit["current_compound_source_scoped_aliases"] == 1


def test_represented_historical_family_can_retain_actual_source_cells_only():
    reference = {"ἀντεπίοι": "ἀντέπειμι"}
    target = {}
    audit = restore_attested_compound_aliases(reference, {}, target,
        [("glaux_pairs.json", [{"form": "ἀντεπῇσαν", "lemma": "ἀντέπειμι"},
                               {"form": "ὑπῆν", "lemma": "ὕπειμι"}])],
        {"ἀντέπειμι", "ὕπειμι"})
    assert target["αντεπησαν"] == "ἀντέπειμι"
    assert "ὑπῆν" not in target
    assert "ἀντεπίοι" not in target
    assert audit["additional_historical_family_source_cell_scope"] == 2


def test_source_precedence_retains_all_exact_cell_readings():
    reference = {"διέξιών": "διέξειμι", "διεξιων": "διέξειμι"}
    sources = [
        ("perseus_pairs.json", [{"form": "διεξιὼν", "lemma": "διέξειμι"}]),
        ("glaux_pairs.json", [{"form": "διεξιών", "lemma": "διεξέρχομαι"}]),
    ]
    target = {}
    audit = restore_attested_compound_aliases(
        reference, {}, target, sources, {"διέξειμι", "διεξέρχομαι"})
    assert target["διεξιων"] == "διέξειμι"
    row = next(row for row in audit["recovered"] if row["form"] == "διεξιων")
    assert [r["owner"] for r in row["cells"][0]["readings"]] == [
        "διέξειμι", "διεξέρχομαι"]


def test_automatic_owner_disputes_do_not_replace_correct_bare_fallbacks():
    reference = {"συνηγμενον": "σύνειμι", "ανηγμενος": "ἄνειμι",
                 "παρηγμενα": "πάρειμι"}
    sources = [("glaux_pairs.json", [
        {"form": "συνηγμένον", "lemma": "σύνειμι"},
        {"form": "ἀνηγμένος", "lemma": "ἄνειμι"},
        {"form": "παρηγμένα", "lemma": "πάρειμι"},
    ]), ("diorisis_pairs.json", [
        {"form": "συνηγμένον", "lemma": "συνάγω"},
        {"form": "ἀνηγμένος", "lemma": "ἀνάγω"},
        {"form": "παρηγμένα", "lemma": "παράγω"},
    ])]
    target = {"συνηγμένον": "σύνειμι", "ἀνηγμένος": "ἄνειμι",
              "παρηγμένα": "πάρειμι"}
    audit = restore_attested_compound_aliases(reference, {}, target, sources,
        {"σύνειμι", "ἄνειμι", "πάρειμι", "συνάγω", "ἀνάγω", "παράγω"})
    assert not set(reference) & target.keys()
    assert audit["automatic_owner_conflict_count"] == 3
    assert target["συνηγμένον"] == "σύνειμι"
    assert all(row["cells"][0]["automatic_owner_conflict"]
               for row in audit["ambiguous"])


def test_unsupported_perfect_middle_annotation_needs_direct_owner_evidence():
    reference = {"παρηγμενα": "πάρειμι"}
    pair = {"form": "παρηγμένα", "lemma": "πάρειμι", "pos": "verb",
            "tags": ["perfect", "participle", "middle", "plural", "neuter"]}
    target = {"παρηγμένα": "πάρειμι"}
    audit = restore_attested_compound_aliases(reference, {}, target,
        [("glaux_pairs.json", [pair])], {"πάρειμι"})
    assert "παρηγμενα" not in target
    assert audit["unsupported_annotation_morphology_count"] == 1
    assert target["παρηγμένα"] == "πάρειμι"
    directly_reviewed = [("hunspell_grc_postclassical_review.json", [pair])]
    restore_attested_compound_aliases(reference, {}, target, directly_reviewed, {"πάρειμι"})
    assert target["παρηγμενα"] == "πάρειμι"


@pytest.mark.parametrize("tense", ["aorist", "perfect", "pluperfect"])
def test_defective_compound_tense_annotations_do_not_license_new_aliases(tense):
    target = {"προσυποσχεῖν": "προσυποσχάω"}
    audit = restore_attested_compound_aliases(
        {"προσυποσχειν": "προσύπειμι"}, {}, target,
        [("glaux_pairs.json", [{"form": "προσυποσχεῖν", "lemma": "προσύπειμι",
                                "tags": [tense, "infinitive", "active"]}])],
        {"προσύπειμι"})
    assert "προσυποσχειν" not in target
    assert target["προσυποσχεῖν"] == "προσυποσχάω"
    assert audit["unsupported_annotation_morphology_count"] == 1


def test_longer_positive_compound_shape_withholds_without_inferring_its_owner():
    target = {"συνανιόντε": "συνανίω"}
    audit = restore_attested_compound_aliases(
        {"συνανιοντε": "ἄνειμι"}, {}, target,
        [("ag_pairs.json", [{"form": "ἰόντε", "lemma": "εἶμι"}]),
         ("glaux_pairs.json", [{"form": "συνανιόντε", "lemma": "ἄνειμι",
                                 "tags": ["present", "participle", "active"]}])],
        {"ἄνειμι", "συνάνειμι", "εἶμι"})
    assert target["συνανιόντε"] == "συνανίω"
    assert "συνανιοντε" not in target
    assert audit["unsupported_compound_owner_shape_count"] == 1
    row = next(row for row in audit["ambiguous"] if row["form"] == "συνανιοντε")
    assert row["cells"][0]["potential_compound_shape_owners"] == ["συνάνειμι"]


@pytest.mark.parametrize("form,bare,monotonic,old_owner,automatic_owner,reviewed_owner", [
    ("συνανιόντε", "συνανιοντε", "συνανιόντε", "συνανίω", "ἄνειμι", "συνάνειμι"),
    ("προσυποσχεῖν", "προσυποσχειν", "προσυποσχείν", "προσυποσχάω", "προσύπειμι", "προσυπέχω"),
])
def test_cited_exact_cells_correct_bad_annotations_without_generating_a_family(
        form, bare, monotonic, old_owner, automatic_owner, reviewed_owner,
        actual_reviewed_alias_sources):
    target = {form: old_owner}
    reviewed = [actual_reviewed_alias_sources[0], ("diorisis_pairs.json", [
        {"form": form, "lemma": automatic_owner, "tags": ["aorist"]}])]
    audit = restore_attested_compound_aliases({bare: automatic_owner}, {}, target,
        reviewed, {automatic_owner, reviewed_owner})
    assert target[form] == target[bare] == reviewed_owner
    assert {key for key, owner in target.items() if owner == reviewed_owner} == {
        form, bare, monotonic}
    correction = next(row for row in audit["canonicalized"] if row["form"] == form)
    assert correction["owner"] == reviewed_owner
    assert any(row.get("citation") for row in correction["cells"][0]["readings"]
               if row["source"] == "hunspell_grc_postclassical_review.json")


@pytest.fixture
def actual_reviewed_alias_sources(tmp_path):
    from overlay_lsj import SOURCE_PAIR_NAMES

    review_name = "hunspell_grc_postclassical_review.json"
    (tmp_path / review_name).write_bytes((ROOT / "data" / review_name).read_bytes())
    for name in SOURCE_PAIR_NAMES:
        (tmp_path / name).write_text("[]")
    return load_alias_source_pairs(tmp_path)[0]


def test_cited_elision_resolves_annotations_and_keeps_their_readings(
        actual_reviewed_alias_sources):
    sources = [actual_reviewed_alias_sources[0], ("glaux_pairs.json", [
        {"form": "ἀπῆσθ’", "lemma": "ἄπειμι", "tags": ["plural", "imperfect"]}
    ]), ("diorisis_pairs.json", [
        {"form": "ἀπῆσθ᾽", "lemma": "ἄφημαι", "tags": ["plural", "imperfect"]}
    ])]
    target = {"ἀπῆσθ᾽": "ἄπειμι"}
    audit = restore_attested_compound_aliases({"απησθ’": "ἄπειμι"}, {}, target,
        sources, {"ἄπειμι", "ἄφημαι"})
    assert target["απησθ’"] == target["απησθ᾽"] == "ἄπειμι"
    assert "απησθ" not in target
    recovered = next(row for row in audit["recovered"] if row["form"] == "απησθ’")
    assert {row["owner"] for row in recovered["cells"][0]["readings"]} == {
        "ἄπειμι", "ἄφημαι"}


def test_automatic_capitalized_verb_annotations_preserve_names_and_lower_readings():
    target = {"Προσεῖς": "Προσεύς", "Σύνεις": "Σύνεις",
              "προσεῖς": "πρόσω", "προσεις": "πρόσω",
              "σύνεις": "συνίημι", "συνεις": "συνίημι"}
    original = dict(target)
    audit = restore_attested_compound_aliases(
        {"Προσεις": "πρόσειμι", "Συνεις": "σύνειμι"}, {}, target,
        [("diorisis_pairs.json", [
            {"form": "Προσεῖς", "lemma": "πρόσειμι"},
            {"form": "Σύνεις", "lemma": "σύνειμι"}])],
        {"πρόσειμι", "σύνειμι", "πρόσω", "συνίημι"})
    assert all(target[key] == value for key, value in original.items())
    assert "Προσεις" not in target and "Συνεις" not in target
    assert audit["unsupported_capitalized_annotation_count"] == 2


def test_matching_lowercase_cell_can_support_a_sentence_initial_verb_alias():
    target = {}
    restore_attested_compound_aliases({"Απην": "ἄπειμι", "απην": "ἄπειμι"}, {},
        target, [("perseus_pairs.json", [{"form": "ἀπῆν", "lemma": "ἄπειμι"}]),
                 ("diorisis_pairs.json", [{"form": "Ἀπῆν", "lemma": "ἄπειμι"}])],
        {"ἄπειμι"})
    assert target["Απην"] == target["Ἀπῆν"] == "ἄπειμι"


def test_lower_verb_proof_cannot_fold_away_a_distinct_capital_target():
    target = {"Ἀπῆν": "Ἄπειμι"}
    restore_attested_compound_aliases({"Απην": "ἄπειμι", "απην": "ἄπειμι"}, {},
        target, [("perseus_pairs.json", [{"form": "ἀπῆν", "lemma": "ἄπειμι"}]),
                 ("diorisis_pairs.json", [{"form": "Ἀπῆν", "lemma": "ἄπειμι"}])],
        {"ἄπειμι"})
    assert target["Ἀπῆν"] == "Ἄπειμι"
    assert "Απην" not in target


@pytest.mark.parametrize("sources", [
    ["diorisis_pairs.json"], ["glaux_pairs.json", "diorisis_pairs.json"],
])
def test_automatic_owners_cannot_rewrite_unsupported_metadata_without_direct_proof(sources):
    target = {"διέξει": "διεξέω"}
    restore_attested_compound_aliases({"διεξει": "διέξειμι"}, {}, target,
        [(source, [{"form": "διέξει", "lemma": "διέξειμι"}]) for source in sources],
        {"διέξειμι"})
    assert target["διέξει"] == "διεξέω"


def test_shared_new_testament_annotations_are_not_independent_owner_proof():
    target = {"διέξει": "διεξέω"}
    pair = {"form": "διέξει", "lemma": "διέξειμι"}
    restore_attested_compound_aliases({"διεξει": "διέξειμι"}, {}, target,
        [(source, [pair]) for source in (
            "nt_pairs.json", "ptnk_pairs.json", "tagnt_pairs.json")], {"διέξειμι"})
    assert target["διέξει"] == "διεξέω"


def test_source_backed_terminal_elision_keeps_original_and_bare_aliases():
    reference = {"απησθ’": "ἄπειμι"}
    target = {}
    audit = restore_attested_compound_aliases(reference, {}, target,
        [("glaux_pairs.json", [{"form": "ἀπῆσθ’", "lemma": "ἄπειμι"}])], {"ἄπειμι"})
    assert target["ἀπῆσθ’"] == target["απησθ’"] == "ἄπειμι"
    assert target["ἀπῆσθ᾽"] == target["απησθ᾽"] == "ἄπειμι"
    assert "απησθ" not in target
    elided = next(row for row in audit["recovered"] if row["form"] == "ἀπῆσθ’")
    reading = elided["cells"][0]["readings"][0]
    assert reading["original_form"] == "ἀπῆσθ’"
    assert reading["form"] == "ἀπῆσθ᾽"


def test_baseline_mapping_conflicts_allow_independent_source_owner_repairs(tmp_path):
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"schema_version": 1, "reviews": [
        {"form": "δεἰσί", "decision": "reject"}]}))
    exclusions = {"nonlexical": {"δηα": "abbreviation"},
                  "baseline_conflicts": {"ἀντεξῄεσαν": "ἀντί-ἔξειμι"}}
    blocked = alias_spelling_exclusions(exclusions, review)
    assert "ἀντεξῄεσαν" not in blocked
    assert {"δηα", "δεἰσί"} <= blocked
    # The pre-overlay AG table lacks this actual source cell; its older
    # pseudo-owner occurs in the combined lookup, not this target table.
    target = {}
    audit = restore_attested_compound_aliases(
        {"αντεξηεσαν": "ἀντέξειμι"}, {}, target,
        [("glaux_pairs.json", [{"form": "ἀντεξῄεσαν", "lemma": "ἀντέξειμι"}]),
         ("diorisis_pairs.json", [{"form": "ἀντεξῄεσαν", "lemma": "ἀντέξειμι"}])],
        {"ἀντέξειμι"}, excluded_forms=blocked,
        historical_conflicts=exclusions["baseline_conflicts"])
    assert target["αντεξηεσαν"] == "ἀντέξειμι"
    assert target["ἀντεξῄεσαν"] == "ἀντέξειμι"
    row = next(row for row in audit["recovered"] if row["form"] == "ἀντεξῄεσαν")
    assert row["historical_conflict_owner"] == "ἀντί-ἔξειμι"


def test_reviewed_split_fragment_cannot_license_source_aliases():
    blocked = alias_spelling_exclusions(
        {"nonlexical": {}, "baseline_conflicts": {}},
        ROOT / "data" / "hunspell_grc_spelling_review.json")
    assert "ἐνεί" in blocked
    assert "ἔνει" not in blocked
    target = {}
    audit = restore_attested_compound_aliases(
        {"ενει": "ἔνειμι"}, {}, target,
        [("glaux_pairs.json", [{"form": "ἐνεί", "lemma": "ἔνειμι"}])],
        {"ἔνειμι"}, excluded_forms=blocked)
    assert not {"ενει", "ένει", "ἐνεί", "ἐνεὶ"} & target.keys()
    assert all(row["form"] not in {"ἐνεί", "ἐνεὶ"} for row in audit["recovered"])


def test_reviewed_transcription_does_not_make_a_false_verb_alias():
    blocked = alias_spelling_exclusions(
        {"nonlexical": {}, "baseline_conflicts": {}},
        ROOT / "data" / "hunspell_grc_spelling_review.json")
    target = {"επει": "ἐπεί", "ἐπεί": "ἐπεί", "ἔπει": "ἔπειμι"}
    before = dict(target)
    restore_attested_compound_aliases(
        {"επει": "ἔπειμι"}, {}, target,
        [("glaux_pairs.json", [{"form": "ἐπεῖ", "lemma": "ἔπειμι"}]),
         ("diorisis_pairs.json", [{"form": "ἐπεῖ", "lemma": "ἔπειμι"}])],
        {"ἔπειμι", "ἐπεί"}, excluded_forms=blocked)
    assert "ἐπεῖ" not in target
    assert all(target[key] == owner for key, owner in before.items())


@pytest.mark.parametrize("hard,source", [("νί", "νὶ"), ("νὶ", "νί")])
def test_hard_rejections_apply_to_both_contextual_accent_directions(tmp_path, hard, source):
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"schema_version": 1, "reviews": []}))
    blocked = alias_spelling_exclusions({"nonlexical": {hard: "abbreviation"}}, review)
    assert "νί" in blocked
    target = {}
    restore_attested_compound_aliases({source: "πρόειμι"}, {}, target,
        [("glaux_pairs.json", [{"form": source, "lemma": "νί"}])], {"νί"},
        excluded_forms={hard})
    assert not target


def test_distinct_marked_owners_and_independent_target_readings_are_preserved():
    reference = {"ἄνειμι": "πρόειμι", "ανειμι": "πρόειμι"}
    sources = [("ag_pairs.json", [
        {"form": "ἄνειμι", "lemma": "ἄνειμι"},
        {"form": "ἀνεῖμι", "lemma": "ἀνίημι"},
    ])]
    target = {"ἄνειμι": "ἀνέρχομαι"}
    audit = restore_attested_compound_aliases(
        reference, {}, target, sources, {"ἄνειμι", "ἀνίημι", "ἀνέρχομαι"})
    assert "ανειμι" not in target
    assert target["ἄνειμι"] == "ἀνέρχομαι"
    assert any(row["form"] == "ανειμι" for row in audit["ambiguous"])


def test_direct_source_alias_recovery_can_repair_exact_unsupported_metadata():
    reference = {"ἀντέξηεσαν": "ἀντέξειμι", "αντεξηεσαν": "ἀντέξειμι"}
    sources = [("perseus_pairs.json", [
        {"form": "ἀντεξῄεσαν", "lemma": "ἀντέξειμι"},
    ])]
    target = {"ἀντεξῄεσαν": "ἀντί-ἔξειμι"}
    audit = restore_attested_compound_aliases(
        reference, {}, target, sources, {"ἀντέξειμι"})
    assert target["ἀντεξῄεσαν"] == "ἀντέξειμι"
    assert target["αντεξηεσαν"] == "ἀντέξειμι"
    assert audit["canonicalized_count"] == 1


def test_source_alias_recovery_requires_positive_heads_and_clean_source_forms():
    reference = {"ὕπον": "ὕπειμι", "υπον": "ὕπειμι",
                 "ἄλλλως": "πρόειμι", "αλλλως": "πρόειμι",
                 "ἐνΘήβαις": "πρόειμι", "ενΘηβαις": "πρόειμι"}
    sources = [("glaux_pairs.json", [
        {"form": "ὑπόν", "lemma": "invented-owner"},
        {"form": "ἄλλλως", "lemma": "ἄλλως"},
        {"form": "ἐνΘήβαις", "lemma": "Θῆβαι"},
        {"form": "γράφει", "lemma": "γράφω"},
    ])]
    target = {}
    audit = restore_attested_compound_aliases(
        reference, {}, target, sources, {"ἄλλως", "Θῆβαι", "γράφω"})
    assert not target
    assert len(audit["rejected_source_rows"]) == 3


def test_source_alias_recovery_does_not_cross_case_or_reinfer_a_marked_owner():
    reference = {"μετάποντος": "μετάπειμι", "μεταποντος": "μετάπειμι",
                 "ὑπῆν": "ὕπειμι", "υπην": "ὕπειμι"}
    sources = [("glaux_pairs.json", [
        {"form": "μετάποντος", "lemma": "μεταποντος"},
        {"form": "ὑπῆν", "lemma": "ὑπείμι"},
    ])]
    target = {}
    restore_attested_compound_aliases(
        reference, {}, target, sources, {"Μετάποντος", "ὕπειμι", "ὑπείμι"})
    assert "μεταποντος" not in target
    assert target["υπην"] == "ὑπείμι"


def test_alias_recovery_preserves_same_source_ties_independently_of_row_order():
    reference = {"ἄνοντος": "ἄνειμι", "ανοντος": "ἄνειμι"}
    pairs = [{"form": "ἄνοντος", "lemma": "ἄνω"},
             {"form": "ἄνοντος", "lemma": "ἄνειμι"}]
    for ordered in (pairs, list(reversed(pairs))):
        target = {}
        audit = restore_attested_compound_aliases(
            reference, {}, target, [("ag_pairs.json", ordered)], {"ἄνω", "ἄνειμι"})
        assert "ανοντος" not in target and "ἄνοντος" not in target
        assert audit["same_source_tied_cells"][0]["selected_owner"] is None
        selected = {"ἄνοντος": "ἄνω"}
        audit = restore_attested_compound_aliases(
            reference, {}, selected, [("ag_pairs.json", ordered)], {"ἄνω", "ἄνειμι"})
        assert selected["ανοντος"] == "ἄνω"
        assert audit["same_source_tied_cells"][0]["selected_owner"] == "ἄνω"


def test_explicit_exclusion_blocks_a_colliding_cell_and_its_alias():
    reference = {"ὕπην": "ὕπειμι", "υπην": "ὕπειμι"}
    sources = [("glaux_pairs.json", [{"form": "ὑπῆν", "lemma": "ὕπειμι"}])]
    for excluded in ({"ὑπῆν"}, {"υπην"}):
        target = {}
        restore_attested_compound_aliases(reference, {}, target, sources,
                                         {"ὕπειμι"}, excluded_forms=excluded)
        assert not set(target) & excluded


def test_alias_lexical_heads_require_nonempty_ancient_evidence(tmp_path):
    path = tmp_path / "heads.json"
    path.write_text(json.dumps({"schema_version": 2,
                               "heads": {"ὕπειμι": ["lsj9_headwords"], "ἄνω": []},
                               "vernacular_heads": {"κάθομαι": ["wiktionary_en_el"]}}))
    assert load_alias_lexical_heads(path) == {"ὕπειμι"}


def test_exact_reviewed_cells_preserve_lookup_aliases_and_citation_proof(tmp_path):
    import overlay_lsj
    for name in overlay_lsj.SOURCE_PAIR_NAMES:
        (tmp_path / name).write_text("[]")
    spec = {"lemma": "συμπερίειμι", "method": "source_forms",
            "forms": ["συμπεριόντα"], "citation": "Choricius, Tyrannicida, page 54",
            "evidence": "Exact form independently verified in the printed edition."}
    review = {"schema_version": 1, "paradigms": [spec, {
        "lemma": "συμπερίειμι", "method": "lookup_paradigm"}]}
    (tmp_path / "hunspell_grc_postclassical_review.json").write_text(json.dumps(review))
    pairs, pins = load_alias_source_pairs(tmp_path)
    assert len(pairs[0][1]) == 1
    assert "hunspell_grc_postclassical_review.json" in pins
    reference = {"σύμπεριόντα": "συμπερίειμι", "συμπεριοντα": "συμπερίειμι"}
    target = {}
    audit = restore_attested_compound_aliases(reference, {}, target, pairs,
                                             {"συμπερίειμι"})
    assert target["συμπεριόντα"] == target["συμπεριοντα"] == "συμπερίειμι"
    assert all(row["cells"][0]["readings"][0]["citation"] == spec["citation"]
               for row in audit["recovered"] if row["form"] in
               {"συμπεριόντα", "συμπεριοντα"})


def test_reviewed_archimedes_grave_preserves_acute_and_grave_aliases(tmp_path):
    import overlay_lsj
    source = Path(overlay_lsj.__file__).parent / "data" / "hunspell_grc_postclassical_review.json"
    spec = next(row for row in json.loads(source.read_text())["paradigms"]
                if row["lemma"] == "ἀπόχρη")
    assert spec["forms"] == ["ἀποχρὴ"]
    for name in overlay_lsj.SOURCE_PAIR_NAMES:
        (tmp_path / name).write_text("[]")
    (tmp_path / source.name).write_text(json.dumps({"schema_version": 1,
                                                   "paradigms": [spec]}))
    pairs, _pins = load_alias_source_pairs(tmp_path)
    target = {}
    audit = restore_attested_compound_aliases(
        {"ἀποχρή": "ἀπόχρη", "ἀποχρὴ": "ἀπόχρη"}, {}, target,
        pairs, {"ἀπόχρη"})
    assert target["ἀποχρή"] == target["ἀποχρὴ"] == "ἀπόχρη"
    assert all(row["cells"][0]["readings"][0]["original_form"] == "ἀποχρὴ"
               for row in audit["recovered"] if row["form"] in
               {"ἀποχρή", "ἀποχρὴ"})


def test_overlay_adds_only_expansion_entries_and_preserves_current_values():
    reference_base = {"base": "old-base", "base-conflict": "old"}
    reference = {
        **reference_base,
        "ἀποκλείεται": "historical-lemma",
        "ῥαψάμενοι": "ῥάπτω",
        "δακτύλι": "historical-lemma",
        "κατεβρεχθῶσι(ν": "βρέχω",
        "unmarked": "generated-fallback",
    }
    target = {
        "base": "new-base",
        "δακτύλι": "current-lemma",
        "new": "new-lemma",
    }

    stats = overlay_expansion(
        reference, reference_base, target, excluded_forms={"ἀποκλείεται"}
    )

    assert target == {
        "base": "new-base",
        "δακτύλι": "current-lemma",
        "new": "new-lemma",
        "ῥαψάμενοι": "ῥάπτω",
    }
    assert stats["expansion_entries"] == 5
    assert stats["unmarked_entries"] == 1
    assert stats["editorial_entries"] == 1
    assert stats["excluded_entries"] == 1
    assert stats["added"] == 1
    assert stats["conflicts_preserved"] == 1


@pytest.mark.parametrize("lemma,hyphenated", [
    ("Βακχ-εία", True),            # LSJ9's export divides the head
    ("καθ-ηδύνω", True),           # the Sophocles lexicon divides a compound
    ("Σινα-ῖτις", True),           # an OCR line-break hyphen
    ("προσ-ανα-στίζω", True),
    ("ἐπί-εἰσκομίζω", True),       # a corpus lemma naming the preverb apart
    ("παρά,εἰσ-δυάω", True),
    ("στρέφω—τρέπω", True),        # a dash between two words
    ("ἐμ\u0301-βλῴσκω", True),      # a stray accent before the hyphen
    ("βου\u00adμα\u00adστόω", True),  # soft hyphens from LSJ9's headword list
    ("γναμπτ\u00ad\u00adω", True),
    ("ἐπιθρύξω—ἐω", True),
    ("α-⏑β", False),
    ("Βακχεία", False),
    ("καθηδύνω", False),
    ("ὁτ-", False),                # a hyphen at either end is not inside
    ("-σον", False),
    ("–⏑τατος", False),
    ("", False),
    ("-", False),
])
def test_hyphenated_lemma_has_a_hyphen_between_two_letters(lemma, hyphenated):
    from overlay_lsj import hyphenated_lemma
    assert hyphenated_lemma(lemma) is hyphenated


def test_overlay_never_restores_a_row_under_a_hyphenated_lemma():
    reference = {
        # LSJ: the historical layer has the divided and the joined head
        "Βοιωτ-άρχου": "Βοιωτ-άρχης", "Βοιωτάρχου": "Βοιωτάρχης",
        # Sophocles: a head the generator now reads as καθηδύνω, and one it rejects
        "καθ-ήδυνε": "καθ-ηδύνω", "Σινα-ίτιδος": "Σινα-ῖτις",
        # a spelling without a hyphen under such a lemma
        "ἐπεισεκόμισαν": "ἐπί-εἰσκομίζω",
        "Βακχ-είας": "Βακχ-εία",
        "ὁτέων": "ὁτ-",
    }
    target = {"Βακχ-είας": "current"}
    stats = overlay_expansion(
        reference, {}, target,
        # naming the printed head superseded changes nothing
        superseded_noun_lemmas={"Σινα-ῖτις"},
        withheld_noun_lemmas={"Σινα-ῖτις": ("ιδος",)},
        carried={"Σινα-ίτιδος": "Σινα-ῖτις"}, lexical_heads={"Σινα-ῖτις"})
    assert target == {"Βακχ-είας": "current", "Βοιωτάρχου": "Βοιωτάρχης",
                      "ὁτέων": "ὁτ-"}
    assert stats["hyphenated_lemma_entries"] == 4
    assert stats["already_present"] == 1 and stats["conflicts_preserved"] == 1
    assert stats["superseded_noun_entries"] == 0
    assert stats["withheld_noun_entries_kept"] == 0
    assert stats["added"] == 2


def test_unaccented_key_under_a_hyphenated_lemma_stays_out():
    from overlay_lsj import restore_unmarked_keys
    reference = {"Βοιωτ-άρχου": "Βοιωτ-άρχης", "Βοιωτ-αρχου": "Βοιωτ-άρχης",
                 "ἐπεισεκόμισαν": "ἐπί-εἰσκομίζω",
                 "επεισεκομισαν": "ἐπί-εἰσκομίζω",
                 "Βοιωτάρχου": "Βοιωτάρχης", "Βοιωταρχου": "Βοιωτάρχης"}
    # even a table that still has the marked rows gives the keys no backing
    target = {"Βοιωτ-άρχου": "Βοιωτ-άρχης", "ἐπεισεκόμισαν": "ἐπί-εἰσκομίζω",
              "Βοιωτάρχου": "Βοιωτάρχης"}
    carried = {key: lemma for key, lemma in reference.items()
               if key in {"Βοιωτ-αρχου", "επεισεκομισαν", "Βοιωταρχου"}}
    stats = restore_unmarked_keys(reference, {}, target, carried)
    assert target["Βοιωταρχου"] == "Βοιωτάρχης"
    assert "Βοιωτ-αρχου" not in target and "επεισεκομισαν" not in target
    assert stats["unmarked_candidates"] == 1 and stats["unmarked_restored"] == 1


def test_target_editorial_cleanup_resolves_only_movable_nu():
    target = {
        "κατεβρεχθῶσι(ν": "βρέχω",
        "κατεβρεχθῶσι": "current-lemma",
        "ἀ[ζηχὲς": "ἀζηχής",
        "plain": "plain",
    }

    stats = resolve_target_editorial_entries(target)

    assert target == {
        "κατεβρεχθῶσι": "current-lemma",
        "plain": "plain",
        "κατεβρεχθῶσιν": "βρέχω",
    }
    assert stats == {
        "target_editorial_removed": 2,
        "target_movable_nu_entries": 1,
        "target_movable_nu_added": 1,
        "target_movable_nu_present": 1,
        "target_movable_nu_conflicts": 1,
    }


def test_overlay_does_not_restore_superseded_verb_generation():
    reference_base = {"ἀβακίζομαι": "ἀβακίζομαι"}
    reference = {
        **reference_base,
        "ἀβακιζόομαι": "ἀβακίζομαι",
        "ἀβακιζόμεθα": "ἀβακίζομαι",
        "σύμεφην": "σύμφημι",
        "φῆμι": "φημί",
        "ἀμφιένυμι": "ἀμφιέννυμι",
        "ἀμφιέννυμι": "ἀμφιέννυμι",
        "λύεις": "λύω",
    }
    target = {
        # Current independent evidence and corrected generation always win.
        "ἀβακίζομαι": "ἀβακίζομαι",
        "ἀβακιζόμεθα": "ἀβακίζομαι",
        "ἀμφιέννυμι": "ἀμφιέννυμι",
        "σύμεφην": "independent-current-analysis",
    }
    before = dict(target)
    stats = overlay_expansion(reference, reference_base, target)
    assert target == {**before, "λύεις": "λύω"}
    assert stats["superseded_verb_entries"] == 3
    assert stats["already_present"] == 3


def test_pin_verification_rejects_changed_bytes(tmp_path):
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(b"known bytes")
    pin = {
        "size": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    verify_file(artifact, pin, "fixture")

    artifact.write_bytes(b"changed bytes")
    with pytest.raises(SystemExit, match="mismatch"):
        verify_file(artifact, pin, "fixture")


def test_manifest_pins_reference_base_and_traced_forms():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert len(manifest["reference"]["revision"]) == 40
    assert len(manifest["reference"]["sha256"]) == 64
    assert len(manifest["reference_base"]["sha256"]) == 64
    assert len(manifest["behavior_baseline"]["sha256"]) == 64
    assert len(manifest["carried_baseline"]["revision"]) == 40
    assert len(manifest["carried_baseline"]["sha256"]) == 64
    assert len(manifest["exclusions"]["sha256"]) == 64
    assert manifest["expected_counts"]["expansion_entries"] == 7_581_466
    assert manifest["expected_counts"]["unmarked_entries"] == 3_839_827
    assert manifest["expected_counts"]["editorial_entries"] == 247
    assert manifest["expected_counts"]["excluded_entries"] == 36_839
    assert manifest["artifact_floor"]["grc_source_rows"] == 10_000_000
    assert manifest["sentinels"]["ῥαψάμενοι"] == "ῥάπτω"
    verify_sentinels(manifest["sentinels"], manifest["sentinels"])


def test_superseded_noun_family_keeps_independent_and_current_spellings():
    reference = {"ωνός": "ἀγκών", "ἀγκῶνος": "ἀγκών", "παλαιός": "παλαιός"}
    base = {"ἀγκῶνος": "ἀγκών"}
    target = {"ωνός": "independent-current"}
    stats = overlay_expansion(reference, base, target,
                              superseded_noun_lemmas={"ἀγκών"})
    assert target == {"ωνός": "independent-current", "παλαιός": "παλαιός"}
    assert stats["superseded_noun_entries"] == 0
    target = {}
    stats = overlay_expansion(reference, base, target,
                              superseded_noun_lemmas={"ἀγκών"})
    assert "ωνός" not in target and "ἀγκῶνος" not in target
    assert stats["superseded_noun_entries"] == 1


def test_noun_audit_rejects_different_inputs_even_with_identical_source(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "build"))
    import expand_lsj
    monkeypatch.setattr(expand_lsj, "noun_generation_input_hashes", lambda: {"fixture": "current"})
    path = tmp_path / "audit.json"
    audit = {"schema_version": 1, "generator_sha256": hashlib.sha256(
        (ROOT / "build/expand_lsj.py").read_bytes()).hexdigest(),
        "input_sha256": {"fixture": "old"}, "superseded_lemmas": ["ἀγκών"]}
    path.write_text(json.dumps(audit))
    with pytest.raises(SystemExit, match="noun inputs changed"):
        load_noun_generation_audit(path)
    audit["input_sha256"] = {"fixture": "current"}
    path.write_text(json.dumps(audit))
    assert load_noun_generation_audit(path) == frozenset({"ἀγκών"})


@pytest.mark.skipif(not LOOKUP_DB.exists(), reason="lookup.db not downloaded")
def test_shipped_lookup_contains_recovered_expansion_sentinels():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    connection = sqlite3.connect(f"file:{LOOKUP_DB}?mode=ro", uri=True)
    try:
        grc_rows = connection.execute(
            "SELECT COUNT(*) FROM lookup WHERE src = 'grc'"
        ).fetchone()[0]
        assert grc_rows >= manifest["artifact_floor"]["grc_source_rows"]
        for form, lemma in manifest["sentinels"].items():
            rows = connection.execute(
                "SELECT l.text FROM lookup k "
                "JOIN lemmas l ON l.id = k.lemma_id "
                "WHERE k.form = ? AND k.src = 'grc'",
                (form,),
            ).fetchall()
            assert (lemma,) in rows, f"missing {form} -> {lemma}"
    finally:
        connection.close()


@pytest.mark.skipif(not LOOKUP_DB.exists(), reason="lookup.db not downloaded")
def test_shipped_lookup_contains_no_editorial_sigla():
    connection = sqlite3.connect(f"file:{LOOKUP_DB}?mode=ro", uri=True)
    try:
        count = connection.execute(
            "SELECT COUNT(*) FROM lookup k "
            "JOIN lemmas l ON l.id = k.lemma_id "
            "WHERE instr(k.form, '[') OR instr(k.form, ']') "
            "OR instr(k.form, '(') OR instr(k.form, ')') "
            "OR instr(l.text, '[') OR instr(l.text, ']') "
            "OR instr(l.text, '(') OR instr(l.text, ')')"
        ).fetchone()[0]
        assert count == 0
    finally:
        connection.close()


def test_historical_nonlexical_lemmas_do_not_regenerate_source_notation():
    reference = {
        "έφʼωμαι": "ἐφʼ ᾧ",
        "λεχέτω": "λέγω =⟩ λέχω",
        "οὐδεπέτω": "οὐδέ πω",
        "παρέσται": "παρά,εἰσ-εἰμί",
        "ἀναθαρρρέω": "ἀναθαρρρέω",
        "μέλλλω": "μέλλλω",
        "ἀπορρρέω": "ἀπορρρέω",
        "ὑποδ̣έω": "ὑποδ̣έω",
        "γῶνται": "γῶ",
        "ὕεται": "ὕω",
        "τἀμά": "ἐμός",
    }
    target = {}
    stats = overlay_expansion(reference, {}, target)
    assert target == {"γῶνται": "γῶ", "ὕεται": "ὕω", "τἀμά": "ἐμός"}
    assert stats["nonlexical_lemma_entries"] == 8


def test_historical_nonlexical_guard_preserves_base_and_current_evidence():
    reference = {
        "έφʼωμαι": "ἐφʼ ᾧ",
        "μέλλλω": "μέλλλω",
        "παρέσται": "παρά,εἰσ-εἰμί",
    }
    base = {"μέλλλω": "independent-base-analysis"}
    target = {"έφʼωμαι": "independent-current-analysis"}
    stats = overlay_expansion(reference, base, target)
    assert target == {"έφʼωμαι": "independent-current-analysis"}
    assert stats["already_present"] == 1
    assert stats["nonlexical_lemma_entries"] == 1


def test_overlay_withholds_irregular_compounds_but_preserves_independent_rows():
    reference_base = {"ἐπανῄει": "ἐπάνειμι"}
    reference = {
        **reference_base,
        "ἐπάνηει": "ἐπάνειμι",
        "ἄνεστί": "ἄνειμι",
        "πάρεισι": "πάρειμι",
        "σύνηδει": "σύνοιδα",
        "ᾔει": "εἶμι",
    }
    target = {"πάρεισι": "independent-source", "ἄνειμι": "ἄνειμι"}
    before = dict(target)
    stats = overlay_expansion(reference, reference_base, target)
    assert target == {**before, "ᾔει": "εἶμι"}
    assert stats["superseded_verb_entries"] == 3
    assert stats["already_present"] == 1


def test_marked_citation_owner_does_not_change_accent_to_another_head():
    target = {"ἑνῶ": "ἑνῶ", "ενώ": "ἑνῶ"}
    audit = restore_attested_compound_aliases(
        {"ἑνῶ": "ἔνειμι", "ενώ": "ἔνειμι"}, {}, target,
        [("ag_pairs.json", [{"form": "ἑνῶ", "lemma": "ἑνῶ",
                             "pos": "verb", "tags": []}]),
         ("glaux_pairs.json", [{"form": "ἑνῶ", "lemma": "ἑνόω",
                                "pos": "verb", "tags": ["present"]}])],
        {"ἕνω", "ἑνόω", "ἔνειμι"})
    assert target["ἑνῶ"] == target["ενώ"] == "ἑνῶ"
    assert "ἕνω" not in target.values()
    assert not audit["canonicalized"]
    rejected = [row for row in audit["rejected_source_rows"]
                if row["lemma"] == "ἑνῶ"]
    assert rejected and all(row["reason"] == "unsupported_owner"
                            for row in rejected)


def test_missing_tone_in_citation_owner_can_use_unique_positive_head():
    target = {}
    restore_attested_compound_aliases(
        {"υπην": "ὕπειμι"}, {}, target,
        [("glaux_pairs.json", [{"form": "ὑπῆν", "lemma": "ὑπειμι"}])],
        {"ὕπειμι"})
    assert target["ὑπῆν"] == target["υπην"] == "ὕπειμι"


@pytest.mark.parametrize("form,raw_owner,head", [
    ("συνεισιέναι", "σύνείσειμι", "συνείσειμι"),
    ("ξύνιον", "συν\u0301ιω", "συνίω"),
])
def test_impossible_citation_accents_use_unique_same_marked_head(form, raw_owner, head):
    target = {form: raw_owner}
    audit = restore_attested_compound_aliases(
        {form: "σύνειμι"}, {}, target,
        [("glaux_pairs.json", [{"form": form, "lemma": raw_owner}])],
        {head, "σύνειμι"})
    assert target[form] == head
    assert audit["canonicalized"]
    reading = audit["canonicalized"][0]["cells"][0]["readings"][0]
    assert reading["form"] == form
    assert reading["lemma"] == raw_owner


def test_impossible_citation_accents_do_not_choose_between_positive_heads():
    target = {"ξύνιον": "συν\u0301ιω"}
    audit = restore_attested_compound_aliases(
        {"ξύνιον": "σύνειμι"}, {}, target,
        [("glaux_pairs.json", [{"form": "ξύνιον", "lemma": "συν\u0301ιω"}])],
        {"συνίω", "σύνιω", "σύνειμι"})
    assert target["ξύνιον"] == "συν\u0301ιω"
    assert not audit["canonicalized"]


def test_impossible_citation_accents_do_not_replace_breathing():
    target = {"ὑπῆν": "ὑπ\u0301ειμι"}
    audit = restore_attested_compound_aliases(
        {"ὑπῆν": "ὕπειμι"}, {}, target,
        [("glaux_pairs.json", [{"form": "ὑπῆν", "lemma": "ὑπ\u0301ειμι"}])],
        {"ὔπειμι"})
    assert target == {"ὑπῆν": "ὑπ\u0301ειμι"}
    assert not audit["canonicalized"]


def test_valid_enclitic_citation_accent_is_not_treated_as_malformed():
    target = {"πρόσειμί": "πρόσειμί"}
    audit = restore_attested_compound_aliases(
        {"πρόσειμί": "πρόσειμι"}, {}, target,
        [("glaux_pairs.json", [{"form": "πρόσειμί", "lemma": "πρόσειμί"}])],
        {"πρόσειμι"})
    assert target["πρόσειμί"] == "πρόσειμί"
    assert not audit["canonicalized"]


def test_baseline_conflicts_release_only_what_the_published_table_carried():
    from overlay_lsj import carried_baseline_conflicts
    reference = {"ἐκπλαγῇ": "ἐκπλαγής", "ἀμέτρητον": "ἀμέτρητος",
                 "Δημήτρειον": "Δημήτρειος", "Γὰρ": "Γάρ", "Ἤ": "Ἤ"}
    conflicts = {"ἐκπλαγῇ": "ἐκπλάσσω", "ἀμέτρητον": "ἀμετρητος",
                 "Δημήτρειον": "Δημήτρειοι", "Γὰρ": "γάρ", "Ἤ": "ἤ",
                 "ἄλλο": "ἄλλος"}                # not an expansion form
    carried = {"ἀμέτρητον": "ἀμέτρητος",        # carried the expansion reading
               "Δημήτρειον": "Δημήτρειος",
               "ἐκπλαγῇ": "ἐκπλάσσω",           # carried the recorded reading
               "ἄλλο": "ἄλλος"}                 # Γὰρ and Ἤ: no key at all
    assert carried_baseline_conflicts(conflicts, reference, carried) == {
        "ἀμέτρητον", "Δημήτρειον"}
    assert carried_baseline_conflicts(conflicts, reference, {}) == frozenset()


def test_released_conflicts_take_the_expansion_and_stay_counted():
    reference = {"ἀμέτρητον": "ἀμέτρητος", "ἐκπλαγῇ": "ἐκπλαγής"}
    target = {}
    stats = overlay_expansion(reference, {}, target,
                              excluded_forms={"ἀμέτρητον", "ἐκπλαγῇ"},
                              released_forms={"ἀμέτρητον"})
    assert target == {"ἀμέτρητον": "ἀμέτρητος"}
    assert stats["excluded_entries"] == 2
    assert stats["released_conflict_entries"] == 1


def test_unaccented_keys_return_where_every_marked_form_agrees():
    from overlay_lsj import restore_unmarked_keys
    reference = {"ἀθήλυντος": "ἀθήλυντος", "αθηλυντος": "ἀθήλυντος",
                 "παμπόλλων": "πάμπολλος", "παμπολλων": "πάμπολλος",
                 "νοερῶν": "νοερόω", "νοερων": "νοερόω",
                 "ορφνινος": "ὄρφνινος",
                 "τοσ": "τόσος"}
    target = {"ἀθήλυντος": "ἀθήλυντος",
              "παμπόλλων": "πάμπολλος", "Παμπόλλων": "πάμπολλος",
              "νοερῶν": "νοερός"}                 # a newer marked reading
    carried = {key: lemma for key, lemma in reference.items()
               if key in {"αθηλυντος", "παμπολλων", "νοερων", "ορφνινος", "τοσ"}}
    stats = restore_unmarked_keys(reference, {}, target, carried,
                                  excluded_forms={"τοσ"})
    assert target["αθηλυντος"] == "ἀθήλυντος"
    assert target["παμπολλων"] == "πάμπολλος"
    assert "νοερων" not in target and "ορφνινος" not in target
    assert "τοσ" not in target
    assert stats == {"unmarked_candidates": 4, "unmarked_restored": 2,
                     "unmarked_disputed": 1, "unmarked_unbacked": 1,
                     "unmarked_lemma_respelled": 0,
                     "unmarked_oxytone_cell_restored": 0,
                     "unmarked_corrected_head_restored": 0,
                     "unmarked_nonhead_readings_ignored": 0,
                     "unmarked_grave_twins_skipped": 0}


def test_unaccented_keys_the_published_table_lacked_stay_out():
    from overlay_lsj import restore_unmarked_keys
    reference = {"ἀθήλυντος": "ἀθήλυντος", "αθηλυντος": "ἀθήλυντος",
                 "ἀνώτερος": "ἀνώτερος", "ανωτερω": "ἀνώτερος",
                 "ξυνίων": "ξυνίημι", "ξυνιων": "ξυνίημι"}
    target = {"ἀθήλυντος": "ἀθήλυντος", "ἀνωτέρω": "ἀνώτερος",
              "ξυνίων": "ξυνίημι"}
    carried = {"αθηλυντος": "ἀθήλυντος",          # carried: comes back
               "ξυνιων": "συνίημι"}                # carried another reading
    stats = restore_unmarked_keys(reference, {}, target, carried)
    assert target["αθηλυντος"] == "ἀθήλυντος"
    assert "ανωτερω" not in target and "ξυνιων" not in target
    assert stats["unmarked_candidates"] == 1 and stats["unmarked_restored"] == 1
    assert restore_unmarked_keys(reference, {}, dict(target), {})[
        "unmarked_candidates"] == 0


def test_restored_unaccented_keys_are_the_table_tail_and_digested():
    import hashlib
    from overlay_lsj import restore_unmarked_keys, unmarked_restore_record
    reference = {"ἀθήλυντος": "ἀθήλυντος", "αθηλυντος": "ἀθήλυντος",
                 "παμπόλλων": "πάμπολλος", "παμπολλων": "πάμπολλος"}
    target = {"ἀθήλυντος": "ἀθήλυντος", "παμπόλλων": "πάμπολλος"}
    rows = []
    carried = {"αθηλυντος": "ἀθήλυντος", "παμπολλων": "πάμπολλος"}
    restore_unmarked_keys(reference, {}, target, carried, restored_rows=rows)
    assert rows == [("αθηλυντος", "ἀθήλυντος"), ("παμπολλων", "πάμπολλος")]
    assert list(target.items())[-len(rows):] == rows
    record = unmarked_restore_record(rows)
    text = "αθηλυντος\tἀθήλυντος\nπαμπολλων\tπάμπολλος\n"
    assert record == {"count": 2,
                      "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    assert unmarked_restore_record([]) == {
        "count": 0, "sha256": hashlib.sha256(b"").hexdigest()}


def test_released_conflict_with_editorial_notation_stays_excluded():
    # The pinned counts file such a form under the exclusions; releasing it
    # would count it as notation too and add nothing.
    reference = {"ἀμέτρητ(ον)": "ἀμέτρητος", "ἀμέτρητον": "ἀμέτρητος"}
    target = {}
    stats = overlay_expansion(reference, {}, target,
                              excluded_forms={"ἀμέτρητ(ον)", "ἀμέτρητον"},
                              released_forms={"ἀμέτρητ(ον)", "ἀμέτρητον"})
    assert target == {"ἀμέτρητον": "ἀμέτρητος"}
    assert stats["excluded_entries"] == 2
    assert stats["released_conflict_entries"] == 1
    assert stats["editorial_entries"] == 0


def test_noun_without_a_fresh_paradigm_keeps_its_published_keys():
    # λεκτόν was superseded but its generation withheld; ἀγκών got a fresh
    # paradigm, so its old keys stay out.
    reference = {"λεκτοῦ": "λεκτόν", "λεκτά": "λεκτόν", "λεκτοῖς": "λεκτόν",
                 "ἀγκῶνος": "ἀγκών", "παλαιός": "παλαιός"}
    carried = {"λεκτοῦ": "λεκτόν", "λεκτά": "λεκτός",    # another reading
               "ἀγκῶνος": "ἀγκών"}                       # λεκτοῖς: no key
    target = {}
    stats = overlay_expansion(
        reference, {}, target, superseded_noun_lemmas={"λεκτόν", "ἀγκών"},
        carried=carried, withheld_noun_lemmas={"λεκτόν": ("οῦ",)},
        lexical_heads={"λεκτόν", "ἀγκών"})
    assert target == {"λεκτοῦ": "λεκτόν", "παλαιός": "παλαιός"}
    assert stats["withheld_noun_entries_kept"] == 1
    assert stats["withheld_noun_lemmas_kept"] == 1
    assert stats["superseded_noun_entries"] == 3
    assert stats["added"] == 1


def test_withheld_noun_keys_need_a_positive_head_and_a_possible_spelling():
    reference = {"κλῃσίν": "κλείς", "κλει-δὸς": "κλείς", "κλεῖδὸς": "κλείς",
                 "Ἀρτέμονος": "᾿Αρτέμων",        # the head list has Ἀρτέμων
                 "Καλαβρῶν": "Καλαβρὰ"}          # a head in grave spelling
    carried = dict(reference)
    target = {}
    stats = overlay_expansion(
        reference, {}, target,
        superseded_noun_lemmas={"κλείς", "᾿Αρτέμων", "Καλαβρὰ"},
        carried=carried,
        withheld_noun_lemmas={"κλείς": ("κλειδός",), "᾿Αρτέμων": (),
                              "Καλαβρὰ": ()},
        lexical_heads={"κλείς", "Ἀρτέμων", "Καλαβρὰ"})
    assert target == {"κλῃσίν": "κλείς"}
    assert stats["withheld_noun_entries_kept"] == 1
    assert stats["withheld_noun_lemmas_kept"] == 1
    assert stats["superseded_noun_entries"] == 4


def test_wrongly_declined_withheld_noun_keeps_only_its_citation_spelling():
    # The old generator declined Θεοπρεπότης, -τος by its nominative. The
    # printed ending is not among the keys, so only the citation stays.
    # κλείς has its printed genitive among them and keeps the paradigm.
    reference = {"Θεοπρεπότης": "Θεοπρεπότης", "Θεοπρεπότὴς": "Θεοπρεπότης",
                 "Θεοπρεπότου": "Θεοπρεπότης", "Θεοπρεπότῃ": "Θεοπρεπότης",
                 "κλείς": "κλείς", "κλειδός": "κλείς", "κλειδί": "κλείς",
                 "ἀπεικός": "ἀπεικός", "ἀπεικὸς": "ἀπεικός",
                 "ἀπεικοῦ": "ἀπεικός"}
    target = {}
    stats = overlay_expansion(
        reference, {}, target,
        superseded_noun_lemmas={"Θεοπρεπότης", "κλείς", "ἀπεικός"},
        carried=dict(reference),
        withheld_noun_lemmas={"Θεοπρεπότης": ("τος",), "κλείς": ("κλειδός",),
                              "ἀπεικός": ("ότος",)},
        lexical_heads={"Θεοπρεπότης", "κλείς", "ἀπεικός"})
    assert target == {"Θεοπρεπότης": "Θεοπρεπότης",
                      "κλείς": "κλείς", "κλειδός": "κλείς", "κλειδί": "κλείς",
                      "ἀπεικός": "ἀπεικός", "ἀπεικὸς": "ἀπεικός"}
    assert stats["withheld_noun_citation_only_lemmas"] == 2
    assert stats["withheld_noun_lemmas_kept"] == 3
    assert stats["withheld_noun_entries_kept"] == 6
    assert stats["superseded_noun_entries"] == 4


def test_withheld_noun_lemmas_are_superseded_records_without_a_genitive(tmp_path):
    from overlay_lsj import load_withheld_noun_lemmas
    lsj = tmp_path / "lsj.json"
    lsj.write_text(json.dumps({"resolutions": {
        "λεκτόν": {"supplied": "οῦ", "resolved": "",
                   "reason": "unsupported-genitive-abbreviation"},
        "σωματεῖον": {"supplied": "", "resolved": "",
                      "reason": "missing-genitive"},
        "ἀγκών": {"supplied": "ῶνος", "resolved": "ἀγκῶνος",
                  "reason": "completed-genitive"},
        "κλυσμός": {"supplied": "οὔ", "resolved": "",
                    "reason": "unsupported-genitive-abbreviation"},
        "Α α": {"supplied": "", "resolved": "",
                "reason": "nonlexical-noun-headword"},
        "ἄλλος": {"supplied": "ου", "resolved": "",
                  "reason": "missing-genitive"}}}), encoding="utf-8")
    sophocles = tmp_path / "sophocles.json"
    sophocles.write_text(json.dumps({"resolutions": {
        "λεκτόν": {"supplied": "τό", "resolved": "",
                   "reason": "unsupported-genitive-abbreviation"},
        "κλυσμός": {"supplied": "", "resolved": "κλυσμου",
                    "reason": "completed-genitive"}}}), encoding="utf-8")
    superseded = {"λεκτόν", "σωματεῖον", "ἀγκών", "κλυσμός", "Α α"}
    assert load_withheld_noun_lemmas((lsj, sophocles), superseded) == {
        "λεκτόν": ("οῦ", "τό"), "σωματεῖον": ()}


def test_published_reading_stays_over_an_accentless_copy_of_its_head():
    from overlay_lsj import keep_published_over_unaccented_heads
    target = {"κληζεται": "κληζω", "κληζω": "κληζω",
              "κληθεις": "κληζω",            # the published table read καλέω
              "ἔξηλθεν": "ἐξέρχομαι",        # a marked fresh lemma stays
              "φορω": "φορω"}                # published head is no head here
    carried = {"κληζεται": "κλῄζω", "κληθεις": "καλέω",
               "ἔξηλθεν": "ἐξερύω", "φορω": "φορῶ"}
    kept = keep_published_over_unaccented_heads(
        target, carried, {"κλῄζω", "καλέω", "ἐξερύω"})
    assert kept == 1
    assert target == {"κληζεται": "κλῄζω", "κληζω": "κληζω",
                      "κληθεις": "κληζω", "ἔξηλθεν": "ἐξέρχομαι",
                      "φορω": "φορω"}
    assert keep_published_over_unaccented_heads(dict(target), {}, {"κλῄζω"}) == 0


def test_oxytone_cells_are_the_regular_masculine_and_neuter_endings():
    from overlay_lsj import oxytone_os_cells
    cells = oxytone_os_cells("θελητός")
    assert {"θελητός", "θελητὸς", "θελητοῦ", "θελητῷ", "θελητόν", "θελητὸν",
            "θελητέ", "θελητώ", "θελητοῖν", "θελητοί", "θελητῶν",
            "θελητοῖς", "θελητούς", "θελητά", "θελητὰ",
            "θελητος", "θελητου", "θελητων", "θελητα"} <= cells
    assert not {"θελητή", "θελητῆς", "θελητάς", "θελήτου", "θελητοι̃ς"} & cells
    assert "Ἀγαρηνῶν" in oxytone_os_cells("Ἀγαρηνός")
    # Only an oxytone in -ός with no other accent has such cells.
    assert oxytone_os_cells("ἄνθρωπος") == frozenset()
    assert oxytone_os_cells("θελητόν") == frozenset()
    assert oxytone_os_cells("σοφῶς") == frozenset()


def test_sophocles_only_head_is_not_trusted_beside_its_twin():
    from overlay_lsj import confirmed_head, ocr_twin_head
    evidence = {"Θερινός": ("sophocles/0550.xml",),
                "θερινός": ("lsj9_headwords",),
                "Ἱωνικός": ("sophocles/0600.xml",),
                "Ἰωνικός": ("lsj9_headwords", "sophocles/0600.xml"),
                "Κρητικός": ("pd_headwords", "sophocles/0650.xml"),
                "κρητικός": ("lsj9_headwords",),
                "Σοδομηνός": ("sophocles/1000.xml",)}
    assert ocr_twin_head("Θερινός", evidence)
    assert ocr_twin_head("Ἱωνικός", evidence)
    assert not ocr_twin_head("Ἰωνικός", evidence)
    assert not ocr_twin_head("Κρητικός", evidence)      # another list confirms
    assert not ocr_twin_head("Σοδομηνός", evidence)     # no twin
    assert not ocr_twin_head("θερινός", evidence)
    assert not ocr_twin_head("ἄγνωστος", evidence)      # no head at all
    assert confirmed_head("Κρητικός", evidence)
    assert not confirmed_head("Σοδομηνός", evidence)
    assert not confirmed_head("ἄγνωστος", evidence)


def test_breathed_spelling_adds_the_correction_s_breathing_to_a_form():
    from overlay_lsj import breathed_spelling
    assert breathed_spelling("Ρωμαϊκόν", "Ρωμαϊκός", "Ῥωμαϊκός") == "Ῥωμαϊκόν"
    assert breathed_spelling("ολικοῦ", "ολικός", "ὁλικός") == "ὁλικοῦ"
    assert breathed_spelling("ουρανοφόρους", "ουρανοφόρος",
                             "οὐρανοφόρος") == "οὐρανοφόρους"
    assert breathed_spelling("ονειρατευώμεθα", "ονειρατεύομαι",
                             "ὀνειρατεύομαι") == "ὀνειρατευώμεθα"
    assert breathed_spelling("Αδάμου", "Άδαμος", "Ἄδαμος") == "Ἀδάμου"
    # An augmented or already breathed form has no such spelling, nor has
    # a correction that is more than one added breathing.
    assert breathed_spelling("ωνειρατευόμην", "ονειρατεύομαι",
                             "ὀνειρατεύομαι") is None
    assert breathed_spelling("ὁλικοῦ", "ολικός", "ὁλικός") is None
    assert breathed_spelling("Ρωμαϊκόν", "Ρωμαϊκός", "Ῥωμαικός") is None
    assert breathed_spelling("Ρωμαϊκόν", "Ρωμαϊκός", "Ῥωμαϊκὸς") is None


def test_published_oxytone_cells_return_under_a_trusted_self_mapped_head():
    from overlay_lsj import carry_published_cells
    evidence = {"ἀρχοντικός": ("lsj9_headwords",),
                "θελητός": ("lsj9_headwords",),
                "Ἀντιοχικός": ("lsj9_headwords",),
                "Θερινός": ("sophocles/0550.xml",),
                "θερινός": ("lsj9_headwords",),
                "νοερός": ("lsj9_headwords",),
                "λογισμός": ("lsj9_headwords",),
                "σοφός": ("lsj9_headwords",)}
    carried = {
        "ἀρχοντικόν": "ἀρχοντικός", "ἀρχοντικὸν": "ἀρχοντικός",
        "αρχοντικον": "ἀρχοντικός",
        "ἀρχοντικοῦ": "ἀρχοντικός",          # present with this reading
        "θελητόν": "θελητός",                # displaced by another reading
        "θελητοῦ": "θελητός",
        "θελητής": "θελητός",                # not a masculine/neuter cell
        "Ἀντιοχικῶν": "᾿Αντιοχικός",          # spacing breathing in the lemma
        "Θερινόν": "Θερινός",                # OCR twin of θερινός
        "νοερῶν": "νοερός",                  # head not self-mapped
        "λογισμὸν": "λογισμός",              # superseded noun
        "σοφοῦ": "σοφός", "σοφόν": "σοφός",  # σοφόν: pinned exclusion
        "ἀγνώστου": "ἀγνωστός",              # no positive head
    }
    target = {"ἀρχοντικός": "ἀρχοντικός", "ἀρχοντικοῦ": "ἀρχοντικός",
              "θελητός": "θελητός", "θελητόν": "θελητόν",
              "Ἀντιοχικός": "Ἀντιοχικός", "Θερινός": "Θερινός",
              "νοερός": "νοερόω", "λογισμός": "λογισμός", "σοφός": "σοφός"}
    before = dict(target)
    stats, unmarked = carry_published_cells(
        target, carried, evidence, {}, excluded_forms={"σοφόν"},
        superseded_noun_lemmas={"λογισμός"})
    added = {form: lemma for form, lemma in target.items() if form not in before}
    assert added == {"ἀρχοντικόν": "ἀρχοντικός", "ἀρχοντικὸν": "ἀρχοντικός",
                     "θελητοῦ": "θελητός", "Ἀντιοχικῶν": "Ἀντιοχικός",
                     "σοφοῦ": "σοφός"}
    assert {form: target[form] for form in before} == before
    assert unmarked == {"αρχοντικον": ("ἀρχοντικός", "oxytone_cell")}
    assert stats == {"oxytone_cell_entries": 5, "oxytone_cell_lemmas": 4,
                     "corrected_head_entries": 0, "corrected_head_lemmas": 0}


def test_published_keys_move_to_a_breathing_corrected_head():
    from overlay_lsj import carry_published_cells
    corrections = {"Ρωμαϊκός": "Ῥωμαϊκός", "ολικός": "ὁλικός",
                   "ονειρατεύομαι": "ὀνειρατεύομαι"}
    carried = {"Ρωμαϊκόν": "Ρωμαϊκός", "Ρωμαικον": "Ρωμαϊκός",
               "Ρωμαϊκῆς": "Ρωμαϊκός",           # no fresh breathed cell
               "ολικοῦ": "ολικός",               # breathed cell reads another
               "ονειρατευώμεθα": "ονειρατεύομαι",
               "ωνειρατευόμην": "ονειρατεύομαι",  # no such spelling
               "Ρωμαϊκός": "Ρωμαϊκός"}           # present: left as it is
    target = {"Ῥωμαϊκόν": "Ῥωμαϊκός", "ὁλικοῦ": "ὁλικόω",
              "ὀνειρατευώμεθα": "ὀνειρατεύομαι", "Ρωμαϊκός": "Ρωμαϊκός"}
    before = dict(target)
    stats, unmarked = carry_published_cells(target, carried, {}, corrections)
    added = {form: lemma for form, lemma in target.items() if form not in before}
    assert added == {"Ρωμαϊκόν": "Ῥωμαϊκός",
                     "ονειρατευώμεθα": "ὀνειρατεύομαι"}
    assert {form: target[form] for form in before} == before
    assert unmarked == {"Ρωμαικον": ("Ῥωμαϊκός", "corrected_head")}
    assert stats == {"oxytone_cell_entries": 0, "oxytone_cell_lemmas": 0,
                     "corrected_head_entries": 2, "corrected_head_lemmas": 2}


def test_carried_head_records_hold_evidence_and_corrections(tmp_path):
    from overlay_lsj import load_carried_head_records
    path = tmp_path / "heads.json"
    path.write_text(json.dumps({
        "schema_version": 2,
        "heads": {"Ῥωμαϊκός": ["lsj9_headwords", "sophocles/0950.xml"],
                  "Θερινός": ["sophocles/0550.xml"], "ἄκυρος": []},
        "sophocles_breathing": {"schema_version": 1, "corrected": {
            "Ρωμαϊκός": {"head": "Ῥωμαϊκός", "rule": "same_head"}}}}),
        encoding="utf-8")
    evidence, corrections = load_carried_head_records(path)
    assert evidence == {"Ῥωμαϊκός": ("lsj9_headwords", "sophocles/0950.xml"),
                        "Θερινός": ("sophocles/0550.xml",)}
    assert corrections == {"Ρωμαϊκός": "Ῥωμαϊκός"}
    path.write_text(json.dumps({"schema_version": 2,
                                "heads": {"ἄκυρος": ["lsj9_headwords"]}}))
    assert load_carried_head_records(path) == (
        {"ἄκυρος": ("lsj9_headwords",)}, {})
    path.write_text("{}")
    with pytest.raises(SystemExit, match="invalid positive lexical-head"):
        load_carried_head_records(path)


def test_unaccented_key_compares_lemmas_with_the_breathing_normalised():
    from overlay_lsj import restore_unmarked_keys
    # The reference spells the lemma with a spacing breathing; the fresh
    # marked form reads the precomposed spelling.
    reference = {"Αγαρηνων": "᾿Αγαρηνός", "Αδριατικων": "᾿Αδριατικός",
                 "αβαφη": "αβαφής", "ομοπιστοις": "ομόπιστος"}
    target = {"Ἀγαρηνῶν": "Ἀγαρηνός", "\u0313Αγαρηνῶν": "᾿Αγαρηνός",
              "\u0313Αδριατικῶν": "᾿Αδριατικός",   # only the old spelling
              "αβαφή": "αβαφής", "ἀβαφῆ": "ἀβαφής",
              "ομοπίστοις": "ομόπιστος"}
    rows = []
    stats = restore_unmarked_keys(
        reference, {}, target, dict(reference), restored_rows=rows,
        breathing_corrections={"αβαφής": "ἀβαφής", "ομόπιστος": "ὁμόπιστος"})
    assert rows == [("Αγαρηνων", "Ἀγαρηνός"), ("Αδριατικων", "᾿Αδριατικός"),
                    ("αβαφη", "ἀβαφής"), ("ομοπιστοις", "ομόπιστος")]
    assert stats["unmarked_restored"] == 4
    assert stats["unmarked_lemma_respelled"] == 2
    # Without the corrections the uncorrected and corrected heads differ.
    plain = {key: lemma for key, lemma in target.items() if key not in reference}
    stats = restore_unmarked_keys(reference, {}, plain, dict(reference))
    assert "αβαφη" not in plain and plain["Αγαρηνων"] == "Ἀγαρηνός"
    assert stats["unmarked_disputed"] == 1


def test_unaccented_key_ignores_a_reading_that_is_no_lexical_head():
    from overlay_lsj import restore_unmarked_keys
    reference = {"απεπνιγη": "ἀποπνίγω", "ιστρικος": "ἱστρικός",
                 "διαστατου": "διαστατός", "ασμοδαιε": "ἀσμοδαιός"}
    target = {"ἀπεπνίγη": "ἀποπνίγω", "ἀπεπνιγῆ": "ἀποπενιγ͂υμι",
              "ἱστρικός": "ἱστρικός", "ἰστρικός": "ἰστρικός",
              "διαστατοῦ": "διαστατός", "διαστάτου": "διαστατόω",
              "ἀσμοδαιέ": "ἀσμοδαιός", "ἀσμοδαῖε": "ἀσμοδαῖος"}
    evidence = {"ἀποπνίγω": ("lsj9_headwords",),
                "ἱστρικός": ("sophocles/0600.xml",),     # unconfirmed head
                "διαστατός": ("lsj9_headwords",),
                "διαστατόω": ("lsj9_headwords",)}        # a head disputes
    carried = dict(reference)
    with_heads = dict(target)
    stats = restore_unmarked_keys(reference, {}, with_heads, carried,
                                  head_evidence=evidence)
    assert {key: with_heads[key] for key in with_heads if key not in target} == {
        "απεπνιγη": "ἀποπνίγω"}
    assert stats["unmarked_restored"] == 1
    assert stats["unmarked_nonhead_readings_ignored"] == 1
    assert stats["unmarked_disputed"] == 3
    without = dict(target)
    stats = restore_unmarked_keys(reference, {}, without, carried)
    assert without == target and stats["unmarked_disputed"] == 4


def test_unaccented_key_takes_the_acute_form_over_its_grave_twin():
    from overlay_lsj import restore_unmarked_keys
    reference = {"ευθης": "εὐθής", "νερον": "νερόν", "καλον": "καλός"}
    target = {"εὐθής": "εὐθής", "εὐθὴς": "εὐθύς",
              "νερὸν": "νύς",                    # no acute twin: it counts
              "καλόν": "καλός", "καλὸν": "καλός", "κᾶλον": "κᾶλον"}
    stats = restore_unmarked_keys(reference, {}, target, dict(reference))
    assert target["ευθης"] == "εὐθής"
    assert "νερον" not in target and "καλον" not in target
    assert stats["unmarked_restored"] == 1
    assert stats["unmarked_grave_twins_skipped"] == 2
    assert stats["unmarked_disputed"] == 2


def test_carried_cells_return_only_with_their_own_lemma_spelling():
    from overlay_lsj import restore_unmarked_keys
    reference = {"αθηλυντος": "ἀθήλυντος"}
    target = {"ἀθήλυντος": "ἀθήλυντος",
              "ἀρχοντικόν": "ἀρχοντικός", "ἀρχοντικὸν": "ἀρχοντικός",
              "Ῥωμαϊκόν": "Ῥωμαϊκός", "Ρωμαϊκόν": "Ῥωμαϊκός",
              "ολικοῦ": "ολικός",                 # only the uncorrected head
              "θελητόν": "θελητόν"}
    cells = {"αρχοντικον": ("ἀρχοντικός", "oxytone_cell"),
             "Ρωμαικον": ("Ῥωμαϊκός", "corrected_head"),
             "ολικου": ("ὁλικός", "corrected_head"),
             "θελητον": ("θελητός", "oxytone_cell"),
             "αθηλυντος": ("ἀθηλυντός", "oxytone_cell"),   # the reference wins
             "σοφου": ("σοφός", "oxytone_cell")}           # pinned exclusion
    rows = []
    stats = restore_unmarked_keys(
        reference, {}, target, {"αθηλυντος": "ἀθήλυντος"},
        excluded_forms={"σοφου"}, restored_rows=rows,
        breathing_corrections={"ολικός": "ὁλικός"}, carried_cells=cells)
    assert rows == [("αθηλυντος", "ἀθήλυντος"), ("αρχοντικον", "ἀρχοντικός"),
                    ("Ρωμαικον", "Ῥωμαϊκός")]
    assert list(target.items())[-3:] == rows
    assert stats["unmarked_candidates"] == 5
    assert stats["unmarked_restored"] == 3
    assert stats["unmarked_oxytone_cell_restored"] == 1
    assert stats["unmarked_corrected_head_restored"] == 1
    assert stats["unmarked_disputed"] == 1 and stats["unmarked_unbacked"] == 1
