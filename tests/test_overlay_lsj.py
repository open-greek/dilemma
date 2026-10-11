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


def test_overlay_adds_only_expansion_entries_and_preserves_current_values():
    reference_base = {"base": "old-base", "base-conflict": "old"}
    reference = {
        **reference_base,
        "ἀποκλείεται": "historical-lemma",
        "ῥαψάμενοι": "new-historical-lemma",
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
        "ῥαψάμενοι": "new-historical-lemma",
    }
    assert stats["expansion_entries"] == 5
    assert stats["unmarked_entries"] == 1
    assert stats["editorial_entries"] == 1
    assert stats["excluded_entries"] == 1
    assert stats["added"] == 1
    assert stats["conflicts_preserved"] == 1


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
