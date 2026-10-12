#!/usr/bin/env python3
"""Restore the pinned historical LSJ/Sophocles expansion layer.

``build_data.py`` regenerates ``ag_lookup.json`` from Wiktionary and therefore
removes the expensive LSJ/Sophocles paradigm expansion. The expansion is
recoverable from a known-good historical lookup, but the whole historical
lookup must not be overlaid blindly: its base mappings may have been corrected
since it was built.

The historical expansion derives ``reference - reference_base`` and adds
missing marked keys, preserving existing target mappings. Separately, source
evidence can retain attested cells and lookup aliases in the bounded withheld
irregular-compound scope; it never reconstructs an approximate paradigm. The
reference and its exact base are byte-pinned in ``data/ag_expansion_reference.json``.

Usage:
    python overlay_lsj.py
    python overlay_lsj.py --kaikki kaikki
    python overlay_lsj.py --dry-run
    python overlay_lsj.py --reference /path/to/pinned/ag_lookup.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path

from dilemma.form_sanitize import (
    has_editorial_sigla, nonlexical_generated_lemma, resolve_editorial_form,
    sanitize_form, canonicalize_final_elision,
)
from dilemma import grave_to_acute, to_monotonic


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "data" / "ag_expansion_reference.json"
DEFAULT_TARGET = ROOT / "data" / "ag_lookup.json"
DEFAULT_NOUN_AUDIT = ROOT / "data" / "ag_noun_generation_audit.json"
DEFAULT_SOPHOCLES_NOUN_AUDIT = ROOT / "data" / "ag_sophocles_noun_generation_audit.json"
DEFAULT_LEXICAL_HEADS = ROOT / "data" / "grc_lexical_heads.json"
# The same source order as the lookup builder, with the fresh merged
# Wiktionary pairs first. Annotation alternatives are retained in the audit.
SOURCE_PAIR_NAMES = (
    "ag_pairs.json", "perseus_pairs.json", "glaux_pairs.json",
    "diorisis_pairs.json", "nt_pairs.json", "ptnk_pairs.json",
    "tagnt_pairs.json", "pedalion_pairs.json",
)
# Paths remain relative to the user-supplied dump directory in portable audits.
CLOSED_CLASS_DUMPS = {
    "wiktionary_en_grc": ("en", "en-el/kaikki.org-dictionary-AncientGreek.jsonl"),
    "wiktionary_el_grc": ("el", "el/kaikki.org-dictionary-AncientGreek.jsonl"),
}
_CHUNK = 1 << 20
_SPACING_MARKS = frozenset({0x1FBD, 0x1FBF, 0x1FFE})


def has_orthographic_mark(form: str) -> bool:
    """Whether a historical key is a marked Greek surface spelling.

    The expansion also stores a fully accent-stripped key for every generated
    form. Those keys are lookup conveniences rather than spellings and can
    shadow newer analyses that share the same bare skeleton, so recovery is
    deliberately limited to marked surface forms.
    """
    return any(
        unicodedata.category(char) == "Mn" or ord(char) in _SPACING_MARKS
        for char in unicodedata.normalize("NFD", form)
    )


def withheld_irregular_compound(lemma: str) -> bool:
    """Whether the old dispatcher approximated this prefixed irregular."""
    plain = strip_marks(lemma).lower()
    return (plain.endswith(("ειμι", "οιδα", "χρη"))
            and plain not in {"ειμι", "οιδα", "χρη"})


def strip_marks(text: str) -> str:
    return unicodedata.normalize("NFC", "".join(
        char for char in unicodedata.normalize("NFD", text)
        if not unicodedata.combining(char)))


def superseded_verb_generation(lemma: str) -> bool:
    """Whether this historical generated family used a corrected dispatcher.

    Historical middle lemmas used an extra thematic vowel and unrestricted
    active output; -φημι used an unsupported regular or compound route;
    -ννυμι lost the first nu. Irregular compounds used an approximate prefix
    joiner that lost subscripts and combined incompatible base paradigms.
    Rebuild supported families and withhold unsupported compound generation
    instead of recovering old generated rows. Independent base entries
    and any spelling already in the current target remain untouched.
    """
    plain = "".join(
        char for char in unicodedata.normalize("NFD", lemma)
        if not unicodedata.combining(char)
    )
    return (plain.endswith(("μαι", "φημι", "ννυμι"))
            or withheld_irregular_compound(lemma))


def lookup_aliases(form: str) -> tuple[str, ...]:
    """Case-preserving lookup variants, not newly generated paradigm cells."""
    form = unicodedata.normalize("NFC", sanitize_form(form))
    canonical = canonicalize_final_elision(form)
    return tuple(dict.fromkeys(alias for spelling in (form, canonical)
        for alias in (spelling, grave_to_acute(spelling),
                      to_monotonic(spelling), strip_marks(spelling))))


def restore_attested_compound_aliases(
    reference: dict[str, str], reference_base: dict[str, str],
    target: dict[str, str], source_pairs, lexical_heads: set[str], *,
    excluded_forms: set[str] | frozenset[str] = frozenset(),
    historical_conflicts: dict[str, str] | None = None,
) -> dict:
    """Recover attested cells and aliases, never the approximate paradigm.

    Historical compound families bound recovery. Existing reference/base
    keys can seed it through their owner or a literal base-cell letter shape.
    Actual source cells under a positive compound head represented in those
    historical families can also retain coverage, even when the old reference
    lacks that cell. Neither scope rule generates a spelling or licenses it
    without source proof.
    A marked cell needs a separately defined lexical
    owner and an actual Wiktionary/corpus pair. Acute/grave spellings share
    the lookup builder's source precedence. Distinct marked cells that imply
    different owners for one alias remain ambiguous; independent target
    readings always win. Unsupported citation metadata is repaired only for
    an actual source cell. All alternatives and conflict decisions are audited.
    """
    # Reuse the dictionary's structural spelling checks, including genuine
    # crasis and productive enclitic accents, rather than inventing a second
    # orthographic policy for the recovery path.
    from export_hunspell import (
        corpus_lexical_evidence_is_clean, grc_orthography_reason,
        lexical_head_key, TONAL_MARKS,
    )

    hard_exclusions = {grave_to_acute(canonicalize_final_elision(form))
                       for form in excluded_forms}

    def is_excluded(form):
        return grave_to_acute(canonicalize_final_elision(form)) in hard_exclusions

    def bare_key(form):
        return strip_marks(canonicalize_final_elision(sanitize_form(form)))

    heads = {grave_to_acute(sanitize_form(head)) for head in lexical_heads
             if corpus_lexical_evidence_is_clean(head)}
    from build.expand_lsj import _IRREGULAR_BASE_FORMS

    # Some independently attested old cells have already been re-owned
    # outside the withheld family (e.g. ἄνοντος -> ἄνω). Match their letter
    # shape only among existing historical keys, then require source proof
    # below. Never use the former marked prefix joiner to make spellings.
    patterns = set()
    for head in heads:
        if not withheld_irregular_compound(head):
            continue
        plain = strip_marks(head)
        for base, forms in _IRREGULAR_BASE_FORMS.items():
            ending = strip_marks(base)
            if plain.endswith(ending):
                prefix = plain[:-len(ending)]
                patterns.update(prefix + strip_marks(form) for form in forms)
                patterns.add(plain)
    tonal_heads, bare_heads = defaultdict(set), defaultdict(set)
    for head in heads:
        tonal_heads[lexical_head_key(head)].add(head)
        bare_heads[strip_marks(head)].add(head)

    # A separately defined compound citation form is itself a source cell.
    # Preserve its self-map/aliases even when no corpus token names it.
    sources = list(source_pairs)
    head_position = (1 if sources and sources[0][0] ==
                     "hunspell_grc_postclassical_review.json" else 0)
    sources.insert(head_position, ("grc_lexical_heads.json", [
        {"form": head, "lemma": head} for head in sorted(heads)
        if withheld_irregular_compound(head)
    ]))
    direct_sources = {"hunspell_grc_postclassical_review.json",
                      "grc_lexical_heads.json", "ag_pairs.json",
                      "perseus_pairs.json"}

    def canonical_owner(raw):
        lemma = grave_to_acute(sanitize_form(raw))
        if not corpus_lexical_evidence_is_clean(lemma):
            return None
        if lemma in heads:
            return lemma
        if lemma[:1].isupper() and lemma.lower() in heads:
            lemma = lemma.lower()
            return lemma
        # Missing or structurally impossible citation accents can be
        # supplied by a unique positive head with the same remaining marks.
        # A valid accent can distinguish another lexeme or a contracted
        # form, so it must not be silently replaced. This repairs citation
        # metadata only; actual source spellings keep their original marks.
        has_tone = any(ord(char) in TONAL_MARKS
                       for char in unicodedata.normalize("NFD", lemma))
        malformed_tone = has_tone and grc_orthography_reason(lemma) in {
            "tonal_mark_on_consonant", "duplicate_tonal_marks",
            "duplicate_tonal_marks_on_syllable", "misplaced_second_accent",
        }
        options = tonal_heads[lexical_head_key(lemma)]
        if (not has_tone or malformed_tone) and len(options) == 1:
            return next(iter(options))
        # Only a completely unmarked annotation can use a unique marked
        # head. A marked conflict must not lose breathing, subscript or case.
        if strip_marks(lemma) == lemma:
            options = bare_heads[lemma]
            if len(options) == 1:
                return next(iter(options))
        return None

    # Literal base cells from merged Wiktionary/edited treebank evidence can
    # expose an omitted preverb in an automatic owner (e.g. συνανιόντε named
    # as ἄνειμι despite independently defined συνάνειμι). The shape is an
    # ambiguity guard only; it must never infer that longer owner.
    base_shapes = {strip_marks(form) for base, forms in _IRREGULAR_BASE_FORMS.items()
                   if strip_marks(base) == "ειμι" for form in forms}
    for source, pairs in sources:
        if source not in {"ag_pairs.json", "perseus_pairs.json"}:
            continue
        for pair in pairs:
            if pair["lemma"] in {"εἰμί", "εἶμι"}:
                form = canonicalize_final_elision(sanitize_form(pair["form"]))
                if (corpus_lexical_evidence_is_clean(form)
                        and grc_orthography_reason(form) is None):
                    base_shapes.add(strip_marks(form))
    compound_shape_owners = defaultdict(set)
    for head in heads:
        plain = strip_marks(head)
        if withheld_irregular_compound(head) and plain.endswith("ειμι"):
            prefix = plain[:-4]
            for form in base_shapes:
                compound_shape_owners[prefix + form].add(head)

    # A current source may correct the old owner or use a stem-altered cell
    # absent from the literal base-shape inventory. It can seed scope only
    # for an existing historical key and a positively defined compound owner;
    # full spelling, ambiguity, and source checks still apply below.
    source_compound_bare = set()
    represented_owners = {canonical_owner(lemma)
        for lemma in set(reference.values()) | set(reference_base.values())
        if withheld_irregular_compound(lemma)}
    represented_owners.discard(None)
    source_family_scope = set()
    for _source, pairs in sources:
        for pair in pairs:
            if not withheld_irregular_compound(pair["lemma"]):
                continue
            owner = canonical_owner(pair["lemma"])
            if owner and withheld_irregular_compound(owner):
                source_compound_bare.add(bare_key(pair["form"]))
                if owner in represented_owners:
                    source_family_scope.add(bare_key(pair["form"]))
    scope, patterned_scope, sourced_scope = set(), set(), set()
    for historical in (reference, reference_base):
        for form, lemma in historical.items():
            if is_excluded(form):
                continue
            bare = bare_key(form)
            if withheld_irregular_compound(lemma):
                scope.update(lookup_aliases(form))
            elif bare in patterns or bare in source_compound_bare:
                aliases = lookup_aliases(form)
                scope.update(aliases)
                if bare in patterns:
                    patterned_scope.update(aliases)
                if bare in source_compound_bare:
                    sourced_scope.update(aliases)
    scoped_bare = {bare_key(form) for form in scope}
    additional_source_cell_scope = source_family_scope - scoped_bare
    scoped_bare.update(source_family_scope)

    cells, rejected = {}, []
    for priority, (source, pairs) in enumerate(sources):
        for pair in pairs:
            raw = canonicalize_final_elision(
                unicodedata.normalize("NFC", sanitize_form(pair["form"])))
            if bare_key(raw) not in scoped_bare:
                continue
            owner = canonical_owner(pair["lemma"])
            record = {"form": raw, "lemma": pair["lemma"],
                      "source": source, "source_priority": priority,
                      "owner": owner, "original_form": pair["form"]}
            for field in ("pos", "tags"):
                if field in pair:
                    record[field] = pair[field]
            for field in ("evidence", "citation", "source_url"):
                if field in pair:
                    record[field] = pair[field]
            # Validate textual elision in the same canonical storage spelling
            # used by the lookup builder, preserving the source reading above.
            spelling_reason = grc_orthography_reason(raw)
            if (is_excluded(raw) or not owner or not has_orthographic_mark(raw)
                    or not corpus_lexical_evidence_is_clean(raw)
                    or spelling_reason is not None):
                record["reason"] = ("excluded_form" if is_excluded(raw)
                                    else "unsupported_owner" if not owner
                                    else spelling_reason or "unmarked_or_malformed_cell")
                rejected.append(record)
                continue
            key = grave_to_acute(raw)
            if key not in cells:
                cells[key] = {"priority": priority, "tier_owners": set(),
                              "readings": [], "forms": set()}
            if cells[key]["priority"] == priority:
                cells[key]["tier_owners"].add(owner)
            cells[key]["readings"].append(record)
            cells[key]["forms"].add(raw)
            cells[key]["forms"].add(unicodedata.normalize("NFC", sanitize_form(pair["form"])))

    tied_cells = []
    for key, cell in cells.items():
        cell["readings"].sort(key=lambda row: (
            row["source_priority"], row["form"], row["lemma"], row["owner"]))
        owners = cell["tier_owners"]
        if len(owners) == 1:
            cell["owner"] = next(iter(owners))
        else:
            # Merged Wiktionary morphology can carry several readings in
            # one source. Row order is not precedence: use its already
            # selected positive target owner only when the cell supports it.
            selected = {canonical_owner(target[form])
                        for form in {key, *cell["forms"]} if form in target}
            selected &= owners
            cell["owner"] = next(iter(selected)) if len(selected) == 1 else None
            tied_cells.append({"form": key, "owners": sorted(owners),
                               "selected_owner": cell["owner"],
                               "readings": cell["readings"]})
        # Automatic corpus annotations can disagree about lexical ownership
        # even when the combined marked lookup has selected one convention.
        # That winner alone cannot justify a new bare alias: it may replace
        # a correct fallback (e.g. συνηγμενον -> συνάγω) with σύνειμι.
        # Exact reviewed forms, independent citation heads, Wiktionary cells,
        # and the edited treebank may resolve a lower-tier annotation dispute.
        first_source = cell["readings"][0]["source"]
        all_owners = {row["owner"] for row in cell["readings"]}
        disputed = len(all_owners) > 1 and first_source not in direct_sources
        # Automatic annotations sometimes group aorist/perfect/pluperfect
        # forms under an εἰμί/εἶμι compound (e.g. παρηγμένα -> πάρειμι).
        # Those defective base families have no such paradigm. Preserve an
        # existing marked convention, but require direct source ownership
        # before extending it to missing cells or bare aliases.
        incompatible = (first_source not in direct_sources and any(
            strip_marks(row["owner"]).endswith("ειμι")
            and {"aorist", "perfect", "pluperfect"}.intersection(row.get("tags", ()))
            for row in cell["readings"] if row["source"] == first_source))
        shape_owners = compound_shape_owners[bare_key(key)]
        longer_shape_owners = {owner for owner in shape_owners
            if cell["owner"] is not None
            and len(strip_marks(owner)) > len(strip_marks(cell["owner"]))}
        misplaced_preverb = (first_source not in direct_sources
            and cell["owner"] is not None
            and withheld_irregular_compound(cell["owner"])
            and strip_marks(cell["owner"]).endswith("ειμι")
            and bool(longer_shape_owners) and cell["owner"] not in shape_owners)
        cell["alias_owner"] = (None if disputed or incompatible or misplaced_preverb
                               else cell["owner"])
        cell["automatic_owner_conflict"] = disputed
        cell["unsupported_annotation_morphology"] = bool(incompatible)
        cell["unsupported_compound_owner_shape"] = bool(misplaced_preverb)
        cell["potential_compound_shape_owners"] = sorted(shape_owners)
        cell["direct_source_support"] = first_source in direct_sources

    # A sentence-initial verb can have independent lowercase morphology,
    # but a lower lexical head alone cannot disprove a capitalized name.
    # Do not extend a lone capitalized automatic annotation to new aliases
    # or use it to erase an existing capitalized target reading.
    for key, cell in cells.items():
        lowercase = cells.get(key.lower()) if key[:1].isupper() else None
        corroborated = (lowercase is not None and cell["alias_owner"] is not None
                        and lowercase["alias_owner"] == cell["alias_owner"])
        possible_name = any(target[form][:1].isupper()
            and target[form] != cell["owner"]
            for form in {key, *cell["forms"]} if form in target)
        uncorroborated = (key[:1].isupper() and not cell["direct_source_support"]
                         and (not corroborated or possible_name))
        cell["unsupported_capitalized_annotation"] = bool(uncorroborated)
        if uncorroborated:
            cell["alias_owner"] = None

    proposals = defaultdict(dict)
    for key, cell in cells.items():
        for form in sorted(cell["forms"]):
            for alias in lookup_aliases(form):
                if (not is_excluded(alias)
                        and (alias in scope or bare_key(alias) in scoped_bare)):
                    proposals[alias][key] = cell
    actual_forms = {form for cell in cells.values() for form in cell["forms"]}
    recovered, ambiguous, preserved, canonicalized = [], [], [], []
    for alias, evidence in sorted(proposals.items()):
        owners = {cell["alias_owner"] for cell in evidence.values()
                  if cell["alias_owner"]}
        has_tie = any(cell["alias_owner"] is None for cell in evidence.values())
        detail = {"form": alias, "owners": sorted(owners),
                  "cells": [{"form": key, "owner": cell["owner"],
                             "alias_owner": cell["alias_owner"],
                             "automatic_owner_conflict": cell["automatic_owner_conflict"],
                             "unsupported_annotation_morphology": cell["unsupported_annotation_morphology"],
                             "unsupported_compound_owner_shape": cell["unsupported_compound_owner_shape"],
                             "unsupported_capitalized_annotation": cell["unsupported_capitalized_annotation"],
                             "direct_source_support": cell["direct_source_support"],
                             "potential_compound_shape_owners": cell["potential_compound_shape_owners"],
                             "readings": cell["readings"]}
                            for key, cell in sorted(evidence.items())]}
        if historical_conflicts and alias in historical_conflicts:
            detail["historical_conflict_owner"] = historical_conflicts[alias]
        if alias in target:
            detail["current_owner"] = target[alias]
            current = canonical_owner(target[alias])
            direct = any(cell["direct_source_support"] for cell in evidence.values())
            if (not has_tie and len(owners) == 1 and target[alias] not in owners
                    and (current in owners
                         or (current is None and alias in actual_forms and direct))):
                owner = next(iter(owners))
                target[alias] = owner
                detail["owner"] = owner
                canonicalized.append(detail)
            else:
                preserved.append(detail)
        elif has_tie or len(owners) != 1:
            ambiguous.append(detail)
        else:
            owner = next(iter(owners))
            target[alias] = owner
            detail["owner"] = owner
            recovered.append(detail)
    return {"schema_version": 1,
            "scope_aliases": len(scope), "source_cells": len(cells),
            "pattern_scoped_aliases": len(patterned_scope),
            "current_compound_source_scoped_aliases": len(sourced_scope),
            "additional_historical_family_source_cell_scope": len(additional_source_cell_scope),
            "automatic_owner_conflict_count": sum(
                cell["automatic_owner_conflict"] for cell in cells.values()),
            "unsupported_annotation_morphology_count": sum(
                cell["unsupported_annotation_morphology"] for cell in cells.values()),
            "unsupported_compound_owner_shape_count": sum(
                cell["unsupported_compound_owner_shape"] for cell in cells.values()),
            "unsupported_capitalized_annotation_count": sum(
                cell["unsupported_capitalized_annotation"] for cell in cells.values()),
            "recovered_count": len(recovered), "ambiguous_count": len(ambiguous),
            "canonicalized_count": len(canonicalized),
            "existing_preserved_count": len(preserved),
            "recovered": recovered, "ambiguous": ambiguous,
            "canonicalized": canonicalized,
            "same_source_tied_cells": sorted(tied_cells, key=lambda row: row["form"]),
            "existing_preserved": preserved, "rejected_source_rows": rejected}


def load_alias_source_pairs(data: Path):
    """Load the merged source evidence and record portable byte pins."""
    pairs, pins = [], {}
    # Independently reviewed exact textual forms outrank automatic annotation
    # conventions; derived paradigms never enter this source-evidence tier.
    review_path = data / "hunspell_grc_postclassical_review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("schema_version") != 1:
        raise SystemExit("invalid exact source-form review for alias recovery")
    pins[review_path.name] = {
        "sha256": file_sha256(review_path), "size": review_path.stat().st_size}
    reviewed_pairs = []
    for spec in review["paradigms"]:
        if spec["method"] != "source_forms":
            continue
        for form in spec["forms"]:
            reviewed_pairs.append({"form": form, "lemma": spec["lemma"],
                                   **{field: spec[field] for field in
                                      ("evidence", "citation", "source_url")
                                      if field in spec}})
    pairs.append((review_path.name, reviewed_pairs))
    for name in SOURCE_PAIR_NAMES:
        path = data / name
        if not path.is_file():
            raise SystemExit(f"source pairs missing for alias recovery: {name}")
        pins[name] = {"sha256": file_sha256(path), "size": path.stat().st_size}
        pairs.append((name, json.loads(path.read_text(encoding="utf-8"))))
    return pairs, pins


def load_alias_lexical_heads(path: Path) -> set[str]:
    """Only positively evidenced Ancient heads license source recovery."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2 or not payload.get("heads"):
        raise SystemExit("invalid positive lexical-head evidence for alias recovery")
    return {head for head, evidence in payload["heads"].items() if evidence}


def load_closed_class_entries(kaikki: Path):
    """Read both exact fresh dumps, retaining own Ancient PRON/DET entries."""
    pins, paths, entries = {}, {}, []
    for label, (source, relative) in CLOSED_CLASS_DUMPS.items():
        path = kaikki / relative
        if not path.is_file():
            raise SystemExit(f"fresh closed-class source missing: {relative}")
        paths[label] = path
        pins[label] = {"path": relative, "sha256": file_sha256(path),
                       "size": path.stat().st_size}
    for label, (source, relative) in CLOSED_CLASS_DUMPS.items():
        with paths[label].open(encoding="utf-8") as handle:
            for line, text in enumerate(handle, 1):
                entry = json.loads(text)
                if entry.get("lang_code") == "grc" and entry.get("pos") in {"pron", "det"}:
                    entries.append({"source": source, "line": line, "entry": entry})
    verify_closed_class_inputs(paths, pins)
    return entries, pins, paths


def verify_closed_class_inputs(paths, pins):
    """Do not publish evidence read while either dump changed."""
    for label, path in paths.items():
        actual = {"path": pins[label]["path"], "sha256": file_sha256(path),
                  "size": path.stat().st_size}
        if actual != pins[label]:
            raise SystemExit(f"fresh closed-class source changed: {pins[label]['path']}")


def prepare_closed_class_elisions(kaikki, reference, reference_base, predecessor,
                                  source_pairs, lexical_heads):
    """Plan before historical additions, with imported source and dump pins."""
    if kaikki is None:
        return {"schema_version": 1, "enabled": False}, {}
    import build_data
    from build import closed_class_elisions
    imported = {
        "build/closed_class_elisions.py": Path(closed_class_elisions.__file__),
        "build_data.py": Path(build_data.__file__),
    }
    source_pins = {}
    for relative, path in imported.items():
        if path.resolve() != (ROOT / relative).resolve():
            raise SystemExit(f"closed-class classifier imported outside source tree: {relative}")
        source_pins[relative] = {"sha256": file_sha256(path), "size": path.stat().st_size}
    entries, pins, paths = load_closed_class_entries(kaikki)
    decisions = closed_class_elisions.plan_repairs(
        reference, reference_base, predecessor, source_pairs, lexical_heads, entries)
    return {"schema_version": 1, "enabled": True,
            "external_input_sha256": pins, "source_sha256": source_pins,
            "fresh_entry_count": len(entries), "decisions": decisions,
            "planned_count": sum(row["accepted"] for row in decisions)}, paths


def merge_closed_class_elisions(target, alias_stats, closed_class):
    """Keep all proof decisions and merge only reversible still-matching rows."""
    if not closed_class["enabled"]:
        alias_stats["closed_class_elisions"] = closed_class
        return
    from build.closed_class_elisions import apply_repairs
    applied = apply_repairs(target, closed_class["decisions"])
    for row in applied["canonicalized"] + applied["existing_preserved"]:
        row["repair_kind"] = "historical_closed_class_elision"
    existing_forms = {row["form"] for row in alias_stats["recovered"]
                      + alias_stats["canonicalized"]}
    if existing_forms & {row["form"] for row in applied["canonicalized"]}:
        raise SystemExit("closed-class repair duplicates another alias change")
    alias_stats["canonicalized"].extend(applied["canonicalized"])
    alias_stats["existing_preserved"].extend(applied["existing_preserved"])
    alias_stats["canonicalized_count"] = len(alias_stats["canonicalized"])
    alias_stats["existing_preserved_count"] = len(alias_stats["existing_preserved"])
    closed_class.update(canonicalized=applied["canonicalized"],
                        existing_preserved=applied["existing_preserved"],
                        canonicalized_count=len(applied["canonicalized"]),
                        existing_preserved_count=len(applied["existing_preserved"]))
    alias_stats["closed_class_elisions"] = closed_class


def alias_spelling_exclusions(exclusions: dict, review_path: Path) -> frozenset[str]:
    """Historical row conflicts are not absolute bans on a source spelling."""
    from export_hunspell import load_grc_spelling_review

    reviewed = load_grc_spelling_review(review_path)
    hard = frozenset(exclusions["nonlexical"]) | reviewed
    return hard | frozenset(grave_to_acute(canonicalize_final_elision(form))
                            for form in hard)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, pin: dict, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")
    expected_size = pin["size"]
    if path.stat().st_size != expected_size:
        raise SystemExit(
            f"{label} size mismatch: expected {expected_size}, "
            f"got {path.stat().st_size}: {path}"
        )
    actual_hash = file_sha256(path)
    if actual_hash != pin["sha256"]:
        raise SystemExit(
            f"{label} SHA-256 mismatch: expected {pin['sha256']}, "
            f"got {actual_hash}: {path}"
        )


def load_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise SystemExit(f"unsupported expansion manifest: {path}")
    return manifest


def resolve_hf_file(pin: dict, label: str, explicit: Path | None = None) -> Path:
    if explicit is not None:
        path = explicit
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise SystemExit(
                "huggingface_hub is required to download the pinned expansion "
                "reference; install the project build dependencies or pass "
                "--reference"
            ) from exc
        path = Path(hf_hub_download(
            repo_id=pin["repo"],
            filename=pin["path"],
            revision=pin["revision"],
        ))
    verify_file(path, pin, label)
    return path


def resolve_reference(manifest: dict, explicit: Path | None) -> Path:
    return resolve_hf_file(
        manifest["reference"], "expansion reference", explicit
    )


def overlay_expansion(
    reference: dict[str, str],
    reference_base: dict[str, str],
    target: dict[str, str],
    excluded_forms: set[str] | frozenset[str] = frozenset(),
    superseded_noun_lemmas: set[str] | frozenset[str] = frozenset(),
) -> dict[str, int]:
    """Add only historical expansion keys, preserving current mappings."""
    stats = {
        "reference_entries": len(reference),
        "reference_base_entries": len(reference_base),
        "target_entries_before": len(target),
        "expansion_entries": 0,
        "unmarked_entries": 0,
        "editorial_entries": 0,
        "excluded_entries": 0,
        "already_present": 0,
        "conflicts_preserved": 0,
        "nonlexical_lemma_entries": 0,
        "superseded_verb_entries": 0,
        "superseded_noun_entries": 0,
        "added": 0,
    }
    for form, lemma in reference.items():
        if form in reference_base:
            continue
        stats["expansion_entries"] += 1
        if not has_orthographic_mark(form):
            stats["unmarked_entries"] += 1
            continue
        if form in excluded_forms:
            stats["excluded_entries"] += 1
            continue
        if has_editorial_sigla(form) or has_editorial_sigla(lemma):
            stats["editorial_entries"] += 1
            continue
        if form in target:
            stats["already_present"] += 1
            if target[form] != lemma:
                stats["conflicts_preserved"] += 1
            continue
        if nonlexical_generated_lemma(lemma):
            stats["nonlexical_lemma_entries"] += 1
            continue
        if superseded_verb_generation(lemma):
            stats["superseded_verb_entries"] += 1
            continue
        if lemma in superseded_noun_lemmas:
            stats["superseded_noun_entries"] += 1
            continue
        target[form] = lemma
        stats["added"] += 1
    stats["target_entries_after"] = len(target)
    return stats


def load_noun_generation_audit(path: Path, generator="expand_lsj.py") -> frozenset[str]:
    """Require the current generator's completed audit before recovering nouns."""
    if not path.exists():
        raise SystemExit(f"noun generation audit missing; run build/{generator} --expand first")
    audit = json.loads(path.read_text(encoding="utf-8"))
    source = ROOT / "build" / generator
    if (audit.get("schema_version") != 1
            or audit.get("generator_sha256") != file_sha256(source)):
        raise SystemExit(f"noun generation audit is stale; rerun build/{generator} --expand")
    if (generator != "expand_lsj.py" and audit.get("shared_generator_sha256") !=
            file_sha256(ROOT / "build" / "expand_lsj.py")):
        raise SystemExit(f"shared noun generator changed; rerun build/{generator} --expand")
    sys.path.insert(0, str(ROOT / "build"))
    if generator == "expand_lsj.py":
        from expand_lsj import noun_generation_input_hashes
        current_inputs = noun_generation_input_hashes()
    else:
        from expand_sophocles import sophocles_noun_input_hashes
        current_inputs = sophocles_noun_input_hashes()
    if audit.get("input_sha256") != current_inputs:
        raise SystemExit(f"noun inputs changed; rerun build/{generator} --expand")
    lemmas = audit.get("superseded_lemmas")
    if not isinstance(lemmas, list) or not all(isinstance(x, str) for x in lemmas):
        raise SystemExit("invalid noun generation audit")
    return frozenset(lemmas)


def resolve_target_editorial_entries(target: dict[str, str]) -> dict[str, int]:
    """Remove notation from a built target without guessing at restorations.

    Terminal movable-nu notation expands to its two real spellings. All other
    bracketed forms, and every bracketed lemma, are discarded. Existing target
    mappings always win when a resolved spelling collides.
    """
    contaminated = [
        (form, lemma)
        for form, lemma in target.items()
        if has_editorial_sigla(form) or has_editorial_sigla(lemma)
    ]
    stats = {
        "target_editorial_removed": len(contaminated),
        "target_movable_nu_entries": 0,
        "target_movable_nu_added": 0,
        "target_movable_nu_present": 0,
        "target_movable_nu_conflicts": 0,
    }
    for form, _lemma in contaminated:
        del target[form]

    for form, lemma in contaminated:
        if has_editorial_sigla(lemma):
            continue
        resolved = resolve_editorial_form(form)
        if not resolved:
            continue
        stats["target_movable_nu_entries"] += 1
        for spelling in resolved:
            if spelling not in target:
                target[spelling] = lemma
                stats["target_movable_nu_added"] += 1
            else:
                stats["target_movable_nu_present"] += 1
                if target[spelling] != lemma:
                    stats["target_movable_nu_conflicts"] += 1
    return stats


def verify_sentinels(lookup: dict[str, str], sentinels: dict[str, str]) -> None:
    wrong = {
        form: {"expected": lemma, "actual": lookup.get(form)}
        for form, lemma in sentinels.items()
        if lookup.get(form) != lemma
    }
    if wrong:
        rendered = ", ".join(
            f"{form!r}: {values['actual']!r} != {values['expected']!r}"
            for form, values in wrong.items()
        )
        raise SystemExit(f"expansion sentinel mismatch: {rendered}")


def write_json_atomic(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.chmod(output_mode)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore the pinned historical AG expansion layer"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--reference", type=Path,
        help="Local copy of the pinned reference (otherwise downloaded)",
    )
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--noun-audit", type=Path, default=DEFAULT_NOUN_AUDIT)
    parser.add_argument("--sophocles-noun-audit", type=Path,
                        default=DEFAULT_SOPHOCLES_NOUN_AUDIT)
    parser.add_argument("--source-pairs", type=Path, default=ROOT / "data",
                        help="Directory with all merged Wiktionary/corpus pair files")
    parser.add_argument("--lexical-heads", type=Path, default=DEFAULT_LEXICAL_HEADS)
    parser.add_argument("--source-alias-audit", type=Path,
                        help="Save portable alias evidence, conflicts and input hashes")
    parser.add_argument("--kaikki", type=Path,
                        help="Fresh dump directory; require both Ancient dumps for "
                             "source-proved historical closed-class elision repairs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.kaikki is not None:
        os.environ["KAIKKI_DIR"] = str(args.kaikki)
    superseded_nouns = load_noun_generation_audit(args.noun_audit)
    superseded_nouns |= load_noun_generation_audit(args.sophocles_noun_audit,
                                                  "expand_sophocles.py")

    manifest = load_manifest(args.manifest)
    reference_path = resolve_reference(manifest, args.reference)
    base_path = ROOT / manifest["reference_base"]["path"]
    verify_file(base_path, manifest["reference_base"], "reference base")
    exclusions_path = ROOT / manifest["exclusions"]["path"]
    verify_file(exclusions_path, manifest["exclusions"], "expansion exclusions")
    with exclusions_path.open(encoding="utf-8") as handle:
        exclusions = json.load(handle)
    excluded_forms = frozenset(exclusions["baseline_conflicts"]) | frozenset(
        exclusions["nonlexical"]
    )

    print(f"Loading pinned reference: {reference_path}")
    with reference_path.open(encoding="utf-8") as handle:
        reference = json.load(handle)
    print(f"  {len(reference):,} entries")

    print(f"Loading exact reference base: {base_path}")
    with base_path.open(encoding="utf-8") as handle:
        reference_base = json.load(handle)
    print(f"  {len(reference_base):,} entries")

    print(f"Loading current target: {args.target}")
    with args.target.open(encoding="utf-8") as handle:
        target = json.load(handle)
    print(f"  {len(target):,} entries")

    source_pairs, alias_inputs = load_alias_source_pairs(args.source_pairs)
    lexical_heads = load_alias_lexical_heads(args.lexical_heads)
    alias_inputs["grc_lexical_heads.json"] = {
        "sha256": file_sha256(args.lexical_heads),
        "size": args.lexical_heads.stat().st_size,
    }
    closed_class, fresh_paths = prepare_closed_class_elisions(
        args.kaikki, reference, reference_base, target, source_pairs, lexical_heads)
    target_editorial_stats = resolve_target_editorial_entries(target)
    print(
        "Resolved target editorial notation: "
        f"{target_editorial_stats['target_movable_nu_added']:,} movable-nu "
        f"spellings added; "
        f"{target_editorial_stats['target_editorial_removed']:,} notation "
        "entries removed"
    )

    stats = overlay_expansion(
        reference, reference_base, target, excluded_forms=excluded_forms,
        superseded_noun_lemmas=superseded_nouns,
    )
    alias_stats = restore_attested_compound_aliases(
        reference, reference_base, target, source_pairs,
        lexical_heads,
        excluded_forms=alias_spelling_exclusions(
            exclusions, args.source_pairs / "hunspell_grc_spelling_review.json"),
        historical_conflicts=exclusions["baseline_conflicts"],
    )
    merge_closed_class_elisions(target, alias_stats, closed_class)
    alias_stats["historical_overlay_counts"] = stats
    if closed_class["enabled"]:
        verify_closed_class_inputs(fresh_paths, closed_class["external_input_sha256"])
        for relative, expected in closed_class["source_sha256"].items():
            actual = {"sha256": file_sha256(ROOT / relative),
                      "size": (ROOT / relative).stat().st_size}
            if actual != expected:
                raise SystemExit(f"closed-class source changed: {relative}")
    spelling_review = args.source_pairs / "hunspell_grc_spelling_review.json"
    alias_inputs[spelling_review.name] = {
        "sha256": file_sha256(spelling_review), "size": spelling_review.stat().st_size}
    alias_stats["input_sha256"] = alias_inputs
    alias_stats["reference_sha256"] = manifest["reference"]["sha256"]
    alias_stats["reference_base_sha256"] = manifest["reference_base"]["sha256"]
    alias_stats["generator_sha256"] = file_sha256(Path(__file__))
    alias_stats["cell_pattern_source_sha256"] = file_sha256(ROOT / "build/expand_lsj.py")
    for key, expected in manifest["expected_counts"].items():
        if stats[key] != expected:
            raise SystemExit(
                f"expansion count mismatch for {key}: expected "
                f"{expected:,}, got {stats[key]:,}"
            )
    verify_sentinels(target, manifest["sentinels"])

    print(f"Historical expansion layer: {stats['expansion_entries']:,}")
    print(f"Unmarked fallback keys skipped: {stats['unmarked_entries']:,}")
    print(f"Editorial-notation entries skipped: {stats['editorial_entries']:,}")
    print(f"Pinned exclusions: {stats['excluded_entries']:,}")
    print(f"Already present: {stats['already_present']:,}")
    print(f"Current mapping conflicts preserved: {stats['conflicts_preserved']:,}")
    print(f"Nonlexical generated lemma entries skipped: {stats['nonlexical_lemma_entries']:,}")
    print(f"Superseded generated verb entries skipped: {stats['superseded_verb_entries']:,}")
    print(f"Superseded generated noun entries skipped: {stats['superseded_noun_entries']:,}")
    print(f"Recovered entries: {stats['added']:,}")
    print(f"Attested compound cells/aliases recovered: {alias_stats['recovered_count']:,}")
    print(f"Attested owners canonicalized: {alias_stats['canonicalized_count']:,}")
    print(f"Historical closed-class elisions canonicalized: "
          f"{closed_class.get('canonicalized_count', 0):,}")
    print(f"Ambiguous aliases withheld: {alias_stats['ambiguous_count']:,}")
    print(f"Independent existing readings preserved: {alias_stats['existing_preserved_count']:,}")
    print(f"Final size: {len(target):,}")
    if args.source_alias_audit:
        write_json_atomic(args.source_alias_audit, alias_stats)

    if args.dry_run:
        print("Dry run: target was not changed")
        return
    print(f"Saving atomically to {args.target}...")
    write_json_atomic(args.target, target)
    print(f"  {args.target.stat().st_size / (1024 * 1024):.1f} MB written")


if __name__ == "__main__":
    main()
