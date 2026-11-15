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
CARRIED_BASELINE_ENV = "DILEMMA_AG_EXPANSION_CARRIED"
# The same source order as the lookup builder, with the fresh merged
# Wiktionary pairs first. Annotation alternatives are retained in the audit.
SOURCE_PAIR_NAMES = (
    "ag_pairs.json", "perseus_pairs.json", "glaux_pairs.json",
    "diorisis_pairs.json", "nt_pairs.json", "ptnk_pairs.json",
    "tagnt_pairs.json", "pedalion_pairs.json",
)
# label -> (edition, flat dump name). Either dump layout is read
# (dilemma_build.kaikki_dumps); audits record the flat name, so both layouts
# give the same audit bytes.
CLOSED_CLASS_DUMPS = {
    "wiktionary_en_grc": ("en", "kaikki.org-en-dictionary-AncientGreek.jsonl"),
    "wiktionary_el_grc": ("el", "kaikki.org-el-dictionary-AncientGreek.jsonl"),
}
_CHUNK = 1 << 20
_SPACING_MARKS = frozenset({0x1FBD, 0x1FBF, 0x1FFE})
_ACCENT_MARKS = frozenset("\u0300\u0301\u0342")
_BREATHING_MARKS = ("\u0313", "\u0314")
_LEADING_BREATHINGS = frozenset("\u0313\u0314\u1fbf\u1ffe")
# Regular masculine and neuter cells of a second-declension oxytone.
_OXYTONE_OS_ENDINGS = ("ός", "οῦ", "ῷ", "όν", "έ", "ώ", "οῖν", "οί", "ῶν",
                       "οῖς", "ούς", "ά")
# What a consonant-stem or s-stem entry prints after its headword: a
# genitive in -ος, -ως or -ους, or the neuter -ες of an adjective in -ης.
_THIRD_DECLENSION_ENDINGS = ("ος", "ως", "ους", "ες")


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
    from dilemma_build.expand_lsj import _IRREGULAR_BASE_FORMS

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
    from dilemma_build.kaikki_dumps import resolve_dump

    pins, paths, entries = {}, {}, []
    for label, (source, filename) in CLOSED_CLASS_DUMPS.items():
        path = resolve_dump(filename, kaikki)
        if not path.is_file():
            raise SystemExit(f"fresh closed-class source missing: {filename}")
        paths[label] = path
        # The flat dump name, whichever layout holds the file, so equal
        # dump bytes give equal audit bytes.
        pins[label] = {"path": filename, "sha256": file_sha256(path),
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
    from dilemma_build import closed_class_elisions
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
    from dilemma_build.closed_class_elisions import apply_repairs
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


def resolve_carried_baseline(manifest: dict, explicit: Path | None) -> Path:
    return resolve_hf_file(
        manifest["carried_baseline"], "carried expansion baseline", explicit
    )


def carried_baseline_conflicts(conflicts, reference: dict[str, str],
                               carried: dict[str, str]) -> frozenset[str]:
    """Baseline conflicts the published table carried with the expansion reading.

    The pinned exclusions keep an expansion reading out of every form whose
    1.3.1 resolution disagreed with it, so that resolution's sources fill
    the form later in the lookup build (ἐκπλαγῇ ἐκπλάσσω, not the generated
    ἐκπλαγής). The table published from 1.3.7 to 1.3.25 nevertheless read
    some of those forms from the expansion (ἀμέτρητον ἀμέτρητος, Δημήτρειον
    Δημήτρειος), because it was built on top of earlier tables. A build from
    fresh dumps withholds them, and the corpus gap-fill then supplies the
    recorded reading (ἀμετρητος, Δημήτρειοι). Such a conflict is released:
    the form keeps the reading the published table gave it. Every other
    conflict stays excluded, so a capitalised function word (Γὰρ, Ἤ) still
    reads its lowercase lemma and not a generated name.
    """
    return frozenset(
        form for form in conflicts
        if form in reference and carried.get(form) == reference[form])


def fold_leading_breathing(lemma: str) -> str:
    """A lemma whose breathing is written before its capital, precomposed.

    The reference spells part of its capitalised lemmas with a spacing
    breathing in front of the letter (᾿Αγαρηνός for Ἀγαρηνός).
    """
    if lemma and lemma[0] in _LEADING_BREATHINGS:
        return sanitize_form(lemma)
    return lemma


def kept_withheld_noun_forms(
    pending: dict[str, list[str]],
    withheld_noun_lemmas: dict[str, tuple[str, ...]],
    lexical_heads: set[str] | frozenset[str],
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Choose the published keys a noun without a fresh paradigm keeps.

    ``pending`` holds, per superseded lemma whose generation was withheld,
    the marked expansion keys the published table carried and the target
    lacks. They are kept when the lemma is a positive lexical head and the
    key is a structurally possible polytonic spelling (the Hunspell
    export's grc_orthography_reason, with the elision mark normalised).

    One class of old paradigm is known to be wrong. The old generator
    declined a headword by its nominative when it could not use the printed
    genitive, so Θεοπρεπότης, -τος became Θεοπρεπότου and ἀδιτίων, -ωνος
    became ἀδιτίονος. Where the lemma's own citation key is among the keys
    (the whole paradigm was generated) and the entry prints a
    third-declension ending that none of the other keys ends with, only the
    citation spelling is kept.
    """
    from export_hunspell import grc_orthography_reason

    kept: list[tuple[str, str]] = []
    stats = {"withheld_noun_lemmas_kept": 0, "withheld_noun_entries_kept": 0,
             "withheld_noun_citation_only_lemmas": 0, "left": 0}
    for lemma, forms in pending.items():
        candidates = [
            form for form in forms
            if lemma in lexical_heads and grc_orthography_reason(
                canonicalize_final_elision(sanitize_form(form))) is None
        ]
        citation = [form for form in candidates if grave_to_acute(form) == lemma]
        if citation:
            letters = [strip_marks(form).lower() for form in candidates
                       if form not in citation]
            printed = [strip_marks(text).lower()
                       for text in withheld_noun_lemmas[lemma]]
            if any(ending.endswith(_THIRD_DECLENSION_ENDINGS)
                   and not any(form.endswith(ending) for form in letters)
                   for ending in printed):
                candidates = citation
                stats["withheld_noun_citation_only_lemmas"] += 1
        stats["left"] += len(forms) - len(candidates)
        if candidates:
            stats["withheld_noun_lemmas_kept"] += 1
            stats["withheld_noun_entries_kept"] += len(candidates)
            kept.extend((form, lemma) for form in candidates)
    return kept, stats


def overlay_expansion(
    reference: dict[str, str],
    reference_base: dict[str, str],
    target: dict[str, str],
    excluded_forms: set[str] | frozenset[str] = frozenset(),
    superseded_noun_lemmas: set[str] | frozenset[str] = frozenset(),
    released_forms: set[str] | frozenset[str] = frozenset(),
    carried: dict[str, str] | None = None,
    withheld_noun_lemmas: dict[str, tuple[str, ...]] | None = None,
    lexical_heads: set[str] | frozenset[str] = frozenset(),
) -> dict[str, int]:
    """Add only historical expansion keys, preserving current mappings.

    ``released_forms`` are excluded forms (released baseline conflicts) that
    still take the expansion reading; they count among the pinned
    exclusions and separately as released.

    A superseded noun lemma loses its old paradigm to the fresh one. A
    lemma in ``withheld_noun_lemmas`` has no fresh paradigm (the generator
    could not use its printed genitive), so it keeps the expansion keys
    ``carried`` (the published table) has with the same reading, as
    kept_withheld_noun_forms() selects them. They are counted as
    ``withheld_noun_entries_kept``, not as ``added``.
    """
    carried = carried or {}
    withheld_noun_lemmas = withheld_noun_lemmas or {}
    pending: dict[str, list[str]] = {}
    stats = {
        "reference_entries": len(reference),
        "reference_base_entries": len(reference_base),
        "target_entries_before": len(target),
        "expansion_entries": 0,
        "unmarked_entries": 0,
        "editorial_entries": 0,
        "excluded_entries": 0,
        "released_conflict_entries": 0,
        "already_present": 0,
        "conflicts_preserved": 0,
        "nonlexical_lemma_entries": 0,
        "superseded_verb_entries": 0,
        "superseded_noun_entries": 0,
        "withheld_noun_lemmas_kept": 0,
        "withheld_noun_entries_kept": 0,
        "withheld_noun_citation_only_lemmas": 0,
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
            # A released form with editorial notation stays excluded: the
            # pinned counts file it under the exclusions, not the notation.
            if (form not in released_forms or has_editorial_sigla(form)
                    or has_editorial_sigla(lemma)):
                continue
            stats["released_conflict_entries"] += 1
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
            if lemma in withheld_noun_lemmas and carried.get(form) == lemma:
                pending.setdefault(lemma, []).append(form)
            else:
                stats["superseded_noun_entries"] += 1
            continue
        target[form] = lemma
        stats["added"] += 1
    if pending:
        kept, kept_stats = kept_withheld_noun_forms(
            pending, withheld_noun_lemmas, lexical_heads)
        stats["superseded_noun_entries"] += kept_stats.pop("left")
        stats.update(kept_stats)
        for form, lemma in kept:
            target[form] = lemma
    stats["target_entries_after"] = len(target)
    return stats


def oxytone_os_cells(lemma: str) -> frozenset[str]:
    """Regular masculine and neuter cells of a second-declension oxytone.

    For a head in -ός with no other accent these are the twelve endings of
    the masculine and the neuter (θελητός: θελητοῦ, θελητόν, θελητά ...),
    each in its accented spelling, its grave spelling where the final
    syllable carries an acute, and its unaccented spelling. Any other
    lemma has none.
    """
    decomposed = unicodedata.normalize("NFD", lemma)
    if (not decomposed.endswith("ός")
            or sum(char in _ACCENT_MARKS for char in decomposed) != 1):
        return frozenset()
    stem = decomposed[:-3]
    cells = set()
    for ending in _OXYTONE_OS_ENDINGS:
        cell = stem + unicodedata.normalize("NFD", ending)
        cells.add(unicodedata.normalize("NFC", cell))
        cells.add(unicodedata.normalize("NFC", cell.replace("́", "̀")))
        cells.add(strip_marks(cell))
    return frozenset(cells)


def confirmed_head(lemma: str, head_evidence: dict[str, tuple[str, ...]]) -> bool:
    """Whether a list other than the Sophocles lexicon attests the head.

    That lexicon is read from OCR, which capitalises entry words and
    misreads breathings and accents, so a head only it attests can be a
    misspelling of another word.
    """
    return any(not name.startswith("sophocles/")
               for name in head_evidence.get(lemma, ()))


def ocr_twin_head(lemma: str, head_evidence: dict[str, tuple[str, ...]]) -> bool:
    """Whether a head only the Sophocles lexicon has is another head's twin.

    Such a head is not trusted when its lowercase spelling (Θερινός beside
    θερινός) or its spelling with the other breathing (Ἱωνικός beside
    Ἰωνικός) is itself a positive head. A head another list confirms keeps
    its spelling (Κρητικός beside κρητικός).
    """
    if lemma not in head_evidence or confirmed_head(lemma, head_evidence):
        return False
    lowercase = lemma[:1].lower() + lemma[1:]
    if lowercase != lemma and lowercase in head_evidence:
        return True
    decomposed = unicodedata.normalize("NFD", lemma)
    for mark, other in (_BREATHING_MARKS, _BREATHING_MARKS[::-1]):
        if mark in decomposed:
            return unicodedata.normalize(
                "NFC", decomposed.replace(mark, other, 1)) in head_evidence
    return False


def breathed_spelling(form: str, head: str, corrected: str) -> str | None:
    """``form`` with the breathing a correction added to its head.

    The breathing review turns ``head`` into ``corrected`` by adding one
    breathing (Ρωμαϊκός to Ῥωμαϊκός, ολικός to ὁλικός). The same mark goes
    on the same letter of an unbreathed form whose letters up to there are
    the head's (Ρωμαϊκόν to Ῥωμαϊκόν); any other form has no such spelling.
    """
    head_nfd = unicodedata.normalize("NFD", head)
    corrected_nfd = unicodedata.normalize("NFD", corrected)
    form_nfd = unicodedata.normalize("NFD", form)
    if (len(corrected_nfd) != len(head_nfd) + 1
            or any(mark in form_nfd for mark in _BREATHING_MARKS)):
        return None
    for index, char in enumerate(corrected_nfd):
        if (char in _BREATHING_MARKS
                and corrected_nfd[:index] + corrected_nfd[index + 1:] == head_nfd):
            break
    else:
        return None
    letters = [char for char in head_nfd[:index] if not unicodedata.combining(char)]
    position = matched = 0
    while matched < len(letters):
        if position == len(form_nfd):
            return None
        char = form_nfd[position]
        if not unicodedata.combining(char):
            if char != letters[matched]:
                return None
            matched += 1
        position += 1
    return unicodedata.normalize(
        "NFC", form_nfd[:position] + corrected_nfd[index] + form_nfd[position:])


def keep_published_over_unaccented_heads(
    target: dict[str, str], carried: dict[str, str],
    lexical_heads: set[str] | frozenset[str],
) -> int:
    """Keep a published reading an accent-less copy of its lemma took over.

    LSJ9 exports some heads without any diacritic (κληζω for κλῄζω). The
    fresh paradigm of such a head claims keys the published table read
    under the properly spelled head (κληζεται: κλῄζω). Where the fresh
    lemma has no diacritic and the published lemma is a positive head with
    exactly those letters, the published reading stays. Returns the number
    of readings put back.
    """
    replaced = 0
    for form, lemma in target.items():
        published = carried.get(form)
        if (published is None or published == lemma
                or has_orthographic_mark(lemma)
                or published not in lexical_heads
                or strip_marks(published) != lemma):
            continue
        target[form] = published
        replaced += 1
    return replaced


def carry_published_cells(
    target: dict[str, str],
    carried: dict[str, str],
    head_evidence: dict[str, tuple[str, ...]],
    breathing_corrections: dict[str, str],
    excluded_forms: set[str] | frozenset[str] = frozenset(),
    superseded_noun_lemmas: set[str] | frozenset[str] = frozenset(),
) -> tuple[dict[str, int], dict[str, tuple[str, str]]]:
    """Carry two kinds of key that only the published table has.

    Neither kind is in the expansion reference: an earlier Sophocles
    expansion made them, and no current generator does. Both are written
    only for a key the target lacks; an existing reading is never replaced.

    Oxytone adjective cells. A key the published table read under a head
    in -ός comes back when the head is a positive lexical head the target
    maps to itself, is not an OCR twin (ocr_twin_head), and the key is one
    of its regular masculine or neuter cells (oxytone_os_cells):
    ἀρχοντικόν ἀρχοντικός, θελητόν θελητός. The published lemma is read
    with a leading spacing breathing folded into its capital. A superseded
    noun is left alone: it has its fresh paradigm, or the keys
    overlay_expansion() kept for it.

    Breathing-corrected heads. The published table read these keys under a
    head whose breathing the Sophocles OCR lost (Ρωμαϊκόν Ρωμαϊκός). The
    breathing review (``sophocles_breathing`` in grc_lexical_heads.json)
    corrected the head, and the key now reads the corrected head when the
    target has its breathed spelling with that head (Ῥωμαϊκόν Ῥωμαϊκός).

    Marked keys are written here. Unaccented keys are returned as
    ``{key: (lemma, rule)}`` for restore_unmarked_keys(), which adds one
    only where the marked spellings of the finished table agree with it.
    """
    stats = {"oxytone_cell_entries": 0, "oxytone_cell_lemmas": 0,
             "corrected_head_entries": 0, "corrected_head_lemmas": 0}
    marked: list[tuple[str, str, str]] = []
    unmarked: dict[str, tuple[str, str]] = {}
    cells: dict[str, frozenset[str]] = {}
    for form, published in carried.items():
        if (form in target or form in excluded_forms
                or has_editorial_sigla(form) or has_editorial_sigla(published)):
            continue
        corrected = breathing_corrections.get(published)
        if corrected is not None:
            if not has_orthographic_mark(form):
                unmarked[form] = (corrected, "corrected_head")
            elif target.get(breathed_spelling(form, published, corrected)) == corrected:
                marked.append((form, corrected, "corrected_head"))
            continue
        lemma = fold_leading_breathing(published)
        if lemma not in cells:
            cells[lemma] = (
                oxytone_os_cells(lemma)
                if (lemma in head_evidence and target.get(lemma) == lemma
                    and published not in superseded_noun_lemmas
                    and lemma not in superseded_noun_lemmas
                    and not ocr_twin_head(lemma, head_evidence))
                else frozenset())
        if form in cells[lemma]:
            if has_orthographic_mark(form):
                marked.append((form, lemma, "oxytone_cell"))
            else:
                unmarked[form] = (lemma, "oxytone_cell")
    for rule in ("oxytone_cell", "corrected_head"):
        rows = [(form, lemma) for form, lemma, kind in marked if kind == rule]
        stats[f"{rule}_entries"] = len(rows)
        stats[f"{rule}_lemmas"] = len({lemma for _form, lemma in rows})
    for form, lemma, _rule in marked:
        target[form] = lemma
    return stats, unmarked


def restore_unmarked_keys(
    reference: dict[str, str],
    reference_base: dict[str, str],
    target: dict[str, str],
    carried: dict[str, str],
    excluded_forms: set[str] | frozenset[str] = frozenset(),
    restored_rows: list[tuple[str, str]] | None = None,
    head_evidence: dict[str, tuple[str, ...]] | None = None,
    breathing_corrections: dict[str, str] | None = None,
    carried_cells: dict[str, tuple[str, str]] | None = None,
) -> dict[str, int]:
    """Give back the unaccented expansion keys the published table carried.

    The historical expansion stored a fully unaccented key beside every
    generated form (αθηλυντος beside ἀθήλυντος); an input typed without
    accents reaches the paradigm only through it. overlay_expansion() never
    copies those keys, because one can shadow a newer reading of a marked
    form with the same letters. The table published from 1.3.7 to 1.3.25
    still carried part of them from earlier tables, and a build from fresh
    dumps has none of those. Here a missing key comes back only where
    ``carried`` (that published table) has it with the expansion's lemma and
    the marked target forms that strip to it read that lemma, so it adds no
    key the published table lacked and agrees with the current table. A key
    no marked form backs, or whose forms read another head, stays out.

    The marked forms (the key's backers) are compared as follows.

    * Two spellings of one lemma agree: a leading spacing breathing is
      folded into its capital (᾿Αγαρηνός is Ἀγαρηνός), and a head the
      breathing review corrected (``breathing_corrections``) is its
      corrected spelling (Ρωμαϊκός is Ῥωμαϊκός). The key is written with
      the spelling its backers read, the normalised one when they differ,
      so a paradigm keeps one lemma spelling across its keys.
    * A grave form whose acute twin is a target key is not a backer of its
      own: the acute form's lemma stands for both (ζημιωθῇς, not the
      generated ζημιωθὴς).
    * With ``head_evidence``, a backer that reads another lemma does not
      dispute a key whose lemma is a confirmed head (confirmed_head) when
      its own lemma is no positive lexical head (ἀπεπνιγῆ under the
      generated ἀποπενιγ͂υμι beside ἀπεπνίγη ἀποπνίγω). A backer that agrees
      always counts.

    ``carried_cells`` are the unaccented keys carry_published_cells()
    returned (``{key: (lemma, rule)}``). They are not in the reference and
    come back under the same test, and only with exactly that lemma
    spelling among their backers.

    The keys are appended to ``target`` in order, and ``restored_rows``
    receives each (key, lemma), so the caller can record a digest that lets
    scripts/build_source_alias_manifest.py remove exactly these rows again.
    """
    corrections = breathing_corrections or {}

    def canonical(lemma: str) -> str:
        lemma = fold_leading_breathing(lemma)
        return corrections.get(lemma, lemma)

    wanted: dict[str, str] = {}
    published: dict[str, str] = {}
    for form, lemma in reference.items():
        if (form in reference_base or has_orthographic_mark(form)
                or form in target or form in excluded_forms
                or carried.get(form) != lemma
                or has_editorial_sigla(form) or has_editorial_sigla(lemma)):
            continue
        wanted[form] = canonical(lemma)
        published[form] = lemma
    rules: dict[str, str] = {}
    for form, (lemma, rule) in (carried_cells or {}).items():
        if form in wanted or form in target or form in excluded_forms:
            continue
        wanted[form] = lemma
        rules[form] = rule
    # bare key -> normalised lemma -> the spellings its backers read
    readings: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set))
    grave_twins = 0
    for form, lemma in target.items():
        if not has_orthographic_mark(form):
            continue
        bare = strip_marks(form)
        if bare not in wanted:
            continue
        acute = grave_to_acute(form)
        if acute != form and acute in target:
            grave_twins += 1
            continue
        readings[bare][canonical(lemma)].add(lemma)
    stats = {"unmarked_candidates": len(wanted), "unmarked_restored": 0,
             "unmarked_disputed": 0, "unmarked_unbacked": 0,
             "unmarked_lemma_respelled": 0,
             "unmarked_oxytone_cell_restored": 0,
             "unmarked_corrected_head_restored": 0,
             "unmarked_nonhead_readings_ignored": 0,
             "unmarked_grave_twins_skipped": grave_twins}
    for bare, lemma in wanted.items():
        current = readings.get(bare, {})
        others = set(current) - {lemma}
        disputing = others
        if head_evidence is not None and confirmed_head(lemma, head_evidence):
            disputing = {other for other in others if other in head_evidence}
        spellings = current.get(lemma, set())
        if bare in rules:
            spellings = spellings & {lemma}
        if spellings and not disputing:
            if lemma not in spellings and len(spellings) == 1:
                (lemma,) = spellings
            target[bare] = lemma
            stats["unmarked_restored"] += 1
            if bare in rules:
                stats[f"unmarked_{rules[bare]}_restored"] += 1
            elif lemma != published[bare]:
                stats["unmarked_lemma_respelled"] += 1
            if others:
                stats["unmarked_nonhead_readings_ignored"] += 1
            if restored_rows is not None:
                restored_rows.append((bare, lemma))
        elif disputing:
            stats["unmarked_disputed"] += 1
        else:
            stats["unmarked_unbacked"] += 1
    return stats


def unmarked_restore_record(rows: list[tuple[str, str]]) -> dict:
    """Count and digest of the unaccented keys appended last to the table.

    The digest covers ``key<TAB>lemma<LF>`` for each row in table order, so
    the alias manifest can check the table's last ``count`` rows against it
    and remove them before it reverses the alias changes.
    """
    hasher = hashlib.sha256()
    for key, lemma in rows:
        hasher.update(f"{key}\t{lemma}\n".encode("utf-8"))
    return {"count": len(rows), "sha256": hasher.hexdigest()}


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


def load_withheld_noun_lemmas(
    paths, superseded_noun_lemmas: set[str] | frozenset[str],
) -> dict[str, tuple[str, ...]]:
    """Superseded nouns that the current generators gave no paradigm.

    The noun audits record, per headword, the genitive text the entry
    prints and the genitive the generator resolved from it. A superseded
    lemma is withheld when it has a record, none of its records resolved a
    genitive, and none rules the headword out as a noun. The value is the
    genitive text its records print, which may be empty.
    """
    records: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        audit = json.loads(Path(path).read_text(encoding="utf-8"))
        for lemma, record in audit.get("resolutions", {}).items():
            if lemma in superseded_noun_lemmas:
                records[lemma].append(record)
    return {
        lemma: tuple(record["supplied"] for record in rows if record["supplied"])
        for lemma, rows in records.items()
        if not any(record["resolved"]
                   or record["reason"] == "nonlexical-noun-headword"
                   for record in rows)
    }


def load_carried_head_records(
    path: Path,
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """Head evidence and breathing corrections for the carried keys.

    Returns the source lists of every positive head, and the corrected
    spelling of each head the Sophocles breathing review corrected.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2 or not payload.get("heads"):
        raise SystemExit("invalid positive lexical-head evidence for carried keys")
    evidence = {head: tuple(sources)
                for head, sources in payload["heads"].items() if sources}
    review = payload.get("sophocles_breathing") or {}
    if review and review.get("schema_version") != 1:
        raise SystemExit("unsupported Sophocles breathing review record")
    corrections = {head: record["head"]
                   for head, record in review.get("corrected", {}).items()}
    return evidence, corrections


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
    parser.add_argument(
        "--carried", type=Path,
        help="Local copy of the pinned published table whose expansion keys "
             "are carried over (otherwise $DILEMMA_AG_EXPANSION_CARRIED, "
             "else downloaded)",
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
    parser.add_argument("--lsj9-dir", type=Path,
                        help="LSJ9 export directory the noun audit pins "
                             "(default: $DILEMMA_LSJ9_DIR, else ~/Documents/lsj9)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.lsj9_dir is not None:
        os.environ["DILEMMA_LSJ9_DIR"] = str(args.lsj9_dir)
    if args.kaikki is not None:
        # build/expand_lsj.py reads the dump directory from the environment
        # when overlay imports it.
        os.environ["DILEMMA_KAIKKI_DIR"] = str(args.kaikki)
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

    carried: dict[str, str] = {}
    if "carried_baseline" in manifest:
        carried_path = args.carried
        if carried_path is None and os.environ.get(CARRIED_BASELINE_ENV):
            carried_path = Path(os.environ[CARRIED_BASELINE_ENV])
        carried_path = resolve_carried_baseline(manifest, carried_path)
        print(f"Loading pinned published table: {carried_path}")
        with carried_path.open(encoding="utf-8") as handle:
            carried = json.load(handle)
        print(f"  {len(carried):,} entries")
    released_forms = carried_baseline_conflicts(
        exclusions["baseline_conflicts"], reference, carried
    ) - frozenset(exclusions["nonlexical"])

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

    # The rules that carry published keys need the published table; a
    # manifest without one runs the plain overlay.
    withheld_nouns: dict[str, tuple[str, ...]] = {}
    head_evidence: dict[str, tuple[str, ...]] = {}
    breathing_corrections: dict[str, str] = {}
    if carried:
        withheld_nouns = load_withheld_noun_lemmas(
            (args.noun_audit, args.sophocles_noun_audit), superseded_nouns)
        head_evidence, breathing_corrections = load_carried_head_records(
            args.lexical_heads)

    stats = overlay_expansion(
        reference, reference_base, target, excluded_forms=excluded_forms,
        superseded_noun_lemmas=superseded_nouns, released_forms=released_forms,
        carried=carried, withheld_noun_lemmas=withheld_nouns,
        lexical_heads=lexical_heads,
    )
    stats["unaccented_head_readings_kept"] = (
        keep_published_over_unaccented_heads(target, carried, lexical_heads))
    cell_stats, carried_cells = carry_published_cells(
        target, carried, head_evidence, breathing_corrections,
        excluded_forms=excluded_forms - released_forms,
        superseded_noun_lemmas=superseded_nouns)
    stats.update(cell_stats)
    alias_stats = restore_attested_compound_aliases(
        reference, reference_base, target, source_pairs,
        lexical_heads,
        excluded_forms=alias_spelling_exclusions(
            exclusions, args.source_pairs / "hunspell_grc_spelling_review.json"),
        historical_conflicts=exclusions["baseline_conflicts"],
    )
    merge_closed_class_elisions(target, alias_stats, closed_class)
    unmarked_rows: list[tuple[str, str]] = []
    stats.update(restore_unmarked_keys(
        reference, reference_base, target, carried,
        excluded_forms=excluded_forms - released_forms,
        restored_rows=unmarked_rows,
        head_evidence=head_evidence if carried else None,
        breathing_corrections=breathing_corrections,
        carried_cells=carried_cells))
    stats["unmarked_withheld_noun_restored"] = sum(
        lemma in withheld_nouns for _key, lemma in unmarked_rows)
    del carried
    alias_stats["historical_overlay_counts"] = stats
    alias_stats["unmarked_key_restore"] = unmarked_restore_record(unmarked_rows)
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
    alias_stats["carried_baseline_sha256"] = manifest.get(
        "carried_baseline", {}).get("sha256")
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
    print(f"Released baseline conflicts: {stats['released_conflict_entries']:,}")
    print(f"Recovered entries: {stats['added']:,}")
    print(f"Published keys of nouns without a fresh paradigm kept: "
          f"{stats['withheld_noun_entries_kept']:,} "
          f"({stats['withheld_noun_lemmas_kept']:,} lemmas, "
          f"{stats['withheld_noun_citation_only_lemmas']:,} citation only)")
    print(f"Published readings kept over accent-less heads: "
          f"{stats['unaccented_head_readings_kept']:,}")
    print(f"Published oxytone -ός cells carried: "
          f"{stats['oxytone_cell_entries']:,} "
          f"({stats['oxytone_cell_lemmas']:,} lemmas)")
    print(f"Published keys moved to breathing-corrected heads: "
          f"{stats['corrected_head_entries']:,} "
          f"({stats['corrected_head_lemmas']:,} lemmas)")
    print(f"Unaccented fallback keys restored: {stats['unmarked_restored']:,} "
          f"({stats['unmarked_disputed']:,} disputed, "
          f"{stats['unmarked_unbacked']:,} without a marked form)")
    print(f"  of them: {stats['unmarked_withheld_noun_restored']:,} of nouns "
          f"without a fresh paradigm, "
          f"{stats['unmarked_oxytone_cell_restored']:,} oxytone -ός cells, "
          f"{stats['unmarked_corrected_head_restored']:,} of breathing-corrected "
          f"heads, {stats['unmarked_lemma_respelled']:,} with a normalised "
          f"lemma spelling, {stats['unmarked_nonhead_readings_ignored']:,} past "
          f"a reading that is no lexical head; "
          f"{stats['unmarked_grave_twins_skipped']:,} grave twins not consulted")
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
