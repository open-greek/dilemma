#!/usr/bin/env python3
"""Test suite for Dilemma Greek lemmatizer.

Tests lookup table integrity, model predictions, and coverage across
Greek varieties. Run after build_data.py and train.py.

Usage:
    python test_dilemma.py                  # all tests
    python test_dilemma.py --lookup-only    # skip model tests
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# The file also supports direct execution from outside the repository.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from build_data import (training_pair_headword_errors, validate_noun_head_inventory,
                        validate_verb_head_inventory, validate_closed_class_head_inventory)


def check_lookup_no_chains(lookup: dict) -> tuple[int, int, list[str]]:
    """Verify no chained lookups: every lemma should map to itself."""
    failures = 0
    total = 0
    examples = []
    for form, lemma in lookup.items():
        if lemma != form:  # skip identity entries
            total += 1
            if lemma in lookup and lookup[lemma] != lemma:
                failures += 1
                if len(examples) < 10:
                    examples.append(f"{form} -> {lemma} -> {lookup[lemma]}")
    return failures, total, examples


def check_lookup_lemmas_are_headwords(lookup: dict) -> tuple[int, int, list[str]]:
    """Verify every lemma in the lookup is itself a headword (maps to itself)."""
    lemmas = set(lookup.values())
    headwords = {k for k, v in lookup.items() if k == v}
    missing = lemmas - headwords
    examples = sorted(list(missing))[:10]
    return len(missing), len(lemmas), examples


def built_closed_class_heads(pairs: list, lookup: dict, data_dir: Path) -> dict:
    """Require published positive evidence for typed non-flat closed heads."""
    if not any(p.get("pos") in {"det", "pron"}
               and p["lemma"] in lookup and lookup[p["lemma"]] != p["lemma"]
               for p in pairs):
        return {}
    return validate_closed_class_head_inventory(
        json.loads((data_dir / "el_closed_class_heads.json").read_text(encoding="utf-8")),
        json.loads((data_dir / "mg_pos_lookup.json").read_text(encoding="utf-8")))



def test_modern_training_pair_heads_have_real_source_evidence():
    """Run the actual pair/flat-lookup invariant in the normal pytest suite."""
    import pytest

    pairs_path, lookup_path = DATA_DIR / "mg_pairs.json", DATA_DIR / "mg_lookup.json"
    if not pairs_path.exists() or not lookup_path.exists():
        pytest.skip("Modern Greek built pairs and lookup are not installed")
    evidence_path = DATA_DIR / "el_noun_heads.json"
    assert evidence_path.exists(), "Missing el_noun_heads.json; rebuild Modern Greek data"
    noun_heads = validate_noun_head_inventory(
        json.loads(evidence_path.read_text(encoding="utf-8")))
    verb_path = DATA_DIR / "el_verb_heads.json"
    assert verb_path.exists(), "Missing el_verb_heads.json; run build/build_el_verb_heads.py"
    verb_heads = validate_verb_head_inventory(json.loads(verb_path.read_text(encoding="utf-8")))
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
    closed_heads = built_closed_class_heads(pairs, lookup, DATA_DIR)
    bad, examples = training_pair_headword_errors(pairs, lookup, noun_heads, verb_heads,
                                                 closed_heads)
    assert bad == 0, f"{bad} unsupported training pair heads: {examples}"

def get_known_pairs() -> list[dict]:
    """Curated test cases across varieties. Returns list of {form, lemma, variety}."""
    return [
        # ── SMG ──
        {"form": "τρέχουν", "lemma": "τρέχω", "variety": "SMG"},
        {"form": "έτρεξε", "lemma": "τρέχω", "variety": "SMG"},
        {"form": "τρομερό", "lemma": "τρομερός", "variety": "SMG"},
        {"form": "πολεμούσαν", "lemma": "πολεμάω", "variety": "SMG"},
        {"form": "ανθρώπων", "lemma": "άνθρωπος", "variety": "SMG"},
        {"form": "γυναίκες", "lemma": "γυναίκα", "variety": "SMG"},
        {"form": "κάθισε", "lemma": "κάθομαι", "variety": "SMG"},  # Wiktionary headword
        {"form": "ποτήρια", "lemma": "ποτήρι", "variety": "SMG"},
        {"form": "φέρνοντας", "lemma": "φέρνω", "variety": "SMG"},  # MG headword (not AG φέρω)

        # ── AG Epic/Homeric ──
        {"form": "θεοί", "lemma": "θεός", "variety": "Epic"},
        {"form": "μῆνιν", "lemma": "μῆνις", "variety": "Epic"},
        {"form": "ἄειδε", "lemma": "ἀείδω", "variety": "Epic"},
        {"form": "Ἀχιλῆος", "lemma": "Ἀχιλλεύς", "variety": "Epic"},
        {"form": "πολύτροπον", "lemma": "πολύτροπος", "variety": "Epic"},
        {"form": "Πηλείδεω", "lemma": "Πηλείδης", "variety": "Epic"},

        # ── AG Attic ──
        {"form": "ἐποίησεν", "lemma": "ποιέω", "variety": "Attic"},
        {"form": "ἔλεγον", "lemma": "λέγω", "variety": "Attic"},
        {"form": "ἔλυσε", "lemma": "λύω", "variety": "Attic"},

        # ── AG Koine ──
        {"form": "ἐβασίλευσεν", "lemma": "βασιλεύω", "variety": "Koine"},

        # ── AG misc ──
        {"form": "σβέννυσι", "lemma": "σβέννυμι", "variety": "AG"},
        {"form": "τριῶν", "lemma": "τρεῖς", "variety": "AG"},
        {"form": "παίδων", "lemma": "παῖς", "variety": "AG"},

        # ── Medieval/Byzantine ──
        # From Swaelens et al. (2024) epigram examples
        {"form": "φλόγα", "lemma": "φλόγα", "variety": "Byzantine"},  # acc self-maps (φλόξ not in lookup)
        {"form": "Αἶνος", "lemma": "Αἶνος", "variety": "Byzantine"},

        # ── Katharevousa ──
        {"form": "ἐξετέλεσεν", "lemma": "ἐκτελέω", "variety": "Katharevousa"},

        # ── Crasis (lookup test only checks raw lookup, not crasis table) ──
        {"form": "τοὔνομα", "lemma": "τοὔνομα", "variety": "Crasis"},  # self-maps in lookup
        {"form": "κἀγώ", "lemma": "κἀγώ", "variety": "Crasis"},       # self-maps in lookup
        {"form": "τἀνδρός", "lemma": "ἀνήρ", "variety": "Crasis"},    # in lookup via form_of
        {"form": "κἄν", "lemma": "κἄν", "variety": "Crasis"},         # self-maps in lookup

        # ── Articles/pronouns ──
        {"form": "τους", "lemma": "τους", "variety": "SMG article/pronoun"},
        {"form": "της", "lemma": "της", "variety": "SMG article/pronoun"},
    ]


def get_e2e_cases() -> list[dict]:
    """End-to-end test cases using the Dilemma class (lookup + crasis + model).

    Returns list of {form, lemma, variety, source} where source indicates
    which resolution path should handle it (lookup, crasis, model).
    """
    return [
        # ── SMG (lookup) ──
        {"form": "τρομερό", "lemma": "τρομερός", "variety": "SMG", "source": "lookup"},
        {"form": "ανθρώπων", "lemma": "άνθρωπος", "variety": "SMG", "source": "lookup"},
        {"form": "εσκόρπισαν", "lemma": "σκορπίζω", "variety": "SMG", "source": "lookup"},
        {"form": "χολωμένο", "lemma": "χολωμένος", "variety": "SMG", "source": "lookup"},

        # ── AG (lookup) ──
        {"form": "μῆνιν", "lemma": "μῆνις", "variety": "Epic", "source": "lookup"},
        {"form": "θεοί", "lemma": "θεός", "variety": "Epic", "source": "lookup"},
        {"form": "ἔλυσε", "lemma": "λύω", "variety": "Attic", "source": "lookup"},
        {"form": "σβέννυσι", "lemma": "σβέννυμι", "variety": "AG", "source": "lookup"},

        # ── Crasis (crasis table) ──
        {"form": "τοὔνομα", "lemma": "ὄνομα", "variety": "Crasis", "source": "crasis"},
        {"form": "κἀγώ", "lemma": "ἐγώ", "variety": "Crasis", "source": "crasis"},
        {"form": "ταὐτός", "lemma": "αὐτός", "variety": "Crasis", "source": "crasis"},
        {"form": "ἅνδρες", "lemma": "ἀνήρ", "variety": "Crasis", "source": "crasis"},

        # ── Model fallback (forms not in lookup) ──
        {"form": "εκαθαρίζονταν", "lemma": "καθαρίζω", "variety": "SMG", "source": "model"},
        {"form": "εφώναξε", "lemma": "φωνάζω", "variety": "SMG", "source": "model"},
    ]


def check_no_greek_junk(pairs: list[dict]) -> tuple[int, int, list[str]]:
    """Check that all training forms contain only Greek characters."""
    failures = 0
    total = len(pairs)
    examples = []
    for p in pairs:
        form = p["form"]
        for ch in form:
            if ch.isalpha():
                cp = ord(ch)
                if not (0x0370 <= cp <= 0x03FF or 0x1F00 <= cp <= 0x1FFF
                        or 0x0300 <= cp <= 0x036F):
                    failures += 1
                    if len(examples) < 10:
                        examples.append(f"{form} (U+{cp:04X} '{ch}')")
                    break
    return failures, total, examples


def run_lookup_tests():
    """Run all lookup/data integrity tests."""
    print("=" * 60)
    print("LOOKUP TABLE TESTS")
    print("=" * 60)

    results = []

    for prefix, name in [("mg", "MG"), ("ag", "AG"), ("med", "Medieval")]:
        path = DATA_DIR / f"{prefix}_lookup.json"
        if not path.exists():
            print(f"\n  SKIP {name}: {path.name} not found")
            continue

        lookup = json.load(open(path, encoding="utf-8"))
        print(f"\n  {name} lookup: {len(lookup):,} entries")

        # Test: no chains
        fail, total, ex = check_lookup_no_chains(lookup)
        status = "PASS" if fail == 0 else "FAIL"
        results.append((status, f"{name} no chains"))
        print(f"  [{status}] No chained lookups: {fail}/{total} failures")
        for e in ex:
            print(f"         {e}")

        # Test: lemmas are headwords
        fail, total, ex = check_lookup_lemmas_are_headwords(lookup)
        status = "PASS" if fail == 0 else "WARN"
        results.append((status, f"{name} lemma headwords"))
        print(f"  [{status}] Lemmas are headwords: {fail}/{total} missing")
        for e in ex[:5]:
            print(f"         {e}")

    # Test: training pair lemmas are headwords (no dirty chains)
    for prefix, name in [("mg", "MG"), ("ag", "AG")]:
        pairs_path = DATA_DIR / f"{prefix}_pairs.json"
        lookup_path = DATA_DIR / f"{prefix}_lookup.json"
        if not pairs_path.exists() or not lookup_path.exists():
            continue
        pairs = json.load(open(pairs_path, encoding="utf-8"))
        lookup = json.load(open(lookup_path, encoding="utf-8"))
        noun_heads, verb_heads, closed_heads = set(), set(), {}
        if prefix == "mg":
            evidence_path = DATA_DIR / "el_noun_heads.json"
            try:
                noun_heads = validate_noun_head_inventory(
                    json.loads(evidence_path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as error:
                results.append(("FAIL", "MG noun-head evidence"))
                print(f"  [FAIL] MG noun-head evidence: {error}")
            try:
                verb_heads = validate_verb_head_inventory(json.loads(
                    (DATA_DIR / "el_verb_heads.json").read_text(encoding="utf-8")))
            except (OSError, ValueError) as error:
                results.append(("FAIL", "MG verb-head evidence"))
                print(f"  [FAIL] MG verb-head evidence: {error}")
            try:
                closed_heads = built_closed_class_heads(pairs, lookup, DATA_DIR)
            except (OSError, ValueError) as error:
                results.append(("FAIL", "MG closed-class head evidence"))
                print(f"  [FAIL] MG closed-class head evidence: {error}")
        bad, bad_ex = training_pair_headword_errors(pairs, lookup, noun_heads, verb_heads,
                                                   closed_heads)
        status = "PASS" if bad == 0 else "FAIL"
        results.append((status, f"{name} training pair lemmas clean"))
        print(f"\n  [{status}] {name} training pair lemmas clean: {bad}/{len(pairs)} dirty")
        for e in bad_ex:
            print(f"         {e}")

    # Test: training data has no non-Greek
    for prefix, name in [("mg", "MG"), ("ag", "AG")]:
        path = DATA_DIR / f"{prefix}_pairs.json"
        if not path.exists():
            continue
        pairs = json.load(open(path, encoding="utf-8"))
        fail, total, ex = check_no_greek_junk(pairs)
        status = "PASS" if fail == 0 else "FAIL"
        results.append((status, f"{name} Greek-only"))
        print(f"\n  [{status}] {name} pairs Greek-only: {fail}/{total} failures")
        for e in ex:
            print(f"         {e}")

    return results


def run_known_pair_tests():
    """Test known form->lemma pairs against the lookup."""
    print("\n" + "=" * 60)
    print("KNOWN PAIR TESTS (lookup)")
    print("=" * 60)

    # Load all lookups
    lookup = {}
    for path in [DATA_DIR / "mg_lookup.json", DATA_DIR / "med_lookup.json",
                 DATA_DIR / "ag_lookup.json"]:
        if path.exists():
            data = json.load(open(path, encoding="utf-8"))
            for k, v in data.items():
                if k not in lookup:
                    lookup[k] = v

    results = []
    cases = get_known_pairs()
    for case in cases:
        form = case["form"]
        expected = case["lemma"]
        variety = case["variety"]

        # Try: exact, lowercase, monotonic, stripped
        from dilemma import to_monotonic, strip_accents
        actual = (lookup.get(form) or lookup.get(form.lower())
                  or lookup.get(to_monotonic(form.lower()))
                  or lookup.get(strip_accents(form.lower())))

        if actual == expected:
            status = "PASS"
        elif actual is None:
            status = "MISS"
        else:
            status = "FAIL"

        results.append((status, f"{form} -> {expected} ({variety})"))
        if status != "PASS":
            print(f"  [{status}] {form} -> expected '{expected}', got '{actual}' ({variety})")

    passed = sum(1 for s, _ in results if s == "PASS")
    print(f"\n  {passed}/{len(results)} passed")
    return results


def run_e2e_tests(scale=None):
    """End-to-end tests using the full Dilemma class."""
    print("\n" + "=" * 60)
    print(f"END-TO-END TESTS (scale={scale})")
    print("=" * 60)

    from dilemma import Dilemma
    d = Dilemma(scale=scale)

    results = []
    cases = get_e2e_cases()

    # Group by variety for display
    from collections import defaultdict
    by_variety = defaultdict(list)
    for case in cases:
        by_variety[case["variety"]].append(case)

    for variety in ["SMG", "Epic", "Attic", "AG", "Crasis", "Byzantine"]:
        group = by_variety.get(variety, [])
        if not group:
            continue
        print(f"\n  {variety}:")
        for case in group:
            form = case["form"]
            expected = case["lemma"]
            source = case["source"]
            actual = d.lemmatize(form)

            if actual == expected:
                status = "PASS"
            else:
                status = "FAIL"

            results.append((status, f"{form} -> {expected} ({variety}/{source})"))
            mark = "" if status == "PASS" else f" (got '{actual}')"
            print(f"    [{status}] {form:20s} -> {expected:20s} [{source}]{mark}")

    passed = sum(1 for s, _ in results if s == "PASS")
    print(f"\n  {passed}/{len(results)} passed")
    return results


def main():
    parser = argparse.ArgumentParser(description="Test Dilemma")
    parser.add_argument("--lookup-only", action="store_true",
                        help="Skip model/e2e tests, only test data/lookup")
    parser.add_argument("--scale", type=int, default=None,
                        help="Scale for e2e tests (default: auto-detect)")
    args = parser.parse_args()

    all_results = []
    all_results.extend(run_lookup_tests())
    all_results.extend(run_known_pair_tests())

    if not args.lookup_only:
        all_results.extend(run_e2e_tests(scale=args.scale))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for s, _ in all_results if s == "PASS")
    failed = sum(1 for s, _ in all_results if s == "FAIL")
    warned = sum(1 for s, _ in all_results if s == "WARN")
    missed = sum(1 for s, _ in all_results if s == "MISS")
    print(f"  PASS: {passed}  FAIL: {failed}  WARN: {warned}  MISS: {missed}")

    if failed > 0:
        print("\nFailed tests:")
        for s, name in all_results:
            if s == "FAIL":
                print(f"  {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
