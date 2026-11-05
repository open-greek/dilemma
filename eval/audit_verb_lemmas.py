#!/usr/bin/env python3
"""Compare public lemma APIs on source-reviewed words in real passages.

Run from a checkout with the usual Dilemma data available:

    python eval/audit_verb_lemmas.py --output verb-lemma-audit.json

The fixture preserves passage tokens and source references separately from
reviewed expectations. Corpus annotations are evidence, not an automatic
gold standard. Accepted citation alternatives prevent dictionary conventions
from being counted as errors. This is a regression audit, not a held-out
accuracy benchmark. No model guesses are used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "eval" / "fixtures" / "verb_lemma_passages.json"


def _contains(actual, expected):
    """Match reviewed fields while allowing additional source metadata."""
    return all(actual.get(key) == value for key, value in expected.items())


def audit_case(engine, case):
    tokens = case["tokens"]
    index = case["target_index"]
    word = tokens[index]
    tags = ["X"] * len(tokens)
    tags[index] = "VERB"
    verbose = engine.lemmatize_verbose(word, guess=False)
    results = {
        "scalar": engine.lemmatize(word, guess=False),
        "batch": engine.lemmatize_batch(tokens, guess=False)[index],
        "pos": engine.lemmatize_pos(word, "VERB", guess=False),
        "batch_pos": engine.lemmatize_batch_pos(tokens, tags, guess=False)[index],
    }
    issues = []
    accepted = set(case["accepted_lemmas"])
    for name, lemma in results.items():
        if lemma not in accepted:
            issues.append({"api": name, "reason": "unsupported_lemma", "actual": lemma})
    if results["scalar"] != results["batch"]:
        issues.append({"api": "batch", "reason": "differs_from_scalar"})
    if results["pos"] != results["batch_pos"]:
        issues.append({"api": "batch_pos", "reason": "differs_from_pos"})
    if not any(candidate.lemma in accepted for candidate in verbose):
        issues.append({"api": "verbose", "reason": "missing_accepted_lemma"})

    grammar = [candidate.grammar for candidate in verbose if candidate.grammar]
    for reading in grammar:
        if reading["base_lemma"] in case.get("forbidden_bases", []):
            issues.append({"api": "verbose", "reason": "wrong_homonym_analysis",
                           "lemma": reading["lemma"], "base_lemma": reading["base_lemma"]})
    for expected in case.get("required_readings", []):
        matches = [reading for reading in grammar if reading["lemma"] == expected["lemma"]]
        if not matches:
            issues.append({"api": "verbose", "reason": "missing_grammatical_reading",
                           "lemma": expected["lemma"]})
            continue
        analyses = [analysis for reading in matches for analysis in reading["analyses"]]
        for analysis in expected["analyses"]:
            if not any(_contains(actual, analysis) for actual in analyses):
                issues.append({"api": "verbose", "reason": "missing_analysis",
                               "lemma": expected["lemma"], "analysis": analysis})
        # The fixture reviews the complete person/number/mood/tense set for
        # these forms. A source-backed alternative must not conceal an
        # additional unsupported parse.
        for analysis in analyses:
            if not any(_contains(analysis, reviewed) for reviewed in expected["analyses"]):
                issues.append({"api": "verbose", "reason": "unsupported_analysis",
                               "lemma": expected["lemma"], "analysis": analysis})

    return {"id": case["id"], "word": word, "source": case["source"],
            "accepted_lemmas": case["accepted_lemmas"], "results": results,
            "verbose": [asdict(candidate) for candidate in verbose], "issues": issues}


def audit_cases(engine, cases):
    rows = [audit_case(engine, case) for case in cases]
    reasons = Counter(issue["reason"] for row in rows for issue in row["issues"])
    return {
        "schema_version": 1,
        "summary": {"cases": len(rows), "spellings": len({row["word"].lower() for row in rows}),
                    "passed": sum(not row["issues"] for row in rows),
                    "failed": sum(bool(row["issues"]) for row in rows),
                    "reasons": dict(sorted(reasons.items()))},
        "cases": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lang", choices=("grc", "all"), default="grc")
    args = parser.parse_args(argv)
    payload = args.cases.read_bytes()
    fixture = json.loads(payload)
    sys.path.insert(0, str(ROOT))
    from dilemma import Dilemma

    # POS tables stay enabled. skip_pos=True would silently test fallback
    # lookup instead of the POS disambiguation path this audit is checking.
    engine = Dilemma(lang=args.lang, convention=None, normalize=False,
                     resolve_articles=False, skip_pos=False)
    result = audit_cases(engine, fixture["cases"])
    result["inputs"] = {"cases_sha256": hashlib.sha256(payload).hexdigest(),
                        "language": args.lang, "pos_tables_enabled": True,
                        "pos_entries": len(engine._pos_lookup),
                        "ancient_pos_entries": len(engine._pos_ag_lookup),
                        "fixture_sources": fixture["sources"]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 1 if result["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
