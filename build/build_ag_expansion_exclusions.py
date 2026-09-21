#!/usr/bin/env python3
"""Build exclusions for the pinned historical AG expansion recovery.

The historical expansion may contain a different lemma for a form already
resolved by the 1.3.1 lookup, or a form/lemma the current nonlexical classifier
rejects. Record those rows once so the normal overlay remains deterministic
without downloading the 1.3.1 lookup on every corpus build.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dilemma.nonlexical import classify_nonlexical  # noqa: E402
from overlay_lsj import (  # noqa: E402
    has_orthographic_mark,
    load_manifest,
    resolve_hf_file,
    resolve_reference,
    verify_file,
    write_json_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data" / "ag_expansion_reference.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data" / "ag_expansion_exclusions.json",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    reference_path = resolve_reference(manifest, None)
    base_path = ROOT / manifest["reference_base"]["path"]
    verify_file(base_path, manifest["reference_base"], "reference base")
    baseline_path = resolve_hf_file(
        manifest["behavior_baseline"], "behavior baseline"
    )

    with reference_path.open(encoding="utf-8") as handle:
        reference = json.load(handle)
    with base_path.open(encoding="utf-8") as handle:
        reference_base = json.load(handle)

    connection = sqlite3.connect(f"file:{baseline_path}?mode=ro", uri=True)
    baseline = dict(connection.execute(
        "SELECT k.form, l.text FROM lookup k "
        "JOIN lemmas l ON l.id = k.lemma_id WHERE k.lang = 'all'"
    ))
    connection.close()

    conflicts: dict[str, str] = {}
    nonlexical: dict[str, str] = {}
    for form, historical_lemma in reference.items():
        if form in reference_base:
            continue
        if not has_orthographic_mark(form):
            continue
        baseline_lemma = baseline.get(form)
        if baseline_lemma is not None and baseline_lemma != historical_lemma:
            conflicts[form] = baseline_lemma
            continue
        reason = classify_nonlexical(form) or classify_nonlexical(
            historical_lemma
        )
        if reason:
            nonlexical[form] = reason

    output = {
        "schema_version": 1,
        "purpose": (
            "Historical expansion forms excluded to preserve the pinned "
            "1.3.1 resolved lookup and current nonlexical classifications."
        ),
        "baseline_conflicts": dict(sorted(conflicts.items())),
        "nonlexical": dict(sorted(nonlexical.items())),
    }
    write_json_atomic(args.output, output)
    print(f"Baseline conflicts: {len(conflicts):,}")
    print(f"Nonlexical rows: {len(nonlexical):,}")
    print(f"Total exclusions: {len(conflicts) + len(nonlexical):,}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
