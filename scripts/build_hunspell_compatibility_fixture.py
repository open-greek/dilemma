#!/usr/bin/env python3
"""Pin the reviewed acceptance surface of the dictionary Tonos ships.

The source of truth is the compiled blob the keyboard actually reads, not raw
Hunspell stems: the April export (Dilemma 0.4.1) as Tonos compiled and shipped
it. The downstream candidate gate classifies structural junk, reviewed
must-reject forms, accepted losses, and common-word respellings; the remaining
surface becomes Dilemma's deterministic no-regression contract. Every input is
an explicit argument, because the gate and its review lists live with the
keyboard, not in this repository.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_hunspell_frequency import sha256  # noqa: E402

DEFAULT_OUTPUT = ROOT / "data" / "hunspell_grc_april_compat.json.gz"


def _load_gate(path: Path):
    spec = importlib.util.spec_from_file_location("tonos_dictionary_gate", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load Tonos gate: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_version_sidecar(baseline: Path) -> dict[str, str]:
    """Return the ``key: value`` lines of ``<baseline>.version``."""
    sidecar = baseline.with_suffix(".version")
    fields: dict[str, str] = {}
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        if value.strip():
            fields[key.strip()] = value.strip()
    return fields


def build_payload(
    baseline: Path,
    blob: Path,
    lm: Path,
    gate_path: Path,
    gate_dir: Path,
) -> dict:
    gate = _load_gate(gate_path)
    version = read_version_sidecar(baseline)
    entries = gate.blob_entries(gate.read_blob_bytes(blob))
    counts = gate.read_lm_counts(lm)
    structural_names = {name for name, _rule, _description in gate.STRUCTURAL}
    exceptions = gate.load_forms(gate_dir / "structural_exceptions.txt")
    must_reject = gate.load_forms(gate_dir / "must_reject.txt")
    accepted_losses = gate.load_forms(gate_dir / "accepted_losses.txt")
    respelled = gate.near_duplicates(entries, counts)

    rejected = Counter()
    required: list[str] = []
    for form in entries:
        if form in must_reject:
            rejected["must_reject"] += 1
            continue
        if form in accepted_losses:
            rejected["accepted_loss"] += 1
            continue
        if form in respelled:
            rejected["common_word_respelling"] += 1
            continue
        classes = set(gate.classify(form))
        structural = sorted(classes & structural_names)
        if structural and form not in exceptions:
            rejected[structural[0]] += 1
            continue
        required.append(form)
    required.sort()

    fixture_paths = [
        gate_dir / "structural_exceptions.txt",
        gate_dir / "must_reject.txt",
        gate_dir / "accepted_losses.txt",
    ]
    return {
        "schema_version": 1,
        "baseline": {
            "version": version["version"],
            "commit": version["commit"],
            "dic_sha256": sha256(baseline.with_suffix(".dic")),
            "aff_sha256": sha256(baseline.with_suffix(".aff")),
            "compiled_blob_sha256": sha256(blob),
            "compiled_entries": len(entries),
        },
        "policy": {
            "description": (
                "compiled acceptance of the shipped dictionary minus the "
                "candidate gate's structural junk classes, must-reject "
                "forms, accepted losses, and LM respellings"
            ),
            "gate_sha256": sha256(gate_path),
            "lm_sha256": sha256(lm),
            "review_lists": {
                path.name: sha256(path) for path in fixture_paths
            },
            "rejected_by_class": dict(sorted(rejected.items())),
        },
        "forms": required,
    }


def write_payload(path: Path, payload: dict) -> None:
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, required=True,
        help="path stem of the shipped grc_polytonic .dic/.aff/.version",
    )
    parser.add_argument(
        "--baseline-blob", type=Path,
        help="compiled blob of the baseline (default: <baseline>.dict.lzma)",
    )
    parser.add_argument(
        "--lm", type=Path, required=True,
        help="shipped language model whose counts define respellings",
    )
    parser.add_argument(
        "--gate", type=Path, required=True,
        help="the downstream candidate gate script",
    )
    parser.add_argument(
        "--gate-lists", type=Path,
        help="directory of the gate's review lists (default: a "
             "dictionary_gate directory beside the gate script)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = build_payload(
        args.baseline,
        args.baseline_blob or args.baseline.with_suffix(".dict.lzma"),
        args.lm,
        args.gate,
        args.gate_lists or args.gate.parent / "dictionary_gate",
    )
    write_payload(args.output, payload)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(
        f"wrote {args.output}: {len(payload['forms']):,} required forms, "
        f"sha256={digest}"
    )
    print("rejected: " + json.dumps(
        payload["policy"]["rejected_by_class"], ensure_ascii=False
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
