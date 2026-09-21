#!/usr/bin/env python3
"""Restore the pinned historical LSJ/Sophocles expansion layer.

``build_data.py`` regenerates ``ag_lookup.json`` from Wiktionary and therefore
removes the expensive LSJ/Sophocles paradigm expansion. The expansion is
recoverable from a known-good historical lookup, but the whole historical
lookup must not be overlaid blindly: its base mappings may have been corrected
since it was built.

This script derives only ``reference - reference_base`` and adds missing keys
to the current target. Existing target mappings always win. The reference and
its exact base are byte-pinned in ``data/ag_expansion_reference.json``.

Usage:
    python overlay_lsj.py
    python overlay_lsj.py --dry-run
    python overlay_lsj.py --reference /path/to/pinned/ag_lookup.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import unicodedata
from pathlib import Path

from dilemma.form_sanitize import has_editorial_sigla, resolve_editorial_form


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "data" / "ag_expansion_reference.json"
DEFAULT_TARGET = ROOT / "data" / "ag_lookup.json"
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
        target[form] = lemma
        stats["added"] += 1
    stats["target_entries_after"] = len(target)
    return stats


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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

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

    target_editorial_stats = resolve_target_editorial_entries(target)
    print(
        "Resolved target editorial notation: "
        f"{target_editorial_stats['target_movable_nu_added']:,} movable-nu "
        f"spellings added; "
        f"{target_editorial_stats['target_editorial_removed']:,} notation "
        "entries removed"
    )

    stats = overlay_expansion(
        reference, reference_base, target, excluded_forms=excluded_forms
    )
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
    print(f"Recovered entries: {stats['added']:,}")
    print(f"Final size: {stats['target_entries_after']:,}")

    if args.dry_run:
        print("Dry run: target was not changed")
        return
    print(f"Saving atomically to {args.target}...")
    write_json_atomic(args.target, target)
    print(f"  {args.target.stat().st_size / (1024 * 1024):.1f} MB written")


if __name__ == "__main__":
    main()
