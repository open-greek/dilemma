#!/usr/bin/env python3
"""Merge per-source form frequency files into data/corpus_freq.json.

Sums element-wise [total, philosophy, poetry, history, oratory, science,
narrative, epistles, religion, commentary, other] vectors across every
configured input. Missing inputs are skipped with a warning, so this
runs out-of-the-box on a partial build.

Default sources:
    data/glaux_freq.json              GLAUx (17M, 10 genres)
    data/diorisis_freq.json           Diorisis (10M, 10 genres)
    data/pg_freq.json                 Patrologia Graeca / Migne
    data/first1kgreek_freq.json       First1KGreek
    data/pta_freq.json                Patristic Text Archive
    data/canonical_greeklit_freq.json Perseus canonical-greekLit

The post-GLAUx-Diorisis sources don't carry a genre breakdown, so their
counts all land in their build-time default bucket (PG/PTA in religion,
First1KGreek + canonical-greekLit in other). The genre-aware ranking
that consumers do via GLAUx/Diorisis still works -- they just see extra
ungenred volume from the unlemmatized corpora.

Output: data/corpus_freq.json (drop-in replacement for old GLAUx+Diorisis-only file)

Usage:
    python build/merge_corpus_freq.py
    python build/merge_corpus_freq.py --include glaux diorisis pg
"""

import argparse
import json
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"

GENRE_ORDER = [
    "philosophy", "poetry", "history", "oratory", "science",
    "narrative", "epistles", "religion", "commentary", "other",
]
VEC_LEN = 1 + len(GENRE_ORDER)

# (key, path, label, default_bucket). `default_bucket` is used when the
# source file has no per-genre breakdown (a [total]-only vector) -- the
# full count gets attributed to that bucket. Sources that *do* ship a
# breakdown ignore this field.
SOURCES = [
    ("glaux", DATA_DIR / "glaux_freq.json", "GLAUx", None),
    ("diorisis", DATA_DIR / "diorisis_freq.json", "Diorisis", None),
    ("pg", DATA_DIR / "pg_freq.json", "Patrologia Graeca", "religion"),
    ("first1kgreek", DATA_DIR / "first1kgreek_freq.json", "First1KGreek", "other"),
    ("pta", DATA_DIR / "pta_freq.json", "PatristicTextArchive", "religion"),
    ("canonical_greeklit", DATA_DIR / "canonical_greeklit_freq.json",
     "Perseus canonical-greekLit", "other"),
]

OUTPUT_PATH = DATA_DIR / "corpus_freq.json"


def load_freq(path: Path):
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    total = data.get("_total_tokens", 0)
    forms = data.get("forms", {})
    genres = data.get("_genres", [])
    if genres and genres != GENRE_ORDER:
        print(f"  WARN: genre order mismatch in {path.name}: {genres}")
    return total, forms


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--include",
        nargs="*",
        choices=[s[0] for s in SOURCES],
        default=[s[0] for s in SOURCES],
        help="Subset of sources to merge (default: all available).",
    )
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = p.parse_args()

    selected = [s for s in SOURCES if s[0] in args.include]

    t0 = time.time()
    merged: dict[str, list[int]] = {}
    sources_used: list[str] = []
    total_tokens = 0

    for key, path, label, default_bucket in selected:
        if not path.exists():
            print(f"skip: {path.name} not found")
            continue
        n, forms = load_freq(path)
        print(f"  {label:30s}  {n:>12,} tokens  {len(forms):>8,} forms"
              f"{f'  -> {default_bucket}' if default_bucket else ''}")
        sources_used.append(f"{label} ({n // 1_000_000}M tokens)")
        total_tokens += n
        bucket_offset = (1 + GENRE_ORDER.index(default_bucket)
                         if default_bucket else None)
        for form, vec in forms.items():
            existing = merged.get(form)
            if existing is None:
                existing = [0] * VEC_LEN
                merged[form] = existing
            if len(vec) == VEC_LEN:
                for i in range(VEC_LEN):
                    existing[i] += vec[i]
            else:
                # Short / totals-only vector: add to total + default bucket.
                cnt = vec[0]
                existing[0] += cnt
                if bucket_offset is not None:
                    existing[bucket_offset] += cnt

    print(f"\nMerged forms: {len(merged):,}")
    print(f"Total tokens (sum across sources): {total_tokens:,}")

    print(f"\nGenre distribution:")
    for i, g in enumerate(GENRE_ORDER):
        gt = sum(v[1 + i] for v in merged.values())
        print(f"  {g:15s} {gt:>14,}  ({100 * gt / max(1, total_tokens):>5.2f}%)")

    out = {
        "_total_tokens": total_tokens,
        "_genres": GENRE_ORDER,
        "_n_forms": len(merged),
        "_sources": sources_used,
        "forms": merged,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"\nWrote {args.output} ({args.output.stat().st_size / 1e6:.0f} MB, "
          f"{time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
