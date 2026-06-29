"""Extract form frequencies from the Patristic Text Archive TEI corpus.

PatristicTextArchive (https://github.com/PatristicTextArchive/pta_data)
ships ~200 Greek patristic works in TEI XML, all CC BY-SA. Overlaps the
existing pg_freq.json (Migne) on much of the same author set but uses
more recent critical editions where available.

All counts go in the 'religion' bucket since the corpus is uniformly
patristic. Downstream merging into corpus_freq.json then exposes these
as religion-bucket tokens.

Output: data/pta_freq.json (same 11-element shape as corpus_freq.json).

Usage:
    python build/build_pta_freq.py
    python build/build_pta_freq.py --repo /path/to/pta_data
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from tei_tokenize import tokenize_tei

REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
OUT_PATH = DATA_DIR / "pta_freq.json"

DEFAULT_REPO = Path.home() / "Documents" / "corpus-of-open-greek" / "sources" / "pta"

GENRE_ORDER = [
    "philosophy", "poetry", "history", "oratory", "science",
    "narrative", "epistles", "religion", "commentary", "other",
]
DEFAULT_BUCKET = "religion"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    p.add_argument("--bucket", default=DEFAULT_BUCKET, choices=GENRE_ORDER)
    args = p.parse_args()

    if not args.repo.exists():
        print(f"error: {args.repo} not found. Clone PTA first:", file=sys.stderr)
        print("  git clone --depth 1 https://github.com/PatristicTextArchive/pta_data "
              f"{args.repo}", file=sys.stderr)
        return 1

    from nc_filter import tei_is_noncommercial
    files = sorted((args.repo / "data").rglob("*.pta-grc*.xml"))
    # Openly licensed: drop the per-file NonCommercial PTA texts (e.g. pta0036).
    before = len(files)
    files = [f for f in files if not tei_is_noncommercial(f.read_bytes())]
    print(f"Found {len(files):,} Greek TEI files under "
          f"{args.repo} (excluded {before - len(files)} NonCommercial)")

    t0 = time.time()
    total: Counter = Counter()
    for i, f in enumerate(files, 1):
        total.update(tokenize_tei(f))
        if i % 50 == 0:
            print(f"  {i}/{len(files)}  tokens={sum(total.values()):,}  "
                  f"forms={len(total):,}  elapsed={time.time() - t0:.0f}s")

    bucket_idx = GENRE_ORDER.index(args.bucket)
    forms = {}
    for form, cnt in total.items():
        vec = [0] * (1 + len(GENRE_ORDER))
        vec[0] = cnt
        vec[1 + bucket_idx] = cnt
        forms[form] = vec

    out = {
        "_total_tokens": sum(total.values()),
        "_genres": GENRE_ORDER,
        "_n_forms": len(forms),
        "_sources": [f"PatristicTextArchive (bucket={args.bucket})"],
        "forms": forms,
    }
    DATA_DIR.mkdir(exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"\nWrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.0f} MB)")
    print(f"  Tokens: {sum(total.values()):,}  Forms: {len(forms):,}  "
          f"Time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
