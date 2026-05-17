"""Extract form frequencies from the First1KGreek TEI corpus.

First1KGreek (https://github.com/OpenGreekAndLatin/First1KGreek) hosts
~1100 ancient Greek works in TEI XML, primarily authors from the first
millennium that aren't in Perseus's canonical-greekLit. CC BY-SA.

We mostly need it to close the post-classical / late-antique coverage gap
that GLAUx and Diorisis don't reach, so all counts here land in the
'other' bucket — downstream merging into corpus_freq.json then keeps
the genre breakdown that does exist for GLAUx + Diorisis intact and
exposes the new tokens as ungenred totals.

Output: data/first1kgreek_freq.json (same shape as corpus_freq.json,
[total, philosophy, poetry, history, oratory, science, narrative,
epistles, religion, commentary, other] per form).

Usage:
    python build/build_first1kgreek_freq.py
    python build/build_first1kgreek_freq.py --repo /path/to/First1KGreek
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
OUT_PATH = DATA_DIR / "first1kgreek_freq.json"

DEFAULT_REPO = Path.home() / "Documents" / "corpora" / "First1KGreek"

GENRE_ORDER = [
    "philosophy", "poetry", "history", "oratory", "science",
    "narrative", "epistles", "religion", "commentary", "other",
]
DEFAULT_BUCKET = "other"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    p.add_argument("--bucket", default=DEFAULT_BUCKET, choices=GENRE_ORDER)
    args = p.parse_args()

    if not args.repo.exists():
        print(f"error: {args.repo} not found. Clone First1KGreek first:", file=sys.stderr)
        print("  git clone --depth 1 https://github.com/OpenGreekAndLatin/First1KGreek "
              f"{args.repo}", file=sys.stderr)
        return 1

    files = sorted((args.repo / "data").rglob("*-grc*.xml"))
    print(f"Found {len(files):,} Greek TEI files under {args.repo}")

    t0 = time.time()
    total: Counter = Counter()
    for i, f in enumerate(files, 1):
        total.update(tokenize_tei(f))
        if i % 100 == 0:
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
        "_sources": [f"First1KGreek (bucket={args.bucket})"],
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
