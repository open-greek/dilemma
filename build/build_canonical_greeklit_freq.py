"""Extract form frequencies from Perseus's canonical-greekLit TEI corpus.

canonical-greekLit (https://github.com/PerseusDL/canonical-greekLit)
ships the Perseus Digital Library's TEI editions of classical Greek
authors (Plato, Aristotle, Demosthenes, etc.). CC BY-SA. ~800 works.

Overlaps GLAUx and Diorisis substantially -- both lemmatized the same
Perseus texts -- but the form-count union dedupes those naturally and
still adds previously-unseen forms from works the lemmatizers skipped.
All counts go in the 'other' bucket (the genre breakdown for the
overlapping content already lives in GLAUx/Diorisis entries).

Output: data/canonical_greeklit_freq.json
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
OUT_PATH = DATA_DIR / "canonical_greeklit_freq.json"

# canonical-greekLit now lives under the corpus-of-open-greek (cog) source clones
DEFAULT_REPO = Path.home() / "Documents" / "corpus-of-open-greek" / "sources" / "perseus"

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
        print(f"error: {args.repo} not found. Clone canonical-greekLit first:",
              file=sys.stderr)
        print("  git clone --depth 1 https://github.com/PerseusDL/canonical-greekLit "
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
        "_sources": [f"Perseus canonical-greekLit (bucket={args.bucket})"],
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
