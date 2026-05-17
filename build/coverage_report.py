"""Report ag_lookup.json's coverage of a form-frequency corpus.

Loads any *_freq.json file (default: data/corpus_freq.json), checks each
form against the lookup, and prints:

    - exact-match coverage (form present in ag_lookup as-is)
    - relaxed coverage (form present after accent-strip)
    - top-N missing forms by corpus frequency

Useful for identifying the highest-leverage lookup gaps so we can target
elision-form coverage, Byzantine vocabulary, etc., without needing the
licensed TLG corpus.

Output:
- prints summary to stdout
- writes data/coverage_report.json
"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_FREQ = DATA_DIR / "corpus_freq.json"
LOOKUP_PATH = DATA_DIR / "ag_lookup.json"
OUT_PATH = DATA_DIR / "coverage_report.json"
TOP_MISSING = 200


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("freq", nargs="?", type=Path, default=DEFAULT_FREQ,
                   help=f"Frequency file to measure (default: {DEFAULT_FREQ.name})")
    p.add_argument("--lookup", type=Path, default=LOOKUP_PATH)
    p.add_argument("--top", type=int, default=TOP_MISSING)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    args = p.parse_args()

    for path in (args.freq, args.lookup):
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            return 1

    print(f"Loading {args.freq.name} ...")
    freq_data = json.load(args.freq.open())
    forms_raw = freq_data.get("forms", freq_data)
    forms = {f: (v[0] if isinstance(v, list) else v) for f, v in forms_raw.items()
             if not f.startswith("_")}
    total_tokens = sum(forms.values())
    print(f"  {len(forms):,} forms, {total_tokens:,} tokens")

    print(f"Loading {args.lookup.name} ...")
    lookup = json.load(args.lookup.open())
    lookup_keys = set(lookup.keys())
    lookup_stripped = {strip_accents(k) for k in lookup_keys}
    print(f"  {len(lookup_keys):,} keys ({len(lookup_stripped):,} accent-stripped)")

    n_exact = n_relaxed = tok_exact = tok_relaxed = 0
    missing: list[tuple[str, int]] = []
    for form, count in forms.items():
        in_exact = form in lookup_keys
        in_relaxed = in_exact or strip_accents(form) in lookup_stripped
        if in_exact:
            n_exact += 1
            tok_exact += count
        if in_relaxed:
            n_relaxed += 1
            tok_relaxed += count
        else:
            missing.append((form, count))
    missing.sort(key=lambda kv: -kv[1])
    top = missing[:args.top]

    summary = {
        "freq_file": args.freq.name,
        "unique_forms": len(forms),
        "total_tokens": total_tokens,
        "exact_match": {
            "forms_covered": n_exact,
            "forms_pct": round(100 * n_exact / len(forms), 2),
            "tokens_covered": tok_exact,
            "tokens_pct": round(100 * tok_exact / max(1, total_tokens), 2),
        },
        "relaxed_accent_strip": {
            "forms_covered": n_relaxed,
            "forms_pct": round(100 * n_relaxed / len(forms), 2),
            "tokens_covered": tok_relaxed,
            "tokens_pct": round(100 * tok_relaxed / max(1, total_tokens), 2),
        },
        "missing_unique_forms": len(missing),
        "missing_tokens": total_tokens - tok_relaxed,
        "top_missing_by_frequency": [
            {"form": f, "count": c} for f, c in top
        ],
    }
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    em = summary["exact_match"]
    rm = summary["relaxed_accent_strip"]
    print()
    print(f"Coverage of {args.freq.name} by {args.lookup.name}")
    print("-" * 60)
    print(f"  Exact:   {em['forms_covered']:>9,} / {len(forms):>9,} forms "
          f"({em['forms_pct']:>5.2f}%)  "
          f"{em['tokens_covered']:>11,} / {total_tokens:,} tokens ({em['tokens_pct']:>5.2f}%)")
    print(f"  Relaxed: {rm['forms_covered']:>9,} / {len(forms):>9,} forms "
          f"({rm['forms_pct']:>5.2f}%)  "
          f"{rm['tokens_covered']:>11,} / {total_tokens:,} tokens ({rm['tokens_pct']:>5.2f}%)")
    print(f"  Missing: {summary['missing_unique_forms']:,} forms, "
          f"{summary['missing_tokens']:,} tokens")
    print()
    print(f"Top {min(20, len(top))} missing forms by frequency:")
    for f, c in top[:20]:
        print(f"  {c:>10,}  {f}")
    print()
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
