#!/usr/bin/env python3
"""Build data/cog_public_freq.json from cog's corrected, license-filtered
open-text frequency rollup (corpus-of-open-greek/data/public_lexicon.tsv).

This is the openly-licensed replacement for dilemma's three home-grown
open-text frequency builders (build_first1kgreek_freq / build_pg_freq /
build_canonical_greeklit_freq). cog already computes OCR-corrected,
license-filtered form counts over First1KGreek, the Patrologia Graeca, Perseus
canonical-greekLit, and a Byzantine-historian corpus, keyed by accented surface
form. We fold each form to corpus_freq's accent-stripped lowercase key and sum.

Replacing those three sources (not adding alongside) in merge_corpus_freq.py
avoids double-counting and picks up cog's corrected PG + Byzantine vocabulary.

Input:  ~/Documents/corpus-of-open-greek/data/public_lexicon.tsv  (form<TAB>count)
Output: data/cog_public_freq.json  (consumed by build/merge_corpus_freq.py)
Run:    python build/build_cog_public_freq.py [public_lexicon.tsv]
"""
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_LEXICON = (Path.home() / "Documents" / "corpus-of-open-greek"
                   / "data" / "public_lexicon.tsv")
OUT = DATA_DIR / "cog_public_freq.json"


def _key(s: str) -> str:
    """corpus_freq key: accent-stripped, lowercased (matches the merge keys)."""
    nfd = unicodedata.normalize("NFD", s.lower())
    return unicodedata.normalize(
        "NFC", "".join(c for c in nfd if not unicodedata.combining(c)))


def main() -> int:
    lex = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LEXICON
    if not lex.exists():
        raise SystemExit(f"public_lexicon not found: {lex}")
    forms: dict[str, int] = defaultdict(int)
    total = 0
    for line in lex.open(encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        form, cnt = parts[0], parts[1]
        if not form or not cnt.isdigit():
            continue
        k = _key(form)
        if not k:
            continue
        c = int(cnt)
        forms[k] += c
        total += c
    out = {
        "_total_tokens": total,
        "_n_forms": len(forms),
        "_sources": [f"cog public_lexicon ({total // 1_000_000}M tokens)"],
        "forms": {k: [v] for k, v in forms.items()},
    }
    json.dump(out, OUT.open("w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {OUT} ({len(forms):,} forms, {total:,} tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
