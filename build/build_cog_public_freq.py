#!/usr/bin/env python3
"""Build data/cog_public_freq.json from cog's corrected, license-filtered
open-text frequency rollup (open-greek-corpus/data/public_lexicon.tsv).

This is the openly-licensed replacement for dilemma's three home-grown
open-text frequency builders (build_first1kgreek_freq / build_pg_freq /
build_canonical_greeklit_freq). cog already computes OCR-corrected,
license-filtered form counts over First1KGreek, the Patrologia Graeca, Perseus
canonical-greekLit, and a Byzantine-historian corpus, keyed by accented surface
form. We fold each form to corpus_freq's accent-stripped lowercase key and sum.

Replacing those three sources (not adding alongside) in merge_corpus_freq.py
avoids double-counting and picks up cog's corrected PG + Byzantine vocabulary.

Input:  ~/Documents/open-greek-corpus/data/public_lexicon.tsv  (form<TAB>count)
Output: data/cog_public_freq.json  (consumed by build/merge_corpus_freq.py)
Run:    python build/build_cog_public_freq.py [public_lexicon.tsv]
"""
import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from corpus_freq_key import corpus_freq_key

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_LEXICON = (Path.home() / "Documents" / "open-greek-corpus"
                   / "data" / "public_lexicon.tsv")
OUT = DATA_DIR / "cog_public_freq.json"


def _key(s: str) -> str:
    """corpus_freq key: accent-stripped, lowercased (matches the merge keys)."""
    return corpus_freq_key(s)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path.parent), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lexicon", nargs="?", type=Path, default=DEFAULT_LEXICON)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument(
        "--ogc-commit",
        help="override the OGC git commit auto-detected from the lexicon path",
    )
    args = parser.parse_args()
    lex = args.lexicon
    if not lex.exists():
        raise SystemExit(f"public_lexicon not found: {lex}")
    commit = args.ogc_commit or _git_commit(lex)
    if not commit:
        raise SystemExit(
            "could not determine the OGC commit; pass --ogc-commit explicitly"
        )
    lexicon_sha256 = _sha256(lex)
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
        "_sources": [
            f"OGC public_lexicon ({total // 1_000_000}M tokens; "
            f"commit={commit}; sha256={lexicon_sha256})"
        ],
        "_source_provenance": {
            "ogc_commit": commit,
            "public_lexicon_sha256": lexicon_sha256,
        },
        "forms": {k: [v] for k, v in forms.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, args.output.open("w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {args.output} ({len(forms):,} forms, {total:,} tokens)")
    print(f"  OGC commit: {commit}")
    print(f"  lexicon sha256: {lexicon_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
