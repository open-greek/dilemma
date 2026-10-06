#!/usr/bin/env python3
"""Count the spellings of the language model's held-out sentences.

The grc Hunspell export admits a spelling on GLAUx or Diorisis counts when no
lemma source proposes it, and takes a Modern-Greek-only lookup row the same
way. Those counts leave out the held-out GLAUx and Diorisis sentences of
``train_lm.py``'s dev split, so the sentences a keyboard is measured on do
not vouch for the words it is measured on. The export reads the counts from
``data/hunspell_grc_lm_dev_spellings.json.gz``, which this script writes from
``build/lm/dev_sentences_{glaux,diorisis}.txt``; rebuild it after
``train_lm.py`` draws a new split. It holds spelling counts and the source
files' sha256, not sentences.

    python scripts/build_hunspell_dev_spellings_fixture.py
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from export_hunspell import (  # noqa: E402
    LM_DEV_SENTENCES,
    LM_DEV_SPELLINGS,
    count_dev_spellings,
)


def build_payload(paths: tuple[Path, ...] = LM_DEV_SENTENCES) -> dict:
    sources = {}
    for path in paths:
        data = path.read_bytes()
        sources[path.name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "sentences": data.count(b"\n"),
        }
    counts = count_dev_spellings(paths)
    return {
        "schema_version": 1,
        "description": (
            "NFC spelling counts of the language model's held-out GLAUx and "
            "Diorisis sentences (train_lm.py dev split)"
        ),
        "sources": sources,
        "tokens": sum(counts.values()),
        "forms": dict(sorted(counts.items())),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=LM_DEV_SPELLINGS)
    args = ap.parse_args()
    missing = [str(p) for p in LM_DEV_SENTENCES if not p.exists()]
    if missing:
        print(f"ERROR: {', '.join(missing)} not found; run train_lm.py",
              file=sys.stderr)
        return 1
    payload = build_payload()
    encoded = json.dumps(payload, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    with args.out.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                           mtime=0) as stream:
            stream.write(encoded)
    print(f"wrote {args.out}: {len(payload['forms']):,} spellings, "
          f"{payload['tokens']:,} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
