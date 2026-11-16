#!/usr/bin/env python3
"""Remove the hyphenated strings from data/ag_headwords.json.

The list is a published snapshot of the lemma values of an older Ancient
lookup, and no generator for it is in the repository. It carries the
spellings that table had for divided dictionary heads (συν-αγανακτέω,
κρυσταλλο-ειδής), for AGDT's preverb lemmas (παρά-ἐπιδημέω) and for
line-broken corpus tokens (ἐμπιπλά-ζομαι). None is a citation form
(dilemma.citation.hyphenated_lemma), yet the list vouched for them:
corpus lemma validation in build_lookup_db.py let them through, and the
runtime's compound step accented a hyphenated input from them and returned
the hyphenated string as a lemma (προ-δουλεύοιμι -> προ-δουλεύω).

This script drops every such string and records each one, with the reason
and the hashes of the list before and after, in
data/changes/ag_headwords_hyphenated.json. Nothing is added: the joined
spelling of a removed string is in the list already, or in another headword
list, when it is a word. Run on a list that has no such string, the script
changes nothing.

    python scripts/prune_ag_headwords.py           # prune and record
    python scripts/prune_ag_headwords.py --check   # exit 1 if not pruned
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dilemma.citation import HEAD_HYPHENS, hyphenated_lemma  # noqa: E402

LIST_PATH = ROOT / "data" / "ag_headwords.json"
RECORD_PATH = ROOT / "data" / "changes" / "ag_headwords_hyphenated.json"
RULE = "dilemma.citation.hyphenated_lemma"
REASON = (
    "A string with a hyphen, soft hyphen or dash between two letters is no "
    "Ancient citation form. The list took these from an older lookup's "
    "lemma column: dictionary heads keyed as printed, with the stem divided "
    "(συν-αγανακτέω), AGDT lemmas with the preverb apart (παρά-ἐπιδημέω) "
    "and line-broken corpus tokens (ἐμπιπλά-ζομαι). Listed, they passed "
    "corpus lemma validation and let the runtime return a hyphenated "
    "lemma for a hyphenated input.")


def serialize(headwords) -> bytes:
    """The list as the published file writes it: one line, no escapes."""
    return json.dumps(list(headwords), ensure_ascii=False).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def joined(headword: str) -> str:
    return "".join(char for char in headword if char not in HEAD_HYPHENS)


def prune(headwords):
    """``(kept, removed)``: the list without its hyphenated strings, in the
    same order, and the removed strings."""
    kept = [head for head in headwords if not hyphenated_lemma(head)]
    removed = [head for head in headwords if hyphenated_lemma(head)]
    return kept, removed


def record(source: bytes, headwords, kept, removed) -> dict:
    listed = set(headwords)
    return {
        "schema_version": 1,
        "list": "data/ag_headwords.json",
        "rule": RULE,
        "reason": REASON,
        "source": {"sha256": sha256(source), "size": len(source),
                   "entries": len(headwords)},
        "result": {"sha256": sha256(serialize(kept)),
                   "size": len(serialize(kept)), "entries": len(kept)},
        "removed_count": len(removed),
        "joined_spelling_listed": sum(joined(head) in listed
                                      for head in removed),
        "removed": sorted(removed),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", type=Path, default=LIST_PATH)
    parser.add_argument("--record", type=Path, default=RECORD_PATH)
    parser.add_argument("--check", action="store_true",
                        help="change nothing; exit 1 when the list still has "
                             "a hyphenated string or does not match the record")
    args = parser.parse_args(argv)

    source = args.list.read_bytes()
    headwords = json.loads(source)
    if serialize(headwords) != source:
        print(f"{args.list.name} is not in the published serialization; "
              "refusing to rewrite it", file=sys.stderr)
        return 2
    kept, removed = prune(headwords)
    previous = (json.loads(args.record.read_text(encoding="utf-8"))
                if args.record.exists() else None)

    if not removed:
        if previous is None:
            print(f"{args.list.name}: no hyphenated string, no record")
            return 0
        if previous["result"]["sha256"] != sha256(source):
            print(f"{args.list.name} has no hyphenated string but is not "
                  f"the list {args.record.name} records", file=sys.stderr)
            return 1
        print(f"{args.list.name}: already pruned "
              f"({previous['removed_count']:,} strings recorded)")
        return 0
    if args.check:
        print(f"{args.list.name} has {len(removed):,} hyphenated strings "
              f"(e.g. {removed[0]})", file=sys.stderr)
        return 1

    entry = record(source, headwords, kept, removed)
    args.list.write_bytes(serialize(kept))
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(
        json.dumps(entry, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    print(f"{args.list.name}: {len(headwords):,} -> {len(kept):,} entries, "
          f"{len(removed):,} hyphenated strings removed "
          f"({entry['joined_spelling_listed']:,} with the joined spelling "
          f"listed); recorded in {args.record.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
