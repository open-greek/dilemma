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
list, when it is a word.

A second step drops the fragments that rule leaves (#149): strings with a
hyphen or dash at an edge (-γον, ὁτ-) or beside a metrical sign
(–⏑τατος), and strings with a metrical sign (⏑λέων). They are endings and
scansion notes from a source's apparatus, not words. The step records them
in data/changes/ag_headwords_fragments.json the same way. The two records
chain: the second one's source is the first one's result. Run on a list
that has no such string, the script changes nothing.

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
FRAGMENT_RECORD_PATH = ROOT / "data" / "changes" / "ag_headwords_fragments.json"
RULE = "dilemma.citation.hyphenated_lemma"
FRAGMENT_RULE = "scripts/prune_ag_headwords.py:headword_fragment"
# Metrical signs of the Miscellaneous Technical block (⏑ ⏒ ⏓ ⏔ ⏕ ⏖ ...).
METRICAL_SIGNS = frozenset(chr(code) for code in range(0x23D1, 0x23DA))
FRAGMENT_REASON = (
    "A string with a hyphen or dash at an edge (-γον, ὁτ-), or beside a "
    "metrical sign (–⏑τατος), or with a metrical sign at all (⏑λέων), is an "
    "ending or a scansion note from a source's apparatus, not an Ancient "
    "citation form. The hyphenated-lemma rule leaves these, since their "
    "hyphen is not between two letters.")
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


def headword_fragment(headword: str) -> bool:
    """Whether a string has a hyphen or dash anywhere, or a metrical sign:
    after the hyphenated-lemma step, an affix or a scansion fragment."""
    return any(char in HEAD_HYPHENS or char in METRICAL_SIGNS
               for char in headword)


# (rule, reason, predicate, default record path), applied in this order.
STEPS = (
    (RULE, REASON, hyphenated_lemma, RECORD_PATH),
    (FRAGMENT_RULE, FRAGMENT_REASON, headword_fragment, FRAGMENT_RECORD_PATH),
)


def prune(headwords, predicate=hyphenated_lemma):
    """``(kept, removed)``: the list without the strings ``predicate``
    matches (the hyphenated ones by default), in the same order, and the
    removed strings."""
    kept = [head for head in headwords if not predicate(head)]
    removed = [head for head in headwords if predicate(head)]
    return kept, removed


def record(source: bytes, headwords, kept, removed, rule=RULE,
           reason=REASON) -> dict:
    listed = set(headwords)
    return {
        "schema_version": 1,
        "list": "data/ag_headwords.json",
        "rule": rule,
        "reason": reason,
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
    parser.add_argument("--record", type=Path, default=RECORD_PATH,
                        help="record of the hyphenated-lemma step")
    parser.add_argument("--fragment-record", type=Path,
                        default=FRAGMENT_RECORD_PATH,
                        help="record of the fragment step")
    parser.add_argument("--check", action="store_true",
                        help="change nothing; exit 1 when the list still has "
                             "a string to remove or does not match the records")
    args = parser.parse_args(argv)
    records = (args.record, args.fragment_record)

    source = args.list.read_bytes()
    headwords = json.loads(source)
    if serialize(headwords) != source:
        print(f"{args.list.name} is not in the published serialization; "
              "refusing to rewrite it", file=sys.stderr)
        return 2

    current, current_bytes, pending = headwords, source, []
    for (rule, reason, predicate, _default), path in zip(STEPS, records):
        kept, removed = prune(current, predicate)
        if not removed:
            continue
        if args.check:
            print(f"{args.list.name} has {len(removed):,} strings for "
                  f"{rule} (e.g. {removed[0]})", file=sys.stderr)
            return 1
        entry = record(current_bytes, current, kept, removed, rule, reason)
        pending.append((path, entry))
        current, current_bytes = kept, serialize(kept)

    if not pending:
        previous = [json.loads(path.read_text(encoding="utf-8"))
                    for path in records if path.exists()]
        if not previous:
            print(f"{args.list.name}: nothing to remove, no record")
            return 0
        if previous[-1]["result"]["sha256"] != sha256(source):
            print(f"{args.list.name} has nothing to remove but is not the "
                  f"list the records end with", file=sys.stderr)
            return 1
        print(f"{args.list.name}: already pruned ("
              + ", ".join(f"{entry['removed_count']:,} strings for "
                          f"{entry['rule']}" for entry in previous) + ")")
        return 0

    args.list.write_bytes(current_bytes)
    for path, entry in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entry, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
        print(f"{args.list.name}: {entry['source']['entries']:,} -> "
              f"{entry['result']['entries']:,} entries, "
              f"{entry['removed_count']:,} strings removed by {entry['rule']} "
              f"({entry['joined_spelling_listed']:,} with the joined spelling "
              f"listed); recorded in {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
