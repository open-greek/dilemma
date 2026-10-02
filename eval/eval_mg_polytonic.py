#!/usr/bin/env python3
"""Coverage of held-out polytonic Modern Greek by the grc_mg_polytonic list.

The word list (``export_mg_polytonic.py``) reads the language model's
TRAINING sentences of the polytonic Modern Greek slice. The language model's
dev split is per sentence, so a dev sentence's document, and its author, also
supply training sentences: the shipping list has seen the very texts the dev
sentences come from, and its coverage of them overstates what it does for a
writer it has never read.

This script measures the share of dev-sentence words the dictionary accepts
(grc alone, grc with the list) under four ways of building the list:

* ``train split``: the shipping list, every document's training sentences;
* ``documents held out``: only documents without any dev sentence, so no
  dev sentence's document contributes (only 16% of the slice is left);
* ``author held out``: authors hashed into ``--folds`` folds; a dev word is
  checked against the list built without its author's fold, so neither its
  document nor anything else by its author contributes;
* ``other fold held out``: the control for the last one, the list built
  without a different fold, so it loses as much data but keeps the author.
  The gap between the two is what the author's own texts give the list.

A word counts as accepted the way Hunspell accepts it: as spelled, or, for a
capitalized word, as its lowercase spelling. Dev sentences from monotonic
documents, and dev sentences with a monotonic signal, are left out, as the
list leaves them out of its sources: their words are not polytonic spellings
to accept.

Usage:
    python eval/eval_mg_polytonic.py
    python eval/eval_mg_polytonic.py --write-lists DIR   # lists for a harness
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import export_mg_polytonic as mg  # noqa: E402
from export_hunspell import load_grc_spelling_review  # noqa: E402


def dev_words(counts: mg.CorpusCounts) -> list[tuple[int, str]]:
    """(document, word) for every plain word of the clean dev sentences of
    the polytonic documents."""
    out: list[tuple[int, str]] = []
    for doc, sentences in counts.dev.items():
        if counts.documents[doc].monotonic:
            continue
        for tokens in sentences:
            if any(mg.monotonic_signal(t) for t in tokens):
                continue
            out.extend((doc, t.form) for t in tokens
                       if t.plain and mg._position(t) is not None)
    return out


def accepted(word: str, words: set[str]) -> bool:
    if word in words:
        return True
    lower = mg._lowercase(word)
    return lower != word and lower in words


def build_list(counts, grc_words, known_word, reviewed, **holdout) -> dict:
    sources = mg.source_documents(counts, **holdout)
    candidates = mg.gather_candidates(counts, sources)
    return mg.select_forms(candidates, grc_words, reviewed, known_word).entries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--json", type=Path, help="write the measurements here")
    ap.add_argument("--write-lists", type=Path, metavar="DIR",
                    help="also write the evaluation lists under DIR")
    args = ap.parse_args()

    counts = mg.count_corpus()
    grc_words = mg.read_grc_words()
    known_word = mg.load_known_words()
    reviewed = load_grc_spelling_review()
    words = dev_words(counts)
    folds = args.folds
    fold_of = {d: mg.author_fold(info.author, folds)
               for d, info in enumerate(counts.documents)}

    lists = {"train split": build_list(counts, grc_words, known_word, reviewed)}
    lists["documents held out"] = build_list(
        counts, grc_words, known_word, reviewed, holdout_dev_documents=True)
    by_fold = {k: build_list(counts, grc_words, known_word, reviewed,
                             holdout_author_fold=(k, folds))
               for k in range(folds)}

    def coverage(choose) -> dict:
        n = covered = by_list = 0
        missing: Counter = Counter()
        for doc, word in words:
            n += 1
            if accepted(word, grc_words):
                covered += 1
                continue
            if accepted(word, choose(doc)):
                covered += 1
                by_list += 1
            else:
                missing[word] += 1
        return {"words": n, "accepted": covered, "share": covered / n,
                "accepted_through_list": by_list,
                "top_missing": missing.most_common(20)}

    empty: set[str] = set()
    results = {
        "grc alone": coverage(lambda doc: empty),
        "train split": coverage(lambda doc: lists["train split"]),
        "documents held out": coverage(lambda doc: lists["documents held out"]),
        "author held out": coverage(lambda doc: by_fold[fold_of[doc]]),
        "other fold held out": coverage(
            lambda doc: by_fold[(fold_of[doc] + 1) % folds]),
    }
    sizes = {"train split": len(lists["train split"]),
             "documents held out": len(lists["documents held out"]),
             "author held out": {k: len(v) for k, v in by_fold.items()}}

    base = results["grc alone"]
    print(f"{base['words']:,} words in the clean dev sentences of "
          f"{len({d for d, _ in words}):,} polytonic documents")
    print(f"{'list':22s} {'accepted':>9s} {'share':>7s} {'via list':>9s} "
          f"{'OOV cut':>8s}")
    for name, r in results.items():
        cut = ((r["accepted"] - base["accepted"])
               / (base["words"] - base["accepted"]))
        print(f"{name:22s} {r['accepted']:9,} {r['share']:7.2%} "
              f"{r['accepted_through_list']:9,} {cut:8.1%}")
    print(f"list sizes: {sizes}")

    if args.json:
        args.json.write_text(json.dumps(
            {"results": results, "sizes": sizes, "folds": folds},
            ensure_ascii=False, indent=1), encoding="utf-8")
    if args.write_lists:
        source = mg.corpus_identity()
        mg.write_list(lists["documents held out"],
                      args.write_lists / "documents_held_out",
                      variant="grc-mg (evaluation: documents with a dev "
                              "sentence held out)", source=source)
        for k, entries in by_fold.items():
            mg.write_list(entries, args.write_lists / f"author_fold_{k}",
                          variant=f"grc-mg (evaluation: author fold {k} of "
                                  f"{folds} held out)", source=source)


if __name__ == "__main__":
    main()
