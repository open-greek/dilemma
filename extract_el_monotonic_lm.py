#!/usr/bin/env python3
"""Extract the openly licensed monotonic Modern Greek corpus into compact
shards for the monotonic next-word model and the monotonic word list.

Reads every enabled source of ``data/el_lm_corpora.json`` (pinned inputs,
verified by sha256), normalizes, filters, sentence-splits and tokenizes
each document with ``build/el_text.py``, and writes::

    build/el_corpus/manifest.json                 counts per source and split
    build/el_corpus/<source>/train-00000.jsonl.gz shards (see build/el_corpus.py)
    build/el_corpus/<source>/dev-00000.jsonl.gz
    build/el_corpus/<source>/abbreviations.tsv.gz            train counts
    build/el_corpus/<source>/abbreviation_candidates.tsv.gz  words before a
        period that a lowercase word follows (train)
    build/el_corpus/<source>/homoglyph_repairs.tsv.gz        audit of repairs

Split and leakage
-----------------
Documents go to dev by a hash of their group (sitting, consultation,
article, entry; ``el_corpus.split_of``). The run makes two passes. The dev
pass collects the case-folded key of every kept dev sentence of every
source; the train pass then drops each training sentence whose key is a dev
key, so repeated speeches, campaign comments and quotations cannot leak. In
both splits a sentence is kept at most ``max_sentence_repeats`` times per
source, which bounds procedural formulas and copy-pasted comments.

Caps and weights
----------------
Every training document is written; the token cap of a source is recorded
as a bucket threshold over ``el_corpus.cap_bucket`` (a fixed, document-level
sample), and ``el_corpus.iter_corpus_sentences`` applies it. Weights are
recorded for the trainer. Neither changes the shards, so caps can be tuned
without extracting again.

Usage
-----
    python extract_el_monotonic_lm.py --download     # fetch missing inputs
    python extract_el_monotonic_lm.py --sanity       # 2,000 documents per
                                                     # source, into
                                                     # build/el_corpus_sanity
    python extract_el_monotonic_lm.py                # full run
    python extract_el_monotonic_lm.py --sources tatoeba,common_voice \\
        --out-dir build/el_corpus_small

A run with ``--sanity``, ``--max-docs`` or ``--sources`` is refused in
``build/el_corpus``, so a partial run cannot replace the full corpus.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

from dilemma_build import el_corpus, el_text

ROOT = Path(__file__).resolve().parent

SANITY_DOCS = 2000
BATCH_CHARS = 2_000_000
BATCH_DOCS = 5000

_REGISTER_INDEX = {w: i for i, w in enumerate(el_text.REGISTER_WORDS)}
# Repeat caps whose effect the manifest reports, so the cap can be chosen
# from measurements without extracting again.
REPEAT_CAP_PROBES = (1, 2, 3, 5, 10, 20, 50, 100)
TOP_REPEATS = 25


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------

def _process_batch(args):
    """Process a batch of documents in a worker.

    Returns per-document results and the batch's merged statistics.
    """
    options, docs = args
    stats: Counter = Counter()
    repairs: Counter = Counter()
    candidates: Counter = Counter()
    out = []
    for doc_id, group, text in docs:
        pd = el_text.process_document(text, **options)
        stats.update(pd.stats)
        repairs.update(pd.homoglyph_repairs)
        candidates.update(pd.abbreviation_candidates)
        sents = []
        for toks in pd.sentences:
            words = 0
            num = foreign = abbr = 0
            reg = None
            for t in toks:
                if t == el_text.NUM_TOKEN:
                    num += 1
                elif t == el_text.FOREIGN_TOKEN:
                    foreign += 1
                elif t.endswith("."):
                    abbr += 1
                else:
                    words += 1
                    r = _REGISTER_INDEX.get(el_text.lower_el(t))
                    if r is not None:
                        reg = (reg or ()) + (r,)
            sents.append((" ".join(toks), el_text.sentence_key(toks),
                          len(toks), words, num, foreign, abbr, reg))
        out.append((doc_id, group, sents))
    return out, stats, repairs, candidates


# --------------------------------------------------------------------------
# Main process helpers
# --------------------------------------------------------------------------

def git_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT, stderr=subprocess.DEVNULL).decode()
    except Exception:
        return "unknown"
    return f"{commit}-dirty" if status.strip() else commit


def _batches(docs, wanted_split, dev_fraction, read_counter):
    batch, chars = [], 0
    for doc in docs:
        read_counter["documents"] += 1
        if el_corpus.split_of(doc.group, dev_fraction) != wanted_split:
            continue
        batch.append((doc.id, doc.group, doc.text))
        chars += len(doc.text)
        if chars >= BATCH_CHARS or len(batch) >= BATCH_DOCS:
            yield batch
            batch, chars = [], 0
    if batch:
        yield batch


def write_tsv_gz(path: Path, counter: Counter) -> str:
    """``key \\t count`` lines, most frequent first, ties by key; byte-stable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        for k, c in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            key = "\t".join(k) if isinstance(k, tuple) else k
            gz.write(f"{key}\t{c}\n".encode("utf-8"))
    return el_corpus.file_sha256(path)


def _empty_split():
    return {"documents": 0, "sentences": 0, "tokens": 0, "words": 0,
            "num": 0, "foreign": 0, "abbreviations": 0,
            "dropped_repeats": {"sentences": 0, "tokens": 0},
            "dropped_dev_duplicates": {"sentences": 0, "tokens": 0}}


def run_pass(src, paths, split, pool, args, dev_keys, state):
    """Run one split of one source; updates ``state`` in place."""
    key = src["key"]
    dev_fraction = src.get("dev_fraction", state["split_fraction"])
    options = el_corpus.text_options(src)
    max_repeats = state["max_repeats"]
    writer = el_corpus.ShardWriter(args.out_dir / key, split, args.shard_tokens)
    acc = state["sources"][key]["splits"][split]
    repeats: dict[int, int] = {}
    repeated_len: dict[int, int] = {}      # tokens of sentences seen twice
    repeated_text: dict[int, str] = {}
    reg = state["register"][key]
    read = Counter()
    docs = el_corpus.iter_documents(src, paths, args.max_docs)
    batches = ((options, b) for b in _batches(docs, split, dev_fraction, read))
    t0 = time.time()
    for out, stats, repairs, candidates in pool.imap(_process_batch, batches):
        state["filters"][key].update(stats)
        state["repairs"][key].update(repairs)
        if split == "train":
            state["candidates"][key].update(candidates)
        for doc_id, group, sents in out:
            kept = []
            doc_tokens = 0
            for (s, skey, ntok, words, num, foreign, abbr, rhits) in sents:
                if split == "train" and skey in dev_keys:
                    acc["dropped_dev_duplicates"]["sentences"] += 1
                    acc["dropped_dev_duplicates"]["tokens"] += ntok
                    continue
                seen = repeats.get(skey, 0)
                repeats[skey] = seen + 1
                if seen == 1:
                    repeated_len[skey] = ntok
                    repeated_text[skey] = s
                if seen >= max_repeats:
                    acc["dropped_repeats"]["sentences"] += 1
                    acc["dropped_repeats"]["tokens"] += ntok
                    continue
                if split == "dev":
                    dev_keys.add(skey)
                kept.append(s)
                doc_tokens += ntok
                acc["sentences"] += 1
                acc["tokens"] += ntok
                acc["words"] += words
                acc["num"] += num
                acc["foreign"] += foreign
                acc["abbreviations"] += abbr
                if split == "train":
                    if rhits:
                        for r in rhits:
                            reg[r] += 1
                    if abbr:
                        for t in s.split(" "):
                            if t.endswith("."):
                                state["abbreviations"][key][t] += 1
                    state["cap_hist"][key][el_corpus.cap_bucket(doc_id)] += ntok
            if kept:
                writer.write(doc_id, group, kept, doc_tokens)
                acc["documents"] += 1
    acc["shards"] = writer.close()
    probes = {}
    for cap in REPEAT_CAP_PROBES:
        probes[str(cap)] = sum((repeats[k] - cap) * n
                               for k, n in repeated_len.items()
                               if repeats[k] > cap)
    acc["repeat_cap_tokens_removed"] = probes
    top = sorted(repeated_len, key=lambda k: (-repeats[k], repeated_text[k]))
    acc["most_repeated"] = [[repeats[k], repeated_text[k]]
                            for k in top[:TOP_REPEATS]]
    state["sources"][key]["documents_read"][split] = read["documents"]
    print(f"  {key} {split}: {read['documents']:,} documents read, "
          f"{acc['documents']:,} kept, {acc['sentences']:,} sentences, "
          f"{acc['words']:,} words ({time.time() - t0:.0f} s)", flush=True)


class _InProcess:
    """A stand-in for a pool that maps in this process (``--workers 0``)."""

    def imap(self, fn, iterable):
        return map(fn, iterable)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _pool(workers: int):
    if workers <= 0:
        return _InProcess()
    return get_context("spawn").Pool(workers)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=el_corpus.DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--sources", default=None,
                    help="comma-separated source keys (default: all enabled)")
    ap.add_argument("--max-docs", type=int, default=None,
                    help="read at most this many documents per source")
    ap.add_argument("--sanity", action="store_true",
                    help=f"same as --max-docs {SANITY_DOCS}")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="worker processes; 0 processes in this process")
    ap.add_argument("--shard-tokens", type=int, default=20_000_000)
    ap.add_argument("--download", action="store_true",
                    help="download pinned inputs that are missing")
    args = ap.parse_args(argv)

    if args.sanity and args.max_docs is None:
        args.max_docs = SANITY_DOCS
    partial = args.max_docs is not None or args.sources is not None
    if args.out_dir is None:
        args.out_dir = (el_corpus.DEFAULT_OUT_DIR.with_name("el_corpus_sanity")
                        if partial else el_corpus.DEFAULT_OUT_DIR)
    if partial and args.out_dir.resolve() == el_corpus.DEFAULT_OUT_DIR.resolve():
        ap.error(f"a partial run may not write {el_corpus.DEFAULT_OUT_DIR}")

    manifest = el_corpus.load_manifest(args.manifest)
    sources = [s for s in manifest["sources"] if s.get("enabled", True)]
    if args.sources:
        wanted = args.sources.split(",")
        unknown = set(wanted) - {s["key"] for s in sources}
        if unknown:
            ap.error(f"unknown or disabled sources: {sorted(unknown)}")
        sources = [s for s in sources if s["key"] in wanted]

    print(f"Resolving inputs for {len(sources)} sources", flush=True)
    inputs = {s["key"]: el_corpus.resolve_inputs(s, download=args.download)
              for s in sources}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "split_fraction": manifest["split"]["dev_fraction"],
        "max_repeats": manifest["max_sentence_repeats"],
        "sources": {s["key"]: {"splits": {"dev": _empty_split(),
                                          "train": _empty_split()},
                               "documents_read": {}} for s in sources},
        "filters": {s["key"]: Counter() for s in sources},
        "repairs": {s["key"]: Counter() for s in sources},
        "candidates": {s["key"]: Counter() for s in sources},
        "abbreviations": {s["key"]: Counter() for s in sources},
        "register": {s["key"]: [0] * len(el_text.REGISTER_WORDS) for s in sources},
        "cap_hist": {s["key"]: [0] * el_corpus.BUCKETS for s in sources},
    }
    dev_keys: set[int] = set()
    t0 = time.time()
    with _pool(args.workers) as pool:
        print("Pass 1: dev documents", flush=True)
        for src in sources:
            run_pass(src, inputs[src["key"]], "dev", pool, args, dev_keys, state)
        print(f"Pass 2: train documents ({len(dev_keys):,} dev sentence keys)",
              flush=True)
        for src in sources:
            run_pass(src, inputs[src["key"]], "train", pool, args, dev_keys, state)

    out_sources = {}
    totals = {"dev": Counter(), "train": Counter()}
    under_caps = Counter()
    for src in sources:
        key = src["key"]
        entry = state["sources"][key]
        side = {
            "abbreviations": write_tsv_gz(
                args.out_dir / key / "abbreviations.tsv.gz",
                state["abbreviations"][key]),
            "abbreviation_candidates": write_tsv_gz(
                args.out_dir / key / "abbreviation_candidates.tsv.gz",
                state["candidates"][key]),
            "homoglyph_repairs": write_tsv_gz(
                args.out_dir / key / "homoglyph_repairs.tsv.gz",
                state["repairs"][key]),
        }
        train = entry["splits"]["train"]
        threshold, kept = el_corpus.cap_threshold(
            state["cap_hist"][key], src.get("token_cap"))
        words = max(train["words"], 1)
        register = {w: round(state["register"][key][i] * 1e6 / words, 1)
                    for i, w in enumerate(el_text.REGISTER_WORDS)}
        out_sources[key] = {
            "title": src.get("title"),
            "license": src["license"],
            "role": src["role"],
            "register": src.get("register"),
            "weight": src.get("weight", 1),
            "dev_fraction": src.get("dev_fraction", state["split_fraction"]),
            "inputs": [{"file": p.name, "sha256": pin["sha256"]}
                       for p, pin in zip(inputs[key], (
                           src["fetch"]["files"] if "files" in src["fetch"]
                           else [src["fetch"]]))],
            "documents_read": entry["documents_read"],
            "splits": entry["splits"],
            "cap": {"token_cap": src.get("token_cap"),
                    "bucket_threshold": threshold,
                    "train_tokens_under_cap": kept},
            "register_per_million_words": register,
            "filters": dict(sorted(state["filters"][key].items())),
            "side_tables_sha256": side,
        }
        for split in ("dev", "train"):
            s = entry["splits"][split]
            for f in ("documents", "sentences", "tokens", "words"):
                totals[split][f] += s[f]
        under_caps["tokens"] += kept
        under_caps["weighted_tokens"] += kept * src.get("weight", 1)

    result = {
        "schema_version": 1,
        "created_by": "extract_el_monotonic_lm.py",
        "commit": git_commit(),
        "corpus_manifest": str(args.manifest.resolve().relative_to(ROOT))
        if args.manifest.resolve().is_relative_to(ROOT) else str(args.manifest),
        "corpus_manifest_sha256": el_corpus.file_sha256(args.manifest),
        "el_text_sha256": el_corpus.file_sha256(ROOT / "build" / "el_text.py"),
        "max_docs_per_source": args.max_docs,
        "split": {"seed": el_corpus.SPLIT_SEED, "cap_seed": el_corpus.CAP_SEED,
                  "buckets": el_corpus.BUCKETS,
                  "default_dev_fraction": state["split_fraction"],
                  "max_sentence_repeats": state["max_repeats"],
                  "dev_sentence_keys": len(dev_keys)},
        "placeholders": sorted(el_text.PLACEHOLDERS),
        "totals": {"dev": dict(totals["dev"]), "train": dict(totals["train"]),
                   "train_under_caps": dict(under_caps)},
        "sources": out_sources,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_dir / 'manifest.json'} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
