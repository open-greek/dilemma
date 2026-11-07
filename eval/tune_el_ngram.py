#!/usr/bin/env python3
"""Size against quality of the monotonic Modern Greek next-word model, over
a grid of pruning settings and training-set sizes.

For each training size (a document-level sample of the capped mixture, or
all of it), the corpus is counted once (``el_ngram.count_corpus``); then for
every pruning setting the intermediates are written, exported with
``export_lm.py --lang el`` and measured with ``eval/eval_el_ngram.py`` on a
fixed sample of the dev split. The pruning knobs are those of the Ancient
Greek trainer: ``min_count_bi`` (bigram entries seen fewer times go),
``min_count_tri`` (likewise trigrams) and ``min_bigram_for_tri`` (trigram rows
whose context was seen fewer times go).

Usage
-----
    python eval/tune_el_ngram.py --samples 20000000,50000000
    python eval/tune_el_ngram.py --samples full --ctx 40,60,80 --tri 1,2 --bi 1
    python eval/tune_el_ngram.py --samples full --vocab 80000,120000,150000,200000 \\
        --ctx 60 --tri 2 --bi 2

Results go to ``build/lm_el_tuning/results.jsonl`` (one line per model) and
a Markdown table on stdout.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from dilemma_build import el_ngram  # noqa: E402
import eval_el_ngram  # noqa: E402

DEFAULT_OUT = ROOT / "build" / "lm_el_tuning"


def ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--samples", default="20000000,50000000",
                    help="Comma-separated weighted training sizes, or 'full'.")
    ap.add_argument("--vocab", default=str(el_ngram.DEFAULT_VOCAB_SIZE))
    ap.add_argument("--ctx", default="3,5,10,20,40",
                    help="min_bigram_for_tri values")
    ap.add_argument("--tri", default="1,2", help="min_count_tri values")
    ap.add_argument("--bi", default="1,2", help="min_count_bi values")
    ap.add_argument("--no-caps", action="store_true")
    ap.add_argument("--max-sentences", type=int, default=3000,
                    help="Dev sentences per source for each evaluation.")
    ap.add_argument("--corpus-dir", default=str(el_ngram.DEFAULT_CORPUS))
    ap.add_argument("--dictionary", default=str(el_ngram.DEFAULT_DICTIONARY))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--keep", action="store_true",
                    help="Keep each model's directory (binaries and "
                         "intermediates); by default only its reports stay.")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    entries = el_ngram.load_dictionary(Path(args.dictionary))
    shards = el_ngram.encode_corpus(Path(args.corpus_dir), Path(args.dictionary),
                                    ROOT / "build" / "lm_el_encoded", entries)
    dev_dir = out / "dev"
    rows = []
    for sample, vocab_size in itertools.product(args.samples.split(","),
                                                ints(args.vocab)):
        tokens = None if sample == "full" else int(sample)
        c = el_ngram.count_corpus(
            Path(args.corpus_dir), Path(args.dictionary), vocab_size=vocab_size,
            caps=not args.no_caps, sample_tokens=tokens, entries=entries,
            shards=shards)
        if not (dev_dir / "dev_sentences.txt").exists():
            dev_dir.mkdir(parents=True, exist_ok=True)
            el_ngram.write_dev(dev_dir, el_ngram.selection(shards, split="dev"),
                               entries)
        for ctx, tri, bi in itertools.product(ints(args.ctx), ints(args.tri),
                                              ints(args.bi)):
            name = (f"{sample}_v{vocab_size}_ctx{ctx}_tri{tri}_bi{bi}"
                    + ("_nocaps" if args.no_caps else ""))
            d = out / name
            t0 = time.time()
            stats = el_ngram.write_intermediates(
                c, d, min_count_bi=bi, min_count_tri=tri,
                min_bigram_for_tri=ctx, dev=False)
            subprocess.run(
                [sys.executable, str(ROOT / "export_lm.py"), "--lang", "el",
                 "--in", str(d), "--out", str(d / "el_ngram.bin"),
                 "--version-out", str(d / "el_ngram.version")],
                check=True, capture_output=True)
            rep = eval_el_ngram.main(
                ["--bin", str(d / "el_ngram.bin"), "--dev-dir", str(dev_dir),
                 "--max-sentences", str(args.max_sentences),
                 "--out", str(out / f"{name}.eval.json")])
            version = json.loads((d / "el_ngram.version").read_text(encoding="utf-8"))
            row = {
                "name": name, "sample": sample,
                "train_tokens_weighted": stats["n_train_words_weighted"],
                "vocab": version["vocab_size"], "caps": not args.no_caps,
                "min_bigram_for_tri": ctx, "min_count_tri": tri,
                "min_count_bi": bi,
                "size_bytes": version["artifact_size_bytes"],
                "bigram_contexts": version["bigram_contexts"],
                "trigram_contexts": version["trigram_contexts"],
                "bigram_entries": rep["model"]["bigram_entries"],
                "trigram_entries": rep["model"]["trigram_entries"],
                "oov_unigrams": version["oov_unigrams"],
                "pooled": rep["pooled"],
                "per_source": rep["per_source"],
                "seconds": round(time.time() - t0, 1),
            }
            rows.append(row)
            with open(results, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            (out / f"{name}.version").write_text(
                json.dumps(version, ensure_ascii=False, indent=2), encoding="utf-8")
            if not args.keep:
                shutil.rmtree(d)
        del c

    cols = ["sample", "vocab", "min_bigram_for_tri", "min_count_tri",
            "min_count_bi", "MB", "tri ctx", "ppl all", "top1 all", "top3 all",
            "ppl conv", "top1 conv", "top3 conv", "comp2@3", "comp3@3"]
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        a, cv = r["pooled"]["all"], r["pooled"].get("conversational", {})
        vals = [r["sample"], f"{r['vocab']:,}", r["min_bigram_for_tri"],
                r["min_count_tri"], r["min_count_bi"],
                f"{r['size_bytes'] / 1e6:.1f}", f"{r['trigram_contexts']:,}",
                f"{a['pseudo_perplexity']:.0f}", f"{100 * a['next_top1_all']:.2f}",
                f"{100 * a['next_top3_all']:.2f}",
                f"{cv.get('pseudo_perplexity', 0):.0f}",
                f"{100 * cv.get('next_top1_all', 0):.2f}",
                f"{100 * cv.get('next_top3_all', 0):.2f}",
                f"{100 * a['completion2_top3_listed']:.2f}",
                f"{100 * a['completion3_top3_listed']:.2f}"]
        print("| " + " | ".join(str(v) for v in vals) + " |")


if __name__ == "__main__":
    main()
