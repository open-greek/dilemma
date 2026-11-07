#!/usr/bin/env python3
"""Next-word prediction and mid-word completion of the monotonic Modern
Greek model, measured on the held-out dev split.

Reads ``build/lm/el_ngram.bin`` (``export_lm.py --lang el``) with
``dilemma_build.gnlm_reader``, which answers the way the keyboard's Swift
reader does, and the dev sentences ``train_lm.py --lang el`` wrote
(``build/lm_el/dev_sentences_<source>.txt``: resolved word-list spellings,
``<UNK>`` for tokens with none). The dev split is held out by document
group, so no dev document contributed a count.

Measures
--------
* **Next word.** After each word of a sentence (and after ``<s>``), the
  first 1 and 3 suggestions of ``LMReader.suggest`` with the two previous
  words as context, against the word that follows. ``</s>`` is never a
  target. The keyboard never offers a double-accented enclitic form
  (όνομά), so suggestions leave them out and such a target counts as its
  host (όνομα). ``all words`` counts every word target (an ``<UNK>`` or a
  spelling out of the vocabulary is a miss); ``in vocabulary`` only
  targets the model can propose. A context word out of the vocabulary
  resolves to nothing, as on the keyboard.
* **Completion recall.** For each word target with more than k letters
  (k = 2, 3), the first k letters typed without accents in lowercase,
  then the first 1 and 3 completions of ``LMReader.suggestCompletions``:
  a hit when the word is among them. Over word-list targets and over
  vocabulary targets.
* **Pseudo-perplexity.** exp of the mean negative ``LMReader.score`` over
  vocabulary targets: stupid backoff from the rows the file holds, so a
  word cut from a row is scored by backoff, exactly as the keyboard scores
  it. Not a normalized perplexity; it compares models of the same format.
* **Predictions after common contexts**, for a quick look.

Usage
-----
    python eval/eval_el_ngram.py                       # whole dev split
    python eval/eval_el_ngram.py --max-sentences 3000  # per source, quick
    python eval/eval_el_ngram.py --bin X.bin --dev-dir build/lm_el --out X.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dilemma_build.el_ngram import fold_enclitic_accent  # noqa: E402
from dilemma_build.gnlm_reader import GNLMReader, normalize  # noqa: E402

DEFAULT_BIN = ROOT / "build" / "lm" / "el_ngram.bin"
DEFAULT_DEV = ROOT / "build" / "lm_el"
CONVERSATIONAL = ("tatoeba", "common_voice", "wiktionary_examples")
SANITY_CONTEXTS = [
    (None, "θα"), (None, "δεν"), (None, "να"), (None, "είναι"),
    (None, "καλή"), ("τι", "κάνεις"), ("ευχαριστώ", "πολύ"),
    ("<s>", "θα"), (None, "σας"), ("<s>", "καλημέρα"),
]
SANITY_COMPLETIONS = [
    ("θα", "ήθελα", "να"), ("τι", "κάνεις", "σ"), ("<s>", "ευχαριστώ", "π"),
    (None, "<s>", "καλ"), ("να", "σας", "ευχ"), (None, "δεν", "ξ"),
]
STEM_LENGTHS = (2, 3)
TOP = (1, 3)


def letters(token: str) -> int:
    return sum(1 for c in token if c.isalpha())


def load_dev(dev_dir: Path, max_sentences: int | None) -> dict[str, list[list[str]]]:
    """Dev sentences per source, every n-th one when capped, so the sample
    spans the whole file."""
    out = {}
    for path in sorted(dev_dir.glob("dev_sentences_*.txt")):
        source = path.stem[len("dev_sentences_"):]
        lines = path.read_text(encoding="utf-8").splitlines()
        if max_sentences and len(lines) > max_sentences:
            step = len(lines) / max_sentences
            lines = [lines[int(k * step)] for k in range(max_sentences)]
        out[source] = [ln.split(" ") for ln in lines if ln]
    return out


class Evaluator:
    """Fast paths over a :class:`GNLMReader` for bare lowercase stems."""

    def __init__(self, lm: GNLMReader):
        self.lm = lm
        norm, _, _ = lm._normalized_index()
        self.norm = norm
        counts = lm.counts.astype(np.int64)
        self.prefix_codes = {}
        self.prefix_top = {}
        for k in STEM_LENGTHS:
            codes: dict[str, int] = {}
            arr = np.full(lm.V, -1, dtype=np.int64)
            groups: dict[int, list[int]] = defaultdict(list)
            for i, n in enumerate(norm):
                if len(n) < k or i in lm.reserved_ids or i in lm.hidden_ids:
                    continue
                c = codes.setdefault(n[:k], len(codes))
                arr[i] = c
                groups[c].append(i)
            top = {}
            for c, ids in groups.items():
                ids.sort(key=lambda i: (-counts[i], lm.vocab[i]))
                top[c] = ids
            self.prefix_codes[k] = (codes, arr)
            self.prefix_top[k] = top

    def completions(self, prev_id, curr_id, stem: str, limit: int) -> list[int]:
        """``suggestCompletions`` for a bare lowercase stem of length k:
        the typed-marks rule keeps every candidate, so the vocabulary scan
        ranks by count; only its first entries can reach the result."""
        lm = self.lm
        k = len(stem)
        codes, arr = self.prefix_codes[k]
        code = codes.get(stem)
        if code is None:
            return []
        pool: list[tuple[float, int, int]] = []
        seen: set[int] = set()
        for row, backoff in (
                (lm.trigram_row(prev_id, curr_id)
                 if prev_id is not None and curr_id is not None else None, 0.0),
                (lm.bigram_row(curr_id) if curr_id is not None else None,
                 math.log(0.4))):
            if row is None:
                continue
            ids, q = row
            sel = np.flatnonzero(arr[ids] == code)
            for j in sel.tolist():
                w = int(ids[j])
                if w in seen or w in lm.hidden_ids:
                    continue
                seen.add(w)
                pool.append((backoff + q[j] / 1024.0, int(lm.counts[w]), w))
        need = limit
        for w in self.prefix_top[k].get(code, ()):
            if w in seen:
                continue
            seen.add(w)
            c = int(lm.counts[w])
            pool.append((lm.unigram_score(c), c, w))
            need -= 1
            if need <= 0:
                break
        pool.sort(key=lambda x: (-x[0], -x[1], lm.vocab[x[2]]))
        return [w for _, _, w in pool[:limit]]


def evaluate(lm: GNLMReader, dev: dict[str, list[list[str]]]) -> dict:
    ev = Evaluator(lm)
    index = lm.index
    unk_names = {"<UNK>"}
    results = {}
    for source, sents in dev.items():
        r = defaultdict(float)
        for toks in sents:
            ids = [None if t in unk_names else index.get(t) for t in toks]
            for i in range(1, len(toks)):
                target = toks[i]
                if target == "</s>":
                    continue
                prev_id = ids[i - 2] if i >= 2 else None
                curr_id = ids[i - 1]
                wid = ids[i]
                host = fold_enclitic_accent(target)
                if host != target:
                    # The keyboard offers the host of a double-accented
                    # form and writes the second accent itself.
                    target, wid = host, index.get(host)
                listed = target not in unk_names
                in_vocab = wid is not None
                r["words"] += 1
                r["listed"] += listed
                r["in_vocab"] += in_vocab
                sug = lm.suggest_ids(prev_id, curr_id, max(TOP))
                for n in TOP:
                    hit = in_vocab and wid in sug[:n]
                    r[f"next_top{n}"] += hit
                if in_vocab:
                    sc = lm.score_id(wid, prev_id, curr_id)
                    r["logp_sum"] += sc
                if not listed:
                    continue
                n_letters = letters(target)
                stem_full = normalize(target)
                for k in STEM_LENGTHS:
                    if n_letters <= k:
                        continue
                    r[f"long{k}_listed"] += 1
                    r[f"long{k}_in_vocab"] += in_vocab
                    if not in_vocab:
                        continue
                    comp = ev.completions(prev_id, curr_id, stem_full[:k], max(TOP))
                    for n in TOP:
                        r[f"comp{k}_top{n}"] += wid in comp[:n]
        results[source] = dict(r)
    return results


def summarize(raw: dict) -> dict:
    s = {}
    w = raw.get("words", 0)
    v = raw.get("in_vocab", 0)
    s["word_targets"] = int(w)
    s["in_vocab_share"] = v / w if w else 0.0
    s["listed_share"] = raw.get("listed", 0) / w if w else 0.0
    for n in TOP:
        hits = raw.get(f"next_top{n}", 0)
        s[f"next_top{n}_all"] = hits / w if w else 0.0
        s[f"next_top{n}_in_vocab"] = hits / v if v else 0.0
    s["pseudo_perplexity"] = math.exp(-raw.get("logp_sum", 0) / v) if v else float("inf")
    for k in STEM_LENGTHS:
        L = raw.get(f"long{k}_listed", 0)
        LV = raw.get(f"long{k}_in_vocab", 0)
        for n in TOP:
            hits = raw.get(f"comp{k}_top{n}", 0)
            s[f"completion{k}_top{n}_listed"] = hits / L if L else 0.0
            s[f"completion{k}_top{n}_in_vocab"] = hits / LV if LV else 0.0
    return s


def pooled(results: dict, sources) -> dict:
    total = defaultdict(float)
    for src in sources:
        for k, v in results.get(src, {}).items():
            total[k] += v
    return dict(total)


def report(lm: GNLMReader, results: dict) -> dict:
    out = {"per_source": {}, "pooled": {}}
    for src, raw in sorted(results.items()):
        out["per_source"][src] = summarize(raw)
    out["pooled"]["all"] = summarize(pooled(results, results))
    conv = [s for s in CONVERSATIONAL if s in results]
    if conv:
        out["pooled"]["conversational"] = summarize(pooled(results, conv))
    return out


def sanity_table(lm: GNLMReader) -> dict:
    nxt = {" ".join(x for x in ctx if x): lm.suggest(*ctx, limit=6)
           for ctx in SANITY_CONTEXTS}
    comp = {f"{' '.join(x for x in (p, c) if x)} | {stem}":
            lm.suggest_completions(p, c, stem, 3)
            for p, c, stem in SANITY_COMPLETIONS}
    return {"next_word": nxt, "completions": comp}


def print_report(rep: dict, header: str) -> None:
    cols = ["word_targets", "in_vocab_share", "next_top1_all", "next_top3_all",
            "next_top1_in_vocab", "next_top3_in_vocab", "pseudo_perplexity",
            "completion2_top3_listed", "completion3_top3_listed"]
    print(header)
    print("| set | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    rows = list(rep["per_source"].items()) + [
        (f"POOLED {k}", v) for k, v in rep["pooled"].items()]
    for name, s in rows:
        vals = []
        for c in cols:
            v = s[c]
            vals.append(f"{v:,}" if c == "word_targets" else
                        f"{v:.1f}" if c == "pseudo_perplexity" else f"{100 * v:.2f}")
        print(f"| {name} | " + " | ".join(vals) + " |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bin", default=str(DEFAULT_BIN))
    ap.add_argument("--dev-dir", default=str(DEFAULT_DEV))
    ap.add_argument("--max-sentences", type=int, default=None,
                    help="Per source, every n-th dev sentence up to this many.")
    ap.add_argument("--out", default=None,
                    help="JSON report (default: <bin name>.eval.json in "
                         "--dev-dir, so build/lm holds only the artifact "
                         "and its sidecar).")
    args = ap.parse_args(argv)
    t0 = time.time()
    lm = GNLMReader(Path(args.bin))
    lm.hide_double_accented()
    dev = load_dev(Path(args.dev_dir), args.max_sentences)
    results = evaluate(lm, dev)
    rep = report(lm, results)
    rep["sanity"] = sanity_table(lm)
    rep["model"] = {"bin": str(args.bin), "size_bytes": len(lm.data),
                    "vocab": lm.V, "bigram_rows": len(lm.bi_rows),
                    "bigram_entries": len(lm.bi_sug),
                    "trigram_rows": len(lm.tri_rows),
                    "trigram_entries": len(lm.tri_sug),
                    "oov_entries": lm.oov_n, "total_tokens": lm.total_tokens}
    rep["dev"] = {"dir": str(args.dev_dir), "max_sentences": args.max_sentences,
                  "sentences": {k: len(v) for k, v in dev.items()}}
    rep["seconds"] = round(time.time() - t0, 1)
    out = Path(args.out) if args.out else (
        Path(args.dev_dir) / (Path(args.bin).stem + ".eval.json"))
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(rep, f"{args.bin}: {len(lm.data):,} bytes, vocab {lm.V:,}")
    print("\nNext word after common contexts:")
    for ctx, sug in rep["sanity"]["next_word"].items():
        print(f"  {ctx:>18}: {', '.join(sug)}")
    print("Completions:")
    for ctx, sug in rep["sanity"]["completions"].items():
        print(f"  {ctx:>24}: {', '.join(sug)}")
    print(f"\nWrote {out} in {rep['seconds']}s")
    return rep


if __name__ == "__main__":
    main()
