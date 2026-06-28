#!/usr/bin/env python3
"""Phase A for the Modern Greek tagger: convert UD_Greek-GDT (.conllu) to the
sentences.jsonl format train_tagger.py consumes.

GreBerta is an Ancient-Greek model, so MG needs its own tagger (Greek-BERT
encoder + the same per-feature-head architecture). Trained on GDT - small
(~43K train tokens) but the standard MG UD treebank.

Syntactic-word level: multiword-token range rows (id "n-m", e.g. στο = σε+το)
and empty nodes (id "n.m") are skipped; the component rows carry the real tags.
MWT splitting at inference is handled in the el backend.

Output: data/tagger_mg/sentences.jsonl  (+ prints the feature inventory)
Usage: python build_mg_tagger_data.py
"""
import json
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent   # dilemma repo root
GDT = _ROOT / "data" / "treebanks" / "UD_Greek-GDT"
OUT = _ROOT / "data" / "tagger_mg"
SPLITS = {"train": "el_gdt-ud-train.conllu",
          "dev": "el_gdt-ud-dev.conllu",
          "test": "el_gdt-ud-test.conllu"}


def parse_feats(col: str) -> dict:
    if not col or col == "_":
        return {}
    out = {}
    for kv in col.split("|"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
    return out


def read_conllu(path: Path, split: str, mwt: dict | None = None):
    sents = []
    toks = []
    pending = None   # (surface, lo, hi, [component forms]) for an open MWT range
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if toks:
                sents.append({"split": split, "tokens": toks})
                toks = []
            pending = None
            continue
        if line.startswith("#"):
            continue
        f = line.split("\t")
        tid = f[0]
        if "-" in tid:   # MWT range row: record surface + span, collect parts
            lo, hi = (int(x) for x in tid.split("-"))
            pending = (f[1], lo, hi, [])
            continue
        if "." in tid:   # empty node
            continue
        form, lemma, upos, feats = f[1], f[2], f[3], f[5]
        # collect component forms of an open MWT range, then close + record it
        if pending and pending[1] <= int(tid) <= pending[2]:
            pending[3].append(form)
            if int(tid) == pending[2] and mwt is not None:
                mwt[pending[0].lower()] = pending[3]
                pending = None
        if upos == "_" or not form:
            continue
        toks.append({"form": form, "lemma": lemma, "upos": upos,
                     "feats": parse_feats(feats)})
    if toks:
        sents.append({"split": split, "tokens": toks})
    return sents


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_sents = []
    feat_keys = Counter()
    upos = Counter()
    counts = {}
    mwt = {}
    for split, fname in SPLITS.items():
        s = read_conllu(GDT / fname, split, mwt)
        all_sents.extend(s)
        counts[split] = sum(len(x["tokens"]) for x in s)
        for sent in s:
            for t in sent["tokens"]:
                upos[t["upos"]] += 1
                for k in t["feats"]:
                    feat_keys[k] += 1
    with (OUT / "sentences.jsonl").open("w", encoding="utf-8") as fh:
        for s in all_sents:
            fh.write(json.dumps(s, ensure_ascii=False, separators=(",", ":")) + "\n")
    (OUT / "mwt.json").write_text(
        json.dumps(mwt, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(f"wrote {OUT / 'sentences.jsonl'}: {len(all_sents)} sentences")
    print(f"  tokens by split: {counts}")
    print(f"  multiword tokens: {len(mwt)}  e.g. "
          f"{dict(list(sorted(mwt.items()))[:6])}")
    print(f"  UPOS ({len(upos)}): {[u for u, _ in upos.most_common()]}")
    print(f"  feature keys ({len(feat_keys)}): "
          f"{[f'{k}({n})' for k, n in feat_keys.most_common()]}")


if __name__ == "__main__":
    main()
