#!/usr/bin/env python3
"""Phase A of the Ancient-Greek tagger: assemble openly licensed GLAUx gold for
fine-tuning the GreBerta tagger + dependency parser.

Morpheus-free: emits only the treebank gold (UPOS + UD features via
convert_treebank, plus the native AGDT head + relation for the dep-parse head).
No candidate prior (the ablation showed it adds ~1.3%, so the trained model is
--no-prior and needs no Morpheus binary). GLAUx is CC BY-SA except a handful of
NonCommercial texts, which are excluded so the tagger is openly licensed.

GLAUx lives at $DILEMMA_GLAUX_DIR (default ~/Documents/glaux); it is an external
corpus input, like the kaikki dumps build_data.py reads.

Output: data/tagger/sentences.jsonl, one sentence per line:
  {"work": tlg, "split": train|dev|test,
   "tokens": [{"form","lemma","upos","feats","head","deprel"}, ...]}
head is the 1-based position of the syntactic head within the sentence (0 =
root / attaches above an elided node); deprel is the native AGDT relation.

Usage: python build/build_tagger_data.py [--limit N]
"""
import argparse
import csv
import json
import os
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
import convert_treebank as CT   # convert_postag(postag, lemma) -> (upos, feats)

GLAUX = Path(os.environ.get("DILEMMA_GLAUX_DIR",
                            str(Path.home() / "Documents" / "glaux")))
GLAUX_XML = GLAUX / "xml"
GLAUX_META = GLAUX / "metadata.txt"
OUT = Path(__file__).resolve().parent.parent / "data" / "tagger"

HELD_OUT_TEST = {"0012-001"}   # Homer, Iliad
# Dev was Herodotus 0016-001, whose manual trees are PROIEL-derived and now
# excluded entirely; Epictetus (Pedalion-derived manual, Koine prose, 92K
# tokens) replaces it. Dev metrics are NOT comparable across that change.
HELD_OUT_DEV = {"0557-001"}    # Epictetus, Dissertationes

sys.path.insert(0, str(Path(__file__).resolve().parent))  # build/ (nc_filter)
from nc_filter import excluded_glaux_stems, gorman_glaux_stems


def extract(limit: int = 0):
    nc = excluded_glaux_stems(GLAUX_META)
    gorman = gorman_glaux_stems(GLAUX_META)
    files = sorted(GLAUX_XML.glob("*.xml"))
    if limit:
        files = files[:limit]
    sentences = []
    n_nc = 0
    for xf in files:
        stem = xf.stem
        if stem in nc:
            n_nc += 1
            continue
        split = ("test" if stem in HELD_OUT_TEST
                 else "dev" if stem in HELD_OUT_DEV else "train")
        try:
            root = ET.parse(xf).getroot()
        except ET.ParseError:
            continue
        # Gorman-derived works: the manual sentences ARE Gorman's trees
        # (held-out gold, never ingested); the auto sentences pass.
        skip_manual = stem in gorman
        for sent in root.findall(".//sentence"):
            if skip_manual and sent.get("analysis") == "manual":
                continue
            # First pass: keep real surface words (drop "z" artificial ellipsis
            # nodes), assigning each a 1-based position; map word id -> position.
            kept = []
            id2pos = {}
            for w in sent.findall(".//word"):
                if (w.get("postag") or "")[:1] == "z":
                    continue
                form = unicodedata.normalize("NFC", w.get("form") or "")
                if not form:
                    continue
                id2pos[w.get("id")] = len(kept) + 1   # 1-based
                kept.append(w)
            toks = []
            for w in kept:
                form = unicodedata.normalize("NFC", w.get("form") or "")
                lemma = unicodedata.normalize("NFC", w.get("lemma") or "")
                upos, feats = CT.convert_postag(w.get("postag") or "", lemma)
                # head: 1-based position of the head word, or 0 if it is the
                # sentence root or attaches above a dropped (elided) node.
                head = id2pos.get(w.get("head"), 0)
                deprel = w.get("relation") or "_"
                toks.append({"form": form, "lemma": lemma, "upos": upos,
                             "feats": feats, "head": head, "deprel": deprel})
            if toks:
                sentences.append({"work": stem, "split": split, "tokens": toks})
    return sentences, n_nc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N GLAUx files (smoke test)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Reading GLAUx from {GLAUX_XML} (openly licensed)...", flush=True)
    sents, n_nc = extract(args.limit)
    n_tok = sum(len(s["tokens"]) for s in sents)
    sp = Counter(s["split"] for s in sents)
    rel = Counter(t["deprel"] for s in sents for t in s["tokens"])
    print(f"  {len(sents):,} sentences, {n_tok:,} tokens; {n_nc} text(s) "
          f"excluded (NonCommercial or PROIEL-derived)")
    print(f"  splits (sentences): {dict(sp)}")
    print(f"  {len(rel)} deprels; top: {[r for r, _ in rel.most_common(12)]}")
    with (OUT / "sentences.jsonl").open("w", encoding="utf-8") as f:
        for s in sents:
            f.write(json.dumps(s, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"  wrote {OUT / 'sentences.jsonl'}")


if __name__ == "__main__":
    main()
