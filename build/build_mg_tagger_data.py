#!/usr/bin/env python3
"""Phase A for the Modern Greek tagger: convert the COMMERCIAL-SAFE Modern Greek
UD treebanks (.conllu) to the sentences.jsonl format train_tagger.py consumes.

We deliberately do NOT use UD_Greek-GDT: it is CC BY-NC-SA (NonCommercial). The
treebanks here are all CC BY-SA 4.0, gold (manually native-validated), and
independent of GDT:
  - UD_Greek-GUD       Standard Modern Greek (fiction); the base, ~25K tokens.
  - UD_Greek-Cretan    East Cretan dialect (augmentation).
  - UD_Greek-Lesbian   Lesbos (Northern) dialect (augmentation).
  - UD_Greek-Messinian Messenian (Southern) dialect (augmentation).
(UD_Greek-Cypriot / -Griko are CC BY-SA too but are still data-less placeholder
repos; they get folded in automatically once they ship .conllu files.)

GreBerta is an Ancient-Greek model, so MG needs its own tagger (Greek-BERT
encoder + the same per-feature-head architecture + biaffine dep head). The
dialect treebanks all go to train; GUD's split is held out for dev/test so the
reported numbers are Standard Modern Greek.

Syntactic-word level: multiword-token range rows (id "n-m", e.g. στο = σε+το)
and empty nodes (id "n.m") are skipped; the component rows carry the real tags.
MWT splitting at inference is handled in the el backend.

Output: data/tagger_mg/sentences.jsonl  (+ mwt.json + prints the inventory)
Usage: python build/build_mg_tagger_data.py
"""
import json
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent   # dilemma repo root
TB = _ROOT / "data" / "treebanks"
OUT = _ROOT / "data" / "tagger_mg"

# Standard Modern Greek base (held out for dev/test).
GUD = TB / "UD_Greek-GUD"
# Dialect treebanks: everything they contain goes to train (augmentation).
DIALECTS = ["UD_Greek-Cretan", "UD_Greek-Lesbian", "UD_Greek-Messinian",
            "UD_Greek-Cypriot", "UD_Greek-Griko"]
DEV_EVERY = 8   # hold out every Nth GUD-train sentence as dev (dev != test)


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
        if pending and pending[1] <= int(tid) <= pending[2]:
            pending[3].append(form)
            if int(tid) == pending[2] and mwt is not None:
                mwt[pending[0].lower()] = pending[3]
                pending = None
        if upos == "_" or not form:
            continue
        try:
            head = int(f[6])
        except (ValueError, IndexError):
            head = 0
        deprel = f[7] if len(f) > 7 and f[7] != "_" else "_"
        toks.append({"form": form, "lemma": lemma, "upos": upos,
                     "feats": parse_feats(feats), "head": head, "deprel": deprel})
    if toks:
        sents.append({"split": split, "tokens": toks})
    return sents


DIALECT_TEST_EVERY = 5   # hold out every 5th dialect sentence (~20%) for per-dialect eval


def main(gud_only=False):
    OUT.mkdir(parents=True, exist_ok=True)
    all_sents = []
    mwt = {}
    src_counts = Counter()

    # GUD: train (minus a held-out dev) + test, all Standard Modern Greek. The
    # GUD test split is the headline "test" (honest Standard-MG accuracy).
    gud_train = read_conllu(GUD / "el_gud-ud-train.conllu", "train", mwt)
    for i, s in enumerate(gud_train):
        s["split"] = "dev" if i % DEV_EVERY == 0 else "train"
    all_sents.extend(gud_train)
    gud_test = GUD / "el_gud-ud-test.conllu"
    if gud_test.exists():
        all_sents.extend(read_conllu(gud_test, "test", mwt))
    src_counts["GUD"] = sum(len(s["tokens"]) for s in all_sents)

    # Dialect treebanks. Each dialect's sentences are split ~80/20: 80% -> train
    # (augmentation; skipped entirely with --gud-only, for the ablation), 20% ->
    # a held-out per-dialect eval split test_<dialect>. mwt=None: the dialects'
    # multiword-token splits differ from Standard MG (Lesbian στο -> σ+του vs SMG
    # στο -> σ+το) and the runtime mwt.json runs on ALL el input, so it stays
    # Standard-MG only; dialect sentences are still word-split for training.
    for name in DIALECTS:
        d = TB / name
        short = name.replace("UD_Greek-", "").lower()
        sents = []
        for cf in sorted(d.glob("*.conllu")):
            sents.extend(read_conllu(cf, "train", None))
        kept = 0
        for i, s in enumerate(sents):
            if i % DIALECT_TEST_EVERY == 0:
                s["split"] = f"test_{short}"
                all_sents.append(s)
                kept += len(s["tokens"])
            elif not gud_only:
                s["split"] = "train"
                all_sents.append(s)
                kept += len(s["tokens"])
        src_counts[name] = kept

    feat_keys = Counter()
    upos = Counter()
    deprels = Counter()
    by_split = Counter()
    for s in all_sents:
        by_split[s["split"]] += len(s["tokens"])
        for t in s["tokens"]:
            upos[t["upos"]] += 1
            deprels[t["deprel"]] += 1
            for k in t["feats"]:
                feat_keys[k] += 1

    out_name = "sentences_gudonly.jsonl" if gud_only else "sentences.jsonl"
    with (OUT / out_name).open("w", encoding="utf-8") as fh:
        for s in all_sents:
            fh.write(json.dumps(s, ensure_ascii=False, separators=(",", ":")) + "\n")
    # mwt.json is the same Standard-MG map for both variants; write once.
    (OUT / "mwt.json").write_text(
        json.dumps(mwt, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    print(f"wrote {OUT / out_name}: {len(all_sents)} sentences")
    print(f"  tokens by source: {dict(src_counts)}")
    print(f"  tokens by split:  {dict(by_split)}")
    print(f"  multiword tokens: {len(mwt)}  e.g. "
          f"{dict(list(sorted(mwt.items()))[:6])}")
    print(f"  UPOS ({len(upos)}): {[u for u, _ in upos.most_common()]}")
    print(f"  deprels ({len(deprels)}): {[d for d, _ in deprels.most_common()]}")
    print(f"  feature keys ({len(feat_keys)}): "
          f"{[f'{k}({n})' for k, n in feat_keys.most_common()]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gud-only", action="store_true",
                    help="Ablation: emit sentences_gudonly.jsonl with GUD train "
                         "only (no dialect augmentation); same dev + test + "
                         "per-dialect test splits as the full build.")
    main(gud_only=ap.parse_args().gud_only)
