#!/usr/bin/env python3
"""Phase B: fine-tune GreBerta as a lexicon-informed morphological tagger.

GreBerta encoder + per-token classification heads (UPOS + each UD feature). The
Morpheus candidate set (data/tagger/candidates.json) is fed in as a SOFT PRIOR:
a multi-hot over allowed UPOS/feature values, projected and added to each token's
representation. The model predicts UPOS + features freely (learning GLAUx's
convention), biased by the prior. Gold feats are normalize_feats'd so the model
learns the canonical convention used by measure_book_morph.py.

Trained on the commercial-safe GLAUx gold (data/tagger/sentences.jsonl).

Usage:
  python train_tagger.py --smoke               # tiny subset, validate the pipeline
  python train_tagger.py --epochs 2            # full train
  python train_tagger.py --no-prior            # ablation: plain GreBerta tagger
"""
import argparse
import os
# Fast (Rust) tokenizer + DataLoader fork workers would otherwise warn/deadlock.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

DATA = Path(__file__).resolve().parent / "data" / "tagger"
MODEL_NAME = "bowphs/GreBerta"
FEATURES = ["Case", "Number", "Gender", "Person", "Tense", "Mood", "Voice",
            "VerbForm", "Aspect", "Degree"]
NONE = "_"          # "feature absent" class
MAXLEN = 256        # subword cap per sentence
# Runtime-configurable (set in main from CLI). AG uses the Morpheus prior +
# normalize_feats + the RoBERTa add_prefix_space; MG (Greek-BERT) turns those off.
NORMALIZE = True
PREFIX_SPACE = True


def nfeats(upos, lemma, feats):
    # AG canonicalizes feats onto the gold treebank convention via
    # convert_treebank.normalize_feats (pure Python, no Morpheus). MG /
    # --no-normalize skips it entirely.
    if not NORMALIZE:
        return feats
    try:
        from convert_treebank import normalize_feats
        return normalize_feats(upos, lemma, feats)
    except Exception:
        return feats


def derive_features(path):
    """Feature keys present in the train split (for non-AG corpora)."""
    keys = set()
    for line in open(path, encoding="utf-8"):
        s = json.loads(line)
        if s["split"] != "train":
            continue
        for t in s["tokens"]:
            keys.update(t["feats"])
    return sorted(keys)


# --------------------------------------------------------------------------
# Label space
# --------------------------------------------------------------------------
def build_label_space(path, limit=None):
    upos, fv = set(), {f: set([NONE]) for f in FEATURES}
    n = 0
    for line in open(path, encoding="utf-8"):
        s = json.loads(line)
        if s["split"] != "train":
            continue
        for t in s["tokens"]:
            u = t["upos"]
            upos.add(u)
            nf = nfeats(u, t["lemma"], t["feats"])
            for f in FEATURES:
                fv[f].add(nf.get(f, NONE))
        n += 1
        if limit and n >= limit:
            break
    upos_list = sorted(upos)
    fv_list = {f: sorted(fv[f]) for f in FEATURES}
    return upos_list, fv_list


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class TaggerData(Dataset):
    def __init__(self, sents, tok, upos2i, fv2i, prior_index, cands, use_prior):
        self.sents = sents
        self.tok = tok
        self.upos2i = upos2i
        self.fv2i = fv2i           # {feature: {value: idx}}
        self.prior_index = prior_index   # {(kind,value): col}  for the multi-hot
        self.prior_dim = len(prior_index)
        self.cands = cands
        self.use_prior = use_prior

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, i):
        toks = self.sents[i]["tokens"]
        words = [t["form"] for t in toks]
        enc = self.tok(words, is_split_into_words=True, truncation=True,
                       max_length=MAXLEN, return_tensors="pt")
        word_ids = enc.word_ids()
        # first-subword index per word (within cap)
        first = {}
        for pos, wid in enumerate(word_ids):
            if wid is not None and wid not in first:
                first[wid] = pos
        n_words = max(first) + 1 if first else 0

        upos_y = torch.full((n_words,), -100, dtype=torch.long)
        feat_y = {f: torch.full((n_words,), -100, dtype=torch.long) for f in FEATURES}
        sub_idx = torch.zeros(n_words, dtype=torch.long)
        prior = torch.zeros(n_words, self.prior_dim)
        for w in range(n_words):
            if w not in first:
                continue
            sub_idx[w] = first[w]
            t = toks[w]
            u = t["upos"]
            nf = nfeats(u, t["lemma"], t["feats"])
            upos_y[w] = self.upos2i.get(u, -100)
            for f in FEATURES:
                feat_y[f][w] = self.fv2i[f].get(nf.get(f, NONE), -100)
            if self.use_prior:
                for c in self.cands.get(t["form"], []):
                    col = self.prior_index.get(("UPOS", c[1]))
                    if col is not None:
                        prior[w, col] = 1.0
                    cf = nfeats(c[1], c[0], c[2])
                    for f in FEATURES:
                        v = cf.get(f)
                        if v is None:
                            continue
                        for vv in str(v).split(","):
                            col = self.prior_index.get((f, vv))
                            if col is not None:
                                prior[w, col] = 1.0
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "sub_idx": sub_idx, "upos_y": upos_y,
            "feat_y": feat_y, "prior": prior, "n_words": n_words,
        }


def collate(batch, pad_id):
    maxsub = max(b["input_ids"].size(0) for b in batch)
    maxw = max(b["n_words"] for b in batch)
    B = len(batch)
    ids = torch.full((B, maxsub), pad_id, dtype=torch.long)
    am = torch.zeros((B, maxsub), dtype=torch.long)
    sub = torch.zeros((B, maxw), dtype=torch.long)
    uy = torch.full((B, maxw), -100, dtype=torch.long)
    fy = {f: torch.full((B, maxw), -100, dtype=torch.long) for f in FEATURES}
    pr = torch.zeros((B, maxw, batch[0]["prior"].size(1)))
    wmask = torch.zeros((B, maxw), dtype=torch.bool)
    for b, ex in enumerate(batch):
        L = ex["input_ids"].size(0); W = ex["n_words"]
        ids[b, :L] = ex["input_ids"]; am[b, :L] = ex["attention_mask"]
        sub[b, :W] = ex["sub_idx"]; uy[b, :W] = ex["upos_y"]
        pr[b, :W] = ex["prior"]; wmask[b, :W] = True
        for f in FEATURES:
            fy[f][b, :W] = ex["feat_y"][f]
    return {"input_ids": ids, "attention_mask": am, "sub_idx": sub,
            "upos_y": uy, "feat_y": fy, "prior": pr, "wmask": wmask}


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class MorphTagger(nn.Module):
    def __init__(self, n_upos, n_feat_vals, prior_dim, use_prior):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL_NAME)
        h = self.enc.config.hidden_size
        self.use_prior = use_prior
        if use_prior:
            self.prior_proj = nn.Linear(prior_dim, h)
        self.drop = nn.Dropout(0.1)
        self.upos_head = nn.Linear(h, n_upos)
        self.feat_heads = nn.ModuleDict(
            {f: nn.Linear(h, n_feat_vals[f]) for f in FEATURES})

    def forward(self, input_ids, attention_mask, sub_idx, prior):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        B, _, H = out.shape
        idx = sub_idx.unsqueeze(-1).expand(-1, -1, H)
        tok = torch.gather(out, 1, idx)               # (B, W, H) first-subword reps
        if self.use_prior:
            tok = tok + self.prior_proj(prior)
        tok = self.drop(tok)
        return self.upos_head(tok), {f: self.feat_heads[f](tok) for f in FEATURES}


def main():
    global MODEL_NAME, FEATURES, NORMALIZE, PREFIX_SPACE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--no-prior", action="store_true")
    ap.add_argument("--workers", type=int, default=8,
                    help="DataLoader workers (parallel tokenize + prior build)")
    ap.add_argument("--out", default=str(DATA / "tagger_model.pt"))
    ap.add_argument("--model", default=MODEL_NAME, help="HF encoder id")
    ap.add_argument("--sentences", default=str(DATA / "sentences.jsonl"))
    ap.add_argument("--candidates", default=str(DATA / "candidates.json"))
    ap.add_argument("--features-auto", action="store_true",
                    help="derive the feature set from the train data (non-AG)")
    ap.add_argument("--no-normalize", action="store_true",
                    help="skip Morpheus normalize_feats (non-AG corpora)")
    ap.add_argument("--no-prefix-space", action="store_true",
                    help="tokenizer without add_prefix_space (BERT, not RoBERTa)")
    args = ap.parse_args()
    MODEL_NAME = args.model
    NORMALIZE = not args.no_normalize
    PREFIX_SPACE = not args.no_prefix_space
    if args.features_auto:
        FEATURES = derive_features(args.sentences)
    use_prior = not args.no_prior
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={dev} model={MODEL_NAME} prior={use_prior} "
          f"features={len(FEATURES)} smoke={args.smoke}", flush=True)

    sents_path = Path(args.sentences)
    print("building label space...", flush=True)
    upos_list, fv_list = build_label_space(sents_path, limit=2000 if args.smoke else None)
    upos2i = {u: i for i, u in enumerate(upos_list)}
    fv2i = {f: {v: i for i, v in enumerate(fv_list[f])} for f in FEATURES}
    prior_index = {}
    for u in upos_list:
        prior_index[("UPOS", u)] = len(prior_index)
    for f in FEATURES:
        for v in fv_list[f]:
            if v != NONE:
                prior_index[(f, v)] = len(prior_index)
    print(f"  upos={len(upos_list)} feat-values={ {f: len(fv_list[f]) for f in FEATURES} } prior_dim={len(prior_index)}")

    print("loading sentences...", flush=True)
    train, dev_s = [], []
    for k, line in enumerate(open(sents_path, encoding="utf-8")):
        s = json.loads(line)
        if s["split"] == "train":
            train.append(s)
        elif s["split"] == "dev":
            dev_s.append(s)
        if args.smoke and len(train) >= 400 and len(dev_s) >= 80:
            break
    if args.smoke:
        train = train[:400]; dev_s = dev_s[:80]
    cands = json.load(open(args.candidates)) if use_prior else {}
    print(f"  train {len(train):,} dev {len(dev_s):,}")

    tok = AutoTokenizer.from_pretrained(
        MODEL_NAME, **({"add_prefix_space": True} if PREFIX_SPACE else {}))
    n_feat_vals = {f: len(fv_list[f]) for f in FEATURES}
    model = MorphTagger(len(upos_list), n_feat_vals, len(prior_index), use_prior).to(dev)
    ds_tr = TaggerData(train, tok, upos2i, fv2i, prior_index, cands, use_prior)
    ds_dv = TaggerData(dev_s, tok, upos2i, fv2i, prior_index, cands, use_prior)
    pad = tok.pad_token_id
    # The per-item work (tokenize + build the prior multi-hot) is CPU-bound and
    # was starving the GPU at num_workers=0 (~50% util). Parallel workers keep
    # the GPU fed; pin_memory + persistent_workers cut transfer + respawn cost.
    pin = (dev == "cuda")
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True,
                       collate_fn=lambda b: collate(b, pad),
                       num_workers=args.workers, pin_memory=pin,
                       persistent_workers=args.workers > 0,
                       prefetch_factor=4 if args.workers > 0 else None)
    dl_dv = DataLoader(ds_dv, batch_size=args.bs, shuffle=False,
                       collate_fn=lambda b: collate(b, pad),
                       num_workers=min(args.workers, 4), pin_memory=pin,
                       persistent_workers=min(args.workers, 4) > 0,
                       prefetch_factor=4 if args.workers > 0 else None)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    cel = nn.CrossEntropyLoss(ignore_index=-100)

    def run_eval():
        model.eval()
        up_ok = up_tot = strict_ok = 0
        feat_ok = feat_tot = 0
        with torch.no_grad():
            for b in dl_dv:
                b = {k: (v.to(dev) if torch.is_tensor(v) else
                         {f: vv.to(dev) for f, vv in v.items()}) for k, v in b.items()}
                ul, fl = model(b["input_ids"], b["attention_mask"], b["sub_idx"], b["prior"])
                up = ul.argmax(-1); m = b["upos_y"] != -100
                up_ok += ((up == b["upos_y"]) & m).sum().item(); up_tot += m.sum().item()
                strict = (up == b["upos_y"]) & m
                for f in FEATURES:
                    fp = fl[f].argmax(-1); fm = b["feat_y"][f] != -100
                    feat_ok += ((fp == b["feat_y"][f]) & fm).sum().item()
                    feat_tot += fm.sum().item()
                    strict = strict & ((fp == b["feat_y"][f]) | (~fm))
                strict_ok += (strict & m).sum().item()
        return up_ok / max(up_tot, 1), feat_ok / max(feat_tot, 1), strict_ok / max(up_tot, 1)

    print("training...", flush=True)
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for it, b in enumerate(dl_tr):
            b = {k: (v.to(dev) if torch.is_tensor(v) else
                     {f: vv.to(dev) for f, vv in v.items()}) for k, v in b.items()}
            ul, fl = model(b["input_ids"], b["attention_mask"], b["sub_idx"], b["prior"])
            loss = cel(ul.reshape(-1, ul.size(-1)), b["upos_y"].reshape(-1))
            for f in FEATURES:
                loss = loss + cel(fl[f].reshape(-1, fl[f].size(-1)), b["feat_y"][f].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
            if it % 200 == 0:
                print(f"  ep{ep} it{it}/{len(dl_tr)} loss={loss.item():.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        u, fa, st = run_eval()
        print(f"EPOCH {ep}: dev UPOS={u*100:.1f}% feats={fa*100:.1f}% strict={st*100:.1f}% "
              f"avg_loss={tot/len(dl_tr):.3f}", flush=True)
        # Checkpoint after every epoch so a mid-run interruption (e.g. a remote
        # box dropping) keeps the latest completed-epoch weights. The canonical
        # --out is always overwritten with the most recent epoch.
        ckpt = {"state": model.state_dict(), "upos_list": upos_list,
                "fv_list": fv_list, "prior_index": list(prior_index),
                "use_prior": use_prior, "epoch": ep,
                # config so eval/export/runtime reconstruct without CLI flags
                "model_name": MODEL_NAME, "features": FEATURES,
                "normalize": NORMALIZE, "prefix_space": PREFIX_SPACE,
                "max_len": MAXLEN}
        torch.save(ckpt, f"{args.out}.ep{ep}")
        torch.save(ckpt, args.out)
        print(f"saved {args.out} (epoch {ep})", flush=True)


if __name__ == "__main__":
    main()
