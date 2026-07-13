#!/usr/bin/env python3
"""Phase B: fine-tune GreBerta as a lexicon-informed morphological tagger.

GreBerta encoder + per-token classification heads (UPOS + each UD feature). The
Morpheus candidate set (data/tagger/candidates.json) is fed in as a SOFT PRIOR:
a multi-hot over allowed UPOS/feature values, projected and added to each token's
representation. The model predicts UPOS + features freely (learning GLAUx's
convention), biased by the prior. Gold feats are normalize_feats'd so the model
learns the canonical convention used by measure_book_morph.py.

Trained on the openly licensed GLAUx gold (data/tagger/sentences.jsonl).

Usage:
  python train_tagger.py --smoke               # tiny subset, validate the pipeline
  python train_tagger.py --epochs 2            # full train
  python train_tagger.py --no-prior            # ablation: plain GreBerta tagger
"""
import argparse
import functools
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
# Biaffine dependency-parse head (Dozat & Manning 2017), enabled by --dep.
ARC_DIM = 256
REL_DIM = 128


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


def derive_deprels(path):
    """Dependency relation labels present in the train split."""
    rels = set()
    for line in open(path, encoding="utf-8"):
        s = json.loads(line)
        if s["split"] != "train":
            continue
        for t in s["tokens"]:
            rels.add(t.get("deprel", "_"))
    return sorted(rels)


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
    def __init__(self, sents, tok, upos2i, fv2i, prior_index, cands, use_prior,
                 deprel2i=None):
        self.sents = sents
        self.tok = tok
        self.upos2i = upos2i
        self.fv2i = fv2i           # {feature: {value: idx}}
        self.prior_index = prior_index   # {(kind,value): col}  for the multi-hot
        self.prior_dim = len(prior_index)
        self.cands = cands
        self.use_prior = use_prior
        self.deprel2i = deprel2i   # {relation: idx}; None disables dep parsing

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
        item = {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "sub_idx": sub_idx, "upos_y": upos_y,
            "feat_y": feat_y, "prior": prior, "n_words": n_words,
        }
        if self.deprel2i is not None:
            # head: 1-based gold head position (0 = root), matching the W+1
            # head dimension (index 0 = ROOT). Heads truncated out of the
            # window are ignored.
            head_y = torch.full((n_words,), -100, dtype=torch.long)
            deprel_y = torch.full((n_words,), -100, dtype=torch.long)
            for w in range(n_words):
                if w not in first:
                    continue
                t = toks[w]
                h = t.get("head", 0)
                if 0 <= h <= n_words:
                    head_y[w] = h
                deprel_y[w] = self.deprel2i.get(t.get("deprel", "_"), -100)
            item["head_y"] = head_y
            item["deprel_y"] = deprel_y
        return item


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
    dep = "head_y" in batch[0]
    hy = torch.full((B, maxw), -100, dtype=torch.long) if dep else None
    ry = torch.full((B, maxw), -100, dtype=torch.long) if dep else None
    for b, ex in enumerate(batch):
        L = ex["input_ids"].size(0); W = ex["n_words"]
        ids[b, :L] = ex["input_ids"]; am[b, :L] = ex["attention_mask"]
        sub[b, :W] = ex["sub_idx"]; uy[b, :W] = ex["upos_y"]
        pr[b, :W] = ex["prior"]; wmask[b, :W] = True
        for f in FEATURES:
            fy[f][b, :W] = ex["feat_y"][f]
        if dep:
            hy[b, :W] = ex["head_y"]; ry[b, :W] = ex["deprel_y"]
    out = {"input_ids": ids, "attention_mask": am, "sub_idx": sub,
           "upos_y": uy, "feat_y": fy, "prior": pr, "wmask": wmask}
    if dep:
        out["head_y"] = hy; out["deprel_y"] = ry
    return out


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class Biaffine(nn.Module):
    """Dozat & Manning biaffine scorer: s[b,i,j] = x[b,i]^T U y[b,j] (+ bias),
    with out_dim relation channels. x are dependents (W), y are heads (W+1)."""
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.out_dim = out_dim
        self.U = nn.Parameter(torch.zeros(out_dim, in_dim + 1, in_dim + 1))
        nn.init.xavier_uniform_(self.U)

    def forward(self, x, y):                          # x (B,Wx,d), y (B,Wy,d)
        x = torch.cat([x, x.new_ones(*x.shape[:-1], 1)], -1)
        y = torch.cat([y, y.new_ones(*y.shape[:-1], 1)], -1)
        s = torch.einsum("bxi,oij,byj->boxy", x, self.U, y)   # (B,out,Wx,Wy)
        return s.squeeze(1) if self.out_dim == 1 else s.permute(0, 2, 3, 1)


class MorphTagger(nn.Module):
    def __init__(self, n_upos, n_feat_vals, prior_dim, use_prior, n_deprels=0):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL_NAME)
        h = self.enc.config.hidden_size
        self.use_prior = use_prior
        self.n_deprels = n_deprels
        if use_prior:
            self.prior_proj = nn.Linear(prior_dim, h)
        self.drop = nn.Dropout(0.1)
        self.upos_head = nn.Linear(h, n_upos)
        self.feat_heads = nn.ModuleDict(
            {f: nn.Linear(h, n_feat_vals[f]) for f in FEATURES})
        if n_deprels:
            self.root = nn.Parameter(torch.zeros(1, 1, h))   # ROOT (head idx 0)
            nn.init.normal_(self.root, std=0.02)
            self.arc_dep = nn.Sequential(nn.Linear(h, ARC_DIM), nn.ReLU())
            self.arc_head = nn.Sequential(nn.Linear(h, ARC_DIM), nn.ReLU())
            self.arc_bi = Biaffine(ARC_DIM, 1)
            self.rel_dep = nn.Sequential(nn.Linear(h, REL_DIM), nn.ReLU())
            self.rel_head = nn.Sequential(nn.Linear(h, REL_DIM), nn.ReLU())
            self.rel_bi = Biaffine(REL_DIM, n_deprels)

    def forward(self, input_ids, attention_mask, sub_idx, prior):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        B, _, H = out.shape
        idx = sub_idx.unsqueeze(-1).expand(-1, -1, H)
        tok = torch.gather(out, 1, idx)               # (B, W, H) first-subword reps
        if self.use_prior:
            tok = tok + self.prior_proj(prior)
        tok = self.drop(tok)
        ups = self.upos_head(tok)
        feats = {f: self.feat_heads[f](tok) for f in FEATURES}
        if not self.n_deprels:
            return ups, feats
        tok_r = torch.cat([self.root.expand(B, -1, -1), tok], 1)   # (B, W+1, H)
        arc = self.arc_bi(self.arc_dep(tok), self.arc_head(tok_r))  # (B, W, W+1)
        rel = self.rel_bi(self.rel_dep(tok), self.rel_head(tok_r))  # (B,W,W+1,R)
        return ups, feats, arc, rel


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
    ap.add_argument("--dep", action="store_true",
                    help="also train a biaffine dependency-parse head (head + relation)")
    ap.add_argument("--init-ckpt",
                    help="warm-start from an existing checkpoint: adopts its "
                         "label spaces, feature set, and config, loads its "
                         "weights, then fine-tunes on --sentences")
    args = ap.parse_args()
    MODEL_NAME = args.model
    NORMALIZE = not args.no_normalize
    PREFIX_SPACE = not args.no_prefix_space
    if args.features_auto:
        FEATURES = derive_features(args.sentences)
    use_prior = not args.no_prior
    init_ck = None
    if args.init_ckpt:
        init_ck = torch.load(args.init_ckpt, map_location="cpu",
                             weights_only=False)
        MODEL_NAME = init_ck.get("model_name", MODEL_NAME)
        if init_ck.get("features"):
            FEATURES = init_ck["features"]
        NORMALIZE = init_ck.get("normalize", NORMALIZE)
        PREFIX_SPACE = init_ck.get("prefix_space", PREFIX_SPACE)
        use_prior = init_ck.get("use_prior", use_prior)
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={dev} model={MODEL_NAME} prior={use_prior} "
          f"features={len(FEATURES)} smoke={args.smoke}", flush=True)

    sents_path = Path(args.sentences)
    if init_ck:
        # adopt the checkpoint's label spaces so the state dict fits; labels
        # in the fine-tune data outside these spaces map to -100 (ignored)
        print("adopting label space from --init-ckpt...", flush=True)
        upos_list = init_ck["upos_list"]
        fv_list = init_ck["fv_list"]
        prior_index = {(tuple(k) if isinstance(k, list) else k): i
                       for i, k in enumerate(init_ck["prior_index"])}
        deprel_list = init_ck.get("deprels") or []
    else:
        print("building label space...", flush=True)
        upos_list, fv_list = build_label_space(sents_path, limit=2000 if args.smoke else None)
        prior_index = {}
        for u in upos_list:
            prior_index[("UPOS", u)] = len(prior_index)
        for f in FEATURES:
            for v in fv_list[f]:
                if v != NONE:
                    prior_index[(f, v)] = len(prior_index)
        deprel_list = derive_deprels(sents_path) if args.dep else []
    upos2i = {u: i for i, u in enumerate(upos_list)}
    fv2i = {f: {v: i for i, v in enumerate(fv_list[f])} for f in FEATURES}
    deprel2i = ({r: i for i, r in enumerate(deprel_list)}
                if deprel_list else None)
    print(f"  upos={len(upos_list)} feat-values={ {f: len(fv_list[f]) for f in FEATURES} } "
          f"prior_dim={len(prior_index)} deprels={len(deprel_list)}")

    print("loading sentences...", flush=True)
    train, dev_s = [], []
    eval_splits: dict = {}   # split name ("test", "test_<dialect>") -> sentences
    for k, line in enumerate(open(sents_path, encoding="utf-8")):
        s = json.loads(line)
        sp = s["split"]
        if sp == "train":
            train.append(s)
        elif sp == "dev":
            dev_s.append(s)
        elif sp.startswith("test"):
            eval_splits.setdefault(sp, []).append(s)
        if args.smoke and len(train) >= 400 and len(dev_s) >= 80:
            break
    if args.smoke:
        train = train[:400]; dev_s = dev_s[:80]; eval_splits = {}
    cands = json.load(open(args.candidates)) if use_prior else {}
    print(f"  train {len(train):,} dev {len(dev_s):,}  eval splits: "
          + (", ".join(f"{k}={len(v)}" for k, v in sorted(eval_splits.items()))
             or "none"))

    tok = AutoTokenizer.from_pretrained(
        MODEL_NAME, **({"add_prefix_space": True} if PREFIX_SPACE else {}))
    n_feat_vals = {f: len(fv_list[f]) for f in FEATURES}
    model = MorphTagger(len(upos_list), n_feat_vals, len(prior_index), use_prior,
                        n_deprels=len(deprel_list)).to(dev)
    if init_ck:
        model.load_state_dict(init_ck["state"])
        print(f"warm-started from {args.init_ckpt} "
              f"(epoch {init_ck.get('epoch')})", flush=True)
    ds_tr = TaggerData(train, tok, upos2i, fv2i, prior_index, cands, use_prior, deprel2i)
    ds_dv = TaggerData(dev_s, tok, upos2i, fv2i, prior_index, cands, use_prior, deprel2i)
    pad = tok.pad_token_id
    # The per-item work (tokenize + build the prior multi-hot) is CPU-bound and
    # was starving the GPU at num_workers=0 (~50% util). Parallel workers keep
    # the GPU fed; pin_memory + persistent_workers cut transfer + respawn cost.
    pin = (dev == "cuda")
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True,
                       collate_fn=functools.partial(collate, pad_id=pad),
                       num_workers=args.workers, pin_memory=pin,
                       persistent_workers=args.workers > 0,
                       prefetch_factor=4 if args.workers > 0 else None)
    dl_dv = DataLoader(ds_dv, batch_size=args.bs, shuffle=False,
                       collate_fn=functools.partial(collate, pad_id=pad),
                       num_workers=min(args.workers, 4), pin_memory=pin,
                       persistent_workers=min(args.workers, 4) > 0,
                       prefetch_factor=4 if args.workers > 0 else None)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    cel = nn.CrossEntropyLoss(ignore_index=-100)

    def masked_arc(arc, wmask):
        # mask invalid head positions (root idx 0 + real words are valid)
        amask = torch.cat([wmask.new_ones(wmask.size(0), 1), wmask], 1)  # (B,W+1)
        return arc.masked_fill(~amask.unsqueeze(1), -1e9)

    def rel_at(rel, head):    # gather relation logits at the given head index
        h = head.clamp(min=0)[:, :, None, None].expand(-1, -1, 1, rel.size(-1))
        return rel.gather(2, h).squeeze(2)            # (B, W, R)

    def run_eval(dl):
        model.eval()
        up_ok = up_tot = strict_ok = 0
        feat_ok = feat_tot = 0
        uas = las = dep_tot = 0
        with torch.no_grad():
            for b in dl:
                b = {k: (v.to(dev) if torch.is_tensor(v) else
                         {f: vv.to(dev) for f, vv in v.items()}) for k, v in b.items()}
                out = model(b["input_ids"], b["attention_mask"], b["sub_idx"], b["prior"])
                ul, fl = out[0], out[1]
                up = ul.argmax(-1); m = b["upos_y"] != -100
                up_ok += ((up == b["upos_y"]) & m).sum().item(); up_tot += m.sum().item()
                strict = (up == b["upos_y"]) & m
                for f in FEATURES:
                    fp = fl[f].argmax(-1); fm = b["feat_y"][f] != -100
                    feat_ok += ((fp == b["feat_y"][f]) & fm).sum().item()
                    feat_tot += fm.sum().item()
                    strict = strict & ((fp == b["feat_y"][f]) | (~fm))
                strict_ok += (strict & m).sum().item()
                if model.n_deprels:
                    ph = masked_arc(out[2], b["wmask"]).argmax(-1)   # pred head
                    pr = rel_at(out[3], ph).argmax(-1)               # pred rel
                    hy = b["head_y"]; hm = hy != -100
                    uas += ((ph == hy) & hm).sum().item()
                    las += ((ph == hy) & (pr == b["deprel_y"]) & hm).sum().item()
                    dep_tot += hm.sum().item()
        return (up_ok / max(up_tot, 1), feat_ok / max(feat_tot, 1),
                strict_ok / max(up_tot, 1),
                uas / max(dep_tot, 1), las / max(dep_tot, 1))

    print("training...", flush=True)
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for it, b in enumerate(dl_tr):
            b = {k: (v.to(dev) if torch.is_tensor(v) else
                     {f: vv.to(dev) for f, vv in v.items()}) for k, v in b.items()}
            out = model(b["input_ids"], b["attention_mask"], b["sub_idx"], b["prior"])
            ul, fl = out[0], out[1]
            loss = cel(ul.reshape(-1, ul.size(-1)), b["upos_y"].reshape(-1))
            for f in FEATURES:
                loss = loss + cel(fl[f].reshape(-1, fl[f].size(-1)), b["feat_y"][f].reshape(-1))
            # a batch with no dep-annotated tokens (e.g. all-Iliad fine-tune
            # sentences, head = -1 everywhere) would make CE(all-ignored) NaN
            if model.n_deprels and (b["head_y"] != -100).any():
                arc = masked_arc(out[2], b["wmask"])
                loss = loss + cel(arc.reshape(-1, arc.size(-1)), b["head_y"].reshape(-1))
                ra = rel_at(out[3], b["head_y"])      # relation logits at gold head
                loss = loss + cel(ra.reshape(-1, ra.size(-1)), b["deprel_y"].reshape(-1))
            opt.zero_grad(); loss.backward()
            # Without clipping, one pathological batch can blow up the
            # weights unrecoverably (observed 2026-07-12: loss 0.27 -> 8.3
            # in the last 500 iterations of epoch 2, dev UPOS 99.6 -> 53.0).
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            if it % 200 == 0:
                print(f"  ep{ep} it{it}/{len(dl_tr)} loss={loss.item():.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        u, fa, st, uas, las = run_eval(dl_dv)
        dep_str = f" UAS={uas*100:.1f}% LAS={las*100:.1f}%" if model.n_deprels else ""
        print(f"EPOCH {ep}: dev UPOS={u*100:.1f}% feats={fa*100:.1f}% strict={st*100:.1f}%"
              f"{dep_str} avg_loss={tot/len(dl_tr):.3f}", flush=True)
        # Checkpoint after every epoch so a mid-run interruption (e.g. a remote
        # box dropping) keeps the latest completed-epoch weights. The canonical
        # --out is always overwritten with the most recent epoch.
        ckpt = {"state": model.state_dict(), "upos_list": upos_list,
                "fv_list": fv_list, "prior_index": list(prior_index),
                "use_prior": use_prior, "epoch": ep,
                # config so eval/export/runtime reconstruct without CLI flags
                "model_name": MODEL_NAME, "features": FEATURES,
                "normalize": NORMALIZE, "prefix_space": PREFIX_SPACE,
                "max_len": MAXLEN, "deprels": deprel_list,
                "arc_dim": ARC_DIM, "rel_dim": REL_DIM}
        torch.save(ckpt, f"{args.out}.ep{ep}")
        torch.save(ckpt, args.out)
        print(f"saved {args.out} (epoch {ep})", flush=True)

    # Final eval on the held-out test split(s). "test" is the headline GUD
    # (Standard MG) accuracy; "test_<dialect>" splits give per-dialect numbers.
    if eval_splits:
        print("\n=== final test eval ===", flush=True)
        for name in sorted(eval_splits):
            ds = TaggerData(eval_splits[name], tok, upos2i, fv2i, prior_index,
                            cands, use_prior, deprel2i)
            dl = DataLoader(ds, batch_size=args.bs, shuffle=False,
                            collate_fn=functools.partial(collate, pad_id=pad), num_workers=0)
            u, fa, st, uas, las = run_eval(dl)
            dep_str = (f" UAS={uas*100:.1f}% LAS={las*100:.1f}%"
                       if model.n_deprels else "")
            print(f"  {name:16} UPOS={u*100:.1f}% feats={fa*100:.1f}% "
                  f"strict={st*100:.1f}%{dep_str}  "
                  f"({len(eval_splits[name])} sents)", flush=True)


if __name__ == "__main__":
    main()
