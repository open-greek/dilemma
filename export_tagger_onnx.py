#!/usr/bin/env python3
"""Export the trained GreBerta morphology tagger to ONNX for the dilemma package.

The Morpheus-candidate prior contributes ~1.3% (ablation), so we BAKE its
constant contribution (prior_proj(0) = prior_proj.bias) into the graph and drop
the prior input entirely. The exported model is a clean 3-input tagger:

    (input_ids, attention_mask, sub_idx) -> upos_logits, <feat logits x10>

so the dilemma runtime needs only onnxruntime + a BPE tokenizer - NO Morpheus,
no torch. Also emits tagger_labels.json (upos_list, fv_list, FEATURES) and saves
the GreBerta tokenizer next to the model.

Usage: ~/mlx-env/bin/python export_tagger_onnx.py
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

import train_tagger as T

DATA = Path(__file__).resolve().parent / "data" / "tagger"
CKPT = DATA / "tagger_model.pt"
OUT_ONNX = DATA / "tagger.onnx"
OUT_LABELS = DATA / "tagger_labels.json"
OUT_TOK = DATA / "tokenizer"


class ExportTagger(nn.Module):
    """Inference-only wrapper: first-subword gather + baked prior bias + heads,
    returning a flat tuple (ONNX has no dict outputs). When the model has a
    dependency head, also returns arc (B,W,W+1) and rel (B,W,W+1,R) scores."""
    def __init__(self, model):
        super().__init__()
        self.enc = model.enc
        self.upos_head = model.upos_head
        self.feat_heads = model.feat_heads
        self.n_deprels = model.n_deprels
        H = self.enc.config.hidden_size
        bias = (model.prior_proj.bias.detach().clone()
                if model.use_prior else torch.zeros(H))
        self.register_buffer("prior_bias", bias)
        if self.n_deprels:
            self.root = model.root
            self.arc_dep, self.arc_head = model.arc_dep, model.arc_head
            self.arc_bi = model.arc_bi
            self.rel_dep, self.rel_head = model.rel_dep, model.rel_head
            self.rel_bi = model.rel_bi

    def forward(self, input_ids, attention_mask, sub_idx):
        out = self.enc(input_ids=input_ids,
                       attention_mask=attention_mask).last_hidden_state
        H = out.size(-1)
        idx = sub_idx.unsqueeze(-1).expand(-1, -1, H)
        tok = torch.gather(out, 1, idx) + self.prior_bias
        ups = self.upos_head(tok)
        feats = [self.feat_heads[f](tok) for f in T.FEATURES]
        if not self.n_deprels:
            return (ups, *feats)
        tok_r = torch.cat([self.root.expand(tok.size(0), -1, -1), tok], 1)
        arc = self.arc_bi(self.arc_dep(tok), self.arc_head(tok_r))   # (B,W,W+1)
        rel = self.rel_bi(self.rel_dep(tok), self.rel_head(tok_r))   # (B,W,W+1,R)
        return (ups, *feats, arc, rel)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--out-dir", default=str(DATA))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_onnx = out_dir / "tagger.onnx"
    out_labels = out_dir / "tagger_labels.json"
    out_tok = out_dir / "tokenizer"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # reconstruct the train-time config from the checkpoint (encoder + feature
    # set differ between the AG GreBerta and MG Greek-BERT models).
    T.MODEL_NAME = ck.get("model_name", T.MODEL_NAME)
    T.FEATURES = ck.get("features", T.FEATURES)
    max_len = ck.get("max_len", T.MAXLEN)
    prefix_space = ck.get("prefix_space", True)
    upos_list, fv_list = ck["upos_list"], ck["fv_list"]
    prior_index = ck["prior_index"]
    nfv = {f: len(fv_list[f]) for f in T.FEATURES}
    deprels = ck.get("deprels", [])
    T.ARC_DIM = ck.get("arc_dim", T.ARC_DIM)
    T.REL_DIM = ck.get("rel_dim", T.REL_DIM)
    model = T.MorphTagger(len(upos_list), nfv, len(prior_index), ck["use_prior"],
                          n_deprels=len(deprels))
    model.load_state_dict(ck["state"])
    model.eval()
    wrap = ExportTagger(model).eval()

    tok = AutoTokenizer.from_pretrained(
        T.MODEL_NAME, **({"add_prefix_space": True} if prefix_space else {}))
    # a real 2-sentence batch so dynamic axes trace correctly
    words = [["μῆνιν", "ἄειδε", "θεά"], ["ἄνδρα", "μοι", "ἔννεπε", "Μοῦσα"]]
    enc = tok(words, is_split_into_words=True, padding=True, truncation=True,
              max_length=max_len, return_tensors="pt")
    sub = []
    for b in range(len(words)):
        wid = enc.word_ids(b)
        first = {}
        for pos, w in enumerate(wid):
            if w is not None and w not in first:
                first[w] = pos
        sub.append([first[i] for i in range(len(words[b]))]
                   + [0] * (max(len(x) for x in words) - len(words[b])))
    sub_idx = torch.tensor(sub, dtype=torch.long)

    out_names = ["upos"] + [f"feat_{f}" for f in T.FEATURES]
    dyn = {"input_ids": {0: "batch", 1: "subwords"},
           "attention_mask": {0: "batch", 1: "subwords"},
           "sub_idx": {0: "batch", 1: "words"}}
    for nm in out_names:
        dyn[nm] = {0: "batch", 1: "words"}
    if deprels:
        out_names += ["arc", "rel"]
        dyn["arc"] = {0: "batch", 1: "words", 2: "heads"}
        dyn["rel"] = {0: "batch", 1: "words", 2: "heads"}

    with torch.no_grad():
        ref = wrap(enc["input_ids"], enc["attention_mask"], sub_idx)
    torch.onnx.export(
        wrap, (enc["input_ids"], enc["attention_mask"], sub_idx), str(out_onnx),
        input_names=["input_ids", "attention_mask", "sub_idx"],
        output_names=out_names, dynamic_axes=dyn, opset_version=17,
        dynamo=False)
    print(f"wrote {out_onnx} ({out_onnx.stat().st_size/1e6:.0f} MB)")

    out_labels.write_text(json.dumps(
        {"upos_list": upos_list, "fv_list": fv_list, "features": T.FEATURES,
         "none": T.NONE, "model_name": T.MODEL_NAME, "max_len": max_len,
         "deprels": deprels},
        ensure_ascii=False), encoding="utf-8")
    tok.save_pretrained(str(out_tok))
    print(f"wrote {out_labels} + tokenizer -> {out_tok}")

    # validate ONNX == torch
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
    onx = sess.run(None, {"input_ids": enc["input_ids"].numpy().astype("int64"),
                          "attention_mask": enc["attention_mask"].numpy().astype("int64"),
                          "sub_idx": sub_idx.numpy()})
    maxerr = max(float(np.abs(r.numpy() - o).max()) for r, o in zip(ref, onx))
    ua = (ref[0].argmax(-1).numpy() == onx[0].argmax(-1)).all()
    print(f"validation: max|torch-onnx|={maxerr:.2e}  upos argmax match={bool(ua)}")


if __name__ == "__main__":
    main()
