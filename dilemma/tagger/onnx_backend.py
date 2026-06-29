"""Accent-preserving ONNX morphology backend shared by the Ancient-Greek and
Modern-Greek tagger paths.

Both languages use the same architecture (a transformer encoder + per-feature
classification heads) exported to a 3-input ONNX model
(``input_ids, attention_mask, sub_idx`` -> ``upos`` + one logit tensor per UD
feature). Only the encoder + label set differ, and both are captured in the
weights directory, so one runtime serves both:

- ``grc`` / ``med``: GreBerta (Apache-2.0, polytonic-preserving byte-level BPE).
- ``el``:            Greek-BERT (nlpaueb), trained on the openly licensed
                     UD_Greek-GUD + CC BY-SA dialect treebanks (Cretan/Lesbian/
                     Messinian), not the NonCommercial UD_Greek-GDT.

Pure runtime: onnxruntime + tokenizers + numpy (no torch, no transformers). It
does contextual UPOS + UD-feature tagging, and - when the model carries a
biaffine dependency head (a non-empty ``deprels`` in ``tagger_labels.json``,
as both the ``el`` and ``grc``/``med`` models now do) - greedy dependency
parsing, so each token gets a ``head`` (0 = ROOT) and ``deprel``. Older
heads-free models leave ``head``/``deprel`` as ``None``. Lemmas are added
downstream by ``Tagger`` via the Dilemma lemmatizer.
"""
import json
from pathlib import Path

import numpy as np

PAD_ID = 1  # RoBERTa <pad>; BERT uses 0 (set from the tokenizer below)


class OnnxMorphTagger:
    """ONNX UPOS + UD-feature tagger over whitespace-split words."""

    REQUIRED = ("tagger.onnx", "tagger_labels.json")

    @classmethod
    def available(cls, model_dir) -> bool:
        d = Path(model_dir)
        return all((d / f).exists() for f in cls.REQUIRED) \
            and (d / "tokenizer" / "tokenizer.json").exists()

    def __init__(self, model_dir):
        d = Path(model_dir)
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:  # pragma: no cover - optional extra
            raise ImportError(
                "The ONNX tagger needs `onnxruntime` and `tokenizers` "
                "(`pip install onnxruntime tokenizers`).") from e
        self.sess = ort.InferenceSession(
            str(d / "tagger.onnx"), providers=["CPUExecutionProvider"])
        lab = json.loads((d / "tagger_labels.json").read_text(encoding="utf-8"))
        self.upos_list = lab["upos_list"]
        self.fv_list = lab["fv_list"]
        self.features = lab["features"]
        self.none = lab["none"]
        self.max_len = lab["max_len"]
        self.deprels = lab.get("deprels", [])   # non-empty -> model has a dep head
        self.tok = Tokenizer.from_file(str(d / "tokenizer" / "tokenizer.json"))
        # pad id from the tokenizer (RoBERTa=1, BERT=0)
        pad = self.tok.token_to_id("<pad>")
        if pad is None:
            pad = self.tok.token_to_id("[PAD]")
        self.pad_id = pad if pad is not None else PAD_ID
        # Optional multiword-token split map (Modern Greek: στο -> σ + το). The
        # model is trained on syntactic words, so MWT surfaces are expanded here.
        mwt_path = d / "mwt.json"
        self.mwt = (json.loads(mwt_path.read_text(encoding="utf-8"))
                    if mwt_path.exists() else {})

    def _split_mwt(self, words):
        if not self.mwt:
            return words
        out = []
        for w in words:
            out.extend(self.mwt.get(w.lower(), [w]))
        return out

    def tag_sentences(self, sentences, bs: int = 32):
        """sentences: list[str]. Returns list (per sentence) of token dicts
        {form, raw_form, upos, feats, head, deprel} aligned to whitespace words.
        When the model has a dep head (self.deprels non-empty), head is the
        greedy biaffine arc (0 = ROOT, 1..n = the n words) and deprel the
        relation label; otherwise both are None."""
        token_lists = [self._split_mwt(s.split()) for s in sentences]
        out = [None] * len(token_lists)
        order = sorted(range(len(token_lists)), key=lambda i: len(token_lists[i]))
        for s in range(0, len(order), bs):
            idxs = [i for i in order[s:s + bs] if token_lists[i]]
            for i in (i for i in order[s:s + bs] if not token_lists[i]):
                out[i] = []
            if not idxs:
                continue
            encs = [self.tok.encode(token_lists[i], is_pretokenized=True)
                    for i in idxs]
            maxsub = min(self.max_len, max(len(e.ids) for e in encs))
            maxw = max(len(token_lists[i]) for i in idxs)
            B = len(idxs)
            ids = np.full((B, maxsub), self.pad_id, dtype=np.int64)
            mask = np.zeros((B, maxsub), dtype=np.int64)
            sub = np.zeros((B, maxw), dtype=np.int64)
            nwords = []
            for r, (i, e) in enumerate(zip(idxs, encs)):
                eid = e.ids[:maxsub]
                ids[r, :len(eid)] = eid
                mask[r, :len(eid)] = 1
                first = {}
                for pos, w in enumerate(e.word_ids[:maxsub]):
                    if w is not None and w not in first:
                        first[w] = pos
                nwords.append(len(token_lists[i]))
                for w in range(nwords[-1]):
                    sub[r, w] = first.get(w, 0)
            res = self.sess.run(None, {"input_ids": ids, "attention_mask": mask,
                                       "sub_idx": sub})
            ups, feats = res[0], res[1:]
            nf = len(self.features)
            arc = res[1 + nf] if self.deprels else None   # (B, maxw, maxw+1)
            rel = res[2 + nf] if self.deprels else None   # (B, maxw, maxw+1, R)
            for r, i in enumerate(idxs):
                row = []
                words = token_lists[i]
                nw = nwords[r]
                for w in range(nw):
                    pu = self.upos_list[int(ups[r, w].argmax())]
                    pf = {}
                    for fi, f in enumerate(self.features):
                        v = self.fv_list[f][int(feats[fi][r, w].argmax())]
                        if v != self.none:
                            pf[f] = v
                    head = deprel = None
                    if self.deprels:
                        # valid heads: ROOT (index 0) + the nw real words
                        hi = int(arc[r, w, :nw + 1].argmax())   # 0=root, 1..nw
                        head = hi
                        deprel = self.deprels[int(rel[r, w, hi].argmax())]
                    row.append({"form": words[w], "raw_form": words[w],
                                "upos": pu, "feats": pf,
                                "head": head, "deprel": deprel})
                out[i] = row
        return out
