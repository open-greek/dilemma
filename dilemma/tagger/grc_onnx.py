"""Accent-preserving GreBerta morphology backend for the Ancient-Greek path.

This replaces the de-accenting Ancient-Greek-BERT model on the ``grc`` path with
a fine-tuned GreBerta encoder (Apache-2.0, polytonic-preserving byte-level BPE).
It predicts UPOS + the core UD features IN CONTEXT and, unlike the joint morphy
model, does NOT de-accent the input, so breathing marks and accents - which
distinguish real minimal pairs - survive into the tagger.

Pure runtime: onnxruntime + tokenizers + numpy (no torch, no transformers). It
produces the same per-token dict shape the joint decoder does, except it does
not do dependency parsing, so ``head``/``deprel`` are ``None``. Lemmas are added
downstream by ``Tagger`` via the Dilemma lemmatizer (shared with the MG path).
"""
import json
from pathlib import Path

import numpy as np

PAD_ID = 1  # GreBerta / RoBERTa <pad>


class GreBertaTagger:
    """ONNX GreBerta UPOS + UD-feature tagger over whitespace-split words."""

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
                "The GreBerta tagger needs `onnxruntime` and `tokenizers` "
                "(`pip install onnxruntime tokenizers`).") from e
        self.sess = ort.InferenceSession(
            str(d / "tagger.onnx"), providers=["CPUExecutionProvider"])
        lab = json.loads((d / "tagger_labels.json").read_text(encoding="utf-8"))
        self.upos_list = lab["upos_list"]
        self.fv_list = lab["fv_list"]
        self.features = lab["features"]
        self.none = lab["none"]
        self.max_len = lab["max_len"]
        self.tok = Tokenizer.from_file(str(d / "tokenizer" / "tokenizer.json"))

    def tag_sentences(self, sentences, bs: int = 32):
        """sentences: list[str]. Returns list (per sentence) of token dicts
        {form, raw_form, upos, feats, head, deprel} aligned to whitespace words.
        head/deprel are None (no dependency parsing on this path)."""
        token_lists = [s.split() for s in sentences]
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
            ids = np.full((B, maxsub), PAD_ID, dtype=np.int64)
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
            for r, i in enumerate(idxs):
                row = []
                words = token_lists[i]
                for w in range(nwords[r]):
                    pu = self.upos_list[int(ups[r, w].argmax())]
                    pf = {}
                    for fi, f in enumerate(self.features):
                        v = self.fv_list[f][int(feats[fi][r, w].argmax())]
                        if v != self.none:
                            pf[f] = v
                    row.append({"form": words[w], "raw_form": words[w],
                                "upos": pu, "feats": pf,
                                "head": None, "deprel": None})
                out[i] = row
        return out
