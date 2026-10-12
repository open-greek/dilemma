"""ONNX inference backend for Dilemma.

Drop-in replacement for the PyTorch model's generate() method.
Uses ONNX Runtime instead of PyTorch, eliminating the ~2GB dependency.

Usage (called automatically by Dilemma when ONNX files exist):
    from dilemma.onnx_inference import OnnxLemmaModel
    model = OnnxLemmaModel("/path/to/model_dir")
    results = model.generate(src_ids, src_pad_mask, num_beams=4)
"""

import json
import numpy as np
from pathlib import Path


class CharVocabLight:
    """Minimal vocab class that doesn't need PyTorch."""

    PAD = 0
    BOS = 1
    EOS = 2

    def __init__(self, vocab_path: Path):
        with open(vocab_path, encoding="utf-8") as f:
            d = json.load(f)
        self.char2id = d["char2id"]
        self.id2char = {int(k): v for k, v in d["id2char"].items()}

    def encode(self, text: str, add_bos=False, add_eos=False) -> list[int]:
        ids = []
        if add_bos:
            ids.append(self.BOS)
        for ch in text:
            ids.append(self.char2id.get(ch, self.PAD))
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids) -> str:
        chars = []
        for i in ids:
            i = int(i)
            if i == self.EOS:
                break
            if i in (self.PAD, self.BOS):
                continue
            chars.append(self.id2char.get(i, ""))
        return "".join(chars)

    def __len__(self):
        return len(self.char2id)


class OnnxLemmaModel:
    """ONNX-backed inference for the Dilemma transformer.

    Loads encoder.onnx and decoder_step.onnx, implements beam search
    in pure Python/NumPy. No PyTorch required.

    Optionally loads morphology head weights (heads.npz) for POS,
    nominal (gender/number/case), and verbal (tense/mood/voice)
    prediction via simple numpy matmul on pooled encoder output.

    The ONNX models use fixed sequence lengths to avoid MHA reshape issues:
    - Encoder: ONNX_MAX_LEN (48) for source
    - Decoder: ONNX_MAX_OUT (32) for target
    Inputs are padded to these lengths; the causal mask handles the rest.
    """

    ONNX_MAX_OUT = 32  # must match export_onnx.py ONNX_TGT_LEN

    def __init__(self, model_dir: str | Path):
        from ._ort_providers import make_session

        model_dir = Path(model_dir)
        self.encoder = make_session(model_dir / "encoder.onnx")
        self.decoder = make_session(model_dir / "decoder_step.onnx")
        self.on_gpu = bool(getattr(self.encoder, "dilemma_on_gpu", False))

        # Load morphology heads (optional, exported by export_onnx.py)
        self._heads = {}
        self._head_labels = {}
        heads_path = model_dir / "heads.npz"
        labels_path = model_dir / "head_labels.json"
        if heads_path.exists():
            data = np.load(str(heads_path))
            for name in ["pos_head", "nom_head", "verb_head"]:
                w_key = f"{name}_weight"
                b_key = f"{name}_bias"
                if w_key in data:
                    self._heads[name] = (data[w_key], data[b_key])
        if labels_path.exists():
            with open(labels_path, encoding="utf-8") as f:
                self._head_labels = json.load(f)

    @property
    def has_pos_head(self):
        return "pos_head" in self._heads

    @property
    def has_morph_heads(self):
        return "nom_head" in self._heads or "verb_head" in self._heads

    def _run_head(self, head_name: str, pooled: np.ndarray) -> np.ndarray:
        """Run a linear head: logits = pooled @ W^T + b."""
        weight, bias = self._heads[head_name]
        return pooled @ weight.T + bias

    def _pool_memory(self, memory: np.ndarray,
                     src_mask: np.ndarray) -> np.ndarray:
        """Mean-pool encoder output over non-padding positions.

        Args:
            memory: (batch, seq_len, d_model)
            src_mask: (batch, seq_len) bool, True = padding

        Returns:
            (batch, d_model)
        """
        mask = (~src_mask).astype(np.float32)  # (batch, seq_len), 1=real
        mask_3d = mask[:, :, None]  # (batch, seq_len, 1)
        pooled = (memory * mask_3d).sum(axis=1) / mask.sum(axis=1, keepdims=True).clip(min=1)
        return pooled

    def predict_pos(self, src: np.ndarray,
                    src_mask: np.ndarray = None) -> list[str]:
        """Predict POS tag for each word in the batch.

        Returns list of POS label strings (e.g. "verb", "noun", "adj").
        """
        if "pos_head" not in self._heads:
            return ["" for _ in range(src.shape[0])]
        if src_mask is None:
            src_mask = (src == 0)
        memory = self._encode(src, src_mask)
        pooled = self._pool_memory(memory, src_mask)
        logits = self._run_head("pos_head", pooled)
        labels = self._head_labels.get("pos_labels", {})
        return [labels.get(str(int(idx)), f"tag{idx}")
                for idx in logits.argmax(axis=1)]

    def predict_morph(self, src: np.ndarray,
                      src_mask: np.ndarray = None) -> list[dict]:
        """Predict nominal + verbal morphology for each word.

        Returns list of dicts with keys "nom" and/or "verb" (label strings
        like "m.s.n" or "pre.ind.act"), plus "nom_conf"/"verb_conf" scores.
        """
        if src_mask is None:
            src_mask = (src == 0)
        memory = self._encode(src, src_mask)
        pooled = self._pool_memory(memory, src_mask)

        results = [{} for _ in range(src.shape[0])]
        nom_labels = self._head_labels.get("nom_labels", {})
        verb_labels = self._head_labels.get("verb_labels", {})

        if "nom_head" in self._heads:
            logits = self._run_head("nom_head", pooled)
            # softmax for confidence
            exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_l / exp_l.sum(axis=1, keepdims=True)
            for i, idx in enumerate(logits.argmax(axis=1)):
                results[i]["nom"] = nom_labels.get(str(int(idx)), f"nom{idx}")
                results[i]["nom_conf"] = float(probs[i, idx])

        if "verb_head" in self._heads:
            logits = self._run_head("verb_head", pooled)
            exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_l / exp_l.sum(axis=1, keepdims=True)
            for i, idx in enumerate(logits.argmax(axis=1)):
                results[i]["verb"] = verb_labels.get(str(int(idx)), f"verb{idx}")
                results[i]["verb_conf"] = float(probs[i, idx])

        return results

    def _encode(self, src: np.ndarray, src_mask: np.ndarray) -> np.ndarray:
        return self.encoder.run(None, {
            "src": src,
            "src_key_padding_mask": src_mask,
        })[0]

    def _decode_step(self, tgt_ids: list[int], memory: np.ndarray,
                     mem_mask: np.ndarray) -> np.ndarray:
        """Run one decoder step. Pads tgt to fixed length, returns logits
        for the last real token position.

        Args:
            tgt_ids: list of token IDs (the partial output so far)
            memory: (1, src_len, d_model) encoder output
            mem_mask: (1, src_len) bool encoder padding mask

        Returns:
            logits: (vocab_size,) for the next token prediction
        """
        real_len = len(tgt_ids)
        padded = tgt_ids + [0] * (self.ONNX_MAX_OUT - real_len)
        tgt = np.array([padded[:self.ONNX_MAX_OUT]], dtype=np.int64)

        # Decoder returns (1, ONNX_MAX_OUT, vocab_size)
        all_logits = self.decoder.run(None, {
            "tgt": tgt,
            "memory": memory,
            "memory_key_padding_mask": mem_mask,
        })[0]

        # Extract logits at the last real position (causal mask ensures
        # this position only sees tokens before it, not the padding)
        return all_logits[0, real_len - 1, :]

    def generate(self, src: np.ndarray, src_key_padding_mask: np.ndarray = None,
                 max_len=32, bos_id=1, eos_id=2, num_beams=1):
        """Beam search decoding using ONNX sessions.

        Args:
            src: (batch, src_len) int64 array of source token IDs
            src_key_padding_mask: (batch, src_len) bool array (True = padding)
            num_beams: beam width (1 = greedy)

        Returns:
            list of lists of (token_ids, score) tuples, matching the
            PyTorch model's beam search return format.
        """
        if isinstance(src, list):
            src = np.array(src, dtype=np.int64)
        if src_key_padding_mask is None:
            src_key_padding_mask = (src == 0)
        if isinstance(src_key_padding_mask, list):
            src_key_padding_mask = np.array(src_key_padding_mask, dtype=bool)
        if len(src) > 128:
            results = []
            for begin in range(0, len(src), 128):
                results.extend(self.generate(
                    src[begin:begin + 128], src_key_padding_mask[begin:begin + 128],
                    max_len=max_len, bos_id=bos_id, eos_id=eos_id, num_beams=num_beams))
            return results

        memory = self._encode(src, src_key_padding_mask)
        batch_size = src.shape[0]

        beams = [[([bos_id], 0.0)] for _ in range(batch_size)]
        complete = [[] for _ in range(batch_size)]
        active = list(range(batch_size))
        # Bound decoder activations independently of the caller's word count.
        # Each word keeps its own beam ranking and stopping condition.
        decoder_batch_size = 128
        for _ in range(max_len):
            candidates = {i: [] for i in active}
            jobs = []
            for i in active:
                for ids, score in beams[i]:
                    if ids[-1] == eos_id:
                        complete[i].append((ids, score))
                    else:
                        jobs.append((i, ids, score))
            for begin in range(0, len(jobs), decoder_batch_size):
                chunk = jobs[begin:begin + decoder_batch_size]
                indexes = [job[0] for job in chunk]
                lengths = [len(job[1]) for job in chunk]
                tgt = np.zeros((len(chunk), self.ONNX_MAX_OUT), dtype=np.int64)
                for row, (_, ids, _) in enumerate(chunk):
                    tgt[row, :len(ids)] = ids
                logits = self.decoder.run(None, {
                    "tgt": tgt, "memory": memory[indexes],
                    "memory_key_padding_mask": src_key_padding_mask[indexes],
                })[0]
                for row, (i, ids, score) in enumerate(chunk):
                    last = logits[row, lengths[row] - 1, :]
                    max_val = last.max()
                    log_probs = last - max_val - np.log(np.exp(last - max_val).sum())
                    top = np.argpartition(log_probs, -num_beams)[-num_beams:]
                    top = top[np.argsort(log_probs[top])[::-1]]
                    for index in top:
                        candidates[i].append((ids + [int(index)],
                                              score + float(log_probs[index])))
            remaining = []
            for i in active:
                if not candidates[i]:
                    continue
                candidates[i].sort(key=lambda x: x[1], reverse=True)
                beams[i] = candidates[i][:num_beams]
                if all(ids[-1] == eos_id for ids, _ in beams[i]):
                    complete[i].extend(beams[i])
                else:
                    remaining.append(i)
            active = remaining
            if not active:
                break
        for i in range(batch_size):
            complete[i].extend(b for b in beams[i] if b[0][-1] != eos_id)
            complete[i].sort(key=lambda x: x[1], reverse=True)
        return [rows[:num_beams] for rows in complete]
