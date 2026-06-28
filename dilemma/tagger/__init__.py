"""Dilemma POS tagger and dependency parser.

    from dilemma import Tagger

    tagger = Tagger(device="cuda")              # Modern Greek (default)
    tagger = Tagger(lang="grc", device="cuda")  # Ancient Greek
    results = tagger.tag(["Ο Αχιλλέας πολεμά"])
"""

import os
from pathlib import Path

import torch
from transformers import AutoModel

from .model import TaggerModel
from .labels import EL_POS_LABEL_COUNTS, EL_DP_LABEL_COUNT
from .weights import load_weights
from .tokenize import batch_tokenize
from .decode import decode_batch
from .segment import segment
from ._revisions import BERT_REVISIONS, TAGGER_WEIGHTS_REV

__version__ = "0.4.0"

# Maximum subwords per dynamic batch before flushing to GPU
_DEFAULT_MAX_SUBWORDS = 2048

# BERT models per language
_BERT_MODELS = {
    "el": "nlpaueb/bert-base-greek-uncased-v1",
    "grc": "pranaydeeps/Ancient-Greek-BERT",
    "med": "pranaydeeps/Ancient-Greek-BERT",  # Medieval/Byzantine Greek
}


def _resolve_weights_dir() -> Path:
    """Where to find local tagger weights.

    Resolution order, mirroring dilemma/core.py's data/model resolution:
      1. $DILEMMA_TAGGER_DIR
      2. ~/.cache/dilemma/tagger_model/
      3. <dilemma-repo-root>/weights/   (dev mode)
      4. Fallback: ~/.cache/dilemma/tagger_model/ even if it does not exist yet.
    """
    env = os.environ.get("DILEMMA_TAGGER_DIR")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    cache = Path.home() / ".cache" / "dilemma" / "tagger_model"
    if cache.exists():
        return cache
    repo_root = Path(__file__).resolve().parent.parent.parent
    dev = repo_root / "weights"
    if dev.exists():
        return dev
    return cache


_WEIGHTS_DIR = _resolve_weights_dir()


class Tagger:
    """Greek POS tagger and dependency parser with integrated lemmatization.

    Supports Modern Greek (el) via gr-nlp-toolkit weights and Ancient Greek
    (grc) via custom-trained heads on UD Perseus + PROIEL treebanks.

    Args:
        lang: "el" (Modern Greek, default) or "grc" (Ancient Greek).
        device: "cuda", "cpu", or None (auto-detect).
        pos_path: Path to POS weights. None = auto-detect.
        dp_path: Path to DP weights. None = auto-detect (MG only).
        checkpoint: Path to a joint checkpoint (AG). Overrides pos/dp_path.
        max_subwords: Maximum subwords per batch before flushing.
        lemmatize: Whether to include lemmas in output (requires Dilemma).
        lemma_cache: Pre-built {form: lemma} dict. Forms found in the
            cache skip Dilemma entirely. Useful for large corpora where
            most tokens map to known forms.
        dialect: Ancient Greek dialect for Dilemma normalization. One of
            "ionic", "doric", "aeolic", "koine", "auto", or None
            (default). Passed through to Dilemma to normalize dialect
            forms to Attic equivalents before lemma lookup.
    """

    def __init__(
        self,
        lang: str = "el",
        device: str | None = None,
        pos_path: str | None = None,
        dp_path: str | None = None,
        checkpoint: str | None = None,
        max_subwords: int = _DEFAULT_MAX_SUBWORDS,
        lemmatize: bool = True,
        lemma_cache: dict[str, str] | None = None,
        dialect: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_subwords = max_subwords
        self.lang = lang
        self._lemmatize = lemmatize
        self._lemma_cache = lemma_cache
        self._lemmatizer = None
        self._dialect = dialect
        self._using_greberta = False   # accent-preserving ONNX backend (grc)
        self._greberta = None

        if lang == "el":
            self._init_el(pos_path, dp_path)
        elif lang in ("grc", "med"):
            self._init_grc(checkpoint)
        else:
            raise ValueError(f"Unsupported language: {lang}. Use 'el', 'grc', or 'med'.")

        # The GreBerta backend is an ONNX session, not a torch module.
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

        if self._lemmatize:
            self._init_lemmatizer()

    def _init_el(self, pos_path, dp_path):
        """Initialize MG model.

        Checks for a fine-tuned single-backbone checkpoint first (from
        train.py --lang el). Falls back to gr-nlp-toolkit dual-backbone
        weights if no checkpoint exists.
        """
        # Prefer fine-tuned single-backbone checkpoint
        finetuned = _WEIGHTS_DIR / "el" / "tagger_el.pt"
        if finetuned.exists():
            self._init_grc_from_file(str(finetuned), fallback_bert="el")
            return

        # Fall back to gr-nlp-toolkit dual-backbone weights
        bert_name = _BERT_MODELS["el"]
        bert_rev = BERT_REVISIONS.get(bert_name)
        pos_bert = AutoModel.from_pretrained(bert_name, revision=bert_rev)
        dp_bert = AutoModel.from_pretrained(bert_name, revision=bert_rev)
        # Use MG-sized label counts for gr-nlp-toolkit weight compatibility
        self.model = TaggerModel(
            pos_bert, dp_bert,
            feat_sizes=EL_POS_LABEL_COUNTS,
            num_deprels=EL_DP_LABEL_COUNT,
        )
        load_weights(self.model, pos_path=pos_path, dp_path=dp_path, device="cpu")

    def _init_grc_from_file(self, checkpoint: str, fallback_bert: str = "grc"):
        """Load a single-backbone checkpoint (shared by grc, med, and fine-tuned el)."""
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

        bert_name = ckpt.get("bert_model", _BERT_MODELS[fallback_bert])
        feat_sizes = ckpt.get("feat_sizes")
        num_deprels = ckpt.get("num_deprels")

        bert = AutoModel.from_pretrained(
            bert_name, revision=BERT_REVISIONS.get(bert_name)
        )
        self.model = TaggerModel(
            bert,
            feat_sizes=feat_sizes,
            num_deprels=num_deprels,
        )
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)

    def _init_grc(self, checkpoint):
        """Initialize AG/Medieval model with single BERT (jointly trained).

        Prefers the accent-preserving GreBerta ONNX backend when its weights
        are present (grc); else the joint Ancient-Greek-BERT ONNX, else the
        PyTorch checkpoint.
        """
        # Accent-preserving GreBerta backend (preferred for grc, and for med -
        # Byzantine literary Greek is classicizing, so the AG model serves it).
        # Only auto-selected when no explicit checkpoint was requested.
        if checkpoint is None:
            from .grc_onnx import GreBertaTagger
            # med has no model of its own; fall back to the grc GreBerta weights.
            candidates = [self.lang, "grc"] if self.lang == "med" else [self.lang]
            for cand in candidates:
                gdir = _WEIGHTS_DIR / cand
                if GreBertaTagger.available(gdir):
                    self._greberta = GreBertaTagger(gdir)
                    self._using_greberta = True
                    self.model = None
                    return

        # Try ONNX if explicitly requested via checkpoint="onnx"
        onnx_dir = _WEIGHTS_DIR / self.lang / "onnx"
        if checkpoint == "onnx" and (onnx_dir / "tagger_joint.onnx").exists():
            try:
                from .onnx_model import TaggerONNX
                self.model = TaggerONNX(onnx_dir)
                self._using_onnx = True
                return
            except ImportError:
                pass  # onnxruntime not installed, fall back to PyTorch

        self._using_onnx = False

        if checkpoint is None:
            default = _WEIGHTS_DIR / self.lang / f"tagger_{self.lang}.pt"
            if default.exists():
                checkpoint = str(default)
            else:
                try:
                    from huggingface_hub import hf_hub_download
                    checkpoint = hf_hub_download(
                        repo_id="ciscoriordan/dilemma",
                        filename=f"tagger/{self.lang}/tagger_{self.lang}.pt",
                        revision=TAGGER_WEIGHTS_REV,
                    )
                except Exception:
                    raise FileNotFoundError(
                        f"Weights not found locally ({default}) or on HuggingFace. "
                        f"Train with: python train.py --lang {self.lang}"
                    )

        self._init_grc_from_file(checkpoint)

    def _init_lemmatizer(self):
        """Initialize Dilemma lemmatizer."""
        try:
            from dilemma import Dilemma
        except ImportError:
            # Try sibling directory (development layout)
            import sys
            dilemma_path = Path(__file__).parent.parent.parent / "dilemma"
            if dilemma_path.exists():
                sys.path.insert(0, str(dilemma_path))
                try:
                    from dilemma import Dilemma
                except ImportError:
                    self._lemmatize = False
                    return
            else:
                self._lemmatize = False
                return
        dilemma_lang = "all"
        self._lemmatizer = Dilemma(lang=dilemma_lang, device="cpu",
                                   dialect=self._dialect)
        self._lemmatizer.preload()

    def tag(
        self,
        sentences: list[str] | str,
        segment_text: bool = False,
    ) -> list[list[dict]]:
        """Tag a list of sentences, returning per-sentence token dicts.

        Handles dynamic batching internally - pass any number of sentences.

        Args:
            sentences: List of raw text strings (one sentence each), or a
                single string. When a single string is passed with
                segment_text=True, it is automatically split into sentences
                using Greek punctuation rules.
            segment_text: If True and input is a single string, auto-segment
                it into sentences before tagging. Has no effect when input
                is already a list.

        Returns:
            List of sentence results. Each sentence is a list of token dicts:
            {"form", "upos", "lemma", "feats", "head", "deprel"}
        """
        if isinstance(sentences, str):
            if segment_text:
                sentences = segment(sentences)
            else:
                sentences = [sentences]

        if not sentences:
            return []

        all_results = [None] * len(sentences)
        batch_indices = []
        batch_sentences = []
        est_subwords = 0

        for i, sent in enumerate(sentences):
            est = int(len(sent.split()) * 1.3) + 2
            if batch_sentences and est_subwords + est > self.max_subwords:
                results = self._tag_batch(batch_sentences)
                for idx, res in zip(batch_indices, results):
                    all_results[idx] = res
                batch_indices = []
                batch_sentences = []
                est_subwords = 0

            batch_indices.append(i)
            batch_sentences.append(sent)
            est_subwords += est

        if batch_sentences:
            results = self._tag_batch(batch_sentences)
            for idx, res in zip(batch_indices, results):
                all_results[idx] = res

        return all_results

    def _tag_batch(self, sentences: list[str]) -> list[list[dict]]:
        """Process a single batch through the model."""
        if self._using_greberta:
            # Accent-preserving GreBerta path: its own tokenization + decode,
            # then the shared (language-agnostic) lemmatization step.
            results = self._greberta.tag_sentences(sentences)
            self._add_lemmas(results)
            return results

        enc = batch_tokenize(sentences)

        if getattr(self, "_using_onnx", False):
            # ONNX: pass numpy arrays, get back torch tensors
            pos_logits, arc_scores, rel_scores = self.model(
                enc.input_ids, enc.attention_mask)
        else:
            with torch.inference_mode():
                input_ids = enc.input_ids.to(self.device)
                attention_mask = enc.attention_mask.to(self.device)
                pos_logits, arc_scores, rel_scores = self.model(
                    input_ids, attention_mask)

        results = decode_batch(
            pos_logits, arc_scores, rel_scores,
            enc.word_masks, enc.subword2word, enc.word_forms,
            raw_forms=enc.raw_forms,
        )

        self._add_lemmas(results)
        return results

    def _add_lemmas(self, results: list[list[dict]]) -> None:
        """Attach lemmas to decoded tokens in place (shared by all backends).

        Uses the polytonic ``raw_form`` (Dilemma's lookup is keyed on polytonic
        forms) and the predicted UPOS for POS-aware disambiguation.
        """
        if self._lemmatize and self._lemmatizer is not None:
            # Use raw (polytonic) forms for lemmatization when available,
            # since Dilemma's lookup tables are keyed on polytonic forms.
            all_forms = [
                t.get("raw_form", t["form"]) for sent in results for t in sent
            ]

            # Check lemma cache first, only send misses to Dilemma
            cache = self._lemma_cache
            if cache:
                all_lemmas = [cache.get(f) for f in all_forms]
                miss_indices = [i for i, l in enumerate(all_lemmas)
                                if l is None]
                if miss_indices:
                    miss_forms = [all_forms[i] for i in miss_indices]
                    if hasattr(self._lemmatizer, "lemmatize_batch_pos"):
                        all_upos = [t["upos"] for sent in results
                                    for t in sent]
                        miss_upos = [all_upos[i] for i in miss_indices]
                        miss_lemmas = self._lemmatizer.lemmatize_batch_pos(
                            miss_forms, miss_upos)
                    else:
                        miss_lemmas = self._lemmatizer.lemmatize_batch(
                            miss_forms)
                    for idx, lemma in zip(miss_indices, miss_lemmas):
                        all_lemmas[idx] = lemma
            else:
                # No cache - send everything to Dilemma
                if hasattr(self._lemmatizer, "lemmatize_batch_pos"):
                    all_upos = [t["upos"] for sent in results
                                for t in sent]
                    all_lemmas = self._lemmatizer.lemmatize_batch_pos(
                        all_forms, all_upos)
                else:
                    all_lemmas = self._lemmatizer.lemmatize_batch(all_forms)

            idx = 0
            for sent_tokens in results:
                for token in sent_tokens:
                    token["lemma"] = all_lemmas[idx]
                    idx += 1
        elif self._lemmatize:
            for sent_tokens in results:
                for token in sent_tokens:
                    token["lemma"] = token["form"]
