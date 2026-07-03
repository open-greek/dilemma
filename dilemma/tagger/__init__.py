"""Dilemma POS tagger and dependency parser.

    from dilemma import Tagger

    tagger = Tagger()             # Modern Greek (default)
    tagger = Tagger(lang="grc")   # Ancient Greek
    results = tagger.tag(["Ο Αχιλλέας πολεμά"])

The runtime is torch-free: a single ONNX morphological tagger + biaffine
dependency parser per language (onnxruntime + tokenizers), served by
``OnnxMorphTagger``, plus the Dilemma lemmatizer. torch / transformers are
needed only to (re)train and export the weights (``train_tagger.py`` /
``export_tagger_onnx.py``), not to run them.
"""

import os
from pathlib import Path

from .segment import segment
from ._revisions import TAGGER_WEIGHTS_REV

__version__ = "0.7.0"

# Maximum subwords per dynamic batch before flushing through the ONNX session.
_DEFAULT_MAX_SUBWORDS = 2048


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


def _download_onnx_morph(lang: str):
    """Fetch the ONNX morph tagger weights for ``lang`` from HuggingFace and
    return the directory holding them (``tagger.onnx`` + ``tagger_labels.json``
    + ``tokenizer/`` [+ ``mwt.json`` for el]), or ``None`` if unavailable.

    Used when the weights are not already present in ``_WEIGHTS_DIR`` (e.g. a
    fresh install that skipped ``python -m dilemma download``). Pinned to
    ``TAGGER_WEIGHTS_REV`` for reproducibility; downloads land in the standard
    HuggingFace cache.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:  # pragma: no cover - optional dependency
        return None
    try:
        root = snapshot_download(
            repo_id="ciscoriordan/dilemma",
            allow_patterns=[f"tagger/{lang}/*"],
            revision=TAGGER_WEIGHTS_REV,
        )
    except Exception:
        return None
    d = Path(root) / "tagger" / lang
    return d if d.exists() else None


class Tagger:
    """Greek POS tagger and dependency parser with integrated lemmatization.

    Supports Modern Greek (el, default) via Greek-BERT trained on the
    openly licensed UD_Greek-GUD + dialect treebanks (Cretan/Lesbian/Messinian),
    and Ancient Greek (grc) / Medieval-Byzantine Greek (med) via GreBerta
    trained on GLAUx + AGDT. None of these use the NonCommercial UD_Greek-GDT;
    the MG tagger was rebuilt off GUD + dialect treebanks so every training
    source is openly licensed.

    Args:
        lang: "el" (Modern Greek, default), "grc" (Ancient Greek), or "med".
        device: Accepted for backwards compatibility; advisory only. The ONNX
            tagger runs on CPU (onnxruntime CPUExecutionProvider).
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
        max_subwords: int = _DEFAULT_MAX_SUBWORDS,
        lemmatize: bool = True,
        lemma_cache: dict[str, str] | None = None,
        dialect: str | None = None,
    ):
        if lang not in ("el", "grc", "med"):
            raise ValueError(
                f"Unsupported language: {lang}. Use 'el', 'grc', or 'med'.")
        self.device = device or "cpu"   # advisory; the ONNX runtime is CPU
        self.max_subwords = max_subwords
        self.lang = lang
        self._lemmatize = lemmatize
        self._lemma_cache = lemma_cache
        self._lemmatizer = None
        self._dialect = dialect
        self._morph_onnx = None

        self._init_backend()

        if self._lemmatize:
            self._init_lemmatizer()

    def _init_backend(self):
        """Load the torch-free ONNX morph backend for this language.

        Uses local weights from ``_WEIGHTS_DIR/<lang>`` when present, else
        fetches them from HuggingFace. Medieval Greek (``med``) has no model of
        its own - Byzantine literary Greek is classicizing, so it falls back to
        the grc GreBerta weights. The same single-encoder architecture serves
        ``el`` (Greek-BERT) and ``grc``/``med`` (GreBerta): contextual UPOS +
        UD-feature tagging plus a biaffine dependency head.
        """
        from .onnx_backend import OnnxMorphTagger
        candidates = [self.lang, "grc"] if self.lang == "med" else [self.lang]
        for cand in candidates:
            gdir = _WEIGHTS_DIR / cand
            # med has no published model, so don't HF-fetch it (it would be a
            # 0-file round-trip); a local med/ is still honored, else fall to grc.
            if not OnnxMorphTagger.available(gdir) and cand != "med":
                gdir = _download_onnx_morph(cand)   # fetch from HF; dir|None
            if gdir is not None and OnnxMorphTagger.available(gdir):
                self._morph_onnx = OnnxMorphTagger(gdir)
                return
        raise FileNotFoundError(
            f"No tagger weights for '{self.lang}' found locally ({_WEIGHTS_DIR}) "
            f"or on HuggingFace. Run `python -m dilemma download` to fetch the "
            f"ONNX tagger, or train one with `python train_tagger.py`."
        )

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
        # Use the MG lookup for Modern Greek (avoids AG lemmas like την->ὅς,
        # Τρώες->Τρώς); the "all" lookup for AG/Medieval.
        dilemma_lang = "el" if self.lang == "el" else "all"
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
        """Process a single batch through the ONNX morph backend (grc/med/el):
        its own tokenization + decode, then the shared (language-agnostic)
        lemmatization step."""
        results = self._morph_onnx.tag_sentences(sentences)
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
