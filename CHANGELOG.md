# Changelog

All notable changes to Dilemma are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-07-03

### Changed
- The Ancient Greek (`grc`) tagger weights were fine-tuned on the Iliad
  composite gold plus a GLAUx mixture. No regression on general Greek
  (GLAUx test strict 93.6% -> 94.1%, UAS 83.6% -> 85.1%, LAS 78.3% ->
  79.9%); large gains on Homeric text. Pinned revision updated; tagger
  sub-version 0.6.0.
- `Tagger.tag` now segments leading/trailing punctuation into standalone
  `PUNCT` tokens ("ἄειδε," -> "ἄειδε" + ","), so the output can contain
  more tokens than `text.split()` (as MG multiword tokens already could).
  Elision/aphaeresis apostrophes stay attached (δ’, ῥ’, ’γώ); runs of one
  character stay one token ("..."). Lemma lookups also improve, since the
  lemmatizer now sees the clean word instead of "word+comma". Tagger
  sub-version 0.7.0.

## [1.0.0] - 2026-06-29

First stable release.

### Added
- Ancient Greek (`grc`/`med`) and Modern Greek (`el`) dependency parsing: the
  taggers now carry a biaffine dependency head, so every token has `head` and
  `deprel` (previously `el` had no shipped parser and `grc`/`med` had none).
- `dilemma.__version__`, resolved from the installed package metadata.

### Changed
- The tagger runtime is now torch-free: importing `dilemma.tagger` pulls in only
  `onnxruntime` + `tokenizers` (+ numpy). `torch`/`transformers` are needed only
  to (re)train and export weights.
- The `[tagger]` extra now installs the runtime plus torch + transformers, so
  `pip install dilemma-nlp[tagger]` followed by `Tagger()` works. For inference
  only, use `[tagger-onnx]`.
- Tagger weights auto-download from HuggingFace when not present locally.

### Removed (breaking)
- `Tagger()` no longer accepts `checkpoint`, `pos_path`, or `dp_path`; `device`
  is accepted but advisory (the ONNX runtime is CPU). Point `DILEMMA_TAGGER_DIR`
  at a directory to use custom weights.
- The legacy dual-BERT joint tagger stack (`TaggerModel`, the joint ONNX model,
  and the gr-nlp-toolkit weight loader) has been removed.

### Licensing
- Openly licensed by default: NonCommercial sources are never ingested. The
  committed NonCommercial PROIEL data was removed; PROIEL is dropped, UD Perseus
  is replaced by the AGDT original (CC BY-SA), and the NonCommercial GLAUx and
  PTA texts are filtered out. See NOTICE for the full per-source list.

[1.1.0]: https://github.com/open-greek/dilemma/releases/tag/1.1.0
[1.0.0]: https://github.com/open-greek/dilemma/releases/tag/1.0.0
