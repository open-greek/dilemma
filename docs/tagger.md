# POS tagging and dependency parsing

[README](../README.md)

- [Installation and use](#installation-and-use)
- [Languages and tokenization](#languages-and-tokenization)
- [Training sources and recorded evaluation](#training-sources-and-recorded-evaluation)
- [Device selection](#device-selection)
- [Throughput planning](#throughput-planning)

The `Tagger` API segments text, predicts POS and morphological features, and
adds dependency heads and relations. With `lemmatize=True` (the default), it
uses the predicted POS tags to select Dilemma lemmas.

## Installation and use

```bash
python3 -m pip install "dilemma-nlp[tagger-onnx]"
python3 -m dilemma download
```

```python
from dilemma import Tagger

tagger = Tagger(lang="grc")
for token in tagger.tag(["μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος"])[0]:
    print(token)
```

Each token has `form`, `raw_form`, `upos`, `feats`, `lemma`, `head`, and
`deprel` fields. Available dependency fields depend on the installed model.
Set `lemmatize=False` if you need only the tagger/parser output. The
`tagger-onnx` extra supplies the inference runtime (`onnxruntime` and
`tokenizers`); the `tagger` extra also installs PyTorch and `transformers`
for training and export.

## Languages and tokenization

| Language option | Model |
|---|---|
| `el` (default) | Modern Greek, using Greek-BERT |
| `grc` | Ancient Greek, using accent-preserving GreBerta |
| `med` | Medieval/Byzantine text, using the Ancient Greek backend |

Leading and trailing punctuation become separate `PUNCT` tokens. Elision and
aphaeresis apostrophes remain attached to their words. Modern Greek multiword
tokens expand (`στο` → `σ` + `το`), so output token counts can differ from
`text.split()`.

The Ancient model uses AGDT dependency relations (`PRED`, `SBJ`, `OBJ`,
`ATR`, `ADV`, and coordination/apposition variants). The Modern model uses
Universal Dependencies relations. See
[tagger initialization](../dilemma/tagger/__init__.py) for the current API.

## Training sources and recorded evaluation

The Modern model uses UD_Greek-GUD and the Cretan, Lesbian, and Messinian
dialect treebanks. The Ancient model uses GLAUx with the AGDT relation set.
NonCommercial training sources and Gorman's held-out annotations are excluded;
see [NOTICE](../NOTICE). The taggers do not require Morpheus.

These are recorded results from the held-out GUD and GLAUx test splits, not
measurements rerun for each Dilemma package release:

| Modern Greek tagger | UPOS | feats | UAS | LAS |
|------|:--:|:--:|:--:|:--:|
| **Dilemma `el`** (GUD + CC BY-SA dialects, ~38K tokens) | **97.9%** | **99.4%** | **89.5%** | **86.1%** |


| Ancient Greek tagger | UPOS | feats | UAS | LAS |
|------|:--:|:--:|:--:|:--:|
| **Dilemma `grc`** (GLAUx + AGDT, ~19M training tokens) | **96.3%** | **99.5%** | **83.6%** | **78.3%** |


Weights are built with `train_tagger.py` and `export_tagger_onnx.py`. Source
readers are `build/build_tagger_data.py`, `build/build_mg_tagger_data.py`, and
`convert_treebank.py`. The training extra and the source corpora are needed
to reproduce those builds.

## Device selection

ONNX execution providers are selected by `dilemma/_ort_providers.py`: CUDA
when a compatible `onnxruntime-gpu` installation is available, otherwise CPU.
Use `DILEMMA_ORT_PROVIDERS` to request a provider order, for example
`CUDAExecutionProvider,CPUExecutionProvider` or
`CoreMLExecutionProvider,CPUExecutionProvider`.

The constructor's `device` argument is advisory. Check `tagger.providers`
and `tagger.on_gpu` to see which provider is actually in use. ONNX Runtime
can fall back to CPU when a requested GPU provider is unavailable.

## Throughput planning

Tagging is a batched encoder workload. Lemmatization also performs SQLite
lookups, grammatical checks, and beam search on unresolved words. Measure the
two stages separately on representative text before choosing worker counts.
Rare vocabulary and OCR errors can increase the proportion needing beam
search, so ordinary-prose throughput is a poor estimate for a lexicon or
scholia run.

Earlier annotation runs recorded the following rates on a roughly
50,000-token Ancient Greek sample. They describe those workloads and machines;
they are not general hardware limits:

| Box | lemma, 1 worker | lemma saturation | tag (CPU, 1 proc) | coupled tag+lemma per shard on GPU |
|---|--:|--:|--:|--:|
| EPYC 7B12 (64c / 2.5-3.3 GHz, DDR4) | 54 tok/s | ~1,100 tok/s @ ~32 workers | 745 tok/s | 361 tok/s (12 shards, 90% GPU util) |
| Threadripper PRO 7995WX (96c/192t, 5.39 GHz, DDR5 8-ch) | 110 tok/s | ~2,460 tok/s @ ~48-61 workers | 2,531 tok/s | not measured |


In those runs, throughput leveled off as workers competed for memory
bandwidth. A separate CUDA session in every worker also used substantial
GPU memory. Start with a few workers, increase the count while measuring
aggregate throughput, and stop when additional workers no longer help.
A shared or separately batched tagger can avoid duplicating sessions across
lemmatizer workers. Large works may need to be split to keep one slow work
from determining the completion time of the whole run.
