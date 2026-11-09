# Monotonic Modern Greek next-word model

[README](../README.md)

- [Build](#build)
- [Output](#output)
- [Which spellings are counted](#which-spellings-are-counted)
- [Counting](#counting)
- [Export](#export)
- [Pruning](#pruning)
- [Vocabulary size](#vocabulary-size)
- [Recorded model](#recorded-model)
- [Evaluation](#evaluation)
- [Tests](#tests)
- [Known limitations](#known-limitations)

`train_lm.py --lang el` and `export_lm.py --lang el` build the trigram
next-word model of a monotonic Modern Greek keyboard bundle,
`build/lm/el_ngram.bin`, in the same GNLM v2 format as the Ancient Greek
model (see [Next-word prediction](language-model.md)). It is trained on the
training split of the [monotonic corpus](el-monotonic-corpus.md), and its
vocabulary holds only spellings of the
[monotonic word list](el-monotonic-word-list.md), so the model never
proposes a word the list rejects.

## Build

```bash
python extract_el_monotonic_lm.py      # the corpus (el-monotonic-corpus.md)
python export_el_monotonic.py          # the word list (el-monotonic-word-list.md)
python train_lm.py --lang el --sanity  # first shard of every source, 5% sample,
                                       # into build/lm_el_sanity, under a minute
python train_lm.py --lang el           # build/lm_el, about 1.5 minutes the first
                                       # time, 25 s once the shards are encoded
python export_lm.py --lang el          # build/lm/el_ngram.{bin,version}, about 15 s
python eval/eval_el_ngram.py           # whole dev split, about 6 minutes
python -m pytest tests/test_el_ngram.py -q
```

The first training run encodes every shard of the corpus (below) into
`build/lm_el_encoded/` with one worker process per core; later runs reuse
those arrays while the shards, the word list and the resolution code are
unchanged. A run with `--sanity`, `--sample-tokens` or `--sample-fraction`
cannot write `build/lm_el`. Counting the whole capped corpus needs about
12 GB of memory.

`eval/tune_el_ngram.py` repeats train, export and evaluation over a grid of
pruning settings, training sizes and vocabulary sizes; it produced the
tables below.

## Output

`train_lm.py --lang el` writes the intermediates of the Ancient Greek
trainer to `build/lm_el/`: `vocab.json`, `unigrams.json`, `bigrams.tsv.gz`,
`trigrams.tsv.gz`, `type_counts.tsv.gz` (the training count of every
word-list spelling seen, in the vocabulary or not), the dev sentences
(`dev_sentences.txt` and `dev_sentences_<source>.txt`, resolved as below)
and `stats.json` (resolution counts, per-source tokens, vocabulary
coverage, the pruning settings, and the sha256 of the corpus manifest, the
corpus registry and the word list).

`export_lm.py --lang el` writes `build/lm/el_ngram.bin` and the JSON sidecar
`build/lm/el_ngram.version`. Besides the fields of the Ancient Greek
sidecar (`semver`, `dilemma_commit`, `format_version` 2, the top-K values,
`vocab_size`, the context counts, `total_tokens`, `artifact_size_bytes`,
`oov_dictionary_sha256`), it records `lang`, `artifact_sha256`,
`corpus_manifest_sha256`, `el_lm_corpora_sha256`,
`vocab_dictionary_sha256`, `train_commit`, `train_words_weighted`,
`seen_spellings`, `in_vocab_word_share`, `unk_word_share`, the pruning
settings, the per-source token counts and weights, and
`oov_counts_saturated`. `dilemma_commit` is the full commit hash, with
`-dirty` when tracked files differed from it.

Export refuses intermediates counted against another word list than the
one it chooses the out-of-vocabulary table from (`dictionary_sha256` in
`stats.json` against the sha256 of `build/hunspell/el_monotonic.dic`), so
the table and the vocabulary always come from the same list. Re-run both
steps after every rebuild of the word list. The binary and the
intermediates are build products and are not committed.

## Which spellings are counted

The keyboard looks a word up as an NFC string, first as typed and then
lowercased with the final-sigma rule (`LMReader.idCaseInsensitive`). There
is no accent-insensitive lookup. Training resolves every corpus token to
the word-list spelling the keyboard would look up, or to `<UNK>`:

| Token | Counted as | Why |
|---|---|---|
| `<NUM>`, `<FOREIGN>` | `<UNK>` | Not words of the list |
| Abbreviation with its period (`κ.`, `π.χ.`) | `<UNK>` | The list keeps abbreviations apart |
| `όνομά` in `το όνομά μου` | `όνομά` | The double-accented form of a list spelling is a token of its own, so the keyboard can compare `όνομά μου` with `όνομα μου` (below) |
| `Θα` at the start of a sentence | `θα` | The lowercase is a list spelling |
| `Αθήνα` at the start of a sentence | `Αθήνα` | Only the capitalized spelling is in the list |
| `Νίκη` inside a sentence | `Νίκη` | The name is a list spelling of its own beside `νίκη` |
| `Βουλή` inside a sentence | `βουλή` | Not a list spelling as written; the lowercase is, which is what the reader's fallback finds |
| `ΠΑΣΟΚ`, `ΕΣΥ`, `ΝΑΤΟ` | `<UNK>` | Capitals in running text are acronyms and names; resolving them by their letters would count the health service ΕΣΥ as `εσύ` |
| `Ο`, `Η`, `Ή` | `ο`, `η`, `ή` | The capital articles and the disjunctive; any other one-letter capital is a label (`τμήμα Α`) |
| Anything else | `<UNK>` | Not a list spelling |

A double-accented token is kept as it is when it is the form the keyboard
writes before an enclitic for a list spelling (`el_accent.enclitic_twin`:
one tonos, stressed on the third syllable from the end with or without
synizesis, the second tonos on the last vowel), resolved like the
spelling it doubles. The list has 267,461 such forms. One seen at least 3
times whose host is in the vocabulary joins the vocabulary, outside its
150,000 types, with its id after the list spellings; a host ranks by its
count plus its twin's. A rarer one is counted as its host. These forms
are no words of the list: they are never homograph spellings, never go
into the out-of-vocabulary table, and the keyboard must never offer them
as predictions or completions (the evaluations hide every vocabulary
token with two tonos, `GNLMReader.hide_double_accented`). Any other
double-accented token keeps only its first tonos (`fold_enclitic_accent`).

The recorded run resolves 195,633,147 tokens under the corpus caps,
before source weighting and without sentence boundaries. Its resolution counts
are below; the two enclitic rows are subsets of the other outcomes.

| Resolution | Tokens |
|---|---|
| exact | 169,678,270 |
| lowered sentence initial | 8,951,394 |
| lowered inside sentence | 6,168,060 |
| unk placeholder | 5,425,695 |
| unk abbreviation | 1,244,243 |
| unk capitals | 1,670,885 |
| unk not in word list | 2,494,600 |
| enclitic folded | 39,642 |
| enclitic twin | 474,882 |

After weighting and sentence boundaries, training counts
221,240,505 tokens. The vocabulary holds
5,791 double-accented forms.

## Counting

`build/el_ngram.py` counts with numpy instead of the `Counter` of the
Ancient Greek path, which would need tens of gigabytes for 200M tokens:

1. **Encode.** Each shard becomes an int32 array: the spelling's index in
   the sorted word list (after the 4 reserved ids) times 16, plus a 4-bit
   code that says how the token was resolved. The arrays carry the start of
   every document and its cap and sample buckets.
2. **Select.** A training document is kept when its cap bucket is below
   the threshold the corpus manifest records for its source
   (`el_corpus.cap_bucket`), the same documents `el_corpus.iter_corpus_sentences`
   yields. A source's weight (Tatoeba 10, Wiktionary examples 5, Common
   Voice 2, the rest 1) repeats its counts. `--sample-tokens N` keeps an
   independent document-level sample (its own hash seed) of about N
   weighted tokens, for the subsamples the pruning was tuned on.
3. **Vocabulary.** The most frequent word-list spellings by weighted count,
   plus the four reserved tokens, plus the double-accented forms of the
   spellings in it seen at least 3 times.
4. **Count.** Bigram and trigram keys are packed into uint64, 21 bits per
   id, counted with `np.unique` over chunks of whole sentences, and the
   sorted runs are merged.

Counting is exact: the tests compare it with `Counter` counts at chunk
sizes that split sentences every way.

### The out-of-vocabulary count table

The exporter appends the training counts of the word-list spellings that
are out of the vocabulary but share their letters with another spelling
(`πότε` and `ποτέ`), so the keyboard can choose between two such spellings
by evidence. The table stores one byte per count, which `LMReader` reads as
it is. At 200M tokens such a spelling can occur more than 255 times outside
a small vocabulary, and a capped count would make two frequent twins look
equally likely. The trainer therefore adds every such spelling seen more
than 255 times to the vocabulary, where counts are 32-bit, and export
reports `oov_counts_saturated` (0 in every recorded build). At the sizes
measured here, 80,000 to 200,000 types, the vocabulary already holds all
of them.

## Export

Export uses the Ancient Greek settings: rows of the 30 most likely
continuations after a word and the 15 most likely after two, 10 global
fallbacks, homograph spellings seen at least 3 times kept in a row past
that cut, and the bigram row of a homograph spelling completed with every
continuation seen at least 3 times. In this vocabulary 6,270
spellings share their letters with another (`που` and `πού`, `πότε` and
`ποτέ`, `όρος` and `ορός`).

Three things differ from the Ancient Greek export:

- **No rows headed by `<UNK>`.** The reader resolves a context word to an
  id only when the word is in the vocabulary, so a bigram row headed by
  `<UNK>`, or a trigram row with `<UNK>` in its context, is never read.
  `--drop-unknown-contexts` (the default for `el`) leaves them out; rows
  headed by `<s>` stay, because the reader asks for `<s>` at the start of a
  sentence.
- **Pre-pruned intermediates.** At full size the counts hold 58M distinct
  trigrams. The trainer writes only what export can keep for
  the cuts it was run with: in every row the entries down to the count of
  the last top-K entry (ties included), the homograph spellings, the
  completed homograph rows, and the bigram of every kept trigram context.
  Export cuts a row by count and then by id, so these files give the same
  binary as the full tables; a test checks they are byte-identical.
  `stats.json` records the cuts, and export refuses settings that would
  need entries the files do not have.
- **Host and double-accented form pairs.** Past the cut, the bigram row
  of a host or its double-accented form keeps every enclitic the keyboard
  decides the accent for (`μου`, `σου`, `μας`, `σας`, `του`, `της`,
  `τους`, `το`, `τα`, `τον`, `την`, `τις`) seen at least 3 times after
  it; a trigram row whose second word is one of the pair keeps those
  enclitics seen at least 3 times; and a row that keeps one of the pair
  keeps the other when it has it. The keyboard scores both readings from
  rows of the same depth. The first 30 or 15 entries of every row are
  unchanged.

## Pruning

These tuning experiments use the earlier word list of commit `2951b0e`,
before the double-accented vocabulary was added. They explain the defaults;
current results appear under [Evaluation](#evaluation).

Three knobs, the same as for Ancient Greek, decide the size:

- `--min-count-tri`: trigram entries seen fewer times are dropped.
- `--min-bigram-for-tri`: trigram rows whose two-word context was seen
  fewer times are dropped; the keyboard backs off to the bigram row.
- `--min-count-bi`: bigram entries seen fewer times are dropped.

Counts are weighted, so a trigram seen once in Tatoeba counts 10 and a
trigram seen once in Common Voice counts 2. A minimum of 4 keeps single
occurrences in Tatoeba and the Wiktionary examples but drops those in
Common Voice and the large sources.

The knobs were first compared on document samples of 20M and 50M weighted
tokens (150,000 types, every combination of `--min-bigram-for-tri` 3, 5,
10, 20, 40, `--min-count-tri` 1, 2 and `--min-count-bi` 1, 2), evaluated on
3,000 dev sentences per source. Pseudo-perplexity is defined under
[Evaluation](#evaluation); top-1 and top-3 count every word target, so a
target outside the vocabulary is a miss. A selection:

| Sample | Min. context count | Min. trigram count | Min. bigram count | MB | Trigram rows | Pseudo-perplexity | Top-1 | Top-3 | Conversational top-1 | Conversational top-3 | Completion after 2 letters, top-3 | After 3 letters, top-3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20M | 3 | 1 | 1 | 42.9 | 607,033 | 1306 | 15.04 | 24.94 | 10.25 | 18.62 | 57.59 | 56.25 |
| 20M | 3 | 2 | 1 | 25.0 | 374,335 | 1415 | 15.25 | 25.38 | 10.41 | 19.00 | 57.20 | 55.82 |
| 20M | 3 | 2 | 2 | 20.0 | 374,335 | 1491 | 15.30 | 25.41 | 10.48 | 19.02 | 56.79 | 55.40 |
| 20M | 10 | 1 | 1 | 27.7 | 180,134 | 1410 | 14.92 | 24.91 | 10.11 | 18.64 | 57.34 | 55.99 |
| 20M | 10 | 2 | 1 | 20.4 | 168,440 | 1477 | 14.94 | 25.09 | 10.13 | 18.74 | 57.08 | 55.69 |
| 20M | 40 | 1 | 1 | 17.9 | 44,467 | 1617 | 14.42 | 24.51 | 9.63 | 18.27 | 56.83 | 55.44 |
| 20M | 40 | 2 | 1 | 16.6 | 44,405 | 1631 | 14.42 | 24.51 | 9.63 | 18.27 | 56.77 | 55.38 |
| 50M | 3 | 1 | 1 | 79.3 | 1,246,314 | 1111 | 15.79 | 26.15 | 11.28 | 19.96 | 58.96 | 57.69 |
| 50M | 3 | 2 | 1 | 43.9 | 809,671 | 1197 | 15.94 | 26.49 | 11.38 | 20.35 | 58.57 | 57.27 |
| 50M | 3 | 2 | 2 | 37.6 | 809,671 | 1234 | 15.97 | 26.52 | 11.44 | 20.40 | 58.32 | 57.01 |
| 50M | 10 | 1 | 1 | 49.4 | 397,783 | 1191 | 15.63 | 26.08 | 11.07 | 19.92 | 58.74 | 57.45 |
| 50M | 10 | 2 | 1 | 34.0 | 375,913 | 1245 | 15.64 | 26.22 | 11.10 | 20.12 | 58.47 | 57.16 |
| 50M | 40 | 1 | 1 | 28.7 | 107,409 | 1355 | 15.13 | 25.66 | 10.51 | 19.51 | 58.22 | 56.91 |
| 50M | 40 | 2 | 1 | 25.8 | 107,318 | 1368 | 15.13 | 25.66 | 10.51 | 19.51 | 58.15 | 56.85 |

On both samples, raising `--min-count-tri` from 1 to 2 removes almost half
of the file and slightly raises top-1 and top-3, at a cost in
pseudo-perplexity: a trigram seen once is a poor first suggestion, and the
reader lists trigram entries before bigram entries. Raising
`--min-bigram-for-tri` lowers top-1 at every size, because it removes
trigram rows that are read. `--min-count-bi 2` saves about 5 MB at a small
cost in completion recall.

At full size (200.7M weighted tokens):

| Min. context count | Min. trigram count | Min. bigram count | MB | Trigram rows | Pseudo-perplexity | Top-1 | Top-3 | Conversational pseudo-perplexity | Conversational top-1 | Conversational top-3 | Completion after 2 letters, top-3 | After 3 letters, top-3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 3 | 1 | 85.3 | 1,766,646 | 989 | 16.87 | 27.85 | 2689 | 12.88 | 22.32 | 60.08 | 58.89 |
| 5 | 3 | 1 | 80.4 | 1,543,372 | 994 | 16.82 | 27.81 | 2699 | 12.84 | 22.29 | 60.07 | 58.88 |
| 3 | 3 | 2 | 77.1 | 1,766,646 | 1001 | 16.87 | 27.86 | 2767 | 12.88 | 22.33 | 59.96 | 58.78 |
| 5 | 3 | 2 | 72.2 | 1,543,372 | 1006 | 16.82 | 27.82 | 2778 | 12.84 | 22.29 | 59.95 | 58.77 |
| 10 | 3 | 1 | 70.3 | 1,085,677 | 1011 | 16.66 | 27.69 | 2774 | 12.56 | 22.07 | 60.02 | 58.84 |
| 3 | 4 | 1 | 69.7 | 1,306,501 | 1026 | 16.78 | 27.71 | 2765 | 12.79 | 22.26 | 59.88 | 58.70 |
| 5 | 4 | 1 | 68.6 | 1,255,748 | 1027 | 16.77 | 27.70 | 2768 | 12.77 | 22.24 | 59.88 | 58.70 |
| 10 | 3 | 2 | 62.1 | 1,085,677 | 1023 | 16.66 | 27.69 | 2854 | 12.57 | 22.08 | 59.91 | 58.72 |
| 10 | 4 | 1 | 61.7 | 940,227 | 1041 | 16.63 | 27.60 | 2837 | 12.53 | 22.05 | 59.85 | 58.67 |
| 3 | 4 | 2 | 61.5 | 1,306,501 | 1038 | 16.78 | 27.71 | 2845 | 12.79 | 22.26 | 59.77 | 58.59 |
| 3 | 5 | 1 | 61.4 | 1,063,900 | 1055 | 16.70 | 27.59 | 2819 | 12.72 | 22.15 | 59.75 | 58.57 |
| 5 | 4 | 2 | 60.4 | 1,255,748 | 1040 | 16.77 | 27.70 | 2848 | 12.78 | 22.25 | 59.77 | 58.59 |
| 10 | 5 | 1 | 56.2 | 828,301 | 1065 | 16.61 | 27.52 | 2884 | 12.49 | 21.97 | 59.72 | 58.54 |
| 10 | 4 | 2 | 53.5 | 940,227 | 1053 | 16.64 | 27.61 | 2919 | 12.53 | 22.06 | 59.74 | 58.55 |
| 3 | 5 | 2 | 53.2 | 1,063,900 | 1067 | 16.71 | 27.60 | 2901 | 12.73 | 22.16 | 59.64 | 58.46 |
| 3 | 6 | 1 | 52.7 | 772,428 | 1086 | 16.61 | 27.47 | 2956 | 12.48 | 21.92 | 59.59 | 58.42 |
| 10 | 6 | 1 | 51.1 | 698,574 | 1090 | 16.57 | 27.44 | 2969 | 12.44 | 21.89 | 59.58 | 58.41 |
| 10 | 5 | 2 | 48.0 | 828,301 | 1078 | 16.61 | 27.53 | 2967 | 12.50 | 21.98 | 59.61 | 58.43 |
| 3 | 8 | 1 | 46.0 | 572,692 | 1126 | 16.49 | 27.29 | 3047 | 12.38 | 21.77 | 59.40 | 58.22 |
| 10 | 8 | 1 | 45.7 | 557,696 | 1127 | 16.48 | 27.28 | 3050 | 12.38 | 21.76 | 59.40 | 58.21 |
| 3 | 6 | 2 | 44.5 | 772,428 | 1099 | 16.62 | 27.48 | 3041 | 12.49 | 21.93 | 59.48 | 58.30 |
| 10 | 6 | 2 | 42.9 | 698,574 | 1104 | 16.57 | 27.45 | 3055 | 12.44 | 21.90 | 59.47 | 58.29 |
| 3 | 8 | 2 | 37.8 | 572,692 | 1139 | 16.49 | 27.30 | 3135 | 12.39 | 21.78 | 59.29 | 58.10 |
| 10 | 8 | 2 | 37.5 | 557,696 | 1141 | 16.48 | 27.29 | 3138 | 12.38 | 21.77 | 59.29 | 58.10 |

A minimum context count at or below the minimum trigram count has no
effect, since a trigram seen n times has a context seen at least n times,
so such settings appear once in the table.

The defaults are a minimum context count of 5, a minimum trigram count of
4 and a minimum bigram count of 2: 60.4 MB, the best top-1 and top-3 of the
settings near 60 MB. Against the largest setting measured (85.3 MB) they
lose 0.10 points of top-1, 0.15 of top-3 and 0.3 of completion recall.
Raising the trigram minimum is the way to a smaller file: from 4 to 8 it
saves 23 MB for 0.28 points of top-1. At about the same size, the full
corpus is a point of top-1 ahead of the 50M sample (16.77% against 15.79%
at about 60 MB).

## Vocabulary size

With the pruning fixed, the vocabulary was measured at 80,000, 120,000,
150,000 and 200,000 types on the same 3,000 dev sentences per source.
Completion recall counts every word-list target, so a target outside the
vocabulary is a miss. Pseudo-perplexity is over vocabulary targets, a set
that grows with the vocabulary, so it does not compare across rows.

| Vocabulary | MB | Word targets in vocabulary | Top-1 | Top-3 | Pseudo-perplexity | Completion after 2 letters, top-1 | top-3 | After 3 letters, top-1 | top-3 | Conversational top-1 | Conversational top-3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 80,000 | 54.0 | 89.94 | 16.65 | 27.52 | 958 | 47.63 | 59.32 | 45.20 | 58.02 | 12.37 | 21.69 |
| 120,000 | 58.1 | 90.75 | 16.74 | 27.65 | 1014 | 47.89 | 59.62 | 45.54 | 58.40 | 12.67 | 22.09 |
| 150,000 | 60.4 | 91.09 | 16.77 | 27.70 | 1040 | 48.02 | 59.77 | 45.73 | 58.59 | 12.78 | 22.25 |
| 200,000 | 63.3 | 91.45 | 16.79 | 27.74 | 1073 | 48.08 | 59.89 | 45.85 | 58.73 | 12.86 | 22.35 |

Recall after 2 and 3 letters rises by 0.30 and 0.38 points from 80,000 to
120,000 types, by 0.15 and 0.19 to 150,000, and by 0.12 and 0.14 to
200,000, and each step adds 2 to 4 MB. The last type of the vocabulary is
seen 50, 22, 13 and 6 times (weighted), and the vocabulary covers 98.63%,
99.34%, 99.61% and 99.84% of the occurrences of word-list spellings. The
default is 150,000 types. At none of these sizes does a contested spelling
outside the vocabulary occur more than 255 times, so the trainer adds no
types for the out-of-vocabulary table.

A reader that completes a stem by scanning the whole vocabulary does work
in proportion to its size; see [Known limitations](#known-limitations).

## Recorded model

Built at commit `6cb8d11` on 2026-11-08 from the dictionary with sha256
`ef508ecffe7cf3a2b9696e8de3edea791fd3b81f28b7977146479a427948464f`.
Recounting and exporting against the follow-up word list with sha256
`a9d955ed10d7301f228065faf028977a18d2c65d755fee7fd476b5dcfd81fe93`
produces the same binary byte for byte: spelling preferences and C-to-M name
ratings do not change the model's vocabulary or counts. The current version
file records the new dictionary digest. The binary's sha256 is
`46564a1f26beaa83f90ef569563b3dfeab479df2053aac85553a8e7afcc0fe0e`.

| Part | Measure |
|---|---|
| Size | 61,308,873 bytes |
| Vocabulary | 155,791 types, including 5,791 double-accented forms |
| Bigram rows | 153,141 |
| Trigram rows | 1,269,046 |
| Homograph spellings | 6,270 |
| Out-of-vocabulary counts | 12,945, 0 saturated |
| Weighted training words | 200,543,373 |
| Training vocabulary share | 94.10% |
| Training unknown share | 5.52% |
| Distinct bigrams counted | 13,890,568 |
| Distinct trigrams counted | 58,653,612 |

Corpus shards and side tables match the previous build byte for byte; the
held-out split is unchanged. The word list changed, so the counts and model
were rebuilt in order.
`export_lm.py --lang el --check-dictionary` passes, and the Python reader
reports no structural problem. The native Swift reader resolved every vocabulary
entry and reserved id, checked all dictionary words outside the vocabulary, and
matched Python on all 18,000 method answers from 6,000 dev queries, including
capitalized contexts and ASCII apostrophes. These checks cover the GNLM v2
format without requiring a keyboard or a simulator.

## Evaluation

`eval/eval_el_ngram.py` replays the whole dev split, held out by document group.
Next-word top-1 and top-3 include unknown and out-of-vocabulary targets as misses.
Completion takes the first two or three letters without accents and is measured
over listed word targets longer than that prefix. A double-accented target counts
as its host, and suggestions hide double-accented forms.

The score uses stupid backoff, so pseudo-perplexity compares files of this format;
it is not a normalized language-model probability. Pooled results weight sources
by their targets. Conversational pools Tatoeba, Common Voice and Wiktionary examples.

| Set | Word targets | In vocabulary, % | Top-1, % | Top-3, % | Completion after 2 letters, top-3, % | After 3 letters, top-3, % | Pseudo-perplexity |
|---|---|---|---|---|---|---|---|
| Common Voice | 2,086 | 95.73 | 9.44 | 17.50 | 50.12 | 49.32 | 4156 |
| EELLAK articles | 56,120 | 90.08 | 15.57 | 26.75 | 58.65 | 56.29 | 1119 |
| European Parliament | 1,904,781 | 94.80 | 22.35 | 34.74 | 64.17 | 61.33 | 505 |
| OpenGov comments | 1,856,015 | 91.90 | 17.56 | 28.96 | 62.48 | 60.89 | 762 |
| Parliament | 3,773,347 | 96.63 | 19.77 | 32.40 | 67.67 | 67.90 | 489 |
| Tatoeba | 12,295 | 98.19 | 15.68 | 27.34 | 57.01 | 61.14 | 1437 |
| Wikipedia | 1,924,319 | 88.79 | 14.71 | 25.57 | 55.82 | 54.80 | 1940 |
| Wikisource | 107,987 | 73.31 | 4.41 | 10.05 | 41.86 | 38.89 | 9494 |
| Wiktionary examples | 42,202 | 88.51 | 11.99 | 20.93 | 49.72 | 49.08 | 3601 |
| All, pooled | 9,679,152 | 93.47 | 18.61 | 30.50 | 63.37 | 62.38 | 723 |
| Conversational, pooled | 56,583 | 90.88 | 12.70 | 22.20 | 51.36 | 51.73 | 2919 |

The three-letter completion set has longer and harder words than the two-letter
set, so its recall can be lower. The pruning experiments above use smaller dev
samples and a different historical vocabulary; their pooled scores are not
the current scores in this table.

[Monotonic keyboard data](monotonic-keyboard-data.md) measures accent restoration,
typing slips, enclitics and a three-slot suggestion bar using these same files.

### Predictions after common contexts

The first six suggestions:

| Context | Suggestions |
|---|---|
| θα | πρέπει, είναι, ήθελα, έπρεπε, σας, το |
| δεν | είναι, θα, μπορεί, έχει, υπάρχει, έχουν |
| να | μην, είναι, το, γίνει, σας, έχει |
| είναι | η, το, ο, ένα, ότι, και |
| καλή | διάθεση, επιτυχία, η, και, χρονιά, γνώση |
| τι κάνεις | εδώ, εκεί, στον, απόψε, και, στη |
| ευχαριστώ πολύ | κύριε, κυρία, για, και, τον, που |
| θα at the start of a sentence | πρέπει, ήθελα, σας, το, μου, ήταν |
| σας | πω, ευχαριστώ, παρακαλώ, και, το, είπα |
| καλημέρα at the start of a sentence | σας, σε, και, κύριε, τι, θα |

## Tests

`tests/test_el_ngram.py` builds a two-source fixture corpus and word list,
trains, exports and reads the binary back. It checks the resolution table
above, that streaming counts equal `Counter` counts, the token caps and
weights, that the vocabulary holds only list spellings, their
double-accented forms and the reserved tokens at the ids the header names,
the structure of every row (sorted contexts, contiguous offsets, rows in
descending probability, no reserved or out-of-range ids, no `<UNK>` rows),
the out-of-vocabulary footer and its counts, the sidecar fields,
byte-identical output from pre-pruned and full intermediates (also on
randomized counts with host and double-accented pairs) and across reruns,
the export refusals, and the reader's refusals of malformed files. For
the double-accented forms it checks that they resolve to tokens of their
own, join the vocabulary outside its budget, are no homographs of their
hosts, keep both halves of a pair and their enclitics in the rows, and are
hidden from the reader's suggestions.

## Known limitations

- Parliament, OpenGov and Wikipedia dominate training. Conversational results are
  lower, and suggestions can favor the formal register (`κύριε` after `ευχαριστώ πολύ`).
- Acronyms and abbreviations remain in companion lists. The model does not propose
  them; a keyboard needs a separate path for those suggestions.
- The reader lowercases context but does not restore its accents. A keyboard must
  restore an unaccented capitalized context before lookup when needed.
- Capitalized institution names have entries of their own and can appear where a
  lowercase word is intended.
- A double-accented twin seen fewer than three times has no model token. Enclitic
  decisions for it fall back to the documented syllable and context rules.
- Trigram suggestions precede bigram suggestions. A weak trigram can outrank a
  stronger bigram; the pruning threshold removes only the weakest counts.
- Prefix completion scans the vocabulary, so cost grows with its size. This
  validation establishes reader agreement, not worst-case latency on every device.
- Older literary and dialect text in Wikisource remains less covered than the
  contemporary sources. Pseudo-perplexity is not comparable to normalized models.
