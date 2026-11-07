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
spelling it doubles. The list has 265,200 such forms. One seen at least 3
times whose host is in the vocabulary joins the vocabulary, outside its
150,000 types, with its id after the list spellings; a host ranks by its
count plus its twin's. A rarer one is counted as its host. These forms
are no words of the list: they are never homograph spellings, never go
into the out-of-vocabulary table, and the keyboard must never offer them
as predictions or completions (the evaluations hide every vocabulary
token with two tonos, `GNLMReader.hide_double_accented`). Any other
double-accented token keeps only its first tonos (`fold_enclitic_accent`).

In the full training run, of the 195,633,147 training tokens under the
caps (before weighting), 169,505,613 were list spellings or their
double-accented forms as written, 8,948,242 were lowercased at the start
of a sentence and 6,163,654 inside one, and 11,015,638 became `<UNK>`
(5,425,695 placeholders, 1,244,243 abbreviations, 1,670,885 words in
capitals, 2,674,815 not in the list). 474,750 tokens were double-accented
forms of list spellings, and 39,774 other double-accented tokens lost
their second accent. The vocabulary holds 5,784 double-accented forms.

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
continuation seen at least 3 times. In this vocabulary 6,058
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

Built at commit cc4cdaa on 2026-11-07, from the corpus extracted at
commit 3a45179 and the word list with sha256 `b21c4047` (631,330 entries);
the binary's sha256 is `8f62be97`:

| Part | Measure |
|---|---|
| Size | 61,065,151 bytes (58.2 MiB), 20.8 MB under `xz -6` |
| Vocabulary | 155,784 types: 150,000 plus 5,784 double-accented forms (4.2 MB with counts and the 10 global fallbacks) |
| Bigram rows | 153,020 rows (1.8 MB), 2,218,343 entries (13.3 MB) |
| Trigram rows | 1,263,685 rows (20.2 MB), 3,567,076 entries (21.4 MB) |
| Homograph spellings | 6,058 kept past the cut; 1,156 bigram rows completed with 263,637 entries |
| Host and double-accented form pairs | 5,784 |
| Out-of-vocabulary table | 12,831 counts (0.1 MB), none capped |
| Training words | 200.5M weighted (195.6M tokens under the caps); 94.02% vocabulary spellings, 5.61% `<UNK>`, 0.37% list spellings outside the vocabulary |
| Distinct n-grams counted | 13.8M bigrams, 58.5M trigrams |

The pruning and vocabulary tables above were measured with the word list
of commit 2951b0e (630,688 entries) and without the double-accented forms.
The list of this build differs from it in about 3,000 entries, most of
them names and rare forms (see
[Monotonic keyboard data](monotonic-keyboard-data.md#changes-to-the-word-list)),
and the corpus has since dropped Wiktionary's editorial notes and keeps
`ό,τι` as one token; the evaluation below is of this build. The previous
build (sha256 `227240c4`, 60,428,378 bytes, 150,000 types) scored 18.52%
top-1 and 30.36% top-3 on its dev sentences, which split `ό,τι`.

Training took 31 s with the encoded shards cached (56 s when the new list
made it encode them again) and export 16 s, on an Apple M5 Max. `build/gnlm_reader.py`
finds no structural problem in the file, and `export_lm.py --lang el
--check-dictionary` confirms the out-of-vocabulary table matches the list.
The format is unchanged: the double-accented forms are ordinary vocabulary
tokens. The keyboard's Swift `LMReader` was last checked on the previous
build (sha256 `227240c4`): it resolved each of its 150,000 vocabulary
entries to its own id and the four reserved tokens to the ids the header
names, found a count for each list word outside the vocabulary that the
table covers, and gave the same answers as `build/gnlm_reader.py` on
6,000 queries drawn from the dev sentences (`suggest`, `score` and
`suggestCompletions`, with context words in capitals and with ASCII
apostrophes).

## Evaluation

`eval/eval_el_ngram.py` reads the binary with `build/gnlm_reader.py`, a
Python reader that validates a file the way `LMReader` does and answers
`suggest`, `score` and `suggestCompletions` step for step as it does, and
replays the dev sentences of `build/lm_el` (held out by document group,
never counted):

- **Next word.** After every word of a sentence, and after `<s>`, the first
  1 and 3 suggestions with the two previous words as context, against the
  word that follows. `</s>` is never a target. *All words* counts every
  word target, so `<UNK>` and spellings outside the vocabulary are misses;
  *in vocabulary* counts only targets the model can propose.
- **Completion recall.** For every word target longer than k letters
  (k = 2, 3), the first k letters typed in lowercase without accents, then
  the first 1 and 3 completions, against the word. Over word-list targets.
- **Pseudo-perplexity.** exp of the mean negative `LMReader.score` over
  vocabulary targets. The score is stupid backoff over the rows the file
  holds, so a word cut from a row is scored at a lower tier, as on the
  keyboard. The scores are not normalized probabilities, so the value only
  compares models of this format with each other.

The pooled results weight sources by their dev word targets, so
Parliament dominates them; *conversational* pools Tatoeba, Common Voice and the
Wiktionary examples.

Whole dev split, 449,360 sentences (`python eval/eval_el_ngram.py`, about
7 minutes on an idle machine). Double-accented forms are hidden from the
suggestions as on the keyboard, and a double-accented target counts as its
host:

| Set | Word targets | In vocabulary | Top-1 | Top-3 | Top-1, vocabulary targets | Top-3, vocabulary targets | Pseudo-perplexity | Completion after 2 letters, top-1 | top-3 | After 3 letters, top-1 | top-3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Common Voice | 2,086 | 95.69 | 9.44 | 17.50 | 9.87 | 18.29 | 4164 | 36.96 | 50.24 | 36.51 | 49.48 |
| EELLAK articles | 56,120 | 90.05 | 15.56 | 26.74 | 17.28 | 29.69 | 1116 | 47.64 | 58.71 | 42.92 | 56.33 |
| European Parliament | 1,904,781 | 94.62 | 22.28 | 34.65 | 23.55 | 36.62 | 503 | 53.37 | 64.19 | 49.62 | 61.34 |
| OpenGov comments | 1,856,015 | 91.72 | 17.50 | 28.88 | 19.08 | 31.48 | 760 | 50.45 | 62.55 | 48.40 | 60.98 |
| Parliament | 3,773,347 | 96.55 | 19.74 | 32.37 | 20.44 | 33.52 | 488 | 55.18 | 67.72 | 54.47 | 67.95 |
| Tatoeba | 12,295 | 98.22 | 15.68 | 27.35 | 15.97 | 27.85 | 1439 | 44.39 | 57.05 | 44.84 | 61.17 |
| Wikipedia | 1,924,319 | 88.79 | 14.71 | 25.56 | 16.57 | 28.79 | 1940 | 43.58 | 55.84 | 41.95 | 54.84 |
| Wikisource | 107,987 | 73.17 | 4.40 | 10.04 | 6.02 | 13.72 | 9397 | 28.49 | 42.01 | 24.19 | 39.09 |
| Wiktionary examples | 42,202 | 88.51 | 11.99 | 20.93 | 13.55 | 23.65 | 3601 | 38.36 | 49.77 | 36.52 | 49.15 |
| All, pooled | 9,679,152 | 93.37 | 18.57 | 30.45 | 19.89 | 32.62 | 722 | 51.36 | 63.41 | 49.57 | 62.43 |
| Conversational, pooled | 56,583 | 90.89 | 12.70 | 22.20 | 13.97 | 24.43 | 2920 | 39.65 | 51.41 | 38.34 | 51.79 |

Completion after 3 letters can score below completion after 2 letters
because it is measured over words of more than 3 letters, a harder set
than words of more than 2. The tuning tables above use 3,000 dev sentences
per source, which gives the small sources more weight, so their pooled
numbers are lower than these.

[Monotonic keyboard data](monotonic-keyboard-data.md) measures the model
together with the word list the way a keyboard uses them: choosing between
spellings that differ only in accents, and the keystrokes a three-slot
suggestion bar saves.

### Predictions after common contexts

The first six suggestions (`suggest`):

| Context | Suggestions |
|---|---|
| `θα` | πρέπει, είναι, ήθελα, έπρεπε, σας, το |
| `δεν` | είναι, θα, μπορεί, έχει, υπάρχει, έχουν |
| `να` | μην, είναι, το, γίνει, σας, έχει |
| `είναι` | η, το, ο, ένα, ότι, και |
| `καλή` | διάθεση, επιτυχία, η, και, χρονιά, γνώση |
| `τι κάνεις` | εδώ, εκεί, στον, απόψε, και, στη |
| `ευχαριστώ πολύ` | κύριε, κυρία, για, και, τον, που |
| `θα` at the start of a sentence | πρέπει, ήθελα, σας, το, μου, ήταν |
| `σας` | πω, ευχαριστώ, παρακαλώ, και, το, είπα |
| `καλημέρα` at the start of a sentence | σας, σε, και, κύριε, τι, θα |

The first three completions (`suggestCompletions`) of a stem typed without accents:

| Context | Stem | Completions |
|---|---|---|
| `θα ήθελα` | `να` | να, ναι, Ναυτιλίας |
| `τι κάνεις` | `σ` | στον, στη, συνήθως |
| `ευχαριστώ` at the start of a sentence | `π` | πολύ, πάρα, που |
| start of a sentence | `καλ` | καλό, καλά, καλύτερα |
| `να σας` | `ευχ` | ευχαριστώ, ευχαριστούμε, εύχομαι |
| `δεν` | `ξ` | ξέρω, ξέρετε, ξεχνά |

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

- **Register.** Parliament, OpenGov and Wikipedia make up most of the
  training text. The conversational sets score far lower (top-1 12.70%
  against 18.57% pooled), and some predictions show the Parliament
  register: after `ευχαριστώ πολύ` the first suggestions are `κύριε` and
  `κυρία`. The open sources hold little casual conversation (see the
  corpus's limitations).
- **Capitals, acronyms and abbreviations.** Words in capitals are never
  counted, so the model never proposes `ΗΠΑ` or `ΕΣΥ`, and abbreviations
  with their period (`π.χ.`) are `<UNK>`. The word list's companion lists
  hold them.
- **Context typed in capitals.** The reader lowercases a context word but
  has no accent-insensitive lookup. `ΘΑ` finds `θα`, but a word typed in
  capitals without its tonos (`ΕΙΝΑΙ`) resolves to nothing unless the
  keyboard restores the accent first.
- **Capitalized names of institutions.** Spellings that two sources
  write with their capital inside sentences are list entries of their own
  (`Ένωση`, `Ναυτιλίας` of `Υπουργείο Ναυτιλίας`), so the model counts
  them apart from the lowercase word and can propose them in their
  capitalized form: after `θα ήθελα` the third completion of `να` is
  `Ναυτιλίας`.
- **Rare double-accented forms.** A double-accented form seen fewer than
  3 times in training is counted as its host, so for it the keyboard has
  no evidence and falls back to the syllable rule.
- **Trigram entries first.** The reader lists a trigram row before the
  bigram row, so a weakly attested trigram entry outranks a strong bigram
  entry. The minimum trigram count of 4 removes the weakest; ranking the
  two tiers together by backoff score may do better.
- **Completion cost grows with the vocabulary.** For a Greek stem the
  reader scans the whole vocabulary, which is sorted by its accented
  bytes. The first version of the keyboard's Swift reader decoded every
  entry and stripped its diacritics: on an Apple M5 Max a completion took a
  median 150, 230, 288 and 385 ms at 80,000, 120,000, 150,000 and 200,000
  types, over the 50 ms its comments allow. Its current version folds each
  entry's letters in place and rejects most entries at the first letter; on
  the same machine a completion takes 0.7 ms on average at 150,000 types
  (4,000 stems of two and three letters from the Parliament dev
  sentences). The scan is still linear in the vocabulary size, but the
  format needs no index.
- **Pooled numbers.** Parliament has 39% of the dev word targets, so it
  dominates the pooled results; the per-source rows are the better guide.
- **Wikisource.** Only 73.17% of its dev words are vocabulary spellings:
  literary and dialect text, much of it older, that the list and the
  model cover poorly.
- **Pseudo-perplexity.** Stupid backoff scores are not normalized, so the
  value compares models of this format with each other and nothing else.
