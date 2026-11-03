# Next-word language model

[README](../README.md)

- [Build and export](#build-and-export)
- [Corpus readers and source validation](#corpus-readers-and-source-validation)
- [Generated files](#generated-files)
- [Binary format](#binary-format)
- [Export settings](#export-settings)
- [Recorded size and evaluation](#recorded-size-and-evaluation)
- [Interpreting prediction scores](#interpreting-prediction-scores)
- [Reader compatibility](#reader-compatibility)

`train_lm.py` and `export_lm.py` build a compact trigram model for
next-word prediction. It is used by the Tonos keyboard and exported as
`build/lm/grc_ngram.bin`. The default corpus registry includes GLAUx, Diorisis,
polytonic Modern Greek from Wikisource, and Byzantine vernacular texts.
Sources can be disabled individually for comparison runs.

The model uses stupid backoff with a configurable context table size. Its
binary format supports memory-mapped lookup without a neural runtime. The
measured results below describe particular builds; size and latency depend
on the corpus, export settings, and reader implementation.

## Build and export

```bash
python train_lm.py --sanity         # few files per corpus, <1 min
python train_lm.py                  # all enabled sources
python train_lm.py --no-diorisis --no-polytonic-mg --no-byzantine  # GLAUx-only baseline
python export_lm.py                 # writes grc_ngram.bin + .version, ~30 s
python eval_lm.py                   # writes eval_results.txt, ~90 s
```

`export_lm.py` reads the grc Hunspell dictionary
(`build/hunspell/grc_polytonic.dic`, from `export_hunspell.py --variant
grc`) to choose which out-of-vocabulary counts it carries (see below),
so export the dictionary first, or pass `--oov-min-count 0` to leave
that table out. **Re-export the LM after every rebuild of the grc
dictionary**: the table follows the dictionary it was chosen against,
whose sha256 `grc_ngram.version` records. `export_hunspell.py` prints a
note when the dictionary it wrote differs from that one, and

```bash
python export_lm.py --check-dictionary   # exit 1 when the LM needs re-exporting
```

checks the pair without exporting anything.

## Corpus readers and source validation

Corpus loaders live in ``train_lm.py`` (GLAUx, inline) and
``extract_diorisis_lm.py`` (Diorisis, beta-code to NFC). To add
another Ancient Greek corpus, write a loader that yields
``(sentence_id, [<s>, ...NFC tokens..., </s>])`` and append one
entry to ``build_corpus_sources`` in ``train_lm.py``; the counting,
vocab, split, and eval stages do not need any changes.

The Diorisis pairs, frequency, language-model, and form-attestation readers
share `build/diorisis_source.py` for quotation context. A terminal Beta Code
apostrophe is removed as a duplicated closing quote only when the following
punctuation repeats it and the same document independently contains the
resulting form under every annotated lemma, part of speech, and morphology
reading. Genuine elision, including `)` notation, remains intact; unresolved
quote/elision readings are preserved. The pairs builder writes the original
and normalized forms, witness locations, source hashes, and generator
revision to `data/changes/diorisis_token_normalizations.json` by default;
`--token-audit PATH` chooses another audit destination. Each recorded change
can be reversed with `restore_source_beta`.
The same-cell rule also resolves a final apostrophe before plain sentence
punctuation when the next sentence begins with a consonant. A vowel-initial
continuation preserves possible elision. This distinguishes complete
`ἐπάιον` followed by punctuation from the genuine elision `ὤιετ᾽ εἶναι`.

When normalization brings source cells together, the pair extractor retains
their distinct part-of-speech and morphology readings. POS disambiguation
counts each form/lemma/POS edge once per corpus, so additional morphology
readings do not multiply that source's vote.

Preserving an unresolved corpus reading does not establish a new dictionary
spelling. The Ancient export uses that audit to require an exact independent
lexical head, a cited source-form review, or a closed grammatical exception
before admitting an elision implicated in quotation punctuation. Corpus
counts and generated lookup rows do not provide that exception. Spellings
in the curated compatibility fixture remain eligible; explicit spelling rejections
still take precedence. The export audit records the withheld forms and the
hashes of its quotation and lexical evidence. Cited exact forms preserve
genuine elision and printed prodelision spellings even when the same spelling
also occurs with quotation punctuation in a corpus.

The GLAUx loader reads through the license filter in
`build/nc_filter.py`, the same one the pair, frequency, attestation
and tagger builders use: the 7 NonCommercial and 25 PROIEL-derived
works are dropped whole, and the 40 Gorman-derived works contribute
only their automatic sentences, since their manual sentences are the
project's held-out gold. That removes 841,551 of the 17.0M tokens
GLAUx contributes, so the counts above, measured before the filter
reached this loader, fall by that much on the next rebuild. The
filter reads GLAUx's `metadata.txt`, which the loader expects next
to the `--glaux` directory; `--glaux-metadata` points it elsewhere,
and a run stops rather than proceed without it.

The GLAUx pair builder also repairs malformed name lemmas when their source
group contains one exact nominative singular independently defined as a name
in the merged lexical evidence. It preserves ambiguous groups and competing
readings. Run `python build/build_glaux_pairs.py` first; it writes corrected
pairs and `data/changes/glaux_name_lemmas.json` with original rows, corrected
rows, source fingerprints and the generator commit. The lemma-attestation
and tagger-data builders validate and apply that same audit through
`--name-corrections PATH` (the audit above by default). Use
`--lexical-heads PATH` and `--name-pairs PATH` for alternate pinned inputs.
The audit can reconstruct the original pairs with `unapply_name_corrections`.
Pair extraction and audit loading require the source `metadata.txt`; missing
or changed exclusion metadata stops the build, including statistics-only runs.

## Generated files

Output layout:

```
build/lm/
  grc_ngram.bin                 memory-mapped binary (size varies by export)
  grc_ngram.version             semver + dilemma commit + vocab/context counts
  eval_results.txt              from eval_lm.py (combined dev split)
  eval_results_glaux.txt        eval restricted to GLAUx dev sentences
  eval_results_diorisis.txt     eval restricted to Diorisis dev sentences
  vocab.json                    intermediate from train_lm.py
  unigrams.json                 intermediate
  bigrams.tsv.gz                intermediate
  trigrams.tsv.gz               intermediate
  type_counts.tsv.gz            intermediate: training count of every type, in the vocabulary or not
  dev_sentences.txt             held-out dev set, deterministic split (seed 4242)
  dev_sentences_glaux.txt       same split, restricted to GLAUx sentences
  dev_sentences_diorisis.txt    same split, restricted to Diorisis sentences
  stats.json                    training corpus statistics
```

## Binary format

Artifact layout at a glance (full spec in the docstring of
`export_lm.py`): a 128-byte little-endian header, a sorted UTF-8
vocab (binary-searchable by the Swift reader), a per-vocab unigram
count column, and three flat sorted tables: unigram top-K,
bigram top-K-per-w1, and trigram top-K-per-w1,w2. Each suggestion
entry is 6 bytes: a `u32` word id and an `i16` fixed-point log
probability (scale 1024). Lookup is a binary search on the trigram
context (w1, w2); on miss, fall back to bigram (w1); on miss, fall
back to the global unigram top-K. All tables are mmap'd and accessed
by offset; nothing needs to be read into RAM.

Typeahead v2 (format_version = 2) adds two things versus v1:

- Independent top-K per table. Bigram contexts now store the top 30
  continuations and trigram contexts the top 15, so the keyboard's
  mid-word prefix filter has enough candidates to work with even
  after a user has typed a character or two. The global unigram
  fallback stays at 10. The v2 reader refuses any file with
  `format_version` below 2, so old v1 binaries must be rebuilt.
- A per-vocab unigram count column, one `u32` per vocab entry,
  indexed by sorted-vocab id. The Swift reader uses this to rank
  global prefix completions by corpus frequency when the current
  bigram/trigram top-K doesn't cover the user's stem, so the bar is
  never starved of useful entries even for rare context / common
  stem combinations.

Three later additions keep `format_version` at 2 and leave the header and
every section above byte-for-byte what a v2 reader expects:

- **Homograph spellings kept past the cut.** A context row also keeps,
  after its top 30 (bigram) or 15 (trigram), every continuation seen at
  least 3 times there whose bare letters another vocabulary spelling
  shares with more than a grave, case or elision-glyph difference: ἢ
  and ἡ, ἐκείνῃ and ἐκείνη, αὐτοῦ and αὑτοῦ. A keyboard choosing
  between two spellings of the same letters then scores both from the
  same row instead of one by backoff. The first 30 / 15 entries of every
  row are unchanged, so next-word lists are too (20,000 of 20,000
  held-out contexts gave the same six words). Mid-word completions
  change, because a kept spelling counts as a context entry and so
  ranks above completions found by scanning the vocabulary: on 4,000
  held-out one- and two-letter prefixes, 1,207 top-3 lists changed and
  the word that followed was in them 1,485 times against 1,473.
  `--keep-homograph-min-count 0` turns this off.
- **Rows completed after a homograph spelling.** To choose between `ἢ` and
  `ἡ`, a keyboard scores the word typed next after each of them, and the
  cut left almost every next word out of both bigram rows, so both
  readings fell back to that word's unigram probability and the choice was
  made without evidence. The bigram row of every homograph spelling now
  also keeps, past its cut, every continuation seen at least 3 times after
  it: 266,694 continuations in 1,491 rows, 1.6 MB. Trigram rows are cut as
  before, and the first 30 entries of every row, with them the next-word
  lists, are unchanged. `--homograph-context-min-count 0` turns this off;
  with both minimums at 0 the exporter writes the plain top-K rows.
- **Out-of-vocabulary unigram table.** When no spelling of the typed
  letters is among the 80,000 vocabulary types, every spelling scored
  nothing and the dictionary's order decided. `train_lm.py` now writes
  the training count of every type, and the exporter appends the counts
  of the out-of-vocabulary types that are a dictionary spelling with a
  same-letter twin: 125,357 entries at 9 bytes each (a 64-bit FNV-1a
  hash of the spelling and a count byte), found through a 16-byte footer
  that ends the file. A type a reader resolves to a vocabulary id
  through its lowercase or its U+2019 elision spelling is left out, and
  a word's elision-glyph variants are summed under U+2019, the spelling
  a reader asks for. A reader that knows the footer scores such a
  spelling at the unigram tier, `α² · c / N`, as it scores a vocabulary
  word no row holds; one that does not ignores the trailing bytes.
  `--oov-min-count 0` leaves the table out.

Measured by replaying the 26,906 held-out Ancient Greek sentences
(474,220 words) of the dev split through a keyboard that types each
word as bare letters and restores its marks, against the shipped 1.3.6
model: the homograph rows correct 1,972 words and break 549 (net
+1,423); the table on top of them corrects 2,741 and breaks 424 (net
+2,317). GLAUx and Diorisis annotate largely the same texts, so most
dev sentences have a twin in the training split; on the 7,449
sentences with under 20% of their 5-grams in training (137,525 words)
the two gains are +3.2 and +4.5 corrected words per 1,000. The completed
rows, measured the same way against the model with the other two, correct
259 words and break 23 (net +236), and 104 and 2 on the low-overlap
sentences (+0.74 per 1,000); typed with their punctuation, 265 and 20 (116
and 1). Every corpus of running text gains, and so does polytonic Modern
Greek with each author held out.

## Export settings

Defaults from `export_lm.py`:

| Knob | Value | Effect |
|------|------:|--------|
| vocab size | 80,000 | Reduces UNK target rate to 7.6% on the dev split |
| top-K uni  | 10 | Global fallback depth |
| top-K bi   | 30 | Per-bigram context depth (mid-word prefix filter wants this deep) |
| top-K tri  | 15 | Per-trigram context depth |
| min bigram count | 1 | Keep every observed bigram |
| min trigram count | 1 | Keep every observed trigram... |
| min bigram count for trigram | 3 | ...but only when the (w1, w2) bigram context is well attested. Rare contexts fall back to the bigram table at inference. This is the dominant size/quality knob. |
| keep homograph spellings | 3 | A homograph spelling seen this often in a context stays in its row past the top-K cut (0: off) |
| complete homograph rows | 3 | The bigram row of a homograph spelling keeps every continuation seen this often after it (0: off) |
| out-of-vocabulary count table | 1 | Minimum training count of a contested dictionary spelling the table carries (0: no table) |

## Recorded size and evaluation

The recorded v2 build was 61,892,317 bytes (59.0 MiB; 20.2 MB under xz -6), 80K
vocab, ~1.1M trigram contexts, ~80K bigram contexts. The plain top-K
rows are 56.9 MB of that, the kept homograph spellings 2.3 MB, the
completed homograph rows 1.6 MB and the out-of-vocabulary table 1.1 MB.

Held-out evaluation, keyboard-realistic regime (exclude `</s>` and
UNK targets):

| Dev split | training data | sentences | preds scored | top-1 | top-3 | top-5 | top-6 |
|-----------|--------------|----------:|-------------:|------:|------:|------:|------:|
| GLAUx only        | GLAUx           | 19,578 | 315,385 | 13.56% | 23.08% | 27.75% | 29.59% |
| GLAUx only        | GLAUx+Diorisis  | 19,578 | 315,278 | 14.86% | 28.14% | 34.52% | 36.56% |
| Diorisis only     | GLAUx+Diorisis  | 10,672 | 188,199 | 15.17% | 30.46% | 37.63% | 39.88% |
| Combined          | GLAUx+Diorisis  | 30,250 | 503,477 | 14.97% | 29.00% | 35.68% | 37.77% |
| Combined          | v2 (bi 30/tri 15) | 32,724 | 534,541 | 14.69% | 28.30% | 34.83% | 37.12% |

The v2 row reports on the newer combined corpus (GLAUx + Diorisis +
Katharevousa Wikisource + Byzantine vernacular). Top-1 / top-3 /
top-5 are unchanged from the v1 row's combined baseline (as
expected: widening top-K per context does not move top-N for N
smaller than K); the new number to look at is top-6 at 37.1%.

Adding Diorisis lifts top-3 accuracy on the identical held-out
GLAUx sentences by +5.1 points and top-5 by +6.8 points; top-1 moves
by +1.3 points. Dev sentences with identical sentence-id hashes
across runs make this an apples-to-apples comparison.

Backoff level breakdown: ~74% of predictions hit the trigram table,
~26% fall back to bigrams, <0.01% fall to unigram. Perplexity (stupid
backoff, α=0.4) is reported in `eval_results.txt` but is **not** a
clean perplexity: because the binary only stores the top-K
continuations per context, off-list targets are assigned a floor
probability, which drives PPL up. The top-K accuracies are the real
quality metric for this artifact; PPL is a sanity signal, not a
headline number.

## Interpreting prediction scores

The top-1 and top-K figures above are measurements on these development
splits. Greek inflection and variable word order make the prediction task
difficult, but they do not set a general accuracy ceiling for another model
or corpus. Compare changes on the same held-out sentences and score the
number of proposals that the consumer actually displays.

- **UNK rate is 7.9%** on the combined dev split even at 80K vocab.
  Polytonic inflection multiplies forms (a typical verb has 300+
  forms), so long-tail coverage is inherently hard. Tonos can let
  the user type any form; the LM just won't propose it.
- **Homographs are preserved.** ἦ (past indicative) and ἤ
  (disjunction) remain distinct. No accent stripping is done at any
  stage.
- **Diorisis beta code is converted to NFC polytonic** via the
  `betacode` PyPI package. A small cleanup handles elision-after-
  consonant encoded as `)` rather than `'` (e.g. `par)` for παρ’).
  Elision-after-vowel (`ou)` = οὐ with smooth breathing) stays
  untouched.
- **Corpus frequency keys preserve elision.** GLAUx, Diorisis, PTA, and the
  OGC public-lexicon adapter use `build/corpus_freq_key.py` to strip accents
  while canonicalizing textual apostrophes and trailing combining psili to
  U+2019. `merge_corpus_freq.py` applies the same normalization again, so
  source-specific spellings cannot split counts across parallel keys. After a
  tokenizer change, regenerate that source's frequency artifact before
  merging; a lost apostrophe cannot be reconstructed from an old bare key.

## Reader compatibility

Reader contract for Tonos: the binary format is versioned in the
header (`format_version = 2`) and in `grc_ngram.version`. Any on-disk
layout change will bump `format_version`. The sidecar version file
also carries the dilemma commit hash, vocab size, and trigram /
bigram context counts so a Tonos build can detect staleness and
refuse mismatched artifacts. The Swift reader rejects any file whose
`format_version` doesn't match the version it was compiled against;
the v2 reader does not read v1 files (and vice versa).
