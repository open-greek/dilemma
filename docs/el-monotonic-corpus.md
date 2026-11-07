# Monotonic Modern Greek corpus

[README](../README.md)

- [Build](#build)
- [Sources](#sources)
- [Normalization and tokenization](#normalization-and-tokenization)
- [Filters](#filters)
- [Split, repeats and caps](#split-repeats-and-caps)
- [Output](#output)
- [Recorded corpus](#recorded-corpus)
- [Known limitations](#known-limitations)

`extract_el_monotonic_lm.py` builds the corpus of monotonic Modern Greek used
for a monotonic next-word model and for the corpus routes of the monotonic
word list. It reads the pinned, openly licensed sources of
`data/el_lm_corpora.json`, normalizes and tokenizes every document with
`build/el_text.py` exactly as a keyboard segments the words a person types,
and writes a document-level train/dev split under `build/el_corpus/`. The
shards are an intermediate: what is distributed later is counts computed from
them, not the sentences.

## Build

```bash
python extract_el_monotonic_lm.py --download    # fetch missing pinned inputs
python extract_el_monotonic_lm.py --sanity      # 2,000 documents per source, about 20 s,
                                                # into build/el_corpus_sanity
python extract_el_monotonic_lm.py               # full corpus into build/el_corpus
python -m pytest tests/test_el_text.py tests/test_el_corpus.py -q
```

Hugging Face inputs come from the Hugging Face cache at the pinned revision.
URL inputs are kept in `data/raw/<source>/` (ignored by git), or in
`$DILEMMA_EL_RAW_DIR`. The Wiktionary examples are read from the kaikki.org
dumps that `build_data.py` uses (`kaikki/`, or `$DILEMMA_KAIKKI_DIR`). Every
input's size and sha256 must match the manifest, and a run stops on any
difference. A run limited by `--sanity`, `--max-docs` or `--sources` cannot
write `build/el_corpus`. The full run takes about 5 minutes with 16 worker
processes.

## Sources

Every source in `data/el_lm_corpora.json` has a license, the evidence for
that license, an attribution, a role, a token cap and a weight. The extractor
reads only licenses on the allowed list in `build/el_corpus.py` (CC0, CC BY,
CC BY-SA) and refuses any license with NC or ND in it. Corpora that were
considered and left out are listed under `excluded` with the reason. They
include the NonCommercial or NoDerivatives theatre, press and oral-history
sets, OpenSubtitles (no license for the subtitle text), web crawls (the
license covers the compilation, not the text), and administrative and
legislative text whose size would dominate the mixture. The Triantafyllidis
headwords are for evaluation only and are never read as text.

| Source | License | Register | Token cap | Weight |
|---|---|---|---|---|
| Greek Parliament proceedings 1989-2020 | CC BY 4.0 | transcribed debate | 100M | 1 |
| OpenGov citizen comments | CC BY 4.0 | written comments on draft laws | none | 1 |
| Greek Wikipedia | CC BY-SA 4.0 | encyclopedic | 30M | 1 |
| European Parliament documents | CC BY-SA 4.0 | formal, largely translated | 5M | 1 |
| EELLAK articles | CC BY-SA 4.0 | articles on open technology | none | 1 |
| Wikisource, monotonic texts | CC BY-SA 3.0 | literary | none | 1 |
| Tatoeba | CC BY 2.0 FR | short everyday sentences | none | 10 |
| Common Voice sentence collection | CC0 1.0 | read-aloud sentences | none | 2 |
| Wiktionary usage examples | CC BY-SA 4.0 | short example sentences | none | 5 |

The registers of spoken debate and of citizens' comments are the core;
encyclopedic and formal text is capped so that it cannot dominate. The
weights are recorded for the trainer and do not change the shards.
OpenCouncil (municipal council transcripts) is in the manifest but disabled:
its terms are not a named open license and the dataset is gated.

## Normalization and tokenization

`build/el_text.py` has no input or output of its own; the tests in
`tests/test_el_text.py` pin its behavior.

- Text is normalized to NFC. Letters with oxia become the letters with tonos,
  the Greek question mark becomes `;` and the ano teleia becomes U+00B7.
  Soft hyphens, zero-width characters and byte-order marks are removed.
- The apostrophe U+0027, U+02BC, U+2018, the spacing tonos, the acute accent
  and the Greek koronis are folded to U+2019 when they mark an elision or an
  aphaeresis. `σ' αγαπώ` gives `σ’` and `αγαπώ`; `σ΄αγαπώ` stays one token,
  `σ’αγαπώ`, because the keyboard treats U+2019 as part of a word. A mark
  between quoted words is dropped.
- A spacing tonos before the first vowel of a word is joined to it:
  `΄Οταν` becomes `Όταν`. No-break and thin spaces become spaces.
- Latin letters that look like Greek letters inside a Greek word are
  replaced (`τo` becomes `το`). Every repair is counted, and the
  per-source table `homoglyph_repairs.tsv.gz` lists each original and
  corrected spelling.
- Hyphens, digits, spaces and punctuation separate words, as on the keyboard,
  so `γερο-Ανέστης` gives `γερο` and `Ανέστης`.
- A word-final σ becomes ς, except before a hyphen.
- Numbers and Greek numerals (`Β΄`, `ΙΑ΄`, list markers such as `ΙΑ.`)
  become `<NUM>`. Latin words, other scripts, URLs and e-mail addresses
  become `<FOREIGN>`.
- Sentences end at the keyboard's sentence breakers: a line break, `.`, `!`,
  `?`, `;`, `·` and `…`. Known abbreviations keep their period and do not end
  the sentence (`κ.`, `π.χ.`, `Ε.Ε.`, `Τ.Ε.Ι.`, `δισ.`). An abbreviation that
  often closes a sentence (`κ.λπ.`, `δισ.`) ends it when a capital follows.
  Words that a lowercase word follows after a period are counted in
  `abbreviation_candidates.tsv.gz` for review.
- Case is kept. `el_text.lower_el` lowercases with the final-sigma rule, and
  `el_text.sentence_key` is the case-folded key used for duplicates.

## Filters

Each dropped document or sentence is counted by reason, with its tokens,
under `filters` in the output manifest.

A document is dropped when:

- more than 2% of its Greek words are polytonic (at least two);
- more than 1% of its words are Katharevousa markers (at least three), such
  as `ταις`, `ούτω`, `ήτο`, or the endings -ειν, -ουσι, -ωσι;
- more than half of its words are foreign;
- more than a quarter of its polysyllables lack an accent (at least five),
  which marks text typed without accents.

A sentence is dropped when it has a polytonic word, no Greek word, more than
one Katharevousa marker, a foreign majority, only capitals, or a quarter or
more unaccented polysyllables. For Wikipedia, a sentence must also end with
punctuation, which removes headings, list items and table rows, and the
article stops at headings such as `Παραπομπές`. The unaccented test counts
vowel groups and glides, so `μια`, `ποιος`, `γεια` and `δυο` count as one
syllable and need no accent, and it ignores words in capitals, proclitic
prefixes such as `γερο-` and aphaeresis forms such as `’θελες`. Modern verb
forms such as `κλαις` and the adverb `εντούτοις` are not Katharevousa
markers.

## Split, repeats and caps

- A document goes to dev by a hash of its group (seed 4242): a sitting of the
  Parliament, an OpenGov consultation, a European Parliament file, a
  Wiktionary entry, or the document itself. A group is never on both sides.
  The dev share is 2%, and 5% for the three small conversational sources.
- The run makes two passes. The first collects the case-folded key of every
  dev sentence of every source. The second removes every training sentence
  with a dev key, so repeated speeches, campaign comments and quotations do
  not leak into dev.
- In both splits a sentence is kept at most 5 times per source
  (`max_sentence_repeats`). This bounds procedural formulas and copied
  comments. The manifest records the tokens other limits would remove
  (`repeat_cap_tokens_removed`) and the most repeated sentences.
- Both rules apply only to sentences of at least 5 tokens
  (`dedup_min_tokens`). Short sentences such as `Ευχαριστώ`, `Το ξέρω`,
  `Τι να κάνουμε` and `Γεια σου` recur naturally in every source: removing
  their repeats, or every training copy of one that also occurs in dev,
  would remove the most conversational sentences from training. Their count
  is `short_sentences`.
- A token cap keeps a fixed sample of training documents: those whose cap
  bucket (`el_corpus.cap_bucket`, a hash of the document id with its own
  seed) is below the recorded `bucket_threshold`. All training documents are
  written; `el_corpus.iter_corpus_sentences` applies the cap, so caps can be
  changed without extracting again.

## Output

```
build/el_corpus/
  manifest.json                            counts per source and split, filters,
                                           caps, register rates, input hashes
  <source>/train-00000.jsonl.gz            {"id", "group", "s": [sentences]};
  <source>/dev-00000.jsonl.gz              tokens joined by single spaces
  <source>/abbreviations.tsv.gz            abbreviation counts (train)
  <source>/abbreviation_candidates.tsv.gz  words to review as abbreviations
  <source>/homoglyph_repairs.tsv.gz        original and repaired spellings
```

Shards rotate every 20M tokens. Their gzip headers carry no name or time, so
a rerun of the same inputs and code writes the same bytes; `manifest.json`
records the commit, the sha256 of `data/el_lm_corpora.json` and of
`build/el_text.py`, and the sha256 of every shard and table.

## Recorded corpus

Full extraction at commit 7be7f22, 2026-11-06, 5 minutes on an 18-core Mac.
Train words exclude `<NUM>`, `<FOREIGN>` and abbreviations; tokens include
them. "Filtered" is the share of the source's tokens removed by the document
and sentence filters. The last three columns are occurrences per million
training words, case-folded.

| Source | Documents | Train words | Train tokens | Under cap | Dev tokens | Filtered | Dev duplicates removed | Repeats removed (tokens) | είμαι | ξέρω | εντάξει |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Parliament | 1,280,927 | 166.6M | 170.0M | 100.0M | 3.78M | 0% | 179,219 | 0.45M | 334.2 | 271.6 | 79.7 |
| OpenGov comments | 441,886 | 52.4M | 54.5M | 54.5M | 1.86M | 7% | 14,680 | 21.13M | 262.6 | 44.8 | 17.3 |
| Wikipedia | 226,834 | 83.3M | 89.8M | 30.0M | 1.92M | 11% | 24,711 | 0.21M | 23.8 | 5.2 | 4.1 |
| European Parliament | 154,876 | 85.1M | 88.0M | 5.0M | 1.90M | 21% | 169,416 | 2.84M | 8.2 | 0.1 | 4.7 |
| EELLAK articles | 7,284 | 2.75M | 2.97M | 2.97M | 56K | 28% | 549 | 9K | 35.6 | 18.5 | 14.9 |
| Wikisource | 4,806 | 2.23M | 2.25M | 2.25M | 108K | 83% | 122 | 1K | 552.5 | 403.3 | 1.3 |
| Tatoeba | 42,985 | 237K | 237K | 237K | 12K | 0% | 26 | 0 | 5,279.2 | 1,452.8 | 156.3 |
| Common Voice | 6,461 | 41K | 41K | 41K | 2K | 0% | 0 | 0 | 710.8 | 857.9 | 24.5 |
| Wiktionary examples | 56,919 | 674K | 705K | 705K | 44K | 11% | 2,759 | 31K | 368.1 | 148.4 | 49.0 |

In total, train has 408.5M tokens (393.4M words, 18.75M sentences) and dev
has 9.68M tokens (449,676 sentences); 195.7M training tokens are under the
caps. Wikisource loses most of its text because most of its documents are
polytonic. The repeat cap removes 21.1M tokens of OpenGov comments: copies of
the same text submitted by many people.

The register rates show what the sources contribute. Parliament has the
first person and the spoken `εντάξει` at rates near those of the
conversational sets, OpenGov the first person but little `ξέρω`, and
Wikipedia and the European Parliament documents almost neither. Exempting
short sentences from the repeat rules raised the Parliament rate of
`εντάξει` from 58.8 to 79.7 per million, because it is often a sentence by
itself.

## Known limitations

- The duplicate rules match exact case-folded token sequences. OpenGov
  campaign comments that change a few words survive, and their sentences
  can still be in train and in dev.
- `εντάξει` is also a form of the verb `εντάσσω` (`να εντάξει`). After `να`,
  `θα`, `μην` or `δεν` it is about 5% of the Parliament occurrences, 24% in
  OpenGov and 29% in Wikipedia, so its rate is not a pure register measure.
- Tatoeba replaces its per-language export every week, so the pinned file
  can no longer be downloaded once it changes; a fresh download needs a new
  pin. The Wiktionary pins depend on the local kaikki.org dumps.
- The European Parliament documents are formal and largely translated. They
  are capped at 5M tokens until a comparison shows whether they help.
- The open sources hold little casual written conversation. The best
  candidates found (theatre dialogue, oral history, subtitles) are
  NonCommercial, NoDerivatives or unlicensed, and OpenCouncil is disabled
  until its terms are a named open license.
