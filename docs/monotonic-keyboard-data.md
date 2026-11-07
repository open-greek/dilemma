# Monotonic Modern Greek keyboard data

[README](../README.md)

- [Reproduce](#reproduce)
- [Text](#text)
- [Accent restoration](#accent-restoration)
- [Typing slips](#typing-slips)
- [The enclitic second accent](#the-enclitic-second-accent)
- [Keystroke savings](#keystroke-savings)
- [What these numbers mean for a keyboard](#what-these-numbers-mean-for-a-keyboard)
- [Changes to the word list](#changes-to-the-word-list)
- [Tests](#tests)
- [Known limitations](#known-limitations)

`eval/eval_el_accent_restoration.py` measures the two files of the
monotonic keyboard bundle, the [word list](el-monotonic-word-list.md)
`build/hunspell/el_monotonic.dic` and the
[next-word model](el-monotonic-lm.md) `build/lm/el_ngram.bin`, the way a
keyboard uses them, on the held-out dev split of the
[monotonic corpus](el-monotonic-corpus.md):

- restoring accents to words typed without them;
- what common typing slips turn into;
- the second accent a word takes before an enclitic pronoun
  (`το όνομά μου`);
- the keystrokes a suggestion bar with three slots saves.

It reads nothing else: no lexicon, no tagger, no counts beyond those the
two files hold. The one exception is a variant of the enclitic measure
that reads the training split; it is marked where it appears. The
restoration rule is `build/el_accent.py`, a module that reads no files,
so a keyboard implementation can be checked against it word for word.

The numbers below are for the word list (631,696 entries, sha256
`abef641c`) and the model trained from it (sha256 `227240c4`), both built
at commit 489336b on 2026-11-07. The export at commit e2315cd wrote the
same `.dic` and dropped seven abbreviations.

## Reproduce

The bundle is built in four steps, each reading the output of the one
before. Besides the standard library they need numpy, pyarrow and
huggingface_hub (`python -m pip install numpy pyarrow huggingface_hub`).
From a fresh clone:

```bash
hf download ciscoriordan/dilemma data/lookup.db data/mg_pairs.json \
    data/el_participles.json --local-dir .       # Wiktionary inputs of the word list
python extract_el_monotonic_lm.py --download    # fetch the pinned corpus inputs
python extract_el_monotonic_lm.py --sanity      # 2,000 documents per source, about 20 s
python extract_el_monotonic_lm.py               # 1. the corpus, build/el_corpus, about 6 minutes
python export_el_monotonic.py                   # 2. the word list, build/hunspell, about 3 minutes
python train_lm.py --lang el                    # 3. the counts, build/lm_el, under a minute
python export_lm.py --lang el                   # 4. build/lm/el_ngram.{bin,version}, about 15 s
python export_lm.py --lang el --check-dictionary
                                                # exits 1 if the model is older than the list
```

The word list refuses Hugging Face inputs whose sha256 differs from
`data/hf_manifest.json`. The Wiktionary usage examples of the corpus are
read from the kaikki.org dumps that `build_data.py` uses (`kaikki/`, or
`$DILEMMA_KAIKKI_DIR`), and the Tatoeba export is replaced upstream every
week, so a machine without the pinned copies needs new pins first (see the
corpus page's [known limitations](el-monotonic-corpus.md#known-limitations)).
The model takes its vocabulary from the word list, so steps 3 and 4 follow
every export of the list; `export_el_monotonic.py` prints a note when the
model in `build/lm` was built from another list.

A rebuild from a fresh clone at commit 3862f50 on 2026-11-07 (about 10
minutes on an 18-core Mac) reproduced the corpus shards, the `.dic`, the
two companion lists, the evidence table and `el_ngram.bin` byte for byte.
The `.aff`, the `.version` files, the audit and the corpus manifest differ
only in the commit they record.

The evaluation:

```bash
python eval/eval_el_accent_restoration.py --quick
                                                # small samples, about 1 minute
python eval/eval_el_accent_restoration.py --enclitic-evidence --workers 12
                                                # build/el_monotonic/accent_eval.json, about 6 minutes
python -m pytest tests/test_el_accent_restoration.py -q
```

The full run prints the tables of this page and writes every number, with
the most frequent groups, errors and examples, to
`build/el_monotonic/accent_eval.json`. `--margin`, `--word-margin` and
`--policy` change the reported setting, `--max-sentences` limits the
restoration runs, and `--tune-sentences`, `--slip-sentences` and
`--keystroke-sentences` set the sample sizes of the other measures (0
skips one). Without `--enclitic-evidence` the run skips the variant that
reads the training split.

## Text

The dev split is held out from both exporters by document group. The
script splits its documents once more by a hash of the document id. The
*tune* half (a sample of 3,000 sentences per source) is used only to choose
the margins below. Every other number comes from the *test* half: 215,483
sentences and 4.48M typed words.

A token is typed when it has only Greek letters (and U+2019 for elision)
and is not written in capitals, except the one-letter words `Ο`, `Η` and
`Ή`. Placeholders, numbers, abbreviations and words in capitals stay as
written and are not counted.

Pooled numbers weight the sources by their words, so Parliament, the
European Parliament, OpenGov comments and Wikipedia dominate them.
*Conversational* pools Tatoeba, Common Voice and the Wiktionary examples.

The corpus spelling is the reference. It is not always right: writers
leave out the enclitic second accent, and some write `απο` for `από`. The
measures below say where this matters.

## Accent restoration

Every typed word loses its tonos and dialytika, keeping its case, and is
then restored word by word, left to right, with `el_accent.decide`:

1. The candidates are the list entries with the same accent-free key
   (`el_lexicon.bare_key`: lowercase, no tonos or dialytika, ς as σ), the
   key a keyboard's compiled dictionary is indexed by. The list has
   605,403 keys, and 23,286 of them have two or more spellings. When the
   word is typed in lowercase and a lowercase candidate exists,
   capitalized candidates (names) are left out.
2. No candidate: the word stays as typed (*no candidate*).
3. One candidate: its accents are applied (*unique*).
4. Several candidates: the model scores each after the two restored words
   before it (`<s>` at the start of a sentence), with the score
   `LMReader.score` gives, in natural-log units. The best candidate is
   applied when it leads the second by the *margin*, 1.0. When the typed
   letters are a word themselves (`η`, `που`, `πως`, `δυο`), replacing
   them needs the larger *word margin*, 3.0. Below that the word stays as
   typed (*left bare*). That is policy `keep`. Policy `top` applies the
   best candidate anyway when the typed letters are not a word, since a
   polysyllable typed without its accent is misspelled whatever is chosen.
5. After the sentence is restored, a host word takes the enclitic second
   accent by the keyboard rule of [the enclitic second
   accent](#the-enclitic-second-accent).

A third run, *with the next word*, adds to each candidate's score the best
score of the next typed word's candidates after it. This is what a
keyboard can do when it revises a word once the next one is typed.

A word that is right as typed (a monosyllable, an unaccented word, the
article `η`) is *damaged* when restoration changes it. The damage rate is
over such words whose spelling the list holds. Typed words the list lacks
are counted apart when restoration changes them (5,675 words): corpus
misspellings (`απο`, `οτι`), sentence-initial words written without the
tonos on their capital (`Ενωση`, `Οταν`), Katharevousa words (`μοι`, `αι`)
and prefixes cut from a hyphenated compound (`αντι-`, `κατα-`).

Test half, policy `keep`, margin 1.0, word margin 3.0 (percentages):

| Set | Words | Accuracy | Accuracy, listed spellings | Damage rate | Left bare | No candidate |
|---|---|---|---|---|---|---|
| Common Voice | 1,050 | 96.00 | 97.77 | 0.24 | 2.67 | 1.43 |
| EELLAK articles | 22,870 | 98.64 | 98.96 | 0.00 | 1.77 | 0.29 |
| European Parliament | 879,658 | 98.42 | 98.84 | 0.01 | 1.38 | 0.54 |
| OpenGov comments | 867,858 | 97.32 | 98.53 | 0.02 | 1.72 | 1.01 |
| Parliament | 1,791,366 | 98.17 | 98.79 | 0.01 | 1.59 | 0.46 |
| Tatoeba | 6,177 | 97.67 | 98.64 | 0.00 | 1.60 | 0.58 |
| Wikipedia | 873,321 | 95.94 | 98.52 | 0.01 | 1.82 | 2.59 |
| Wikisource | 19,474 | 82.32 | 97.82 | 0.05 | 2.57 | 14.82 |
| Wiktionary examples | 20,643 | 95.21 | 98.36 | 0.03 | 2.08 | 2.96 |
| All, pooled | 4,482,417 | 97.54 | 98.69 | 0.01 | 1.63 | 1.07 |
| Conversational, pooled | 27,870 | 95.79 | 98.40 | 0.03 | 2.00 | 2.38 |

*Accuracy* is the share of typed words restored to the corpus spelling;
*accuracy, listed spellings* counts only words whose corpus spelling is an
entry (98.63% of the words; a double-accented word counts when its spelling
without the second accent is one). The Common Voice damage rate is one word
of 411. Wikisource is older literary text, and 14.82% of its words have no
candidate (see [Known limitations](#known-limitations)).

The three runs, pooled:

| Run | Accuracy | Accuracy, listed spellings | Damage rate | Damaged words | Left bare | Conversational accuracy |
|---|---|---|---|---|---|---|
| `keep` | 97.54 | 98.69 | 0.012 | 191 of 1,616,537 | 1.63 | 95.79 |
| `top` | 97.72 | 98.88 | 0.012 | 193 of 1,616,537 | 1.32 | 96.09 |
| `keep`, with the next word | 97.89 | 99.05 | 0.025 | 408 of 1,616,537 | 0.97 | 96.14 |

By decision, policy `keep`: one candidate 2,469,233 words, 99.28% right;
typed letters already the only spelling 1,426,049 words, 99.77% right;
chosen by the model 466,167 words, 98.06% right; left bare 72,918 words,
44.25% right (mostly `η` kept where the corpus has `η`); no candidate
48,050 words, 17.76% right.

### Choosing the margins

On the tune half, with the margin and the word margin equal:

| Margin | `keep`: accuracy, listed | damage rate | left bare | `top`: accuracy, listed | damage rate | left bare |
|---|---|---|---|---|---|---|
| 0 | 98.96 | 0.35 | 0.03 | 98.96 | 0.35 | 0.02 |
| 0.5 | 98.87 | 0.25 | 0.46 | 98.95 | 0.25 | 0.29 |
| 1 | 98.72 | 0.14 | 0.96 | 98.93 | 0.14 | 0.61 |
| 1.5 | 98.57 | 0.08 | 1.30 | 98.89 | 0.08 | 0.81 |
| 2 | 98.35 | 0.03 | 1.92 | 98.84 | 0.03 | 1.21 |
| 3 | 97.98 | 0.01 | 2.48 | 98.78 | 0.01 | 1.43 |
| 4 | 97.49 | 0.01 | 3.17 | 98.76 | 0.01 | 1.62 |
| 6 | 96.27 | 0.00 | 5.54 | 98.74 | 0.00 | 2.74 |

Damage only happens when the typed letters are a word, so it depends on
the word margin alone; the margin trades accuracy against words left bare.
With the margin at 1.0, the word margin:

| Word margin | Left context: accuracy, listed | damage rate | left bare | With the next word: accuracy, listed | damage rate | left bare |
|---|---|---|---|---|---|---|
| 1 | 98.72 | 0.14 | 0.96 | 99.05 | 0.11 | 0.57 |
| 2 | 98.63 | 0.03 | 1.56 | 98.99 | 0.06 | 0.86 |
| 3 | 98.58 | 0.01 | 1.78 | 98.93 | 0.03 | 1.05 |
| 4 | 98.55 | 0.01 | 1.96 | 98.87 | 0.01 | 1.26 |
| 5 | 98.54 | 0.00 | 2.73 | 98.82 | 0.01 | 1.56 |
| 6 | 98.53 | 0.00 | 3.09 | 98.78 | 0.00 | 1.88 |
| 8 | 98.52 | 0.00 | 4.15 | 98.71 | 0.00 | 2.66 |
| never | 98.51 | 0.00 | 5.33 | 98.63 | 0.00 | 5.23 |

A word margin of 3 brings the damage rate to about 1 in 10,000 for 0.14
points of accuracy against a word margin of 1; never replacing a typed word
costs 0.07 points more and leaves three times as many words bare.

### Ambiguous pairs

Pooled test half; *right* and *left bare* in percent; the last column is
how often the corpus writes the second spelling of the pair:

| Pair | Occurrences | Right, left context | Left bare | Right, with the next word | Left bare | Corpus share of the second spelling |
|---|---|---|---|---|---|---|
| η, ή | 94,399 | 83.0 | 35.1 | 92.5 | 11.6 | 18.2 |

- **η, ή.** The disjunctive `ή` is 18.2% of the occurrences. In left
  context the best candidate often does not lead by the word margin (35.1%
  left bare), so `η` stays and is wrong for 15,942 words, the largest
  single source of errors. The next
  word settles most of them (`η Ελλάδα ή η Ιταλία`): 92.5% right.
- **που, πού; πως, πώς.** The question words differ from the conjunctions
  in what follows (`πώς είσαι`, `πού είσαι`), so left context leaves
  71.4% of `πως` bare, and the next word raises `πως`, `πώς` only to
  61.1% right. The questions are rare in this text (`πού` is 1.2% of its
  pair).
- **ποτέ, πότε** and **παρά, πάρα.** Both spellings need accents, so a
  wrong choice is not damage, and the next word settles them (`πάρα
  πολύ`): 84.4% and 95.7% right.
- **μια, μία; δυο, δύο.** Both spellings of each pair are standard, and
  writers choose by taste (`μία` 35.6%, `δύο` 89.2%), so no context
  decides. The word margin leaves them as typed. These two pairs account
  for about 10,000 of the 110,313 errors of the `keep` run.

### Errors and damage

The commonest errors of the `keep` run, as (corpus, output, count):
`ή`→`η` left bare 15,022; `μία`→`μια` 5,797 and `δύο`→`δυο` 4,012 left
bare; `πώς`→`πως` 1,414; `ό`→`ο` 1,336 (the `ό` of `ό,τι`, which the
tokenizer splits off); `ή`→`η` chosen by the model 920; `τού`→`του` 573
(an old accented monosyllable the list rejects); `φυτοπροστατευτικών`
456 (no candidate: the word list lacks it).

All 191 damaged words of the `keep` run are replacements of one valid
word by its twin: `η`→`ή` 86, `δυο`→`δύο` 66, `μια`→`μία` 26, `πως`→`πώς` 4,
`που`→`πού` 3, `μιας`→`μίας` 2, `βιο`→`βίο` 2, `θειο`→`θείο` 1,
`μιαν`→`μίαν` 1. The run with the next word damages more of the same
words (`η`→`ή` 125, `δυο`→`δύο` 119, `μια`→`μία` 78).

## Typing slips

On 3,000 test-half sentences per source, about one typed word in seven
(rate 0.15, seed 5151) gets one common slip before its accents are
stripped. The question is what restoration then does with it: if the
slipped letters are the key of another entry, restoration writes that word
and nothing flags the slip; if they are no key, the word is left for typo
correction. A *same-sound lookup* (`el_lexicon.sound_key`, which merges
the spellings of /i/, /o/ and /e/ and single and double consonants)
stands for a simple correction step.

| Slip | Slipped words | Now another word's key | No candidate | Same-sound lookup recovers the word |
|---|---|---|---|---|
| One spelling of /i/ for another (ι, η, υ, ει, οι) | 8,843 | 13.6 | 86.4 | 79.8 |
| ο for ω, ω for ο | 6,490 | 29.5 | 70.5 | 67.0 |
| ε for αι, αι for ε | 5,970 | 10.0 | 90.0 | 86.8 |
| Doubled or single consonant | 7,305 | 3.4 | 96.6 | 90.6 |
| Neighboring key | 20,865 | 12.1 | 87.9 | 0.3 |
| All | 49,473 | 13.1 | 86.9 | 47.0 |

The slips that land on another word are mostly short function words and
their twins: `το`→`τω` 594, `των`→`τον` 397, `με`→`μάι` 391,
`να`→`μα` 384, `του`→`ρου` 203, `και`→`ξάι` 195, `τον`→`των` 193,
`από`→`άπω` 189, `τα`→`ρα` 176, `της`→`τοις` 153, `της`→`τις` 150. Some
are real confusions only context can catch (`των` and `τον`, `της` and
`τις`, and the learned `τω` and `τοις` of `εν τω μεταξύ` and `τοις
εκατό`). Others land on rare entries from Wiktionary that are hardly
words of modern text: a dialect noun (`μάι`), interjections (`ρα`, `άε`)
and dialect forms (`ξάι`). A keyboard's correction should weigh such an
entry by its rating and the context before taking the typed letters as
meant.

The same-sound lookup recovers most sound slips but also changes 1,251 of
the 6,524 unslipped words that have no candidate (19.2%): words the list
lacks that sound like an entry. It needs a frequency and context check
too. Unslipped words in the same sentences are restored as well as
elsewhere (96.43% right, damage 0.012%).

## The enclitic second accent

A word stressed on the third syllable from the end takes a second tonos on
its last syllable before an enclitic pronoun: `ο άνθρωπός μου`, `άκουσέ
με`. Syllables are counted with synizesis (`el_lexicon.is_proparoxytone`):
an unstressed ι or ε before a vowel is a glide, so `σπίτια` and
`οικογένεια` count as stressed on the second-to-last syllable and take no
second accent. For a few words (`δωμάτιο`, `προσπάθεια`) both
pronunciations are heard, so the spelling cannot decide
(`el_accent.needs_synizesis_call`).

The word list holds no double accents, and the model folds them in
training, so the second accent is added by rule. Three rules are compared
over every dev occurrence of a typed host followed by an enclitic (whole
dev split, 1,029,299 hosts):

- **Every proparoxytone:** the syllable rule before any enclitic.
- **Keyboard rule** (`el_accent.takes_enclitic_accent`): `μου`, `σου`,
  `μας` and `σας` are always pronouns after a noun or verb, so the rule
  applies before them. `του`, `της`, `τους`, `το`, `τα`, `τον`, `την`, `τη`
  and `τις` are also articles (`ο δάσκαλος του σχολείου`), and `με` and
  `σε` are also prepositions, so before those the rule applies only at
  the end of a sentence, the one case left context and the next token
  settle without a part-of-speech tagger.
- **Training text first** (reads the training split, not the bundle): the
  majority spelling of the host before that enclitic and the next word in
  the training split, when it was seen at least 5 times; else of the host
  before that enclitic; else the keyboard rule.

*Precision* is the share of added accents the corpus has; *recall* the
share of the corpus's second accents the rule adds; *agreement* the share
of all hosts where rule and corpus agree.

| Rule | Enclitics | Hosts | Corpus has the second accent | Rule adds it | Precision | Recall | Agreement |
|---|---|---|---|---|---|---|---|
| Every proparoxytone | μου, σου, μας, σας | 35,650 | 5,477 | 6,423 | 76.7 | 89.9 | 94.24 |
| Every proparoxytone | του, της, τους, το, τα, τον, την, τη, τις | 811,451 | 19,366 | 195,116 | 8.9 | 90.1 | 77.87 |
| Every proparoxytone | με, σε | 182,198 | 67 | 63,841 | 0.0 | 46.3 | 64.96 |
| Keyboard rule | μου, σου, μας, σας | 35,650 | 5,477 | 6,423 | 76.7 | 89.9 | 94.24 |
| Keyboard rule | του, της, τους, το, τα, τον, την, τη, τις | 811,451 | 19,366 | 3,834 | 88.5 | 17.5 | 97.98 |
| Keyboard rule | με, σε | 182,198 | 67 | 21 | 38.1 | 11.9 | 99.96 |
| Training text first | μου, σου, μας, σας | 35,650 | 5,477 | 6,343 | 83.4 | 96.5 | 96.50 |
| Training text first | του, της, τους, το, τα, τον, την, τη, τις | 811,451 | 19,366 | 15,400 | 85.2 | 67.7 | 98.95 |
| Training text first | με, σε | 182,198 | 67 | 8 | 87.5 | 10.4 | 99.97 |

- Before `μου`, `σου`, `μας` and `σας`, the syllable rule agrees with the
  corpus 94.24% of the time. Most disagreements are hosts the writer left
  without the second accent (precision 76.7%), and the misses are mostly
  glide words written with it (`οικογένειά μου`, `προσπάθειά μας`), which
  the synizesis count excludes.
- Before the article-like enclitics, adding the accent after every
  proparoxytone would be wrong nine times in ten (`η απόφαση του
  υπουργού`). The keyboard rule adds it only at the end of a sentence:
  88.5% precision, but it finds 17.5% of the accents the corpus has.
- Following the training text's majority spelling raises that recall to
  67.7% at 85.2% precision, because the pair and the next word usually
  tell the pronoun from the article (`η απόφασή του να` against `η
  απόφαση του υπουργού`). The bundle cannot do this today: the model
  folds the second accent in training, so it holds no evidence for it.
  Of the host and enclitic pairs seen at least 5 times in the training
  split, 3,686 pairs and 8,986 triples with the next word are spelled, by
  majority, against the keyboard rule; a table of that size, or a model
  that keeps the two spellings apart, would carry the gain.
- Before `μου`, `σου`, `μας` and `σας` a proparoxytone host should take
  the second accent, yet the corpus leaves it out on 1,499 of 6,423 such
  hosts (23.3%). Those are omissions, so precision and recall against the
  corpus understate a correct rule. Over all enclitics, the share of
  proparoxytone hosts written with the second accent ranges from 7.1%
  (EELLAK articles) to 22.8% (Wikisource).

## Keystroke savings

On 3,000 test-half sentences per source, each word is typed letter by
letter without accents. Before the first letter the bar shows the model's
first three next-word predictions (`LMReader.suggest`); after each letter
it shows completions of the typed stem (`LMReader.suggestCompletions`).
Tapping the word when it appears costs one keystroke and inserts the space;
without the bar a word costs its letters plus the space. The context is
the corpus spelling of the words before, as if each had come out right.
Four layouts:

- **Predictions only:** the bar is used only before the first letter.
- **Completions only:** only after a letter.
- **Three completions:** both, every slot holding a suggestion.
- **Commit-aware:** when pressing space would change the typed letters
  (an accent added), one slot shows the letters as typed and one the
  correction, so one slot is left for completions, or none when the typed
  word has an accent alternative.

Keystrokes saved, percent:

| Set | Words | Predictions only | Completions only | Three completions | Commit-aware |
|---|---|---|---|---|---|
| Common Voice | 1,050 | 10.0 | 34.4 | 37.5 | 36.0 |
| EELLAK articles | 22,870 | 17.0 | 45.5 | 49.7 | 48.9 |
| European Parliament | 98,784 | 22.4 | 48.8 | 54.1 | 53.3 |
| OpenGov comments | 70,407 | 18.9 | 47.0 | 51.4 | 50.6 |
| Parliament | 53,585 | 22.1 | 48.4 | 53.6 | 52.8 |
| Tatoeba | 6,177 | 15.9 | 39.5 | 44.4 | 43.3 |
| Wikipedia | 57,963 | 15.4 | 40.8 | 45.0 | 44.2 |
| Wikisource | 19,474 | 6.5 | 21.8 | 24.1 | 23.2 |
| Wiktionary examples | 20,643 | 12.1 | 36.4 | 39.7 | 38.8 |
| All, pooled | 350,953 | 18.7 | 44.7 | 49.2 | 48.4 |
| Conversational, pooled | 27,870 | 12.8 | 36.9 | 40.6 | 39.6 |

Pooled, the word is among the three predictions before its first letter
for 30.6% of words, and among the completions at a later letter for
another 59.6% (57.1% with the commit-aware bar). Predictions before the
first letter add 4.5 points to completions alone, and the commit-aware
layout costs 0.8 points. The first predictions most often shown are `και`, `η`, `την`,
`να`, `του`, `της`, `το`, `των`, `που` and `θα`; the first completions
after one letter `και`, `το`, `είναι`, `την`, `σε`, `με`, `που`, `να`,
`από` and `του`. No junk word is among them. The script's
completion lookup matches `gnlm_reader` on all 400 sampled queries.

These savings assume the user taps every suggestion the moment it
appears, so they are an upper bound for the three-slot bar, not a
prediction of what people save.

## What these numbers mean for a keyboard

- **Restore with a margin and a larger word margin.** Margin 1.0 and word
  margin 3.0 restore 98.69% of listed spellings with 191 damaged words in
  1.6M (0.012%). When the model is unsure, the bar is the place for the
  candidates.
- **Revise a word when the next one is typed.** Scoring with the next word
  raises `η`, `ή` from 83.0% to 92.5% right, `πότε`, `ποτέ` from 57.0% to
  84.4% and `πάρα`, `παρά` from 66.2% to 95.7%, at a damage rate of
  0.025%, still far below 0.5%.
- **Leave both-valid variants as typed.** `μια`, `μία` and `δυο`, `δύο`
  are both correct; no context chooses between them.
- **Fold the second accent before asking the model.** The model's
  vocabulary has no double-accented words, so a context word such as
  `όνομά` must be looked up as `όνομα`.
- **Do not restore a word typed before a hyphen.** `αντι-` and `κατα-` in
  a hyphenated compound are prefixes, not the words `αντί` and `κατά`.
- **Treat `ό,τι` as one unit.** The list has no `ό`, so the keyboard must
  keep `ό,τι` together for restoration and suggestions.
- **The enclitic accent needs more than the bundle has** for the
  article-like enclitics; see the section above.
- **Check slip corrections against frequency and context.** 13.1% of
  simulated slips land on another entry, often a rare one.

## Changes to the word list

The slip simulation found misspellings that the corpus-word route of
`export_el_monotonic.py` had admitted: one letter away from a very common
word (`είνα`, `είανι`, `πρέπε`, `λοπόν`, `γαιτί`, `φία`, `αποτέλσμα`,
`συγκεριμένα`), plus Katharevousa endings of common words (`έχειν`,
`χώραν`, `κόσμω`). Each one hides a slip of the common word from
correction. The exporter now rejects a corpus word when a known word one
typing slip away (a letter left out, added, replaced, or swapped with the
next one) is at least 300 times as common in both the open corpus and
FrequencyWords. That removed 227 corpus words. The real Modern Greek words
the rule caught are accepted, each with a reason, in
`data/el_monotonic_review.json` (`έδωκε`, `ήτον`, `πολύν`, `καλήν`,
`τρίτιο`, `πόρταλ` and 17 more).

The elision `μεθ’` had been admitted only through the misspelling `μεθά`.
The elision route now accepts an elided τ, π or κ written θ, φ or χ before
a word that had a rough breathing, which admits `μεθ’` (from `μετά`) and
`υφ’` (from `υπό`). With `κανέν` gone, the elision `κανέν’` (from
`κανένα`) is admitted too; the elision route refuses an elision whose
letters are an entry.

Together: 630,688 entries before, 630,463 after. Against the list before
the change, the share of simulated slips that land on another entry fell
from 14.10% to 13.82%, word-list coverage of the dev split stayed at
98.34%, and restoration accuracy stayed at 97.53%. Wikisource lost some
older forms (`ήκουσα`, `είδον`, `ταύτης`): its share of words with no
candidate rose from 15.14% to 15.75%.

A review of the list on 2026-11-07 changed it again: 630,463 entries
before, 631,696 after (1,531 added, 298 removed). The
[word list](el-monotonic-word-list.md) page describes each rule. The
changes that move the measures on this page:

- `κε`, an abbreviation of `κύριε` that FrequencyWords counts often
  enough to rate C, took 1,048 slips of `και`, the most of any entry. The
  list now rejects it, and ε and αι slips that land on another entry fell
  from 28.1% to 10.0%.
- `τοις` and `τω` are now function words, for `τοις εκατό` and `εν τω
  μεταξύ`. They take slips of `της` and `το`, so ο and ω slips that land
  on another entry rose from 19.7% to 29.5%, and slips between the
  spellings of /i/ from 12.0% to 13.6%.
- The list no longer reads the accent-stripped aliases of `lookup.db`, so
  `ξαι` (for `ξάι`) and `σοι` (for `σόι`) are no longer entries.
- Names whose letters are also a word now keep their capital when the
  text writes them with one most of the time (`Ρώμη`, `Βουλή`), and
  capitalized spellings of common words are left out (`Μέσα`, `Κυρία`).
- The corpus no longer reads Wiktionary's editorial notes as usage
  examples, so the Wiktionary examples of the test half have 20,643 typed
  words instead of 21,181.

Together, the share of simulated slips that land on another entry fell
from 13.82% to 13.12%, restoration accuracy rose from 97.53% to 97.54%
(98.69% of listed spellings before and after), damaged words fell from 192
to 191, and Wikisource's share of words with no candidate fell from 15.75%
to 14.82%.

## Tests

`tests/test_el_accent_restoration.py` covers `build/el_accent.py`
(stripping and transferring marks, the candidate filter by case, each
decision kind, the word margin, the enclitic accent with synizesis and the
keyboard rule) and the evaluation on the fixture model of
`tests/test_el_ngram.py` (restoration in left context, damage counted only
on listed words, completions that match the reader, keystroke counts, the
next-word score, the enclitic tables). `tests/test_el_monotonic_list.py`
covers the typing-slip rule and the aspirated elisions.

## Known limitations

- **The reference has errors.** Corpus spellings without the enclitic
  second accent, and misspellings such as `απο`, count against a correct
  restoration or are set apart as unlisted; the measures cannot tell a
  writer's omission from a wrong rule.
- **Register.** Parliament, the European Parliament, OpenGov and Wikipedia
  make up nearly all of the test half. The conversational sets have 27,870
  words, so their numbers are less certain, and casual text with its
  questions (`πού`, `πώς`) is underrepresented.
- **Older text.** Wikisource restores at 82.32%, because 14.82% of its
  words have no candidate: older spellings, Katharevousa forms and dialect
  words that the list leaves out or that the typing-slip rule removes.
- **Simulated typing.** The slips are drawn from five kinds at a fixed
  rate, not from logs of real typing, and the keystroke savings assume a
  user who taps every suggestion at once.
- **Rare short entries.** Wiktionary forms of two or three letters that
  are rare in modern text (`μάι`, `ξάι`, `άε`, `ρα`) remain in the list
  and absorb slips of common words, and the learned `τω` and `τοις`
  absorb slips of `το`, `της` and `τις`. Capitalized entries admitted as
  names can be misspellings too (`Προέδρε`, with the accent of the
  genitive).
- **The second accent before article-like enclitics** is added only at the
  end of a sentence; the better rule needs evidence the bundle does not
  carry.
