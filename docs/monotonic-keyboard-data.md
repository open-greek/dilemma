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
  (`το όνομά μου`), which the keyboard decides from the model;
- the keystrokes a suggestion bar with three slots saves.

It reads nothing else: no lexicon, no tagger, no counts beyond those the
two files hold. The one exception is a variant of the enclitic measure
that reads the training split; it is marked where it appears. The
restoration rule is `build/el_accent.py`, a module that reads no files,
so a keyboard implementation can be checked against it word for word.

The numbers below are for the word list (631,330 entries, sha256
`b21c4047`) and the model trained from it (sha256 `8f62be97`), both built
at commit cc4cdaa on 2026-11-07.

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
   605,371 keys, and 22,806 of them have two or more spellings. When the
   word is typed in lowercase and a lowercase candidate exists,
   capitalized candidates (names) are left out.
   When only names have the letters, a lowercase typed word can take
   accents only from names rated C or M. R and X names are excluded;
   deliberately capitalized input can match names of every rating. This
   keeps the slip `μαι` from becoming `μάι` through the rare name `Μάι`.
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
   accent as the keyboard decides it, from the model's evidence (see [the
   enclitic second accent](#the-enclitic-second-accent)).

A third run, *with the next word*, adds to each candidate's score the best
score of the next typed word's candidates after it. This is what a
keyboard can do when it revises a word once the next one is typed.

A word that is right as typed (a monosyllable, an unaccented word, the
article `η`) is *damaged* when restoration changes it. The damage rate is
over such words whose spelling the list holds. Typed words the list lacks
are counted apart when restoration changes them (5,637 words): corpus
misspellings (`απο`, `οτι`), sentence-initial words written without the
tonos on their capital (`Ενωση`, `Οταν`), Katharevousa words (`μοι`, `αι`)
and prefixes cut from a hyphenated compound (`αντι-`, `κατα-`).

Test half, policy `keep`, margin 1.0, word margin 3.0 (percentages):

| Set | Words | Accuracy | Accuracy, listed spellings | Damage rate | Left bare | No candidate |
|---|---|---|---|---|---|---|
| Common Voice | 1,049 | 96.00 | 97.67 | 0.24 | 2.67 | 1.43 |
| EELLAK articles | 22,867 | 98.67 | 98.97 | 0.00 | 1.77 | 0.29 |
| European Parliament | 879,572 | 98.58 | 98.99 | 0.01 | 1.38 | 0.54 |
| OpenGov comments | 867,804 | 97.38 | 98.58 | 0.02 | 1.72 | 1.02 |
| Parliament | 1,790,286 | 98.28 | 98.85 | 0.01 | 1.59 | 0.46 |
| Tatoeba | 6,173 | 97.76 | 98.67 | 0.00 | 1.62 | 0.58 |
| Wikipedia | 873,267 | 96.02 | 98.60 | 0.01 | 1.82 | 2.59 |
| Wikisource | 19,463 | 82.37 | 97.85 | 0.05 | 2.55 | 14.84 |
| Wiktionary examples | 20,637 | 95.26 | 98.38 | 0.03 | 2.09 | 2.95 |
| All, pooled | 4,481,118 | 97.64 | 98.77 | 0.01 | 1.63 | 1.07 |
| Conversational, pooled | 27,859 | 95.84 | 98.42 | 0.03 | 2.01 | 2.37 |

*Accuracy* is the share of typed words restored to the corpus spelling;
*accuracy, listed spellings* counts only words whose corpus spelling is an
entry (98.65% of the words; a double-accented word counts when its spelling
without the second accent is one). The Common Voice damage rate is one word
of 410. Wikisource is older literary text, and 14.84% of its words have no
candidate (see [Known limitations](#known-limitations)).

The three runs, pooled:

| Run | Accuracy | Accuracy, listed spellings | Damage rate | Damaged words | Left bare | Conversational accuracy |
|---|---|---|---|---|---|---|
| `keep` | 97.64 | 98.77 | 0.012 | 191 of 1,615,184 | 1.63 | 95.84 |
| `top` | 97.83 | 98.96 | 0.012 | 193 of 1,615,184 | 1.32 | 96.15 |
| `keep`, with the next word | 97.99 | 99.13 | 0.025 | 406 of 1,615,184 | 0.97 | 96.20 |

By decision, policy `keep`: one candidate 2,491,772 words, 99.41% right;
typed letters already the only spelling 1,424,303 words, 99.86% right;
chosen by the model 444,033 words, 98.01% right; left bare 72,909 words,
44.30% right (mostly `η` kept where the corpus has `η`); no candidate
48,101 words, 17.83% right.

### Choosing the margins

On the tune half, with the margin and the word margin equal:

| Margin | `keep`: accuracy, listed | damage rate | left bare | `top`: accuracy, listed | damage rate | left bare |
|---|---|---|---|---|---|---|
| 0 | 99.03 | 0.35 | 0.03 | 99.04 | 0.35 | 0.02 |
| 0.5 | 98.95 | 0.25 | 0.46 | 99.03 | 0.25 | 0.29 |
| 1 | 98.80 | 0.14 | 0.97 | 99.01 | 0.14 | 0.62 |
| 1.5 | 98.65 | 0.08 | 1.30 | 98.97 | 0.08 | 0.81 |
| 2 | 98.43 | 0.03 | 1.92 | 98.92 | 0.03 | 1.21 |
| 3 | 98.06 | 0.01 | 2.48 | 98.86 | 0.01 | 1.43 |
| 4 | 97.58 | 0.01 | 3.15 | 98.84 | 0.01 | 1.61 |
| 6 | 96.38 | 0.00 | 5.49 | 98.81 | 0.00 | 2.73 |

Damage only happens when the typed letters are a word, so it depends on
the word margin alone; the margin trades accuracy against words left bare.
With the margin at 1.0, the word margin:

| Word margin | Left context: accuracy, listed | damage rate | left bare | With the next word: accuracy, listed | damage rate | left bare |
|---|---|---|---|---|---|---|
| 1 | 98.80 | 0.14 | 0.97 | 99.12 | 0.11 | 0.57 |
| 2 | 98.71 | 0.03 | 1.56 | 99.07 | 0.06 | 0.87 |
| 3 | 98.65 | 0.01 | 1.78 | 99.01 | 0.03 | 1.05 |
| 4 | 98.63 | 0.01 | 1.96 | 98.94 | 0.01 | 1.26 |
| 5 | 98.62 | 0.00 | 2.73 | 98.90 | 0.01 | 1.56 |
| 6 | 98.61 | 0.00 | 3.09 | 98.85 | 0.00 | 1.88 |
| 8 | 98.60 | 0.00 | 4.15 | 98.79 | 0.00 | 2.67 |
| never | 98.59 | 0.00 | 5.32 | 98.71 | 0.00 | 5.22 |

A word margin of 3 brings the damage rate to about 1 in 10,000 for 0.15
points of accuracy against a word margin of 1; never replacing a typed word
costs 0.06 points more and leaves three times as many words bare.

### Ambiguous pairs

Pooled test half; *right* and *left bare* in percent; the last column is
how often the corpus writes the second spelling of the pair:

| Pair | Occurrences | Right, left context | Left bare | Right, with the next word | Left bare | Corpus share of the second spelling |
|---|---|---|---|---|---|---|
| η, ή | 94,399 | 83.0 | 35.1 | 92.5 | 11.6 | 18.2 |

- **η, ή.** The disjunctive `ή` is 18.2% of the occurrences. In left
  context the best candidate often does not lead by the word margin (35.1%
  left bare), so `η` stays and is wrong for 15,945 words, the largest
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
  for about 10,000 of the 105,594 errors of the `keep` run.

### Errors and damage

The commonest errors of the `keep` run, as (corpus, output, count):
`ή`→`η` left bare 15,024; `μία`→`μια` 5,799 and `δύο`→`δυο` 4,009 left
bare; `πώς`→`πως` 1,414; `ή`→`η` chosen by the model 921; `πού`→`που`
left bare 580; `τού`→`του` 573 (an old accented monosyllable the list
rejects); `φυτοπροστατευτικών` 456 (no candidate: the word list lacks
it). The `ό` of `ό,τι`, which the tokenizer used to split off, is no
longer among them: `ό,τι` is one token and one entry.

All 191 damaged words of the `keep` run are replacements of one valid
word by its twin: `η`→`ή` 86, `δυο`→`δύο` 66, `μια`→`μία` 26, `πως`→`πώς` 4,
`που`→`πού` 3, `μιας`→`μίας` 2, `βιο`→`βίο` 2, `θειο`→`θείο` 1,
`μιαν`→`μίαν` 1. The run with the next word damages more of the same
words (`η`→`ή` 124, `δυο`→`δύο` 119, `μια`→`μία` 78).

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
| One spelling of /i/ for another (ι, η, υ, ει, οι) | 8,814 | 13.5 | 86.5 | 80.0 |
| ο for ω, ω for ο | 6,491 | 29.5 | 70.5 | 66.9 |
| ε for αι, αι for ε | 5,963 | 10.1 | 89.9 | 86.5 |
| Doubled or single consonant | 7,294 | 3.4 | 96.6 | 90.6 |
| Neighboring key | 20,911 | 10.6 | 89.4 | 0.3 |
| All | 49,473 | 12.5 | 87.5 | 46.9 |

The slips that land on another word are mostly short function words and
their twins: `το`→`τω` 599, `των`→`τον` 398, `με`→`μάι` 392,
`να`→`μα` 386, `του`→`ρου` 207, `τον`→`των` 197, `από`→`άπω` 188,
`τα`→`ρα` 171, `της`→`τις` 154, `της`→`τοις` 153. Some are real
confusions only context can catch (`των` and `τον`, `της` and `τις`, and
the learned `τω` and `τοις` of `εν τω μεταξύ` and `τοις εκατό`). Others
land on rare entries: the name `Μάι`, which restoration writes in
lowercase when no lowercase entry has the letters, and the interjections
`ρα` and `ρου`. The rarest two- and three-letter forms that a slip of a
very common word produces are no longer entries (`μάι`, `ξάι`, `άε`; see
[changes to the word list](#changes-to-the-word-list)), so slips of `και`
no longer land on `ξάι`. A keyboard's correction should still weigh an
entry by its rating and the context before taking the typed letters as
meant.

The same-sound lookup recovers most sound slips but also changes 1,253 of
the 6,526 unslipped words that have no candidate (19.2%): words the list
lacks that sound like an entry. It needs a frequency and context check
too. Unslipped words in the same sentences are restored as well as
elsewhere (96.52% right, damage 0.013%).

## The enclitic second accent

A word stressed on the third syllable from the end takes a second tonos on
its last syllable before an enclitic pronoun: `ο άνθρωπός μου`, `άκουσέ
με`. Syllables are counted with synizesis (`el_lexicon.is_proparoxytone`):
an unstressed ι or ε before a vowel is a glide, so `σπίτια` and
`οικογένεια` count as stressed on the second-to-last syllable and take no
second accent. For a few words (`δωμάτιο`, `προσπάθεια`) both
pronunciations are heard, so the spelling cannot decide
(`el_accent.needs_synizesis_call`).

The word list holds no double accents. The model keeps them as tokens of
their own: the double-accented form of every list spelling that can take
one (`όνομά` beside `όνομα`, `el_accent.enclitic_twin`), seen at least 3
times in training and with its host in the vocabulary, 5,784 forms (see
the [next-word model](el-monotonic-lm.md#which-spellings-are-counted)).
So the model can say which reading is likelier, and four rules are
compared over every dev occurrence of a typed host followed by an
enclitic (whole dev split, 1,029,422 hosts):

- **Every proparoxytone:** the syllable rule before any enclitic.
- **End-of-sentence rule** (`el_accent.takes_enclitic_accent`), the
  keyboard's rule before the model kept the double-accented forms: `μου`,
  `σου`, `μας` and `σας` are always pronouns after a noun or verb, so the
  rule applies before them. `του`, `της`, `τους`, `το`, `τα`, `τον`,
  `την`, `τη` and `τις` are also articles (`ο δάσκαλος του σχολείου`),
  and `με` and `σε` are also prepositions, so before those the rule
  applies only at the end of a sentence, the one case left context and
  the next token settle without a part-of-speech tagger.
- **Keyboard with model evidence** (`el_accent.keyboard_enclitic_decision`),
  what the keyboard does: before `μου`, `σου`, `μας`, `σας`, `του`, `της`,
  `τους`, `το`, `τα`, `τον`, `την` and `τις`, the model scores both
  readings, the host form after the two words before it plus the enclitic
  after the form, plus the next word after the form and the enclitic when
  one follows and both readings score it (`el_accent.enclitic_evidence`).
  The reading that leads by 1.0 (natural log) is written. Where the model
  has no say (a form it has not seen, or a smaller lead), a pronoun takes
  the syllable rule, and an article homograph takes it at the end of a
  sentence and, before a word, only when that word marks the enclitic
  reading (`και`, `να`, `που`, `στο` and the rest of
  `el_accent.POSSESSIVE_SIGNALS`). `με`, `σε` and `τη` never take it.
- **Training text first** (reads the training split, not the bundle): the
  majority spelling of the host before that enclitic and the next word in
  the training split, when it was seen at least 5 times; else of the host
  before that enclitic; else the end-of-sentence rule.

*Precision* is the share of added accents the corpus has; *recall* the
share of the corpus's second accents the rule adds; *agreement* the share
of all hosts where rule and corpus agree.

| Rule | Enclitics | Hosts | Corpus has the second accent | Rule adds it | Precision | Recall | Agreement |
|---|---|---|---|---|---|---|---|
| Every proparoxytone | μου, σου, μας, σας | 35,702 | 5,477 | 6,423 | 76.7 | 89.9 | 94.25 |
| Every proparoxytone | του, της, τους, το, τα, τον, την, τη, τις | 811,501 | 19,366 | 195,116 | 8.9 | 90.1 | 77.87 |
| Every proparoxytone | με, σε | 182,219 | 67 | 63,841 | 0.0 | 46.3 | 64.96 |
| End-of-sentence rule | μου, σου, μας, σας | 35,702 | 5,477 | 6,423 | 76.7 | 89.9 | 94.25 |
| End-of-sentence rule | του, της, τους, το, τα, τον, την, τη, τις | 811,501 | 19,366 | 3,834 | 88.5 | 17.5 | 97.98 |
| End-of-sentence rule | με, σε | 182,219 | 67 | 21 | 38.1 | 11.9 | 99.96 |
| Keyboard with model evidence | μου, σου, μας, σας | 35,702 | 5,477 | 5,855 | 83.0 | 88.7 | 95.48 |
| Keyboard with model evidence | του, της, τους, το, τα, τον, την, τη, τις | 811,501 | 19,366 | 12,795 | 86.5 | 57.1 | 98.76 |
| Keyboard with model evidence | με, σε | 182,219 | 67 | 0 | none added | 0.0 | 99.96 |
| Training text first | μου, σου, μας, σας | 35,702 | 5,477 | 6,343 | 83.4 | 96.5 | 96.51 |
| Training text first | του, της, τους, το, τα, τον, την, τη, τις | 811,501 | 19,366 | 15,400 | 85.2 | 67.7 | 98.95 |
| Training text first | με, σε | 182,219 | 67 | 8 | 87.5 | 10.4 | 99.97 |

- Before the article-like enclitics, adding the accent after every
  proparoxytone would be wrong nine times in ten (`η απόφαση του
  υπουργού`). The end-of-sentence rule adds it only there: 88.5%
  precision, but it finds 17.5% of the accents the corpus has. With the
  model's evidence the keyboard finds 57.1% at 86.5% precision, because
  the pair and the next word usually tell the pronoun from the article
  (`η απόφασή του να` against `η απόφαση του υπουργού`). That is most of
  the gain of following the training text itself (67.7% at 85.2%), carried
  by the bundle. By enclitic: `τους` 80.6% recall at 87.8% precision,
  `του` 50.0% at 87.9%, `της` 36.0% at 86.4%. After `το`, `τα`, `τον`,
  `την` and `τις` the corpus has 371 second accents in all; the keyboard
  adds 302, of which 112 are right.
- Before `μου`, `σου`, `μας` and `σας`, the syllable rule agrees with the
  corpus 94.25% of the time. Most disagreements are hosts the writer left
  without the second accent (precision 76.7%), and the misses are mostly
  glide words written with it (`οικογένειά μου`, `προσπάθειά μας`), which
  the synizesis count excludes. The model's evidence raises agreement to
  95.48% (precision 83.0%, recall 88.7%): it adds the accent to some glide
  words the writers accent, and it leaves it out where the writers mostly
  do. Standard spelling wants the accent on every proparoxytone before a
  pronoun, so part of that precision is the model following the writers'
  omissions: it adds the accent to 5,855 hosts where the syllable rule
  adds it to 6,423.
- Of the 224,813 dev hosts that have a double-accented form, the model
  knew both forms for 165,125; the other 59,688 were decided by rule
  alone.
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
| Common Voice | 1,049 | 9.7 | 34.5 | 37.5 | 36.0 |
| EELLAK articles | 22,867 | 17.0 | 45.5 | 49.7 | 48.9 |
| European Parliament | 98,768 | 22.3 | 48.7 | 54.0 | 53.3 |
| OpenGov comments | 70,402 | 18.9 | 46.9 | 51.3 | 50.5 |
| Parliament | 53,546 | 22.0 | 48.4 | 53.6 | 52.8 |
| Tatoeba | 6,173 | 15.7 | 39.5 | 44.4 | 43.3 |
| Wikipedia | 57,962 | 15.4 | 40.8 | 45.0 | 44.1 |
| Wikisource | 19,463 | 6.5 | 21.7 | 24.1 | 23.1 |
| Wiktionary examples | 20,637 | 12.1 | 36.4 | 39.7 | 38.8 |
| All, pooled | 350,867 | 18.6 | 44.6 | 49.2 | 48.4 |
| Conversational, pooled | 27,859 | 12.7 | 36.9 | 40.5 | 39.6 |

Pooled, the word is among the three predictions before its first letter
for 30.6% of words, and among the completions at a later letter for
another 59.7% (57.2% with the commit-aware bar). Predictions before the
first letter add 4.6 points to completions alone, and the commit-aware
layout costs 0.8 points. The first predictions most often shown are `και`, `η`, `την`,
`να`, `του`, `της`, `το`, `των`, `που` and `θα`; the first completions
after one letter `και`, `το`, `είναι`, `την`, `σε`, `με`, `που`, `να`,
`από` and `του`. No junk word is among them, and no double-accented
form: the evaluation hides those as a keyboard must
(`GNLMReader.hide_double_accented`). The script's completion lookup
matches `gnlm_reader` on all 400 sampled queries.

These savings assume the user taps every suggestion the moment it
appears, so they are an upper bound for the three-slot bar, not a
prediction of what people save.

## What these numbers mean for a keyboard

- **Restore with a margin and a larger word margin.** Margin 1.0 and word
  margin 3.0 restore 98.77% of listed spellings with 191 damaged words in
  1.6M (0.012%). When the model is unsure, the bar is the place for the
  candidates.
- **Revise a word when the next one is typed.** Scoring with the next word
  raises `η`, `ή` from 83.0% to 92.5% right, `πότε`, `ποτέ` from 57.0% to
  84.4% and `πάρα`, `παρά` from 66.3% to 95.7%, at a damage rate of
  0.025%, still far below 0.5%.
- **Leave both-valid variants as typed.** `μια`, `μία` and `δυο`, `δύο`
  are both correct; no context chooses between them.
- **Look up a double-accented word as typed, and never offer one.** The
  model holds 5,784 double-accented forms (`όνομά`) beside their hosts,
  so a context word such as `όνομά` is looked up as it is, and only a
  double-accented word the model lacks is looked up without its second
  accent. These forms are no entries of the list: predictions and
  completions must skip them (any vocabulary token with two tonos).
- **Do not restore a word typed before a hyphen.** `αντι-` and `κατα-` in
  a hyphenated compound are prefixes, not the words `αντί` and `κατά`.
- **Treat `ό,τι` as one unit.** The list has `ό,τι` as one entry and the
  model as one token, and has no `ό`, so the keyboard must not split
  `ό,τι` at its comma for restoration or suggestions.
- **Decide the enclitic accent from the model.** Comparing the two
  readings finds 57.1% of the second accents before the article-like
  enclitics at 86.5% precision, against 17.5% for the end-of-sentence
  rule.
- **Check slip corrections against frequency and context.** 12.5% of
  simulated slips land on another entry, often a rare one.
- **Read accent alternatives as suggestions.** An entry's `ac:spelling`
  metadata points to a much commoner accent alternative. The rare entry
  remains a valid word, so this is evidence for a proposal, not permission
  to discard the typed reading.

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

A second review the same day changed the list and the model: 631,696
entries before, 631,330 after (618 removed, 252 added, 1,411 moved from C
to M). The [word list](el-monotonic-word-list.md) page describes each
rule. The changes that move the measures on this page:

- The model keeps the double-accented forms as tokens of their own, and
  the keyboard decides the second accent from them (see [the enclitic
  second accent](#the-enclitic-second-accent)). Before the article-like
  enclitics, recall rose from 17.5% to 57.1%.
- `ό,τι` is one token and one entry. The test half has 4,481,118 typed
  words instead of 4,482,417, and the 1,336 errors on its `ό` are gone.
- Rare entries beside a far commoner spelling of the same letters with
  another accent are left out (537, such as `καλήμερα` and `Σαββάτο`), so
  22,806 keys have two or more spellings instead of 23,286, and 2,491,772
  words have one candidate instead of 2,469,233.
- The rarest two- and three-letter forms one sound or one key from a very
  common word are left out (27, such as `μάι`, `ξάι` and `άε`).
  Neighboring-key slips that land on another entry fell from 12.1% to
  10.6%.
- Names that two sources write with their capital (`Ένωση`, `Εθνών`,
  `Επιτροπή`) are entries; capitals that only the Parliament minutes use
  (`Βουλευτή`, `Αίθουσα`) are not.

Together, restoration accuracy rose from 97.54% to 97.64% (98.69% to
98.77% of listed spellings), damaged words stayed at 191, and the share of
simulated slips that land on another entry fell from 13.12% to 12.47%.

## Tests

`tests/test_el_accent_restoration.py` covers `build/el_accent.py`
(stripping and transferring marks, the candidate filter by case, each
decision kind, the word margin, the enclitic accent with synizesis, the
end-of-sentence rule, the double-accented form of a host, and the
keyboard's decision from a model's evidence, with and without the next
word) and the evaluation on the fixture model of `tests/test_el_ngram.py`
(restoration in left context, damage counted only on listed words,
completions that match the reader, keystroke counts, the next-word score,
the enclitic tables). `tests/test_el_monotonic_list.py` covers the
typing-slip rule, the aspirated elisions and the rules of the second
review.

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
- **Rare short entries.** The rarest short slips of common words are no
  longer entries, but names (`Μάι`) and interjections that FrequencyWords
  rates high (`ρα`) still absorb slips of common words, and the learned
  `τω` and `τοις` absorb slips of `το`, `της` and `τις`.
- **The second accent follows the writers.** The model learned the
  corpus spellings, omissions included, so before a pronoun it leaves the
  accent off some hosts that standard spelling accents (5,855 accents
  against 6,423 for the syllable rule), and before the article-like
  enclitics it still misses 42.9% of the corpus's accents.
