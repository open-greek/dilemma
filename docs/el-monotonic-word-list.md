# Monotonic Modern Greek word list

[README](../README.md)

- [Build](#build)
- [Output](#output)
- [Inputs and licenses](#inputs-and-licenses)
- [Spelling rules](#spelling-rules)
- [Admission routes](#admission-routes)
- [Frequency ratings](#frequency-ratings)
- [Companion lists](#companion-lists)
- [Recorded export](#recorded-export)
- [Evaluation](#evaluation)
- [The LibreOffice Greek dictionary](#the-libreoffice-greek-dictionary)
- [Known limitations](#known-limitations)

`export_el_monotonic.py` writes the word list of a monotonic Modern Greek
keyboard bundle: every spelling the keyboard should accept as a word, with a
frequency rating, in monotonic spelling (one tonos, the dialytika, final ς,
U+2019 for elision). It is separate from the older `el_GR_monotonic` variant
of `export_hunspell.py`, which is a projection of `lookup.db` with no corpus
evidence.

## Build

```bash
python extract_el_monotonic_lm.py                 # the corpus, once (see el-monotonic-corpus.md)
python export_el_monotonic.py --max-lemmas 5000 --out-dir /tmp/el_check
                                                  # sanity run, about 1 minute
python export_el_monotonic.py                     # build/hunspell, several minutes
python eval/eval_el_monotonic.py --download       # first run fetches the pinned elspell files
python eval/eval_el_monotonic.py                  # build/el_monotonic/eval.json, about 30 s
python -m pytest tests/test_el_monotonic_list.py -q
```

The first export counts every token of the corpus's training split per
source (`build/el_type_counts.py`, about 40 s with 10 worker processes) into
`build/el_monotonic/type_counts/train/`; the counts are reused until the
corpus manifest's shard digests change. The evaluation counts the dev split
the same way. A run with `--max-lemmas` is refused in `build/hunspell`.
The Wiktionary inputs must match the sha256 pins of `data/hf_manifest.json`
unless `--allow-unpinned` is given (`hf download ciscoriordan/dilemma
data/lookup.db data/mg_pairs.json data/el_participles.json --local-dir .`
fetches them). After writing `build/hunspell`, the exporter prints a note
when the [next-word model](el-monotonic-lm.md) in `build/lm` was built from
another list, since the model takes its vocabulary from the list.

## Output

Under `build/hunspell/`:

| File | Contents |
|---|---|
| `el_monotonic.dic` | The entry count, then `form<TAB>fr:C\|M\|R\|X` lines sorted by UTF-8 bytes; `fr:X mg:generated` for a paradigm form no frequency source has. `ac:spelling` records a preferred accent alternative, `sp:spelling` an alternative with different letters, and `us:label` a reviewed usage restriction. No affix flags. |
| `el_monotonic.aff` | `SET UTF-8`, `LANG el`, `FLAG num`, and `TRY` with the lowercase monotonic letters by their frequency in open text. |
| `el_monotonic.version` | `version`, `commit` (with `-dirty` when tracked files differ from it), `variant: el`, `entries`, bucket counts and edges, route counts, frequency sources, the sha256 of the four outputs and of every input. Any change of input, code or output changes it. |
| `el_monotonic_abbreviations.txt` | Abbreviations with their periods (`π.χ.`, `κ.λπ.`, `σελ.`). |
| `el_monotonic_acronyms.txt` | All-capital acronyms (`ΗΠΑ`, `ΟΗΕ`, `ΣΥΡΙΖΑ`). |
| `el_monotonic.audit.json` | Per route, what was admitted and rejected, with the most frequent examples of each rejection reason. |
| `el_monotonic.evidence.tsv.gz` | Per entry: route, rating, winning source, lemma and the counts in every frequency source. |

Every file is byte-identical across runs with the same inputs and commit.
The outputs are build products and are not committed; the keyboard project
takes the `.dic`, `.aff`, `.version` and the two companion lists. The
[next-word model](el-monotonic-lm.md) takes its vocabulary from the `.dic`.

## Inputs and licenses

| Input | Use | License |
|---|---|---|
| `data/mg_pairs.json`, `data/el_participles.json` (Wiktionary via kaikki.org, pinned on Hugging Face) | Lemmas and their inflected forms | CC BY-SA 4.0 |
| `data/lookup.db`, `el` rows (pinned on Hugging Face) | Forms `mg_pairs.json` lacks (εμείς, συνιστά) | CC BY-SA 4.0 |
| `data/mg_freq.txt` (FrequencyWords, OpenSubtitles2018) | Counts only, for ratings and the corpus-word route | CC BY-SA 4.0 |
| `build/el_corpus`, training split | Counts, spread and context of every token | The open licenses of `data/el_lm_corpora.json` |
| `data/el_monotonic_function_words.json` | Closed-class words that are always entries | Reviewed in this repository |
| `data/el_monotonic_review.json` | Reviewed rejects and exceptions, each with its reason | Reviewed in this repository |
| LibreOffice el_GR dictionary | Evaluation only | MPL 1.1, GPL 2.0 or LGPL 2.1 |
| `data/triantafyllidis_headwords.json` | Evaluation only | CC BY-NC-ND 4.0 |

The list holds words and ratings derived from counts, not corpus sentences.
The source archives used to reproduce the corpus are distributed separately
under their source licenses. The dev split is never read by the exporter.

## Spelling rules

`build/el_lexicon.py` decides which spellings may be entries, on every
route:

- NFC, the monotonic letters of the Greek block only: no Greek Extended
  (polytonic) letters, no Latin look-alikes, no digits.
- Lowercase, or a capital initial for names. All-capital spellings go to the
  acronym list.
- ς only at the end; σ before the elision mark (`σ’`).
- A polysyllable carries exactly one tonos, within the last three syllables
  (loanwords from the lexicon may carry it earlier: `κάμεραμαν`).
  Monosyllables carry none, except `ή`, `πού` and `πώς`.
- Syllables are counted with synizesis: an unaccented ι, υ, ει or οι before
  another vowel is a glide (`μια`, `ποιος`, `σπίτια`, `βοήθεια`), and so is an
  unaccented ι after an accented vowel (`κορόιδο`). So `ποιός` and `μιά` are
  accented monosyllables, while `μία` and `υιός` are two syllables.
- The dialytika only on ι or υ, after a vowel, and only where the two letters
  would otherwise read as a digraph (`προϊόν`, `ταΐζω`; not `τσάϊ`, `πρωϊνό`).
- No letter three times in a row, no double accents. The second accent before
  an enclitic (`όνομά μου`) is added by the keyboard, never listed; the
  [next-word model](el-monotonic-lm.md) keeps the double-accented forms
  so the keyboard can decide it.
- A lowercase spelling with no tonos that synizesis makes a monosyllable
  must be Wiktionary's own (`μια`, `δυο`, `γεια`); otherwise it is an
  accent-stripped alias (`τρια`, `πλοιο`).

## Admission routes

1. **Paradigms.** Every form of a Wiktionary lemma, provided one lowercase
   form of the lemma has at least 3 tokens in one frequency source. A form no
   source has is rated X and marked `mg:generated`. A lemma's capitalized
   forms (names, months) need their own evidence: written with the capital
   inside sentences at least 3 times, in at least 2 documents. A capitalized
   form that is no Wiktionary name, whose lowercase is a word, and that is
   written with its capital in less than a quarter of at least 20
   mid-sentence uses (`Μέσα`, `Αυτού`, `Νέοι`) is left out. `lookup.db`
   forms join only when `mg_pairs.json` lacks them, their lemma is lowercase,
   they start like their lemma, and they are not the unmarked letters of a
   Wiktionary form with a tonos or a dialytika (`πλαι` for `πλάι`, `σοι` for
   `σόι`, which the spelling check reads as monosyllables).
2. **Function words.** The reviewed closed list (`και`, `ο`, `ή`, `πού`), which
   Wiktionary files partly on its Ancient Greek side.
3. **Corpus words.** A lowercase word the lexicon lacks (`συμβαίνουν`) with at
   least 50 FrequencyWords tokens and at least 5 open-corpus tokens in 2
   documents, or, for formal words subtitles rarely have (`επικροτεί`,
   `θεσπιστεί`, forms Wiktionary's tables lack), at least 3 FrequencyWords
   tokens and 20 open-corpus tokens in 10 documents. It must have one tonos
   and be written in lowercase in most of its mid-sentence uses. Without
   subtitle evidence, it needs at least 100 open-corpus tokens in 20
   documents, with at least 5 tokens in 5 documents in each of two
   independent sources. This route rejects Katharevousa markers, obsolete
   infinitives and personal endings, and older case endings when their
   Modern equivalent is already known. It keeps formal participles and uses
   the same spelling, variant, typing-slip and joined-word checks.
   Unless Wiktionary has it (Katharevousa datives such as
   `εν μέρει`, which the paradigm reader drops), it is rejected when a far
   commoner known word has its letters without the dialytika (`προυπόθεση`) or
   with the accent elsewhere (`έιναι`, `ποίος`), or its pronunciation with the
   same stressed syllable (`κάθησε`, `νοιώθεις`, `βαθειά`). A Katharevousa
   ending that sounds like a modern one (`εν γνώσει`, `εν καιρώ`) is kept.
   It is also rejected when a known word one typing slip away (a letter left
   out, added, replaced, or swapped with the next one) is at least 300 times
   as common in both the open corpus and FrequencyWords: misspellings
   (`είνα`, `πρέπε`, `συγκεριμένα`) and Katharevousa endings of common
   words (`έχειν`, `χώραν`), which would hide a slip of the common word
   from correction. The review file accepts the Modern Greek words this
   catches (`ήτον`, `πολύν`, `τρίτιο`). Last, a word that starts with
   `θα`, `να`, `δεν`, `που`, `από`, `είναι`, `πρέπει`, `μπορεί`, `αυτό`,
   `κατά` or `για` and continues with a C entry or a function word of at
   least two letters is two words written without the space (`είναιτο`,
   `πρέπεινα`, `απότι`, `θαήθελα`) and is rejected.
4. **Names.** A capitalized spelling the lexicon lacks, written with its
   capital inside sentences at least 20 times in 3 documents, with a 95%
   Wilson lower bound of the capitalized share of at least 0.9. When its
   lowercase is a word, the lower bound must be at least 0.75 and the
   sources must agree. Each frequency source with at least 20 mid-sentence
   uses of the two spellings votes for the one whose share has a lower
   bound of at least 0.75. A form of a Wiktionary name page, which the
   paradigm reader drops (`Ρώμη` beside `ρώμη`, `Τσίπρα` beside `τσίπρα`),
   needs no more sources voting for the lowercase than for the capital;
   any other spelling needs at least two sources voting for the capital
   and more than vote for the lowercase (`Ένωση` of `Ευρωπαϊκή Ένωση`,
   `Εθνών` of `Ηνωμένων Εθνών`, `Επιτροπή`). So a title or an institution
   one source capitalizes (`Υπουργός`, `Πρόεδρε`, `Βουλευτή` and `Αίθουσα`
   in the Parliament minutes) is no name. A name is also rejected when a
   capitalized spelling of the same letters is far commoner inside
   sentences: 100 times with the accent elsewhere (`Βρετάνια` beside
   `Βρετανία`), 4 times when only the dialytika differs (`Γουϊνέα` beside
   `Γουινέα`).
5. **Elisions.** A spelling with U+2019 seen at least 10 times in 3
   documents. An elision (`απ’`, `είν’`, `κατ’`) must be an attested entry cut
   before its final α, ε, ι, ο, υ, αι or ου, accent kept, with τ, π or κ
   written θ, φ or χ before a word that had a rough breathing (`μεθ’` from
   `μετά`, `υφ’` from `υπό`), and must stand
   before a vowel or, colloquially, before an article or weak pronoun
   (`πάρ’ το`) in most of its uses; that rejects numbering marks (`αβ’`, `ββ’`).
   An aphaeresis (`’ναι`, `’χει`, `’κει`) must be an attested entry without
   its first vowel, with the accent matching what is left, and must follow a
   word that ends in a vowel; that rejects quotation marks folded into
   apostrophes (`’το`, `’και`). An elision without a tonos is dropped when
   an elision of the same letters with one is at least twice as common:
   `ολ’` (passing as an elision of `ολέ`) beside `όλ’`, `πρωτ’` beside
   `πρώτ’`; `αλλ’` beside the rarer `άλλ’` stays.
6. **Reviewed loanwords** of `data/hunspell_mg_reviewed_loanwords.json` with
   at least 50 FrequencyWords tokens.

Last, three cleanups drop entries that would hide a typing slip of a
common word:

- An X entry is dropped when a C or M entry has its letters with another
  accent (`αερόπλανα` beside `αεροπλάνα`): a mistyped accent would
  otherwise pass as a word nobody writes.
- A lowercase R or X entry in fewer than 10 open-corpus documents is
  dropped when a spelling of its letters with another accent is at least
  50 times as common (plus one) in both the open corpus and FrequencyWords
  (`καλήμερα` beside `καλημέρα`, `Γαλλιά` beside `Γαλλία`), and a
  capitalized entry when a capitalized spelling of its letters is at least
  40 times as common inside sentences (`Σαββάτο` beside `Σάββατο`).
  Function words, elisions and reviewed loanwords stay, and so does a form
  of the commoner spelling's own lemma (the colloquial `ακούσαν` beside
  `άκουσαν`) or of a lemma with a C or M form (`τροπών` beside `τρόπων`):
  those are words, not misplaced accents. The 2026-11-07 review removed 537 entries.
- A paradigm form of at most three letters, rated R or X and in fewer
  than 10 documents, is dropped when its letters are a same-sound
  spelling or a neighboring-key slip on the same keyboard row of a word
  at least 300 times as common in both sources (`μάι` for `με`, `ξάι`
  for `και`, `άε` for `σε`). A letter left out or added does not count
  here, since a short word is one letter away from many others. The 2026-11-07 review
  removed 27 entries.

## Frequency ratings

A rare entry with a much commoner accent alternative can still be a real
word: `πινάκες` is also a plural of `πινάκα`, and `κάνονες` of `κάνονας`.
The exporter keeps it and adds `ac:πίνακες` or `ac:κανόνες` when the other
spelling is rated C or M and has at least 50 times its count (plus one)
in both the open corpus and FrequencyWords. All qualifying alternatives
are recorded. Forms of the same lemma and distinct lemmas with their own
C/M form stay unmarked; a cell shared with the common lemma is not
independent evidence. Consumers may offer the alternative, but must not
treat the metadata as proof that the rare word is wrong. Existing readers
that ignore unknown metadata continue to accept it.

Reviewed preferences also cover source-listed variants that frequency alone
cannot distinguish: dialectal `τσαι` beside `τσάι`, rare `γινεί` beside
`γίνει`, and the archaizing spelling `διευκρίνηση` beside standard
`διευκρίνιση`. Each decision in `data/el_monotonic_review.json` records its
sources and reason. The exporter keeps the observed frequency and the real
reading, adding `ac:`, `sp:` and `us:` metadata only when the preferred form
has C/M evidence. The [reference proposal filter](monotonic-keyboard-data.md#consumer-requirements)
keeps these out of routine alternatives unless the typed marks or real context
support them.

A form's rating is the highest of its rates per million words in the
frequency sources: C at 37 and above, M at 3.7, R below, X when no source has
it. These are the edges of the polytonic Modern Greek list. The sources are
FrequencyWords (262.8M words) and each open-corpus source with at least 1M
training words (Parliament, OpenGov comments, Wikipedia, European Parliament,
EELLAK articles, monotonic Wikisource); Tatoeba, Common Voice and the
Wiktionary examples are pooled (932,062 words). FrequencyWords lowercases and
strips apostrophes, so it rates lowercase words only, and a name only when
its lowercase is no word. In the open corpus, a sentence-initial capital is
shared between a word and its capitalized twin in proportion to their
mid-sentence uses. An open-corpus rate counts for C only when the spelling
is in at least 10 of the source's documents, and for M only in at least 3;
otherwise it is held just under that edge, so that one novel repeating a
character's name (`Κατίγκω`) or one article repeating a term does not make
a word common. A source of fewer than 10M words (EELLAK articles,
Wikisource and the pooled small sources) rates C only when a larger source
or FrequencyWords also reaches C; otherwise its rate is held just under
the C edge. A capitalized name additionally needs C evidence from a case-preserving
open source of at least 10M words: case-folded subtitles and a small set of
sentence examples cannot establish that `Τομ` is common in general text.
In the 2026-11-07 export the small-source rule moved 1,411 entries from C to M, mostly words that a few
thousand Tatoeba sentences repeat. An elided word written without its mark (`απ` for `απ’`,
which FrequencyWords counts) is never an entry.

## Companion lists

Abbreviations are the tokens the corpus tokenizer kept with their period
inside sentences, at least 10 times in 3 documents, with a lowercase letter
(all-capital spellings are left out), written with the period more
often than without when they have one period and two or more letters; plus
words a lowercase word follows after a period. Spellings that differ only in
case are counted together and the commonest is listed; another spelling
with a lowercase letter and its own 10 tokens in 3 documents is listed as
well (`π.Χ.`, before Christ, beside `π.χ.`, for example; `υπ.` beside
`Υπ.`), unless it is a listed spelling with a capital first letter (`Παρ.`
at the start of a sentence). A single letter with a tonos or a final
sigma, and a dotted token with a word of three or more letters inside
(`του.Β.`, a missing space), are not abbreviations. Shorter pairs written
without the space between them (`κλπ.Β.`, `βλ.Π.Δ.`) are rejected by name
in `data/el_monotonic_review.json`.
When the commonest spelling is uppercase (`Ε.Ε.`), a lowercase variant with
its own 10 tokens in three documents is still listed (`ε.ε.`). No lowercase
spellings are inferred from uppercase use. Single capital initials remain excluded.

Acronyms are all-capital tokens of two or more letters with at least 20
tokens in 5 documents, no tonos on a capital, a 95% Wilson lower bound of at
least 0.9 for their all-capital share against lowercase and capitalized
spellings of the same letters, and letters that are no entry of the list,
unless the acronym is at least 10 times as common in capitals as the
letters are in lowercase in the open corpus, and at least 10 times as
common, per million words, as the entry is in FrequencyWords (`ΣΥΡΙΖΑ`
beside the adverb `σύριζα`, `ΑΕΙ`, `ΑΠΕ`, `ΧΥΤΑ`).
The FrequencyWords test can also be satisfied by two independent open
sources, each with 20 capital tokens in 5 documents and at least 10 times
as many capitals as other spellings. The 95% Wilson lower bound of at
least 0.9 is measured over the full corpus. Subtitle counts fold case,
so they cannot veto that evidence
(`ΝΑΤΟ` beside the spoken `νάτο`). The audit names the confirming sources.
Reviewed dual readings such as `ΕΚ`/`εκ` and `ΠΑ`/`πα` need not dominate
their lowercase words. Their source-identified acronym entries still require
20 capital tokens in five documents from each of two independent sources.
The review file records the institution's own usage; rejects and the guard
against two words run together still apply. Lowercase words remain entries.

## Recorded export

The follow-up rules in `1139bde` retain all 637,216 accepted spellings from
the `6cb8d11` build, add preferred-spelling metadata, and move 35 capitalized
names from C to M. The dictionary's sha256 is
`a9d955ed10d7301f228065faf028977a18d2c65d755fee7fd476b5dcfd81fe93`. The build reads the present-tense completion
of 9,450 pairs across 1,904 heads, while preserving prior analyses and preferred
lookups. The corpus's documents and train/dev split are unchanged.

| Measure | Value |
|---|---|
| Entries | 637,216 |
| C, M, R, X | 6,220, 41,096, 368,750, 221,150 |
| Attested, generated | 416,066, 221,150 |
| Corpus words | 7,053 |
| Elisions | 189 |
| Function words | 5 |
| Names | 17,859 |
| Paradigm entries | 612,110 |
| Acronyms | 1,734 |
| Compiled dictionary | 35,251,000 bytes |
| Entries with common accent alternatives | 83 |
| Entries with preferred spellings of different letters | 1 |
| Lexicon lemmas read | 187,894 |

The native Swift compiler produced the dictionary above and its reader found
every entry, frequency bucket and generated-form marker. The model reader also
matched every vocabulary id, reserved id and out-of-vocabulary count, and all
18,000 answers across 6,000 dev queries. The query set includes capitalized
contexts and ASCII apostrophes. These checks establish format and reader
agreement; they do not establish how a keyboard ranks corrections.

This export took about six minutes under shared machine load. Runtime depends
on the available CPU and whether corpus type counts are cached.

## Evaluation

`eval/eval_el_monotonic.py` reads 9,300,478 dev word tokens
in 49,310 documents held out from the exporter. A word with
an enclitic's second accent counts as accepted when its host spelling is an entry.

| Measure | List | elspell |
|---|---|---|
| Word tokens accepted as spelled | 98.51% | 97.20% |
| Lowercase tokens accepted as spelled | 99.06% | 98.68% |
| Accepted with accents ignored | 98.79% | 97.46% |
| Capitalized tokens accepted | 95.00% | 89.76% |
| All-capital tokens accepted | 93.95% | 65.52% |

Abbreviations are evaluated separately: 61.55%
of abbreviation tokens are in the companion list.

| Source | Exact coverage |
|---|---|
| Common Voice | 97.99% |
| EELLAK articles | 99.47% |
| European Parliament | 99.46% |
| OpenGov comments | 98.27% |
| Parliament | 99.41% |
| Tatoeba | 99.17% |
| Wikipedia | 97.01% |
| Wikisource | 79.68% |
| Wiktionary examples | 96.30% |

Of 34 reviewed misspellings, the accepted spellings are
`νοιώθω`.
Wiktionary lists the old spelling `νοιώθω` as a form of `νιώθω`.
Moving the tonos of a C or M entry to another vowel produces
120,435 spellings, of which
3,001 (2.49%)
are entries; many are distinct real words such as `πότε` and `ποτέ`.
The list covers 82.05% of
44,396 single-word
Triantafyllidis headwords, an evaluation-only source.

[Monotonic keyboard data](monotonic-keyboard-data.md) measures restoration,
typing slips, the enclitic second accent and keystroke savings on the same data.

## The LibreOffice Greek dictionary

The el_GR dictionary of LibreOffice (elspell 0.9,
828,806 fully expanded words, pinned at commit
`74d8e826359dbf38224fac1b9c02254b7fe0186a`) is evaluation-only. Its
MPL 1.1, GPL 2.0 or LGPL 2.1 license is separate from this word list's
source licenses; it supplies no words to the exporter.

On dev, elspell accepts 7,560
types (29,328 tokens)
that this list rejects. This list accepts
21,823 types
(151,175 tokens) that
elspell rejects. The JSON report includes the most frequent differences and
comparisons by frequency bucket. Neither dictionary is a gold spelling reference.

## Known limitations

- The keyboard must keep `ό,τι` together at its comma, and treat the companion
  abbreviation and acronym lists separately from ordinary dictionary entries.
- Open-only formal words need evidence in two independent sources. A word found
  only in one source remains out even when frequent; subtitle counts cannot
  supply missing evidence for it.
- Wiktionary retains some older spellings (`νοιώθω`, `καμμία`), and the
  corpus filters do not catch every misspelling. The reviewed reject file records
  the exceptions found so far. Older literary and dialect text remains less covered.
- Capitalized institution names can be proposed where a lowercase word is meant.
  A lowercase restoration must apply the case and frequency policy documented in
  [monotonic keyboard data](monotonic-keyboard-data.md#accent-restoration).
- A rare word can share letters with a common one with a different accent.
  `ac:` metadata supports a proposal; it does not invalidate the rare spelling.
- `τοις` and `τω`, used in `τοις εκατό` and `εν τω μεταξύ`, are also plausible
  slips of common words. Correction needs frequency and context checks.
- 221,150 X entries are generated forms unseen in the frequency
  sources. A dictionary hit alone is weak evidence that one was intended.
- Acronyms colliding with a word need strong capitalized evidence. Two independent
  open sources can confirm them even when case-folded subtitle counts cannot.
