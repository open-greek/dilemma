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
python export_el_monotonic.py                     # build/hunspell, about 2.5 minutes
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
| `el_monotonic.dic` | The entry count, then `form<TAB>fr:C\|M\|R\|X` lines sorted by UTF-8 bytes; `fr:X mg:generated` for a paradigm form no frequency source has. No affix flags. |
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

No sentence of the corpus is distributed: the list holds words and ratings
derived from counts. The dev split is never read by the exporter.

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
  an enclitic (`όνομά μου`) is added by the keyboard, never listed.
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
3. **Corpus words.** A lowercase word the lexicon lacks (`ρωτάω`) with at
   least 50 FrequencyWords tokens and at least 5 open-corpus tokens in 2
   documents, or, for formal words subtitles rarely have (`επικροτεί`,
   `θεσπιστεί`, forms Wiktionary's tables lack), at least 3 FrequencyWords
   tokens and 20 open-corpus tokens in 10 documents. It must have one tonos
   and be written in lowercase in most of its mid-sentence uses. Unless Wiktionary has it (Katharevousa datives such as
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
   catches (`ήτον`, `πολύν`, `τρίτιο`).
4. **Names.** A capitalized spelling the lexicon lacks, written with its
   capital inside sentences at least 20 times in 3 documents, with a 95%
   Wilson lower bound of the capitalized share of at least 0.9. When its
   lowercase is a word, it must be a form of a Wiktionary name page, which
   the paradigm reader drops (`Ρώμη` beside `ρώμη`, `Τσίπρα` beside
   `τσίπρα`), with a lower bound of at least 0.75; a title or an
   institution some sources capitalize (`Υπουργός`, `Πρόεδρε`) is no name.
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
   apostrophes (`’το`, `’και`).
6. **Reviewed loanwords** of `data/hunspell_mg_reviewed_loanwords.json` with
   at least 50 FrequencyWords tokens.

Last, an X entry is dropped when a C or M entry has its letters with another
accent (`αερόπλανα` beside `αεροπλάνα`): a mistyped accent would otherwise
pass as a word nobody writes.

## Frequency ratings

A form's rating is the highest of its rates per million words in the
frequency sources: C at 37 and above, M at 3.7, R below, X when no source has
it. These are the edges of the polytonic Modern Greek list. The sources are
FrequencyWords (262.8M words) and each open-corpus source with at least 1M
training words (Parliament, OpenGov comments, Wikipedia, European Parliament,
EELLAK articles, monotonic Wikisource); Tatoeba, Common Voice and the
Wiktionary examples are pooled (0.95M words). FrequencyWords lowercases and
strips apostrophes, so it rates lowercase words only, and a name only when
its lowercase is no word. In the open corpus, a sentence-initial capital is
shared between a word and its capitalized twin in proportion to their
mid-sentence uses. An open-corpus rate counts for C only when the spelling
is in at least 10 of the source's documents, and for M only in at least 3;
otherwise it is held just under that edge, so that one novel repeating a
character's name (`Κατίγκω`) or one article repeating a term does not make
a word common. An elided word written without its mark (`απ` for `απ’`,
which FrequencyWords counts) is never an entry.

## Companion lists

Abbreviations are the tokens the corpus tokenizer kept with their period
inside sentences, at least 10 times in 3 documents, with a lowercase letter
(all-capital dotted acronyms are left out), written with the period more
often than without when they have one period and two or more letters; plus
words a lowercase word follows after a period. Spellings that differ only in
case are counted together and the commonest is listed; another spelling
with a lowercase letter and its own 10 tokens in 3 documents is listed as
well (`π.Χ.`, before Christ, beside `π.χ.`, for example; `υπ.` beside
`Υπ.`), unless it is a listed spelling with a capital first letter (`Παρ.`
at the start of a sentence). A single letter with a tonos or a final
sigma, and a dotted token with a word of three or more letters inside
(`του.Β.`, a missing space), are not abbreviations.

Acronyms are all-capital tokens of two or more letters with at least 20
tokens in 5 documents, no tonos on a capital, a 95% Wilson lower bound of at
least 0.9 for their all-capital share against lowercase and capitalized
spellings of the same letters, and letters that are no entry of the list.

## Recorded export

Export of 2026-11-07, commit 489336b, version 1.3.15 (sha256 of the
`.dic` `abef641c`):

| Measure | Value |
|---|---|
| Entries | 631,696 |
| C, M, R, X | 7,613, 39,353, 365,531, 219,199 |
| Attested, generated | 412,497, 219,199 |
| Paradigm, corpus word, name, elision, function word | 607,247, 6,605, 17,645, 195, 4 |
| Abbreviations, acronyms | 226, 1,691 |
| Compiled size estimate | 34.9 MB (34,930,430 bytes), 55.3 bytes per entry |
| Lexicon read | 187,846 lemmas, 1.10M forms; 53,811 lemmas attested |

The compiled size follows the keyboard's dictionary format (raw and stripped
string pools with offsets, a stripped index, one frequency byte per entry).
For the earlier export at commit f08cf8f (630,463 entries), the keyboard's
Swift dictionary compiler produced a file of exactly the estimated size
(34,881,785 bytes) in about 5 s on an Apple M5 Max, and its reader found
each entry with the bucket of its `fr:` field.
The largest rejections were unattested capitalized forms (246,100), forms
of unattested lemmas (215,678) and unaccented polysyllables (10,675). The
lexicon no longer reads the 698,350 accent-stripped aliases of `lookup.db`
(see route 1), which the spelling check used to reject one by one.

## Evaluation

`eval/eval_el_monotonic.py` on the dev split (9.30M word tokens in 49,310
documents the exporter never read):

| Measure | List | elspell |
|---|---|---|
| Word tokens accepted as spelled | 98.36% | 97.20% |
| Lowercase tokens accepted as spelled | 98.89% | 98.68% |
| Accepted with accents ignored | 98.69% | 97.49% |
| Capitalized tokens accepted | 94.96% | |
| All-capital tokens accepted | 93.94% | |
| Abbreviation tokens in the abbreviation list | 61.6% | |

Per source, exact coverage is 99.3% for Parliament, 99.3% for the European
Parliament, 98.1% for OpenGov comments, 97.0% for Wikipedia and 79.4% for
monotonic Wikisource, whose older literary spelling the list does not
accept. A token with an enclitic's second accent counts as accepted when
the entry without it is one. The commonest misses are numbering letters
(`β`, `γ`, and the capital `Ι` of Roman numerals, which the letter `ι` no
longer covers), the `ό` of `ό,τι` (tokenized apart), the old accented
monosyllable `τού`, `διά`, technical words FrequencyWords lacks
(`φυτοπροστατευτικών`) and mixed-case acronyms (`ΣτΕ`).

False accepts: of 34 reviewed misspellings (unaccented, accented
monosyllables, double accents, Latin look-alikes, wrong accents, wrong or
missing dialytika, old spellings, ASCII apostrophes), the list accepts one,
`νοιώθω`, which Wiktionary lists as a form of `νιώθω`. Of the 119,679
spellings made by moving the tonos of a C or M entry to another vowel, 3,124
(2.6%) are entries, most of them other words (`πότε`, `ποτέ`). Triantafyllidis
headwords: 81.9% of 44,396 single-word headwords are entries; the missing ones
are mostly rare derived adjectives no source attests (`άβαλτος`, `άγλωσσος`).

[Monotonic keyboard data](monotonic-keyboard-data.md) measures the list
together with the next-word model the way a keyboard uses them: accent
restoration, typing slips, the enclitic second accent and keystroke
savings. Its slip simulation found the misspellings the typing-slip rule
of the corpus-word route now rejects.

## The LibreOffice Greek dictionary

The el_GR dictionary of LibreOffice (elspell 0.9, 828,806 fully expanded
words, pinned at commit `74d8e82` of LibreOffice/dictionaries with sha256 in
the evaluation script) is licensed MPL 1.1, GPL 2.0 or LGPL 2.1. GPL and LGPL
terms cannot be met by an app distributed through the App Store, and MPL 1.1
keeps the covered file under MPL, which a list that also carries CC BY-SA
content cannot satisfy. So it is used only to evaluate the list, never to
admit words. On the dev split, it accepts 8,662 token types (37,407 tokens)
that the list rejects, mostly single letters, old accented monosyllables
(`τού`, `τούς`, `μάς`), `διά`, abbreviations written without their period
(`κλπ`, `πχ`) and formal participles (`καταταγέντων`). The list accepts
21,621 types (145,459 tokens) it rejects, mostly acronyms, elisions, names,
recent words (`πάροχοι`, `οικοτόπων`) and old spellings Wiktionary lists
(`καμμία`). It lacks 132 C entries of the list: names (`Τσίπρα`, `Βοστώνη`),
elisions, recent technical words (`ανοιχτότητα`) and dialect or older forms
(`ήτο`, `απάνου`).

## Known limitations

- `ό,τι` is tokenized as `ό` and `τι`; the keyboard must treat `ό,τι` as a
  unit, since `ό` alone is not an entry.
- The corpus-word route needs at least 3 FrequencyWords tokens, so technical
  words seen only in open text (`φυτοπροστατευτικών`, 682 dev tokens) are
  missing unless Wiktionary has their form.
- Wiktionary lists some old spellings as forms (`νοιώθω`, `καμμία`), and they
  pass with their own ratings. Corpus words are also an imperfect filter for
  merged spellings and misspellings; `data/el_monotonic_review.json` rejects
  the ones found (`γιαυτό`, `τιδήποτε`).
- The typing-slip rule also rejects older forms of common words that
  Wiktionary lacks (`ήκουσα`, `είδον`, `ταύτης`), so older literary text
  has more words the list rejects.
- A capitalized spelling whose lowercase is a word is a name only with a
  Wiktionary name page. Names without one stay lowercase (`Ευρωπαϊκή
  Ένωση`, `Ηνωμένων Εθνών`; the model predicts `ένωση` after
  `Ευρωπαϊκή`), as does `Όρος` of `Άγιο Όρος`, which text mostly writes
  as the word `όρος`. A few titles that Parliament minutes capitalize and
  Wiktionary files as names are entries (`Βουλευτή`, `Αίθουσα`).
- `τοις` and `τω` are function words for `τοις εκατό` and `εν τω
  μεταξύ`, which make up most of their uses, but they are also slips of
  `τις`, `της`, `τους` and `το`. In the typing-slip simulation of
  [monotonic keyboard data](monotonic-keyboard-data.md), `το` typed `τω`
  is 64 of 795 o and ω slips; a keyboard can accept the two only where the
  next-word model expects them.
- Monotonic Wikisource is a rating source of its own, so 979 entries, some
  of them older literary forms (`ψυχήν`, `όλην`, `ώραν`, `οπού`), are
  rated C on its counts.
- X entries (219,007) are Wiktionary forms no source has. Rare legitimate
  forms and Wiktionary's dialectal or colloquial variants are among them.
- Acronyms whose letters are an entry (`ΝΑΤΟ` beside `νάτο`) are left out of
  the acronym list.
