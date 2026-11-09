# Monotonic Modern Greek keyboard data

[README](../README.md)

`eval/eval_el_accent_restoration.py` measures the
[word list](el-monotonic-word-list.md) and
[next-word model](el-monotonic-lm.md) as a keyboard uses them: restoring marks,
recognizing typing slips, adding the enclitic second accent and saving keystrokes.
It reads the bundle alone, except for an explicitly labeled comparison that reads
the training text for enclitic evidence.

The current files were built at `6cb8d11` on 2026-11-08:
637,216 dictionary entries (sha256 `ef508ecffe7cf3a2b9696e8de3edea791fd3b81f28b7977146479a427948464f`), with model sha256
`46564a1f26beaa83f90ef569563b3dfeab479df2053aac85553a8e7afcc0fe0e`. The corpus and document-level train/dev split are
unchanged from the prior bundle. The word list and model were rebuilt after the
source-backed present-tense completion.

## Reproduce

Each step reads the previous step's output. Install numpy, pyarrow and
huggingface_hub along with the project's build dependencies. From a clean checkout:

```bash
hf download ciscoriordan/dilemma data/lookup.db data/mg_pairs.json \
    data/el_participles.json --local-dir .
python extract_el_monotonic_lm.py --download
python extract_el_monotonic_lm.py --sanity
python extract_el_monotonic_lm.py
python export_el_monotonic.py
python train_lm.py --lang el
python export_lm.py --lang el
python export_lm.py --lang el --check-dictionary
python eval/eval_el_monotonic.py --download
python eval/eval_el_ngram.py
python eval/eval_el_accent_restoration.py --quick
python eval/eval_el_accent_restoration.py --enclitic-evidence --workers 12
```

The word list checks Wiktionary inputs against `data/hf_manifest.json`.
The original Tatoeba export and Wiktionary usage examples are archived at the
immutable Hub revision in `data/el_lm_corpora.json`. Their original text, document
ids and groups are preserved. The 2026-11-08 clean-checkout corpus rebuild at
`c4dace2` reproduced all 36 shards and all side tables byte for byte. It fetched
the archived inputs and reused verified caches for the other pinned sources.

The model vocabulary depends on the word list: retrain and re-export after every
list change. `--check-dictionary` refuses a model trained from a different list.
The two `.version` files record source commits, input and output hashes, settings
and counts. The evaluation JSON files retain all measurements and examples.

The native Swift compiler and readers passed every entry, frequency bucket,
generated marker, vocabulary id, reserved id and out-of-vocabulary count check.
All 18,000 `suggest`, `score` and completion answers matched the Python reader on
6,000 dev queries, including capitalized contexts and ASCII apostrophes. This
checks data interchange; it does not assert a consumer has implemented every
correction policy below.

## Text

Documents are held out by group. The evaluator splits dev again by a hash of
document id: the tune half selects margins, and the test half measures accuracy.
The current test half has 215,483 sentences
and 4,481,118 typed words. A typed word has Greek letters and
possibly U+2019; capitals, placeholders, numbers and abbreviations stay as written.
The one-letter words `Ο`, `Η` and `Ή` count as typed words.

Pooled results weight sources by their words. Conversational pools Tatoeba,
Common Voice and Wiktionary examples. The corpus spelling is the reference,
including its occasional misspellings and omitted enclitic accents.

## Accent restoration

The evaluator removes tonos and dialytika, keeps case, and restores words from
left to right with `build/el_accent.py`:

1. Match entries by `el_lexicon.bare_key`: lowercase letters without marks,
   with final sigma folded. There are 611,094 keys,
   22,966 with more than one spelling.
2. For a lowercase input, if any matching lowercase entry exists, exclude every
   capitalized candidate. Keep lowercase entries of every frequency rating.
   If only names match, keep only C and M names (numeric ratings 3 and 2).
   R and X names (1 and 0) are excluded. Deliberately capitalized input may
   match names of every rating. Thus the slip `μαι` cannot acquire `μάι` through
   the rare name `Μάι`.
3. With no candidate, leave the word as typed; with one, apply its marks.
4. With several, score them after the previous two restored words. Apply the best
   when its natural-log score leads by at least 1.0. If the typed letters already
   form a word, require the larger word margin, 3.0. The `keep` policy otherwise
   leaves them as typed; `top` chooses the best when the bare input is not a word.
5. Apply the enclitic second-accent rule below after restoration.

`keep_next_word` also scores the next word's best candidate, modeling revision
after another word is typed. The margins remain 1.0 and 3.0; this change adds no
new score ratio. The margin sweep in the JSON uses the tune half only.

Accuracy counts corpus spellings restored correctly. Listed accuracy includes
only words whose corpus spelling is accepted by the list. Damage is a changed
word that was already correct without marks, measured over such listed words.
Left bare and no candidate are shares of all typed words. Percentages:

| Set | Words | Accuracy | Listed accuracy | Damage | Left bare | No candidate |
|---|---|---|---|---|---|---|
| Common Voice | 1,049 | 96.09 | 97.67 | 0.244 | 2.67 | 1.33 |
| EELLAK articles | 22,867 | 98.70 | 98.96 | 0.000 | 1.78 | 0.25 |
| European Parliament | 879,572 | 98.75 | 98.94 | 0.005 | 1.40 | 0.36 |
| OpenGov comments | 867,804 | 97.53 | 98.54 | 0.019 | 1.76 | 0.84 |
| Parliament | 1,790,286 | 98.36 | 98.83 | 0.012 | 1.60 | 0.39 |
| Tatoeba | 6,173 | 97.76 | 98.67 | 0.000 | 1.62 | 0.58 |
| Wikipedia | 873,267 | 96.04 | 98.60 | 0.009 | 1.82 | 2.57 |
| Wikisource | 19,463 | 82.55 | 97.84 | 0.052 | 2.55 | 14.91 |
| Wiktionary examples | 20,637 | 95.28 | 98.38 | 0.029 | 2.09 | 2.95 |
| All, pooled | 4,481,118 | 97.74 | 98.75 | 0.012 | 1.64 | 0.97 |
| Conversational, pooled | 27,859 | 95.86 | 98.42 | 0.031 | 2.01 | 2.37 |

| Policy | Accuracy | Listed accuracy | Damage, % | Damaged words | Left bare, % |
|---|---|---|---|---|---|
| keep | 97.74 | 98.75 | 0.012 | 191 | 1.64 |
| top | 97.93 | 98.94 | 0.012 | 193 | 1.32 |
| keep_next_word | 98.09 | 99.10 | 0.025 | 406 | 0.98 |

Expanding the word list changes which targets count toward listed accuracy;
compare that measure together with overall accuracy and coverage.

Both `μια` and `μία`, and `δυο` and `δύο`, are standard. A context score is not
proof that a user's preferred valid variant is wrong. The proposal list is the
place for ambiguous readings.

## Typing slips

On 3,000 test-half sentences per source, one word in about seven receives a
simulated slip (rate 0.15, seed 5151), then loses its marks. A slip landing on
another entry's letters can be hidden by restoration. Same-sound lookup merges
the spellings of /i/, /o/ and /e/ and single and doubled consonants. It is a
candidate-generation comparison, not an implemented general correction engine.
Percentages are of the slipped words in each row:

| Slip | Words | Another entry | No candidate | Same-sound recovery |
|---|---|---|---|---|
| Doubled consonant | 7,536 | 2.97 | 97.03 | 90.91 |
| ε and αι | 5,803 | 2.64 | 97.36 | 94.33 |
| Spellings of /i/ | 8,929 | 12.42 | 87.58 | 80.00 |
| Neighboring key | 21,047 | 10.03 | 89.97 | 0.29 |
| ο and ω | 6,483 | 28.88 | 71.12 | 66.85 |
| All | 49,798 | 10.98 | 89.02 | 47.92 |

Same-sound lookup also changes 1,285
of 6,193 unslipped words with no candidate.
It needs frequency and context checks. Real words such as `τω` and `τοις` still
absorb slips, even after rare-name candidates are filtered.

## The enclitic second accent

A proparoxytone takes a second tonos before a pronoun: `ο άνθρωπός μου`.
Syllable counting accounts for synizesis; for words with variable pronunciation,
spelling alone cannot decide the accent. The list has no double-accented entries.
The model holds 5,791
attested twins so the keyboard can compare both readings.

The model-evidence rule scores the host and its twin after the two previous
words, the enclitic after each, and the next word where both readings score it.
A reading that leads by 1.0 wins. Otherwise pronouns `μου`, `σου`, `μας`, `σας`
use the syllable rule. Article homographs `του`, `της`, `τους`, `το`, `τα`, `τον`,
`την`, `τις` use it at sentence end or before a possessive signal listed in
`el_accent.POSSESSIVE_SIGNALS` (such as `και`, `να`, `που`, `στο`).
`με`, `σε` and `τη` never receive the second accent under this rule.

The table compares four rules on the whole dev split. `always` means `μου`,
`σου`, `μας`, `σας`; `article` means the article-like enclitics including `τη`;
`preposition` means `με`, `σε`. Precision is the share of added accents present
in the corpus; recall is the share of corpus accents recovered. Training text
first is a separate comparison reading the majority training spelling (at least
five examples), then backing off to the end-of-sentence rule. Percentages:

| Rule | Group | Hosts | Corpus accented | Rule adds | Precision | Recall | Agreement |
|---|---|---|---|---|---|---|---|
| Every proparoxytone | always | 35,702 | 5,477 | 6,423 | 76.66 | 89.90 | 94.25 |
| Every proparoxytone | article | 811,501 | 19,366 | 195,116 | 8.95 | 90.13 | 77.87 |
| Every proparoxytone | preposition | 182,219 | 67 | 63,841 | 0.05 | 46.27 | 64.96 |
| End-of-sentence rule | always | 35,702 | 5,477 | 6,423 | 76.66 | 89.90 | 94.25 |
| End-of-sentence rule | article | 811,501 | 19,366 | 3,834 | 88.55 | 17.53 | 97.98 |
| End-of-sentence rule | preposition | 182,219 | 67 | 21 | 38.10 | 11.94 | 99.96 |
| Training text first | always | 35,702 | 5,477 | 6,343 | 83.35 | 96.53 | 96.51 |
| Training text first | article | 811,501 | 19,366 | 15,400 | 85.17 | 67.73 | 98.95 |
| Training text first | preposition | 182,219 | 67 | 8 | 87.50 | 10.45 | 99.97 |
| Model evidence | always | 35,702 | 5,477 | 5,855 | 83.01 | 88.73 | 95.48 |
| Model evidence | article | 811,501 | 19,366 | 12,788 | 86.47 | 57.10 | 98.76 |
| Model evidence | preposition | 182,219 | 67 | 0 | none added | 0.00 | 99.96 |

Writers often omit a required second accent. Agreement with the corpus therefore
does not establish normative correctness: the model can learn those omissions.
The syllable-only rule also over-accents article constructions such as
`η απόφαση του υπουργού`.

## Keystroke savings

On 3,000 test-half sentences per source, the user types each word without marks
and taps it the moment it appears. A tap costs one keystroke and inserts a space.
Context uses the correct preceding words. Predictions-only uses the bar before
the first letter, completions-only after it, and three completions uses both.
The commit-aware layout reserves slots for the typed text and correction when
space would change the spelling, leaving fewer slots for completion.

| Set | Words | Predictions only, % | Completions only, % | Three completions, % | Commit-aware, % |
|---|---|---|---|---|---|
| Common Voice | 1,049 | 9.7 | 34.5 | 37.5 | 36.1 |
| EELLAK articles | 22,867 | 17.0 | 45.5 | 49.7 | 48.9 |
| European Parliament | 98,768 | 22.5 | 49.0 | 54.3 | 53.6 |
| OpenGov comments | 70,402 | 19.0 | 47.1 | 51.5 | 50.7 |
| Parliament | 53,546 | 22.1 | 48.4 | 53.6 | 52.9 |
| Tatoeba | 6,173 | 15.7 | 39.5 | 44.4 | 43.3 |
| Wikipedia | 57,962 | 15.4 | 40.8 | 45.0 | 44.2 |
| Wikisource | 19,463 | 6.5 | 21.8 | 24.2 | 23.2 |
| Wiktionary examples | 20,637 | 12.0 | 36.4 | 39.7 | 38.8 |
| All, pooled | 350,867 | 18.7 | 44.7 | 49.3 | 48.6 |
| Conversational, pooled | 27,859 | 12.7 | 36.9 | 40.5 | 39.6 |

These savings are an upper bound for immediate suggestion selection, not a
measurement of human typing speed. The evaluation's completion lookup matched
the reader on all 400 sampled queries.

## Consumer requirements

- Preserve valid typed variants; use the larger margin before changing an existing word.
- Apply the lowercase/name policy above before scoring. Frequency filtering names
  must not accidentally filter rare lowercase words.
- Read `ac:spelling` as a proposed common accent alternative. `πινάκες` and
  `κάνονες` also have legitimate rare readings; metadata does not invalidate them.
  A compiler that ignores unknown fields remains compatible but cannot use this
  metadata until it explicitly preserves it.
- Preserve `sp:spelling` preferences with different letters and `us:label` usage
  restrictions too. `el_accent.proposal_options` excludes a marked variant from
  routine proposals when its preferred form is available, while keeping a
  separate typed-as-is choice. Explicitly typed marks or a contextual lead of
  3.0 natural log units retain the variant. Pass actual left-context scores to
  this filter, never context-free unigram scores. Do not filter genuine
  unmarked ambiguities such as `σχεδία`/`σχέδια` by this rule.
- Format uppercase words with Greek uppercase spelling rules: `τσάι` becomes
  `ΤΣΑΪ`, `Μάιος` becomes `ΜΑΪΟΣ`, and `είναι` becomes `ΕΙΝΑΙ`. Preserve the
  disjunctive one-letter `Ή`. `transfer_marks` implements these rules.
- Recognize both cases of dotted abbreviations, including `Ε.Ε.` and `ε.ε.`.
  Acronym membership preserves an all-capital reading such as `ΕΚ`; it does
  not turn lowercase `εκ` into an acronym.
- Look up a double-accented context word as typed when the model knows it, but
  hide every double-accented vocabulary item from predictions and completions.
- Treat `ό,τι` as one word and U+2019 as part of an elided word. Do not restore
  prefixes immediately before a hyphen (`αντι-`, `κατα-`).
- Check typo proposals against frequency and context even when the slipped letters
  are a valid entry. The experiments above do not replace that correction logic.

## Tests and limitations

The five focused test files cover normalization, archived corpus inputs,
word admission and ratings, acronym confirmation, accent-alternative metadata,
model resolution and export, candidate filtering, restoration margins, enclitics
and the evaluation calculations:

```bash
python -m pytest tests/test_el_text.py tests/test_el_corpus.py \
    tests/test_el_monotonic_list.py tests/test_el_ngram.py \
    tests/test_el_accent_restoration.py -q
```

Formal speech and writing dominate the data; casual questions and short informal
messages are less well represented. Wikisource contains older and dialect text
outside the intended Modern spelling coverage. Slips are simulated rather than
drawn from keyboard logs. Corpus spelling is not a manually corrected reference,
particularly for enclitics. Keep these limits separate from reader compatibility.
