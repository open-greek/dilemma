# Changelog

All notable changes to Dilemma are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.3.4] - 2026-09-21

### Fixed
- Reject editorial square brackets and parentheses at GLAUx pair ingestion
  and again in both movable-nu and elision morphology derivation. This keeps
  contaminated corpus forms such as `)λπίζουσι` out of `grc_morph.json` and
  protects morphology exports made from older pair artifacts.

## [1.3.3] - 2026-09-21

### Fixed
- Resolve terminal Wiktionary movable-nu notation (`(ν` / `(ν)`) into both
  conventional spellings while rejecting every other square-bracket or
  parenthesis siglum in expansion forms and citation lemmas. The final overlay
  removes 93,084 contaminated target rows, adds 159,083 clean movable-nu
  spellings, skips 247 contaminated historical-reference rows, and applies the
  same guard to Byzantine gap-fill data and Hunspell export. The rebuilt
  `lookup.db` and `grc_polytonic` files contain no such delimiters, and the
  Hunspell dictionary now loads in regex-based consumers such as `spylls`.

## [1.3.2] - 2026-09-21

### Fixed
- Restore the revision-pinned historical LSJ/Sophocles expansion layer after
  current expansion runs. The recovery derives only forms absent from the
  exact historical Wiktionary base, preserves all current mapping conflicts,
  and guards representative polytonic paradigms in the shipped lookup. On the
  1.3.1 inputs this restores 3,551,231 marked AG mappings while deliberately
  omitting 3,839,827 generated accent-stripped fallback keys
  (`ag_lookup.json` grows from 5,314,819 to 8,866,050 entries). The repaired
  `grc` Hunspell export contains 1,617,436 dictionary entries and 26,521 affix
  rules, up from 1,157,778 and 7,816 in 1.3.1.

## [1.3.1] - 2026-09-20

### Added
- Add a reproducible, headword-only importer for the Mozilla Data Collective
  export of the Triantafyllidis dictionary. The Parquet read projects only the
  `lemma` column, recognizes only audited display conventions, rejects
  ambiguous headings rather than guessing, and records source and validation
  checksums with aggregate audit counts. The Triantafyllidis convention now
  unions that dictionary authority with the existing Modern Greek Wiktionary
  inventory for broader coverage.
- Add a revision- and byte-pinned GlossAPI coverage auditor for nine
  permissively licensed candidate corpora. It reports `guess=False` coverage,
  unknown forms, conservative Unicode/OCR defects, and Modern/Ancient lookup
  conflicts, and can build a balanced Modern Greek frequency experiment
  without modifying `mg_freq.txt`. Historical candidates remain audit-only
  until Open Greek Corpus admits cleaned, identified, and deduplicated text.
- Add a separate optional evaluation for GlossAPI's Greek variety classifier.
  GreekBERT and Transformers remain outside Dilemma's core dependencies, and
  supplied Gutenberg variety labels are not treated as independent gold data.

### Fixed
- Stop ingesting the placeholders a treebank uses for "not a word". GLAUx
  marks an editorial gap, where the manuscript is damaged or unreadable, with
  `relation="GAP"`, postag `z` (unanalyzable) and the gap character in the
  lemma field, leaving the surviving letters in `form`: `Gδου` is what is left
  of the genitive of Hades in Plutarch's Comparatio Cimonis et Luculli after
  its first two letters were lost. `build_glaux_pairs.py` read those as
  ordinary form-and-lemma pairs, and `is_greek` did not stop them because it
  asks whether ANY character is Greek and the surviving letters are, so bare
  `G` was rejected while `Gδου` and `μάχηG` were not. The reader now skips a
  token the annotation itself marks as a gap, which removes 1,868 of the
  624,822 pairs in `glaux_pairs.json`. Separately, the cog standardized
  exports use the CoNLL-U convention of `_` for a field the annotator left
  empty; `is_clean_lemma` accepted it as a lemma string, so 1,108 of the
  18,552 Pedalion pairs carried it, metrical scansion patterns among them.
  `_` is now rejected for every cog export, not just Pedalion. Measured on
  the shipped artifacts before the fix: 1,019 `lookup.db` rows resolved to
  the lemma `_` and nothing else, so `Dilemma().lemmatize("διατείνας")`
  answered `"_"` for a real aorist participle of διατείνω, and `Gδου`
  answered `"G"`. Both fall through to the rule and model layers instead
  after a rebuild.
- Reject the corpus word-tails that entered `lookup.db` as words of their
  own. A corpus tokenizer split one printed word at its elision mark and gave
  the second half the whole word's lemma, so the ending arrived at ingestion
  looking exactly like an elided word: `νοντ᾽` under βαρύνω (the end of
  `βαρύνοντ᾽`), `εντ᾽` under πευκήεις, `λευμ᾽` under γαμήλευμα, `γάσασθ᾽`
  under ἐργάζομαι. Nothing about the pair alone gives one away, so no per-pair
  filter can catch it; what gives it away is the rest of the lemma's paradigm,
  which makes this a whole-table check. `build_lookup_db.py` now runs one
  (`drop_corpus_word_tails`, the rule in `dilemma/elision_tails.py`) over the
  merged table and both per-language tables, after every ingestion path has
  contributed and before any row is written. Every rejection is recorded in
  `data/citation_hygiene_rejections.tsv` with the form it was cut from as its
  evidence.

  Measured against `data/lookup.db` and `data/form_profile.db`: 10,814 of the
  7,097,421 Ancient Greek rows end in an elision mark; 234 of those are also
  the ending of another form of the same lemma (228 once the six single-letter
  stems are set aside); 8 of those 234 have no corpus attestation; and 4 of
  those 8 are cut from their sibling by two or more characters. Those 4 are
  exactly the tails above, and the rule rejects nothing else - not `μ᾽ σ᾽ ν᾽
  κ᾽ τιν᾽ ποτ᾽ φησ᾽ σφ᾽ ᾽ς ᾽ν κατ᾽ δ᾽ ἀλλ᾿ γράψατ᾽ φης᾽ βούλεσθ᾽`, and nothing
  at all on the Modern Greek side.

  The last two conditions carry different work, because Ancient Greek builds
  two word-pairs that look like a word and its tail. The augment makes a
  one-character pair: `γράψατ᾽` is `ἐγράψατ᾽` minus `ἐ` and `φης᾽` is `έφης`
  minus its augment, both real words that no corpus wrote, so only the cut
  length saves them. Reduplication makes a two-character pair that the cut
  length cannot separate: `δοῦσ᾽` is `διδοῦσ᾽` minus `δι`, `μνήσομ᾽` is
  `μεμνήσομ` minus `με`, `θεῖσ᾽` is `τιθεῖσ᾽` minus `τι`. Of the 23 rows cut by
  two or more characters, 4 are the tails and the other 19 are real words held
  only by their corpus attestation; reduplication is the commonest reason one
  lands there, with crasis and an augment before a doubled consonant giving
  the same two-character cut. The stem is looked up with the mark attached and never
  bare, because the same tokenizer error left the bare stem in the corpus it
  annotated (`νοντ` occurs 4 times). The elision mark is recognized in all
  eight spellings a source can use, including U+0313 COMBINING COMMA ABOVE,
  which is how treebank exports write it; U+1FFE GREEK DASIA is excluded
  because it is the rough breathing and elision leaves a smooth one.

  Rebuild impact, computed read-only against the current `data/lookup.db`: 4
  of 9,854,919 rows go, all four keys disappear entirely (each had exactly one
  row), and no surviving key changes lemma. The runtime does not start
  answering "no lemma" for them - only `λευμ᾽` does. `νοντ᾽` becomes μιαίνω
  and `εντ᾽` becomes εἰμί from the elision expander, and `γάσασθ᾽` falls to
  the transformer, whose answer depends on the model version and is not
  recorded here. What the fix removes is the lookup's
  high-confidence assertion that these are words, and with it their place in
  `spell_index.db` and their candidacy for the Ancient Greek Hunspell
  dictionary, which three of the four currently clear the diacritic gate for.

  Two residues remain and want a different rule: `G2α᾽` (lemma `G`) and `G?᾽`
  (lemma `_`) are optical-character-recognition placeholders, not tails. They
  are cut from their siblings by one character, so the cut-length test lets
  them through. They need a test on the characters themselves - a form or
  lemma that is not Greek - which belongs with the other per-pair checks in
  `build_lookup_db._citation_artifact_reason`.
- Stop the Hunspell exporter from dropping every Ancient Greek form whose
  only mark is a spacing elision or breathing character. `has_any_diacritic`
  tested for Unicode combining marks (category Mn), while U+1FBD GREEK
  KORONIS and the spacing breathings are Sk, so `δ᾽`, `κατ᾽`, `μετ᾽` and
  `μηδ᾽` were rejected although `lookup.db` holds them. They passed
  before `build_lookup_db.py` started running `sanitize_form` at ingestion,
  which rewrites a trailing combining psili to the koronis. A spacing mark
  counts only on a form that opens on a Greek consonant, the way an elided
  or aphaeresized word does, because polytonic Greek writes a breathing over
  every word-initial vowel. That keeps out the Milesian numerals a source
  wrote with a koronis in place of the keraia (`ε᾽` five, `α᾽` one, `ο᾽`
  seventy) and the accent-stripped lookup keys of vowel-initial words
  (`ημειβετ᾿` beside `ἠμείβετ᾿`). Stranded acutes, the numeral
  signs U+00B4, U+0384 and U+0375, and quotation-mark apostrophes are still
  not read as marks at all, and unaccented spellings are still rejected.
  Measured against `data/lookup.db`, the Ancient Greek variant gains 45
  (form, lemma) pairs and loses none. Three kinds of residue still get
  through, none separable from a real spelling by any rule on the string:
  a numeral whose letters open on a consonant (`δ᾽` is four as well as the
  elided δέ), the accent-stripped lookup key of an accented elision
  (`ταυτ᾽` beside `ταῦτ᾽`, the same shape as the correct `κατ᾽`), and
  word-tails that entered the lookup as rows of their own (`νοντ᾽` under
  βαρύνω, the end of βαρύνοντ᾽; `μα᾽` under μής, a lemma whose other
  forms are OCR residue). The second belongs to `build_data.py`'s
  `_add_lookup`, whose stripped key drops the accent but keeps the elision
  mark. The third is a corpus tokenization error; `νοντ᾽` is now rejected at
  ingestion by the word-tail filter described above, while `μα᾽` is not,
  because it is the ending of no other form of μής - that lemma's whole
  paradigm is residue, which is a different defect.
- Apply the GLAUx license filter (`build/nc_filter.py`) to the corpus behind
  the next-word prediction language model. `train_lm.iter_glaux_sentences`
  read every GLAUx file, so the n-gram counts and the held-out dev sentences
  carried the 7 NonCommercial and 25 PROIEL-derived works, plus the manual
  sentences of the 40 Gorman-derived works that are the project's held-out
  gold: 841,551 of the 17.0M tokens GLAUx contributes, 4.95%. Excluded works
  are now dropped whole and Gorman-derived works keep only their automatic
  sentences, matching every other GLAUx reader. `--glaux-metadata` says where
  the filter reads GLAUx's metadata.txt, and a run stops rather than proceed
  without it. The shipped `grc_ngram.bin` has not been rebuilt.

## [1.3.0] - 2026-09-16

### Added
- Add a deterministic Dilemma-only benchmark regression gate for the
  committed Classical, Katharevousa, and Demotic gold sets. The gate pins
  fixture hashes and token predictions, reports token-level changes, and
  fails CI on strict or equivalence-adjusted accuracy regressions.
- Add `scripts/release.py` as the single release command. It updates version
  metadata and the changelog, waits for the exact release commit to pass main
  CI, creates the numeric tag, and watches GitHub and PyPI publication.

### Changed
- Make the citation-hygiene audit lookup-source-aware, reporting Ancient and
  Modern Greek residue separately and classifying broad multi-accent results
  as single-token or multiword/punctuated diagnostics.
- Reuse successful main CI when publishing a tag instead of running the full
  artifact-backed suite twice, and update `actions/setup-python` to v7.
- Document that additive voice-aware Modern Greek canonical cells are owned
  by Klisy's canonical producer, while its legacy collapsed cells remain
  backward compatible; Dilemma does not duplicate that downstream exporter.

## [1.2.4] - 2026-09-16

### Fixed
- Reject citation lemmas whose tonal accents are structurally impossible:
  duplicate tonal marks on one vowel or tonal marks attached to a non-vowel.
  Legitimate multi-accent phrases and enclitic-bearing forms remain allowed.
- Restore propagation of Ancient Greek verb tense from Wiktionary table
  headers onto individual morphology pairs, completing work that had remained
  isolated on the stale `grc-verb-tense-tags` branch.

### Changed
- Split structurally malformed tonal residue from the broad multi-accent audit
  bucket and preserve source-file provenance for build-time citation rejects.
- Add installed-wheel smoke tests across every declared Python version
  (3.10-3.14), while retaining the full artifact-backed test run on 3.12;
  modernize the package's SPDX license metadata for current setuptools.

## [1.2.3] - 2026-09-14

### Fixed
- Reject final elision marks and final keraia/prime marks as citation lemmas,
  while preserving explicit numeral and other nonlexical pass-through tokens.
- Rebuild lookup artifacts without final-elision, final-keraia/prime, or
  nonlexical numeral/siglum values in the lemma table.

### Changed
- `scripts/audit_citation_hygiene.py` now separates nonlexical mark-bearing
  tokens from rejected citation-lemma residue, so the final-keraia bucket only
  reports unresolved citation artifacts.

### Tests
- Add a citation-hygiene audit guard for the generated lookup artifact so
  grave, overline, leading-combining, final-elision, and final-keraia citation
  residue cannot silently re-enter CI.

## [1.2.2] - 2026-09-13

### Fixed
- Reject overline abbreviation marks and stranded leading combining marks as
  invalid citation lemmas, matching the existing proof-based grave-accent gate.
- Apply the same citation-form hygiene while rebuilding `lookup.db`, including
  late LBG headword and generated-pair additions, so rebuilt lookup artifacts no
  longer carry grave, overline, or leading-combining citation lemmas.
- Allow grave-to-acute normalization for trusted single-token independent
  lexicon headwords only; unproven graves, including multiword grave phrases,
  remain rejected rather than blindly normalized.

### Changed
- `scripts/audit_citation_hygiene.py` now separates Greek numeral and final
  keraia/prime residue from true final elision marks, and reports
  accepted/rejected citation-status reason counts.
- Refreshed HuggingFace-pinned `lookup.db` and `spell_index.db` artifacts in
  `data/hf_manifest.json`.

### Documentation
- Document `citation_policy="strict_ag"`, `citation_status()`, and the
  `LemmaCandidate.citation` validation note.

## [1.2.1] - 2026-09-13

### Added
- Device-agnostic ONNX execution: `dilemma._ort_providers` auto-selects CUDA
  when `onnxruntime-gpu` is present (an NVIDIA box), else CPU, with a
  `DILEMMA_ORT_PROVIDERS` override. `Tagger.on_gpu` / `Tagger.providers` report
  the REAL execution device, and `make_session()` warns on a silent CPU
  fallback (onnxruntime-gpu missing / CUDA-cuDNN mismatch) - the trap where a
  GPU box runs the tagger on CPU with the card idle. README's "Device and
  throughput planning" section now carries measured numbers: a full-corpus pass
  is CPU + memory-bandwidth bound, not GPU-bound. A 64-core EPYC 7B12 saturates
  the lemmatizer at ~1,100 tok/s around 32 workers (tag-on-CPU 745 tok/s/proc),
  and coupling tag+lemma as one CUDA session per shard is *worse* (361 tok/s,
  VRAM-bound) than the CPU lemmatizer alone; a Threadripper PRO 7995WX
  (96c/192t, DDR5 8-ch) hits ~2,460 tok/s at ~48-61 workers (2.2x). Verdict:
  buy cores + memory bandwidth, not a GPU; use a decoupled pure-CPU scale-out
  (never one GPU tagger session per worker); real throughput drops on
  beam-search-heavy text (lexica/scholia) and a run's wall-clock is floored by
  its largest single work.
- NON-LEXICAL token classifier (`dilemma.nonlexical`), exposed as
  `classify_nonlexical(token)` / `is_lexical(token)` (module level and as
  `Dilemma` methods). It recognizes the editorial/typographic residue that
  dominates OCR'd lexica, scholia, and the Patrologia Graeca - γράφεται
  variant marks (`γρ`), Greek numerals (`κζ'`, `,αφ'`), bracket references
  (`[76]`, `[49-59]`), Latin/citation abbreviations (`fr.`, `Herod.`), lone
  punctuation/sigla, and vowel-less consonant fragments - so a caller can
  tell "not a lexical word" from "failed to lemmatize a real word". Pure
  stdlib, no model. `lemmatize`/`lemmatize_batch` return non-lexical tokens
  unchanged and skip the transformer fallback (which otherwise manufactures
  spurious lemmas); `lemmatize_verbose` returns a single candidate with
  `source="nonlexical"`, the class label in `via`, and `tag="X"`.
  `LemmaCandidate` gains a `tag` field and an `is_lexical` property. The
  classifier is conservative by construction (a Greek word always has a
  vowel; numerals are unaccented with strictly descending place-value tiers),
  so elided monosyllables (`δ᾿`) and numeral-shaped real words (`τε`) stay
  lexical; no movement on `bench_fast.py`.

### Fixed
- Ancient Greek lemmatization no longer emits grave-accented citation lemmas
  from contaminated lookup, model, or POS-table values. Grave values are
  normalized to acute only when the acute form is backed by an independent
  lexicon headword inventory; otherwise that candidate is dropped. The
  validation is applied consistently across single-word, batch, verbose, and
  POS-aware entry points.
- `lemmatize_pos` / `lemmatize_batch_pos` no longer prefer a capitalized
  proper-noun twin over the common-word lemma for a lowercase, non-PROPN
  token. Many common lemmas have a capitalized personification/name twin
  as a separate headword (θυμός vs Θυμός, ἔρις vs Ἔρις, τύχη vs Τύχη),
  surfaced both by the lookup ("+case_alt") and by the POS tables; the
  POS-matching step could return the twin whose capitalization disagreed
  with the input. A capitalization-agreement tiebreak now runs on every
  ranking path: a lowercase non-PROPN form gets the lowercase lemma, a
  PROPN tag still reaches the capitalized twin, and capitalized or
  all-caps input (sentence-initial, titles) keeps either twin reachable
  by POS. It is a re-rank between case twins, never a filter. Measured
  +0.76 lemma-equivalence points on the Persae gold calibration set
  (91.83% -> 92.59%); no movement on `bench_fast.py`.

### Changed
- `python -m dilemma download` (and `dilemma.download()`) now opts in to
  HuggingFace's high-performance transfer automatically: Xet
  (`HF_XET_HIGH_PERFORMANCE=1`) on modern `huggingface_hub`, or the
  `hf_transfer` Rust downloader on older versions when installed. An
  explicit user setting (including `0`) is respected.

## [1.1.0] - 2026-07-03

### Changed
- The Ancient Greek (`grc`) tagger weights were fine-tuned on the Iliad
  composite gold plus a GLAUx mixture. No regression on general Greek
  (GLAUx test strict 93.6% -> 94.1%, UAS 83.6% -> 85.1%, LAS 78.3% ->
  79.9%); large gains on Homeric text. Pinned revision updated; tagger
  sub-version 0.6.0.
- `Tagger.tag` now segments leading/trailing punctuation into standalone
  `PUNCT` tokens ("ἄειδε," -> "ἄειδε" + ","), so the output can contain
  more tokens than `text.split()` (as MG multiword tokens already could).
  Elision/aphaeresis apostrophes stay attached (δ’, ῥ’, ’γώ); runs of one
  character stay one token ("..."). Lemma lookups also improve, since the
  lemmatizer now sees the clean word instead of "word+comma". Tagger
  sub-version 0.7.0.

## [1.0.0] - 2026-06-29

First stable release.

### Added
- Ancient Greek (`grc`/`med`) and Modern Greek (`el`) dependency parsing: the
  taggers now carry a biaffine dependency head, so every token has `head` and
  `deprel` (previously `el` had no shipped parser and `grc`/`med` had none).
- `dilemma.__version__`, resolved from the installed package metadata.

### Changed
- The tagger runtime is now torch-free: importing `dilemma.tagger` pulls in only
  `onnxruntime` + `tokenizers` (+ numpy). `torch`/`transformers` are needed only
  to (re)train and export weights.
- The `[tagger]` extra now installs the runtime plus torch + transformers, so
  `pip install dilemma-nlp[tagger]` followed by `Tagger()` works. For inference
  only, use `[tagger-onnx]`.
- Tagger weights auto-download from HuggingFace when not present locally.

### Removed (breaking)
- `Tagger()` no longer accepts `checkpoint`, `pos_path`, or `dp_path`; `device`
  is accepted but advisory (the ONNX runtime is CPU). Point `DILEMMA_TAGGER_DIR`
  at a directory to use custom weights.
- The legacy dual-BERT joint tagger stack (`TaggerModel`, the joint ONNX model,
  and the gr-nlp-toolkit weight loader) has been removed.

### Licensing
- Openly licensed by default: NonCommercial sources are never ingested. The
  committed NonCommercial PROIEL data was removed; PROIEL is dropped, UD Perseus
  is replaced by the AGDT original (CC BY-SA), and the NonCommercial GLAUx and
  PTA texts are filtered out. See NOTICE for the full per-source list.

[1.3.4]: https://github.com/open-greek/dilemma/releases/tag/1.3.4
[1.3.3]: https://github.com/open-greek/dilemma/releases/tag/1.3.3
[1.3.2]: https://github.com/open-greek/dilemma/releases/tag/1.3.2
[1.3.1]: https://github.com/open-greek/dilemma/releases/tag/1.3.1
[1.3.0]: https://github.com/open-greek/dilemma/releases/tag/1.3.0
[1.2.4]: https://github.com/open-greek/dilemma/releases/tag/1.2.4
[1.2.3]: https://github.com/open-greek/dilemma/releases/tag/1.2.3
[1.2.2]: https://github.com/open-greek/dilemma/releases/tag/1.2.2
[1.2.1]: https://github.com/open-greek/dilemma/releases/tag/1.2.1
[1.2.0]: https://github.com/open-greek/dilemma/releases/tag/1.2.0
[1.1.0]: https://github.com/open-greek/dilemma/releases/tag/1.1.0
[1.0.0]: https://github.com/open-greek/dilemma/releases/tag/1.0.0
