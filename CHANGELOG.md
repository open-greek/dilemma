# Changelog

All notable changes to Dilemma are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- The dative -ι and -σι no longer elide. They elide only in epic (Smyth 72),
  and rarely there: GLAUx elides them in about 1% of epic tokens before a
  vowel and 0.02% of prose ones, and the elided spelling it does attest is
  nearly always another case's (`ἄνδρ᾽` is `ἄνδρα`, `πάντ᾽` is `πάντα`), so
  a keyboard applying the table turned `παντί` into the accusative. The
  tagged corpora say which forms are datives; the first file that has a form
  decides, since they disagree on some. `τῷδε` and `ἔμοιγε` keep their
  entries, because what they lose is the ε of δε and γε. 1,729 entries go,
  carrying 195,574 corpus occurrences, and the table ends at 15,252.
- A participle's dative plural in -σι takes movable nu like any other
  dative plural (`οὖσιν`, `ἔχουσιν`); the movable-nu list had excluded every
  participle. 2,456 forms join it.
- Long final vowels that the spelling does not show no longer elide: the
  Attic accusative of a noun in -εύς (`βασιλέα`, whose ᾱ is long; Smyth 276),
  the contracted neuter plurals `κρέα`, `κέρα` and `γέρα`, and the rest of the
  deictic -ί paradigm (`τονδί`, `τοιονδί`, `τοιαδί`). Neither does a
  monosyllable that does not end in ε (Smyth 72), so `σά` and the fragments
  `λα`, `πί` and `στι` go; `ῥα`, `κα`, `γα` and `σφι` stay. An accented ε or
  ο before a final ι is a hiatus rather than a diphthong, so the epic
  `βασιλέι` elides again, and a circumflex before the penult hosts no
  enclitic's accent, so the crasis `ὦγαθέ` retracts to `ὦγάθ᾽`. More
  crasis forms and dialect words elide bare (`τἀπί`, `μἀλλά`, `κἀντί`,
  `κοὐδέ`, `σφωέ`, `ὑπά`). That leaves 16,981 entries.
- Words Attic never elides no longer have an elision entry, although their
  final vowel is short: `ὅτι`, `περί`, `πρό`, `ἄχρι` and `μέχρι` (Smyth 72),
  `διό` and `καθά`, which already contain one (`δι᾽ ὅ`, `καθ᾽ ἅ`), and the
  emphatic and deictic -ί, which is long (`οὐχί`, `ὁδί`, `τοδί`). A keyboard
  applying the table rewrote them before any vowel, and `ὅτ᾽` reads as `ὅτε`,
  `δι᾽` as `διά`. GLAUx elides `ὅτι` 5 times in 20,231 before a vowel and
  `οὐχί` never in 462. 32 entries go, whose 14 distinct corpus spellings
  carry 476,588 occurrences.
- The elision table now emits a pair only when the corpora attest the
  spelling the rules of elision give: the final vowel goes, an oxytone throws
  its accent back as an acute unless it is a preposition, conjunction or
  enclitic, and every other mark stays. The corpus candidates used to be
  ranked instead, and the ranking only broke ties among spellings that were
  often all wrong, so a key whose one candidate belonged to another word
  still got it. 32,333 entries fall to 17,030. 13,156 of the keys dropped are
  spellings the orthography rules reject (a grave before the last syllable, a
  missing breathing), whose values were correcting the spelling as well as
  eliding it. Among the 2,147 well-formed ones, a long final α took the
  neuter plural's elision (`αἰτία` -> `αἴτι᾽`, `ἑτέρα` -> `ἕτερ᾽`; 266 keys,
  39,259 occurrences), an oxytone in -ι took another case's accent
  (`γυναικί` -> `γυναῖκ᾽`, the accusative's; 105 keys, 35,291), and the
  deictic -ί, which is long, was elided into the plain pronoun (`τουτί` ->
  `τοῦτ᾽`). The epic `ἐνί` now elides to `ἐν᾽` rather than the numeral's
  `ἕν᾽`, and the dialect prepositions and particles that elide bare are
  listed (`προτί`, `κοτέ`, `ποκά`, `πεδά`, `κἀπί`). `τοτέ`, "at times", is
  an accented adverb rather than an enclitic, so it retracts to `τότ᾽`
  instead of eliding bare.
- The elision table gave `κατὰ` the value `Κατ᾽` and `μετὰ` the value
  `Μετ᾽`, and a keyboard writes the value into the user's text as it stands,
  so the two commonest mid-sentence spellings of those prepositions came out
  capitalized: 217,784 and 88,664 corpus occurrences of those spellings. A form's elision was
  assigned lemma by lemma, and the last lemma to claim the form won whatever
  the ranking said. The corpora carry a capitalized lemma `Κατά` whose one
  elided token opened a sentence, and it came after `κατά`. A form's
  candidates are now pooled across every lemma that claims it and ranked
  once, and each candidate first takes the full form's case, so a lowercase
  `ἑλλάδα` no longer gets `Ἑλλάδ᾽` either. 1,683 entries change case and
  nothing else; 763 of them went from a capital to lowercase.
- An enclitic's accent on the last syllable was read as the word's own, so
  the retraction rule moved the real one: `χεῖρά` took `χείρ᾽`, `εἶπέ` took
  `εἴπ᾽`, and `ἄλλὰ` was taken for the conjunction and lost its accent. The
  acute an enclitic throws onto a proparoxytone or properispomenon leaves with
  the elided vowel, and the word keeps its own accent where it was. 32 entries
  carrying 5,663 occurrences change.
- Three defects in the elision table, all reported by Tonos, which reads it to
  rewrite the user's text and diffs it by hand because its dictionary gate
  never sees the file.
  - A grave oxytone was not retracting. The grave is only the contextual
    spelling of an oxytone, and it is the spelling a word carries
    mid-sentence, which is exactly where elision happens, so `αὐτὸ` took
    `αὐτ᾽` where `αὐτή` correctly took `αὔτ᾽`. 59 entries by Tonos's count,
    carrying 119,486 corpus occurrences: `αὐτὸ`, `πολλὰ`, `αὐτὰ`, `ἐμὲ`.
  - A word elision cannot touch was still given an entry. Elision removes a
    short final vowel, so η and ω are out, a circumflex or an iota subscript
    marks a long vowel, and the second element of a diphthong goes with the
    first. That is 10,957 keys, including `αὐτῷ`, `αὐτῇ`, `δεῖ` (which was
    paired with `δέ᾽`) and `ἤδη`.
  - A pair whose every candidate is junk was emitting the least bad one, so
    `ὅσδε` gave `ὃσδ᾽`, a grave on an elided word. 30 entries, now dropped
    rather than written into someone's text.
  The known-bare class is also stated rather than left to the absence of an
  accented candidate: prepositions, conjunctions and, newly, enclitics, which
  have no accent of their own to throw back (Smyth 183). That separates the
  enclitic `ποτέ` from the interrogative `πότε`. The orthotone `εἰμί` and
  `φημί` are not enclitics here, so `εἰμί` keeps `εἴμ᾽`.
- `tests/test_export_morphology.py` now checks the built table against those
  three invariants, pinning the oxytone one by corpus weight rather than entry
  count, because weight is what separates a rule that stopped firing from the
  tail of the known-bare class.

### Fixed
- An elided oxytone throws its accent back onto the penult as an acute
  (Smyth 174), and the elision table was leaving it bare. The ranking added
  in the previous entry prefers the candidate that keeps the full form's own
  stem marks, and an oxytone's stem carries no mark, so `ἀνδρί` took `ἀνδρ᾽`
  over `ἄνδρ᾽`, `αὐτή` took `αὐτ᾽` over `αὔτ᾽`, and `Διί` took `Δι᾽`, which
  also collides with `δι᾽` for `διά`. The retraction now ranks above the
  stem-marks test, and applies only to oxytones, so every other form still
  keeps its marks where they were. Prepositions and conjunctions lose the
  accent outright instead, which is the other half of the same rule: beyond
  the ten pinned particles that covers `ἀνά`, `ἀμφί`, `περί`, `οὐδέ`, `μηδέ`
  and `τε`, and it corrects `οὐδέ` from `οὔδ᾽` to `οὐδ᾽`.

  Reported by Tonos, which held the table back from its second swap-in rather
  than ship it. Its gate reads only the dictionary, so nothing on either side
  tests this file automatically.

### Fixed
- The elision table in `build/hunspell/grc_morph.json`, which the keyboard
  reads to offer an elided spelling, picked between candidates on casing,
  breathing and whether an accent survived, and broke ties on set iteration
  order. Python randomizes that per process, so the table was not
  reproducible: two builds of the same inputs differed in 316 entries.
  Candidates are now ranked on whether the orthography rules accept the
  spelling at all, then on whether it keeps the full form's own stem marks,
  then the existing profile, then the corpus count, then the spelling itself.
  606 entries change against the table Tonos ships: `αὔτ᾽` becomes `αὐτ᾽`,
  `γέλοι᾽` becomes `γελοῖ᾽`, `διδῶσ᾽` becomes `δίδωσ᾽`, and the structurally
  invalid spellings among them fall from 14 to none. `εἶπε` keeps `εἶπ᾽` and
  `εἰπέ` keeps `εἴπ᾽`, which a corpus count alone gets backwards, because the
  two elided spellings belong to different full forms rather than competing
  for one.

### Changed
- The compatibility baseline is now the dictionary Tonos currently ships,
  rather than the April 0.4.1 export it started from, and
  `data/hunspell_grc_april_compat.json.gz` is renamed
  `hunspell_grc_shipped_compat.json.gz` to stop claiming otherwise. It moves
  forward with each swap-in; the contract it encodes is unchanged, that an
  export must not lose what the keyboard already accepts. Because the gate
  strips its structural classes before pinning, the new baseline grandfathers
  far less junk than the old one: 18 truncated stems against 8,971, and no
  missing-breathing forms at all against 166,226. 1,363,274 required forms.
- A respelling both treebanks annotate independently no longer has to clear
  the raw-count floor. GLAUx and Diorisis lemmatize and tag separately, so
  both landing on one spelling more than once is evidence that counting OCR
  tokens cannot give: the dual `πρώτω` has 40 corpus tokens against 4,006 for
  the dative `πρώτῳ`, which asks for 25 treebank tokens where it has 9. Doric
  `γλώσσᾳ` and the contract `πειρᾷς` were failing the same way, as were Doric
  `δεσπότᾳ` and epic `γένεϊ`.

  The confirmation is bounded the same way the floor is, because treebanks
  mis-tag a common word in proportion to how common it is: it counts only
  when the treebank support is at least one token per thousand of the
  dominant spelling. Without that bound the treebanks' own stray taggings of
  `ὅτι`, `εἶναι` and `τοῦτο` - two or three tokens against 241,498, 140,919
  and 176,293 - let `ὄτι`, `εἴναι` and `τούτο` into the dictionary, which
  Tonos's gate caught. `scripts/audit_hunspell_frequency.py` applies the same
  rule, so the audit and the export agree on what a weak respelling is.

### Added
- The Patrologia Graeca now comes from three tiers of evidence rather than
  one, taking it from 22 Migne volumes and 4,212,976 tokens to 81 volumes and
  15,200,132. `data/form_profile.db` grows from 1,350,920 exact forms to
  1,945,872.
  - The corpus's Qwen3.6-27B re-OCR of the public-domain scans adds 48
    volumes that nothing else covered: Eusebius, Athanasius, Basil, both
    Gregories, Chrysostom, Cyril of Alexandria, Theodoret, John of Damascus,
    George Monachus. Its rows carry `source: "ocr"` and loci
    `pg<volume>_<page>.<line>`, which the reader had no case for, so they
    were skipped although the same pass's output was already ingested for
    In Matthaeum through a hardcoded path. Each of those volumes gets its
    author's century, since `by_century` is a shipped field.
  - The raw first-generation calfa-co dump is read again for the 11 volumes
    neither other tier covers, and for the 22 the corpus serves carved into
    the works of its carve plan it is read as SECONDARY evidence: it credits
    `source_counts` and citations for the part of a volume the carve leaves
    out, without touching the deduplicated totals the better text decides.
    That part of the volume is where 1,504 of the forms the previous build
    lost their only attestation.

### Fixed
- Diorisis beta code read the elision apostrophe after an unaccented alpha,
  iota or upsilon as a macron on that vowel, so `di'` became `δῑ` instead of
  `δι’` for 8,136 tokens. `extract_diorisis_lm.beta_to_nfc` now converts the
  word without the apostrophe and appends the elision mark U+2019, and writes
  the medial sigma a Greek word takes before it (`λέγουσ’`, `πᾶσ’`). Forms
  carrying a spurious quantity mark in `data/form_profile.db` fall from 1,537
  to 445, and the next-word language model is rebuilt from the corrected text.
  `build/build_diorisis_pairs.py` separately dropped the trailing `)` that
  Diorisis writes for the same elision, which the converter turned into a
  breathing (`δἰ`).
- The open-greek-corpus repository now stores the Patrologia Graeca one file
  per work, with a locus of the form `PG<volume>.<page>`, rather than one file
  per volume, and stores In Matthaeum under its author and work slug rather
  than its TLG identifier. `build/build_form_attestation.py` looked only for
  the old layout, so with the raw per-volume OCR clone no longer on disk it
  ingested no Patrologia Graeca at all. It now groups the per-work rows back
  into their Migne volumes, recovering 4,212,976 tokens and In Matthaeum's
  212,906 citations.
- GLAUx records a dialect for each work but `build/build_glaux_pairs.py`
  dropped it, so Epic, Ionic, Doric and Aeolic forms competed for the Attic
  cells of a verb paradigm. A form attested only in works of one of those
  dialects now carries that dialect tag, which routes it to that dialect's
  paradigm slice. Works marked Attic, Attic/Koine or Koine share the default
  slice.

### Changed
- `build/build_grc_verb_paradigms.py` picks the cell forms of
  `data/ag_verb_paradigms.json` on more evidence, which corrects 64 of the 88
  cells the textbook review in `data/hunspell_grc_textbook_review.json` had
  recorded as wrong:
  - A Wiktionary table row that reaches kaikki with its person and number
    collapsed onto the first cell is dropped. 637 of 918 fully tagged cells
    across 117 verbs were affected: τιμάω's whole present active indicative
    arrived tagged first-person singular, and on corpus frequency `τιμάς`
    (far commoner as the accusative plural of τιμή) then took the cell from
    `τιμῶ`.
  - A subjunctive cell rejects a form whose ending belongs to the
    corresponding indicative, since the subjunctive lengthens the thematic
    vowel. GLAUx labels the future indicative `κολακεύσεις` an aorist
    subjunctive, and on frequency it beat the real `κολακεύσῃς`.
  - The pass that takes a cell's accent from Wiktionary now reads which cell
    Wiktionary puts each spelling in, so it keeps an accent that is the only
    thing telling two cells apart (`λύσαι` is λύω's aorist optative
    third-person singular, `λῦσαι` its aorist middle imperative) while still
    correcting a spelling both cells share (`βαίνον` to `βαῖνον`).
  - A form is ranked on its corpus attestation, an augment in a past
    indicative cell, and whether Wiktionary lists it for the verb; forms of
    εἰμί left behind by a periphrastic cell are dropped; and the filter that
    kept only polytonic spellings no longer discards an Attic form whose
    rival happens to carry a circumflex (`δίδως` against Ionic `διδοῖς`).
- The textbook review withdraws its eight entries against the ει stem of the
  perfect of τίθημι. `τέθεικα` is the Attic perfect (Smyth 777, LSJ s.v.), and
  `data/form_profile.db` attests `τεθείκασι` 21 times against none at all for
  the `τεθήκασι` the review proposed. The `withdrawn` block records the
  evidence.

## [1.3.6] - 2026-09-22

### Fixed
- Replace the 1.3.5 Hunspell acute-only admission boundary (the 76,153-form
  language-model vocabulary, whose minimum count was 20) with the full
  accent-preserving `form_profile.db`. Sparse attested forms and complete
  textbook paradigms now survive, including `λύω`, `ποιέω`, epic/dialect
  forms, New Testament and Septuagint vocabulary, and future passives. The top
  2,000 LSJ9 headwords are normalized out of dictionary display notation and
  retained as citation forms.
- Emit the acute citation/before-pause twin of every retained contextual-grave
  form. Productive enclitic second accents are retained when removing the
  final acute yields an independently attested proparoxytone or
  properispomenon host (`θάλασσάν`, `δῶρόν`); a paroxytone host never takes
  one, so `λύκοί` and `ὕπνῴ` are rejected.
- Reject punctuation-bearing entries, internal breathings left by bad preverb
  joins, impossible accent placement including a grave before the final
  syllable (`καὶτοὺς`) or on an elided word (`γὰρ᾽`), malformed sigma, and
  known high-impact accent errors.
  New post-April forms also require exact attestation or membership in a
  pinned citation, textbook, grammar, or productive second-accent class,
  preventing orthographically plausible generator noise from riding in on a
  lemma's frequency. Structural checks run over the complete expanded
  artifact, not only the LM vocabulary.
- Reject weak new respellings of common words. A new spelling that differs
  only in accent, breathing, diaeresis, or iota subscript from a well-formed
  spelling with at least 1,000 corpus tokens needs GLAUx or Diorisis support:
  25 tokens below a 1% share of the common spelling's count, 5 below 5%.
  This removes OCR copies such as `ἑγώ`, `ταΐς`, `ἐκεί`, and `ὄσα` (`ταΐς` had
  outranked `ταῖς` as Tonos's correction for `ταις`) while keeping Doric
  `τᾷ` and `τῶ`.
- Accept valid elided forms the April dictionary lacked, such as `δι᾽`,
  `παρ᾽`, `γ᾽`, and `θ᾽`, and the complete consonant-final words `ἔκ` and
  `παρέκ`.
- Pin the 971 reviewed forms of the language-model head, so the corpus
  evidence filters cannot drop a form the release audit requires, such as the
  polytonic Modern article `τή`, which the Ancient Greek corpora behind
  `form_profile.db` barely attest. In the head fixture, `κα-` is no longer a
  reviewed nonword (its spelling `κα` is the Doric particle), and `δῑ`, a
  Diorisis Beta Code artifact of elided `δι᾽`, is.
- Keep valid spellings the structural rules would otherwise misread, all of
  which the April dictionary accepted: accentless dialect proclitics and
  enclitics (`ἁ`, `αἰ`, `εἰν`, `ἐντι`), crasis of `καί` or the article with a
  proclitic (`κἀν`, `κοὐκ`, `χὠ`, `τἀν`), crasis with a later coronis or with
  vocative `ὦ` (`ἐγᾦμαι`, `μέντἄν`, `καλοκἀγαθία`, `ὦνθρωπε`), fused
  enclitics that leave the host's accent in place (`οὗτινος`, `ᾧτινι`,
  `τοῖσιδε`), iota adscript after a long vowel (`ζῶια`, `τῆιδε`), Ionic `ωυ`
  crasis (`τωὐτό`), `πῃ`, Homeric `ὑπέκ` and `διέκ`, and indeclinable
  Septuagint loanwords (`χερουβίμ`, `ἐφούδ`, `σαβαώθ`).
- Stop giving textbook cells and closed-list forms the common frequency
  bucket: only curated forms, top citation headwords, and the language-model
  head are pinned to `fr:C`. Reviewed April forms are no longer added a
  second time under a placeholder lemma, which had written 949 words twice
  with conflicting buckets.
- Treat a spelling as attested when any single source attests it.
  `form_profile.db` deduplicates totals by work, so 199,082 spellings found
  only in a lower-priority copy of a work had a total of 0 and were dropped as
  unattested.
- Generalize same-lemma bare-elision detection while limiting the mechanism to
  suspicious final consonants, so real inflections ending in `ν`, `ρ`, `ς`,
  `ξ`, or `ψ` are not mistaken for fallbacks. Preserve the reviewed Homeric
  apocope forms, `οὐκ`/`οὐχ`, and the complete unaccented enclitic lists,
  including `εἰμί` forms; keep `του` and `τῳ` rejected for Tonos correction.

### Changed
- CI now downloads `data/form_profile.db` (tracked in `scripts/hf_data.py`)
  so it can build and audit the grc Hunspell export, and the test job's time
  limit is 45 minutes.

### Added
- Add a content-pinned whole-artifact compatibility fixture: the dictionary
  Tonos shipped in April (the Dilemma 0.4.1 export as the keyboard compiled
  it) minus the forms Tonos's candidate gate classes as junk. The
  release audit requires all 1,179,659 forms, every top-2,000 LSJ9 headword
  except three two-word phrases and one extraction artifact, all 2,170 pinned
  textbook paradigm forms, complete grave/acute twins, and zero synthetic
  flagged bases, new weak respellings, or structurally invalid entries outside
  the reviewed April surface (984 of whose forms fail Dilemma's structural
  rules and are left for review downstream).
- Pin the textbook paradigms of `λύω`, `παιδεύω`, `τίθημι`, `δίδωμι`,
  `ἵστημι`, `τιμάω`, `ποιέω`, and `δηλόω` from the generated paradigm data
  after a recorded review (`data/hunspell_grc_textbook_review.json`) that
  removes 95 wrong cells (`λύ` for `λύεις`, `λύσαν` for `λῦσαν`, Doric and
  Ionic forms in Attic slots) and adds standard cells the generator lacks
  (`δίδωσι`, `τίθησι`, `ἔδωκα`, `ποιεῖσθαι`), each corroborated by
  `lookup.db` or corpus attestation. Vowel-length marks are removed.
- Add exact-form regression gates for the New Testament (Perseus), the
  Septuagint (First1KGreek), Iliad 1, Herodotus 1, and a Katharevousa sample.
  Every source file is pinned by SHA-256, and a candidate may not exceed the
  April `.dic`/`.aff` pair's rejection count on either tokens or types. The
  corpora are in `form_profile.db`, so the gate guards against regression; it
  does not measure coverage of unseen text.

## [1.3.5] - 2026-09-21

### Fixed
- Restore common Ancient Greek spellings in the `grc` Hunspell export. The
  selector admits language-shared rows only for lemmas with `grc` evidence,
  uses a revision-pinned accent-preserving full-LM map for acute-only forms,
  rejects vowel- or rho-initial spellings without a breathing, and preserves a
  reviewed closed list of correctly unaccented enclitics. Bare lemmatizer
  fallbacks for elision remain excluded. The resulting export has 1,284,405
  entries and 105 zero-strip suffix rules.
- Stop emitting truncated common prefixes as flagged Hunspell entries. Affix
  compression now occurs only when the base is itself a paradigm form; all
  other paradigms are inlined, requiring neither `NEEDAFFIX` nor nonzero-strip
  support from Tonos.
- Restore high-frequency valid forms that Dilemma resolves through grammar or
  POS authorities but which lookup source collisions hid from the exporter,
  including `τ᾽`, `μεθ᾽`, `δῑ`, `εἶνε`, `μαῦρον`, `μαῦροι`, and
  `ἑκατέρως`. `ἀνάμεσα` is retained through exact LM attestation, while
  `του`, `τῳ`, monotonic `ότι`/`αυτός`, and unrelated Modern forms remain
  rejected.
- Normalize elision marks through one shared corpus-frequency key in the
  GLAUx, Diorisis, Patristic Text Archive, and OGC rollup builders, and again
  at merge time. Diorisis no longer deletes the apostrophe and decomposed TEI
  psili is preserved before accents are stripped, preventing parallel `δ` and
  `δ’` keys. The rebuilt OGC rollup is pinned to commit `f4062a50` and
  public-lexicon SHA-256 `02d6de71...`; merge provenance retains both values.
- Generalize the Hunspell bare-elision rejection beyond a finite stem list.
  A bare spelling with a more frequent marked counterpart is excluded unless
  DGE or Cunliffe independently establishes it as a headword, preserving
  genuine collisions such as `ἄν`. Preserve the closed set of Homeric
  shortened prepositions (`κὰτ`/`κάτ` through `κὰγ`/`κάγ`, `ἂμ`/`ἄμ`).

### Added
- Add a revision-pinned Hunspell frequency auditor and CI regression fixture.
  It loads the expanded dictionary with `spylls`, folds contextual grave and
  final elision marks, and checks the top 1,000 Greek-bearing LM tokens:
  971 required forms accepted and 29 reviewed nonwords rejected, representing
  16,543,630 tokens. It also rejects synthetic flagged bases and emitted forms
  without required initial breathings. JSON fixture generation hard-fails on
  sanity runs; format-v2 LM binaries supply exact vocabulary and counts. An
  optional full-vocabulary comparison produces a reviewable LM-weighted delta
  against a previously shipped dictionary. The current candidate gains 4,510
  full-LM forms (1,232,520 tokens) and loses 213 (73,046 tokens) against the
  April Tonos artifact; the newly removed forms are reviewed elision fragments.

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

[1.3.6]: https://github.com/open-greek/dilemma/releases/tag/1.3.6
[1.3.5]: https://github.com/open-greek/dilemma/releases/tag/1.3.5
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
