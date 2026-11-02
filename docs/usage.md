# Using Dilemma

[README](../README.md)

- [Basic usage](#basic-usage)
- [Language and convention options](#language-and-convention-options)
- [Conventions](#conventions)
- [Citation-form validation](#citation-form-validation)
- [Unknown words and model inference](#unknown-words-and-model-inference)
- [Nonlexical tokens](#nonlexical-tokens)
- [Batch processing](#batch-processing)
- [Verbose mode](#verbose-mode)
- [POS-aware disambiguation](#pos-aware-disambiguation)
- [Spelling correction](#spelling-correction)
- [Paradigm generation](#paradigm-generation)
- [Elision expansion](#elision-expansion)
- [Corpus attestation](#corpus-attestation)
- [Attested forms and example passages](#attested-forms-and-example-passages)
- [Repairing corrupt lemma annotations](#repairing-corrupt-lemma-annotations)

## Basic usage

```python
from dilemma import Dilemma

d = Dilemma()                                  # all periods (default, AG-priority)
d.lemmatize("ἀνθρώπους")                      # "ἄνθρωπος"
d.lemmatize("ἔγραψα")                         # "γράφω"
d.lemmatize_batch(["λέγουσι", "θεούς"])       # ["λέγω", "θεός"]

# Elision expansion (AG elided forms resolved via Wiktionary lookup)
d.lemmatize("ἀλλ̓")                            # "ἀλλά"
d.lemmatize("ἔφατ̓")                           # "φημί"
d.lemmatize("δ̓")                              # "δέ"
d.lemmatize("κατ̓")                            # "κατά"

# Single period
d_mg = Dilemma(lang="el")                     # MG only
d_mg.lemmatize("τραγουδάει")                  # "τραγουδάω"
d_mg.lemmatize("έγραψε")                      # "γράφω"
d_grc = Dilemma(lang="grc")                   # AG only

# Treebank evaluation mode: resolve articles and personal/interrogative pronouns
d_eval = Dilemma(resolve_articles=True)
d_eval.lemmatize("τῆς")                       # "ὁ" (not "τῆς")
d_eval.lemmatize("μοι")                       # "ἐγώ" (not "μοι")
d_eval.lemmatize("αὐτοῦ")                     # "αὐτός"
d_eval.lemmatize("τι")                        # "τις" (indefinite)
d_eval.lemmatize("τί")                        # "τίς" (interrogative)

# Byzantine text with orthographic normalization
d_byz = Dilemma(normalize=True, period="byzantine")
d_byz.lemmatize("εκκλησια")                   # "ἐκκλησία" (restores accents and breathing)
```

By default, marked Ancient articles retain their own forms, such as
`τῆς` → `τῆς`. Pronoun forms follow the dictionary evidence, including
independent adverb and particle readings. Set `resolve_articles=True` to map
closed-class forms
to canonical lemmas (`ὁ`, `ἐγώ`, `σύ`, `αὐτός`, `τις`, `τίς`), matching
treebank conventions (AGDT, DiGreC). This mode uses accent placement to
distinguish indefinite `τις` from interrogative `τίς`; a final grave is
normalized to acute. Default lookup can retain independent adverb and
particle readings of forms such as `αὐτοῦ` and `τί`.
Marked Ancient articles are recognized in `grc` and `all` modes even when
their lookup entries are absent or conflicting. They keep their own forms
(`ἡ` stays `ἡ`, distinct from `ἤ`); positional graves become acute
(`τὸ` becomes `τό`). Verbose results identify these readings as `article`
with the `DET` tag. Strict citation-headword checks still apply.
The opt-in mode also recognizes monotonic projections of polytonic articles,
such as `τών` from `τῶν`, without removing accents from unrelated words.
The `triantafyllidis` convention auto-enables
article resolution (articles to `ο`, skipping AG pronoun resolution
for forms like `σε`/`με` that are MG prepositions).

## Language and convention options

```python
from dilemma import Dilemma

d = Dilemma()                                  # all periods (default)
d_mg = Dilemma(lang="el")                     # MG only
d_grc = Dilemma(lang="grc")                   # AG only

# LSJ lemma convention
d_lsj = Dilemma(convention="lsj")

# Cunliffe convention
d_cun = Dilemma(convention="cunliffe")

# Triantafyllidis convention (recommended for MG)
d_mg = Dilemma(convention="triantafyllidis")

# Strict AG citation validation for audit/dictionary surfaces
d_audit = Dilemma(lang="grc", citation_policy="strict_ag")
```

## Conventions

Different dictionaries cite the same Greek word differently. LSJ uses
`λέγω` for the εἶπον family; Cunliffe uses `γίγνομαι` rather than
`γίνομαι`; Triantafyllidis uses `είμαι` rather than `εἰμί`. A tool
that outputs `εἰμί` against a gold standard expecting `είμαι` looks
wrong, even though both answers are correct for their respective
conventions.

The `convention` parameter remaps Dilemma's output to whichever
standard you're targeting. The same model can serve an LSJ workflow, a
Cunliffe workflow, and a Triantafyllidis workflow without retraining.

| Convention | Target | Example mappings |
|------------|--------|-----------------|
| `None` (default) | Wiktionary headwords | `εἶπον`→`εἶπον`, `θεούς`→`θεός`, `σπήλαια`→`σπήλαιον` |
| `lsj` | [LSJ](https://github.com/ciscoriordan/lsj9) dictionary | `εἶπον`→`λέγω`, `αἰνῶς`→`αἰνός`, `σπηλαίου`→`σπήλαιον` |
| `cunliffe` | [Cunliffe](https://archive.org/details/lexiconofhomeric0000cunn) Homeric Lexicon | `γίνεται`→`γίγνομαι`, `θέλει`→`ἐθέλω`, `νοῦν`→`νόος` |
| `triantafyllidis` | [Triantafyllidis](http://www.greek-language.gr/greekLang/modern_greek/tools/lexica/triantafyllides/) MG dictionary | `ὁ`→`ο`, `εἰμί`→`είμαι`, `σπήλαιον`→`σπήλαιο`, `εἷς`→`ένας` |

The mapping is built automatically from `data/lemma_equivalences.json`
cross-referenced against the convention's headword list, with explicit
overrides in `data/convention_{name}.json`. The Triantafyllidis inventory uses
only the source dictionary's bare citation headwords and unions them with the
existing Modern Greek Wiktionary inventory for words outside the dictionary;
definitions, examples, pronunciation, grammar, etymology, and source-page
content are neither read nor shipped. Lemma equivalences also
group valid alternative lemmatizations (comparative/positive adjective
forms, active/deponent pairs, spelling variants) so that benchmarks
score them as correct rather than penalizing convention disagreements.

Note: `data/lemma_equivalences.json` covers Ancient Greek equivalences.
For Modern Greek, the lookup merge in `build_data.py` prefers EL
(Greek) Wiktionary lemma forms over EN (English) Wiktionary, since
EL uses the modern contracted forms that reflect actual usage
(`τρώω`, `λέω`) while EN tends toward fuller morphological stems
(`τρώγω`, `λέγω`). For the remaining cases where the two sources
use genuinely different lemmas, downstream consumers like
[Lemma](https://github.com/open-greek/lemma) use a separate MG
equivalences file generated by cross-referencing
`mg_lookup_scored.json` against Wiktionary headwords, with corpus
frequency as a tiebreaker for the canonical form.

```python
# LSJ lemma convention: remap output to LSJ dictionary headwords
d_lsj = Dilemma(convention="lsj")
d_lsj.lemmatize("αἰνῶς")                     # "αἰνός" (adverb -> adjective)
d_lsj.lemmatize("εἶπον")                      # "λέγω" (aorist -> present stem)

# Cunliffe convention: remap to Cunliffe Homeric Lexicon headwords
d_cun = Dilemma(convention="cunliffe")
d_cun.lemmatize("γίνεται")                    # "γίγνομαι" (Homeric form)
d_cun.lemmatize("θέλει")                      # "ἐθέλω" (Homeric form)
d_cun.lemmatize("νοῦν")                       # "νόος" (uncontracted Homeric form)

# Triantafyllidis convention: remap to Modern Greek monotonic forms
d_mg = Dilemma(convention="triantafyllidis")
d_mg.lemmatize("σπήλαια")                     # "σπήλαιο" (not σπήλαιον)
d_mg.lemmatize("Είναι")                       # "είμαι" (not εἰμί)
d_mg.lemmatize("εργαλεία")                    # "εργαλείο" (not ἐργαλεῖον)
d_mg.lemmatize("τα")                          # "ο" (not ὁ)
```

## Citation-form validation

Ancient Greek grave accents are positional in running text, not dictionary
citation accents. By default, Dilemma rejects grave-accented citation lemmas
unless replacing the grave with an acute produces an independently attested
Ancient/Byzantine headword. It also rejects citation candidates with overline
abbreviation marks, stranded combining marks, final elision/keraia marks, or
structurally impossible accent placement (a tonal mark on a non-vowel or two
tonal marks on one vowel). Multiple accents across a phrase or
enclitic-bearing form are not rejected merely for being multiple. These are
treated as bad lemma evidence, not spelling variants to normalize blindly.

For downstream dictionary/headword audits, use `citation_policy="strict_ag"`.
That policy keeps the default artifact checks and additionally requires AG
outputs to be backed by the independent headword inventory:

```python
d = Dilemma(lang="grc", citation_policy="strict_ag")

d.citation_status("οὐρανοστεγὴς", lang="grc")
# {"normalized": "οὐρανοστεγής", "ok": True, "reason": "grave_to_trusted_headword", ...}

d.citation_status("Παντὸς", lang="grc")
# {"normalized": None, "ok": False, "reason": "untrusted_grave", ...}

d.citation_status("α̅", lang="grc")
# {"normalized": None, "ok": False, "reason": "overline", ...}
```

`strict_ag` can reduce coverage because an unverified candidate becomes
`None` rather than a guessed headword. It is intended for audit pipelines and
dictionary lookup surfaces where no lemma is safer than a false citation form.
`python scripts/audit_citation_hygiene.py` audits the built database. Each
lookup rebuild also writes `data/citation_hygiene_rejections.tsv`, whose
entries retain the source input, table, form, lemma, and rejection reason for
every citation value removed by the build-time gate.

One rejection cannot be made per pair. A corpus tokenizer that splits a
printed word at its elision mark and gives the second half the whole word's
lemma leaves an ending that looks exactly like an elided word - `νοντ᾽` under
βαρύνω, which is the end of `βαρύνοντ᾽`. What gives it away is the rest of the
lemma's paradigm, so `build_lookup_db.py` runs one whole-table pass
(`dilemma/elision_tails.py`) over the merged and per-language tables once
every source has contributed. A form is rejected only when all four hold: it
ends in an elision mark, it is the ending of a longer form of the same lemma,
no corpus in `form_profile.db` wrote it, and it is cut from that longer form
by two or more characters. The last two conditions are what keep real words:
an unaugmented form such as `γράψατ᾽` is its augmented sibling minus one
character, and an unreduplicated form such as `δοῦσ᾽` is its reduplicated
sibling minus two but is written by the corpus. Rejections go into the same
`citation_hygiene_rejections.tsv`, carrying the form each tail was cut from as
the evidence, so the build is reversible from the report.

That pass needs `data/form_profile.db`, which is an opt-in download
(`python -m dilemma download --with-attestation`) and is not a step of the
rebuild order above. Rebuild without it and the filter does nothing: the
word-tails are written to `lookup.db` as though they were words, and the same
code over the same corpora produces a different artifact depending on what is
on disk. The build prints a banner when that happens, and the per-table
"Dropped N corpus word-tails" line is absent.

The audit reports lookup-source-aware `grc` and `el` sections because both
languages share the database's deduplicated lemma table. Ancient Greek strict
headword validation is applied only to AG-sourced lemmas; valid Modern Greek
multiword names are not misclassified as untrusted AG headwords. The broad
multi-accent diagnostic is also split into single-token and
multiword/punctuated classes, but neither class is rejected without an
independent structural or headword-backed reason.


## Unknown words and model inference

Construction does not import ONNX Runtime or PyTorch. Backend checks, native
imports, missing-backend warnings, and model loading wait until inference is
requested. This also applies when model files are installed and when the
instance is created on a worker thread.

By default the lemmatizer always answers: when nothing structured resolves a
word it falls back to the transformer, and when even that fails it returns the
input unchanged. For a dictionary-lookup caller a wrong lemma is worse than no
lemma (a guessed headword puts a false definition in front of the reader), and
an echoed input is indistinguishable from a successful identity lemmatization.
All five lemmatization entry points therefore take `guess=False`, which skips
the transformer and returns `None` (or an empty candidate list) instead of a
guess or an echo:

```python
d.lemmatize("μάκαπ")                         # "μάκαπ" (echo - OCR junk, unresolvable)
d.lemmatize("μάκαπ", guess=False)            # None ("I don't know")
d.lemmatize_batch(words, guess=False)        # None at unresolvable positions
d.lemmatize_verbose("μάκαπ", guess=False)    # [] - no identity/model candidates
```

Non-Greek and NON-LEXICAL tokens (see `classify_nonlexical`) still pass through
unchanged: those are classifications, not lemma guesses. For a per-candidate
confidence signal, use `lemmatize_verbose()` - each candidate's `score` (1.0
lookup ... 0.5 model ... 0.0 identity echo) and `source` say how much to trust
it.

## Nonlexical tokens

Real corpora - especially OCR'd lexica, scholia, and the Patrologia Graeca -
are full of tokens that are not words: γράφεται variant marks (`γρ`), Greek
numerals (`κζ'`, `,αφ'`), editorial references (`[76]`, `[49-59]`), Latin or
citation abbreviations (`fr.`, `Herod.`), lone punctuation/sigla, and
vowel-less consonant fragments. These have no lemma. Dilemma classifies them
structurally (pure stdlib, no model) so they can be told apart from a real
word that failed to lemmatize, and are never sent to the transformer fallback
(which would only manufacture a spurious lemma):

```python
from dilemma import Dilemma, classify_nonlexical, is_lexical

classify_nonlexical("γρ")     # -> "variant-mark"
classify_nonlexical("κζ'")    # -> "numeral"
classify_nonlexical("[76]")   # -> "bracket-ref"
classify_nonlexical("fr.")    # -> "abbreviation"
classify_nonlexical("πλ")     # -> "consonant-cluster"
classify_nonlexical("λόγος")  # -> None  (a real word)
is_lexical("λόγος")           # -> True
is_lexical("γρ")              # -> False

d = Dilemma()
d.lemmatize("γρ")             # -> "γρ"  (returned unchanged, not sent to the model)
d.lemmatize_verbose("γρ")     # -> [LemmaCandidate(source="nonlexical", via="variant-mark", tag="X")]
```

A census-style consumer filters non-words out of a failure count with
`d.is_lexical(token)` (or the module-level `is_lexical`). The class labels are
`dilemma.nonlexical.NONLEXICAL_CLASSES`. The classifier is conservative: an
elided monosyllable (`δ᾿`), a word that merely resembles a numeral (`τε`), and
any accented real word stay lexical.

## Batch processing

When processing a large corpus (thousands of words), call `preload()` to
enable query-level caching on the SQLite lookup tables. This avoids
repeated SQLite round trips for forms that appear multiple times:

```python
d = Dilemma()
d.preload()  # enable query cache - ~40x faster for repeated lookups

for word in corpus:
    d.lemmatize_verbose(word)  # second lookup of same form is instant
```

`preload()` is safe to call multiple times (idempotent) and does not
change output - it only affects performance. It caches query results
on demand rather than loading the full 14.5M-entry table into memory.

## Verbose mode

For ambiguous forms, `lemmatize_verbose` returns all candidates with
metadata so downstream tools can disambiguate using context:

```python
from dilemma import Dilemma

d = Dilemma()

# Proper noun vs common noun: Ἔρις (goddess) vs ἔρις (strife)
candidates = d.lemmatize_verbose("ἔριδι")
for c in candidates:
    print(f"{c.lemma:10s} lang={c.lang} proper={c.proper} via={c.via}")
# Ἔρις       lang=grc proper=True  via=exact

# Multiple candidates: common noun + proper-noun capitalization variant
candidates = d.lemmatize_verbose("πόλεμο")
# -> [LemmaCandidate(lemma="πόλεμος", lang="grc", via="exact"),
#     LemmaCandidate(lemma="Πόλεμος", lang="grc", via="exact+case_alt")]

# Elision with multiple valid expansions
candidates = d.lemmatize_verbose("δ̓")
# -> [LemmaCandidate(lemma="δέ", source="elision", via="elision:ε"),
#     LemmaCandidate(lemma="δή", source="elision", via="elision:η"), ...]
```

**Article-agreement disambiguation:** When multiple candidates exist, pass
the preceding word to rank by gender/number agreement with a Greek article:

```python
# Prefer candidates matching masculine article τόν
candidates = d.lemmatize_verbose("λόγου", prev_word="τοῦ")
# -> masculine λόγος ranked before proper Λόγος
```

This only re-ranks candidates, never excludes them. If the preceding word is
not a recognized article form, it has no effect.

Each `LemmaCandidate` has:
- `lemma` - the lemma string
- `lang` - `"el"` (MG, including medieval), `"grc"` (AG), `"med"` (medieval provenance label in output)
- `proper` - `True` if lemma is a proper noun (capitalized headword)
- `source` - `"lookup"`, `"elision"`, `"crasis"`, `"particle_strip"`, `"verb_morphology"`, `"compound"`, `"article"`, `"normalize"`, `"byzantine_norm"`, `"prefix_strip"`, `"model"`, `"identity"`, `"nonlexical"`
- `via` - how it matched: `"exact"`, `"lower"`, `"elision:ε"`, `"suffix_strip"`, `"augment_strip"`, `"θεο+φθόγγος"`, `"+case_alt"`, etc. For a non-lexical token, the non-lexical class label.
- `score` - `1.0` for lookup, `0.5` for model, `0.0` for identity fallback
- `tag` - `"X"` for a non-lexical token, `""` otherwise
- `citation` - citation-form validation note such as `"trusted_headword"`, `"grave_to_trusted_headword"`, `"modern_greek"`, or `"unchanged"`
- `is_lexical` - `False` for a non-lexical token (`source == "nonlexical"`), `True` otherwise

## POS-aware disambiguation

When a POS tagger (e.g. Dilemma's [POS tagger module](tagger.md#pos-tagging-and-dependency-parsing))
provides UPOS tags, `lemmatize_pos` uses POS to disambiguate between
multiple candidates from the regular lookup:

```python
d = Dilemma()
d.lemmatize_pos("αὐτοῦ", "ADV")    # "αὐτοῦ" (adverb: here/there)
d.lemmatize_pos("αὐτοῦ", "PRON")   # "αὐτός" (pronoun: genitive)
d.lemmatize_pos("λευκόν", "NOUN")  # "λευκόν" (noun: white-of-egg)
d.lemmatize_pos("λευκόν", "ADJ")   # "λευκός" (adjective: white)
```

Curated POS entries can select among regular lookup candidates or override
a single candidate when the dictionaries and corpus evidence support a
different lemma for that part of speech.
An explicit verb tag also overrides a single competing reading represented
by lowercase and capitalized candidates, such as `Λάμπε` selecting `λάμπω`
instead of the independent name `Λάμπος`. Untagged name readings remain
available.

Batch POS calls apply the same overrides when all candidates are case or
accent variants of the input. Agreement with a self-map still compares the
available citation spellings. If no POS entry matches, case twins retain
the scalar API's candidate order and capitalization rule; a single accent
variant retains its scalar citation spelling. This fallback checks only
structured readings, so unresolved words keep their batched model inference.

With `convention="triantafyllidis"` or `lang="el"`, POS tags also fix MG
self-map issues for adjective and verb inflections. MG lookup tables
sometimes return self-maps for inflected forms (e.g. `ανθρώπινα` maps to
itself instead of `ανθρώπινος`). When POS is ADJ, the masculine nominative
citation form (-ος, -ής, -ύς) is preferred. When POS is VERB, the
infinitive/1sg form (-ω, -ώ, -μαι) is preferred. Adverbs and nouns keep
their MG self-maps unchanged.

The POS lookup tables (~1.82M AG-only entries, ~1.86M combined) are built
from six sources in priority order: AGDT originals and licensed corpus votes, LSJ9
indeclinables (2.2K adverbs, prepositions, conjunctions, particles,
interjections with unambiguous POS), GLAUx corpus (8.7K entries), MG
Wiktionary, AG Wiktionary, LSJ9 grammar. For polytonic input (breathing
marks, circumflex), the AG-only POS entries are checked first to avoid
MG lemma overrides on Ancient Greek text, mirroring the main lookup's
AG-first logic.

Rebuild the corpus POS table after a source refresh with
`python build/build_treebank_pos_lookup.py --kaikki PATH_TO_KAIKKI`.
Corpus lemma targets must occur in the merged `grc_lexical_heads.json`
inventory; membership in the generated flat lookup alone is insufficient.
The builder requires that inventory before writing outputs and records
rejected targets, source fingerprints and all tied owner votes in
`data/treebank_pos_lookup.audit.json`. Use `--audit-output PATH` to choose
another audit destination. A tied owner does not enter the scalar POS table,
and rejected source claims still protect against conflicting projections.
Explicit Ancient Greek POS calls also reject those documented unsupported
targets if an older POS table or the ordinary lookup offers them as a fallback.
This applies equally to scalar and batch calls; it does not change untyped
`lemmatize()` or Modern Greek behavior. The rejection audit must match the
exact `grc_lexical_heads.json` bytes. Missing or mismatched evidence raises a
data-mismatch error rather than trusting an unbound rejection list. Refresh
both files with `python -m dilemma download --no-tagger` (use `--dir PATH`
for a custom data directory), or rebuild the corpus POS table against the
current merged lexical-head inventory. Validation is cached for each model.
When a POS entry supplies a distinct supported citation spelling for an
inflection, it takes precedence over an accent-stripped lookup collision.
An exact independent lookup entry keeps its own spelling, such as `οὖ`
beside the differently accented adverb `οὐ`.
The audit also retains every source-supported capitalized name and ethnic
adjective owner before the runtime table's ambiguity filter. The dictionary
export combines those owners with exact cased corpus counts and lexical-head
evidence, preserving attested names hidden by a flat lowercase owner.
Capitalization checks recognize both Unicode uppercase and titlecase letters,
including Greek capitals with a written iota such as `ᾍ` in `ᾍδης`, without
changing their spelling or treating lowercase iota-subscript letters as capitals.
A corpus vote that agrees with the bare lookup can still be needed when a
lower-priority POS table, a projected spelling, or a capitalized scalar query
would select another lemma. A case variant already corrected by the effective
POS lookup does not authorize a redundant recovery.
These additional entries require at least two weighted corpus votes, a unique
highest-weight corpus owner, and an independent definition of the exact lemma
and part of speech in the English
or Greek Ancient Greek extracts. Tied corpus owners remain in the source
evidence and cannot authorize an additional scalar override.
Without `--kaikki`, this additional recovery is disabled. Lowercase and
monotonic aliases retain each unambiguous part of speech, and withhold a part
of speech with competing owners. Newly recovered entries also account for
conflicting corpus claims that did not qualify for recovery. Exact spellings
remain available. An exact entry overrides a projected reading only for its
own part of speech; an exact noun does not suppress a projected adjective
with independent adjective evidence. Adding another POS beside an
exact spelling requires independent lexical evidence for that POS, so inherited
corpus-tag errors do not introduce unsupported targets. Fresh Wiktionary definitions
qualify; `--lsj9-headwords PATH` also admits explicit LSJ noun/adjective grammar
when the file matches the fingerprint and head membership in the merged lexical
evidence. Without either source, the existing exact POS scope is retained.

Explicit `DET` tags retain the established article lemma (`ὁ` or `ο`) when
an auxiliary POS table instead returns a noncanonical form. This uses the
same article inventory and accent-sensitive normalization as the optional
article resolver and applies before candidate selection in both scalar and
batch calls; untagged words, pronouns, and other POS tags are unchanged.

## Spelling correction

For unknown or misspelled words, `suggest_spelling` returns candidate
corrections from the lookup table ranked by edit distance:

```python
d = Dilemma()
d.suggest_spelling("θεός")       # [("θεός", 0), ...]  (exact match)
d.suggest_spelling("θεος")       # [("θεος", 0), ("θεός", 0), ...]  (diacritic error = free)
d.suggest_spelling("θδός")       # [("θεός", 1), ...]  (letter-level ED1)
```

The approach works in two layers. First, diacritics are stripped from both
the input and the dictionary, collapsing the 14.5M-entry lookup into 6.97M
unique base forms. ED0/ED1/ED2 matches are found on these stripped forms,
then expanded back to their original polytonic variants and ranked by true
Levenshtein distance. This means accent and breathing errors (wrong accent,
missing breathing mark) are corrected for free, while letter-level errors
(θ/δ, ρ/ν) use standard edit distance. The spell index is built lazily on
first call.

By default, suggestions include all forms in the lookup table (inflected
forms and lemmata from all sources). Two filtering options reduce false
positives when resolving to a specific dictionary:

```python
# Only return known LSJ headwords (strictest - ~119K entries)
d.suggest_spelling("ἀγωνιστήριον", max_distance=1, headwords_only="lsj")

# Only return lemmata/citation forms (less strict - ~700K entries)
d.suggest_spelling("ἀγωνιστήριον", max_distance=1, lemmata_only=True)
```

You can also check headword membership directly:

```python
d.is_headword("θεός")              # True  (LSJ headword)
d.is_headword("θεοί")              # False (inflected form, not a headword)
d.is_headword("θεός", "cunliffe")  # check against Cunliffe headwords
```

## Paradigm generation

`dilemma.paradigm` resolves Ancient Greek inflection cells by source
precedence (jtauber > Morpheus > dilemma_corpus > template), with a
small template fallback for the simplest regular cases. Useful when
a build pipeline needs full inflection paradigms for every lemma but
the kaikki / corpus extracts only carry the cells that happened to
be attested.

```python
from dilemma.paradigm import generate, generate_paradigm, ParadigmSlot

slot = ParadigmSlot.verb_finite(
    voice="active", tense="aorist", mood="indicative",
    person="1", number="sg",
)
generate("γράφω", slot)
# ParadigmForm(form="ἔγραψα", source="jtauber")

generate_paradigm("γράφω", "verb")
# {"active_aorist_indicative_1sg": ParadigmForm(...), ...}
```

Each form is wrapped in `ParadigmForm(form, source)` so callers can
decide whether to trust a cell. Corpus and paradigm-table sources are
safe to ship as-is; template fallbacks are best-effort. Inflection
keys match the canonical shapes:

| Shape | Example |
|-------|---------|
| Verb finite | `active_aorist_indicative_1sg` |
| Verb infinitive | `active_aorist_infinitive` |
| Verb participle | `active_aorist_participle_nom_m_sg` |
| Noun | `genitive_pl` |
| Adjective | `nominative_m_sg` |

The orchestrator reads paradigm JSONs from `$DILEMMA_PARADIGM_DATA`
(set this to a directory containing `jtauber_ag_paradigms.json`,
`ag_verb_paradigms.json`, `ag_noun_paradigms.json`,
`dilemma_ag_verb_paradigms.json`, and/or
`dilemma_ag_noun_paradigms.json`). Missing files yield empty per-source
dicts; the orchestrator falls through to whichever sources are
available and finally to the template fallback.

Templates handle thematic -ω verbs (present-system active only),
vowel-stem 1st / 2nd-declension nouns, and three-termination -ος /
-η / -ον adjectives. Anything that needs accent re-placement or stem
allomorphy (contract -άω / -έω / -όω, athematic μι-verbs, suppletive
verbs, 3rd-decl consonant stems) returns None so callers can leave
the cell empty rather than ship a bogus form. Pass `allow_template=False`
to skip the template path entirely and only return cells with explicit
source attribution.

A CLI entry point fills missing inflection cells across a build's
canonical hash, mapping `{filepath: entry}`:

```sh
python -m dilemma paradigm fill --in pending.json --out filled.json [--with-templates]
```

Each filled cell is recorded in a sidecar `inflections_source.<dialect>.<key>`
field on the entry so downstream consumers can render generator-filled
cells distinctly from corpus-attested ones.

## Elision expansion

Ancient Greek texts frequently elide final vowels before a following
vowel, marking the elision with an apostrophe (U+0313 in polytonic
encoding, U+1FBD/U+02B9/U+02BC/U+1FBF/U+2019/ASCII `'`/U+0060 in other
encodings). All of these resolve identically: the lemma never depends on
which apostrophe codepoint the source text used.

**Exact elided entries first:** the lookup table stores corpus-derived
lemmas for elided forms under one canonical key (U+1FBD), and these win
before any expansion heuristic: `ὅτ᾽` -> `ὅτε` (not the more frequent
homograph ὅτι, which never elides in Homer), `μ᾽` -> `ἐγώ`, `σ᾽` -> `σύ`,
`κ᾽` -> `ἄν`, `θ᾽` -> `τε`, `ποτ᾽` -> `ποτέ`, `αὖτ᾽` -> `αὖτε`. One
residual static-lookup limit: a few Homeric `ὅθ᾽` tokens are the locative
ὅθι rather than temporal ὅτε, which only syntactic context can settle.

Verbose and POS-aware calls prioritize an exact entry over reconstructed
expansions when its original lemma is supported by the merged lexical
inventory for the requested language. This checks the original citation
before convention remapping, including proper names. Unsupported exact
targets retain their former ordering, and every alternative remains
available. A matching POS reading can still select another candidate.

Forms without an exact entry fall through to the expander, which strips
the elision mark and tries each Greek vowel against the lookup table:

| Elided | Expanded | Lemma |
|--------|----------|-------|
| `ἀλλ̓` | `ἀλλά` | `ἀλλά` |
| `δ̓` | `δέ` | `δέ` |
| `τ̓` | `τε` | `τε` |
| `ἐπ̓` | `ἐπί` | `ἐπί` |
| `ἔφατ̓` | `ἔφατο` | `φημί` |
| `κατ̓` | `κατά` | `κατά` |
| `καθ᾿` | `κατά` | `κατά` |
| `ἀφ᾿` | `ἀπό` | `ἀπό` |
| `βάλλ̓` | `βάλλε` | `βάλλω` |

**Consonant de-assimilation:** Before rough breathing, Greek assimilates
voiceless stops to aspirates (τ->θ, π->φ, κ->χ). The elision expander
reverses this: `καθ᾿` tries both `καθ-` and `κατ-`, `ἀφ᾿` tries both
`ἀφ-` and `ἀπ-`, recovering prepositions like κατά and ἀπό.

**Frequency ranking:** When multiple expansions match the lookup table,
candidates are ranked by lemma-keyed corpus attestation (see
`attestation()` below), so common prepositions like κατά always beat
obscure verbs like κάθω. Function words are further prioritized when the
stem matches a known elision pattern, and proper nouns are deprioritized.

Polytonic input automatically restricts expansion to the AG lookup
table, avoiding false matches from MG monotonic forms.

## Corpus attestation

`Dilemma.attestation(lemma)` returns a per-lemma diachronic profile built from
the GLAUx and Diorisis corpora plus the OGA and Pedalion corpora consumed via the
[Open Greek Corpus](https://github.com/open-greek/open-greek-corpus)
standardized annotation export (34M deduped tokens, 304K lemmas): how often,
when, where, and in what genre a lemma is attested. It is keyed by the corpora's
own NFC polytonic lemma annotation and returns `None` for an unattested lemma.

```python
d.attestation("ἀείδω")
# {
#   "total": 1942,                                          # deduped frequency
#   "source_counts": {"glaux": 1869, "diorisis": 1240},    # independent, NOT summed
#   "by_genre":   {"philosophy": 367, "poetry": 441, "history": 263, ...},
#   "by_century": {"-8": 49, "-7": 20, "-6": 40, ...},      # signed; -8 = 8th century BCE
#   "by_dialect": {"Aeolic": 2, "Attic": 249, "Attic/Koine": 662, ...},  # GLAUx only
#   "dominant_pos": "verb",
# }
```

The artifact is `data/lemma_attestation.json`, built by the standalone
`build/build_lemma_attestation.py` pass. The three sources annotate many of
the same texts, so the frequency (`total` and the `by_*` breakdowns) is
deduplicated at the work level by TLG id (each work counted once, with
priority GLAUx > Diorisis > OGA > Pedalion; GLAUx preferred for its dialect and
richer metadata) - a union, not a sum. OGA is all-automatic annotation (a form of
model output), so it is the lowest-priority source; its ~17M tokens cover
795 works - the Suda, the Iliad scholia, Eusebius, patristic homilies -
that no hand-checked treebank annotates, and works the Gorman treebanks
annotate are excluded from it entirely (Gorman is the held-out gold).
Pedalion (manual-gold AGDT trees, PROIEL already dropped by cog) adds ~39K
deduped tokens from works no other source covers (Sextus Empiricus, Menander,
papyri, poetic fragments); its Gorman-provenance rows are skipped and its
constructed example-sentence collections feed only lookup/POS, not attestation.
`source_counts` separately keeps each lemmatizer's full independent count
(overlapping, never summed): agreement is a confidence signal, and a lemma seen
only in a non-preferred source's reading of a shared work has `total` 0 but a
non-empty `source_counts`. It is distinct from the form-keyed, genre-only
`corpus_freq.json`: this one is lemma-keyed and adds the century and dialect
axes. Because the corpora are not
sense-disambiguated there is one profile per lemma string, so `dominant_pos`
disambiguates a noun-vs-verb homograph but not same-POS senses. The file's
`_meta` block documents the full schema, the signed-century scheme, and the
caveats.

## Attested forms and example passages

The lemma profile above has a surface-FORM sibling: which exact inflected forms
actually occur in the corpus, with their own usage distribution and the passages
that attest them. `Dilemma.form_attestation(form)` returns it, or `None` if the
form is unattested:

```python
rec = d.form_attestation("μῆνιν")
# {
#   "total_count": 349, "n_works": 128,
#   "source_counts": {"diorisis": 104, "first1k": 219, "glaux": 128, "pg": 28, "pta": 4},
#   "by_century": {-8: 14, -6: 5, -5: 12, ...},     # usage-by-year axis
#   "by_genre":   {"poetry": 40, "history": 34, "philosophy": 13, ...},
#   "by_century_genre": {-8: {"poetry": 14}, ...},  # joint, for a usage heatmap
#   "dominant_pos": "other",   # untagged raw-text sources (first1k/pta/pg) -> "other"
#   "citations": [                                 # example passages (work + locus)
#     {"author": "Homerus", "title": "Ilias", "source": "glaux",
#      "century": -8, "locus": "1.1", "locus_scheme": "line", "count": 1}, ...]
# }
```

Both directions of the lemmatizer/generator take an `attested_only` flag (off by
default, so nothing changes unless you ask for it):

```python
# Input: only lemmatize forms that actually occur in the corpus (exact NFC,
# elision resolved); an unattested form returns None instead of a guess.
d.lemmatize("μῆνιν", attested_only=True)          # "μῆνις"
d.lemmatize("υπολογιστές", attested_only=True)    # None (modern, unattested)
d.lemmatize_batch(words, attested_only=True)      # None at unattested positions
# attach the citations to each candidate:
d.lemmatize_verbose("μῆνιν", with_attestation=True)

# Output: only generate paradigm cells whose form is attested (grave/case
# folded, since generated forms are citation-style), with optional citations.
from dilemma.paradigm import generate, generate_paradigm, ParadigmSlot
generate_paradigm("λύω", "verb", attested_only=True, with_attestation=True)
```

The data lives in two SQLite artifacts built by the standalone
`build/build_form_attestation.py` pass. It reads the lemmatized GLAUx + Diorisis
treebanks plus the raw-text First1KGreek, PTA, Patrologia Graeca and
byzantine-vernacular corpora (for late-antique, patristic and Byzantine
coverage), deduping each work once by source priority (so `total_count` is a
deduped union while `citations` keep every source's passages). The Patrologia
Graeca and byzantine-vernacular texts are read from the
[Open Greek Corpus](https://github.com/open-greek/open-greek-corpus).

The Patrologia Graeca comes in three tiers of evidence, best first, and a
Migne volume is counted once under the best tier that covers it:

1. The CC-BY [calfa-co](https://github.com/calfa-co/Patrologia-Graeca) text
   as the corpus serves it, with whole-token OCR corrections applied and each
   volume carved into its works. 22 volumes, loci `PG<volume>.<page>`.
2. The corpus's Qwen3.6-27B re-OCR of the public-domain Migne scans, which
   replaced a retired first-generation OCR pass that misread whole sorts
   (`ϖ` as `σ`). 48 further volumes, loci `pg<volume>_<page>.<line>` -
   Eusebius, Athanasius, Basil, both Gregories, Chrysostom, Cyril of
   Alexandria, Theodoret, John of Damascus.
3. The raw first-generation calfa-co dump, read from
   `<corpus>/sources/cgpg/PG*/PG*_text.txt`. It is the only text for 11
   volumes nothing else covers, and for the 22 carved volumes it is read as
   secondary evidence: it credits `source_counts` and citations for the part
   of a volume the carve plan leaves out, without touching the deduplicated
   totals the better text decides.

The raw-text sources (especially the OCR'd Patrologia Graeca) are noisier than
the treebanks, so a form attested only there is lower-confidence;
`source_counts` tells you which corpora attest each form. Both artifacts are opt-in downloads, kept out of the base
`dilemma download`:

```bash
python -m dilemma download --with-attestation   # form_profile.db: gate + distribution
python -m dilemma download --with-citations      # + form_citations.db: example loci
```

`--with-attestation` is enough for the gate and the usage distribution that
powers the usage-by-year graph and heatmap; `--with-citations` adds the example
loci (and implies the profile). Without them, `attested_only` and
`form_attestation` raise a clear "download it" error.

Example loci require matching schema versions, content hashes and source hashes in the two
databases. If the citation file belongs to another build or lacks this
metadata, `form_attestation` still returns the usage profile, with an empty
`citations` list and a `citations_note` asking you to download both files
together. `max_citations=0` reads only the profile.

## Repairing corrupt lemma annotations

Lemma annotations in treebanks and OCR-derived data break in learnable ways:
OCR letter confusions (μάκαπ for μάκαρ), digamma noise (ϝέκταρ for νέκταρ,
πρόσθεϝ for πρόσθεν), diaeresis/gemination variation (ἐυμελίης for ἐϋμμελίης),
and truncation (ἐπιτροχάδη for ἐπιτροχάδην). `repair_lemma()` validates a lemma
against the canonical AG headword inventory (LSJ + Cunliffe, shipped in
`data/`) and repairs a corrupt one - preferring evidence from the surface form
it was assigned to, which is usually correct even when the annotation is not:

```python
d.repair_lemma("μάκαρ")                      # "μάκαρ"  (already canonical)
d.repair_lemma("μάκαπ")                      # "μάκαρ"  (edit-distance repair)
d.repair_lemma("ϝέκταρ")                     # "νέκταρ" (digamma normalization)
d.repair_lemma("Πειραί", form="Πειραΐδης")   # "Πειραΐδης" (form-guided)
d.repair_lemma("διαπέταμαι", form="διέπτατο") # "διαπέτομαι"
d.repair_lemma("ξζψχβγ")                     # None - never echoes junk back
```

The same digamma handling also works on surface forms in plain lemmatization:
`lemmatize("πρόσθεϝ")` -> `πρόσθεν`, `lemmatize("ϝάναξ")` -> `ἄναξ` (a variant
is accepted only if it itself resolves in the lookup, so this never invents a
lemma).
