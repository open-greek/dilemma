# Sources and lookup construction

[README](../README.md)

- [Sources and scale](#sources-and-scale)
- [HuggingFace Hub dataset](#huggingface-hub-dataset)
- [Confidence tiers](#confidence-tiers)
- [Quality controls](#quality-controls)
- [Lookup construction](#lookup-construction)
- [Lexical readings and source validation](#lexical-readings-and-source-validation)

## Sources and scale

The counts below describe recorded builds and overlap across sources. Use
`data/hf_manifest.json` to identify the artifacts used by a particular commit.

| Source | Forms | Notes |
|--------|------:|-------|
| EN + EL Wiktionary (MG) | 2.8M | From kaikki.org dumps |
| EN + EL Wiktionary (AG) | 2.4M | From kaikki.org dumps |
| EL Wiktionary (Medieval) | 6.9K | From kaikki.org dumps |
| LSJ + Sophocles expansion | Millions of AG additions | Current Wiktionary-Lua expansion plus marked, revision-pinned historical recovery |
| [Perseus / AGDT](https://github.com/PerseusDL/treebank_data) (CC BY-SA 3.0 US) | 81K | The 33 Greek AGDT works (Sophocles, Aeschylus, Homer, Hesiod, Herodotus, Thucydides, Plutarch, Polybius, Athenaeus); the original, not the NC UD release |
| GLAUx corpus | 557K | 17M tokens, 98.8% accuracy ([Keersmaekers 2021](https://github.com/alekkeersmaekers/glaux)) |
| Diorisis corpus | 76K new | 10M tokens, 91.4% accuracy ([Vatri & McGillivray 2018](https://figshare.com/articles/dataset/The_Diorisis_Ancient_Greek_Corpus/6187256)) |
| HNC Golden Corpus | 1K new | 88K-token gold MG corpus ([CLARIN:EL](https://inventory.clarin.gr/corpus/870), [openUnderPSI](https://www.clarin.eu/content/licenses-and-clarin-categories)) |
| DGE headwords | 52K | Headword filter coverage from Diccionario Griego-Espanol |
| LGPN names | 44K | Proper noun coverage from Lexicon of Greek Personal Names |
| Perseus Digital Library headwords | 176K | Headword filter from L&S, Pape, Bailly, etc. |
| **Total lookup** | **14.5M** | Deduped combined lookup rows (source rows above overlap) |

All Wiktionary data is extracted automatically from
[kaikki.org](https://kaikki.org/) JSONL dumps. LSJ and Sophocles
expansions use wikitextprocessor to run Wiktionary's grc-decl and grc-conj
Lua modules on headwords extracted from lexicon XML/TEI files.

The [GLAUx corpus](https://github.com/alekkeersmaekers/glaux) provides
the largest single source of new form-lemma pairs outside Wiktionary.
GLAUx is the primary corpus source due to its 98.8% lemma accuracy.
The [Diorisis corpus](https://figshare.com/articles/dataset/The_Diorisis_Ancient_Greek_Corpus/6187256)
(Vatri & McGillivray, 2018; 10M tokens, Homer - 5th century CE) is used
as a secondary source: its 456K form-lemma pairs add 76K new entries
not found in GLAUx. Because Diorisis has lower lemma accuracy (91.4%),
its pairs are only added when they don't conflict with existing entries
from Wiktionary, LSJ, or GLAUx. For form-frequency data, GLAUx and
Diorisis are merged with the Patristic Text Archive and the Open Greek
Corpus open-text rollup into the 94.0M-token `corpus_freq.json` (see
[Form-frequency corpora](#form-frequency-corpora-and-coverage-diagnostics)
below).

We chose not to integrate one other large corpus:
The lookup builder also reads PTNK, TAGNT, and Pedalion exports from the
Open Greek Corpus. These sources fill missing entries without replacing
existing dictionary readings. Pedalion contributes corpus POS evidence and
lemma attestations, with Gorman-provenance and constructed examples excluded
from attestation. OGA contributes lemma attestations through its standardized
export; it is not a standalone lookup source. See [NOTICE](../NOTICE) for the
source licenses and exclusions.

Each form is indexed under its original, monotonic, and accent-stripped
variants for fuzzy matching.

### Form-frequency corpora and coverage diagnostics

`data/corpus_freq.json` is the merged form-frequency table consumed by
`rank_forms.py` and `vesuvius.py`. It unions four sources (per-form
counts summed, genre vectors element-wise added):

| Source | Tokens | License | Genre breakdown |
|---|---:|---|---|
| GLAUx | 16.2M | CC BY-SA | yes (10 genres, lemma-aware) |
| Diorisis | 10.2M | CC BY 4.0 | yes (10 genres, lemma-aware) |
| Patristic Text Archive | 2.5M | Per-text; see NOTICE | bucket = religion |
| [Open Greek Corpus](https://github.com/open-greek/open-greek-corpus) rollup | 65.1M | Source-specific; see NOTICE | bucket = other |
| **Total** | **94.0M tokens / 1.99M forms** | | |

The Open Greek Corpus rollup (its `public_lexicon.tsv`) covers First1KGreek,
the corrected Patrologia Graeca OCR, Perseus canonical-greekLit, byzantium.gr
historians and the Byzantine vernacular corpus, with OCR corrections and
license filtering already applied. It replaces the three uncorrected per-repo
builders (PG, First1KGreek, canonical-greekLit) that Dilemma used before.

Build:

```
python build/build_glaux_freq.py              # if not already built
python build/build_diorisis_freq.py
python build/build_pta_freq.py                # needs an Open Greek Corpus clone (sources/pta)
python build/build_cog_public_freq.py /path/to/pinned-ogc/data/public_lexicon.tsv
python build/merge_corpus_freq.py             # writes data/corpus_freq.json
```

The OGC adapter auto-detects the input checkout commit, hashes the generated
`public_lexicon.tsv`, and records both values in `cog_public_freq.json`.
`merge_corpus_freq.py` carries that source string into `corpus_freq.json`.
The recorded rollup uses OGC commit `f4062a50` and public-lexicon SHA-256
`02d6de7181e8a108abe224081352d817c2064edf6234a51ecc63a5b8c83e6c2d`.

The PTA TEI corpus is tokenized by `build/tei_tokenize.py`, which extracts
text from `<text>` subtrees, drops apparatus / notes / bibliography, and
splits on non-Greek characters with accent-stripped lowercase NFC keys -- the
same shape GLAUx and Diorisis already use, so the merge is a straight
element-wise sum. The Open Greek Corpus rollup arrives pre-tokenized in the
same shape from that corpus's own build.

To inspect lookup coverage against the merged corpus run:

```
python build/coverage_report.py
```

As of the current pipeline, accent-relaxed `ag_lookup.json` coverage is
**96.57% of running tokens** in the merged 94.0M-token corpus and **36.77%
of unique forms**. `coverage_report.py` uses the same canonical elision key as
the frequency builders, so U+2019 corpus apostrophes and U+1FBD lookup
koronides compare correctly. None of the merged sources is lemmatized
end-to-end by Dilemma,
so they all contribute frequency and coverage signal but not new
form-lemma pairs (lemma pairs come from Wiktionary plus the
GLAUx/Diorisis/Perseus treebanks; the Gorman treebanks are held out
as evaluation gold).

### GlossAPI corpus audits

`data/glossapi_corpora.json` pins nine permissively licensed GlossAPI
candidates by Hub commit, byte size, and SHA-256. The audit measures
`guess=False` lookup coverage, unknown forms, conservative Unicode/OCR
defects, and forms whose effective `el` and `grc` lookup rows disagree:

```bash
pip install -e '.[glossapi-audit]'
python scripts/audit_glossapi_coverage.py --source stream
```

For a resumable audit, add `--sample-dir data/glossapi_samples`. The saved
counters are keyed by the source revision, artifact pins, reader, and sample
limits; coverage is recomputed against the current Dilemma inputs. Reports
record the code revision and the SHA-256 of the actual lookup and supporting
tables. Generated reports and sample counters stay local and uncommitted.

Sampling assigns a separate token budget to each artifact and selected Parquet
row group, and reads one group at a time. A row is a procedure, article,
consultation, or volume depending on the source, so document counts are not
directly comparable. The Parliament reader counts `section` once; its
`json_section` field duplicates that text and adds serialization escapes.
Unknown-form categories are triage hints, not confirmed dictionary omissions.
The report distinguishes database language disagreements from disagreements
that remain after removing citation spelling differences.

Frequency downsampling preserves the observed distribution instead of taking
only its most frequent forms. The experiment retains per-source counts, source
shares, and the checksum of the production frequency list. It still requires
contextual evaluation before changing ambiguous-lemma ranking.

The October 20, 2026 manifest supersedes the unavailable OpenGov v2 repository
with a verified pin of `glossAPI/OpenGov_Deliberations`. It retains the old pin
for comparison and samples the replacement's archive and recent consultations
separately. Equal artifact budgets balance those periods; they do not estimate
their population proportions. Any inaccessible source is reported explicitly
and keeps the frequency experiment incomplete.

Large sources can be deferred explicitly, without making a partial frequency
experiment look complete:

```bash
python scripts/audit_glossapi_coverage.py --source stream \
  --exclude-corpus diavgeia
```

The generated coverage and frequency reports are diagnostics and never
overwrite `data/mg_freq.txt`. Gutenberg, Ekklisiastika, and Archetai are
audit-only: historical evidence must first pass through Open Greek Corpus
provenance, language, OCR-quality, and deduplication gates. In particular,
raw Ekklisiastika rubrics, joined tokens, and repeated hymn cycles must not
feed language-model or frequency counts. OGC additionally blocks Ekklisiastika
admission pending written permission for the underlying GOARCH text or an
independently licensed replacement. A dataset-level license does not remove
that gate.

Evaluate cleaned Modern Greek frequencies separately with:

```bash
python eval/eval_glossapi_frequency.py --lookup PATH_TO_PINNED_MG_LOOKUP
```

The evaluator retains the raw token windows and removes repeated lines,
explicit field labels, and enumeration markers. It restores stress only in
all-capitals words with one accented spelling in the pinned Modern lookup.
Counts and reports remain local; `--reuse-counts` requires matching collector,
source-manifest, lookup, and sampling pins. After reviewing an unchanged
cleaning recipe, `--collector-sha SHA256` can explicitly pin counters from an
older collector while testing a new reranker. Runtime ranking inputs must
match `data/hf_manifest.json`; missing attestation or POS tables stop the run.
A smaller sanity run can use
`--corpus mitos --max-tokens 20000 --max-documents 32 --limit-eval 100` with
separate `--counts` and `--output` paths.

The pinned UD Greek GDT development split selects among fixed mixture weights;
the test split measures that choice and a predeclared 10% mixture. This is an
experimental reranker of tied lookup candidates and source-recorded noun,
verb, adjective, and adverb alternatives accepted by the existing POS
validator. Gold POS is not used to select candidates, and preceding-article
contexts are preserved. It reports paired gains/losses and compares against
both production
surface-frequency priors and the shipped runtime. Dictionary-mapped surface
counts are uncertain lemma priors. They do not replace the runtime's annotated
lemma attestation or `data/mg_freq.txt`. Adoption requires a positive test
effect and replication; insufficient evidence leaves the experiment separate.

The October 20, 2026 run collected all six Modern sources at their declared
caps. On the 9,243-token GDT development split it selected a 25% domain prior.
On the separate 9,452-token test split, that mixture scored 8,200 correct
lemmas, versus 8,193 for the production-frequency comparator and 8,213 for
the shipped runtime. The gain over the frequency comparator did not exceed
the runtime, so the mixture was not adopted and `mg_freq.txt` remains unchanged.

The GlossAPI variety classifier has a separate optional evaluation so
GreekBERT does not become a Dilemma runtime dependency:

```bash
pip install -e '.[variety-eval]'
python eval/eval_glossapi_variety.py
```

### Extraction sources

Form-lemma pairs come from three sources per Wiktionary entry:

1. **Inflection tables** (primary). Every cell in a verb conjugation or
   noun declension table becomes a form-lemma pair. Covers all tenses,
   moods, cases, numbers. Multi-form cells (e.g. `Πηλείδᾱο / Πηλείδεω`)
   are split into separate pairs.
2. **`form_of` references**. When a page says "form of X", that gives
   us an additional pair. Adds ~44K MG and ~6K AG pairs not found in
   inflection tables.
3. **`alt_of` references**. Alternative/variant spellings. Adds ~1K
   pairs.

## HuggingFace Hub dataset

Generated data files (ranked form lists, form frequencies, scored
lookups) are hosted at
[`open-greek/dilemma-data`](https://huggingface.co/datasets/open-greek/dilemma-data)
on HuggingFace Hub. `rank_forms.py` downloads these by default instead
of regenerating locally.

Available files:

| File | Description |
|------|-------------|
| `mg_ranked_forms.json` | MG lemma to frequency-ranked form list |
| `mg_form_freq.json` | MG form to frequency count |
| `mg_ranked_forms_verbose.json` | MG ranked forms with per-corpus breakdown |
| `mg_lookup_scored.json` | MG scored lookup table |
| `ag_ranked_forms.json` | AG lemma to frequency-ranked form list |
| `ag_form_freq.json` | AG form to frequency count |
| `ag_ranked_forms_verbose.json` | AG ranked forms with per-corpus breakdown |
| `med_ranked_forms.json` | Medieval lemma to frequency-ranked form list |
| `med_form_freq.json` | Medieval form to frequency count |
| `mg_polytonic_freq.json` | MG polytonic form frequencies from Wikisource (38M tokens) |
| `mg_polytonic_ranked.json` | Monotonic MG form to ranked polytonic variants |

This is separate from the main
[`ciscoriordan/dilemma`](https://huggingface.co/ciscoriordan/dilemma)
model repo, which hosts `lookup.db`, model weights, and other core
data files.

## Confidence tiers

Not all lookup entries are equally trustworthy. Forms from inflection
tables are template-generated and may be wrong for irregular words.
Each entry is scored on a 5-point scale:

| Tier | Condition | MG count | AG count |
|:----:|-----------|:--------:|:--------:|
| 5 | Both EN + EL Wiktionary have a page for this form | 63K | 17K |
| 4 | EN Wiktionary has a page (no EL page) | 22K | 50K |
| 3 | EL Wiktionary has a page (no EN page) | 1.05M | 131K |
| 2 | Both EN + EL tables agree on the lemma | 199K | 64K |
| 1 | Single source, table-only | 1.49M | 2.21M |

Higher confidence wins when two sources map the same form to different
lemmas.

### Dialect tagging

Ancient Greek forms from EN Wiktionary carry dialect tags extracted from
inflection table headers (e.g. "Epic declension-1", "Attic contracted
present"). These are propagated to every form in that table section:

| Dialect | Tagged forms |
|---------|:------------:|
| Attic | 262K |
| Epic | 102K |
| Ionic | 15K |
| Doric | 10K |
| Koine | 37K |
| Aeolic | 4K |
| Laconian | 672 |
| Boeotian | 555 |
| Arcadocypriot | 407 |

## Quality controls

### Missing Modern Greek present tables

Some independently defined verbs have no conjugation table in either pinned
Wiktionary dump. `build_data.py` completes their missing present-active
indicative cells with `dilemma/el_present_paradigms.py`. It requires a defined,
self-mapped head, preserves existing pairs and lookup readings, and handles
regular -ω, contracted -άω and the reviewed ακούω, καίω, κλαίω, φταίω, λέω and
τρώω families separately. It does not infer a past, passive or imperative.
Unsupported classes are listed rather than guessed.

Every added pair and lookup key is recorded in
`data/changes/el_present_completion.json`, with the original value, rule,
source-head evidence and input hashes. The bounded refresh command uses the
same rules against a manifest-pinned snapshot, preserving unrelated data:

```bash
python scripts/complete_el_present_snapshot.py --base BASE --output OUTPUT --kaikki KAIKKI
```

Its audit also records the exact inserted SQLite rows and each spelling-index
value before and after. Pass `--undo AUDIT --output OUTPUT` to restore the
original logical rows; the command refuses files modified after the refresh.

- **Greek-only filter**. All forms must contain only Greek Unicode
  characters (U+0370-03FF, U+1F00-1FFF, U+0300-036F). Removes Latin
  letters, digits, template artifacts.
- **Chain-breaking**. If form A maps to lemma B, and B maps to C, the
  chain is resolved to the real headword at build time. Fixes ~65K entries
  caused by accent-stripped key collisions and treebank convention differences.
- **Pronoun cross-contamination**. Greek Wiktionary dumps the entire
  pronoun paradigm table into each pronoun entry (e.g. `εσύ` lists
  `εγώ` as a "form"). Articles and determiners are restricted to
  headword-only. Pronoun forms that are headwords of other closed-class
  entries are skipped.
- **Proper noun plural filter**. EL Wiktionary generates plural forms
  for proper nouns via templates (413K junk entries like `Αχιλλείς`).
  These are skipped unless EN Wiktionary also lists them (which
  indicates a human editor intentionally added them, e.g. `Έλληνες`).
- **Training pair validation**. Lemmas must resolve to supported lexical
  heads. Independently defined homographs can retain separate paradigms even
  when the flat lookup chooses another reading. The `el_*_heads.json` and
  `grc_reading_heads.json` companions record the source evidence for those
  exceptions; unsupported targets are resolved or dropped.

## Lookup construction

The lookup table combines forms from multiple sources:

| Source | Forms | Notes |
|--------|------:|-------|
| **Wiktionary** (EN + EL, all periods) | 5.2M | Baseline from kaikki.org dumps |
| **LSJ + Sophocles expansion** | 6.4M net AG additions | Current Wiktionary-Lua expansion plus the marked surface forms from the revision-pinned historical expansion delta; 32K LSJ nouns, 22K verbs, 14K adjectives, and 13.5K Sophocles nouns + 4.6K verbs |
| **[GLAUx](https://github.com/alekkeersmaekers/glaux)** (Keersmaekers, 2021) | 557K | 17M-token corpus, 8th century BCE - 4th century CE, 98.8% lemma accuracy |
| **[Diorisis](https://figshare.com/articles/dataset/The_Diorisis_Ancient_Greek_Corpus/6187256)** (Vatri & McGillivray, 2018) | 76K new | 10M-token corpus, Homer - 5th century CE, 91.4% lemma accuracy. Low-priority pairs (only added when no conflict with existing sources). Contributes to the merged 94.0M-token `corpus_freq.json` alongside GLAUx, PTA, and the [Open Greek Corpus](https://github.com/open-greek/open-greek-corpus) open-text rollup. |
| **[HNC Golden Corpus](https://inventory.clarin.gr/corpus/870)** (CLARIN:EL) | 1K new | 88K-token gold-standard MG corpus, 11K unique form-lemma pairs. Low priority (only added when not in Wiktionary). Also used for MG evaluation. |
| **[Perseus / AGDT](https://github.com/PerseusDL/treebank_data)** (CC BY-SA 3.0 US) | 81K | The 33 Greek AGDT works: Sophocles, Aeschylus, Homer, Hesiod, Herodotus, Thucydides, Plutarch, Polybius, Athenaeus. Sourced from the original, not the NonCommercial UD release. |
| **DGE** (Diccionario Griego-Espanol) | 52K | Headword filter coverage for spell-check |
| **LGPN** (Lexicon of Greek Personal Names) | 44K | Proper noun headword coverage |
| **Perseus Digital Library** (L&S, Pape, Bailly, etc.) | 176K | Headword filter from multiple classical lexica |
| **VLG** (Vocabolario della Lingua Greca, Montanari) | 133K | Headword filter coverage; 12K new beyond LSJ/DGE/Perseus |
| **Words in Progress** (Aristarchus supplementary lexicon, Montanari & Perrone) | 4K | Curated new/rare AG headwords with morphology; ~1,135 new |
| Closed-class handling | Grammatical rules | Article recognition and optional canonical mappings; see the API guide |

**Wiktionary Lua expansion.** Most of the non-Wiktionary forms above
come from running Wiktionary's own
[`grc-decl`](https://en.wiktionary.org/wiki/Module:grc-decl) and
[`grc-conj`](https://en.wiktionary.org/wiki/Module:grc-conj) Lua modules
over LSJ and Sophocles headwords, via
[wikitextprocessor](https://github.com/tatuylonen/wikitextprocessor).
These are the same modules Wiktionary editors invoke through
`{{grc-decl}}` / `{{grc-conj}}` templates when they paste in a headword
plus its grammatical metadata; the modules emit the full noun, verb, or
adjective paradigm. Wiktionary itself only renders those paradigms for
headwords an editor has manually written up. Running the modules over
the 32K nouns + 22K verbs + 14K adjectives in LSJ and the 13.5K nouns +
4.6K verbs in Sophocles produces millions of inflected forms for
classical and Byzantine vocabulary that no editor has touched, which
is what lifts the lookup table from 5.3M Wiktionary-only forms to 14.5M
total and is why rare-vocabulary coverage on classical and Patristic
texts is competitive with rule-based morphological analyzers. Cunliffe's
Homeric Lexicon (~12K headwords) isn't expanded this way because its
headwords are a subset of LSJ and already covered by the LSJ expansion
plus the GLAUx Homeric corpus (557K pairs). See
[LSJ/Sophocles expansion](lexicon-expansion.md#lsj-and-sophocles-paradigm-expansion) for the build step.

Verb expansion rejects source entries explicitly identified as prefixes,
imperative forms of another verb, or defective verbs whose entry explicitly
limits the available persons. Those restrictions preserve source-attested
forms without inventing an unrestricted paradigm. It checks the original
headword spelling before quantity marks are removed, so the prefix ἡμῐ cannot
receive a verb paradigm merely because its letters end in -μι. Ordinary verb entries that
describe their conjugation remain eligible.

Synthesized uncontracted active imperatives receive recessive accent from
the complete form, rather than inheriting the citation or future stem's
accent. For example, `ἀκούω` and `ἀκούσω` give `ἄκουε` and `ἄκουσον`.
The independently documented futures of `ἁνδάνω` and `ἐνίπτω` do not
supply an aorist stem: their source aorists use different stems. Their
observed aorist cells and future participles remain available; irregular
aorist parts are not reinterpreted through regular sigmatic participle templates.
Explicit vowel lengths in independent Wiktionary verb citations and first-person
indicatives are retained in `grc_lexical_heads.json`. Synthesis uses matching
principal-part spellings to recover these lengths without copying their accent.
Conflicting lengths remain separate source readings and supply no inferred
length. Nominal headers and contradictory grammatical tags cannot supply
verb quantities. A declared verb-form page needs a separate lexical definition;
quantity notation is removed after accent placement.

The verb expansion is more than just the present system. For each
verb headword, an LSJ-entry parser (`build/lsj_principal_parts.py`)
walks the head paragraph and pulls out whatever principal parts LSJ
has labeled (`fut.`, `aor.` 1 / 2, `pf.`, `pf. m./p.`, `aor. p.`,
`impf.`, `plpf.`), feeding each into `Module:grc-conj` as a
positional argument. A principal part must carry a tonal accent. The parser
skips an unaccented token when an intact accented form immediately follows
in the same clause; it never supplies a missing accent to the source text.
About 712 LSJ-only verbs receive at least one
parsed principal part this way, lifting them from present-tense-only
expansion to full paradigm coverage and adding ~+55 unique forms per
eligible verb on average. The classifier itself
(`build/expand_lsj.py::_classify_verb`) routes athematic and
irregular verbs explicitly: -ννυμι / -νυμι / -ημι / -ωμι / -αμι /
-μι suffix dispatch for the regular cases, plus a hand-coded Attic
core paradigm for εἰμί, εἶμι, οἶδα, χρή, φημί and their preverbed
compounds (πάρειμι, σύνειμι, εἴσειμι, ...) joined via a backtracking
preverb splitter. Across the 28,745-candidate LSJ verb set the full
pipeline succeeds on 28,535 (99.27%); the residual failures are all
OCR-corrupt or non-Attic dialect entries with no canonical Attic
base.

The lookup table is built from Wiktionary [kaikki dumps](https://kaikki.org/)
(EN and EL editions for MG and AG, plus EL Medieval Greek), expanded with
inflected forms from LSJ (via Wiktionary Lua modules) and the Sophocles
lexicon of Roman and Byzantine Greek, then augmented with form-lemma pairs
from gold-standard treebanks (AGDT). Each form is indexed under
its original, monotonic, and accent-stripped variants, so `θεοὶ` (polytonic
with grave), `θεοί` (monotonic with acute), and `θεοι` (stripped) all
resolve to `θεός`. Input can be polytonic, monotonic, or unaccented. AG
forms take priority over MG, ensuring classical lemma forms (βιβλίον,
φύσις, θεῖος) are preferred over their MG equivalents (βιβλίο, φύση,
θείο). Medieval Wiktionary entries are merged into the MG table at
build time. When `lang="el"` is used, 150K MG-specific entries
override the AG-first defaults with MG lemma forms (ο instead of ὁ,
είμαι instead of εἰμί). For polytonic input (breathings/circumflex),
an additional AG-only lookup pass runs first.

When the transformer handles an unseen form, beam search generates
multiple candidates and picks the first that matches a known headword
from the citation-safe filter. In default mode that includes validated lookup
self-maps plus independent Ancient/Byzantine lexicon headwords; with
`citation_policy="strict_ag"` it uses only the independent headword inventory.
If nothing matches, the input is returned unchanged. (DGE, LGPN, and the
Perseus Digital Library lexica feed the build-time lemma-validation and
spell-check filters, not this runtime output filter.)

**Wiktionary as upstream:** Because Dilemma's lookup tables are built
directly from Wiktionary, any missing or incorrect lemmatization can
often be fixed by editing the Wiktionary entry itself. When the kaikki
dumps are next regenerated and `build_data.py` re-run, the fix flows
into Dilemma automatically. This means the coverage and accuracy of
Dilemma improve over time as Wiktionary's Greek coverage improves,
without any changes to Dilemma's code.

## Lexical readings and source validation

### Ancient lexical readings

Cross-source form references preserve cycles in every Greek language dataset.
A noun and verb can each have an inflection spelled like the other's headword:
`ἔργον` (work) and `ἔργω` (enclose) retain separate lemmas and paradigms instead
of collapsing according to the order of the source entries.
Ancient training pairs retain each source morphological reading. Their table
owners and same-part-of-speech references resolve to independently defined
heads before the flat lookup: `ἀλέομαι` (shun) keeps its own table, while the
same spelling in the conjugation of `ἀλέω` (grind) still belongs to `ἀλέω`.
An explicit reference belonging only to another part of speech cannot consume
an independently defined head: the noun reading of `κρίνω` points to `κρίνον`
(lily), while the verb `κρίνω` (judge) keeps its own conjugation. References
already applied within one source receive the same check. This also preserves
the verb `δύω` (plunge) beside the Epic numeral variant of `δύο` (two).
Dialect, contracted-form and voice-only descriptions alone do not establish a
separate lexical head, including when the source labels the entry as a verb.
Their inflection tables resolve to the supported parent without treating the
variant description as an independent definition. A separate lexical meaning
in the same entry still supports its headword.
Nominal case descriptions also recognize slash-separated cases, gender and
contraction labels. Verb-form descriptions with explicit grammatical qualifiers
likewise remain grammatical evidence, not independent lexical definitions.
Explicit spelling references, sacred abbreviations, and
unexpanded inflection templates do not supply independent definitions. A form
header still permits a genuine lexical meaning, such as a lexicalized plural.
Reference targets can follow a colon or an explicit word label.
The entire description must be a reference or a recognized grammatical note;
an appended definition remains evidence of an independent meaning.
English form senses may nest a translation beneath their grammatical
description. Such translations do not define another head: `ἦλθον` retains
its aorist readings under `ἔρχομαι`. Greek present-tense head descriptions
followed by their own lexical meanings remain independent.
Alternative-spelling references from the original word to a supported head
of the same part of speech still canonicalize normally. When a source assigns
the alternative to a different numbered etymology from an independently
defined homograph, only that alternative entry follows the reference. For
example, the binding and needing senses of `δέω` retain their own tables;
the separate alternative spelling of `δήω` still points to `δήω`.
A spelling-reference chain also follows explicit references before a competing
inflection-table mapping, provided it reaches one positively defined head of
the same part of speech. Thus the pitch spelling `πίτα` follows `πίττα` to
`πίσσα`, while the independent pie reading keeps `πίτα`. Ambiguous or
unsupported chains retain the previous fallback.
Scalar lookup also follows an exact form page's unambiguous, same-part-of-speech
reference chain when an intermediate word belongs to an unrelated lexical
head. Thus `παρόντος` reaches the verb `πάρειμι` through its participle `παρών`,
while the separate noun `παρών` and its genitive `παρῶνος` retain their own
reading. Independent source words and their normalized keys are protected;
conflicting references and cycles retain the previous lookup. Name references
count as conflicting evidence without redirecting names, so the name forms
`Ἀθηναίην`, `Ἀθηναίης`, and `Ἀθηναίῃ` retain their Athena reading.
This priority applies to the explicit target recorded in a source reading,
including structured spelling and inflection references;
scalar part-of-speech mappings keep their existing choice between homographs.
An Ancient comparative or superlative description alone does not establish
an independent adjective head, even when it includes a translation after a
colon. A separately supplied lexical meaning still establishes that head.
Synthesized first-aorist active infinitives and present or future neuter
participles retain their accent position while taking a circumflex on a
known long penult before a short final syllable (`τμῆξαι`, `παιδεῦον`).
Unmarked vowel length is not guessed, and finite optatives keep their
distinct accent. A diphthong consumes only two vowels; a following vowel
starts another syllable when determining the accent's permitted position.
An inflected reading
does not replace an independently defined homograph: `δίω` keeps its own
conjugation as well as its separate reading in the conjugation of `δείδω`.
`data/grc_reading_heads.json` records the exact source definitions, parts of
speech, and source hashes for surviving Ancient training heads that differ
from their flat lookup target or are absent from the flat map. The training-data
integrity check requires this evidence; an unrelated definition under another part of speech cannot
authorize an exception.

### Modern Greek homographs

In the Modern Greek dumps a capitalized page (a name, an abbreviation, a
heading in sentence case) reaches its lowercase keys only where no
lowercase page or table claims them: the female name `Ερωτώ` does not take
`ερωτώ` from the verb, nor the abbreviation `ΔΕΣ` take `δες` from `βλέπω`.
And a form-of reference of one Wiktionary that would point a lemma back at
a word the other files under it is not followed: EN files `ρωτώ` as a form
of `ρωτάω` while EL's `ρωτώ` table lists `ρωτάω`, and the cycle made the
chain-breaking delete both conjugations. The same priority applies across
the two dictionaries: a weak lowercase key derived from a name in one
source does not override the other's verb form (`λεν`, `τρων`, `μιλάνε`).
A form page's head line and its verb inflection table resolve through the
word that page names. They do not inherit a noun homograph's lemma:
`ερωτήσεις` remains a form of `ερωτώ` on verb rows, and the participle
forms of `αντιπαθών` belong to `αντιπαθώ`.
The same POS-specific references are followed through spelling pages, so
`ἀντιπαθών` also reaches the verb. A row tagged as a verb is omitted when
its target is attested only as another part of speech; this removes
misclassified noun and adjective forms without losing their genuine
nominal rows. Existing noun, adjective and ethnonym paradigms remain.
Only a sense with an explicit `form_of` reference can supply verb evidence
from its gloss; descriptive mentions of a verb do not redirect passive
heads or create new verb lemmas. When the gloss names that same reference,
it can correct an incorrect head-page POS (`δηώ`). A broken reference does
not remove a separately attested verb head or proper-name head form.
### Participles and finite verbs

A participle's top-level reference can resolve its verb only when that
target reaches an independent lexical verb entry in either dictionary,
including through an explicit spelling-reference chain. All explicit verb
targets from both dictionaries are retained separately from inflection-table
claims. A successful sense-level reference chain from the participle's own
entries takes priority over top-level pointers. A finite-form homograph's
references do not replace that parent, and a sense labeled as a synonym is
not a spelling alias. Other alias nodes use their explicit verb references;
table evidence is used only when neither supplies a finite endpoint.
Conflicting successful targets remain unresolved. Participles tagged on an
entry or one of its senses remain intermediate forms, and an explicit-reference
cycle must have one unambiguous exit to a finite verb. A standalone participle
that would otherwise remain a verb lemma also enters this check, even without
a top-level pointer. Its actual references are followed before an unsupported
verb reading is omitted. This check runs after both sources have been read,
so entry order does not affect it. A resolved participle keeps that finite
verb through POS-specific cleanup instead of inheriting a flat homograph's
lemma; independent noun and adjective readings remain. Its own recorded
table also keeps that supported verb when a spelling-alias pointer names
an unresolved participle (`καθούμενος` to `κάθομαι`, despite `καθήμενος`).
Without a supported own-head path, the unresolved alias still blocks the
verb reading.
Nominal labels in top-level references cannot supply verb endpoints through
unrelated inflection tables. An unresolved participle also cannot remove an
independently defined finite verb through a reverse table or lookup alias.
The build also writes `data/el_participles.json` for explicitly recorded
participles whose finite parent remains unresolved. This is morphology for
dictionary export, separate from the verb training pairs: it preserves the
source headword and its own nominal inflection cells without inventing a
finite verb or relabeling it as an adjective. A headword without a table
retains only its own spelling, with no inferred case or descendants.
The version-1 JSON envelope has `language: "el"` and a `cells` array. Each
row records `form`, `lemma`, original `pos: "verb"`, `kind` (`headword` or
`inflection`), grammatical `tags` including `participle`, the unmodified
`source_form` and `source_lemma`, and sorted source editions in `sources`.
Inflection rows require an explicit case and number and exclude finite
person or mood cells. Different source spellings and cells remain separate;
only identical readings merge their source editions.
### Noun and adjective ownership

An independently defined noun's own table also retains its source lemma
in the training pairs, even when the flat lookup chooses an inflectional
homograph. The masculine noun `σκιάς` and the feminine genitive `σκιάς`
of `σκιά` therefore keep separate paradigms. Form-of and alternative-spelling
entries still follow their explicit references; no ownership marker enters
the output JSON, and flat lookup arbitration is unchanged.
A Modern noun form page with an explicit case and number does not supply a
full table when that table assigns incompatible cells to the page's own
headword. Its exact form reference remains: noun `κρατών` stays the genitive
plural of `κράτος`, while the independently defined adjective `κρατών` owns
`κρατούν` and `κρατούσα`. Spelling-variant pages, tables that reproduce the
stated cell, and entries with a separate lexical meaning retain their tables.
`data/el_noun_heads.json` documents the independent noun heads for which that
flat choice differs, with exact source headwords, noun senses, tags, and
SHA-256 hashes of the source editions. The version-1 envelope has
`language: "el"`, `source_sha256`, and `heads`; each head retains all distinct
qualifying source entries in `evidence`. Integrity checks permit a non-self-mapped
noun lemma only when this companion contains valid independent evidence for
that exact head. Inflection, alternative-spelling, and misspelling entries
cannot authorize an exception. Both the command-line data check and pytest
validate the built pairs against the lookup and this companion.
Independently defined Modern adjectives retain their own source tables through
the same flat homographs. Adjective `κρατών` therefore owns `κρατούν`,
`κρατούσα`, and `κρατούσες`, while noun `κρατών` remains the genitive plural
of `κράτος`. A new typed adjective field preserves the previously published
non-adjective fields and their scalar fallbacks; an adjective correction does
not expose unrelated internal noun or adverb table mappings.
`data/el_adjective_heads.json` records exact adjective definitions,
declared head types, source forms, and source-edition hashes for non-flat owners.
Certification also requires an explicit adjective head template, forms in at
least two genders, or an explicitly indeclinable header. A raw adjective POS
label alone cannot certify a duplicated noun or adverb entry. Definitions with
insufficient morphology retain their previous processing. Grammatical form
descriptions, spelling references, and empty headers cannot authorize an
exception. The extractor publishes the supported ADJ target in
`mg_pos_lookup.json` whenever it differs from the scalar fallback, including
forms with only one typed reading. The scalar lookup remains unchanged.
The companion is generated by the normal Modern build, included in the base
data download and CI registry, and required by the built-pair integrity check.
### Finite-verb validation

The corresponding finite-verb exceptions are recorded separately in
`data/el_verb_heads.json`. Generate it after extraction with
`python build/build_el_verb_heads.py --kaikki PATH_TO_DUMPS` (or pass
`--data PATH_TO_OUTPUT`). The builder selects only non-self-mapped verb
lemmas from the final pairs, reads the same source editions, and requires
an independently defined finite verb for every selected head. Participle,
form-only, and nominal entries cannot supply that evidence. A header or
no-gloss relation alone also cannot make a terminal finite verb: its explicit
source references must reach a positively defined finite parent. Missing support
fails the build. These companions validate morphology ownership; they do
not change the runtime lookup or its POS table.
### Determiners and pronouns

Modern reference rows retain their own sense's grammatical and usage tags,
rather than reducing every reference to `form-of` or `alt-of`. Distinct source
cells remain separate. Independently defined determiner and pronoun heads also
retain their own POS-specific morphology when the flat homograph is a verb:
`πάση` keeps its dative tags and determiner owner `πας`, while the verb `πας`
still resolves to `πηγαίνω`. The extraction-generated
`data/el_closed_class_heads.json` records exact same-POS source definitions and
input hashes for non-flat heads. Built-data integrity checks require this
positive evidence and an exact matching DET or PRON self-target in
`mg_pos_lookup.json`; form-only, spelling-reference, and no-gloss entries cannot
authorize an exception. The scalar lookup remains unchanged.
Own determiner and pronoun sense references follow same-POS chains to a
positively defined head before competing table claims: `εσένα` and `εσένανε`
resolve to `εσύ`, without treating weak form `σε` as a lexical head.
Different accented references keep their own roots; a normalized key with
multiple supported roots retains its previous mapping.
`lookup.db` takes these changes up when `build_lookup_db.py` next rebuilds
it. A pronoun table that spans
several grammatical persons is the shared personal-pronoun grid, whose forms
do not all belong to that page, and is skipped whatever its
template's name: in the 2026-09 dumps EN's `τα` page carries it under a
generic template and EL's `εγώ` and `εσείς` under a renamed one, which
made `εγώ`, `εσύ`, `αυτός` and `μου` forms of `τα`.

### Modern spelling references

A Modern Greek page that only gives another spelling of a word (the Greek
Wiktionary's `γιός`, "πολυτονική γραφή του γιος", or `μιλῶ`, whose link to
`μιλώ` kaikki does not carry; the English Wiktionary's "polytonic spelling
of" pages) is read as that word's alternative form, not as a lemma: as a
lemma it took the word's monotonic and accentless keys, made `γιος` a form
of `γιός` in a cycle that deleted its paradigm, and filed the forms of
`μιλώ` and `αγχωτικός` under `μιλῶ` and `ἀγχωτικός`. Such a page is one
every sense of which names the page's own word without the marks monotonic
writing drops, and for a monosyllable (one run of vowels, `γιος`, `γεια`)
without its accent, by an `alt_of` or `form_of` reference or by that gloss
(`γιος` for `γιός`, `μουστακάτος` for `μουστακᾶτος`): 76 pages of the Greek
dump, 50 of them known only by the gloss, and 47 of the English one
(`build_data.spelling_target`). A page that names a word of other letters
(`ἐτρώθη`, a form of `τιτρώσκω`), a `form_of` reference of a sense tagged
with a case (`σοφίᾳ`, "dative singular of σοφία"), and a page with a sense
of its own are read as before, and so are the Ancient and Medieval Greek
dumps, where a page whose monotonic spelling is another word's is an
inflected form of it (the dative `φωνῇ` of `φωνή`). In each Modern Greek
dump's own pass two pages that name each other as forms (`αγχωτικός` and
`αγχώδης`, each "συνώνυμο του" the other, which kaikki reads as `alt_of`),
or a page naming a word whose key already maps back to it, both stay
lemmas, where following the reference made a cycle that the chain-breaking
deleted with both paradigms. The merged lookup checks all proposed
references before applying any of them. Reciprocal synonyms such as
`νησοπυριτικός` and `ορθοπυριτικός` therefore keep both paradigms regardless
of their order in the dumps. The weak pronouns `μόνος` shows on its
headword line (`μόνος μου`, `μας`, `σας`) are not read as its forms, which
had made `σας` a form of `μόνος`.

### Abbreviations

An abbreviation is not an inflection of the word it abbreviates, and in the Modern
Greek dumps its reference does not move the key of a lowercase word with a
sense of its own (`build_data.Abbreviations`). The English Wiktionary gives
`δις` and `δ` as abbreviations of `δεσποινίς`; once `δεσποινίς` kept its
paradigm, following them made the adverb `δις` ("twice", and the "billion"
of `δισ.`, 20,825 monotonic tokens) and the letter `δ` its forms. They stay
themselves, and so do `α`, `β`, `ν` and `κ`, which the same pages of
letters had made forms of `άντε`, `βόρειος`, `νότος` and `κυρία`. The other
abbreviations the page of one abbreviation lists as alternatives are forms
of the word it abbreviates, not of the page's headword: `δ`'s `δις` is a
form of `δεσποινίς`, where filed under `δ`, the letter, it was a noun of
lemma `δ`. A key no lowercase word claims still follows its abbreviation:
`κα` goes to `κυρία`, whose Greek Wiktionary page is an abbreviation too,
`μαρ` to `Μάρτιος` beside the surname `Μαρ`, and the contraction `στο`
("σε + το"), which the English Wiktionary also tags as an abbreviation, to
`σε`.

### Grammatical cell tags

Modern Greek pair deduplication retains the part of speech and complete
grammatical tags of each table cell. Identical rows are removed, but the
genitive, accusative and vocative singular rows of `καθηγητή` remain
separate even though their monotonic spelling is identical. The exporter
can then distinguish `καθηγητῆ` from `καθηγητή`. The JSON fields remain
`form`, `lemma`, `pos` and `tags`; internal reference metadata is not
written to the data files.

For individual form-to-lemma corrections where the lookup table returns
the wrong lemma due to ambiguity (e.g., proper nouns beating common
verbs), `build_lookup_db.py` has a `_LOOKUP_OVERRIDES` dict that
hard-corrects specific entries in the database.
