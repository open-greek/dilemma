# Word lists and Hunspell export

[README](../README.md)

- [Frequency-ranked inflection lists](#frequency-ranked-inflection-lists)
- [Hunspell spell-check export](#hunspell-spell-check-export)
- [Boundary rewrites (`grc_morph.json`)](#boundary-rewrites-grc_morphjson)

## Frequency-ranked inflection lists

`rank_forms.py` produces per-lemma ranked form lists, sorted by corpus
frequency. This is useful for downstream consumers (e.g.
[Lemma](https://github.com/open-greek/lemma)) that need to know which
inflections of a word are most common.

```bash
python rank_forms.py --lang el     # Modern Greek (default)
python rank_forms.py --lang grc    # Ancient Greek
python rank_forms.py --lang mgr    # Medieval/Byzantine Greek
python rank_forms.py --lang all    # All three
```

For each language, the script produces:
- `{prefix}_ranked_forms.json` - lemma to list of forms sorted by frequency
- `{prefix}_form_freq.json` - form to raw frequency count

Frequency sources (used for primary ranking):
- **MG**: [FrequencyWords/OpenSubtitles](https://github.com/hermitdave/FrequencyWords) (1.49M forms)
- **AG**: merged `corpus_freq.json` (94.0M tokens, 1.99M unique forms) from
  GLAUx + Diorisis + PatristicTextArchive + the
  [Open Greek Corpus](https://github.com/open-greek/open-greek-corpus)
  open-text rollup (First1KGreek, corrected Patrologia Graeca, Perseus
  canonical-greekLit, byzantium.gr)
- **Medieval**: AG corpus frequencies as proxy

Additional corpora available in `--verbose` per-form breakdowns:
- **Patrologia Graeca** (`freq_pg`): 3.14M tokens of Byzantine/Patristic Greek from [PG corpus](https://zenodo.org/records/15780625) (Church Fathers, PG071-PG158)
- **Byzantine vernacular** (`freq_byz_vern`): 191K tokens from the [Byzantine Vernacular Corpus](https://github.com/open-greek/byzantine-early-modern-corpus) (Digenes Akritas, Chronicle of Moreas, Erotokritos, etc.)

By default, `rank_forms.py` downloads pre-built files from the
[`open-greek/dilemma-data`](https://huggingface.co/datasets/open-greek/dilemma-data)
HuggingFace dataset. Use `--rebuild` to regenerate locally from the
lookup and frequency source files:

```bash
python rank_forms.py --lang el --rebuild            # regenerate MG locally
python rank_forms.py --lang all --verbose --rebuild  # all languages with per-corpus breakdown
python rank_forms.py --lang el --polytonic --rebuild  # MG with polytonic variant ranking
```

The `--polytonic` flag generates `mg_polytonic_ranked.json`, which maps
each monotonic MG form to its attested polytonic variants ranked by
corpus frequency. This is built from `mg_polytonic_freq.json` (see
`build/build_polytonic_freq.py`), which extracts polytonic word
frequencies from the [glossAPI/Wikisource Greek texts](https://huggingface.co/datasets/glossAPI/Wikisource_Greek_texts)
dataset (~38M tokens, 5,394 texts). Forms appearing fewer than 3 times
are filtered out.

The `--verbose` flag adds a `{prefix}_ranked_forms_verbose.json` file
with per-corpus frequency breakdowns for each form. Each entry includes
frequencies from all available corpora (OpenSubtitles, GLAUx, Diorisis,
Patrologia Graeca, Byzantine vernacular), letting consumers re-rank for
mixed-period use cases (e.g., a Modern Greek dictionary for a book about
ancient topics could boost forms with high `freq_glaux`).

## Hunspell spell-check export

`export_hunspell.py` produces compact Hunspell `.dic` + `.aff` pairs from
`lookup.db`, aimed at mobile consumers (primarily the
[Tonos](https://tonospolytonic.com/) iOS polytonic keyboard)
where the full 1.07 GB `lookup.db` and 560 MB `spell_index.db` do not
fit inside the ~48 MB memory ceiling of a keyboard extension. Affix
compression uses a zero-strip SFX rule only when its dictionary base is itself
a real member of the paradigm. A mechanical common prefix such as `λόγ` or
`ἀνθρώπ` is inlined instead, because both Hunspell and Tonos accept every
`.dic` base as a word. Every flagged base therefore has an identity rule.
Editorial brackets and parentheses are excluded before affix compression, so
the generated `.aff` remains loadable by consumers that compile affixes as
regular expressions, including `spylls`.

Default output is the **grc** variant (Ancient + Medieval polytonic),
which is what Tonos ships. An optional **el** variant (Modern Greek
monotonic) is retained for other downstream consumers via
`--variant el`. Output layout under `build/hunspell/`:

The `el` and `both` exports require `data/mg_form_freq.json`. Fetch it with
`hf download open-greek/dilemma-data mg_form_freq.json --repo-type dataset
--local-dir data`. If it is absent or empty, the exporter exits with an error
before writing a dictionary. The `grc` export does not need this file.

| Variant | Script name | Lang tag | Contents |
|--------|------------|---------|---------|
| `grc_polytonic.{dic,aff,version}` | `grc` | `grc` | Ancient + Medieval polytonic forms. Rows owned by `src='grc'` are eligible, as are language-shared rows for a lemma that also has `grc` evidence; this preserves AG headwords such as `λέγω`, `πατήρ`, and `γῆ` without admitting unrelated Modern lemmas. New forms require exact evidence from the complete accent-preserving `form_profile.db`, including hapaxes, or membership in a pinned citation, textbook-paradigm, grammar, or productive second-accent class; arbitrary unattested generator output is not admitted. A productive second accent needs a proparoxytone or properispomenon host (`θάλασσάν`, `δῶρόν`, but not `λύκοί`). A new spelling that differs only in accent, breathing, diaeresis, or iota subscript from a well-formed spelling with at least 1,000 corpus tokens (`ἑγώ` beside `ἐγώ`, `ταΐς` beside `ταῖς`) must also be attested in the GLAUx or Diorisis treebanks: at least 25 tokens when it has under 1% of the common spelling's count, at least 5 when it has under 5%. A spelling both treebanks annotate independently, more than once each and at a rate of at least one treebank token per thousand of the dominant spelling, is confirmed instead of counted, which keeps the dual `πρώτω`, Doric `γλώσσᾳ` and the contract `πειρᾷς`. This keeps genuine dialect spellings such as Doric `τᾷ` and `τῶ`. Outside the reviewed April surface (see the release audit below), a vowel- or rho-initial form without a breathing is rejected, and so are a grave accent before the final syllable (`καὶτοὺς`, `ὓστερον`) or on an elided word (`γὰρ᾽`), and a second accent anywhere but on the ultima of a proparoxytone or properispomenon, or before a fused enclitic (`Αἴγυπτόνδε`). Valid elisions normalize their final textual apostrophe to U+1FBD koronis. Same-lemma bare elision fallbacks are excluded unless an independent DGE/Cunliffe headword proves the collision, while the reviewed historical fallback set catches malformed self-headwords and preserves genuine `ἄν`. The closed Homeric apocope set preserves `κὰτ`/`κάτ` through `κὰγ`/`κάγ` and `ἂμ`/`ἄμ`. A closed grammatical list preserves legitimate unaccented words: enclitics including the `τις`, `φημί`, and `εἰμί` forms, dialect proclitics (`ἁ`, `αἰ`), and crasis with a proclitic (`κἀν`, `χὠ`), including the legitimate indefinite-pronoun forms `του` and `τῳ`. The same list carries the article's graves (`τὸ`, `τὼ`), which `lookup.db` leaves to `resolve_articles`, the relative pronoun's (`ὃ`, `ἣ`, `οἳ`, `αἳ`), and the crasis of `καί` written with an iota subscript that GLAUx or Diorisis has at least 4 times (`κᾂν`, `κᾀπὶ`, `κᾀπειδὰν`, `κᾄπειτ᾽`, the unaccented `κᾀκ` and `κᾀξ`), for which `lookup.db` has no row. A spelling GLAUx or Diorisis has at least 5 times outside the language model's held-out sentences comes in even when no lemma source proposes it, under itself, or when only the Modern Greek lexicon does, under that lexicon's lemma (`σπανίως`, `κτήσεις`, `χαίρετε`); the counts are case-sensitive, a contextual grave counts with its acute, and the held-out sentences' own occurrences are left out (`data/hunspell_grc_lm_dev_spellings.json.gz`, written from `build/lm/dev_sentences_{glaux,diorisis}.txt` by `scripts/build_hunspell_dev_spellings_fixture.py`), so those sentences stay a fair test of the dictionary. An elided spelling, an inner capital, an iota adscript, a vowel-length mark, a breathing after a consonant (`δὔ`, a converted elision mark), a second accent, or the grave of an explicitly rejected spelling (`τοὺ`) keeps a treebank spelling out, a capitalized one comes in only when its lowercase is not a word, and the respelling rule above applies to them as to every new form; `--treebank-spelling-min-count 0` turns both rules off. Crasis inside a word may carry its breathing past the first syllable (`ἐγᾦμαι`, `καλοκἀγαθία`, `ταὧν`), crasis with `ὦ` keeps the interjection's circumflex wherever it lands (`ὦνθρωπε`, `ὦλεθρε`), and reviewed lists keep complete words that end in a consonant (`ἔκ`, `παρέκ`, `ὑπέκ`, `χερουβίμ`). A fused enclitic's host keeps its own accent window: the `-τις` of `ὅστις` (`οὗτινος`), `-περ` (`οἷονπερ`), and the `-δε` of the epic and Ionic datives of `ὅδε` (`τῇσιδε`); any other `-δε` takes the second accent (`πόλεμόνδε`). In a Latin name `ου` between two vowels is the consonant v, not a syllable (`Ὀκτάουιος`), and the Ionic enclitics `τεο`, `τεῳ` and `τεων` are written without an accent. Every retained contextual grave receives its citation/before-pause acute twin unless the twin fails a structural rule. |
| `el_GR_monotonic.{dic,aff,version}` | `el` | `el_GR` | Modern Greek monotonic forms, including MG-relevant vocabulary drawn from the AG side of `lookup.db` (articles, common verbs, proper names). Not shipped in Tonos. |

A third list, `grc_mg_polytonic.{dic,aff,version}`, holds polytonic Modern
Greek spellings, attested ones and generated ones for the frequent forms of
verbs, nouns, adjectives and the other inflected words, for a keyboard to
merge into grc. It comes from
`export_mg_polytonic.py`; see
[below](modern-greek.md#polytonic-modern-greek-word-list).

Each dictionary entry carries a morphological field `fr:<bucket>` where
the bucket is one of `C` (common), `M` (medium), `R` (rare), or `X`
(unseen in corpus). This lets consumers rank spelling candidates
without shipping the full frequency table.

The bucket is chosen from three signals, in priority order:

1. **Canonical seed.** `data/canonical_ag_forms.json` pins ~194 iconic
   polytonic surface forms to `C` (Iliad/Odyssey/Herodotus incipits,
   Olympians, Homeric heroes). These are low-token-count forms whose
   cultural weight exceeds their corpus frequency - `ἄειδε` appears
   only 71 times in corpus but is the opening word of the Iliad. The
   grc export also pins the citation forms of the 2,000 most frequent
   LSJ9 headwords and the 971 reviewed forms of the language-model head,
   so unaccented `λυω` ranks `λύω` first. Textbook paradigm cells and
   closed-list forms are exempt from the evidence filters but keep their
   corpus bucket.
2. **Canonical lemmas.** ~168 famous lemmas (`ἀείδω`, `μῆνις`, `Πλάτων`,
   etc.) promote any polytonic-marked form of theirs to `C`. This
   catches canonical inflections beyond what the seed enumerates.
3. **Lemma aggregate.** For polytonic forms, the lemma's total corpus
   count promotes the form: `>= 20K -> C`, `>= 5K -> M`. Highly-
   inflected lemmas (`ἀείδω` has 1289 surface forms) would otherwise
   dilute frequency across the paradigm so no single form crosses the
   per-form threshold.
4. **Per-form count fallback.** `count >= 1000 -> C`, `>= 100 -> M`,
   `>= 1 -> R`, else `X`.

Only polytonic-marked forms receive the canonical promotions, so
monotonic leaks like `άειδε` (acute-only, no breathing) stay at `R`
and downstream rankers can pick the polytonic variant when the two
otherwise tie.

```bash
python export_hunspell.py                 # grc polytonic (default)
python export_hunspell.py --variant both  # grc + el
python export_hunspell.py --variant el    # el monotonic only
python export_hunspell.py --sanity 10000  # 10K-lemma sanity pass (no April surface)
```

The release audit loads the expanded `.dic`/`.aff` pair with `spylls` and
checks both the corpus head and the complete artifact. The LM fixture lists
the 1,000 most frequent Greek-bearing words of the language model (ranked
with a contextual grave read as the acute) in each of the 1,215 spellings the
model writes them with, `καί` and `καὶ` apart (16,546,390 LM tokens). Each
spelling is looked up as written: a keyboard writes a grave only where the
dictionary has the grave spelling, so an acute twin does not stand in for it.
All must be accepted except 30 reviewed Modern spellings (among them the
graves `τὴ` and `στὴν`, which the polytonic Modern Greek list carries),
fragments, numerals, bare stems and corpus misspellings (`ὃτι`, `ἢδη`),
which must remain rejected. The fixture checks all 1,185 remaining spellings,
including their contextual grave variants. The whole-artifact gates additionally
check the 1,363,275 spellings in the reviewed baseline (the export of commit
`2e633cc`, as the keyboard compiled it), with documented exclusions for
unsupported case variants, bare elision fragments, reviewed respellings, and
explicit source errors. The current export has 1,371,104 expanded
spellings. The gates also require all 1,996 valid citation spellings
among the top 2,000 LSJ9 lemmas, and all 2,154 pinned cells of the textbook
paradigms for `λύω`, `παιδεύω`, `τίθημι`,
`δίδωμι`, `ἵστημι`, `τιμάω`, `ποιέω`, and `δηλόω`. They also reject
punctuation, internal breathings, impossible accent placement, synthetic
flagged bases, bare elision stems, and new weak respellings of common
spellings, and require an acute twin for every contextual grave whose acute
spelling passes the structural rules.

The baseline is the dictionary Tonos ships now, not the April 0.4.1 export it
started from: it moves forward with each swap-in, and what it encodes is that
an export must not lose what the keyboard already accepts. Because the gate
strips its structural classes before pinning, a baseline taken from a recent
export grandfathers far less than one taken from April: 18 truncated stems
rather than 8,971, and no missing-breathing forms rather than 166,226. The
reviewed surface is exempt from Dilemma's structural rules, and whatever
still fails them stays until it is reviewed in Tonos, because dropping it
would count as a loss in the gate. The acute twin of a reviewed grave
inherits its review.

`data/hunspell_grc_spelling_review.json` records spellings a review
rejected, which the exporter drops with their grave or acute twins whatever
protects them, and which the audit no longer requires. Two reviewers
labeled each candidate independently, from its attestation by source and
up to three corpus sentences, and a form is listed only where both rejected
it. It covers two groups: respellings of common words that OCR'd text
admitted (191 of them, such as `στρατηγού` for `στρατηγοῦ` and `βοής` for
`βοῆς`), since the respelling guard's thresholds were calibrated on
labeled forms and are not to be tightened by guess; and reviewed-baseline
entries that fail the structural rules and are not words anyone types
(89: editorial underdots, synizesis ties, spurious extra accents such
as `δόκιμάζω`). Two breathing variants both reviewers rejected, `αὗθις`
and `ἄπαντα`, stay, because GLAUx attests them in held-out text. The
exporter also accepts the polytonic Modern Greek particles `θὰ` and `γιὰ`,
which no lemma source proposes.

The reviewed source specifications also preserve `παῖσι` as the old Attic
dative plural explicitly recorded under `παῖς` in LSJ. Its dictionary evidence
is kept separate from mistaken accents in later corpus transcriptions.

A finite reviewed source-form list can preserve a rare spelling beside its
more common form. For example, `ὤοντο` remains beside `ᾤοντο` only while
its exact spelling is present in both cited Diorisis passages and the
merged lexical inventory supports `οἴομαι`. The audit distinguishes the
reviewed grammatical interpretation from the corpus's unknown lemma.
The export requires matching logical content hashes and source hashes in
`form_profile.db` and `form_citations.db` before joining their form IDs;
rebuild both with `build/build_form_attestation.py` when either changes.

A content-hashed fixture measures rejection counts over the New Testament
(Perseus), the Septuagint (First1KGreek), Iliad 1, Herodotus 1, and a
Katharevousa sample. These are regression corpora, not held-out text:
`form_profile.db` contains the same works, so the exact-attestation rule
admits almost all of their forms. Candidate token and type rejection counts,
measured with `spylls`, may not exceed those of the April `.dic`/`.aff` read
the same way. `data/hunspell_grc_shipped_compat.json.gz` records the reviewed
baseline surface; `data/hunspell_grc_textbook.json.gz` pins the textbook
paradigms, taken from `data/ag_verb_paradigms.json` (built by
`build/build_grc_verb_paradigms.py`; the fixture records its SHA-256);
`tests/fixtures/hunspell_heldout.json.gz` records only exact form counts and
source hashes, not source prose. The compatibility fixture is built from files
that ship with Tonos, not with Dilemma: its compiled dictionary and language
model, and its candidate gate with the gate's review lists, each recorded by
SHA-256. Rebuild the fixtures only when deliberately changing the baseline or
source revisions:

```bash
python scripts/build_hunspell_compatibility_fixture.py \
  --baseline /path/to/shipped/grc_polytonic \
  --lm /path/to/shipped/grc_ngram.bin \
  --gate /path/to/candidate_gate.py
python scripts/build_hunspell_textbook_fixture.py
python scripts/build_hunspell_heldout_fixture.py \
  --baseline /path/to/shipped/grc_polytonic \
  --ogc-root /path/to/open-greek-corpus
```

The corpus-head fixture is built from the full `train_lm.py` run behind the
current language model (`--source json`: `vocab.json`, `unigrams.json` and
`stats.json`, 30,933,396 training tokens), not a `train_lm.py --sanity`
output.
JSON fixture regeneration reads `stats.json` and fails unless
`"sanity": false`; the binary route reads the exact vocabulary and unigram
counts embedded by `export_lm.py`. The exporter itself uses
`data/form_profile.db` (`python -m dilemma download --with-attestation`) for
sparse exact-form evidence; `data/hunspell_grc_form_freq.json.gz` is retained
only as an auditable LM artifact and is not an admission boundary.

Sparse inflections can use an exact dictionary POS reading when a flat lookup
self-map hides their defined headword. `data/grc_contracted_verb_aliases.json`
records explicit contracted-verb references separately from independent definitions,
so an attested form under `καταμετρῶ` can retain the defined `καταμετρέω` owner.
The exporter requires one supported target and preserves tied references
without choosing between them; it does not infer a verb from its ending.
Contracted references retain the source spelling's accent; only contextual
grave and acute variants share a key.
An accent-tolerant lookup alias cannot borrow a contracted citation's evidence
when the spelling itself differs from that citation only in its marks.
Build this artifact with `python build/build_grc_contracted_aliases.py --kaikki DIR`
after generating the lexical inventory from those same source extracts.
The export excludes the monotonic lookup aliases of the genitives and datives
of `λιγυστικόν` and `περσικόν`; their circumflex spellings and the acute dual
`περσικώ` remain available.

```bash
# Audit a fresh Hunspell export against the committed full-run fixture.
python scripts/audit_hunspell_frequency.py --source fixture

# Regenerate after a full train_lm.py run; refuses sanity outputs.
python scripts/audit_hunspell_frequency.py --source json --write-fixture

# Or regenerate from a format-v2 binary and its provenance sidecar.
python scripts/audit_hunspell_frequency.py --source binary \
  --lm-binary build/lm/grc_ngram.bin \
  --lm-version build/lm/grc_ngram.version \
  --write-fixture \
  --attestation-out data/hunspell_grc_form_freq.json.gz

# Produce the full LM-weighted review against a previously shipped artifact.
python scripts/audit_hunspell_frequency.py --source binary \
  --lm-binary build/lm/grc_ngram.bin \
  --lm-version build/lm/grc_ngram.version \
  --baseline-dictionary /path/to/shipped/grc_polytonic \
  --comparison-report build/hunspell/baseline_comparison.json
```

Output layout, with one sidecar `.version` file per variant so the
consumer can detect updates:

```
build/hunspell/
  el_GR_monotonic.dic
  el_GR_monotonic.aff
  el_GR_monotonic.version   # semver + dilemma commit hash + entry count
  grc_polytonic.dic
  grc_polytonic.aff
  grc_polytonic.version
  grc_mg_polytonic.dic      # from export_mg_polytonic.py
  grc_mg_polytonic.aff
  grc_mg_polytonic.version
  eval_results.txt          # from eval_hunspell.py
```

`eval_hunspell.py` is the quality gate. It samples mid-frequency real
Greek words (ranks 100..10K in the corpus), generates synthetic typos
at edit distance 1 and 2, and measures top-1/top-5 correction accuracy
against the compact artifact. Add `--compare-full` to also benchmark
Dilemma's own `suggest_spelling()` on the full `lookup.db`.

```bash
python eval_hunspell.py                    # default 500 targets per variant
python eval_hunspell.py --n 100            # quick sanity eval
python eval_hunspell.py --variant el       # only MG
python eval_hunspell.py --compare-full     # compare vs full Dilemma (slower)
```

Requires `pip install spylls` for the Python-side Hunspell consumer.
To regenerate both artifacts and the eval report end-to-end:

```bash
python export_hunspell.py
python eval_hunspell.py --n 200
```

## Boundary rewrites (`grc_morph.json`)

`export_morphology.py` writes `build/hunspell/grc_morph.json`, the table a
keyboard uses to respell a word once the next word turns out to start with a
vowel. `nu` lists the forms that take movable nu (Smyth 134): third-person
singulars in -ε, in -σι, and in -στι (`ἐστί`, its compounds, crasis and
prodelided spellings, but not the Doric `ἐντί`); third-person plurals in -σι,
the subjunctive's included (`ὦσι`, epic `ἐθέλῃσι`); every dative plural in
-σι, -ξι or -ψι, the article's and numerals' as much as a noun's or a
participle's; and adverbs in -σι, and adverbs of place in -θε (Smyth 134 D),
that the texts write with a nu in at least a tenth of their tokens
(`παντάπασιν`, `Ἀθήνησιν`, `πέρυσιν`, `πρόσθεν`, `ὄπισθεν`, but not the
deictic `οὑτωσί` or `εἴθε`). Where a spelling has another reading that cannot take nu,
GLAUx's token counts decide: `φέρε` is an imperative 956 times and an
unaugmented imperfect 45, so it is left out, while `μέλλε`, an imperative 14
times and an imperfect 7, stays, because the imperfect is spelled `μέλλεν` 17
more times before a vowel. A capitalized spelling is weighed with its
lowercase word; a lowercase one only with its lowercase spellings, so a name's
vocative (`Κέλσε`) does not decide the verb (`κέλσε`). Diorisis lists every
candidate analysis on each token, so it shows that a spelling can take nu but
does not vote, and the export refuses a GLAUx pairs file without counts.
An unaugmented imperfect or thematic aorist is spelled like the imperative
(`κατένεγκε`, `ἐντύγχανε`), and the taggers do not tell them apart, so such a
spelling is left out unless the texts write it with its nu somewhere: GLAUx
writes the nu on 99.6% of these past forms before a vowel, so one never written
with it stood there only as the imperative. The augment is read after the
verb's preverbs, from an accent it holds on the penult (`συνεξεῦρε`, Smyth
426), and for a stem that cannot show it (`ἐνεγκ-`) from the augmented spelling
filed under the same verb or its simple verb. A liquid verb's first aorist
(`σήμηνε`) has its imperative in -ον and stays. The aorist optative in -ειε
takes the nu like any third-person singular in -ε (`δόξειεν`); the pluperfect
in -ει is left out, since GLAUx writes it with the nu 11 times in 4,317.
`el` maps a full form to its elided form, ending in U+1FBD. An
elision is written only in the spelling the rules give: the short final vowel
goes, an oxytone throws its accent back onto the new last vowel as an acute
unless it is a preposition, conjunction or enclitic, and every other mark
stays. It is written only when the tagged corpora or the lookup table attest
exactly that spelling among the elided forms of a lemma that claims the full
form, and the value takes the key's case. No entry is made for a final vowel
that cannot elide: a long one, marked or not (η, ω, a circumflex, an iota
subscript or adscript, the Attic accusative in -έᾱ of nouns in -εύς, the
deictic -ί), the words Attic never elides (`ὅτι`, `περί`, `πρό`, `ἄχρι`,
`μέχρι`; Smyth 72), a monosyllable not ending in ε, and the dative -ι and -σι,
which elide only in epic. The tagged corpora say which spellings are datives;
for a form they do not analyze (the lookup table's paradigm expansions, such
as `δμητῆρι`), the forms of its lemma decide: a genitive in -ος beside it, the
uncontracted -εϊ of an s-stem, or a third-declension stem that yields the -σι.
A present participle's dative plural is also the verb's third person plural
(`κεύθουσι`, `τιμῶσι`), so those keep their elision.
`el_modern` holds the Modern Greek elisions in the same `{full: elided}`
shape, read from the polytonic Modern Greek slice as the word list reads it:
39 pairs such as `γιὰ` -> `γι᾽`, `στὸ` -> `στ᾽`, `τώρα` -> `τώρ᾽`,
`ὅλα` -> `ὅλ᾽`, `ποὺ` -> `π᾽`, `εἶναι` -> `εἶν᾽`, `νὰ` -> `ν᾽`, and `κι` ->
`κι᾽`, which the slice writes for `κι` about half the times it precedes a
vowel. An elided spelling qualifies with 10 tokens from 2 authors, and its
full form is the commonest attested word that spells its letters and marks
plus one vowel or vowel digraph (unaccented when the elided spelling keeps
an accent, accented when it has lost it, as an oxytone's does) and is at
least as common; an unaccented preposition's aspirated spelling (`ἀφ᾽`,
`καθ᾽`) is left to the plain one. Only the pairs `el` lacks or spells
otherwise are listed: 51 Modern Greek pairs agree with `el` (`διὰ`, `ἀπὸ`,
`μέσα`, `σὲ`), and one differs (`μιά` is `μί᾽` in `el` by the Ancient rule,
`μι᾽` here). A keyboard writing Modern Greek reads `el` with `el_modern`
laid over it, and one writing Ancient Greek reads `el` alone. The pairs are
kept out of `el` because a keyboard that elides automatically rewrites the
word before every vowel, and Ancient Greek never elides the article `τὸ` or
`ὅτι`.

Modern Greek elides optionally, and most of these elisions are the minority
spelling, so `el_modern_share` gives each `el_modern` key its rate: of the
times the slice writes a word the elided spelling stands for right before a
vowel-initial word, the share it writes it elided (`στ᾽` counts `στὸ` and
`στὰ` together). A word the elided spelling stands for only through a long
ending does not count: the article's `τοῦ` and the dative `τῷ` are not
`τ᾽`. Only `γιὰ` (0.65) and `στὸ` (0.58) pass
half, and `κι` is just under it (0.49); `νὰ` is at 0.29, `τὸ` at 0.20,
`τώρα` at 0.06 and `ὅτι` at 0.01. With automatic elision off, a keyboard
reads the table only to decide whether an apostrophe after a word is an
elision mark. A reader that knows only `nu` and `el` ignores both keys;
`--no-modern-greek` leaves them out, as the slice's parquet is needed to
build them.

`tests/test_export_morphology.py` checks the built table against these rules.
