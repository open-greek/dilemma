# Evaluation and limitations

[README](../README.md)

- [Reading the results](#reading-the-results)
- [Multi-period benchmarks](#multi-period-benchmarks)
- [Elided forms (AGDT Iliad)](#elided-forms-agdt-iliad)
- [Held-out gold: the Gorman treebanks](#held-out-gold-the-gorman-treebanks)
- [Dilemma vs the TLG lemmatizer](#dilemma-vs-the-tlg-lemmatizer)
- [Rare vocabulary coverage](#rare-vocabulary-coverage)
- [DiGreC treebank](#digrec-treebank)
- [HNC Modern Greek](#hnc-modern-greek)
- [Related work](#related-work)
- [Ambiguity and citation conventions](#ambiguity-and-citation-conventions)

## Reading the results

The comparison tables preserve earlier benchmark runs. They have not been
rerun for every package release and do not establish the accuracy of current
third-party packages. The release
passed CI's [deterministic regression gate](../scripts/check_benchmark_regressions.py),
which uses `guess=False`, checks the gold-file hashes, and rejects decreases
in strict or equivalence-adjusted correctness on the compact Classical,
Katharevousa, and Demotic fixtures. Those fixtures also have lookup overlap;
they are regression checks, not an unseen-text accuracy estimate.

Broader evaluations below have different sample sizes and source overlap.
Recognition, citation-form validity, and agreement with a gold lemma measure
different things. Keep those measures separate when comparing results.

## Multi-period benchmarks

The tables below retain the earlier full comparison across tools and
conventions. The 1.3.7 artifact recheck with `bench_fast.py` scores
99.7% on Classical Greek and 95.0% on Katharevousa with `all`/
`wiktionary`, and 96.0% on Demotic with `el`/`triantafyllidis`.

Equiv-adjusted accuracy across four periods of Greek. All tools
evaluated with the same normalization (case-folded, accent-stripped)
and lemma equivalence groups (see `data/benchmarks/bench_all.py`).

**Test sets:**
- **AG Classical**: Sextus Empiricus, *Pyrrhoniae Hypotyposes* 1.1-1.8 (357 tokens, [First1KGreek](https://opengreekandlatin.github.io/First1KGreek/), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)). Not in any UD treebank or Gorman.
- **Byzantine**: [Swaelens et al. (2024)](https://aclanthology.org/2024.lrec-main.899/) DBBE gold standard (8,342 tokens of unedited Byzantine epigrams, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Katharevousa**: Konstantinos Sathas, *Neoelliniki Filologia* (1868), biography of Bessarion (318 tokens, [el.wikisource.org](https://el.wikisource.org/), [public domain](https://en.wikipedia.org/wiki/Public_domain)).
- **Demotic MG**: Greek Wikipedia articles "[Σπήλαιο Πετραλώνων](https://el.wikipedia.org/wiki/Σπήλαιο_Πετραλώνων)" and "[Ελαιόλαδο](https://el.wikipedia.org/wiki/Ελαιόλαδο)" (400 tokens, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)). Not in any MG treebank. A separate dev set (251 tokens from "[Μέλισσα](https://el.wikipedia.org/wiki/Μέλισσα)" and "[Σαμοθράκη](https://el.wikipedia.org/wiki/Σαμοθράκη)") is also included at `data/benchmarks/demotic_dev.txt`.

| Tool | AG Classical | Byzantine (literary) | Katharevousa | Demotic MG |
|------|:--------:|:--------:|:--------:|:--------:|
| [spaCy](https://spacy.io/) `el` | -- | 31.7% | 44.6% | 79.9% |
| [stanza](https://stanfordnlp.github.io/stanza/) `el` | -- | 37.4% | 48.4% | 87.0% |
| [Swaelens et al. (2024)](https://aclanthology.org/2024.lrec-main.899/) | -- | 65.8% | -- | -- |
| [CLTK](https://github.com/cltk/cltk) | 81.2% | 66.6% | 74.8% | -- |
| [Morpheus](https://github.com/perseids-tools/morpheus-perseids-api) (oracle) | -- | 71.1% | -- | -- |
| [stanza](https://stanfordnlp.github.io/stanza/) `grc` | 92.2% | 71.3% | 85.2% | -- |
| [Swaelens et al. (2025)](https://aclanthology.org/2025.findings-acl.399/) | -- | ~74-75% | -- | -- |
| **Dilemma** (best convention per period) | **99.7%** | **89.2%**‡ | **94.7%** | **94.8%**† |

<sub>†For Demotic MG, `lang="el"` with `triantafyllidis` (94.8%) matches `lang="all"` with `triantafyllidis`. For MG-only workloads, `lang="el"` with `triantafyllidis` is recommended since it avoids AG lemmas (e.g. σπήλαιον) being returned for MG words that have an AG lookalike. ‡Byzantine is 89.2% from the static lookup alone (no POS); with per-token POS (gold POS in this measurement) it reaches 91.8%, since much of the residual is genuine syncretism (θεῶ = dat. θεός vs θεάομαι) that only context resolves.</sub>

Cells marked `--` indicate the tool doesn't support that period or
wasn't tested. Morpheus "oracle" picks the best candidate from all its
analyses, representing the ceiling for rule-based morphology. (The TLG
lemmatizer is compared separately [below](#dilemma-vs-the-tlg-lemmatizer),
since it reports a different metric and no per-era breakdown.)

**Dilemma detail by convention:**

| Lang | Convention | POS | AG Classical | Byzantine (literary) | Katharevousa | Demotic MG |
|------|------------|-----|:--------:|:--------:|:--------:|:--------:|
| `all` | `wiktionary` (default) | -- | 99.7% | 89.2% | 94.7% | 79.5%* |
| `all` | `wiktionary` (default) | gold | -- | 91.8% | -- | -- |
| `all` | `triantafyllidis` | -- | 87.7% | 79.3% | 89.6% | 94.8%† |
| `grc` | `wiktionary` (default) | -- | 99.7% | 88.7% | 94.7% | 79.5%* |
| `grc` | `triantafyllidis` | -- | 91.0% | 83.3% | 91.8% | 89.2% |
| `el` | `wiktionary` (default) | -- | 95.2% | 84.5% | 91.5% | 90.0% |
| `el` | `triantafyllidis` | -- | 87.7% | 79.3% | 89.6% | 94.8% |

<sub>\*Demotic MG scores with `wiktionary` convention are convention mismatches, not real accuracy gaps: AG citation forms like `σπήλαιον` don't match the MG gold standard `σπήλαιο`. Using `convention="triantafyllidis"` fixes this. The AG and Byzantine benchmarks run with `resolve_articles=True` (the correct setting for gold that lemmatizes articles, e.g. τὸν -> ὁ); with resolution disabled, marked Ancient articles now retain their own forms (see [article behavior](usage.md#basic-usage)).</sub>

`lang="all"` searches both AG and MG lookup tables for every token.
The `wiktionary` convention outputs polytonic AG citation forms. The
`triantafyllidis` convention outputs monotonic MG lemma forms and is
the recommended setting for Modern Greek text (see
[Conventions](usage.md#conventions)).

POS column: `--` means Dilemma disambiguates on its own (default).
`gold` means gold-standard POS tags from the dataset are fed in. Only
DBBE provides gold POS; the +2.6pp gain there (89.2% -> 91.8%) is
genuine syncretism that only context can resolve (θεῶ = dat. θεός vs
θεάομαι), which requires an appropriate POS reading; the gold-POS score is not a measurement of predicted tags.

The eval scripts (`eval/eval_dbbe.py`, `eval/eval_digrec.py`,
`eval/eval_hnc.py`, `eval/bench_dbbe.py`, `eval/eval_iliad_elision.py`,
`eval/eval_gorman_gold.py`, `eval/eval_ptnk.py`) provide per-POS
breakdowns and error categorization. `eval_ptnk.py` measures Koine/LXX
gold agreement against the manually lemmatized UD_Ancient_Greek-PTNK
Septuagint treebank (96.8% with PTNK's lemma conventions neutralized
locally; its text, though not its annotation, overlaps dilemma's
corpora).

## Elided forms (AGDT Iliad)

Elided forms can leave several plausible expansions. This evaluation
separates raw annotation agreement from a reviewed adjustment for
elided `ὅτ'` and `ὅθ'`.
Measured over every elision-marked token in the AGDT Iliad treebank
(10,312 tokens, `eval/eval_iliad_elision.py`, point `ILIAD_TREEBANK`
at your AGDT `tlg0012.tlg001` file):

| | agreement with AGDT gold |
|---|:--:|
| raw | 99.0% |
| counting elided ὅτ'/ὅθ' -> ὅτε as correct | 99.3% |

The adjusted score treats 35 AGDT `ὅτ'`/`ὅθ'` tokens annotated as `ὅτι`
as expansions of `ὅτε`. The raw score retains the source annotations. The result
is independent of which apostrophe codepoint the text uses (U+2019,
U+1FBD, U+02BC, ASCII, or combining U+0313). Remaining disagreements include a handful of locative ὅθ' (= ὅθι) and τόθ' (= τόθι)
readings, and AGDT tokenization quirks (οὐ + δ' with gold οὐδέ on the
δ' token).

## Held-out gold: the Gorman treebanks

The [Gorman treebanks](https://github.com/perseids-publications/gorman-trees)
(Vanessa Gorman, CC BY-SA 4.0; 554K hand-annotated tokens across 18
classical prose authors) are deliberately never ingested into any
Dilemma artifact - not the lookup, not the POS tables. They serve as
the project's held-out gold standard, so agreement with their lemma
annotation is measured by `eval/eval_gorman_gold.py`, using the equivalence-adjusted
scoring described above. `build/nc_filter.py` also excludes GLAUx sentences
annotated from
Gorman's trees. The benchmark spans oratory, history, philosophy, and Ionic
prose. The table records the earlier evaluation run.

| Author (tokens) | Gold agreement |
|---|:--:|
| all 18 authors (554,440) | **94.0%** |
| Herodotus, Ionic (29,327) | 92.2% |
| Thucydides (28,923) | 93.6% |
| Xenophon, *Cyropaedia* (36,116) | 92.4% |
| Polybius (95,531) | 94.6% |
| Diodorus (23,159) | 96.2% |

The residual is dominated by lemmatization-convention differences
(κεῖνος vs ἐκεῖνος, plural ethnonym lemmas, voice conventions), the
same class the equivalence table cannot fully absorb.

## Dilemma vs the TLG lemmatizer

The [Thesaurus Linguae Graecae](https://stephanus.tlg.uci.edu/history.php)
(TLG) provides a diachronic Greek lemmatizer within its corpus search service.
It is proprietary, bound to the subscription TLG corpus and search engine,
and cannot be run on external text. Pantelia (2022) reports a
*recognition* rate (the share of wordforms its lexicon + paradigm analyzer
can identify): 98.4%, including Byzantine and early-modern forms (Pantelia
2022). That figure does not measure lemma disambiguation.

The recorded Dilemma recognition run measures the share of wordforms it resolves via lookup + deterministic rules,
with neural inference excluded. It used 24.4M First1KGreek word tokens
(Ancient and Patristic texts). First1KGreek also supplies frequency and
attestation data, so this is a recognition measurement on those texts, not
a held-out lemma-disambiguation test:

| | TLG lemmatizer | Dilemma |
|---|:--:|:--:|
| Recognition rate (wordforms analyzed) | 98.4%¹ | 98.8%² |
| Lemma accuracy (correct lemma chosen) | not published | 94.6%³ |
| Era coverage | ancient → Byzantine (to ~1669) | ancient → Byzantine → Modern (incl. Demotic) |
| Open source | No | Yes |
| Runs on arbitrary text | No (TLG corpus only) | Yes |

<sub>¹ Recognition over the TLG's own subscription corpus (Pantelia,
*Thesaurus Linguae Graecae: A Bibliographic Guide to the Canon of Greek
Authors and Works*, UC Press, 2022). ² Dilemma lexicon + rules only (no
neural model, to match TLG), token-level over 24.4M First1KGreek
tokens (Dilemma normalizes lunate sigma ϲ→σ/ς internally, since these
editions use it throughout). Unrecognized tokens include editorial sigla, Greek numerals, and
abbreviations (γρ, ΙΙ, κζ). A model prediction does not, by itself, establish
that an unfamiliar spelling has been recognized correctly. ³ Unweighted mean of the four per-period accuracy
scores above (token-weighted, 90.0% - the DBBE epigrams dominate the
token count) - a metric TLG does not publish.</sub>

The recognition percentages come from different corpora and are not a
controlled comparison. Dilemma's recorded First1KGreek run is useful as a
coverage check, not as evidence that it disambiguates lemmas as accurately as
another system.

## Rare vocabulary coverage

Following [SIGMORPHON](https://sigmorphon.github.io/) shared task
methodology for out-of-vocabulary evaluation, we take every distinct
non-capitalized word of a text, stratify by corpus frequency, and check
whether the output lemma is a valid LSJ/Wiktionary headword
(`bench_rare.py`). This checks whether rare forms produce a recognized citation, not whether
the selected lemma is correct in context.

| Text | Period | Morpheus | Stanza | Dilemma |
|------|--------|:--------:|:------:|:-------:|
| Xenophon, *Cyropaedia* | Attic | 99.5% | 84% | **99.6%** |
| Kresadlo, *Astronautilia* 13 | Epic | 74% | 74% | **84%** |
| Herodotus, *Histories* | Ionic | 99.5% | 88% | **99.3%** |

<sub>Herodotus is measured over the full *Histories* text
([Perseus canonical-greekLit](https://github.com/PerseusDL/canonical-greekLit)
TEI, tlg0016.tlg001, [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/);
24,862 distinct non-capitalized forms). The figure holds across the
frequency range: 99.6% for high-frequency forms down to 99.0% for
forms attested 1-9 times, and 64/64 corpus-unseen forms. The
Cyropaedia and Astronautilia rows and the Morpheus/Stanza columns are
pinned to their original 2026-03 measurement. For gold-match accuracy
against hand-annotated treebanks (including Cyropaedia at 92.4% and
Herodotus at 92.2%, both genuinely held out), see the
[Gorman held-out gold benchmark](#held-out-gold-the-gorman-treebanks).</sub>

## DiGreC treebank

On the [DiGreC treebank](https://github.com/mdm33/digrec) (Homer
through 15th century Byzantine Greek), Dilemma reaches 93.3%
equiv-adjusted (89.3% strict) over 87,672 tokens. The gap accounts for
convention differences between annotation schemes (e.g.
`εἶπον`/`λέγω`, `ἐγώ`/`ἡμεῖς`). DiGreC's Herodotus, New Testament,
Septuagint, and Sphrantzes sections derive from the PROIEL treebank
(CC BY-NC-SA), which this project avoids entirely, evaluation
included, so those sections are excluded
(`eval/eval_digrec.py --exclude-author ...`).

## HNC Modern Greek

`eval_hnc.py` evaluates against the
[HNC Golden Corpus](https://inventory.clarin.gr/corpus/870) (88K tokens
of gold-standard Modern Greek from CLARIN:EL).

## Related work

- [Vatri and McGillivray (2020)](https://brill.com/view/journals/jgl/20/2/article-p179_4.xml)
  evaluate Ancient Greek lemmatization methods.
- [Swaelens et al. (2024)](https://aclanthology.org/2024.lrec-main.899/)
  examine lemmatization of unedited Byzantine epigrams and the difficulties
  caused by inconsistent orthography.
- [Swaelens et al. (2025)](https://aclanthology.org/2025.findings-acl.399/)
  study joint lemmatization and morphological analysis, reporting improvements
  from multi-task training on the DBBE gold standard.
- [Celano (2025)](https://aclanthology.org/2025.lm4dh-1.5/)
  compare morphosyntactic parsers and lemmatizers using the Ancient Greek
  Dependency Treebank annotation scheme.

## Ambiguity and citation conventions

Some disagreements reflect genuine ambiguity or different dictionary
conventions. The token counts below belong to the recorded evaluation sets.
Report incorrect behavior in [GitHub issues](https://github.com/open-greek/dilemma/issues);
source corrections enter released artifacts after extraction and rebuilding.

| Issue | Tokens | Notes |
|-------|--------|-------|
| **αὐτοῦ ambiguity** | ~200 | Genuine lexical ambiguity: both an adverb ("here/there") and genitive of αὐτός. Resolved when POS context is available via `lemmatize_pos()`. |
| **μιν → ὅς** | ~340 | Convention difference. Wiktionary maps μιν to the 3rd person pronoun. Perseus treebank uses μιν as its own lemma. |
| **Lemma convention differences** | ~400 | αὐτάρ vs ἀτάρ, κε vs ἄν - Wiktionary and Perseus use different citation forms for some Homeric particles. Handled by lemma equivalence groups for evaluation. |
