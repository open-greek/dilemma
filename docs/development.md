# Development

[README](../README.md)

- [Set up a checkout](#set-up-a-checkout)
- [Tests](#tests)
- [Training](#training)
- [Export to ONNX](#export-to-onnx)
- [Shipping rebuilt data artifacts](#shipping-rebuilt-data-artifacts)
- [Release](#release)
- [CI checks](#ci-checks)

Repository build helpers use the `dilemma_build` import namespace, separate
from PyPA's `build` package. Their source scripts and generated outputs remain
under `build/`, so existing commands such as `python build/expand_lsj.py` keep
working alongside `python -m build`. The development extra includes PyPA
`build`, and CI runs the source-validation tests with it installed.

## Set up a checkout

```bash
git clone https://github.com/open-greek/dilemma.git
cd dilemma
python3 -m pip install -e ".[onnx,lm,dev]"
./scripts/install_git_hooks.sh
```

Download the manifest-pinned artifacts for tests without rebuilding them:

```bash
hf download ciscoriordan/dilemma $(python3 scripts/hf_data.py files) --local-dir .
python3 scripts/hf_data.py verify
```

The manifest includes the model and corpus evidence used by CI. A normal
`python3 -m dilemma download --no-tagger` is sufficient for library use, but
it does not fetch all optional inputs needed by the full test suite.

## Tests

Run the fast checks without downloading corpus data:

```bash
python3 -m pytest tests/test_lazy_model_backend.py tests/test_article_self_mappings.py tests/test_pronoun_resolution.py tests/test_builder_namespace.py -q
```

With the manifest artifacts installed, run the full suite and the deterministic
benchmark gate. The dictionary audit requires a fresh Hunspell export first:

```bash
python3 export_hunspell.py --variant grc
python3 scripts/audit_hunspell_frequency.py --source fixture
python3 scripts/run_test_chunks.py
python3 scripts/check_benchmark_regressions.py
```

`scripts/run_test_chunks.py` runs the suite as five pytest processes (four
chunks of test files, then `tests/test_new_features.py` alone), the way CI
does. A single `pytest tests/` process also works but peaks at about 17.5 GB,
more than a 16 GB machine or CI runner has.

Use `--run-slow` for tests that load model weights. Some source-specific tests
need additional corpora and skip when those corpora are absent. See
[CI checks](#ci-checks) for the checks run on GitHub.

## Training

### 1. Build data

Downloads all 5 kaikki dumps and extracts every form-lemma pair from
inflection tables. Non-Greek characters are filtered out.

```bash
pip install -e ".[lm]"                       # build-data deps (lxml, betacode, beta-code, pygtrie)
python build_data.py --download             # downloads + extracts (~1.5GB total)
```

The Cretan and Cypriot sections of Greek Wiktionary are not in those dumps.
Run `python build/extract_el_dialect_sections.py --download --kaikki KAIKKI`
(118 MB raw extract) before `build_data.py` to include them; see
[Modern Greek noun headwords](data.md#modern-greek-noun-headwords).

#### Dump directories

Every builder that reads the kaikki dumps (`build_data.py`,
`build/build_grc_lexical_heads.py`, `build/build_grc_contracted_aliases.py`,
`build/build_treebank_pos_lookup.py`, `build/build_el_verb_heads.py`,
`build/build_verb_subjects.py`, `build/expand_lsj.py`,
`build/expand_sophocles.py`, `overlay_lsj.py`,
`scripts/build_source_alias_manifest.py`) takes `--kaikki DIR` and reads
either layout: flat (`DIR/kaikki.org-en-dictionary-AncientGreek.jsonl`, as
`scripts/archive_kaikki_inputs.py download` writes the pinned snapshot) or
nested (`DIR/en-el/kaikki.org-dictionary-AncientGreek.jsonl`,
`DIR/el/kaikki.org-dictionary-AncientGreek.jsonl`). The names live in
`build/kaikki_dumps.py`. Without `--kaikki`, `$DILEMMA_KAIKKI_DIR` names the
directory (the older `$KAIKKI_DIR` is still read when it is unset) for
`build_data.py`, `build/build_el_verb_heads.py`, `build/expand_lsj.py`,
`build/expand_sophocles.py` and `overlay_lsj.py`. The other builders do not
read it: `build_grc_lexical_heads.py`, `build_grc_contracted_aliases.py` and
`build_verb_subjects.py` require `--kaikki`, and
`build_treebank_pos_lookup.py` without `--kaikki` builds without the
Wiktionary definition check for corpus votes (a different table and audit),
so pass it explicitly. `expand_lsj.py` and `expand_sophocles.py` also take
`--lsj9-dir DIR` (`$DILEMMA_LSJ9_DIR`, default `~/Documents/lsj9`) for
`lsj9_forms.tsv`, `lsj9_headwords.json`, `lsj9_glosses.jsonl` and
`lsj9_glosses_earlier.jsonl` (the glosses file at the earlier lsj9 commit the
pin names, read for principal parts only); the verb pass refuses either
glosses file when it does not match `data/lsj9_glosses_pin.json`.

### 2. Build the lookup database

After extraction, rebuild the SQLite lookup from the generated sources:

```bash
python3 build_lookup_db.py
```

For expanded Ancient and Byzantine paradigms, follow the
[LSJ/Sophocles expansion guide](lexicon-expansion.md) before this step.
POS tables and corpus-attestation data have separate builders described in
the [data guide](data.md) and [API guide](usage.md#pos-aware-disambiguation).
Changing lookup data does not by itself require retraining the transformer.

### 3. Train

Install the training backend, then train on the extracted pairs. Use
`--scale` to control the training size.

```bash
python3 -m pip install -e ".[torch]"
python train.py --scale test                # quick sanity check (20K pairs, ~15 sec)
python train.py --scale full                # all data (default)
python train.py                             # same as --scale full
```

Legacy `--scale 1/2/3` flags are still accepted for compatibility.

### Training scales

Every scale includes **100% of non-standard varieties** (Medieval,
Katharevousa, Cypriot, Cretan, Maniot, Heptanesian, archaic, dialectal).
The remaining budget is split 50/50 between Ancient Greek and standard MG.
Underrepresented tense categories are oversampled to compensate for
their rarity in Wiktionary's paradigm tables, following
[Swaelens et al. (2025)](https://aclanthology.org/2025.findings-acl.399/)'s
finding that perfects are underrepresented in training data relative
to Byzantine text. Aorist forms (3x, critical for stem-changing 2nd
aorist), perfect (3x), future (3x), imperfect (2x), and pluperfect
(5x, rarest at 0.15% of pool) are oversampled proportionally to
their rarity and the degree of stem change from the present form.

The table records the earlier training run; elapsed times depend on hardware.

| Scale | Training pairs | Varieties | AG | SMG | Time (RTX 2080) |
|:-----:|---------------:|----------:|-------:|-------:|:--------------:|
| test | 20K | 9K (100%) | 5.5K | 5.5K | ~15 sec |
| full | 3.5M (all) | 9K (100%) | 1.5M (100%) | 1.7M (100%) | ~95 min |

Models save to `model/{lang}-test/` (test scale) or `model/{lang}/`
(full scale).

Eval accuracy is the model's score on held-out pairs *without* the
lookup table. In practice, the lookup resolves most forms instantly
and the model only handles truly novel words. When the model is used,
beam search generates 4 candidates and the first one that matches a
headword allowed by the citation policy is selected. If none match, the input is returned
unchanged .

### Multi-task learning

When training pairs include POS tags (from Wiktionary) and morphological
features (from GLAUx), the model jointly predicts POS, nominal morphology
(gender/number/case, 45 labels), and verbal morphology (tense/mood/voice,
69 labels) alongside the lemma via auxiliary classification heads on the
encoder output. This follows
[Swaelens et al. (2025)](https://aclanthology.org/2025.findings-acl.399/)'s
finding that multi-task learning (joint POS + morphology + lemma)
improved Byzantine Greek lemmatization by ~9 percentage points. Each
auxiliary loss is weighted at 0.1x relative to the lemmatization loss.
The recorded full-scale run reached 90.4% POS, 81.5% nominal, and 91.2%
verbal accuracy on its held-out set.

Training uses a linear warmup LR scheduler (500 steps warmup, then linear
decay) and gradient clipping (max norm 1.0) for stable convergence.

## Export to ONNX

Generates ONNX model files so inference works without PyTorch.

```bash
python3 -m pip install onnx
python export_onnx.py                  # exports encoder.onnx + decoder_step.onnx
```

## Shipping rebuilt data artifacts

`build_lookup_db.py` fills reviewed missing Modern Greek present-active cells
for `εμπίπτω`, `απορρέω`, and `συνιστώ`, plus the masculine noun paradigm of
`αιτών`. The rules in `dilemma/el_paradigm_gaps.py` require an existing
self-mapped Modern head and never overwrite an existing Modern reading or
infer another tense. The noun readings `αιτούντος` and `αιτούντες` belong to
`αιτών`; they are not relabeled as finite verbs.

For a bounded refresh of an existing release, run
`python scripts/complete_el_paradigms.py --base PATH_TO_BASE --manifest PATH_TO_BASE_MANIFEST`.
The base directory must contain `lookup.db`, `spell_index.db`, and
`mg_lookup.json` whose hashes match that release's manifest. The command uses
the same rules as the full builder, preserves existing combined and Ancient
rows, and writes `data/changes/el_paradigm_completion.json` with input/output
hashes, source references, generator hashes, and every added row. Reversing the
exact refresh with `--undo PATH_TO_AUDIT` restores the original logical rows;
it refuses files changed by subsequent work. A full rebuild remains the
standard path when other lexical inputs change.

A change to the Modern Greek generator reaches the published snapshot the same
way. Run the old and the new `build_data.py --lang el` over the same dumps into
two separate directories, then record and apply their difference:

```bash
python scripts/refresh_el_generator_delta.py diff OLD/data NEW/data \
    --dumps KAIKKI_DIR --old-commit OLD_SHA --new-commit NEW_SHA \
    --input triantafyllidis_headwords.json \
    --out data/changes/el_generator_refresh.json
python scripts/refresh_el_generator_delta.py apply --base BASE_DATA \
    --manifest data/hf_manifest.json \
    --delta data/changes/el_generator_refresh.json --output OUT
```

The delta records both commits, the dumps' checksums, and each changed lookup
key, POS reading, pair, head and participle cell. `apply` checks the base files
against the manifest and makes a change only where the base still holds the
old generator's value; anything else is reported as a conflict and left as
published. Its audit, `el_generator_refresh_audit.json`, lists every change
with the input and output hashes, and `undo --data OUT --audit AUDIT` restores
the base's content. Rebuild `lookup.db` and `spell_index.db` from the refreshed
`mg_lookup.json` afterwards.

`build_data.py` writes into its own checkout's `data/`, so run each side in
a separate clone, with the same dumps and `PYTHONHASHSEED=0`. The dumps are
pinned in `data/kaikki_inputs.json`; restore them into a flat directory with
`scripts/archive_kaikki_inputs.py download --out KAIKKI_DIR` (it verifies
every sha256), check an existing directory with `verify KAIKKI_DIR`, and stage
a new snapshot for the Hub with `stage --src KAIKKI_DIR --out-dir STAGE`
followed, after the upload, by `pin-revision FULL_SHA --out-dir STAGE`. Give
`diff` a `--dumps` directory holding only the three files the Modern build
reads (`kaikki.org-en-dictionary-Greek.jsonl`,
`kaikki.org-el-dictionary-Greek.jsonl`,
`kaikki.org-el-dictionary-GreekDialects.jsonl`), so the record pins exactly
those.

The 1.3.25 change that skips Modern Greek noun and adjective table cells
without a stem of their word (`is_modern_nominal_fragment` in
`dilemma/nominal_header_forms.py`: αρχαΐζων's `ούσης`, `όντων`, `ουσών`,
αισώπειος's `ος` and `ε`, βρόντημα's header letter `γ`) is recorded this way
in `data/changes/el_nominal_fragments.json` (old generator 1.3.24, new
generator the release branch, dumps of the 2026-11-14 snapshot).
`scripts/derive_el_nominal_fragment_delta.py` is a removal-only cross-check
of such a rule for when the generator's dumps are not available: it applies
the rule to the dumps' cells and removes a fragment's readings from the
pinned tables. It cannot see what a full run does besides (a freed key going
to its next source, a POS entry left with one reading being dropped, training
pairs), so write its output to a scratch path and compare.

```bash
python scripts/derive_el_nominal_fragment_delta.py --dumps KAIKKI_DIR \
    --base data --manifest data/hf_manifest.json --out /path/to/check.json
```

The artifacts CI tests against (including `lookup.db`, `spell_index.db`,
`lemma_attestation.json`, `form_profile.db`, and the ONNX/PyTorch files in
`model/combined-s3/`) live on HuggingFace, not in git, and must match
the source commit that uses them. So must every other file the Hub serves to
`python -m dilemma download`. `data/hf_manifest.json` pins their
sha256 and `scripts/hf_data.py` does the bookkeeping:

```bash
python scripts/hf_data.py push -n   # what push would upload and re-pin
python scripts/hf_data.py push      # upload whatever changed, refresh the manifest
git add data/hf_manifest.json       # commit it with the code change
python scripts/hf_data.py check     # checkout, manifest and Hub agree
./scripts/install_git_hooks.sh      # once per clone: pre-push sync check
```

`python -m dilemma download` serves every file under `data/` and `model/`,
which is more than CI downloads, so the manifest has two lists and every
served file belongs in one of them:

| List | In `scripts/hf_data.py` | Holds | Covered by |
|---|---|---|---|
| `files` | `TRACKED` | Everything CI downloads: every table and model file the runtime reads, and the evidence the tests read | `push`, `check`, `wait`, `files`, `verify` |
| `pinned_only` | `PINNED_ONLY` | Served files CI does not download: the Ancient and Medieval build tables (`ag_lookup.json`, `ag_pairs.json`, `med_lookup.json`, `med_pairs.json`, the source-alias audit and manifest), the corpus pair tables (`glaux_pairs.json`, `lbg_pairs.json`, `nt_pairs.json` and the three gap-fill tables), the per-source frequency tables, `ag_verb_paradigms.json`, `vesuvius_index.json.gz`, `med_pos_lookup.json` and the Ancient-only model `model/grc-s3/` | `push`, `check`, `wait` |

A table the runtime reads goes in `TRACKED`, also when git tracks it: an
install reads the Hub copy, so the three Ancient POS tables
(`ag_pos_lookup.json`, `treebank_pos_lookup.json`, `glaux_pos_lookup.json`)
are pinned there and CI downloads them over the checkout. A test reads the
file names out of the runtime package and fails when one is in neither list,
or is only in `PINNED_ONLY`; another compares the pin of every git-tracked
file with the committed blob, so changing such a table fails that test until
`push` has run. `med_pos_lookup.json` is in git as well, but no code reads it
(`build_data.py` writes it beside the Medieval lookup), so it is pinned in
`PINNED_ONLY`.

`push` uploads a changed file of either list and rewrites its pin. It hashes
the `PINNED_ONLY` files the checkout has and keeps the pins of the rest, so a
checkout that rebuilt only the runtime artifacts can still push. `check` also
lists what the Hub serves and fails on a file under `data/`, `model/` or
`tagger/` that is in neither list: a file uploaded by hand must be added to a
list (and pushed) or deleted from the Hub. Two prefixes are exempt, with their
reasons in `UNPINNED`: `data/kaikki_inputs/` (pinned by Hub revision and
sha256 in `data/kaikki_inputs.json`) and `tagger/` (the tagger weights, which
the manifest does not pin yet).

CI waits for the Hub to serve the manifest's bytes, both lists, before
downloading (`hf_data.py wait`) and re-checks the downloaded files afterwards
(`hf_data.py verify`). That way a `git push` that beats a slow upload just
makes the test job wait rather than run new tests against the previous
artifacts, and an upload that never happened fails with a message saying so.
The pre-push hook runs `hf_data.py check` and catches both cases before they
reach CI; `git push --no-verify` skips it.

## Release

Run `scripts/release.py` from a clean `main` synchronized with `origin/main`.
It updates `VERSION`, `pyproject.toml`, and `CHANGELOG.md` together,
commits and pushes `main`, and waits for the Tests workflow on that exact SHA.
Only after CI succeeds does it create and push the numeric annotated tag. The
tag workflow verifies that successful main run, then publishes the GitHub
Release and PyPI artifacts without running the full suite a second time.

Replace `NEW_VERSION` with a semantic version greater than the current `VERSION`.

```bash
python3 scripts/release.py NEW_VERSION --dry-run
python3 scripts/release.py NEW_VERSION
```

Before bumping, regenerate and HF-upload any data outputs the new
version is meant to ship (see the section above), so the release tag
points at a SHA whose data files match what's on HuggingFace.

## CI checks

Tests run automatically on GitHub-hosted runners for pushes and pull requests
to `main`. Startup, article and pronoun recognition, and build-import
regression tests run before artifact downloads, using temporary test data.
CI downloads data
files from HuggingFace (`lookup.db`, `spell_index.db`,
`lemma_attestation.json`, and the matching `form_profile.db` and
`form_citations.db` pair for the Hunspell export,
plus source tables including the elision-preserving `diorisis_pairs.json`)
and the complete `combined-s3` model, all pinned by sha256 in `data/hf_manifest.json`.
ONNX Runtime and PyTorch are installed, and model/data integrity and real-model
batched-decoding checks run before the full suite so missing or incompatible
model files fail early. The Ancient
export and coverage audit also run first, using the pinned merged corpus
frequency table. The exporter creates its output directory before writing
audit files, including on a clean checkout. The job has a 75-minute limit to
cover both the full dictionary export and the test suite
(see [Shipping rebuilt data artifacts](#shipping-rebuilt-data-artifacts)).

```bash
python -m pytest tests/ -v                  # run all tests via pytest (recommended)
python scripts/check_benchmark_regressions.py  # deterministic accuracy gate
python bench_fast.py                      # compact comparison across three periods
python tests/test_integrity.py              # data integrity + model inference checks
python tests/test_dilemma.py                # lookup table + end-to-end lemmatization tests
python tests/test_dilemma.py --lookup-only  # skip model tests
```

The compact benchmark counts missing predictions as errors and continues scoring
the remaining forms.

The suite covers the runtime, source builders, exports, and their regression fixtures.
`tests/test_comprehensive.py` is one of the largest single files (~170 tests)
and covers core lemmatization, particle suffix stripping, verb
morphology stripping, article-agreement disambiguation, crasis
resolution, elision handling, orthographic normalization, dialect
normalization (Ionic, Doric, Aeolic, Koine), convention switching,
language filtering, spelling suggestions, batch operations, and
edge cases. The remaining files cover
paradigm builders, athematic verb expansion, LSJ principal-parts
extraction, the `morph_diff` annotator, and end-to-end integrity
checks.

`tests/test_integrity.py` runs 7 structural checks: ONNX/vocab dimension
match, DB table presence, model load, inference, and ONNX/PyTorch
parity. `tests/test_dilemma.py` validates lookup correctness and known
form-lemma pairs across Greek varieties.
