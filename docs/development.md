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
python3 -m pytest tests/ -x -v
python3 scripts/check_benchmark_regressions.py
```

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

The artifacts CI tests against (including `lookup.db`, `spell_index.db`,
`lemma_attestation.json`, `form_profile.db`, and the ONNX/PyTorch files in
`model/combined-s3/`) live on HuggingFace, not in git, and must match
the source commit that uses them. `data/hf_manifest.json` pins their
sha256 and `scripts/hf_data.py` does the bookkeeping:

```bash
python scripts/hf_data.py push      # upload whatever changed, refresh the manifest
git add data/hf_manifest.json       # commit it with the code change
./scripts/install_git_hooks.sh      # once per clone: pre-push sync check
```

CI waits for the Hub to serve the manifest's bytes before downloading
(`hf_data.py wait`) and re-checks them afterwards (`hf_data.py verify`).
That way a `git push` that beats a slow upload just makes the test job
wait rather than run new tests against the previous artifacts, and an
upload that never happened fails with a message saying so. The pre-push
hook catches both cases before they reach CI; `git push --no-verify`
skips it.

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
