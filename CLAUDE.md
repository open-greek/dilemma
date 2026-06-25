# Dilemma - Project Instructions

## Data Storage

Large data files and models are stored on HuggingFace at `hf.co/ciscoriordan/dilemma` (public model repo), not in git. This includes lookup.db, all *_lookup.json/*_pairs.json/*_freq.json files, lemma_attestation.json, form_profile.db, form_citations.db, spell_index.db, vesuvius_index.json.gz, and ONNX/PyTorch model files. Download with `huggingface-cli download ciscoriordan/dilemma --local-dir . --include "data/*" "model/*"`. Both form-attestation DBs are excluded from the base `python -m dilemma download`: `--with-attestation` pulls form_profile.db (gate + usage distribution), `--with-citations` additionally pulls the large form_citations.db (example loci) and implies the profile.

## Build Pipeline

### Lookup DB rebuild order
1. `build_data.py --kaikki kaikki` - extract Wiktionary data from kaikki dumps
2. `build/expand_lsj.py --expand` - expand LSJ noun paradigms via Wiktionary Lua modules
3. `build/expand_lsj.py --expand-verbs` - expand LSJ verb paradigms
4. `build_lookup_db.py` - build SQLite lookup from JSON files
5. `train.py --lang all --scale 3` - retrain transformer
6. `export_onnx.py` - export ONNX models

### raw_lookups.db vs JSON priority
`build_lookup_db.py` loads from both `raw_lookups.db` (SQLite, from `build_data.py`) and JSON files, and prefers whichever has more entries. This means `ag_lookup.json` with LSJ expansion (~9.97M AG) automatically wins over `raw_lookups.db` (~2.36M AG base Wiktionary). No need to manually delete `raw_lookups.db` before rebuilding.

### Concurrent build_lookup_db.py
Never run multiple `build_lookup_db.py` instances simultaneously. They corrupt the SQLite output. Kill any stale processes before rebuilding.

## Corpus attestation (lemma-keyed)

`build/build_lemma_attestation.py` reads GLAUx (`~/Documents/glaux/`) and Diorisis (`data/diorisis/xml/`) directly and emits `data/lemma_attestation.json`: for each AG lemma, token counts stratified by source, genre (the 10 corpus_freq bins), century (signed; -8 = 8th c. BC), and dialect (GLAUx only), plus `dominant_pos`. This is a standalone corpus-aggregation pass (~3 min, 131K lemmas, 17.5M deduped tokens) - it does NOT need the lookup-db build or transformer retrain, just the corpora on disk. Only GLAUx + Diorisis are used (the other corpus_freq sources lack clean date/dialect metadata). GLAUx and Diorisis annotate largely the same texts, so the frequency (total + by_*) is deduplicated at the work level by TLG id (each work counted once, GLAUx preferred); totals are a union, not a sum. `source_counts` separately keeps each lemmatizer's full independent count (overlapping, not summed) as a confidence/recall signal - this is dedup AND multi-source, not either/or. Generalizes to N sources (add a key); token-level ROVER voting would be a per-work alignment pass layered on top (the corpora are ~98% but not token-identical, so alignment is non-trivial). Output is byte-stable (sorted keys, SHA-256 source hashes in `_meta`, no wall-clock). It is keyed by surface FORM nowhere - keys are the corpora's own NFC polytonic lemma annotation, restricted to lexical Greek. Distinct from `corpus_freq.json`, which is form-keyed and genre-only. Public API: `Dilemma.attestation(lemma) -> dict | None`. Run `python -m pytest tests/test_attestation.py` after changes.

## Form attestation ("attested only" + citations)

`build/build_form_attestation.py` is the surface-FORM sibling of the lemma-keyed attestation pass. It reads GLAUx + Diorisis (lemmatized treebanks) plus First1KGreek, PTA, the byzantine-vernacular corpus, and the Patrologia Graeca (raw text: TEI via `build/tei_locus.py`, PG via calfa-co's `$0`/`$8`/`$9` volume/page markers), and emits two SQLite artifacts (both opt-in downloads). `data/form_profile.db` (~207 MB, `--with-attestation`) holds, per exact NFC polytonic form, the keys (`form`, grave/case-folded `form_norm`), a deduped usage distribution (`total_count`, `source_counts`, `by_century`, `by_genre`, `by_dialect`, a `century x genre` joint for the heatmap, `dominant_pos`), and a `works` table. `data/form_citations.db` (~443 MB, `--with-citations`) holds up to `--cap` (default 200) example loci per form. Like the lemma pass, totals are deduped at the work level (GLAUx preferred) so `total_count != SUM(citations.count)`; citations keep both sources. Diorisis forms are Beta Code, converted via `beta_to_nfc` from `extract_diorisis_lm.py` (matches GLAUx's U+2019 elision mark). GLAUx loci are the single finest present `div_*`/`line` value (the attributes are cumulative); Diorisis loci are sentence-granular. Reproducibility is asserted via a logical content hash (`meta.content_hash`), not file bytes. It is a standalone pass (no lookup-db build, no retrain); run on the local Mac where the corpora live. Run `python -m pytest tests/test_form_attestation.py` after changes.

The runtime lives in `dilemma/_attest_db.py` (pure stdlib, no torch; the single source of truth for the three key functions, imported by the builder, core, and paradigm). Public API:
- `Dilemma.form_attestation(form, *, max_citations=20) -> dict | None` (form-keyed sibling of `attestation(lemma)`).
- Input gate: `lemmatize` / `lemmatize_verbose` / `lemmatize_batch` / `lemmatize_pos` / `lemmatize_batch_pos` take `attested_only=True` (exact NFC, via elision expansion; gated words yield `None` / `[]`). `lemmatize_verbose(..., with_attestation=True)` attaches each candidate's matched-form attestation.
- Output gate: `paradigm.generate` / `generate_paradigm` take `attested_only=True` (grave/case-folded match, filter-then-pick across the source precedence) and `with_attestation=True`. `fill_canonical_dict` is intentionally NOT gated (Klisy wants full coverage).

Multi-source dedup: each work is counted once in `total`/`by_*` by source priority (glaux > diorisis > first1k > pta > pg > byz), keyed by TLG id; lower-priority sources of a claimed work contribute only `source_counts` + citations. TLG-bearing TEI works inherit GLAUx's century/genre/dialect; others use a per-source genre bucket (first1k=other, pta/pg=religion, byz=poetry) plus a composition century from the PTA `<creation>` header, the byzantine `manifest.json`, or PG's per-volume `PG_CENTURY` map. PG is keyed by Migne volume (`work_id` like `PG151`, locus = Migne page), has no TLG id so it never dedups against PTA (its patristic overlap is kept as independent evidence), and is noisy 19th-c. OCR - so a form attested ONLY in `pg`/`first1k` is lower-confidence; check `source_counts` to require a treebank (glaux/diorisis) source. `--sources` selects a subset; canonical-greekLit (classical, ~fully overlapping GLAUx/Diorisis) is not in the default set; the calfa-co PG corpus lives at `~/Documents/corpora/PG`. Memory is bounded by streaming citations to a temp SQLite `_raw` table (the `CiteSink`), not a Python list.

## Benchmarks

### Convention matching
Each benchmark must run with the correct convention:
- AG Classical, Katharevousa, Byzantine: `convention='wiktionary'` (default)
- Demotic MG: `convention='triantafyllidis'`, `lang='el'`
- HNC MG: `convention='triantafyllidis'`, `lang='el'` + equivalences for HNC-specific conventions

Running Demotic with wiktionary convention gives ~83% instead of ~96% due to convention mismatch (AG lemma forms like σπήλαιον vs MG σπήλαιο).

### bench_fast.py
Runs AG Classical, Katharevousa, and Demotic with correct conventions. Use this for quick validation after any pipeline change.

## Pipeline fixes that don't work

### Corpus self-map overrides (reverted)
Replacing Wiktionary self-maps with corpus evidence (GLAUx/PROIEL/Gorman/Diorisis) proved too aggressive. It overrides correct headword entries with corpus convention preferences (e.g., δεῖ -> δέομαι instead of δέω). Use targeted manual overrides in `_LOOKUP_OVERRIDES` for known bugs instead.

### Corpus consensus overrides (reverted)
Using 2+ corpus agreement to override Wiktionary entries also proved too aggressive for the same reasons.

## What does work

### GLAUx/Diorisis lemma validation
Validating corpus lemmas against the AG headword set (`ag_headwords_exact`) filters out annotation errors (Ἔσθι, corrupt -δήποτε forms). This is safe and should always be enabled.

### Proper noun confidence lowering
In `build_data.py`, proper noun entries (`pos="name"`) get lower confidence for stripped/unaccented keys, so common verbs beat place names (φασιν -> φημί, not Φᾶσις).

## Training

- Use `--lang all` (not `--lang combined`, which doesn't exist)
- Two scales: `--scale test` (20K pairs, ~15 sec sanity check) and `--scale full` (all data, default). `--scale 3` is a legacy alias for full.
- Training on RTX 4090: ~35 min for 3 epochs, RTX 2080 Ti: ~95 min
- The model trains on pairs from `build_data.py`, not from the lookup table directly

## Testing

- ~1,000 tests across the files in `tests/`
- Tests run on self-hosted GitHub Actions runner on CORSAIRONE (WSL2)
- `python -m pytest tests/ -x -v` to run locally
- Always run tests after any pipeline change before committing

## morph_diff annotator

`dilemma/morph_diff.py` is a small, pure-Python stem-change annotator. Given
`(lemma, form, lang)` it returns a `MorphDiff` whose `roles` array
classifies each NFC code point of the form as one of STEM, ENDING, AUGMENT,
REDUPLICATION, IRREGULAR_STEM, or UNCHANGED.

```python
from dilemma import diff_form, diff_paradigm, MorphDiff, Role

diff_form("γράφω", "ἔγραψα", lang="grc")
# MorphDiff(form="ἔγραψα", roles=[AUGMENT, STEM, STEM, STEM,
#          IRREGULAR_STEM, ENDING], irregular_indices=[0, 4],
#          stem_change=True, ending_start=5)
```

Use it when you need a millisecond-fast per-character diff between a
citation form and an inflected form (e.g. for a flashcard UI that
highlights irregular root changes). It is intentionally pure: no lookup
DB queries, no model loads, no torch / onnxruntime imports. Heuristic
rules cover the common AG / MG cases of syllabic and temporal augment,
Cε- reduplication (with φ/θ/χ -> π/τ/κ deaspiration), and basic stem
allomorphy. What the heuristic cannot align via LCS falls into
IRREGULAR_STEM.
