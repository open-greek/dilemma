# How Dilemma works

[README](../README.md)

- [Pipeline overview](#pipeline-overview)
- [Lookup table](#lookup-table)
- [Rule-based fallback layer](#rule-based-fallback-layer)
- [Dialect normalization](#dialect-normalization)
- [Orthographic normalizer](#orthographic-normalizer)
- [Transformer model](#transformer-model)
- [Batching and caching](#batching-and-caching)
- [Earlier byte-level model](#earlier-byte-level-model)

Dilemma combines dictionary lookups, grammatical rules, and a character-level
model. The lookup covers known spellings; the rules recover readings from
systematic transformations; the model proposes lemmas for unresolved forms.
The exact order is implemented in [core.py](../dilemma/core.py).

<p align="center">
  <img src="https://raw.githubusercontent.com/open-greek/dilemma/main/diagram.svg" width="700" alt="Dilemma components">
</p>

## Pipeline overview

Closed-class forms, crasis, numerals, and elisions are handled before ordinary
lookup. Exact spellings take priority over progressively normalized keys.
Particle and verb-morphology rules and optional dialect/orthographic
normalization then try further readings. Neural inference runs when the
previous steps have not resolved a word and `guess=True`.

Compound decomposition and additional Byzantine normalization can also run
when the model leaves a form unresolved. These fallbacks remain available
with `guess=False`; that setting skips neural inference and suppresses the
final unchanged-word result. Scalar, batch, and verbose methods share these
rules but return different amounts of candidate metadata.

Spelling suggestions use the separate `suggest_spelling()` API. They are not
an automatic edit-distance correction pass over every lemmatized word.

## Lookup table

The SQLite lookup combines dictionary paradigms and corpus readings. It tries
exact spelling, lowercase, grave-to-acute, monotonic, and accent-stripped keys.
Ancient-only and Modern-only tables help preserve the requested citation
convention when spellings overlap. The JSON files are a fallback when the
SQLite database is absent.

See [Sources and lookup construction](data.md) for the input inventories,
merge priorities, homograph handling, and validation evidence. Lookup data
comes from multiple sources; generated paradigms and corpus annotations can
contain errors and do not replace independent lexical evidence.

## Rule-based fallback layer

- **Particles:** recognize suffixes such as `-περ`, `-γε`, `-δε`, and deictic
  `-ι`, then look up the remaining form.
- **Verb morphology:** remove a supported augment or reduplication and check
  the resulting stem. Grammatical evidence prevents a noun homograph from
  being used as a verb stem.
- **Elision and crasis:** expand abbreviated forms such as `δ’` or recover the
  components of `κἀγώ`. Multiple supported readings can remain ambiguous.
- **Compounds:** try supported prefixes and linking vowels, then look up the
  base. Reconstruction must retain the grammatical reading of that base.

## Dialect normalization

For Ancient Greek dialect texts (Herodotus, Pindar, Sappho, etc.),
the normalizer maps dialect-specific forms to their Attic equivalents
so the Attic-heavy lookup table can match them.

**Ionic** (highest coverage): η/ᾱ alternation after ε, ι, ρ
(ἱστορίης → ἱστορίας), uncontracted vowels (ποιέειν → ποιεῖν,
τιμέω → τιμῶ), κ/π interrogative interchange (κῶς → πῶς,
ὅκου → ὅπου), σσ/ττ alternation (θάλασσα → θάλαττα),
ρσ/ρρ alternation (θάρσος → θάρρος), and common word mappings
(μοῦνος → μόνος, ξεῖνος → ξένος, κεῖνος → ἐκεῖνος).

**Doric**: ᾱ/η alternation (Ἀθάνα → Ἀθήνη), word mappings
(ποτί → πρός, τύ → σύ), Doric futures (-σέω → -σω).

**Aeolic**: smooth breathing mapped back to Attic rough breathing (Aeolic systematically drops the rough breathing, a feature known as psilosis).

**Koine**: σσ/ττ alternation (overlaps with the Ionic rule above and with the period-specific normalization rules in the orthographic normalizer).

```python
d = Dilemma(dialect="ionic")                              # Ionic texts
d = Dilemma(dialect="doric")                              # Doric texts
d = Dilemma(dialect="auto")                               # try all dialects
d = Dilemma(dialect="ionic", period="hellenistic")        # combined
```

Dialects can be combined with period profiles. Setting `dialect`
implicitly enables the normalizer (no need for `normalize=True`).

## Orthographic normalizer

For texts with non-standard spelling, Dilemma includes an optional
orthographic normalizer that generates candidate normalized forms before
lookup. This handles:

- **Itacism**: η/ει/οι/υ all pronounced [i] and interchanged by scribes
- **αι/ε merger**: αι pronounced [e] and confused with ε
- **ο/ω confusion**: loss of vowel length distinction
- **Missing iota subscripta**: ᾳ/ῃ/ῳ written as α/η/ω
- **Spirantization**: β/υ interchange, φ/π, θ/τ, χ/κ confusion
- **Geminate simplification**: λλ→λ, νν→ν, etc.

Period-specific profiles (hellenistic, late_antique, byzantine) weight
rules by historical probability.

```python
d = Dilemma(normalize=True, period="byzantine")
```

## Transformer model

The lemmatizer model is a character-level encoder-decoder trained from
scratch on explicit `(form, lemma)` pairs. The training sources are
Wiktionary inflection tables, including the Ancient verb expansion, and
GLAUx. Other dictionary expansions and treebanks contribute lookup data;
Gorman's annotations are held out for evaluation.

The model reads Greek characters directly. Its shared encoder also supports
auxiliary POS, nominal-morphology, and verbal-morphology classification heads.
The reference configuration is:

| Component | Config |
|-----------|--------|
| Encoder | 3 transformer layers, 256 hidden, 4 heads |
| Decoder | 3 transformer layers, 256 hidden, 4 heads |
| POS head | Linear (256 -> 10 tags), auxiliary task |
| Nominal head | Linear (256 -> 45 labels), gender/number/case |
| Verbal head | Linear (256 -> 69 labels), tense/mood/voice |
| FFN | 512 dim |
| Vocabulary | ~381 Greek characters + special tokens |
| Parameters | ~4.2M |
| Inference | ONNX or PyTorch, beam search with headword filter |


Beam search proposes candidate lemmas. The citation policy controls which
headwords can be returned: the default accepts validated lookup self-maps
and independent Ancient/Byzantine heads; `strict_ag` requires independent
Ancient/Byzantine headword evidence. If no acceptable result is found, the
ordinary API returns the input unchanged. With `guess=False`, it returns a
missing result instead.

ONNX Runtime is preferred when ONNX files and the runtime are available;
PyTorch is the alternative. Construction leaves both backends unloaded.
Backend checks, warnings, imports, and weight loading occur on the first
inference request. The parity tests compare the two implementations on the
shipped model.

## Batching and caching

ONNX decoding batches active beams from multiple words in groups of at most
128. Each word keeps its own beam ranking and stopping condition. Encoder
batches are capped at 128 words. This bounds batch memory and reduces repeated
decoder calls without changing the model weights or beam width.

Each instance caches up to 65,536 context-independent model predictions.
Beam widths use separate cache entries. `tests/test_batched_decoder.py`
compares ranked results with serial decoding, including a real-model check
under `--run-slow`.

## Earlier byte-level model

An earlier implementation fine-tuned ByT5-small. ByT5 tokenizes UTF-8 bytes;
the current model uses Greek characters, so a multibyte Greek character needs
one vocabulary position. The smaller model also avoids the `transformers`
dependency for lemmatizer inference. Training times from older experiments
are hardware-specific and should not be used to estimate a new run.

<p align="center">
  <img src="https://raw.githubusercontent.com/open-greek/dilemma/main/examples.svg" width="700" alt="Greek lemmatization examples">
</p>
