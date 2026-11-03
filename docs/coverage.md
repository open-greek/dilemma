# Greek language coverage

[README](../README.md)

- [Language codes](#language-codes)
- [Modern Greek varieties](#modern-greek-varieties)
- [Ancient Greek varieties](#ancient-greek-varieties)
- [Medieval/Byzantine Greek](#medievalbyzantine-greek)

## Language codes

| Code | Period | ISO standard |
|------|--------|-------------|
| `el` | Modern Greek (including vernacular medieval, Katharevousa, regional) | ISO 639-1 |
| `grc` | Ancient Greek (Homer through Byzantine literary Greek) | ISO 639-2 |

Code and API calls use ISO 639 language codes: **`el`** for Modern Greek
and **`grc`** for Ancient Greek. In English text we often use the
shorthands **MG** (Modern Greek) and **AG** (Ancient Greek).

For Dilemma's purposes, MG (`el`) includes Katharevousa, even though
Katharevousa often benefits from AG lemmatization due to its archaizing
vocabulary and morphology. Medieval/Byzantine Greek has two components:
vernacular medieval Greek (ancestor of Modern Greek, merged into `el`)
and literary Byzantine Greek (classicizing, Atticist-influenced, resolved
via the AG lookup under `grc`).

For lemmatization, the two-way split works because Byzantine literary
Greek is classicizing (handled by `grc`), while vernacular medieval
Greek is the ancestor of Modern Greek (handled by `el`). The `med`
label still appears in `LemmaCandidate.lang` for forms from the
medieval Wiktionary dump, but these are merged into the `el` lookup
at build time.

The [POS tagger and dependency parser](tagger.md) also accept `lang="med"`
for Medieval/Byzantine text. That model uses the Ancient Greek backend. The
lemmatizer itself accepts only `all`, `grc`, and `el`; `med` is also a source
label in verbose candidates.

## Modern Greek varieties

| Variety | Wiktionary-tagged headwords |
|---------|---------------|
| **Standard Modern Greek (SMG/Demotic)** | 877K entries (core) |
| **Katharevousa** | 283+ tagged, hundreds more formal/place terms |
| **Cretan** | 273 |
| **Cypriot** | 199 |
| **Heptanesian (Ionian)** | 18 |
| **Maniot** | 3 |
| **Medieval/Byzantine (vernacular)** | 3K ([merged into MG](#medievalbyzantine-greek) - vernacular medieval is the ancestor of MG; literary Byzantine is Atticist-influenced and resolves via the AG lookup, not this table) |

## Ancient Greek varieties

| Variety | Wiktionary-tagged headwords |
|---------|---------------|
| **Epic/Homeric** | 3,755 |
| **Ionic** | 1,638 |
| **Attic** | 1,279 |
| **Koine** | 1,209 |
| **Byzantine (literary)** | 496 |
| **Doric** | 456 |
| **Aeolic** | 163 |
| **Laconian** | 52 |
| **Boeotian** | 15 |
| **Arcadocypriot** | 11 |

The counts above are Wiktionary headwords explicitly labeled with a
dialect tag. Each headword generates a full inflection paradigm (10-40
forms for verbs, 4-8 for nouns), so Wiktionary-derived form coverage is
much larger than the headword count suggests.

However, Wiktionary tags are only a fraction of Dilemma's actual dialect
coverage. Corpus-derived form-lemma pairs add substantially more:
GLAUx contributes 76K Ionic pairs from Herodotus and the Hippocratic
corpus, with Diorisis and AGDT adding more across Herodotus,
Thucydides, Xenophon, Demosthenes, and others. The dialect normalization layer (Ionic, Doric, Aeolic, Koine)
then maps remaining dialectal forms to their Attic equivalents for
lookup, catching forms that no corpus or dictionary has catalogued.

Katharevousa forms are the primary non-SMG target for Modern Greek -
they mix AG morphology (augments, 3rd declension genitives) with MG
vocabulary. The strong Epic/Homeric coverage (3,755 tagged headwords
plus extensive GLAUx corpus data) is directly relevant for literary
texts based on Homer.

## Medieval/Byzantine Greek

<a id="why-medieval-is-mg"></a>
Medieval/Byzantine Greek has two distinct registers that Dilemma handles
differently. Vernacular medieval forms are merged into Modern Greek
(`el`) since they are the direct ancestor of MG. Literary Byzantine
forms are classicizing and resolve via the AG (`grc`) lookup.
EL Wiktionary's "Medieval Greek"
category (3,158 entries, 3,080 headwords) is roughly 71% vernacular
and 29% literary Byzantine, based on presence of polytonic diacritics:

- **Vernacular** (~71%): δέρνω, θυμώνω, χτενίζω, βρίσκω, γούνα,
  ναράντζι, βουρκόλακας, ξεχαρβαλώνω - early MG vocabulary
- **Literary Byzantine** (~29%): ἀποφθέγγομαι, αἰθεροπόρος,
  περικαλλής, κριθάλευρον - Atticist-influenced forms
- **Medieval-specific**: μαξιλάριν, ἀδελφάτον, κασσίδιον, ἴνδικτος,
  γαστάλδος - neither pure AG nor modern MG

Merging all into `el` works because the AG lookup runs first. The 29%
literary forms typically already exist in the AG table and resolve
there; only the vernacular and medieval-specific forms actually fall
through to the MG lookup. On the DBBE benchmark, only 2 of 8,342
tokens resolved via the medieval table, while 92.8% came from the AG
lookup.
