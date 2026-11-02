# LSJ and Sophocles paradigm expansion

[README](../README.md)

- [Build commands](#build-commands)
- [Historical expansion overlay](#historical-expansion-overlay)
- [Corpus and lexical evidence](#corpus-and-lexical-evidence)
- [Verb expansion](#verb-expansion)
- [Finite-form synthesis](#finite-form-synthesis)
- [Non-Attic form filtering](#non-attic-form-filtering)
- [Choosing the form that fills a cell](#choosing-the-form-that-fills-a-cell)

## Build commands

To regenerate the expanded lookup table from LSJ and Sophocles sources:

```bash
pip install --force-reinstall --no-deps git+https://github.com/tatuylonen/wikitextprocessor.git
python build/expand_lsj.py --setup           # build Wiktionary Lua module database
python build/expand_lsj.py --expand          # expand LSJ nouns
python build/expand_lsj.py --expand-verbs    # expand LSJ verbs
python build/expand_sophocles.py --expand    # expand Sophocles nouns
python build/expand_sophocles.py --expand-verbs  # expand Sophocles verbs
python overlay_lsj.py                        # recover pinned historical expansion gaps
python build/expand_lbg.py                    # inflect Byzantine headwords (gated gap-fill)
```

This requires LSJ9 data from [lsj9](https://github.com/ciscoriordan/lsj9)
(included in `data/lsjgr_bridges.json` and `data/lsj9_frequency.json`) and
the Sophocles TEI data (included in `data/sophocles/`).

## Historical expansion overlay

The final overlay is intentional. A May 2026 base-data refresh replaced the
9,947,605-entry expanded AG table with a 2,354,300-entry Wiktionary base, and
the June re-expansion recovered only part of the prior paradigms. The overlay
derives only the expansion delta from the known-good, revision- and
SHA-256-pinned March artifact in `data/ag_expansion_reference.json`; it never
restores a historical base mapping and never overrides a current mapping.
It restores marked surface spellings only, not the generated accent-stripped
fallback keys that can shadow unrelated analyses sharing the same bare
skeleton. The generated `data/ag_expansion_exclusions.json` also removes
historical rows that conflict with the pinned 1.3.1 resolved lookup or with
Dilemma's nonlexical classifier. Its generator is
`build/build_ag_expansion_exclusions.py`.
Running it without `--reference` downloads and verifies that exact artifact.

After withholding approximate irregular-compound paradigms, the overlay
separately preserves actual source cells and their original, monotonic, and
unaccented lookup aliases within the historically represented families. This
requires a clean merged Wiktionary, corpus, or reviewed textual cell and a
positively defined Ancient lexical owner; a historical letter pattern alone
never generates a spelling. Existing independent readings and hard spelling
rejections take precedence. Unresolved source or marked-cell ownership ties,
and unsupported aorist, perfect, or pluperfect annotations under εἰμί/εἶμι
compound owners do not license new aliases without direct ownership evidence.
An automatic shorter compound owner is also withheld when only a longer,
independently defined compound accounts for the actual source cell's preverb;
that comparison never infers a replacement owner. Replacing an unsupported
existing target owner requires direct evidence. Missing citation accents may
be supplied from a unique positive head. Structurally impossible citation
accents (on a consonant, duplicated within a syllable, or an invalid second
accent) can use that same unique-head path while preserving breathing,
subscript, and case. Actual source spellings remain unchanged in the audit.
A valid existing accent is preserved because it can distinguish a different
lexeme or contracted form, including a productive enclitic accent. A solely capitalized
automatic cell also needs matching lowercase or direct evidence before gaining
aliases, and cannot erase a possible existing name. Final
elision is normalized to Greek koronis while the original source spelling is
retained in the audit. `--source-alias-audit FILE` records all readings,
restorations, canonicalization, ambiguity skips, and portable input/source
hashes. The merged source pairs, reviewed exact-form specification, and
`grc_lexical_heads.json` are required inputs for this step.

With `--kaikki DIR`, the overlay also checks historical-only elisions against
both fresh Ancient Greek dumps in `en-el/` and `el/`, and binds noun input
validation to that same directory. It can replace an
unsupported historical owner only when all positively defined current source
readings agree on one PRON or DET owner, and every reading fits an explicit
case/number cell in that owner's independently defined entry of the same POS.
The complete cell must be the marked stem plus one literal vowel. Source
quantity notation and initial capitalization are normalized; accents,
breathings, and other letters are preserved. Competing owners or POS readings,
incompatible cells, a positive historical marker variant, and a matching
complete unelided word all withhold the change. Existing predecessor mappings
and later changes remain protected. When rebuilding from a reversed alias
audit, an exact historical cell with its unchanged historical owner is checked
again against current evidence, so a still-supported repair is reproducible.
Independent marker variants remain protected. This check creates no bare lookup aliases.

The historical overlay counts remain unchanged. Proven replacements run after
the normal alias recovery, only while the former historical owner still
matches, and join the reversible `canonicalized` audit. The audit retains all
source claims, own-entry cells, definitions, and withheld decisions. It pins
both fresh dumps and the imported definition classifier and repair helper.
Omitting `--kaikki` disables this additional check; providing it requires both
dumps and rejects missing, malformed, or changed input files.

## Corpus and lexical evidence

The Ancient export retains independently cited proper-name forms even when
the flat lookup assigns their spelling to a common-word lemma. The finite
`source_forms` specifications in `data/hunspell_grc_postclassical_review.json`
record the spelling, owner, citation, and source. They preserve forms such as
`Πῶλε` and `Ὀνησίμῳ` without exempting unrelated capitalized common nouns from
the case filter.

An already reviewed compatibility spelling also survives the case filter when
it has positive exact-case corpus evidence or no lowercase spelling survives.
This retains inherited capitalized spellings and work titles without inventing
a proper-name lemma. The audit distinguishes those two reasons and records the
retained spelling, its count, and the existing ownership decision. Unattested
lowercase name aliases and redundant sentence capitals remain filtered;
explicit spelling, quotation, and bare-fragment rejections still take priority.
Restored capitals with unresolved ownership must also pass the structural and
corpus respelling checks; exact-case counts alone do not waive those checks.

Exact reviewed forms also retain Archimedes' Doric `ἀποχρὴ` under `ἀπόχρη`,
with its contextual acute twin. This textual exception does not remove accents
from other valid citation heads when recovering aliases.

The Diorisis pair extractor preserves terminal Beta Code elision as Greek
koronis, with medial σ before the mark. An elided source token therefore cannot
silently become a complete word; unelided homographs remain separate readings.
After compatibility forms are restored, the exporter checks bare fragments
again, including capitalized spellings. A terminal-consonant fragment with a
matching case-preserving elided lookup reading is excluded unless a separate
unelided owner or an independently edited headword establishes a complete word.
Closed apocope and consonant-final word lists remain available. The export audit
records these rejections; raw corpus counts do not override the check because
transcriptions can lose apostrophes.

Exact reviewed corpus fragments are also excluded after compatibility and
from source-alias recovery. The review for `ἐνεί` records the split complete
words `ἐνείραντα` and `ἐνείκασθαι` and the remaining OCR loci; it preserves
the independent headword `ἔνει`. The authored spelling review is reversible
and leaves the original source readings unchanged.

The exact `ἐπεῖ` review distinguishes erroneous transcription from genuine
`ἔπει`, `ἐπεί`/`ἐπεὶ`, and `ἐκεῖ`. The printed Herodotus witness has `ἐπεὶ`;
the reviewed corpus contexts do not establish a circumflex verb form.
The decision does not choose one replacement for unrelated damaged passages.

The optional `data/ag_source_alias_audit.json` and
`data/ag_source_alias_manifest.json` preserve source readings and reversible
alias changes, and pin the matching lookup, predecessor, source revision,
and input hashes. `scripts/build_source_alias_manifest.py` verifies those
pins and reverses each change before writing the manifest. Use
`--pre-alias-lookup` for the first manifest, or `--pre-alias-manifest` to
verify it against the recorded predecessor hash. When the audit includes
historical closed-class elision checks, also pass the same `--kaikki DIR` to
verify both external dump pins. The manifest verifies the imported classifier
and helper against the source revision and retains strict byte reversal.
These files are provenance artifacts and are not required at runtime.

Expansion sources sometimes encode restored letters with Leiden square
brackets or spelling alternatives with parentheses. These delimiters are
metadata, not Greek spelling. The build resolves only the structurally safe
terminal movable-nu notation (`(ν` or `(ν)`) into both real spellings; all
other bracketed forms or lemmas are rejected rather than guessed. The final
overlay also cleans an already-expanded target, and `build_lookup_db.py` plus
the Hunspell exporter repeat the rejection as defense in depth.

The GLAUx pair builder applies the same rejection before its shared tagged
artifact is written. `export_morphology.py` repeats the check while deriving
both movable-nu and elision tables, so `grc_morph.json` remains clean even
when it is built from an older local pair file.

## Verb expansion

`--expand-verbs` does three things:

1. Classifies the headword via `_classify_verb` (suffix dispatch for
   thematic and athematic types, explicit table for εἰμί / εἶμι /
   οἶδα / χρή / φημί and their preverbed compounds) and runs
   `Module:grc-conj` on the present-system paradigm.
2. Parses the LSJ entry head paragraph with
   `build/lsj_principal_parts.py` to extract whatever principal
   parts LSJ has labeled (`fut.`, `aor.`, `pf.`, `pf. m./p.`,
   `aor. p.`, `impf.`, `plpf.`).
3. Re-invokes `Module:grc-conj` for each extracted tense, merging the
   results into the verb's paradigm. Verbs without parsable principal
   parts fall back to present-only behavior, so the path is strictly
   additive.

About 712 LSJ-only verbs receive at least one parsed principal part
through path 2; full-pipeline success on the 28,745 LSJ-only verb
candidates is 99.27%.

## Finite-form synthesis

A separate post-processing pass in `build/build_grc_verb_paradigms.py`
takes the same parsed principal parts and procedurally synthesizes the
missing finite-mood cells (subjunctive / optative / imperative / aorist
infinitive) for thematic -ω verbs via `build/synth_verb_moods.py`, plus
the full case×gender×number declension of every participle slot via
`build/synth_verb_participles.py`, plus the past-indicative 1sg cells
(active / middle imperfect; active / middle / passive aorist) for any
verb where kaikki dropped the tense tag and the unaugmented Homeric
variants got correctly filtered out into the epic dialect slice. Stem-
templating only fills cells that aren't already attested by Wiktionary
or GLAUx; real corpus cells are never overwritten.
Corpus-only lemmas need an independent headword in `data/grc_lexical_heads.json`
before this pass infers a citation or synthesizes forms. Without that evidence,
the observed cells and dialects are retained, but no conjugation is extrapolated.
Build the lexical-head inventory before building canonical verb paradigms.

Coverage: on the 27K-verb output the synthesis adds ~278K finite-mood
cells, ~775K participle cells, ~15K aor-2 cells, ~430K contract cells
(active + middle present participle), and ~36K past-indicative 1sg
cells. Comparing against jtauber/greek-inflexion's hand-curated
paradigms on the 3.6K shared lemmas, dilemma's per-mood coverage is
currently:

| mood        | dilemma / jtauber | notes                             |
|-------------|-------------------|-----------------------------------|
| indicative  | 0.83              | mostly corpus / Wiktionary cells  |
| subjunctive | 1.17              | full thematic + contract synth    |
| optative    | 0.95              | full thematic + contract synth    |
| imperative  | 1.03              | full thematic + contract synth    |
| infinitive  | 1.49              | over-broad sigmatic synthesis     |
| participle  | 0.95              | thematic + aor-2 + ε/α-contract   |
| overall     | 0.97              | well past the 95% full-flip mark  |

Synthesis covers thematic -ω verbs (regular and aor-2 / strong-aorist),
α-/ε-/ο-contract verbs (present system), and the mixed-α aor-2 class
(πίπτω → ἔπεσα, λέγω → εἶπα, εὑρίσκω → εὗρα) that uses α-style endings
on the active and middle indicative but regular ο-thematic endings
elsewhere. The participle ending tables encode jtauber's macron / breve
quantity-mark conventions on feminine forms (-ουσᾰ / -ουσᾰν / -ούσᾱς
present-active, -σᾱσα / -σᾱ́σᾱς aorist-active, -μενᾱς middle-passive,
-υίᾱς perfect-active) so synthesised cells match jtauber verbatim.

Known limitations: athematic μι-verbs are handled in the upstream LSJ
expansion (path 2) rather than the synth module; perfect contract
participles aren't synthesised because they'd need separate perfect-
stem extraction; ο-contract middle participles are skipped because
jtauber's pattern is too inconsistent cell-by-cell; and a small set of
verbs with idiosyncratic accent (e.g. ὁρᾷς) are deferred to corpus.

## Non-Attic form filtering

GLAUx provides AGDT 9-position morph tags but no dialect axis: a
Homeric unaugmented imperfect / aorist gets the same active-imperfect
or aorist tag as its Attic counterpart, and a sandhi crasis form like
κἄβλεψας gets the same active-aorist-2sg tag as a regular ἔβλεψας
would. The build pass recovers dialect / sandhi status from the
surface form itself before populating the canonical Attic slice:

- Crasis forms (consonant-initial words with a breathing mark on a
  non-initial vowel: κἄβλεψας, τοὔνομα, χἠμεῖς) are dropped entirely
  as textual artifacts. ~650 forms are removed across the 27K-verb
  output.
- Homeric iterative imperfects (-εσκον / -ασκον / -οσκον infix forms
  on verbs whose lemma doesn't natively end in -σκω) are routed to
  the ``epic`` dialect slice. ~370 forms.
- Athematic root-aorist passives (ἐλύμην / ἔλυντο / λύτο series, with
  middle-voice personal endings on a slot tagged passive aorist) go
  to ``epic``. ~130 forms.
- Unaugmented past-indicatives (aorist / imperfect / pluperfect cells
  whose form lacks the augment that Attic mandates) go to ``epic``.
  ~27K forms.

Detection is conservative: lemmas starting with ε- whose forms also
start with ε- are skipped to avoid clobbering compound-prefix verbs
where the augment is internal (ἐκμολεῖν → ἐξέμολεν), and lemmas with
long-vowel initials (η-, ω-) are skipped because their temporal
augment is morphologically invisible.

GLAUx does record a dialect per work, which `build/build_glaux_pairs.py`
now carries onto the pairs: a form attested only in works marked Ionic,
Epic, Doric or Aeolic is tagged with that dialect, so it fills that
dialect's slice instead of competing for the Attic one. Works marked
Attic, Attic/Koine or Koine share the default slice.
Each pair also carries its token count (`count`) and, when its tokens were
tagged more than one way, every analysis with its own count (`analyses`,
`[pos, tags, count]`, commonest first); `pos` and `tags` stay the first
token's analysis.

## Choosing the form that fills a cell

Source tags are interpreted deterministically. A single ambiguous grammatical
axis contributes each explicit reading, and a form tagged for several dialects
contributes to each slice. Tags with several ambiguous axes cannot establish
which values belong together; they stay in the raw data but do not populate
canonical cells. Filtering a collapsed cell does not discard another valid
reading of the same form.

An unbreathed citation form can acquire a breathing from a unique surviving
present first-person source cell when all other letters and marks agree.
Existing breathings, accents, subscripts, and ambiguous candidates are preserved.
This corrects the canonical citation without changing the original source rows.

For a verb whose own citation is explicitly marked defective, procedural
synthesis is limited to the tense and voice families present in retained source
cells, including dialect slices. An automatically inserted citation cell does
not authorize a present paradigm, and a defective tag on a different referenced
head is not inherited. Supplied cells remain unchanged.

Several sources can offer a form for the same cell, and the sources
disagree. The pass ranks the candidates and, before ranking, discards
two kinds of evidence that are wrong rather than merely weaker:

- A Wiktionary table row that reaches kaikki with its person and number
  collapsed onto the first cell. A real cell holds one form and at most
  a few spellings of it, so four or more distinct letter sequences in
  one cell of the same dialect means the row lost its person-number axis.
  Independently tagged Attic, Epic, Doric, and Koine variants are counted
  separately; rejecting one dialect's damaged cell preserves the others.
  The detector counts
  letters without diacritics; polytonic and simplified copies of contracted
  and uncontracted forms do not multiply that count. All surviving source
  variants remain available for choosing the canonical spelling. For example, τιμάω's entire
  present active indicative arrives tagged first-person singular, and
  `τιμάς` — far commoner as the accusative plural of the noun τιμή —
  then wins the cell on corpus frequency.
- In a subjunctive cell, a form whose ending belongs to the
  corresponding indicative. The subjunctive lengthens the thematic
  vowel, so `-εις`, `-ομεν` and `-ονται` rule a form out where `-ῃς`,
  `-ωμεν` and `-ωνται` are required. Endings the two moods genuinely
  share, such as the alpha contracts' `-ᾶτε` and the middle `-ῃ`, are
  left alone.

What survives is ranked by how many sources assign it to the cell,
then (in a past indicative cell) whether it carries the augment, then
whether Wiktionary lists it for this verb, then its corpus count in
`form_profile.db`.

Finally, Wiktionary's conjugation tables decide the accent of any cell
that a corpus token or the synthesizer spelled the same letters
differently. That pass reads which cell Wiktionary puts each spelling
in, so an accent that is the only thing telling two cells apart
survives — `λύσαι` is λύω's aorist optative third-person singular and
`λῦσαι` its aorist middle imperative — while a spelling both cells
share is still corrected, as `βαίνον` is to `βαῖνον`. When grammatical
tags exist, the replacement must fit the current cell, including an
explicit tense. A missing tense imposes no extra restriction. An infinitive
cannot reaccent a generated optative merely because the optative's
own source row was filtered out. Untagged spellings cannot change a cell's
accent, because they provide no grammatical evidence for that cell. Accent
evidence also stays within the default paradigm's dialect: an Epic
`κυλίσαι` does not replace Attic `κυλῖσαι`. A paradigm drawn entirely
from another dialect uses that dialect's evidence. A unique matching
cell takes precedence over the corpus's preferred accent.
If a damaged table row survives in only some cells or dialects, only
those retained readings can supply accent evidence.

Synthesized first-aorist middle imperatives in the second-person singular
use recessive accent, with final `-αι` counted as short: `κέλευσαι` and
`ὄζησαι`, compared with the active infinitives `κελεῦσαι` and `ὀζῆσαι`.
Long penults written with η or ω, a diphthong, or explicit source quantity
take a circumflex in two-syllable forms (`κρῶξαι`). The existing source-cell
correction still supplies vowel quantity that an unmarked α, ι, or υ
cannot establish. A diaeresis keeps adjacent vowels in separate syllables
when placing the recessive accent (`ἄϊξαι`).
Second-aorist mood synthesis requires a matching second-aorist principal
part. An `-α` ending or a difference from the present stem is insufficient:
sigmatic variants (`ἔπαιξα`) and liquid first aorists (`ἔμεινα`) must not
generate second-aorist optatives or infinitives. Mixed-alpha synthesis is
limited to established lemma/form pairs; other source forms remain in
their attested cells. If the selected principal part does not fit the
active second-aorist template, both mood and participle synthesis try the
separately labeled second aorist. A middle-only principal part cannot
displace an available active form.
Passive aorist endings from a source table with an empty stem cannot
supply a complete verb cell. For example, τύπτω's incomplete Ionic
passive table lists `ῆν` and `εῖτε`; those readings are withheld while
the complete active Ionic and passive Attic/Epic forms remain. This
checks the specific grammatical cell and keeps independent homographs
such as ἵημι's active aorist `εἷτε` and εἰμί's imperfect `ἦν`.
The shared check also runs during Ancient source-table extraction, before
creating lookup, POS, or training records. Raw filtering requires explicit,
unambiguous grammatical axes. Gender-only participle headings are filtered
only when the entry's matching `grc-conj` template explicitly lacks both
passive stems and no complete competing template uses that dialect.
Other missing or conflicting labels remain auditable source data. A
separately identified active or middle reading survives when the same
row's passive reading is invalid.
For `grc-conj aor-1`, an augmented indicative stem without its nonpast
stem can also produce bare active or middle endings. Extraction withholds
only matching nonpast endings when every competing template for that
voice and dialect proves the missing stem. Indicative-only declarations,
complete competing templates, augmented indicative forms, uncertain labels,
and independently supplied voice readings remain. This prevents incomplete
οἰδάνω, δύπτω, and παρακελεύομαι tables from claiming forms such as `ῶ`
and `αίμην`; it does not ban those spellings in other entries.

Periphrastic forms contribute the main verb, not the separate auxiliary
εἰμί. The filter also recognizes the source's monotonic `είμεν`, `είτε`,
and `είεν` copies when they explicitly carry perfect-optative tags.
The distinct aorist forms of ἵημι and imperative forms of ἵστημι remain.
Each marked auxiliary also needs its corresponding mood, preserving the
aorist middle indicative `ἔστω` of ἵστημι beside the imperative of εἰμί.
Conflicting mood labels, or conflicting tense labels on a monotonic copy,
remain available for later disambiguation.
Ancient Wiktionary extraction applies the same check before creating training
pairs and lookup entries, so a compound tense cannot teach the isolated
auxiliary as a form of the main verb. Original source dumps remain unchanged.

Ancient nominal headers can abbreviate a genitive ending, such as `-ῆς`
beside `αὐλή`. An omitted hyphen does not make that ending a complete
inflection. Unqualified header endings are withheld from training, lookup,
and POS data; complete case forms, marked alternatives, and independent
parts of speech remain available.
