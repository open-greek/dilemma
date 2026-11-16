# LSJ and Sophocles paradigm expansion

[README](../README.md)

- [Build commands](#build-commands)
- [Sophocles heads](#sophocles-heads)
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
python build/expand_lsj.py --expand --kaikki kaikki --lsj9-dir LSJ9        # expand LSJ nouns
python build/expand_lsj.py --expand-verbs --kaikki kaikki --lsj9-dir LSJ9  # expand LSJ verbs
python build/expand_sophocles.py --expand --kaikki kaikki        # expand Sophocles nouns
python build/expand_sophocles.py --expand-verbs --kaikki kaikki  # expand Sophocles verbs
python overlay_lsj.py --lsj9-dir LSJ9          # recover pinned historical expansion gaps
python build/expand_lbg.py                    # inflect Byzantine headwords (gated gap-fill)
```

`--expand-verbs` also reads `data/form_profile.db` (`python -m dilemma
download --with-attestation`), `data/grc_lexical_heads.json`
(`build/build_grc_lexical_heads.py`) and the treebank pair files in `data/`
(`perseus_pairs.json`, `glaux_pairs.json`, `diorisis_pairs.json`,
`nt_pairs.json`, `ptnk_pairs.json`, `tagnt_pairs.json`,
`pedalion_pairs.json`), and stops when one is missing; see
[Principal parts that no entry prints](#principal-parts-that-no-entry-prints).

This requires LSJ9 data from [lsj9](https://github.com/ciscoriordan/lsj9)
(included in `data/lsjgr_bridges.json` and `data/lsj9_frequency.json`; the
exports `lsj9_forms.tsv`, `lsj9_headwords.json` and `lsj9_glosses.jsonl` are
read from `--lsj9-dir`, `$DILEMMA_LSJ9_DIR` or `~/Documents/lsj9`), the
Sophocles TEI data (included in `data/sophocles/`), and a current
`data/grc_lexical_heads.json`, whose `sophocles_breathing` record supplies the
breathings the Sophocles OCR lost (`expand_sophocles.py` refuses to run
without it) and whose heads decide two of the [head rules](#sophocles-heads).
`--kaikki` names the dump directory in either layout
([Dump directories](development.md#dump-directories)).

The verb pass reads each headword's LSJ text from `lsj9_glosses.jsonl` to add
principal parts and to skip entries that are not paradigms (prefixes,
cross-references, inflected imperatives, person-limited verbs). A glosses
export that lacks an entry's text cannot skip it that way, so a headword that
is all conjugation ending (the prefix `ἡμι-`, whose stem is empty) is never
expanded, with or without its text; it would otherwise map bare endings such
as `ῆς` and `εῖτε` to the prefix. Because the export decides which verbs are
expanded and how, `data/lsj9_glosses_pin.json` pins its sha256 and size, and
`expand_lsj.py --expand-verbs` stops when `lsj9_glosses.jsonl` is missing or
differs.

The current export ends many verb entries' head text at the first citation
(`κατατέμνω`: "fut. -τεμῶ Ar.Ach.301: aor."), before the remaining principal
parts. The pin therefore names a second file, `lsj9_glosses_earlier.jsonl`: the
same export at an earlier lsj9 commit, which keeps those parts and drops the
citations. The verb pass reads it only for principal parts: a tense the current
text gives stands, a tense it lacks is taken from the earlier text, and the
three aorist slots are filled only together. 978 verb entries gain 3,018
principal parts this way. Entry rejections still read the current text alone.
Both files give the text under the entry's own spelling, which the verb pass
matches to the headword list as described under
[Verb expansion](#verb-expansion).

## Sophocles heads

`build/expand_sophocles.py` reads each entry's head from its `<orth>` element.
The text is OCR, so the first word is not taken as it stands: five rules
decide what is generated, and under which head. Nouns, adverbs and verbs
are read through the same rules.

- **A phrase is not an entry.** When another Greek word follows the first
  one inside the `<orth>`, before its first comma, the head is a phrase or a
  word the OCR split, and nothing is generated from its first word:
  `Νέα Ῥώμη` gives no paradigm of `Νέα`, and `κατα- φθάνω` none of `κατα`.
  Whatever follows a full stop, colon, semicolon, dash, equals sign or run of
  hyphens is entry text, and so is a parenthesis, a digit or a Latin word
  (`or`, `for`, a citation); the first word then stays the head
  (`βιοθανέω ---- βιαιοθανατέω`, `γιγαντιάω (γίγας)`,
  `Ἱριάδιτζα or Τριαδίτζα`).
- **A stated gender wins over the ending.** An article that is the first
  field after the head, or that follows one field (the genitive, which the
  OCR may garble), states the noun's gender: `καισάριης, ἡ`,
  `ἀτρεπτότης, ητος; ἡ`. The ending classes then supply a genitive only
  through a class of that gender (`ἀντίστροφος, ου, ἡ` is declined as a
  feminine in -ος). Where the ending has no such class the shared genitive
  resolver decides, and the noun is withheld when it finds no genitive
  either: `καισάριης, ἡ` (Latin *caesaries*) used to get the masculine
  paradigm in -ου. An article further on belongs to the definition and stays
  a guess, as does the `ἡ` the OCR reads for an adjective's feminine ending
  (`ἀρθρικός, ἡ. ὄν`).
- **A malformed head is rejected.** A head with two accents (`Ἔρετριάς`,
  `οἴηματικός`) or with a hyphen or dash inside it (`Σινα-ῖτις`,
  `ὕπο-κατασκευή`) is not generated. The one repair is to drop the hyphens,
  and only when the joined word has a single accent and a lexical source
  other than Sophocles lists it in `grc_lexical_heads.json`: `καθ-ηδύνω` is
  read as `καθηδύνω` and expanded under that head, unless the table already
  has it.
- **A capitalized common word goes under its lowercase head.** When the
  other lexical sources list the lowercase spelling of a capitalized head and
  none lists the capitalized one, the capital is the lexicon's typography:
  the entry `᾿Αθεότης` is read as `ἀθεότης`, and generated there unless the
  table already has that head. A breathing the OCR prints as a separate mark
  before the capital is joined to it for this comparison. A capitalized head
  another source lists (`Πέτρα` beside `πέτρα`), or one without a lowercase
  twin, is a name and stays.
- **A capitalized unaccented key does not shadow a lowercase reading.** Every
  generated form also gets an accent-stripped key. A capitalized one is
  withheld when its lowercase spelling already reads another lemma:
  `Εστιας` for `Ἑστιάς` would take the unaccented reading of `ἑστία`.

The noun step records the decisions in
`data/ag_sophocles_noun_generation_audit.json`, each record with its `count`:

- `head_rules.multi_word_heads`: every phrase and its first word;
- `head_rules.stated_gender`: every noun whose stated gender is not the
  gender of its ending's class, with both and the result (`expanded`, or the
  reason the noun was withheld);
- `head_rules.malformed_heads`: every rejected head and the reason
  (`two-accents`, `inner-hyphen`);
- `head_rules.repaired_heads` and `head_rules.lowercased_heads`: every
  printed head and the head it is read under;
- `withheld_capitalized_keys`: every withheld key with its lemma and the
  lowercase reading.

The head records cover verbs and adverbs as well, because every step reads
the same heads. The verb step writes its own withheld keys to
`data/ag_sophocles_verb_generation_audit.json`. The lexical heads the repair
and the lowercase rule read are pinned among the noun audit's inputs
(`input_sha256.lexical_heads`), so the overlay refuses an audit made against
other heads.

The rules change what the generator writes, not what the overlay recovers.
`superseded_lemmas` still names a lemma as the lexicon prints it, because the
historical layer keyed its rows that way, so a rejected or re-keyed head that
is not superseded keeps the marked rows the pinned reference has for it.

## Historical expansion overlay

The final overlay is intentional. A May 2026 base-data refresh replaced the
9,947,605-entry expanded AG table with a 2,354,300-entry Wiktionary base, and
the June re-expansion recovered only part of the prior paradigms. The overlay
derives only the expansion delta from the known-good, revision- and
SHA-256-pinned March artifact in `data/ag_expansion_reference.json`; it never
restores a historical base mapping and never overrides a current mapping
(the one exception is the accent-less head rule below).
It restores marked surface spellings first, not the generated accent-stripped
fallback keys that can shadow unrelated analyses sharing the same bare
skeleton. The generated `data/ag_expansion_exclusions.json` also removes
historical rows that conflict with the pinned 1.3.1 resolved lookup or with
Dilemma's nonlexical classifier. Its generator is
`build/build_ag_expansion_exclusions.py`.

The table published from 1.3.7 to 1.3.25 was built on top of earlier tables,
so it carried keys that a build from fresh dumps withholds or that no current
generator makes. The manifest pins that table as `carried_baseline`, and the
overlay gives such a key back only under one of the rules below. Each rule
states its own condition; there is no general "carry every published key"
rule, because part of the published readings were wrong and the rebuild fixed
them (`ἔξηλθεν` read `ἐξερύω`, `φᾶτε` read `φάω`). No rule adds a key the
published table lacked.

- An excluded 1.3.1 conflict (`carried_baseline_conflicts`). `ἀμέτρητον`
  keeps `ἀμέτρητος` and `Δημήτρειον` keeps `Δημήτρειος`; without the release
  the corpus gap-fill supplies the recorded readings `ἀμετρητος` and
  `Δημήτρειοι`. Every other conflict stays excluded: `ἐκπλαγῇ` stays
  `ἐκπλάσσω`, and a capitalised function word (`Γὰρ`, `Ἤ`) still reads its
  lowercase lemma. Released forms count among the pinned `excluded_entries`
  and separately as `released_conflict_entries`.
- A noun without a fresh paradigm (`kept_withheld_noun_forms`). A superseded
  noun lemma loses its old paradigm to the one the current generators make.
  Where the noun audits record no resolved genitive for the lemma (the
  generator could not use the printed one: `λεκτόν`, `κλείς`, `σωματεῖον`),
  there is no fresh paradigm, and the lemma keeps the marked expansion keys
  the published table had with the same reading (`λεκτοῦ`, `κλῃσίν`,
  `σωματεῖα`). The lemma must be a positive lexical head in its citation
  spelling (not one with a grave, `Καλαβρὰ`) and the key a structurally
  possible polytonic spelling (the Hunspell export's check), so hyphenated
  and doubly accented OCR keys stay out. One class of old paradigm
  is wrong: the old generator declined such a headword by its nominative
  (`Θεοπρεπότης, -τος` as `Θεοπρεπότου`). Where the lemma's citation key is
  among the keys and the entry prints an ending in -ος, -ως, -ους or -ες that
  no other key ends with, only the citation spelling is kept. A headword the
  audit rules out as a noun, and a lemma with a fresh paradigm, keep nothing.
  The kept keys count as `withheld_noun_entries_kept`; the pinned counts do
  not change.
- A regular cell of an oxytone in -ός (`carry_published_cells`). No generator
  makes the masculine and neuter cells of the Sophocles adjectives in -ικός
  and -τός, and they are not in the reference either: an earlier Sophocles
  expansion made them. A key the published table read under a head in -ός
  comes back when the target lacks the key, the head is a positive lexical
  head that the target maps to itself, and the key is one of the head's
  twelve regular masculine and neuter cells in its accented, grave or
  unaccented spelling (`ἀρχοντικόν`, `θελητόν`, `καρπωτοῦ`, `Κρητικον`). A
  head only the Sophocles OCR attests is skipped when its lowercase spelling
  or its spelling with the other breathing is a positive head (`Θερινός`
  beside `θερινός`, `Ἱωνικός` beside `Ἰωνικός`), and so is a superseded noun.
  A key that already has another reading keeps it (`νοερῶν` stays `νοερόω`).
- A breathing-corrected head (`carry_published_cells`). The published table
  read some keys under a head whose breathing the Sophocles OCR lost
  (`Ρωμαϊκόν` under `Ρωμαϊκός`). Where the breathing review in
  `grc_lexical_heads.json` corrected the head and the target has the key's
  breathed spelling under the corrected head (`Ῥωμαϊκόν` `Ῥωμαϊκός`), the
  published key comes back with the corrected head. A key without such a
  fresh cell stays out.
- A historical unaccented key (`αθηλυντος`, `παμπολλων`), after the marked
  spellings and the source aliases are in (`restore_unmarked_keys`). It comes
  back only where the marked spellings of the current table that strip to it
  read the key's lemma, so the key agrees with the table and unaccented input
  reaches the paradigm. A key whose marked spellings read another head, that
  no marked spelling backs, or that the published table lacked stays out.
  Three refinements decide what agreement is. Two spellings of one lemma
  agree: a leading spacing breathing folds into its capital (`᾿Αγαρηνός` is
  `Ἀγαρηνός`) and a breathing-corrected head is its corrected spelling; the
  key takes the spelling its marked forms read. A grave form whose acute twin
  is in the table is not consulted (`ευθης` follows `εὐθής`, not the reading
  of `εὐθὴς`). A marked form that reads a lemma no list in
  `grc_lexical_heads.json` has does not dispute a key whose own lemma a list
  other than the Sophocles OCR confirms (`απεπνιγη`, `ζημιωθης`). The
  unaccented keys of the three rules above come back under the same test.
  The audit records the count and a digest of these rows
  (`unmarked_key_restore`); `scripts/build_source_alias_manifest.py` removes
  exactly those trailing rows before it reverses the alias changes.
- A reading under an accent-less head (`keep_published_over_unaccented_heads`).
  LSJ9 exports some heads without any diacritic (`κληζω` for `κλῄζω`), and the
  fresh paradigm of such a head claims keys the published table read under
  the properly spelled one. Where the fresh lemma has no diacritic and the
  published lemma is a positive lexical head with exactly those letters, the
  published reading stays (`κληζεται` reads `κλῄζω`). This is the only rule
  that replaces a current reading.

The overlay prints each rule's count, and the alias audit records them under
`historical_overlay_counts`.

Running it without `--reference` downloads and verifies that exact artifact;
the carried table likewise comes from its pinned Hub revision unless
`--carried` or `DILEMMA_AG_EXPANSION_CARRIED` names a local copy.

`build_lookup_db.py` checks the readings this layer supplies once more: a
lemma without any diacritic that no list in `grc_lexical_heads.json` has
(transliterated Latin such as `αελιυς`, OCR such as `λαβλυμι`, hyphenated
compounds such as `προ-ανακηρυσσω`) is dropped, or moved to the single
Ancient head with the same letters when the lookup maps that head to itself
(`συνοχος` → `σύνοχος`; a cross-reference head such as `κλεῖδα` drops the
reading instead). A corpus pair naming one of these lemmas gets the same
verdict. Each such reading is listed in
`data/citation_hygiene_rejections.tsv`.

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
both fresh Ancient Greek dumps (flat or nested layout, see
[Dump directories](development.md#dump-directories)), and binds noun input
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
   results into the verb's paradigm. For a verb without parsable
   principal parts the pass infers them from its simplex or its class and
   keeps the cells a corpus attests
   ([Principal parts that no entry prints](#principal-parts-that-no-entry-prints)).

794 of the 28,503 LSJ-only verb candidates receive at least one parsed
principal part through path 2 (217 of them from the earlier export), and
27,754 candidates are expanded.

The entry text is found by the headword's spelling. The headword list drops
quantity marks, and prints `θρώσκω` for the entry `θρῴσκω`, while the glosses
keep the entry's own spelling (`ἀγῑνέω`), so the exact spelling alone misses
the text of 6,883 candidates. `head_text_finder` takes the exact spelling
first, then the one entry with the same letters without quantity marks, then
the one without iota subscript either. A spelling that two entries share
(`θᾱλέω`, `θᾰλέω`) gets no text.

The parser reads Arabic and Roman aorist numbers (`aor. 1`, `aor. I`), a text
that opens inside a section (`Pass., aor. ἀντετάφην`), and two numbered
passive aorists of one verb (`ἀπεπνίχθην`, `ἀπεπνίγην`). A token that begins
with a vowel and has no breathing is the tail of a word (`fut. ύσω`), never a
form. Where the current text stops right after a label (`ἀνοιδέω`: "fut.
-ήσω: aor.") and the earlier text opens with a Greek word (`ἀνῴδησα`), the two
are read as one text. Hyphenated abbreviations (`fut. -ήσω`, `aor. -έτεμον`)
are still not expanded.

### Stems of the past tenses

`Module:grc-conj` takes two stems for a past tense: the augmented one for the
indicative and the unaugmented one for every other mood, the infinitive and
the participle (`aor-1|ἐλυσ|λυσ`, and for the passive `aor-1|||ἐλυθ|λυθ`).
LSJ prints the augmented indicative, so `unaugmented_stem` derives the other
stem from it and from the headword:

- In a compound the augment follows the preverb, which stands before it in
  its prevocalic form; removing the augment restores the preverb of the
  present: `ἀντετάφην` gives `ἀντιταφ-`, `ἐξέβαλον` `ἐκβαλ-`, `συνέλεξα`
  `συλλεξ-`, `ἐνέθορον` `ἐνθορ-`, `κατέστησα` `καταστησ-`.
- A temporal augment restores the vowel of the headword (`ἤκουσα` `ἀκουσ-`,
  `ἤλπισα` `ἐλπισ-`, `ἀνῴδησα` `ἀνοιδησ-`), never a fixed vowel.
- The suppletive aorists have their own stems (`εἰσήνεγκα` `εἰσενεγκ-`,
  `ἀπῆλθον` `ἀπελθ-`, `συνεῖδον` `συνιδ-`, `ἀφεῖλον` `ἀφελ-`).
- A stem that cannot be read against the headword is not guessed: a second
  augment after the preverb (`ἠμφεσβήτησα`, `ἐπηνώρθωσα`), a contracted one
  (`προὔβαλον`), a syllabic augment before a vowel (`ἔωσα`). Then only the
  indicative is built. The same holds for the κ aorists of the -μι verbs in
  the active (`ἔθηκα`, not `θήκας`) and for root aorists that go like a
  passive in the indicative alone (`ἀνθυπέστην`, `ὑπέφθην`).

A compound also passes `prefix=`, so that the accent of the indicative, the
perfect and the pluperfect stays on or after the augment or reduplication
(`παρέσχον`, `ἀπῆρα`, `ἀφῖγμαι`). The printed part decides: where LSJ has the
accent on the preverb (`ἄφευσα`, epic `περίχευα`), none is passed. For a
perfect the preverb is taken only where the stem shows a perfect of the rest
after it (`δια-πεφυκ`, `μετ-αγηοχ`), because the letters often allow two
divisions (`ἀπαν-αλίσκω`, `ἀπανα-λίσκω`).

Other tenses:

- A middle principal part builds the middle alone (`ἐδεξάμην`, `δέξομαι`,
  `ἀφικόμην`: `form=mid`), and a passive aorist in `-θην` under the plain
  label is a passive one (`ἐβουλήθην`).
- A future with a circumflex is contracted (`βαλῶ` `fut-con-e|βαλ`, `βιβῶ`
  of a verb in `-άζω` `fut-con-a|βιβ`); read as an ordinary future it gave
  `βάλεις`.
- The pluperfect is built on the perfect stems as before (`λελύκειν`) and,
  for a visibly reduplicated stem, with the augment too (`ἐλελύκειν`,
  `ἀπελελύκειν`).
- A contracted imperfect takes the contract template of its present
  (`ἀτίμων`, `ἠγνόουν`).

The present system of every verb is written before any principal part, so a
cell a headword spells out itself (`φιαλοῦμαι` of `φιαλόω`) outranks the same
form derived from another verb's printed part (the future `φιαλῶ` of
`φιάλλω`). A headword that another verb prints as its principal part
(`ὑπομνήσω` under `ὑπομιμνήσκω`) is written after the printed parts, and the
inferred parts below come last.

### Principal parts that no entry prints

Most LSJ verbs print no principal parts: a compound leaves them to its
simplex, a denominative to its class. Such a verb had its present system
only, so `εἰκόνισεν` had no Ancient row. The pass states the missing parts
as hypotheses and keeps the cells a corpus attests. On the 1.3.26 inputs,
8,206 compounds take parts from a simplex and 12,443 verbs from their class;
8,953 of them keep at least one cell, 50,331 cells in all (99,406 keys with
the unaccented spellings).

- A compound takes the parts its simplex prints (`compound_parts`):
  `συγκατατρίβω` gets `συγκατέτριψα` and `συγκατατρίψω` from `ἔτριψα` and
  `τρίψω` under `τρίβω`. The last preverb is spelled for the sound that
  follows: elided before a vowel and aspirated before a rough one
  (`κατ-έτριψα`, `καθ-είλκυσα`), assimilated before a consonant
  (`συλ-λήψομαι`, `ἐγ-κεχλίαγκα`), `ἐξ` before a vowel, with the `ρ` doubled
  after a vowel (`καταρρεύσομαι`). The simplex is the verb left after the
  longest chain of preverbs, and only when every shorter chain that also
  leaves a verb leaves a compound of it; two readings that disagree
  (`ὑπ-ερείδω`, `ὑπερ-είδω`) or two homographs give none, and an aspirated
  preverb needs a rough verb after it (`καθ-έλκω`).
- A verb in `-ίζω`, `-άζω`, `-εύω`, `-έω`, `-όω` or `-άω`, or the middle of
  one, takes the regular parts of its class (`regular_parts`): `εἰκονίζω`
  gets `εἰκονίσω`, `ᾐκόνισα` and also `εἰκόνισα` (a diphthong is tried with
  and without the augment), `ᾐκόνικα`, `ᾐκόνισμαι`, `ᾐκονίσθην`, and the
  Attic future `εἰκονιῶ`. The augment stands in front and, in further
  candidate sets, after each preverb the letters allow (`ἐπιστατέω`:
  `ἠπιστάτησα` and `ἐπεστάτησα`). A compound whose simplex prints nothing
  takes the regular parts of the simplex.
- Nothing is inferred from a class for a stem under three letters (`ἐρέω`,
  `ὁράω`, `ἀρόω`), for a headword that may be a compound of such a verb
  (`ἀπο-θέω`: `ἀποθήσω` is the future of `ἀποτίθημι`), or where another
  dictionary verb would share the tenses (`regular_rivals`): `-άω` beside
  `-άζω`, `-έω` or `-όω` on the same stem (`βοάζω`, `βοάω`; `δαπανῶσαι` is the
  aorist of `δαπανόω` and a participle of `δαπανάω`), or the same verb with
  the other of `ἀ-` and `ἐ-`, which take one augment (`ἠπόρημαι`). Nothing is
  inferred at all for a capitalised headword (`Βακχεύω`, which the corpora
  file under `βακχεύω`) or for a verb whose other voice has an entry of its
  own (`θεάω` beside `θεάομαι`). A tense the entry prints itself is never
  inferred.

These parts are hypotheses, so they are expanded like printed ones but only
the cells a corpus attests are kept: the exact spelling must be in
`data/form_profile.db` (grave folded to acute, case folded, as in the output
gate of the attestation runtime). The template gives a participle in its
nominatives only, and the runtime resolves the other cases of a participle in
`-μενος` through the masculine nominative, so that cell is also kept when a
corpus has another case of it (`ἐσκοτισμένους` for `ἐσκοτισμένος`). A
template call is skipped when no attested form begins with one of its stems.
Further:

- a cell the table already has, or that is a dictionary headword itself, is
  left alone (the "perfect participle" `ἱππικός` of `ἱππίζω`; a perfect
  without visible reduplication is not inferred for a dental stem at all);
- a cell that two verbs infer goes to neither;
- a cell that a noun also spells is kept only where a treebank reads it as
  this verb: a future in `-σει` or `-σεις` (`-ξει`, `-ψει`) is the dative or
  plural of the action noun in `-σις` (`ἐπισκιάσει`, `ἀμοιρήσεις`), and the
  participle of an Attic future in `-ιῶν`, or of its uncontracted `-ιέων`, a
  genitive plural (`κοσμιῶν`, `πεδιέων`).

### Treebank readings

The lookup build fills gaps from the treebank pair files in a fixed order
(`perseus_pairs.json`, `glaux_pairs.json`, `diorisis_pairs.json`,
`nt_pairs.json`, `ptnk_pairs.json`, `tagnt_pairs.json`,
`pedalion_pairs.json`): the first file that has a form speaks for it, and
within the file its first pair. The verb pass reads the pair files the same
way (`load_treebank_readings`). The grave counts as the acute. A file that
has the form only with a capital (`Κατεδέξατο`) speaks for it when no file
has it as spelled, unless the lemma is capitalised too (`Φιαλέων` of
`Φιαλεύς` says nothing about `φιαλέων`). The reading that counts is the
first usable one (`treebank_reading`): the verb itself, by its letters
(`ἐνθρῴσκω` for `ἐνθρώσκω`), a headword of some dictionary in
`data/grc_lexical_heads.json` that is no reviewed non-lemma, or the verb in
its other voice (`ἐπιλανθάνομαι` for `ἐπιλανθάνω`). A lemma that no
dictionary heads (`σύν-εἶδον`, `ὑπαλειψάννυμι`) is passed over.

A principal-part cell is checked against that reading, and how much the
reading weighs depends on the stem under the cell:

- **A stem no entry prints.** A cell of an inferred part, a cell on the
  unaugmented stem that the generator derives for a printed past tense
  (`καταδέξας` beside `κατέδεξα`, `ἀγαγών` beside `ἤγαγον`) and the augmented
  pluperfect are the generator's own. A derived stem that the entry prints
  in its future is not among them (`ἀμφιεσ-` of `ἠμφίεσα` stands in
  `ἀμφιέσω`). Such a cell is left out when the
  treebanks read the form as another word, and the lookup build takes the
  treebank reading as before: `ἠβουλήθη` stays with `βούλομαι`, not
  `ἀβουλέω`, and `ἀνακλάσεις` with the noun `ἀνάκλασις`, which the first
  pair names.
- **A stem the entry prints.** The future, the perfect, the imperfect and the
  augmented indicative of a printed past tense are the dictionary's. So are
  the other moods of a past tense with a plain `ἐ-` before the stem
  (`ἐζημίωσα`, `ζημιῶσαι`), which the conjugation module takes off by
  itself. Another verb in a treebank does not move such a cell: `ἀμφιέσομαι`
  stays the future LSJ prints under `ἀμφιέννυμι`, though a treebank files it
  under `ἀμφιάζω`. The cell is left out only when the verb the treebanks
  read has it among its own principal parts, printed or inferred
  (`κατεδέξατο` is the aorist of `καταδέχομαι` in the treebanks, a compound of
  `δέχομαι`, while LSJ prints `κατέδεξα` as the Ionic aorist of
  `καταδείκνυμι`: both verbs give the cell, and the treebanks decide), when
  the treebanks read the form as no verb at all (`φιαλέων` is the genitive of
  `φιάλη` before it is a participle of the future `φιαλῶ`), or when they file
  it under the same verb in its other voice (`ἐνόσφισεν` under `νοσφίζομαι`).
- **A present cell of another verb.** The present systems are written first,
  so a present holds the form when a printed part of another verb spells it
  too (`περιειλόμεθα` of `περιείλω` and of the aorist `περιειλόμην` under
  `περιαιρέω`). Where the treebanks read the form as the verb of the printed
  part, that verb takes it.

The corpus tables only withhold or confirm cells; no form or lemma enters the
table from them.

The pass therefore needs `data/form_profile.db`, `data/grc_lexical_heads.json`
and the seven pair files, and stops when one is missing. `--no-corpus-checks`
runs without them (no inferred parts, no treebank check); that is not the
published build. What the corpus tables decided is written to
`data/ag_verb_generation_audit.json`: the content hash of `form_profile.db`
and the sha256 of the other inputs, the inferred cells by verb with the kind
of inference and the simplex, the cells two verbs inferred, the cells left
to a treebank reading, each with that reading and the kind of stem
(`printed`, `derived`, `inferred`), and the present cells that went to a
printed part.

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
