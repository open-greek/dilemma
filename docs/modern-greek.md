# Polytonic Modern Greek word list

[README](../README.md)

- [Build and source selection](#build-and-source-selection)
- [Accents and shared vocabulary](#accents-and-shared-vocabulary)
- [Spellings to avoid in Modern text](#spellings-to-avoid-in-modern-text)
- [Recorded attested-list reviews](#recorded-attested-list-reviews)
- [Generating missing forms](#generating-missing-forms)
- [Choosing lexical owners](#choosing-lexical-owners)
- [Subjunctive spellings](#subjunctive-spellings)
- [Frequency classes](#frequency-classes)
- [Rare forms and reviewed vocabulary](#rare-forms-and-reviewed-vocabulary)
- [Generated-form evaluation](#generated-form-evaluation)
- [Author and document holdouts](#author-and-document-holdouts)
- [Keyboard replay comparisons](#keyboard-replay-comparisons)

This guide describes the exporter rules and records dictionary and keyboard
evaluations from earlier builds. Counts and results refer to those builds;
the exported `.version` file identifies the inputs to a new run.

## Build and source selection

The grc dictionary is built from Ancient and Medieval Greek sources, so it
has almost no Modern Greek vocabulary, and on purpose it rejects the
unaccented `του` and the Modern spellings among the language model's
commonest words (`κι`, `της`, `μας`, `τώρα`). A keyboard that ships only grc
turns polytonic Modern Greek (Modern Greek written with breathings and
accents, as in print before 1982, in church texts and by some writers
today) into Ancient Greek neighbors: `τώρα` becomes `τἄρα`, `ἀκόμη`
becomes `ἀκμὴ`, `σπίτι` becomes `σπίτιν`. `export_mg_polytonic.py` writes a
separate list that such a keyboard can merge into grc, and leaves
`grc_polytonic.dic` as it is:

```bash
python export_mg_polytonic.py     # build/hunspell/grc_mg_polytonic.{dic,aff,version}
python export_mg_polytonic.py --holdout-author-fold 0/5 --out-dir DIR   # evaluation list
```

The `.dic` is a plain list, one spelling per line with the grc `fr:` field
and no affix flags; the `.aff` only makes the pair loadable on its own. No
entry is a spelling `grc_polytonic.dic` accepts, as written or, for a
capitalized spelling, through its lowercase entry, so the whole list is
Modern-only. The `.version` sidecar records the sha256 of the grc dictionary
the list was selected against, and the exporter must be rerun after every
grc rebuild. An evaluation variant needs `--out-dir` and is refused in
`build/hunspell`, where it would replace the shipping list. The source is the
polytonic Modern Greek slice of Wikisource that the next-word language model
also reads (`extract_polytonic_mg.py`), and only the language model's
training sentences, so its dev sentences stay held out; a training sentence
that repeats a dev sentence word for word, case aside, as Wikisource's
duplicate copies of a text do, is left out too. A spelling is listed when:

- it comes from a polytonic source. A document is dropped when more than 1%
  of its lowercase words, with 95% confidence (the Wilson lower bound), and
  at least three, read as monotonic: a vowel-initial word without a
  breathing, or a monosyllable polytonic writing always accents (`και`,
  `να`, `δεν`, `στο`, ...), counted on the training sentences. The bound
  keeps a short polytonic poem with three OCR slips; it drops 39 of 2,213
  documents, 10.5% of the slice's words. Inside the rest, a sentence with any
  such word is skipped, which also drops the monotonic license note
  Wikisource adds to its pages. Elision and aphaeresis marks are read as
  the texts write them: the apostrophes, the koronis, and the spacing psili
  and dasia (`νά ῾βρω`), which in front of a vowel is that vowel's breathing
  (`῾Η` is `Ἡ`);
- it has at least 3 tokens from at least 2 authors, in its own case: the
  lowercase spelling counts its lowercase tokens, a capitalized one (a name)
  its capitals inside a sentence, and a capital that opens a sentence, line,
  quotation or stretch of dialogue, or follows an ellipsis or a closing
  bracket (`[Γελῶσα] Χάχ`), goes with the case in which more authors write
  the word inside a sentence, the word's grave and acute spellings
  together. A capital carrying the iota (`ᾍδη`) is a capital, though
  Unicode makes it a titlecase letter. So a word and a name are both listed
  when each qualifies
  (`διάολος`), and neither passes on the other's authors (`λάζος`, one
  author's "blade", beside the name `Λάζος`); a capitalized spelling whose
  lowercase is listed adds nothing a Hunspell reader does not accept, and is
  left out. Measured with each dev word's author held out (below),
  one-author spellings would add 3,200 entries that cover 0.8% more of the
  dev words when their authors are read and 0.2% more when they are not; a
  floor of 2 tokens would add 9,100 entries, of which 60% have a monotonic
  spelling Dilemma's lexicon knows, against 68% at 3 or 4 tokens;
- it is well-formed polytonic Modern Greek: the grc structural rules and
  explicit rejects, in either case, except that the weak possessive pronouns
  `του`, `της`, `μας`, `σας`, `τους`, `των` and `κι` are correct unaccented,
  syllables are counted with synizesis (`τέλειωσε`, `πλάγιασε`, `γάϊδαρος`
  keep their accent within three), a lowercase word may end in a consonant
  when it is not an attested word that lost its last vowel (`ἄχ`, `κονιάκ`,
  but not `τόμ`), and an elided word may end in the glide ι (`κι᾽`,
  `γι᾽`). An unaccented elided spelling must stand for an attested word
  accented on the vowel it lost (`γι᾽`, `γιατ᾽`, `ἐδ᾽`), so `μητ᾽`, `φερ᾽`
  and `μεσ᾽` are out: a paroxytone keeps its accent (`μήτ᾽`, `μέσ᾽`). The
  weak accusatives `το`, `τα`, `τον`, `την` stay out: a bare-typed `το` is
  almost always the article `τὸ`;
- it is not a misspelling of another spelling in the slice. A spelling whose
  breathing alone differs from another is one when the other is five times
  as common, or at least as common and accepted by grc, however rare both
  are (`ἔτοιμος` beside `ἕτοιμος`, `εἷνε` beside `εἶνε`); a reviewed list
  keeps the real homographs (`ὅντας`, `ἄρματα`, `οὗλα`).
  Any other mark-only respelling with under 5% of the count of a spelling of
  the same letters that has at least 100 tokens is one too (`εἴχε` beside
  `εἶχε`), unless its monotonic spelling differs and Dilemma's lexicon knows
  it, which keeps different words (`χρονιά` beside `χρόνια`, `ἔμενα` beside
  `ἐμένα`); such a word must still not be that rare beside the commonest
  spelling with its own monotonic letters and accent (`ὄποιος` goes beside
  `ὅποιος`). An elided spelling counts as another word only through an
  attested word it stands for (`κάθ᾽`, for `κάθε`, beside `καθ᾽`);
- a recorded review did not reject it: `data/hunspell_grc_spelling_review.json`
  and `data/hunspell_mg_spelling_review.json`, which lists the accent-type
  and iota-subscript misspellings no rule catches (`ἤταν`, `ἐδῷ`, `ξῦλον`,
  `χειμώνα` beside the slice's five times commoner `χειμῶνα`), two words run
  together (`τουκαὶ`, and `εἰδ᾽`, written for `εἰ δ᾽`) and `τουναντίον`
  without its koronis.

## Accents and shared vocabulary

Each oxytone also gets its contextual twin (`στὴν` beside grc's `στήν`).
`fr:` is the higher of two buckets: the grc bucket edges on the spelling's
token count in the slice, the two twins counted together, and the bucket of
its monotonic form's count in monotonic Modern Greek text
(`data/mg_freq.txt`, OpenSubtitles, 263M tokens) at the rates per token of
grc's edges on its 27M-token corpus: `C` from 37 per million (9,724
tokens), `M` from 3.7 (973). A traditional subjunctive in -ῃ or -ῃς counts
with its monotonic spelling in -ει or -εις, and an elided spelling has no
monotonic count. The counts are of lowercased text, so they count a name
with the word of the same letters: a lowercase spelling whose monotonic
form the slice writes more often capitalized inside a sentence, with 95%
confidence (the Wilson lower bound of the capitalized share above one
half), gets no bucket from them (`κίνα`, beside `Κίνα`), nor a capitalized
one the slice does not write so. A handful of capitals is no such
evidence: the slice's 4 `Τσάντα` beside 2 `τσάντα` say nothing of the
noun's 18,966 monotonic tokens. A name the slice does not write still
lends its tokens to a generated form (the imperative `κίνα` is `M`, and
`μάσα`); Wiktionary's list of names is no guard, since it lists rare ones
too (`Σταμάτα`, beside the imperative `σταμάτα`). A generated form ranks no
higher than an attested spelling of its letters, whatever their counts:
the imperative `μίλα`, `C` by its own, is `M` like the attested `μιλᾶ`. A
spelling with the second accent an enclitic gives it (`ὁδήγησέ`) is the
same word as the plain one and sets no such ceiling. The slice is 1.7M
tokens of older prose, so its counts alone put 234 spellings above `R`;
monotonic text raises 3,109 more from `R` (`βρήκαμε`, `ἀγόρια`,
`καλοκαῖρι`, `λέμε`) and 120 from `M` to `C`, and gives the generated forms
below their only bucket. A spelling in `-ῃ` or `-ῃς` takes only its share of
its `-ει` form's count (below). The `.version` records where the buckets come
from (`bucket_source`). A keyboard may take a `C` or `M` word for a sign
that a text is Modern Greek, or write letters that only such a word has as
that word, so these get `fr:R` whatever their count:

- one the GLAUx and Diorisis treebanks attest 20 times (`του`, the Ancient
  enclitic genitive of `τις`), or the contextual twin of a grc spelling they
  attest that often. A treebank count is the larger of the two treebanks'
  counts: they annotate largely the same works, so their sum counts most
  tokens twice (`τὴ`: 15 and 11, 9 of them the same lines of Aristophanes),
  and taking each work from one of them would lose a word the other spells
  differently (`οὔτε`: GLAUx 0, Diorisis 13,323);
- one the patristic texts (PTA) or First1KGreek write, with its twin, 20
  times and at a rate at least a thirtieth of its rate in the Modern Greek
  slice (`μοναχὸς`, beside the patristic and Byzantine `μοναχός`). Below
  that rate a Modern spelling in those OCR'd editions is a misprint or a
  quotation (`στήν`, `δὰ`), so `στὴν` and `στὸν` stay signs of Modern Greek;
- one whose letters grc has under no spelling while the treebanks write
  them 20 times: the interjection `τάκ`, whose letters are those of the
  crasis `τἀκ`, would otherwise turn a passage of Ancient Greek that
  writes `τἀκ` into Modern Greek.

The Patrologia's OCR is not read, nor the Byzantine vernacular corpus, which
is early Modern Greek itself. The counts come from Dilemma's form
attestation (`data/form_profile.db`, `python -m dilemma download
--with-attestation`); without it the exporter warns and caps nothing. In the
current list this caps `του`, `των`, `μοναχὸς`, `μίας`, `κό`, `πὲ` and
`τάκ`, with their twins. The `.version` records
the sha256 of every file that decides the list: the grc dictionary,
`lookup.db`, `form_profile.db`, and for the generated forms and the
monotonic buckets the verb paradigms (`data/mg_pairs.json`) and the
monotonic counts (`data/mg_freq.txt`).

## Spellings to avoid in Modern text

The list also names the grc spellings polytonic Modern Greek writes another
way, with lines of the form `form<TAB>mg:avoid`, so that a keyboard writing
Modern Greek can leave them out of its candidates: the Katharevousa dative
`ἐκείνῃ` beside `ἐκείνη`, misspellings such as `εἷναι` beside `εἶναι` or
`μὰς` beside `μᾶς`, and the weak forms `με`, which the slice writes `μὲ` 25
times as often, `σε`, `που` beside `ποὺ` and `ποῦ`, and `πως`. A lowercase
spelling is marked when it is a respelling of its word, the spellings of the
same letters with the same monotonic spelling (accents as the acute,
breathings and iota subscript dropped):

- its share of the word's lowercase tokens is under a tenth with 95%
  confidence (the Wilson upper bound, which needs at least 53 tokens of the
  word), and the word's commonest spelling takes at least a tenth, is a word
  grc or the list accepts and is written by at least as many authors. Another
  word of the same letters does not count: `ὁπού`, 26% of the demotic
  relative's spellings, is not marked though `ὅπου` fills its letters;
- a spelling of the word that the slice writes more often and grc or the
  list accepts differs from it only in marks, and where they differ in the
  breathing, the accent type or a dropped iota subscript, the GLAUx and
  Diorisis treebanks do not write the rarer one 20 times and a twentieth as
  often as the other, each with its contextual twin and both treebanks
  counted. An added subscript, with the accent it brings, is a respelling
  outright: polytonic Modern Greek writes the Katharevousa dative `ψυχῇ` as
  `ψυχή`, while the treebanks, where the dative is a word, cannot say so
  (taking the datives to the treebanks too unmarks 45 lines and costs 15
  words on the folds below). The test keeps the Katharevousa reflexive
  `αὑτοῦ` beside `αὐτοῦ`, `ὄν` beside `ὅν`, `ἤν` ("if") beside `ἦν` and
  the relative `ἥ` beside the dative `ᾗ`. The breathing misspellings the
  slice writes (`εἷναι`, `ἃν`, `ἄμα`, `ὦν`) are at 3.3% of their word or
  below, but a misspelling whose spelling is also an Ancient word goes
  unmarked (`ὀποῦ`, the genitive of `ὀπός` "sap", beside the relative
  `ὁποῦ`; `σάς` beside `σᾶς`), and the rarest reflexives (`αὑτά`, `αὑτήν`,
  `αὑτοῖς`, 4.0 to 4.6%) stay marked, as does the subjunctive `ᾖ`, which
  adds a subscript to `ἤ`.

So a spelling that is another word is never marked: `ὅ` of `ὅ,τι`, the
numerals `ἕν`, `εἷς` and `ἕξ`, `πάρα`, `ὥς`, `πασᾶν` and the crasis
`ταὐτά` differ from their commoner neighbors in the monotonic spelling. The
weak forms `με`, `σε`, `που` and `πως` are reviewed exceptions: unaccented,
they are words of their own (clitics and Ancient enclitics), but polytonic
Modern Greek writes them accented in most of their uses. So is `τῶ`, the
`τῶν` of Εφταλιώτης and Μαβίλης without its final ν, which has the
monotonic spelling of the dative `τῷ` and is never marked. Nor is a word
with the second acute an enclitic gives it (`γυναῖκά` beside `γυναῖκα`), a
spelling the list or its contextual twin carries, or one of the unaccented
closed list. Elided and aphaeresized spellings are not judged, and only
grc's spellings get a line, with the contextual grave of an avoided acute.
A Hunspell reader that does not know the field accepts the marked
spellings, as grc does.

## Recorded attested-list reviews

The recorded build had 16,774 attested words (759 in `C` and 2,408 in `M`),
49,344 generated spellings (below), 189
`mg:avoid` lines and 6,651 `mg:shared` lines. The attested words include 969 capitalized, 152
elided and 197 aphaeresized spellings (`᾽ναι`,
`᾽ς`), the six weak
pronouns, `κι` and `κι᾽`, `στὴν`, `στὸν`, `στὴ`, `στοὺς`, `τὴ`, `γι᾽`, and
the Modern vocabulary (`τώρα`, `ἀκόμη`, `σπίτι`, `ὄχι`, `μπορεῖ`). `θὰ`,
`νὰ`, `γιὰ`, `στὸ`, `στὰ`, `δὲν`, `ἀπ᾽`, `μου` and `σου` are in grc already.
An adversarial review of the earlier 17,789-entry list judged 500 random
entries and every entry whose breathing, accent-type or iota-subscript twin
is five times commoner, and confirmed 82 errors, about 1 in 200; the rules
and the review file above remove all of them but `ποιητῆ`, which a second
look found correct (the genitive, beside the accusative `ποιητή`). Of a new
random sample of 200 entries none is an Ancient-only or OCR-damaged
spelling; the rarest are older spellings and dialect words (`θᾶμμα`,
`κἀμμία`, `ὁλοκαιρὶς`, `μίλαε`, crasis such as `σὅλους`). A later review found
no error in 300 random entries (every `C`, 100 `M`, 179 `R`) and two in 50
each of the capitalized, elided and aphaeresized ones: `εἰδ᾽` (now in the
review file) and `Χὰχ`, an interjection that only sentence-initial capitals
had made a name; a scan of every capitalized entry added `Χάχ`, `Ἀετὲ`, and
`᾽Σ` beside `᾽ς`. The rules above now leave all of them out.

## Generating missing forms

The slice is mostly older prose, so it rarely attests the second person and
the spoken forms of common verbs, or the words of everyday life: it has
`ἔρθει` and `ἔρθουν` but not `ἔρθεις`, no `μιλήσεις`, `πιεῖς` or
`καταλάβεις`, and no `τηλέφωνο`, `κινητό`, `αὐτοκίνητο` or `ἐνημέρωση`. A
keyboard with the attested list alone turns these into Ancient neighbors
(`μιμήσεις`, `πιεῖν`, the datives `αὐτοκινήτῳ` and `κέντρῳ`), and with the
verbs generated but not the nouns, into verb forms of the same letters
(`τηλεφωνῶ`, `ἐνημερώσῃ`). `mg_polytonic_paradigms.py` writes polytonic
spellings for the inflected forms of every part of speech in Wiktionary's
monotonic paradigms (`data/mg_pairs.json`, written by `build_data.py`):
verbs, nouns, adjectives, names, adverbs, pronouns, numerals and the rest,
but not romanizations, abbreviations or symbols. It does so for every form
that monotonic text has at least 50 times (`data/mg_freq.txt`,
OpenSubtitles). A cell Wiktionary tags rare, archaic, dialectal, obsolete or
dated is generated like any other (`ὑπακούουν`, `συντριβεῖ`,
`ἀπεχθάνεσαι`), unless another part of speech has its letters, whose count
it would borrow (`πιάνου`, the piano's); Katharevousa cells and datives are
not generated. These Wiktionary rows are left out:

- forms filed under an abbreviation in capitals (`ΔΕΣ` for `δες`), whereas
  a page title in a heading's capitals is read in lowercase (`Ερωτώ`, with
  `ρώτησα`);
- the pieces of tables Wiktionary could not split into stem and ending
  (`σαλπάρω`'s `ούμε`, `όμαστε`, `όσουν`). Such a piece starts with another
  letter than its word, even after an augment, and is the very ending of
  the same cell in at least 5% of the words of its part of speech; a
  suppletive form (`φάγατε`, `είπα`, `ήρθα`) also starts otherwise but ends
  no other verbs' forms;
- spellings that are polytonic already (`ἔχομεν`, `ζᾶ`);
- a noun's or adjective's row with no case or number where another row of
  its part of speech places the form (`αρκτικών`, filed only as a form
  under the sentence-case page `Αρκτική`, beside its cell under
  `αρκτικός`);
- a name's form with the letters of another word (the surnames `Δώση`,
  `Γράψη`), which is that word at the start of a sentence, and the names
  Wiktionary does not decline in any form (`Τέσα`, `Τζέιμς`, `Γουίλιαμς`),
  foreign names in transliteration, which Greek writes in many ways.

## Choosing lexical owners

A form several words share gets one spelling, from the word that is a verb
rather than another word's page, then the one whose letters it shares the
longest beginning with (`πουλᾶμε` is `πουλάω`'s, not `πωλώ`'s), reading the
attested spellings of all of them (`ἑπόμενες`, beside the attested
`ἑπόμενη` of `επόμενος`). When independently inflected nouns supply
different explicit case-and-number cells, their whole-paradigm monotonic
counts select the primary owner. Thus common feminine `σκιά` retains its
own nominative, and `σκατά` the neuter plural of `σκατό`. Adjective or
adverb readings of an inflected form do not invalidate its independent
noun head. Nouns supplying the same cells retain the closest lexical
stem, so a diminutive or synonym in another word's table does not acquire
that word's declension. Finite verbs, names, and mixed headword analyses
keep their existing selection rule. An independently defined adjective or
adverb sharing the form does not reset an already selected noun owner:
the noun cells still determine the primary and rare alternative readings.
This refinement leaves a preferred adjective, finite verb, or name in its
existing selection path.

A different explicit cell of another independently inflected noun remains
at `R` with zero pooled count: `χημεῖα`, plural of `χημείο`, remains beside
`χημεία`, and masculine genitive `σκατᾶ` beside neuter plural `σκατά`.
These alternatives do not borrow the common homonym's frequency or
provide evidence for raising another spelling's rating. A monotonic
spelling has the letters, the
accented vowel and any diaeresis of the polytonic one. The rest is decided
as follows:

- the breathing of an initial vowel or `ρ`: the verb's attested polytonic
  forms with the same first two letters (`ἔρθω`, `ἔρθει` -> `ἔρθεις`);
  without them, the grc words with the longest prefix of the word, up to 7
  letters, that at least 5 of them share, when 90% of those agree
  (`ἀναρωτιέμαι` from `ἀνα-`, `εὑρέθηκα` from `εὑρε-`, `ἑστιάζω` from
  `ἑστια-`). A prefix that many words share but do not agree on decides for
  the default, since its words are already further from the word than
  those of the longer one: `εστι-` is split by `ἐστί`, and `εσ-` below it,
  smooth almost throughout, would make `ἐστιάζω`. The default is the smooth
  breathing, the rough one on `υ` and `ρ`. An augment before a consonant is
  smooth (`ἦβρε`, not `ἧβρε` after `ἥβη`). On a diphthong the breathing
  goes on the second vowel;
- the type of the accent, by the Ancient rule that polytonic Modern Greek
  keeps, with syllables counted the Ancient way (`ἤ-πι-ε`): an accented
  antepenult takes the acute; an accented penult the circumflex when its
  vowel is long and the last syllable short (`δῶσε`, `ποῦμε`, `ἦρθα`,
  `εἶσαι`), else the acute; an accented last syllable the circumflex when
  its vowel is long (`μπορεῖς`, `δοθεῖ`, `πιῶ`, `ἐπιζῶν`), else the acute
  (`πές`). Only the aorist passive participle in -είς takes the acute on
  its long last syllable (`γραφείς`, beside the dependent `γραφεῖς`). `η`,
  `ω` and the diphthongs are long, `ε` and `ο` short, and a final `αι` or
  `οι` counts short. For `α`, `ι` and `υ` the verb's attested forms decide,
  where the type of their accent shows the length, if they have at least 3
  tokens, the list's own floor: one token of `γελᾶστε` is a misspelling.
  Before a consonant of a perfective stem only forms with that consonant
  too count, since the long `α` of a contraction (`γελᾶτε`, `κοιτᾶς`) says
  nothing of the stem's (`γελάστε`, `κοιτάξτε`, `κοιτάχτε`); and a
  perfective form's acute (`σκάσε`) says nothing of an imperfective
  contraction (`σκᾶς`, `σκᾶνε`). Without such forms, the contracted endings
  are long: of the `-άω` verbs and of those filed under a polytonic `-ῶ`
  (`μιλᾶμε`, `μιλᾶτε`, `μιλᾶς`, `μιλᾶνε`), of the passive `-άμαι` and
  `-ώμαι` (`θυμᾶμαι`, `αὐταπατᾶσαι`) and of any imperative plural in `-ᾶτε`
  (`ἐλᾶτε`); any other vowel is short (`κάνε`, `σκάσε`). The `-α` of an
  imperative counts long (`ζήτα`, `φεύγα`). Monotonic writing leaves a
  monosyllable unaccented (`πιω`, `πεις`), so its accent goes on its last
  vowel (`πιῶ`, `πιεῖς`). The gerund keeps the acute, which the slice writes
  283 times to the circumflex's 241 in the paradigms' gerunds (`μιλώντας`,
  `ζητῶντας`);
- for a noun, adjective, pronoun or name, the same rule, with the length of
  a last syllable that its vowel does not show taken as Ancient Greek has
  it. A final `-ι` is long (the neuters from `-ιον` keep their acute:
  `συκώτι`, `γιαούρτι`). A feminine singular `-α` is long (`χώρα`, `καρδιά`)
  except after `σσ`, `ττ`, `ζ` or a diphthong and `ρ` (`γλῶσσα`, `μοῖρα`)
  and in the `-εια` of an adjective in `-ύς` (`ταχεῖα`), and a feminine
  genitive `-ας` is long; a neuter plural, a masculine or any other `-α` is
  short (`σχολεῖα`, `μῆνας`, `παιδιά`), and a noun in `-μα` is neuter when
  its tags do not say (`νῆμα`). An accented last syllable takes the acute
  (`ψυχή`, `καλοί`, `κινητό`), and the circumflex, when it is long, only in
  the genitive (`ψυχῆς`, `καλοῦ`, `καρδιᾶς`, `καρδιῶν`, `δικιᾶς`), in a word
  of one syllable, in an adverb (`ἀκριβῶς`), and in the endings Ancient
  Greek contracted: the nouns in `-άς`, `-ούς` and `-ού` in every case
  (`μπαμπᾶς`, `μπαμπᾶ`, `παπποῦς`, `ἀλεποῦ`), the plural `-εῖς` of a noun or
  adjective (`γονεῖς`, `ἀληθεῖς`, `ταχεῖς`, but not an aorist passive
  participle filed under itself, `γραφείς`), and every ending of an
  adjective in `-ής` with a neuter in `-ές` but the nominative singular
  (`ἀληθῆ`, `ἀληθοῦς`). An adverb's `-ά` is short (`ξανά`). A form
  Wiktionary files with no cell, in `-ού` or `-ών`, is a genitive
  (`ἐθνικοῦ`, `πνευματικῶν`). The length of an `α`, `ι` or `υ` before the
  last syllable comes from the word's attested forms with the same letters
  through the next one, as for a verb, and is short without them. A name
  keeps its capital (`Λονδίνο`, `Ἰούνιο`);
- the iota subscript: the verb's attested forms with at least 3 tokens
  carry it over to the same vowel before the same letter (`σῴσει` beside
  `σῴσῃ`, but `ἀγαπάει` beside `ἀγαπᾷ`), and the contracted `-ᾶς` and `-ᾶ`
  take it as most of the slice's tokens of such forms do (69% and 83%:
  `ἀγαπᾷς`), except in a monosyllable (`πᾶς`).

## Subjunctive spellings

The texts write the subjunctive of the second and third person singular
both as the indicative (`νὰ πάρεις`, `νὰ πάρει`) and the traditional way
(`νὰ πάρῃς`, `νὰ πάρῃ`), so both are generated, for the dependent forms, for
the present forms that serve as their imperfective subjunctive, and for any
form in `-ει` or `-εις` that Wiktionary tags only as a form of its verb
(`ξαναδῇ`, `ὁδηγῇς`). Monotonic writing spells the subjunctive `-ει(ς)` and
writes it `-η(ς)` only by mistake, a few times in a hundred; where it has
the traditional spelling's letters and accent 50 times and a tenth as often
as the form, they are another word's, and the twin is left out (the noun
`υπολογιστή`, 12,664 tokens, beside the subjunctive `ὑπολογιστῇ`).

A generated spelling is listed only where neither grc nor the attested list
has a lowercase spelling of its monotonic form (a diaeresis aside:
`χάιδεψε` is the attested `χάϊδεψε`; a name such as `Δροῦσε` covers no
`δροῦσε`), so it fills a gap. A Katharevousa dative of a noun, adjective or
pronoun neither covers nor competes with the plain spelling Modern Greek
writes (`κέντρῳ` beside `κέντρο`), while a verb's subscript is its
contraction's (`σταματᾷ`) and competes like any other spelling. Other
spellings of its letters, in the attested list, in grc or generated before
it, compete with it for the same typed letters by the cell each spells, a
cell being a word's form (``_stops``): two cells of the same letters are
two forms, and both go in for the context to tell apart, whatever their
counts (the imperative `περπάτα` and the present `περπατᾷ`, the noun
`τηλέφωνα` and the imperative `τηλεφώνα`; `ξεκινᾷ` beside the attested
`ξεκίνα`). Two spellings of the same cell are one form spelled two ways,
and the generated one gives way to the attested one (`ἀνέβουμε` to
`ἀνεβοῦμε`), to one generated before it, which monotonic text has at
least as often (`πεθάναν`, 61 tokens, to `πέθαναν`, 8,404), and to a grc
spelling whose monotonic form monotonic text has at least as often as its
own (`φάκελους`, 59, to `φακέλους`, 2,712; `χειροῦργος`, from the
misaccented `χειρούργος`, 1,286, to `χειρουργός`, 1,643). A noun's or
adjective's cell is its case and number, the same under the noun and the
adjective of one word (`ὑπάλληλους` gives way to `ὑπαλλήλους`), and another
word of the same letters is another form (the adjective `ἔνοχη` beside the
noun `ἐνοχή`). A spelling whose monotonic form is in no paradigm (an
Ancient word, a word Wiktionary lacks) stops a generated one only where
monotonic text writes it twice as often (`RIVAL_RATIO`), so `καταλαβεῖς`, of
`καταλαβεύς` "clamp", leaves `καταλάβεις`, and a subjunctive twin, whose
letters monotonic text writes only by mistake, gives way to any spelling of
its letters it writes twice as often (`ἐνημερώσῃ` to the noun `ἐνημέρωση`,
3,970). The generated spellings come with the one whose own letters and
accent monotonic text writes most first. A generated spelling must
pass the list's orthography rules and recorded reviews, and each oxytone
gets its contextual twin; a loanword may end in a consonant (`μπάσκετ`,
`μοτέλ`, `ἀλκοόλ`), unless it is a word of grc or the list that lost its
last vowel. Normally, a capitalized form (a name, a month) is listed where no
spelling of its monotonic form is, capitalized or in lowercase (a spelling
in `-ῃ` too: `γράψῃ` keeps out the surname `Γράψη`), a dative in `-ᾳ` or
`-ῳ` aside (grc's `Ῥωσίᾳ` and `ῥωσίᾳ` leave `Ῥωσία`), where it would not
give way as its lowercase spelling does (`Ἅγιους`, of the page `Άγιος`, to
grc's `ἁγίους`), and then only with a
mark monotonic writing drops (`Ἰούνιο`, `Ὑόρκης`, `Ῥωσία`), where
monotonic text has it often enough for `M` (`Λονδίνο`, 14,283 tokens;
`Τζένη`, 2,315), or where it has two of its declined forms, as it has a
Greek word's (`Δανία`, 676, and `Δανίας`, 214), so that a keyboard takes it
for a word rather than respell it (`Λονδίνῳ`, a dative of the attested
list) or leave it bare. A rarer name in one form (`Τζίμα`, 85), mostly a
foreign one, is right as typed, and in the list it would only draw names
typed near it to itself (`τζιμ`, typed without an accent, corrected to
`Τζίμα`). A generated name is `R`, since its lowercased count is shared
with the words of its letters. An uncounted lowercase Ancient spelling
(`fr:X`) alone does not cover a counted capital noun or name with an
independently defined Modern head and two distinct, explicitly identified
case-and-number forms. Literal capital coverage and counted lowercase
coverage (`C`, `M`, or `R`) retain their guards. An Ancient spelling a review found to have
the letters of the word Modern Greek means by them is left out of the
list, is no evidence for a generated spelling and covers none, and is
marked `mg:avoid` where grc has it (`MG_LETTERS_OF_ANOTHER_WORD`): `εὑρῶ`,
the subjunctive of `εὑρίσκω` (5 tokens in the slice), beside the currency
`εὐρώ` (4,311 monotonic tokens). It is written `form<TAB>fr:B mg:generated`, with
the bucket `B` of its monotonic form's count (above), so a reader can tell
it from the attested spellings; a Hunspell reader that does not know the
field accepts it like any entry. The `.version` counts the `attested` and
`generated` entries and their buckets, and records the sha256 of the
paradigms and the frequencies (`verb_paradigms_sha256`,
`form_frequencies_sha256`). Without those files the exporter says so and
generates nothing, and `--no-generated-verbs` leaves the generated forms
out. In the current export, 80,678 candidate analyses are covered by grc or
the attested list, 2,219 remain beside a spelling of the same letters,
and 1,234 give way to one. The 49,344 generated entries cover verbs,
nouns, adjectives, names, standalone participles, and other source cells,
including subjunctive twins, contextual twins, and plain nominal twins.
Of the generated entries, 274 are in `C` and 4,950 in `M`.

## Frequency classes

The list leaves out what grc accepts, so the commonest Modern Greek words
that keep their Ancient spelling (`θέση`, `πόδια`, `κοίτα`, `ἀγορά`) had no
Modern Greek rating: a keyboard ranking the readings of Modern Greek text
by the list's buckets saw only grc's, and preferred `θέσῃ`, rated from the
monotonic `θέσει`. The list now rates them with lines of their own, `θέση
fr:C mg:shared` (``shared_lines``): a grc spelling the slice attests takes
the bucket an attested entry would, and a generated spelling grc has as
written the bucket of its monotonic form's count, unless the slice writes
that form otherwise (`ἐμᾶς`, not the generated `ἐμάς`). The current list
has 6,651 shared rows (1,422 `C`, 5,105 `M`, 124 `R`). Low-frequency alternatives
remain at `R`. These rows rate spellings already in the Ancient dictionary;
they are no sign that a text is Modern Greek and are not additional
entries: a reader that merges the two lists keeps grc's entry and reads
the field to use the rating.

A spelling in `-ῃ` or `-ῃς` (`θέσῃ`, `ἔρθῃς`), a traditional subjunctive,
is rated by its own use (``subscript_count``). Monotonic text writes it
`-ει` or `-εις`, as it writes the indicative, the perfect and the dative of
those letters (`θα θέσει`, `έχει θέσει`, `εν θέσει`), so the spelling takes
only the part of that form's count which the slice writes with the
subscript, `(n + 1) / (n + m + 2)` of it by the rule of succession, where n
counts the slice's tokens of the spelling and m those of its `-ει`
spellings: `θέσῃ`, 19 tokens beside 19 of `θέσει`, takes half of `θέσει`'s
2,110 and is `M`, below `θέση`'s `C`; `σταματήσῃ`, 21 beside 8, takes 73% of
25,119 and is `C`; a generated twin the slice has in neither spelling takes
half. The whole `-ει` count rated `θέσῃ` before. The slice's count alone,
or the monotonic count of the spelling's own letters, which monotonic text
writes only by mistake, rates almost every such spelling `R`: measured as
below, that broke 12 dev words on the author folds in the Automatic
register, against 1 now, and lost the traditional subjunctive in mixed text
(`πρὶν σταματήσῃ`, `νὰ ἀποφασίσῃ` left bare).

## Rare forms and reviewed vocabulary

Modern forms normally require 50 monotonic tokens for generation. An
explicit plural verb, infinitive in `-ει`, or nominal case-and-number cell may
enter from 10 tokens when the merged Modern lexical-head evidence independently
defines its headword. Nouns also require an owned nominative singular, excluding
ambiguous `-ής` and `-ιάς` heads. Adjectives and names require their own
nominative singular and an antepenultimate accent in the admitted form.
A new low-frequency spelling also gives way to a dominant edited spelling
with the same monotonic stress. A distinct stress remains eligible unless
an established Modern reading meets the ordinary floor and has at least
twice its count; admitting rare `χάκι` must not displace the color `χακί`.
A part-of-speech label alone does not establish a cell,
and rare, obsolete, dialectal, misspelled, or unparsed source rows retain
the ordinary floor. The actual token count and generated provenance remain
unchanged; this admits forms such as `κουταλιοῦ` without promoting their
frequency. Existing orthography, homograph, and review checks still apply.
New rare nouns do not erase a frequent verb's traditional subjunctive
spelling. Nominal final vowels do not inherit a sibling dative's iota
subscript, which belongs to that ending rather than the stem.
An explicitly generated Modern reading also retains its R rating when
the Ancient entry is uncounted (X), as for `ψαρᾶς`. This supplies its
Modern count without turning a shared spelling into a register signal.
An independently defined uncontracted verb in `-νω` with separate first-
and third-person singular past cells may also supply a missing present
plural in `-νουν` when that exact plural has the common token rate.
This is an inferred inflection with generated provenance, not a claimed
source row. It fills incomplete paradigms such as `συμβαίνω` / `συμβαίνουν`.

`data/hunspell_mg_reviewed_loanwords.json` is maintained lexical input for
indeclinable loans missing from the source headword tables. Each form needs
two recorded sources, a review date, and at least 50 tokens in the existing
merged Modern frequency input. For example, the separately written
`σούπερ μάρκετ` documents `μάρκετ`. The exporter adds only the reviewed word,
never guessed components or inflections, and honors existing entries and
spelling rejections. These entries carry `mg:generated mg:reviewed`; they
are counted separately and never reported as polytonic corpus attestations.
The source file's hash is recorded in the export version.

A generated word's frequency ceiling follows its lexical ownership. When
the source paradigms positively assign two spellings to separate Modern
headwords, one word no longer inherits the other's ceiling solely because
their bare letters match (`στυλό` and `στύλο`). Forms of the same lemma and
forms without independent ownership evidence retain the existing cap.

## Generated-form evaluation

`eval/eval_mg_paradigms.py` measures reconstruction of held-out source
spellings. It holds out, in turn,
every form the slice attests at least 3 times with one spelling taking more
than half its tokens (a capitalized form's capitalized tokens inside a
sentence), generates it with that form and its subjunctive spelling left out
of its word's evidence, and compares the two, contextual grave and acute
counted as one and a capital as its small letter. The 81 forms the generator
declines (an unaccented word of several syllables, a clitic of one) are left
out:

| Verb class or part of speech | Spelling | Of them, forms with at least 50 monotonic tokens | Subjunctive spelling |
|------------|--------------------:|---------------------:|---------------------:|
| verbs in `-ω` (`γράφω`) | 99.0% of 5,027 | 99.2% of 3,416 | 91.1% of 1,058 |
| `-άω` (`μιλάω`) | 95.5% of 1,034 | 95.8% of 781 | 93.6% of 157 |
| `-ώ`, `-είς` (`μπορώ`) | 98.1% of 565 | 99.5% of 378 | 86.7% of 181 |
| `-μαι` (`έρχομαι`) | 96.4% of 716 | 97.3% of 444 | 87.3% of 118 |
| other verbs | 92.4% of 118 | 98.5% of 66 | 87.5% of 32 |
| all verbs | 98.1% of 7,460 | 98.5% of 5,085 | 90.5% of 1,546 |
| nouns | 96.5% of 11,871 | 96.7% of 7,444 |  |
| adjectives | 97.9% of 5,967 | 97.8% of 3,603 |  |
| adverbs | 93.5% of 864 | 92.9% of 396 |  |
| pronouns | 90.1% of 152 | 90.4% of 114 |  |
| names | 89.8% of 1,195 | 83.7% of 338 |  |
| standalone participles | 89.1% of 119 | 81.8% of 44 |  |
| all other parts of speech | 96.1% of 20,385 | 96.2% of 12,106 |  |

By cell, the past is right 99.9% of the time, the imperfect 99.3%, the
present 98.9%, the imperative 99.0%, participles 96.2% and dependent forms
96.7%. Examples of competing spellings include the Katharevousa
participle `γραφεὶς`
beside the dependent `γραφεῖς`, `ζητῶντας` beside `ζητώντας`, `μιλᾶς`
beside `μιλᾷς`, the nouns `γραφή`, `λάμψη` and `δύση` beside the
subjunctives `γραφῇ`, `λάμψῃ` and `δύσῃ`, the relative `ὃν` beside the
participle `ὄν`, and subjunctives that some texts write without the
subscript (`κρύβης`). Among the nouns and adjectives they are the same
letters in another cell (the genitive `μαθητῆ` beside the accusative
`μαθητή`, the adjective `καστανά` beside the genitive `καστανᾶ`), Ancient
datives the slice prefers (`ἀφοβίᾳ`, `προτεραίᾳ`), and words whose long vowels polytonic Modern Greek writes
both ways (`σαπούνι` and `σαποῦνι`, `κολώνα` and `κολῶνα`, `βοριάς` and
`βασιλιᾶς`). The names are compared with the slice's capitalized
spellings, which are often another word at the start of a line.

The forms the list generates are the ones the slice does not attest, so a
hand-checked sample of them gives a second number:
`eval/mg_generated_sample.tsv` holds 356 generated verb spellings,
stratified by verb class and cell, and 379 generated noun and adjective
spellings, stratified by part of speech, case and number and `fr:` bucket,
each checked for its breathing, accent type and subscript.
Of the historical sample, 723 spellings remain generated verbatim,
all with a prior correct judgment. One other sample, `μαρή`, is supplied by grc; the
Modern list supplies its contextual grave `μαρὴ`. The other 11 samples are
no longer generated. `eval/eval_mg_paradigms.py` reports the sample's
membership and prior judgments on the current list.

## Author and document holdouts

The language model's dev split is per sentence, so a dev sentence's document
and author also supply training sentences. `eval/eval_mg_polytonic.py`
measures how many of the 34,315 words in its clean dev sentences the
dictionary accepts, with the list built four ways, each with its generated
forms read from its own sources:

| Dictionary | Accepted, attested words only | Accepted, with generated forms | Out-of-dictionary words removed |
|------------|------------------------------:|-------------------------------:|-------------------------------:|
| grc alone | 81.6% | | |
| grc + list from the training split | 92.2% | 93.3% | 63% |
| grc + list without the documents holding a dev sentence | 88.8% | 91.3% | 53% |
| grc + list without each dev word's author (5 author folds) | 91.0% | 92.5% | 59% |
| grc + list without another author fold (control) | 91.9% | 93.0% | 62% |

The authors' own texts account for about 0.5 points of the 11.7 the list
adds. The generated forms add 1.0 to 2.5 points, the most where the list
has the fewest attested words. `--write-lists DIR` writes the author-fold
lists, each with the `mg:avoid` lines and `fr:R` caps read from its own
authors and the verb forms generated from their spellings, for a keyboard
harness. The keyboard measurements below retain the earlier list and engine
snapshots described in each comparison. One such measurement was made with a
keyboard (not part of Dilemma) that has a Modern Greek register, offers the
list's words only in Modern Greek text and reads the `mg:avoid` marks there.
The dev sentences were typed as bare letters, with each author's fold held
out from the list and its marks. Of the 2,415 sentences, 149 (2,493 words)
are monotonic in the gold text itself, so they are left out; on the other
36,129 words the keyboard made these errors:

| Register | Without `mg:avoid` | With `mg:avoid` | Fixed | Broken |
|----------|-------------------:|----------------:|------:|-------:|
| Modern | 5,248 | 4,965 | 332 | 49 |
| Automatic | 7,210 | 7,088 | 136 | 14 |

With the monotonic sentences counted as well, the 38,622 words give 6,210
and 5,943 in the Modern register, 8,250 and 8,137 in the Automatic one. Two
earlier mark rules gave the same within a word: a share of all the letters'
tokens with the lexicon's different-word check (207 lines on the same list)
4,966 and 7,089, and the same share with the respelling test (211 lines)
4,965 and 7,088. The weak forms carry
nearly all of the gain: 126 `με`, 30 `σε`, 49 `που` and 4 `πως` fixed,
against 5 `με`, 6 `σε`, 2 `που` and 4 `πως` that were correct enclitics
(`ἄφησέ με`, `οὕτω πως`), 3 `πῶς` written `πὼς`, and 13 of the older
relative `ποῦ` written `ποὺ`. On 474,220 words of held-out Ancient Greek
sentences, the keyboard's Ancient and Automatic registers gave the same
words with the list as without it.

## Keyboard replay comparisons

A later build of the same keyboard measured the generated verb forms and
the monotonic buckets against the attested words alone, as the earlier
lists had them. On the author folds above (all 38,622 words), its errors
went from 6,661 to 6,457 in the Modern register (209 words fixed, 5 broken)
and from 8,014 to 7,396 in the Automatic one (634 fixed, 16 broken); the
`C` and `M` buckets let it read more text as Modern Greek there, and write
letters only a frequent list word has as that word. The Ancient register
gave the same words. Typed as bare letters after `θα` and `να`, the active
dependent forms of all six persons and the present second and third person
singular of the 400 verbs with the most monotonic tokens (3,562 cases, each
form at least 50 tokens) came out as the intended word 53.8% of the time
before and 99.1% after, in the Modern and Automatic registers alike. Of
1,440 imperatives typed alone (360 forms of the same verbs, opening a
sentence or after `Ἄκου,`, before `!` or `.`), the Modern register wrote
31.3% as intended before and 96.0% after, the Automatic one 13.8% and
44.0%, none lost. On the held-out Ancient Greek sentences and passages the
Ancient register gave the same words, byte for byte. In the Automatic
register the held-out sentences gained 612 words and lost 14, all but 9
gained in the Modern Greek ones (8 Byzantine vernacular, 1 GLAUx), and the
passages gained 287 and lost 17, in Modern Greek, Katharevousa and
Byzantine vernacular but for 5 gained in patristic, capitalized and New
Testament ones; a scholion apparatus that writes the Modern-looking `νότο`
was read as Modern Greek once, so the question `Τίνες` after it came out
`Τινές`.

The next version of that keyboard measured the list with the generated
forms of every part of speech, against the list with generated verb forms
only and against the keyboard's current list, all with each author's fold
held out. On the author folds its errors went from 6,458 (verb forms only)
and 6,663 (current list) to 6,249 in the Modern register (245 fixed and 36
broken against the first), and from 7,406 and 8,024 to 7,240 in the
Automatic one (197 fixed, 31 broken). Most of the broken words are
traditional subjunctives after `νὰ` and `θὰ` (`νὰ προτιμήσῃ`, `νὰ
ἐξετάσῃ`), now written as the noun of the same letters (`προτίμηση`), and
Katharevousa genitives (`δυσαρεσκείας`, `αἰθούσας`). The verb forms typed
after `θα` and `να` came out as intended as often as before (99.1%); of the
1,440 imperatives typed alone, 6 were lost in the Modern register (95.5%)
and 4 in the Automatic one (43.8%), where `τηλεφώνα` gives way to the noun
`τηλέφωνα`, 24 times as common, and `κούνα` ranks no higher than the
attested `κουνᾶ`. Of 550 everyday Modern Greek sentences typed in both
registers, 127 came out better than with the verb forms only and none
worse. The 5,000 commonest monotonic words, typed alone as bare letters in
both registers, came out right 8,580 times of 10,000, against 7,625 with
the verb forms only and 5,691 with the keyboard's current release and its
list. Held-out Ancient Greek sentences and passages, and 1,280 further
passages of Ancient, Koine, patristic, Byzantine and Katharevousa text, came
out the same in the Ancient register, but for one Modern Greek sentence
among the held-out ones in which `ῥόζ`, a loanword the list
now has, made the keyboard respell the typed `ροζ` as `ῥοῦ`. In the
Automatic register the held-out sentences gained 129 words and lost 28 and
the passages gained 173 and lost 25, the losses again mostly subjunctives
after `νὰ`.

Against that list, with the same keyboard build, the list as it is now
fixed 8 words on the author folds in the Modern register and broke 1 (6,184
errors to 6,177), and fixed 3 and broke 1 in the Automatic one (7,190 to
7,188); the broken word, `γιοὺς` written `γιὸς` in the two folds whose
authors attest it, was generated before from a stray row of the pairs,
which lack the paradigm of `γιος`. The 10,000 commonest monotonic words,
typed alone in both registers, came out right 103 more times and 8 fewer:
the forms of `ρωτάω`, `φακέλους`, `συναδέλφους`, `χειρουργός` and
`κινέζικο` gained, and `ἐξάλλου`, `ὁποῖες`, `ἐνοχές` and `ἀξίες` lost,
beside which the rarer word of the same letters now goes in (`ἔξαλλου`,
`ὅποιες`, `ἔνοχες`, `ἄξιες`, rated `R` against `C` and `M`) and is the
keyboard's choice for a word alone at the start of a sentence. Of the 550
and 778 everyday sentences typed in both registers, 35 came out better and
1 worse; after `θα` and `να` 2 more verb forms came out as intended
(99.2%), none fewer; of the imperatives typed alone 6 more and 4 fewer did
in the Modern register, where `τρεχάτε` is now written as the present
`τρέχατε` beside it (157 monotonic tokens against 129). In the Automatic
register the held-out sentences gained 2 words and lost 2, the passages
gained 11 and lost 1, and the further passages gained 7 and lost 4, three
of them subjunctives that their own use now rates `R`, just below `M`
(`ἀπομακρυνθῇ`, `παρουσιάσῃ`, `ἐπικοινωνήσῃς`), and `ἐν τῇ Ῥωσίᾳ`, now
beside the generated `Ῥωσία`. The Ancient register gave the same words,
byte for byte, on every set.
