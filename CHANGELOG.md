# Changelog

All notable changes to Dilemma are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

This release fixes the readings reported on #157 and the rest of #147.
`build_lookup_db.py`, the POS-table steps and the adverb heads are rebuilt on
the same archived 2026-11 kaikki snapshot and the 1.3.29 dictionary expansion;
the Modern Greek tables are unchanged. Measurements compare this release
(rebuilt data and new code) with the published 1.3.29 data and code, with
`guess=False`, unless a line says otherwise.

### Measurements

- **Lookup.** `lookup.db` has 14,997,738 rows (1.3.29: 14,996,447) under 746,619 lemmas (746,623). 183 keys change their lemma, all reviewed readings and respellings below, and 1,291 are new: 1,261 cells of 240 `-ος` heads declined under the genitive rule below and 30 reviewed forms. No key loses its row.
- **Corpus vocabulary** (the 971,304 forms of the corpus form table read with `Dilemma(lang="grc", skip_numeral_ordinals=True).lemmatize_batch(guess=False)`, counted by corpus tokens): 387 forms (38,681 tokens) change their lemma, 33,713 of these tokens `ἐπειδὴ` and `Ἐπειδή`, 2,342 the unaccented `ἀπο` and `Ἀπο`, and 901 `πλάσσω` and its compounds; 6,445 forms (21,342 tokens) that had no answer get one, most of them oblique participles; two forms lose theirs (`περιπλαττομένοις`, `δυνεύω`; 4 tokens). The 300 largest changes and a random 200 were read; the main wrong answers found are under Known gaps, and all of them in #159.
- **Corpus annotations** (the 3,143,756 token and UPOS pairs of the corpus annotations, 81.6 million tokens, read with `Dilemma(lang="all", resolve_articles=True).lemmatize_batch_pos(guess=False)`): 1,471 pairs (53,490 tokens) change their lemma, 38,099 of these tokens `ἐπειδὴ` SCONJ, 1,870 `εἶθ’` ADV (`εἴτε`, now `εἶτα`) and 282 `οὔθ’` ADV (`οὔτι`, now `οὔτε`); 24,358 pairs (101,649 tokens) that had no answer get one: 56,482 tokens of words elided with an acute, a tonos or a prime (`δι´` ADP alone 6,135), about 7,000 through the dictionary-head rule below and the rest mostly oblique participles; ten pairs (12 tokens) lose theirs: OCR errors, the numerals `ῖη΄` and `έπ΄` NUM (which `guess=False` now leaves without an answer) and `περιπλαττομένοις`, `ὑποπλαττομένοις` and `ἀποπληττομένην`.
- **Benchmarks.** `eval_dbbe.py` (equiv): 90.0% without POS and 92.9% with gold POS (1.3.29: 90.0% and 92.8%; with gold POS `γραφεντα` is fixed; without POS `ἐπειδὴ` and `γραφεντα` are fixed and `τούνεκά` reads LSJ's `τοὔνεκα`, where the gold has `ἕνεκα`). `eval_gorman_gold.py`: 93.9% (93.9%). `bench_fast`: AG 99.7, Katharevousa 95.6, Demotic 96.0, as in 1.3.29. The benchmark gate passes.
- **Annotation evaluation** (tagger 0.7.0 with `lemmatize_batch_pos`, as the corpus annotations run it): DBBE 7,399 of 8,299 tokens (1.3.29: 7,398; `ἐπειδὴ` now `ἐπειδή`, as the DBBE gold has it). Gorman test 95.82% (1.3.29: 95.88%): the only three tokens that change are `ἐπειδὴ`, which the Gorman treebanks lemmatize as `ἐπεί`. The synthetic-OCR view moves the same way (4,675 of 5,025; 4,678).

### Fixed

- **Elision written with a spacing acute or a tonos.** Editions and their OCR print the apostrophe of an elided word as a spacing acute (U+00B4, or the Greek oxia U+1FFD), chiefly `δι´` and `Δι´` in the Patrologia Graeca, and less often as the Greek tonos (U+0384, `δι΄`, `παρ΄`) or a prime: no elided key and no elision rule met the form, and a guess read it as `δίς` (6,301 ADP tokens in the corpus annotations). A final acute, tonos or prime after an elided Greek word is now read as the elision mark in all five entry points, the tagged spellings and the `attested_only` gate: `δι´` and `δι΄` give `διά` (ADP; `Δι´` untagged gives `Ζεύς`, as `Δι’` does), `ἀλλ´` gives `ἀλλά`, `δ´` `δέ`. The POS tables key an elided form under ’ or under ᾽, and a cell under either mark now counts as the form's own (`εἶθ´` ADV gives `εἶτα` from the cell of `εἶθ᾽`). An elided word with a breathing or an accent ends in a consonant, so after a final vowel or `ς` in such a word the mark is a stray and is dropped (`ἀλλὰ´` reads `ἀλλὰ`, `ἵνα´` `ἵνα`, `οἱ´` `ὁ`). A word with neither is read as elided only when it is one of the elided words written so (`δι`, `παρ`, `μετ`, `μεθ`, `κατ`, `καθ`, `μηδ`, `ποτ`, `ποθ`, `αλλ`): other unmarked letters with the mark are numerals and keep their spelling, those in order (`ιβ´`, `ιστ΄`, `ις΄`, `κα΄`) and those the tables print in any order of their letters (`δε´`, `βφ΄`). So does a single letter (`ε´`, `Ὄ΄`), except `δ´` and `τ´` with the acute, which are `δέ` and `τε` far more often than 4 and 300. Tagged NUM, these and the numerals the OCR gave a breathing or an accent (`ἰδ´`, `ῖη΄`, and `ἐο´`, 75, in either order of its letters) stay the numeral: the token, or `None` with `guess=False`.
- **Enclitic accents.** A word with the second accent a following enclitic adds, whose keys give nothing or something that is no Ancient headword, reads its own rows without that accent when they give an Ancient headword: `ἐφαπτόμενοί` gives `ἐφάπτω` through `ἐφαπτόμενοι`, not the Modern `εφάπτομαι` of the accent-stripped key, `ῥᾷόν` `ῥᾴδιος` (1.3.29: `ῥάιος`) and `Ἡσίοδός` `Ἡσίοδος` (`ἡσίοδος`). A headword the word's keys give keeps its place (`ταῦτόν` stays `ταὐτός`, `ῥάδιόν` `ῥᾴδιον`). When the keys give nothing, a row that reads the bare spelling itself counts only for a nominative in `-ος` (`πολυστέφανός` `πολυστέφανος`): the dictionaries and word lists head some inflected forms as cross-references, so `ἐκφῦναί` keeps `ἐκφύω` and `ἄνθετό` stays without an answer.
- **Oblique active participles.** The tables give most participles in the nominative only: `συμπτύσσων` read `συμπτύσσω`, `συμπτύσσοντες` and `Συμπτύσσοντες` nothing. When nothing else resolves a word (after the normalizer and the compound and prefix fallbacks), an oblique active participle (`-οντ-`, `-ουντ-`, `-ωντ-`, `-αντ-`, `-εντ-`, `-υντ-`, `-οτ-` and the feminines) is read through its masculine nominative with each accent it can take, when that nominative is an Ancient key reading a verb's citation form a lexicon heads (`verb_morphology`, `via="active_participle"`). Two nominatives that read different verbs give no answer, and so does a form that may be a misaccent of a nominative reading another word (`παροῦντα`, beside `παρῶν` of `παρόω` and `παρών` of `πάρειμι`; `ἄλλοντα` is not `ἀναλόω` through `ἀλλῶν`). In the corpus vocabulary it answers about 5,800 forms (19,000 tokens), such as `φρονούντες` → `φρονέω` (138 tokens), `ποιῶντες` → `ποιέω`, `προβεβασιλευκότος` → `προβασιλεύω`, `εἰσενεγκόντες` → `εἰσφέρω`. Like the other deterministic fallbacks it can find a finite form of the same verb under the nominative's letters (`ἦλθασιν` → `ἔρχομαι` through `ἦλθας`), and in a few forms another verb: `ἀπήρασι` reads `ἀπεράω` through the Wiktionary row `ἀπήρας`, where the text means `ἀπαίρω`.
- **The POS rejections no longer empty an answer that a dictionary heads.** `lemmatize_pos()` and `lemmatize_batch_pos()` drop a candidate whose lemma the treebank POS audit rejects as a corpus target, a lemma no lexicon heads. When that left a word with no answer although `lemmatize()` has one, the annotations kept the form. The five words of #157 now read their LSJ heads through the reviewed readings and respellings below (`αἱμοπτυικοὺς` ADJ `αἱμοπτυϊκός`, `ὀρθοπνοϊκοῖς` ADJ `ὀρθοπνοϊκός`, `λημφθεῖσα` VERB `λαμβάνω`, `ἀναλημφθεῖσα` VERB `ἀναλαμβάνω`, `οἱουτινοσοῦν` PRON `οἱοστισοῦν`). In general, when the rejection leaves a word with nothing and the rejected untagged answer is the one headword of LSJ, the DGE, Cunliffe or the LBG written with other marks (breathing, accent, diaeresis, iota subscript), the tagged call gives that headword: `ἴσταται` VERB `ἵστημι` (untagged `ἴστημι`), `Θράκῃ` PROPN `Θρᾴκη`, `ζωδιακοῦ` ADJ `ζῳδιακός`. In the corpus annotations this answers about 7,000 tokens, such as `Ρωμαῖοι` PROPN `Ῥωμαῖος`, `κᾀγὼ` PRON `ἐγώ`, `τροχαϊκὸν` ADJ `τροχαϊκός` and `ἀρπάσας` VERB `ἁρπάζω`. A rejected answer that spells no such headword stays vetoed. So does one that spells a headword with no row of its own (`ὀδόντ` untagged `ὁδοντ`) or whose own rows read another word, a form the dictionary heads as a cross-reference (`ἱστήκει` untagged `ἵτω`, which spells `ἴτω`, of `εἶμι`; `τοιοῦ` untagged `οιὁ`, which spells `οἷο`), and one with an accent on other letters than the headword's, which is another word (`ταύτῃσιν` untagged `ταῦτος` is not `ταὐτός`, `τῳδὶ` untagged `ᾳδίς` not `ἄδις`): returning the untagged answer itself was tried and revived corpus artefacts (`Ὅρα` PROPN `Ὅρας`, 3,708 tokens; `ἔχόντων` VERB `ἔχοω`; `ἀπάσης` VERB `ἀπασάω`).
- **Action nouns under NOUN.** The dative of an action noun in `-σις`, `-ξις` or `-ψις` has the letters of its verb's future (`τῇ συμπτύξει`). Under NOUN, a form in `-σει`, `-ξει` or `-ψει` whose candidate is the verb reads the noun, when exactly one Ancient headword spells the stem with `-ις` and its stem is the verb's but for the last letter: `συμπτύξει` NOUN gives `σύμπτυξις`; untagged it stays `συμπτύσσω`.
- **Capitalized ordinals under PROPN.** A capitalized ordinal adjective tagged PROPN (`πρῶτος` to `δωδέκατος`, `ἔννατος`, `εἰκοστός`), as a heading or the first word of a sentence often is, keeps the ordinal, not the rare name of its spelling, in the scalar and the batch call: `Ἕβδομον` PROPN gives `ἕβδομος`, `Δέκατον` `δέκατος`.
- **`μειλιχίῃ` NOUN** gives the noun `μειλιχία` (`οὐ μειλιχίῃ πολέμοιο` in Homer); untagged and under ADJ it stays `μειλίχιος` (#147).
- **Reviewed readings** (`build_lookup_db.py`, checked against LSJ): `Χάους` → `χάος` (was the Chaonian `Χάων`); `ἐπειδὴ`, `Ἐπειδή`, `’πειδὴ`, `πειδὴ` and `Κἀπειδὴ` → `ἐπειδή`, LSJ's own head (they read `ἐπεί`; all 56 `ἐπειδὴ` tokens of the two Damascius works had it), and the forms of `ἐπειδήπερ` and `ἐπείπερ` that read `ἐπεί` likewise (`κἀπειδήπερ`, `’πειδήπερ`, `πειδήπερ`, `Ἐπείπερ`); `ἐκεὶ` → `ἐκεῖ` (was `ἐπεί`); `Τεύκριον` → the plant `τεύκριον` (was the adjective); `οἱουτινοσοῦν` → `οἱοστισοῦν` (was the artefact `οί̔ουτινος`); the forms of `συζάω`, live with, that a row gave to `συζέω` (`συζήσας`, `συζήσειεν`, `συζήσεται`); `δηῦτε` → `δηὖτε` (was `δήω`); the grammarian `Ἀπίων` for `Ἀπίωνος`, `Ἀπίωνα`, `Ἀπίωνι` and `Ἀπίωνες` (GLAUx folds the name to `ἀπίων`); and the own keys of `ἀποικέω`, `δινεύω` and `ἑλκέω`, which read `ἀποικίζω`, `δινέω` and `ἕλκω` while their forms read them, with the elided `ἀπῳκεῖτ᾽`, `δινεύεσκ᾽` and `ἑλκήσουσ᾽` and the optative `διείποιμ᾽`, which reads LSJ's aorist `διεῖπον` (`διέπω` has `διέποιμι`) (#147). Elided keys whose rows read another word, which the acute-shaped mark now reaches: `Ἀφ᾽` and `Ἀπ᾽` → `ἀπό` (were the pear `ἄπιος`), `Μεθ᾽` → `μετά` (was `μεθό`), `Εἶθ᾽` → `εἶτα` (was `εἶμι`), as their lowercase keys read; `οὕτ᾽` → `οὔτε` (was `ὅτε`); `ἐκ᾽`, `οὖν᾽` and the unaccented `ἀπο` → `ἐκ`, `οὖν` and `ἀπό` (were `ἐκάς` and the artefacts `οὖνέ` and `ἀπος`).
- **Reviewed forms no table had:** the feminine of `ἑνιαῖος` (`ἑνιαία`, `ἑνιαίᾳ`, `ἑνιαῖαι`, `ἑνιαίαις`; `ἑνιαία` gave the Modern `ενιαία`), the perfect middle participle of `ἀποκεράννυμι` (16 cells; `ἀποκεκραμέναις` gave `ἀποκείρω`; no lexicon heads the compound, and `build_lookup_db.REVIEWED_UNLISTED_HEADS` records it and `ἐπιλεπτολογέω` with the head each compounds), `ἐπιλεπτολογέω` and `ἐπιλεπτολογοῦσι(ν)` (the variant Damascius' editor prints and indexes, which LSJ cites under `ἐπιλεπτουργέω`), `ἐπέτραπται` → `ἐπιτρέπω`, and the feminine cells of `τρεπτός` that its table lacks (`τρεπτῆς`, `τρεπτήν`; with `θρεπτός` now declined, `τρεπτῆς` would have read `θρεπτός` through the Byzantine normalizer).
- **Reviewed respellings** (`dilemma/grc_reviewed_respellings.py`), applied to every reading of the combined and the Ancient lookup table and to every treebank POS cell, so the lookup and the POS lookup agree: `λήμπω` and `ἀναλήμπω`, built from the Koine `λημφθεῖσα` and `ἀναλημφθεῖσα`, are `λαμβάνω` and `ἀναλαμβάνω`; `ὀρθοπνοικός`, `αἱμοπτυικός`, `ἐπιλημψία` and `ἐναπόλημψις` are `ὀρθοπνοϊκός`, `αἱμοπτυϊκός`, `ἐπιληψία` and `ἐναπόληψις`. Readings of `πλαττ-` as `πλήσσω` or one of its compounds are `πλάσσω` and its compound (`πλάττω` is the Attic of `πλάσσω`, and no form of `πλήσσω` has the letters `πλαττ`): `πλαττόμεναι`, `πλάττεται`, and 101 keys such as `καταπλάττειν` (`καταπλήσσω`, now `καταπλάσσω`) and `διαπλάττει`; compounds with no key of their own follow (`ὑποπλάττεται`, `μεταπλάττεται`), and readings of `διηρ-` as `διερέω` are `διαιρέω` (`διηρημένον` VERB, `διηρῆσθαι`). `ἠθμοῦ` keeps `ἡθμός`, the strainer as LSJ heads it ("in codd. usu. (but perh. wrongly) ἠθμός").
- **POS cells** (`data/pos_cell_corrections.json`): `Χάους` NOUN → `χάος` (was `Χάων`) and `λίνῳ` NOUN → `λίνον` (was `λίνος`; untagged it gave `λίνον`). Once the reviewed readings above changed the untagged answers, the rebuilt treebank table kept the GLAUx cells that had agreed with the old ones; 17 rows correct them: `ἐκεὶ` ADV → `ἐκεῖ` (the rebuilt cell read `ἐπεί`; 1.3.29 gave `ἐκεῖ`), `δηῦτε` VERB → `δηὖτε` (was `δήω`), `συζήσας`, `συζήσεται` and `συζήσειεν` VERB → `συζάω` (was `συζέω`), `ἐπειδήπερ`, `κἀπειδήπερ` and `’πειδήπερ` CCONJ → `ἐπειδήπερ`, `ἐπείπερ` CCONJ → `ἐπείπερ` and `’πειδὴ` CCONJ → `ἐπειδή` (was `ἐπεί`), `λίνω` NOUN → `λίνον` (the dative without its subscript), and `ἀπο` ADP → `ἀπό` (was `ἀπος`), with the folded keys of these cells (`δηύτε`, `καπειδήπερ`, `’πειδή`, `επείπερ`, `απο`). Four rows give elided and crasis cells the reading of the untagged lookup: `οὔθ’` ADV → `οὔτε` (was `οὔτι`; 282 tokens of `οὔθ’`, `οὔθ᾽` and `οὔθ´` ADV in the corpus annotations, nearly all of them the correlative `οὔτε`), `κᾆθ’` ADV → `εἶτα` (κᾆτα, which took the cell `καθ᾽` ADV `καθά` once a cell under the other mark counted; 30 tokens), and `δηὖτε` ADV → `δηὖτε` with its folded key (was `αὖτε`, as GLAUx folds the crasis).
- **Second-declension gaps.** A listed `-ος` head was left undeclined when a table gave its genitive singular in `-ου`. A Wiktionary page restored beside a head can give the singular and nothing else, so the plural had no source: `ἀγχίαλοι` read `ἀγχίαλλος` and `ἀξύλοις` nothing (#147). A table now counts only when it gives the nominative plural in `-οι` as well. This declines 240 more heads (1,261 new keys): `ἀγχίαλοι`, `ἀγχιάλοις`, `ἀξύλοις`, `ποικίλοι`, `θρεπτοῦ`, `θρεπτῆς`.
- **Held corpus readings.** The words the corpus held for a lemma in Damascius (`ἑστηκυιῶν`, `ταυτούμενος`, `προϋπάρξει`, `προϋπονοηθήσεται`, `ἅπτουσα`, `σύμπτυξις`, `προπίπτον`, `ἐπέτραπται`, `ἀποπεμπτέον`) and the 35 Dioscorides readings with `δυσεντερικά` (2.102.2) read their LSJ heads untagged, as the corpus lemma step reads them, and with their UPOS but for `Ἰστρικὸς` ADJ, below (`ὀρθοπνοικοὺς` → `ὀρθοπνοϊκός`, `ἐπιλημψίας` → `ἐπιληψία`, `ἐναπολήμψεις` → `ἐναπόληψις`, `προϋπάρξει` NOUN → `προΰπαρξις`).

- Rebuilt and re-pinned in `data/hf_manifest.json`: `lookup.db`, `spell_index.db`, `treebank_pos_lookup.json`, `treebank_pos_lookup.audit.json`, `grc_adverb_heads.json` and `pos_cell_corrections.json`.

### Known gaps

- From #147: `διεῖπον` keeps its reading `διέπω`, which is right: it is the imperfect of `διέπω` as well as the aorist LSJ heads on its own.
- Ionic `-δεχθ-` forms of `δείκνυμι` compounds (`παραδεχθείς` `παραδείκνυμι`, also an aorist of `παραδέχομαι`) are as in 1.3.29, and the participle step follows them (`παραδεχθέντα`).
- The main wrong answers left by the rules of this release; #159 lists all that were found.
  - Enclitic-accent step: it changes 189 forms of the corpus vocabulary (864 tokens), nearly all rightly, but about 15 forms that 1.3.29 read rightly now read another word, each a few tokens: `προεπίστασθαί` `προεφίστημι` (1.3.29: `προεπίσταμαι`), `Συνήσθημέν` `συνέζομαι` (`συνήδομαι`), `ὅτιπέρ` `ὅ` (`ὅτι`), `πλειότεροί` `πλέως` (`πολύς`), `Ἔνετοί` `ἔμετος`, `ἀναπλασθῆναί` `ἀναπολάζω`, `ἀνακεκρᾶσθαί` `ἀνακράζω`, `Συνῄρητό` `συλλέγω`. It accepts the heads of the word lists, which include some inflected forms (`δάκτυλά` gives `δάκτυλα`, `ὑπόποδός` `ὑπόποδος`). `lemmatize_verbose()` has no enclitic step, so `ἐφαπτόμενοί` reads the Modern `εφάπτομαι` there.
  - Elision: a stray mark after a whole word ending in a consonant other than `ς` is read as an elision (`ἂν´` `ἄνα`, `γὰρ´` `γάρον`), and a stray dropped after letters that make no word lets them read as a word, under `lang="all"` sometimes a Modern word or acronym (`ὅι´` PRON `ἕ`, 17 tokens; `εἐδ´` `ΕΕΔΑ`); about 95 tokens in all. `ἐπειδ´` CCONJ reads `ἐφοράω` (13 tokens). A cell under the other elision mark also counts for a form's monotonic and accent-stripped keys, which brings about 80 tokens their right lemma (`ἡδ᾽` CCONJ `ἠδέ`, `τῆδ᾽` ADV `τῇδε`, `οἴσθ᾽` VERB `οἶδα`) and about 35 a wrong one (`μᾶλ’` ADJ `μάλη`, 8 tokens; `ὑφ’` NOUN `ὑφή`; `κᾷθ’` ADV `καθά`); tagged NOUN, `οὐδ’` reads `οὐδός`, as `οὐδ᾽` NOUN did (30 tokens).
  - The dictionary-head fallback maps some artefacts to another headword with the same letters: `παρῳχηκότα`, `-ότων` and `-ότι` VERB `παροχέω` (66 tokens; untagged the artefact `παροͅχέω`; the verb is `παροίχομαι`), `Ἤιδει` and `Ἤιδειν` `αἰδέω` (46), `προὔφαινον` and its forms `προϋφαίνω` (28), `οἷνος` NOUN `ὄνος` (27 tokens with PRON), `Ἁγίαν` PROPN `Ἀγίας` (27), `ὀίσσατο` `οἴζω` (21), `ἡγμένων` VERB `ἄγνυμι` (7).
  - The participle resolver does not check the accent of the stem or the tag (`Μάνεντος` and `Μάνεντα` PROPN `μαίνομαι`, 45 tokens), reads some aorist infinitives in `-σαι` as feminine participles (`ὀργάσαι` `ὀργάω`, 13; `συγχῶσαι` `συγχέω`), and reaches another verb through some keys (`απήρασιν` `ἀπεράω`, 29; `γέλωντος` and `γέλωντα` NOUN `γελάω`, 47, for the noun `γέλως`). `_is_verb_citation_spelling` rejects any iota subscript, so participles of `σῴζω`, `ᾄδω` and `θνῄσκω` are not resolved.
  - The action-noun rule accepts the heads of the word lists, so some futures tagged NOUN read nouns LSJ does not head (`θερίσει` `θέρισις`; `παραλλάσσει` NOUN `παράλλασσις`).
  - `πλάσσω`: 24 `πλασσ-` keys still read compounds of `πλήσσω` (`ἔπλασσε` VERB `πλήσσω`), and the respelling gives four treebank cells `ἀποπλάσσω`, which LSJ and the DGE head as `ἀποπλάσσομαι`. `περιπλαττομένοις` loses its untagged answer (2 tokens), `ὑποπλαττομένοις` and `ἀποπληττομένην` their tagged ones (one token each).
- A tagged word whose untagged answer is a rejected corpus target that no dictionary heads in any spelling still gets no answer: names the dictionaries lack (`Ἰεροβοάμ`, `Σαδώκ`), forms the treebanks lemmatize as themselves (`Ἰστρικός`) and fragments (`ἄνθρω`). The annotations keep the form for these, as before.
- Pre-existing, as in 1.3.29: for `ἀρπάζοντες` VERB the scalar `lemmatize_pos()` gives the Modern `αρπάζω` and `lemmatize_batch_pos()` `ἁρπάζω`. A capitalized preposition tagged PROPN reads the name or the Modern capitalized headword of its spelling: `Διὰ` and `Δι’` PROPN give `Ζεύς` (17,996 tokens in the corpus annotations), and now `Δι´` PROPN as well (177); `Μετ’` PROPN gives the Modern `Μετά`, and now the elided `Μεθ’` and `Μεθ´` PROPN as well (191 tokens, which read `μεθό` or nothing in 1.3.29), and `Παρ´` PROPN `Παρά` (25). The folded key `επείπερ` still reads `ἐπεί` untagged.

## [1.3.29] - 2026-11-20

This release closes the follow-ups to 1.3.27 and 1.3.28 (#132, #144 to #149,
#152, #154 and #155). The Ancient chain is rebuilt from the dictionary
expansion on (steps from `expand_lsj.py` to the treebank POS table) on the same
archived 2026-11 kaikki snapshot; the Modern Greek tables are unchanged.
Measurements compare this release (rebuilt data and new code) with the
published 1.3.28 data and code, with `guess=False`, unless a line says
otherwise. Counts given for one change alone come from a build of that change
on the 1.3.28 inputs.

### Rebuilt lookup

- **Size.** `lookup.db` has 14,996,447 rows (1.3.28: 14,877,178) under 746,623 lemmas (746,424): 14,846,909 rows of the combined table (14,727,748), 138,932 `el`-only rows (138,829) and 10,606 `grc`-only rows (10,601).
- **Agreement with the annotated corpora** (`lang="all"`, answers that become right and wrong against the corpus lemma):

  | Corpus | Tokens | Untagged | Tagged |
  |---|---|---|---|
  | DBBE gold | 8,342 | +20 / −0 | +20 / −1 |
  | Gorman test | 5,025 | +7 / −0 | +8 / −0 |
  | Perseus pairs | 80,911 | +234 / −108 | +47 / −55 |
  | Diorisis pairs (sample) | 30,000 | +46 / −9 | +13 / −7 |

  Nearly all the Perseus losses, tagged and untagged, are AGDT's own lemmas for names, which the new headword check no longer copies into the lookup (`Κλυταιμήστρα`, `Θήβησιν`, `Κασανδδρεύς`, the Beta Code `ξψρυς` for `Κῦρόν`): the AGDT form now reads a listed head or, for a name no list has, nothing.
- **Benchmarks.** `eval_dbbe.py` (equiv): 90.0% without POS and 92.8% with gold POS (1.3.28: 89.9 and 92.7). `eval_gorman_gold.py`: 93.9% (93.8%). `bench_fast`: AG 99.7, Katharevousa 95.6, Demotic 96.0. The benchmark gate passes. In the annotation evaluation (tagger 0.7.0 unchanged), Gorman test lemma accuracy goes from 95.74% to 95.88% normalized (94.05% to 94.21% strict), and the DBBE Byzantine set from 89.06% to 89.14% (7,391 to 7,398 of 8,299; 7 fixed, none broken).

### Added

- Comparatives and superlatives of lexicon adjectives that no table declines, from a reviewed list `data/grc_degree_forms.json` (`build/build_grc_degree_forms.py`, source `grc_degree_forms`) (#152). An `-ος` adjective takes `-ότερος`/`-ότατος` or `-ώτερος`/`-ώτατος` by the length of its stem syllable, judged from LSJ's length marks and the consonants after it. An s-stem adjective takes `-έστερος`/`-έστατος`. LSJ's stated comparison is used where it gives one. A cell is kept only when the form-attestation corpora have its spelling, and it reads the positive. The build adds 4,327 keys for 1,840 adjectives (7,773 tokens in the attestation corpora): `ὁλικωτάτων` and `ὁλικωτέρα` → `ὁλικός`, `θεοπρεπεστάτων` → `θεοπρεπής`, `μυστικωτάτω` → `μυστικός`. A degree word whose form already reads as another lemma is left out with all its cells (153 words, such as `κατώτερος` beside the Byzantine `κάτος`).
- `data/grc_es_adjectives.json` (`build/find_grc_es_adjectives.py`) lists 6,741 s-stem adjectives in `-ής` and `-ης`, read from LSJ 9 (`ές`/`ες`), the Sophocles lexicon (`βληχηματώδης, ες`) and Byzantine heads without an article (#146, #144). A listed adjective gets its contracted cells without an `-έος` row. A barytone accented before `-ης` keeps its accent (`συνήθους`, `συνήθεσι`), and its genitive plural is given both ways (`συνήθων`, `συνηθῶν`). The `-ής` cells go from 1,083 keys for 186 lemmas to 35,806 keys for 5,338 lemmas (`βληχηματώδους`, `πανευσεβοῦς`, `χρυσοφεγγῆ`, `πολυσπερῶν`).
- `skip_numeral_ordinals=True` also covers the digamma (`Ϝον`) and spellings with an overline or a macron on any letter (`Η̄ον`, `Ηο̄ν`, `Β̄ον`); with such a mark the letter may be lowercase (`δ̄ο̄ν̄`). A lowercase `ϝον` without a mark is not covered, nor is the abbreviation `Ση̄` (#152).
- `skip_numeral_ordinals=True` also covers the two-letter numerals for 11-99, a capital tens letter with a capital units letter (`ΙΑον`, `ΚΒον`); a tens letter alone (`Ιον`, `Μον`) is not covered (#152).

### Fixed

- `lemmatize_pos()` and `lemmatize_batch_pos()` read the POS tables under NFC and with a lunate sigma written as `σ`/`ς`, as `lemmatize()` reads the lookup: `Δῖόϲ`/PROPN gives `Ζεύς`, not `Δῖος`. Of the 10,045 DBBE tokens, 2,013 are not in NFC and never met their POS cell before. DBBE with gold tags: 92.7% to 92.8%. (#154)
- The impersonal `δεῖ` and its forms `ἔδει`, `δέοι` and `δεῖν` are reviewed readings of `δεῖ`, untagged and under VERB or AUX (`δεῖν`/VERB gave `δέω`, untagged `ἔδει` gave `ἐσθίω`). `δέον`, `δέῃ` and `δεήσει` keep their table readings. On the Gorman test split: +4 tokens untagged, +8 tagged, none lost (#155).
- `ρροπίω`, `ρραστωνάω`, `ρρυφάνω`, `ρρωζμαίω`, `ρρωμαίζω`, `ρρωννύω`, `ρρᾳθύμνυηι` and `ρρῃίννυμι` are rejected as lemmas: a generator kept the augment's doubled rho in the stem. `ἐρρίφθαι` and `ἐῤῥίφθαι` no longer read `ρροπίω` (#132).
- The normalizer reads every respelling's own row before any respelling's monotonic or stripped key, in `lemmatize()`, `lemmatize_batch()` and `lemmatize_verbose()`, instead of taking the respellings in rule order: with `normalize=True`, `ἔπιναι` gives `πίνω` (the row of `ἔπινε`), not `ἔπειμι` (the stripped key of `ἐπεῖναι`) (#132).
- Untagged, the contractions of σε with the article give `σε` with `lang="all"` as with `lang="el"` when they carry an acute, a grave or no accent (`στον` gave `στον`, `στην` gave `ἵστημι`, `στου` gave `ἕκαστος`); with `lang="el"` the polytonic `στὴν` gave `ἵστημι`. A circumflex form keeps its row (`στῆν` gives `ἵστημι`); tagged lookups are unchanged. +8 of 400 tokens on the Demotic benchmark (#148).
- An ADJ entry stands against a row whose lemma is an indeclinable conjunction, particle, preposition or interjection in LSJ: `ἀλλή`/ADJ gives `ἄλλος`, not `ἀλλά`. A stripped-key entry still does not replace an adjective or pronoun row (`ταύτην`/ADJ stays `οὗτος`) (#148).
- The second-declension list reads the Sophocles lexicon's endings, so 1,824 more `-ος` heads have a class (1,304 of two endings, 377 with `-η`, 143 with `-α`; `θεαρχικός` → `θεαρχικοῦ`, `θεαρχικῆς`), and 194 substantive adjectives move from noun to adjective (`φιλόσοφος`, `νεκρός`) (#144, #146). An OCR `ον` followed by an article is a noun's genitive `ου` and is not read as two endings.
- A listed lowercase head whose key reads as the name of the same spelling is declined in lowercase (#146): 287 more such heads have their cells (`παναιτίου`/ADJ → `παναίτιος`). The name keeps its key and capitalized forms.
- A generated cell whose accentless key reads a verb is added when the texts attest its own lowercase spelling more often than the verb's spellings with the same letters, and do not tag it a verb (#146). This adds 288 attested cells (732 tokens): `διχρόνῳ` → `δίχρονος`, `θεοστυγῆ` → `θεοστυγής`, `παλαιολόγων` → `παλαιολόγος`, `εὐδρόμοις` → `εὔδρομος`. `παραίνων` keeps `παραινέω`.
- Sophocles verb entries that refer to another verb with "see" are expanded under their own head (#144): `ἀναστάζω`, `ὑφέλλω`, `ἐνλιμνάζω`, `ἀποσιρόω`, `μεταμφιέζω`, 626 new keys for 9 verbs. The 174 accentless keys of `ὀφφικίζω`, `ῥέσκω` and `ῥητλιάζω` now read the accented head. A reference to something that is not a verb citation form (`εἰδέτω, see οἶδα`) is not expanded.
- The micro sign (U+00B5) typed for mu is read as Greek mu in all five `lemmatize` methods: `γνωµοδότου` → `γνωμοδότης` (#144).
- `Ἑρμαίου`/NOUN reads `Ἑρμαίας`, as it does untagged, through a reviewed POS cell (#144).
- `expand_abbreviation` cuts no future from the reduplicated present of a `-μι` verb (`ἀφίημι` with `-ήσω` gave `ἀφιήσω`).

### LSJ verb paradigms (#145)

- The LSJ verb pass writes out the principal parts LSJ abbreviates with a hyphen, where no full form is printed for the tense: `fut. -ήσω` under `ἐπιορκέω` gives `ἐπιορκήσω`, `fut. -στήσω` under `προΐστημι` `προστήσω`, `-έβαλον` under `ἐμβάλλω` `ἐνέβαλον`, `-απατηθήσομαι` under `ἐξαπατάω` `ἐξαπατηθήσομαι`. A fragment it cannot read is left out (`-έδραμον` under `ἀνατρέχω`, `-ψω` under `γράφω`). That makes 1,535 parts for 1,238 verbs, most of them futures. A part printed in full still wins (`σαλπίζω` keeps `σαλπίσω` beside `fut. -ιῶ`), and only the full forms decide which tenses are inferred (#145).
- Past-tense stems the verb pass could not read now give the other moods: a preverb kept whole before a vowel (`ἐπιώρκησα`: `ἐπιορκῆσαι`), `προ-` contracted with the augment (`προὔστησα`, `προὔβαλον`), and a double augment (`ἐπαρῳνήθην` of `παροινέω`, `ἐδιῄτησα` of `διαιτάω`). A double augment also gets its indicative with the inner augment alone, so `παρῳνήθη` → `παροινέω`, which had no answer. An Epic iterative that LSJ prints as an aorist (`προβάλεσκον`) builds its indicative only (#145).
- A verb cell that the published expansion reads as a noun, an adjective or a name is left to that reading unless a treebank reads it as the verb: `ἀναισίμους` → `ἀναίσιμος` (not the Ionic imperfect of `ἀναισιμόω`), `ἐπιπόθητον` → `ἐπιπόθητος` (not `ἐπιπίνω`). That covers 281 cells: 208 printed, 16 derived and 57 inferred. An inferred cell also gives way to a regular cell of a nominal head with the same accent (`ἐπικλινεῖς` of `ἐπικλινής`, not the future of `ἐπικλίνω`; `πρασιῇ` → `πρασιά`; 9 cells). It does the same where the corpora spell it only with a capital and that spelling reads a name (`αἰγιεῖς`, `Αἰγιεῖς` of `Αἰγιεύς`, is no future of `αἰγίζω`; 14 cells). The present systems are not checked, so `εὐθαρσεῖ` stays `εὐθαρσέω` (#145).
- `κατεῖπον`, `κατεῖπε`, `κατειπεῖν`, `κατείποι` and `κατειπών` → `κατεῖπον`, the LSJ head, not `κατερέω`, whose entry says it serves "as fut. of aor. κατεῖπον". This fixes 4 Perseus forms. `συνεῖπον` stays `συναγορεύω` ("aor. of συναγορεύω"), and `ὑπειδόμην` stays with `ὑφοράω`, which prints it (#145).
- Measured on a build of this change alone: 74,451 keys gained, 285 of them attested (426 tokens). Almost all come from the written-out abbreviations, which, like other printed parts, are not attestation-gated.
- 417 keys lost (10 attested, 24 tokens): the nominal cells above, which no longer read as a verb. 1,222 keys change lemma (161 attested, 828 tokens): `ἐκδέξεται` `ἐκδείκνυμι` → `ἐκδέχομαι`, `καταγνώσονται` `καταγνοέω` → `καταγιγνώσκω`, `μεταθήσεσθαι` `μεταθέω` → `μετατίθημι`, `ἀναδέξεται` `ἀναδείκνυμι` → `ἀναδέχομαι`, and `κατειπεῖν` → `κατεῖπον`.
- DBBE 89.9% without POS and 92.7% with gold POS, and Gorman 93.8%, all unchanged. On the treebank pairs, Perseus has 6 fixes against 7 breaks untagged and Diorisis (30k) 3 against 5. The breaks are lemma variants where LSJ prints the part under its own head (`προσπτύξομαι` → `προσπτύσσω`, not `ποτιπτύσσω`; `προπεσεῖν` → `προπίπτω`, not `προπίτνω`; `μετεμελήθησαν` → `μεταμέλομαι`).

### Generated cells (#146, #144)

- Measured on a build of these changes alone, against the 1.3.28 lookup, the generator changes gain 45,497 keys, change 175 (the verb keys above) and lose 53. The lost keys are cells now shared by an `-ος` adjective and a newly listed by-form (`πεντάπηχος`/`πενταπήχης`, `ἀτηρός`/`ἀτηρής`, Sophocles' `τογᾶτος` beside `τογάτος`), which go to neither word. DBBE goes from 89.9% to 90.0% without POS and from 92.7% to 92.8% with gold POS (16 fixes, no breaks). Gorman stays at 93.8%, and no Gorman, Perseus or Diorisis answer changes.

### Lookup readings (#147)

The counts below were measured with `build_lookup_db.py` and the steps after it on the 1.3.28 inputs, against the published 1.3.28 data, with `guess=False`.

- **Inflected forms follow a restored head.** The 1.3.27 rule gave a displaced head its own key back but left its forms under the lemma that displaced it (`κλαίω` came back, `κλαίειν` still read `κλείω`). A form now takes the head where the pinned Wiktionary-only base reads it as a restored head and the merged table as the lemma that head was taken from, and GLAUx reads it as the head no less often: `ἕπεται`, `ἕπεσθαι` read `ἕπομαι` (1.3.28: `ἕπω`), `κλαίειν` `κλαίω`, `ἄπειρον` `ἄπειρος` (was `ἤπειρος`), `κηροῦ` `κηρός` (was `κῆρ`), `θεᾶς` `θεά`, `ζῳδίων` `ζῴδιον`, `δάκρυα` `δάκρυον`. 7,957 keys under 260 heads, most of them forms no corpus attests (`κλαυσθήσεσθε`, `ἡγιασάμην`).
- **Lowercase forms of a word Wiktionary files under the name (#147).** A lowercase form that reads as the capitalized twin of a restored lowercase head takes the head where GLAUx has at least two tokens for it and more than for the name: `ἔρωτα`, `ἔρωτος`, `ἔρωτι` read `ἔρως` (was `Ἔρως`; GLAUx 581, 333, 208), `δικαιότερος` `δίκαιος` (was `Δίκαιος`), `μειλιχίοις` `μειλίχιος`, `καρδίαν` `καρδία`. 298 keys; with the looser keys that follow them, 476 keys under 57 heads. `Ἔρωτα` and the other capitalized forms keep the name.
- **Forms of a head that reads itself.** Where the base reads a corpus form as a listed head that reads itself, GLAUx has at least two tokens for that head and at least five times as many as for the merged reading, and the merged reading is no lemma convention, the form takes the head: `οἰκιῶν`, `οἰκίαις` read `οἰκία` (was `οἰκίον`), `πλευρᾶς`, `πλευρῶν` `πλευρά` (was `πλευρόν`), `σημαίας` `σημαία` (was `σημεῖον`), `εἴσεται` `οἶδα` (was `εἶμι`), `ὀποῦ` `ὀπός` (was `ὄψ`), `θαρρῶν` `θαρσέω` (was `θάρσος`), `μάθησιν` `μάθησις` (was `μανθάνω`), `ἔσο` `εἰμί` (was the Modern `εἶμαι`), `τύνη` `σύ` (was `ὁ`), `σφέων` `σφεῖς` (was `ἐγώ`). 138 keys, and 186 looser keys follow the forms of all three kinds.
- **Lemma conventions kept.** Of the 790 keys the 1.3.28 lookup read as another lemma than the base and GLAUx agree on (two tokens or more, fewer for the current reading), 200 remain. 150 are conventions: a form with an entry of its own (`ἡμέρας`, `δίκην`, `καλόν`, `θᾶττον`, `πλείονα` `πλείων`), two spellings of one citation form (`ῥᾳθυμίαν` `ῥᾳθυμία`, `γλωττίδος` `γλωσσίς`), the name and the word (`ἴριδος` `ἶρις`), the compounds of `εἶμι` (`εἰσιέναι` `εἴσειμι`, `ᾔει` `εἶμι`) and the middle and active citation forms (`τιτύσκετο` `τιτύσκω`). The others are reviewed readings (`χωρῇ` `χώρα`), forms both words share (`αἰνέσει` `αἴνεσις`, GLAUx 14 to 6; `ἀφελεῖ` `ἀφαιρέω`), forms GLAUx reads as a third word (`λιταῖς`, `λιτή` 32) and forms Wiktionary defines in an entry of their own (`κεχηνότα` `κεχηνώς`). No corpus-majority rule decides any of them.
- **AGDT lemmas are checked against the headword lists**, as GLAUx and Diorisis lemmas are. 194 AGDT gap-fills under a lemma no list has are dropped (`Κῦρόν` under the Beta Code `ξψρυς`, `Σκιπίωνα` under `Σκιπιων`, `πάντά` under `πα~ς`, `λἐγουσι` under `λἐγω`); a preverb lemma joined to a listed compound is kept. AGDT rows go from 5,341 to 5,172. 122 keys take another source's reading (`Σκιπίωνα` `Σκιπίων`, `Κλυταιμήστρα` `Κλυταιμνήστρα`, `ψηφισάμενοι` `ψηφίζω`, `ῥηθὲν` `λέγω`) and 49 lose their row, most of them names no list has (`Δροπικοί`, `Καιάδης`) or garbled keys (`λἐ`, `ρἰ`, `σβυς`). `Κῦρόν`, `Πλαγγόνα` and `Γλαῦκός` read the name through reviewed readings, where GLAUx's pair would fold to the lowercase `κῦρος`, `πλαγγών`, `γλαῦκος`, and unaccented `εἰναι` reads `εἰμί` (GLAUx: the artefact `ἐναι`).
- **Elided keys (#147).** 138 elided keys still read a lemma GLAUx does not name, nearly all two names of one word (`εἶπ᾽` `εἶπον`, GLAUx `λέγω`; `ἴθ᾽` `εἶμι`, GLAUx `ἔρχομαι`; `μιχθεῖσ᾽` `μίγνυμι`, GLAUx `μείγνυμι`); they keep the reading of their full forms under every `lang`. Six that read another word are corrected: `Δι᾽` reads `Ζεύς` (was the lowercase `ζεύς`), `Σάμιοι᾽` `Σάμιος` (was `σάμος`), `Κωλιάδ᾽` `Κωλιάς`, `Ἆρ᾽` `ἆρα` (was `ἄρ`), `ξυνέρξεθ᾽` `συνέργω` (was the noun `σύνεργον`) and `ἀδάματ᾽` `ἀδάματος` (was the name `Ἀδάματος`); `συνέργω` and `ἀδάματος` read themselves.
- **`Φαεινίδα`** reads `Φαεινίς` (was a Modern row reading the accusative itself). `Στέλλος` and `Συμεώνης` keep the Modern rows, which read each name as itself, and `παρτον` and `μος`, fragments no source names an Ancient lemma for, keep theirs.
- **Size and measurements.** `lookup.db` has 14,877,030 rows (1.3.28: 14,877,178): 8,644 keys change their lemma, 181 keys lose a row (126 of them `lang="el"` rows the combined row now matches) and 33 are new. On the 30,000 sampled Diorisis pairs 42 answers move to the corpus lemma and 4 away from it; on the AGDT pairs 226 and 99 (the 99 are AGDT's own lemmas the check drops, `Κλυταιμήστρα`, `Θήβησιν`). DBBE (`eval_dbbe.py`, equiv) stays at 89.9% without POS and 92.7% with gold POS (4 untagged tokens gained, none lost: `ἥπετο`, `κλαύσατε`, `ἡγιασμένος`, `ἔσο`). Gorman (`eval_gorman_gold.py`): 93.9% overall (1.3.28: 93.8%). `bench_fast`: AG 99.7, Katharevousa 95.6, Demotic 96.0; the benchmark gate passes.

### Hub manifest and hyphenated-lemma follow-ups (#149)

- **Tagger weights pinned.** `python -m dilemma download` fetched `tagger/` at the head of the Hub repository, while the tagger's own fallback download fetched `TAGGER_WEIGHTS_REV` (`876efd48`). The two differed for Ancient Greek: the head has later `grc` weights (uploaded 13 July, 27 fewer dependency labels) that no release pins. The full download now takes the pinned revision too. `data/hf_manifest.json` has a third section, `tagger`, with that revision and the sha256 of its 10 files (1,482 MB); `hf_data.py pin-tagger` writes it from the Hub, and `check` fails when the manifest and `dilemma/tagger/_revisions.py` name different revisions, when the Hub serves other bytes at the revision, or when the revision has a file `TAGGER` does not list. Weights at the head that differ from the pin get a note. `tagger/` is no longer an unpinned prefix.
- **Ancient-only model tested.** `tests/test_grc_model.py` checks that `lang="grc"` picks `model/grc-s3/` over `combined-s3` when both are present and falls back otherwise, and, with `--run-slow` and the model installed, that the copy is the pinned one and that `lang="grc"` reads unseen forms with it through both backends (`λελαμπρυσμένος` `λαμπρύνω`, `μεταπεπεισμένων` `μεταπείθω`). Measured with the Hub copy: ONNX and PyTorch agree.
- **`ag_verb_paradigms.json` rebuilt.** The served file was the 1.3.7 build. `build/build_grc_verb_paradigms.py` now completes LSJ principal parts from the earlier glosses export pinned in `data/lsj9_glosses_pin.json`, the way `build/expand_lsj.py` does: the current export stops many entries at the first citation, so a plain rebuild lost the perfect participles of `γράφω` and the aorist passive `ἐτέθην` of `τίθημι` (38,221 cells). It takes the LSJ9 directory from `--lsj9-dir` or `$DILEMMA_LSJ9_DIR` and checks both exports against their pins. Against the 1.3.7 file: 26,715 to 26,719 verbs, 2,573,894 to 2,584,205 cells, 10,278 cells added, 1,098 removed and 1,024 changed in 172 verbs (`βλώσκω` aorist participle `βλώσξαντα` is now `μολόντα`, `καίω` `καέντες` `καυθέντες`, `κάνούσας` `κανούσας`).
- **Reviewed readings for the #137 gap.** `data/grc_hyphenated_lemma_readings.json` gives 11 of the 27 plain spellings that lost their only (hyphenated) reading a head: `συνεστρέχοντας` `συνειστρέχω`, `συνεσυκοφάντει` `συσσυκοφαντέω`, `ἰητρεύωνται` `ἰατρεύω`, `ὑπεικαθέων` `ὑπείκω`, `παρεισδυῇ` `παρεισδύνω`. The other 16 stay without an answer: the corpus lemma is garbled beyond a clear head (`ὕ-πτυχͅσκω`), the token is two words (`μετεξέτεροίτε`) or a misprint (`δελήλωνται`), or no head list has the compound (`κατενίσατο`, `ἐπισυνήρτυεν`).
- **Treebank POS table.** `build/build_treebank_pos_lookup.py` settles a hyphenated source lemma as `build_lookup_db.py` does: an AGDT preverb lemma votes for the listed compound (`σύν-εἶδον` for `συνεῖδον`, `ἐν-χειρέω` for `ἐγχειρέω`), any other hyphenated lemma casts no vote. The audit records both under `hyphenated_lemmas`. On the 1.3.28 lookup: 274 votes join and 135 are dropped; 96,500 to 96,508 forms (10 added, such as `ἀντεκαθέζοντο` VERB `ἀντικαθέζομαι` and `κατεκέντησαν` `κατακεντάω`; 2 removed; none changed).
- **Ancient headword list.** `data/ag_headwords.json` loses the 63 fragments the hyphenated-lemma rule left: a hyphen or dash at an edge (`-γον`, `ὁτ-`), beside a metrical sign (`–⏑τατος`) or a metrical sign alone (`⏑λέων`); 225,788 to 225,725 entries. `scripts/prune_ag_headwords.py` runs the step after the hyphen rule and records it in `data/changes/ag_headwords_fragments.json`, chained to the earlier record.
- **LSJ9 reference counts.** `data/lsj9_frequency.json` was a copy of the LSJ9 export, keyed by the head as printed: 33,870 keys divided at the stem (`ψῡχ-ή`) and 4,872 more with a length mark (`πᾰρά`), which no lemma matches. `build/build_lsj9_frequency.py` keys each count under the citation spelling and adds the 29 pairs of entries that join (`τροχός` 25 + `τροχ-ός` 1); 55,495 to 55,466 keys, recorded in `data/changes/lsj9_frequency_keys.json`. Only `suggest_spelling` reads the table; no lemma answer changes.

- **LSJ9 POS table.** `data/lsj9_pos_lookup.json` keyed 463,163 forms with a hyphen and pointed 463,031 cells at a divided head such as `ἀασι-φρονία`. `build/build_lsj9_pos_lookup.py` now joins the divided heads of the LSJ9 POS list before matching, and skips any form or lemma with a hyphen. Rebuilt on the 1.3.28 `ag_lookup.json`, it goes from 1,968,772 to 1,501,405 forms, with no hyphen in any key or cell. Four cells were printed endings filed under a divided head (`λιβε`, `λιβες`, `λιβές`, `λιβέ` NOUN `βορρό-λιψ`). None remain: `λιβε` and `λιβες` now give `λίψ`. The cell `γος` NOUN `σμῶδιξ` is gone as well, but the hyphen rule did not remove it: the current `ag_lookup.json` has no `γος` row, so it came from an older lookup. The reviewed drop of that cell in `data/pos_cell_corrections.json` becomes a no-op after the rebuild. It stays, in case the row comes back. Because the table is rebuilt from the current lookup, it also differs where the lookup changed since the old build: 9,408 forms removed with no hyphen involved, 5,206 added, and 1,132 cells with another lemma (`ανιαις` `ἡνία` to `ἀνία`, `περιστερεωνας` `περιστέριον` to `περιστερεών`). LSJ9 tags `ὁ` as a noun, so the article forms now get a NOUN cell for `ὁ`. The lemma is right; only a NOUN-tagged article would reach these cells, since this table only fills tags the earlier tables leave open.
- `build_lookup_db.py` reads `data/grc_hyphenated_lemma_readings.json` and adds a reading only where the form has no row of its own (11 keys).

### Known gaps

- With `lang="grc"` a polytonic Modern Greek word can still read an Ancient lemma through its loose keys or the normalizer (`ἄνοιξη` gives `ἀνοίγω`, `ἀλλιῶς` `ἁλιεύς`). A gate on words whose monotonic spelling is a Modern citation form was measured again: it took the right lemma of 877 attested forms (2,344 tokens) such as `ἐνότητα` (`ἑνότης`) and `ὅποτε` (`ὁπότε`), and is not adopted (#132).
- A PROPN entry still yields to the row of a common noun or adjective (`ἵππων`/PROPN gives `Ἵππος`), and a capitalized NOUN keeps its name cell (`Πόντον`/NOUN gives `Πόντος`): both were measured again and lose more than they fix on the Diorisis pairs. Under SCONJ a form with no SCONJ cell still reads its lookup row (`ἐπειδὴ`/SCONJ gives `ἐπεί`): reading its CCONJ cell changed 168 corpus forms, many to errors in those cells, for no net gain on Gorman (#148).
- `Ση̄` and other overlined abbreviations still reach a lemma (`Ση̄` gives `σός`) (#152).
- Four cells of `ἀγχίαλος` and `ἄξυλος` (`ἀγχίαλα`, `ἀγχίαλοι`, `ἀγχιάλοις`, `ἀξύλοις`) lose their row: the restored Wiktionary genitive now declines the word, so the generated -ος cells are no longer written, and Wiktionary files those cells under the names.
- `μειλιχίῃ` with NOUN gives `μειλίχιος` (1.3.28: `μειλιχία`): the POS entry won against the name reading, not against the adjective.
- `διεῖπον`, `δινεύω`, `ἀποικέω` and `ἑλκέω` read another verb in the Wiktionary-only base too (`διέπω`, `δινέω`, `ἀποικίζω`, `ἕλκω`), and their elided forms follow (`διείποιμ᾽`, `δινεύεσκ᾽`, `ἀπῳκεῖτ᾽`, `ἑλκήσουσ᾽`).

## [1.3.28] - 2026-11-18

### Added

- `Dilemma(skip_numeral_ordinals=True)` returns no lemma for a capital numeral letter for 1-9, or `ϛ`, followed by `ον`, as printed: an ordinal whose superscript ending was flattened (`Ηον`, `Γον`, `Βον`). It applies in all five `lemmatize` methods and is off by default. `Τον`, `Σον` and `Ιον` spell common words and are not covered (#152).

### Fixed

- `ὅμηλυς` and `κάτηλυς` have their `-υδ-` forms (`ὁμήλυδες` → `ὅμηλυς`, `κατήλυδες` → `κάτηλυς`), and `γοσταί` has a row, through the reviewed record `data/grc_headword_corrections.json` (#150).
- `γος` is rejected as a lemma, so `γον` and `γος` no longer answer `γος`. Its rows were the ends of words divided at a line break (#152).
- More tagged readings from the #151 comparison: `ἀξίαν`/NOUN → `ἀξία`, `ἱερὰ`/ADJ → `ἱερός`, `ἱερωμέναις`/VERB → `ἱερόω`, `μοναχῶ`/ADJ → `μοναχός`, `θεοφάνην`/NOUN → `Θεοφάνης`, `ποδηγοῖς`/NOUN → `ποδηγός`, and `εὐτονήσας`/VERB → `εὐτονέω`, which had no answer. `ἀστοχήσει`/VERB gives `ἀστοχέω` with `lang="all"` too. `βάλει`/VERB gives `βάλλω` with `lang="grc"`; `lang="all"` keeps the Modern `βάζω`.
- Tagged lookups keep more of the form's own reading against a POS entry found only at a looser key (#151, after #105). An unaccented pronoun keeps a row the POS tables record under PRON for its letters (`ταυτης`/PRON → `οὗτος`, `συ`/PRON → `σύ`), while `α`/PRON stays `ὅς` and an entry with the row's letters stands (`οιαν`/PRON → `οἷος`).
- A PROPN entry with other letters no longer replaces a common word and its name twin when the capitalized form has a polytonic accent or breathing (`Ἡδὺς`/PROPN → `Ἡδύς`, not `Ἡδεῖα`). Unaccented names and spellings with only an acute keep their entries (`Διος`/PROPN → `Ζεύς`, `Βάρδα`/PROPN → `Βάρδας`). `Δὰν` and `Δάν` are the biblical `Δάν`; Doric `Δᾶν` stays `Ζεύς`.
- A bare letter tagged NOUN or ADJ answers with `lang="grc"` too (`α` gave no answer, because the POS rejection list holds it as a corpus lemma).
- `κύρου`/NOUN reads the name `Κῦρος` through a reviewed POS cell. The noun `κῦρος` is neuter with genitive `κύρους`.
- `λάτρης`/NOUN stays `λάτρης` with `lang="grc"` through a reviewed POS cell, not the `λατρεύς` of the plural `λατρῆς`.

### Changed

- The `meta` key `dilemma_version` of `lookup.db` becomes `built_from_version`. It holds the `VERSION` of the checkout that built the file, which a release bumps after the data is pinned, so 1.3.27's file said 1.3.26. No code reads the key; `data/hf_manifest.json` ties a file to a release. Published files keep the old key until the next rebuild.

## [1.3.27] - 2026-11-16

This release closes the follow-ups to 1.3.26 (#127, #129 to #132, #134 and
#136 to #141). The Ancient and Medieval chain is rebuilt on the same archived
2026-11 kaikki snapshot; the Modern Greek tables are still the 1.3.25 build.
Unless a line says otherwise, measurements compare this release (rebuilt data
and new code) with the published 1.3.26 data and code, with `guess=False`.
Counts given for one change alone are from a build of that change on the
1.3.26 inputs; the totals of the release build are under "Rebuilt lookup".

### Rebuilt lookup

- **Size.** `lookup.db` has 14,877,141 rows (1.3.26: 15,687,214) under 746,425 lemmas (777,245): 14,727,711 rows of the combined table (15,539,307), 138,829 `el`-only rows (137,259) and 10,601 `grc`-only rows (10,648).
- **Keys against 1.3.26** (combined table). 1,087,433 keys leave the table, 275,837 are new and 8,135 change their lemma. Of the keys that leave, 1,034,040 read a hyphenated lemma and 25,873 were spelled with the micro sign. Of the other 27,520, 98 occur in the form-attestation corpora (319 tokens); 27 of those still get the same lemma, and most of the rest were fragments (`μό`, `μιον`) or ill-formed cells (the augmented imperative `κατέχεσον`). The new keys are 132,209 generated adjective cells, 116,153 cells of the merged Ancient layer and 27,261 of the dictionary expansion; 22,415 of them occur in the form-attestation corpora (38,159 tokens). Of the new keys that GLAUx or Diorisis annotate, 1,288 agree with the corpus lemma and 57 do not, most of those against a corpus lemma no dictionary lists. Of the keys that change their lemma, 3,909 change only its spelling; where the corpora vote, 235 keys (24,902 tokens) move to the corpus lemma and 416 (829 tokens) away from it. 111 listed heads that read themselves now read another word, most of them principal parts that read their verb (`ὑπεριδεῖν`: `ὑπεροράω`).
- **Agreement with the annotated corpora** (`lang="all"`, answers that become right and wrong against the corpus lemma):

  | Corpus | Tokens | Untagged | Tagged |
  |---|---|---|---|
  | DBBE gold | 8,342 | +23 / −0 | +23 / −0 |
  | Gorman test | 5,025 | +2 / −1 | +2 / −0 |
  | Perseus pairs | 80,911 | +28 / −180 | +102 / −117 |
  | Diorisis pairs (sample) | 30,000 | +17 / −3 | +25 / −5 |

  161 of the 180 untagged Perseus losses and 92 of the 117 tagged ones are tokens whose AGDT lemma is itself hyphenated (`διά-βραβεύω`); the answer is now the joined headword. Most of the others answer a better lemma than the AGDT one (`μαχώμεθ᾽`: `μάχομαι`, AGDT `συμμαχέω`) or follow LSJ (`καθεῖρξαν`: `καθείργνυμι`, `ἄλαλκε`: `ἀλέξω`). With `lang="el"` DBBE gains 53 answers and loses none.
- **DBBE** (`eval_dbbe.py`, equiv): 89.9% without POS and 92.7% with gold POS (1.3.26: 89.7 and 92.5). `bench_fast`: AG 99.7, Katharevousa 95.3, Demotic 96.0, as in 1.3.26. The benchmark gate passes; two baseline tokens change, both to the gold lemma (`ὑποτυπωτικῶς`, `ἐνδυθεὶς`).

### Hyphenated lemmas (#137)

1.3.26 had 1,034,264 lookup rows whose lemma has a hyphen between two letters, and a plain spelling filed under one returned it as a citation form (`κατεκληρούχησαν` read `κατά-κληρουχέω`). The rebuilt lookup has 53: 52 Modern Greek compounds from HNC, which are spelled that way (`Ελλάδα-Βουλγαρία`), and one garbled headword of the Byzantine list (`ἐκστή-θισιζ`).

- **LSJ heads.** `build/expand_lsj.py` reads a head that `lsj9_forms.tsv` prints divided at the stem (`Βακχ-εία`, `καθ-ηδύνω`) under LSJ's joined headword. Up to 1.3.26 each of the 41,956 divided heads also got a paradigm of its own with the hyphen in every key and in the lemma (454,352 keys), which no token matches. Seven heads whose gender only the divided row gives now get their noun paradigm (`κεγχρίτης`, `ἐνναέτης`, `νεουργός`).
- **Headword-list artefacts.** 76 LSJ headwords broken by soft hyphens and 39 with a dash are no longer entries.
- **Overlay.** `overlay_lsj.py` no longer restores a historical row whose lemma has a hyphen, a soft hyphen or a dash between two letters (579,380 keys in 1.3.26, most of them under Sophocles heads the generator now reads joined or rejects). The rows are dropped, not re-keyed: 919,827 of the removed keys have their joined spelling in the table under the joined head (`Βοιωτάρχου` under `Βοιωτάρχης`).
- **Corpus pairs.** `build_lookup_db.py` no longer takes an AGDT, GLAUx, Diorisis, NT or cog pair under such a lemma as it stands (336 pairs). AGDT writes some compound verbs with the preverbs apart (`ἐπί-εἰσκομίζω`, `ἀντί,κατά-ἕζομαι`). Such a lemma is joined the way a lexicon writes the compound, with the preverb's elision, aspiration and assimilation (`dilemma/preverb_lemma.py`), and the pair is kept under the compound when `grc_lexical_heads.json` lists it: 210 pairs under 149 lemmas (`ἐπεισεκόμισαν` reads `ἐπεισκομίζω`, `ἐφημίλλητ᾽` reads `ἐφαμιλλάομαι`). The other 126 pairs are dropped; `data/citation_hygiene_rejections.tsv` records both.
- **Ancient headword list.** `data/ag_headwords.json` loses its 6,560 hyphenated strings (232,348 to 225,788 entries). The list vouched for them: corpus lemma validation passed a pair under one, and the runtime's compound step answered a hyphenated input with the hyphenated string. `scripts/prune_ag_headwords.py` makes the edit and `data/changes/ag_headwords_hyphenated.json` records every removed string.
- **Answers.** Of 208 plain spellings that returned a hyphenated lemma, 181 return a citation form (`κατεκληρούχησαν` reads `κατακληρουχέω`, `διεξαγαγόντες` `διεξάγω`) and 27 nothing. Typed with its hyphen, a removed key returns the joined head in nine cases of ten (`Βοιωτ-άρχου` reads `Βοιωτάρχης`) and never a hyphenated lemma.

### Sophocles lexicon: OCR readings and skipped entries (#139, #136)

The digitized Sophocles lexicon is OCR output. 1.3.26 generated from its heads as printed, kept the rows earlier expansions had made from misread heads, and skipped two kinds of entry it could have used.

- **Micro sign.** The OCR prints the mu of 1,047 heads as the micro sign (U+00B5), which is no Greek letter and which NFC leaves alone, so 42,710 keys of the Ancient table matched no Greek text. The chunks and the earlier expansion are now read with the Greek letter: no Ancient row is spelled with the sign, and the heads get their forms (`γνωμοδότου` reads `γνωμοδότης`, `μαντίου` reads `μαντίον`). The earlier expansion had declined part of these heads in the wrong class (an adjective in `-ής, -ές` as a noun of the first declension, one in `-ικός` as a consonant stem), so a paradigm it spelled with the sign is read only when its rows show the ending the entry implies: 654 of 753 lemmas are read, and the 1,252 rows of the other 99 are dropped (`μελισταγοῦ`, `μυριστικκά`).
- **Reviewed head corrections.** `data/sophocles_head_corrections.json` records 164 heads no rule can settle, each with its chunk and a reason: a second accent (`εἶλεώδης` is `εἰλεώδης`), an accent for a breathing, a misread diaeresis (`ϊεραρχέω` is `ἱεραρχέω`), Latin look-alike letters, the phi symbol U+03D5, a mu the OCR gave twice (`κληρονομμέω`), and letters misread in heads the micro sign had hidden (`μεταναστείω` is `μεταναστεύω`). A record whose head is not one accented Greek word, or that gives no reason, stops the build.
- **Respelled heads.** Where the head rules or a correction change only the accents, the breathing or the capital of a printed head, the rows an earlier expansion made under the printed spelling are read under the present head when those rows confirm it: one of them is the present head itself, and another has the ending the entry prints (83 heads, 1,643 marked keys; `Θεοτηρήτων` reads `θεοτήρητος`, `ἀμαντεύτου` `ἀμάντευτος`). An old paradigm built on the misread accent, breathing or diaeresis, or declined in the wrong class, is dropped with the misread lemma (`ἐκκηρυκτοῦ`, `Θεαρχιξί`, `Βληχηματώδου` have no rows), and so is one whose head changed a letter.
- **Proper names.** A capitalized head is read as a common word when the other dictionaries list only its lowercase spelling. 36 such heads are names of a person, a god or a place (`Νοητός`, `Σκόπελος`, `Κοπρώνυμος`); they are listed in the `names` section of `data/sophocles_head_corrections.json`, and the rows under the name stay the name's.
- **Phrases.** The rule that skips a head followed by a second Greek word skipped 102 entries. 21 are one word with a run-on ending, a qualifier or an adverb idiom and are generated again (`κομιτίου` reads `κομίτιον`), 25 are one word the OCR split and are joined when another dictionary lists the joined word (`κηνσο φύλαξ` is `κηνσοφύλαξ`), and 56 stay phrases (`Κοίλη Συρία`, `ἕως οὗ`).
- **Verbs glossed by an equals sign (#136).** The verb step skipped every entry whose gloss begins with `=`, and 1.3.26 dropped the older forms of those verbs with nothing to replace them. Such an entry is now expanded when its head is a verb. Of the 14,056 keys of 183 lemmas the issue describes, 13,998 are back (`ἀνακοσμοποιεῖ` reads `ἀνακοσμοποιέω`); the other 58 belong to `ἐπισηκρήτῳ`, a dative. Nine more of the dropped lemmas were misread heads, and their forms are read under the corrected head (`ἱεραρχεῖ` reads `ἱεραρχέω`).

### LSJ verb paradigms (#136, #134)

- **Past tenses stand on two stems.** `Module:grc-conj` wants the augmented stem for the indicative and the unaugmented one for every other mood. LSJ prints the augmented indicative, and the generator used that stem throughout, so the other moods were ill-formed (`εἰσηνέγκωμεν`) or fell to another verb. The unaugmented stem is now derived from the printed form and the headword: `ἀντετάφην` gives `ἀντιταφείς`, `ἀνῴδησα` `ἀνοιδήσας`, `συνεῖδον` `συνιδεῖν`. Where the augment cannot be read, only the indicative is built. An `ε` that belongs to the stem stays (`ἀπένεικα`, `ἕσσα`), and the augmented pluperfect of a compound in `εἰσ-` keeps its breathing.
- **Parser.** Roman and Arabic aorist numbers, a text that opens inside a section (`Pass., aor. ἀντετάφην`), two passive aorists of one verb, a contracted future (`βαλῶ`), a contracted imperfect (`ἠγνόουν`), middle parts as middle only, and a second aorist taken from the earlier glosses export when the current text stops after the label.
- **The 5,678 keys of 83 LSJ verbs that 1.3.26 lost (#136).** 651 are back under the same lemma (`ἀνῴδησα`, `ἀντιταφείς`, `ἐνθορών`), 49 read another, well-formed lemma, and 4,978 stay out: of their 2,755 accented keys, 48 spellings occur in any corpus (227 tokens), and those were wrong readings or readings under a variant headword.
- **Principal parts that no entry prints (#134).** Most LSJ verbs print none, so they had a present system only and `εἰκόνισεν` had no Ancient row. A compound now takes the parts its simplex prints (`συγκατατρίβω`: `συγκατέτριψα`) and a verb in `-ίζω`, `-άζω`, `-εύω`, `-έω`, `-όω` or `-άω` the regular ones of its class. These are hypotheses: only the cells whose exact spelling `form_profile.db` attests are kept. 8,206 verbs take their parts from a simplex and 12,439 from a regular class; of 1,839,607 generated cells 50,329 are attested and kept (99,405 keys with their looser spellings), for 8,953 of those verbs. A cell two verbs infer goes to neither, and a cell a noun also spells (`ἐπισκιάσει`, `κοσμιῶν`) is kept only where a treebank reads it as the verb.
- **Treebank readings decide ties, and only ties.** A cell on a stem no entry prints is left out when the treebanks read the form as another word (`ἠβουλήθη` stays with `βούλομαι`, not `ἀβουλέω`). A cell on a stem the entry prints stands, except where the verb the treebanks read has the cell among its own parts (`κατεδέξατο`: `καταδέχομαι`), reads it as no verb or is the same verb in its other voice. A headword that reads itself keeps its reading (`μεθήκω`). No form or lemma enters the table from the corpus tables; `data/ag_verb_generation_audit.json` records every decision.
- **Declined participles.** An inferred participle in `-μενος` is kept when a corpus has another case of it, and the stem test no longer skips stems with an iota subscript (`συνῄρημαι`).
- `build/expand_lsj.py --expand-verbs` now needs `data/form_profile.db`, `data/grc_lexical_heads.json` and the seven treebank pair files, and stops when one is missing. `--no-corpus-checks` runs without them and is not the published build.

### Generated cells for undeclined `-ος` words and `-ής` adjectives (#131)

- **Second-declension `-ος` words that no table declines get their regular cells.** Most are Byzantine adjectives, which the Byzantine list gives without an article, and LSJ adjectives with no dictionary table: `ἐμφλόγου`, `κολοβορίνων` and `αὐτοσυστάτου` read `ἔμφλογος`, `κολοβόρινος` and `αὐτοσύστατος`. 16,674 lemmas gain 131,305 keys: 178 for 32 nouns, and for adjectives 93,873 for 13,529 of two endings or with an unsettled feminine, 23,301 for 1,802 with a feminine in `-η` and 13,953 for 1,311 with one in `-α` (sources `grc_adjective_gaps` and `grc_noun_gaps`).
- **The class comes from a lexicon, not from the spelling.** `build/find_grc_second_declension.py` writes `data/grc_second_declension.json`: 48,650 plain `-ος` headwords of LSJ 9 and the Byzantine list as `noun`, `adjective`, `adjective_eta` or `adjective_alpha`, each with its evidence. A word any entry calls neuter is left out, and so are stems in `-ο`.
- **s-stem adjectives in `-ής` with Ionic or epic tables get their Attic contracted cells**: `δυσπενθοῦς`, `ἀκαμπῆ` and `εὐαγέσι` read `δυσπενθής`, `ἀκαμπής` and `εὐαγής` (1,082 keys for 185 lemmas).
- **The cells are keyed by their accented form only.** An accentless key would answer every other accentuation of a rare headword's letters, and in the corpora that is as often another word or a misprint (with such keys `οἴκτον`, for `οἶκτον`, read `οἰκτός`). A grave accent and an enclitic's second accent reach the cells (`νοερὰν`, `ἔμφλογόν`).
- **What is not generated.** The dual and the vocative singular, stems under four letters, and any cell whose accentless key already reads as another word (`δικῆς` beside `δίκης`; `θεοφόροις` stays with `θεοφορέω`). An accent doublet's shared cells go to neither word (`εὐπροσόδων` keeps `εὐπρόσοδος`). `μεγάλος`, `τοσαῦτος` and `μισοτοιοῦτος` are not declined, and five attested cells that are forms of another word are withheld.
- **Measured.** 1.6% of the cells occur in the form-attestation corpora. With `guess=False`, about two thirds of them had no answer, a quarter already resolved to the same lemma through a fallback, and 6,175 resolved to another word through a respelling fallback (`ἀθήκτους` read `ἄτηκτος`) and now read their own headword.

### Lookup readings (#127, #141)

- **Capitalized pronoun forms beside a Modern name (#127).** A capitalized key that the pronoun-twin rule or the reviewed non-lemma record emptied takes the Ancient reading of its lowercase spelling when the Modern table holds the same key: `Πάσας` and `Τίνας` read `πᾶς` and `τίς` under `lang="all"` and `lang="grc"` (1.3.26: the Modern names).
- **Dropped expansion lemmas no longer shut out GLAUx (#127).** An unaccented expansion lemma that the build drops used to hold its key until the sanitizer removed it. A GLAUx pair now takes such a key: 19 keys, Roman names and places (`Κέλσου` reads `Κέλσος`, `Μαικήνου` `Μαικήνας`). `Μοδέστου` and `Πίου` read `Μόδεστος` and `Πῖος` through reviewed readings, and `Βάβυρσά` reads `Βάβυρσα`.
- **No treebank reading answers `lang="el"` alone (#141).** The copy of an Ancient treebank pair in the Modern table is removed wherever the combined row reads the key otherwise (294 rows in 1.3.26): `Ἀβράμου` reads `Ἀβράμης` and `ᾔει` `εἶμι` under every `lang`. Where that copy was the only correct reading, the combined row is corrected instead (61 elided keys and `οἰκείοις`): `χαίρετ᾽` reads `χαίρω`, `καθέζετ᾽` `καθέζομαι`, `κείν᾽` `ἐκεῖνος`.
- **Medieval entries spelled like Ancient forms (#141).** A Medieval entry is not merged into the Modern table when GLAUx reads the spelling as an Ancient form at least twice and the Medieval lemma is no Modern word (27 keys). With `lang="el"`, `παρέξω` reads `παρέχω` again and `ἀκούεις` `ἀκούω`.
- **Lowercase words that read as a name.** A lowercase dictionary head whose only Wiktionary page is the capitalized name takes its own reading back when GLAUx reads the lowercase form as the lowercase lemma at least twice and uses that lemma at least as often overall (`εὔδηλος`, `ἰατρικός`, `εἰρηνικός`, `θύτης`; the grave spelling counts, so `θεατρικός` and `γυμνικός` come back on it). 354 heads take 840 keys back.
- **Forms of `ἐγώ` and `σύ`.** `ἐμοῦ`, `ἐμεῖο`, `ἐμέο`, `ἐμεῦ`, `ἐμέθεν`, `μευ` and their looser keys read `ἐγώ` (1.3.25 and 1.3.26: `μου`, the title of a Greek Wiktionary form-of page that carries the pronoun grid). `μου` reads `ἐγώ` under `lang="grc"` and stays `μου` under `lang="all"` and `lang="el"`. `σοι` and `σοί` read `σύ` (was `σός`).
- **`νεῶν`, `ὥς`, `δεῖνος`, `Νέαν`.** `νεῶν` reads `ναῦς` (was `νέος`; GLAUx 974), accented `ὥς` reads `ὡς` (was `ὅς`), `δεῖνος` reads `δεῖνα`, and `Νέαν`, `Νέᾳ` and `Νέης` read `νέος` (GLAUx files the `Νέα` of place names under a name lemma that the build lowercased to `νέη`).
- **Reviewed readings.** Five readings that cells of the LSJ verb pass or of a Sophocles verb took are set back: `ἀπέδει` reads `ἀποδέω` (GLAUx 11; the cell was a future of `ἀπεσθίω`), `ὑπαγορεύσει` `ὑπαγόρευσις`, `ἐνῆκα` `ἐνίημι`, `προσεικώς` `προσέοικα` and `τυμπανόεις` itself. `Ἐφέσω` and `Ἐφέσον` read `Ἔφεσος` and `Γραϊῶν` `Γραῖα`, where the capitalized form had fallen to a lowercase verb cell. `Κεῖται` reads `κεῖμαι` (1.3.26: `κέω`, from one Diorisis row).

### POS-aware lemmatization (#129, #130, #134, #138, #141)

- **Tagged answers of restored heads (#141).** The treebank table builder dropped a corpus vote that agrees with the bare lookup unless Wiktionary defines the lemma for the tag, so a head whose own lookup row 1.3.26 restored lost its cell and a lower table answered the tagged request with another lexeme. The builder now also keeps the vote of a head itself, where a lower table answers the form with another word (audit key `listed_head_votes`, 134 forms): `ταμεῖον` NOUN gives `ταμεῖον` (was `ταμιεῖον`), `ψηφίς` NOUN `ψηφίς`, `μεγάλος` ADJ `μεγάλος`, `ἰθύνω` VERB `ἰθύνω`.
- **One verb cell per verb.** The runtime resolves a participle through the verbs the POS tables name, and a compound verb whose every vote agrees with the lookup had no cell left (`προσκαθημένας` lost `προσκάθημαι`). The builder keeps one agreeing VERB or AUX vote for each verb no table names (audit key `verb_witness_cells`, 15,475 verbs).
- **Capitalized names under NOUN (#138).** The builder writes an exact NOUN cell for a name when GLAUx has at least 10 tokens of the capitalized spelling, reads at least nine tenths of them as that name, and has at least a quarter as many name tokens as tokens of the lowercase common noun or adjective (audit key `capital_name_cells`, 528 cells): `Θέρμον` NOUN gives `Θέρμος`, `Εὐριπίδου` NOUN `Εὐριπίδης`, `Τρύφων` NOUN `Τρύφων` (was `τρυφή`), `Δελφῶν` NOUN `Δελφοί`. `Τύχη`, `Νίκη` and `Ἔρις` NOUN still give the lowercase noun, and `Φίλων` NOUN `φίλος`.
- **PROPN entries against the capitalized twin of a verb row (#138).** A lowercase form tagged PROPN whose row reads a verb answered that verb with a capital (`φάων` gave `Φάω`). A PROPN entry under the form's own spelling that names a capitalized lemma now decides: `Φάων`, `Μέλητος`, `κλεῖτον` PROPN `Κλεῖτος`.
- **Accent-stripped POS entries under ADJ and PRON (#130).** An entry found only under the form's accent-stripped key no longer replaces an exact lookup reading that an Ancient dictionary lists: `ταύτην` ADJ gives `οὗτος` (was the `ταὐτός` of `ταὐτήν`), `ἀθληταί` ADJ `ἀθλητής`, `οὕ` PRON `ἕ`, `αὕται` PRON `οὗτος`. A pronoun with a wrong accent or breathing keeps its lemma (`ὁδὲ` PRON stays `ὅδε`). Not extended to DET and NUM, and not applied with `lang="el"` or the triantafyllidis convention.
- **Closed-class entries against a reading and its capitalized twin (#129).** Under PRON, DET, ADP, CCONJ, SCONJ, PART and INTJ an entry under the form's own spelling outranks a lookup reading that has only its case alternate beside it: `αἶ` CCONJ gives `εἰ` (was `αἶα`), `ἄνα` ADP `ἀνά` (was `ἄναξ`), `ἐμέο` PRON `ἐγώ`. `αἴ` and `αἲ` SCONJ give `εἰ`, as under CCONJ, through reviewed cells. `αἶ` SCONJ gives `εἰ` and `αἶ` PART `αἴ` in the same way; without a tag it stays `αἶα`.
- **Contractions of `σε` with the article (#134).** With `lang="all"` or `"el"`, `στο`, `στον`, `στην`, `στη`, `στου`, `στης`, `στα`, `στους`, `στις` and `στων` give `ο` under DET and `σε` under ADP, monotonic or polytonic. Until now `στον` DET gave `στον` and `στην` DET gave `ἵστημι`.
- **Smooth breathing written for rough (#134).** A form the Ancient lookup lacks as written and holds with a rough breathing under the same reading no longer takes a Modern Greek POS entry: `ἄπαν` NOUN gives `ἅπας` (was the Modern noun `άπαν`), `ἤμισυ` NOUN `ἥμισυς`.
- **`lang="el"` and capitalized Ancient names (#138): no change, documented.** `Εὐξένου` NOUN gives `εὔξενος` with `lang="el"`; PROPN gives `Εὔξενος`. Letting the word's own row decide there capitalized the words that open a sentence on the UD Greek GUD treebank. Read Ancient text with `lang="all"` or `"grc"`.
- **Reviewed cells.** `data/pos_cell_corrections.json` goes from 169 to 157 rows: the 32 `add` rows whose cell the builder now writes itself are retired. Twenty rows are new. Five add a cell: `αἴ`, `αἲ` and `αἶ` SCONJ (`εἰ`), `αἶ` PART (`αἴ`) and `παρέξω` ADV in the Modern table. Fifteen drop one that a mistagged token or a misfiled spelling wrote and that the new rules would otherwise answer with: `κεῖται` PROPN and NOUN `Κίτης` (one GLAUx token against 1,285 of `κεῖμαι`), `καλόν` PROPN and NOUN `Κηλόν`, `ποίησιν` and `ποίησι` DET `ποῖος` (the Ionic `ποίῃσιν`, listed without its subscript), `ὑμέν` PRON `σύν`, `πᾶσι` PRON `πᾶν`, `κάλλιστον` ADP `κατά` and `Κήτεος` NOUN `Κήτευς`.

### Runtime lookup and fallbacks (#132, #134, #141)

- **A double rho printed with breathings reads as plain `ρρ` (#141, #134).** Older editions print a medial double rho as `ῤῥ`, and their OCR also gives half-marked and confused pairs. Every method now rewrites such a pair inside a word before the lookup, after the `attested_only` gate, which still matches the spelling as printed. `ἀπεῤῥάγησαν` gives `ἀπορρήγνυμι` (1.3.26: `None`) and `Βέῥῥοιαν` `Βέροια`. 5,816 attested forms (13,361 tokens) are printed this way: 1,017 that had no lemma now have one.
- **A capitalized word reads the capitalized keys of the generated paradigms (#141).** The dictionary expansion keys the unaccented spelling of a capitalized form with its capital (`Σαμαρειτις`), and the lookup cascade only tried the lowercase keys. A capitalized word that no lowercase key answers now tries both keys with the capital and takes a capitalized Ancient citation form from them: `Σαμαρείτις` gives `Σαμαρεῖτις` under every `lang` (1.3.26: `None`).
- **The Byzantine fallback prefers a respelling that is a row as written.** Respellings are tried against their own rows first, then under their monotonic keys, then under their accent-stripped keys: `περιμέναι` gives `περιμένω`, not `περάω`. A breathing slip is tried before any respelling of a letter (`ἅπατον` gives `ἄπατος`).
- **A lowercase word is not read as a Modern Greek name or abbreviation (#132).** A form with a breathing may not take a lemma that drops the breathing; a capitalized lemma was exempt, also for a lowercase form, whose accent-stripped key is then the row of a Modern Greek surname or abbreviation: `ἀνάβη` read `Ανάβης`, `ὅτ` `Οτ`. The exemption now needs a capitalized form. With `lang="grc"` 2,032 lowercase attested forms lose such a lemma, nearly all fragments; 454 take another reading (`ἄντι` gives `ἀντί`).
- **`χρυσόστομον` (#141)** is answered by the generated adjective cell of #131.

### Hub manifest (#140)

`python -m dilemma download` serves every file under `data/` and `model/`, and 1.3.26 pinned 63 of them. 30 more (1,555 MB) had no pin, so nothing checked that the Hub served a release's own bytes.

- **Runtime POS tables.** `ag_pos_lookup.json`, `treebank_pos_lookup.json` and `glaux_pos_lookup.json` join the list CI waits for, downloads and verifies (66 files).
- **A second pinned list.** `data/hf_manifest.json` has a `pinned_only` list (`PINNED_ONLY` in `scripts/hf_data.py`, 28 files) for served files CI does not download: the Ancient and Medieval build tables, the corpus pair tables, the per-source frequency tables, `ag_verb_generation_audit.json`, `ag_verb_paradigms.json`, `vesuvius_index.json.gz`, `med_pos_lookup.json` and the Ancient-only model `model/grc-s3/`. `push` uploads a changed file of either list and rewrites its pin, `check` and `wait` compare both lists with the Hub, and `files` and `verify` stay with the CI list.
- **Served files without a pin.** `hf_data.py check`, which the pre-push hook runs, lists what the Hub serves and fails on a file under `data/` or `model/` that is in neither list. `push --dry-run` says what would be uploaded and re-pinned and writes nothing.

### Known gaps

- Hyphenated lemmas (#137). 27 plain spellings whose only reading was a hyphenated lemma have no answer without a guess: 18 sat under a garbled corpus lemma (`ὑπεικαθέων` under `ὑπέι-κ̓αθέω`), 7 under a compound no head list has (`κατενίσατο`), and 2 came from the historical table. `treebank_pos_lookup.json` still rejects the AGDT lemmas the lookup joins, so those forms have no POS edge there. `data/ag_headwords.json` keeps 58 strings with a hyphen at an edge or beside a metrical sign, and `lsj9_pos_lookup.json` and `lsj9_frequency.json` still carry hyphenated keys copied from older tables.
- Sophocles lexicon (#139). Entries that only refer to another head with "see" are still not expanded; seven of the lemmas of #136 are of this kind (`φαγέω`, `ἀναστάζω`, `ὑφέλλω`). The noun step makes no paradigm for Sophocles adjectives, so a respelled or lowercased adjective has only the older paradigm, under the keys it was made with (`Θεοτηρήτων` is read, `θεοτηρήτων` is not), and none where that paradigm was not confirmed (`θεαρχικός`, `βληχηματώδης`, and the adjectives whose micro-sign paradigm was dropped). `Ἑρμαίου` under the tag NOUN answers `ἕρμαιον`. `μίον` and its forms no longer answer: they were the tail of the misread head `χημίον`. A word typed with the micro sign itself is not read as its Greek spelling.
- LSJ verbs (#136, #134).
  - An inferred cell is an exact row, so it answers before the runtime's looser fallbacks. For the inferred cells the 1.3.26 data answered the same verb for about 36,600, nothing for 11,900 and another lemma for 1,853. Most of those were loose answers that the row corrects (`ζυμῶσαι`: `ζυμόω`, was `ζυμώνω`). Where 1.3.26 gave a dictionary headword and no treebank reads the verb (978 cells, 1,602 tokens), some of the new answers are wrong: a contracted future beside a nominal form (`πρασιῇ`: `πρασίζω`, was `πρασιά`; `ἐπικλινεῖς`: `ἐπικλίνω`, was `ἐπικλινής`) and a lowercase cell that the corpora have only as a capitalized name (`αἰγιεῖς` beside `Αἰγιεῖς`, about 17 cells; the attestation test folds case). About 12,000 kept cells rest on one raw-text token; in a sample of 190, seven were dubious (`τερησάντων` under `τερέω`, `ἐφείσατε` under `ἐφίζω`).
  - A verb cell that a nominal form also spells takes the key when only the published expansion has the nominal reading: `ἀναισίμους` (`ἀναίσιμος`) reads `ἀναισιμόω`, `ἐπιπόθητον` `ἐπιπίνω`, `προτύποις` `προτύπτω`. About ten accented forms.
  - Lemma choices that follow LSJ against the treebanks: the cells of `καθεῖρξα` read `καθείργνυμι` (GLAUx: `κατείργω`, about 190 tokens), `κατεῖπον` and `κατεῖπε` read `κατερέω`, `συνεῖπον` and its cells `συναγορεύω`. `προμαρτυρεῖ`, `προμαρτυρεῖται` and `προμαρτυρούμενον` read `προμαρτύρομαι` (1.3.26: `προμαρτυρέω`).
  - With `lang="all"`, a monotonic form that an Ancient cell also spells gives the Ancient verb: `συμπαγείς` gives `συμπήγνυμι`, not `συμπαγής`. `lang="el"` is unchanged.
  - The runtime resolves an oblique participle through its nominative only when the verb is a verb target of the POS tables, so `ἐσκοτισμένους` still has no answer although `ἐσκοτισμένος` has a row. About 33,000 attested oblique forms in `-μεν-` stay unresolved (1.3.26: 34,628), most of them single misspellings.
  - Three kinds of derived stem are still ill-formed outside the indicative, as in 1.3.26: a preverb kept whole before a vowel (`ἐπιορκέω`: `ἐπιωρκῆσαι`), a crasis read as no augment (`προὔστησα`), and an iterative read as an aorist (`προβάλεσκε`). `παρῳνήθη` (2 tokens) has no answer: LSJ prints the double augment, which is built for the indicative only. Hyphenated abbreviations (`fut. -ήσω`) are still not expanded.
- Generated cells (#131). About 4,900 Ancient `-ος` heads known only as bare headword strings, with no class in LSJ 9 or the Byzantine list, are still headword-only, and so are 3,242 listed words whose genitive already reads as another word and about 1,150 whose headword key reads as a name of the same spelling (`ἐπάρατος` reads `Ἐπάρατος`). The cells have no accentless key. A generated cell whose letters are a verb's form keeps the verb's reading (`θεοφόροις` reads `θεοφορέω`; about 195 attested forms). Barytone `-ης` adjectives and `-ής` adjectives without an `-έος` row get no contracted cells.
- Lookup readings (#127, #141). The lookup still reads about 640 forms as another lemma than the pinned Wiktionary-only base and GLAUx agree on (`ἕπεται` reads `ἕπω`, `κλαίειν` `κλείω`); some are lemma conventions, others errors, and they need a reviewed rule, not a corpus majority. 109 elided keys answer `lang="el"` with the combined reading where the removed treebank copy named another lemma, most of them two names of one word (`εἶπ᾽` reads `εἶπον`, GLAUx `λέγω`). AGDT pairs get no headword check: 268 of the 5,533 AGDT rows name a lemma outside the headword lists. An inflected form of a restored lowercase head can still read the name where Wiktionary files it there (`ἔρωτα` reads `Ἔρως`). `Φαεινίδα`, `Στέλλος`, `Συμεώνης`, `παρτον` and `μος` still fall to a Modern row.
- POS-aware lemmatization.
  - A PROPN entry does not decide against the row of a noun or an adjective (#138): `ἵππων` PROPN gives `Ἵππος` where the tables hold `Ἵππων`. Measured and not adopted: it moved 33 answers to the corpus lemma and 86 away.
  - A name cell answers every NOUN request of its capitalized spelling with the name, also where a text capitalizes the common noun at the start of a sentence (`Πόντον` NOUN gives `Πόντος`).
  - Where the lookup reads an acute or monotonic citation spelling as another word, the tagged answer is the head and the untagged one is not (`μαθητρίς` NOUN gives `μαθητρίς`, untagged `μαθητής`): 35 lowercase spellings.
  - Under ADJ a wrong lookup row now stands against the entry of the accent-stripped key (#130): `ἀλλή` ADJ gives `ἀλλά` (was `ἄλλος`). About ten of the 141 corpus forms that change under ADJ are of this kind.
  - A form with a CCONJ cell and no SCONJ cell can answer differently under SCONJ (`ἐπειδὴ` SCONJ gives `ἐπεί`): 162 forms.
  - Untagged `στον` gives `στον` and `στην` gives `ἵστημι` with `lang="all"` (#134): lookup rows, not the POS path.
- Runtime fallbacks (#132). With `lang="grc"` a polytonic Modern Greek word that no Ancient head spells can still get an Ancient lemma: `ἄνοιξη` gives `ἀνοίγω` through its own accent-stripped key, `ἀλλιῶς` gives `ἁλιεύς` and `ἀνάβει` `ἀναβαίνω` through the Byzantine fallback. The same keys read Ancient and Byzantine forms printed with a misplaced accent or breathing (`ἕξης` gives `ἑξῆς`), and a gate that kept the fallback from re-accenting a word spelled as a Modern Greek one removed more right lemmas than wrong ones on the attested forms; it was measured and not kept. The lowercase-name rule also drops the Modern Greek name of a Byzantine form printed in lowercase (`ἀχρίδος` reads `ἀκρίς`). A marked double rho reads whatever the plain spelling reads, including rows whose lemma is wrong (`ἐῤῥίφθαι` gives `ρροπίω`, like `ἐρρίφθαι`). The optional normalizer (`normalize=True`) still takes its respellings in rule order.
- Hub manifest (#140). The tagger weights under `tagger/` (10 files, 1,479 MB) are served without a pin. `model/grc-s3/` is pinned but not tested: a full download makes `lang="grc"` load it, and CI tests `lang="grc"` with `combined-s3`. `ag_verb_paradigms.json` is the 1.3.7 build, older than the `ag_pairs.json` it is built from.

### Release notes

- Rebuilt and re-pinned in `data/hf_manifest.json`: `lookup.db`, `spell_index.db`, `ag_headwords.json`, `grc_lexical_heads.json`, `grc_contracted_verb_aliases.json`, `grc_adverb_heads.json`, `pos_cell_corrections.json`, `treebank_pos_lookup.json`, `ag_noun_generation_audit.json`, `ag_sophocles_noun_generation_audit.json`, `treebank_pos_lookup.audit.json`, and the build tables `ag_lookup.json`, `ag_source_alias_audit.json` and `ag_source_alias_manifest.json`. `ag_verb_generation_audit.json` is new on the Hub. `ag_pairs.json`, `verb_subjects.json` and `grc_reading_heads.json` were rebuilt to the same bytes.
- Not rebuilt: the Modern Greek tables, `lemma_attestation.json`, `form_profile.db`, the models, and `lsj9_pos_lookup.json`, as in 1.3.26.
- Full suite on the rebuilt data with `--run-slow`: 7,126 passed, 27 skipped, 3 xfailed, 0 failed (run in seven chunks).

## [1.3.26] - 2026-11-15

This release rebuilds the Ancient and Medieval Greek lookup on the archived
2026-11 kaikki snapshot (#125). It also fixes the merged Ancient layer (#126),
the `lang="all"` POS answers lost with the Modern fragments (#128) and the
fallback readings that skipped the reviewed non-lemma filter (#133). The
Modern Greek tables are not regenerated (`build_data.py --lang grc` and
`--lang mgr` only). Unless a line says otherwise, measurements compare this
release (rebuilt data and new code) with the published 1.3.25 data and code,
with `guess=False`.

### Rebuilt lookup (#125)

The kaikki dumps that 1.3.7 to 1.3.25 were built on no longer exist. The
Ancient and Medieval builders now read the dumps of 2026-10-30 (English
Wiktionary) and 2026-11-02 (Greek Wiktionary), archived on the Hub and pinned
in `data/kaikki_inputs.json`. The whole Ancient chain was rerun on them:
`build_data.py`, the lexical heads, the contracted aliases, the LSJ and
Sophocles expansions, the overlay, the alias manifest, `build_lookup_db.py`,
the treebank POS table and the adverb heads.

- **Size.** `lookup.db` has 15,687,214 rows (1.3.25: 15,703,890): 2,292,678 from `wiktionary_grc`, 6,014,364 from `grc_lexicon_expansion` and 2,350,226 from `grc_merged`. `ag_lookup.json` has 10,665,147 keys (10,640,468).
- **Keys against 1.3.25.** The Ancient table gains about 72,000 expansion keys and 13,000 Wiktionary keys, and about 9,200 expansion keys change their reading. It drops about 100,600 keys:
  - 77,168 are spellings of hyphenated or starred Sophocles composition entries (`*πάρ-ισος`), which no token matches.
  - 17,455 are verb forms of 237 lemmas. Part of them were wrong readings (`ἔξηλθεν` read `ἐξερύω`, `φᾶτε` read `φάω`); the rest are listed under "Known gaps".
  - About 5,200 are nominal spellings that no generator makes any more, mostly misaccented ones (`ῥήμα`, `ἁμαρτια`, `ἀδελφην`), and 714 are capitalized unaccented keys (`Ειρηνην`, `Ιερων`).
- **Agreement with the annotated corpora** (`lang="all"`, answers that become right and wrong against the corpus lemma):

  | Corpus | Tokens | Untagged | Tagged |
  |---|---|---|---|
  | DBBE gold | 8,342 | +9 / −3 | +14 / −2 |
  | Gorman test | 5,025 | +2 / −0 | +4 / −0 |
  | Perseus pairs | 80,911 | +96 / −24 | +50 / −9 |
  | Diorisis pairs (sample) | 30,000 | +23 / −5 | +17 / −4 |

  `lang="grc"` and `lang="el"` are net positive on every corpus as well. The three `bench_fast` fixtures do not change under `lang="all"`.
- **DBBE** (`eval_dbbe.py`, equiv): 89.7% without POS and 92.5% with gold POS (1.3.25: 89.6 and 92.4). `bench_fast`: AG 99.7, Katharevousa 95.3, Demotic 96.0 (unchanged). `scripts/check_benchmark_regressions.py` passes.

### Carrying the published expansion forward

The published table was built on top of earlier tables, so it holds keys that
a build from fresh dumps withholds or that no current generator makes.
`overlay_lsj.py --carried` now takes the published 1.3.25 table as a pinned
input and gives such a key back only under a stated rule. No rule adds a key
the published table lacked, and there is no general "carry every published
key" rule, because part of the published readings were wrong and the rebuild
fixed them (`ἔξηλθεν` read `ἐξερύω`). Full conditions are in
docs/lexicon-expansion.md, "Historical expansion overlay". Counts are from
the release build:

- Excluded 1.3.1 conflicts whose published reading the corpus gap-fill would otherwise replace with a recorded artefact: 7,675 keys (`ἀμέτρητον` keeps `ἀμέτρητος`).
- Nouns that get no fresh paradigm because the generator cannot use their printed genitive: 1,539 keys of 253 lemmas (`λεκτοῦ`, `κλῃσίν`, `σωματεῖα`).
- Regular masculine and neuter cells of oxytone adjectives in -ός, which an earlier Sophocles expansion made: 3,747 keys of 457 lemmas (`ἀρχοντικόν`, `θελητόν`).
- Keys of heads whose lost breathing was corrected, moved to the corrected head: 1,316 keys of 119 lemmas (`Ρωμαϊκόν` under `Ῥωμαϊκός`).
- Readings under a properly spelled head that a fresh paradigm of an accent-less LSJ9 head claimed: 57 keys (`κληζεται` reads `κλῄζω`, not `κληζω`).
- Unaccented fallback keys, only where the marked spellings of the rebuilt table that strip to the key read its lemma: 386,533 keys.

### Dictionary expansion

- **Principal parts.** The current LSJ9 glosses export ends many verb entries' head text at the first citation, before the remaining principal parts. `build/expand_lsj.py` now also reads the same export at an earlier lsj9 commit, for principal parts only: a tense the current text gives stands, a missing one is taken from the earlier text, and the three aorist slots are filled only together. 978 verb entries gain 3,018 principal parts, and 178 verbs that only LSJ lists gain 19,363 forms. Both files are pinned in `data/lsj9_glosses_pin.json`, and the verb pass stops when either is missing or differs.
- **Sophocles heads.** `build/expand_sophocles.py` reads each head through five rules (docs/lexicon-expansion.md, "Sophocles heads"), and its audits record every decision:
  - The first word of a phrase is not an entry (`Νέα Ῥώμη` gives no paradigm of `Νέα`).
  - A stated gender wins over the ending (`καισάριης, ἡ` no longer gets a masculine paradigm in -ου).
  - A head with two accents is rejected (77 heads). A head with an inner hyphen is read under the joined word when another lexical source lists it (`καθ-ηδύνω` as `καθηδύνω`: 6,159 heads) and rejected otherwise (978).
  - A capitalized common word is generated under its lowercase head (`᾿Αθεότης` as `ἀθεότης`).
  - A capitalized unaccented key is withheld when its lowercase spelling already reads another lemma (`Εστιας` for `Ἑστιάς` beside `ἑστία`): 270 keys.
- **Sophocles breathings** (#107). `build/sophocles_breathing.py` restores the initial breathing of each unbreathed ο- and ρ-chunk head from evidence (rho before a vowel, another source's head with the same letters, or the agreeing heads that share its longest initial stretch). It withholds the head when the evidence is split or missing. 1,490 heads are corrected (`ὀφφίκιον`, `ῥῆσις`, `ῥομφαία`) and 88 withheld. The decisions are recorded in `grc_lexical_heads.json` (`sophocles_breathing`).
- **Lexical heads.** Wiktionary pages whose only gloss names a nonfinite cell of another verb ("uncontracted present active infinitive of ἐπαινέω") are no longer lexical heads: `ἐπαινέειν`, `νοέειν`, `ἐπαινέεσθαι` (new in the 2026-11 dumps) and 12 such heads that 1.3.25 lists (`χέειν`, `ποιέεσθαι`, `δοκέειν`).
- **Pronoun grid.** The Greek Wiktionary's Ancient Greek page for the determiner `ὅς` carries the personal-pronoun grid in the 2026-11 dump. On a page that is not a pronoun, the grid now contributes only the headword line, so `ἡμῖν`, `ἡμῶν`, `ὑμῖν`, `ὑμᾶς` and `σύ` do not become DET forms of `ὅς` (596 pair rows).
- A stemless LSJ verb headword (the prefix `ἡμι-`) is never expanded, so bare endings such as `ῆς` and `εῖτε` are not mapped to it.
- **Generated noun cells.** The dictionary expansion now declines 60 nouns in -σις that only the generated layer declined in 1.3.25 (`σύναρσις`). Their readings are the same, and `grc_noun_gaps` has 39,397 rows (40,149). `ἀδράνεος`, an adjective in LSJ, joins the reviewed lemmas that are not s-stem neuters: the expansion layer declines it as a neuter, and its second-declension cells (`ἀδράνεον`), which had kept it out of the rule, are among the dropped keys.

### Lookup readings

- **Displaced heads** (#126). Ancient headwords that another lemma had taken over in the lookup get their own reading back. `ἄπειρος` reads `ἄπειρος` again instead of `ἤπειρος`. Other examples: `κλαίω` (was `κλείω`), `γραφή` (`γραφεύς`), `μήν` (`μείς`), `εἴσω` (`εἴδομαι`), `δῆτα` (`δή`), `θεά` (`θεός`), `δῖος` (`Δῖος`), `ἀγαπητός` (`Ἀγαπητός`), `λιτός` (`λίς`), `γονή` (`γονεύς`) and `σπαθίον` (`σπάθη`). The cause: when a word has a second Wiktionary entry, such as `κλαίω` being also a future of `κλείω`, or appears in another word's table, the merged Ancient table filed the word under that other lemma.
  - `build_lookup_db.py` applies the rule before the corpus gap-fill (`restore_displaced_lexical_heads`). A head that Wiktionary defines gets its own reading back when GLAUx reads the form as the head more often than as the other lemma, or when GLAUx has no or tied evidence and the pinned Wiktionary-only base reads it as itself. A head that only a dictionary lists needs both.
  - Where GLAUx prefers the other lemma, the reading stays (`τοῦ`, `ἴσον`, `θρόνον`). `ag_pos_lookup.json` still keeps both readings of a homograph.
  - The restored rows carry the `wiktionary_grc` source, and each is listed in `citation_hygiene_rejections.tsv` under `displaced_lexical_head_restored_to_<head>`.
  - The release build restores 581 heads and 1,084 keys.
- **Reviewed readings.** `KAIKKI_2026_11_READING_OVERRIDES` in `build_lookup_db.py` holds 274 keys whose reading the rebuild changed for the worse, each with its evidence. Unless noted there, the reading is the one 1.3.25 returned. Together with the older reviewed overrides, the build changes 404 readings and adds 6 absent keys.
  - Readings the dumps changed: `ην` and `ήν` → `εἰμί` (accented `ἤν` stays `ἐάν`), `μεγάλα` → `μέγας`, `εἴσομαι` → `οἶδα`, `τοῖ` → `σύ`, `κἆτα` → `εἶτα`, `ἐπαινεῖν` → `ἐπαινέω`.
  - Nominal cells that regenerated contract verbs took: `νοερῶν` → `νοερός` (not `νοερόω`), `γνάθων` → `γνάθος` (not the name `Γνάθων`), the paradigm of `παστάς`, the participles of `ἕψω` and `οἰκίζω`, and cells of 24 other nominals (`νεκροποιοῦ` → `νεκροποιός`, not `νεκροποιέω`).
  - Nouns in -σις where the treebanks read the noun: `ἑστιάσεις` → `ἑστίασις`, `ἐμφράξεις` → `ἔμφραξις`.
  - Keys that fell to a treebank gap-fill row with a wrong lemma: `εὑρέτιν` → `εὑρέτις`, `Δωρίδ᾽` → `Δωρίς`, `ἔσσαν` → `ἕννυμι`.
  - `νη` reads the oath particle `νή` under `lang="all"` (1.3.25: the Modern `νη`); `lang="el"` keeps its Modern row.
- **Absent keys.** A reviewed reading is written even when no source has the key for the 17 forms in `REVIEWED_ABSENT_KEY_READINGS`, whose only reading named a rejected lemma (`πεπαρῴνηκεν` → `παροινέω`, `ἠμφεσβήτουν` → `ἀμφισβητέω`).
- **Rejected lemmas.** Seven corpus lemmatizer artefacts join `dilemma/grc_rejected_lemmas.py`: `παραπαροινέω`, `ἀμφιαμφισβητέω`, `γιχρώσκω`, `ταθήχω`, `χειρίω`, `ἀρικνέομαι` and `διασμήγνυμι`.
- **Modern spellings under `lang="all"`.** Seven monotonic and unaccented keys that fresh pages gave an Ancient reading keep the Modern lemma 1.3.25 returned (`αυτοκινήτου` → `αυτοκίνητο`, `Βιέννης` → `Βιέννη`). Only the combined reading changes; `lang="grc"` keeps the Ancient one.
- **Corpus rows in the Modern table.** For a form with a reviewed reading, an el-only row that an Ancient corpus gap-fill supplied with another lemma is removed, so `lang="el"` returns the reviewed reading too (`εἴσομαι` no longer gives `εἶμι`): 20 rows.

### POS-aware lemmatization

- **`lang="all"` POS answers that only Modern Greek paradigm fragments used to reach** (#128). The Modern POS table lists only forms with several Modern readings, so dropping the fragments in 1.3.25 (#122) also dropped the cells of their real readings. `data/pos_cell_corrections.json` now also covers the Modern Wiktionary and LSJ9 tables and has an `add` action, which supplies a cell only while the table has none for that form and tag:
  - `ας` PART gives the Modern particle `ας` again (was `ὅς`); the head-line fragment cell `ας` NOUN → `μονοτονία` is dropped.
  - `ονων` NOUN gives `ὄνος` (was `ὀνή`), under `lang="grc"` too. Accented `ὀνῶν` NOUN keeps `ὀνή`.
  - `όντων` NOUN keeps `ὄντα`, the Ancient Wiktionary noun.
  - `γ` VERB gives `γ` (was `Γ`): a single candidate that differs from the untagged answer only in case no longer replaces it when its case disagrees with the form and tag.
- **Rejected lemmas in fallback readings** (#133). The reviewed non-lemma filter of #108 now also applies to the grammatical readings behind the Byzantine-normalisation lookups and the participle nominative fallback, so a stale table cell targeting `Ταύτη` or `Τοῦτος` cannot surface through them.
- **Capitalized words.** `lemmatize_pos()` and `lemmatize_batch_pos()` now give a capitalized word the same lemma under every tag (checked on 74,841 capitalized GLAUx types):
  - A POS entry that names one of the word's case twins decides: `Πῶς` ADV gives `πῶς`, `Δίκαιον` ADJ `δίκαιος`, `Λόγος` NOUN `λόγος`.
  - Where no POS entry names one, the reading of the word's own lookup row stands for Ancient text (`Εὐξένου` NOUN gives `Εὔξενος`, `Βαβυλωνίην` ADJ `Βαβυλώνιος`).
  - With `lang="el"` or `convention="triantafyllidis"`, and for a word without polytonic marks whose lowercase twin is a Modern citation form, the lowercase reading is kept (`Χέρι` NOUN gives `χέρι`, `Κάποιος` DET `κάποιος`). PROPN reads the capitalized twin under every setting.
- **Reviewed cells.** `data/pos_cell_corrections.json` grows from 96 to 169 rows. The treebank builder drops most corpus votes that agree with the bare lookup, and the restored heads (#126) make many more votes agree, so reviewed cells whose loss let a lower table answer are added back (`δήτα` ADV gives `δῆτα`, `ζώδιον` NOUN `ζῴδιον`), and new cells that displaced a 1.3.25 answer are corrected or dropped (`εισήεσαν` VERB gives `εἴσειμι`, `ήμεν` VERB `εἰμί`). An Ancient name whose common word the tables hold under the lowercase key keeps its capital under NOUN or ADJ through a reviewed cell (`Πρόξενος` NOUN gives `Πρόξενος`).

### Build

- Every builder that reads the kaikki dumps accepts both the flat layout (`scripts/archive_kaikki_inputs.py download` restores the archived snapshot into it) and the nested `en-el/`, `el/` layout. The dump names and `resolve_dump` moved from `build_data.py` to `build/kaikki_dumps.py` (`build_data` re-exports them).
- `build/expand_lsj.py` and `build/expand_sophocles.py` take `--kaikki DIR` and `--lsj9-dir DIR`; `overlay_lsj.py` takes `--lsj9-dir` and `--carried`. The dump directory comes from `$DILEMMA_KAIKKI_DIR` in every builder (`$KAIKKI_DIR` is still read when it is unset), the LSJ9 exports from `$DILEMMA_LSJ9_DIR`. Without either, `expand_lsj.py` reads `./kaikki` instead of `~/Documents/kaikki`.
- `build_data.py` output is identical under different `PYTHONHASHSEED` values (measured on the 2026-11 dumps for every language); the documented `PYTHONHASHSEED=0` is kept for the other builders.

### Known gaps

- Verb forms of 1.3.25 that are not regenerated (#136). The lookup loses 17,455 such keys of 237 lemmas, including the wrong readings named above and artefacts of the older tables (`ἑφρόντιζον`).
  - 154 lemmas (11,777 keys) get no fresh paradigm. 146 of them are Sophocles entries whose text is a cross-reference (`ἀνακοσμοποιέω, = preceding`), which the Sophocles verb pass skips.
  - 83 LSJ9 verbs (5,678 keys) are still expanded and lose cells, mostly outside the present system (`ἀνῴδησα`, `εἰσήνεγκα`).
  - With `lang="all"`, 9,438 of the forms still return their 1.3.25 lemma through fallback keys, 6,204 return `None` and 1,813 return another lemma.
- About 1.03M keys still read a hyphenated lemma (1.3.25: 1.11M; #137).
  - About 0.58M read a Sophocles composition head (1.3.25: 0.66M). The overlay names a superseded lemma as the lexicon prints it, so it returns the rows of a rejected or re-keyed head.
  - About 0.45M read a hyphenated LSJ head, as in 1.3.25: `build/expand_lsj.py` keys such a head as the lexicon prints it.
- With `lang="el"`, 147 form and tag pairs of the measured corpora that 1.3.25 read as a capitalized Ancient name now read its lowercase twin, 130 of them under a nominal tag (`Εὐξένου` NOUN gives `εὔξενος`), because Modern text capitalizes every sentence. `Τύχη` and `Νίκη` NOUN give the lowercase noun under every setting (#138).
- Small regressions against 1.3.25 with `guess=False` (#141):
  - With `lang="el"`, five forms read a new GLAUx row (`Ἀβράμου` gives `Ἀβράμος`, not `Ἀβράμης`), and untagged `παρέξω` gives the Medieval adverb instead of `παρέχω`.
  - `ἀπεῤῥάγησαν`, `χρυσόστομον` and `Σαμαρείτις` give `None`.
  - About 35 tagged cells of restored heads give another lexeme than the untagged answer (`ταμεῖον` NOUN gives `ταμιεῖον`, `σπαθίον` NOUN `σπάθη`).

### Release notes

- Rebuilt and re-pinned in `data/hf_manifest.json`: `lookup.db`, `spell_index.db`, `grc_lexical_heads.json`, `grc_reading_heads.json`, `grc_contracted_verb_aliases.json`, `verb_subjects.json`, `grc_adverb_heads.json`, `pos_cell_corrections.json`, `ag_noun_generation_audit.json`, `ag_sophocles_noun_generation_audit.json`, `treebank_pos_lookup.audit.json` and `changes/glaux_name_lemmas.json`. `lemma_attestation.json` has the same counts as before; only the hash it records for the name corrections changes.
- `python -m dilemma download` also serves files that the manifest does not pin. Their Hub copies are updated with this release: the POS tables `ag_pos_lookup.json`, `treebank_pos_lookup.json` and `glaux_pos_lookup.json` (the last had not been updated for 1.3.25), and the build tables `ag_lookup.json`, `ag_pairs.json`, `med_lookup.json`, `med_pairs.json`, `ag_source_alias_audit.json` and `ag_source_alias_manifest.json`.
- `lsj9_pos_lookup.json` is not rebuilt: a trial table on the new exports drops the key `μόνον` and adds `τό` NOUN → `ὁ`, and it was not benchmarked.
- Full suite on the rebuilt data with `--run-slow`: 6,074 passed, 26 skipped, 3 xfailed, 0 failed (run in seven chunks).

## [1.3.25] - 2026-11-14

This release rebuilds the lookup and the Modern Greek tables once for all of
its data fixes (#103, #108, #110, #114, #115, #122); the code fixes apply to
the published data as it stands. Each measurement below was taken for its own
fix against 1.3.24 with `guess=False`, unless it says otherwise.

### Lookup and table data

`λαβηι` no longer returns `λαβλυμι` (#103). The dictionary-expansion layer of
the Ancient lookup held lemmas that are source-text artefacts rather than
citation forms: Latin words read as Beta Code (`αελιυς`, `σερvιλια`), OCR
garble (`λαβλυμι`, `λαβ’μι`), hyphenated LSJ composition lemmas
(`προ-ανακηρυσσω`) and unaccented doublets of real heads (`συνοχος`).
`build_lookup_db.py` now checks every expansion reading whose lemma carries no
diacritic: a lemma that `grc_lexical_heads.json` lists (`τε`, `που`) is kept,
one with a single same-letter Ancient head that the lookup maps to itself is
moved to it (`σύνοχα` → `σύνοχος`), and the rest are dropped, including those
whose only same-letter head is a cross-reference entry of the head lists
(`κληϊδα` → `κλειδα` was moved to `κλεῖδα`, an accusative of `κλείς`) or has
no row of its own; all are listed in `data/citation_hygiene_rejections.tsv`.
GLAUx, Diorisis and other corpus pairs naming one of these judged lemmas get
the same verdict, because `ag_headwords.json` lists some of them: GLAUx
annotates `λαβῃ` with `λαβλυμι` and refilled the key the check emptied.
Other Wiktionary, corpus and merged readings are not checked this way, so real
unaccented LBG and LSJ10 headwords stay. The 1.3.25 build drops 1,032
expansion rows (138 lemmas) and 10 more whose head is no citation form, and
repairs 13 rows (9 lemmas); it rejects 269 corpus pairs (GLAUx 255, Diorisis
13, Pedalion 1) and repairs 24, so no lookup row carries a dropped lemma.
`λαβηι` and `λαβῃ` give `λαβή` with `lang="all"`, `"grc"` and `"el"`, and
`κληϊδα` gives `κλείς`. Three repairs of the first rebuild became drops
because their heads have no row (`Βάβυρσα`, `μός`, `πέδιος`), so `Βάβυρσά`
gives `None`. Rejecting the corpus pairs also fixes DBBE `Πολλῶν` untagged
(`πολύς`, was GLAUx's `πολλες`). On 119,894 annotated Perseus/Diorisis pairs only `σύνοχα` and
`βιατὰς` change among this issue's rows, both to the gold lemma.

Sentence-initial forms of οὗτος, πᾶς and the personal pronouns no longer
return other headwords or names that do not exist (#108): `Αὕτη` PRON gives
`οὗτος` (was `ἄτη`), `Τούτου` gives `οὗτος` (was `τοῦτος`), `Πάντα` ADJ and
untagged `Πάντες` give `πᾶς` (were `πάν` and `πάνετες`), `Ταύτης` gives
`οὗτος` (was `Ταύτη`), `Ταύτην` PROPN and `ταύτην` NOUN give `οὗτος`, and
`Ἡμῶν` PROPN gives `ἐγώ`. `Ταύτη`, `Τοῦτος` and `Ἡμων` have Ancient pages only
in the Greek Wiktionary, and GLAUx files forms of οὗτος under the name lemmas
`Ταύτης`, `Τοῦτης`, `Τούτῖς`, `Τουτό`, `Αὕτῃ` and `Τούτων`; these are listed
with their evidence in `dilemma/grc_rejected_lemmas.py`. The lookup build
drops their readings and rejects corpus pairs that name them,
`build/build_glaux_pos_lookup.py` skips them, and the runtime ignores POS-table
cells that target them (which already fixes the PROPN/NOUN cases on the
1.3.24 data). A capitalised GLAUx or Diorisis pair is also skipped when the
form's lowercase spelling reads, from a dictionary source, as a closed-class
pronoun and the pair names another lemma (about 60 rows: `Αὕτη` → `ἄτη`,
`Ἡμῶν` → `ἡμός`, `Τίνες` → `τίνω`). `ἄτη`, `πάν`, `πάνετες` and `τοῦτος` keep
their own readings. A reviewed name lemma is checked before the corpus
capitalisation repair lowercases it, so GLAUx's `Ταύτῃ` → `Ταύτης` no longer
passes as the lowercase `ταύτης`; untagged `Ταύτῃ` gives `οὗτος` from
Diorisis.

The second-person pronoun and the γραφή paradigm no longer resolve to σῦς and
γραφεύς (#110): `σύ`, `σοῦ`, `σεῖο`, `σέο`, `σέθεν` and `τεΐν` give `σύ`, and
`γραφή`, `γραφῆς`, `γραφήν`, `γραφαί` give `γραφή`, tagged or not. The merged
layer of `ag_lookup.json` had moved 19 keys of γραφή and 17 of σύ that the
Wiktionary base reads correctly, and a NOUN or PRON tag could not override a
reading with a capitalized twin; `_LOOKUP_OVERRIDES` restores them and also
corrects `ἱερέων` → `ἱερεύς`, `ἄριστα`/`ἀρίστοις` → `ἀγαθός` and `ἀρὰν` →
`ἀρά`. The case-twin rule of #42 stays limited to VERB: extending it to the
other tags measured 4 to 5 tokens worse on the held-out Gorman test, as the
issue reported. Together with #103, #108 and #122, measured on 119,894
annotated Perseus/Diorisis pairs (all of Perseus and every tenth Diorisis
pair) on an emulated rebuild: tagged, 19 answers change in both `lang="all"`
and `lang="grc"`, 14 to the gold lemma and 1 away from it (`ἀρὰν`, whose AGDT
gold is the Homeric `ἀρή`); untagged, 23 change, 15 to the gold lemma and 3
away (`ἀρὰν` and two gold rows that are themselves corpus artefacts, `Αιδης`
and `ἡμός`). DBBE accent-free equivalence rises from 89.0% to 89.3% untagged
and from 91.9% to 92.1% with gold POS. The benchmark gate is unchanged.

Ancient participles no longer pick up paradigm fragments from the Modern Greek
POS table (#122): `οὐσῶν`, `οὖσης` and `ὅντων` tagged ADJ give `εἰμί` (were
`αρχαΐζων`). Wiktionary's table for `αρχαΐζων` lists the bare participle
endings `ούσης`, `όντων` and `ουσών`, and other Modern noun and adjective
tables list bare endings (`αισώπειος`: `ος`, `ε`, `ους`), stray header letters
(`βρόντημα`: `γ`) or another word's cells (`κρεατομηχανή`: `μηχανή`).
`build_data.py` now skips a Modern case cell or headword form that is shorter
than its word and is either a bare ending or, for a common noun or adjective,
starts with another letter (`is_modern_nominal_fragment`); suppletive cells,
short name forms and headword alternatives stay. The published tables carry
the change as the recorded generator delta `data/changes/el_nominal_fragments.json`:
`scripts/refresh_el_generator_delta.py diff` over the 1.3.24 and the new
`build_data.py --lang el`, both run on the kaikki.org dumps pinned in the new
`data/kaikki_inputs.json` (the same Greek dumps and dialect extract
`el_generator_refresh.json` records), then `apply`ed to the published tables
with no conflict. 479 fragment cells change 26 `mg_lookup.json` keys (19 lose
their reading; 7 now take the key's next source, e.g. `ος` → `Ος`, `ες` →
`εις`, `ια` → `ίο`, `κο` → `ΚΟ`), 39 `mg_pos_lookup.json` keys (8 lose one ADJ
or NOUN reading; 31 are dropped because one reading is left and the table
holds only ambiguous forms, e.g. `όντων`, `ούσης`, `παίχτες`), and
`mg_pairs.json` (127 fragment pairs removed; the two `γένη` pairs whose lemma
the generator resolves through the key `ος` now name `Ος`). Nothing else
moved. With `lang="el"`, `ὄντων` and `οὐσῶν` tagged ADJ also give `εἰμί`.
Under `lang="all"`, `ας` tagged PART now gives `ὅς`, as it does untagged: the
Modern particle reading was reachable only through `αισώπειος`'s fragment
`ας` (follow-up).

The kaikki.org extracts the builders read are pinned in `data/kaikki_inputs.json`
(per file: URL, kaikki's Last-Modified date, size, sha256, license and Hub
archive path), and `scripts/archive_kaikki_inputs.py` stages them gzipped for
the Hub, restores them from a pinned Hub revision into a flat dump directory,
and verifies a dump directory against the pins.

s-stem neuters known only from their Ionic or uncontracted table now have the
Attic contracted cells (#114). The dictionary expansion and the Homeric
Wiktionary tables give στέφεος, στέφεα and στεφέων but no στέφους, so
`στέφους` NOUN fell to the Byzantine normaliser and returned στέφω (likewise
τέκους → τίκτω, κέρνους → κερνέω, σκέπους → σκεπός, γλάγους/θύους/κύους →
None). `build_lookup_db.py` now generates the contracted cells (-ους -ει -η
-ῶν -εσι(ν) -οῖν, acute on the stem's last vowel) for every lowercase -ος
lemma with an accented -εος row and a neuter plural row under it from Ancient
sources (`dilemma/grc_noun_gaps.py`, new lookup source `grc_noun_gaps`, not
share-alike, like `grc_adjective_gaps`), filling only keys no source
supplies: 543 keys for 124 lemmas. All eight issue forms now give their noun
in both modes. The plural-evidence check keeps κῦδος and σθένος out; Modern
Greek rows count for nothing (υπέρβαρος), a vowel- or rho-initial lemma needs
its breathing, a lemma an Ancient dictionary table also declines in the
second declension is a masculine or an adjective (λόχος, πρᾶος, λάϊνος,
ὄχθος, ὦρος; also the genuine neuter γάνος), and φῖλος is a reviewed
exclusion. A cell whose lowercase, grave-to-acute or monotonic spelling
already reads as another lemma is not added, so `λαθῶν` keeps `λανθάνω` and
`φιλοῖν` `φίλος`; the accentless key does not block a cell, so `σκέπους`
still gives `σκέπος`. That gate costs `ἔσθη`, which keeps `ἐσθίω` through
`έσθη` (DBBE gold `ἔσθος`), so of the two DBBE gains of the first emulation
only `στέφους` remains.

Nouns in -σις known only from a headword list now have their regular cells
(#115). σύναρσις had only its VLG headword row, so συνάρσει, συνάρσεως,
συνάρσεις and σύναρσιν gave None; the Byzantine paradigm expansion skips every
-ις noun because it cannot infer an -ις genitive. The same `grc_noun_gaps`
step gives every lowercase, singly accented, self-mapped -σις head the Attic
cells -εως -ει -εις -εων -εσι(ν) -έοιν and the accusative -ιν (no vocative,
which meets corpus names such as Κίμωσι), again only for absent keys, and a
key two accent doublets would share stays empty, and a head without the
breathing its first letter needs (όδευσις beside ὅδευσις) is skipped: 39,606
keys for 3,024 lemmas. συνάρσει NOUN → σύναρσις, ῥητρεύσει → ῥήτρευσις, and αἰτίωσιν →
αἰτίωσις (was αἰτιάω through the accent-stripped key); keys that already read
as another word keep it (κινήσει → κινέω untagged). Measured on an emulated
rebuild: benchmark gate and annotated pairs unchanged, DBBE +2 (συνάρσει,
Ῥητρεύσεών), no breaks.

### POS-aware lemmatization

A POS entry found only under a form's accent-stripped key no longer replaces
the reading of the form as written (#105). With a tag, `ὄρους` NOUN gave
`ὀρός` and `ὅρους` NOUN gave `ὀρός` (the entry under `ορους` belongs to
`ὀροῦς`), `ὅλων` NOUN gave `ὀλός`, `Ἄργους` PROPN gave `Ἀργώ`, `μᾶζαν` PROPN
gave `Μαζάν`, `αὐτή` NOUN gave `ἀϋτή` and `πόθῳ` VERB gave `πίνω`, although
the untagged call returned `ὄρος`, `ὅρος`, `ὅλος`, `Ἄργος`, `μᾶζα`, `αὐτός`
and `πόθος`. Such an entry no longer respells a reading the lookup holds for
the form's own spelling, no longer replaces an article or crasis reading, and
under NOUN or PROPN no longer replaces exact lookup readings that are not
verbs; the verb rule for case twins (#42) now needs an entry under the form's
own or monotonic spelling. Entries under the monotonic key, which keeps the
accent's place, are unchanged, and so are the overrides Byzantine misspellings
rely on (`μοχθῶ` NOUN gives `μόχθος`, `τῆδε` PRON gives `ὅδε`, `ἰδῶν` VERB
gives `ὁράω`). The numerals also no longer take the multiplicative adverb
under ADV: `ἐκτός` and `ἕκτος` ADV gave `ἑξάκις`, because 20 numeral cells of
the Wiktionary POS table carried it (`ἑπτά`, `ἕβδομος`, `πέμπτος`,
`ἑξήκοντα`, `ἕξ` ...). Measured with `lemmatize_batch_pos`, `lang="grc"`,
accent-sensitive scoring (#105 and #106 together): DBBE gold POS +12 tokens
(12 fixed, 0 broken), Gorman test +3 with predicted and +5 with gold tags (0
broken), the AGDT pairs +180 (191/11) and a 30,000-pair Diorisis sample +47
(48/1); every break is a reference that itself uses another spelling of the
same lemma (`ἄμαξα` beside `ἅμαξα`, `εὔξοος` beside `ἐΰξοος`) or the AGDT typo
`μάκαπ`. With `lang="all"`: DBBE +10 (10/0), Gorman +5/+9 (0 broken).
Untagged results and the benchmark gate are unchanged.

Wrong cells of the treebank, GLAUx and Wiktionary POS tables no longer
replace the lookup's correct lemma (#106): `μάκαρ` ADJ gave the AGDT typo
`μάκαπ`, `οὗπερ` PRON gave `ὄσπερ`, `καί` INTJ gave `αἴ` and VERB `φαίνω`,
`και` ADV gave `παι`, `τούς` ADJ gave `τού`, `αὐτόν` ADJ gave `αὐτος`, `μία`
and `Μία` NUM gave `εἶς`, and with `lang="grc"` `κλήσει` and `κλήσεις` NOUN
gave `κλῇσις`. They now give `μάκαρ`, `ὅσπερ`, `καί`, `ὁ`, `αὐτός`, `εἷς` and
`κλῆσις`. The fixes are a reviewed record, `data/pos_cell_corrections.json`
(78 rows, each with its reason and evidence), that Dilemma applies to the
tables as it loads them, so no table is rebuilt; a row applies only while its
table still holds the wrong target. The record also drops the 38 treebank
cells that put the adjective `γλυκερός` under `γλυκύς`, so `γλυκερός` ADJ now
gives `γλυκερός` as the untagged call does (AGDT references that use `γλυκύς`
count as 2 of the 11 AGDT breaks above). A general check of the tables'
targets against the lexical-head lists was measured and not adopted: it also
removed real lemmas the head lists lack (`βουλευτέος`, `ἐγγυτέρω`, `δείρη`)
and lost 7 net on the AGDT pairs.

A POS-table target whose initial rho has no breathing no longer replaces the
lemma of an Ancient form (#107). Some GLAUx and treebank texts write an
initial rho without its breathing, so the corpus POS tables held `ρῆσις` and
`ρέω` under `ρήσεις`, `ρύω` under `ρυήσονται`, `ρόιζος` under `ροίζῳ` and
`ρομφαία` under `ρομφαίαν`, and polytonic input reached them through its
monotonic key: `ῥήσεις` NOUN gave `ρῆσις` and VERB `ρέω`, `ῥυήσονται` VERB
`ρύω`, `ῥοίζῳ` ADJ `ρόιζος` and `ῥομφαῖαν` NOUN `ρομφαία`, with `lang="all"`
and `lang="grc"`. For Ancient input such a target of the Ancient tables is now
skipped and the lookup decides: `ῥῆσις`, `ῥῆσις`, `ῥέω`, `ῥοῖζος`,
`ῥομφαία`. Modern Greek entries with a bare rho are unaffected (`ῥα` INTJ
still gives `ρε`). Measured with `lemmatize_batch_pos`: DBBE gold POS +4
tokens accent-sensitive (`ῥήσεις` three times and `ῥομφαῖαν`; 0 broken), a
150,000-pair Diorisis sample +1, the AGDT pairs and the benchmark gate
unchanged.

A form written with a breathing whose lemma is found only through the
lookup's monotonic or accent-stripped key no longer gets a lemma without that
breathing (#109). Those keys hold many Modern Greek rows, so with `lang="all"`
and `lang="grc"` alike `εὐσεβεστάτῃ` gave `ευσεβής`, `ἀστοχήσει` `αστοχώ`,
`ἀνεξαντλήτου` and (through the Byzantine normalizer) `ἀνεξάντλητον`
`ανεξάντλητος`, and `ὀφφικίου` `οφφίκιον`. Such a lemma is now replaced by the
Ancient headword with the same letters, with the form's breathing or the other
one and the accent as written or an acute turned circumflex (`εὐσεβής`,
`ἀνεξάντλητος`; `ἐδραία` gives `ἑδραῖος`, `ἀνευροῦς` `ἄνευρος`); a
breathingless Ancient head takes the form's breathing (`ὀφφίκιον`); a
capitalized name only Modern Greek lists keeps its lemma (`Ῥαχὶλ` gives
`Ραχίλ`); with `lang="all"` a Modern Greek lexical head with no Ancient twin
is kept, so polytonic vernacular Greek still reads as Modern as in 1.3.24
(`ἀκόμη` gives `ακόμη`, `ἀξίζει` `αξίζω`, `ἀλλιῶς` `αλλιώς`, `ἀστοχήσει`
`αστοχώ`); otherwise there is no lemma (`ἀστοχήσει` gives `None` with
`lang="grc"` and `guess=False`). `lang="el"` and
`convention="triantafyllidis"` are unchanged.
Measured: DBBE untagged strict 6,799 -> 6,809 with `lang="all"`
(equivalence-adjusted 7,299, unchanged) and 6,808 -> 6,819 with `lang="grc"`
(7,295 -> 7,296); DBBE gold POS strict 7,096 -> 7,105 (`all`, #107 included;
equivalence-adjusted 7,541 -> 7,539) and 7,108 -> 7,118 (`grc`; 7,541 ->
7,540); Katharevousa gate +1 (`ἐνδυθεὶς` gives `ἐνδύω`, not `ενδύομαι`); the
AGDT pairs unchanged; Diorisis +1. The two equivalence-adjusted losses are
tagged calls the existing POS rejection audit now turns to `None` (`ὀφφικίου`
NOUN, `ἀββακοὺμ` NOUN), because the heads list holds only the Sophocles
spelling `οφφίκιον` and no `Ἀββακούμ`; untagged they give `ὀφφίκιον` and
`Ἀββακούμ`. Keeping capitalized names avoided the three further losses the
plain rule had (`Ραχίλ`, `Αγιοστεφανίτης`, `Απόκρεω`). Keeping the Modern
head under `lang="all"` was measured on the rebuilt data against the rule
without it: of the 15,000 most frequent forms of the Wikisource polytonic
Modern Greek slice, 50 answers change (7,068 tokens), 37 of them from `None`
to the Modern lemma (`ἀκόμη` 1,659 tokens, `ἀκόμα` 956, `ἔξαφνα`, `ἀγκαλιά`,
`ὡστόσο`, `αὔριο`, `ἔργο`), and others from an unrelated Byzantine-normalizer
answer to it (`ἁμάξι` was `ἁμαξίς`); on DBBE 4 answers change with and
without gold POS and one leaves the gold lemma (`ὠδικοῦ` gives `ωδικός`, not
`ᾠδικός`); `lang="grc"` is unchanged.

The polytonic Modern Greek POS rule of 1.3.24 (`lang="all"`) gives the
Ancient answer again for three kinds of Ancient forms (#123). A misaccented
word with hiatus counted as a monosyllable, so its bare key counted as its own
spelling (`προῖὸν` NOUN gave `προϊών`); monosyllables are now counted with
diphthongs, the diaeresis and synizesis (two adjacent nuclei whose first is an
unaccented, non-initial ι, υ, ει or οι), so `μιὰ`, `ποιὸς` and `γριὰ` still
count and `προϊόν`, `πρωΐας` and `Κάιν` do not; `προῖὸν` gives `πρόειμι`. When
a Modern reading stands against an Ancient reading that is a capitalized name,
the lowercase Ancient headword with the Modern lemma's letters is returned
(`ἄξυλος` ADJ gives `ἄξυλος`, not `Ἄξυλος`; `ἐκδηλος` gives `ἔκδηλος`, not
`έκδηλος`; `εὐήνεμός` gives `εὐήνεμος`). A capitalized form tagged PROPN whose
Ancient reading is a lowercase word now keeps it (`Ὀρᾷς` gives `ὁράω`, not
`Όρας`; `Τοῦτου` `οὗτος`; `Γένῃ` `γίγνομαι`), which reverses the `Γένῃ` PROPN
-> `Γένη` behaviour 1.3.24 tested. Measured over the 26,290 attested or
annotated (form, tag) pairs the rule can reach: the monosyllable count
changes 37 (94 corpus tokens), the lowercase headword 32 (68 tokens; 2
annotated pairs, both fixed), the PROPN rule 129 (626 tokens, 127 of them to
the untagged answer, e.g. `Ὀρᾷς` over 151 tokens, `Ὅρᾳ` over 123); a handful
of genuine names are lost (`Καροῦ`, `Λειχοῦδης`). DBBE, the AGDT pairs and the
benchmark gate are unchanged by these three; Diorisis +1 (`ἄξυλος`).
Correctly spelled forms the lookup lacks (`συνῃρημένῃ`, `Βοῤῥᾶ`) and accent
slips (`ἄπαν`) still take a Modern headword, and Modern-origin rows stored for
both languages still count as known Ancient spellings (`ἄντρες` NOUN keeps
`ἄνδρας`); counting only Ancient-origin rows changed 24 pairs with mixed
results, so both stay documented gaps in docs/usage.md.

With `lang="all"` and `convention="triantafyllidis"`, a Modern Greek POS
entry that the polytonic rule lets stand is returned when the lookup offers
the same lemma from its Modern and its Ancient side, as under the default
convention (#119): `σεῖς` PRON gives `εσείς`, not `σέω`; `κυροῦ` NOUN gives
`κυρός`, not `κυρώ`; `σὰν` CCONJ gives `σαν`, not `σός` (`νὰ` CCONJ still
gives `νά`, as with `lang="el"`). Measured: no change under the default
convention; under triantafyllidis 0 of 465,573 annotated Ancient (form, UPOS)
pairs, 8 of 660,000 tag requests (e.g. `ἑπομένῃ` ADJ `επόμενος`, `ῥὰ` INTJ
`ρε`) and 1 DBBE token (`κυροῦ`, now correct).

`lemmatize_pos` and `lemmatize_batch_pos` agree when no POS entry chooses a
candidate. The scalar call returned the first verbose candidate and the batch
call kept the `lemmatize_batch()` answer, and under `convention="triantafyllidis"`
the two rank differently: with `lang="all"`, `προῖὸν` NOUN gave `πρόειμι`
from one and `προϊών` from the other, `γραφῆς` ADJ `γραφή` and `γραφεύς`.
`lemmatize_pos` now returns the `lemmatize_batch()` answer with the batch
call's adjustments. On 4,800 requests (400 test words, 12 tags each) the
disagreements under `lang="all"` with triantafyllidis fall from 347 to 0;
`lang="all"`, `"grc"` and `"el"` with the default convention and `"el"` with
triantafyllidis had none, and no answer changes there.

Polytonic Modern Greek conjunctions keep their Modern reading under SCONJ as
under CCONJ (#120): an SCONJ request reads the Modern CCONJ entry when the
Ancient POS tables have no CCONJ entry for the form (`νὰ` SCONJ gives `να`,
not `νή`; `σὰν` SCONJ gives `σαν`, not `σός`; `ὄντας` SCONJ gives `είμαι`).
Ancient conjunctions are unchanged (`ἐπειδὴ` SCONJ gives `ἐπεί`). Measured: 0
of 465,573 annotated Ancient pairs, 0 DBBE tokens and 0 gate tokens changed.

With `lang="all"`, polytonic vernacular contractions of σε with the article
(`στὸ`, `στὸν`, `στὴ(ν)`, `στὰ`, `στοῦ`, `στῆς`, `στῶν`, `στοὺς`, `στὶς`) give
`ο` under DET and `σε` under ADP, under every convention (#121). `στὸ` DET
gave no lemma since 1.3.23, `στὸν` DET gave `ἀστός` and `στὴν` DET `ἵστημι`.
Other tags keep the Ancient readings (`στὸν` ADJ gives `ἀστός`). Measured: 0
of 465,573 annotated Ancient pairs and 0 DBBE tokens changed.

A single unaccented Greek letter tagged NOUN or ADJ (a numeral, siglum or
list label) no longer returns an unrelated word from a one-letter POS-table
cell or an accent-stripped lookup row (#116): `α` ADJ gave `Λητογενής`, `ι`
NOUN `ζευγίτης`, `υ` NOUN `ὗς`, `ε` NOUN `ἔ`, `δ` NOUN `δεσποινίς` (with
`lang="all"`) and `γ` NOUN `βρόντημα` (with `lang="el"`). `lemmatize_pos` and
`lemmatize_batch_pos` now return the letter itself under those two tags, also
with `guess=False`; with `lang="grc"`, `α` gives `None`, because the treebank
audit rejects it as a target. Other tags (`η` DET → `ὁ`, `κ` X → `Κ`),
accented or breathed letters (`ἡ`) and untagged lookup are unchanged. On the
AGDT/Perseus pairs three types change (`ι` ADJ now matches its gold `ι`; `α`
ADJ and `ο` ADJ leave the artefact gold lemmas `Λητογενής` and `εὐρύπορος`
that created the cells); the gate fixtures and DBBE are unchanged by this fix.

A NOUN tag on a dative that has an adverbial self-map row now returns the
noun (#111). `σιγῇ`, `βίᾳ`, `ἡμέρᾳ`, `σιωπῇ`, `παρρησίᾳ` and `κομιδῇ` under
NOUN came back unchanged: the stripped comparison matched the Wiktionary
self-map to the POS target, and the safeguard that keeps `οὖ` beside `οὐ` then
kept the self-map. A self-map that differs from a NOUN or PROPN target only by
an added iota subscript now defers to the target (`σιγῇ` NOUN gives `σιγή`);
under ADV, without a tag and in Modern Greek the adverb row is kept (`σιγῇ`
ADV gives `σιγῇ`), and `οὖ` ADV still gives `οὖ`. Measured: gate fixtures
unchanged; the only changed answers in 110,862 tagged Perseus and Diorisis
pairs are six Perseus NOUN datives (`βίᾳ`, `εὐκαιρίᾳ`, `σιγῇ`, `σιωπῇ`,
`σχολῇ`, `ἡμέρᾳ`), each now the gold lemma exactly instead of only
accent-equivalent.

### Articles, participles and adverbs

With `resolve_articles=True`, smooth-breathing `αι`/`οι` forms are no longer
read as the article (#112). The article check compared monotonic spellings,
and because the article set holds `αἵ`/`οἵ` (the article accented before an
enclitic), `αἴ`, `αἲ`, `αἶ`, `αἰ` and `οἴ` all collapsed onto it: `αἴ` tagged
PART or SCONJ, `οἴ` tagged INTJ and the untagged forms gave `ὁ`. A smooth
breathing on `αι` or `οι` now ends the article check, since the plural
articles have a rough one; `αἴ` PART gives `αἴ`, `οἴ` gives `οἴ`, and `αἶ`
gives the lookup reading `αἶα`. A combining psili after the final vowel's
own breathing or accent is an elision mark, not a smooth breathing, so the
elided `αἶ̓` and `οἷ̓` (AGDT Sophocles) are not articles either, and
`lemmatize()` now tries the elided entry and the vowel expander first for
them, as `lemmatize_batch()` already did: both give `αἶα` and `οἷος`, as
with `’` (`βαί̓` and `ὁποῖ̓` no longer give `βάι` and `ὄψ` from the scalar
call). Monotonic and rough-breathing spellings (`αί`,
`αἵ`, `Αἱ`, `ΑΙ`) still give `ὁ`, and so do the Byzantine scribal spellings
`ὀ` and `τἠς`, which the DBBE gold annotates as the article (excluding every
smooth breathing broke those 7 tokens, so the rule is limited to `αι`/`οι`).
Measured: gate fixtures unchanged; Perseus 80,911 pairs +5 correct and 0
broken (with and without tags); a 30K Diorisis sample 0 broken; DBBE gold +2
(`αἲ` twice), 0 broken.

An oblique case of a `-μενος` participle with no row of its own now gives the
lemma of its nominative (#113). Participles are usually listed in the
nominative singular only, so `ἐλευσομένῳ`, `ἐλευσομένου`, `ἐλευσομένοις` and
`ἐλευσόμενοι` gave `λεύσσω` (the augment strip read them as `λευσομένῳ`) and
`ἐλευσομένας`, `ἐλευσόμενα`, `ἐλευσόμεναι` and `ἐλευσομέναις` gave `None`. The
form's masculine nominative singular is rebuilt with the recessive accent,
then with the perfect's `-μένος`, and when it is a lookup key whose reading is
a verb its lemma is returned (all eight give `ἔρχομαι` through `ἐλευσόμενος`;
`lemmatize_verbose` reports `via="participle_nominative"`). The augment strip
no longer applies to a present, future or aorist participle, and a temporal
augment is accepted only when the lemma begins with the restored vowel, so
`εἰκόνισεν` no longer gives `κονίω` (from `ἐκόνισεν`); it now gives the Modern
Greek `εικονίζω`, because no source lists the Ancient aorist of `εἰκονίζω`.
Measured: gate fixtures unchanged; DBBE gold +7 (`ἐλευσομένῳ`, `εἰκόνισεν`
accent-equivalent, `πεπλησμένην`, `προσκαθημένας`, `λυτρουμένην`,
`διωρθωμένου`, `ὠρθωμένοις`) and 1 neutral change (`ἱερωμέναις` gives
`ἱεράομαι` against gold `ἱερόω`), 0 broken, with and without gold POS; a 30K
Diorisis sample +1 (`πεφοινιγμένας`); Perseus unchanged.

Regular Ancient adverbs in -ως now have one lemma: their adjective, in the
tagged and the untagged calls, with `lang="all"` and `lang="grc"`, and under
every Ancient convention (#117). Before, the answer depended on the form and
on the tag: `καλῶς` gave `καλός` under ADV but `καλῶς` untagged, `ὀρθῶς` gave
`ὀρθῶς` both ways, and `ἀληθῶς` gave `ἀληθῶς` under ADV but `ἀληθής`
untagged; `convention="lsj"` mapped only three of them, because its adverb
remap read `lsj9_headword_pos.json`, which was never published. The new
generated map `data/grc_adverb_heads.json` (`build/build_grc_adverb_heads.py`,
15,357 adverbs from the POS tables and headword lists, with input hashes) is
applied in one place, so `καλῶς`, `ὀρθῶς`, `ἀληθῶς`, `συντόμως` and `ταχέως`
give `καλός`, `ὀρθός`, `ἀληθής`, `σύντομος` and `ταχύς` everywhere.
Pronominal and participial adverbs (`οὕτως`, `πῶς`, `ὄντως`, `πάντως`) and the
reviewed lexicalized adverbs of `data/grc_adverb_review.json` (`ἴσως`,
`ἰδίως`, `ἄλλως`, ...) keep their own headword; `lang="el"` and
`convention="triantafyllidis"` keep Modern Greek adverbs (`κυρίως`,
`διεθνώς`). With `lang="all"`, an adverb spelled the same in both languages
(`κυρίως`, `δικαίως`) is read as Ancient and gives the adjective. Equivalence
groups no longer resolve to an adverb, so `convention="cunliffe"` stops
mapping `ἄδικος` to `ἀδίκως`. The dead LSJ adverb/neuter remap is removed.
Measured: Diorisis -ως pairs 3,596 -> 4,189 correct untagged and 3,955 ->
4,374 tagged; Perseus +99/-12 untagged and +30/-13 tagged; DBBE +29/-2
untagged and +11/-0 tagged; a random 30K Diorisis sample +43/-0 and +28/-0;
the Ancient gate fixture gains `ὑποτυπωτικῶς` -> `ὑποτυπωτικός`, Katharevousa
and Demotic are unchanged. The losses are adverbs the references themselves
lemmatize both ways (`ταχέως`, `ἀκριβῶς`, `κυρίως`). See "Adverbs in -ως" in
docs/usage.md.

### Data directory check

A data directory that lacks a list lemmatization reads now says so (#118).
Before, every reader of the headword lists treated a missing file as empty, so
a directory assembled from part of the published files loaded without a
message and answered differently: with the eleven `*_headwords.json` lists
missing, `ὀλεθροτόκον` ADJ gave `None` instead of `ὀλεθροτόκος` and `καὶ` INTJ
gave `αἴ` instead of `καί`. `Dilemma()` now emits one `DilemmaDataWarning` per
data directory naming each missing file (the Ancient lists,
`grc_lexical_heads.json`, `grc_adverb_heads.json` and
`pos_cell_corrections.json` for `grc`/`all`, the two
Modern Greek lists for `el`/`all`, and `lemma_equivalences.json` with the
convention's own file) and the command that fetches them,
`python -m dilemma download --no-tagger`; `dilemma.core.missing_data_files()`
lists them. It is a warning rather than an error, so Modern-Greek-only and
minimal directories still load; a full download is silent and no answer
changes. `pos_cell_corrections.json` is now also in `scripts/hf_data.py`'s
tracked set, so it reaches the Hub and `python -m dilemma download`; without
it the #106 corrections were silently absent from installed data.

### Development

`scripts/run_test_chunks.py` runs the test suite as several pytest processes,
and CI now uses it: one process over the whole suite peaks at about 17.5 GB,
more than the 16 GB CI runner has. The Ancient Greek, Modern Greek and
Medieval Greek kaikki dumps the data build reads are archived on the Hub and
pinned in `data/kaikki_inputs.json`; `scripts/archive_kaikki_inputs.py
download` restores them and checks every sha256.

### Known limits

- The lookup holds one reading per key, so genuine homographs keep that
  reading under every tag: `ἀξίαν` gives `ἄξιος` and `ἱερὰ` gives `ἱερόν`.
- Headword-only Ancient adjectives and nouns in -ος (about 16,900 lemmas,
  mostly Byzantine entries without a gender) still have no generated cells.
- With `lang="grc"`, a polytonic Modern Greek word that no Ancient head
  spells gets no lookup lemma and can reach the Byzantine normalizer, which
  may find an unrelated word (`ἀλλιῶς` gives `ἁλιεύς`, `ἀνάβει` `Ανάβης`,
  `ἄνοιξη` `ἀνοίγω`); with `lang="all"` they keep their Modern lemma.
- A key that the #103 or #108 rules empty can be refilled by a Modern Greek
  self row: untagged `Πάσας` gives `Πάσας`, not `πᾶς`, and `Τούτος` gives the
  Modern `Τούτος`.
- `αἴ` SCONJ gives `αἴ`, not `εἰ`: the Ancient POS tables hold only a CCONJ
  cell for it, and the SCONJ-to-CCONJ reading of #120 applies to Modern Greek
  entries only.
- With `lang="grc"`, a bare `α` tagged NOUN or ADJ gives `None`, because the
  treebank audit rejects `α` as a target.
- The Sophocles lexicon writes its omicron and rho heads without breathings,
  and `grc_lexical_heads.json` and the POS rejection audit carry those
  spellings, so `ὀφφικίου` NOUN gives `None` and breathingless rho heads
  remain in the lists. Repairing them needs a rebuild of
  `grc_lexical_heads.json` from inputs this release does not rebuild.

## [1.3.24] - 2026-11-13

With `lang="all"`, vernacular Greek written with polytonic accents keeps its
Modern Greek POS reading again. 1.3.23 stopped reading the Modern Greek POS
entries for every polytonic form, so `νὰ` tagged CCONJ gave `νή`, `κυροῦ`
tagged NOUN gave the verb `κυρόω`, `σεῖς` tagged PRON gave `σέω`, and the
misaccented `ἀλλην` came back unchanged. A Modern Greek entry now stands for
polytonic input when the Ancient side offers nothing of the requested kind:
nothing Ancient reads the form; the reading is an unaccented echo that is no
Ancient headword; the Ancient POS tables record it only under a conflicting
part of speech, and the tag is a closed class or the accented spelling is
unknown to them; the reading is a capitalized name for a lowercase word not
tagged as a noun (`κακὸ` ADJ gave the name `Κακό`, `πιὰ` ADV gave `Πῖα`); or
the Ancient side records nothing, and either the reading was found only by
discarding the form's accents while the Modern Greek entry is recorded for
them, or the lookup holds the spelling but its reading has a verb ending
under a closed-class tag. `docs/usage.md` states the rule and its known
gaps.

The 1.3.23 cases are unchanged: the same word in monotonic spelling gives the
Ancient headword (`ἄλλην` PRON gives `ἄλλος`), an Ancient reading of a
compatible part of speech is kept (`Ἑλλήνων` PROPN gives `Ἕλλην`), an
open-class tag does not displace a spelling the Ancient side knows (`ὄντων`
ADJ gives `εἰμί`), and an Ancient form written without its accent, or with
the second accent of a following enclitic, is not read through an
accent-stripped Modern Greek entry (`ἐφη` VERB gives `φημί`, `Πᾶνά` PROPN
gives `Πάν`). The rule weighs the same Ancient evidence under every
`convention`. With `lang="all"` and `convention="triantafyllidis"`, though, a
Modern Greek entry it lets stand is often not returned, as in 1.3.22: `νὰ`
CCONJ gives `νά`, but `σεῖς` PRON still gives `σέω` and `κυροῦ` NOUN `κυρώ`.
`lang="el"` is not affected.

Measured against 1.3.23 on 521,484 polytonic (form, tag) pairs from the
Diorisis and AGDT annotations and the Ancient and Katharevousa benchmark
forms (build sources, not held out): under a form's annotated tag one answer
reaches the annotated lemma and none leaves it; under the other tags a tagger
may give that annotation nothing changes; under every other tag the POS
tables offer for the form, 84 answers leave it and 2 reach it. 1.3.22 left it
on 88, 431 and 1,794 of those pairs. On the DBBE gold standard with gold POS
this restores 7 of 8,342 tokens. Vernacular forms that are also plausible
Ancient spellings still get the Ancient reading (`σᾶς` PRON gives `σός`,
`Νὰ` CCONJ gives `ἵνα`, `στὸν` DET gives `ἀστός`), and `στὸ` DET still gives
no lemma. `νὰ` is restored under CCONJ and PART only: no Modern Greek entry
carries SCONJ, so `νὰ` SCONJ gives `νή` as before. An Ancient form whose
spelling the lookup lacks, through an accent slip or a gap, can take a Modern
Greek headword when Modern Greek records a word with that accent (`ἄπαν`
NOUN gives `άπαν`).

With `guess=False` the compound fallback returns only a lemma that a
dictionary lists. 1.3.23 let monotonic Modern Greek input without a headword
keep the accent of its base lemma, and that reconstruction joins unrelated
pieces as readily as real ones: `βράβευσον` gave `βραβευσός` (`βραβευ` +
`σός`) and `σελίδεσιν` gave `σελιδέσις`. Among the 291,613 most frequent
Modern Greek forms, 3,270 `guess=False` answers were such reconstructions,
and most of a 70-form sample were non-words. Of the 3,266 answers that
change, 2,861 now come back as `None`, or as no candidates; the other 405
are answered by the fallbacks that `guess=False` keeps, Byzantine
normalization and prefix stripping, which the reconstruction used to
pre-empt. `repair_lemma` reads the same candidates and follows. With
`guess=True` nothing changes: the reconstruction still replaces a form the
transformer leaves unresolved (`μονοακόρεστα` gives `μονοακόρεστος`). In the
deterministic benchmark gate, which runs with `guess=False`, Demotic loses
that one token (383 of 400 equivalent, from 384) and the Katharevousa
`Νικέρωνα` no longer gives `νικερώνω`; the reviewed baseline is refreshed to
match (#103). The refresh also takes in two answers 1.3.23 changed without
refreshing it: Katharevousa `θρυλλουμένου` gives `θρυλέω` (297 strict and 302
equivalent of 318, from 296 and 301), and the hyphenated
`Παλαιοανθρωπολογίας-Σπηλαιολογίας` gives no lemma.

## [1.3.23] - 2026-11-13

With `lang="all"`, POS-aware lemmatization no longer turns polytonic Ancient
forms into monotonic Modern Greek headwords (`ἄλλην` PRON gave `άλλος`,
`Ἑλλήνων` PROPN gave `Έλληνας`). For input with breathings or polytonic
accents only the Ancient POS entries are read; when they have no reading the
lookup candidates decide, as with `lang="grc"`. Monotonic input still reads
the Modern Greek entries (#100).

Compound analysis returns accented citation forms. The reconstruction joined
accent-stripped pieces (`ὀλεθροτόκον` gave `ολεθροτοκος`); it now takes the
spelling of the matching headword from LSJ, LSJ10, Cunliffe, DGE, LBG or
Wiktionary (`ὀλεθροτόκος`), Modern Greek headword lists for monotonic input.
Monotonic Modern Greek input without a headword keeps the accent of its base
lemma (`μονοακόρεστα` gives `μονοακόρεστος`). Otherwise the compound
abstains, so `guess=False` returns `None` (#101).

## [1.3.22] - 2026-11-12

`lookup.db` attributes the readings of a merged input to the layer that
recorded them, where byte-pinned records tell the layers apart. The Ancient
rows that all named `wiktionary_grc` now name `wiktionary_grc` (2.29M rows,
readings of the pinned Wiktionary-only base), `grc_lexicon_expansion` (6.03M,
paradigms generated from LSJ, Sophocles and Cunliffe headwords) or
`grc_merged` (2.31M, readings no pinned record assigns). Modern Greek present
and imperfect forms generated by rule name `el_rule_paradigms` (16K rows).
`wiktionary_grc` therefore means less than it did in 1.3.21, and the three
new sources are appended to the `sources` table, so existing ids are
unchanged. The build verifies the layer files against their pins and stops on
a mismatch. No reading changes because of this; the schema version stays 3.

63 more Ancient `-ικός` adjectives regain their feminine and neuter-plural
forms, from a hand review recorded with reasons in
`data/grc_ikos_review.json`: 52 lowercase heads and 11 capitalized ethnic or
proper adjectives such as `Πλατωνικός`. The generated cells are now added
after every other Ancient source, so a form a corpus supplies keeps its
corpus reading (`γενετική` is again the noun). The lookup holds 9,699 such
keys for 563 lemmas.

## [1.3.21] - 2026-11-12

Ancient `-ικός` adjectives that the pinned Sophocles expansion declined as
masculine nouns regain their feminine and neuter-plural forms (#95):
`μοναχικὴν` now gives `μοναχικός`. `build_lookup_db.py` fills 8,532 absent
keys for the 507 lemmas listed in `data/grc_ikos_adjectives.json`, each with
independent evidence that it is a three-ending adjective, and the Sophocles
parser now recognizes the OCR's renderings of `ή, όν`.

`lookup.db` names the source of every reading. Each row's `source_id` refers
to a new `sources` table with the source's name, components, licenses,
share-alike flag and attribution, copied from the reviewed registry
`data/lookup_sources.json`; the build stops if a reading comes from an input
the registry does not list. The schema version is 3, and `meta` adds
`sources_sha256`. NOTICE now also lists the Cunliffe lexicon, the headword-only
lexica, the Byzantine headwords and the Nestle 1904 New Testament.

The compound fallback no longer invents lemmas from unrelated parts. It took
any string ending in a linking vowel as a first element and any lookup hit as
the second, so `μοναχικὴν` gave `μοναχιεν` (`μοναχι` + `ἐν`, from the crasis
`κἠν`). The second element must now begin like its lemma.

The convention remap tables (`convention_lsj.json`, `convention_cunliffe.json`,
`convention_triantafyllidis.json`) are published with the downloaded data.
Without them an install with only the download applied no convention remap.

## [1.3.20] - 2026-11-12

The published data now carries the Modern Greek headword fixes and the
accented Byzantine forms. `χιονόνερο` and `μπαλωθιά` are headwords again, with
their inflections, and `lemmatize_pos("ατσίδας", "NOUN")` gives `ατσίδας`
(#87, #88, #89). The generator change reached the pinned Modern Greek snapshot
as a bounded refresh: `scripts/refresh_el_generator_delta.py` recorded the old
and new generators' difference on the Wiktionary dumps of 2026-10-30 and
2026-11-03 and applied all 6,748 lookup keys, 360 POS readings and 2,132 pair
changes without a conflict (`data/changes/el_generator_refresh*.json`).
`lookup.db` was then rebuilt in full, so it also gains the `meta` table, the
`lemma_id` index, about a million accented Byzantine forms with their
accentless keys, and the reviewed grammatical readings, and drops repair-era
rows that repeated the combined reading. The file grows from 1.22 GB to
1.50 GB.

Generator changes for Modern Greek headwords. The published
artifacts are unchanged until the next rebuild.

A dictionary headword list now comes before the other head rules: every
headword of the Triantafyllidis list has a key that ends at a headword, with no
page or table required. A lowercase headword whose key ended at a capitalized
word is its own lemma again (`κλέφτης` was under the name `Κλέφτης`, `θεότητα`
under `Θεός`). Any other key that already ends at a headword keeps its
Wiktionary lemma.

Read the Cretan and Cypriot sections of Greek Wiktionary, which the extract
files outside the Greek language file. `build/extract_el_dialect_sections.py`
collects them from the edition's raw extract. A dialect entry fills a gap only
and its pairs are tagged with the dialect.

Restore headwords whose Greek Wiktionary page the extract omits. `χιονόνερο`
has a malformed noun heading and `μπαλωθιά` is headed as Cretan, so neither is
in the Greek language file. Reviewed records of the cited page revisions supply
the definition and declension, and retire when the extract has the page again.
A reference from an inflection page alone still defines no headword.

Keep `ατσίδας` as a noun headword. A noun given as the other gender of another
noun, with its own declension, no longer becomes a form of that noun, and a
defined noun keeps its key when one of its senses names a word that has no
entry. The feminine genitive `ατσίδας` remains under `ατσίδα`, and the surname
`Ατσίδας` is unchanged.

`build_lookup_db.py` now stamps `lookup.db` with a `meta(key, value)` table:
`schema_version`, the `dilemma_version` that built it, the `lemmas` and
`lookup_rows` counts, and `lookup_sha256`, a hash of the logical rows that a
consumer can pin instead of the file's bytes. It also indexes `lookup` by
`lemma_id`, so every form of a lemma is read without a table scan. Both appear
with the next published database; files built earlier have neither.

`data/ag_headwords.json` and the headword lists and small tables the runtime
reads (LSJ, Cunliffe, DGE, LBG, Triantafyllidis and the others under `data/`)
are now published with the downloaded data. The wheel carries no data files,
so an install with only the download had no headword lists and a lemma
convention such as `lsj` or `triantafyllidis` loaded an empty headword set.

`build/expand_lbg.py` refuses to write a Byzantine pairs file whose forms are
mostly unaccented, and `build_lookup_db.py` warns when it reads one. The pairs
file behind recent databases had lost its accents (807 of 1,561,683 keys
carried a diacritic), so about 41,000 Byzantine lemmas have only accentless
inflections in the lookup. Lemmatization is unaffected, because lookups fall
back to the accentless key, but a reader that lists a lemma's forms gets none
that match polytonic text. The current expansion code produces accented forms;
the fix reaches the data with the next `lbg_pairs.json` and lookup rebuild.

## [1.3.19] - 2026-11-11

When Ancient Greek article resolution is enabled, a noun or proper noun tag
no longer overrides the canonical lemma for a marked article. Both scalar and
batch lemmatization still enforce input attestation and rejected-target checks.

## [1.3.18] - 2026-11-09

Correct the source form-page reading that labeled άχθηκα as imperfect. It now
uses the aorist, while the original source gloss and correction evidence remain
in the audit. Other readings, including the shared imperfect and aorist of
παρέσυρα, are unchanged. The pinned snapshot can be repaired and restored
without rebuilding unrelated data.

## [1.3.17] - 2026-11-09

Keep explicit verb-cell tenses ahead of a table's heading, so imperfect forms
do not acquire present-tense analyses. Preserve independently supported
readings rather than merging their tense tags.
Present/future alternatives remain separate analyses, and imperfective future
labels no longer become imperfect past tense. Pooled head-line labels do not
assign every listed tense to each spelling.

Complete missing active imperfect cells from source-recorded past stems,
retaining both augment variants of συλλαμβάνω. Form-of pages and unknown past
stems are not used to infer new paradigms. Keep the dialect conjunction τσαι
separate from the noun τσάι without changing canonical spelling variants such
as δυο/δύο. Explicit Modern Greek POS tags select the corresponding accented
owner when both readings are recorded. The published snapshot includes reversible corrections, archived
source evidence, and exact input hashes.

## [1.3.16] - 2026-11-08

Dilemma can now build a monotonic Modern Greek keyboard bundle: a rated word
list, abbreviation and acronym lists, and a contextual next-word model.
Corpus inputs are openly licensed, pinned and split by document. Archived
Tatoeba and Wiktionary examples can be downloaded from an immutable Hub
revision, preserving the original train/dev split.

The exporters admit independently attested formal words and acronyms, retain
rare words with metadata for common accent alternatives, and restrict
lowercase accent restoration to common or medium-frequency names when no
lowercase word matches. Evaluations cover held-out coverage, predictions,
completion, accent restoration, typing slips, enclitic accents and keystroke
savings, with Python and Swift reader checks.

Retain source-listed dialect and older spellings with explicit preferences for
standard proposals. Restore uppercase dialytika (`ΤΣΑΪ`, `ΜΑΪΟΣ`), recognize
lowercase dotted initialisms (`ε.ε.`), and preserve acronyms that are also
lowercase words (`ΕΚ`/`εκ`, `ΠΑ`/`πα`). Common ratings for capitalized names
require corroboration from a large source that preserves case.

Complete missing present-active cells beneath independently defined Modern
Greek verb heads, preserving source-supplied forms and existing lookup
readings. The refreshed data adds 9,450 pairs across 1,904 heads with
reversible provenance. The older Modern Greek Hunspell export now refuses
missing or empty form-frequency data instead of writing an empty dictionary.

## [1.3.15] - 2026-11-05

### Fixed
- Extend reviewed grammatical analyses to additional `εἶμι` compounds,
  `οἶδα` pluperfects, and copula forms, including person and number variants.
  Correct related-headword conflations in older POS tables without replacing
  independently supported lexical owners.
- Preserve separate verb readings and their tenses for ambiguous forms such
  as `ᾖσαν`, `ἴσθι`, `ἔπεσαν`, and `ξύνισαν`. Keep grammatical identities
  when a requested dictionary convention maps them to the same citation.

### Added
- Add a source-reviewed passage audit across all five lemma APIs, retaining
  original corpus annotations and valid citation alternatives. Future lookup
  builds store all reviewed grammatical readings separately from the flat
  lookup preference.

## [1.3.14] - 2026-11-05

### Fixed
- Preserve the grammatical lemmas of reviewed `εἶμι` imperfects and compounds,
  including `ἀπῇμεν`, across lookup, verbose, batch, and POS-aware calls.
  Verbose candidates expose the grammatical analysis and the separate
  relationship to `ἔρχομαι` and its compounds, including with older local data.

## [1.3.13] - 2026-11-03

### Added
- Expose optional, source-backed verb-subject evidence in the Python API and
  paradigm exports. Preserve person, thing, and impersonal readings by sense
  and source; leave unmarked uses unknown. Build the table reproducibly from
  both English and Greek Wiktionary editions without changing lookup data.

## [1.3.12] - 2026-11-02

### Changed
- Put installation and common usage first in the README, with detailed API,
  data, evaluation, and build documentation in linked guides. Update corpus
  descriptions, citations, and language-model command examples.

## [1.3.11] - 2026-11-02

### Fixed
- Defer ONNX Runtime and PyTorch availability checks until model inference is
  requested, keeping construction and `guess=False` lookups free of native
  backend imports.
- Preserve marked Ancient Greek article forms when canonical article
  resolution is disabled, including grave-to-acute normalization, without
  confusing `ἡ` with `ἤ`.
- Import repository build helpers through `dilemma_build` so installing PyPA
  `build` does not hide extraction and source-validation modules. Existing
  script paths and packaging commands remain available.

## [1.3.10] - 2026-10-20

### Fixed
- Fill reviewed Modern Greek noun and present-active paradigm gaps, including
  αιτούντος, αιτούντες, εμπίπτουν, συνιστούν, and απορρέουν, without replacing
  existing readings. Record the bounded lookup refresh in a reversible audit.

### Added
- Evaluate cleaned GlossAPI frequencies against pinned development and test
  data, with explicit source caps, paired ranking changes, and separate local
  outputs. Experimental counts do not replace production frequencies.
  The selected mixture remained below the shipped runtime on the test split.

## [1.3.9] - 2026-10-20

### Fixed
- Use an absolute public URL for the README image so it renders on PyPI.
- Audit the replacement OpenGov archive and recent consultations at verified
  revisions, and count Parliament passages once without their JSON duplicate.
- Preserve source balance and rare forms in GlossAPI frequency experiments,
  record actual lookup inputs, and retain resumable, revision-pinned samples.

## [1.3.8] - 2026-10-12

### Fixed
- Require merged lexical-head evidence for exact corpus POS targets; retain
  rejected claims and tied owner votes in a deterministic companion audit.
- Keep noun homographs from supplying unsupported verb stems during fallback
  lemmatization, and honor explicit verb tags beside case variants of a name.
- Preserve an independent exact lookup spelling beside a differently accented
  POS homograph, including `οὖ` beside `οὐ`.
- Count exact spellings separately from ranked words in the Hunspell frequency
  audit summary, including acute and grave variants.
- Preserve attested capitalized names and ethnic adjectives when a flat
  lowercase lemma hides their independently supported source reading.
- Recognize Greek titlecase letters with a written iota in lexical ownership,
  corpus name handling, and runtime case matching. Retain reviewed capitalized
  spellings with exact-case attestation or no surviving lowercase spelling,
  without inventing a proper-name owner.
- Keep structural and corpus respelling checks in force when restoring an
  inherited capitalized spelling with unresolved lexical ownership.
- Retain the directly cited forms `Κορῆ` and `Ὦρα` without deriving additional
  name spellings or treating ambiguous corpus counts as name frequencies.
- Correct malformed corpus name lemmas through a reversible source audit
  shared by pair, lemma-attestation and tagger-data generation.
- Distinguish terminal quotation punctuation from elision using complete
  morphological witnesses in the same source document.
- Preserve the LSJ-listed old Attic dative plural `παῖσι` under `παῖς`,
  independently of mistaken corpus accents.
- Preserve the rare printed spelling `ὤοντο` from two cited passages, with
  matching profile and citation provenance required for the export.
- Reapply supported historical pronoun repairs when regenerating source
  aliases from a reversed audit, preserving independent and intervening owners.
- Require GLAUx exclusion metadata during pair extraction and name-audit
  loading, rejecting missing or changed metadata before accepting a rebuild.
- Reject mismatched citation caches at runtime while retaining the usage
  profile, preventing form IDs from another build from returning wrong loci.

## [1.3.7] - 2026-10-12

### Fixed
- Create the Hunspell output directory before writing audit files on a clean
  checkout, and include merged corpus frequencies in the CI export inputs.
- Download checksum-pinned model files and install ONNX Runtime in CI, with
  an early integrity check before the full dictionary tests.
- Preserve tonal marks when matching explicit contracted-verb references,
  preventing an acute spelling from borrowing a circumflex form's evidence.
- Exclude five monotonic noun lookup aliases from the polytonic export while
  preserving their source-table circumflex forms and the distinct acute dual.
- Preserve attested sparse inflections when a flat self-map or contracted
  verb citation hides an independently defined owner. Keep explicit contracted
  references separate from lexical definitions and require a unique target.
- Preserve independently defined projected readings for other parts of speech
  when a spelling gains an exact corpus POS entry, retaining exact owners for
  their own tags. Require matching lexical POS evidence from fresh definitions
  or the fingerprint-matched merged LSJ source before broadening an exact entry,
  preventing inherited corpus errors such as πείραι → ἀπολάω.
- Match scalar POS selection in batch calls for input case/accent variants,
  self-map citation comparisons, and ordered case-twin fallbacks, including
  absent POS entries. Preserve batched inference for unresolved words.
- Repair historical-only closed-class elisions only from unique current
  source readings and explicit literal-completion cells in fresh same-POS
  definitions. Preserve competing and complete-word readings, and record
  reversible changes with the matching dump and source hashes.
- Keep explicitly tagged articles from returning a noncanonical POS-table
  target in scalar and batch calls, while preserving the existing Ancient
  and Modern article lemmas.
- Retain independently defined corpus POS votes when a lower-priority table
  or normalized spelling would select another lemma despite a matching bare
  lookup. Require a unique highest-weight owner and fresh same-POS dictionary
  evidence for this recovery; preserve tied claims without selecting one.
- Recognize nominal case descriptions with multiple cases and grammatical
  qualifiers as inflections. Explicit spelling references and broken inflection
  templates no longer establish lexical heads; genuine meanings on form pages
  remain eligible. Apply the same checks to the lexical-head inventory.
- Treat explicitly qualified verb-form descriptions as grammatical
  references while preserving separately defined lexical senses.
- Preserve name-reference ambiguity when following typed form chains,
  preventing Athena's oblique forms from acquiring the adjective's lemma.
- Recognize monotonic projections of polytonic articles in the opt-in
  article-resolution mode, including `τών` from `τῶν`.
- Follow unambiguous, explicitly typed form references through intermediate
  noun homographs, so `παρόντος` reaches `πάρειμι` while the separate noun
  `παρών` and its own declension remain intact.
- Resolve ordinary forms of `αὐτός`, `τις`, and `τίς` to their canonical
  pronouns in the opt-in article-resolution mode, preserving the distinction
  between indefinite and interrogative accents and accepting capitals and graves.
- Retain explicit source vowel lengths through verb synthesis, preserving
  forms such as `θῦνε` and `πρᾶξαι` without borrowing another cell's accent.
- Withhold aorists inferred from the distinct future stems of `ἁνδάνω`
  and `ἐνίπτω`, preserving observed aorists and future participles.
- Apply recessive accent to synthesized uncontracted active imperatives
  instead of retaining their citation or future stem's accent.
- Withhold unrestricted LSJ verb expansion when the entry explicitly limits
  its available persons, preserving attested forms of defective verbs.
- Require an accented source form when parsing LSJ principal parts, skipping
  unaccented OCR tokens without inventing their missing accent.
- Retain observed cells for corpus-only Ancient verb lemmas without expanding
  unsupported annotations into inferred citations and conjugations. Synthesis
  requires a surviving Wiktionary owner or an independently attested headword.
- Treat Ancient comparative and superlative descriptions as grammatical
  references when they supply no independent adjective meaning.
- Follow explicit Ancient spelling-reference chains to their unique defined
  head before competing inflection-table mappings, preserving the pitch
  readings of `πίτα` alongside the independent pie homograph. Keep the
  existing scalar part-of-speech choice when it lacks source-entry context.
- Preserve the long-penult circumflex in synthesized first-aorist active
  infinitives and present or future neuter participles, without moving
  persistent accents or changing finite optatives.
  Count a vowel after a complete diphthong as a separate syllable, avoiding
  a circumflex on the antepenult of three-vowel sequences.
- Keep independent Ancient homographs under their own headword when an
  alternative spelling belongs to a different numbered etymology. The
  binding and needing senses of `δέω` no longer acquire the lemma `δήω`.
- Withhold bare nonpast endings when an Ancient aorist template explicitly
  supplies only its indicative stem, preserving indicative-only tables,
  complete competing templates, and independent grammatical readings.
- Withhold ending-only passive aorist readings from incomplete source
  tables during extraction and canonical assembly, preserving complete
  forms, uncertain raw labels, and independently identified voice readings.
- Require matching grammatical evidence when a conjugation table corrects
  a verb cell's accent, so an infinitive cannot change an optative or
  imperative's distinct accent.
  Explicit tense also keeps an aorist participle from deciding a future
  participle's accent; sources that omit tense can still match.
  Tagless form references cannot change a generated cell's accent; an
  infinitive spelling no longer supplies the accent of a default optative.
- Exclude monotonic copies of perfect-optative auxiliaries from other
  verbs' conjugations while retaining independent aorist homographs.
  Apply the shared auxiliary check during Ancient Wiktionary extraction
  so isolated auxiliaries cannot become training examples for the main verb.
  Match the auxiliary's mood so the distinct indicative `ἔστω` survives.
- Count Ancient verb table variants within each dialect when detecting lost
  person and number labels. Valid Attic, Epic, Doric, and Koine readings no
  longer combine into a false damaged cell, and filtering one dialect keeps
  the other source readings.

### Added
- Independently defined Modern adjectives keep their source morphology through
  flat noun or verb homographs. `el_adjective_heads.json` records the exact
  source definitions, adjective morphology, and input hashes; typed ADJ lookup retains these owners
  without changing scalar lookup. The adjective `κρατών` owns `κρατούν`,
  `κρατούσα`, and `κρατούσες`; the noun reference still names `κράτος`.
  Certification requires explicit adjective morphology, so mislabeled noun or
  adverb entries and definition-only headers cannot supply a new ownership exception.
  Newly published ADJ overrides preserve the previous non-adjective fields
  and scalar fallbacks instead of exposing unrelated internal table mappings.
- Canonical Ancient verb citations can recover a missing breathing from a
  unique matching source cell. Explicitly defective verbs retain supplied cells
  while synthesized forms stay within source-supported tense and voice families.
- Ancient English form senses with nested translations retain their source
  readings under the defined parent verb instead of creating a separate head.
- Exclude the reviewed transcription `ἐπεῖ` without rejecting its valid
  verb, conjunction, and locative counterparts or rewriting source text.
- Exclude the independently reviewed fragment `ἐνεί` from dictionary and
  source-alias admission, retaining complete words and the distinct `ἔνει`.
- Retain the directly cited Archimedes spelling `ἀποχρὴ` and its contextual
  acute twin through the reviewed exact-form route, without changing the
  accent boundary for other source citation heads.
- Ancient source aliases preserve existing citation accents instead of
  matching a different headword after removing them. Missing accents may
  still be supplied from a uniquely supported head. Structurally impossible
  citation accents may use the same unique-head correction while retaining
  breathing, subscript, case, and original source metadata; valid accents
  and ambiguous heads remain protected.
- Modern noun form pages retain their exact grammatical references but do
  not authorize full tables whose own-head cells contradict those references.
  Independent homographs, spelling variants, and separate lexical meanings
  retain their tables.
- Determiner and pronoun sense references resolve through explicit same-POS
  chains to positively defined heads before table claims. Ambiguous normalized
  projections retain the previous mapping instead of choosing entry order.
- Uncounted lowercase Ancient coverage (`fr:X`) no longer suppresses a counted
  capital noun or name with an independent Modern definition and two explicit
  declined forms. Literal capital and counted lowercase coverage remain guards.
- Modern reference rows preserve per-sense grammatical tags and distinct cells.
  Independent determiner and pronoun morphology retains its source POS owner
  even when a flat homograph belongs to a verb. `el_closed_class_heads.json`
  supplies exact same-POS source evidence for the integrity check; scalar
  lookup and VERB resolution are unchanged.
- `el_noun_heads.json` records independent noun senses and source hashes for
  morphology owners that differ from the flat lookup. `el_verb_heads.json`,
  built by `build/build_el_verb_heads.py`, records the corresponding finite
  verb evidence. Data integrity checks require positive source evidence for
  every such noun or verb exception; runtime lookup behavior is unchanged.
- Participle references no longer stop at a bare no-gloss verb header. They
  follow explicit source references to a positively defined finite parent,
  retaining unresolved analyses separately when no such parent is supported.
- The polytonic Modern Greek list rates the frequent Modern Greek spellings
  grc has (`θέση`, `πόδια`, `κοίτα`, `ἀγορά`), which it leaves out, with
  lines of their own, `θέση<TAB>fr:C mg:shared`: 6,650 rows, including
  low-frequency alternatives at `R`, by the slice and monotonic counts.
  They add no entries: a reader that
  merges the lists keeps grc's, and reads the field to rank Modern Greek
  readings by it.
- The polytonic Modern Greek word list carries generated spellings for the
  frequent forms its source texts do not attest, of verbs (`ἔρθεις`,
  `ἔρθῃς`, `μιλήσεις`, `πιεῖς`, `σταμάτα`), nouns (`τηλέφωνο`, `κινητό`,
  `αὐτοκίνητο`, `ἐνημέρωση`, `καρδιᾶς`), adjectives (`ἀσθενῆ`, `ταχεῖα`),
  names (`Λονδίνο`, `Ἰούνιο`) and the other parts of speech, written with an
  `mg:generated` field (`mg_polytonic_paradigms.py`). They come from
  Wiktionary's monotonic paradigms, for the forms monotonic text has at
  least 50 times. Left out are forms filed under abbreviations, the
  fragments of tables Wiktionary could not split, Katharevousa cells and
  datives, spellings already polytonic, a row with no cell where another
  places the form, a name with the letters of another word, a name
  Wiktionary does not decline, and a clitic of one syllable; a form several
  words share gets one spelling, reading all their attested forms. The
  breathing follows the word's attested polytonic forms, then the longest
  prefix the grc words agree on. The accent type follows the Ancient rules,
  with vowel length from attested forms of at least 3 tokens, the
  contraction defaults for verbs, and for nouns and adjectives the Ancient
  lengths of the last syllable (a final `-ι` long, a feminine singular `-α`
  long but for `γλῶσσα`, `μοῖρα` and `ταχεῖα`, the neuters in `-μα`) and the
  circumflex of the genitive, of the adverb and of the endings Ancient Greek
  contracted (`μπαμπᾶς`, `γονεῖς`, `ἀληθῆ`). The iota subscript follows the
  word's forms and, for a verb's `-ᾶς` and `-ᾶ`, what the texts mostly
  write. The second and third person singular of the subjunctive is
  generated both ways (`πάρεις` and `πάρῃς`) unless the traditional
  spelling's letters are another word's (`ὑπολογιστή`). A generated
  spelling only fills a monotonic form that neither grc nor the list covers
  (a noun's Katharevousa dative covers none: `κέντρῳ` beside `κέντρο`),
  goes in beside an attested spelling of the same letters only as another
  paradigm cell that monotonic text has at least twice as often (the
  imperative `σταμάτα` beside the present `σταματᾷ`), and competes the same
  way with the other generated spellings of its letters, whatever their
  parts of speech (the noun `ἐνημέρωση` keeps out the subjunctive
  `ἐνημερώσῃ`). A loanword may end in a consonant (`μπάσκετ`), and a name
  is generated only with a mark monotonic writing drops (`Ἰούνιο`) or a
  count that reaches `M` (`Λονδίνο`). So 49,348 spellings join the 16,786
  attested ones. Held out from their own evidence, source forms are
  reconstructed at 98.1% for verbs (90.5% for subjunctives),
  96.5% for nouns and 97.9% for adjectives. The existing manual
  sample retains 723 generated spellings with correct prior judgments;
  one additional sample now uses its contextual grave and 11 are no longer
  generated. With each dev word's author held out, generated forms raise
  coverage from 91.0% to 92.5%.
  `--no-generated-verbs` leaves them out. The `.version` counts `attested`
  and `generated` entries (`entries` is their sum) and records the sha256
  of the paradigms and the frequencies.
- `εὑρῶ`, the Ancient subjunctive of `εὑρίσκω`, is left out of the list and
  marked `mg:avoid` where grc has it: Modern Greek means the currency
  `εὐρώ` by those letters, which the list now generates
  (`MG_LETTERS_OF_ANOTHER_WORD`). The list's only spelling of `λεωφορεῖο`
  was `λεωφορεῖον`; the generated `λεωφορεῖο` now stands beside it.
- The list's `fr:` buckets also read monotonic Modern Greek text: each
  spelling takes the higher of its slice bucket and the bucket of its
  monotonic form's count in `data/mg_freq.txt` (OpenSubtitles, 263M
  tokens) at the rates per token of grc's edges, `C` from 37 per million
  and `M` from 3.7; a generated form has only the latter, and ranks no
  higher than an attested spelling of its letters (a spelling with an
  enclitic's second accent aside). A spelling the slice writes mostly as a
  name with 95% confidence, or a name it writes mostly in lowercase, keeps
  its slice bucket. The slice's counts alone put 189 spellings above `R`;
  now 3,168 attested and 5,224 generated ones are. A spelling whose
  letters grc has under no lowercase spelling while the treebanks write
  them 20 times (`τάκ`, beside the crasis `τἀκ`) is capped at `R` like an
  Ancient word. The `.version` says where the buckets come from
  (`bucket_source`) and counts them for all entries and for each kind.
  Typed as bare letters into a keyboard outside this repository, with each
  author held out, the polytonic Modern Greek dev words come out wrong
  6,249 times instead of 6,663 in its Modern register and 7,240 instead of
  8,024 in its Automatic one; common verb forms typed after `θα` and `να`
  come out as intended 99.1% of the time instead of 53.6%, and imperatives
  typed alone 95.5% instead of 31.2% in the Modern register and 43.8%
  instead of 13.8% in the Automatic one. Held-out Ancient Greek comes out
  the same in its Ancient register, but for one Modern Greek sentence.
- A word list of attested polytonic Modern Greek spellings,
  `build/hunspell/grc_mg_polytonic.dic` (`export_mg_polytonic.py`), for a
  keyboard to merge into the grc dictionary, which is unchanged. grc holds
  Ancient and Medieval Greek only, so a keyboard shipping it alone rewrote
  polytonic Modern Greek into Ancient neighbors (`τώρα` -> `τἄρα`,
  `ἀκόμη` -> `ἀκμὴ`) and rejected `του`, `κι` and `μας`. The 16,786 words
  come from the training sentences of the language model's polytonic Modern
  Greek slice, from documents that are not monotonic or partly so, with at
  least 3 tokens by 2 authors in their own case, well-formed under the grc
  rules adapted to Modern Greek (unaccented weak pronouns and `κι`,
  synizesis), not misspellings of a commoner spelling in the slice (a
  breathing rule, a mark-level rule and the reviewed rejects in
  `data/hunspell_mg_spelling_review.json`), and not accepted by grc; each
  oxytone gets its acute or grave twin. A spelling that is also an Ancient
  or later Greek word by the GLAUx and Diorisis treebanks, or by the
  patristic texts and First1KGreek at a comparable rate (`του`, the Ancient
  enclitic genitive; `μοναχὸς`), is written with `fr:R`, so that it is no
  sign of Modern Greek. The list also marks 187 grc spellings Modern Greek
  writes another way (`ἐκείνῃ`, `μὰς`, `με`, `που`) with `form<TAB>mg:avoid`
  lines: a respelling of a commoner spelling of the same word (the same
  monotonic spelling, and for a breathing, accent-type or dropped-subscript
  difference no treebank evidence of a word of its own) with a share under
  a tenth of that word's tokens at 95% confidence. Another word is never
  marked (`ὅ`, `ἕν`, the reflexive `αὑτοῦ`, `ὄν`); the weak forms `με`,
  `σε`, `που` and `πως` are reviewed exceptions. Replayed as bare-letter
  typing through a keyboard outside this repository that offers the list
  only in Modern Greek text, with each sentence's author held out from the
  list and its marks and the monotonic dev sentences left out, the 36,129
  words of the polytonic Modern Greek dev sentences come out wrong 5,248
  times in its Modern register without the marks and 4,965 times with them,
  and 474,220 held-out Ancient Greek words come out the same with the list
  as without it.
  `eval/eval_mg_polytonic.py` measures the list's coverage with documents
  or authors held out, and writes the evaluation lists.
- `grc_morph.json` carries an optional `el_modern` table: the Modern Greek
  elisions the same slice attests (`γιὰ` -> `γι᾽`, `στὸ` -> `στ᾽`,
  `τώρα` -> `τώρ᾽`, `κι` -> `κι᾽`, 39 pairs), limited to those `el` lacks or
  spells otherwise, for a keyboard writing Modern Greek to lay over `el`, and
  `el_modern_share`, how often the texts elide the words each elided
  spelling stands for before a vowel (`γιὰ` 0.65, `τὸ` 0.20, `τώρα` 0.06).
  `el` and `nu` are unchanged;
  `export_morphology.py --no-modern-greek` leaves both tables out.
- The next-word language model (`grc_ngram.bin`) carries the training count
  of spellings outside its 80,000-word vocabulary. When every spelling of
  the letters a user typed was outside the vocabulary, the model scored them
  all alike and a keyboard fell back to its dictionary's order; now a rarer
  spelling loses to a commoner one (`κυλικεῖον` over `κυλίκειον`,
  `περιτομῇ` over `περιτομὴ`). The table holds 125,357 dictionary spellings
  that have a same-letter twin, at 9 bytes each (1.1 MB), after the existing
  sections and behind a 16-byte footer, so a reader that does not know it
  reads the file as before. `train_lm.py` writes the counts
  (`type_counts.tsv.gz`, training split only); `export_lm.py` chooses the
  entries against the grc Hunspell dictionary, records its sha256 in
  `grc_ngram.version`, and `export_lm.py --check-dictionary` and
  `export_hunspell.py` report when the dictionary has been rebuilt since, as
  the LM must then be re-exported. On 26,906 held-out Ancient Greek
  sentences typed as bare letters, a keyboard that reads the table restores
  a further 2,317 words correctly (2,741 fixed, 424 broken), and 4.5 more
  per 1,000 words on the sentences with little overlap with training.

### Fixed
- Ancient Greek dialect, contracted-form and voice-only descriptions no longer count as
  independent definitions merely because their entries are labeled as verbs.
  Their source inflection tables resolve to the supported lexical parent.
  Spelling references also recognize explicit word labels and a colon before
  the target, while retaining any appended lexical meaning.
- `data/mg_pairs.json` has the paradigm of `γιος` again, and the words
  whose Greek Wiktionary page only gives their polytonic spelling are no
  longer filed under it (#28). Such a page (`γιός`, "πολυτονική γραφή του
  γιος"; `μιλῶ`, `ἀγχωτικός`, `μουστακᾶτος`) made itself a lemma under its
  word's monotonic and accentless keys, so `γιος` became a form of `γιός`,
  a cycle that the chain-breaking deleted with the whole paradigm, and
  `μιλώ`'s and `αγχωτικός`'s forms went under `μιλῶ` and `ἀγχωτικός`. In
  the Modern Greek dumps a page whose every sense names its own monotonic
  spelling (for a monosyllable, also without its accent), by an `alt_of`
  or `form_of` reference or by that gloss, now maps to that word and makes
  no lemma of its own; a `form_of` reference of a sense tagged with a case
  (`σοφίᾳ`, "dative singular of σοφία") is read as before. Two pages that
  name each other (`αγχωτικός` and `αγχώδης`, each "συνώνυμο του" the
  other, which kaikki reads as `alt_of`; `δεσποινίς` and `δεσποινίδα`) are
  no longer followed into a cycle that deleted both paradigms, and the weak
  pronouns on `μόνος`'s headword line (`μόνος μου`, `μας`, `σας`) are no
  forms of it, so `μας` and `σας` are forms of `εγώ` and `εσύ` again. An
  abbreviation's reference no longer moves the key of a lowercase word
  with a sense of its own: once `δεσποινίς` kept its paradigm, the English
  Wiktionary's abbreviation pages `δις` and `δ` would have made the adverb
  `δις` ("twice", "billion") and the letter `δ` its forms, and `α`, `β`,
  `ν` and `κ` were forms of `άντε`, `βόρειος`, `νότος` and `κυρία`; `κα`
  still goes to `κυρία` and the contraction `στο` to `σε`. The other
  abbreviations such a page lists as alternatives are forms of the word it
  abbreviates (`δ`'s `δις`, of `δεσποινίς`), not of the page's headword.
  Reciprocal synonym references are checked together before the merged
  lookup is changed, so `νησοπυριτικός` and `ορθοπυριτικός` each keep their
  paradigm. `lookup.db` takes the corrected data up when it is rebuilt.
- `data/mg_pairs.json` has the conjugation of `ρωτάω` and the other verbs
  whose forms went under a capitalized page. In the Modern Greek dumps a
  capitalized page (the female name `Ερωτώ`, the abbreviation `ΔΕΣ`, a
  heading in sentence case) took the lowercase key of a word whose own page
  is a form-of page (`ερωτώ`, `δες`), so that word's pairs went under the
  page; it now reaches its lowercase keys only where nothing lowercase
  claims them (`build_data.py`). And a form-of reference of one Wiktionary
  that pointed a lemma back at a word the other files under it (EN's
  `ρωτώ`, a form of `ρωτάω`; EL's `ρωτώ` table, which lists `ρωτάω`) made a
  cycle that the chain-breaking deleted with both conjugations; it is no
  longer followed. This priority also applies across dictionaries, where
  the surnames `Λεν` and `Τρων` previously overrode `λεν` and `τρων` from
  the other source. Form pages resolve verb rows through their explicit
  reference, preserving `ερωτήσεις` under `ερωτώ` and the participle forms
  of `αντιπαθών` under `αντιπαθώ`, instead of a noun or adjective homograph.
  POS-specific resolution also follows intermediate spelling pages;
  source-mislabeled verb rows whose targets are only nominal are omitted.
  Existing noun, adjective and ethnonym paradigms are preserved, and a
  missing reference can use another source's attested verb.
  Inflection senses whose gloss names their explicit reference preserve
  that verb's forms despite a mislabeled head page (`δηώ`). Descriptive
  glosses do not redirect passive heads or create new verb lemmas, and
  broken references leave independently attested head forms in place.
  A participle's top-level reference is used only when the target has an
  explicit reference chain to an independent lexical verb entry, after
  both dictionaries have supplied their POS. An existing sense-level
  reference chain to such a verb takes priority over top-level inflection
  or spelling pointers, with inflection-table evidence used only as a fallback.
  All explicit targets from both dictionaries survive collection; a tagged
  participle uses its own entry's references before those of a finite-form
  homograph, and synonym labels do not count as spelling aliases. Conflicting
  successful targets remain unresolved. Participles tagged on entries or
  senses remain intermediate forms; cycles resolve only when their exits
  identify one finite verb. Standalone participles that would otherwise
  remain verb lemmas enter the same reference check, even without a
  top-level pointer, rather than supplying unsupported verb roots.
  A resolved participle retains that verb through POS-specific cleanup;
  unresolved readings are omitted without removing independent nominal
  readings. A table whose own participle head has a supported verb keeps
  that endpoint even when a spelling alias names an unresolved participle,
  preserving `καθούμενα` under `κάθομαι`.
  Nominal labels cannot supply verb endpoints through unrelated tables,
  and an unresolved participle cannot remove its independently defined
  finite verb through a reverse table or lookup alias.
  Separately, `el_participles.json` preserves explicitly recorded participle
  headwords and nominal table cells whose finite parent remains unresolved.
  Exporters can retain that morphology without adding unsupported verb
  roots or inventing adjective readings. The versioned schema keeps exact
  source spellings, distinct grammatical cells, and all source editions;
  headwords without a table add no inferred inflections.
  The pairs are regenerated from the 2026-10-03 Kaikki dumps; `lookup.db`
  takes the fix up when it is next rebuilt. A pronoun table that spans
  several persons, the shared personal-pronoun grid, is skipped whatever its
  template's name: those dumps carry it on EN's `τα` page under a generic
  template, which made `εγώ`, `εσύ`, `αυτός` and `μου` forms of `τα`.
- Modern Greek pair deduplication preserves each part of speech and
  grammatical tag set (#31). The genitive, accusative and vocative rows of
  `καθηγητή` survive separately, allowing the exporter to produce the
  circumflex genitive and acute accusative and vocative. Exact duplicate
  rows are still removed; the JSON fields are unchanged.
- An independently defined Modern Greek noun's table keeps its source
  lemma when the same headword is also another noun's inflected form.
  The masculine `σκιάς` and feminine `σκιά` retain separate cells instead
  of pooling the masculine genitive under the feminine noun. Explicit
  form-of and alternative-spelling entries still follow their references;
  flat lookup arbitration and verb resolution are unchanged.
- Distinct noun case-and-number cells select their primary owner by
  whole-paradigm frequency, while nouns supplying the same cells retain
  the closest lexical stem. Independent noun heads remain eligible when
  an inflected form also has an adjective or adverb reading. Alternative
  explicit cells retain a rare spelling without borrowing the primary
  homonym's count (`σκατά` and masculine `σκατᾶ`). Diminutives and synonyms
  in another noun's table cannot acquire that noun's declension.
  A separate adjective or adverb head sharing the spelling no longer
  disables arbitration between independently owned noun cells; the
  neuter plural `σκατά` retains its counts and masculine `σκατᾶ` remains
  a distinct rare reading with zero pooled frequency.
- In the polytonic Modern Greek list, two paradigm cells of the same
  letters both go in, whatever their counts: the present `περπατᾷ`,
  `προχωρᾷ` and `τηλεφωνᾷ` are back beside the imperatives `περπάτα`,
  `προχώρα` and `τηλεφώνα`. Within a cell (the same form of the same word;
  a noun's or adjective's case and number), a generated spelling gives way
  to the list's spelling and to a grc spelling monotonic text writes at
  least as often, which grc's spellings did not take part in before
  (`φάκελους` to `φακέλους`, `ὑπάλληλους` to `ὑπαλλήλους`, `χειροῦργος` to
  `χειρουργός`); a spelling of no cell stops it at twice the count, and a
  subjunctive twin gives way to any spelling of its letters at twice its
  own (`ἐνημερώσῃ` to `ἐνημέρωση`).
- A name whose only other spelling is a dative is generated: grc's
  `Ῥωσίᾳ` and `ῥωσίᾳ` no longer keep out `Ῥωσία`, and a name with two
  frequent declined forms is generated though rare (`Δανία`, `Δανίας`). A
  nominative singular in `-α` and a genitive singular in `-ας` are a
  feminine's whatever lemma Wiktionary files them under (`σκούπα`, not
  `σκοῦπα`, under `σκουπόχορτο`). A capitalized form gives way where its
  lowercase spelling does (`Ἅγιους` to grc's `ἁγίους`, `Αὔγουστου` to
  `Αὐγούστου`).
- A spelling in `-ῃ` or `-ῃς` is rated by its own use, not by all of the
  `-ει` or `-εις` form it stands for, which monotonic text also writes for
  the indicative, the perfect and the dative: it takes the share of that
  count which the slice writes with the subscript (`θέσῃ`, half of
  `θέσει`'s 2,110 tokens, is `M`, below `θέση`'s `C`).
- The color adjectives in `-ής`, `-ιά`, `-ί` (`λεμονής`, `κερασής`) are no
  s-stems: their feminine plural `-ιές` was taken for an s-stem's neuter
  `-ές`, and the noun `λεμονιά`, filed under the adjective, was generated
  `λεμονιᾶ`.
- `lookup.db` has the relative pronoun's graves `ὃ`, `ἣ`, `οἳ` and `αἳ`,
  under `ὅς`. `build_lookup_db.py` keeps the first lemma a source gives a
  form, and AGDT, the first treebank it reads, files Homer's demonstrative
  `ὃ`, `ἣ`, `οἳ` and `αἳ` under the article `ὁ`, which the article filter
  then drops; so GLAUx's relative `ὅς` never got in, and the forms had no
  row at all. The article is a proclitic: it has no accent of its own and
  before an enclitic takes an acute (`ὅ γε`), never a grave. A grave spelling
  of its vowel-only letters that an earlier source filed under the article
  now gives way to a later source's reading; every other article spelling,
  the dual `τὼ` among them, stays with the article and out of the lookup.
  The rebuilt `lookup.db` gains exactly those four rows and `ὂ`, a
  misspelling of `ὃ`, also under `ὅς`. Lemmatization is unchanged on the
  fast benchmark (AG Classical 99.7%, Katharevousa 94.7%, Demotic 94.8%), and
  so are the grc dictionary, the morphology table and the polytonic Modern
  Greek list, which had the forms from the exporter's closed list.
  `data/citation_hygiene_rejections.tsv` loses 22 editorial-siglum rows that
  `glaux_pairs.json` has not carried since its last rebuild.
- The grc dictionary has the crasis of `καί` written with an iota subscript,
  as some editions print it: `κᾂν`, `κᾄν`, `κᾀπί`, `κᾀπειδάν`, `κᾄπειτ᾽`,
  `κᾄπειθ᾽`, `κᾀνταῦθ᾽`, `κᾀκτός`, `κᾀκεῖσε`, eight forms of `κᾀκεῖνος`,
  and the unaccented `κᾀκ` and `κᾀξ` beside `κἀκ` and `κἀξ` (22 words with
  the contextual graves). These are every such spelling GLAUx or Diorisis
  has at least 4 times that the lookup does not bring in; `κᾂν` has 1,005
  GLAUx tokens, `κᾀπὶ` 356, `κᾀκ` 277. `lookup.db` has no row for them, and
  `κᾂν` and `κᾄν` were only in the polytonic Modern Greek list, so Ancient
  text could not get them; that list now marks them as spellings Modern
  Greek writes otherwise. No held-out sentence has one, and replayed through
  a keyboard outside this repository every held-out and running-text word
  comes out as before.
- The grc dictionary takes the spellings the treebanks attest that no lemma
  source proposes, or that only the Modern Greek lexicon does. A spelling
  GLAUx or Diorisis has at least 5 times, case kept and a contextual grave
  counted with its acute, comes in under itself; a `lookup.db` form whose
  lemma has only Modern Greek rows comes in under that lemma (`σπανίως`, 521
  corpus tokens, and `κτήσεις`, 219, were dropped because the exporter took
  only lemmas with an Ancient row). The counts leave out every occurrence in
  the language model's held-out GLAUx and Diorisis sentences, whose spelling
  counts `data/hunspell_grc_lm_dev_spellings.json.gz` records
  (`scripts/build_hunspell_dev_spellings_fixture.py`), so the dictionary is
  not vouched for by the sentences it is measured on. Elided spellings, inner
  capitals, an iota adscript, a length mark, a breathing after a consonant
  (`δὔ`, an elision mark converted into a breathing), a second accent
  (`τάγαθόν`) and the grave of an explicitly rejected spelling (`τοὺ`) stay
  out, a capitalized spelling comes in only when its lowercase is not a word,
  and the respelling rule applies as to every new form. Below 5 tokens most
  candidates are Modern Greek words of late texts and names written in
  lowercase. The dictionary gains 178 words, mostly names and rare words
  (`Δηριάδη`, `ὁρμόν`, `καταλήμψεται`). Replayed as bare-letter typing
  through a keyboard outside this repository, the 474,220 words of the
  held-out GLAUx and Diorisis sentences come out right 30 more times and
  wrong 3 more (11 and 2 of them in the 137,525 words with little overlap
  with training), with or without their punctuation typed; the running-text
  passages gain 6 words in either register, and polytonic Modern Greek, with
  each author held out, 4 in the Automatic register.
  `--treebank-spelling-min-count 0` turns both rules off.
- `scripts/audit_hunspell_frequency.py` checks the corpus head's graves as
  spelled. It read a contextual grave as its acute, but a keyboard writes the
  grave only where the dictionary has the grave spelling, so a missing `ἣ`
  passed while `ἥ` was there. The fixture now ranks the language model's
  1,000 most frequent words as before and lists each in every spelling the
  model writes it with (1,215 spellings; `καί` and `καὶ` apart), each looked
  up as written. Its exclusions gain the Modern Greek graves `τὴ` and `στὴν`,
  which the polytonic Modern Greek list carries, and four corpus misspellings
  (`ὃτι`, `ἢδη`, `ἣκιστα`, `τοῦτὸ`); `στό` and `στά`, which the model writes
  only with the grave, and `δῑ`, gone from the model since its retraining,
  leave them. The fixture is now read from the training run's JSON files.
  The exporter pins the same head words, except `στήν`, which the model
  writes only as the Modern Greek `στὴν`, and `ἐνδέχεται`, which replaces
  `δῑ` among the 1,000; `στήν` stays in the dictionary, in bucket M rather
  than C.
- The grc Hunspell dictionary has the relative pronoun's graves `ἣ`, `οἳ` and
  `αἳ` and the dual article's `τὼ`, which the next-word language model's
  training text uses 4,144, 9,854, 1,978 and 1,363 times. `lookup.db` has none
  of them: `build_lookup_db.py` leaves out every form it maps to the article
  `ὁ` whose letters are an article's, and AGDT, the first treebank it reads,
  files Homer's demonstrative `ἣ`, `οἳ`, `αἳ` and `ὃ` under `ὁ`; only the
  first lemma of a form is kept, so GLAUx's relative `ὅς` never gets in. The
  exporter's closed list, which puts the article's spellings back, listed `αἵ`
  and `οἵ` (the acute, and under `ὁ`) where the graves belong, and lacked
  `τὼ`. It now carries `ὃ`, `ἣ`, `οἳ` and `αἳ` under `ὅς` and `τὼ` under `ὁ`;
  `ὃ` was in the dictionary only through the reviewed April surface. A
  keyboard that writes a grave only where the dictionary has the grave
  spelling wrote the article `οἱ`, `αἱ`, `ἡ` or the conjunction `ἢ` for them.
- The movable-nu list no longer gives a nu to imperatives the taggers file as
  unaugmented pasts. An unaugmented imperfect or thematic aorist in -ε is
  spelled like the 2sg imperative (`κατένεγκε`, `ἐντύγχανε`, Galen's
  `προσέμβαλλε`), and GLAUx writes the nu on 99.6% of these pasts before a
  vowel, so such a spelling stays only when the texts write it with its nu
  somewhere. The augment is read after the verb's preverbs, from an accent it
  holds on the penult (`συνεξεῦρε`, Smyth 426), or from the augmented
  spelling filed under the same or the simple verb; liquid first aorists
  (`σήμηνε`), σ- and κ-aorists and iteratives are not imperative-shaped, and
  `ἄντε`, Plato's `ἄν τε` filed as a verb, leaves the list. The adverbs of
  place in -θε (`πρόσθε`, `ὄπισθε`) follow the written-nu rule of the adverbs
  in -σι (Smyth 134 D), and the aorist optative in -ειε takes the nu like any
  third-person singular in -ε (`δόξειεν`, 1,942 of 1,950 GLAUx tokens before
  a vowel). 26,277 forms become 27,021: 876 optatives and 57 adverbs come in,
  189 pasts go.
- The elision table no longer elides datives that no tagged corpus analyzes.
  A form in -ι known only from the lookup table's paradigm expansions
  (`δμητῆρι`, `λυθεῖσι`, `παντι`) is read through the forms of its lemmas: a
  genitive in -ος beside it, the uncontracted -εϊ of an s-stem, or a
  third-declension stem that yields the -σι makes it a dative. A present
  participle's dative plural is also the verb's third person plural
  (`κεύθουσι`, `τιμῶσι`) and keeps its elision. Diorisis's nominative-dual
  analysis of the uncontracted datives in -εϊ (`κήδεϊ`) no longer vetoes the
  dative. 15,245 entries become 14,905; none is added or changed.
- The grc dictionary no longer accepts 280 spellings a recorded review
  rejected (`data/hunspell_grc_spelling_review.json`): 191 respellings of
  common words that OCR'd text admitted (`στρατηγού` for `στρατηγοῦ`,
  `βοής`, `ἔκαστος`, `ἐβδομήκοντα`), which the respelling guard's calibrated
  thresholds let through, and 89 reviewed-baseline entries that fail the
  structural rules and are not typed words (underdotted letters, synizesis
  ties, `δόκιμάζω`). Two reviewers labelled each candidate independently from
  its attestation and corpus sentences, and a form is listed only where both
  rejected it; the exporter drops it with its grave or acute twin and the
  audit no longer requires it. It accepts `θὰ` and `γιὰ`, polytonic Modern
  Greek particles no lemma source proposes, while their acute `θά` and `γιά`,
  the monotonic misspelling, stay rejected. 1,363,527 entries become
  1,363,249. The held-out corpus fixture is regenerated for the rules changed
  since it was built: `θά` leaves its coverage, and `τεο`, `τεῳ`, `τῇσιδε`
  and `ᾧχετο` join it.
- The movable-nu list follows what the texts mean by a spelling, not the
  first analysis a pairs file met. `φέρε` is an imperative 956 times in GLAUx
  and an unaugmented imperfect 45, and was on the list, so a keyboard wrote
  `φέρεν εἰπέ`. The GLAUx pairs now carry each analysis's token count, and the
  readings that take nu, with the tokens of the spelling that already has it
  (`μέλλεν`), are weighed against those that cannot; a spelling in -σι, which
  can only be a dative plural or a third person, ignores analyses that say
  otherwise. 162 forms go, 420 GLAUx tokens before a vowel: 142 imperatives
  (`ἄκουε`, `ἴδε`, `λάβε`), vocatives (`Κέλσε`, `βάσκανε`), `πέρατι`, which a
  Diorisis candidate analysis filed as a verb, word pairs a tokenizer fused
  (`βαδιεῖτι`), and the Doric 3sg in -τι (`δίδωτι`, `ἐντί`), since only
  `ἐστί` and its compounds take nu there.
  217 come in: the article's and the numerals' dative plurals (`τῇσι`, 784
  GLAUx tokens, `ταῖσι`, `δεκατέσσαρσι`), the epic third singular
  subjunctive (`ἐθέλῃσι`), adverbs in -σι the texts write with a nu
  (`παντάπασι`, `πέρυσι`, `Ὀλυμπίασι`), and imperfects whose first token was
  tagged an imperative (`πέλε`, 78 tokens to 1). The list ends at 26,277. The
  export stops if the GLAUx pairs file has no counts.
- Four orthography rules misread valid Greek and dropped 18 attested forms
  from the grc dictionary, which the keyboard's own gate had listed as words
  it must accept. In a Latin name `ου` between two vowels is the consonant v,
  not a syllable, so `Ὀκτάουιος` is accented on its antepenult. A fused
  `-περ`, and the `-δε` of the epic and Ionic datives of `ὅδε` (`τῇσιδε`),
  leave the host's accent where it was (`οἷονπερ`). Crasis with `ὦ` keeps the
  interjection's circumflex (`ὦλεθρε`), and crasis inside `ταὧς` carries its
  breathing into its forms (`ταὧνι`). The Ionic enclitics `τεο`, `τεῳ` and
  `τεων` are written without an accent. The rules now say so, and 46 entries
  come back, among them `Φλάουιος`, `Λίουιος`, `ἐγᾦδ᾽` and `ὦνθρωπ᾽`; none
  goes.

### Changed
- The language model completes the bigram row of every homograph spelling.
  To choose between `ἢ` and `ἡ`, or `ὃ` and `ὁ`, a keyboard scores the word
  typed next after each reading, but a row kept only its 30 most frequent
  continuations, so the next word was almost never in either row and both
  readings fell back to its unigram probability: the choice was made
  without evidence, and the commoner spelling won. Past the cut, the row of
  a homograph spelling now also keeps every continuation seen there at
  least 3 times (`--homograph-context-min-count`, 0 to turn it off): 266,694
  continuations in 1,491 rows, 1.6 MB, the file growing from 60.3 to 61.9
  MB. Trigram rows and the rows' first 30 entries, with them the next-word
  lists, are unchanged. Replayed as bare-letter typing through a keyboard
  outside this repository, the 474,220 words of the held-out GLAUx and
  Diorisis sentences come out right 259 more times and wrong 23 more, 104
  and 2 of them in the 137,525 words with little overlap with training;
  typed with their punctuation, 265 and 20 (116 and 1). Every running-text
  corpus gains in both registers (+51 words in the Ancient register, +76 in
  the Automatic), and so does polytonic Modern Greek with each author held
  out (+31 Automatic, +62 Modern).
- The language model keeps both spellings of a homograph in each context.
  A context kept its 30 (bigram) or 15 (trigram) most frequent next words,
  so the rarer of `ἢ` and `ἡ`, or of `ἐκείνῃ` and `ἐκείνη` after `τῇ`, fell
  out of the row and was scored by backoff against its twin's full
  probability. A spelling whose bare letters another vocabulary word shares
  now stays in the row when it was seen there at least 3 times. Next-word
  lists are unchanged (the rows' first entries are the same); the file grows
  from 56.9 to 59.2 MB before the count table. On the same held-out
  sentences a keyboard restores 1,423 more words correctly (1,972 fixed, 549
  broken), 3.2 more per 1,000 words on the low-overlap sentences.
- The compatibility baseline is rebased onto the dictionary Tonos ships now,
  the export of `2e633cc`: 1,363,275 reviewed forms that every export must
  keep.

### Fixed
- The dative -ι and -σι no longer elide. They elide only in epic (Smyth 72),
  and rarely there: GLAUx elides them in about 2% of epic tokens before a
  vowel and 0.2% of the rest, and the elided spelling it does attest is
  nearly always another case's (`ἄνδρ᾽` is `ἄνδρα`, `πάντ᾽` is `πάντα`), so
  a keyboard applying the table turned `παντί` into the accusative. The
  tagged corpora say which forms are datives; the first file that has a form
  decides, since they disagree on some. `τῷδε` and `ἔμοιγε` keep their
  entries, because what they lose is the ε of δε and γε. A capitalized
  homograph's analyses do not decide a lowercase word (GLAUx tags a `Χερσὶ`
  with no case, which would have kept `χερσί` eliding). 1,739 entries go,
  196,008 corpus occurrences of those spellings, and the table ends at
  15,242.
- A participle's dative plural in -σι takes movable nu like any other
  dative plural (`οὖσιν`, `ἔχουσιν`), and so does the subjunctive's third
  person plural (`ὦσιν`, Smyth 134); the movable-nu list had excluded every
  participle and subjunctive. A spelling the orthography rules reject gets
  no nu, so `ἒστι` and `λέγουσὶ` no longer take one, while a prodelided
  form (`’στι`, and the bare `στι` a keyboard reads after an apostrophe)
  keeps it. The list grows from 21,722 forms to 26,221.
- Long final vowels that the spelling does not show no longer elide: the
  Attic accusative of a noun in -εύς (`βασιλέα`, whose ᾱ is long; Smyth 276),
  the contracted neuter plurals `κρέα`, `κέρα` and `γέρα`, and the rest of the
  deictic -ί paradigm (`τονδί`, `τοιονδί`, `τοιαδί`). Neither does a
  monosyllable that does not end in ε (Smyth 72), so `σά` and the fragments
  `λα`, `πί` and `στι` go; `ῥα`, `κα` and `γα` stay. An accented ε or
  ο before a final ι is a hiatus rather than a diphthong, so the epic
  `βασιλέι` is read as `βασιλέϊ` (though as a dative it no longer elides;
  see above), and a circumflex before the penult hosts no
  enclitic's accent, so the crasis `ὦγαθέ` retracts to `ὦγάθ᾽`. More
  crasis forms and dialect words elide bare (`τἀπί`, `μἀλλά`, `κἀντί`,
  `κοὐδέ`, `σφωέ`, `ὑπά`). That leaves 16,981 entries.
- Words Attic never elides no longer have an elision entry, although their
  final vowel is short: `ὅτι`, `περί`, `πρό`, `ἄχρι` and `μέχρι` (Smyth 72),
  `διό` and `καθά`, which already contain one (`δι᾽ ὅ`, `καθ᾽ ἅ`), and the
  emphatic and deictic -ί, which is long (`οὐχί`, `ὁδί`, `τοδί`). A keyboard
  applying the table rewrote them before any vowel, and `ὅτ᾽` reads as `ὅτε`,
  `δι᾽` as `διά`. GLAUx elides `ὅτι` 5 times in 20,231 before a vowel and
  `οὐχί` never in 462. 32 entries go, whose 14 distinct corpus spellings
  carry 476,588 occurrences.
- The elision table now emits a pair only when the corpora attest the
  spelling the rules of elision give: the final vowel goes, an oxytone throws
  its accent back as an acute unless it is a preposition, conjunction or
  enclitic, and every other mark stays. The corpus candidates used to be
  ranked instead, and the ranking only broke ties among spellings that were
  often all wrong, so a key whose one candidate belonged to another word
  still got it. 32,333 entries fall to 17,030. 13,156 of the keys dropped are
  spellings the orthography rules reject (a grave before the last syllable, a
  missing breathing), whose values were correcting the spelling as well as
  eliding it. Among the 2,147 well-formed ones, a long final α took the
  neuter plural's elision (`αἰτία` -> `αἴτι᾽`, `ἑτέρα` -> `ἕτερ᾽`; 266 keys,
  39,259 occurrences), an oxytone in -ι took another case's accent
  (`γυναικί` -> `γυναῖκ᾽`, the accusative's; 105 keys, 35,291), and the
  deictic -ί, which is long, was elided into the plain pronoun (`τουτί` ->
  `τοῦτ᾽`). The epic `ἐνί` now elides to `ἐν᾽` rather than the numeral's
  `ἕν᾽`, and the dialect prepositions and particles that elide bare are
  listed (`προτί`, `κοτέ`, `ποκά`, `πεδά`, `κἀπί`). `τοτέ`, "at times", is
  an accented adverb rather than an enclitic, so it retracts to `τότ᾽`
  instead of eliding bare.
- The elision table gave `κατὰ` the value `Κατ᾽` and `μετὰ` the value
  `Μετ᾽`, and a keyboard writes the value into the user's text as it stands,
  so the two commonest mid-sentence spellings of those prepositions came out
  capitalized: 217,784 and 88,664 corpus occurrences of those spellings. A form's elision was
  assigned lemma by lemma, and the last lemma to claim the form won whatever
  the ranking said. The corpora carry a capitalized lemma `Κατά` whose one
  elided token opened a sentence, and it came after `κατά`. A form's
  candidates are now pooled across every lemma that claims it and ranked
  once, and each candidate first takes the full form's case, so a lowercase
  `ἑλλάδα` no longer gets `Ἑλλάδ᾽` either. 1,683 entries change case and
  nothing else; 763 of them went from a capital to lowercase.
- An enclitic's accent on the last syllable was read as the word's own, so
  the retraction rule moved the real one: `χεῖρά` took `χείρ᾽`, `εἶπέ` took
  `εἴπ᾽`, and `ἄλλὰ` was taken for the conjunction and lost its accent. The
  acute an enclitic throws onto a proparoxytone or properispomenon leaves with
  the elided vowel, and the word keeps its own accent where it was. 32 entries
  carrying 5,663 occurrences change.
- Three defects in the elision table, all reported by Tonos, which reads it to
  rewrite the user's text and diffs it by hand because its dictionary gate
  never sees the file.
  - A grave oxytone was not retracting. The grave is only the contextual
    spelling of an oxytone, and it is the spelling a word carries
    mid-sentence, which is exactly where elision happens, so `αὐτὸ` took
    `αὐτ᾽` where `αὐτή` correctly took `αὔτ᾽`. 59 entries by Tonos's count,
    carrying 119,486 corpus occurrences: `αὐτὸ`, `πολλὰ`, `αὐτὰ`, `ἐμὲ`.
  - A word elision cannot touch was still given an entry. Elision removes a
    short final vowel, so η and ω are out, a circumflex or an iota subscript
    marks a long vowel, and the second element of a diphthong goes with the
    first. That is 10,957 keys, including `αὐτῷ`, `αὐτῇ`, `δεῖ` (which was
    paired with `δέ᾽`) and `ἤδη`.
  - A pair whose every candidate is junk was emitting the least bad one, so
    `ὅσδε` gave `ὃσδ᾽`, a grave on an elided word. 30 entries, now dropped
    rather than written into someone's text.
  The known-bare class is also stated rather than left to the absence of an
  accented candidate: prepositions, conjunctions and, newly, enclitics, which
  have no accent of their own to throw back (Smyth 183). That separates the
  enclitic `ποτέ` from the interrogative `πότε`. The orthotone `εἰμί` and
  `φημί` are not enclitics here, so `εἰμί` keeps `εἴμ᾽`.
- `tests/test_export_morphology.py` now checks the built table against those
  three invariants, pinning the oxytone one by corpus weight rather than entry
  count, because weight is what separates a rule that stopped firing from the
  tail of the known-bare class.

### Fixed
- An elided oxytone throws its accent back onto the penult as an acute
  (Smyth 174), and the elision table was leaving it bare. The ranking added
  in the previous entry prefers the candidate that keeps the full form's own
  stem marks, and an oxytone's stem carries no mark, so `ἀνδρί` took `ἀνδρ᾽`
  over `ἄνδρ᾽`, `αὐτή` took `αὐτ᾽` over `αὔτ᾽`, and `Διί` took `Δι᾽`, which
  also collides with `δι᾽` for `διά`. The retraction now ranks above the
  stem-marks test, and applies only to oxytones, so every other form still
  keeps its marks where they were. Prepositions and conjunctions lose the
  accent outright instead, which is the other half of the same rule: beyond
  the ten pinned particles that covers `ἀνά`, `ἀμφί`, `περί`, `οὐδέ`, `μηδέ`
  and `τε`, and it corrects `οὐδέ` from `οὔδ᾽` to `οὐδ᾽`.

  Reported by Tonos, which held the table back from its second swap-in rather
  than ship it. Its gate reads only the dictionary, so nothing on either side
  tests this file automatically.

### Fixed
- The elision table in `build/hunspell/grc_morph.json`, which the keyboard
  reads to offer an elided spelling, picked between candidates on casing,
  breathing and whether an accent survived, and broke ties on set iteration
  order. Python randomizes that per process, so the table was not
  reproducible: two builds of the same inputs differed in 316 entries.
  Candidates are now ranked on whether the orthography rules accept the
  spelling at all, then on whether it keeps the full form's own stem marks,
  then the existing profile, then the corpus count, then the spelling itself.
  606 entries change against the table Tonos ships: `αὔτ᾽` becomes `αὐτ᾽`,
  `γέλοι᾽` becomes `γελοῖ᾽`, `διδῶσ᾽` becomes `δίδωσ᾽`, and the structurally
  invalid spellings among them fall from 14 to none. `εἶπε` keeps `εἶπ᾽` and
  `εἰπέ` keeps `εἴπ᾽`, which a corpus count alone gets backwards, because the
  two elided spellings belong to different full forms rather than competing
  for one.

### Changed
- The compatibility baseline is now the dictionary Tonos currently ships,
  rather than the April 0.4.1 export it started from, and
  `data/hunspell_grc_april_compat.json.gz` is renamed
  `hunspell_grc_shipped_compat.json.gz` to stop claiming otherwise. It moves
  forward with each swap-in; the contract it encodes is unchanged, that an
  export must not lose what the keyboard already accepts. Because the gate
  strips its structural classes before pinning, the new baseline grandfathers
  far less junk than the old one: 18 truncated stems against 8,971, and no
  missing-breathing forms at all against 166,226. 1,363,274 required forms.
- A respelling both treebanks annotate independently no longer has to clear
  the raw-count floor. GLAUx and Diorisis lemmatize and tag separately, so
  both landing on one spelling more than once is evidence that counting OCR
  tokens cannot give: the dual `πρώτω` has 40 corpus tokens against 4,006 for
  the dative `πρώτῳ`, which asks for 25 treebank tokens where it has 9. Doric
  `γλώσσᾳ` and the contract `πειρᾷς` were failing the same way, as were Doric
  `δεσπότᾳ` and epic `γένεϊ`.

  The confirmation is bounded the same way the floor is, because treebanks
  mis-tag a common word in proportion to how common it is: it counts only
  when the treebank support is at least one token per thousand of the
  dominant spelling. Without that bound the treebanks' own stray taggings of
  `ὅτι`, `εἶναι` and `τοῦτο` - two or three tokens against 241,498, 140,919
  and 176,293 - let `ὄτι`, `εἴναι` and `τούτο` into the dictionary, which
  Tonos's gate caught. `scripts/audit_hunspell_frequency.py` applies the same
  rule, so the audit and the export agree on what a weak respelling is.

### Added
- The Patrologia Graeca now comes from three tiers of evidence rather than
  one, taking it from 22 Migne volumes and 4,212,976 tokens to 81 volumes and
  15,200,132. `data/form_profile.db` grows from 1,350,920 exact forms to
  1,945,872.
  - The corpus's Qwen3.6-27B re-OCR of the public-domain scans adds 48
    volumes that nothing else covered: Eusebius, Athanasius, Basil, both
    Gregories, Chrysostom, Cyril of Alexandria, Theodoret, John of Damascus,
    George Monachus. Its rows carry `source: "ocr"` and loci
    `pg<volume>_<page>.<line>`, which the reader had no case for, so they
    were skipped although the same pass's output was already ingested for
    In Matthaeum through a hardcoded path. Each of those volumes gets its
    author's century, since `by_century` is a shipped field.
  - The raw first-generation calfa-co dump is read again for the 11 volumes
    neither other tier covers, and for the 22 the corpus serves carved into
    the works of its carve plan it is read as SECONDARY evidence: it credits
    `source_counts` and citations for the part of a volume the carve leaves
    out, without touching the deduplicated totals the better text decides.
    That part of the volume is where 1,504 of the forms the previous build
    lost their only attestation.

### Fixed
- Diorisis beta code read the elision apostrophe after an unaccented alpha,
  iota or upsilon as a macron on that vowel, so `di'` became `δῑ` instead of
  `δι’` for 8,136 tokens. `extract_diorisis_lm.beta_to_nfc` now converts the
  word without the apostrophe and appends the elision mark U+2019, and writes
  the medial sigma a Greek word takes before it (`λέγουσ’`, `πᾶσ’`). Forms
  carrying a spurious quantity mark in `data/form_profile.db` fall from 1,537
  to 445, and the next-word language model is rebuilt from the corrected text.
  `build/build_diorisis_pairs.py` separately dropped the trailing `)` that
  Diorisis writes for the same elision, which the converter turned into a
  breathing (`δἰ`).
- The open-greek-corpus repository now stores the Patrologia Graeca one file
  per work, with a locus of the form `PG<volume>.<page>`, rather than one file
  per volume, and stores In Matthaeum under its author and work slug rather
  than its TLG identifier. `build/build_form_attestation.py` looked only for
  the old layout, so with the raw per-volume OCR clone no longer on disk it
  ingested no Patrologia Graeca at all. It now groups the per-work rows back
  into their Migne volumes, recovering 4,212,976 tokens and In Matthaeum's
  212,906 citations.
- GLAUx records a dialect for each work but `build/build_glaux_pairs.py`
  dropped it, so Epic, Ionic, Doric and Aeolic forms competed for the Attic
  cells of a verb paradigm. A form attested only in works of one of those
  dialects now carries that dialect tag, which routes it to that dialect's
  paradigm slice. Works marked Attic, Attic/Koine or Koine share the default
  slice.

### Changed
- `build/build_grc_verb_paradigms.py` picks the cell forms of
  `data/ag_verb_paradigms.json` on more evidence, which corrects 64 of the 88
  cells the textbook review in `data/hunspell_grc_textbook_review.json` had
  recorded as wrong:
  - A Wiktionary table row that reaches kaikki with its person and number
    collapsed onto the first cell is dropped. 637 of 918 fully tagged cells
    across 117 verbs were affected: τιμάω's whole present active indicative
    arrived tagged first-person singular, and on corpus frequency `τιμάς`
    (far commoner as the accusative plural of τιμή) then took the cell from
    `τιμῶ`.
  - A subjunctive cell rejects a form whose ending belongs to the
    corresponding indicative, since the subjunctive lengthens the thematic
    vowel. GLAUx labels the future indicative `κολακεύσεις` an aorist
    subjunctive, and on frequency it beat the real `κολακεύσῃς`.
  - The pass that takes a cell's accent from Wiktionary now reads which cell
    Wiktionary puts each spelling in, so it keeps an accent that is the only
    thing telling two cells apart (`λύσαι` is λύω's aorist optative
    third-person singular, `λῦσαι` its aorist middle imperative) while still
    correcting a spelling both cells share (`βαίνον` to `βαῖνον`).
  - A form is ranked on its corpus attestation, an augment in a past
    indicative cell, and whether Wiktionary lists it for the verb; forms of
    εἰμί left behind by a periphrastic cell are dropped; and the filter that
    kept only polytonic spellings no longer discards an Attic form whose
    rival happens to carry a circumflex (`δίδως` against Ionic `διδοῖς`).
- The textbook review withdraws its eight entries against the ει stem of the
  perfect of τίθημι. `τέθεικα` is the Attic perfect (Smyth 777, LSJ s.v.), and
  `data/form_profile.db` attests `τεθείκασι` 21 times against none at all for
  the `τεθήκασι` the review proposed. The `withdrawn` block records the
  evidence.

## [1.3.6] - 2026-09-22

### Fixed
- Replace the 1.3.5 Hunspell acute-only admission boundary (the 76,153-form
  language-model vocabulary, whose minimum count was 20) with the full
  accent-preserving `form_profile.db`. Sparse attested forms and complete
  textbook paradigms now survive, including `λύω`, `ποιέω`, epic/dialect
  forms, New Testament and Septuagint vocabulary, and future passives. The top
  2,000 LSJ9 headwords are normalized out of dictionary display notation and
  retained as citation forms.
- Emit the acute citation/before-pause twin of every retained contextual-grave
  form. Productive enclitic second accents are retained when removing the
  final acute yields an independently attested proparoxytone or
  properispomenon host (`θάλασσάν`, `δῶρόν`); a paroxytone host never takes
  one, so `λύκοί` and `ὕπνῴ` are rejected.
- Reject punctuation-bearing entries, internal breathings left by bad preverb
  joins, impossible accent placement including a grave before the final
  syllable (`καὶτοὺς`) or on an elided word (`γὰρ᾽`), malformed sigma, and
  known high-impact accent errors.
  New post-April forms also require exact attestation or membership in a
  pinned citation, textbook, grammar, or productive second-accent class,
  preventing orthographically plausible generator noise from riding in on a
  lemma's frequency. Structural checks run over the complete expanded
  artifact, not only the LM vocabulary.
- Reject weak new respellings of common words. A new spelling that differs
  only in accent, breathing, diaeresis, or iota subscript from a well-formed
  spelling with at least 1,000 corpus tokens needs GLAUx or Diorisis support:
  25 tokens below a 1% share of the common spelling's count, 5 below 5%.
  This removes OCR copies such as `ἑγώ`, `ταΐς`, `ἐκεί`, and `ὄσα` (`ταΐς` had
  outranked `ταῖς` as Tonos's correction for `ταις`) while keeping Doric
  `τᾷ` and `τῶ`.
- Accept valid elided forms the April dictionary lacked, such as `δι᾽`,
  `παρ᾽`, `γ᾽`, and `θ᾽`, and the complete consonant-final words `ἔκ` and
  `παρέκ`.
- Pin the 971 reviewed forms of the language-model head, so the corpus
  evidence filters cannot drop a form the release audit requires, such as the
  polytonic Modern article `τή`, which the Ancient Greek corpora behind
  `form_profile.db` barely attest. In the head fixture, `κα-` is no longer a
  reviewed nonword (its spelling `κα` is the Doric particle), and `δῑ`, a
  Diorisis Beta Code artifact of elided `δι᾽`, is.
- Keep valid spellings the structural rules would otherwise misread, all of
  which the April dictionary accepted: accentless dialect proclitics and
  enclitics (`ἁ`, `αἰ`, `εἰν`, `ἐντι`), crasis of `καί` or the article with a
  proclitic (`κἀν`, `κοὐκ`, `χὠ`, `τἀν`), crasis with a later coronis or with
  vocative `ὦ` (`ἐγᾦμαι`, `μέντἄν`, `καλοκἀγαθία`, `ὦνθρωπε`), fused
  enclitics that leave the host's accent in place (`οὗτινος`, `ᾧτινι`,
  `τοῖσιδε`), iota adscript after a long vowel (`ζῶια`, `τῆιδε`), Ionic `ωυ`
  crasis (`τωὐτό`), `πῃ`, Homeric `ὑπέκ` and `διέκ`, and indeclinable
  Septuagint loanwords (`χερουβίμ`, `ἐφούδ`, `σαβαώθ`).
- Stop giving textbook cells and closed-list forms the common frequency
  bucket: only curated forms, top citation headwords, and the language-model
  head are pinned to `fr:C`. Reviewed April forms are no longer added a
  second time under a placeholder lemma, which had written 949 words twice
  with conflicting buckets.
- Treat a spelling as attested when any single source attests it.
  `form_profile.db` deduplicates totals by work, so 199,082 spellings found
  only in a lower-priority copy of a work had a total of 0 and were dropped as
  unattested.
- Generalize same-lemma bare-elision detection while limiting the mechanism to
  suspicious final consonants, so real inflections ending in `ν`, `ρ`, `ς`,
  `ξ`, or `ψ` are not mistaken for fallbacks. Preserve the reviewed Homeric
  apocope forms, `οὐκ`/`οὐχ`, and the complete unaccented enclitic lists,
  including `εἰμί` forms; keep `του` and `τῳ` rejected for Tonos correction.

### Changed
- CI now downloads `data/form_profile.db` (tracked in `scripts/hf_data.py`)
  so it can build and audit the grc Hunspell export, and the test job's time
  limit is 45 minutes.

### Added
- Add a content-pinned whole-artifact compatibility fixture: the dictionary
  Tonos shipped in April (the Dilemma 0.4.1 export as the keyboard compiled
  it) minus the forms Tonos's candidate gate classes as junk. The
  release audit requires all 1,179,659 forms, every top-2,000 LSJ9 headword
  except three two-word phrases and one extraction artifact, all 2,170 pinned
  textbook paradigm forms, complete grave/acute twins, and zero synthetic
  flagged bases, new weak respellings, or structurally invalid entries outside
  the reviewed April surface (984 of whose forms fail Dilemma's structural
  rules and are left for review downstream).
- Pin the textbook paradigms of `λύω`, `παιδεύω`, `τίθημι`, `δίδωμι`,
  `ἵστημι`, `τιμάω`, `ποιέω`, and `δηλόω` from the generated paradigm data
  after a recorded review (`data/hunspell_grc_textbook_review.json`) that
  removes 95 wrong cells (`λύ` for `λύεις`, `λύσαν` for `λῦσαν`, Doric and
  Ionic forms in Attic slots) and adds standard cells the generator lacks
  (`δίδωσι`, `τίθησι`, `ἔδωκα`, `ποιεῖσθαι`), each corroborated by
  `lookup.db` or corpus attestation. Vowel-length marks are removed.
- Add exact-form regression gates for the New Testament (Perseus), the
  Septuagint (First1KGreek), Iliad 1, Herodotus 1, and a Katharevousa sample.
  Every source file is pinned by SHA-256, and a candidate may not exceed the
  April `.dic`/`.aff` pair's rejection count on either tokens or types. The
  corpora are in `form_profile.db`, so the gate guards against regression; it
  does not measure coverage of unseen text.

## [1.3.5] - 2026-09-21

### Fixed
- Restore common Ancient Greek spellings in the `grc` Hunspell export. The
  selector admits language-shared rows only for lemmas with `grc` evidence,
  uses a revision-pinned accent-preserving full-LM map for acute-only forms,
  rejects vowel- or rho-initial spellings without a breathing, and preserves a
  reviewed closed list of correctly unaccented enclitics. Bare lemmatizer
  fallbacks for elision remain excluded. The resulting export has 1,284,405
  entries and 105 zero-strip suffix rules.
- Stop emitting truncated common prefixes as flagged Hunspell entries. Affix
  compression now occurs only when the base is itself a paradigm form; all
  other paradigms are inlined, requiring neither `NEEDAFFIX` nor nonzero-strip
  support from Tonos.
- Restore high-frequency valid forms that Dilemma resolves through grammar or
  POS authorities but which lookup source collisions hid from the exporter,
  including `τ᾽`, `μεθ᾽`, `δῑ`, `εἶνε`, `μαῦρον`, `μαῦροι`, and
  `ἑκατέρως`. `ἀνάμεσα` is retained through exact LM attestation, while
  `του`, `τῳ`, monotonic `ότι`/`αυτός`, and unrelated Modern forms remain
  rejected.
- Normalize elision marks through one shared corpus-frequency key in the
  GLAUx, Diorisis, Patristic Text Archive, and OGC rollup builders, and again
  at merge time. Diorisis no longer deletes the apostrophe and decomposed TEI
  psili is preserved before accents are stripped, preventing parallel `δ` and
  `δ’` keys. The rebuilt OGC rollup is pinned to commit `f4062a50` and
  public-lexicon SHA-256 `02d6de71...`; merge provenance retains both values.
- Generalize the Hunspell bare-elision rejection beyond a finite stem list.
  A bare spelling with a more frequent marked counterpart is excluded unless
  DGE or Cunliffe independently establishes it as a headword, preserving
  genuine collisions such as `ἄν`. Preserve the closed set of Homeric
  shortened prepositions (`κὰτ`/`κάτ` through `κὰγ`/`κάγ`, `ἂμ`/`ἄμ`).

### Added
- Add a revision-pinned Hunspell frequency auditor and CI regression fixture.
  It loads the expanded dictionary with `spylls`, folds contextual grave and
  final elision marks, and checks the top 1,000 Greek-bearing LM tokens:
  971 required forms accepted and 29 reviewed nonwords rejected, representing
  16,543,630 tokens. It also rejects synthetic flagged bases and emitted forms
  without required initial breathings. JSON fixture generation hard-fails on
  sanity runs; format-v2 LM binaries supply exact vocabulary and counts. An
  optional full-vocabulary comparison produces a reviewable LM-weighted delta
  against a previously shipped dictionary. The current candidate gains 4,510
  full-LM forms (1,232,520 tokens) and loses 213 (73,046 tokens) against the
  April Tonos artifact; the newly removed forms are reviewed elision fragments.

## [1.3.4] - 2026-09-21

### Fixed
- Reject editorial square brackets and parentheses at GLAUx pair ingestion
  and again in both movable-nu and elision morphology derivation. This keeps
  contaminated corpus forms such as `)λπίζουσι` out of `grc_morph.json` and
  protects morphology exports made from older pair artifacts.

## [1.3.3] - 2026-09-21

### Fixed
- Resolve terminal Wiktionary movable-nu notation (`(ν` / `(ν)`) into both
  conventional spellings while rejecting every other square-bracket or
  parenthesis siglum in expansion forms and citation lemmas. The final overlay
  removes 93,084 contaminated target rows, adds 159,083 clean movable-nu
  spellings, skips 247 contaminated historical-reference rows, and applies the
  same guard to Byzantine gap-fill data and Hunspell export. The rebuilt
  `lookup.db` and `grc_polytonic` files contain no such delimiters, and the
  Hunspell dictionary now loads in regex-based consumers such as `spylls`.

## [1.3.2] - 2026-09-21

### Fixed
- Restore the revision-pinned historical LSJ/Sophocles expansion layer after
  current expansion runs. The recovery derives only forms absent from the
  exact historical Wiktionary base, preserves all current mapping conflicts,
  and guards representative polytonic paradigms in the shipped lookup. On the
  1.3.1 inputs this restores 3,551,231 marked AG mappings while deliberately
  omitting 3,839,827 generated accent-stripped fallback keys
  (`ag_lookup.json` grows from 5,314,819 to 8,866,050 entries). The repaired
  `grc` Hunspell export contains 1,617,436 dictionary entries and 26,521 affix
  rules, up from 1,157,778 and 7,816 in 1.3.1.

## [1.3.1] - 2026-09-20

### Added
- Add a reproducible, headword-only importer for the Mozilla Data Collective
  export of the Triantafyllidis dictionary. The Parquet read projects only the
  `lemma` column, recognizes only audited display conventions, rejects
  ambiguous headings rather than guessing, and records source and validation
  checksums with aggregate audit counts. The Triantafyllidis convention now
  unions that dictionary authority with the existing Modern Greek Wiktionary
  inventory for broader coverage.
- Add a revision- and byte-pinned GlossAPI coverage auditor for nine
  permissively licensed candidate corpora. It reports `guess=False` coverage,
  unknown forms, conservative Unicode/OCR defects, and Modern/Ancient lookup
  conflicts, and can build a balanced Modern Greek frequency experiment
  without modifying `mg_freq.txt`. Historical candidates remain audit-only
  until Open Greek Corpus admits cleaned, identified, and deduplicated text.
- Add a separate optional evaluation for GlossAPI's Greek variety classifier.
  GreekBERT and Transformers remain outside Dilemma's core dependencies, and
  supplied Gutenberg variety labels are not treated as independent gold data.

### Fixed
- Stop ingesting the placeholders a treebank uses for "not a word". GLAUx
  marks an editorial gap, where the manuscript is damaged or unreadable, with
  `relation="GAP"`, postag `z` (unanalyzable) and the gap character in the
  lemma field, leaving the surviving letters in `form`: `Gδου` is what is left
  of the genitive of Hades in Plutarch's Comparatio Cimonis et Luculli after
  its first two letters were lost. `build_glaux_pairs.py` read those as
  ordinary form-and-lemma pairs, and `is_greek` did not stop them because it
  asks whether ANY character is Greek and the surviving letters are, so bare
  `G` was rejected while `Gδου` and `μάχηG` were not. The reader now skips a
  token the annotation itself marks as a gap, which removes 1,868 of the
  624,822 pairs in `glaux_pairs.json`. Separately, the cog standardized
  exports use the CoNLL-U convention of `_` for a field the annotator left
  empty; `is_clean_lemma` accepted it as a lemma string, so 1,108 of the
  18,552 Pedalion pairs carried it, metrical scansion patterns among them.
  `_` is now rejected for every cog export, not just Pedalion. Measured on
  the shipped artifacts before the fix: 1,019 `lookup.db` rows resolved to
  the lemma `_` and nothing else, so `Dilemma().lemmatize("διατείνας")`
  answered `"_"` for a real aorist participle of διατείνω, and `Gδου`
  answered `"G"`. Both fall through to the rule and model layers instead
  after a rebuild.
- Reject the corpus word-tails that entered `lookup.db` as words of their
  own. A corpus tokenizer split one printed word at its elision mark and gave
  the second half the whole word's lemma, so the ending arrived at ingestion
  looking exactly like an elided word: `νοντ᾽` under βαρύνω (the end of
  `βαρύνοντ᾽`), `εντ᾽` under πευκήεις, `λευμ᾽` under γαμήλευμα, `γάσασθ᾽`
  under ἐργάζομαι. Nothing about the pair alone gives one away, so no per-pair
  filter can catch it; what gives it away is the rest of the lemma's paradigm,
  which makes this a whole-table check. `build_lookup_db.py` now runs one
  (`drop_corpus_word_tails`, the rule in `dilemma/elision_tails.py`) over the
  merged table and both per-language tables, after every ingestion path has
  contributed and before any row is written. Every rejection is recorded in
  `data/citation_hygiene_rejections.tsv` with the form it was cut from as its
  evidence.

  Measured against `data/lookup.db` and `data/form_profile.db`: 10,814 of the
  7,097,421 Ancient Greek rows end in an elision mark; 234 of those are also
  the ending of another form of the same lemma (228 once the six single-letter
  stems are set aside); 8 of those 234 have no corpus attestation; and 4 of
  those 8 are cut from their sibling by two or more characters. Those 4 are
  exactly the tails above, and the rule rejects nothing else - not `μ᾽ σ᾽ ν᾽
  κ᾽ τιν᾽ ποτ᾽ φησ᾽ σφ᾽ ᾽ς ᾽ν κατ᾽ δ᾽ ἀλλ᾿ γράψατ᾽ φης᾽ βούλεσθ᾽`, and nothing
  at all on the Modern Greek side.

  The last two conditions carry different work, because Ancient Greek builds
  two word-pairs that look like a word and its tail. The augment makes a
  one-character pair: `γράψατ᾽` is `ἐγράψατ᾽` minus `ἐ` and `φης᾽` is `έφης`
  minus its augment, both real words that no corpus wrote, so only the cut
  length saves them. Reduplication makes a two-character pair that the cut
  length cannot separate: `δοῦσ᾽` is `διδοῦσ᾽` minus `δι`, `μνήσομ᾽` is
  `μεμνήσομ` minus `με`, `θεῖσ᾽` is `τιθεῖσ᾽` minus `τι`. Of the 23 rows cut by
  two or more characters, 4 are the tails and the other 19 are real words held
  only by their corpus attestation; reduplication is the commonest reason one
  lands there, with crasis and an augment before a doubled consonant giving
  the same two-character cut. The stem is looked up with the mark attached and never
  bare, because the same tokenizer error left the bare stem in the corpus it
  annotated (`νοντ` occurs 4 times). The elision mark is recognized in all
  eight spellings a source can use, including U+0313 COMBINING COMMA ABOVE,
  which is how treebank exports write it; U+1FFE GREEK DASIA is excluded
  because it is the rough breathing and elision leaves a smooth one.

  Rebuild impact, computed read-only against the current `data/lookup.db`: 4
  of 9,854,919 rows go, all four keys disappear entirely (each had exactly one
  row), and no surviving key changes lemma. The runtime does not start
  answering "no lemma" for them - only `λευμ᾽` does. `νοντ᾽` becomes μιαίνω
  and `εντ᾽` becomes εἰμί from the elision expander, and `γάσασθ᾽` falls to
  the transformer, whose answer depends on the model version and is not
  recorded here. What the fix removes is the lookup's
  high-confidence assertion that these are words, and with it their place in
  `spell_index.db` and their candidacy for the Ancient Greek Hunspell
  dictionary, which three of the four currently clear the diacritic gate for.

  Two residues remain and want a different rule: `G2α᾽` (lemma `G`) and `G?᾽`
  (lemma `_`) are optical-character-recognition placeholders, not tails. They
  are cut from their siblings by one character, so the cut-length test lets
  them through. They need a test on the characters themselves - a form or
  lemma that is not Greek - which belongs with the other per-pair checks in
  `build_lookup_db._citation_artifact_reason`.
- Stop the Hunspell exporter from dropping every Ancient Greek form whose
  only mark is a spacing elision or breathing character. `has_any_diacritic`
  tested for Unicode combining marks (category Mn), while U+1FBD GREEK
  KORONIS and the spacing breathings are Sk, so `δ᾽`, `κατ᾽`, `μετ᾽` and
  `μηδ᾽` were rejected although `lookup.db` holds them. They passed
  before `build_lookup_db.py` started running `sanitize_form` at ingestion,
  which rewrites a trailing combining psili to the koronis. A spacing mark
  counts only on a form that opens on a Greek consonant, the way an elided
  or aphaeresized word does, because polytonic Greek writes a breathing over
  every word-initial vowel. That keeps out the Milesian numerals a source
  wrote with a koronis in place of the keraia (`ε᾽` five, `α᾽` one, `ο᾽`
  seventy) and the accent-stripped lookup keys of vowel-initial words
  (`ημειβετ᾿` beside `ἠμείβετ᾿`). Stranded acutes, the numeral
  signs U+00B4, U+0384 and U+0375, and quotation-mark apostrophes are still
  not read as marks at all, and unaccented spellings are still rejected.
  Measured against `data/lookup.db`, the Ancient Greek variant gains 45
  (form, lemma) pairs and loses none. Three kinds of residue still get
  through, none separable from a real spelling by any rule on the string:
  a numeral whose letters open on a consonant (`δ᾽` is four as well as the
  elided δέ), the accent-stripped lookup key of an accented elision
  (`ταυτ᾽` beside `ταῦτ᾽`, the same shape as the correct `κατ᾽`), and
  word-tails that entered the lookup as rows of their own (`νοντ᾽` under
  βαρύνω, the end of βαρύνοντ᾽; `μα᾽` under μής, a lemma whose other
  forms are OCR residue). The second belongs to `build_data.py`'s
  `_add_lookup`, whose stripped key drops the accent but keeps the elision
  mark. The third is a corpus tokenization error; `νοντ᾽` is now rejected at
  ingestion by the word-tail filter described above, while `μα᾽` is not,
  because it is the ending of no other form of μής - that lemma's whole
  paradigm is residue, which is a different defect.
- Apply the GLAUx license filter (`build/nc_filter.py`) to the corpus behind
  the next-word prediction language model. `train_lm.iter_glaux_sentences`
  read every GLAUx file, so the n-gram counts and the held-out dev sentences
  carried the 7 NonCommercial and 25 PROIEL-derived works, plus the manual
  sentences of the 40 Gorman-derived works that are the project's held-out
  gold: 841,551 of the 17.0M tokens GLAUx contributes, 4.95%. Excluded works
  are now dropped whole and Gorman-derived works keep only their automatic
  sentences, matching every other GLAUx reader. `--glaux-metadata` says where
  the filter reads GLAUx's metadata.txt, and a run stops rather than proceed
  without it. The shipped `grc_ngram.bin` has not been rebuilt.

## [1.3.0] - 2026-09-16

### Added
- Add a deterministic Dilemma-only benchmark regression gate for the
  committed Classical, Katharevousa, and Demotic gold sets. The gate pins
  fixture hashes and token predictions, reports token-level changes, and
  fails CI on strict or equivalence-adjusted accuracy regressions.
- Add `scripts/release.py` as the single release command. It updates version
  metadata and the changelog, waits for the exact release commit to pass main
  CI, creates the numeric tag, and watches GitHub and PyPI publication.

### Changed
- Make the citation-hygiene audit lookup-source-aware, reporting Ancient and
  Modern Greek residue separately and classifying broad multi-accent results
  as single-token or multiword/punctuated diagnostics.
- Reuse successful main CI when publishing a tag instead of running the full
  artifact-backed suite twice, and update `actions/setup-python` to v7.
- Document that additive voice-aware Modern Greek canonical cells are owned
  by Klisy's canonical producer, while its legacy collapsed cells remain
  backward compatible; Dilemma does not duplicate that downstream exporter.

## [1.2.4] - 2026-09-16

### Fixed
- Reject citation lemmas whose tonal accents are structurally impossible:
  duplicate tonal marks on one vowel or tonal marks attached to a non-vowel.
  Legitimate multi-accent phrases and enclitic-bearing forms remain allowed.
- Restore propagation of Ancient Greek verb tense from Wiktionary table
  headers onto individual morphology pairs, completing work that had remained
  isolated on the stale `grc-verb-tense-tags` branch.

### Changed
- Split structurally malformed tonal residue from the broad multi-accent audit
  bucket and preserve source-file provenance for build-time citation rejects.
- Add installed-wheel smoke tests across every declared Python version
  (3.10-3.14), while retaining the full artifact-backed test run on 3.12;
  modernize the package's SPDX license metadata for current setuptools.

## [1.2.3] - 2026-09-14

### Fixed
- Reject final elision marks and final keraia/prime marks as citation lemmas,
  while preserving explicit numeral and other nonlexical pass-through tokens.
- Rebuild lookup artifacts without final-elision, final-keraia/prime, or
  nonlexical numeral/siglum values in the lemma table.

### Changed
- `scripts/audit_citation_hygiene.py` now separates nonlexical mark-bearing
  tokens from rejected citation-lemma residue, so the final-keraia bucket only
  reports unresolved citation artifacts.

### Tests
- Add a citation-hygiene audit guard for the generated lookup artifact so
  grave, overline, leading-combining, final-elision, and final-keraia citation
  residue cannot silently re-enter CI.

## [1.2.2] - 2026-09-13

### Fixed
- Reject overline abbreviation marks and stranded leading combining marks as
  invalid citation lemmas, matching the existing proof-based grave-accent gate.
- Apply the same citation-form hygiene while rebuilding `lookup.db`, including
  late LBG headword and generated-pair additions, so rebuilt lookup artifacts no
  longer carry grave, overline, or leading-combining citation lemmas.
- Allow grave-to-acute normalization for trusted single-token independent
  lexicon headwords only; unproven graves, including multiword grave phrases,
  remain rejected rather than blindly normalized.

### Changed
- `scripts/audit_citation_hygiene.py` now separates Greek numeral and final
  keraia/prime residue from true final elision marks, and reports
  accepted/rejected citation-status reason counts.
- Refreshed HuggingFace-pinned `lookup.db` and `spell_index.db` artifacts in
  `data/hf_manifest.json`.

### Documentation
- Document `citation_policy="strict_ag"`, `citation_status()`, and the
  `LemmaCandidate.citation` validation note.

## [1.2.1] - 2026-09-13

### Added
- Device-agnostic ONNX execution: `dilemma._ort_providers` auto-selects CUDA
  when `onnxruntime-gpu` is present (an NVIDIA box), else CPU, with a
  `DILEMMA_ORT_PROVIDERS` override. `Tagger.on_gpu` / `Tagger.providers` report
  the REAL execution device, and `make_session()` warns on a silent CPU
  fallback (onnxruntime-gpu missing / CUDA-cuDNN mismatch) - the trap where a
  GPU box runs the tagger on CPU with the card idle. README's "Device and
  throughput planning" section now carries measured numbers: a full-corpus pass
  is CPU + memory-bandwidth bound, not GPU-bound. A 64-core EPYC 7B12 saturates
  the lemmatizer at ~1,100 tok/s around 32 workers (tag-on-CPU 745 tok/s/proc),
  and coupling tag+lemma as one CUDA session per shard is *worse* (361 tok/s,
  VRAM-bound) than the CPU lemmatizer alone; a Threadripper PRO 7995WX
  (96c/192t, DDR5 8-ch) hits ~2,460 tok/s at ~48-61 workers (2.2x). Verdict:
  buy cores + memory bandwidth, not a GPU; use a decoupled pure-CPU scale-out
  (never one GPU tagger session per worker); real throughput drops on
  beam-search-heavy text (lexica/scholia) and a run's wall-clock is floored by
  its largest single work.
- NON-LEXICAL token classifier (`dilemma.nonlexical`), exposed as
  `classify_nonlexical(token)` / `is_lexical(token)` (module level and as
  `Dilemma` methods). It recognizes the editorial/typographic residue that
  dominates OCR'd lexica, scholia, and the Patrologia Graeca - γράφεται
  variant marks (`γρ`), Greek numerals (`κζ'`, `,αφ'`), bracket references
  (`[76]`, `[49-59]`), Latin/citation abbreviations (`fr.`, `Herod.`), lone
  punctuation/sigla, and vowel-less consonant fragments - so a caller can
  tell "not a lexical word" from "failed to lemmatize a real word". Pure
  stdlib, no model. `lemmatize`/`lemmatize_batch` return non-lexical tokens
  unchanged and skip the transformer fallback (which otherwise manufactures
  spurious lemmas); `lemmatize_verbose` returns a single candidate with
  `source="nonlexical"`, the class label in `via`, and `tag="X"`.
  `LemmaCandidate` gains a `tag` field and an `is_lexical` property. The
  classifier is conservative by construction (a Greek word always has a
  vowel; numerals are unaccented with strictly descending place-value tiers),
  so elided monosyllables (`δ᾿`) and numeral-shaped real words (`τε`) stay
  lexical; no movement on `bench_fast.py`.

### Fixed
- Ancient Greek lemmatization no longer emits grave-accented citation lemmas
  from contaminated lookup, model, or POS-table values. Grave values are
  normalized to acute only when the acute form is backed by an independent
  lexicon headword inventory; otherwise that candidate is dropped. The
  validation is applied consistently across single-word, batch, verbose, and
  POS-aware entry points.
- `lemmatize_pos` / `lemmatize_batch_pos` no longer prefer a capitalized
  proper-noun twin over the common-word lemma for a lowercase, non-PROPN
  token. Many common lemmas have a capitalized personification/name twin
  as a separate headword (θυμός vs Θυμός, ἔρις vs Ἔρις, τύχη vs Τύχη),
  surfaced both by the lookup ("+case_alt") and by the POS tables; the
  POS-matching step could return the twin whose capitalization disagreed
  with the input. A capitalization-agreement tiebreak now runs on every
  ranking path: a lowercase non-PROPN form gets the lowercase lemma, a
  PROPN tag still reaches the capitalized twin, and capitalized or
  all-caps input (sentence-initial, titles) keeps either twin reachable
  by POS. It is a re-rank between case twins, never a filter. Measured
  +0.76 lemma-equivalence points on the Persae gold calibration set
  (91.83% -> 92.59%); no movement on `bench_fast.py`.

### Changed
- `python -m dilemma download` (and `dilemma.download()`) now opts in to
  HuggingFace's high-performance transfer automatically: Xet
  (`HF_XET_HIGH_PERFORMANCE=1`) on modern `huggingface_hub`, or the
  `hf_transfer` Rust downloader on older versions when installed. An
  explicit user setting (including `0`) is respected.

## [1.1.0] - 2026-07-03

### Changed
- The Ancient Greek (`grc`) tagger weights were fine-tuned on the Iliad
  composite gold plus a GLAUx mixture. No regression on general Greek
  (GLAUx test strict 93.6% -> 94.1%, UAS 83.6% -> 85.1%, LAS 78.3% ->
  79.9%); large gains on Homeric text. Pinned revision updated; tagger
  sub-version 0.6.0.
- `Tagger.tag` now segments leading/trailing punctuation into standalone
  `PUNCT` tokens ("ἄειδε," -> "ἄειδε" + ","), so the output can contain
  more tokens than `text.split()` (as MG multiword tokens already could).
  Elision/aphaeresis apostrophes stay attached (δ’, ῥ’, ’γώ); runs of one
  character stay one token ("..."). Lemma lookups also improve, since the
  lemmatizer now sees the clean word instead of "word+comma". Tagger
  sub-version 0.7.0.

## [1.0.0] - 2026-06-29

First stable release.

### Added
- Ancient Greek (`grc`/`med`) and Modern Greek (`el`) dependency parsing: the
  taggers now carry a biaffine dependency head, so every token has `head` and
  `deprel` (previously `el` had no shipped parser and `grc`/`med` had none).
- `dilemma.__version__`, resolved from the installed package metadata.

### Changed
- The tagger runtime is now torch-free: importing `dilemma.tagger` pulls in only
  `onnxruntime` + `tokenizers` (+ numpy). `torch`/`transformers` are needed only
  to (re)train and export weights.
- The `[tagger]` extra now installs the runtime plus torch + transformers, so
  `pip install dilemma-nlp[tagger]` followed by `Tagger()` works. For inference
  only, use `[tagger-onnx]`.
- Tagger weights auto-download from HuggingFace when not present locally.

### Removed (breaking)
- `Tagger()` no longer accepts `checkpoint`, `pos_path`, or `dp_path`; `device`
  is accepted but advisory (the ONNX runtime is CPU). Point `DILEMMA_TAGGER_DIR`
  at a directory to use custom weights.
- The legacy dual-BERT joint tagger stack (`TaggerModel`, the joint ONNX model,
  and the gr-nlp-toolkit weight loader) has been removed.

### Licensing
- Openly licensed by default: NonCommercial sources are never ingested. The
  committed NonCommercial PROIEL data was removed; PROIEL is dropped, UD Perseus
  is replaced by the AGDT original (CC BY-SA), and the NonCommercial GLAUx and
  PTA texts are filtered out. See NOTICE for the full per-source list.

[1.3.29]: https://github.com/open-greek/dilemma/releases/tag/1.3.29
[1.3.28]: https://github.com/open-greek/dilemma/releases/tag/1.3.28
[1.3.27]: https://github.com/open-greek/dilemma/releases/tag/1.3.27
[1.3.26]: https://github.com/open-greek/dilemma/releases/tag/1.3.26
[1.3.25]: https://github.com/open-greek/dilemma/releases/tag/1.3.25
[1.3.24]: https://github.com/open-greek/dilemma/releases/tag/1.3.24
[1.3.23]: https://github.com/open-greek/dilemma/releases/tag/1.3.23
[1.3.22]: https://github.com/open-greek/dilemma/releases/tag/1.3.22
[1.3.21]: https://github.com/open-greek/dilemma/releases/tag/1.3.21
[1.3.20]: https://github.com/open-greek/dilemma/releases/tag/1.3.20
[1.3.19]: https://github.com/open-greek/dilemma/releases/tag/1.3.19
[1.3.18]: https://github.com/open-greek/dilemma/releases/tag/1.3.18
[1.3.17]: https://github.com/open-greek/dilemma/releases/tag/1.3.17
[1.3.16]: https://github.com/open-greek/dilemma/releases/tag/1.3.16
[1.3.15]: https://github.com/open-greek/dilemma/releases/tag/1.3.15
[1.3.14]: https://github.com/open-greek/dilemma/releases/tag/1.3.14
[1.3.13]: https://github.com/open-greek/dilemma/releases/tag/1.3.13
[1.3.12]: https://github.com/open-greek/dilemma/releases/tag/1.3.12
[1.3.11]: https://github.com/open-greek/dilemma/releases/tag/1.3.11
[1.3.10]: https://github.com/open-greek/dilemma/releases/tag/1.3.10
[1.3.9]: https://github.com/open-greek/dilemma/releases/tag/1.3.9
[1.3.8]: https://github.com/open-greek/dilemma/compare/3651c5b76b0c5c186eccee27b29cdf7d35e3bfdc...7353f606173259441f2816ee50fbe126d6af90fc
[1.3.7]: https://github.com/open-greek/dilemma/compare/1.3.6...3651c5b76b0c5c186eccee27b29cdf7d35e3bfdc
[1.3.6]: https://github.com/open-greek/dilemma/releases/tag/1.3.6
[1.3.5]: https://github.com/open-greek/dilemma/releases/tag/1.3.5
[1.3.4]: https://github.com/open-greek/dilemma/releases/tag/1.3.4
[1.3.3]: https://github.com/open-greek/dilemma/releases/tag/1.3.3
[1.3.2]: https://github.com/open-greek/dilemma/releases/tag/1.3.2
[1.3.1]: https://github.com/open-greek/dilemma/releases/tag/1.3.1
[1.3.0]: https://github.com/open-greek/dilemma/releases/tag/1.3.0
[1.2.4]: https://github.com/open-greek/dilemma/releases/tag/1.2.4
[1.2.3]: https://github.com/open-greek/dilemma/releases/tag/1.2.3
[1.2.2]: https://github.com/open-greek/dilemma/releases/tag/1.2.2
[1.2.1]: https://github.com/open-greek/dilemma/releases/tag/1.2.1
[1.2.0]: https://github.com/open-greek/dilemma/releases/tag/1.2.0
[1.1.0]: https://github.com/open-greek/dilemma/releases/tag/1.1.0
[1.0.0]: https://github.com/open-greek/dilemma/releases/tag/1.0.0
