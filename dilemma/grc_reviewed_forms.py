"""Reviewed rows for Ancient forms that no lookup table has a row for.

Corrected corpus readings bring in forms that no table reads and that no rule
of the lemmatizer reaches: Homeric and Ionic endings (ἀίξασκε, εὐπατρίδῃσι,
ἀνείησι), Doric ones (καθυπερτάταν, ποτέοικε), and verb forms the generated
paradigms lack (ἀπήχθημεν, Ἐνήργηκε). Each form here was read in its place on
the printed page (#161), and its lemma is a head of a dictionary or word list
that Dilemma reads.

The rows answer before the tables, for ``lang="grc"`` and ``lang="all"``.
None of the forms has a row of its own in any table, Ancient or Modern, so no
Modern reading is replaced. A form is listed as the lookup cascade first reads
it: lowercase, with an acute for a final grave, with the elision mark it lost.
"""

from __future__ import annotations

GRC_REVIEWED_FORM_RECORDS: dict[str, dict[str, str]] = {
    "ἀίξασκε": {
        "lemma": "ἀΐσσω",
        "reason": "Homeric iterative aorist of ἀΐσσω.",
        "evidence": "Archestratus fr., Brandt, Corpusculum poesis epicae ludibundae: archive.org corpusculumpoesi01lipsuoft, scan n114 (#161)",
    },
    "ἀπέσμηξε": {
        "lemma": "ἀποσμήχω",
        "reason": "Aorist of ἀποσμήχω, wipe off.",
        "evidence": "Chrysostom, In epistulam ad Galatas: PG 61, archive.org patrologia-volumes/061, scan n757 (#161)",
    },
    "ἀνείησι": {
        "lemma": "ἀνίημι",
        "reason": "Subjunctive of ἀνίημι with the epic -ῃσι ending.",
        "evidence": "Philodemus, De musica (Kemke): archive.org philodemidemusic00phil, scan n101 (#161)",
    },
    "ἀπήχθημεν": {
        "lemma": "ἀπάγω",
        "reason": "Aorist passive of ἀπάγω, first person plural.",
        "evidence": "Chrysostom, Epistulae: PG 52, archive.org patrologia-volumes/052, scan n335 (#161)",
    },
    "παρήχθητε": {
        "lemma": "παράγω",
        "reason": "Aorist passive of παράγω, second person plural.",
        "evidence": "Georgius Monachus, Chronicon: PG 110, archive.org patrologia-volumes/110, scan n396 (#161)",
    },
    "ἐνήργηκε": {
        "lemma": "ἐνεργέω",
        "reason": "Perfect of ἐνεργέω (printed capitalized, Ἐνήργηκε).",
        "evidence": "Cyril of Alexandria, Contra Nestorium: PG 76, archive.org patrologia-volumes/076, scan n50 (#161)",
    },
    "κατωθήσας": {
        "lemma": "κατωθέω",
        "reason": "Aorist participle of κατωθέω, push down.",
        "evidence": "Cyril of Alexandria, In Isaiam: PG 70, archive.org patrologia-volumes/070, scan n172 (#161)",
    },
    "αὐχεῖθ’": {
        "lemma": "αὐχέω",
        "reason": "αὐχεῖτε elided before a rough breathing.",
        "evidence": "Isidorus Scholasticus (Cougny, Appendix nova): archive.org epigrammatumant00couggoog, scan n78 (#161)",
    },
    "αὐχεῖθ": {
        "lemma": "αὐχέω",
        "reason": "αὐχεῖθ’ with the elision mark lost.",
        "evidence": "Isidorus Scholasticus (Cougny, Appendix nova): archive.org epigrammatumant00couggoog, scan n78 (#161)",
    },
    "εὐπατρίδῃσι": {
        "lemma": "εὐπατρίδης",
        "reason": "Ionic dative plural of εὐπατρίδης.",
        "evidence": "Isidorus Scholasticus (Cougny, Appendix nova): archive.org epigrammatumant00couggoog, scan n537 (#161)",
    },
    "χοροιτυπίην": {
        "lemma": "χοροιτυπία",
        "reason": "Ionic accusative of χοροιτυπία.",
        "evidence": "Isidorus Scholasticus (Cougny, Appendix nova): archive.org epigrammatumant00couggoog, scan n41 (#161)",
    },
    "καθυπερτάταν": {
        "lemma": "καθυπέρτατος",
        "reason": "Doric accusative feminine of καθυπέρτατος.",
        "evidence": "Stobaeus, Anthologium I (Wachsmuth): archive.org adw8682.0001.001.umich.edu, scan n318 (#161)",
    },
    "ποτέοικε": {
        "lemma": "ποτέοικα",
        "reason": "Doric ποτέοικε (προσέοικε), headed as ποτέοικα.",
        "evidence": "Theages fr. (Mullach, FPG II): archive.org fragmentaphiloso02mulluoft, scan n128 (#161)",
    },
    "χριστέμποροι": {
        "lemma": "χριστέμπορος",
        "reason": "Plural of χριστέμπορος (printed capitalized, Χριστέμποροι).",
        "evidence": "Gregory of Nazianzus, Oratio 40: PG 36, archive.org patrologia-volumes/036, scan n209 (#161)",
    },
    "ἐϋκμήτῳ": {
        "lemma": "εὔκμητος",
        "reason": "Homeric ἐΰκμητος, the head εὔκμητος (as ἐϋτρήτοισι εὔτρητος).",
        "evidence": "Didymus (Schmidt): archive.org didymichalcente00schmgoog, scan n149 (#161)",
    },
    "νεικεογεννήτοισιν": {
        "lemma": "νεικεογέννητος",
        "reason": "Epic dative plural of νεικεογέννητος (Empedocles).",
        "evidence": "Hierocles, In Aureum carmen (Mullach, FPG I): archive.org fragmentaphilo01mull, scan n97 (#161)",
    },
    "ταυτοποιόν": {
        "lemma": "ταὐτοποιός",
        "reason": "ταὐτοποιόν printed without the coronis.",
        "evidence": "Proclus, In Timaeum II (Diehl): archive.org diehl-procli-diadochi-in-platonis-timaevm-commentaria-gr-lat-t.-2-1904, scan n23 (#161)",
    },
    "παρέπλαττεν": {
        "lemma": "παραπλάσσω",
        "reason": "Imperfect of παραπλάττω, the Attic spelling of παραπλάσσω.",
        "evidence": "Excerpta de virtutibus et vitiis II.1: archive.org constporphexcerptahistoricavol2.1devirtutibus, scan n195 (#161)",
    },
    "τηυγέτη": {
        "lemma": "Τηϋγέτη",
        "reason": "Τηϋγέτη, the Ionic name, printed without the diaeresis.",
        "evidence": "Hesiod fr. (Rzach): archive.org bub_gb_y3g7AQAAMAAJ, scan n425 (#161)",
    },
    "προὐταξάμην": {
        "lemma": "προτάσσω",
        "reason": "προετάξαμην with the crasis of προ and the augment.",
        "evidence": "Themistius, Oratio 1 (Dindorf): archive.org themistiioratio01dindgoog, scan n53 (#161)",
    },
    "προϋπεῖναι": {
        "lemma": "προΰπειμι",
        "reason": "Infinitive of προΰπειμι, be there before.",
        "evidence": "Eustathius, In Iliadem III: archive.org commentariiadhom03eust, scan n54 (#161)",
    },
    "προϋποτεθεῖσαν": {
        "lemma": "προϋποτίθημι",
        "reason": "Aorist passive participle of προϋποτίθημι.",
        "evidence": "Proclus, In Timaeum III (Diehl): archive.org diehl-procli-diadochi-in-platonis-timaevm-commentaria-gr-lat-t.-3-1906, scan n188 (#161)",
    },
    "προϋφεστήκασιν": {
        "lemma": "προϋφίστημι",
        "reason": "Perfect of προϋφίστημι, third person plural.",
        "evidence": "Gregory of Nyssa, De anima et resurrectione: PG 46, archive.org patrologia-volumes/046, scan n69 (#161)",
    },
    "μεσσούριον": {
        "lemma": "μεσούριον",
        "reason": "μεσούριον with the epic double σ.",
        "evidence": "Aristarchus (Ludwich): archive.org aristarchshomer00gramgoog, scan n568 (#161)",
    },
}

GRC_REVIEWED_FORMS: dict[str, str] = {
    form: record["lemma"] for form, record in GRC_REVIEWED_FORM_RECORDS.items()
}
