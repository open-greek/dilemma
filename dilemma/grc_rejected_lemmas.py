"""Reviewed Ancient Greek lemma values that are no citation form.

Each entry is a spelling that some source gives as a lemma but that no
dictionary heads, with the reason and the evidence it was reviewed on. The
lookup build (``build_lookup_db.py``) drops every reading whose lemma is
listed, ``build/build_glaux_pos_lookup.py`` skips GLAUx pairs with such a
lemma, and the runtime drops POS-table cells that target one, so tables
built before the record (the treebank and Wiktionary POS tables) cannot
return them either.

Only exact spellings are listed: ``τοῦτος`` (an LBG headword, the late
by-form of οὗτος) and ``ἄτη`` keep their own readings.
"""

from __future__ import annotations

REJECTED_LEMMA_TARGETS: dict[str, dict[str, str]] = {
    "Ταύτη": {
        "reason": "wiktionary_el_only_head",
        "evidence": "Only a Greek Wiktionary Ancient Greek page heads it "
                    "(grc_lexical_heads.json lists it under wiktionary_el_grc "
                    "alone); ταύτῃ and ταύτην are forms of οὗτος. GLAUx also "
                    "files four οὗτος forms under it as a name (#108).",
    },
    "Τοῦτος": {
        "reason": "wiktionary_el_only_head",
        "evidence": "Only a Greek Wiktionary Ancient Greek page heads it "
                    "(wiktionary_el_grc alone); GLAUx files Τούτου, Τούτοις, "
                    "Τοῦτο, Τοῦτον under it as a name. Lowercase τοῦτος (LBG) "
                    "is not affected (#108).",
    },
    "Ἡμων": {
        "reason": "wiktionary_el_only_head",
        "evidence": "Only a Greek Wiktionary Ancient Greek page heads it "
                    "(wiktionary_el_grc alone); ἡμῶν is the genitive of "
                    "ἡμεῖς (ἐγώ) (#108).",
    },
    "Ταύτης": {
        "reason": "glaux_name_lemma",
        "evidence": "GLAUx annotates Ταύτῃ, Ταύτην, ταύτῃ, ταύτης (forms of "
                    "οὗτος) as a proper noun with this lemma; no dictionary "
                    "heads it (#108).",
    },
    "Τοῦτης": {
        "reason": "glaux_name_lemma",
        "evidence": "GLAUx annotates Τοῦτο as a proper noun with this lemma; "
                    "no dictionary heads it (#108).",
    },
    "Τούτῖς": {
        "reason": "glaux_name_lemma",
        "evidence": "GLAUx annotates Τούτων (pron) with this lemma; no "
                    "dictionary heads it (#108).",
    },
    "Τουτό": {
        "reason": "glaux_name_lemma",
        "evidence": "GLAUx annotates Τοῦτο (adv) with this lemma; no "
                    "dictionary heads it (#108).",
    },
    "Αὕτῃ": {
        "reason": "glaux_name_lemma",
        "evidence": "GLAUx annotates Αὕτη (adv) with this lemma, a form of "
                    "οὗτος; no dictionary heads it (#108).",
    },
    "Τούτων": {
        "reason": "glaux_name_lemma",
        "evidence": "GLAUx annotates τούτων as a noun with this lemma, the "
                    "genitive plural of οὗτος; no dictionary heads it (#108).",
    },
    # Lemmas a treebank's automatic annotation built from an augmented,
    # reduplicated or misprinted form. No head list has them, but
    # ag_headwords.json does, so headword validation lets the corpus pairs
    # through. The forms keep a reading of their own through
    # REVIEWED_ABSENT_KEY_READINGS in build_lookup_db.py.
    "παραπαροινέω": {
        "reason": "corpus_lemmatizer_artifact",
        "evidence": "GLAUx and Diorisis file the perfect and pluperfect of "
                    "παροινέω under it (πεπαρῴνηκα, πεπαρῴνηκας, πεπαρῴνηκεν, "
                    "πεπαρῳνήκεσαν), reading the reduplication as a second "
                    "prefix; no dictionary heads it.",
    },
    "ἀμφιαμφισβητέω": {
        "reason": "corpus_lemmatizer_artifact",
        "evidence": "GLAUx files the double-augmented past tenses of "
                    "ἀμφισβητέω under it (ἠμφεσβήτουν 4, ἠμφεσβήτησεν 4, "
                    "ἠμφεσβήτησαν 3 and six more forms), and the dictionary "
                    "expansion carries the same readings; no dictionary "
                    "heads it.",
    },
    "γιχρώσκω": {
        "reason": "corpus_lemmatizer_artifact",
        "evidence": "GLAUx files χρωσθήσεται, the future passive of χρῴζω, "
                    "under it; no dictionary heads it.",
    },
    "ταθήχω": {
        "reason": "corpus_lemmatizer_artifact",
        "evidence": "GLAUx files τέθηκται, the perfect middle of θήγω, under "
                    "it (Diorisis: θήγω); no dictionary heads it.",
    },
    "χειρίω": {
        "reason": "corpus_lemmatizer_artifact",
        "evidence": "GLAUx files κεχειρικέναι, the perfect infinitive of "
                    "χειρίζω, under it; no dictionary heads it.",
    },
    "ἀρικνέομαι": {
        "reason": "corpus_lemmatizer_artifact",
        "evidence": "GLAUx files ἀρίκηται under it, a misprint of Ionic "
                    "ἀπίκηται (ἀφικνέομαι) in a Hippocratic text (ἄχρις ἂν "
                    "ἀρίκηται ἐς τοὺς ἑκατὸν σταδίους, TLG 0627.032); no "
                    "dictionary heads it.",
    },
    "διασμήγνυμι": {
        "reason": "corpus_lemmatizer_artifact",
        "evidence": "GLAUx files διέσμηκται, the perfect middle of διασμήχω, "
                    "under it; no dictionary heads it.",
    },
    # A head-list entry that is no word. The lookup has no row for it, so
    # the rule that repairs an unaccented expansion lemma to its accented
    # head would otherwise send word fragments to it (#127).
    "μός": {
        "reason": "dictionary_run_on_entry",
        "evidence": "The LSJ headword list carries it as the bare run-on "
                    "entry -μός, with no grammar and no gloss; no other list "
                    "has it. GLAUx files the line-break fragments μος, μον "
                    "and μῳ under the unaccented μος.",
    },
    # The same kind of fragment, with no dictionary behind it at all. It
    # gave the only reading of γον, which capitalized Γον (the ordinal Γ
    # with a superscript ον in printed Greek) then reached by lowercasing.
    "γος": {
        "reason": "line_break_fragment",
        "evidence": "No lexicon heads it (LSJ, DGE and the merged lexical "
                    "heads lack it); Greek Wiktionary has it only as the "
                    "genitive ending in the headers of σμῶδιξ and ὕσπληξ. "
                    "GLAUx files three tokens under it, all word fragments: "
                    "λό γος and λό γον split at a line break, and γος "
                    "repeated after σάλπιγγος.",
    },
}

# Lemmas that begin with a double rho, which no Greek word does: the past
# tenses of verbs in ῥ- double the rho after the augment (ἔρριψα), and a
# misread augment left the doubled rho on the lemma. Each gave the only row
# of one or two augmented spellings, so ἐρρίφθαι read ρροπίω while
# ἐρρῖφθαι and the accent-stripped ερριφθαι read ῥίπτω (#132).
_DOUBLE_RHO_LEMMAS = {
    "ρροπίω": "ἐρρίφθαι (dictionary expansion); ἐρρῖφθαι reads ῥίπτω",
    "ρραστωνάω": "ἐρραστώνησεν (dictionary expansion); ερραστωνησεν reads "
                 "ῥᾳστωνέω",
    "ρρυφάνω": "ἐρρύφανεν (dictionary expansion)",
    "ρρωζμαίω": "ἐρρωζμάιον (dictionary expansion)",
    "ρρωμαίζω": "ἐρρωμάιζον (dictionary expansion)",
    "ρρωννύω": "ἐρρώννυεν, ἐρρώννυέ (dictionary expansion)",
    "ρρᾳθύμνυηι": "ἐρρᾳθυμηένον (dictionary expansion)",
    "ρρῃίννυμι": "ἐρρῄισε (GLAUx)",
}
for _lemma, _forms in _DOUBLE_RHO_LEMMAS.items():
    REJECTED_LEMMA_TARGETS[_lemma] = {
        "reason": "double_rho_augment_artifact",
        "evidence": "No dictionary heads it and no Greek word begins with "
                    "ρρ: the doubled rho of an augmented past tense was "
                    f"kept on the lemma. Its only rows: {_forms}.",
    }

REJECTED_LEMMAS = frozenset(REJECTED_LEMMA_TARGETS)

# Closed-class pronominal heads. A capitalised corpus form whose lowercase
# spelling already reads as one of these (from a dictionary source) takes
# that reading rather than the corpus's: GLAUx gives sentence-initial Αὕτη
# the lemma Ἄτη, which the build lowercases to the real but wrong head ἄτη.
CLOSED_PRONOMINAL_LEMMAS = frozenset({
    "οὗτος", "ὅδε", "ἐκεῖνος", "αὐτός", "ἑαυτοῦ", "ἐμαυτοῦ", "σεαυτοῦ",
    "ἐγώ", "σύ", "ἡμεῖς", "ὑμεῖς", "ὅς", "ὅστις", "τίς", "τις", "πᾶς",
    "ἅπας", "εἷς", "οὐδείς", "μηδείς", "ἄλλος", "ἕτερος", "ὅσος", "οἷος",
    "σός", "ἐμός", "ἀλλήλων", "τοιοῦτος", "τοσοῦτος",
})
