"""Reviewed lemma spellings that the lookup and POS-table builds correct.

Some sources give a form a lemma that LSJ spells otherwise, or the lemma of
another verb whose letters are close. ``build_lookup_db.py`` rewrites the
readings of the lookup tables with ``respelled_lemma`` and
``build/build_treebank_pos_lookup.py`` its POS cells, so the lookup and the
POS lookup give the same lemma (#157).
"""

from __future__ import annotations

import unicodedata

# Lemma -> the spelling LSJ heads. λημφθεῖσα and ἀναλημφθεῖσα are the Koine
# spellings of ληφθεῖσα and ἀναληφθεῖσα: λήμπω and ἀναλήμπω are no verbs,
# but expansion rows built from the Koine forms. ὀρθοπνοϊκός, αἱμοπτυϊκός,
# ἐπιληψία and ἐναπόληψις are LSJ's spellings of the other four. (LSJ heads
# the strainer ἡθμός, with the manuscripts' ἠθμός as a variant, so
# ἠθμοῦ keeps the lemma ἡθμός.)
REVIEWED_LEMMA_RESPELLINGS = {
    "λήμπω": "λαμβάνω", "ἀναλήμπω": "ἀναλαμβάνω",
    "ὀρθοπνοικός": "ὀρθοπνοϊκός", "αἱμοπτυικός": "αἱμοπτυϊκός",
    "ἐπιλημψία": "ἐπιληψία", "ἐναπόλημψις": "ἐναπόληψις",
    # LSJ and the DGE head the middle ἀποπλάσσομαι; ἀποπλάσσω came from
    # respelling the treebank cells of ἀποπλήσσω (#159).
    "ἀποπλάσσω": "ἀποπλάσσομαι",
    # GLAUx gives the capitalized Περὶ and Περί the bare letters περι, which
    # no dictionary heads: they are the preposition (#159).
    "περι": "περί",
    # The heads of the προϋ- verbs and adjectives have the diaeresis; rows
    # that drop it give a second, misspelled head beside the right one
    # (προὐπάρχειν προυπάρχω, προὐπάρχων προϋπάρχω; #161). GLAUx's
    # προυφαίνω is the lemma it gives the imperfect προὔφαινε, the crasis
    # of προέφαινε: the verb is προφαίνω (προϋφαίνω, weave before, has the
    # diaeresis; #162).
    "προυπάρχω": "προϋπάρχω", "προυποτίθημι": "προϋποτίθημι",
    "προυπαλείφω": "προϋπαλείφω", "προυφίστημι": "προϋφίστημι",
    "προύπειμι": "προΰπειμι", "πραυντικός": "πραϋντικός",
    "προυφαίνω": "προφαίνω",
    # Rows that read a form as itself, or as a lemma no dictionary heads,
    # where the dictionaries head another spelling (#161): προὐννέπω is the
    # crasis of προεννέπω, προέργου the uncontracted προὔργου, ἆισμα the
    # unsubscripted ᾆσμα, εὐύδροις the dative of εὔυδρος, Ἰάπυγες the
    # plural of Ἰᾶπυξ, τοῦνομα the crasis of τὸ ὄνομα (as τοὔνομα in the
    # crasis table), κυανόχρωτος the stem of κυανόχρως, and προνοοῦμαι
    # the middle of προνοέω, which the lookup reads for προνοοῦμαι itself.
    "προὐννέπω": "προεννέπω", "προέργου": "προὔργου", "ἆισμα": "ᾆσμα",
    "εὐύδροις": "εὔυδρος", "Ἰάπυγες": "Ἰᾶπυξ", "τοῦνομα": "ὄνομα",
    "κυανόχρωτος": "κυανόχρως", "προνοοῦμαι": "προνοέω",
}

# (initial letters of the form, lowercase and without marks; the lemma a
# source gives; the lemma). διῃρημένος and διῄρηται, also printed without
# the subscript, are the perfect of διαιρέω; GLAUx gave διηρημένον and
# διήρηται to διερέω.
REVIEWED_STEM_READINGS = (
    (("διηρ",), "διερέω", "διαιρέω"),
    # Expansion rows that gave the forms of one verb to another whose LSJ
    # entry names it (#159): ἀναπλασθῆναι and ἀναπλάσας are ἀναπλάσσω,
    # not ἀναπολάζω; ἀνακεκρᾶσθαι is the perfect of ἀνακεράννυμι (ἀνακράζω
    # has ἀνακέκραγα); προεπίστασθαι and προεπιστάμενος are προεπίσταμαι,
    # know beforehand (προεφίστημι has προεπιστῆσαι); πλειότερος is the
    # comparative of πολύς, not of πλέως; θαυματουργέω is LSJ's own head
    # beside θαυμασιουργέω; καταπνεύσας is καταπνέω, not καταπονέω;
    # διαρραγείς is διαρρήγνυμι, not διαράσσω; ἀποφράξας is ἀποφράσσω, not
    # ἀποφράζω, whose aorist is ἀπέφρασα.
    (("αναπλασ", "ανεπλασ"), "ἀναπολάζω", "ἀναπλάσσω"),
    (("ανακεκρασ", "ανακεκραμ"), "ἀνακράζω", "ἀνακεράννυμι"),
    (("προεπισταμ", "προεπιστασθ", "προεπιστασαι", "προεπιστανται",
      "προεπιστατ"),
     "προεφίστημι", "προεπίσταμαι"),
    (("πλειοτερ", "πλειοτατ"), "πλέως", "πολύς"),
    (("θαυματουργ", "τεθαυματουργ", "εθαυματουργ"), "θαυμασιουργέω",
     "θαυματουργέω"),
    (("καταπνευ", "κατεπνευ"), "καταπονέω", "καταπνέω"),
    (("διαρραγ", "διερραγ"), "διαράσσω", "διαρρήγνυμι"),
    (("αποφραξ", "απεφραξ", "αποφραχθ", "απεφραχθ"), "ἀποφράζω", "ἀποφράσσω"),
    # The folded key επειπερ of the unaccented επείπερ reads ἐπεί, as the
    # tagged cell did before 1.3.30 (#159).
    (("επειπερ",), "ἐπεί", "ἐπείπερ"),
    # The -περ words of the real-OCR evaluation (#159). ὅπερ, as ἅπερ,
    # ὅνπερ and the other cases, is ὅσπερ, LSJ's head and the Gorman
    # trees' lemma, where a Wiktionary row gives ὅπερ and the treebank
    # cells ὅς; ὡσπερ and ὡςπερ are ὥσπερ, not ὡς; and the AGDT cells give
    # ὅτιπερ the ὅ of ὅστις, where LSJ heads ὅτιπερ itself.
    (("οπερ",), "ὅπερ", "ὅσπερ"),
    (("οπερ", "ηπερ", "απερ", "οιπερ", "αιπερ", "ουπερ", "ωπερ", "ησπερ",
      "ονπερ", "ηνπερ", "ωνπερ", "οισπερ", "αισπερ", "ουσπερ", "ασπερ",
      "ανπερ"),
     "ὅς", "ὅσπερ"),
    (("ωσπερ", "ωςπερ"), "ὡς", "ὥσπερ"),
    (("οτιπερ",), "ὅ", "ὅτιπερ"),
    # Rows that read the relative's accented ἅ, αἵ and οἵ as themselves:
    # they are ὅς, as ὅ, ἥ and οἵ are (#159).
    (("α",), "ἅ", "ὅς"), (("αι",), "αἵ", "ὅς"), (("οι",), "οἵ", "ὅς"),
    # Wrong answers of the 1.3.30 corpus generation (#159), each a form of
    # the second verb by its stem: ὑποσχεθείς is the aorist passive of
    # ὑπισχνέομαι, παραδεχθείς of παραδέχομαι, ἀναρρηθείς of ἀνεῖπον and
    # στοιχηθείς of στοιχέω; ἁπλοῦντες is ἁπλόω; εἰρηνεύσας εἰρηνεύω;
    # ἐγχρώσας ἐγχρώννυμι; περιχώσας and συγχῶσαι have the -ωσ- of περιχόω
    # and συγχόω (χέω has ἔχεα); ἐνιζήσας is ἐνιζάνω, συρράξας συρράσσω,
    # ἐπιρρίψας ἐπιρρίπτω, ἐπαφείς ἐπαφίημι, φλοίουσα φλοίω; καταλείψας
    # and ἀπολείψας are the late sigmatic aorist of καταλείπω and ἀπολείπω,
    # παραλειφθείς the aorist passive of παραλείπω; καλλιώτερος is the late
    # comparative of καλός, θάτερος the crasis of ὁ ἕτερος, δάκτυλα the
    # plural of δάκτυλος (δάχτυλο is the Modern word), πορπατοῦσι the
    # vernacular περιπατοῦσι, and απήρασιν the vernacular third plural of
    # the aorist of ἀπαίρω (ἀπῆραν).
    (("υποσχεθ",), "ὑπέχω", "ὑπισχνέομαι"),
    (("παραδεχθ",), "παραδείκνυμι", "παραδέχομαι"),
    (("αναρρηθ",), "ἀναιρέω", "ἀνεῖπον"),
    (("στοιχηθ",), "στοιχίζω", "στοιχέω"),
    (("απλουντ", "απλουσ"), "ἀπολόω", "ἁπλόω"),
    (("ειρηνευσ",), "εἰρηνέω", "εἰρηνεύω"),
    (("εγχρωσ",), "ἐγχράω", "ἐγχρώννυμι"),
    (("περιχωσ",), "περιχέω", "περιχόω"),
    (("συγχωσ",), "συγχέω", "συγχόω"),
    (("ενιζησ",), "ἐνζάω", "ἐνιζάνω"),
    (("συρραξ",), "συναράσσω", "συρράσσω"),
    (("επιρριψ",), "ἐπιρριπτέω", "ἐπιρρίπτω"),
    (("επαφεντ", "επαφεισ"), "ἐφάπτω", "ἐπαφίημι"),
    (("φλοιουσ", "φλοιοντ"), "φλοιόω", "φλοίω"),
    (("καταλειψ",), "καταλείβω", "καταλείπω"),
    (("απολειψ",), "ἀπολείβω", "ἀπολείπω"),
    (("παραλειφθ",), "παραλείφω", "παραλείπω"),
    (("καλλιωτερ",), "κάλλιον", "καλός"),
    (("θατερο",), "θατέρως", "ἕτερος"),
    (("δακτυλ",), "δάχτυλο", "δάκτυλος"),
    (("πορπατ",), "πορπάω", "περιπατέω"),
    (("απηρασι",), "ἀπεράω", "ἀπαίρω"),
    # Rows of the bare spellings the enclitic step reads (#159): συνήσθημεν
    # is the aorist of συνήδομαι, παραινοῦντι παραινέω, and παραλειφθεῖσι
    # the aorist passive of παραλείπω.
    (("συνησθ",), "συνέζομαι", "συνήδομαι"),
    # κᾀντεῦθεν, the crasis of καὶ ἐντεῦθεν, read back as ἐͅντεῦθα.
    (("καντευθεν",), "ἐντεῦθα", "ἐντεῦθεν"),
    (("παραινουντ",), "παρανέω", "παραινέω"),
    (("παραλειφθ",), "παραλαμβάνω", "παραλείπω"),
    # The elided μάλιστ’ and μάλισθ’, and μάλιστα under ADV, are the
    # superlative μάλιστα, which LSJ and Cunliffe head and the untagged
    # lookup reads; GLAUx folds it into μάλα, the Gorman trees and the AGDT
    # do not (μάλιστα and its elided forms: μάλιστα 520 times, μάλα 23 in
    # the Gorman trees; 260 and 45 in the AGDT) (#161).
    (("μαλιστ", "μαλισθ"), "μάλα", "μάλιστα"),
    # Wrong answers of the corpus generation (#161, #162), each a form of
    # the second word by its letters: ἐξηγόρασε is the aorist of ἐξαγοράζω
    # (ἐξηγορέω has -ησ-), συναπιοῦσα the participle of συνάπειμι (εἶμι),
    # περιειλήσας of περιειλέω, προϋποστάσης, with the accent on -στά-,
    # the aorist participle of προϋφίστημι (the Modern noun προϋπόσταση
    # has it on -πό-), the word εὐοῖ alone the Bacchic cry (εὕοι, εὕοιμι
    # are εὕω's optative), πολυσήμαντος LSJ's own head beside πολύσημος,
    # βῆλα the Byzantine βῆλον, curtain (βάλλω has no βηλ-), τιμῶσα, τιμῶσι the
    # participle and the present of τιμάω, φυράσω the future of φυράω
    # (φύρω has φύρσω), ὑδαταινούσῃσιν the participle of ὑδαταίνω (the row
    # reads the crasis artefact ὑδαταινούσει), and παρώσασθε the aorist of
    # παρωθέω.
    (("εξηγορασ",), "ἐξηγορέω", "ἐξαγοράζω"),
    (("συναπι",), "συναπέρχομαι", "συνάπειμι"),
    (("περιειλη",), "περιείλω", "περιειλέω"),
    (("προυποστάσ",), "προϋπόσταση", "προϋφίστημι"),
    (("ευοι$",), "εὕω", "εὐοῖ"),
    (("πολυσημαντ",), "πολύσημος", "πολυσήμαντος"),
    (("βηλ",), "βάλλω", "βῆλον"),
    (("τιμωσ",), "τιμή", "τιμάω"),
    (("φυρασ",), "φύρω", "φυράω"),
    (("υδαταινουσ",), "ὑδαταινούσει", "ὑδαταίνω"),
    (("παρωσ",), "παρόω", "παρωθέω"),
    # ἅνπερ is the Doric accusative of ὅσπερ (ἥνπερ), not ἄνπερ, if (with
    # the rough breathing only; _ROUGH_RELATIVES; #162).
    (("ανπερ",), "ἄνπερ", "ὅσπερ"),
)

# The perfect of αἱρέω and its compounds has ᾑρη- (ἐξῄρηται, ἐξῃρημένον,
# συνῄρητο), and no form of λέγω or of its compounds has the letters ηρη:
# its perfect is εἴρηκα, εἴλεγμαι. GLAUx and the expansion rows gave these
# perfects to the λέγω compound with the same preverb (#159).
_LEGO_AIREO_COMPOUNDS = {
    "ἐκλέγω": "ἐξαιρέω", "συλλέγω": "συναιρέω", "διαλέγω": "διαιρέω",
    "προλέγω": "προαιρέω", "παραλέγω": "παραιρέω", "ἀναλέγω": "ἀναιρέω",
    "περιλέγω": "περιαιρέω", "καταλέγω": "καθαιρέω", "ἀπολέγω": "ἀφαιρέω",
    "ὑπολέγω": "ὑφαιρέω", "ἐπιλέγω": "ἐφαιρέω", "λέγω": "αἱρέω",
}
_AIREO_PERFECT_LETTERS = "ηρη"

# πλάττω is the Attic spelling of πλάσσω, while the Attic of πλήσσω is
# πλήττω and its aorist passive has πλαγ-: no form of πλήσσω or of its
# compounds has the letters πλαττ. Expansion rows and GLAUx gave πλαττόμεναι,
# πλάττεται and the rest to πλήσσω, and the compounds alike (διαπλάττει
# διαπλήσσω, καταπλάττειν καταπλήσσω, ἐμπλάττεται ἐμπλήσσω).
# The same holds for πλασσ- (ἔπλασσε, καταπλάσσοντα; #159).
_PLASSO_LETTERS, _PLESSO, _PLASSO = ("πλαττ", "πλασσ"), "πλήσσω", "πλάσσω"


def _letters(form: str) -> str:
    nfd = unicodedata.normalize("NFD", form)
    return "".join(ch for ch in nfd if not unicodedata.combining(ch)).lower()


def _accented_letters(form: str) -> str:
    """*form* in lowercase with its accents (a grave read as an acute) and
    no other mark."""
    nfd = unicodedata.normalize("NFD", form.lower()).replace("\u0300", "\u0301")
    return unicodedata.normalize("NFC", "".join(
        ch for ch in nfd
        if not unicodedata.combining(ch) or ch in "\u0301\u0342"))


def _stem_reads(form: str, letters: str, initials: tuple) -> bool:
    """Whether *form* opens with one of a stem reading's *initials*: its
    bare letters, or the letters with their accents for an initial written
    with one (προυποστάσ), or the whole word for one ending in $ (ευοι$)."""
    for initial in initials:
        if initial.endswith("$"):
            if letters == initial[:-1]:
                return True
        elif initial != _letters(initial):
            if _accented_letters(form).startswith(initial):
                return True
        elif letters.startswith(initial):
            return True
    return False


_STEM_READING_LEMMAS = frozenset(given for _, given, _ in REVIEWED_STEM_READINGS)


def _without_epsilon_subscript(lemma: str) -> str:
    """*lemma* without the iota subscripts that stand under an ε."""
    out, last = [], ""
    for ch in unicodedata.normalize("NFD", lemma):
        if ch == "\u0345" and last == "ε":
            continue
        if not unicodedata.combining(ch):
            last = ch.lower()
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


# The relative read from its own letters needs the word's rough breathing:
# ἆ and ἄ, with a smooth one, are the interjection (#159).
_ROUGH_RELATIVES = frozenset(("ἅ", "αἵ", "οἵ", "ἄνπερ"))


def respelled_lemma(form: str, lemma: str) -> str:
    """The reviewed lemma for ``form`` read as ``lemma``: ``lemma`` itself
    unless one of the records above corrects it."""
    if lemma in REVIEWED_LEMMA_RESPELLINGS:
        return REVIEWED_LEMMA_RESPELLINGS[lemma]
    if "\u0345" in lemma and len(_letters(lemma)) > 2:
        # An iota subscript under ε, which no Greek word has, is the crasis
        # ᾀ of κᾀγώ or κᾀπειτα read back onto the word: ἐͅγώ is ἐγώ (#159).
        # A precomposed ᾳ, ῃ or ῳ has no separate U+0345 in NFC. The bare
        # artefact ἐͅ (κᾀν, ᾐ) is no word either way and stays as it is.
        lemma = _without_epsilon_subscript(lemma)
    if lemma in _STEM_READING_LEMMAS:
        letters = _letters(form)
        for initials, given, corrected in REVIEWED_STEM_READINGS:
            if (lemma == given and _stem_reads(form, letters, initials)
                    and (given not in _ROUGH_RELATIVES
                         or "\u0314" in unicodedata.normalize("NFD", form))):
                return corrected
    elif lemma.endswith(_PLESSO):
        letters = _letters(form)
        if any(stem in letters for stem in _PLASSO_LETTERS):
            return respelled_lemma(form, lemma[:-len(_PLESSO)] + _PLASSO)
    elif lemma in _LEGO_AIREO_COMPOUNDS:
        if _AIREO_PERFECT_LETTERS in _letters(form):
            return _LEGO_AIREO_COMPOUNDS[lemma]
    return lemma
