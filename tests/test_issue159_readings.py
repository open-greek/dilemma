"""Wrong answers left after 1.3.30, reported on #159."""

import pytest

from dilemma import core
from dilemma.core import (
    _has_misplaced_marks, _is_modern_or_acronym, _is_verb_citation_spelling,
    _single_consonants, to_standard_elision)
from dilemma.grc_reviewed_respellings import respelled_lemma


# Helpers that need no data ---------------------------------------------------

@pytest.mark.parametrize("spelling, misplaced", [
    ("παροͅχέω", True),       # ο with a subscript: an augmented οι read back
    ("αἱͅδω", False),         # a subscript under ι: a stray of the encoding
    ("εὐρίͅσκω", False),
    ("ἐͅγώ", True),           # a subscript under ε
    ("Ρ̓ᾶρος", True),         # a smooth breathing on a capital rho
    ("χὠμός", False),         # a coronis: a crasis head of its own
    ("ζῳδιακός", False), ("ᾄδω", False), ("σῴζω", False),
    ("αἰδέω", False), ("Ῥωμαῖος", False), ("ὄις", False),
])
def test_marks_greek_spelling_puts_nowhere(spelling, misplaced):
    assert _has_misplaced_marks(spelling) is misplaced


def test_rejected_answer_may_have_a_smooth_initial_upsilon():
    # OCR gives ὐ for ὑ; a head never has it.
    assert _has_misplaced_marks("ὐπατεύω") is True
    assert _has_misplaced_marks("ὐπατεύω", initial_smooth=False) is False
    assert _has_misplaced_marks("παροͅχέω", initial_smooth=False) is True


@pytest.mark.parametrize("lemma, modern", [
    ("όπερα", True), ("ευτός", True), ("ΕΕΔΑ", True), ("ΑΠΣΙ", True),
    ("ὅσπερ", False), ("αὐτός", False), ("Αἰγύπτιος", False),
    ("ῥέω", False), ("μετά", False),
])
def test_modern_and_acronym_readings(lemma, modern):
    assert _is_modern_or_acronym(lemma) is modern


@pytest.mark.parametrize("lemma, verb", [
    # An iota subscript inside the word is the verb's own.
    ("σῴζω", True), ("ᾄδω", True), ("θνῄσκω", True), ("λύω", True),
    ("ἐγώ", False), ("λόγῳ", False), ("Ζεύς", False),
])
def test_verb_citation_spellings(lemma, verb):
    assert _is_verb_citation_spelling(lemma) is verb


def test_doubled_consonants_count_once():
    assert _single_consonants("αππιωνα") == "απιωνα"
    assert _single_consonants("βαλλω") == "βαλω"
    assert _single_consonants("ζωοο") == "ζωοο"


@pytest.mark.parametrize("printed, read", [
    # A grave never stands before an elision: the mark is a stray.
    ("ἂν´", "ἂν"), ("γὰρ´", "γὰρ"), ("ὃν´", "ὃν"), ("παρὰ´", "παρὰ"),
    # Elided words keep the mark as ’.
    ("ἐπειδ´", "ἐπειδ’"), ("ὅπερ´", "ὅπερ’"), ("τοῦτ´", "τοῦτ’"),
])
def test_a_grave_before_the_mark_makes_it_a_stray(printed, read):
    assert to_standard_elision(printed) == read


@pytest.mark.parametrize("form, given, corrected", [
    ("ἀναπλασθῆναι", "ἀναπολάζω", "ἀναπλάσσω"),
    ("ἀνακεκρᾶσθαι", "ἀνακράζω", "ἀνακεράννυμι"),
    ("προεπίστασθαι", "προεφίστημι", "προεπίσταμαι"),
    ("πλειότεροι", "πλέως", "πολύς"),
    ("θαυματουργήσαντος", "θαυμασιουργέω", "θαυματουργέω"),
    ("καταπνεύσαντος", "καταπονέω", "καταπνέω"),
    ("διαρραγεῖσαν", "διαράσσω", "διαρρήγνυμι"),
    ("ἀποφράξαντα", "ἀποφράζω", "ἀποφράσσω"),
    ("επειπερ", "ἐπεί", "ἐπείπερ"),
    # The perfects of the αἱρέω compounds read as λέγω compounds.
    ("ἐξῄρητο", "ἐκλέγω", "ἐξαιρέω"),
    ("Συνῄρητο", "συλλέγω", "συναιρέω"),
    # πλάσσω, not πλήσσω, wherever the form has πλασσ- or πλαττ-.
    ("ἔπλασσε", "πλήσσω", "πλάσσω"),
    ("καταπλάσσοντα", "καταπλήσσω", "καταπλάσσω"),
    ("ἀποπλάσσεται", "ἀποπλήσσω", "ἀποπλάσσομαι"),
    # The subscript under ε of a crasis read back onto the word.
    ("κᾀγώ", "ἐͅγώ", "ἐγώ"), ("κάπειτ᾽", "ἔͅπειτα", "ἔπειτα"),
    ("Περὶ", "περι", "περί"),
])
def test_reviewed_respellings(form, given, corrected):
    assert respelled_lemma(form, given) == corrected


@pytest.mark.parametrize("form, lemma", [
    # Other forms keep what they read.
    ("ἐξελέγετο", "ἐκλέγω"), ("ἀναπολάζει", "ἀναπολάζω"),
    ("ἐπεί", "ἐπεί"), ("πλήσσει", "πλήσσω"), ("ἐπλήγη", "πλήσσω"),
])
def test_reviewed_respellings_leave_other_forms(form, lemma):
    assert respelled_lemma(form, lemma) == lemma


# Readings of the shipped data ------------------------------------------------

_MODELS = {}


def _model(lang):
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    if lang not in _MODELS:
        _MODELS.clear()
        _MODELS[lang] = (core.Dilemma(lang="all", resolve_articles=True)
                         if lang == "all" else
                         core.Dilemma(lang="grc", skip_numeral_ordinals=True))
    return _MODELS[lang]


# (form, lemma) read untagged with lang="grc", as the corpus lemma step reads
# its vocabulary.
_UNTAGGED = [
    # The participle resolver checks the accent and the verb's class.
    ("Μάνεντος", None), ("Βάλεντι", None), ("γέλωντος", None),
    ("συγχῶσαι", "συγχόω"), ("περιχῶσαι", "περιχόω"),
    ("νοοῦντά", "νοέω"), ("ἀθυμοῦντά", "ἀθυμέω"),
    ("καταπνεύσαντος", "καταπνέω"),
    ("σῴζοντος", "σῴζω"), ("θνῄσκοντος", "θνήσκω"),
    ("θαυματουργήσαντος", "θαυματουργέω"),
    ("διαρραγεῖσαν", "διαρρήγνυμι"), ("ἀναπλασθέντα", "ἀναπλάσσω"),
    ("ἀποφράξαντα", "ἀποφράσσω"),
    # The enclitic-accent step.
    ("προεπίστασθαί", "προεπίσταμαι"), ("Συνήσθημέν", "συνήδομαι"),
    ("ὅτιπέρ", "ὅτιπερ"), ("Σῶνόν", "Σῶνος"), ("πλειότεροί", "πολύς"),
    ("ἔλλειπές", "ἐλλείπω"), ("Ἔνετοί", "Ἔνετος"),
    ("ἀναπλασθῆναί", "ἀναπλάσσω"), ("ἀνακεκρᾶσθαί", "ἀνακεράννυμι"),
    ("Συνῄρητό", "συναιρέω"), ("Κλόνιός", "Κλόνιος"), ("ἥττηταί", "ἡσσάομαι"),
    ("Ἐποίησέ", "ποιέω"),
    ("ἡρμήνευσέ", "ἑρμηνεύω"), ("ἐλεοῦντός", "ἐλεέω"), ("Ῥεῖάν", "Ῥεία"), ("Ἐβίωνός", "Ἐβίων"),
    ("δυοκαιεξηκοντάεδρά", "δυοκαιεξηκοντάεδρος"),
    ("ἐφαπτόμενοί", "ἐφάπτω"), ("ῥᾷόν", "ῥᾴδιος"),
    ("Σιδώνιόν", "Σιδώνιος"), ("βακχεύεσθαί", "βακχεύω"),
    ("Ἡσίοδός", "Ἡσίοδος"), ("πολυστέφανός", "πολυστέφανος"),
    ("Ἀππίωνά", "Ἀπίων"), ("ἐκφῦναί", "ἐκφύω"), ("ἄνθετό", None),
    # Elided stems and stray marks.
    ("Εἶθ", "εἶτα"), ("Μεθ", "μετά"), ("ἂν´", "ἄν"), ("γὰρ´", "γάρ"),
    ("ὃν´", "ὅς"), ("ἐπειδ´", "ἐπειδή"), ("ἐπεῖδ’", "ἐφοράω"),
    ("κᾀγὼ", "ἐγώ"), ("επείπερ", "ἐπείπερ"), ("ὡςπερ", "ὥσπερ"),
    ("τοιόςδε", "τοιόσδε"), ("ὡςδὲ", "ὡς"), ("ἡνίδε", "ἠνίδε"), ("ἀποπλάσσεται", "ἀποπλάσσομαι"),
    # An acronym is no elided stem.
    ("ΜΕΘ", "ΜΕΘ"),
    # -ευσ- of πνέω and πλέω.
    ("ἀποπλεύσαντος", "ἀποπλέω"), ("ἀναπνεύσαντες", "ἀναπνέω"),
]


def test_untagged_answers():
    words = [form for form, _ in _UNTAGGED]
    want = [lemma for _, lemma in _UNTAGGED]
    model = _model("grc")
    assert model.lemmatize_batch(words, guess=False) == want
    assert [model.lemmatize(word, guess=False) for word in words] == want


@pytest.mark.parametrize("word, lemma", [
    ("ἐφαπτόμενοί", "ἐφάπτω"), ("βακχεύεσθαί", "βακχεύω"),
    ("Ἡσίοδός", "Ἡσίοδος"), ("Σιδώνιόν", "Σιδώνιος"),
    ("Εἶθ", "εἶτα"), ("Μεθ", "μετά"),
])
def test_verbose_reads_as_lemmatize(word, lemma):
    model = _model("grc")
    assert model.lemmatize_verbose(word, guess=False)[0].lemma == lemma
    # lemmatize_verbose() reads the word in NFC, as lemmatize() does.
    import unicodedata
    decomposed = unicodedata.normalize("NFD", word)
    assert model.lemmatize_verbose(decomposed, guess=False)[0].lemma == lemma


# (form, UPOS, lemma) read with lang="all" and the tag, as the annotations
# read the corpus.
_TAGGED = [
    # Capitalized prepositions tagged PROPN, and elided stems under any tag.
    ("Διὰ", "PROPN", "διά"), ("Δι’", "PROPN", "διά"),
    ("Δι´", "PROPN", "διά"), ("Δι´", "INTJ", "διά"), ("Δι´", "ADV", "διά"),
    ("δι´", "NOUN", "διά"), ("Μετ’", "PROPN", "μετά"),
    ("Μεθ’", "PROPN", "μετά"), ("Μεθ´", "PROPN", "μετά"),
    ("Μεθ'", "PROPN", "μετά"), ("Παρ´", "PROPN", "παρά"),
    ("Κατὰ", "PROPN", "κατά"), ("Περὶ", "PROPN", "περί"),
    ("Εἰς", "PROPN", "εἰς"),
    # The name stays a name.
    ("Δία", "PROPN", "Ζεύς"), ("Διὸς", "PROPN", "Ζεύς"),
    ("Δι’", "NOUN", "Ζεύς"),
    ("οὐδ’", "NOUN", "οὐδέ"), ("οὔδ’", "NOUN", "οὐδέ"),
    ("ὑφ’", "NOUN", "ὑπό"), ("καθ’", "ADV", "κατά"),
    ("καθ᾽", "CCONJ", "κατά"), ("παρ’", "ADJ", "παρά"),
    ("παρ᾽", "CCONJ", "παρά"), ("ἄρ’", "VERB", "ἄρα"),
    ("μάλ’", "ADJ", "μάλα"), ("Εἶθ", "ADV", "εἶτα"), ("Μεθ", "ADP", "μετά"),
    ("παρὰ´", "VERB", "παρά"), ("δὲ´", "VERB", "δέ"),
    ("ἐπειδ´", "CCONJ", "ἐπειδή"), ("κάπειτ᾽", "ADV", "ἔπειτα"),
    ("ὅπερ´", "PRON", None), ("Ἀπσ´", "X", None), ("εἐδ´", "X", None),
    # The dictionary-head fallback.
    ("παρῳχηκότα", "VERB", None), ("Ἤιδει", "VERB", None),
    ("προὑπάρχει", "VERB", "προϋπάρχω"), ("οἷνος", "NOUN", None),
    ("ΦΛΕΓΟΝΤΟΣ", "PROPN", "φλέγω"), ("ὐπατεύων", "VERB", "ὑπατεύω"),
    ("ὀίσσατο", "VERB", None),
    # Which breathing the word has does not count: OCR confuses them.
    ("Ἅβραμος", "PROPN", "Ἄβραμος"), ("ἧλθεν", "VERB", "ἔρχομαι"),
    ("ἀπλοϊκῶς", "ADV", "ἁπλοϊκός"), ("εὐρίσκεται", "VERB", "εὑρίσκω"),
    ("ᾁδετε", "VERB", None),
    ("Ρᾶροϲ", "PROPN", None), ("ταυτολογεῖν", "VERB", "ταὐτολογέω"),
    ("ὁταν", "CCONJ", "ὅταν"),
    ("ζωδιακοῦ", "ADJ", "ζῳδιακός"), ("ἀρπάσας", "VERB", "ἁρπάζω"),
    ("Θράκῃ", "PROPN", "Θρᾴκη"), ("Ρωμαῖοι", "PROPN", "Ῥωμαῖος"),
    ("τροχαϊκὸν", "ADJ", "τροχαϊκός"), ("ἴσταται", "VERB", "ἵστημι"),
    ("κᾀγὼ", "PRON", "ἐγώ"),
    # Action nouns need a head of LSJ or of Wiktionary.
    ("θερίσει", "NOUN", "θερίζω"), ("σώσει", "NOUN", "σῴζω"),
    ("παραλλάσσει", "NOUN", "παραλλάσσω"),
    ("συμπτύξει", "NOUN", "σύμπτυξις"),
    ("προϋπάρξει", "NOUN", "προΰπαρξις"),
    # πλάσσω and the αἱρέω perfects.
    ("ἔπλασσε", "VERB", "πλάσσω"), ("καταπλάσσοντά", "VERB", "καταπλάσσω"),
    ("ἀποπλάσσεται", "VERB", "ἀποπλάσσομαι"),
    ("ἐξῄρητο", "VERB", "ἐξαιρέω"), ("ἐξῄρηται", "VERB", "ἐξαιρέω"),
    ("ἐξῃρημένον", "VERB", "ἐξαιρέω"), ("ἐξῃρημένην", "VERB", "ἐξαιρέω"),
    # The relative's accented forms under PRON, and the -περ words.
    ("ὃ", "PRON", "ὅς"), ("ὅ", "PRON", "ὅς"), ("οἵ", "PRON", "ὅς"),
    ("Ὃ", "PRON", "ὅς"), ("ὃ", "DET", "ὁ"),
    ("ὅπερ", "PRON", "ὅσπερ"), ("ἅπερ", "PRON", "ὅσπερ"),
    ("ὡσπερ", "CCONJ", "ὥσπερ"), ("ὡςπερ", "SCONJ", "ὥσπερ"), ("ὅτιπερ", "PRON", "ὅτιπερ"),
    # A polytonic word reads the Ancient spelling of a Modern answer.
    ("ὅποτε", "PRON", "ὁπότε"),
]


def test_tagged_answers():
    words = [form for form, _, _ in _TAGGED]
    tags = [upos for _, upos, _ in _TAGGED]
    want = [lemma for _, _, lemma in _TAGGED]
    model = _model("all")
    assert model.lemmatize_batch_pos(words, tags, guess=False) == want
    assert [model.lemmatize_pos(word, upos, guess=False)
            for word, upos in zip(words, tags)] == want


@pytest.mark.parametrize("word, upos, lemma", [
    # The accent of a preposition's letters counts: Δία is Zeus.
    ("Δία", "PROPN", "Ζεύς"), ("Διὰ", "PROPN", "διά"), ("Περὶ", "PROPN", "περί"),
    # Capitals are left to the other steps.
    ("ΜΕΘ", "PROPN", "ΜΕΘ"),
])
def test_tagged_answers_of_the_ancient_model(word, upos, lemma):
    model = _model("grc")
    assert model.lemmatize_pos(word, upos, guess=False) == lemma
    assert model.lemmatize_batch_pos([word], [upos], guess=False) == [lemma]


def test_modern_elided_words_keep_their_modern_reading():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    model = core.Dilemma(lang="el")
    assert model.lemmatize("απ’", guess=False) == "από"
    assert model.lemmatize("ΜΕΘ", guess=False) == "ΜΕΘ"
