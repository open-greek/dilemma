"""Readings of 1.3.32: the corrected corpus words of #161 and the wrong
answers left after 1.3.31 (#161, #162)."""

import unicodedata

import pytest

from dilemma import core
from dilemma.crasis import crasis_restorations, is_crasis_shaped
from dilemma.grc_reviewed_forms import GRC_REVIEWED_FORM_RECORDS, GRC_REVIEWED_FORMS
from dilemma.grc_reviewed_respellings import respelled_lemma
from dilemma.nonlexical import classify_nonlexical


# Helpers that need no data ---------------------------------------------------

def test_kai_compendium_is_a_word():
    # ϗ has no vowel but is καί (U+03D7, a letter).
    assert classify_nonlexical("ϗ") is None
    assert classify_nonlexical("Ϗ") is None
    assert classify_nonlexical("κζ") == "consonant-cluster"


@pytest.mark.parametrize("word, tiers", [
    ("κᾀκείνων", (("ἐκείνων",), ("ἀκείνων",))),
    ("κἄνθρωπον", (("ἄνθρωπον", "ἔνθρωπον"),)),
    ("Κᾆτα", (("εἶτα",), ("ἆτα",))),
    ("ταὐτόματόν", (("αὐτόματόν",),)),
    ("τοὔνομα", (("ὄνομα", "ἔνομα", "οὔνομα"),)),
    ("κᾄπειτ’", (("ἔπειτ’",), ("ἄπειτ’",))),
])
def test_crasis_restorations(word, tiers):
    word = unicodedata.normalize("NFC", word)
    assert crasis_restorations(word) == tuple(
        tuple(unicodedata.normalize("NFC", s) for s in tier) for tier in tiers)


@pytest.mark.parametrize("word", [
    "καί", "τοῦ", "κεἰ", "ἀκεῖ", "προὐταξάμην", "κάτω", "τὰ", "κᾷνταῦθα"])
def test_words_that_are_no_crasis(word):
    assert not is_crasis_shaped(word)


@pytest.mark.parametrize("form, given, lemma", [
    ("μάλιστ", "μάλα", "μάλιστα"), ("μάλισθ᾽", "μάλα", "μάλιστα"),
    ("μάλιστα", "μάλα", "μάλιστα"),
    ("προὐπάρχειν", "προυπάρχω", "προϋπάρχω"),
    ("προὔφαινον", "προυφαίνω", "προφαίνω"),
    ("ἐξηγόρασέ", "ἐξηγορέω", "ἐξαγοράζω"),
    ("εὐοΐ", "εὕω", "εὐοῖ"), ("εὖοἱ", "εὕω", "εὐοῖ"),
    ("βῆλα", "βάλλω", "βῆλον"), ("ἅνπερ", "ἄνπερ", "ὅσπερ"),
    ("προϋποστάσης", "προϋπόσταση", "προϋφίστημι"),
    ("προὑποστάσης", "προϋπόσταση", "προϋφίστημι"),
    ("παρώσασθε", "παρόω", "παρωθέω"),
])
def test_reviewed_readings(form, given, lemma):
    assert respelled_lemma(form, given) == lemma


@pytest.mark.parametrize("form, lemma", [
    ("μᾶλλον", "μάλα"), ("εὕει", "εὕω"), ("βάλε", "βάλλω"), ("εἶ", "εἶ"),
    # εὕω's optative; the Modern noun, with the accent on -πό-; ἄνπερ, if.
    ("εὕοιμι", "εὕω"), ("εὐοῖς", "εὕω"), ("προϋπόστασης", "προϋπόσταση"),
    ("προϋπόσταση", "προϋπόσταση"), ("ἄνπερ", "ἄνπερ"),
])
def test_reviewed_readings_leave_other_forms(form, lemma):
    assert respelled_lemma(form, lemma) == lemma


def test_reviewed_form_records():
    for form, record in GRC_REVIEWED_FORM_RECORDS.items():
        assert form == unicodedata.normalize("NFC", form)
        assert form == form.lower()
        assert record["lemma"] and record["reason"] and record["evidence"]
        assert GRC_REVIEWED_FORMS[form] == record["lemma"]


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


def test_reviewed_forms_have_no_row_of_their_own():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    for lang in ("all", "grc", "el"):
        table = core.LookupDB(core.LOOKUP_DB_PATH, lang=lang, respell=False)
        try:
            for form in GRC_REVIEWED_FORMS:
                for spelling in (form, form[:1].upper() + form[1:]):
                    assert table._get(spelling) is None, (lang, spelling)
        finally:
            table.close()


# (form, lemma) read untagged with lang="grc", as the corpus lemma step reads
# its vocabulary.
_UNTAGGED = [
    # Corrected corpus words with no answer in 1.3.31 (#161).
    ("προϋπηργμένου", "προϋπάρχω"), ("προϋφεστακόσιν", "προϋφίστημι"),
    ("προϋπέστησεν", "προϋφίστημι"), ("προϊστορήσαντες", "προϊστορέω"),
    ("προϋπαντῶντες", "προϋπαντάω"), ("προϋπεῖναι", "προΰπειμι"),
    ("προϋποτεθεῖσαν", "προϋποτίθημι"), ("προὐταξάμην", "προτάσσω"),
    ("κἀνακουφίζει", "ἀνακουφίζω"), ("κἄνθρωπον", "ἄνθρωπος"),
    ("κᾀετὸς", "ἀετός"), ("τοὐπονείδιστον", "ἐπονείδιστος"),
    ("τοὐπιχειρεῖν", "ἐπιχειρέω"), ("τἀμφίβληστρα", "ἀμφίβληστρον"),
    ("τἀπίσημα", "ἐπίσημος"), ("ταὐτόματόν", "αὐτόματον"),
    ("κἀμαθῆ", "ἀμαθής"), ("τοὐπτάνειον", "ὀπτανεῖον"),
    ("κἀνιδρύουσα", None),   # ἀνιδρύω or ἐνιδρύω
    ("τἀναντί", None),       # ἐναντίος (τἀναντί’) or ἔναντι
    ("προὐξένησε", "προξενέω"), ("Τοὐντεῦθεν", "ἐντεῦθεν"),
    ("ἡνίδε", "ἠνίδε"), ("ἡνιδὲ", "ἠνίδε"), ("ἐπεῖπερ", "ἐπείπερ"),
    ("Θυραῖόν", "Θυραῖον"), ("Κλόνιός", "Κλόνιος"), ("τἀρχαῖ", None),
    ("περατοῦντες", "περατόω"),
    ("ποιοῦντες", "ποιέω"), ("γελοῦντας", "γελάω"),
    ("ἀπήχθημεν", "ἀπάγω"), ("Ἐνήργηκε", "ἐνεργέω"),
    ("ταυτοποιὸν", "ταὐτοποιός"), ("αὐχεῖθ", "αὐχέω"),
    # Wrong answers (#161).
    ("προὐπάρχειν", "προϋπάρχω"), ("προϋποτιθέντες", "προϋποτίθημι"),
    ("προϋποστάσης", "προϋφίστημι"), ("προὐννέπω", "προεννέπω"),
    ("προὔφαινον", "προφαίνω"), ("Κᾆτα", "εἶτα"), ("Κἀκεῖ", "ἐκεῖ"),
    ("εὐοΐ", "εὐοῖ"), ("ἆισμα", "ᾆσμα"), ("Ἰάπυγές", "Ἰᾶπυξ"),
    ("μάλιστ", "μάλιστα"), ("μάλισθ", "μάλιστα"), ("μάλιστ᾽", "μάλιστα"),
    ("μάλιστα", "μάλιστα"),
    ("Κλήμεντά", "Κλήμης"), ("Πολύευκτός", "Πολύευκτος"), ("Ἆγίς", "Ἆγις"),
    ("Ἀστυπάλαιά", "Ἀστυπάλαια"), ("πολυσήμαντόν", "πολυσήμαντος"),
    # Wrong answers (#162).
    ("Εἴπατέ", "εἶπον"), ("Κάλεσόν", "καλέω"),
    ("Ἀποκοτταβίζουσί", "ἀποκοτταβίζω"),
    ("κᾀκείνων", "ἐκεῖνος"), ("κᾄπειτ’", "ἔπειτα"), ("κᾀνταῦθ’", "ἐνταῦθα"),
    ("ὑδαταινούσῃσιν", "ὑδαταίνω"), ("παραλειφθεῖσί", "παραλείπω"),
    ("βῆλά", "βῆλον"), ("φυράσωντι", "φυράω"), ("παρώσασθέ", "παρωθέω"),
    ("πὸτ", "πρός"), ("ἅνπερ", "ὅσπερ"),
    ("καλοῦντόν", "καλέω"), ("ἀνασκινδυλευομένων", "ἀνασκινδυλεύω"),
    # Kept as in 1.3.31.
    ("Σῶνόν", "Σῶνος"), ("Ἔνετοί", "Ἔνετος"), ("ὅτιπέρ", "ὅτιπερ"),
    ("επείπερ", "ἐπείπερ"), ("ὡςπερ", "ὥσπερ"), ("κἀκεῖ", "ἐκεῖ"),
    ("κᾆτα", "εἶτα"), ("κἀγώ", "ἐγώ"), ("Εἶθ", "εἶτα"), ("Μεθ", "μετά"),
    # εἶγε is εἴ γε in the corpus (εἶγε μέλλοι τις), the accented article
    # before an enclitic is left as the untagged call leaves οἱ, and the
    # particle words read their heads (#162 review).
    ("εἶγε", "εἴγε"), ("οἵ", "οἵ"), ("αἵ", "αἵ"), ("οἳ", "ὅς"),
    ("τοῖοςδε", "τοιόσδε"), ("Εὥσπερ", "ἕωσπερ"),
    # No -ουντ- reading where the -έω spelling has no head of its own.
    ("ἐπινούντες", "ἐπινέω"),
]


@pytest.mark.parametrize("word, wrong", [
    ("τοὐτ", ("ἐπί", "ἔτι")), ("τἀτὸ", ("ἀπατός",)), ("τοὔμ", ("ὄμμα",)),
])
def test_short_crasis_stems_read_no_other_word(word, wrong):
    assert _model("grc").lemmatize(word, guess=False) not in wrong


def test_refused_reading_keeps_the_answer(monkeypatch):
    # A reading the citation policy refuses leaves the answer as it was, and
    # lemmatize_verbose() does not fail on it.
    model = _model("grc")
    words = ["κᾀκείνων", "προὐταξάμην", "μάλιστ"]
    before = model.lemmatize_batch(words, guess=False)
    apply = model._apply_convention
    monkeypatch.setattr(
        model, "_apply_convention",
        lambda lemma, *args, **kwargs: (None if lemma in before
                                        else apply(lemma, *args, **kwargs)))
    after = model.lemmatize_batch(words, guess=False)
    assert after == [model.lemmatize(word, guess=False) for word in words]
    assert not set(after) & set(before)
    for word in words:
        model.lemmatize_verbose(word, guess=False)


def test_untagged_answers():
    model = _model("grc")
    words = [word for word, _ in _UNTAGGED]
    want = [lemma for _, lemma in _UNTAGGED]
    assert model.lemmatize_batch(words, guess=False) == want
    assert [model.lemmatize(word, guess=False) for word in words] == want
    for word, lemma in _UNTAGGED:
        candidates = model.lemmatize_verbose(word, guess=False)
        assert (candidates[0].lemma if candidates else None) == lemma, word


# (form, UPOS, lemma) read with lang="all" and the tag, as the annotations
# read the corpus.
_TAGGED = [
    ("ϗ", "X", "καί"),
    ("Διὰ", "NOUN", "διά"), ("Διὰ", "ADJ", "διά"), ("Διὰ", "PROPN", "διά"),
    ("Δία", "NOUN", "Ζεύς"), ("Κἀκεῖ", "PROPN", "ἐκεῖ"),
    ("μάλιστα", "ADV", "μάλιστα"), ("μάλιστ’", "ADV", "μάλιστα"),
    ("πὸτ", "ADV", "πρός"), ("πὸτ", "ADP", "πρός"),
    ("οθ", "NUM", None), ("υπ", "NUM", None), ("αντ", "NUM", "ἀντί"),
    ("ουδ", "NUM", "οὐδέ"), ("ἀμφ", "NUM", "ἀμφί"),
    ("Κἀκεῖσε", "ADV", "ἐκεῖσε"), ("Ταὐτὸν", "PRON", "αὐτός"),
    ("Κἂν", "INTJ", "ἄν"), ("τοὐναν", "ADJ", None),
    ("βδελύττεσθαί", "NOUN", "βδελύσσομαι"),
    ("Κατάσχοντες", "VERB", "κατέχω"),
    ("προϋπόσταση", "NOUN", "προϋπόσταση"),
    ("προϋπάρξῃ", "VERB", "προϋπάρχω"), ("προϊστάμενα", "VERB", "προΐστημι"),
    ("κᾀκείνων", "VERB", "ἐκεῖνος"), ("κᾄπειτ’", "VERB", "ἔπειτα"),
    ("κᾀνταῦθ’", "ADV", "ἐνταῦθα"),
    ("σχήματό", "NOUN", None), ("νηπίαχαί", "ADJ", None),
    ("ἐνανθρωπήσει", "NOUN", "ἐνανθρώπησις"),
    ("ἐνανθρωπήσει", "VERB", "ἐνανθρωπέω"),
    ("θερίσει", "NOUN", "θερίζω"), ("σώσει", "NOUN", "σῴζω"),
    ("προὔφαινον", "VERB", "προφαίνω"),
    ("προὑπάρχει", "VERB", "προϋπάρχω"),
    ("Εἴπατέ", "PROPN", "εἶπον"),
]


def test_tagged_answers():
    words = [form for form, _, _ in _TAGGED]
    tags = [upos for _, upos, _ in _TAGGED]
    want = [lemma for _, _, lemma in _TAGGED]
    model = _model("all")
    assert model.lemmatize_batch_pos(words, tags, guess=False) == want
    assert [model.lemmatize_pos(word, upos, guess=False)
            for word, upos in zip(words, tags)] == want


def test_modern_words_need_no_ancient_model():
    # Under lang="all", a vernacular word with προ and ι or υ keeps its
    # answer, and only a word the readings may read loads the Ancient model.
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    _MODELS.clear()
    model = core.Dilemma(lang="all")
    words = ["προϊόν", "προίκα", "προϋπολογισμός", "προϋπόθεση",
             "προϊστάμενος", "καλημέρα"]
    assert model.lemmatize_batch(words, guess=False) == [
        model._lemmatize_steps(word, guess=False) for word in words]
    assert "_ancient_sibling" not in model.__dict__


def test_vernacular_model_reads_no_crasis():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    _MODELS.clear()
    model = core.Dilemma(lang="el")
    assert model._whole_word_fixes("κᾀκείνων", "x") == "x"
    assert model._whole_word_fixes("προϋπηργμένου", None) is None
