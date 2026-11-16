"""Shipped rows after the Sophocles OCR fixes (#139) and the expansion of
the verb entries glossed by an equals sign (#136).

These read data/lookup.db as rebuilt for 1.3.27.
"""
import sqlite3

import pytest

from dilemma import core

MICRO = "\u00b5"


@pytest.fixture(scope="module")
def db():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    conn = sqlite3.connect(f"file:{core.LOOKUP_DB_PATH}?mode=ro", uri=True)
    yield conn
    conn.close()


def lemmas(db, form):
    """The Ancient readings of a key: lemmas of its `grc` and `all` rows."""
    return {text for (text,) in db.execute(
        "SELECT l.text FROM lookup k JOIN lemmas l ON k.lemma_id = l.id "
        "WHERE k.form = ? AND k.lang IN ('grc', 'all')", (form,))}


def lemma_rows(db, lemma):
    return db.execute(
        "SELECT COUNT(*) FROM lookup k JOIN lemmas l ON k.lemma_id = l.id "
        "WHERE l.text = ?", (lemma,)).fetchone()[0]


def test_no_ancient_row_is_spelled_with_the_micro_sign(db):
    # The Modern corpus rows (source hnc) are not part of the Ancient build.
    rows = db.execute(
        "SELECT s.code, COUNT(*) FROM lookup k JOIN lemmas l ON k.lemma_id = l.id "
        "JOIN sources s ON k.source_id = s.id "
        "WHERE (k.form LIKE ? OR l.text LIKE ?) AND s.code != 'hnc' "
        "GROUP BY s.code", (f"%{MICRO}%", f"%{MICRO}%")).fetchall()
    assert rows == []


@pytest.mark.parametrize("form,lemma", [
    ("μέρους", "μέρος"),
    ("σχηματοποιεῖ", "σχηματοποιέω"),
    ("γνωμοδότου", "γνωμοδότης"),
    ("γνωμοδοτῶν", "γνωμοδοτέω"),
])
def test_micro_sign_heads_read_with_mu(db, form, lemma):
    assert lemmas(db, form) == {lemma}


def test_the_hen_coop_has_its_accent(db):
    assert lemmas(db, "ὀρνιθεῖον") == {"ὀρνιθεῖον"}
    assert lemmas(db, "ὀρνιθεῖα") == {"ὀρνιθεῖον"}
    for misread in ("ὀρνιθείον", "ορνιθείον"):
        assert lemma_rows(db, misread) == 0
        assert lemmas(db, misread) == set()
    # the oblique cells it shares with the adjective stay the adjective's
    for form in ("ὀρνιθείου", "ὀρνιθείων", "ὀρνιθείῳ", "ὀρνιθείοις"):
        assert lemmas(db, form) == {"ὀρνίθειος"}


@pytest.mark.parametrize("misread", [
    "ούβρικος", "ῶνσος", "λημέντια", "λινυργία", "ἐπληπτικῶς", "αῦ",
    "*σχῆμα", "*πιάζω", "«ἀθεμιστία", "\u1fbdΑντινώ", "ἐκνεωτερίξω",
    "κατηγρρησείω", "ἐπιπραῦνω", "γονιμοποίός", "μυριοπαθῆς", "παραπλασμμός",
    # mu printed twice, and letters misread in heads the micro sign had hidden
    "κληρονομμέω", "διαπλασμμός", "σμμικρόφυλλος", "μεταναστείω", "γραμματείω",
    "τερμοτίζω", "κατακοιμεστής", "μανρότριχος", "ἀναμελωδέω",
    # a dropped nu in a verb glossed by an equals sign
    "ῥιψοκιδυνέω", "ριψοκιδυνέω",
    # the phi symbol (U+03D5) for the letter
    "Ἀπ\u03d5ία", "Ἀφ\u03d5ιανός", "Ἄ\u03d5\u03d5ιον",
    # a second accent, a wrong breathing or the OCR's capital
    "εἶλεώδης", "Ἕρμαῖος", "ίλλέκεβρα", "ϊεραρχέω", "Θεοτήρητος",
])
def test_misread_heads_are_no_lemmas(db, misread):
    assert lemma_rows(db, misread) == 0


@pytest.mark.parametrize("form,lemma", [
    ("Κουβρίκου", "Κούβρικος"),
    ("ἐκνεωτερίζει", "ἐκνεωτερίζω"),
    ("ἐπιπραΰνει", "ἐπιπραΰνω"),
    ("γονιμοποιός", "γονιμοποιός"),
    # a head the phrase rule used to skip: the paradigm is generated again
    ("κομιτίου", "κομίτιον"),
    ("δωρίτου", "δωρίτης"),
    ("ἑτερολέκτως", "ἑτερολέκτως"),
])
def test_corrected_and_one_word_heads_have_their_forms(db, form, lemma):
    assert lemmas(db, form) == {lemma}


@pytest.mark.parametrize("form,lemma", [
    ("ἀνακοσμοποιεῖ", "ἀνακοσμοποιέω"),
    ("ἀνακοσμοποιοῦσι", "ἀνακοσμοποιέω"),
    ("ἀμβλισκάνει", "ἀμβλισκάνω"),
    ("ἐπιφροντίζει", "ἐπιφροντίζω"),
    ("ἐπιθηριοῦται", "ἐπιθηριόομαι"),
])
def test_verbs_glossed_by_an_equals_sign_have_their_cells(db, form, lemma):
    assert lemmas(db, form) == {lemma}


@pytest.mark.parametrize("form,lemma", [
    # the older paradigm of a head that was respelled, under the present head
    ("ἀμαντεύτου", "ἀμάντευτος"),
    ("Θεοτηρήτων", "θεοτήρητος"),
    ("Γυμνοσοφισταῖς", "γυμνοσοφιστής"),
    ("ἱεραρχεῖν", "ἱεραρχέω"),
    # the corrected head's fresh paradigm
    ("ἰλλεκέβρας", "ἰλλέκεβρα"),
    ("ἐπιπραΰνει", "ἐπιπραΰνω"),
])
def test_rows_of_a_respelled_head_are_read_under_the_present_head(db, form, lemma):
    assert lemmas(db, form) == {lemma}


@pytest.mark.parametrize("form", [
    # an adjective in -ικός declined as a consonant stem, one in -ώδης and a
    # noun in -ότης declined like πολίτης, a noun in -ίς, -ίδος like πόλις
    "Θεαρχιξί", "Βληχηματώδου", "εἰλεώδου", "Θεοπρεπότου", "Εἰλωτέως",
    # built on another accent, on the rough breathing, without the diaeresis
    "ἐκκηρυκτοῦ", "οἱηκοφόρου", "Ἰουδαισμοῦ", "ἀραχνουφοῦ", "μυριοπαθαῖς",
])
def test_an_old_paradigm_the_entry_does_not_confirm_stays_out(db, form):
    assert lemmas(db, form) == set()


@pytest.mark.parametrize("form,lemma", [
    # a name the other lexica list only as a common word keeps its capital
    ("Κοπρωνύμου", "Κοπρώνυμος"),
    ("Νοητοῦ", "Νοητός"),
    ("Μιλίῳ", "Μίλιον"),
    ("Σκοπέλῳ", "Σκόπελος"),
    ("Μαϊουμᾶν", "Μαϊουμᾶς"),
    ("Ἀργυροῦ", "Ἀργυρός"),
    # the common words are untouched
    ("νοητοῦ", "νοητός"),
    ("σκοπέλου", "σκόπελος"),
])
def test_proper_names_keep_their_capital(db, form, lemma):
    assert lemmas(db, form) == {lemma}


@pytest.mark.parametrize("form,lemma", [
    ("κληρονομεῖ", "κληρονομέω"),
    ("μεταναστεύει", "μεταναστεύω"),
    ("ῥιψοκινδυνεῖ", "ῥιψοκινδυνέω"),
    ("ἀναμελῳδεῖ", "ἀναμελῳδέω"),
    ("μαυροτρίχου", "μαυρότριχος"),
    ("στρουθιομήλου", "στρουθιόμηλον"),
])
def test_heads_corrected_after_the_micro_sign_have_their_forms(db, form, lemma):
    assert lemmas(db, form) == {lemma}


@pytest.mark.parametrize("form,lemma", [
    # a word the OCR split in two, which the phrase rule used to skip
    ("κηνσοφύλακος", "κηνσοφύλαξ"),
    ("μοιχοσυνόδου", "μοιχοσύνοδος"),
    ("συμψάλλει", "συμψάλλω"),
    ("προνύσσει", "προνύσσω"),
])
def test_split_words_have_their_forms(db, form, lemma):
    assert lemmas(db, form) == {lemma}


@pytest.mark.parametrize("first", ["κηνσο", "μοιχο", "σωματο", "χρυσο", "ραφαν"])
def test_the_first_part_of_a_split_word_is_no_lemma(db, first):
    assert lemma_rows(db, first) == 0


def test_artefacts_of_the_older_tables_stay_out(db):
    assert lemmas(db, "ἑφρόντιζον") == set()
    # a dative the lexicon glosses with an equals sign is no verb
    assert lemmas(db, "ἐπισηκρήτουσιν") == set()
    assert lemma_rows(db, "Ἰεύω") == 0


@pytest.mark.parametrize("form,lemma", [
    # rows of other sources under a head the lexicon's OCR had misread
    ("σχήματός", "σχῆμα"),
    ("διανομὴν", "διανομή"),
    ("μελέτᾳ", "μελέτη"),
])
def test_other_sources_rows_of_a_reread_head_stay(db, form, lemma):
    assert lemmas(db, form) == {lemma}


@pytest.mark.parametrize("form,lemma", [
    # cells of μαντίον and μετάθετος, heads the micro sign had hidden
    ("μαντίων", "μάντις"),
    ("μεταθέτω", "μετατίθημι"),
    ("μετάθετε", "μετατίθημι"),
    # cells of ὑποπτερόω, παιδομανέω and ἐπεκλύω, verbs glossed by "="
    ("ὑποπτέρου", "ὑπόπτερος"),
    ("παιδομανεῖ", "παιδομανής"),
    ("ἐπέκλυε", "ἐπικλύω"),
])
def test_treebank_readings_keep_their_keys(db, form, lemma):
    assert lemmas(db, form) == {lemma}
    assert lemmas(db, "μαντίου") == {"μαντίον"}
