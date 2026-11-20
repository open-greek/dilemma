"""The shipped lookup files no Ancient form under a hyphenated lemma (#137).

Up to 1.3.26 about a million rows had a lemma with a hyphen between two
letters: a dictionary head keyed as printed (Βοιωτ-άρχης), or a treebank's
compound verb with the preverb apart (ἐπί-εἰσκομίζω). These tests read
data/lookup.db as rebuilt for 1.3.27.
"""
import sqlite3

import pytest

from dilemma import core
from dilemma.citation import hyphenated_lemma

# Sources whose lemmas are spelled with a hyphen: Modern Greek compounds
# (Ελλάδα-Βουλγαρία) and an LBG headword as that lexicon prints it.
HYPHENATED_SOURCES = {"hnc", "lbg_headwords"}


@pytest.fixture(scope="module")
def lookup():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    connection = sqlite3.connect(f"file:{core.LOOKUP_DB_PATH}?mode=ro", uri=True)
    yield connection
    connection.close()


def _lemmas(lookup, form):
    return {lemma for (lemma,) in lookup.execute(
        "SELECT m.text FROM lookup l JOIN lemmas m ON m.id = l.lemma_id "
        "WHERE l.form = ?", (form,))}


def test_only_modern_compounds_and_lbg_keep_a_hyphenated_lemma(lookup):
    hyphenated = [(lemma_id, text) for lemma_id, text in lookup.execute(
        "SELECT id, text FROM lemmas") if hyphenated_lemma(text)]
    rows = {}
    for lemma_id, text in hyphenated:
        for (source,) in lookup.execute(
                "SELECT s.code FROM lookup l JOIN sources s "
                "ON s.id = l.source_id WHERE l.lemma_id = ?", (lemma_id,)):
            rows.setdefault(source, []).append(text)
    assert set(rows) <= HYPHENATED_SOURCES, {
        source: lemmas[:5] for source, lemmas in rows.items()
        if source not in HYPHENATED_SOURCES}
    assert len(rows.get("lbg_headwords", ())) <= 1


@pytest.mark.parametrize("form,lemma", [
    # AGDT's preverb lemma, joined to the listed compound
    ("ἐπεισεκόμισαν", "ἐπεισκομίζω"),
    ("κατεκληρούχησαν", "κατακληρουχέω"),
    ("καθεσθέντα", "καθέζομαι"),
    ("ἐμπεριέλαβεν", "ἐμπεριλαμβάνω"),
    ("προϊστορημένων", "προϊστορέω"),
    # the list's accent for a short verb in composition
    ("Παρεισιόντι", "παρείσειμι"),
    # a divided dictionary head, under the joined headword
    ("Βοιωτάρχου", "Βοιωτάρχης"),
    ("καθηδύνει", "καθηδύνω"),
])
def test_form_is_filed_under_the_joined_lemma(lookup, form, lemma):
    assert _lemmas(lookup, form) == {lemma}


@pytest.mark.parametrize("form", [
    # the divided spelling of a dictionary head is no key
    "Βοιωτ-άρχου", "Βακχ-είας", "καθ-ηδύνει",
    # a form whose only reading was a compound no head list has
    "κατενίσατο",
])
def test_form_has_no_row(lookup, form):
    assert _lemmas(lookup, form) == set()
