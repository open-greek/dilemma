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
# but expansion rows built from the Koine forms. LSJ heads ἠθμός, a
# strainer, with the smooth breathing, and gives ἡθμός as a variant of
# inscriptions and manuscripts. ὀρθοπνοϊκός, αἱμοπτυϊκός, ἐπιληψία and
# ἐναπόληψις are LSJ's spellings of the other four.
REVIEWED_LEMMA_RESPELLINGS = {
    "λήμπω": "λαμβάνω", "ἀναλήμπω": "ἀναλαμβάνω", "ἡθμός": "ἠθμός",
    "ὀρθοπνοικός": "ὀρθοπνοϊκός", "αἱμοπτυικός": "αἱμοπτυϊκός",
    "ἐπιλημψία": "ἐπιληψία", "ἐναπόλημψις": "ἐναπόληψις",
}

# (initial letters of the form, lowercase and without marks; the lemma a
# source gives; the lemma). διῃρημένος and διῄρηται, also printed without
# the subscript, are the perfect of διαιρέω; GLAUx gave διηρημένον and
# διήρηται to διερέω.
REVIEWED_STEM_READINGS = (
    (("διηρ",), "διερέω", "διαιρέω"),
)

# πλάττω is the Attic spelling of πλάσσω, while the Attic of πλήσσω is
# πλήττω and its aorist passive has πλαγ-: no form of πλήσσω or of its
# compounds has the letters πλαττ. Expansion rows and GLAUx gave πλαττόμεναι,
# πλάττεται and the rest to πλήσσω, and the compounds alike (διαπλάττει
# διαπλήσσω, καταπλάττειν καταπλήσσω, ἐμπλάττεται ἐμπλήσσω).
_PLASSO_LETTERS, _PLESSO, _PLASSO = "πλαττ", "πλήσσω", "πλάσσω"


def _letters(form: str) -> str:
    nfd = unicodedata.normalize("NFD", form)
    return "".join(ch for ch in nfd if not unicodedata.combining(ch)).lower()


def respelled_lemma(form: str, lemma: str) -> str:
    """The reviewed lemma for ``form`` read as ``lemma``: ``lemma`` itself
    unless one of the records above corrects it."""
    if lemma in REVIEWED_LEMMA_RESPELLINGS:
        return REVIEWED_LEMMA_RESPELLINGS[lemma]
    letters = _letters(form)
    for initials, given, corrected in REVIEWED_STEM_READINGS:
        if lemma == given and letters.startswith(initials):
            return corrected
    if lemma.endswith(_PLESSO) and _PLASSO_LETTERS in letters:
        return lemma[:-len(_PLESSO)] + _PLASSO
    return lemma
