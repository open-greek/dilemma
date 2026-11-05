"""Reviewed inflections whose grammatical lemma differs from a related headword.

The imperfect of ἄπειμι (from εἶμι, 'go') belongs to that verb. Its use in
Attic as the future of ἀπέρχομαι concerns present forms, not these imperfects.
Keep the relationship separate from the grammatical analysis.
"""
from __future__ import annotations

import unicodedata
from dataclasses import asdict

from .paradigm import ParadigmSlot


APEIMI_EVIDENCE = (
    "https://atlas.perseus.tufts.edu/dictionaries/entry/"
    "urn:cite2:scaife-viewer:dictionaries.v1:lsj-n11710/",
    "https://en.wiktionary.org/w/index.php?title=ἄπειμι&oldid=93465794",
)

# Person and number follow ParadigmSlot. Both first- and third-person
# singular readings of ἀπῄειν survive. The circumflex spelling ἀπῇμεν is
# the form in Plato, Republic 327b, reported and independently parsed in
# https://github.com/open-greek/dilemma/issues/61.
APEIMI_IMPERFECT = {
    "ἀπῇα": (("1", "sg"),),
    "ἀπῄειν": (("1", "sg"), ("3", "sg")),
    "ἀπῄεις": (("2", "sg"),),
    "ἀπῄεισθα": (("2", "sg"),),
    "ἀπῄει": (("3", "sg"),),
    "ἀπῇτον": (("2", "du"),),
    "ἀπῄτην": (("3", "du"),),
    "ἀπῄμεν": (("1", "pl"),),
    "ἀπῇμεν": (("1", "pl"),),
    "ἀπῇτε": (("2", "pl"),),
    "ἀπῇσαν": (("3", "pl"),),
    "ἀπῄεσαν": (("3", "pl"),),
}


def reviewed_verb_grammar(form: str) -> dict | None:
    """Return fresh metadata for an exact, case-insensitive reviewed spelling.

    NFC handles decomposed text. Accents are retained so that ἀπῆμεν,
    from ἄπειμι meaning 'be absent', does not acquire the 'go' analysis.
    """
    persons = APEIMI_IMPERFECT.get(unicodedata.normalize("NFC", form).lower())
    if persons is None:
        return None
    analyses = [asdict(ParadigmSlot.verb_finite(
        voice="active", tense="imperfect", mood="indicative",
        person=person, number=number)) for person, number in persons]
    return {
        "lemma": "ἄπειμι",
        "base_lemma": "εἶμι",
        "analyses": analyses,
        "related_headwords": [{
            "lemma": "ἀπέρχομαι",
            "relation": "present_used_as_future",
            "dialect": "attic",
            "applies_to_input": False,
        }],
        "evidence": list(APEIMI_EVIDENCE),
    }


def correct_grammatical_lemma(form: str, lemma: str) -> str:
    """Correct the reviewed headword conflation without changing other owners."""
    if lemma == "ἀπέρχομαι" and unicodedata.normalize("NFC", form).lower() in APEIMI_IMPERFECT:
        return "ἄπειμι"
    return lemma
