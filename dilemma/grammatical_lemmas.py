"""Reviewed εἶμι inflections retain grammar beside connected headwords.

Dictionary and corpus conventions can group εἶμι and its compounds with
ἔρχομαι and its compounds. A headword convention does not supply the
grammatical tense or erase the independently inflected verb's identity.
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

# The additional first-plural readings were checked separately after the
# wider lookup/POS audit. Do not construct unattested compounds by prefixing
# this table, or remove accents to match homonyms of εἰμί ('be').
_REVIEWED_FAMILIES = [
    ("ἄπειμι", "ἀπέρχομαι", APEIMI_IMPERFECT, APEIMI_EVIDENCE),
    ("εἶμι", "ἔρχομαι", {"ᾖμεν": (("1", "pl"),),
                        "ᾔειμεν": (("1", "pl"),)},
     ("https://en.wiktionary.org/wiki/εἶμι",
      "https://en.wiktionary.org/wiki/ἔρχομαι")),
    ("ἔξειμι", "ἐξέρχομαι", {"ἐξῇμεν": (("1", "pl"),)},
     ("https://en.wiktionary.org/wiki/ἔξειμι",
      "https://en.wiktionary.org/wiki/ἐξέρχομαι")),
    ("εἴσειμι", "εἰσέρχομαι", {"εἰσῇμεν": (("1", "pl"),)},
     ("https://atlas.perseus.tufts.edu/dictionaries/entry/"
      "urn:cite2:scaife-viewer:dictionaries.v1:lsj-n31333/",
      "https://en.wiktionary.org/wiki/εἶμι")),
    ("πρόσειμι", "προσέρχομαι", {"προσῇμεν": (("1", "pl"),),
                               "πρόσῃμεν": (("1", "pl"),)},
     ("https://en.wiktionary.org/wiki/πρόσειμι",
      "https://en.wiktionary.org/wiki/προσέρχομαι")),
    ("πάρειμι", "παρέρχομαι", {"παρῇμεν": (("1", "pl"),),
                             "πάρῃμεν": (("1", "pl"),)},
     ("https://en.wiktionary.org/wiki/πάρειμι",
      "https://atlas.perseus.tufts.edu/dictionaries/entry/"
      "urn:cite2:scaife-viewer:dictionaries.v1:lsj-n79484/")),
    ("πρόειμι", "προέρχομαι", {"προῇμεν": (("1", "pl"),)},
     ("https://en.wiktionary.org/wiki/πρόειμι",
      "https://en.wiktionary.org/wiki/εἶμι")),
]
_REVIEWED_FORMS = {
    form: (lemma, headword, persons, evidence)
    for lemma, headword, forms, evidence in _REVIEWED_FAMILIES
    for form, persons in forms.items()
}


def reviewed_verb_lemmas() -> dict[str, str]:
    """Grammatical owners for the same reviewed forms in lookup builds."""
    return {form: reading[0] for form, reading in _REVIEWED_FORMS.items()}


def reviewed_verb_grammar(form: str) -> dict | None:
    """Return fresh metadata for an exact, case-insensitive reviewed spelling.

    NFC handles decomposed text. Accents are retained so that ἀπῆμεν,
    from ἄπειμι meaning 'be absent', does not acquire the 'go' analysis.
    """
    reading = _REVIEWED_FORMS.get(unicodedata.normalize("NFC", form).lower())
    if reading is None:
        return None
    lemma, headword, persons, evidence = reading
    analyses = [asdict(ParadigmSlot.verb_finite(
        voice="active", tense="imperfect", mood="indicative",
        person=person, number=number)) for person, number in persons]
    return {
        "lemma": lemma,
        "base_lemma": "εἶμι",
        "analyses": analyses,
        "related_headwords": [{
            "lemma": headword,
            "relation": "present_used_as_future",
            "dialect": "attic",
            "applies_to_input": False,
        }],
        "evidence": list(evidence),
    }


def correct_grammatical_lemma(form: str, lemma: str) -> str:
    """Correct the reviewed headword conflation without changing other owners."""
    reading = _REVIEWED_FORMS.get(unicodedata.normalize("NFC", form).lower())
    if reading is not None and lemma == reading[1]:
        return reading[0]
    return lemma
