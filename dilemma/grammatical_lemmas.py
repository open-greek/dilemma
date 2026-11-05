"""Reviewed verbal inflections retain grammar beside connected headwords.

Dictionary and corpus conventions can group εἶμι and its compounds with
ἔρχομαι and its compounds. A headword convention does not supply the
grammatical tense or erase the independently inflected verb's identity.
"""
from __future__ import annotations

import unicodedata
from copy import deepcopy
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

LSJ_EIMI = ("https://atlas.perseus.tufts.edu/dictionaries/entry/"
            "urn:cite2:scaife-viewer:dictionaries.v1:lsj-n31131/")
LSJ_OIDA = ("https://atlas.perseus.tufts.edu/dictionaries/entry/"
            "urn:cite2:scaife-viewer:dictionaries.v1:lsj-n72202/")
LSJ_EIMI_BE = ("https://atlas.perseus.tufts.edu/dictionaries/entry/"
               "urn:cite2:scaife-viewer:dictionaries.v1:lsj-n31130/")
DIORISIS = "https://figshare.com/articles/dataset/The_Diorisis_Ancient_Greek_Corpus/6187256"


def _persons(*cells):
    """Index explicitly reviewed spellings by their paradigm cells.

    No prefix or accent variants are generated here. A repeated spelling,
    such as first- and third-singular ᾔειν, retains both analyses.
    """
    positions = (("1", "sg"), ("2", "sg"), ("3", "sg"), ("2", "du"),
                 ("3", "du"), ("1", "pl"), ("2", "pl"), ("3", "pl"))
    assert len(cells) == len(positions)
    result = {}
    for forms, position in zip(cells, positions):
        for form in forms.split():
            result.setdefault(form, []).append(position)
    return {form: tuple(readings) for form, readings in result.items()}


# The Attic tables were reviewed separately, including their recessive
# accents. Do not derive additional compounds or strip accents across
# homonyms of εἰμί ('be'). LSJ confirms the underlying εἶμι paradigm;
# the individual Wiktionary tables supply the listed compound spellings.
_REVIEWED_FAMILIES = [
    ("ἄπειμι", "ἀπέρχομαι", APEIMI_IMPERFECT, APEIMI_EVIDENCE),
    ("εἶμι", "ἔρχομαι", _persons(
        "ᾖα ᾔειν ἤϊα ἤια", "ᾔεις ᾔεισθα", "ᾔει ᾔειν ᾖε ᾖεν ἴε ἴεν",
        "ᾖτον ᾔειτον", "ᾔτην ᾐείτην ἴτην", "ᾖμεν ᾔειμεν ᾔομεν ἤιομεν ἦιμεν",
        "ᾖτε ᾔειτε", "ᾖσαν ᾔεσαν ᾔεισαν εἶσαν ᾖν ἴν ἴσαν"),
     (LSJ_EIMI, "https://en.wiktionary.org/w/index.php?title=εἶμι&oldid=93357206")),
    ("ἔξειμι", "ἐξέρχομαι", _persons(
        "ἐξῇα ἐξῄειν", "ἐξῄεισθα ἐξῄεις", "ἐξῄειν ἐξῄει", "ἐξῇτον",
        "ἐξῄτην", "ἐξῇμεν", "ἐξῇτε", "ἐξῇσαν ἐξῄεσαν"),
     (LSJ_EIMI, "https://en.wiktionary.org/wiki/ἔξειμι")),
    ("εἴσειμι", "εἰσέρχομαι", _persons(
        "εἰσῄειν", "", "εἰσῄειν εἰσῄει", "", "", "εἰσῇμεν", "",
        "εἰσῇσαν εἰσῄεσαν"),
     ("https://atlas.perseus.tufts.edu/dictionaries/entry/"
      "urn:cite2:scaife-viewer:dictionaries.v1:lsj-n31333/",
      "https://en.wiktionary.org/wiki/εἶμι")),
    ("πρόσειμι", "προσέρχομαι", _persons(
        "πρόσῃα προσῄειν", "προσῄεισθα προσῄεις", "προσῄειν προσῄει",
        "πρόσῃτον", "προσῄτην", "πρόσῃμεν προσῇμεν", "πρόσῃτε προσῇτε",
        "πρόσῃσαν προσῄεσαν προσῇσαν"),
     (LSJ_EIMI, "https://en.wiktionary.org/w/index.php?title=πρόσειμι&oldid=91588095")),
    ("πάρειμι", "παρέρχομαι", _persons(
        "πάρῃα παρῄειν παρήϊα παρήϊά", "παρῄεισθα παρῄεις", "παρῄειν παρῄει", "πάρῃτον",
        "παρῄτην", "πάρῃμεν παρῇμεν", "πάρῃτε", "πάρῃσαν παρῄεσαν παρῇσαν"),
     ("https://en.wiktionary.org/wiki/πάρειμι",
      "https://atlas.perseus.tufts.edu/dictionaries/entry/"
      "urn:cite2:scaife-viewer:dictionaries.v1:lsj-n79484/")),
    ("πρόειμι", "προέρχομαι", _persons(
        "προῄειν", "", "προῄειν προῄει", "", "", "προῇμεν", "", ""),
     ("https://bailly.app/proeimi", LSJ_EIMI)),
    ("ἔπειμι", "ἐπέρχομαι", _persons(
        "ἐπῇα ἐπῄειν", "ἐπῄεισθα ἐπῄεις", "ἐπῄειν ἐπῄει ἐπήϊεν", "ἐπῇτον",
        "ἐπῄτην", "ἐπῄμεν", "ἐπῇτε", "ἐπῇσαν ἐπῄεσαν ἐπήισαν ἐπήϊσαν"),
     (LSJ_EIMI, "https://en.wiktionary.org/wiki/ἔπειμι")),
    ("κάτειμι", "κατέρχομαι", _persons(
        "κάτῃα κατῄειν", "κατῄεισθα κατῄεις", "κατῄειν κατῄει κατήιεν", "κάτῃτον",
        "κατῄτην", "κάτῃμεν", "κάτῃτε", "κάτῃσαν κατῄεσαν"),
     (LSJ_EIMI, "https://en.wiktionary.org/wiki/κάτειμι")),
    ("περίειμι", "περιέρχομαι", _persons(
        "περίῃα περιῄειν", "περιῄεισθα περιῄεις", "περιῄειν περιῄει", "περίῃτον",
        "περιῄτην", "περίῃμεν", "περίῃτε", "περίῃσαν περιῄεσαν"),
     (LSJ_EIMI, "https://en.wiktionary.org/wiki/περίειμι")),
    ("δίειμι", "διέρχομαι", _persons(
        "διῄειν", "διῄεισθα", "διῄειν διῄει", "", "", "διῇμεν", "", ""),
     (LSJ_EIMI, "https://atlas.perseus.tufts.edu/dictionaries/entry/"
      "urn:cite2:scaife-viewer:dictionaries.v1:lsj-n26473/")),
    ("διέξειμι", "διεξέρχομαι", _persons(
        "διεξῄειν", "διεξῄεις", "διεξῄειν διεξῄει διεξήϊε διεξήιε", "", "",
        "διεξῇμεν διεξῄειμεν", "", "διεξῄεσαν"),
     ("https://bailly.app/diexeimi", "https://atlas.perseus.tufts.edu/lemma/32717/")),
    ("ἄνειμι", "ἀνέρχομαι", _persons(
        "ἀνῄειν ἀνήϊον ἀνήιον", "", "ἀνῄειν ἀνῄει", "", "", "", "",
        "ἀνήϊον ἀνήιον"),
     ("https://bailly.app/aneimi", LSJ_EIMI, DIORISIS)),
    ("σύνειμι", "συνέρχομαι", _persons(
        "", "", "συνῄει", "", "συνίτην", "", "", "συνῇσαν συνῄεσαν σύνισαν ξύνισαν"),
     ("https://bailly.app/syneimi", "https://en.wiktionary.org/wiki/σύνειμι", DIORISIS)),
]


def _grammar(lemma, base, persons, evidence, *, tense="imperfect",
             mood="indicative", related=()):
    return {
        "lemma": lemma, "base_lemma": base,
        "analyses": [asdict(ParadigmSlot.verb_finite(
            voice="active", tense=tense, mood=mood, person=p, number=n))
            for p, n in persons],
        "related_headwords": list(related), "evidence": list(evidence),
    }


_REVIEWED_FORMS = {}
_SHARED_PARADIGMS = {"διέξειμι", "ἄνειμι", "σύνειμι"}
for _lemma, _headword, _forms, _evidence in _REVIEWED_FAMILIES:
    for _form, _people in _forms.items():
        _REVIEWED_FORMS.setdefault(_form, []).append(_grammar(
            _lemma, "εἶμι", _people, _evidence, related=[{
                "lemma": _headword,
                "relation": ("shared_paradigm" if _lemma in _SHARED_PARADIGMS
                             else "present_used_as_future"),
                "dialect": "attic", "applies_to_input": False}]))

_OIDA_EVIDENCE = (LSJ_OIDA, "https://en.wiktionary.org/wiki/οἶδα")
for _form, _people in _persons(
        "ᾔδειν ᾔδη", "ᾔδεις ᾔδησθα", "ᾔδει ᾔδειν", "ᾖστον ᾔδετον",
        "ᾔστην ᾐδέτην", "ᾖσμεν ᾔδεμεν", "ᾖστε ᾔδετε", "ᾖσαν ᾔδεσαν").items():
    _REVIEWED_FORMS.setdefault(_form, []).append(_grammar(
        "οἶδα", "οἶδα", _people, _OIDA_EVIDENCE, tense="pluperfect"))
_REVIEWED_FORMS["ἴσαν"].append(_grammar(
    "οἶδα", "οἶδα", (("3", "pl"),), _OIDA_EVIDENCE, tense="pluperfect"))
for _readings in _REVIEWED_FORMS.values():
    for _reading in _readings:
        if _reading["lemma"] == "οἶδα":
            _reading["semantic_tense"] = "imperfect"

# ᾖσαν has three lexical readings, not three interchangeable headwords.
# The contracted singing form is recorded under the database's ἀείδω
# citation. Its οἶδα analysis is morphologically pluperfect, even though
# οἶδα and its pluperfect commonly have present and imperfect meanings.
_REVIEWED_FORMS["ᾖσαν"].append(_grammar(
    "ἀείδω", "ἀείδω", (("3", "pl"),),
    ("https://en.wiktionary.org/w/index.php?title=ᾖσαν&oldid=89169001",
     "https://en.wiktionary.org/wiki/ἀείδω"), tense="aorist", related=[{
         "lemma": "ᾄδω", "relation": "contracted_present", "dialect": "attic",
         "applies_to_input": False}]))

_REVIEWED_FORMS["ἴσθι"] = [
    _grammar("εἰμί", "εἰμί", (("2", "sg"),),
             (LSJ_EIMI_BE, "https://en.wiktionary.org/wiki/ἴσθι"),
             tense="present", mood="imperative"),
    _grammar("οἶδα", "οἶδα", (("2", "sg"),),
             (LSJ_OIDA, "https://en.wiktionary.org/wiki/ἴσθι"),
             tense="perfect", mood="imperative"),
]
_REVIEWED_FORMS["ἦν"] = [
    _grammar("εἰμί", "εἰμί", (("1", "sg"), ("3", "sg")),
             (LSJ_EIMI_BE, "https://en.wiktionary.org/wiki/εἰμί")),
    _grammar("ἠμί", "ἠμί", (("1", "sg"),),
             ("https://en.wiktionary.org/wiki/ἠμί",)),
]
_REVIEWED_FORMS["ἴσθι"][1]["semantic_tense"] = "present"
# LSJ distinguishes Doric plural morphology from singular syntax used
# with a plural subject; do not silently classify every such subject as
# evidence for a third-plural parse.
_doric = asdict(ParadigmSlot.verb_finite(voice="active", tense="imperfect",
                                       mood="indicative", person="3", number="pl"))
_doric["dialect"] = "doric"
_REVIEWED_FORMS["ἦν"][0]["analyses"].append(_doric)

# A compound can coincide with a different verb, not just a different
# citation convention. Keep the independent owners and their tenses.
for _form in ("σύνισαν", "ξύνισαν"):
    _reading = _grammar("σύνοιδα", "οἶδα", (("3", "pl"),),
                        ("https://en.wiktionary.org/wiki/σύνοιδα",
                         "https://bailly.app/synoida"), tense="pluperfect")
    _reading["semantic_tense"] = "imperfect"
    _REVIEWED_FORMS[_form].append(_reading)
for _form, _tense in (("ἐπήϊεν", "imperfect"),
                       ("ἐπήισαν", "aorist"), ("ἐπήϊσαν", "aorist")):
    _REVIEWED_FORMS[_form].append(_grammar(
        "ἐπαΐω", "ἐπαΐω", (("3", "sg" if _tense == "imperfect" else "pl"),),
        ("https://en.wiktionary.org/wiki/ἐπαΐω", DIORISIS), tense=_tense))
_REVIEWED_FORMS["ἔπεσαν"] = [
    _grammar("ἔπειμι", "εἰμί", (("3", "pl"),),
             ("https://bailly.app/epeimi",
              "https://atlas.perseus.tufts.edu/dictionaries/entry/"
              "urn:cite2:scaife-viewer:dictionaries.v1:lsj-n38487/")),
    _grammar("πίπτω", "πίπτω", (("3", "pl"),),
             ("https://en.wiktionary.org/wiki/πίπτω",
              "https://logeion.uchicago.edu/morpho/ἔπεσαν"), tense="aorist"),
]
_REVIEWED_FORMS["ἦ"] = [
    _grammar("εἰμί", "εἰμί", (("1", "sg"),),
             (LSJ_EIMI_BE, "https://en.wiktionary.org/wiki/εἰμί")),
    _grammar("ἠμί", "ἠμί", (("3", "sg"),),
             ("https://en.wiktionary.org/wiki/ἠμί",)),
]
_REVIEWED_FORMS["ἦτε"] = [_grammar(
    "εἰμί", "εἰμί", (("2", "pl"),),
    (LSJ_EIMI_BE, "https://en.wiktionary.org/wiki/εἰμί"))]
_REVIEWED_FORMS["ἦτε"][0]["analyses"].append(asdict(ParadigmSlot.verb_finite(
    voice="active", tense="present", mood="subjunctive", person="2", number="pl")))
for _form, _people in _persons(
        "ἦ ἦν", "ἦς ἦσθα", "ἦν", "ἦστον ἦτον", "ἤστην ἤτην",
        "ἦμεν", "ἦτε", "ἦσαν").items():
    if _form not in _REVIEWED_FORMS:
        _REVIEWED_FORMS[_form] = [_grammar(
            "εἰμί", "εἰμί", _people,
            (LSJ_EIMI_BE, "https://en.wiktionary.org/wiki/εἰμί"))]


def reviewed_verb_readings(form: str) -> list[dict]:
    """Return all reviewed readings, retaining case-insensitive exact accents."""
    readings = _REVIEWED_FORMS.get(unicodedata.normalize("NFC", form).lower())
    return deepcopy(readings) if readings else []


def reviewed_verb_forms() -> tuple[str, ...]:
    """Stable form order for reproducible exports and coverage audits."""
    return tuple(sorted(_REVIEWED_FORMS))


def reviewed_verb_lemmas() -> dict[str, str]:
    """Unambiguous grammatical owners for single-lemma lookup builds.

    Ambiguous forms are deliberately absent. Their complete readings are
    exported separately rather than resolved by insertion order.
    """
    return {form: readings[0]["lemma"] for form, readings in _REVIEWED_FORMS.items()
            if len(readings) == 1}


def reviewed_verb_grammar(form: str) -> dict | None:
    """Return fresh metadata only when the reviewed lemma is unambiguous.

    NFC handles decomposed text. Accents are retained so that ἀπῆμεν,
    from ἄπειμι meaning 'be absent', does not acquire the 'go' analysis.
    Use reviewed_verb_readings for forms with several lexical readings.
    """
    readings = _REVIEWED_FORMS.get(unicodedata.normalize("NFC", form).lower(), [])
    return deepcopy(readings[0]) if len(readings) == 1 else None


def correct_grammatical_lemma(form: str, lemma: str | None) -> str | None:
    """Correct the reviewed headword conflation without changing other owners."""
    readings = _REVIEWED_FORMS.get(unicodedata.normalize("NFC", form).lower(), [])
    for reading in readings:
        if any(head["lemma"] == lemma and head["relation"] in (
                "present_used_as_future", "shared_paradigm")
               for head in reading["related_headwords"]):
            return reading["lemma"]
    return lemma


def preferred_reviewed_lemma(form: str, preferred: str | None = None) -> str | None:
    """Keep an existing supported preference when a form is ambiguous."""
    readings = _REVIEWED_FORMS.get(unicodedata.normalize("NFC", form).lower(), [])
    if not readings:
        return preferred
    corrected = correct_grammatical_lemma(form, preferred)
    if any(reading["lemma"] == corrected for reading in readings):
        return corrected
    for reading in readings:
        if any(head["lemma"] == preferred for head in reading["related_headwords"]):
            return reading["lemma"]
    return readings[0]["lemma"]
