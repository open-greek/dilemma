"""Reviewed readings that the 2026-11 kaikki dumps changed for the worse."""
import inspect
import json
from pathlib import Path

import pytest

import build_lookup_db as B
from dilemma.grc_rejected_lemmas import REJECTED_LEMMA_TARGETS, REJECTED_LEMMAS

ROOT = Path(__file__).resolve().parents[1]


def test_overrides_restore_the_treebank_readings():
    o = B.KAIKKI_2026_11_READING_OVERRIDES
    assert o["ην"] == o["ήν"] == "εἰμί"
    assert "ἤν" not in o          # accented ἤν is ἐάν, as the dumps have it
    assert o["μεγάλα"] == o["μεγαλα"] == "μέγας"
    assert o["εἴσομαι"] == o["είσομαι"] == o["εισομαι"] == "οἶδα"
    assert o["ἑψόμενα"] == "ἕψω"
    assert o["τοῖ"] == "σύ"
    assert o["κἆτα"] == "εἶτα"
    for key in ("ἐπαινεῖν", "επαινείν", "επαινειν", "ἐπαινέειν", "επαινέειν", "επαινεειν"):
        assert o[key] == "ἐπαινέω", key


def test_overrides_are_applied_with_the_reviewed_lookup_overrides():
    source = inspect.getsource(B.main) if hasattr(B, "main") else inspect.getsource(B)
    assert "**KAIKKI_2026_11_READING_OVERRIDES" in source


def test_override_targets_are_lexical_heads():
    path = ROOT / "data" / "grc_lexical_heads.json"
    if not path.exists():
        pytest.skip("grc_lexical_heads.json is not downloaded")
    heads = json.loads(path.read_text(encoding="utf-8"))["heads"]
    for lemma in (set(B.KAIKKI_2026_11_READING_OVERRIDES.values())
                  | set(B.REVIEWED_ABSENT_KEY_READINGS.values())):
        assert lemma in heads, lemma
    # The corpus lemmatizer artefacts are in no head list.
    for lemma, entry in REJECTED_LEMMA_TARGETS.items():
        if entry["reason"] == "corpus_lemmatizer_artifact":
            assert lemma not in heads, lemma


def test_rebuild_regressions_keep_their_treebank_readings():
    o = B.KAIKKI_2026_11_READING_OVERRIDES
    assert o["νοερῶν"] == o["νοεροῖς"] == "νοερός"
    assert o["παστάδα"] == o["παστάδος"] == "παστάς"
    assert o["πορρωτάτω"] == o["ποῤῥωτάτω"] == "πόρρω"
    assert o["ἑψομένης"] == "ἕψω" and o["μηνας"] == "μείς"


# Readings the review of the rebuilt table against 1.3.25 found worse --------

def _forms_of(lemma):
    return {form for form, target in B.KAIKKI_2026_11_READING_OVERRIDES.items()
            if target == lemma}


def _bare(form):
    return B.strip_accents(form.lower())


def test_whole_paradigms_follow_the_keys_already_reviewed():
    assert _forms_of("γνάθος") == {"γνάθων", "γναθων"}
    assert _forms_of("παστάς") == {
        "παστάδα", "παστάδος", "παστάδας", "παστάδων", "παστάδι", "παστάδες",
        "παστάσι", "παστάσιν", "παστάδε", "παστάδοιν", "πασταδα", "πασταδος",
        "πασταδας", "πασταδων", "πασταδι", "πασταδες", "παστασι", "παστασιν",
        "πασταδε", "πασταδοιν", "παστας"}
    assert _forms_of("ἕψω") == {
        "ἑψόμενα", "ἑψομένης", "ἑψόμενοι", "ἑψόμεναι", "ἑψομένου", "ἑψομένων",
        "ἑψομένῳ", "ἑψομένῃ", "ἑψομένοις", "ἑψομέναις", "ἑψομένην",
        "ἑψομένους", "ἑψομένας"}
    # The table of the future participle οἰκιῶν: polytonic, monotonic and
    # unaccented keys of the same 21 cells, and οἰκιοῦντες from before. The
    # noun's own genitive οἰκίου is not a key of it.
    participle = _forms_of("οἰκίζω")
    assert len(participle) == 64 and "οἰκιοῦντες" in participle
    assert {"οἰκιοῦσι", "οἰκιοῦσιν", "οἰκιοῦσα", "οἰκιοῦν", "οικιούσι",
            "οικιουσι", "οἰκιοῦντος", "οικιουντες"} <= participle
    for form in participle:
        assert _bare(form).startswith(("οικιουν", "οικιουσ")), form
    assert len({_bare(form) for form in participle}) == 21
    assert not {"οἰκίου", "οικίου", "οικιου", "οἰκιῶν"} & participle


def test_nominal_cells_are_not_read_as_regenerated_contract_verbs():
    assert _forms_of("νοερός") == {"νοερῶν", "νοεροῖς", "νοεροῦ", "νοερου",
                                   "νοερων", "νοεροις", "νοεροι", "νοερω"}
    assert _forms_of("ἔναρθρος") == {"ἐνάρθρου", "εναρθροις", "εναρθρω",
                                     "εναρθρων"}
    assert _forms_of("ἀντίπερα") == {"ἀντιπέρα", "αντιπερα", "αντιπεραν"}
    for stem, lemma in (("χριστομαχ", "χριστομάχος"), ("φιληδον", "φιλήδονος"),
                        ("διψυχ", "δίψυχος")):
        assert _forms_of(lemma) == {stem + ending for ending in
                                    ("οι", "οις", "ου", "ω", "ων")}, lemma
    assert _forms_of("νομομαθής") == {"νομομαθης", "νομομαθη", "νομομαθων"}
    # Genuine forms of the verbs keep the verb.
    o = B.KAIKKI_2026_11_READING_OVERRIDES
    assert not {"ἀντιπερῶν", "αντιπερων", "νομομαθοῦ", "νομομαθῇ"} & o.keys()
    genitives = {
        "ὁλοκαύτου": "ὁλόκαυτος", "ὁλοκάρπου": "ὁλόκαρπος", "γωνίου": "γώνιος",
        "καστέλλου": "κάστελλον", "κατασάρκου": "κατάσαρκος",
        "κατασπίλου": "κατάσπιλος", "φάλσου": "φάλσος", "χάμου": "χάμος",
        "ἀπομμάτου": "ἀπόμματος", "ἀφήβου": "ἄφηβος", "ἐμμόρφου": "ἔμμορφος",
        "ἐνουσίου": "ἐνούσιος"}
    for form, lemma in genitives.items():
        assert o[form] == lemma, form
    # With the unaccented keys of the same nominals that moved with them.
    assert sum(len(_forms_of(lemma)) for lemma in genitives.values()) == 47
    for lemma in genitives.values():
        stem = _bare(lemma)[:-2]
        for form in _forms_of(lemma):
            assert _bare(form).startswith(stem), (form, lemma)
            assert _bare(form)[len(stem):] in {"ου", "οι", "οις", "ω", "ων"}, form


def test_single_forms_keep_the_reading_the_treebanks_give():
    o = B.KAIKKI_2026_11_READING_OVERRIDES
    assert o["ἀγλαΐσας"] == "ἀγλαΐζω" and o["ἀττικίζων"] == "ἀττικίζω"
    assert o["δρύας"] == "δρῦς" and o["νη"] == "νή"
    assert "δρυας" not in o and "νή" not in o and "νὴ" not in o
    assert _forms_of("Κορινθιακός") == {"Κορινθιακόν", "Κορινθιακοῦ",
                                        "Κορινθιακοῖς", "Κορινθιακῷ"}
    assert _forms_of("ἑστίασις") == {"ἑστιάσεις", "ἑστιάσει"}
    assert _forms_of("ἔμφραξις") == {"ἐμφράξεις", "ἐμφράξει"}
    assert o["ὀχλήσεις"] == "ὄχλησις"
    assert o["ῥυέντα"] == o["ῥυέντας"] == "ῥέω"
    assert o["ἀπόμυξαι"] == o["ἀπομύξῃ"] == "ἀπομύσσω"
    assert o["Φόρκυνος"] == "Φόρκυς"
    assert o["αυλων"] == o["αυλών"] == "αὐλή" and "αὐλῶν" not in o
    assert o["Ἀπολλώνιον"] == o["Απολλώνιον"] == "Ἀπολλώνιος"
    assert _forms_of("κλῄζω") == {"κλήζεσθαι", "κληζεσθαι", "κλήζουσι",
                                  "κληζουσι", "κλήζομεν", "κληζομεν", "κλήζων",
                                  "κληζων"}
    # Elided keys carry the mark the build folds every elision spelling onto.
    assert o["Δωρίδ᾽"] == "Δωρίς" and o["Ἰλιάδ᾽"] == "Ἰλιάς"
    assert not [form for form in o if form[-1] in "’'ʼ`"]


def test_override_keys_and_lemmas_are_nfc_and_no_lemma_is_rejected():
    import unicodedata
    for table in (B.KAIKKI_2026_11_READING_OVERRIDES,
                  B.REVIEWED_ABSENT_KEY_READINGS, B.MODERN_READING_OVERRIDES):
        for form, lemma in table.items():
            assert unicodedata.normalize("NFC", form) == form, form
            assert unicodedata.normalize("NFC", lemma) == lemma, lemma
            assert lemma not in REJECTED_LEMMAS, lemma
    assert len(B.KAIKKI_2026_11_READING_OVERRIDES) == 233


# The override loop ----------------------------------------------------------

GLAUX, DIORISIS = B.GLAUX_PAIRS_PATH.name, B.DIORISIS_PAIRS_PATH.name


def _apply(overrides, *, combined, ag=None, el=None, el_sources=None, **scope):
    tables = {"combined": dict(combined), "combined_sources": {},
              "ag": dict(combined if ag is None else ag), "ag_sources": {},
              "el": dict(el or {}), "el_sources": dict(el_sources or {})}
    counts = B.apply_lookup_overrides(overrides, **tables, **scope)
    return tables, counts


def test_override_removes_the_el_corpus_reading_it_corrects():
    # The Ancient corpus gap-fill copied these pairs into the el table before
    # the overrides ran; written as el rows they would answer lang="el".
    tables, counts = _apply(
        {"εἴσομαι": "οἶδα", "ἀπεικός": "ἀπεικός", "Μάγων": "μάγος",
         "κἆτα": "εἶτα"},
        combined={"εἴσομαι": "εἶμι", "ἀπεικός": "ἀφίημι", "Μάγων": "Μάγων",
                  "κἆτα": "ἄτη"},
        el={"εἴσομαι": "εἶμι", "ἀπεικός": "ἀφίημι", "Μάγων": "Μάγων",
            "κἆτα": "ἄτη"},
        el_sources={"εἴσομαι": DIORISIS, "ἀπεικός": GLAUX, "Μάγων": GLAUX,
                    "κἆτα": GLAUX})
    assert tables["el"] == {} and tables["el_sources"] == {}
    assert tables["combined"] == tables["ag"] == {
        "εἴσομαι": "οἶδα", "ἀπεικός": "ἀπεικός", "Μάγων": "μάγος",
        "κἆτα": "εἶτα"}
    assert counts == {"changed": 4, "added": 0, "el_removed": 4}
    # With no el reading of its own, lang="el" reads the combined row.
    for form in ("εἴσομαι", "ἀπεικός", "Μάγων"):
        assert form in B.EL_CORPUS_ROW_OVERRIDE_FORMS


def test_override_keeps_every_other_el_reading():
    tables, counts = _apply(
        {"μεγάλα": "μέγας", "ᾔει": "εἶμι", "εἴσομαι": "οἶδα"},
        combined={"μεγάλα": "μεγάλα", "ᾔει": "ἔρχομαι", "εἴσομαι": "οἶδα"},
        el={"μεγάλα": "μεγάλος", "ᾔει": "ἔρχομαι", "εἴσομαι": "οἶδα"},
        el_sources={"μεγάλα": "mg_lookup.json", "ᾔει": GLAUX,
                    "εἴσομαι": DIORISIS})
    # A Modern Wiktionary reading stays; so does the el corpus row of an
    # override older than the 2026-11 review (the published tables answer
    # lang="el" with it), and a corpus reading that agrees with the override.
    assert "ᾔει" not in B.EL_CORPUS_ROW_OVERRIDE_FORMS
    assert tables["el"] == {"μεγάλα": "μεγάλος", "ᾔει": "ἔρχομαι",
                            "εἴσομαι": "οἶδα"}
    assert counts == {"changed": 2, "added": 0, "el_removed": 0}


def test_override_adds_only_the_listed_absent_keys():
    tables, counts = _apply({"μήτ": "μήτε", "φασιν": "φημί"}, combined={})
    assert tables["combined"] == {"μήτ": "μήτε"} and tables["ag"] == {}
    assert tables["combined_sources"] == {"μήτ": B.LOOKUP_OVERRIDES_INPUT}
    assert counts == {"changed": 0, "added": 1, "el_removed": 0}
    # A listed key that a source does supply is corrected like any other.
    tables, counts = _apply({"μήτ": "μήτε"}, combined={"μήτ": "μήτ"})
    assert tables["combined"] == tables["ag"] == {"μήτ": "μήτε"}
    assert counts == {"changed": 1, "added": 0, "el_removed": 0}


def test_absent_key_readings_are_overrides_too():
    source = inspect.getsource(B.build)
    assert "**REVIEWED_ABSENT_KEY_READINGS" in source
    assert B.REVIEWED_ABSENT_KEY_READINGS["μήτ"] == "μήτε"
    shared = (B.KAIKKI_2026_11_READING_OVERRIDES.keys()
              & B.REVIEWED_ABSENT_KEY_READINGS.keys())
    assert not shared, shared
    assert B.EL_CORPUS_ROW_OVERRIDE_FORMS == (
        set(B.KAIKKI_2026_11_READING_OVERRIDES)
        | set(B.REVIEWED_ABSENT_KEY_READINGS) | {"Μάγων"})


def test_forms_of_rejected_corpus_artefacts_keep_a_reading():
    # Rejecting the lemma empties the key, so the form's verb is written
    # whether or not another source brings the key back.
    artefacts = {lemma for lemma, entry in REJECTED_LEMMA_TARGETS.items()
                 if entry["reason"] == "corpus_lemmatizer_artifact"}
    assert artefacts == {"παραπαροινέω", "ἀμφιαμφισβητέω", "γιχρώσκω", "ταθήχω",
                         "χειρίω", "ἀρικνέομαι", "διασμήγνυμι"}
    a = B.REVIEWED_ABSENT_KEY_READINGS
    assert a["πεπαρῴνηκεν"] == a["πεπαρῴνηκας"] == a["πεπαρῳνήκεσαν"] == "παροινέω"
    for form in ("ἠμφεσβήτουν", "ἠμφεσβήτεις", "ἠμφεσβήτησεν", "ἠμφεσβήτησέ",
                 "ἠμφεσβήτησάς", "ἠμφεσβήτησαν", "ἠμφεσβητήσαμεν",
                 "ἠμφεσβητήθη", "ἠμφεσβήτηκεν"):
        assert a[form] == "ἀμφισβητέω", form
    assert a["χρωσθήσεται"] == "χρῴζω" and a["κεχειρικέναι"] == "χειρίζω"
    assert a["διέσμηκται"] == "διασμήχω" and a["ἀρίκηται"] == "ἀφικνέομαι"
    assert not set(a.values()) & REJECTED_LEMMAS
    # Real dictionary heads stay out of the rejected lemmas: Sophocles heads
    # ῥυέντης and ἀπόμυξα, whose colliding cells are overridden instead.
    assert not {"ῥυέντης", "ἀπόμυξα"} & REJECTED_LEMMAS


# Modern readings under lang="all" -------------------------------------------

def test_modern_readings_are_pinned_and_apart_from_the_ancient_overrides():
    m = B.MODERN_READING_OVERRIDES
    assert m["αυτοκινήτου"] == m["αυτοκινητου"] == "αυτοκίνητο"
    assert m["εστιάσεις"] == "εστίαση" and m["εμφράξεις"] == "έμφραξη"
    assert m["δακτυλίδι"] == m["δακτυλιδι"] == "δαχτυλίδι"
    assert m["Βιέννης"] == "Βιέννη"
    # Modern lemmas are monotonic, and no key has an Ancient override too.
    for lemma in m.values():
        assert lemma == B.to_monotonic(lemma), lemma
    assert not m.keys() & (B.KAIKKI_2026_11_READING_OVERRIDES.keys()
                           | B.REVIEWED_ABSENT_KEY_READINGS.keys())
    assert "MODERN_READING_OVERRIDES, combined=combined" in inspect.getsource(B.build)


def test_modern_reading_changes_the_combined_table_only():
    combined = {"αυτοκινήτου": "αὐτοκίνητος", "Βιέννης": "Βίεννα",
                "εστιάσεις": "εστίαση"}
    ag = dict(combined)
    el = {"αυτοκινήτου": "αυτοκίνητο", "εστιάσεις": "εστίαση"}
    el_sources = {"αυτοκινήτου": "mg_lookup.json", "εστιάσεις": "mg_lookup.json"}
    sources = {}
    changed = B.apply_modern_reading_overrides(
        {"αυτοκινήτου": "αυτοκίνητο", "Βιέννης": "Βιέννη",
         "εστιάσεις": "εστίαση", "δακτυλίδι": "δαχτυλίδι"},
        combined=combined, combined_sources=sources, el=el, el_sources=el_sources)
    # An absent key is not added, and a reading that is right already stays.
    assert changed == ["αυτοκινήτου", "Βιέννης"]
    assert combined == {"αυτοκινήτου": "αυτοκίνητο", "Βιέννης": "Βιέννη",
                        "εστιάσεις": "εστίαση"}
    # The row is the Modern table's where that has the reading.
    assert sources == {"αυτοκινήτου": "mg_lookup.json",
                       "Βιέννης": B.LOOKUP_OVERRIDES_INPUT}
    assert ag["αυτοκινήτου"] == "αὐτοκίνητος" and ag["Βιέννης"] == "Βίεννα"
    assert el == {"αυτοκινήτου": "αυτοκίνητο", "εστιάσεις": "εστίαση"}
