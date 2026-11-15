"""Reviewed readings that the 2026-11 kaikki dumps changed for the worse."""
import inspect
import json
from pathlib import Path

import pytest

import build_lookup_db as B

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
    for lemma in set(B.KAIKKI_2026_11_READING_OVERRIDES.values()):
        assert lemma in heads, lemma


def test_rebuild_regressions_keep_their_treebank_readings():
    o = B.KAIKKI_2026_11_READING_OVERRIDES
    assert o["νοερῶν"] == o["νοεροῖς"] == "νοερός"
    assert o["παστάδα"] == o["παστάδος"] == "παστάς"
    assert o["πορρωτάτω"] == o["ποῤῥωτάτω"] == "πόρρω"
    assert o["ἑψομένης"] == "ἕψω" and o["μηνας"] == "μείς"


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
