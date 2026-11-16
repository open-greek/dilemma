"""Principal parts that no LSJ entry prints: a compound takes those of its
simplex, a denominative those of its class, and only the cells a corpus
attests are kept. A principal-part cell gives way to a treebank reading."""

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))
import expand_lsj
from lsj_principal_parts import (
    compound_parts, compound_simplexes, derive_grc_conj_args, other_voice,
    regular_compound_parts, regular_parts, regular_rivals,
)


@pytest.mark.parametrize("headword,simplex,parts,expected", [
    # elision before the augment, the full preverb before a consonant
    ("συγκατατρίβω", "τρίβω",
     {"fut": "τρίψω", "aor": "ἔτριψα", "pf": "τέτρῐφα"},
     {"fut": "συγκατατριψω", "aor": "συγκατετριψα",
      "pf": "συγκατατετρῐφα"}),
    # a rough stem aspirates the preverb, a smooth one does not
    ("καθέλκω", "ἕλκω", {"fut": "ἕλξω", "aor": "εἵλκῠσα"},
     {"fut": "καθελξω", "aor": "καθειλκῠσα"}),
    ("ἀφοράω", "ὁράω", {"fut": "ὄψομαι", "aor2": "εἶδον", "pf": "ἑώρακα"},
     {"fut": "ἀποψομαι", "aor2": "ἀπειδον", "pf": "ἀφεωρακα"}),
    # ἐν and σύν before the augment and before a consonant
    ("ἐγχλιαίνω", "χλιαίνω", {"aor": "ἐχλίηνα", "pf": "κεχλίαγκα"},
     {"aor": "ἐνεχλιηνα", "pf": "ἐγκεχλιαγκα"}),
    ("συλλαμβάνω", "λαμβάνω",
     {"fut": "λήψομαι", "aor2": "ἔλαβον", "pf": "εἴληφα"},
     {"fut": "συλληψομαι", "aor2": "συνελαβον", "pf": "συνειληφα"}),
    # ἐκ is ἐξ before a vowel
    ("προεκπηδάω", "πηδάω", {"aor": "ἐπήδησα", "pf": "πεπήδηκα"},
     {"aor": "προεξεπηδησα", "pf": "προεκπεπηδηκα"}),
    # a contracted future keeps its circumflex
    ("ἀποδέρω", "δέρω", {"fut": "δερῶ", "aor": "ἔδειρα"},
     {"fut": "ἀποδερῶ", "aor": "ἀπεδειρα"}),
    # the doubled ρ after a vowel, the breathing of εἰσ-
    ("καταρρέω", "ῥέω", {"fut": "ῥεύσομαι", "aor": "ἔρρευσα"},
     {"fut": "καταρρευσομαι", "aor": "κατερρευσα"}),
    ("εἰσφέρω", "φέρω", {"fut": "οἴσω", "aor1": "ἤνεγκα"},
     {"fut": "εἰσοισω", "aor1": "εἰσηνεγκα"}),
    # the imperfect is the compound's own
    ("ἐπιδοκέω", "δοκέω", {"impf": "ἐδόκουν"}, {}),
    # not a compound of that verb
    ("λύω", "τρίβω", {"aor": "ἔτριψα"}, {}),
])
def test_compound_parts(headword, simplex, parts, expected):
    assert compound_parts(headword, simplex, parts) == expected


def test_compound_parts_resolve_like_printed_ones():
    parts = compound_parts("συγκατατρίβω", "τρίβω", {"aor": "ἔτριψα"})
    assert derive_grc_conj_args(parts, "συγκατατρίβω")["aor-1"] == [
        "aor-1", "συγκατετριψ", "συγκατατριψ", "prefix=συγκατ"]
    parts = compound_parts("εἰσφέρω", "φέρω", {"aor1": "ἤνεγκα"})
    assert derive_grc_conj_args(parts, "εἰσφέρω")["aor-1"] == [
        "aor-1", "εἰσηνεγκ", "εἰσενεγκ", "prefix=εἰσ"]


def test_compound_simplexes():
    verbs = {"τριβω": ["τρίβω"], "κατατριβω": ["κατατρίβω"],
             "ερειδω": ["ἐρείδω"], "ειδω": ["εἴδω"], "ελκω": ["ἕλκω"],
             "οραω": ["ὁράω"], "ρεω": ["ῥέω"]}
    # the verb after the longest chain; κατατρίβω is a compound of it too
    assert compound_simplexes("συγκατατρίβω", verbs) == ["τρίβω"]
    # ὑπ-ερείδω or ὑπερ-είδω: two readings, none taken
    assert compound_simplexes("ὑπερείδω", verbs) == []
    # the breathing of the verb must fit the preverb
    assert compound_simplexes("καθέλκω", verbs) == ["ἕλκω"]
    assert compound_simplexes("κατέλκω", verbs) == []
    assert compound_simplexes("ἀφοράω", verbs) == ["ὁράω"]
    assert compound_simplexes("ἀποράω", verbs) == []
    assert compound_simplexes("καταρρέω", verbs) == ["ῥέω"]
    assert compound_simplexes("τρίβω", verbs) == []


def test_regular_parts_by_class():
    assert regular_parts("ποιέω") == [{
        "fut": "ποιησω", "aor": "ἐποιησα", "pf": "πεποιηκα",
        "pf_mp": "πεποιημαι", "aor_p": "ἐποιηθην",
        "fut_p": "ποιηθησομαι"}]
    assert regular_parts("δηλόω")[0]["aor"] == "ἐδηλωσα"
    assert regular_parts("παιδεύω")[0]["pf"] == "πεπαιδευκα"
    assert regular_parts("ἀγοράζω")[0]["aor_p"] == "ἠγορασθην"
    # long α after ε, ι, ρ
    assert regular_parts("πειράομαι")[0] == {
        "fut_med": "πειρᾱσομαι", "aor_med": "ἐπειρᾱσαμην",
        "pf_mp": "πεπειρᾱμαι", "aor_p": "ἐπειρᾱθην",
        "fut_p": "πειρᾱθησομαι"}
    assert regular_parts("ὁρμάω")[0]["aor"] == "ὡρμησα"
    # no reduplication before a cluster or ῥ, the aspirate plain
    assert regular_parts("στρατεύω")[0]["pf"] == "ἐστρατευκα"
    assert regular_parts("ῥιπίζω")[0]["aor"] == "ἐρριπισα"
    assert regular_parts("φλυαρέω")[0]["pf"] == "πεφλυαρηκα"
    assert regular_parts("γνωρίζω")[0]["pf_mp"] == "ἐγνωρισμαι"


def test_regular_parts_of_a_diphthong_come_with_and_without_the_augment():
    lengthened, plain, attic = regular_parts("εἰκονίζω")
    assert lengthened["aor"] == "ᾐκονισα" and lengthened["pf"] == "ᾐκονικα"
    assert plain["aor"] == "εἰκονισα"
    # εἰκονικ- is the adjective εἰκονικός, not a perfect
    assert "pf" not in plain and plain["pf_mp"] == "εἰκονισμαι"
    assert attic == {"fut": "εἰκονιῶ"}
    assert [parts["aor"] for parts in regular_parts("οἰκοδομέω")] == [
        "ᾠκοδομησα", "οἰκοδομησα"]
    # ἱππικ- is ἱππικός
    assert "pf" not in regular_parts("ἱππίζω")[0]
    assert regular_parts("ψηφίζομαι")[-1] == {"fut_med": "ψηφιουμαι"}


@pytest.mark.parametrize("verb", ["χέω", "θλάω", "πνέω", "βάλλω", "τίθημι",
                                  "λύω", "γράφω", "ἐρέω", "ὁράω", "ἀρόω"])
def test_no_regular_parts_outside_the_classes(verb):
    assert regular_parts(verb) == []


def test_regular_rivals():
    # -άω shares -ησα with -έω, -ᾱσα with -άζω, and its feminine participle
    # with the aorist of -όω
    assert {"βοαζω", "βοεω", "βοοω", "βοαζομαι"} <= set(regular_rivals("βοάω"))
    assert "δαπαναω" in regular_rivals("δαπανόω")
    assert "μυαω" in regular_rivals("μυέω")
    assert "σκιαω" in regular_rivals("σκιάζω")
    # ἀ- and ἐ- take one augment
    assert {"απορεω", "απορεομαι"} <= set(regular_rivals("ἐπορέομαι"))
    assert regular_rivals("νομίζω") == []
    assert other_voice("θεαω") == "θεαομαι"
    assert other_voice("διαζευγνυμαι") == "διαζευγνυμι"


def test_regular_compound_parts_put_the_augment_after_the_preverb():
    sets = regular_compound_parts("ἐπιστατέω")
    assert sets[0]["aor"] == "ἐπεστατησα"
    assert sets[0]["fut"] == "ἐπιστατησω"
    assert regular_compound_parts("ποιέω") == []


def test_inferred_part_sets():
    verbs = {"τριβω": ["τρίβω"], "δοκεω": ["δοκέω"], "πηδαω": ["πηδάω"]}
    printed = {"τρίβω": {"fut": "τρίψω", "aor": "ἔτριψα"},
               "δοκέω": {"impf": "ἐδόκουν"}}
    parts_of = lambda verb: printed.get(verb, {})
    infer = expand_lsj.inferred_part_sets
    assert infer("κατατρίβω", {}, verbs, parts_of) == (
        "compound", "τρίβω", [{"fut": "κατατριψω", "aor": "κατετριψα"}])
    # The tenses the entry prints itself are not inferred.
    assert infer("κατατρίβω", {"aor": "κατέτριψα"}, verbs, parts_of)[2] == [
        {"fut": "κατατριψω"}]
    # A simplex that prints nothing usable, or nothing: its class.
    kind, source, sets = infer("ἐπιδοκέω", {}, verbs, parts_of)
    assert (kind, source) == ("regular", "δοκέω")
    assert sets[0]["aor"] == "ἐπεδοκησα"
    kind, source, sets = infer("ἐκπηδάω", {}, verbs, parts_of)
    assert (kind, source) == ("regular", "πηδάω")
    assert sets[0]["aor"] == "ἐξεπηδησα"
    # No simplex: the verb's own class, augment in front or after a preverb.
    kind, source, sets = infer("ἐπιστατέω", {}, verbs, parts_of)
    assert (kind, source) == ("regular", "")
    assert {parts.get("aor") for parts in sets} >= {
        "ἠπιστατησα", "ἐπεστατησα"}
    assert infer("βάλλω", {}, verbs, parts_of) == ("", "", [])
    # ἀπο-θέω, whatever the list of verbs knows: ἀποθήσω is not its future.
    assert infer("ἀποθέω", {}, verbs, parts_of) == ("", "", [])
    assert infer("ἐμπλέω", {}, verbs, parts_of) == ("", "", [])


def test_nothing_is_inferred_where_another_verb_shares_the_tenses():
    infer = expand_lsj.inferred_part_sets
    none = ("", "", [])
    parts_of = lambda verb: {}
    assert infer("σκιάζω", {}, {}, parts_of)[0] == "regular"
    # σκιάσω is the future of σκιάω too
    assert infer("σκιάζω", {}, {"σκιαω": ["σκιάω"]}, parts_of) == none
    # ἠπόρημαι is the perfect of ἀπορέω
    assert infer("ἐπορέομαι", {}, {"απορεω": ["ἀπορέω"]}, parts_of) == none
    # the compound has no rival, its simplex has
    verbs = {"θλεω": ["θλέω"], "θλαω": ["θλάω"], "τριβεω": ["τριβέω"],
             "τριβαω": ["τριβάω"]}
    assert infer("κατατριβέω", {}, verbs, parts_of) == none
    # the other voice has an entry of its own
    assert infer("πειράω", {}, {"πειραομαι": ["πειράομαι"]}, parts_of) == none
    # the corpora file a capitalised verb under its lowercase
    assert infer("Βακχεύω", {}, {}, parts_of) == none
    assert infer("βακχεύω", {}, {}, parts_of)[0] == "regular"


def profile(tmp_path, forms):
    path = tmp_path / "form_profile.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE forms (form_id INTEGER PRIMARY KEY, "
                       "form TEXT NOT NULL, form_norm TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO forms (form, form_norm) VALUES (?, ?)",
        [(form, expand_lsj.norm_key(form)) for form in forms])
    connection.commit()
    connection.close()
    return path


def test_stems_attested(tmp_path):
    attested, spellings = expand_lsj.load_attested_forms(
        profile(tmp_path, ["εἰκόνισεν", "Γέγραπται", "ἐλύθη"]))
    assert "εἰκόνισεν" in attested and "γέγραπται" in attested
    assert expand_lsj.stems_attested(["aor-1", "ᾐκονισ", "εἰκονισ"], spellings)
    assert not expand_lsj.stems_attested(["aor-1", "ᾐκονισ", "ᾐκονισ"],
                                         spellings)
    # γεγραφ- gives γέγραπται: the perfect middle stem loses its consonant
    assert expand_lsj.stems_attested(["perf", "γεγραφ", "γεγραφ"], spellings)
    assert not expand_lsj.stems_attested(["perf", "γεγραφ", ""], spellings)
    assert expand_lsj.stems_attested(
        ["aor-1", "", "", "ἐλυθ", "λυθ", "form=pass"], spellings)
    assert not expand_lsj.stems_attested(["fut", "λυσ", "prefix=ἐλυθ"],
                                         spellings)


def test_only_attested_cells_of_an_inferred_part_are_kept(tmp_path,
                                                          monkeypatch):
    attested, spellings = expand_lsj.load_attested_forms(
        profile(tmp_path, ["εἰκόνισεν", "εἰκονίσας", "εἰκονίσαντος"]))
    calls = []

    def expand(_wtp, headword, tense_args):
        calls.append(tuple(tense_args))
        return {"εἰκόνισεν", "εἰκονίσας", "εἰκόνισα", "εἰκονίσαιμι"}

    monkeypatch.setattr(expand_lsj, "_expand_principal_part", expand)
    stats = Counter()
    cells = expand_lsj.expand_inferred_parts(
        None, "εἰκονίζω", regular_parts("εἰκονίζω"), attested, spellings,
        stats)
    # Lua gives the cells; the corpus decides which are kept.
    assert cells == {"εἰκόνισεν", "εἰκονίσας"}
    assert ("aor-1", "ᾐκονισ", "εἰκονισ") in calls
    # No attested form begins with ᾐκονισθ- or εἰκονισθ-: no passive aorist.
    assert not any("form=pass" in call for call in calls)
    assert stats["inferred_calls_skipped"] > 0
    assert stats["inferred_calls"] == len(calls)


def test_a_cell_gives_way_to_a_treebank_reading(tmp_path):
    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps([
        {"form": "κατεδέξατο", "lemma": "καταδέχομαι"},
        {"form": "Κατεδέξατο", "lemma": "καταδέχομαι"},
        {"form": "συνεῖδον", "lemma": "σύν-εἶδον"},
        {"form": "ἔνθορε", "lemma": "ἐνθρῴσκω"},
        {"form": "κατέδεξε", "lemma": "καταδείκνυμι"},
        {"form": "κατέδεξε", "lemma": "καταδέχομαι"},
        {"form": "ἄλλο", "lemma": "ἄλλος"},
    ], ensure_ascii=False), encoding="utf-8")
    later = tmp_path / "later.json"
    later.write_text(json.dumps([
        {"form": "κατεδέξατο", "lemma": "καταδείκνυμι"},
        {"form": "ἠβουλήθη", "lemma": "ἀβουλέω"},
        {"form": "ἐπελάθοντο", "lemma": "ἐπιλανθάνομαι"},
    ], ensure_ascii=False), encoding="utf-8")
    earlier = tmp_path / "earlier.json"
    earlier.write_text(json.dumps([
        {"form": "ἠβουλήθη", "lemma": "βούλομαι"}], ensure_ascii=False),
        encoding="utf-8")
    key = expand_lsj.norm_key
    wanted = {key(form) for form in (
        "κατεδέξατο", "συνεῖδον", "ἔνθορε", "κατέδεξε", "κατέδεξα",
        "ἠβουλήθη", "ἐπελάθοντο")}
    readings = expand_lsj.load_treebank_readings(
        [earlier, pairs, later], wanted)
    # the first file that has a form speaks for it
    assert readings == {
        key("κατεδέξατο"): ["καταδέχομαι"], key("συνεῖδον"): ["σύν-εἶδον"],
        key("ἔνθορε"): ["ἐνθρῴσκω"], key("ἠβουλήθη"): ["βούλομαι"],
        key("κατέδεξε"): ["καταδείκνυμι", "καταδέχομαι"],
        key("ἐπελάθοντο"): ["ἐπιλανθάνομαι"]}
    heads = {"καταδέχομαι", "καταδείκνυμι", "ἐνθρῴσκω", "συνοράω",
             "βούλομαι", "ἀβουλέω"}
    left = expand_lsj.left_to_treebank
    # the treebank reads it as another headword
    assert left("κατεδέξατο", "καταδείκνυμι", readings, heads)
    assert left("ἠβουλήθη", "ἀβουλέω", readings, heads)
    # or as this verb in its other voice
    assert left("ἐπελάθοντο", "ἐπιλανθάνω", readings, heads)
    # it reads it as this verb too
    assert not left("κατέδεξε", "καταδείκνυμι", readings, heads)
    # no treebank has it
    assert not left("κατέδεξα", "καταδείκνυμι", readings, heads)
    # a treebank lemma that is no headword decides nothing
    assert not left("συνεῖδον", "συνοράω", readings, heads)
    # the same verb in another spelling
    assert not left("ἔνθορε", "ἐνθρώσκω", readings, heads)
    assert expand_lsj.treebank_agrees("ἔνθορε", "ἐνθρώσκω", readings)
    assert not expand_lsj.treebank_agrees("κατέδεξα", "καταδείκνυμι",
                                          readings)


def test_verb_pass_infers_and_checks(monkeypatch, tmp_path):
    entries = {hw: {"orth_orig": hw, "gender": ""}
               for hw in ("εἰκονίζω", "καταδείκνυμι", "καταδέχομαι")}
    entries["εἰκονικός"] = {"orth_orig": "εἰκονικός", "gender": "m"}
    monkeypatch.setattr(expand_lsj, "parse_lsj_entries", lambda: entries)
    monkeypatch.setattr(expand_lsj, "load_wiktionary_forms", lambda: {})
    monkeypatch.setattr(expand_lsj, "load_lsj_head_texts", lambda: {
        "καταδείκνυμι": "Ion. aor. κατέδεξα Hdt.: discover."})
    monkeypatch.setattr(expand_lsj, "load_lsj_supplement_head_texts",
                        lambda: {})
    monkeypatch.setattr(expand_lsj, "verify_lsj_glosses",
                        lambda: {"sha256": "0" * 64, "size": 0})
    monkeypatch.setattr(expand_lsj, "get_wtp", lambda: object())
    lookup = tmp_path / "ag_lookup.json"
    lookup.write_text("{}")
    monkeypatch.setattr(expand_lsj, "AG_LOOKUP", lookup)
    monkeypatch.setattr(expand_lsj, "FORM_PROFILE", profile(tmp_path, [
        "εἰκόνισεν", "εἰκονικός", "κατεδέξατο", "κατέδεξε", "εἰκονίσει",
        "εἰκονίσεις"]))
    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps([
        {"form": "κατεδέξατο", "lemma": "καταδέχομαι"},
        {"form": "εἰκονίσεις", "lemma": "εἰκονίζω"}], ensure_ascii=False),
        encoding="utf-8")
    monkeypatch.setattr(expand_lsj, "TREEBANK_PAIRS", (pairs,))
    audit = tmp_path / "audit.json"
    monkeypatch.setattr(expand_lsj, "VERB_AUDIT", audit)

    def expand(_wtp, word, head_text="", supplement_text="",
               principal_part_forms=None):
        if word == "καταδείκνυμι":
            principal_part_forms |= {"κατέδεξε", "κατεδέξατο"}
        return {word}, ""

    monkeypatch.setattr(expand_lsj, "expand_verb", expand)
    monkeypatch.setattr(
        expand_lsj, "_expand_principal_part",
        lambda _wtp, headword, tense_args: (
            {"εἰκόνισεν", "εἰκόνισα", "εἰκονικός", "εἰκονίσει",
             "εἰκονίσεις"}
            if headword == "εἰκονίζω" else set()))
    expand_lsj.expand_verbs()
    result = json.loads(lookup.read_text(encoding="utf-8"))
    # an attested cell of an inferred aorist
    assert result["εἰκόνισεν"] == "εἰκονίζω"
    # an unattested one, and one that is a headword itself
    assert "εἰκόνισα" not in result and "εἰκονικός" not in result
    # a future that the noun in -σις also spells needs a treebank
    assert "εἰκονίσει" not in result
    assert result["εἰκονίσεις"] == "εἰκονίζω"
    # a printed part keeps the cell no treebank reads otherwise
    assert result["κατέδεξε"] == "καταδείκνυμι"
    assert "κατεδέξατο" not in result and "κατεδεξατο" not in result
    recorded = json.loads(audit.read_text(encoding="utf-8"))
    assert recorded["inferred"] == {"εἰκονίζω": {
        "kind": "regular", "from": "",
        "cells": ["εἰκονίσεις", "εἰκόνισεν"]}}
    assert recorded["left_to_treebank"] == [{
        "form": "κατεδέξατο", "verb": "καταδείκνυμι", "part": "printed",
        "treebank": ["καταδέχομαι"]}]
    assert recorded["counts"]["inferred_noun_twins"] == 1
    assert set(recorded["inputs"]) == {"form_profile.db", "pairs.json"}


def test_verb_pass_names_a_missing_corpus_table(monkeypatch, tmp_path):
    monkeypatch.setattr(expand_lsj, "FORM_PROFILE",
                        tmp_path / "form_profile.db")
    monkeypatch.setattr(expand_lsj, "TREEBANK_PAIRS", ())
    with pytest.raises(SystemExit) as stopped:
        expand_lsj.expand_verbs()
    assert "form_profile.db" in str(stopped.value)
