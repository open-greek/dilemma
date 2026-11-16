"""Principal parts that no LSJ entry prints: a compound takes those of its
simplex, a denominative those of its class, and only the cells a corpus
attests are kept. A principal-part cell gives way to a treebank reading:
one on a stem no entry prints to any headword, one on a printed stem to a
verb that has the cell too."""

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))
import expand_lsj
from lsj_principal_parts import (
    INDICATIVE_ONLY, compound_parts, compound_simplexes, derive_grc_conj_args,
    derived_cells, other_voice, regular_compound_parts, regular_parts,
    regular_rivals,
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
        return {"εἰκόνισε(ν)", "εἰκονίσας", "εἰκόνισα", "εἰκονίσαιμι"}

    monkeypatch.setattr(expand_lsj, "_expand_principal_part", expand)
    stats = Counter()
    cells = expand_lsj.expand_inferred_parts(
        None, "εἰκονίζω", regular_parts("εἰκονίζω"), attested, spellings,
        stats)
    # Lua gives the cells; the corpus decides which are kept. The movable
    # nu makes two spellings, and only εἰκόνισεν is attested.
    assert cells == {"εἰκόνισεν", "εἰκονίσας"}
    assert ("aor-1", "ᾐκονισ", "εἰκονισ") in calls
    # No attested form begins with ᾐκονισθ- or εἰκονισθ-: no passive aorist.
    assert not any("form=pass" in call for call in calls)
    assert stats["inferred_calls_skipped"] > 0
    assert stats["inferred_calls"] == len(calls)


def write_pairs(path, rows):
    path.write_text(json.dumps(
        [{"form": form, "lemma": lemma, "pos": pos}
         for form, lemma, pos in rows], ensure_ascii=False), encoding="utf-8")
    return path


def test_treebank_readings_follow_the_gap_fill(tmp_path):
    earlier = write_pairs(tmp_path / "earlier.json", [
        ("ἠβουλήθη", "βούλομαι", "verb"),
        ("Φιαλέων", "Φιαλεύς", "noun"),
        ("Κατέδεξα", "καταδείκνυμι", "verb")])
    pairs = write_pairs(tmp_path / "pairs.json", [
        ("κατεδέξατο", "καταδέχομαι", "verb"),
        ("Κατεδέξατο", "καταδείκνυμι", "verb"),
        ("συνεῖδον", "σύν-εἶδον", "verb"),
        ("ἔνθορε", "ἐνθρῴσκω1", "verb"),
        ("κατέδεξε", "καταδείκνυμι", "verb"),
        ("κατέδεξε", "καταδέχομαι", "verb"),
        ("λαβὼν", "λαμβάνω", "verb"),
        ("ἄλλο", "ἄλλος", "adj")])
    later = write_pairs(tmp_path / "later.json", [
        ("κατεδέξατο", "καταδείκνυμι", "verb"),
        ("ἠβουλήθη", "ἀβουλέω", "verb"),
        ("φιαλέων", "φιάλη", "noun")])
    readings = expand_lsj.load_treebank_readings(
        [earlier, pairs, later],
        {"κατεδέξατο", "συνεῖδον", "ἔνθορε", "κατέδεξε", "κατέδεξα",
         "ἠβουλήθη", "φιαλέων", "λαβών", "ἀπέδεξα"})
    key = expand_lsj.reading_key
    assert readings == {
        # the first file that has a form speaks for it, in its own order
        key("ἠβουλήθη"): [("βούλομαι", "verb")],
        key("κατεδέξατο"): [("καταδέχομαι", "verb")],
        key("κατέδεξε"): [("καταδείκνυμι", "verb"), ("καταδέχομαι", "verb")],
        key("συνεῖδον"): [("σύν-εἶδον", "verb")],
        # a homograph number is no part of the lemma, a grave is an acute
        key("ἔνθορε"): [("ἐνθρῴσκω", "verb")],
        key("λαβών"): [("λαμβάνω", "verb")],
        # the form as spelled before a capitalised one: Φιαλέων is a name
        key("φιαλέων"): [("φιάλη", "noun")],
        # a capitalised form speaks where no file has the form itself
        key("κατέδεξα"): [("καταδείκνυμι", "verb")],
    }


def test_a_cell_gives_way_to_a_treebank_reading():
    key = expand_lsj.reading_key
    readings = {
        key("κατεδέξατο"): [("καταδέχομαι", "verb")],
        key("ἠβουλήθη"): [("βούλομαι", "verb")],
        key("ἐπελάθοντο"): [("ἐπιλανθάνομαι", "verb")],
        key("κατέδεξε"): [("καταδείκνυμι", "verb"), ("καταδέχομαι", "verb")],
        key("ἀνακλάσεις"): [("ἀνάκλασις", "noun"), ("ἀνακλάω", "verb")],
        key("συνεῖδον"): [("σύν-εἶδον", "verb")],
        key("συνιδεῖν"): [("σύν-εἶδον", "verb"), ("συνοράω", "verb")],
        key("ἔνθορε"): [("ἐνθρῴσκω", "verb")],
        key("ἦλθεν"): [("Ἔρχομαι", "verb")],
        key("ταύτης"): [("Ταύτη", "noun")],
        key("φιαλέων"): [("φιάλη", "noun")],
        key("ἀμφιέσομαι"): [("ἀμφιάζω", "verb")],
    }
    heads = frozenset({
        "καταδέχομαι", "καταδείκνυμι", "ἐνθρῴσκω", "συνοράω", "βούλομαι",
        "ἀβουλέω", "ἀνάκλασις", "ἀνακλάω", "ἔρχομαι", "φιάλη", "ἀμφιάζω"})
    reading = expand_lsj.treebank_reading
    left = expand_lsj.left_to_treebank
    agrees = expand_lsj.treebank_agrees
    # the treebank reads it as another headword
    assert reading("κατεδέξατο", "καταδείκνυμι", readings, heads) == (
        "καταδέχομαι", "verb")
    assert left("κατεδέξατο", "καταδείκνυμι", readings, heads)
    assert left("ἠβουλήθη", "ἀβουλέω", readings, heads)
    # or as this verb in its other voice
    assert left("ἐπελάθοντο", "ἐπιλανθάνω", readings, heads)
    # The first usable reading decides: this verb's,
    assert not left("κατέδεξε", "καταδείκνυμι", readings, heads)
    assert agrees("κατέδεξε", "καταδείκνυμι", readings, heads)
    # or the noun's, though the verb is read further down.
    assert left("ἀνακλάσεις", "ἀνακλάω", readings, heads)
    assert not agrees("ἀνακλάσεις", "ἀνακλάω", readings, heads)
    # no treebank has it
    assert not left("κατέδεξα", "καταδείκνυμι", readings, heads)
    assert not agrees("κατέδεξα", "καταδείκνυμι", readings, heads)
    # a treebank lemma that no dictionary heads is passed over
    assert not left("συνεῖδον", "συνοράω", readings, heads)
    assert agrees("συνιδεῖν", "συνοράω", readings, heads)
    # the same verb in another spelling
    assert not left("ἔνθορε", "ἐνθρώσκω", readings, heads)
    assert agrees("ἔνθορε", "ἐνθρώσκω", readings, heads)
    # a capitalised lemma counts as its lowercase headword
    assert left("ἦλθεν", "ἤλθω", readings, heads)

    # A cell on a stem the entry prints gives way to a verb that has the
    # cell among its own principal parts, and to a reading that is no verb.
    rival = expand_lsj.left_to_rival
    yields = {key("κατεδέξατο"): {"καταδεικνυμι", "καταδεχομαι"},
              key("ἠβουλήθη"): {"αβουλεω"},
              key("κατέδεξε"): {"καταδεικνυμι", "καταδεχομαι"}}
    assert rival("κατεδέξατο", "καταδείκνυμι", readings, heads, yields)
    assert rival("φιαλέων", "φιάλλω", readings, heads, yields)
    # βούλομαι and ἀμφιάζω are treebank readings no part of theirs gives
    assert not rival("ἠβουλήθη", "ἀβουλέω", readings, heads, yields)
    assert not rival("ἀμφιέσομαι", "ἀμφιέννυμι", readings, heads, yields)
    # the treebank reads it as this verb, or not at all
    assert not rival("κατέδεξε", "καταδείκνυμι", readings, heads, yields)
    assert not rival("κατέδεξα", "καταδείκνυμι", readings, heads, yields)


def test_lexical_heads_leave_out_the_reviewed_non_lemmas(tmp_path):
    path = tmp_path / "grc_lexical_heads.json"
    path.write_text(json.dumps({"heads": {
        "οὗτος": ["lsj9_headwords"], "Ταύτη": ["wiktionary_el_grc"]}},
        ensure_ascii=False), encoding="utf-8")
    assert expand_lsj.load_lexical_heads(path) == {"οὗτος"}


def test_noun_twins():
    twin = expand_lsj.noun_twin
    assert twin("ἐπισκιάσει") and twin("ἀμοιρήσεις") and twin("συνόψει")
    assert twin("κοσμιῶν") and twin("πεδιέων")
    # the aorist participle of πίνω is no genitive plural
    assert not twin("καταπιών") and not twin("εἰκονίσας")


def test_derived_cells():
    cells = {"κατέδεξε(ν)", "κατεδέξατο", "καταδέξας", "καταδέξασθαι"}
    assert derived_cells(
        "aor-1", ["aor-1", "κατεδεξ", "καταδεξ", "prefix=κατα"], cells) == {
            "καταδέξας", "καταδέξασθαι"}
    # the passive stems stand after two empty ones
    assert derived_cells(
        "aor-pass", ["aor-1", "", "", "ἀντεταφ", "ἀντιταφ", "form=pass"],
        {"ἀντετάφην", "ἀντιταφείς"}) == {"ἀντιταφείς"}
    assert derived_cells("aor-2", ["aor-2", "ἠγαγ", "ἀγαγ"],
                         {"ἤγαγον", "ἀγαγών"}) == {"ἀγαγών"}
    # A plain ἐ- before the stem: the module takes it off by itself.
    assert derived_cells("aor-1", ["aor-1", "ἐζημιωσ", "ζημιωσ"],
                         {"ἐζημίωσα", "ζημιῶσαι"}) == set()
    # one stem with and without the augment, or the indicative alone
    assert derived_cells("aor-2", ["aor-2", "κατειπ", "κατειπ"],
                         {"κατεῖπον", "κατειπεῖν"}) == set()
    assert derived_cells("aor-2", ["aor-2", "ἠγαγ", "ἠγαγ", INDICATIVE_ONLY],
                         {"ἤγαγον"}) == set()
    # the augmented pluperfect is built from the printed perfect
    assert derived_cells("plup-aug", ["plup", "ἐλελυκ", "ἐλελυ"],
                         {"ἐλελύκειν"}) == {"ἐλελύκειν"}
    # the future, the perfect and the imperfect stand on printed stems
    assert derived_cells("fut", ["fut", "λυσ"], {"λύσω"}) == set()
    assert derived_cells("perf", ["perf", "λελυκ", "λελυ"],
                         {"λέλυκα"}) == set()
    assert derived_cells("imperf", ["imperf", "ἐλυ"], {"ἔλυον"}) == set()


def test_cells_only_a_derived_stem_gives(monkeypatch):
    cells = {"aor-2": {"ἤγαγον", "ἄξω", "ἀγαγών"}, "fut": {"ἄξω", "ἄξεις"}}
    monkeypatch.setattr(
        expand_lsj, "_expand_principal_part",
        lambda _wtp, headword, tense_args: cells[tense_args[0]])
    # ἄξω is the future too, on the printed stem
    assert expand_lsj.derived_principal_part_cells(
        None, "ἄγω", "fut. ἄξω: aor. 2 ἤγαγον") == {"ἀγαγών"}
    assert expand_lsj.derived_principal_part_cells(None, "ἄγω", "") == set()


def test_verb_pass_infers_and_checks(monkeypatch, tmp_path):
    entries = {hw: {"orth_orig": hw, "gender": ""}
               for hw in ("εἰκονίζω", "καταδείκνυμι", "καταδέχομαι",
                          "δέχομαι", "ἀμφιέννυμι", "ἀμφιάζω")}
    entries["εἰκονικός"] = {"orth_orig": "εἰκονικός", "gender": "m"}
    monkeypatch.setattr(expand_lsj, "parse_lsj_entries", lambda: entries)
    monkeypatch.setattr(expand_lsj, "load_wiktionary_forms", lambda: {})
    monkeypatch.setattr(expand_lsj, "load_lsj_head_texts", lambda: {
        "καταδείκνυμι": "Ion. aor. κατέδεξα Hdt.: discover.",
        "δέχομαι": "fut. δέξομαι: aor. ἐδεξάμην: take.",
        "ἀμφιέννυμι": "fut. ἀμφιέσω: put round."})
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
    pairs = write_pairs(tmp_path / "pairs.json", [
        ("κατεδέξατο", "καταδέχομαι", "verb"),
        ("κατέδεξεν", "καταδέχομαι", "verb"),
        ("καταδέξασθαι", "καταδέχομαι", "verb"),
        ("ἀμφιέσομαι", "ἀμφιάζω", "verb"),
        ("εἰκονίσεις", "εἰκονίζω", "verb")])
    monkeypatch.setattr(expand_lsj, "TREEBANK_PAIRS", (pairs,))
    heads = tmp_path / "grc_lexical_heads.json"
    heads.write_text(json.dumps(
        {"heads": {hw: ["lsj9_headwords"] for hw in entries}},
        ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(expand_lsj, "LEXICAL_HEADS", heads)
    audit = tmp_path / "audit.json"
    monkeypatch.setattr(expand_lsj, "VERB_AUDIT", audit)

    cells = {
        "εἰκονίζω": {"εἰκόνισε(ν)", "εἰκόνισα", "εἰκονικός", "εἰκονίσει",
                     "εἰκονίσεις"},
        "καταδείκνυμι": {"κατέδεξε(ν)", "κατεδέξατο", "καταδέξασθαι"},
        "καταδέχομαι": {"κατεδέξατο"},
        "ἀμφιέννυμι": {"ἀμφιέσομαι"},
    }

    def expand(_wtp, word, head_text="", supplement_text="",
               principal_part_forms=None):
        if head_text and word in cells:
            principal_part_forms |= cells[word]
        return {word}, ""

    monkeypatch.setattr(expand_lsj, "expand_verb", expand)
    monkeypatch.setattr(
        expand_lsj, "_expand_principal_part",
        lambda _wtp, headword, tense_args: cells.get(headword, set()))
    expand_lsj.expand_verbs()
    result = json.loads(lookup.read_text(encoding="utf-8"))
    # an attested cell of an inferred aorist
    assert result["εἰκόνισεν"] == "εἰκονίζω"
    # an unattested one, and one that is a headword itself
    assert "εἰκόνισα" not in result and "εἰκονικός" not in result
    assert "εἰκόνισε" not in result
    # a future that the noun in -σις also spells needs a treebank
    assert "εἰκονίσει" not in result
    assert result["εἰκονίσεις"] == "εἰκονίζω"
    # A cell on the stem the entry prints stands, whatever a treebank
    # reads, one spelling at a time,
    assert result["κατέδεξε"] == "καταδείκνυμι"
    assert result["κατέδεξεν"] == "καταδείκνυμι"
    assert result["ἀμφιέσομαι"] == "ἀμφιέννυμι"
    # unless the verb the treebank reads has the cell too: καταδέχομαι
    # takes ἐδεξάμην from δέχομαι.
    assert result["κατεδέξατο"] == "καταδέχομαι"
    assert result["κατεδεξατο"] == "καταδέχομαι"
    # A cell on the unaugmented stem, which no entry prints, gives way to
    # the headword the treebank reads.
    assert "καταδέξασθαι" not in result
    recorded = json.loads(audit.read_text(encoding="utf-8"))
    assert recorded["inferred"] == {
        "εἰκονίζω": {"kind": "regular", "from": "",
                     "cells": ["εἰκονίσεις", "εἰκόνισεν"]},
        "καταδέχομαι": {"kind": "compound", "from": "δέχομαι",
                        "cells": ["κατεδέξατο"]}}
    assert recorded["left_to_treebank"] == [
        {"form": "καταδέξασθαι", "verb": "καταδείκνυμι", "part": "derived",
         "treebank": "καταδέχομαι"},
        {"form": "κατεδέξατο", "verb": "καταδείκνυμι", "part": "printed",
         "treebank": "καταδέχομαι"}]
    assert recorded["counts"]["inferred_noun_twins"] == 1
    assert set(recorded["inputs"]) == {
        "form_profile.db", "grc_lexical_heads.json", "pairs.json"}


def test_verb_pass_names_a_missing_corpus_table(monkeypatch, tmp_path):
    monkeypatch.setattr(expand_lsj, "FORM_PROFILE",
                        tmp_path / "form_profile.db")
    monkeypatch.setattr(expand_lsj, "TREEBANK_PAIRS", ())
    with pytest.raises(SystemExit) as stopped:
        expand_lsj.expand_verbs()
    assert "form_profile.db" in str(stopped.value)
