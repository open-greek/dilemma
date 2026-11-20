"""Stems the LSJ verb pass hands to the conjugation module.

A past tense needs its augmented stem for the indicative and its
unaugmented one for everything else. These tests cover reading the augment
after a preverb, the cases where only the indicative is built, the parser
fixes that feed it, and the entry-text lookup by spelling.
"""
import importlib.util
import sys
import unicodedata
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))
import expand_lsj
from lsj_principal_parts import (
    INDICATIVE_ONLY, derive_grc_conj_args, ends_at_label,
    parse_principal_parts, single_augment_stem, stem_preverb,
    unaugmented_stem,
)


def nfc(text):
    return unicodedata.normalize("NFC", text)


@pytest.mark.parametrize("headword,augmented,plain,prefix", [
    # simplex
    ("λύω", "ἐλυσ", "λυσ", ""),
    ("ἀκούω", "ἠκουσ", "ἀκουσ", ""),
    ("ἐλπίζω", "ἠλπισ", "ἐλπισ", ""),
    ("αἰτέω", "ᾐτησ", "αἰτησ", ""),
    ("οἰκέω", "ᾠκησ", "οἰκησ", ""),
    ("αὐξάνω", "ηὐξησ", "αὐξησ", ""),
    ("ἐργάζομαι", "εἰργασ", "ἐργασ", ""),
    ("ἔχω", "ἐσχ", "σχ", ""),
    ("ῥήγνυμι", "ἐρρηξ", "ῥηξ", ""),
    ("ἱδρύω", "ἱδρυσ", "ἱδρυσ", ""),
    ("βούλομαι", "ἠβουληθ", "βουληθ", ""),
    ("λῃστεύω", "ἐλῃστευθ", "λῃστευθ", ""),
    # suppletive aorists
    ("ἔρχομαι", "ἠλθ", "ἐλθ", ""),
    ("αἱρέω", "εἱλ", "ἑλ", ""),
    ("ὁράω", "εἰδ", "ἰδ", ""),
    ("αἴρω", "ἠρ", "ἀρ", ""),
    # the augment follows the preverb, which regains its present spelling
    ("ἀπολύω", "ἀπελυσ", "ἀπολυσ", "ἀπ"),
    ("ἀντιθάπτω", "ἀντεταφ", "ἀντιταφ", "ἀντ"),
    ("ἐκβάλλω", "ἐξεβαλ", "ἐκβαλ", "ἐξ"),
    ("συλλέγω", "συνελεξ", "συλλεξ", "συν"),
    ("ἐμπίπτω", "ἐνεπεσ", "ἐμπεσ", "ἐν"),
    ("ἐνθρώσκω", "ἐνεθορ", "ἐνθορ", "ἐν"),
    ("ἀνοιδέω", "ἀνῳδησ", "ἀνοιδησ", "ἀν"),
    ("ἀπάγω", "ἀπηγαγ", "ἀπαγαγ", "ἀπ"),
    ("εἰσφέρω", "εἰσηνεγκ", "εἰσενεγκ", "εἰσ"),
    ("ἐκφέρω", "ἐξηνεγκ", "ἐξενεγκ", "ἐξ"),
    ("συμφέρω", "συνηνεγκ", "συνενεγκ", "συν"),
    ("ἀφαιρέω", "ἀφειλ", "ἀφελ", "ἀφ"),
    ("ἀπέρχομαι", "ἀπηλθ", "ἀπελθ", "ἀπ"),
    ("ἐπαίρω", "ἐπηρ", "ἐπαρ", "ἐπ"),
    ("παρέχω", "παρεσχ", "παρασχ", "παρ"),
    ("καθίστημι", "κατεστησ", "καταστησ", "κατ"),
    ("συνίστημι", "συνεστησ", "συστησ", "συν"),
    ("ἐξίστημι", "ἐξεστησ", "ἐκστησ", "ἐξ"),
    ("συντρέχω", "συνεδραμ", "συνδραμ", "συν"),
    ("ἀπορρήγνυμι", "ἀπερρηξ", "ἀπορρηξ", "ἀπ"),
    ("συρράπτω", "συνερραψ", "συρραψ", "συν"),
    ("ξυμβάλλω", "ξυνεβαλ", "ξυμβαλ", "ξυν"),
    ("ἐγκαταλείπω", "ἐγκατελιπ", "ἐγκαταλιπ", "ἐγκατ"),
    ("προβάλλω", "προεβαλ", "προβαλ", "προ"),
    ("περιβάλλω", "περιεβαλ", "περιβαλ", "περι"),
    # an augment before the preverb, or a word that only looks compound
    ("καθίζω", "ἐκαθισ", "καθισ", ""),
    ("ἐπείγω", "ἠπειξ", "ἐπειξ", ""),
    ("ἀμφιέννυμι", "ἠμφιεσ", "ἀμφιεσ", ""),
    # no augment to remove
    ("ἀφικνέομαι", "ἀφικ", "ἀφικ", "ἀφ"),
    ("περιχέω", "περιχευ", "περιχευ", ""),
    ("ἐντύνω", "ἐντῡν", "ἐντῡν", ""),
    # Ionic ἔνεικα beside ἤνεικα: the ε is the stem's, not an augment
    ("φέρω", "ἐνεικ", "ἐνεικ", ""),
    ("ἀποφέρω", "ἀπενεικ", "ἀπενεικ", ""),
    ("προσφέρω", "προσενεικ", "προσενεικ", ""),
    ("ἀποφέρω", "ἀπηνεικ", "ἀπενεικ", "ἀπ"),
    # ... while the passive of λύω has one
    ("λύω", "ἐλυθ", "λυθ", ""),
    # ἕσσα beside ἕννυμι: nothing begins with σσ, so the ε is the stem's
    ("ἕννυμι", "ἑσσ", "ἑσσ", ""),
    ("περιέννυμι", "περιεσσ", "περιεσσ", "περι"),
    ("ἐφέννυμι", "ἐφεσσ", "ἐφεσσ", "ἐφ"),
    # ... while ἔσχον and ἔστησα lose theirs
    ("ἔχω", "ἐσχ", "σχ", ""),
    ("ἵστημι", "ἐστησ", "στησ", ""),
])
def test_unaugmented_stem(headword, augmented, plain, prefix):
    assert unaugmented_stem(headword, augmented) == (plain, prefix)
    decomposed = unicodedata.normalize("NFD", augmented)
    assert unaugmented_stem(headword, decomposed) == (plain, prefix)


@pytest.mark.parametrize("headword,augmented", [
    ("ἀμφισβητέω", "ἠμφεσβητησ"),   # second augment after the preverb
    ("ἐνοχλέω", "ἠνωχλησ"),
    ("ἐπανορθόω", "ἐπηνωρθωσ"),
    ("ὠθέω", "ἐωσ"),                # syllabic augment before a vowel
    ("διαΐσσω", "διῃξ"),
    # ἕσσα beside ἵζω: neither an augment nor the present's vowel
    ("ἵζω", "ἑσσ"),
])
def test_unreadable_augment_is_not_guessed(headword, augmented):
    assert unaugmented_stem(headword, augmented)[0] is None


@pytest.mark.parametrize("headword,augmented,plain,prefix", [
    # The augment contracted with the ο of προ- (crasis).
    ("προβάλλω", "προὐβαλ", "προβαλ", "προ"),
    ("προΐστημι", "προὐστησ", "προστησ", "προ"),
    # A preverb kept whole before a vowel, the augment after it.
    ("ἐπιορκέω", "ἐπιωρκησ", "ἐπιορκησ", "ἐπι"),
    # A double augment: both are removed.
    ("παροινέω", "ἐπαρῳνηθ", "παροινηθ", ""),
    ("διαιτάω", "ἐδιῃτησ", "διαιτησ", ""),
])
def test_crasis_hiatus_and_double_augment(headword, augmented, plain, prefix):
    assert unaugmented_stem(headword, augmented) == (plain, prefix)


def test_single_augment_of_a_double_augment():
    assert single_augment_stem("παροινέω", "ἐπαρῳνηθ") == "παρῳνηθ"
    assert single_augment_stem("διαιτάω", "ἐδιῃτησ") == "διῃτησ"
    assert single_augment_stem("λύω", "ἐλυσ") is None
    assert single_augment_stem("παροινέω", "παρῳνηθ") is None
    args = derive_grc_conj_args({"aor_p": "ἐπαρῳνήθην"}, "παροινέω")
    assert args["aor-pass"] == ["aor-1", "", "", "ἐπαρῳνηθ", "παροινηθ",
                                "form=pass"]
    assert args["aor-pass-single"] == ["aor-1", "", "", "παρῳνηθ", "παρῳνηθ",
                                       "form=pass", INDICATIVE_ONLY]


def test_iterative_printed_as_an_aorist_has_an_indicative_only():
    args = derive_grc_conj_args({"aor2": "προβάλεσκον"}, "προβάλλω")
    assert args["aor-2"][-1] == INDICATIVE_ONLY


def test_unreadable_stem_still_names_its_preverb():
    assert unaugmented_stem("διαφίημι", "διαφηκ") == (None, "διαφ")
    assert unaugmented_stem("ἐνωθέω", "ἐνεωσ") == (None, "ἐν")


@pytest.mark.parametrize("headword,stem,prefix", [
    ("ἀπολύω", "ἀπολελυκ", "ἀπο"),
    ("ἀποστέλλω", "ἀπεσταλκ", "ἀπ"),
    ("ἀφικνέομαι", "ἀφῑγ", "ἀφ"),
    ("διαφύομαι", "διαπεφῡκ", "δια"),       # not δι-απ-
    ("μετάγω", "μεταγηοχ", "μετ"),          # not μετα-
    ("παραπόλλυμι", "παραπολωλ", "παραπ"),
    ("ἀπαναλίσκω", "ἀπαναλωκ", "ἀπαν"),     # not ἀπανα-λίσκω
    ("παρίημι", "παρεικ", "παρ"),
    ("λύω", "λελυκ", ""),
])
def test_stem_preverb(headword, stem, prefix):
    assert stem_preverb(headword, stem) == prefix


def test_past_tenses_put_the_augmented_stem_first():
    parts = {"fut": "λύσω", "aor": "ἔλυσα", "pf": "λέλυκα",
             "pf_mp": "λέλυμαι", "aor_p": "ἐλύθην", "fut_p": "λυθήσομαι"}
    assert derive_grc_conj_args(parts, "λύω") == {
        "fut": ["fut", "λυσ", "λυθ"],
        "aor-pass": ["aor-1", "", "", "ἐλυθ", "λυθ", "form=pass"],
        "aor-1": ["aor-1", "ἐλυσ", "λυσ"],
        "perf": ["perf", "λελυκ", "λελυ"],
        "plup": ["plup", "λελυκ", "λελυ"],
        "plup-aug": ["plup", "ἐλελυκ", "ἐλελυ"],
    }


def test_middle_parts_build_no_active():
    parts = {"fut": "δέξομαι", "aor": "ἐδεξάμην", "aor_p": "ἐδέχθην"}
    args = derive_grc_conj_args(parts, "δέχομαι")
    assert args["fut"] == ["fut", "δεξ", "form=mid"]
    assert args["aor-1"] == ["aor-1", "ἐδεξ", "δεξ", "form=mid"]
    assert args["aor-pass"] == ["aor-1", "", "", "ἐδεχθ", "δεχθ", "form=pass"]
    assert derive_grc_conj_args({"aor2": "ἀφικόμην"}, "ἀφικνέομαι") == {
        "aor-2": ["aor-2", "ἀφικ", "ἀφικ", "form=mid", "prefix=ἀφ"]}


def test_contracted_futures_use_the_contract_templates():
    assert derive_grc_conj_args({"fut": "βαλῶ"}, "βάλλω")["fut"] == [
        "fut-con-e", "βαλ"]
    assert derive_grc_conj_args({"fut": "νομιῶ"}, "νομίζω")["fut"] == [
        "fut-con-e", "νομι"]
    assert derive_grc_conj_args({"fut": "βιβῶ"}, "βιβάζω")["fut"] == [
        "fut-con-a", "βιβ"]
    assert derive_grc_conj_args({"fut": "πεσοῦμαι"}, "πίπτω")["fut"] == [
        "fut-con-e", "πεσ", "form=mid"]
    assert derive_grc_conj_args({"fut": "λύσω"}, "λύω")["fut"] == ["fut", "λυσ"]


def test_compound_passive_aorist_alone():
    args = derive_grc_conj_args({"aor_p": "ἀντετάφην"}, "ἀντιθάπτω")
    assert args == {"aor-pass": ["aor-1", "", "", "ἀντεταφ", "ἀντιταφ",
                                 "form=pass", "prefix=ἀντ"]}


def test_passive_deponent_aorist_under_the_plain_label():
    args = derive_grc_conj_args({"aor": "ἐβουλήθην"}, "βούλομαι")
    assert args == {"aor-pass": ["aor-1", "", "", "ἐβουληθ", "βουληθ",
                                 "form=pass"]}


def test_two_passive_aorists_are_both_built():
    text = ("Pass., fut. -πνῐγήσομαι : aor. I ἀπεπνίχθην Aret.SA1.7: "
            "aor. 2 ἀπεπνίγην [ῐ] (v. infr.)")
    parts = parse_principal_parts(text, "ἀποπνίγω")
    assert parts["aor_p"] == "ἀπεπνίχθην"
    assert parts["aor_p2"] == "ἀπεπνίγην"
    args = derive_grc_conj_args(parts, "ἀποπνίγω")
    assert args["aor-pass"][3:5] == ["ἀπεπνιχθ", "ἀποπνιχθ"]
    assert args["aor-pass-2"][3:5] == ["ἀπεπνιγ", "ἀποπνιγ"]


def test_indicative_only_where_the_other_moods_have_another_stem():
    # ἔθηκα: θῶ, θεῖναι, θείς are not built on the κ stem.
    args = derive_grc_conj_args({"aor": "ἔθηκα"}, "τίθημι")
    assert args["aor-1"] == ["aor-1", "ἐθηκ", "θηκ", "form=act", INDICATIVE_ONLY]
    assert args["aor-mid"] == ["aor-1", "ἐθηκ", "θηκ", "form=mid"]
    # Root aorists go like a passive only in the indicative.
    for part, headword in (("ἀνθυπέστην", "ἀνθυφίσταμαι"), ("ὑπέφθην", "ὑποφθάνω")):
        args = derive_grc_conj_args({"aor": part, "aor_p": part}, headword)
        assert args["aor-pass"][-1] == INDICATIVE_ONLY
    # An augment that cannot be read against the present.
    args = derive_grc_conj_args({"aor": "ἠμφεσβήτησα"}, "ἀμφισβητέω")
    assert args["aor-1"] == ["aor-1", "ἠμφεσβητησ", "ἠμφεσβητησ", INDICATIVE_ONLY]


def test_printed_accent_decides_whether_the_preverb_is_passed():
    assert "prefix=παρ" in derive_grc_conj_args({"aor2": "παρέσχον"}, "παρέχω")["aor-2"]
    assert "prefix=ἐν" in derive_grc_conj_args({"aor": "ἐνεῖρα"}, "ἐνείρω")["aor-1"]
    # LSJ prints ἄφευσα, with the accent on the preverb.
    assert derive_grc_conj_args({"aor1": "ἄφευσα"}, "ἀφεύω")["aor-1"] == [
        "aor-1", "ἀφευσ", "ἀφευσ"]
    assert derive_grc_conj_args({"pf_mp": "ἀφῖγμαι"}, "ἀφικνέομαι")["perf"] == [
        "perf", "", "ἀφῑγ", "prefix=ἀφ"]


def test_augmented_pluperfect_only_for_a_reduplicated_stem():
    args = derive_grc_conj_args({"pf": "ἀπολέλυκα"}, "ἀπολύω")
    assert args["plup-aug"] == ["plup", "ἀπελελυκ", "", "prefix=ἀπ"]
    assert "plup-aug" not in derive_grc_conj_args({"pf": "ἤγγελκα"}, "ἀγγέλλω")
    assert "plup-aug" not in derive_grc_conj_args(
        {"pf": "ἀπανάλωκα"}, "ἀπαναλίσκω")


def test_augmented_pluperfect_keeps_the_breathing_of_a_diphthong():
    # εἰσ- carries its breathing on the second vowel.
    args = derive_grc_conj_args({"pf": "εἰσδέδορκα"}, "εἰσδέρκομαι")
    assert args["plup-aug"] == ["plup", "εἰσεδεδορκ", "", "prefix=εἰσ"]
    args = derive_grc_conj_args({"pf": "ἐκλέλυκα"}, "ἐκλύω")
    assert args["plup-aug"] == ["plup", "ἐξελελυκ", "", "prefix=ἐξ"]


def test_contracted_imperfect_takes_the_contract_template():
    assert derive_grc_conj_args({"impf": "ἀτίμων"}, "ἀτιμάω")["imperf"] == [
        "imperf-con-a", "ἀτιμ"]
    assert derive_grc_conj_args({"impf": "ἐβουλόμην"}, "βούλομαι")["imperf"] == [
        "imperf", "ἐβουλ", "form=mp"]
    # δίδωμι is no contract verb: ἐδίδουν is left alone.
    assert "imperf" not in derive_grc_conj_args({"impf": "ἐδίδουν"}, "δίδωμι")


def test_roman_numerals_number_the_aorists():
    parts = parse_principal_parts(
        "fut. εἰσοίσω E.Ba.367: aor. I εἰσήνεγκα", "εἰσφέρω")
    assert parts["aor1"] == "εἰσήνεγκα"
    parts = parse_principal_parts("aor. II ἔλιπον", "λείπω")
    assert parts["aor2"] == nfc("ἔλιπον")


def test_text_that_opens_inside_a_section():
    parts = parse_principal_parts(
        "Pass. (v. infr.), aor. ἐλῃστεύθην D.:—practise robbery", "λῃστεύω")
    assert parts == {"aor_p": "ἐλῃστεύθην"}
    parts = parse_principal_parts("Pass., aor. ἀντετάφην", "ἀντιθάπτω")
    assert parts == {"aor_p": "ἀντετάφην"}


def test_ending_without_a_breathing_is_not_a_form():
    assert parse_principal_parts("fut. ύσω [ῡ], E.Alc.294", "φιτύω") == {}
    assert parse_principal_parts("fut. ανῶ Pl.R.501e", "ἀγριαίνω") == {}
    assert parse_principal_parts("fut. ἀγριανῶ", "ἀγριαίνω") == {
        "fut": "ἀγριανῶ"}


def test_first_person_plural_is_not_an_aorist_stem():
    assert derive_grc_conj_args({"aor": "ποτεδασσάμεθα"}, "ποδδατέομαι") == {}


def test_label_at_the_end_is_completed_by_the_earlier_text():
    current = "fut. -ήσω: aor."
    earlier = "ἀνῴδησα , e: pf. ἀνῴδηκα :—swell up"
    assert ends_at_label(current)
    assert not ends_at_label("fut. ἀνοιδήσω: swell up")
    parts = expand_lsj.merged_principal_parts("ἀνοιδέω", current, earlier)
    assert parts["aor"] == "ἀνῴδησα"
    assert parts["pf"] == "ἀνῴδηκα"
    # An earlier text that starts over is not a continuation.
    restart = "fut. -πνίξω : aor. inf. ἀποπνεῖξαι:—choke:—Pass., aor. 2 ἀπεπνίγην"
    parts = expand_lsj.merged_principal_parts(
        "ἀποπνίγω", "[ῑ], fut. -πνίξω Pl.Com.198: aor.", restart)
    # The future is the abbreviation written out (-πνίξω after ἀπο-).
    assert parts == {"aor_p": "ἀπεπνίγην", "fut": "ἀποπνίξω"}


def test_earlier_text_continues_the_aorists_it_repeats():
    current = "aor. ἀνέθηλα Ael.VH5.4, NA2.25: aor. 2"
    earlier = ("aor. ἀνέθηλα Ael.VH5.4, NA2.25: aor. 2 ἀνέθαλον Ep.Phil.4.10"
               ":—shoot up again")
    parts = expand_lsj.merged_principal_parts("ἀναθάλλω", current, earlier)
    assert parts["aor"] == "ἀνέθηλα"
    assert parts["aor2"] == "ἀνέθαλον"
    args = derive_grc_conj_args(parts, "ἀναθάλλω")
    assert args["aor-1"][:3] == ["aor-1", "ἀνεθηλ", "ἀναθηλ"]
    assert args["aor-2"][:3] == ["aor-2", "ἀνεθαλ", "ἀναθαλ"]
    # An earlier text with another aorist under the same label adds none.
    parts = expand_lsj.merged_principal_parts(
        "παύω", "aor. ἔπαυσα: pf. πέπαυκα", "aor. ἔπαυσα: aor. 2 ἔπαυον")
    assert parts["aor2"] == "ἔπαυον"
    parts = expand_lsj.merged_principal_parts(
        "παύω", "aor. ἔπαυσα: pf. πέπαυκα", "aor. ἔπαυα: aor. 2 ἔπαυον")
    assert "aor2" not in parts


def test_root_aorist_keeps_its_cells_beside_a_numbered_aorist():
    parts = {"aor": "ὑπέφθην", "aor1": "ὑπέφθᾰσα"}
    args = derive_grc_conj_args(parts, "ὑποφθάνω")
    assert args["aor-1"][:3] == ["aor-1", "ὑπεφθᾰσ", "ὑποφθᾰσ"]
    assert args["aor-pass"][3:5] == ["ὑπεφθ", "ὑποφθ"]


def test_head_text_by_spelling_without_quantity_marks():
    texts = {
        nfc("ἀγῑνέω"): "impf. ἠγίνεον",
        nfc("θρῴσκω"): "aor. 2 ἔθορον",
        nfc("θᾱλέω"): "Dor. for θηλέω",
        nfc("θᾰλέω"): "= θάλλω",
        "λύω": "fut. λύσω",
    }
    find = expand_lsj.head_text_finder(texts)
    assert find("λύω") == "fut. λύσω"
    assert find("ἀγινέω") == "impf. ἠγίνεον"
    assert find("θρώσκω") == "aor. 2 ἔθορον"
    assert find("θαλέω") == ""          # two entries share the spelling
    assert find("παύω") == ""


def test_indicative_cells_drops_the_other_moods():
    table = "\n".join([
        '! rowspan="4" | active',
        "! indicative",
        '| class="grc-conj-finite-form" | ἔθηκᾰ',
        "|-",
        "! subjunctive",
        '| class="grc-conj-finite-form" | θήκω',
        "|-",
        '! colspan="2" | infinitive',
        '| colspan="3" class="grc-conj-nonfinite-form" | θῆκαι',
        "|-",
        '! rowspan="4" | middle',
        "! indicative",
        '| class="grc-conj-finite-form" | ἐθηκᾰ́μην',
        "",
    ])
    kept = expand_lsj._indicative_cells(table)
    assert "ἔθηκᾰ" in kept and "ἐθηκᾰ́μην" in kept
    assert "θήκω" not in kept and "θῆκαι" not in kept


def test_principal_part_forms_can_be_kept_apart(monkeypatch):
    monkeypatch.setattr(expand_lsj, "_VERB_CACHE",
                        {("λύω", "pres", "λυ", ""): frozenset({"λύω", "λύεις"})})
    monkeypatch.setattr(expand_lsj, "expand_principal_parts",
                        lambda *args: {"ἔλυσα", "λύεις"})
    extra = set()
    forms, error = expand_lsj.expand_verb(
        None, "λύω", "aor. ἔλυσα", principal_part_forms=extra)
    assert (forms, error) == ({"λύω", "λύεις"}, "")
    assert extra == {"ἔλυσα"}
    forms, _ = expand_lsj.expand_verb(None, "λύω", "aor. ἔλυσα")
    assert forms == {"λύω", "λύεις", "ἔλυσα"}


needs_lua = pytest.mark.skipif(
    not expand_lsj.WTP_DB.exists()
    or importlib.util.find_spec("wikitextprocessor") is None,
    reason="requires the optional pinned Wiktionary Lua cache and runtime",
)


def lua_forms(word, parts):
    expand_lsj._PP_CACHE.clear()
    wtp = expand_lsj.get_wtp()
    forms = set()
    for args in derive_grc_conj_args(parts, word).values():
        forms.update(expand_lsj._expand_principal_part(wtp, word, args))
    return forms


@needs_lua
@pytest.mark.parametrize("word,parts,present,absent", [
    ("ἀκούω", {"aor1": "ἤκουσα", "aor_p": "ἠκούσθην"},
     ["ἤκουσα", "ἀκοῦσαι", "ἀκούσας", "ἠκούσθην", "ἀκουσθῆναι", "ἀκουσθείς"],
     ["ἀκούσθην", "ἠκουσθείς", "ἠκουσθῆναι", "ἠκουσθῶ"]),
    ("ἀντιθάπτω", {"aor_p": "ἀντετάφην"},
     ["ἀντετάφην", "ἀντετάφη", "ἀντιταφῆναι", "ἀντιταφείς"],
     ["ἀντεταφείς", "ἀντεταφῆναι", "ἀντιτάφην"]),
    ("εἰσφέρω", {"aor1": "εἰσήνεγκα"},
     ["εἰσήνεγκα", "εἰσήνεγκας", "εἰσενέγκαι", "εἰσενέγκας"],
     ["εἰσηνέγκας", "εἰσηνέγκαι"]),
    ("ἐνθρώσκω", {"aor2": "ἐνέθορον"},
     ["ἐνέθορον", "ἐνθορεῖν", "ἐνθορών"], ["νεθορεῖν", "ἐνεθορεῖν"]),
    ("παρέχω", {"aor2": "παρέσχον"},
     ["παρέσχον", "παρασχεῖν", "παρασχών"], ["πάρεσχον", "παρεσχεῖν"]),
    ("ἀφικνέομαι", {"pf_mp": "ἀφῖγμαι"}, ["ἀφῖγμαι", "ἀφῖχθαι"], ["ἄφιγμαι"]),
    ("τίθημι", {"aor": "ἔθηκα"},
     ["ἔθηκα", "ἔθηκας", "ἐθηκάμην", "θηκάμενος"], ["θήκας", "θήκαιμι", "θήκω"]),
    ("βάλλω", {"fut": "βαλῶ"}, ["βαλῶ", "βαλεῖς", "βαλεῖν"], ["βάλεις"]),
    ("δέχομαι", {"aor": "ἐδεξάμην"},
     ["ἐδεξάμην", "δέξασθαι", "δεξάμενος"], ["ἔδεξα", "δέξας"]),
    ("λύω", {"pf": "λέλυκα"}, ["λέλυκα", "λελύκειν", "ἐλελύκειν"], []),
    ("εἰσδέρκομαι", {"pf": "εἰσδέδορκα"},
     ["εἰσδέδορκα", "εἰσεδεδόρκειν", "εἰσεδεδορκέτην"],
     ["εισεδεδόρκειν", "εισεδεδορκέτην"]),
    ("ἀποφέρω", {"aor": "ἀπένεικα"},
     ["ἀπένεικα", "ἀπενεῖκαι", "ἀπενείκας"],
     ["ἀπονεῖκαι", "ἀπονείκας", "ἀπονείκειε"]),
])
def test_real_lua_cells(word, parts, present, absent):
    forms = lua_forms(word, parts)
    assert not [form for form in present if nfc(form) not in forms]
    assert not [form for form in absent if nfc(form) in forms]
