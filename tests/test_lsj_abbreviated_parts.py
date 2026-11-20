"""Hyphenated principal parts of LSJ written out from the headword (#145)."""
import sys
import unicodedata
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))
import expand_lsj  # noqa: E402
from lsj_principal_parts import (  # noqa: E402
    expand_abbreviation, parse_principal_parts)


def nfc(text):
    return unicodedata.normalize("NFC", text)


@pytest.mark.parametrize("headword,fragment,expected", [
    # An ending after the stem vowel, or after the contract vowel it
    # lengthens (not the η before it in ἀμφισβητέω).
    ("ἐπιορκέω", "ήσω", "ἐπιορκήσω"),
    ("ἀμφισβητέω", "ήσω", "ἀμφισβητήσω"),
    ("παιδεύω", "εύσω", "παιδεύσω"),
    ("σαλπίζω", "ιῶ", "σαλπιῶ"),
    # A form of the simplex after the preverb, in the preverb's spelling
    # before it.
    ("προβάλλω", "βᾰλῶ", "προβᾰλῶ"),
    ("προβάλλω", "βέβληκα", "προβέβληκα"),
    ("καταπίπτω", "πεσοῦμαι", "καταπεσοῦμαι"),
    ("προΐστημι", "στήσω", "προστήσω"),
    ("ἐφίστημι", "στήσω", "ἐπιστήσω"),
    # A past tense that begins with the augment.
    ("ἐμβάλλω", "έβαλον", "ἐνέβαλον"),
    ("κατανέω", "ένησα", "κατένησα"),
    ("συμψάω", "εψήσθην", "συνεψήσθην"),
    ("ἐπαινέω", "ῄνεσα", "ἐπῄνεσα"),
    # The part repeats a later preverb.
    ("ἐξαπατάω", "απατηθήσομαι", "ἐξαπατηθήσομαι"),
    ("προσεξανίσταμαι", "ανέστην", "προσεξανέστην"),
])
def test_expand_abbreviation(headword, fragment, expected):
    assert expand_abbreviation(headword, fragment) == nfc(expected)


@pytest.mark.parametrize("headword,fragment", [
    ("γράφω", "ψω"),            # the stem consonant changes
    ("ἀνατρέχω", "έδραμον"),    # a suppletive stem
    ("ἀνατρέχω", "δραμοῦμαι"),
    ("ἀποστρέφω", "έστραμμαι"),  # a perfect is no ending of the present
    ("κατανέω", "ησα"),         # an aorist is not built from an ending
])
def test_unreadable_abbreviation_is_left_out(headword, fragment):
    assert expand_abbreviation(headword, fragment) is None


def test_parser_writes_out_the_abbreviation():
    parts = parse_principal_parts("fut. -στήσω: aor. 1 προὔστησα, part.",
                                  "προΐστημι")
    assert parts["fut"] == "προστήσω"
    assert parts["aor1"] == nfc("προὔστησα")
    assert "fut" not in parse_principal_parts(
        "fut. -στήσω: aor. 1 προὔστησα", "προΐστημι", abbreviations=False)


def test_a_full_form_outranks_an_abbreviation():
    text = ("fut. -ιῶ LXXNu.10.4: aor. ἐσάλπιγξα:—later, fut. σαλπίσω "
            "1 Ep.Cor.15.52")
    assert expand_lsj.merged_principal_parts("σαλπίζω", text)["fut"] == (
        nfc("σαλπίσω"))
    # ... also one the earlier export gives.
    parts = expand_lsj.merged_principal_parts(
        "σαλπίζω", "fut. -ιῶ: aor. ἐσάλπιγξα", "fut. σαλπίσω")
    assert parts["fut"] == nfc("σαλπίσω")
