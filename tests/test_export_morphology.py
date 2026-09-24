import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dilemma.form_sanitize import has_editorial_sigla
from export_hunspell import exact_form_key
from export_morphology import (
    _derive_elision_pairs,
    _derive_nu_forms,
    _load_lemma_forms,
)


VERB_TAGS = [
    "third-person",
    "plural",
    "present",
    "indicative",
    "active",
]


def test_nu_derivation_rejects_editorial_sigla(tmp_path):
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(json.dumps([
        {
            "form": ")λπίζουσι",
            "lemma": ")λπίζω",
            "pos": "verb",
            "tags": VERB_TAGS,
        },
        {
            "form": "ἐλπίζουσι",
            "lemma": "ἐλπίζω",
            "pos": "verb",
            "tags": VERB_TAGS,
        },
    ], ensure_ascii=False), encoding="utf-8")

    forms = _derive_nu_forms([pairs_path])

    assert ")λπίζουσι" not in forms
    assert "ἐλπίζουσι" in forms
    assert not any(has_editorial_sigla(form) for form in forms)


def test_elision_loading_and_derivation_reject_editorial_sigla(tmp_path):
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(json.dumps([
        {"form": "λέγε", "lemma": "λέγω"},
        {"form": "λέγ᾽", "lemma": "λέγω"},
        {"form": ")φέρε", "lemma": "φέρω"},
        {"form": "φέρ᾽", "lemma": ")φέρω"},
    ], ensure_ascii=False), encoding="utf-8")

    lemma_forms = _load_lemma_forms([pairs_path], None)
    assert lemma_forms == {"λέγω": {"λέγε", "λέγ᾽"}}

    # Keep the derivation guard independent of the loader so callers passing
    # an already-built mapping cannot reintroduce contaminated entries.
    lemma_forms[")ἄγω"] = {"ἄγε", "ἄγ᾽"}
    lemma_forms["φέρω"] = {")φέρε", ")φέρ᾽"}
    pairs = _derive_elision_pairs(lemma_forms)

    assert pairs["λέγε"] == "λέγ᾽"
    assert not any(
        has_editorial_sigla(full) or has_editorial_sigla(elided)
        for full, elided in pairs.items()
    )


def test_an_elided_form_keeps_its_own_full_form_stem_marks():
    # εἶπε and εἰπέ elide to different spellings, so neither one's corpus
    # count says anything about which belongs to which full form.
    lemma_forms = {"λέγω": {"Εἶπε", "Εἰπέ", "εἶπ᾽", "εἴπ᾽"}}
    pairs = _derive_elision_pairs(lemma_forms, {exact_form_key("εἴπ᾽"): 138})
    assert pairs["Εἶπε"] == "εἶπ᾽"
    assert pairs["Εἰπέ"] == "εἴπ᾽"


def test_the_corpus_count_settles_an_otherwise_tied_elision():
    # Neither spelling carries Εἴτε's stem marks, and both score the same
    # on case, breathing and accent; εἵτ᾽ has 2 corpus tokens to εἴτ᾽'s 765.
    lemma_forms = {"εἴτε": {"Εἴτε", "εἴτ᾽", "εἵτ᾽"}}
    freq = {exact_form_key("εἴτ᾽"): 765, exact_form_key("εἵτ᾽"): 2}
    assert _derive_elision_pairs(lemma_forms, freq)["Εἴτε"] == "εἴτ᾽"


def test_a_candidate_the_orthography_rules_reject_loses():
    # τὸτ᾽ carries a grave on an elided word. It has the same corpus count
    # as τότ᾽, so nothing below the structural check separates them.
    lemma_forms = {"τότε": {"Τότε", "τότ᾽", "τὸτ᾽"}}
    freq = {exact_form_key("τότ᾽"): 1183, exact_form_key("τὸτ᾽"): 1183}
    assert _derive_elision_pairs(lemma_forms, freq)["Τότε"] == "τότ᾽"


def test_the_table_does_not_depend_on_set_iteration_order():
    # The candidates arrive from a set, whose iteration order varies with
    # the interpreter's hash seed, so ties have to break on the spelling.
    lemma_forms = {"ζζύω": {"Ζζύε", "ζζύ᾽", "ζζὺ᾽"}}
    first = _derive_elision_pairs(lemma_forms, {})
    again = _derive_elision_pairs({"ζζύω": set(reversed(sorted(
        lemma_forms["ζζύω"])))}, {})
    assert first == again


def test_an_elided_oxytone_throws_its_accent_back():
    # Smyth 174: ἀνδρί loses its final vowel and the accent goes back onto
    # the penult as an acute. The full form's own stem carries no mark, so
    # the stem-marks test alone would prefer the bare spelling.
    lemma_forms = {"ἀνήρ": {"ἀνδρί", "ἄνδρ᾽", "ἀνδρ᾽"}}
    assert _derive_elision_pairs(lemma_forms, {})["ἀνδρί"] == "ἄνδρ᾽"


def test_prepositions_and_conjunctions_lose_the_accent_instead():
    # The other half of Smyth 174. οὐδέ is not one of the pinned ten, so
    # nothing else would keep it bare.
    lemma_forms = {"οὐδέ": {"οὐδέ", "οὐδ᾽", "οὔδ᾽"}}
    assert _derive_elision_pairs(lemma_forms, {})["οὐδέ"] == "οὐδ᾽"


def test_a_form_that_is_not_oxytone_keeps_its_marks_where_they_were():
    # The retraction rule has to say nothing here, or it would override the
    # stem-marks test for every paroxytone and properispomenon.
    lemma_forms = {"ἄλλος": {"ἄλλε", "ἄλλ᾽", "ἆλλ᾽"}}
    assert _derive_elision_pairs(lemma_forms, {})["ἄλλε"] == "ἄλλ᾽"
