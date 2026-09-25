import json
import sys

import pytest
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
    assert pairs["Εἶπε"] == "Εἶπ᾽"
    assert pairs["Εἰπέ"] == "Εἴπ᾽"


def test_the_corpus_count_settles_an_otherwise_tied_elision():
    # Neither spelling carries Εἴτε's stem marks, and both score the same
    # on case, breathing and accent; εἵτ᾽ has 2 corpus tokens to εἴτ᾽'s 765.
    lemma_forms = {"εἴτε": {"Εἴτε", "εἴτ᾽", "εἵτ᾽"}}
    freq = {exact_form_key("εἴτ᾽"): 765, exact_form_key("εἵτ᾽"): 2}
    assert _derive_elision_pairs(lemma_forms, freq)["Εἴτε"] == "Εἴτ᾽"


def test_a_candidate_the_orthography_rules_reject_loses():
    # τὸτ᾽ carries a grave on an elided word. It has the same corpus count
    # as τότ᾽, so nothing below the structural check separates them.
    lemma_forms = {"τότε": {"Τότε", "τότ᾽", "τὸτ᾽"}}
    freq = {exact_form_key("τότ᾽"): 1183, exact_form_key("τὸτ᾽"): 1183}
    assert _derive_elision_pairs(lemma_forms, freq)["Τότε"] == "Τότ᾽"


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


def test_a_grave_oxytone_retracts_like_an_acute_one():
    # The grave is only the contextual spelling of an oxytone, and it is
    # the spelling a word carries mid-sentence, which is where elision
    # happens. Reported by Tonos: 59 entries were left bare.
    lemma_forms = {"αὐτός": {"αὐτὸ", "αὔτ᾽", "αὐτ᾽"}}
    assert _derive_elision_pairs(lemma_forms, {})["αὐτὸ"] == "αὔτ᾽"


def test_a_word_elision_cannot_touch_gets_no_entry():
    # Elision removes a short final vowel. η and ω are always long, a
    # circumflex or iota subscript marks length, and the second element of
    # a diphthong goes with the first.
    for full, elided in [("αὐτῷ", "αὐτ᾽"), ("αὐτῇ", "αὐτ᾽"),
                         ("δεῖ", "δέ᾽"), ("ἤδη", "ἠδ᾽"),
                         ("λόγοι", "λόγ᾽"), ("λόγου", "λόγ᾽")]:
        pairs = _derive_elision_pairs({"x": {full, elided}}, {})
        assert full not in pairs, f"{full} cannot elide"


def test_a_pair_is_dropped_when_every_candidate_is_junk():
    # Ranking only picks the least bad one, and writing that into
    # someone's text is worse than offering nothing.
    from export_hunspell import grc_orthography_reason
    assert grc_orthography_reason("ὃσδ᾽") is not None
    assert "ὅσδε" not in _derive_elision_pairs({"ὅσδε": {"ὅσδε", "ὃσδ᾽"}}, {})


def test_a_form_two_lemmas_claim_is_ranked_once():
    # The corpora carry a capitalized lemma Κατά whose one elided token
    # opened a sentence. Assigned lemma by lemma, whichever came last won,
    # and κατὰ took Κατ᾽ from it.
    lemma_forms = {"κατά": {"κατὰ", "κατ᾽", "κάτ᾽"},
                   "Κατά": {"κατὰ", "Κατ᾽"},
                   "Ἑλλάς": {"ἑλλάδα", "Ἑλλάδ᾽"}}
    pairs = _derive_elision_pairs(lemma_forms, {})
    assert pairs["κατὰ"] == "κατ᾽"
    assert pairs["ἑλλάδα"] == "ἑλλάδ᾽"
    reordered = dict(reversed(list(lemma_forms.items())))
    assert _derive_elision_pairs(reordered, {}) == pairs


def test_the_elided_form_takes_the_full_forms_case():
    lemma_forms = {"αὐτός": {"Αὐτὸ", "αὔτ᾽"}}
    assert _derive_elision_pairs(lemma_forms, {})["Αὐτὸ"] == "Αὔτ᾽"


def test_an_enclitics_accent_is_not_the_words_own():
    # χεῖρά τε: the acute on the last syllable came from the enclitic and
    # leaves with the elided vowel, so this is no oxytone to retract.
    lemma_forms = {"χείρ": {"χεῖρά", "χεῖρ᾽", "χείρ᾽"},
                   "λέγω": {"εἶπέ", "εἶπ᾽", "εἴπ᾽"},
                   "ἄλλος": {"ἄλλὰ", "ἄλλ᾽", "ἀλλ᾽"}}
    pairs = _derive_elision_pairs(lemma_forms, {})
    assert pairs["χεῖρά"] == "χεῖρ᾽"
    assert pairs["εἶπέ"] == "εἶπ᾽"
    assert pairs["ἄλλὰ"] == "ἄλλ᾽"


def test_a_grave_before_the_last_syllable_is_not_an_accent():
    # μὲτὰ is a malformed μετά, not a proparoxytone carrying an enclitic's
    # accent, so it stays in the class that elides bare.
    lemma_forms = {"μετά": {"μὲτὰ", "μετ᾽", "μέτ᾽"}}
    assert _derive_elision_pairs(lemma_forms, {})["μὲτὰ"] == "μετ᾽"


ARTIFACT = ROOT / "build" / "hunspell" / "grc_morph.json"


@pytest.mark.skipif(not ARTIFACT.exists(), reason="grc_morph.json not built")
def test_the_built_elision_table_holds_its_invariants():
    """The keyboard rewrites the user's text with this table.

    Tonos's dictionary gate never reads this file, so three separate
    defects reached it before anything here checked them.
    """
    import export_morphology as em
    from export_hunspell import (exact_form_key, grc_orthography_reason,
                                 load_form_profile_freq)

    table = json.loads(ARTIFACT.read_text(encoding="utf-8"))["el"]
    assert table

    non_elidable = [k for k in table if not em._can_elide(k)]
    assert not non_elidable, f"keys elision cannot touch: {non_elidable[:10]}"

    invalid = {k: v for k, v in table.items()
               if grc_orthography_reason(v) is not None}
    assert not invalid, f"values the orthography rules reject: {list(invalid)[:10]}"

    # The keyboard writes the value into the user's text as it stands, so
    # a capital here puts one mid-sentence (κατὰ -> Κατ᾽ did, 226,495
    # occurrences).
    recased = {k: v for k, v in table.items()
               if k[:1].isupper() != v[:1].isupper()}
    assert not recased, f"values that change the key's case: {list(recased.items())[:10]}"

    # Oxytones are pinned by corpus weight rather than by count, because
    # weight is what separates a rule that stopped firing from the tail of
    # the known-bare class. That tail is crasis forms (κἀπί), dialect
    # particles (ποκά, πεδά, προτί) and OCR fragments, and stands at 56
    # entries carrying 11,679 occurrences. The regression Tonos reported
    # carried 119,486.
    freq = load_form_profile_freq().exact
    if not freq:
        pytest.skip("form_profile.db not downloaded")
    bare = [k for k, v in table.items()
            if grc_orthography_reason(k) is None
            and em._is_oxytone(k)
            and em._strip_lower(k) not in em.ELISION_KEEPS_NO_ACCENT
            and not em._has_accent(v)]
    weight = sum(freq.get(exact_form_key(k), 0) for k in bare)
    assert weight < 25_000, (
        f"elided oxytones left bare carry {weight:,} occurrences: "
        f"{sorted(bare, key=lambda k: -freq.get(exact_form_key(k), 0))[:10]}"
    )
