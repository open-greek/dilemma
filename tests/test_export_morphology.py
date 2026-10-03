import json
import sys
import unicodedata

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dilemma.form_sanitize import has_editorial_sigla
from export_morphology import (
    _derive_dative_keys,
    _derive_elision_pairs,
    _is_oxytone,
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
    pairs = _derive_elision_pairs(lemma_forms)
    assert pairs["Εἶπε"] == "Εἶπ᾽"
    assert pairs["Εἰπέ"] == "Εἴπ᾽"


def test_the_rules_not_the_corpus_pick_the_spelling():
    # Εἴτε keeps its smooth breathing whatever the counts say: εἵτ᾽ is
    # another spelling, not a variant of this one.
    lemma_forms = {"εἴτε": {"Εἴτε", "εἴτ᾽", "εἵτ᾽"}}
    assert _derive_elision_pairs(lemma_forms)["Εἴτε"] == "Εἴτ᾽"


def test_a_candidate_the_orthography_rules_reject_loses():
    # τὸτ᾽ carries a grave on an elided word.
    lemma_forms = {"τότε": {"Τότε", "τότ᾽", "τὸτ᾽"}}
    assert _derive_elision_pairs(lemma_forms)["Τότε"] == "Τότ᾽"


def test_the_table_does_not_depend_on_set_iteration_order():
    # The candidates arrive from a set, whose iteration order varies with
    # the interpreter's hash seed, so ties have to break on the spelling.
    lemma_forms = {"ζζύω": {"Ζζύε", "ζζύ᾽", "ζζὺ᾽"}}
    first = _derive_elision_pairs(lemma_forms)
    again = _derive_elision_pairs({"ζζύω": set(reversed(sorted(
        lemma_forms["ζζύω"])))})
    assert first == again


def test_an_elided_oxytone_throws_its_accent_back():
    # Smyth 174: ἀνδρί loses its final vowel and the accent goes back onto
    # the penult as an acute. The full form's own stem carries no mark, so
    # the stem-marks test alone would prefer the bare spelling.
    lemma_forms = {"ἀνήρ": {"ἀνδρί", "ἄνδρ᾽", "ἀνδρ᾽"}}
    assert _derive_elision_pairs(lemma_forms)["ἀνδρί"] == "ἄνδρ᾽"


def test_prepositions_and_conjunctions_lose_the_accent_instead():
    # The other half of Smyth 174. οὐδέ is not one of the pinned ten, so
    # nothing else would keep it bare.
    lemma_forms = {"οὐδέ": {"οὐδέ", "οὐδ᾽", "οὔδ᾽"}}
    assert _derive_elision_pairs(lemma_forms)["οὐδέ"] == "οὐδ᾽"


def test_a_form_that_is_not_oxytone_keeps_its_marks_where_they_were():
    # The retraction rule has to say nothing here, or it would override the
    # stem-marks test for every paroxytone and properispomenon.
    lemma_forms = {"ἄλλος": {"ἄλλε", "ἄλλ᾽", "ἆλλ᾽"}}
    assert _derive_elision_pairs(lemma_forms)["ἄλλε"] == "ἄλλ᾽"


def test_a_grave_oxytone_retracts_like_an_acute_one():
    # The grave is only the contextual spelling of an oxytone, and it is
    # the spelling a word carries mid-sentence, which is where elision
    # happens. Reported by Tonos: 59 entries were left bare.
    lemma_forms = {"αὐτός": {"αὐτὸ", "αὔτ᾽", "αὐτ᾽"}}
    assert _derive_elision_pairs(lemma_forms)["αὐτὸ"] == "αὔτ᾽"


def test_a_word_elision_cannot_touch_gets_no_entry():
    # Elision removes a short final vowel. η and ω are always long, a
    # circumflex or iota subscript marks length, and the second element of
    # a diphthong goes with the first.
    for full, elided in [("αὐτῷ", "αὐτ᾽"), ("αὐτῇ", "αὐτ᾽"),
                         ("δεῖ", "δέ᾽"), ("ἤδη", "ἠδ᾽"),
                         ("λόγοι", "λόγ᾽"), ("λόγου", "λόγ᾽")]:
        pairs = _derive_elision_pairs({"x": {full, elided}})
        assert full not in pairs, f"{full} cannot elide"


def test_words_attic_never_elides_get_no_entry():
    # ὅτ᾽ reads as ὅτε, δι᾽ as διά, and οὐχί ends in a long ί.
    lemma_forms = {"ὅτι": {"ὅτι", "ὅτ᾽"}, "περί": {"περί", "περὶ", "περ᾽"},
                   "διό": {"διό", "δι᾽"}, "οὐχί": {"οὐχί", "οὔχ᾽"},
                   "ὅτε": {"ὅτε", "ὅτ᾽"}}
    pairs = _derive_elision_pairs(lemma_forms)
    for word in ("ὅτι", "περί", "περὶ", "διό", "οὐχί"):
        assert word not in pairs, word
    assert pairs["ὅτε"] == "ὅτ᾽"


def test_a_hiatus_or_a_written_out_subscript_is_read_as_such():
    from export_morphology import _can_elide
    # An accented ε before a final ι is a hiatus: βασιλέι is βασιλέϊ, whose
    # short ι could elide were it not a dative.
    assert _can_elide("βασιλέι")
    # An ι after η or ω is the iota subscript written out, and a macron
    # marks a long vowel.
    assert not _can_elide("λόγωι")
    assert not _can_elide("τῆι")
    assert not _can_elide("χώρᾱ")
    # With a diaeresis the ι is a vowel of its own, not a subscript.
    assert _can_elide("ἥρωϊ")


def test_the_dative_ending_does_not_elide(tmp_path):
    # Smyth 72: the dative -ι and -σι elide only in epic, and the elided
    # spelling the corpora attest is another case's (ἄνδρ᾽ is ἄνδρα). The
    # first tagged file that has a form decides, so GLAUx's verb λέγουσι
    # outvotes Diorisis's dative participle, and Diorisis's stray vocative
    # of ἀνδρίς does not save ἀνδρί. τῷδε is a dative too, but it loses the
    # ε of δε.
    glaux = tmp_path / "glaux.json"
    glaux.write_text(json.dumps([
        {"form": "ἀνδρί", "lemma": "ἀνήρ", "tags": ["singular", "dative"]},
        {"form": "λέγουσι", "lemma": "λέγω",
         "tags": ["third-person", "plural", "indicative"]},
        {"form": "τῷδε", "lemma": "ὅδε", "tags": ["singular", "dative"]},
        {"form": "Χερσὶ", "lemma": "Χείρ", "tags": ["plural", "feminine"]},
        {"form": "χερσὶ", "lemma": "χείρ", "tags": ["plural", "dative"]},
    ], ensure_ascii=False), encoding="utf-8")
    diorisis = tmp_path / "diorisis.json"
    diorisis.write_text(json.dumps([
        {"form": "λέγουσι", "lemma": "λέγω",
         "tags": ["plural", "dative", "participle"]},
        {"form": "πᾶσι", "lemma": "πᾶς", "tags": ["plural", "dative"]},
        {"form": "ἀνδρί", "lemma": "ἀνδρίς",
         "tags": ["feminine", "singular", "vocative"]},
    ], ensure_ascii=False), encoding="utf-8")
    datives = _derive_dative_keys([glaux, diorisis])
    assert {key for key, dative in datives.items() if dative} == {
        "ἀνδρί", "τῷδε", "πᾶσι", "χερσί"}

    lemma_forms = {"χείρ": {"χερσὶ", "Χερσὶ", "χέρσ᾽"},
                   "ἀνήρ": {"ἀνδρί", "ἀνδρὶ", "ἄνδρ᾽"},
                   "λέγω": {"λέγουσι", "λέγουσ᾽"},
                   "ὅδε": {"τῷδε", "τῷδ᾽"},
                   "πᾶς": {"πᾶσι", "πᾶσ᾽"}}
    pairs = _derive_elision_pairs(lemma_forms, datives)
    # The capitalized Χερσὶ, analyzed without a case, is read through its
    # lowercase spelling, and does not keep the dative eliding.
    for word in ("ἀνδρί", "ἀνδρὶ", "πᾶσι", "χερσὶ", "Χερσὶ"):
        assert word not in pairs, word
    assert pairs["λέγουσι"] == "λέγουσ᾽"
    assert pairs["τῷδε"] == "τῷδ᾽"


def test_a_dative_no_treebank_analyzes_is_read_from_its_lemmas_forms(tmp_path):
    # The lookup table's paradigm expansions carry datives no tagged corpus
    # analyzes. A dative singular sits beside its genitive (δμητῆρι,
    # δμητῆρος); an aorist participle's dative plural is no verb form
    # (λυθεῖσι, and the second aorist λαβοῦσι, accented on the ending); a
    # present participle's is also the third person plural (κεύθουσι,
    # τιμῶσι, ἱστᾶσι, διδοῦσι), and an optative is a verb (κάμψαιμι).
    # Diorisis files the uncontracted κήδεϊ as a nominative dual, which the
    # diaeresis rules out.
    diorisis = _write(tmp_path / "diorisis.json", [
        {"form": "κήδεϊ", "lemma": "κῆδος", "pos": "noun",
         "tags": ["neuter", "dual", "nominative"]},
    ])
    datives = _derive_dative_keys([diorisis])
    assert "κήδεϊ" not in datives

    lemma_forms = {
        "δμητήρ": {"δμητῆρι", "δμητῆρος", "δμητῆρα", "δμητῆρ᾽"},
        "λύω": {"λυθεῖσι", "λυθέντος", "λυθέντα", "λυθεῖσ᾽"},
        "λαμβάνω": {"λαβοῦσι", "λαβόντος", "λαβόντα", "λαβοῦσ᾽"},
        "κεύθω": {"κεύθουσι", "κεύθοντος", "κεύθοντα", "κεύθουσ᾽"},
        "τιμάω": {"τιμῶσι", "τιμῶντος", "τιμῶντα", "τιμῶσ᾽"},
        "ἵστημι": {"ἱστᾶσι", "ἱστάντος", "ἱστάντα", "ἱστᾶσ᾽"},
        "δίδωμι": {"διδοῦσι", "διδόντος", "διδόντα", "διδοῦσ᾽"},
        "κάμπτω": {"κάμψαιμι", "κάμψαιμ᾽"},
        "κῆδος": {"κήδεϊ", "κήδε᾽"},
    }
    pairs = _derive_elision_pairs(lemma_forms, datives)
    assert not {"δμητῆρι", "λυθεῖσι", "λαβοῦσι", "κήδεϊ"} & set(pairs)
    for verb in ("κεύθουσι", "τιμῶσι", "ἱστᾶσι", "διδοῦσι"):
        assert pairs[verb] == verb[:-1] + "᾽", verb
    assert pairs["κάμψαιμι"] == "κάμψαιμ᾽"


IMPERATIVE = ["second-person", "singular", "present", "imperative", "active"]
IMPERFECT = ["third-person", "singular", "imperfect", "indicative", "active"]


def _write(path, entries):
    path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    return path


def test_the_reading_the_texts_mostly_mean_decides_movable_nu(tmp_path):
    # GLAUx counts φέρε as an imperative 956 times and an unaugmented
    # imperfect 45; the pairs file keeps the imperfect because it came
    # first, and presence alone put φέρεν εἰπέ on the keyboard.
    glaux = _write(tmp_path / "glaux.json", [
        {"form": "φέρε", "lemma": "φέρω", "pos": "verb", "tags": IMPERFECT,
         "count": 1001,
         "analyses": [["verb", IMPERATIVE, 956], ["verb", IMPERFECT, 45]]},
        # μέλλε is an imperative 14 times and an imperfect 7, but before a
        # vowel the imperfect is spelled μέλλεν, 17 more times, and with
        # those it is the commoner reading.
        {"form": "μέλλε", "lemma": "μέλλω", "pos": "verb", "tags": IMPERFECT,
         "count": 21,
         "analyses": [["verb", IMPERATIVE, 14], ["verb", IMPERFECT, 7]]},
        {"form": "μέλλεν", "lemma": "μέλλω", "pos": "verb", "tags": IMPERFECT,
         "count": 17},
        {"form": "ἔλυε", "lemma": "λύω", "pos": "verb", "tags": IMPERFECT,
         "count": 12},
    ])
    forms = _derive_nu_forms([glaux])
    assert "φέρε" not in forms
    assert "μέλλε" in forms
    assert "ἔλυε" in forms


def test_a_file_without_token_counts_does_not_vote(tmp_path):
    # Diorisis lists every candidate analysis on each token, so its
    # analyses show a spelling can take nu but cannot outvote GLAUx's.
    glaux = _write(tmp_path / "glaux.json", [
        # A tie keeps the nu; one more imperative from Diorisis would drop it.
        {"form": "ἄγε", "lemma": "ἄγω", "pos": "verb", "tags": IMPERFECT,
         "count": 20, "analyses": [["verb", IMPERFECT, 10], ["verb", IMPERATIVE, 10]]},
    ])
    diorisis = _write(tmp_path / "diorisis.json", [
        {"form": "ἄγε", "lemma": "ἄγω", "pos": "verb", "tags": IMPERATIVE},
        {"form": "κέλευε", "lemma": "κελεύω", "pos": "verb", "tags": IMPERFECT},
        # Both are unaugmented, so they are kept only because the texts
        # write the past with its ν (see the imperative test below).
        {"form": "ἄγεν", "lemma": "ἄγω", "pos": "verb", "tags": IMPERFECT},
        {"form": "κέλευεν", "lemma": "κελεύω", "pos": "verb", "tags": IMPERFECT},
    ])
    forms = _derive_nu_forms([glaux, diorisis])
    assert "ἄγε" in forms
    assert "κέλευε" in forms


def test_an_analysis_without_case_or_mood_abstains(tmp_path):
    # GLAUx tags the locative Ἀθήνησι as a bare adverb; that says nothing
    # against the dative reading that takes nu (Ἀθήνησιν).
    glaux = _write(tmp_path / "glaux.json", [
        {"form": "Ἀθήνησι", "lemma": "Ἀθήνησι", "pos": "adv", "count": 403},
    ])
    diorisis = _write(tmp_path / "diorisis.json", [
        {"form": "Ἀθήνησι", "lemma": "Ἀθῆναι", "pos": "noun",
         "tags": ["feminine", "plural", "dative"]},
    ])
    assert "Ἀθήνησι" in _derive_nu_forms([glaux, diorisis])


def test_the_article_numerals_and_epic_subjunctive_take_movable_nu(tmp_path):
    # Smyth 134: every dative plural in -σι, the article's and the
    # numerals' included (τῇσιν, τρισίν), and every third person in -σι,
    # the epic subjunctive singular included (ἐθέλῃσιν).
    glaux = _write(tmp_path / "glaux.json", [
        {"form": "τῇσι", "lemma": "ὁ", "pos": "article",
         "tags": ["plural", "feminine", "dative"], "count": 784},
        {"form": "τρισί", "lemma": "τρεῖς", "pos": "num",
         "tags": ["plural", "dative"], "count": 28},
        {"form": "ἐθέλῃσι", "lemma": "ἐθέλω", "pos": "verb",
         "tags": ["third-person", "singular", "present", "subjunctive", "active"],
         "count": 18},
        {"form": "πέρατι", "lemma": "πέρας", "pos": "noun",
         "tags": ["singular", "neuter", "dative"], "count": 135},
    ])
    diorisis = _write(tmp_path / "diorisis.json", [
        # A candidate analysis Diorisis lists for every token of πέρατι.
        {"form": "πέρατι", "lemma": "περάω", "pos": "verb",
         "tags": ["singular", "present", "indicative", "active", "third-person"]},
    ])
    forms = _derive_nu_forms([glaux, diorisis])
    assert {"τῇσι", "τρισί", "ἐθέλῃσι"} <= forms
    assert "πέρατι" not in forms


def test_any_counted_analysis_can_make_a_spelling_eligible(tmp_path):
    # πέλε's first token was tagged an imperative, but 78 of its 79 are the
    # epic imperfect.
    glaux = _write(tmp_path / "glaux.json", [
        {"form": "πέλε", "lemma": "πέλω", "pos": "verb", "tags": IMPERATIVE,
         "count": 79,
         "analyses": [["verb", IMPERFECT, 78], ["verb", IMPERATIVE, 1]]},
        # Unaugmented, so it also needs its ν spelling in the texts.
        {"form": "πέλεν", "lemma": "πέλω", "pos": "verb", "tags": IMPERFECT,
         "count": 55},
    ])
    assert "πέλε" in _derive_nu_forms([glaux])


def test_a_capital_pools_with_the_lowercase_word_but_not_the_reverse(tmp_path):
    # Εἰσένεγκε at the head of a sentence is the imperative the lowercase
    # word mostly is; the vocative of the name Κέλσος must not veto the
    # verb κέλσε.
    glaux = _write(tmp_path / "glaux.json", [
        {"form": "εἰσένεγκε", "lemma": "εἰσφέρω", "pos": "verb", "tags": IMPERATIVE,
         "count": 5},
        {"form": "Εἰσένεγκε", "lemma": "εἰσφέρω", "pos": "verb", "tags": IMPERFECT,
         "count": 1},
        {"form": "κέλσε", "lemma": "κέλλω", "pos": "verb", "tags": IMPERFECT,
         "count": 1},
        {"form": "Κέλσε", "lemma": "Κέλσος", "pos": "noun",
         "tags": ["singular", "masculine", "vocative"], "count": 8},
    ])
    # Diorisis offers the verb as a candidate for the capitalized spelling
    # too; its own eight vocatives outvote the one verb.
    diorisis = _write(tmp_path / "diorisis.json", [
        {"form": "Κέλσε", "lemma": "κέλλω", "pos": "verb", "tags": IMPERFECT},
    ])
    forms = _derive_nu_forms([glaux, diorisis])
    assert "Εἰσένεγκε" not in forms
    assert "κέλσε" in forms
    assert "Κέλσε" not in forms


def test_a_slip_the_ending_rules_out_does_not_vote(tmp_path):
    # An ending in -σι is a dative plural or a third person; GLAUx files
    # one ζηλῶσί as a first person, which cannot be.
    glaux = _write(tmp_path / "glaux.json", [
        {"form": "ζηλῶσί", "lemma": "ζηλόω", "pos": "verb",
         "tags": ["first-person", "singular", "present", "indicative", "active"],
         "count": 1},
    ])
    diorisis = _write(tmp_path / "diorisis.json", [
        {"form": "ζηλῶσί", "lemma": "ζηλόω", "pos": "verb",
         "tags": ["third-person", "plural", "present", "subjunctive", "active"]},
    ])
    assert "ζηλῶσί" in _derive_nu_forms([glaux, diorisis])


def test_an_adverb_in_si_takes_nu_when_the_texts_write_it(tmp_path):
    # παντάπασιν is written with the nu in 70% of its tokens; οὑτωσίν is a
    # slip, twice in 451, since the deictic -ί never takes it.
    glaux = _write(tmp_path / "glaux.json", [
        {"form": "παντάπασι", "lemma": "παντάπασι", "pos": "adv", "count": 563},
        {"form": "παντάπασιν", "lemma": "παντάπασι", "pos": "adv", "count": 1492},
        {"form": "οὑτωσί", "lemma": "οὕτως", "pos": "adv", "count": 449},
        {"form": "οὑτωσίν", "lemma": "οὕτως", "pos": "adv", "count": 2},
    ])
    forms = _derive_nu_forms([glaux])
    assert "παντάπασι" in forms
    assert "οὑτωσί" not in forms


def test_an_adverb_in_the_takes_nu_when_the_texts_write_it(tmp_path):
    # πρόσθε(ν) and ὄπισθε(ν) alternate (Smyth 134 D), GLAUx filing the word
    # as a preposition when it governs a genitive; εἴθε never has a ν, and
    # the 2pl βούλεσθε and the imperative ἐλθέ are verbs.
    glaux = _write(tmp_path / "glaux.json", [
        {"form": "πρόσθε", "lemma": "πρόσθεν", "pos": "adv", "count": 147,
         "analyses": [["adv", [], 134], ["prep", [], 13]]},
        {"form": "πρόσθεν", "lemma": "πρόσθεν", "pos": "adv", "count": 2855},
        {"form": "ὄπισθε", "lemma": "ὄπισθεν", "pos": "prep", "count": 3},
        {"form": "ὄπισθεν", "lemma": "ὄπισθεν", "pos": "prep", "count": 10},
        {"form": "εἴθε", "lemma": "εἴθε", "pos": "adv", "count": 4},
        {"form": "βούλεσθε", "lemma": "βούλομαι", "pos": "verb",
         "tags": ["second-person", "plural", "present", "indicative", "middle"],
         "count": 200},
        {"form": "ἐλθέ", "lemma": "ἔρχομαι", "pos": "verb",
         "tags": ["second-person", "singular", "aorist", "imperative", "active"],
         "count": 50},
    ])
    forms = _derive_nu_forms([glaux])
    assert {"πρόσθε", "ὄπισθε"} <= forms
    assert not {"εἴθε", "βούλεσθε", "ἐλθέ"} & forms


AORIST = ["third-person", "singular", "aorist", "indicative", "active"]


def test_an_unaugmented_past_the_texts_never_write_with_nu_is_the_imperative(
        tmp_path):
    # An unaugmented imperfect or thematic aorist is spelled like the 2sg
    # imperative, which takes no nu (Smyth 134, 438). GLAUx's tagger files
    # Lucian's κατένεγκε and Galen's προσέμβαλλε as indicatives, and
    # Diorisis lists no imperative for the ἐνεγκ- compounds; since the
    # texts never write them with ν, they are dropped. The augment shows
    # after the preverbs (κατήνεγκε, ἐξήνεγκε), or under the simple verb
    # (ἤνεγκε under φέρω), or as a missing ε- (ἐντύγχανε, δίδασκε).
    glaux = _write(tmp_path / "glaux.json", [
        {"form": "κατένεγκε", "lemma": "καταφέρω", "pos": "verb",
         "tags": AORIST, "count": 3},
        {"form": "Κατένεγκε", "lemma": "καταφέρω", "pos": "verb",
         "tags": AORIST, "count": 1},
        {"form": "κατήνεγκε", "lemma": "καταφέρω", "pos": "verb",
         "tags": AORIST, "count": 10},
        {"form": "ἤνεγκε", "lemma": "φέρω", "pos": "verb",
         "tags": AORIST, "count": 40},
        {"form": "ἐξήνεγκε", "lemma": "ἐκφέρω", "pos": "verb",
         "tags": AORIST, "count": 8},
        {"form": "προσέμβαλλε", "lemma": "προσεμβάλλω", "pos": "verb",
         "tags": IMPERFECT, "count": 2},
        {"form": "δίδασκε", "lemma": "διδάσκω", "pos": "verb",
         "tags": IMPERFECT, "count": 2},
        # Kept: a σ-aorist, an iterative, a pluperfect, an augmented
        # imperfect, and an unaugmented aorist the texts write with ν.
        {"form": "βλάστησε", "lemma": "βλαστάνω", "pos": "verb",
         "tags": AORIST, "count": 2},
        {"form": "ποιέεσκε", "lemma": "ποιέω", "pos": "verb",
         "tags": IMPERFECT, "count": 2},
        {"form": "κατεστήκεε", "lemma": "καθίστημι", "pos": "verb",
         "tags": ["third-person", "singular", "pluperfect", "indicative",
                  "active"], "count": 1},
        {"form": "ἔλεγε", "lemma": "λέγω", "pos": "verb",
         "tags": IMPERFECT, "count": 30},
        {"form": "βάλε", "lemma": "βάλλω", "pos": "verb",
         "tags": AORIST, "count": 20},
        {"form": "βάλεν", "lemma": "βάλλω", "pos": "verb",
         "tags": AORIST, "count": 74},
        # ἄντε is Plato's ἄν τε, filed as an imperfect of ἄντομαι, which is
        # not ἀντ- plus -ομαι; it is never written with ν.
        {"form": "ἄντε", "lemma": "ἄντομαι", "pos": "verb",
         "tags": IMPERFECT, "count": 30},
        # Kept: a liquid first aorist, whose imperative is σήμηνον; an
        # accent the augment holds on the penult (Smyth 426), beside the
        # imperative ἔξευρε; an augmented διανέπαυε a greedy split would
        # read as δια- plus ναπαύω; a prodelided augment; and an enclitic's
        # accent on a spelling whose plain twin is written with ν.
        {"form": "σήμηνε", "lemma": "σημαίνω", "pos": "verb",
         "tags": AORIST, "count": 3},
        {"form": "ἐσήμηνε", "lemma": "σημαίνω", "pos": "verb",
         "tags": AORIST, "count": 9},
        {"form": "συνεξεῦρε", "lemma": "συνεξευρίσκω", "pos": "verb",
         "tags": AORIST, "count": 1},
        {"form": "ηὗρε", "lemma": "εὑρίσκω", "pos": "verb",
         "tags": AORIST, "count": 5},
        {"form": "διανέπαυε", "lemma": "διαναπαύω", "pos": "verb",
         "tags": IMPERFECT, "count": 2},
        {"form": "’βάδιζε", "lemma": "βαδίζω", "pos": "verb",
         "tags": IMPERFECT, "count": 1},
        {"form": "δαῖέ", "lemma": "δαίω", "pos": "verb",
         "tags": IMPERFECT, "count": 1},
        {"form": "δαῖεν", "lemma": "δαίω", "pos": "verb",
         "tags": IMPERFECT, "count": 2},
    ])
    diorisis = _write(tmp_path / "diorisis.json", [
        {"form": "προσανένεγκε", "lemma": "προσαναφέρω", "pos": "verb",
         "tags": AORIST},
        {"form": "ξένεγκε", "lemma": "ἐκφέρω", "pos": "verb", "tags": AORIST},
        {"form": "ἐντύγχανε", "lemma": "ἐντυγχάνω", "pos": "verb",
         "tags": IMPERFECT},
    ])
    forms = _derive_nu_forms([glaux, diorisis])
    assert not {"κατένεγκε", "Κατένεγκε", "προσανένεγκε", "ξένεγκε",
                "προσέμβαλλε", "δίδασκε", "ἐντύγχανε", "ἄντε"} & forms
    assert {"κατήνεγκε", "ἤνεγκε", "βλάστησε", "ποιέεσκε", "κατεστήκεε",
            "ἔλεγε", "βάλε", "σήμηνε", "συνεξεῦρε", "διανέπαυε", "’βάδιζε",
            "δαῖέ"} <= forms


def test_the_aorist_optative_in_eie_takes_movable_nu(tmp_path):
    # A third person singular in -ε like the past's (Smyth 134): GLAUx
    # writes δόξειεν before a vowel on 99.6% of such tokens. The other
    # optatives take none.
    optative = ["third-person", "singular", "aorist", "optative", "active"]
    glaux = _write(tmp_path / "glaux.json", [
        {"form": "ποιήσειε", "lemma": "ποιέω", "pos": "verb",
         "tags": optative, "count": 40},
        {"form": "λύοι", "lemma": "λύω", "pos": "verb",
         "tags": ["third-person", "singular", "present", "optative",
                  "active"], "count": 10},
        {"form": "λύσαι", "lemma": "λύω", "pos": "verb",
         "tags": optative, "count": 5},
    ])
    forms = _derive_nu_forms([glaux])
    assert "ποιήσειε" in forms
    assert not {"λύοι", "λύσαι"} & forms


def test_the_third_singular_in_ti_is_esti_and_its_compounds(tmp_path):
    present = ["third-person", "singular", "present", "indicative", "active"]
    glaux = _write(tmp_path / "glaux.json", [
        {"form": form, "lemma": lemma, "pos": "verb", "tags": present, "count": 3}
        for form, lemma in [("πάρεστι", "πάρειμι"), ("κἀστί", "εἰμί"),
                            ("ἐντί", "εἰμί"), ("δίδωτι", "δίδωμι")]
    ])
    forms = _derive_nu_forms([glaux])
    assert {"πάρεστι", "κἀστί"} <= forms
    # The Doric -τι takes no movable nu.
    assert not {"ἐντί", "δίδωτι"} & forms


def test_an_indeclinable_in_e_votes_against_nu(tmp_path):
    # GLAUx tags one ηὖτε as an imperfect; five times it is the adverb.
    glaux = _write(tmp_path / "glaux.json", [
        {"form": "ηὖτε", "lemma": "αὔω", "pos": "verb", "tags": IMPERFECT, "count": 1},
        {"form": "ηὖτε", "lemma": "ηὖτε", "pos": "adv", "count": 5},
    ])
    assert "ηὖτε" not in _derive_nu_forms([glaux])


def test_a_pairs_file_without_counts_is_refused(tmp_path):
    from export_morphology import _require_token_counts
    stale = _write(tmp_path / "glaux.json", [
        {"form": "φέρε", "lemma": "φέρω", "pos": "verb", "tags": IMPERFECT},
    ])
    with pytest.raises(SystemExit):
        _require_token_counts(stale)
    _require_token_counts(_write(tmp_path / "current.json", [
        {"form": "φέρε", "lemma": "φέρω", "pos": "verb", "tags": IMPERFECT,
         "count": 1},
    ]))


def test_a_participles_dative_plural_takes_movable_nu(tmp_path):
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(json.dumps([
        {"form": "οὖσι", "lemma": "εἰμί", "pos": "verb",
         "tags": ["plural", "present", "participle", "dative"]},
        {"form": "οὖσα", "lemma": "εἰμί", "pos": "verb",
         "tags": ["singular", "present", "participle", "nominative"]},
        {"form": "ὦσι", "lemma": "εἰμί", "pos": "verb",
         "tags": ["third-person", "plural", "present", "subjunctive"]},
        {"form": "ἔχουσἰ", "lemma": "ἔχω", "pos": "verb",
         "tags": ["plural", "present", "participle", "dative"]},
        {"form": "ἒστι", "lemma": "εἰμί", "pos": "verb",
         "tags": ["third-person", "singular", "present", "indicative",
                  "active"]},
    ], ensure_ascii=False), encoding="utf-8")
    forms = _derive_nu_forms([pairs_path])
    assert "οὖσι" in forms
    assert "ὦσι" in forms       # the subjunctive's third plural (Smyth 134)
    assert "οὖσα" not in forms
    # Malformed spellings get no ν appended; a prodelided one is not
    # malformed.
    assert "ἔχουσἰ" not in forms
    assert "ἒστι" not in forms
    assert "στι" in forms


def test_a_pair_is_dropped_when_every_candidate_is_junk():
    # Ranking only picks the least bad one, and writing that into
    # someone's text is worse than offering nothing.
    from export_hunspell import grc_orthography_reason
    assert grc_orthography_reason("ὃσδ᾽") is not None
    assert "ὅσδε" not in _derive_elision_pairs({"ὅσδε": {"ὅσδε", "ὃσδ᾽"}})


def test_a_form_two_lemmas_claim_is_ranked_once():
    # The corpora carry a capitalized lemma Κατά whose one elided token
    # opened a sentence. Assigned lemma by lemma, whichever came last won,
    # and κατὰ took Κατ᾽ from it.
    lemma_forms = {"κατά": {"κατὰ", "κατ᾽", "κάτ᾽"},
                   "Κατά": {"κατὰ", "Κατ᾽"},
                   "Ἑλλάς": {"ἑλλάδα", "Ἑλλάδ᾽"}}
    pairs = _derive_elision_pairs(lemma_forms)
    assert pairs["κατὰ"] == "κατ᾽"
    assert pairs["ἑλλάδα"] == "ἑλλάδ᾽"
    reordered = dict(reversed(list(lemma_forms.items())))
    assert _derive_elision_pairs(reordered) == pairs


def test_the_elided_form_takes_the_full_forms_case():
    lemma_forms = {"αὐτός": {"Αὐτὸ", "αὔτ᾽"}}
    assert _derive_elision_pairs(lemma_forms)["Αὐτὸ"] == "Αὔτ᾽"


def test_an_enclitics_accent_is_not_the_words_own():
    # χεῖρά τε: the acute on the last syllable came from the enclitic and
    # leaves with the elided vowel, so this is no oxytone to retract.
    lemma_forms = {"χείρ": {"χεῖρά", "χεῖρ᾽", "χείρ᾽"},
                   "λέγω": {"εἶπέ", "εἶπ᾽", "εἴπ᾽"},
                   "ἄλλος": {"ἄλλὰ", "ἄλλ᾽", "ἀλλ᾽"}}
    pairs = _derive_elision_pairs(lemma_forms)
    assert pairs["χεῖρά"] == "χεῖρ᾽"
    assert pairs["εἶπέ"] == "εἶπ᾽"
    assert pairs["ἄλλὰ"] == "ἄλλ᾽"


def test_no_entry_when_the_rules_spelling_is_unattested():
    # αἰτία ends in a long α, which cannot elide, and its only candidate is
    # the elided neuter plural αἴτια's. γυναικί would retract to γυναίκ᾽,
    # which no text writes; γυναῖκ᾽ is the accusative's. Either value would
    # replace the user's word with another one.
    lemma_forms = {"αἰτία": {"αἰτία", "αἴτι᾽"},
                   "γυνή": {"γυναικί", "γυναῖκ᾽"}}
    pairs = _derive_elision_pairs(lemma_forms)
    assert "αἰτία" not in pairs
    assert "γυναικί" not in pairs


def test_the_epic_preposition_eni_elides_bare_and_the_numeral_retracts():
    # The corpora file ἐνί under εἷς as well as under ἐν, which is how it
    # took the numeral's ἕν᾽; the answer must not depend on which comes last.
    lemma_forms = {"ἐν": {"ἐνί", "ἐν᾽"},
                   "εἷς": {"ἐνί", "ἑνί", "ἕν᾽"}}
    for order in (lemma_forms, dict(reversed(list(lemma_forms.items())))):
        pairs = _derive_elision_pairs(order)
        assert pairs["ἐνί"] == "ἐν᾽"
        assert pairs["ἑνί"] == "ἕν᾽"


def test_a_rule_spelling_the_orthography_rules_reject_is_not_emitted():
    # ἂρα carries a grave before its last syllable, and the rule keeps it.
    assert "ἂρα" not in _derive_elision_pairs({"ἄρα": {"ἂρα", "ἂρ᾽"}})


def test_a_key_that_opens_with_a_mark_is_skipped():
    # A prodelided ᾽κεῖνα is not a word a keyboard reads.
    lemma_forms = {"ἐκεῖνος": {"᾽κεῖνα", "᾽κεῖν᾽"}}
    assert "᾽κεῖνα" not in _derive_elision_pairs(lemma_forms)


def test_tote_is_an_adverb_that_retracts():
    # τοτέ, "at times", is accented and no enclitic; the lemma τότε lists it.
    lemma_forms = {"τοτέ": {"τοτέ", "τοτ᾽"}, "τότε": {"τότε", "τοτέ", "τότ᾽"}}
    assert _derive_elision_pairs(lemma_forms)["τοτέ"] == "τότ᾽"


def test_dialect_prepositions_and_crasis_elide_bare():
    lemma_forms = {"προτί": {"προτί", "προτ᾽", "πρότ᾽"},
                   "ἐπί": {"τἀπί", "τἀπ᾽", "τἄπ᾽"},
                   "ὑπό": {"ὑπά", "ὑπ᾽", "ὕπ᾽"}}
    pairs = _derive_elision_pairs(lemma_forms)
    assert pairs["προτί"] == "προτ᾽"
    assert pairs["τἀπί"] == "τἀπ᾽"
    assert pairs["ὑπά"] == "ὑπ᾽"


def test_a_circumflex_before_the_penult_hosts_no_enclitic_accent():
    # ὦγαθέ is the crasis ὦ ἀγαθέ, an oxytone, and prints as ὦγάθ᾽.
    lemma_forms = {"ἀγαθός": {"ὦγαθέ", "ὦγάθ᾽"}}
    assert _derive_elision_pairs(lemma_forms)["ὦγαθέ"] == "ὦγάθ᾽"


def test_long_final_vowels_the_spelling_hides_do_not_elide():
    # The deictic -ί (Smyth 333g), the Attic accusative -έᾱ of a noun in
    # -εύς (Smyth 276), a contracted neuter plural in -ᾱ, and a monosyllable
    # not in ε (Smyth 72).
    lemma_forms = {"ὅδε": {"τόνδε", "τονδί", "τόνδ᾽"},
                   "βασιλεύς": {"βασιλέα", "βασιλέ᾽"},
                   "κρέας": {"κρέα", "κρέ᾽"},
                   "σός": {"σά", "σ᾽"},
                   "ἄρα": {"ῥα", "ῥ᾽"}}
    pairs = _derive_elision_pairs(lemma_forms)
    for word in ("τονδί", "βασιλέα", "κρέα", "σά"):
        assert word not in pairs, word
    assert pairs["τόνδε"] == "τόνδ᾽"
    assert pairs["ῥα"] == "ῥ᾽"


def test_a_grave_before_the_last_syllable_is_not_an_accent():
    # μὲτὰ is a malformed μετά, not a proparoxytone carrying an enclitic's
    # accent the way χεῖρά is, so it stays an oxytone.
    assert _is_oxytone("μὲτὰ")
    assert not _is_oxytone("χεῖρά")


ARTIFACT = ROOT / "build" / "hunspell" / "grc_morph.json"


@pytest.mark.skipif(not ARTIFACT.exists(), reason="grc_morph.json not built")
def test_the_built_nu_list_follows_the_texts():
    nu = set(json.loads(ARTIFACT.read_text(encoding="utf-8"))["nu"])
    # Imperatives the texts mostly mean, a deictic, and the Doric -τι;
    # imperatives the taggers file as unaugmented pasts.
    assert not {"φέρε", "ἄκουε", "ἴδε", "οὑτωσί", "ἐντί"} & nu
    assert not {"κατένεγκε", "Κατένεγκε", "προσανένεγκε", "ξένεγκε",
                "ἐντύγχανε", "προσέμβαλλε"} & nu
    # An imperfect the ν spelling carries, the Ionic article's dative
    # plural, an adverb in -σι, an epic subjunctive, ἐστί's compounds, the
    # adverbs in -θε, a σ-aorist and an unaugmented aorist written with ν.
    assert {"μέλλε", "τῇσι", "παντάπασι", "ἐθέλῃσι", "πάρεστι",
            "ἔλεγε", "ἐστί", "λέγουσι", "πρόσθε", "ὄπισθε", "βλάστησε",
            "βάλε", "δόξειε", "ποιήσειε", "σήμηνε"} <= nu
    assert "ἄντε" not in nu


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

    # Datives the tagged corpora do not analyze, read from their lemmas'
    # forms, and the uncontracted -εϊ Diorisis calls a dual (Smyth 72).
    datives = {"δμητῆρι", "διι", "παντι", "κήδεϊ", "λαίφεϊ", "μέλεϊ"}
    assert not datives & set(table), sorted(datives & set(table))
    # Verb forms in -ι still elide.
    assert table.get("κάμψαιμι") == "κάμψαιμ᾽"

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

    # Elision drops the last vowel and may move the accent, nothing else. A
    # value that changes a letter, a breathing or an iota subscript is the
    # elision of another word (ἐνί took the numeral's ἕν᾽), and a key whose
    # accent sits before its last vowel keeps that accent where it was
    # (αἰτία took αἴτι᾽, the neuter plural's). An oxytone may add the acute
    # it throws back (ὦγαθέ gives ὦγάθ᾽), and that is all it may add.
    accents = {"\u0300", "\u0301", "\u0342"}

    def keeps(kept, own):
        return kept == own or any(
            kept[:i] + kept[i + 1:] == own
            for i, c in enumerate(kept) if c == "\u0301")

    def stem(form):
        nfd = unicodedata.normalize("NFD", form)
        last = max(i for i, c in enumerate(nfd) if c.lower() in "αεηιουω")
        return nfd[:last]

    def unaccented(nfd):
        return "".join(c for c in nfd if c not in accents).lower()

    other_word, moved = {}, {}
    for k, v in table.items():
        kept, own = unicodedata.normalize("NFD", v)[:-1], stem(k)
        if unaccented(kept) != unaccented(own):
            other_word[k] = v
        elif any(c in accents for c in own) and not keeps(kept, own):
            moved[k] = v
    assert not other_word, f"values that spell another word: {list(other_word.items())[:10]}"
    assert not moved, f"values that move the key's own accent: {list(moved.items())[:10]}"

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
            and not em._elides_bare(k)
            and not em._has_accent(v)]
    weight = sum(freq.get(exact_form_key(k), 0) for k in bare)
    assert weight < 25_000, (
        f"elided oxytones left bare carry {weight:,} occurrences: "
        f"{sorted(bare, key=lambda k: -freq.get(exact_form_key(k), 0))[:10]}"
    )


# --------------------------------------------------------------------------
# The Modern Greek elision overlay (el_modern)
# --------------------------------------------------------------------------

KORONIS = "\u1FBD"


def test_modern_only_elisions_keep_only_what_the_ancient_table_lacks():
    from export_morphology import modern_only_elisions
    ancient = {"ἀλλά": "ἀλλ" + KORONIS, "μιά": "μί" + KORONIS}
    modern = {"ἀλλά": "ἀλλ" + KORONIS, "μιά": "μι" + KORONIS,
              "τώρα": "τώρ" + KORONIS}
    only, conflicts = modern_only_elisions(ancient, modern)
    assert only == {"μιά": "μι" + KORONIS, "τώρα": "τώρ" + KORONIS}
    assert conflicts == {"μιά": ("μί" + KORONIS, "μι" + KORONIS)}


@pytest.mark.skipif(not ARTIFACT.exists(), reason="grc_morph.json not built")
def test_the_built_modern_greek_elisions_hold_their_invariants():
    """el_modern lays the Modern Greek elisions over el: only pairs el
    lacks or spells otherwise, each a well-formed elided spelling."""
    from export_mg_polytonic import mg_orthography_reason

    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    if "el_modern" not in payload:
        pytest.skip("grc_morph.json built without the Modern Greek table")
    ancient, modern = payload["el"], payload["el_modern"]
    assert modern
    assert not [k for k, v in modern.items() if ancient.get(k) == v]
    bad = {k: v for k, v in modern.items()
           if not v.endswith(KORONIS) or mg_orthography_reason(v) is not None}
    assert not bad, bad
    for full, stem in [("γιὰ", "γι"), ("στὸ", "στ"), ("τώρα", "τώρ"),
                       ("ὅλα", "ὅλ"), ("κι", "κι")]:
        assert modern.get(full) == stem + KORONIS, full
    # The Ancient Greek table is not touched: the article and ὅτι, which
    # Ancient Greek never elides, are elided only in the Modern overlay.
    assert "τὸ" not in ancient and "ὅτι" not in ancient
    # Each pair carries how often the texts elide it before a vowel.
    shares = payload["el_modern_share"]
    assert set(shares) == set(modern)
    assert all(0 <= v <= 1 for v in shares.values())
    assert shares["γιὰ"] > 0.5 > shares["τώρα"]
