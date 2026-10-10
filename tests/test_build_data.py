"""Focused tests for Kaikki build-time morphology extraction."""

from __future__ import annotations

import json

import pytest

from build_data import (_parse_verb_tense, extract_pairs, resolve_cross_source,
                        spelling_target)


@pytest.mark.parametrize(("header", "expected"), [
    ("present", "present"),
    ("Epic aorist", "aorist"),
    ("Attic contracted future", "future"),
    ("future perfect", "future-perfect"),
    ("Attic declension-3", ""),
    ("Epic", ""),
])
def test_parse_verb_tense(header, expected):
    assert _parse_verb_tense(header) == expected


def test_extract_pairs_propagates_table_tense(tmp_path):
    dump = tmp_path / "kaikki.jsonl"
    entry = {
        "word": "λύω",
        "pos": "verb",
        "senses": [{"glosses": ["loosen"]}],
        "forms": [
            {"form": "Epic aorist", "tags": ["table-tags"]},
            {
                "form": "ἔλυσα",
                "tags": [
                    "active", "indicative", "first-person", "singular",
                ],
            },
            {"form": "Attic declension-3", "tags": ["table-tags"]},
            {
                "form": "λύων",
                "tags": ["active", "participle", "nominative"],
            },
        ],
    }
    dump.write_text(json.dumps(entry, ensure_ascii=False) + "\n",
                    encoding="utf-8")

    pairs, *_ = extract_pairs(dump, "grc-en")
    by_form = {pair["form"]: pair for pair in pairs}

    assert "aorist" in by_form["ἔλυσα"]["tags"]
    assert "Epic" in by_form["ἔλυσα"]["tags"]
    assert not ({"present", "imperfect", "future", "aorist", "perfect",
                 "pluperfect", "future-perfect"}
                & set(by_form["λύων"]["tags"]))


def _write(tmp_path, entries):
    dump = tmp_path / "kaikki.jsonl"
    dump.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n"
                            for e in entries), encoding="utf-8")
    return dump


def test_a_capitalized_modern_page_does_not_take_a_lowercase_word(tmp_path):
    # The female name Ερωτώ and the abbreviation ΔΕΣ must not become the
    # lemma of the verb ερωτώ or of δες, whose own pages are form-of pages
    # (the lowest confidence).
    dump = _write(tmp_path, [
        {"word": "Ερωτώ", "pos": "name",
         "senses": [{"glosses": ["γυναικείο όνομα"]}]},
        {"word": "ΔΕΣ", "pos": "verb",
         "senses": [{"glosses": ["συντομογραφία"]}]},
        {"word": "ερωτώ", "pos": "verb",
         "senses": [{"glosses": ["λόγια μορφή του ρωτώ"],
                     "form_of": [{"word": "ρωτώ"}]}]},
        {"word": "δες", "pos": "verb",
         "senses": [{"glosses": ["β' ενικό προστακτικής του βλέπω"],
                     "form_of": [{"word": "βλέπω"}]}]},
    ])
    _, lookup, *_ = extract_pairs(dump, "el-el")
    assert lookup["ερωτώ"][0] == "ρωτώ"
    assert lookup["δες"][0] == "βλέπω"
    # The capitalized pages keep their own keys.
    assert lookup["Ερωτώ"][0] == "Ερωτώ"
    # An Ancient Greek dump is read as before.
    _, lookup, *_ = extract_pairs(dump, "grc-el")
    assert lookup["ερωτώ"][0] == "Ερωτώ"


def test_cross_source_resolution_makes_no_cycle():
    # EL's ρωτώ table lists ρωτάω; EN files ρωτώ as a form of ρωτάω. The
    # EN reference would point ρωτώ back at ρωτάω, a cycle that deletes
    # both: ρωτώ stays the lemma.
    lookup = {"ρωτώ": ("ρωτώ", 3), "ρωτάω": ("ρωτώ", 1),
              "αυτούς": ("αυτούς", 1), "αυτός": ("αυτός", 3)}
    moved = resolve_cross_source(lookup, {"ρωτώ": "ρωτάω",
                                          "αυτούς": "αυτός"}, {}, set())
    assert moved == 1
    assert lookup["ρωτώ"] == ("ρωτώ", 3) and lookup["ρωτάω"][0] == "ρωτώ"
    assert lookup["αυτούς"][0] == "αυτός"


def test_a_pronoun_grid_of_several_persons_is_not_read(tmp_path):
    # EN's τα page carries the whole personal-pronoun grid under a generic
    # template; its first- and second-person forms are no forms of τα.
    grid = [{"form": "inflection-table-top", "tags": ["inflection-template"]},
            {"form": "εγώ", "tags": ["first-person", "nominative",
                                     "singular", "strong"]},
            {"form": "εσύ", "tags": ["second-person", "nominative",
                                     "singular", "strong"]},
            {"form": "αυτού", "tags": ["third-person", "genitive",
                                       "singular", "strong"]}]
    dump = _write(tmp_path, [
        {"word": "τα", "pos": "pron", "forms": grid,
         "senses": [{"glosses": ["they"]}]},
        {"word": "αυτός", "pos": "pron",
         "forms": [{"form": "αυτού", "tags": ["genitive", "singular"]}],
         "senses": [{"glosses": ["he"]}]},
    ])
    pairs, lookup, _, pos_map, _ = extract_pairs(dump, "el-en")
    assert not [p for p in pairs if p["lemma"] == "τα" and p["form"] != "τα"]
    assert pos_map["αυτού"]["PRON"] == "αυτός"


def test_a_polytonic_spelling_page_does_not_become_its_words_lemma(tmp_path):
    # The Greek Wiktionary's γιός only says it is the polytonic spelling of
    # γιος, and comes first in the dump. Its accentless key is γιος, and
    # γιος's own page, whose gloss names a male ("αρσενικού φύλου"), is
    # read as a form-of page that cannot take the key back: γιος became a
    # form of γιός, and its paradigm was dropped.
    dump = _write(tmp_path, [
        {"word": "γιός", "pos": "noun",
         "forms": [{"form": "γιός", "tags": ["masculine"]}],
         "senses": [{"glosses": ["πολυτονική γραφή του γιος"],
                     "alt_of": [{"word": "γιος"}]}]},
        {"word": "μιλῶ", "pos": "verb",
         "senses": [{"glosses": ["πολυτονική γραφή του μιλώ"]}]},
        {"word": "μουστακᾶτος", "pos": "noun",
         "senses": [{"glosses": ["πολυτονική γραφή του μουστακάτος"],
                     "alt_of": [{"word": "μουστακάτος"}]}]},
        {"word": "οἴκων", "pos": "noun",
         "senses": [{"glosses": ["πολυτονική γραφή του οίκων"],
                     "alt_of": [{"word": "οίκων"}]}]},
        {"word": "γιος", "pos": "noun",
         "forms": [{"form": "γιου", "tags": ["genitive", "singular"]},
                   {"form": "γιων", "tags": ["genitive", "plural"]}],
         "senses": [{"glosses": ["πρώτου βαθμού συγγένειας απόγονος "
                                 "αρσενικού φύλου"]}]},
        {"word": "μιλώ", "pos": "verb",
         "forms": [{"form": "μιλάς", "tags": ["present", "second-person",
                                             "singular"]}],
         "senses": [{"glosses": ["μιλάω"]}]},
        {"word": "οίκων", "pos": "noun",
         "senses": [{"glosses": ["γενική πληθυντικού του οίκος"],
                     "form_of": [{"word": "οίκος"}]}]},
        {"word": "οίκος", "pos": "noun",
         "senses": [{"glosses": ["σπίτι"]}]},
        # A polytonic page of a word of other letters is that word's form.
        {"word": "ἐτρώθη", "pos": "verb",
         "senses": [{"glosses": ["πολυτονική γραφή του ετρώθη· γ΄ πρόσωπο "
                                 "ενικού οριστικής παθητικού αορίστου του "
                                 "τιτρώσκω"],
                     "form_of": [{"word": "τιτρώσκω"}]}]},
    ])
    pairs, lookup, *_ = extract_pairs(dump, "el-el")
    assert lookup["γιος"][0] == "γιος" and lookup["γιός"][0] == "γιος"
    assert lookup["γιου"][0] == "γιος"
    assert lookup["μιλώ"][0] == "μιλώ" and lookup["μιλῶ"][0] == "μιλώ"
    assert lookup["μουστακάτος"][0] == "μουστακάτος"
    assert lookup["μουστακᾶτος"][0] == "μουστακάτος"
    # The spelling page does not keep οίκων's own page from recording that
    # it is a form of οίκος.
    assert lookup["οίκων"][0] == "οίκος"
    assert lookup["ἐτρώθη"][0] == "τιτρώσκω"
    assert {"form": "γιός", "lemma": "γιος", "pos": "noun",
            "tags": ["alt-of"]} in pairs
    assert {"form": "μιλῶ", "lemma": "μιλώ", "pos": "verb",
            "tags": ["alt-of"]} in pairs
    # An Ancient Greek dump is read as before: there a page whose monotonic
    # spelling is the word it names is a form of it (φωνῇ of φωνή).
    _, lookup, *_ = extract_pairs(dump, "grc-el")
    assert lookup["γιος"][0] == "γιός" and lookup["μιλῶ"][0] == "μιλῶ"


def test_spelling_target():
    # Every sense names the page's own word without the marks monotonic
    # writing drops, by a reference or by the gloss; for a monosyllable
    # without its accent too.
    assert spelling_target(
        "γιός", [{"glosses": ["πολυτονική γραφή του γιος"]}]) == "γιος"
    assert spelling_target(
        "μουστακᾶτος", [{"glosses": ["πολυτονική γραφή του μουστακάτος"],
                         "alt_of": [{"word": "μουστακάτος"}]}]) == "μουστακάτος"
    assert spelling_target(
        "ἱστορία", [{"glosses": ["Katharevousa form of ιστορία (istoría)"],
                     "alt_of": [{"word": "ιστορία"}]}]) == "ιστορία"
    # A page with a sense of its own is a word of its own.
    assert spelling_target(
        "ἀγορά", [{"glosses": ["πολυτονική γραφή του αγορά"]},
                  {"glosses": ["τόπος όπου πουλιούνται τρόφιμα"]}]) is None
    # A page naming a word of other letters is no spelling of it (κυνηγῶ of
    # κυνηγάω, ἐτρώθη of τιτρώσκω), nor is a page with no senses.
    assert spelling_target(
        "κυνηγῶ", [{"glosses": ["πολυτονική γραφή του κυνηγάω"]}]) is None
    assert spelling_target(
        "ἐτρώθη", [{"glosses": ["πολυτονική γραφή του ετρώθη· γ΄ πρόσωπο "
                                "ενικού οριστικής παθητικού αορίστου του "
                                "τιτρώσκω"],
                    "form_of": [{"word": "τιτρώσκω"}]}]) is None
    assert spelling_target("μιλῶ", []) is None
    assert spelling_target("γιος", [{"glosses": ["απόγονος"]}]) is None


def test_two_pages_that_name_each_other_both_stay_lemmas(tmp_path):
    # αγχώδης and αγχωτικός each have a sense "συνώνυμο του" the other,
    # which kaikki gives as alt_of, so both are form-of pages. Following
    # both references made a cycle that chain-breaking deleted with both
    # tables; the polytonic page ἀγχωτικός hid it by taking the key
    # αγχωτικός as a lemma of its own.
    dump = _write(tmp_path, [
        {"word": "αγχώδης", "pos": "adj",
         "forms": [{"form": "αγχώδεις", "tags": ["nominative", "plural"]}],
         "senses": [{"glosses": ["που καταλαμβάνεται εύκολα από άγχος"]},
                    {"glosses": ["συνώνυμο του αγχωτικός"],
                     "alt_of": [{"word": "αγχωτικός"}]}]},
        {"word": "αγχωτικός", "pos": "adj",
         "forms": [{"form": "αγχωτική", "tags": ["feminine", "nominative",
                                                 "singular"]}],
         "senses": [{"glosses": ["που προκαλεί άγχος"]},
                    {"glosses": ["συνώνυμο του αγχώδης"],
                     "alt_of": [{"word": "αγχώδης"}]}]},
        {"word": "ἀγχωτικός", "pos": "adj",
         "forms": [{"form": "ἀγχωτική", "tags": ["feminine", "nominative",
                                                 "singular"]}],
         "senses": [{"glosses": ["πολυτονική γραφή του αγχωτικός"]}]},
        # A one-way reference is still followed.
        {"word": "χτιστός", "pos": "adj",
         "senses": [{"glosses": ["άλλη μορφή του κτιστός"],
                     "form_of": [{"word": "κτιστός"}]}]},
    ])
    _, lookup, *_ = extract_pairs(dump, "el-el")
    assert lookup["αγχώδης"][0] == "αγχώδης"
    assert lookup["αγχωτικός"][0] == "αγχωτικός"
    assert lookup["αγχώδεις"][0] == "αγχώδης"
    assert lookup["αγχωτική"][0] == "αγχωτικός"
    assert lookup["ἀγχωτικός"][0] == "αγχωτικός"
    assert lookup["χτιστός"][0] == "κτιστός"
    # The Ancient and Medieval Greek dumps follow both references as before.
    for lang in ("grc-el", "mgr-el"):
        _, lookup, *_ = extract_pairs(dump, lang)
        assert lookup["αγχώδης"][0] == "αγχωτικός"


def test_weak_pronouns_on_a_headword_line_are_no_forms(tmp_path):
    # The Greek Wiktionary's μόνος shows the weak pronouns it takes (μόνος
    # μου, σου, μας, σας) as untagged forms, which came first in the dump
    # and made μας and σας forms of μόνος.
    dump = _write(tmp_path, [
        {"word": "μόνος", "pos": "pron",
         "forms": [{"form": "μόνος"}, {"form": "μας"}, {"form": "σας"}],
         "senses": [{"glosses": ["χωρίς συνοδεία ή βοήθεια"]}]},
        {"word": "μας", "pos": "pron",
         "senses": [{"glosses": ["γενική πληθυντικού του εγώ"],
                     "form_of": [{"word": "εγώ"}]}]},
    ])
    pairs, lookup, *_ = extract_pairs(dump, "el-el")
    assert lookup["μας"][0] == "εγώ"
    assert not [p for p in pairs if p["lemma"] == "μόνος"
                and p["form"] in ("μας", "σας")]
