"""Focused tests for Kaikki build-time morphology extraction."""

from __future__ import annotations

import json

import pytest

from build_data import (Abbreviations, _parse_verb_tense, apply_participle_references,
                        clean_training_pairs, deduplicate_pairs,
                        drop_abbreviation_targets, extract_pairs,
                        merge_pos_maps, merge_source_lookups, resolve_cross_source,
                        preserve_finite_verb_heads, resolve_pair_lemma,
                        spelling_target, vowel_groups)


def test_source_merge_prefers_a_lowercase_verb_to_another_sources_name(tmp_path):
    el = _write(tmp_path, [
        {"word": "Λεν", "pos": "name", "senses": [{"glosses": ["surname"]}]},
        {"word": "Μιλάνος", "pos": "name", "senses": [{"glosses": ["surname"]}],
         "forms": [{"form": "Μιλάνε", "tags": ["singular", "vocative"]}]},
    ])
    en = tmp_path / "en.jsonl"
    en.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in [
        {"word": "λεν", "pos": "verb", "senses": [{"form_of": [{"word": "λέω"}]}]},
        {"word": "μιλάνε", "pos": "verb", "senses": [{"form_of": [{"word": "μιλώ"}]}]},
    ]))
    _, el_lookup, *_ = extract_pairs(el, "el-el")
    _, en_lookup, *_ = extract_pairs(en, "el-en")
    merged = {}
    merge_source_lookups(merged, [el_lookup, en_lookup], capitals_last=True)
    assert merged["λεν"][0] == "λέω"
    assert merged["μιλάνε"][0] == "μιλώ"
    assert merged["Λεν"][0] == "Λεν"
    assert merged["Μιλάνε"][0] == "Μιλάνος"
    old = {}
    merge_source_lookups(old, [el_lookup, en_lookup])
    assert old["λεν"][0] == "Λεν"


def test_form_page_head_and_participle_cells_follow_their_verb(tmp_path):
    dump = _write(tmp_path, [
        {"word": "ερωτήσεις", "pos": "noun",
         "forms": [{"form": "ερωτήσεις", "tags": ["feminine"]}],
         "senses": [{"form_of": [{"word": "ερώτηση"}]}]},
        {"word": "ερωτήσεις", "pos": "verb",
         "forms": [{"form": "ερωτήσεις"}],
         "senses": [{"form_of": [{"word": "ερωτώ"}]}]},
        {"word": "αντιπαθών", "pos": "verb",
         "forms": [{"form": "αντιπαθούσα", "tags": ["feminine", "nominative"]}],
         "senses": [{"form_of": [{"word": "αντιπαθώ"}]}]},
    ])
    pairs, _, _, pos_map, *_ = extract_pairs(dump, "el-el")
    lookup = {"ερωτήσεις": ("ερώτηση", 1), "ερώτηση": ("ερώτηση", 3),
              "ερωτώ": ("ερωτώ", 3), "αντιπαθών": ("αντιπαθητικός", 1),
              "αντιπαθητικός": ("αντιπαθητικός", 3), "αντιπαθώ": ("αντιπαθώ", 3)}
    clean = clean_training_pairs(pairs, lookup, preserve_cells=True)
    assert {p["lemma"] for p in clean if p["form"] == "ερωτήσεις" and p["pos"] == "noun"} == {"ερώτηση"}
    assert {p["lemma"] for p in clean if p["form"] == "ερωτήσεις" and p["pos"] == "verb"} == {"ερωτώ"}
    assert {p["lemma"] for p in clean if p["form"] == "αντιπαθούσα"} == {"αντιπαθώ"}
    assert pos_map["αντιπαθούσα"]["VERB"] == "αντιπαθώ"
    assert all("via" not in p for p in clean)


def test_cross_source_reciprocal_synonyms_keep_both_paradigms():
    for order in (("νησοπυριτικός", "ορθοπυριτικός"), ("ορθοπυριτικός", "νησοπυριτικός")):
        lookup = {word: (word, 3) for word in order}
        refs = {order[0]: order[1], order[1]: order[0]}
        assert resolve_cross_source(lookup, {}, refs, set(), preserve_cycles=True) == 0
        assert all(word == lemma for word, (lemma, _) in lookup.items())


def test_pair_deduplication_preserves_each_morphological_cell():
    pairs = [{"form": "καθηγητή", "lemma": "καθηγητής", "pos": "noun", "tags": tags}
             for tags in (["genitive", "singular"], ["accusative", "singular"],
                          ["vocative", "singular"], ["singular", "genitive"])]
    pairs.append({**pairs[0], "pos": "adj"})
    unique = deduplicate_pairs(pairs, preserve_cells=True)
    assert len(unique) == 4
    assert {tuple(p["tags"]) for p in unique if p["pos"] == "noun"} == {
        ("genitive", "singular"), ("accusative", "singular"), ("vocative", "singular")}
    assert deduplicate_pairs(pairs) == pairs[:1]


def test_a_missing_adjective_lemma_does_not_resolve_to_a_name():
    el = {"κυριακή": ("Κυριακή", 0), "Κυριακή": ("Κυριακή", 3),
          "κυριακός": ("Κυριακός", 0), "Κυριακός": ("Κυριακός", 3)}
    en = {"κυριακή": ("κυριακός", 1)}
    merged = {}
    merge_source_lookups(merged, [el, en], capitals_last=True)
    assert merged["κυριακή"][0] == "Κυριακή"
    pair = {"form": "κυριακή", "lemma": "κυριακός", "pos": "adj", "tags": ["form-of"]}
    # Pair-level nominal analyses retain the source's existing fallback;
    # only flat-key ownership is protected here.
    assert clean_training_pairs([pair], merged, preserve_cells=True)[0]["lemma"] == "Κυριακός"
    # A real lowercase adjective page is enough to resolve the reference.
    en["κυριακός"] = ("κυριακός", 3)
    merged = {}
    merge_source_lookups(merged, [el, en], capitals_last=True)
    assert merged["κυριακή"][0] == "κυριακός"


def test_name_fallback_does_not_drop_attested_noun_inflections():
    el = {"βροντόσαυρος": ("Βροντόσαυρος", 0),
          "Βροντόσαυρος": ("Βροντόσαυρος", 3)}
    en = {"βροντόσαυρου": ("βροντόσαυρος", 1)}
    merged = {}
    merge_source_lookups(merged, [el, en], True,
                         {"βροντόσαυρου": {"noun"}, "Βροντόσαυρος": {"name"}})
    assert merged["βροντόσαυρου"][0] == "βροντόσαυρος"
    pairs = [{"form": "βροντόσαυρου", "lemma": "βροντόσαυρος", "pos": "noun", "tags": []}]
    assert clean_training_pairs(pairs, merged, True, {}, {})[0]["lemma"] == "Βροντόσαυρος"


def test_missing_adjective_head_does_not_drop_source_inflections():
    source = {"στρογγύλη": ("στρογγύλος", 1)}
    names = {"στρογγύλος": ("Στρογγύλος", 0), "Στρογγύλος": ("Στρογγύλος", 3)}
    merged = {}
    merge_source_lookups(merged, [names, source], True, {"στρογγύλη": {"adj"}})
    assert merged["στρογγύλη"][0] == "στρογγύλος"
    pairs = [{"form": "στρογγύλη", "lemma": "στρογγύλος", "pos": "adj", "tags": []}]
    assert clean_training_pairs(pairs, merged, True, {}, {})[0]["lemma"] == "Στρογγύλος"


def test_verb_reference_chains_do_not_end_at_a_nominal_homograph():
    lookup = {"ἀντιπαθών": ("αντιπαθητικός", 1),
              "αντιπαθών": ("αντιπαθητικός", 1),
              "αντιπαθητικός": ("αντιπαθητικός", 3),
              "αντιπαθώ": ("αντιπαθώ", 3),
              "απύθμενου": ("απύθμενος", 1),
              "απύθμενος": ("απύθμενος", 3)}
    pos_map = {"ἀντιπαθών": {"VERB": "αντιπαθών"},
               "αντιπαθών": {"VERB": "αντιπαθώ"},
               "απύθμενου": {"VERB": "απύθμενος"}}
    lemma_pos = {"αντιπαθητικός": {"adj"}, "αντιπαθώ": {"verb"},
                 "απύθμενος": {"adj"}}
    assert resolve_pair_lemma("ἀντιπαθών", "verb", lookup, pos_map, lemma_pos) == "αντιπαθώ"
    assert resolve_pair_lemma("απύθμενου", "verb", lookup, pos_map, lemma_pos) is None


def test_capitalized_noun_paradigms_survive_and_legacy_behavior_stays():
    lookup = {"αμερικανός": ("Αμερικανός", 1), "Αμερικανός": ("Αμερικανός", 3),
              "θεός": ("Θεός", 1), "Θεός": ("Θεός", 3)}
    pairs = [{"form": "αμερικάνου", "lemma": "αμερικανός", "pos": "noun", "tags": ["genitive"]},
             {"form": "θεού", "lemma": "θεός", "pos": "noun", "tags": ["genitive"]}]
    clean = clean_training_pairs(pairs, lookup, preserve_cells=True,
                                lemma_pos={"αμερικανός": {"noun"}, "θεός": {"noun"}})
    assert len(clean) == 2
    assert {p["lemma"] for p in clean} == {"Αμερικανός", "Θεός"}
    legacy = [{"form": "λέν", "lemma": "λεν", "pos": "verb", "tags": []}]
    old_lookup = {"λεν": ("Λεν", 1), "Λεν": ("Λεν", 3)}
    assert clean_training_pairs(legacy, old_lookup)[0]["lemma"] == "Λεν"


def test_missing_verb_reference_can_use_another_sources_verb_evidence():
    pair = {"form": "κυμαινόμενα", "lemma": "κυμαινόμενος", "pos": "verb",
            "tags": ["neuter", "plural"], "via": "κυμαίνω"}
    lookup = {"κυμαινόμενος": ("κυμαίνομαι", 1), "κυμαίνομαι": ("κυμαίνομαι", 3)}
    clean = clean_training_pairs([pair], lookup, True, {}, {"κυμαίνομαι": {"verb"}})
    assert clean == [{k: v for k, v in pair.items() if k not in ("via", "lemma")} | {"lemma": "κυμαίνομαι"}]
    lookup["κυμαινόμενος"] = ("κύμα", 1)
    lookup["κύμα"] = ("κύμα", 3)
    assert clean_training_pairs([pair], lookup, True, {}, {"κύμα": {"noun"}}) == []


def test_explicit_verb_reference_can_repair_a_mislabeled_head_page(tmp_path):
    dump = _write(tmp_path, [
        {"word": "δηώ", "pos": "romanization", "senses": [{"tags": ["no-gloss"]}]},
        {"word": "δήωσα", "pos": "verb", "forms": [{"form": "δήωσα"}],
         "senses": [{"glosses": ["α' ενικό οριστικής αορίστου του ρήματος δηώ"],
                     "form_of": [{"word": "δηώ"}]}]},
        {"word": "απύθμενος", "pos": "adj", "senses": [{"glosses": ["bottomless"]}]},
        {"word": "απύθμενου", "pos": "verb", "forms": [{"form": "απύθμενου"}],
         "senses": [{"glosses": ["γενική ενικού του απύθμενος"],
                     "form_of": [{"word": "απύθμενος"}]}]},
        {"word": "χοροστατών", "pos": "adj", "senses": [{"glosses": ["presiding"]}]},
        {"word": "χοροστατώ", "pos": "verb", "senses": [{"glosses": ["preside"]}]},
        {"word": "χοροστατούντος", "pos": "verb", "forms": [{"form": "χοροστατούντος"}],
         "senses": [{"glosses": ["γενική ενικού του χοροστατών (μετοχή του ρήματος χοροστατώ)"],
                     "form_of": [{"word": "χοροστατών"}]}]},
    ])
    lemma_pos = {}
    pairs, lookup, _, pos_map, *_ = extract_pairs(dump, "el-el", lemma_pos=lemma_pos)
    clean = clean_training_pairs(pairs, lookup, True, pos_map, lemma_pos)
    assert any(p["form"] == "δήωσα" and p["lemma"] == "δηώ" and p["pos"] == "verb" for p in clean)
    assert not any(p["form"] == "απύθμενου" and p["pos"] == "verb" for p in clean)
    assert lemma_pos["χοροστατών"] == {"adj"}
    assert any(p["form"] == "χοροστατούντος" and p["lemma"] == "χοροστατώ" for p in clean)


def test_other_sources_verb_reference_can_replace_a_participle_self_map():
    first = {"ισχύων": {"VERB": "ισχύων", "ADJ": "ισχύων"}}
    second = {"ισχύων": {"VERB": "ισχύω", "ADJ": "ισχύς"}}
    merge_pos_maps(first, second, prefer_verb_refs=True)
    assert first["ισχύων"] == {"VERB": "ισχύω", "ADJ": "ισχύων"}
    legacy = {"ισχύων": {"VERB": "ισχύων"}}
    merge_pos_maps(legacy, second)
    assert legacy["ισχύων"]["VERB"] == "ισχύων"


def test_explanatory_verb_glosses_do_not_create_references(tmp_path):
    dump = _write(tmp_path, [
        {"word": word, "pos": "verb", "forms": [{"form": word}],
         "senses": [{"glosses": [gloss]}]}
        for word, gloss in [
            ("παραέχω", "σε ρόλο βοηθητικού ρήματος αντί του έχω για να δηλωθεί υπερβολή"),
            ("χαιρετώ", "χρήση του ρήματος για αποχαιρετισμό"),
            ("αντικαθιστώ", "γράφω τους τύπους ενός ρήματος σε άλλους χρόνους ή εγκλίσεις"),
            ("μοσχοπουλιέμαι", "παθητική φωνή του ρήματος μοσχοπουλάω / μοσχοπουλώ"),
            ("ψιμυθιώνομαι", "παθητική φωνή του ρήματος ψιμυθιώνω: μακιγιάρομαι"),
            ("συμφραζόμενος", "μετοχή ενεστώτα του ρήματος συμφράζομαι"),
            ("ἡττῶμαι", "ρήμα της καθαρεύουσας, συνηρημένος τύπος του αρχαίου ρήματος ἡττάομαι"),
        ]
    ])
    lemma_pos = {}
    pairs, lookup, _, pos_map, *_ = extract_pairs(dump, "el-el", lemma_pos=lemma_pos)
    assert not ({"αντί", "για", "σε", "μοσχοπουλάω", "ψιμυθιώνω", "συμφράζομαι", "ηττάομαι"} & lemma_pos.keys())
    clean = clean_training_pairs(pairs, lookup, True, pos_map, lemma_pos)
    assert {p["form"] for p in clean} >= {"παραέχω", "χαιρετώ", "αντικαθιστώ", "μοσχοπουλιέμαι", "ψιμυθιώνομαι", "συμφραζόμενος", "ἡττῶμαι"}


def test_broken_verb_reference_preserves_an_attested_head():
    lookup = {"ημαρτημένος": ("ημαρτημένος", 3), "αμαρτάνω": ("αμαραταίνω", 1)}
    pos_map = {"ημαρτημένος": {"VERB": "αμαρτάνω"}}
    assert resolve_pair_lemma("ημαρτημένος", "verb", lookup, pos_map,
                              {"ημαρτημένος": {"verb"}, "αμαρτάνω": {"verb"}}) == "ημαρτημένος"


def test_missing_proper_name_reference_preserves_head_forms(tmp_path):
    dump = _write(tmp_path, [{"word": "Άδωνις", "pos": "name", "forms": [{"form": "Άδωνις"}],
                              "senses": [{"alt_of": [{"word": "Ἄδωνις"}]}]}])
    pairs, lookup, _, pos_map, *_ = extract_pairs(dump, "el-el")
    assert any(p["form"] == "Άδωνις" for p in clean_training_pairs(pairs, lookup, True, pos_map))


@pytest.mark.parametrize("reverse", [False, True])
def test_participle_top_reference_requires_an_independent_verb_head(tmp_path, reverse):
    entries = [
        {"word": "αναχρονισμένος", "pos": "verb", "tags": ["participle"],
         "form_of": [{"word": "αναχρονίζω"}],
         "forms": [{"form": "αναχρονισμένα", "tags": ["plural"]}],
         "senses": [{"form_of": [{"word": "αναχρονιστικός"}]}]},
        {"word": "αναχρονιστικός", "pos": "adj"},
        {"word": "αναχρονίζω", "pos": "verb"},
        {"word": "αστροσπαρμένος", "pos": "verb", "tags": ["participle"],
         "form_of": [{"word": "αστρόσπαρτος"}],
         "senses": [{"form_of": [{"word": "αστρόσπαρτος"}]}]},
        {"word": "αστρόσπαρτος", "pos": "adj"},
        {"word": "ποτήρι", "pos": "noun", "form_of": [{"word": "ποτήριον"}],
         "forms": [{"form": "ποτήρια"}]},
        {"word": "ποτήριον", "pos": "noun"},
    ]
    dump = _write(tmp_path, list(reversed(entries)) if reverse else entries)
    refs, heads, lemma_pos = {}, set(), {}
    pairs, lookup, _, pos_map, *_ = extract_pairs(
        dump, "el-el", lemma_pos=lemma_pos, participle_refs=refs, verb_heads=heads)
    apply_participle_references(pos_map, refs, heads)
    clean = clean_training_pairs(pairs, lookup, True, pos_map, lemma_pos)
    assert any(p["form"] == "αναχρονισμένα" and p["lemma"] == "αναχρονίζω" for p in clean)
    assert pos_map["αστροσπαρμένος"]["VERB"] == "αστρόσπαρτος"
    assert not any(p["lemma"] == "αστρόσπαρτος" and p["pos"] == "verb" for p in clean)
    assert any(p["form"] == "ποτήρια" and p["lemma"] == "ποτήρι" for p in clean)


@pytest.mark.parametrize("reverse", [False, True])
def test_participle_sense_verb_wins_over_a_top_level_alternate(tmp_path, reverse):
    entries = [
        {"word": "σωροβολιάζομαι", "pos": "verb", "senses": [{"glosses": ["lie in a heap"]}]},
        {"word": "σωροβολιασμένος", "pos": "verb", "tags": ["participle"],
         "form_of": [{"word": "σοροβολιασμένος"}],
         "senses": [{"form_of": [{"word": "σωροβολιάζομαι"}],
                     "glosses": ["μετοχή παθητικού παρακειμένου του ρήματος σωροβολιάζομαι"]}]},
        {"word": "σοροβολιασμένος", "pos": "verb", "tags": ["participle"],
         "senses": [{"form_of": [{"word": "σοροβολιάζομαι"}],
                     "alt_of": [{"word": "σωροβολιασμένος"}],
                     "glosses": ["μετοχή παθητικού παρακειμένου του ρήματος σοροβολιάζομαι, άλλη γραφή του σωροβολιασμένος"]}]},
    ]
    dump = _write(tmp_path, list(reversed(entries)) if reverse else entries)
    refs, heads, lemma_pos = {}, set(), {}
    pairs, lookup, _, pos_map, *_ = extract_pairs(
        dump, "el-el", lemma_pos=lemma_pos, participle_refs=refs, verb_heads=heads)
    assert heads == {"σωροβολιάζομαι"}
    apply_participle_references(pos_map, refs, heads)
    assert pos_map["σωροβολιασμένος"]["VERB"] == "σωροβολιάζομαι"
    clean = clean_training_pairs(pairs, lookup, True, pos_map, lemma_pos)
    assert any(p["form"] == "σοροβολιασμένος" and p["lemma"] == "σωροβολιάζομαι" for p in clean)


@pytest.mark.parametrize("reverse", [False, True])
def test_direct_verb_reference_wins_over_another_lexical_verb(tmp_path, reverse):
    entries = [
        {"word": "σωροβολιάζομαι", "pos": "verb"},
        {"word": "σωρεύω", "pos": "verb"},
        {"word": "σωροβολιασμένος", "pos": "verb", "tags": ["participle"],
         "form_of": [{"word": "σωρεύω"}],
         "senses": [{"form_of": [{"word": "σωροβολιάζομαι"}]}]},
    ]
    dump = _write(tmp_path, list(reversed(entries)) if reverse else entries)
    refs, heads = {}, set()
    _, _, _, pos_map, *_ = extract_pairs(
        dump, "el-el", participle_refs=refs, verb_heads=heads)
    assert heads == {"σωροβολιάζομαι", "σωρεύω"}
    apply_participle_references(pos_map, refs, heads)
    assert pos_map["σωροβολιασμένος"]["VERB"] == "σωροβολιάζομαι"


@pytest.mark.parametrize("reverse", [False, True])
def test_participle_top_reference_preserves_supported_chains(reverse):
    pos_map = {"γραμμένος": {"VERB": "γεγραμμένος"},
               "γεγραμμένος": {"VERB": "γράφω"},
               "ειπωμένος": {"VERB": "ειπωμένος"},
               "κύκλος": {"VERB": "κύκλοςβ"},
               "κύκλοςβ": {"VERB": "κύκλος"}}
    items = [("γραμμένος", {"λέω"}), ("ειπωμένος", {"λέω"}),
             ("κύκλος", {"γράφω"})]
    refs = dict(reversed(items) if reverse else items)
    apply_participle_references(pos_map, refs, {"γράφω", "λέω", "ειπωμένος"})
    assert pos_map["γραμμένος"]["VERB"] == "γράφω"
    assert pos_map["ειπωμένος"]["VERB"] == "λέω"
    assert pos_map["κύκλος"]["VERB"] == "γράφω"


def test_lexical_verb_head_allows_incidental_form_words_in_gloss(tmp_path):
    dump = _write(tmp_path, [{"word": "τρανώνω", "pos": "verb",
                              "senses": [{"glosses": ["make more of something"]}]}])
    heads = set()
    extract_pairs(dump, "el-en", verb_heads=heads)
    assert heads == {"τρανώνω"}


@pytest.mark.parametrize("reverse", [False, True])
def test_participle_top_reference_can_reach_a_lexical_verb_through_a_spelling(reverse):
    pos_map = {"αγαπημένος": {"VERB": "αγαπημένος"},
               "αγαπώ": {"VERB": "αγαπάω"}}
    items = [("αγαπημένος", {"αγαπώ"}), ("αστροσπαρμένος", {"αστρόσπαρτος"})]
    apply_participle_references(pos_map, dict(reversed(items) if reverse else items),
                               {"αγαπάω"})
    assert pos_map["αγαπημένος"]["VERB"] == "αγαπάω"
    assert "αστροσπαρμένος" not in pos_map


@pytest.mark.parametrize("reverse", [False, True])
def test_participle_top_reference_requires_one_existing_lexical_endpoint(reverse):
    pos_map = {"γράφει": {"VERB": "γράφω"}, "λέει": {"VERB": "λέω"}}
    items = [("έγκυρος", {"γράφει", "γράφω"}),
             ("αμφίβολος", {"γράφει", "λέει"}),
             ("εξαρτώμενος", {"άλλος"}), ("άλλος", {"γράφω"})]
    apply_participle_references(pos_map, dict(reversed(items) if reverse else items),
                               {"γράφω", "λέω"})
    assert pos_map["έγκυρος"]["VERB"] == "γράφω"
    assert "αμφίβολος" not in pos_map
    assert pos_map["εξαρτώμενος"]["VERB"] == "γράφω"
    assert pos_map["άλλος"]["VERB"] == "γράφω"


@pytest.mark.parametrize("reverse", [False, True])
def test_participle_graph_keeps_finite_endpoints_and_rejects_isolated_cycles(reverse):
    pos_map = {"συνάπτω": {"VERB": "συνημμένος"},
               "συνημμένος": {"VERB": "συνημμένος"},
               "πρώτος": {"VERB": "δεύτερος"},
               "δεύτερος": {"VERB": "πρώτος"},
               "τρίτος": {"VERB": "τέταρτος"},
               "τέταρτος": {"VERB": "τρίτος"},
               "πέμπτος": {"VERB": "πέμπτος"},
               "έκτος": {"VERB": "έκτος"}}
    items = [("συνημμένος", {"συνάπτω"}),
             ("πρώτος", {"γράφω"}), ("δεύτερος", {"γράφω"}),
             ("τρίτος", {"γράφω"}), ("τέταρτος", {"λέω"}),
             ("πέμπτος", {"έκτος"}), ("έκτος", {"πέμπτος"})]
    refs = dict(reversed(items) if reverse else items)
    apply_participle_references(pos_map, refs,
                               {"συνάπτω", "συνημμένος", "γράφω", "λέω", "πέμπτος", "έκτος"})
    assert pos_map["συνάπτω"]["VERB"] == "συνάπτω"
    assert pos_map["συνημμένος"]["VERB"] == "συνάπτω"
    assert pos_map["πρώτος"]["VERB"] == pos_map["δεύτερος"]["VERB"] == "γράφω"
    assert pos_map["τρίτος"]["VERB"] == "τέταρτος"
    assert pos_map["τέταρτος"]["VERB"] == "τρίτος"
    assert pos_map["πέμπτος"]["VERB"] == "πέμπτος"
    assert pos_map["έκτος"]["VERB"] == "έκτος"


def test_graph_cleanup_preserves_finite_endpoint_and_omits_unresolved_verbs():
    lookup = {"φέρω": ("φέρνω", 3), "φέρνω": ("φέρνω", 3),
              "φέρων": ("φέρων", 3), "φορτίο": ("φορτίο", 3),
              "πρώτος": ("πρώτος", 3), "δεύτερος": ("δεύτερος", 3)}
    pos_map = {"φέρων": {"VERB": "φέρω"}, "φέρω": {"VERB": "φέρω"},
               "πρώτος": {"VERB": "δεύτερος"}, "δεύτερος": {"VERB": "πρώτος"}}
    refs = {"φέρων": {"φέρω"}, "πρώτος": {"δεύτερος"}, "δεύτερος": {"πρώτος"}}
    unresolved = apply_participle_references(pos_map, refs, {"φέρω", "φέρνω"})
    roots = {word: pos_map[word]["VERB"] for word in refs if word not in unresolved}
    pairs = [{"form": word, "lemma": word, "pos": "verb", "tags": []}
             for word in refs]
    pairs.append({"form": "πρώτος", "lemma": "φορτίο", "pos": "noun", "tags": []})
    clean = clean_training_pairs(pairs, lookup, True, pos_map, {}, roots, unresolved)
    assert {tuple((p["form"], p["lemma"], p["pos"])) for p in clean} == {
        ("φέρων", "φέρω", "verb"), ("πρώτος", "φορτίο", "noun")}
    # A non-graph finite-form reading retains the existing lookup behavior.
    assert resolve_pair_lemma("φέρω", "verb", lookup, pos_map, {}, roots, unresolved) == "φέρνω"


def test_tagged_head_metadata_does_not_expand_candidate_cleanup_scope():
    pos_map = {"πρώτος": {"VERB": "πρώτος"}, "δεύτερος": {"VERB": "δεύτερος"}}
    unresolved = apply_participle_references(
        pos_map, {"πρώτος": {"δεύτερος"}}, {"δεύτερος"}, {"πρώτος", "δεύτερος"})
    assert unresolved == {"πρώτος"}
    assert pos_map["δεύτερος"]["VERB"] == "δεύτερος"


def test_unresolved_graph_root_cannot_return_through_flat_alias():
    lookup = {"τύπος": ("άγνωστος", 1), "άγνωστος": ("άγνωστος", 3)}
    pair = {"form": "άλλος", "lemma": "τύπος", "pos": "verb", "tags": []}
    assert clean_training_pairs([pair], lookup, True, {}, {"άγνωστος": {"verb"}},
                                {}, {"άγνωστος"}) == []


def test_a_sense_tagged_participle_cannot_be_a_finite_graph_endpoint(tmp_path):
    dump = _write(tmp_path, [
        {"word": "σκουντηγμένος", "pos": "verb",
         "senses": [{"glosses": ["nudged, pushed"],
                     "tags": ["participle", "masculine"]}]},
        {"word": "σκουντημένος", "pos": "verb", "tags": ["participle"],
         "form_of": [{"word": "σκουντηγμένος"}],
         "senses": [{"glosses": ["που έχει σκουντηθεί"]}]},
    ])
    refs, heads, participles = {}, set(), set()
    _, _, _, pos_map, *_ = extract_pairs(
        dump, "el-el", participle_refs=refs, verb_heads=heads,
        participle_heads=participles)
    assert participles == {"σκουντηγμένος", "σκουντημένος"}
    assert apply_participle_references(pos_map, refs, heads, participles) == {
        "σκουντημένος"}


def test_unresolved_participle_cannot_remove_its_independent_finite_verbs():
    pos_map = {"ανεμοδέρνω": {"VERB": "ανεμοδαρμένος"},
               "ανεμοδέρνομαι": {"VERB": "ανεμοδαρμένος"}}
    refs = {"ανεμοδαρμένος": {"ανεμοδέρνω", "ανεμοδέρνομαι"}}
    heads = {"ανεμοδέρνω", "ανεμοδέρνομαι"}
    unresolved = apply_participle_references(pos_map, refs, heads)
    assert unresolved == {"ανεμοδαρμένος"}
    roots = {}
    lookup = {word: ("ανεμοδαρμένος", 1) for word in heads | unresolved}
    preserve_finite_verb_heads(pos_map, lookup, heads, roots, unresolved)
    pairs = [{"form": word, "lemma": word, "pos": "verb", "tags": []}
             for word in heads | unresolved]
    clean = clean_training_pairs(pairs, lookup, True, pos_map, {}, roots, unresolved)
    assert {(r["form"], r["lemma"]) for r in clean} == {(w, w) for w in heads}


def test_nominal_top_reference_cannot_create_a_second_verb_endpoint():
    pos_map = {"λόγιο": {"VERB": "συμβουλεύω"}}
    refs = {"χρισμένος": {"χρίω", "λόγιο"}}
    unresolved = apply_participle_references(
        pos_map, refs, {"χρίω", "συμβουλεύω"}, {"χρισμένος"},
        {"λόγιο": {"adj"}, "χρίω": {"verb"}, "συμβουλεύω": {"verb"}})
    assert unresolved == set()
    assert pos_map["χρισμένος"]["VERB"] == "χρίω"


@pytest.mark.parametrize("heads", [["γράφω", "λέω"], ["λέω", "γράφω"]])
def test_finite_head_preservation_uses_one_snapshot(heads):
    pos_map = {"γράφω": {"VERB": "λέω"}, "λέω": {"VERB": "τύπος"}}
    roots = {}
    preserve_finite_verb_heads(pos_map, {}, heads, roots, {"τύπος"})
    assert roots == {"γράφω": "γράφω", "λέω": "λέω"}
    assert pos_map == {word: {"VERB": word} for word in heads}


def test_main_keeps_graph_verb_separate_from_finite_form_homograph(tmp_path, monkeypatch):
    import build_data
    entries = [{"word": "φέρω", "pos": "verb", "senses": [{"glosses": ["carry"]}]},
               {"word": "φέρω", "pos": "verb", "senses": [{"form_of": [{"word": "φέρνω"}]}]},
               {"word": "φέρνω", "pos": "verb", "senses": [{"glosses": ["bring"]}]},
               {"word": "φέρων", "pos": "verb", "tags": ["participle"],
                "form_of": [{"word": "φέρω"}], "forms": [{"form": "φέροντα"}]},
               {"word": "φορτίο", "pos": "noun", "forms": [{"form": "φέρων"}]}]
    dump = _write(tmp_path, entries)
    out = tmp_path / "out"
    monkeypatch.setattr(build_data, "DATA_DIR", out)
    monkeypatch.setattr(build_data, "DUMPS", {"el": {"el": dump.name}})
    monkeypatch.setattr("sys.argv", ["build_data.py", "--lang", "el", "--kaikki", str(tmp_path)])
    build_data.main()
    pairs = json.loads((out / "mg_pairs.json").read_text())
    assert any(p["form"] == "φέροντα" and p["lemma"] == "φέρω" and p["pos"] == "verb" for p in pairs)
    pos = json.loads((out / "mg_pos_lookup.json").read_text())
    assert pos["φέρων"]["VERB"] == "φέρω"


def test_main_keeps_noun_case_cells_after_lemma_resolution(tmp_path, monkeypatch):
    import build_data
    entries = [
        {"word": "καθηγητής", "pos": "noun", "senses": [{"glosses": ["teacher"]}],
         "forms": [{"form": "καθηγητή", "tags": [case, "singular"]}
                   for case in ("genitive", "accusative", "vocative")]},
        {"word": "ερώτηση", "pos": "noun", "senses": [{"glosses": ["question"]}],
         "forms": [{"form": "ερωτήσεις", "tags": ["nominative", "plural"]}]},
        {"word": "ερωτώ", "pos": "verb", "senses": [{"glosses": ["ask"]}]},
        {"word": "ερωτήσεις", "pos": "verb", "forms": [{"form": "ερωτήσεις"}],
         "senses": [{"form_of": [{"word": "ερωτώ"}]}]},
        {"word": "Κυριακή", "pos": "noun", "senses": [{"glosses": ["Sunday"]}]},
        {"word": "Κυριακός", "pos": "name", "senses": [{"glosses": ["male name"]}]},
        {"word": "κυριακή", "pos": "adj", "forms": [{"form": "κυριακή"}],
         "senses": [{"form_of": [{"word": "κυριακός"}]}]},
    ]
    dump = _write(tmp_path, entries)
    (tmp_path / "el.jsonl").write_text(json.dumps({
        "word": "Κυριακή", "pos": "name", "senses": [{"glosses": ["Sunday"]}]
    }, ensure_ascii=False) + "\n")
    out = tmp_path / "out"
    monkeypatch.setattr(build_data, "DATA_DIR", out)
    monkeypatch.setattr(build_data, "DUMPS", {"el": {"en": dump.name, "el": "el.jsonl"}})
    monkeypatch.setattr("sys.argv", ["build_data.py", "--lang", "el", "--kaikki", str(tmp_path)])
    build_data.main()
    pairs = json.loads((out / "mg_pairs.json").read_text())
    cells = [p for p in pairs if p["form"] == "καθηγητή"]
    assert len(cells) == 3
    assert {tuple(p["tags"]) for p in cells} == {
        (case, "singular") for case in ("genitive", "accusative", "vocative")}
    assert {p["lemma"] for p in pairs if p["form"] == "ερωτήσεις" and p["pos"] == "verb"} == {"ερωτώ"}
    lookup = json.loads((out / "mg_lookup.json").read_text())
    assert lookup["κυριακή"] == "Κυριακή"
    assert [p for p in pairs if p["form"] == "κυριακή" and p["pos"] == "adj"]
    pos_map = json.loads((out / "mg_pos_lookup.json").read_text())
    assert not pos_map.get("κυριακή", {}).get("ADJ")


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
    pairs, lookup, _, pos_map, *_ = extract_pairs(dump, "el-en")
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
    # A word of more than one syllable keeps its accent (καλός names no
    # καλος), and a form_of reference of a sense tagged with a case names
    # the word the page is a form of, not a spelling (σοφίᾳ, the dative of
    # σοφία).
    assert spelling_target(
        "καλός", [{"glosses": ["πολυτονική γραφή του καλος"]}]) is None
    assert spelling_target(
        "σοφίᾳ", [{"glosses": ["dative singular of σοφία (sofía)"],
                   "tags": ["dative", "feminine", "form-of", "singular"],
                   "form_of": [{"word": "σοφία"}]}]) is None
    assert spelling_target(
        "γειά", [{"glosses": ["obsolete spelling of γεια (geia)"],
                  "alt_of": [{"word": "γεια"}]}]) == "γεια"


def test_vowel_groups():
    # A monosyllable has one run of vowels, synizesis included; a vowel
    # with a diaeresis starts a run of its own.
    assert [vowel_groups(w) for w in ("γιος", "γεια", "Χουαν", "καλός",
                                      "χαϊ", "ρολόι")] == [1, 1, 1, 2, 2, 2]


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


def test_an_abbreviation_does_not_take_a_words_key(tmp_path):
    # The English Wiktionary gives δις and δ as abbreviations of δεσποινίς.
    # δις has a page of its own ("twice"), and the Greek Wiktionary's δ is
    # the letter: neither becomes a form of δεσποινίς. κα, whose only
    # other page is an abbreviation too, and στο, a contraction the English
    # Wiktionary tags as an abbreviation, still follow theirs.
    (tmp_path / "en").mkdir()
    (tmp_path / "el").mkdir()
    en = _write(tmp_path / "en", [
        {"word": "δεσποινίς", "pos": "noun",
         "senses": [{"glosses": ["miss, young unmarried woman"]}]},
        {"word": "δις", "pos": "adv", "senses": [{"glosses": ["twice"]}]},
        {"word": "δις", "pos": "noun",
         "senses": [{"glosses": ["abbreviation of δεσποινίς: Miss"],
                     "tags": ["abbreviation", "alt-of"],
                     "alt_of": [{"word": "δεσποινίς"}]}]},
        {"word": "δ", "pos": "character",
         "senses": [{"glosses": ["the fourth letter of the Greek alphabet"],
                     "tags": ["letter"]}]},
        {"word": "δ", "pos": "noun",
         "senses": [{"glosses": ["abbreviation of δεσποινίς: Miss"],
                     "tags": ["abbreviation", "alt-of"],
                     "alt_of": [{"word": "δεσποινίς"}]}]},
        {"word": "κα", "pos": "noun",
         "senses": [{"glosses": ["abbreviation of κυρία: Mrs"],
                     "tags": ["abbreviation", "alt-of"],
                     "alt_of": [{"word": "κυρία"}]}]},
        {"word": "κυρία", "pos": "noun", "senses": [{"glosses": ["lady"]}]},
        {"word": "στο", "pos": "contraction",
         "senses": [{"glosses": ["contraction of σε + το"],
                     "tags": ["abbreviation", "alt-of", "contraction"],
                     "alt_of": [{"word": "σε"}]}]},
        {"word": "σε", "pos": "prep", "senses": [{"glosses": ["in, to"]}]},
    ])
    el = _write(tmp_path / "el", [
        {"word": "δ", "pos": "symbol",
         "senses": [{"glosses": ["δέλτα: το τέταρτο γράμμα"]}]},
        {"word": "κα", "pos": "abbrev",
         "senses": [{"glosses": ["κυρία (με κεφαλαίο, η επίσημη γραφή)"]}]},
        {"word": "Μαρ", "pos": "name",
         "senses": [{"glosses": ["επώνυμο (ανδρικό ή γυναικείο)"]}]},
    ])
    _, en_lookup, en_headwords, _, en_targets, en_abbr = extract_pairs(
        en, "el-en")
    assert en_lookup["δις"][0] == "δις"
    assert en_lookup["κα"][0] == "κυρία" and en_lookup["στο"][0] == "σε"
    assert {"δις", "δ", "κα", "στο"} <= en_abbr.keys
    assert "δις" in en_abbr.words and "κα" not in en_abbr.words
    _, el_lookup, _, _, el_targets, el_abbr = extract_pairs(el, "el-el")
    assert {"δ"} <= el_abbr.words and "κα" not in el_abbr.words
    # A capitalized page claims no key against an abbreviation: the
    # surname Μαρ leaves μαρ to the English Wiktionary's "Μαρ.", March.
    assert not {"Μαρ", "μαρ"} & el_abbr.words
    # Merged as main does: the Greek Wiktionary's keys first, then the
    # abbreviation targets of the words of either dump dropped, then the
    # cross-source pass.
    merged = {}
    for lookup in (el_lookup, en_lookup):
        for k, (lemma, _) in lookup.items():
            merged.setdefault(k, (lemma, 1))
    # δις lost its target in the English dump already, δ (whose English
    # page, "the fourth letter of", reads as a form-of page) only to the
    # Greek dump's letter.
    assert "δις" not in en_targets and en_targets["δ"] == "δεσποινίς"
    words = en_abbr.words | el_abbr.words
    assert drop_abbreviation_targets(en_targets, en_abbr, words) == 1
    assert "δ" not in en_targets
    resolve_cross_source(merged, en_targets, el_targets, en_headwords)
    assert merged["δ"][0] == "δ" and merged["δις"][0] == "δις"
    assert merged["κα"][0] == "κυρία" and merged["στο"][0] == "σε"
    # The Ancient Greek dumps are read as before.
    *_, grc_abbr = extract_pairs(en, "grc-en")
    assert grc_abbr == Abbreviations(set(), set())



def test_an_abbreviations_alternatives_are_forms_of_its_word(tmp_path):
    # The English Wiktionary's δ and δις, both "abbreviation of δεσποινίς",
    # list each other as alternatives. Those are forms of δεσποινίς, not of
    # the page's headword: filed under δ, which keeps its key as the letter,
    # δις would be a noun of lemma δ.
    dump = _write(tmp_path, [
        {"word": "δεσποινίς", "pos": "noun",
         "senses": [{"glosses": ["miss, young unmarried woman"]}]},
        {"word": "δ", "pos": "noun",
         "forms": [{"form": "d", "tags": ["romanization"]},
                   {"form": "δις", "tags": ["alternative"]}],
         "senses": [{"glosses": ["abbreviation of δεσποινίς: Miss"],
                     "tags": ["abbreviation", "alt-of"],
                     "alt_of": [{"word": "δεσποινίς"}]}]},
        {"word": "δις", "pos": "adv", "senses": [{"glosses": ["twice"]}]},
        # An abbreviation of a phrase keeps its own forms (the plural of
        # χωριάτικη, "χωριάτικη σαλάτα").
        {"word": "χωριάτικη", "pos": "noun",
         "forms": [{"form": "χωριάτικες", "tags": ["plural"]}],
         "senses": [{"glosses": ["abbreviation of χωριάτικη σαλάτα"],
                     "tags": ["abbreviation", "alt-of"],
                     "alt_of": [{"word": "χωριάτικη σαλάτα"}]}]},
        # So does a page with a sense of its own beside the abbreviation.
        {"word": "κ", "pos": "noun",
         "forms": [{"form": "κκ", "tags": ["alternative"]}],
         "senses": [{"glosses": ["abbreviation of κύριος: Mr"],
                     "tags": ["abbreviation", "alt-of"],
                     "alt_of": [{"word": "κύριος"}]},
                    {"glosses": ["kappa"]}]},
    ])
    pairs, lookup, _, pos_map, *_ = extract_pairs(dump, "el-en")
    by_form = {}
    for p in pairs:
        by_form.setdefault(p["form"], set()).add(p["lemma"])
    assert by_form["δις"] == {"δεσποινίς"}
    assert pos_map["δις"]["NOUN"] == "δεσποινίς"
    # The adverb's own page keeps its key.
    assert lookup["δις"][0] == "δις"
    assert by_form["χωριάτικες"] == {"χωριάτικη"}
    assert by_form["κκ"] == {"κ"}
    # The Ancient Greek dumps are read as before.
    pairs, *_ = extract_pairs(dump, "grc-en")
    assert ("δις", "δ") in {(p["form"], p["lemma"]) for p in pairs}
