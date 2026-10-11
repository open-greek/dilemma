"""Focused tests for Kaikki build-time morphology extraction."""

from __future__ import annotations

import json

import pytest

from build_data import (Abbreviations, _parse_verb_tense, apply_participle_references,
                        clean_training_pairs, deduplicate_pairs,
                        drop_abbreviation_targets, extract_pairs,
                        include_participle_endpoints,
                        merge_pos_maps, merge_source_lookups, resolve_cross_source,
                        preserve_finite_verb_heads, resolve_pair_lemma,
                        spelling_target, standalone_participle_inventory,
                        vowel_groups, noun_head_inventory,
                        validate_noun_head_inventory, training_pair_headword_errors,
                        validate_closed_class_head_inventory, closed_class_head_inventory,
                        closed_class_head_evidence, is_independent_closed_class_entry)



@pytest.mark.parametrize("reverse", [False, True])
def test_noun_form_page_keeps_its_reference_without_borrowing_an_adjective_table(tmp_path, reverse):
    rows = [{"form": "κρατών", "tags": ["nominative", "singular", "masculine"]},
            {"form": "κρατούν", "tags": ["nominative", "singular", "neuter"]}]
    entries = [
        {"word": "κρατών", "pos": "noun", "tags": ["form-of"], "forms": rows,
         "senses": [{"glosses": ["genitive plural of κράτος"],
                     "form_of": [{"word": "κράτος"}], "tags": ["genitive", "plural"]}]},
        {"word": "κρατών", "pos": "adj", "forms": rows,
         "senses": [{"glosses": ["prevailing, dominant"]}]},
    ]
    pairs, _, _, typed, *_ = extract_pairs(
        _write(tmp_path, entries[::-1] if reverse else entries), "el-el")
    nominal = [p for p in pairs if p["pos"] == "noun"]
    assert nominal == [{"form": "κρατών", "lemma": "κράτος", "pos": "noun",
                        "tags": ["form-of", "genitive", "plural"]}]
    assert any(p["form"] == "κρατούν" and p["pos"] == "adj" for p in pairs)
    assert "NOUN" not in typed["κρατούν"]


@pytest.mark.parametrize("kind", ["plural_table", "alternative", "mixed_alternative", "mixed_definition",
                                  "mixed_same_sense", "ordinary_of_gloss", "nominal_head"])
def test_noun_table_guard_retains_supported_tables(tmp_path, kind):
    entry = {"word": "νιάτα", "pos": "noun", "senses": [
        {"form_of": [{"word": "νιάτο"}], "tags": ["nominative", "plural"]}],
        "forms": [{"form": "νιάτα", "tags": ["nominative", "plural"]},
                  {"form": "νιάτων", "tags": ["genitive", "plural"]}]}
    if kind == "alternative": entry["senses"] = [{"alt_of": [{"word": "νιάτο"}]}]
    elif kind == "mixed_alternative":
        entry["forms"][0]["tags"] = ["nominative", "singular"]
        entry["senses"].append({"alt_of": [{"word": "νιάτο"}]})
    elif kind in {"mixed_same_sense", "ordinary_of_gloss"}:
        entry["forms"][0]["tags"] = ["nominative", "singular"]
        entry["senses"][0]["glosses"] = ["nominative plural of νιάτο",
            "youth" if kind == "mixed_same_sense" else "a type of vessel"]
    elif kind == "mixed_definition":
        entry["forms"][0]["tags"] = ["nominative", "singular"]
        entry["senses"].append({"glosses": ["youth"]})
    elif kind == "nominal_head": entry["senses"] = [{"glosses": ["youth"]}]
    pairs, *_ = extract_pairs(_write(tmp_path, [entry]), "el-el")
    assert any(p["form"] == "νιάτων" for p in pairs)



@pytest.mark.parametrize("lexical", [False, True])
def test_noun_table_guard_recognizes_exact_gender_prefixed_case_description(tmp_path, lexical):
    entry = {"word": "μανικιουρίστας", "pos": "noun", "tags": ["form-of"],
        "senses": [{"form_of": [{"word": "μανικιουρίστα"}],
                    "glosses": ["το θηλυκό γενική ενικού του μανικιουρίστα"],
                    "tags": ["genitive", "singular"]}],
        "forms": [{"form": "μανικιουρίστας", "tags": ["nominative", "singular"]},
                  {"form": "μανικιουρίστες", "tags": ["nominative", "plural"]}]}
    if lexical: entry["senses"][0]["glosses"].append("a manicurist")
    pairs, *_ = extract_pairs(_write(tmp_path, [entry]), "el-el")
    assert any(p["form"] == "μανικιουρίστες" for p in pairs) == lexical
    assert any(p["form"] == "μανικιουρίστας" and "genitive" in p["tags"] for p in pairs)

def test_noun_table_guard_does_not_infer_a_form_page_from_incomplete_tags(tmp_path):
    entry = {"word": "τύπου", "pos": "noun", "senses": [
        {"form_of": [{"word": "τύπος"}], "tags": ["genitive"]}],
        "forms": [{"form": "τύποι", "tags": ["nominative", "plural"]}]}
    pairs, *_ = extract_pairs(_write(tmp_path, [entry]), "el-el")
    assert any(p["form"] == "τύποι" for p in pairs)


def test_noun_form_table_guard_is_modern_and_noun_only(tmp_path):
    entry = {"word": "κρατών", "pos": "noun", "senses": [
        {"form_of": [{"word": "κράτος"}], "tags": ["genitive", "plural"]}],
        "forms": [{"form": "κρατούν", "tags": ["nominative", "singular"]}]}
    ancient, *_ = extract_pairs(_write(tmp_path, [entry]), "grc-el")
    assert any(p["form"] == "κρατούν" for p in ancient)
    entry["pos"] = "adj"
    adjectival, *_ = extract_pairs(_write(tmp_path, [entry]), "el-el")
    assert any(p["form"] == "κρατούν" for p in adjectival)

def test_modern_form_reference_preserves_its_own_dative_tags(tmp_path):
    dump = _write(tmp_path, [{
        "word": "πάση", "pos": "det",
        "forms": [{"form": "pási", "tags": ["romanization"]}],
        "senses": [{"glosses": ["feminine dative singular of πας (pas)"],
                    "tags": ["archaic", "dative", "feminine", "form-of", "rare", "singular"],
                    "form_of": [{"word": "πας"}]}],
    }])
    pairs, lookup, *_ = extract_pairs(dump, "el-en")
    pair, = [p for p in pairs if p["form"] == "πάση"]
    assert pair == {"form": "πάση", "lemma": "πας", "pos": "det",
                    "tags": ["archaic", "dative", "feminine", "form-of", "rare", "singular"]}
    # Tags do not change the existing flat closed-class/homograph behavior.
    assert lookup["πάση"][0] == "πας"
    flat = {"πας": ("πηγαίνω", 1), "πηγαίνω": ("πηγαίνω", 3)}
    cleaned, = clean_training_pairs([pair], flat, preserve_cells=True)
    assert cleaned["lemma"] == "πηγαίνω" and "dative" in cleaned["tags"]


@pytest.mark.parametrize("relation", ["form_of", "alt_of"])
def test_modern_reference_cells_stay_per_sense_and_target(tmp_path, relation):
    senses = [
        {relation: [{"word": "α"}], "tags": ["genitive", "singular"]},
        {relation: [{"word": "α"}], "tags": ["accusative", "plural"]},
        {relation: [{"word": "β"}], "tags": ["vocative", "singular"]},
    ]
    expected = {
        ("α", frozenset({relation.replace("_", "-"), "archaic", "genitive", "singular"})),
        ("α", frozenset({relation.replace("_", "-"), "archaic", "accusative", "plural"})),
        ("β", frozenset({relation.replace("_", "-"), "archaic", "vocative", "singular"})),
    }
    for ordered in (senses, list(reversed(senses))):
        dump = _write(tmp_path, [{"word": "τύπος", "pos": "noun", "tags": ["archaic"],
                                 "senses": ordered}])
        pairs, *_ = extract_pairs(dump, "el-en")
        assert {(p["lemma"], frozenset(p["tags"])) for p in pairs
                if relation.replace("_", "-") in p["tags"]} == expected


def test_reference_metadata_keeps_ancient_pairs_and_untagged_modern_rows_stable(tmp_path):
    dump = _write(tmp_path, [{"word": "τύπος", "pos": "noun", "tags": ["archaic"],
                             "senses": [{"form_of": [{"word": "λέξη"}],
                                         "tags": ["dative", "singular"]}]}])
    pairs, *_ = extract_pairs(dump, "grc-en")
    assert next(p for p in pairs if p["lemma"] == "λέξη")["tags"] == ["form-of"]
    dump = _write(tmp_path, [{"word": "τύπος", "pos": "noun",
                             "senses": [{"form_of": [{"word": "λέξη"}]}]}])
    pairs, *_ = extract_pairs(dump, "el-en")
    assert next(p for p in pairs if p["lemma"] == "λέξη")["tags"] == ["form-of"]


@pytest.mark.parametrize("reverse", [False, True])
def test_independent_determiner_and_pronoun_heads_keep_typed_ownership(tmp_path, reverse):
    entries = [
        {"word": "πας", "pos": "det", "senses": [{"glosses": ["every, any"]}],
         "forms": [{"form": "πας", "tags": ["nominative", "singular"]}]},
        {"word": "πας", "pos": "pron", "senses": [{"glosses": ["κάθε, όλος"]}]},
        {"word": "πας", "pos": "verb",
         "senses": [{"form_of": [{"word": "πηγαίνω"}],
                     "tags": ["present", "second-person", "singular"]}]},
        {"word": "πάση", "pos": "det",
         "senses": [{"form_of": [{"word": "πας"}], "tags": ["dative", "singular"]}]},
    ]
    heads = {}
    pairs, _, _, pos_map, *_ = extract_pairs(_write(tmp_path, entries[::-1] if reverse else entries),
                                            "el-en", closed_class_heads=heads)
    assert heads == {"det": {"πας"}, "pron": {"πας"}}
    flat = {"πας": ("πηγαίνω", 1), "πάση": ("πηγαίνω", 1), "πηγαίνω": ("πηγαίνω", 3)}
    cleaned = clean_training_pairs(pairs, flat, preserve_cells=True, pos_map=pos_map,
                                   lemma_pos={"πας": {"det", "pron", "verb"}, "πηγαίνω": {"verb"}},
                                   closed_class_heads=heads)
    assert {p["lemma"] for p in cleaned if p["pos"] in {"det", "pron"}} == {"πας"}
    assert {p["lemma"] for p in cleaned if p["pos"] == "verb"} == {"πηγαίνω"}
    assert "dative" in next(p for p in cleaned if p["form"] == "πάση")["tags"]


def test_closed_class_ownership_requires_positive_same_pos_head(tmp_path):
    heads = {}
    entries = [{"word": "τύπος", "pos": "det",
                "senses": [{"glosses": ["nominative of άλλος"],
                            "form_of": [{"word": "άλλος"}]}]},
               {"word": "γνήσιος", "pos": "pron", "senses": [{"tags": ["no-gloss"]}]}]
    extract_pairs(_write(tmp_path, entries), "el-en", closed_class_heads=heads)
    assert heads == {}
    flat = {"τύπος": ("ρήμα", 1), "άλλος": ("άλλος", 3)}
    pos = {"τύπος": {"DET": "άλλος"}}
    assert resolve_pair_lemma("τύπος", "det", flat, pos,
                              closed_class_heads={"pron": {"άλλος"}}) == "ρήμα"
    assert resolve_pair_lemma("τύπος", "det", flat, pos,
                              closed_class_heads={"det": {"άλλος"}}) == "άλλος"


def test_closed_class_integrity_requires_exact_modern_head_and_typed_self_target():
    pairs = [{"form": "πάση", "lemma": "πας", "pos": "det", "tags": ["dative"]}]
    flat = {"πας": "πηγαίνω"}
    proof = closed_class_head_inventory([{"lemma": "πας", **closed_class_head_evidence(
        {"word": "πας", "pos": "det", "senses": [{"glosses": ["every, any"]}]}, "en")}],
        {"πας": ("πηγαίνω", 1)}, {"en": "a" * 64})
    good = validate_closed_class_head_inventory(proof, {"πας": {"DET": "πας"}})
    assert training_pair_headword_errors(pairs, flat, closed_class_heads=good)[0] == 0
    for table, inventory in [
        ({}, proof), ({"πας": {"VERB": "πας"}}, proof),
        ({"πας": {"DET": "πηγαίνω"}}, proof),
        ({"Πας": {"DET": "Πας"}}, proof),
    ]:
        with pytest.raises(ValueError):
            validate_closed_class_head_inventory(inventory, table)
    assert training_pair_headword_errors([dict(pairs[0], pos="verb")], flat,
                                         closed_class_heads=good)[0] == 1


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("gloss", ["feminine genitive singular of αυτός", "genitive singular of αυτός",
                                    "γενική ενικού του αυτός", "alternative spelling of αυτός"])
def test_closed_class_positive_sense_gate_rejects_unstructured_forms(tmp_path, reverse, gloss):
    senses = [{"tags": ["no-gloss"]}, {"glosses": [gloss]}]
    entry = {"word": "αυτής", "pos": "pron", "senses": senses[::-1] if reverse else senses}
    assert not is_independent_closed_class_entry(entry, "en")
    heads = {}
    extract_pairs(_write(tmp_path, [entry]), "el-en", closed_class_heads=heads)
    assert heads == {}


@pytest.mark.parametrize("pos,template", [("pron", "pronoun form"), ("det", "determiner form")])
@pytest.mark.parametrize("gloss", [
    "his (3rd person masculine singular, genitive)",
    "its (3rd person neuter singular, genitive)",
    "they (3rd person neuter plural, nominative)",
    "them (3rd person feminine plural, accusative)",
    "you (2nd person singular, accusative)",
])
def test_declared_closed_class_forms_do_not_become_independent_heads(tmp_path, pos, template, gloss):
    entry = {"word": "αυτού", "pos": pos,
             "head_templates": [{"name": "head", "args": {"1": "el", "2": template}}],
             "senses": [{"glosses": [gloss]}]}
    assert not is_independent_closed_class_entry(entry, "en")
    heads = {}
    extract_pairs(_write(tmp_path, [entry]), "el-en", closed_class_heads=heads)
    assert heads == {}
    proof = closed_class_head_inventory(
        [{"lemma": "αυτού", **closed_class_head_evidence(entry, "en")}],
        {"αυτού": ("αυτός", 1)}, {"en": "a" * 64})
    with pytest.raises(ValueError, match="independent"):
        validate_closed_class_head_inventory(proof, {"αυτού": {pos.upper(): "αυτού"}})


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("word,gloss", [("μας", "our (belonging to us)"),
                                      ("σας", "your (belonging to you)")])
def test_declared_form_keeps_a_separate_lexical_possessive_sense(tmp_path, reverse, word, gloss):
    senses = [{"glosses": ["our (1st person plural, genitive)"]}, {"glosses": [gloss]}]
    entry = {"word": word, "pos": "pron",
             "head_templates": [{"name": "head", "args": {"1": "el", "2": "pronoun form"}}],
             "senses": senses[::-1] if reverse else senses}
    assert is_independent_closed_class_entry(entry, "en")
    heads = {}
    extract_pairs(_write(tmp_path, [entry]), "el-en", closed_class_heads=heads)
    assert heads == {"pron": {word}}
    evidence = closed_class_head_evidence(entry, "en")
    assert evidence["head_templates"] == entry["head_templates"]
    proof = closed_class_head_inventory([{ "lemma": word, **evidence}],
                                       {word: ("εγώ", 1)}, {"en": "a" * 64})
    assert validate_closed_class_head_inventory(proof, {word: {"PRON": word}})["pron"] == {word}


@pytest.mark.parametrize("templates", [None, {}, [{"name": "head", "args": []}],
                                      [{"name": "head", "args": {"1": 1}}]])
def test_closed_class_proof_rejects_malformed_head_templates(templates):
    proof = closed_class_head_inventory([{"lemma": "πας", **closed_class_head_evidence(
        {"word": "πας", "pos": "det", "senses": [{"glosses": ["every, any"]}]}, "en")}],
        {"πας": ("πηγαίνω", 1)}, {"en": "a" * 64})
    proof["heads"][0]["evidence"][0]["head_templates"] = templates
    with pytest.raises(ValueError, match="head templates"):
        validate_closed_class_head_inventory(proof, {"πας": {"DET": "πας"}})


def test_pronoun_form_proof_does_not_override_structured_parent_or_adverb(tmp_path):
    entries = [
        {"word": "αυτού", "pos": "pron",
         "head_templates": [{"name": "head", "args": {"1": "el", "2": "pronoun form"}}],
         "senses": [{"glosses": ["his (3rd person masculine singular, genitive)"]}]},
        {"word": "αυτού", "pos": "pron", "senses": [{"glosses": ["γενική ενικού του αυτός"],
            "form_of": [{"word": "αυτός"}], "tags": ["genitive", "singular"]}]},
        {"word": "αυτού", "pos": "adv", "senses": [{"glosses": ["there"]}]},
        {"word": "αυτός", "pos": "pron", "senses": [{"glosses": ["he, him"]}]},
    ]
    _, adverb_lookup, *_ = extract_pairs(_write(tmp_path, [entries[2]]), "el-en")
    assert adverb_lookup["αυτού"][0] == "αυτού"  # Exact lexical source evidence remains.
    for ordered in (entries, entries[::-1]):
        heads = {}
        pairs, _, _, typed, *_ = extract_pairs(_write(tmp_path, ordered), "el-en",
                                             closed_class_heads=heads)
        assert heads == {"pron": {"αυτός"}}
        assert typed["αυτού"]["PRON"] == "αυτός"
        cleaned = clean_training_pairs(pairs, {"αυτού": ("αυτός", 1), "αυτός": ("αυτός", 3)},
            preserve_cells=True, pos_map=typed, closed_class_heads=heads)
        assert {p["lemma"] for p in cleaned if p["pos"] == "pron"} == {"αυτός"}


@pytest.mark.parametrize("reverse", [False, True])
def test_structured_pronoun_parent_precedes_an_unproven_weak_table_alias(tmp_path, reverse):
    from build_data import apply_closed_class_form_references
    entries = [
        {"word": "σε", "pos": "pron", "head_templates": [
            {"name": "head", "args": {"1": "el", "2": "pronoun form"}}],
         "senses": [{"glosses": ["you (2nd person singular, accusative)"]}]},
        {"word": "εσένα", "pos": "pron", "tags": ["form-of"], "senses": [
            {"glosses": ["γενική ενικού του εσύ"], "tags": ["genitive", "singular"],
             "form_of": [{"word": "εσύ"}]},
            {"glosses": ["αιτιατική ενικού του εσύ"], "tags": ["accusative", "singular"],
             "form_of": [{"word": "εσύ"}]}]},
        {"word": "εσύ", "pos": "pron", "senses": [{"glosses": ["you, thou"]}]},
    ]
    heads, refs = {}, {}
    pairs, _, _, typed, *_ = extract_pairs(
        _write(tmp_path, entries[::-1] if reverse else entries), "el-en",
        closed_class_heads=heads, closed_class_form_refs=refs)
    assert heads == {"pron": {"εσύ"}}
    assert refs == {("εσένα", "pron"): {"εσύ"}}
    typed["εσένα"] = {"PRON": "σε"}  # Competing form-table claim.
    typed["εσενα"] = {"PRON": "σε"}
    assert apply_closed_class_form_references(typed, refs, heads) == {
        ("εσένα", "pron"): "εσύ"}
    assert typed["εσένα"]["PRON"] == typed["εσενα"]["PRON"] == "εσύ"
    flat = {"εσένα": ("εσύ", 3), "εσύ": ("εσύ", 3), "σε": ("σε", 3)}
    clean = clean_training_pairs(pairs, flat, preserve_cells=True, pos_map=typed,
                                closed_class_heads=heads)
    assert {p["lemma"] for p in clean if p["form"] == "εσένα"} == {"εσύ"}


@pytest.mark.parametrize("pos", ["noun", "verb", "adv"])
def test_structured_closed_class_metadata_does_not_collect_other_parts_of_speech(tmp_path, pos):
    refs = {}
    extract_pairs(_write(tmp_path, [{"word": "τύπος", "pos": pos,
        "senses": [{"form_of": [{"word": "ρίζα"}]}]}]), "el-en",
        closed_class_form_refs=refs)
    assert refs == {}


@pytest.mark.parametrize("reverse", [False, True])
def test_named_weak_pronoun_keeps_its_typed_lexical_parent_without_head_proof(tmp_path, reverse):
    from build_data import apply_closed_class_form_references
    entries = [
        {"word": "τονε", "pos": "pron", "senses": [{"form_of": [{"word": "τον"}]}]},
        {"word": "τον", "pos": "pron", "tags": ["form-of"], "senses": [
            {"glosses": ["αδύνατος τύπος της προσωπικής αντωνυμίας γ' προσώπου: αυτόν"]}]},
        {"word": "αυτόν", "pos": "pron", "senses": [{"form_of": [{"word": "αυτός"}]}]},
        {"word": "αυτός", "pos": "pron", "senses": [{"glosses": ["he, himself"]}]},
    ]
    heads, refs = {}, {}
    pairs, _, _, pos_map, *_ = extract_pairs(_write(tmp_path, entries[::-1] if reverse else entries),
        "el-el", closed_class_heads=heads, closed_class_form_refs=refs)
    assert heads == {"pron": {"αυτός"}}
    assert refs == {("τον", "pron"): {"αυτόν"},
                    ("τονε", "pron"): {"τον"}, ("αυτόν", "pron"): {"αυτός"}}
    pos_map.setdefault("τον", {})["DET"] = "ο"
    assert apply_closed_class_form_references(pos_map, refs, heads) == {
        ("τον", "pron"): "αυτός", ("αυτόν", "pron"): "αυτός",
        ("τονε", "pron"): "αυτός"}
    assert pos_map["τον"] == {"PRON": "αυτός", "DET": "ο"}
    flat = {"τονε": ("τόνος", 3), "τον": ("τον", 3), "αυτόν": ("αυτός", 3),
            "αυτός": ("αυτός", 3), "τόνος": ("τόνος", 3), "ο": ("ο", 3)}
    cleaned = clean_training_pairs(pairs, flat, preserve_cells=True, pos_map=pos_map,
                                   closed_class_heads=heads)
    assert {p["lemma"] for p in cleaned} == {"αυτός"}
    assert flat["τονε"][0] == "τόνος"  # Unqualified homograph lookup is untouched.


@pytest.mark.parametrize("reverse", [False, True])
def test_closed_references_follow_own_chain_before_competing_tables(reverse):
    from build_data import apply_closed_class_form_references
    rows = [(('εσένανε', 'pron'), {'εσένα'}),
            (('εσένα', 'pron'), {'εσύ'}),
            (('κείνες', 'pron'), {'εκείνες'}),
            (('εκείνες', 'pron'), {'εκείνος'})]
    refs = dict(rows[::-1] if reverse else rows)
    table = {'εσένανε': {'PRON': 'εσένα'}, 'εσένα': {'PRON': 'σε'},
             'κείνες': {'PRON': 'κείνος'}, 'εκείνες': {'PRON': 'κείνος'}}
    heads = {'pron': {'εσύ', 'κείνος', 'εκείνος'}}
    selected = apply_closed_class_form_references(table, refs, heads)
    assert selected['εσένανε', 'pron'] == selected['εσένα', 'pron'] == 'εσύ'
    assert selected['κείνες', 'pron'] == selected['εκείνες', 'pron'] == 'εκείνος'
    assert table['εσένανε']['PRON'] == 'εσύ'
    assert table['κείνες']['PRON'] == 'εκείνος'


@pytest.mark.parametrize("reverse", [False, True])
def test_closed_reference_normalized_conflicts_keep_the_original_mapping(reverse):
    from build_data import apply_closed_class_form_references
    rows = [(('όποια', 'pron'), {'όποιος'}), (('οποία', 'pron'), {'οποίος'})]
    refs = dict(rows[::-1] if reverse else rows)
    table = {'οποια': {'PRON': 'οποίος'}, 'όποια': {'PRON': 'οποίος'},
             'οποία': {'PRON': 'όποιος'}}
    selected = apply_closed_class_form_references(
        table, refs, {'pron': {'όποιος', 'οποίος'}})
    assert selected == {('όποια', 'pron'): 'όποιος', ('οποία', 'pron'): 'οποίος'}
    assert table['όποια']['PRON'] == 'όποιος'
    assert table['οποία']['PRON'] == 'οποίος'
    assert table['οποια']['PRON'] == 'οποίος'


@pytest.mark.parametrize("reverse", [False, True])
def test_ambiguous_exact_reference_blocks_a_normalized_single_root_overwrite(reverse):
    from build_data import apply_closed_class_form_references
    rows = [(('όποια', 'pron'), {'όποιος', 'οποίος'}),
            (('οποία', 'pron'), {'οποίος'})]
    table = {'οποια': {'PRON': 'όποιος'}, 'όποια': {'PRON': 'όποιος'}}
    selected = apply_closed_class_form_references(
        table, dict(rows[::-1] if reverse else rows), {'pron': {'όποιος', 'οποίος'}})
    assert selected == {('οποία', 'pron'): 'οποίος'}
    assert table['οποια']['PRON'] == table['όποια']['PRON'] == 'όποιος'
    assert table['οποία']['PRON'] == 'οποίος'


@pytest.mark.parametrize("damage", ['missing', 'cycle', 'divergent', 'wrong_pos'])
def test_closed_own_reference_cannot_be_overruled_by_a_supported_table(damage):
    from build_data import apply_closed_class_form_references
    refs = {('τύπος', 'pron'): {'μορφή'}}
    table = {'τύπος': {'PRON': 'ρίζα'}, 'μορφή': {'PRON': 'ρίζα'}}
    heads = {'pron': {'ρίζα'}}
    if damage == 'missing': refs['μορφή', 'pron'] = {'ανύπαρκτο'}
    elif damage == 'cycle': refs['μορφή', 'pron'] = {'τύπος'}
    elif damage == 'divergent':
        refs['μορφή', 'pron'] = {'ρίζα', 'άλλος'}
        heads['pron'].add('άλλος')
    else:
        refs['μορφή', 'pron'] = {'άλλος'}
        heads['det'] = {'άλλος'}
    assert ('τύπος', 'pron') not in apply_closed_class_form_references(table, refs, heads)
    assert table['τύπος']['PRON'] == 'ρίζα'


def test_main_publishes_named_weak_pronoun_reference_without_form_head_proof(tmp_path, monkeypatch):
    import build_data
    source = tmp_path / "source"
    source.mkdir()
    en = [
        {"word": "τονε", "pos": "pron", "senses": [{"form_of": [{"word": "τον"}]}]},
        {"word": "τονε", "pos": "noun", "senses": [{"alt_of": [{"word": "τόνος"}]}]},
        {"word": "τόνος", "pos": "noun", "senses": [{"glosses": ["tone"]}]},
        {"word": "αυτόν", "pos": "pron", "senses": [{"form_of": [{"word": "αυτός"}]}]},
        {"word": "αυτός", "pos": "pron", "senses": [{"glosses": ["he, himself"]}]},
    ]
    el = [{"word": "τον", "pos": "pron", "tags": ["form-of"], "senses": [
        {"glosses": ["αδύνατος τύπος της προσωπικής αντωνυμίας γ' προσώπου: αυτόν"]}]}]
    for edition, entries in (("en", en), ("el", el)):
        (source / (edition + ".jsonl")).write_text("".join(
            json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8")
    output = tmp_path / "data"
    monkeypatch.setattr(build_data, "DATA_DIR", output)
    monkeypatch.setattr(build_data, "DUMPS", {"el": {"en": "en.jsonl", "el": "el.jsonl"}})
    monkeypatch.setattr(build_data.sys, "argv", ["build_data.py", "--lang", "el", "--kaikki", str(source)])
    build_data.main()
    typed = json.loads((output / "mg_pos_lookup.json").read_text())
    assert typed["τον"]["PRON"] == "αυτός"
    assert typed["τονε"]["PRON"] == "αυτός"
    assert typed["τονε"]["NOUN"] == "τόνος"
    pairs = json.loads((output / "mg_pairs.json").read_text())
    assert {p["lemma"] for p in pairs if p["form"] == "τονε" and p["pos"] == "pron"} == {"αυτός"}
    assert all(h["lemma"] != "τον" for h in json.loads(
        (output / "el_closed_class_heads.json").read_text())["heads"])


@pytest.mark.parametrize("reverse", [False, True])
def test_main_keeps_agreeing_closed_targets_when_scalar_fallback_differs(tmp_path, monkeypatch, reverse):
    import build_data
    source = tmp_path / "source"
    source.mkdir()
    entries = [
        {"word": form, "pos": pos, "senses": [{"form_of": [{"word": "ο"}]}]}
        for form in ("τη", "την") for pos in ("det", "pron")
    ] + [{"word": "ο", "pos": "article", "senses": [{"glosses": ["the"]}]},
         {"word": "συμφωνεί", "pos": "verb", "senses": [{"form_of": [{"word": "συμφωνώ"}]}]},
         {"word": "συμφωνώ", "pos": "verb", "senses": [{"glosses": ["agree"]}]}]
    (source / "en.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n"
                                              for e in (entries[::-1] if reverse else entries)))
    (source / "el.jsonl").write_text("")
    original = build_data.resolve_cross_source

    def preserve_scalar_forms(lookup, *args, **kwargs):
        result = original(lookup, *args, **kwargs)
        # The independently maintained scalar lookup is deliberately not
        # retargeted by a typed closed-class publication decision.
        for form in ("τη", "την"):
            lookup[form] = (form, 3)
        return result

    monkeypatch.setattr(build_data, "resolve_cross_source", preserve_scalar_forms)
    output = tmp_path / "data"
    monkeypatch.setattr(build_data, "DATA_DIR", output)
    monkeypatch.setattr(build_data, "DUMPS", {"el": {"en": "en.jsonl", "el": "el.jsonl"}})
    monkeypatch.setattr(build_data.sys, "argv", ["build_data.py", "--lang", "el", "--kaikki", str(source)])
    build_data.main()
    typed = json.loads((output / "mg_pos_lookup.json").read_text())
    flat = json.loads((output / "mg_lookup.json").read_text())
    for form in ("τη", "την"):
        assert typed[form] == {"DET": "ο", "PRON": "ο"}
        assert flat[form] == form
    assert "ο" not in typed  # Agreement with the scalar fallback needs no field.
    assert "συμφωνεί" not in typed  # The publication exception is closed-class only.


@pytest.mark.parametrize("damage", ["missing", "cycle", "wrong_pos", "divergent"])
def test_named_weak_pronoun_requires_one_supported_same_pos_root(damage):
    from build_data import apply_closed_class_form_references
    refs = {("τον", "pron"): {"αυτόν"}}
    table = {"τον": {"PRON": "τον", "DET": "ο"}, "αυτόν": {"PRON": "αυτός"}}
    heads = {"pron": {"αυτός"}}
    if damage == "missing": table.pop("αυτόν")
    elif damage == "cycle": table["αυτόν"]["PRON"] = "τον"
    elif damage == "wrong_pos": heads = {"det": {"αυτός"}}
    elif damage == "divergent":
        refs["τον", "pron"].add("άλλον")
        table["άλλον"] = {"PRON": "άλλος"}
        heads["pron"].add("άλλος")
    assert apply_closed_class_form_references(table, refs, heads) == {}
    assert table["τον"] == {"PRON": "τον", "DET": "ο"}


@pytest.mark.parametrize("damage", ["wrong_source", "wrong_pos", "no_form_tag", "prose", "related"])
def test_named_weak_pronoun_reference_requires_the_exact_form_description(damage):
    from build_data import _named_weak_pronoun_reference
    entry = {"word": "τον", "pos": "pron", "tags": ["form-of"]}
    sense = {"glosses": ["αδύνατος τύπος της προσωπικής αντωνυμίας γ' προσώπου: αυτόν"]}
    source = "el"
    if damage == "wrong_source": source = "en"
    elif damage == "wrong_pos": entry["pos"] = "det"
    elif damage == "no_form_tag": entry["tags"] = []
    elif damage == "prose": sense["glosses"][0] += " σε άλλη φράση"
    elif damage == "related": sense = {"glosses": ["personal pronoun"], "related": [{"word": "αυτόν"}]}
    assert _named_weak_pronoun_reference(entry, sense, source) is None


@pytest.mark.parametrize("change", ["nominal", "wrong_pos", "form_gloss", "form_ref", "alt_tag", "no_gloss", "hash"])
def test_closed_class_inventory_rejects_forged_or_wrong_pos_proof(change):
    proof = closed_class_head_inventory([{"lemma": "πας", **closed_class_head_evidence(
        {"word": "πας", "pos": "det", "senses": [{"glosses": ["every, any"]}]}, "en")}],
        {"πας": ("πηγαίνω", 1)}, {"en": "a" * 64})
    entry = proof["heads"][0]["evidence"][0]
    if change == "nominal": entry["pos"] = "noun"
    elif change == "wrong_pos": proof["heads"][0]["pos"] = "pron"
    elif change == "form_gloss": entry["senses"][0]["glosses"] = ["genitive singular of άλλος"]
    elif change == "form_ref": entry["senses"][0]["form_of"] = [{"word": "άλλος"}]
    elif change == "alt_tag": entry["senses"][0]["tags"] = ["alt-of"]
    elif change == "no_gloss": entry["senses"][0]["tags"] = ["no-gloss"]
    elif change == "hash": proof["source_sha256"]["en"] = "wrong"
    with pytest.raises(ValueError):
        validate_closed_class_head_inventory(proof, {"πας": {"DET": "πας", "PRON": "πας"}})


def test_main_publishes_closed_class_ownership_without_changing_flat_verb(tmp_path, monkeypatch):
    import build_data
    source = tmp_path / "source"
    source.mkdir()
    entries = [
        {"word": "πας", "pos": "det", "senses": [{"glosses": ["every, any"]}]},
        {"word": "πας", "pos": "pron", "senses": [{"glosses": ["κάθε, όλος"]}]},
        {"word": "πας", "pos": "verb", "senses": [{"form_of": [{"word": "πηγαίνω"}]}]},
        {"word": "πηγαίνω", "pos": "verb", "senses": [{"glosses": ["go"]}]},
        {"word": "πάση", "pos": "det", "senses": [{"form_of": [{"word": "πας"}],
             "tags": ["dative", "feminine", "singular"]}]},
    ]
    (source / "en.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n"
                                              for e in entries), encoding="utf-8")
    (source / "el.jsonl").write_text("", encoding="utf-8")
    output = tmp_path / "data"
    monkeypatch.setattr(build_data, "DATA_DIR", output)
    monkeypatch.setattr(build_data, "DUMPS", {"el": {"en": "en.jsonl", "el": "el.jsonl"}})
    monkeypatch.setattr(build_data.sys, "argv", ["build_data.py", "--lang", "el", "--kaikki", str(source)])
    build_data.main()
    flat = json.loads((output / "mg_lookup.json").read_text())
    typed = json.loads((output / "mg_pos_lookup.json").read_text())
    assert flat["πας"] == "πηγαίνω"
    assert typed["πας"]["DET"] == typed["πας"]["PRON"] == "πας"
    assert typed["πας"]["VERB"] == "πηγαίνω"
    assert typed["πάση"]["DET"] == "πας"
    proof = json.loads((output / "el_closed_class_heads.json").read_text())
    assert validate_closed_class_head_inventory(proof, typed) == {"det": {"πας"}, "pron": {"πας"}}
    pair = next(p for p in json.loads((output / "mg_pairs.json").read_text())
                if p["form"] == "πάση" and p["pos"] == "det")
    assert pair["lemma"] == "πας" and "dative" in pair["tags"]


def test_built_closed_head_evidence_is_required_only_for_nonflat_typed_roots(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    from test_dilemma import built_closed_class_heads
    pairs = [{"form": "πάση", "lemma": "πας", "pos": "det"}]
    assert built_closed_class_heads(pairs, {"πας": "πας"}, tmp_path) == {}
    with pytest.raises(OSError):
        built_closed_class_heads(pairs, {"πας": "πηγαίνω"}, tmp_path)
    (tmp_path / "mg_pos_lookup.json").write_text(json.dumps({"πας": {"DET": "πας"}}))
    proof = closed_class_head_inventory([{"lemma": "πας", **closed_class_head_evidence(
        {"word": "πας", "pos": "det", "senses": [{"glosses": ["every, any"]}]}, "en")}],
        {"πας": ("πηγαίνω", 1)}, {"en": "a" * 64})
    (tmp_path / "el_closed_class_heads.json").write_text(json.dumps(proof))
    assert built_closed_class_heads(pairs, {"πας": "πηγαίνω"}, tmp_path) == {
        "det": {"πας"}, "pron": set()}


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


@pytest.mark.parametrize("reverse", [False, True])
def test_ancient_nominal_reference_keeps_independent_verb_head(tmp_path, monkeypatch, reverse):
    import build_data
    entries = [
        {"word": "κρίνω", "pos": "verb",
         "senses": [{"glosses": ["to distinguish among a group of things"]}],
         "forms": [{"form": "κρίνω", "tags": ["present", "first-person", "singular"]},
                   {"form": "κρίνεις", "tags": ["present", "second-person", "singular"]}]},
        {"word": "κρίνω", "pos": "noun",
         "senses": [{"form_of": [{"word": "κρίνον"}]}]},
        {"word": "κρίνον", "pos": "noun",
         "senses": [{"glosses": ["lily"]}]},
        {"word": "δύω", "pos": "verb",
         "senses": [{"glosses": ["to plunge"]}],
         "forms": [{"form": "δύω", "tags": ["present", "first-person", "singular"]},
                   {"form": "δύεις", "tags": ["present", "second-person", "singular"]}]},
        {"word": "δύω", "pos": "num",
         "senses": [{"alt_of": [{"word": "δύο"}]}]},
        {"word": "δύο", "pos": "num", "senses": [{"glosses": ["two"]}]},
        {"word": "δύνω", "pos": "verb", "senses": [{"glosses": ["to sink"]}],
         "forms": [{"form": "δύνεις", "tags": ["present", "second-person", "singular"]}]},
        {"word": "δύνω", "pos": "verb",
         "senses": [{"alt_of": [{"word": "δύω"}]}]},
    ]
    dump = _write(tmp_path, entries[::-1] if reverse else entries)
    output = tmp_path / "output"
    monkeypatch.setattr(build_data, "DATA_DIR", output)
    monkeypatch.setattr(build_data, "DUMPS", {"grc": {"en": dump.name}})
    monkeypatch.setattr(build_data.sys, "argv", [
        "build_data.py", "--lang", "grc", "--kaikki", str(tmp_path)])
    build_data.main()
    lookup = json.loads((output / "ag_lookup.json").read_text())
    assert lookup["κρίνω"] == lookup["κρινω"] == "κρίνω"
    assert lookup["δύω"] == lookup["δυω"] == "δύω"
    assert lookup["δύνω"] == "δύω"  # A same-POS variant still canonicalizes.
    pairs = json.loads((output / "ag_pairs.json").read_text())
    assert any(r["form"] == "κρίνεις" and r["lemma"] == "κρίνω" for r in pairs)
    assert any(r["form"] == "κρίνω" and r["lemma"] == "κρίνον"
               and r["pos"] == "noun" for r in pairs)
    assert any(r["form"] == "δύεις" and r["lemma"] == "δύω" for r in pairs)
    assert any(r["form"] == "δύνεις" and r["lemma"] == "δύω" for r in pairs)
    assert any(r["form"] == "δύω" and r["lemma"] == "δύο"
               and r["pos"] == "num" for r in pairs)
    pos = json.loads((output / "ag_pos_lookup.json").read_text())
    assert pos["κρίνω"] == {"NOUN": "κρίνον", "VERB": "κρίνω"}


@pytest.mark.parametrize("gloss", ["επικός τύπος του δύο", "δωρικός τύπος του δύο",
                                   "εναλλακτικός τύπος του δύο", "τύπος του δύο",
                                   "ιωνικός & επικός τύπος του παρθενία",
                                   "ιωνικός και επικός τύπος του παρθενία",
                                   "ιωνικός, επικός τύπος του παρθενία",
                                   "ιωνικός/επικός τύπος του παρθενία"])
def test_ancient_dialect_reference_is_not_an_independent_lexical_definition(gloss):
    from build_data import is_independent_lexical_entry
    entry = {"word": "δύω", "pos": "num", "senses": [{"glosses": [gloss]}]}
    assert not is_independent_lexical_entry(entry, "el")
    entry["senses"].append({"glosses": ["δύο"]})
    assert is_independent_lexical_entry(entry, "el")


@pytest.mark.parametrize("pos,gloss", [
    ("noun", "άλλη μορφή του: ανανδρία"),
    ("noun", "άλλη μορφή του : λαδολιά"),
    ("noun", "άλλη μορφή της λέξης συσκευαστήριο"),
    ("noun", "άλλη μορφή του ορυκτού κίσσηρη"),
    ("noun", "άλλη μορφή της λέξης γλοιός / γλοιά"),
    ("adj", "άλλη μορφή της λέξης χρυσόκομος"),
    ("adj", "επικός τύπος της λέξης γλύκιος"),
])
def test_reference_target_syntax_preserves_any_independent_meaning(pos, gloss):
    from build.build_grc_lexical_heads import has_definition
    from build_data import is_independent_lexical_entry
    entry = {"word": "λέξη", "pos": pos, "senses": [{"glosses": [gloss]}]}
    assert not is_independent_lexical_entry(entry, "el")
    assert not has_definition(entry, "el")
    entry["senses"][0]["glosses"][0] += ": μια ανεξάρτητη σημασία"
    assert is_independent_lexical_entry(entry, "el")
    assert has_definition(entry, "el")


@pytest.mark.parametrize("word,source,gloss", [
    ("πρηΰνω", "el", "ιωνικός τύπος του πρᾱΰνω"),
    ("χρέομαι", "el", "ιωνικός τύπος του χράομαι"),
    ("ἀντιόω", "el", "επικός τύπος του ἀντιάω"),
    ("τιμῶ", "el", "συνηρημένη μορφή του τιμάω"),
    ("θωέω", "en", "Delphic form of θωάζω"),
    ("σαμαίνω", "en", "Doric and Pamphylian form of σημαίνω"),
    ("ἀμφέπω", "en", "Tragic form of ἀμφιέπω"),
    ("ἐννέπω", "en", "Epic, Doric, and Tragic form of ἐνέπω"),
    ("ἐννέπω", "el", "επικός τύπος  και δωρικός τύπος του ἐνέπω"),
    ("ψῶ", "el", "συνηρημένος τύπος των ψάω / ψήω / ψώω / ψέω"),
    ("χ", "el", "ιωνικός & παλαιός αττικός τύπος του ψ"),
    ("χ", "el", "κρητικός και λακωνικός τύπος του ψ"),
    ("κολυμβᾶν", "el", "απαρέμφατο ενεργητικού ενεστώτα του κολυμβῶ"),
    ("ἑστάναι", "el", "απαρέμφατο παρακειμένου του ἵστημι"),
    ("ὀρέξεις", "el", "επικός τύπος : β΄ πρόσωπο ενικού υποτακτικής ενεργητικού αορίστου του ὀρέγω"),
    ("μαλακιάω", "el", "μεταγενέστερος τύπος του ρήματος μαλκιάω"),
    ("ῥήσσω", "el", "σπανιότερος τύπος του ρήματος ῥήγνυμι"),
    ("αἱρέομαι", "el", "μεσοπαθητική φωνή του ρήματος αἱρέω"),
    ("αἱροῦμαι", "el", "μεσοπαθητική φωνή του ρήματος αἱρῶ (συνηρημένη μορφή του αἱρέω)"),
    ("διαστέλλομαι", "el", "μέση-παθητική φωνή του διαστέλλω"),
    ("πιέζομαι", "el", "παθητική φωνή του ρήματος πιέζω (παθητική διάθεση)"),
    ("χειρουργοῦμαι", "el", "μεσοπαθητική φωνή του ρήματος χειρουργῶ, συνηρημένη μορφή του χειρουργέομαι"),
    ("χολόομαι", "el", "μέσος και παθητικός τύπος του ρήματος χολόω"),
    ("ὠφελοῦμαι", "el", "παθητική φωνή του ὠφελέω-ὠφελῶ"),
    ("στοναχέω", "el", "μορφή του στενάχω"),
    ("θηλέω", "el", "μορφή του θαλέω (αλλά και των ομόρριζων θάλλω, θαλέθω)"),
    ("φάργνυμι", "el", "μορφή του φράσσω (και φράγνυμι)"),
    ("φράγνυμι", "el", "μορφή του φράσσω (και φάργνυμι)"),
    ("λυπέομαι", "el", 'μέση και παθητική φωνή του λυπέω. Η μέση εκφέρεται και αναλυτικά με την ενεργητική φωνή: "λυπῶ ἐμαυτόν" Ο μέσος μέλλων (λυπήσομαι) συνήθως σύνθετος. Ο παθητικός αόριστος "ἐλυπήθην" είναι και μέσος.'),
])
def test_ancient_verb_reference_does_not_prove_an_independent_head(word, source, gloss):
    from build_data import is_independent_lexical_entry
    entry = {"word": word, "pos": "verb", "senses": [{"glosses": [gloss]}]}
    assert not is_independent_lexical_entry(entry, source)
    entry["senses"].append({"glosses": ["to act"]})
    assert is_independent_lexical_entry(entry, source)


def test_ancient_reference_filter_preserves_an_ordinary_definition():
    from build_data import is_independent_lexical_entry
    assert is_independent_lexical_entry({"word": "ζῶ", "pos": "verb",
        "senses": [{"glosses": ["to live a tragic form of life"]}]}, "en")


@pytest.mark.parametrize("gloss", [
    "παθητική φωνή του γλύφω: λαξεύομαι, σκαλίζομαι",
    "μέση φωνή του ρήματος κτενίζω: χτενίζομαι (χτενίζω τα μαλλιά μου)",
    "μέση-παθητική φωνή του ρήματος προδίδωμι, με εγκαταλείπουν, με προδίδουν",
    "μεσοπαθητική φωνή του ρήματος ὄλλυμι (καταστρέφομαι, χάνομαι, αφανίζομαι)",
    "μορφή του σώματος",
    "άλλη μορφή του ορυκτού άνθρακα",
    "μορφή του γενειάζω, όμως με τη λίγο ειδικότερη σημασία ότι τώρα αρχίζω να βγάζω γένια",
    "μορφή του φώγω (ξεροψήνω, καβουρντίζω, ξεραίνω)",
    "μέση και παθητική φωνή του λυπέω. Στενοχωριέμαι, θλίβομαι.",
    'μέση και παθητική φωνή του λυπέω. Ο μέσος μέλλων (λυπήσομαι) συνήθως σύνθετος. Στενοχωριέμαι.',
    'μέση και παθητική φωνή του λυπέω. Ο παθητικός αόριστος "ἐλυπήθην" είναι και μέσος: στενοχωρήθηκα.',
])
def test_ancient_reference_filter_retains_inline_lexical_meanings(gloss):
    from build_data import is_independent_lexical_entry
    pos = "noun" if gloss in {"μορφή του σώματος", "άλλη μορφή του ορυκτού άνθρακα"} else "verb"
    assert is_independent_lexical_entry({"word": "λέξη", "pos": pos,
        "senses": [{"glosses": [gloss]}]}, "el")


@pytest.mark.parametrize("word,pos,gloss", [
    ("καθάριος", "adj", "μεταγενέστερος τύπος του καθάρειος: καθαρός"),
    ("ἀμυγδαλῆ", "noun", "συνηρημένη μορφή του ἀμυγδαλέα: η αμυγδαλιά"),
    ("φαιδιμόεις", "adj", "άλλη μορφή του φαίδιμος : λαμπερός, λαμπρός, ακτινοβόλος"),
    ("ἐρωά", "noun", "βοιωτικός τύπος του ἐρωή· αποχώρηση, ανάπαυση, αποχή"),
    ("υἱύς", "noun", "παλαιός αττικός τύπος του υἱός, ο γιος"),
    ("πυγά", "noun", "δωρικός και λακωνικός τύπος του πυγή, ο πισινός"),
])
def test_ancient_dialect_description_with_own_definition_retains_head(word, pos, gloss):
    from build_data import is_independent_lexical_entry
    assert is_independent_lexical_entry({"word": word, "pos": pos,
        "senses": [{"glosses": [gloss]}]}, "el")


@pytest.mark.parametrize("pos,gloss", [
    ("verb", "ποιητικός & επικός τύπος του καταμύω (συχνά σε ελληνιστικά κείμενα)"),
    ("verb", "συνηρημένη μορφή του ὁρμάω ἠ του ὁρμέω"),
    ("verb", "συνηρημένος τύπος του ἀλέω (α΄πρόσωπο οριστικής και υποτακτικής ενεστώτα)"),
    ("verb", "επικός & ιωνικός τύπος του λαβέ: β΄ πρόσωπο ενικού προστακτικής ενεργητικού ενεστώτα του λαμβάνω"),
    ("verb", "ποιητικός τύπος του κύσω, α΄ ενικό πρόσωπο οριστικής μέλλοντα του ρήματος κυνέω (φιλώ)"),
    ("verb", "συνηρημένη μορφή του ἀσθενόω (α στερ. + σθένος)"),
    ("verb", "συνηρημένη μορφή του ἀτιμόω (εκ του άτιμος)"),
    ("verb", "τύπος του ρήματος αὐξάνω, όπως και το ποιητικό ἀέξω"),
    ("verb", "δωρικός τύπος του χρῄζω των Μεγαρέων"),
    ("verb", "δωρικός τύπος του χρῄζω στους Σικελιώτες"),
    ("verb", "συνηρημένη μορφή του γελάω και του γελόω"),
    ("verb", "επικός τύπος του γελάω -γελῶ"),
    ("verb", "άλλη μορφή του στενάζω, κυρίως επικός τύπος"),
    ("verb", "συνηρημένη μορφή του φυσάω και φυσέω (ιωνικός τύπος του φυσάω)"),
    ("adj", "συνηρημένη μορφή του επιθέτου φωνήεις"),
    ("verb", "συνηρημένη μορφή του θαμβέω και του μεταγενέστερου θαμβόω"),
    ("verb", "συνηρημένη μορφή του λωβάομαι, αποθετικού ρήματος"),
    ("adj", "συνηρημένη μορφή του πλακόεις (Χρειάζεται επεξεργασία)"),
    ("verb", "ιωνικός & επικός τύπος του προτέμνομαι: μεσοπαθητική φωνή του ρήματος προτάμνω"),
    ("pron", "κρητικός τύπος του ὅσος (στην κεντρική Κρήτη)"),
    ("pron", "κρητικός τύπος του ὅσος (στη Γόρτυνα)"),
    ("verb", "δωρικός τύπος του δωρίζω / Δωρίζω (ελληνιστική κοινή) - μαρτυρείται στο απαρέμφατο ενεστώτα δωρίσδεν"),
    ("verb", "συνηρημένη μορφή του ἀροτριάω (κλίση τιμάω - απαντούν συνηρημένοι τύποι) (Χρειάζεται επεξεργασία)"),
    ("verb", "συνηρημένη μορφή του ἀροτριόω (απαντούν τύποι από την κλίση δηλόω)"),
    ("noun", "εναλλακτική μορφή του ὀπτίλος & ὄπτιλλος"),
    ("verb", "δωρικός τύπος του πιέζω (από τον αόριστο του οποίου, δηλ. το ἔπιασα, θεωρείται ότι προέκυψε το πιάνω)"),
    ("noun", "επικός τύπος του τέκμαρ (στον Όμηρο)"),
    ("adv", "ποιητικός τύπος του επιρρήματος ἑξῆς"),
    ("name", "αττικός τύπος του ᾌδης (επίσης: ᾄδης, Ἅιδης και Ἀΐδας)"),
    ("noun", "αττικός τύπος του νόος (συνηρημένος τύπος)"),
    ("noun", "αττικός τύπος του θρόος, συνηρημένος τύπος"),
    ("noun", "ιωνικός τύπος του θυγατριδοῦς (ασυναίρετος τύπος)"),
    ("noun", "αττικός τύπος του κάνεον (συνηρημένο)"),
    ("pron", "αττικός τύπος του αὐτός (μέσω κράσης με το ὁ)"),
    ("adj", "δωρικός τύπος του *εὐγηθής"),
    ("noun", "ιωνικός τύπος του καύαξ,"),
    ("adj", "επικός τύπος του βελτίων (καλύτερος)· συγκριτικός βαθμός του ἀγαθός"),
])
def test_complete_reference_with_grammar_notes_does_not_prove_a_head(pos, gloss):
    from build.build_grc_lexical_heads import has_definition
    from build_data import is_independent_lexical_entry
    entry = {"word": "λέξη", "pos": pos, "senses": [{"glosses": [gloss]}]}
    assert not is_independent_lexical_entry(entry, "el")
    assert not has_definition(entry, "el")
    entry["senses"].append({"glosses": ["a separate lexical meaning"]})
    assert is_independent_lexical_entry(entry, "el")
    assert has_definition(entry, "el")


@pytest.mark.parametrize("gloss", [
    "κρητικός τύπος του ἡλικιώτης, συνέφηβος",
    "επικός τύπος του ἀγρεύω, κυριεύω, κυνηγώ, επιδιώκω, καταλαμβάνω",
    "άλλη μορφή του λύω (μια ανεξάρτητη σημασία)",
    "άλλη μορφή του λύω, κυρίως επικός τύπος: λύνω",
    "επικός τύπος του ὁδός, μόνο στην Οδύσσεια (σε αιτιατική): ※ Ὅμηρος: δρόμος",
])
def test_unknown_reference_tail_remains_lexical_evidence(gloss):
    from build.build_grc_lexical_heads import has_definition
    from build_data import is_independent_lexical_entry
    entry = {"word": "λέξη", "pos": "verb", "senses": [{"glosses": [gloss]}]}
    assert is_independent_lexical_entry(entry, "el")
    assert has_definition(entry, "el")


@pytest.mark.parametrize("gloss", [
    "κρητικός τύπος του χαλκός, χαλκοῦς",
    "Κρητικός  τύπος\nτου χαλκός, χαλκοῦς",
    "δωρικός τύπος του γᾶ (γῆ)",
    "αιολικός & δωρικός τύπος του ἐκεῖνος, κεῖνος",
    "αιολικός τύπος του ἁμός, -ά, -όν (ἡμέτερος)",
    "αιολικός τύπος του ἐν μέσῳ",
    "άλλη μορφή της λέξης γλοιός, γλία",
    "άλλη μορφή του ορυκτού κίσσηρη",
    'άλλη μορφή του επιθέτου πορτογαλικός. Θεωρείται λιγότερο επίσημη, '
    'χρησιμοποιείται κυρίως γιατί το "πορτογαλικός" προκαλεί σύγχυση με το '
    'δεύτερο συνθετικό του "γαλικός".',
    "επικός τύπος του ὁδός, μόνο στην Οδύσσεια (σε αιτιατική): "
    "※ 8ος αιώνας πκε ⌘ Ὅμηρος, Ὀδύσσεια, 17 "
    "(ρ. Τηλεμάχου καὶ Ὀδυσσέως ἐπάνοδος εἰς Ἰθάκην.), στίχ. 196 (195-196 )",
])
def test_reviewed_reference_keeps_any_appended_definition(gloss):
    from build.build_grc_lexical_heads import has_definition
    from build_data import is_independent_lexical_entry
    entry = {"word": "καυχός", "pos": "noun", "senses": [{"glosses": [gloss]}]}
    assert not is_independent_lexical_entry(entry, "el")
    assert not has_definition(entry, "el")
    entry["senses"][0]["glosses"][0] += ": χαλκός ως υλικό"
    assert is_independent_lexical_entry(entry, "el")
    assert has_definition(entry, "el")


@pytest.mark.parametrize('pos', ['character', 'symbol', 'phrase', 'suffix', 'prefix'])
def test_nonword_source_sections_do_not_establish_lexical_owners(pos):
    from build_data import is_independent_lexical_entry
    assert not is_independent_lexical_entry(
        {'word': 'α', 'pos': pos, 'senses': [{'glosses': ['the letter alpha']}]}, 'en')


@pytest.mark.parametrize("reverse", [False, True])
def test_ancient_other_pos_reference_across_sources_preserves_verb(tmp_path, monkeypatch, reverse):
    import build_data
    en = [
        {"word": "δύω", "pos": "num", "senses": [{"alt_of": [{"word": "δύο"}]}]},
        {"word": "δύω", "pos": "verb", "senses": [{"glosses": ["to plunge"]}],
         "forms": [{"form": "δύεις", "tags": ["present", "second-person", "singular"]}]},
        {"word": "δύο", "pos": "num", "senses": [{"glosses": ["two"]}]},
    ]
    el = [{"word": "δύω", "pos": "num",
           "senses": [{"glosses": ["επικός τύπος του δύο"]}]}]
    source = tmp_path / "source"
    source.mkdir()
    for name, entries in (("en", en), ("el", el)):
        (source/(name+'.jsonl')).write_text(''.join(json.dumps(e, ensure_ascii=False)+'\n'
            for e in (entries[::-1] if reverse else entries)))
    output = tmp_path / "output"
    monkeypatch.setattr(build_data, "DATA_DIR", output)
    monkeypatch.setattr(build_data, "DUMPS", {"grc": {"en": "en.jsonl", "el": "el.jsonl"}})
    monkeypatch.setattr(build_data.sys, "argv", ["build_data.py", "--lang", "grc", "--kaikki", str(source)])
    build_data.main()
    lookup = json.loads((output / "ag_lookup.json").read_text())
    assert lookup["δύω"] == lookup["δυω"] == "δύω"
    pairs = json.loads((output / "ag_pairs.json").read_text())
    assert any(r["form"] == "δύεις" and r["lemma"] == "δύω" for r in pairs)
    assert any(r["form"] == "δύω" and r["lemma"] == "δύο" and r["pos"] == "num" for r in pairs)


@pytest.mark.parametrize("indirect", [False, True])
def test_original_head_same_pos_alternative_prevents_other_pos_recovery(indirect):
    from build_data import recover_cross_pos_heads
    heads = {"verb": {"δίω", "δείδω"}, "adj": {"δῖος"}}
    references = {"δίω": {"adj": {"δῖος"}, "verb": {"δειδῶ" if indirect else "δείδω"}}}
    if indirect:
        references["δειδῶ"] = {"verb": {"δείδω"}}
    lookup = {"δίω": ("δῖος", 1), "διω": ("δῖος", 1)}
    canonical = {"δίω": {"verb": references["δίω"]["verb"]}}
    if indirect:
        canonical["δειδῶ"] = {"verb": {"δείδω"}}
    assert recover_cross_pos_heads(lookup, heads, references, canonical) == 0
    assert lookup["δίω"][0] == "δῖος"
    lookup["δίω"] = ("δίω", 3)
    assert resolve_cross_source(lookup, {"δίω": "δῖος"}, {}, set(),
                               lexical_heads=heads, typed_references=references,
                               canonical_references=canonical) == 1


@pytest.mark.parametrize("reverse", [False, True])
def test_independent_verb_and_same_pos_inflection_keep_both_readings(tmp_path, monkeypatch, reverse):
    import build_data
    entries = [
        {"word": "δίω", "pos": "verb", "senses": [{"glosses": ["run away, flee"]}],
         "forms": [{"form": "δίω", "tags": ["present", "first-person", "singular"]},
                   {"form": "δίεις", "tags": ["present", "second-person", "singular"]}]},
        {"word": "δίω", "pos": "verb", "senses": [{
            "form_of": [{"word": "δείδω"}],
            "tags": ["aorist", "subjunctive", "first-person", "singular"]}]},
        {"word": "δίω", "pos": "adj", "senses": [{"form_of": [{"word": "δῖος"}]}]},
        {"word": "δείδω", "pos": "verb", "senses": [{"glosses": ["fear"]}]},
        {"word": "δῖος", "pos": "adj", "senses": [{"glosses": ["divine"]}]},
    ]
    dump = _write(tmp_path, entries[::-1] if reverse else entries)
    output = tmp_path / "output"
    monkeypatch.setattr(build_data, "DATA_DIR", output)
    monkeypatch.setattr(build_data, "DUMPS", {"grc": {"en": dump.name}})
    monkeypatch.setattr(build_data.sys, "argv", [
        "build_data.py", "--lang", "grc", "--kaikki", str(tmp_path)])
    build_data.main()
    lookup = json.loads((output / "ag_lookup.json").read_text())
    assert lookup["δίω"] in {"δίω", "δείδω"}
    assert lookup["δίω"] == lookup["διω"]
    pairs = json.loads((output / "ag_pairs.json").read_text())
    assert any(r["form"] == "δίεις" and r["lemma"] == "δίω" for r in pairs)
    assert {r["lemma"] for r in pairs if r["form"] == "δίω" and r["pos"] == "verb"} == {
        "δίω", "δείδω"}
    assert {r["lemma"] for r in pairs if r["form"] == "δίω" and r["pos"] == "adj"} == {"δῖος"}
    inventory = json.loads((output / 'grc_reading_heads.json').read_text())
    heads = build_data.validate_ancient_reading_head_inventory(inventory)
    assert build_data.training_pair_headword_errors(pairs, lookup, lexical_heads=heads)[0] == 0


@pytest.mark.parametrize('flat_target', ['δείδω', None])
def test_ancient_reading_evidence_requires_exact_positive_part_of_speech(tmp_path, monkeypatch, flat_target):
    import build_data
    import hashlib

    dump = _write(tmp_path, [{'word': 'δίω', 'pos': 'verb',
                             'senses': [{'glosses': ['run away, flee']}]}])
    pairs = [{'form': 'δίεις', 'lemma': 'δίω', 'pos': 'verb'},
             {'form': 'δίω', 'lemma': 'δίω', 'pos': 'adj'}]
    lookup = {'δίω': (flat_target, 1)} if flat_target else {}
    # Python 3.10 is supported and has no hashlib.file_digest.
    monkeypatch.delattr(hashlib, 'file_digest', raising=False)
    inventory = build_data.ancient_reading_head_inventory(pairs[:1], lookup, {'en': dump})
    assert inventory['source_sha256'] == {'en': hashlib.sha256(dump.read_bytes()).hexdigest()}
    heads = build_data.validate_ancient_reading_head_inventory(inventory)
    assert heads == {'verb': {'δίω'}}
    flat = {'δίω': flat_target} if flat_target else {}
    assert build_data.training_pair_headword_errors(pairs, flat)[0] == (2 if flat_target else 0)
    assert build_data.training_pair_headword_errors(pairs, flat, lexical_heads={})[0] == 2
    assert build_data.training_pair_headword_errors(pairs, flat, lexical_heads=heads)[0] == 1
    with pytest.raises(ValueError, match='Unproven Ancient'):
        build_data.ancient_reading_head_inventory(pairs, lookup, {'en': dump})


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("number_type", [str, int])
def test_separate_alternative_etymology_preserves_lexical_tables(
        tmp_path, monkeypatch, reverse, number_type):
    import build_data

    entries = [
        {"word": "δέω", "pos": "verb", "etymology_number": 1,
         "senses": [{"glosses": ["to bind"]}],
         "forms": [{"form": "δέω", "tags": ["present", "indicative", "active",
                                                "first-person", "singular"]},
                   {"form": "δῶ", "tags": ["present", "indicative", "active",
                                               "first-person", "singular"]}]},
        {"word": "δέω", "pos": "verb", "etymology_number": 2,
         "senses": [{"glosses": ["to need"]}],
         "forms": [{"form": "δέομαι", "tags": ["present", "indicative", "middle",
                                                   "first-person", "singular"]}]},
        {"word": "δέω", "pos": "verb", "etymology_number": 3,
         "senses": [{"glosses": ["alternative form of δήω"],
                     "alt_of": [{"word": "δήω"}]}],
         "forms": [{"form": "δήει", "tags": ["present", "indicative", "active",
                                                  "third-person", "singular"]}]},
        {"word": "δήω", "pos": "verb", "senses": [{"glosses": ["to find"]}]},
        {"word": "δηῶ", "pos": "verb",
         "senses": [{"glosses": ["alternative form of δήω"],
                     "alt_of": [{"word": "δήω"}]}]},
    ]
    for entry in entries:
        if "etymology_number" in entry:
            entry["etymology_number"] = number_type(entry["etymology_number"])
    dump = _write(tmp_path, entries[::-1] if reverse else entries)
    output = tmp_path / "output"
    monkeypatch.setattr(build_data, "DATA_DIR", output)
    monkeypatch.setattr(build_data, "DUMPS", {"grc": {"en": dump.name}})
    monkeypatch.setattr(build_data.sys, "argv", [
        "build_data.py", "--lang", "grc", "--kaikki", str(tmp_path)])
    build_data.main()
    pairs = json.loads((output / "ag_pairs.json").read_text())
    lookup = json.loads((output / "ag_lookup.json").read_text())
    for form in ("δῶ", "δέομαι"):
        assert {p["lemma"] for p in pairs if p["form"] == form} == {"δέω"}
        assert lookup[form] == "δέω"
    assert {p["lemma"] for p in pairs if p["form"] == "δήει"} == {"δήω"}
    assert {p["lemma"] for p in pairs if p["form"] == "δέω"} == {"δέω", "δήω"}
    assert lookup["δέω"] == "δέω"
    assert lookup["δηῶ"] == "δήω"


@pytest.mark.parametrize("reverse", [False, True])
def test_separate_spelling_etymology_follows_its_reference_chain(tmp_path, monkeypatch, reverse):
    import build_data

    forms = [{"form": "πίτας", "tags": ["genitive"]},
             {"form": "πίττα", "tags": ["alternative"]},
             {"form": "πίσσα", "tags": ["alternative"]}]
    entries = [
        {"word": "πίτα", "pos": "noun", "etymology_number": 1,
         "senses": [{"glosses": ["alternative form of πίττα"],
                     "alt_of": [{"word": "πίττα"}]}], "forms": forms},
        {"word": "πίτα", "pos": "noun", "etymology_number": 2,
         "senses": [{"glosses": ["pie"]}], "forms": forms},
        {"word": "πίττα", "pos": "noun",
         "senses": [{"glosses": ["alternative form of πίσσα"],
                     "alt_of": [{"word": "πίσσα"}]}]},
        {"word": "πίσσα", "pos": "noun", "senses": [{"glosses": ["pitch"]}]},
    ]
    dump = _write(tmp_path, entries[::-1] if reverse else entries)
    output = tmp_path / "output"
    monkeypatch.setattr(build_data, "DATA_DIR", output)
    monkeypatch.setattr(build_data, "DUMPS", {"grc": {"en": dump.name}})
    monkeypatch.setattr(build_data.sys, "argv", [
        "build_data.py", "--lang", "grc", "--kaikki", str(tmp_path)])
    build_data.main()
    pairs = json.loads((output / "ag_pairs.json").read_text())
    for form in ("πίτας", "πίττα", "πίσσα"):
        assert {p["lemma"] for p in pairs if p["form"] == form} == {"πίτα", "πίσσα"}


@pytest.mark.parametrize("kind", ["direct", "chain", "ambiguous", "unsupported",
                                 "wrong_pos", "cycle", "no_reference"])
def test_reference_only_spelling_prefers_unique_positive_canonical_endpoint(kind):
    lookup = {word: (word, 3) for word in ("πίτα", "πίσσα", "ἄσφαλτος")}
    lookup["πίττα"] = ("πίτα", 1)
    refs = {"πίττα": {"noun": {"πίσσα"}}}
    if kind == "chain":
        refs = {"πίττα": {"noun": {"πίττη"}}, "πίττη": {"noun": {"πίσσα"}}}
    elif kind == "ambiguous":
        refs["πίττα"]["noun"].add("ἄσφαλτος")
    elif kind == "unsupported":
        refs["πίττα"]["noun"] = {"πίττη"}
    elif kind == "wrong_pos":
        refs = {"πίττα": {"verb": {"πίσσα"}}}
    elif kind == "cycle":
        refs["πίσσα"] = {"noun": {"πίττα"}}
    elif kind == "no_reference":
        refs = {}
    heads = {"noun": {"πίτα", "πίσσα", "ἄσφαλτος"}}
    result = resolve_pair_lemma("πίττα", "noun", lookup,
                                {"πίττα": {"NOUN": "πίτα"}},
                                lexical_heads=heads, canonical_references=refs)
    assert result == ("πίσσα" if kind in {"direct", "chain"} else "πίτα")
    assert resolve_pair_lemma("πίτα", "noun", lookup, lexical_heads=heads,
                              canonical_references=refs) == "πίτα"


@pytest.mark.parametrize('change', ['form_of', 'wrong_pos', 'word', 'hash',
                                  'duplicate', 'gloss', 'missing', 'malformed'])
def test_ancient_reading_evidence_rejects_invalid_support(tmp_path, change):
    import build_data

    dump = _write(tmp_path, [{'word': 'δίω', 'pos': 'verb',
                             'senses': [{'glosses': ['run away, flee']}]}])
    inventory = build_data.ancient_reading_head_inventory(
        [{'form': 'δίεις', 'lemma': 'δίω', 'pos': 'verb'}], {'δίω': ('δείδω', 1)}, {'en': dump})
    entry = inventory['heads'][0]['evidence'][0]['entry']
    if change == 'form_of':
        entry['senses'][0]['form_of'] = [{'word': 'δείδω'}]
    elif change == 'wrong_pos':
        inventory['heads'][0]['pos'] = 'adj'
    elif change == 'word':
        entry['word'] = 'δείδω'
    elif change == 'hash':
        inventory['source_sha256']['en'] = 'not a hash'
    elif change == 'duplicate':
        inventory['heads'].append(inventory['heads'][0])
    elif change == 'gloss':
        entry['senses'][0]['glosses'] = ['first person singular of δείδω']
    elif change == 'missing':
        inventory['heads'][0]['evidence'] = []
    else:
        entry['senses'] = [None]
    with pytest.raises(ValueError):
        build_data.validate_ancient_reading_head_inventory(inventory)


@pytest.mark.parametrize("kind", ["no_reference", "same_pos_endpoint", "same_pos_chain"])
def test_cross_source_head_guard_requires_exclusive_other_pos_evidence(kind):
    lookup = {word: (word, 3) for word in ("κρίνω", "κρίνον", "κρινέω")}
    heads = {"verb": {"κρίνω"}, "noun": {"κρίνον"}}
    references = {"κρίνω": {"noun": {"κρίνον"}}}
    if kind == "no_reference":
        references = {}
    elif kind == "same_pos_endpoint":
        heads["verb"].add("κρίνον")
    else:
        heads["verb"].add("κρινέω")
        references["κρίνον"] = {"verb": {"κρινέω"}}
    assert resolve_cross_source(lookup, {"κρίνω": "κρίνον"}, {}, set(),
                                lexical_heads=heads, typed_references=references) == 1
    assert lookup["κρίνω"][0] == "κρίνον"


@pytest.mark.parametrize("reverse", [False, True])
def test_main_preserves_ancient_noun_and_verb_in_reference_cycle(tmp_path, monkeypatch, reverse):
    import build_data
    entries = [
        {"word": "ἔργον", "pos": "noun",
         "senses": [{"glosses": ["work, deed"]}],
         "forms": [{"form": "ἔργον", "tags": ["canonical"]},
                   {"form": "ἔργα", "tags": ["nominative", "plural"]}]},
        {"word": "ἔργω", "pos": "verb",
         "senses": [{"glosses": ["to enclose"]}],
         "forms": [{"form": "ἔργω", "tags": ["canonical"]},
                   {"form": "ἔργεις", "tags": ["present", "second-person"]}]},
        {"word": "ἔργον", "pos": "verb",
         "senses": [{"form_of": [{"word": "ἔργω"}]}],
         "forms": [{"form": "ἔργουσα", "tags": ["feminine"]}]},
        {"word": "ἔργω", "pos": "noun",
         "senses": [{"form_of": [{"word": "ἔργον"}]}],
         "forms": [{"form": "ἔργω", "tags": ["dual", "nominative"]}]},
        {"word": "λόγον", "pos": "noun",
         "senses": [{"form_of": [{"word": "λόγος"}]}]},
        {"word": "λόγος", "pos": "noun",
         "senses": [{"glosses": ["word, account"]}]},
    ]
    dump = _write(tmp_path, entries[::-1] if reverse else entries)
    output = tmp_path / "output"
    monkeypatch.setattr(build_data, "DATA_DIR", output)
    monkeypatch.setattr(build_data, "DUMPS", {"grc": {"en": dump.name}})
    monkeypatch.setattr(build_data.sys, "argv", [
        "build_data.py", "--lang", "grc", "--kaikki", str(tmp_path)])
    build_data.main()
    lookup = json.loads((output / "ag_lookup.json").read_text())
    assert lookup["ἔργον"] == "ἔργον"
    assert lookup["ἔργω"] == "ἔργω"
    assert lookup["λόγον"] == "λόγος"
    pairs = json.loads((output / "ag_pairs.json").read_text())
    assert any(row["form"] == "ἔργα" and row["lemma"] == "ἔργον"
               and row["pos"] == "noun" for row in pairs)
    assert any(row["form"] == "ἔργεις" and row["lemma"] == "ἔργω"
               and row["pos"] == "verb" for row in pairs)
    assert {row["lemma"] for row in pairs if row["form"] == "ἔργουσα"} == {"ἔργω"}
    assert {row["lemma"] for row in pairs if row["form"] == "ἔργω"
            and row["pos"] == "noun"} == {"ἔργον"}
    pos = json.loads((output / "ag_pos_lookup.json").read_text())
    assert pos["ἔργω"]["NOUN"] == "ἔργον"
    assert pos["ἔργον"]["VERB"] == "ἔργω"
    assert all("via" not in row for row in pairs)


@pytest.mark.parametrize("reverse", [False, True])
def test_ancient_independent_homographs_keep_their_own_tables(tmp_path, reverse):
    entries = [
        {"word": "ταμία", "pos": "noun",
         "senses": [{"glosses": ["female housekeeper"]}],
         "forms": [{"form": "ταμίαι", "tags": ["plural", "nominative"]}]},
        {"word": "ταμία", "pos": "noun",
         "senses": [{"form_of": [{"word": "ταμίας"}]}],
         "forms": [{"form": "ταμία", "tags": ["vocative", "singular"]}]},
        {"word": "ταμίας", "pos": "noun",
         "senses": [{"glosses": ["male steward"]}],
         "forms": [{"form": "ταμία", "tags": ["vocative", "singular"]}]},
        {"word": "ἀλέομαι", "pos": "verb",
         "senses": [{"glosses": ["to shun"]}],
         "forms": [{"form": "ἀλέεται", "tags": ["present", "third-person", "singular"]}]},
        {"word": "ἀλέομαι", "pos": "verb",
         "senses": [{"form_of": [{"word": "ἀλέω"}]}],
         "forms": [{"form": "ἀλέεται", "tags": ["present", "third-person", "singular"]}]},
        {"word": "ἀλέω", "pos": "verb",
         "senses": [{"glosses": ["to grind"]}],
         "forms": [{"form": "ἀλέομαι", "tags": ["present", "first-person", "singular"]}]},
    ]
    heads = {}
    pairs, _, _, pos_map, *_ = extract_pairs(
        _write(tmp_path, entries[::-1] if reverse else entries), "grc-en", lexical_heads=heads)
    # Both lexical heads survived the reciprocal-reference pass. A page's
    # own table and another reading's reference table must remain distinct.
    lookup = {word: (word, 4) for word in ("ταμία", "ταμίας", "ἀλέομαι", "ἀλέω")}
    clean = clean_training_pairs(pairs, lookup, pos_map=pos_map, lexical_heads=heads)
    assert {r["lemma"] for r in clean if r["form"] == "ταμίαι"} == {"ταμία"}
    assert {r["lemma"] for r in clean if r["form"] == "ταμία"
            and "vocative" in r["tags"]} == {"ταμίας"}
    assert {r["lemma"] for r in clean if r["form"] == "ἀλέεται"} == {"ἀλέομαι", "ἀλέω"}
    assert all("via" not in row for row in clean)


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
        {"word": "αναχρονίζω", "pos": "verb", "senses": [{"glosses": ["be anachronistic"]}]},
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
        {"word": "σωροβολιάζομαι", "pos": "verb", "senses": [{"glosses": ["be piled up"]}]},
        {"word": "σωρεύω", "pos": "verb", "senses": [{"glosses": ["pile up"]}]},
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


def test_resolved_graph_root_is_followed_after_a_flat_alias():
    lookup = {"τρισευτυχισμένος": ("ευτυχισμένος", 1),
              "ευτυχισμένος": ("ευτυχισμένος", 3), "ευτυχώ": ("ευτυχώ", 3)}
    pair = {"form": "τρισευτυχισμένη", "lemma": "τρισευτυχισμένος",
            "pos": "verb", "tags": ["feminine", "singular"]}
    result = clean_training_pairs([pair], lookup, True, {}, {},
                                  {"ευτυχισμένος": "ευτυχώ"}, set())
    assert result == [{**pair, "lemma": "ευτυχώ"}]


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


def test_related_participle_annotation_does_not_disqualify_a_finite_head(tmp_path):
    dump = _write(tmp_path, [
        {"word": "πίνω", "pos": "verb", "senses": [
            {"glosses": ["καταναλώνω ένα υγρό"]},
            {"tags": ["no-gloss", "participle", "verb"],
             "related": [{"word": "πιωμένος"}]}]},
        {"word": "πιωμένος", "pos": "verb", "tags": ["participle"],
         "form_of": [{"word": "πίνω"}],
         "senses": [{"glosses": ["που έχει πιει"]}]},
        {"word": "κατηγορούμενος", "pos": "verb",
         "senses": [{"tags": ["no-gloss", "participle"],
                     "related": [{"word": "κατηγορώ"}]}]},
    ])
    refs, heads, participles = {}, set(), set()
    _, _, _, pos_map, *_ = extract_pairs(
        dump, "el-el", participle_refs=refs, verb_heads=heads,
        participle_heads=participles)
    assert "πίνω" in heads and "πίνω" not in participles
    assert "κατηγορούμενος" in participles
    assert apply_participle_references(pos_map, refs, heads, participles) == set()
    assert pos_map["πιωμένος"]["VERB"] == "πίνω"


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


def test_explicit_verb_targets_preserve_all_source_readings(tmp_path):
    dump = _write(tmp_path, [
        {"word": "τύπος", "pos": "verb", "senses": [
            {"form_of": [{"word": "γράφω"}]},
            {"alt_of": [{"word": "λέω"}]}]},
    ])
    references = {}
    extract_pairs(dump, "el-el", verb_refs=references)
    assert references["τύπος"] == {"γράφω", "λέω"}


@pytest.mark.parametrize("reverse", [False, True])
def test_named_verb_in_one_sense_does_not_replace_another_senses_target(tmp_path, reverse):
    senses = [
        {"form_of": [{"word": "γράμμα"}], "glosses": ["τύπος του ρήματος γράφω"]},
        {"form_of": [{"word": "λέω"}]}]
    dump = _write(tmp_path, [{"word": "τύπος", "pos": "verb",
                             "senses": senses[::-1] if reverse else senses}])
    references = {}
    pairs, *_ = extract_pairs(dump, "el-el", verb_refs=references)
    assert references["τύπος"] == {"γράφω", "λέω"}
    assert next(p for p in pairs if p["lemma"] == "γράμμα")["via"] == "γράφω"
    assert "via" not in next(p for p in pairs if p["lemma"] == "λέω")
    flat = {w: (w, 3) for w in ("γράμμα", "γράφω", "λέω")}
    clean = clean_training_pairs(pairs, flat, preserve_cells=True,
                                 pos_map={w: {"VERB": w} for w in ("γράφω", "λέω")})
    assert {p["lemma"] for p in clean} == {"γράφω", "λέω"}


def test_participle_references_keep_entry_and_sense_provenance(tmp_path):
    dump = _write(tmp_path, [
        {"word": "δεσμευθείς", "pos": "verb", "tags": ["participle"],
         "form_of": [{"word": "δεσμεύω"}],
         "senses": [{"glosses": ["που δεσμεύθηκε"]}]},
        {"word": "δεσμευθείς", "pos": "verb", "senses": [
            {"form_of": [{"word": "κατατίθεμαι"}],
             "glosses": ["β' ενικό του ρήματος κατατίθεμαι"]}]},
        {"word": "τύπος", "pos": "verb", "senses": [
            {"tags": ["participle"], "form_of": [{"word": "γράφω"}]},
            {"tags": ["second-person"], "form_of": [{"word": "λέω"}]}]},
    ])
    generic, own, top, heads, participles = {}, {}, {}, set(), set()
    extract_pairs(dump, "el-el", participle_refs=top, verb_heads=heads,
                  participle_heads=participles, verb_refs=generic,
                  participle_verb_refs=own)
    assert generic["δεσμευθείς"] == {"κατατίθεμαι"}
    assert "δεσμευθείς" not in own
    assert generic["τύπος"] == {"γράφω", "λέω"}
    assert own["τύπος"] == {"γράφω"}
    pos_map = {"δεσμευθείς": {"VERB": "κατατίθεμαι"}}
    assert not apply_participle_references(
        pos_map, top, {"δεσμεύω", "κατατίθεμαι"}, participles,
        verb_refs=generic, participle_verb_refs=own)
    assert pos_map["δεσμευθείς"]["VERB"] == "δεσμεύω"


@pytest.mark.parametrize("gloss", ["συνώνυμο του κατανεμημένος", "Synonym of κατανεμημένος"])
def test_synonym_sense_does_not_become_a_graph_alias(tmp_path, gloss):
    dump = _write(tmp_path, [{"word": "διανεμημένος", "pos": "verb",
        "tags": ["participle"], "form_of": [{"word": "διανέμω"}],
        "senses": [{"glosses": ["που έχει διανεμηθεί"]},
                   {"glosses": [gloss], "alt_of": [{"word": "κατανεμημένος"}]}]}])
    generic, own, top, participles = {}, {}, {}, set()
    pairs, _, _, pos_map, *_ = extract_pairs(
        dump, "el-el", participle_refs=top, participle_heads=participles,
        verb_refs=generic, participle_verb_refs=own)
    assert "διανεμημένος" not in generic and "διανεμημένος" not in own
    assert any(p["lemma"] == "κατανεμημένος" for p in pairs)
    pos_map["κατανεμημένος"] = {"VERB": "κατανέμω"}
    assert not apply_participle_references(
        pos_map, top, {"διανέμω", "κατανέμω"}, participles,
        verb_refs=generic, participle_verb_refs=own)
    assert pos_map["διανεμημένος"]["VERB"] == "διανέμω"


def test_participle_top_pointer_can_follow_a_generic_spelling_reference():
    pos_map = {}
    assert not apply_participle_references(
        pos_map, {"τύπος": {"παλιό"}}, {"γράφω"}, {"τύπος"},
        verb_refs={"παλιό": {"γράφω"}}, participle_verb_refs={})
    assert pos_map["τύπος"]["VERB"] == "γράφω"


@pytest.mark.parametrize("reverse", [False, True])
def test_named_participle_reference_precedes_table_evidence(reverse):
    pos_map = {"βγαλμένος": {"VERB": "βγαίνω"}}
    refs = {"βγαλμένος": {"βγάζω"}}
    heads = {"βγάζω", "βγαίνω"}
    explicit = {"βγαλμένος": {"βγάζω"}} if reverse else {}
    if reverse:
        refs = {"βγαλμένος": {"βγαίνω"}}
    unresolved = apply_participle_references(
        pos_map, refs, heads, {"βγαλμένος"}, verb_refs=explicit)
    assert not unresolved
    assert pos_map["βγαλμένος"]["VERB"] == "βγάζω"


def test_ambiguous_explicit_references_cannot_be_replaced_by_a_table_choice():
    pos_map = {"τύπος": {"VERB": "γράφω"}}
    refs = {"τύπος": {"γράφω"}}
    unresolved = apply_participle_references(
        pos_map, refs, {"γράφω", "λέω"}, {"τύπος"},
        verb_refs={"τύπος": {"γράφω", "λέω"}})
    assert unresolved == {"τύπος"}


def test_table_evidence_is_a_fallback_after_unsuccessful_named_references():
    pos_map = {"τύπος": {"VERB": "γράφω"}}
    unresolved = apply_participle_references(
        pos_map, {"τύπος": {"ανύπαρκτος"}}, {"γράφω"}, {"τύπος"},
        verb_refs={})
    assert not unresolved
    assert pos_map["τύπος"]["VERB"] == "γράφω"


@pytest.mark.parametrize("reverse", [False, True])
def test_explicit_reference_cycles_preserve_all_exits(reverse):
    refs = {"πρώτος": {"γράφω"}, "δεύτερος": {"λέω"}}
    explicit = {"πρώτος": {"δεύτερος"}, "δεύτερος": {"πρώτος"}}
    if reverse:
        refs = dict(reversed(list(refs.items())))
        explicit = dict(reversed(list(explicit.items())))
    pos_map = {}
    assert apply_participle_references(
        pos_map, refs, {"γράφω", "λέω"}, set(refs),
        verb_refs=explicit) == set(refs)


def test_explicit_cycle_exit_precedes_its_top_level_and_table_exits():
    pos_map = {"πρώτος": {"VERB": "λέω"}}
    refs = {"πρώτος": {"λέω"}, "δεύτερος": {"λέω"}}
    explicit = {"πρώτος": {"δεύτερος"}, "δεύτερος": {"πρώτος", "γράφω"}}
    assert not apply_participle_references(
        pos_map, refs, {"γράφω", "λέω"}, set(refs), verb_refs=explicit)
    assert pos_map["πρώτος"]["VERB"] == pos_map["δεύτερος"]["VERB"] == "γράφω"


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


@pytest.mark.parametrize("reverse", [False, True])
def test_main_omits_unreferenced_participle_verb_but_keeps_nominal_cells(tmp_path, monkeypatch, reverse):
    import build_data
    entries = [
        {"word": "κβαντοποιημένος", "pos": "verb", "tags": ["participle"],
         "senses": [{"glosses": ["που έχει υποβληθεί σε κβαντοποίηση"]}],
         "forms": [{"form": "κβαντοποιημένη", "tags": ["feminine", "singular"]}]},
        {"word": "κβαντοποιημένος", "pos": "adj",
         "senses": [{"glosses": ["quantized"]}],
         "forms": [{"form": "κβαντοποιημένη", "tags": ["feminine", "singular"]}]},
        {"word": "γραμμένος", "pos": "verb", "tags": ["participle"],
         "senses": [{"form_of": [{"word": "γράφω"}]}],
         "forms": [{"form": "γραμμένη", "tags": ["feminine", "singular"]}]},
        {"word": "γράφω", "pos": "verb", "senses": [{"glosses": ["write"]}],
         "forms": [{"form": "γράφεις", "tags": ["present", "singular", "second-person"]}]},
    ]
    dump = _write(tmp_path, list(reversed(entries)) if reverse else entries)
    out = tmp_path / "out"
    monkeypatch.setattr(build_data, "DATA_DIR", out)
    monkeypatch.setattr(build_data, "DUMPS", {"el": {"el": dump.name}})
    monkeypatch.setattr("sys.argv", ["build_data.py", "--lang", "el", "--kaikki", str(tmp_path)])
    build_data.main()
    pairs = json.loads((out / "mg_pairs.json").read_text())
    assert not any(p["pos"] == "verb" and p["lemma"] == "κβαντοποιημένος" for p in pairs)
    assert any(p["form"] == "κβαντοποιημένη" and p["pos"] == "adj" for p in pairs)
    assert any(p["form"] == "γραμμένη" and p["lemma"] == "γράφω" for p in pairs)
    assert any(p["form"] == "γράφεις" and p["lemma"] == "γράφω" for p in pairs)


def test_standalone_participle_uses_own_reference_before_omission():
    pairs = [{"form": "γραμμένη", "lemma": "γραμμένος", "pos": "verb", "tags": []}]
    lookup = {"γραμμένος": ("γραμμένος", 3), "γράφω": ("γράφω", 3)}
    pos_map = {"γραμμένος": {"VERB": "γραμμένος"}}
    refs = include_participle_endpoints(
        pairs, lookup, pos_map, {"γραμμένος": {"verb"}, "γράφω": {"verb"}},
        {}, {"γραμμένος"})
    assert refs == {"γραμμένος": set()}
    assert not apply_participle_references(
        pos_map, refs, {"γράφω"}, {"γραμμένος"},
        verb_refs={}, participle_verb_refs={"γραμμένος": {"γράφω"}})
    assert pos_map["γραμμένος"]["VERB"] == "γράφω"


def test_standalone_participle_does_not_remove_reverse_aliased_finite_head():
    pairs = [{"form": "γράφεις", "lemma": "γράφω", "pos": "verb", "tags": []}]
    lookup = {"γραμμένος": ("γραμμένος", 3), "γράφω": ("γραμμένος", 1)}
    pos_map = {"γραμμένος": {"VERB": "γραμμένος"}, "γράφω": {"VERB": "γράφω"}}
    refs = include_participle_endpoints(
        pairs, lookup, pos_map, {"γραμμένος": {"verb"}, "γράφω": {"verb"}},
        {}, {"γραμμένος"})
    unresolved = apply_participle_references(
        pos_map, refs, {"γράφω"}, {"γραμμένος"}, verb_refs={}, participle_verb_refs={})
    roots = {}
    preserve_finite_verb_heads(pos_map, lookup, {"γράφω"}, roots, unresolved)
    assert clean_training_pairs(pairs, lookup, True, pos_map, {"γράφω": {"verb"}}, roots, unresolved) == pairs


@pytest.mark.parametrize("reverse", [False, True])
def test_main_keeps_named_verb_over_another_tables_claim(tmp_path, monkeypatch, reverse):
    import build_data
    entries = [
        {"word": "βγαίνω", "pos": "verb", "senses": [{"glosses": ["come out"]}],
         "forms": [{"form": "βγαλμένος", "tags": ["participle"]}]},
        {"word": "βγάζω", "pos": "verb", "senses": [{"glosses": ["take out"]}]},
        {"word": "βγαλμένος", "pos": "verb", "tags": ["participle"],
         "form_of": [{"word": "βγάζω"}],
         "forms": [{"form": "βγαλμένη", "tags": ["feminine"]}]},
    ]
    dump = _write(tmp_path, list(reversed(entries)) if reverse else entries)
    out = tmp_path / "out"
    monkeypatch.setattr(build_data, "DATA_DIR", out)
    monkeypatch.setattr(build_data, "DUMPS", {"el": {"el": dump.name}})
    monkeypatch.setattr("sys.argv", ["build_data.py", "--lang", "el", "--kaikki", str(tmp_path)])
    build_data.main()
    pairs = json.loads((out / "mg_pairs.json").read_text())
    assert any(p["form"] == "βγαλμένη" and p["lemma"] == "βγάζω" for p in pairs)


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


def test_source_participle_cells_are_separate_from_finite_training(tmp_path):
    dump = _write(tmp_path, [
        {"word": "μεμονωμένος", "pos": "verb", "tags": ["participle"],
         "senses": [{"tags": ["no-gloss"]}],
         "forms": [
             {"form": "μεμονωμένος", "tags": ["nominative", "masculine", "singular"]},
             {"form": "μεμονωμένα", "tags": ["nominative", "neuter", "plural"]},
             {"form": "μεμονωμένα", "tags": ["accusative", "neuter", "plural"]},
             {"form": "μεμονωμένῳ", "tags": ["dative", "masculine", "singular"]},
             {"form": "μεμονωμένο", "tags": ["neuter"]},
             {"form": "μονώνω", "tags": ["indicative", "first-person", "singular"]},
             {"form": "κακό", "tags": ["nominative", "singular", "third-person"]},
         ]},
        {"word": "άπτερος", "pos": "adj",
         "forms": [{"form": "άπτερα", "tags": ["nominative", "plural"]}]},
        {"word": "πίνω", "pos": "verb",
         "senses": [{"glosses": ["drink"]},
                    {"tags": ["no-gloss", "participle", "verb"],
                     "related": [{"word": "πιωμένος"}]}],
         "forms": [{"form": "πίνουν", "tags": ["third-person", "plural"]}]},
    ])
    cells = []
    with_cells = extract_pairs(dump, "el-el", participle_cells=cells)
    assert with_cells == extract_pairs(dump, "el-el")
    inventory = standalone_participle_inventory(cells, {"μεμονωμένος", "πίνω"})
    assert inventory["schema_version"] == 1 and inventory["language"] == "el"
    assert {r["lemma"] for r in inventory["cells"]} == {"μεμονωμένος"}
    assert {r["form"] for r in inventory["cells"]} == {
        "μεμονωμένος", "μεμονωμένα", "μεμονωμένῳ"}
    same_form = [r for r in inventory["cells"] if r["form"] == "μεμονωμένα"]
    assert len(same_form) == 2
    assert {tuple(r["tags"]) for r in same_form} == {
        ("accusative", "neuter", "participle", "plural"),
        ("neuter", "nominative", "participle", "plural"),
    }
    assert all(r["pos"] == "verb" and r["sources"] == ["el"]
               for r in inventory["cells"])
    pairs, lookup, *_ = with_cells
    cleaned = clean_training_pairs(pairs, lookup, preserve_cells=True,
                                   unresolved_verbs={"μεμονωμένος"})
    assert not any(r["lemma"] == "μεμονωμένος" for r in cleaned)
    assert any(r["lemma"] == "άπτερος" for r in cleaned)
    ancient_cells = []
    extract_pairs(dump, "grc-el", participle_cells=ancient_cells)
    assert ancient_cells == []


def test_participle_inventory_preserves_headwords_and_source_variants(tmp_path):
    dump = _write(tmp_path, [
        {"word": "ξεθεωμένος", "pos": "verb", "tags": ["participle"],
         "senses": [{"glosses": ["exhausted"]}]},
        {"word": "περιπλανώμενος", "pos": "verb",
         "senses": [{"glosses": ["wandering"], "tags": ["participle", "present"]}],
         "forms": [{"form": "περιπλανώμενη", "tags": ["feminine", "nominative", "singular"]}]},
        {"word": "λυμένος", "pos": "verb", "tags": ["participle"],
         "forms": [{"form": "λυμένη", "tags": ["feminine", "nominative", "singular"]}]},
    ])
    cells = []
    extract_pairs(dump, "el-en", participle_cells=cells)
    extract_pairs(dump, "el-el", participle_cells=cells)
    wanted = {"ξεθεωμένος", "περιπλανώμενος"}
    first = standalone_participle_inventory(cells, wanted)
    assert first == standalone_participle_inventory(list(reversed(cells)), wanted)
    assert {r["lemma"] for r in first["cells"]} == wanted
    assert all(r["sources"] == ["el", "en"] for r in first["cells"])
    head = next(r for r in first["cells"] if r["lemma"] == "ξεθεωμένος")
    assert head["form"] == "ξεθεωμένος" and head["kind"] == "headword"
    assert head["tags"] == ["participle"]
    # A sense's explicit grammatical annotation is retained, not inferred.
    head = next(r for r in first["cells"] if r["kind"] == "headword"
                and r["lemma"] == "περιπλανώμενος")
    assert head["tags"] == ["participle", "present"]


def test_resolved_participle_owner_precedes_its_unresolved_alias(tmp_path):
    # καθούμενος has a supported κάθομαι path, while its spelling alias
    # καθήμενος is a standalone participle with no finite parent.
    dump = _write(tmp_path, [
        {"word": "κάθομαι", "pos": "verb",
         "senses": [{"glosses": ["sit"]}]},
        {"word": "καθήμενος", "pos": "verb", "tags": ["participle"],
         "senses": [{"glosses": ["sitting"]}]},
        {"word": "καθούμενος", "pos": "verb", "tags": ["participle"],
         "form_of": [{"word": "κάθομαι"}],
         "senses": [{"alt_of": [{"word": "καθήμενος"}]}],
         "forms": [{"form": "καθούμενα", "tags": ["neuter", "nominative", "plural"]}]},
    ])
    refs, heads, participles, pos_evidence = {}, set(), set(), {}
    verb_refs, own_refs = {}, {}
    pairs, lookup, _, pos_map, *_ = extract_pairs(
        dump, "el-el", lemma_pos=pos_evidence, participle_refs=refs,
        verb_heads=heads, participle_heads=participles,
        verb_refs=verb_refs, participle_verb_refs=own_refs)
    refs["καθήμενος"] = set()
    unresolved = apply_participle_references(
        pos_map, refs, heads, participles, pos_evidence, verb_refs, own_refs)
    roots = {w: pos_map[w]["VERB"] for w in refs if w not in unresolved}
    assert roots["καθούμενος"] == "κάθομαι" and "καθήμενος" in unresolved
    row = next(p for p in pairs if p["form"] == "καθούμενα")
    assert row["via"] == "καθήμενος"
    cleaned = clean_training_pairs(
        [row], lookup, preserve_cells=True, pos_map=pos_map,
        lemma_pos=pos_evidence, participle_roots=roots,
        unresolved_verbs=unresolved)
    assert cleaned == [{"form": "καθούμενα", "lemma": "κάθομαι",
                        "pos": "verb", "tags": ["neuter", "nominative", "plural"]}]
    # Without positive own-head evidence the unresolved alias still blocks.
    assert clean_training_pairs(
        [row], lookup, preserve_cells=True, pos_map=pos_map,
        lemma_pos=pos_evidence, participle_roots={},
        unresolved_verbs=unresolved) == []


@pytest.mark.parametrize("reverse", [False, True])
def test_independent_noun_table_keeps_its_owner_beside_form_homograph(tmp_path, reverse):
    entries = [
        {"word": "σκιά", "pos": "noun", "senses": [{"glosses": ["shadow"]}],
         "forms": [{"form": "σκιά", "tags": ["nominative", "singular"]},
                   {"form": "σκιάς", "tags": ["genitive", "singular"]}]},
        {"word": "σκιάς", "pos": "noun", "senses": [{"glosses": ["robber"]}],
         "forms": [{"form": "σκιάς", "tags": ["masculine"]},
                   {"form": "σκιάς", "tags": ["nominative", "singular"]},
                   {"form": "σκιά", "tags": ["genitive", "singular"]}]},
        {"word": "σκιάς", "pos": "noun",
         "senses": [{"glosses": ["genitive singular of σκιά"],
                     "form_of": [{"word": "σκιά"}]}],
         "forms": [{"form": "σκιάς", "tags": ["feminine"]}]},
    ]
    pairs, _, *_ = extract_pairs(_write(tmp_path, entries[::-1] if reverse else entries), "el-el")
    # The flat lookup is allowed to choose the feminine reading. It cannot
    # change the separate masculine source table's owner.
    lookup = {"σκιά": ("σκιά", 3), "σκιάς": ("σκιά", 1)}
    cleaned = clean_training_pairs(pairs, lookup, preserve_cells=True)
    cells = {(r["form"], r["lemma"], tuple(sorted(r["tags"]))) for r in cleaned}
    assert ("σκιά", "σκιάς", ("genitive", "singular")) in cells
    assert ("σκιάς", "σκιάς", ("nominative", "singular")) in cells
    assert ("σκιάς", "σκιά", ("genitive", "singular")) in cells
    assert ("σκιάς", "σκιά", ("feminine",)) in cells
    assert all(set(r) == {"form", "lemma", "pos", "tags"} for r in cleaned)


def test_noun_ownership_does_not_override_form_or_alternative_references(tmp_path):
    entries = [
        {"word": "τύπου", "pos": "noun",
         "senses": [{"glosses": ["genitive of τύπος"],
                     "form_of": [{"word": "τύπος"}]}],
         "forms": [{"form": "τύπου", "tags": ["genitive", "singular"]}]},
        {"word": "τυπάκι", "pos": "noun",
         "senses": [{"glosses": ["alternative spelling of τυπάκιο"],
                     "alt_of": [{"word": "τυπάκιο"}]}],
         "forms": [{"form": "τυπάκια", "tags": ["nominative", "plural"]}]},
        {"word": "Μεγαλειότατε", "pos": "noun",
         "tags": ["form-of", "singular", "vocative"],
         "form_of": [{"word": "μεγαλειότατος"}],
         "senses": [{"glosses": ["addressing an emperor or king"]}],
         "forms": [{"form": "Μεγαλειότατε", "tags": ["masculine"]}]},
        {"word": "τίτλε", "pos": "noun",
         "senses": [{"tags": ["form-of"], "glosses": ["addressing someone by title"]}],
         "forms": [{"form": "τίτλε", "tags": ["vocative"]}]},
        {"word": "τύπωμα", "pos": "noun", "tags": ["alternative"],
         "senses": [{"glosses": ["a printed impression"]}],
         "forms": [{"form": "τύπωμα", "tags": ["nominative", "singular"]}]},
    ]
    pairs, _, *_ = extract_pairs(_write(tmp_path, entries), "el-el")
    assert not any(r.get("noun_owner") for r in pairs)
    lookup = {"τύπου": ("τύπος", 1), "τύπος": ("τύπος", 3),
              "τυπάκι": ("τυπάκιο", 1), "τυπάκιο": ("τυπάκιο", 3),
              "Μεγαλειότατε": ("μεγαλειότατος", 1),
              "μεγαλειότατος": ("μεγαλειότατος", 3),
              "τίτλε": ("τίτλος", 1), "τίτλος": ("τίτλος", 3),
              "τύπωμα": ("τύπος", 1)}
    cleaned = clean_training_pairs(pairs, lookup, preserve_cells=True)
    assert {r["lemma"] for r in cleaned} == {"τύπος", "τυπάκιο", "μεγαλειότατος", "τίτλος"}
    ancient, *_ = extract_pairs(_write(tmp_path, entries), "grc-el")
    assert not any(r.get("noun_owner") for r in ancient)


def test_internal_noun_ownership_survives_deduplication():
    row = {"form": "σκιά", "lemma": "σκιάς", "pos": "noun",
           "tags": ["genitive", "singular"]}
    owned = {**row, "noun_owner": True}
    lookup = {"σκιά": ("σκιά", 3), "σκιάς": ("σκιά", 1)}
    expected = {("σκιά", "σκιά"), ("σκιά", "σκιάς")}
    for pairs in ([row, owned], [owned, row]):
        assert len(deduplicate_pairs(pairs, preserve_cells=True)) == 2
        assert {(r["form"], r["lemma"]) for r in clean_training_pairs(
            pairs, lookup, preserve_cells=True)} == expected


def _noun_evidence(tmp_path):
    records = []
    entry = {"word": "σκιάς", "pos": "noun", "senses": [{"glosses": ["robber"]}],
             "forms": [{"form": "σκιά", "tags": ["genitive", "singular"]}]}
    extract_pairs(_write(tmp_path, [entry]), "el-el", noun_heads=records)
    return noun_head_inventory(records, {"σκιάς": ("σκιά", 1)}, {"el": "a" * 64})


def test_noun_head_evidence_is_deterministic_and_authorizes_only_exact_nouns(tmp_path):
    records = []
    entries = [{"word": "σκιάς", "pos": "noun",
                "senses": [{"glosses": [gloss]}]} for gloss in ["robber", "uncouth man"]]
    extract_pairs(_write(tmp_path, entries), "el-el", noun_heads=records)
    lookup = {"σκιάς": ("σκιά", 1), "σκιά": ("σκιά", 3)}
    inventory = noun_head_inventory(records + records, lookup, {"el": "a" * 64})
    assert inventory == noun_head_inventory(records[::-1], lookup, {"el": "a" * 64})
    assert len(inventory["heads"][0]["evidence"]) == 2
    heads = validate_noun_head_inventory(inventory)
    pairs = [{"form": "σκιά", "lemma": "σκιάς", "pos": "noun"},
             {"form": "σκιά", "lemma": "σκιάς", "pos": "verb"}]
    assert training_pair_headword_errors(pairs, {k: v[0] for k, v in lookup.items()}, heads)[0] == 1
    assert training_pair_headword_errors(pairs, {k: v[0] for k, v in lookup.items()})[0] == 2
    assert noun_head_inventory(records, {"σκιάς": ("σκιάς", 3)}, {"el": "a" * 64})["heads"] == []


@pytest.mark.parametrize("change", [
    "wrong_head", "wrong_pos", "wrong_source", "missing_hash", "empty_evidence",
    "missing_tags", "empty_gloss", "no_gloss", "form_tag", "alternative_tag",
    "misspelling_tag", "sense_form", "sense_alternative", "sense_form_tag",
    "reference_gloss", "top_reference", "duplicate_head",
])
def test_noun_head_evidence_rejects_missing_or_forged_support(tmp_path, change):
    inventory = _noun_evidence(tmp_path)
    row = inventory["heads"][0]
    evidence = row["evidence"][0]
    sense = evidence["senses"][0]
    if change == "wrong_head": evidence["headword"] = "σκιά"
    elif change == "wrong_pos": evidence["pos"] = "verb"
    elif change == "wrong_source": evidence["source"] = "other"
    elif change == "missing_hash": inventory["source_sha256"] = {}
    elif change == "empty_evidence": row["evidence"] = []
    elif change == "missing_tags": del evidence["tags"]
    elif change == "empty_gloss": sense["glosses"] = []
    elif change == "no_gloss": sense["tags"] = ["no-gloss"]
    elif change == "form_tag": evidence["tags"] = ["form-of"]
    elif change == "alternative_tag": evidence["tags"] = ["alternative"]
    elif change == "misspelling_tag": evidence["tags"] = ["misspelling"]
    elif change == "sense_form": sense["form_of"] = [{"word": "σκιά"}]
    elif change == "sense_alternative": sense["alt_of"] = [{"word": "σκιά"}]
    elif change == "sense_form_tag": sense["tags"] = ["form-of"]
    elif change == "reference_gloss": sense["glosses"] = ["genitive singular of σκιά"]
    elif change == "top_reference":
        evidence["source"] = "en"
        inventory["source_sha256"] = {"en": "b" * 64}
        evidence["form_of"] = [{"word": "σκιά"}]
    elif change == "duplicate_head": inventory["heads"].append(row)
    with pytest.raises(ValueError, match="noun-head evidence"):
        validate_noun_head_inventory(inventory)


def test_noun_evidence_excludes_form_tags_but_keeps_independent_el_etymology(tmp_path):
    records = []
    entries = [
        {"word": "ποτήρι", "pos": "noun", "form_of": [{"word": "ποτήριον"}],
         "senses": [{"glosses": ["drinking glass"]}]},
        {"word": "Μεγαλειότατε", "pos": "noun", "tags": ["form-of"],
         "senses": [{"glosses": ["addressing an emperor or king"]}]},
        {"word": "τύπου", "pos": "noun", "senses": [{"form_of": [{"word": "τύπος"}]}]},
    ]
    extract_pairs(_write(tmp_path, entries), "el-el", noun_heads=records)
    assert [r["lemma"] for r in records] == ["ποτήρι"]
    inventory = noun_head_inventory(records, {}, {"el": "a" * 64})
    assert validate_noun_head_inventory(inventory) == {"ποτήρι"}


def test_main_emits_noun_head_companion_with_exact_input_hashes(tmp_path, monkeypatch):
    import hashlib
    import build_data

    source = tmp_path / "source"
    source.mkdir()
    en = source / "en.jsonl"
    el = source / "el.jsonl"
    en.write_text(json.dumps({"word": "σκιάς", "pos": "noun", "senses": [
        {"form_of": [{"word": "σκιά"}], "glosses": ["genitive of σκιά"]}
    ]}, ensure_ascii=False) + "\n", encoding="utf-8")
    entries = [
        {"word": "σκιά", "pos": "noun", "senses": [{"glosses": ["shadow"]}],
         "forms": [{"form": "σκιάς", "tags": ["genitive", "singular"]}]},
        {"word": "σκιάς", "pos": "noun", "senses": [{"glosses": ["robber"]}],
         "forms": [{"form": "σκιά", "tags": ["genitive", "singular"]}]},
    ]
    el.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8")
    output = tmp_path / "data"
    monkeypatch.setattr(build_data, "DATA_DIR", output)
    monkeypatch.setattr(build_data, "DUMPS", {"el": {"en": "en.jsonl", "el": "el.jsonl"}})
    monkeypatch.setattr(build_data.sys, "argv", ["build_data.py", "--lang", "el", "--kaikki", str(source)])
    build_data.main()
    inventory = json.loads((output / "el_noun_heads.json").read_text())
    assert inventory["source_sha256"] == {
        "en": hashlib.sha256(en.read_bytes()).hexdigest(),
        "el": hashlib.sha256(el.read_bytes()).hexdigest(),
    }
    assert validate_noun_head_inventory(inventory) == {"σκιάς"}
    pairs = json.loads((output / "mg_pairs.json").read_text())
    lookup = json.loads((output / "mg_lookup.json").read_text())
    assert lookup["σκιάς"] == "σκιά"
    assert training_pair_headword_errors(pairs, lookup)[0] > 0
    assert training_pair_headword_errors(pairs, lookup, {"σκιάς"})[0] == 0
    assert (output / "el_participles.json").exists()
    assert all(set(p) == {"form", "lemma", "pos", "tags"} for p in pairs)


def test_verb_head_evidence_requires_independent_finite_entries(tmp_path):
    from build.build_el_verb_heads import build_inventory, required_verb_heads
    from build_data import validate_verb_head_inventory

    entries = [
        {"word": "μεθάω", "pos": "verb", "senses": [{"glosses": ["become drunk"]}]},
        {"word": "μεθάω", "pos": "noun", "senses": [{"glosses": ["a nominal reading"]}]},
        {"word": "μεθυσμένος", "pos": "verb", "tags": ["participle"],
         "senses": [{"glosses": ["drunk"]}]},
        {"word": "παθών", "pos": "verb", "senses": [{"tags": ["participle"], "glosses": ["sufferer"]}]},
        {"word": "τύπου", "pos": "verb", "senses": [{"glosses": ["form of τύπτω"], "form_of": [{"word": "τύπτω"}]}]},
        {"word": "λόγος", "pos": "noun", "senses": [{"glosses": ["word"]}]},
    ]
    path = _write(tmp_path, entries)
    inventory = build_inventory({"μεθάω"}, {"el": path})
    assert validate_verb_head_inventory(inventory) == {"μεθάω"}
    assert len(inventory["heads"][0]["evidence"]) == 1
    for unsupported in ["μεθυσμένος", "παθών", "τύπου", "λόγος", "ανύπαρκτος"]:
        with pytest.raises(ValueError, match="No independent finite verb evidence"):
            build_inventory({unsupported}, {"el": path})
    lookup = {"μεθάω": "μεθώ", "μεθώ": "μεθώ"}
    pairs = [{"form": "μεθυσμένος", "lemma": "μεθάω", "pos": "verb"},
             {"form": "μεθυσμένος", "lemma": "μεθάω", "pos": "noun"}]
    assert required_verb_heads(pairs, lookup) == {"μεθάω"}
    assert training_pair_headword_errors(pairs, lookup, verb_heads={"μεθάω"})[0] == 1
    assert training_pair_headword_errors(pairs, lookup, noun_heads={"μεθάω"})[0] == 1
    assert training_pair_headword_errors(pairs, lookup)[0] == 2


@pytest.mark.parametrize("kind", ["entry_participle", "sense_participle", "noun", "reference", "missing"])
def test_verb_head_validator_rejects_forged_nonfinite_evidence(tmp_path, kind):
    from build_data import validate_verb_head_inventory
    inventory = _noun_evidence(tmp_path)
    evidence = inventory["heads"][0]["evidence"][0]
    evidence["pos"] = "verb"
    for sense in evidence["senses"]:
        sense["related"] = []
    if kind == "entry_participle": evidence["tags"] = ["participle"]
    elif kind == "sense_participle": evidence["senses"][0]["tags"] = ["participle"]
    elif kind == "noun": evidence["pos"] = "noun"
    elif kind == "reference": evidence["senses"][0]["alt_of"] = [{"word": "άλλο"}]
    elif kind == "missing": inventory["heads"][0]["evidence"] = []
    with pytest.raises(ValueError, match="verb-head evidence"):
        validate_verb_head_inventory(inventory)


def test_verb_head_inventory_keeps_ordinary_head_with_related_participle_annotation(tmp_path):
    from build.build_el_verb_heads import build_inventory
    from build_data import validate_verb_head_inventory
    entries = [{"word": "πίνω", "pos": "verb", "senses": [
        {"glosses": ["drink"]},
        {"tags": ["no-gloss", "participle", "verb"], "related": [{"word": "πιωμένος"}]},
    ]}]
    assert validate_verb_head_inventory(build_inventory({"πίνω"}, {"el": _write(tmp_path, entries)})) == {"πίνω"}


@pytest.mark.parametrize("gloss", [
    "third-person singular present indicative of τρέχω",
    "first-person singular aorist active indicative of τρέχω",
    "άκλιτη μετοχή ενεργητικού ενεστώτα του τρέχω",
    "active nonfinite form of δίδω/δίνω",
    "passive of περιπλέκω",
    "aorist passive participle of γράφω",
])
@pytest.mark.parametrize("empty_first", [False, True])
def test_verb_head_evidence_rejects_unstructured_inflection_senses(tmp_path, gloss, empty_first):
    from build.build_el_verb_heads import build_inventory
    senses = ([{"tags": ["no-gloss"]}] if empty_first else []) + [{"glosses": [gloss]}]
    path = _write(tmp_path, [{"word": "τρέχει", "pos": "verb", "senses": senses}])
    with pytest.raises(ValueError, match="No independent finite verb evidence"):
        build_inventory({"τρέχει"}, {"el": path})


@pytest.mark.parametrize("reverse", [False, True])
def test_no_gloss_verb_headers_follow_their_explicit_finite_parent(tmp_path, reverse):
    entries = [
        {"word": "περιπλέκω", "pos": "verb", "senses": [{"glosses": ["entangle"]}]},
        {"word": "τακτοποιώ", "pos": "verb", "senses": [{"glosses": ["put in order"]}]},
        {"word": "επιτυγχάνω", "pos": "verb", "senses": [{"glosses": [
            "alternative form of πετυχαίνω", "to succeed, accomplish, achieve"]}]},
        {"word": "περιπλέκομαι", "pos": "verb", "senses": [{"tags": ["no-gloss"],
             "related": [{"word": "περιπλέκω"}]}], "forms": [{"form": "περιπλέκομαι", "source": "header"}]},
        {"word": "ταχτοποιώ", "pos": "verb", "senses": [{"tags": ["no-gloss"],
             "related": [{"word": "τακτοποιώ"}]}], "forms": [{"form": "ταχτοποιώ", "source": "header"}]},
        {"word": "περιπλέκομαι", "pos": "verb", "senses": [{"tags": ["form-of", "passive"],
             "form_of": [{"word": "περιπλέκω"}], "glosses": ["passive of περιπλέκω"]}]},
        {"word": "ταχτοποιώ", "pos": "verb", "senses": [{"tags": ["alternative"],
             "alt_of": [{"word": "τακτοποιώ"}], "glosses": ["alternative form of τακτοποιώ"]}]},
        {"word": "περιπεπλεγμένος", "pos": "verb", "tags": ["participle"],
         "form_of": [{"word": "περιπλέκομαι"}], "senses": [{"glosses": ["entangled"]}]},
        {"word": "ταχτοποιημένος", "pos": "verb", "tags": ["participle"],
         "form_of": [{"word": "ταχτοποιώ"}], "senses": [{"glosses": ["put in order"]}]},
    ]
    refs, heads, participles, lemma_pos, verb_refs, own_refs = {}, set(), set(), {}, {}, {}
    _, _, _, pos_map, *_ = extract_pairs(_write(tmp_path, entries[::-1] if reverse else entries), "el-el",
        participle_refs=refs, verb_heads=heads, participle_heads=participles,
        lemma_pos=lemma_pos, verb_refs=verb_refs, participle_verb_refs=own_refs)
    assert "περιπλέκομαι" not in heads and "ταχτοποιώ" not in heads
    assert "επιτυγχάνω" in heads
    assert not apply_participle_references(pos_map, refs, heads, participles, lemma_pos, verb_refs, own_refs)
    assert pos_map["περιπεπλεγμένος"]["VERB"] == "περιπλέκω"
    assert pos_map["ταχτοποιημένος"]["VERB"] == "τακτοποιώ"


def test_finite_definition_containing_of_is_not_an_inflection_reference():
    from build_data import is_independent_finite_entry
    assert is_independent_finite_entry({"word": "διοικώ", "pos": "verb", "senses": [
        {"glosses": ["to be in charge of a group"]}]}, "en")
