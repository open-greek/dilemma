"""Nominal morphology labels do not supply independent lexical definitions."""
import pytest
from build_data import is_independent_lexical_entry
from build.build_grc_lexical_heads import has_definition

@pytest.mark.parametrize("word,pos,gloss", [
    ("δεσμά", "noun", "neuter heteroclitic nominative/accusative/vocative plural of δεσμός m (desmós)"),
    ("τράχηλα", "noun", "neuter heteroclitic nominative/accusative/vocative plural of τρᾰ́χηλος m (trắkhēlos)"),
    ("ἐπιθυμίας", "noun", "genitive singular/accusative plural of ἐπιθῡμίᾱ (epithūmíā)"),
    ("ξένων", "adj", "genitive plural all genders of ξένος (xénos)"),
    ("γένεα", "noun", "nominative/accusative/vocative plural uncontracted of γένος (génos)"),
    ("υἷε", "noun", "nominative /accusative/vocative dual of υἱός (huiós)"),
    ("σχήμασι", "noun", "preconsonantal dative plural of σχῆμᾰ (skhêmă)"),
    ("σχήμασιν", "noun", "prevocalic/clause-final dative plural of σχῆμᾰ (skhêmă)"),
])
def test_nominal_inflection_description_is_not_a_definition(word,pos,gloss):
    entry={"word":word,"pos":pos,"senses":[{"glosses":[gloss]}]}
    assert not is_independent_lexical_entry(entry,"en")
    assert not has_definition(entry,"en")
    entry["senses"].append({"glosses":["a separate lexical meaning"]})
    assert is_independent_lexical_entry(entry,"en")
    assert has_definition(entry,"en")

@pytest.mark.parametrize("word,pos,gloss", [
    ("ἰχθυήματα","noun","scrapings; shavings"),
    ("Σπευσίνιοι","noun","a police force in Athens"),
    ("τύϊγγα","noun","some kind of small bird"),
    ("σεργοί","noun","deer"),
    ("φύσαντες","noun","parents, ancestors"),
    ("ἀραία","adj","narrow, slender"),
])
def test_declared_form_header_does_not_erase_lexical_meaning(word,pos,gloss):
    entry={"word":word,"pos":pos,"head_templates":[{"name":"grc-noun form" if pos=="noun" else "grc-adj form"}],"senses":[{"glosses":[gloss]}]}
    assert is_independent_lexical_entry(entry,"en")
    assert has_definition(entry,"en")

@pytest.mark.parametrize("gloss", [
    "codical variant of ζῦθος (zûthos)",
    "Cypriot spelling of μισθῶν (misthôn); genitive plural of μισθός (misthós).",
    "a textual error for πρώτῳ (prṓtōi)",
    "nomen sacrum form of κῡ́ριος (kū́rios)",
])
def test_explicit_spelling_reference_is_not_a_definition(gloss):
    entry={"word":"λέξη","pos":"noun","senses":[{"glosses":[gloss]}]}
    assert not is_independent_lexical_entry(entry,"en")
    assert not has_definition(entry,"en")
    entry["senses"].append({"glosses":["a distinct lexical meaning"]})
    assert is_independent_lexical_entry(entry,"en")
    assert has_definition(entry,"en")

@pytest.mark.parametrize("template", [
    {"name":"grc-noun form"},
    {"name":"grc-adj form"},
    {"name":"head","args":{"1":"grc","2":"noun form"}},
])
def test_unexpanded_inflection_template_is_not_a_definition(template):
    entry={"word":"λέξη","pos":"noun","head_templates":[template],"senses":[{"glosses":["Template:infl of"]}]}
    assert not is_independent_lexical_entry(entry,"en")
    assert not has_definition(entry,"en")
    entry["senses"].append({"glosses":["a distinct lexical meaning"]})
    assert is_independent_lexical_entry(entry,"en")
    assert has_definition(entry,"en")
