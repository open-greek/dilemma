"""Contracted and uncontracted form labels are not lexical definitions."""
import pytest
from build_data import _is_verb_inflection_gloss, is_independent_finite_entry, is_independent_lexical_entry
from dilemma_build.build_grc_lexical_heads import has_definition

@pytest.mark.parametrize("word,gloss", [
    ("καλέουσι", "uncontracted third-person plural present active indicative of καλέω (kaléō)"),
    ("καλέουσιν", "uncontracted third-person plural present active indicative of καλέω (kaléō) (with movable nu)"),
    ("καλοῦσι", "contracted third-person plural present active indicative of καλέω"),
    ("καλέον", "uncontracted present active participle of καλέω"),
    ('δύναιτο', 'late third-person singular aorist middle optative of δύνω (dúnō)'),
    ('ἔστιν', 'paroxytone third-person singular present indicative active of εἰμί (eimí) with movable nu'),
    ('ἔστι', 'paroxytone third-person singular present indicative active of εἰμῐ́ (eimĭ́)'),
    ('ἐξελῃτούργησεν', 'prevocalic/clause-final third-person singular aorist active indicative of ἐκλῃτουργέω (eklēitourgéō)'),
    ('ἀνέθηκεν', 'prevocalic/clause-final third-person singular aorist active indicative of ᾰ̓νᾰτῐ́θημῐ (ănătĭ́thēmĭ)'),
    ('ἔτεμεν', 'prevocalic/clause-final third-person singular aorist active indicative of τέμνω (témnō)'),
    ('συνέτεμεν', 'prevocalic/clause-final third-person singular aorist active indicative of σῠντέμνω (sŭntémnō) with movable nu'),
    ("βῆ", "unaugmented third-person singular aorist active indicative of βαίνω"),
    ("καλέει", "late uncontracted third-person singular present active indicative of καλέω"),
])
def test_prefixed_verb_form_is_not_a_lexical_head(word, gloss):
    assert _is_verb_inflection_gloss(gloss)
    entry = {"word": word, "pos": "verb", "head_templates": [{"name": "grc-verb form"}],
             "senses": [{"glosses": [gloss]}]}
    assert not is_independent_finite_entry(entry, "en")
    assert not is_independent_lexical_entry(entry, "en")
    assert not has_definition(entry, "en")

@pytest.mark.parametrize("gloss", [
    "uncontracted obligation", "contracted muscles", "contracted to name someone", "late arrival", "paroxytone pronunciation",
    "unaugmented income", "prevocalic/clause-final position",
])
def test_prefix_alone_does_not_establish_verb_inflection(gloss):
    assert not _is_verb_inflection_gloss(gloss)
    entry = {"word": "λέξη", "pos": "verb", "senses": [{"glosses": [gloss]}]}
    assert is_independent_finite_entry(entry, "en")
    assert is_independent_lexical_entry(entry, "en")
    assert has_definition(entry, "en")


def test_verb_form_header_can_retain_a_separate_lexical_sense():
    entry = {"word": "λέξη", "pos": "verb", "head_templates": [{"name": "grc-verb form"}],
             "senses": [{"glosses": ["uncontracted third-person plural present active indicative of καλέω"]},
                        {"glosses": ["to name or summon"]}]}
    assert is_independent_finite_entry(entry, "en")
    assert is_independent_lexical_entry(entry, "en")
    assert has_definition(entry, "en")


@pytest.mark.parametrize("word,gloss", [
    ("ἐγώ", "first person singular personal pronoun: I, me, my"),
    ("σύ", "second person singular personal pronoun: thou, you"),
    ("ὑμεῖς", "second person plural personal pronoun: you, ye"),
    ("σφέτερος", "third-person plural possessive pronoun, including reflexive"),
])
def test_person_pronoun_definition_is_a_lexical_head(word, gloss):
    # The 2026-11 dumps define ἐγώ and σύ by these glosses alone; read as
    # inflection glosses they dropped out of the Wiktionary heads (#125).
    assert not _is_verb_inflection_gloss(gloss)
    entry = {"word": word, "pos": "pron", "senses": [{"glosses": [gloss]}]}
    assert has_definition(entry, "en")
