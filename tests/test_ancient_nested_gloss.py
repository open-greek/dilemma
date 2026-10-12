"""Nested form translations do not establish independent Ancient verb heads."""

import pytest

from build_data import is_independent_lexical_entry


def test_nested_translation_does_not_define_an_ancient_finite_head():
    entry = {'word': 'ἦλθον', 'pos': 'verb', 'senses': [
        {'glosses': ['first-person singular/third-person plural aorist indicative active of ἔρχομαι (érkhomai):', 'came']},
        {'glosses': ['first-person singular/third-person plural aorist indicative active of ἔρχομαι (érkhomai):', 'went']},
    ]}
    assert not is_independent_lexical_entry(entry, 'en')
    entry['senses'].append({'glosses': ['to arrive unexpectedly']})
    assert is_independent_lexical_entry(entry, 'en')


@pytest.mark.parametrize('word,parent,meaning', [
    ('λανθάνομαι', 'λανθάνω', 'ξεχνάω, λησμονώ'),
    ('ἀποδίδομαι', 'ἀποδίδωμι', 'πουλώ'),
    ('τυρβάζομαι', 'τυρβάζω', 'ταράζομαι, αναταράζομαι, αναστατώνομαι'),
])
def test_greek_present_head_retains_its_own_definition(word, parent, meaning):
    entry = {'word': word, 'pos': 'verb', 'senses': [{'glosses': [
        f'α΄ πρόσωπο ενικού οριστικής μεσοπαθητικού ενεστώτα του {parent}', meaning]}]}
    assert is_independent_lexical_entry(entry, 'el')


def test_lexical_definition_preceding_grammatical_note_survives():
    entry = {'word': 'δέω', 'pos': 'verb', 'senses': [{'glosses': [
        'to bind', 'first-person singular present indicative of δέω']}]}
    assert is_independent_lexical_entry(entry, 'en')
