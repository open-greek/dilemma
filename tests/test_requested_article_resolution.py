"""An explicitly resolved marked article cannot become a nominal POS target."""
import unicodedata

import pytest

from dilemma import core


def model(*, enabled=True, convention=None, language='grc', target='Τέ', gate=True):
    instance = core.Dilemma.__new__(core.Dilemma)
    instance.lang = language
    instance._resolve_articles = enabled
    instance._convention_name = convention
    instance._pos_rejected_targets = frozenset()
    instance._apply_convention = lambda lemma: lemma
    instance._resolve_attested_form = lambda word: word if gate else None
    instance._pos_table_lookup = lambda word, upos: target
    instance._lemmatize_verbose_impl = lambda word, *, guess: [
        core.LemmaCandidate(lemma='ὁ', source='article', score=1.0)]
    instance.lemmatize_batch = lambda words, **kwargs: [
        None if kwargs.get('attested_only') and not gate else 'ὁ'] * len(words)
    instance.lemmatize = lambda word, **kwargs: 'ὁ'
    return instance


@pytest.mark.parametrize('form', ['Τὸ', 'Τὸν', 'Τὴν', 'Ταῖς', 'Τοὺς', 'Τοῦ'])
@pytest.mark.parametrize('upos', ['NOUN', 'PROPN'])
@pytest.mark.parametrize('normalization', ['NFC', 'NFD'])
def test_requested_marked_articles_survive_nominal_tags(form, upos, normalization):
    instance = model()
    word = unicodedata.normalize(normalization, form)
    assert instance.lemmatize_pos(word, upos, guess=False) == 'ὁ'
    assert instance.lemmatize_batch_pos([word], [upos], guess=False) == ['ὁ']


def test_batch_case_tiebreak_cannot_replace_the_resolved_article():
    instance = model(target='Ὁ')
    assert instance.lemmatize_pos('Τὸ', 'PROPN', guess=False) == 'ὁ'
    assert instance.lemmatize_batch_pos(['Τὸ'], ['PROPN'], guess=False) == ['ὁ']


@pytest.mark.parametrize('form,upos,target', [
    ('με', 'PRON', 'με'), ('σοῦ', 'ADJ', 'σός'), ('αἳ', 'PRON', 'ὅς'),
    ('αἲ', 'INTJ', 'αἴ'), ('η', 'PROPN', 'Η'), ('το', 'NOUN', 'τόπος'),
])
def test_independent_readings_are_not_overridden(form, upos, target):
    instance = model(target=target)
    assert instance._article_pos_lemma(form, upos) is None
    assert instance.lemmatize_pos(form, upos, guess=False) == target
    assert instance.lemmatize_batch_pos([form], [upos], guess=False) == [target]


@pytest.mark.parametrize('configuration', [
    {'enabled': False}, {'convention': 'triantafyllidis'}, {'language': 'el'},
])
def test_other_resolution_settings_keep_their_pos_selection(configuration):
    instance = model(**configuration)
    assert instance._article_pos_lemma('Τὸ', 'PROPN') is None
    assert instance.lemmatize_pos('Τὸ', 'PROPN', guess=False) == 'Τέ'
    assert instance.lemmatize_batch_pos(['Τὸ'], ['PROPN'], guess=False) == ['Τέ']


def test_input_attestation_gate_still_blocks_a_recognized_article():
    instance = model(gate=False)
    assert instance.lemmatize_pos('Τὸ', 'PROPN', attested_only=True, guess=False) is None
    assert instance.lemmatize_batch_pos(['Τὸ'], ['PROPN'], attested_only=True, guess=False) == [None]


def test_target_veto_still_blocks_the_canonical_article():
    instance = model()
    instance._pos_rejected_targets = frozenset({'ὁ'})
    assert instance.lemmatize_pos('Τὸ', 'PROPN', guess=False) is None
    assert instance.lemmatize_batch_pos(['Τὸ'], ['PROPN'], guess=False) == [None]
