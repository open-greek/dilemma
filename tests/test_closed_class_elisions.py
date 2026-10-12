import copy

import pytest

from build.closed_class_elisions import apply_repairs, plan_repairs


def evidence():
    entry = {'word': 'οὗτος', 'pos': 'pron', 'lang_code': 'grc',
             'senses': [{'glosses': ['this; this one']}],
             'forms': [{'form': 'ταῦτα',
                        'tags': ['neuter', 'plural', 'nominative', 'accusative']}]}
    pair = {'form': 'Ταῦτ᾽', 'lemma': 'οὗτος', 'pos': 'pron',
            'tags': ['neuter', 'plural', 'nominative']}
    return {'reference': {'Ταῦτ’': 'Ταύς'}, 'reference_base': {}, 'predecessor': {},
            'source_pairs': [('diorisis', [pair]), ('glaux', [
                {'form': 'Ταῦτ’', 'lemma': 'Ταύς', 'pos': 'noun',
                 'tags': ['masculine', 'singular', 'accusative']}])],
            'lexical_heads': {'οὗτος'},
            'fresh_entries': [{'source': 'en', 'line': 1, 'entry': entry}]}


def plan(values):
    return plan_repairs(**values)


def test_proven_elision_is_canonicalized_and_reversible():
    values = evidence()
    decisions = plan(values)
    assert len(decisions) == 1 and decisions[0]['accepted']
    assert len(decisions[0]['source_readings']) == 2
    target = dict(values['reference'])
    audit = apply_repairs(target, decisions)
    assert target == {'Ταῦτ’': 'οὗτος'}
    for row in audit['canonicalized']:
        target[row['form']] = row['current_owner']
    assert target == values['reference']


@pytest.mark.parametrize('mark', ["'", '`', 'ʼ', '᾽', '᾿', '’', '̓', '̔'])
def test_actual_source_marker_encodings_keep_the_same_proof(mark):
    values = evidence()
    values['source_pairs'][0][1][0]['form'] = 'Ταῦτ' + mark
    decision = plan(values)[0]
    assert decision['accepted'] and decision['owner'] == 'οὗτος'


def test_reversed_sources_and_entries_preserve_the_owner_decision():
    first = evidence()
    second = copy.deepcopy(first)
    second['source_pairs'].reverse()
    second['fresh_entries'].reverse()
    assert [(row['form'], row['accepted'], row.get('owner')) for row in plan(first)] == [
        (row['form'], row['accepted'], row.get('owner')) for row in plan(second)]


@pytest.mark.parametrize('pos', ['pron', 'det'])
def test_both_supported_closed_pos_are_eligible(pos):
    values = evidence()
    values['fresh_entries'][0]['entry']['pos'] = pos
    values['source_pairs'][0][1][0]['pos'] = pos
    assert plan(values)[0]['accepted']


@pytest.mark.parametrize('where', ['reference_base', 'predecessor'])
@pytest.mark.parametrize('encoding', ['Ταῦτ’', 'Ταῦτ᾽', "Ταῦτ'", 'Ταῦτ̓'])
def test_existing_same_surface_reading_is_preserved(where, encoding):
    values = evidence()
    values[where][encoding] = 'ἄλλος'
    assert plan(values) == []


def test_positive_historical_owner_is_never_displaced():
    values = evidence()
    values['lexical_heads'].add('Ταύς')
    assert plan(values) == []


@pytest.mark.parametrize('change', ['missing_definition', 'reference_definition',
                                    'different_pos', 'missing_cell', 'different_tone',
                                    'different_case', 'different_number', 'different_gender',
                                    'underspecified_source', 'underspecified_cell',
                                    'hyphenated_fragment'])
def test_required_proof_is_not_inferred(change):
    values = evidence()
    entry = values['fresh_entries'][0]['entry']
    pair = values['source_pairs'][0][1][0]
    cell = entry['forms'][0]
    if change == 'missing_definition':
        entry['senses'] = []
    elif change == 'reference_definition':
        entry['senses'] = [{'glosses': ['nominative singular of ὅς']}]
    elif change == 'different_pos':
        entry['pos'] = 'noun'
    elif change == 'missing_cell':
        entry['forms'] = []
    elif change == 'different_tone':
        cell['form'] = 'ταύτα'
    elif change == 'different_case':
        pair['tags'] = ['plural', 'genitive']
    elif change == 'different_number':
        pair['tags'] = ['singular', 'nominative']
    elif change == 'different_gender':
        pair['tags'] = ['masculine', 'plural', 'nominative']
    elif change == 'underspecified_source':
        pair['tags'] = ['plural']
    elif change == 'underspecified_cell':
        cell['tags'] = ['neuter', 'plural']
    elif change == 'hyphenated_fragment':
        cell['form'] = 'ταῦτ-α'
    decision = plan(values)[0]
    assert not decision['accepted']
    target = dict(values['reference'])
    assert apply_repairs(target, [decision])['canonicalized'] == []
    assert target == values['reference']


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('competitor', ['other_owner', 'other_pos', 'unknown_pos'])
def test_all_positive_competing_claims_veto(reverse, competitor):
    values = evidence()
    pair = copy.deepcopy(values['source_pairs'][0][1][0])
    if competitor == 'other_owner':
        pair['lemma'] = 'ἄλλος'
        values['lexical_heads'].add('ἄλλος')
    elif competitor == 'other_pos':
        pair['pos'] = 'noun'
    else:
        pair.pop('pos')
    values['source_pairs'].append(('another_source', [pair]))
    if reverse:
        values['source_pairs'].reverse()
        values['fresh_entries'].reverse()
    assert not plan(values)[0]['accepted']


def test_one_matching_reading_cannot_hide_morphology_conflict():
    values = evidence()
    pair = copy.deepcopy(values['source_pairs'][0][1][0])
    pair['tags'] = ['plural', 'genitive']
    values['source_pairs'].append(('another_source', [pair]))
    decision = plan(values)[0]
    assert decision['matching_cells'] and decision['unmatched_positive_readings']
    assert not decision['accepted']


def test_complete_unelided_cell_vetoes_quotation_marker_ambiguity():
    values = evidence()
    values['reference'] = {'οὐθέν’': 'οὐθές'}
    values['lexical_heads'] = {'οὐδείς'}
    values['source_pairs'] = [('diorisis', [
        {'form': 'οὐθέν᾽', 'lemma': 'οὐδείς', 'pos': 'pron',
         'tags': ['accusative', 'singular']}])]
    values['fresh_entries'][0]['entry'] = {
        'word': 'οὐδείς', 'pos': 'pron', 'lang_code': 'grc',
        'senses': [{'glosses': ['no one']}],
        'forms': [{'form': 'οὐθέν', 'tags': ['singular', 'accusative']},
                  {'form': 'οὐθένα', 'tags': ['singular', 'accusative']}]}
    decision = plan(values)[0]
    assert decision['matching_cells'] and decision['ambiguous_unelided_cells']
    assert not decision['accepted']
    assert decision['reason'] == 'already_complete_nominal_cell_marker_ambiguous'


def test_quantity_notation_does_not_change_literal_letters_or_accent():
    values = evidence()
    values['fresh_entries'][0]['entry']['forms'][0]['form'] = 'ταῦτᾰ'
    assert plan(values)[0]['accepted']


def test_unknown_gender_does_not_contradict_explicit_case_and_number():
    values = evidence()
    values['fresh_entries'][0]['entry']['forms'][0]['tags'] = ['plural', 'nominative']
    assert plan(values)[0]['accepted']


def test_other_marker_positive_historical_reading_is_preserved():
    values = evidence()
    values['reference']['Ταῦτ᾽'] = 'ἄλλος'
    values['lexical_heads'].add('ἄλλος')
    decisions = plan(values)
    assert len(decisions) == 1 and not decisions[0]['accepted']
    assert decisions[0]['reason'] == 'positive_historical_marker_conflict'


@pytest.mark.parametrize('owner', [None, 'ἄλλος', 'οὗτος'])
def test_apply_preserves_any_intervening_owner(owner):
    values = evidence()
    target = {} if owner is None else {'Ταῦτ’': owner}
    prior = dict(target)
    audit = apply_repairs(target, plan(values))
    assert target == prior and not audit['canonicalized']
    assert len(audit['existing_preserved']) == 1


def test_a_positive_verb_cannot_enter_this_repair():
    values = evidence()
    values['source_pairs'][0][1][0]['pos'] = 'verb'
    values['fresh_entries'][0]['entry']['pos'] = 'verb'
    assert not plan(values)[0]['accepted']


@pytest.mark.parametrize('lang', ['el', 'gkm', None])
def test_another_or_missing_source_language_cannot_supply_proof(lang):
    values = evidence()
    values['fresh_entries'][0]['entry']['lang_code'] = lang
    assert not plan(values)[0]['accepted']
