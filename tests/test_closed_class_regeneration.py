"""Regenerating a historical overlay must repeat only still-supported repairs."""
import json

import pytest

from build.closed_class_elisions import apply_repairs, plan_repairs


CELLS = [
    ('Ταῦτ’', 'Ταύς', 'οὗτος', 'ταῦτα', ['neuter', 'plural', 'nominative']),
    ('Τοῦτ’', 'Τούς', 'οὗτος', 'τοῦτο', ['neuter', 'singular', 'nominative']),
    ('οὐδεμί’', 'οὐδεμιός', 'οὐδείς', 'οὐδεμία', ['feminine', 'singular', 'nominative']),
    ('ταῖνδ’', 'εῖνδος', 'ὅδε', 'ταῖνδε', ['feminine', 'dual', 'genitive']),
    ('τοῖνδ’', 'τόνδ’', 'ὅδε', 'τοῖνδε', ['masculine', 'dual', 'genitive']),
    ('τοῖσιδ’', 'τόσιδ’', 'ὅδε', 'τοῖσιδε', ['masculine', 'plural', 'dative']),
    ('τᾶνδ’', 'ε͂νδος', 'ὅδε', 'τᾶνδε', ['feminine', 'plural', 'genitive']),
]


def evidence():
    reference={form:old for form,old,*_ in CELLS}
    entries={}
    rows=[]
    for form,old,owner,complete,tags in CELLS:
        entry=entries.setdefault(owner,{'word':owner,'pos':'pron','lang_code':'grc',
            'senses':[{'glosses':['demonstrative or indefinite pronoun']}],'forms':[]})
        entry['forms'].append({'form':complete,'tags':tags})
        rows.append({'form':form[:-1]+'᾽','lemma':owner,'pos':'pron','tags':tags})
    return dict(reference=reference, reference_base={}, predecessor=dict(reference),
                source_pairs=[('diorisis_pairs.json',rows)], lexical_heads=set(entries),
                fresh_entries=[{'source':'en','line':i,'entry':entry}
                               for i,entry in enumerate(entries.values(),1)])


def target_after(values):
    target=dict(values['predecessor'])
    audit=apply_repairs(target,plan_repairs(**values))
    return target,audit


def test_all_seven_repairs_repeat_after_exact_audit_reversal():
    values=evidence()
    expected={form:owner for form,old,owner,*_ in CELLS}
    target,audit=target_after(values)
    assert target==expected
    assert len(audit['canonicalized'])==7
    first=json.dumps(target,ensure_ascii=False,separators=(',',':'))+'\n'
    for _ in range(2):
        for row in audit['canonicalized']:
            target[row['form']]=row['current_owner']
        assert target==values['reference']
        values['predecessor']=dict(target)
        target,audit=target_after(values)
        assert json.dumps(target,ensure_ascii=False,separators=(',',':'))+'\n'==first
        assert len(audit['canonicalized'])==7


def test_already_correct_output_is_idempotent():
    values=evidence();target,_=target_after(values)
    values['predecessor']=target
    repeated,audit=target_after(values)
    assert repeated==target
    assert audit['canonicalized']==[]


@pytest.mark.parametrize('mark', ['’','᾽',"'",'ʼ'])
@pytest.mark.parametrize('owner',['ἄλλος','οὗτος','Ταῦς'])
def test_intervening_marker_owner_is_preserved_even_if_not_lexically_known(mark,owner):
    values=evidence();values['predecessor']['Ταῦτ'+mark]=owner
    target,audit=target_after(values)
    assert target['Ταῦτ'+mark]==owner
    assert 'Ταῦτ’' not in {row['form'] for row in audit['canonicalized']}
    assert len(audit['canonicalized'])==6


def test_independent_reference_base_reading_is_preserved():
    values=evidence();values['reference_base']['Ταῦτ᾽']='Ταύς'
    target,audit=target_after(values)
    assert target['Ταῦτ’']=='Ταύς'
    assert len(audit['canonicalized'])==6


def test_no_repair_when_its_current_source_reading_is_missing():
    values=evidence()
    values['source_pairs'][0][1][:]=[r for r in values['source_pairs'][0][1] if r['form']!='Ταῦτ᾽']
    target,audit=target_after(values)
    assert target['Ταῦτ’']=='Ταύς'
    assert len(audit['canonicalized'])==6


def test_no_repair_when_its_current_literal_cell_is_missing():
    values=evidence()
    values['fresh_entries'][0]['entry']['forms']=[r for r in values['fresh_entries'][0]['entry']['forms'] if r['form']!='ταῦτα']
    target,audit=target_after(values)
    assert target['Ταῦτ’']=='Ταύς'
    assert len(audit['canonicalized'])==6


def test_competing_current_positive_reading_is_retained_and_blocks_repair():
    values=evidence();values['lexical_heads'].add('ἄλλος')
    values['source_pairs'].append(('independent',[
        {'form':'Ταῦτ᾽','lemma':'ἄλλος','pos':'pron','tags':['neuter','plural','nominative']}]))
    decisions=plan_repairs(**values)
    row=next(r for r in decisions if r['form']=='Ταῦτ’')
    assert set(map(tuple,row['positive_claims']))=={('οὗτος','pron'),('ἄλλος','pron')}
    assert not row['accepted']
    target,audit=target_after(values)
    assert target['Ταῦτ’']=='Ταύς'
    assert len(audit['canonicalized'])==6


def test_newly_supported_historical_owner_is_not_overridden():
    values=evidence();values['lexical_heads'].add('Ταύς')
    target,audit=target_after(values)
    assert target['Ταῦτ’']=='Ταύς'
    assert len(audit['canonicalized'])==6


def test_conflicting_historical_marker_reading_blocks_repair():
    values=evidence();values['reference']['Ταῦτ᾽']='ἄλλος';values['lexical_heads'].add('ἄλλος')
    values['predecessor']['Ταῦτ᾽']='ἄλλος'
    target,audit=target_after(values)
    assert target['Ταῦτ’']=='Ταύς' and target['Ταῦτ᾽']=='ἄλλος'
    assert len(audit['canonicalized'])==6


def test_identical_owner_on_an_independent_marker_variant_is_preserved():
    values=evidence();values['predecessor']['Ταῦτ᾽']='Ταύς'
    target,audit=target_after(values)
    assert target['Ταῦτ’']==target['Ταῦτ᾽']=='Ταύς'
    assert len(audit['canonicalized'])==6


def test_another_exact_historical_reference_cell_can_be_repaired():
    values=evidence();values['reference']['Ταῦτ᾽']='Ταύς';values['predecessor']['Ταῦτ᾽']='Ταύς'
    target,audit=target_after(values)
    assert target['Ταῦτ’']==target['Ταῦτ᾽']=='οὗτος'
    assert len(audit['canonicalized'])==8
