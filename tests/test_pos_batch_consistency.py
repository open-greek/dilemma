"""Source-only scalar/batch controls with no full lookup or model load."""
import pytest
from dilemma import core

def model(candidates,baseline,pos,*,gate=True,convention=lambda lemma: lemma):
    result=core.Dilemma.__new__(core.Dilemma)
    result.lang='grc'
    result._convention_name=None
    result._lemmatize_verbose_impl=lambda word,guess: list(candidates)
    result._pos_table_lookup=lambda word,upos: pos
    result._apply_convention=convention
    result._article_pos_lemma=lambda word,upos: None
    result._resolve_attested_form=lambda word: word if gate else None
    result.lemmatize_batch=lambda words,**kwargs: [baseline for _ in words]
    result.lemmatize=lambda word,**kwargs: baseline
    return result

def candidate(lemma,proper=False):
    return core.LemmaCandidate(lemma=lemma,proper=proper,source='lookup')

@pytest.mark.parametrize('word,upos,candidates,baseline,pos',[
    ('ΆΛΕΑ','NOUN',['ἀλέα','Ἀλέα'],'ἀλέα','ἅλας'),
    ('ΆΛΕΑ','VERB',['ἀλέα','Ἀλέα'],'ἀλέα','ἀλέω'),
    ('ΑΓΟΡΆ','VERB',['ἀγορά','Ἀγορά'],'ἀγορά','ἀγοράζω'),
    ('ΕΣΤΑΥΡΩΜΈΝΟΣ','VERB',['εσταυρωμένος','Εσταυρωμένος'],'εσταυρωμένος','σταυρόω'),
    ('Όσσά','PRON',['ὄσσα','Ὄσσα'],'ὄσσα','ὅσος'),
    ('ΚΆΤΙ','ADV',['κάτι','Κάτι'],'κάτι','ἔτι'),
    ('Παρούση','VERB',['παρούση','Παρούση'],'Παρούση','πάρειμι'),
])
def test_all_input_variants_honor_same_scalar_pos_override(word,upos,candidates,baseline,pos):
    m=model([candidate(value,value[:1].isupper()) for value in candidates],baseline,pos)
    assert m.lemmatize_pos(word,upos,guess=False)==pos
    assert m.lemmatize_batch_pos([word],[upos],guess=False)==[pos]

@pytest.mark.parametrize('upos,expected',[('NOUN','ωρολόγιο'),('PROPN','Ωρολόγιο')])
def test_unmatched_pos_retains_ordered_case_twin_fallback(upos,expected):
    m=model([candidate('ωρολόγιο'),candidate('Ωρολόγιο',True)],'Ωρολόγιο','ὡρολόγιον')
    assert m.lemmatize_pos('Ωρολογίου',upos,guess=False)==expected
    assert m.lemmatize_batch_pos(['Ωρολογίου'],[upos],guess=False)==[expected]

@pytest.mark.parametrize('word,baseline,ancient',[
    ('άλμα','άλμα','ἅλμα'),('ΆΛΜΑ','άλμα','ἅλμα'),
    ('εστία','εστία','ἑστία'),('ΕΣΤΊΑ','εστία','ἑστία'),
    ('όμηρος','όμηρος','ὅμηρος'),('ΌΜΗΡΟΣ','όμηρος','ὅμηρος'),
])
def test_pos_agreement_with_selfmap_still_compares_scalar_citation_candidates(word,baseline,ancient):
    m=model([candidate(ancient),candidate(ancient[:1].upper()+ancient[1:],True)],baseline,baseline)
    assert m.lemmatize_pos(word,'NOUN',guess=False)==ancient
    assert m.lemmatize_batch_pos([word],['NOUN'],guess=False)==[ancient]


def test_nonself_agreement_keeps_fast_path():
    m=model([], 'παῖς','παῖς')
    def unexpected(*args,**kwargs):
        raise AssertionError('Nonself agreement does not need candidates')
    m._lemmatize_verbose_impl=unexpected
    assert m.lemmatize_batch_pos(['παιδός'],['NOUN'],guess=False)==['παῖς']


def test_multiple_distinct_lemmas_are_not_overridden_by_unmatched_pos():
    m=model([candidate('one'),candidate('two')],'one','unmatched')
    assert m.lemmatize_batch_pos(['input'],['NOUN'],guess=False)==['one']

def test_missing_pos_uses_scalar_ordered_case_choice():
    m=model([candidate('ωρολόγιο'),candidate('Ωρολόγιο',True)],'Ωρολόγιο',None)
    assert m.lemmatize_pos('Ωρολογίου','NOUN',guess=False)=='ωρολόγιο'
    assert m.lemmatize_batch_pos(['Ωρολογίου'],['NOUN'],guess=False)==['ωρολόγιο']

def test_attestation_gate_cannot_be_resurrected():
    m=model([candidate('input'),candidate('Input')],None,'other',gate=False)
    assert m.lemmatize_pos('input','NOUN',attested_only=True,guess=False) is None
    assert m.lemmatize_batch_pos(['input'],['NOUN'],attested_only=True,guess=False)==[None]

def test_rejected_convention_target_does_not_trigger_new_override():
    m=model([candidate('input'),candidate('Input')],'input','other',convention=lambda lemma: None)
    assert m.lemmatize_batch_pos(['input'],['NOUN'],guess=False)==['input']

def test_matching_pos_still_wins_over_case_twin_order():
    m=model([candidate('common'),candidate('Common',True)],'Common','Common')
    assert m.lemmatize_batch_pos(['Common'],['NOUN'],guess=False)==['Common']

def test_modern_selfmap_fix_precedes_case_twin_fallback():
    m=model([candidate('θηλυκή'),candidate('Θηλυκή',True)],'Θηλυκή','unmatched')
    m._fix_mg_selfmap=lambda *args,**kwargs: 'θηλυκός'
    assert m.lemmatize_pos('Θηλυκής','ADJ',guess=False)=='θηλυκός'
    assert m.lemmatize_batch_pos(['Θηλυκής'],['ADJ'],guess=False)==['θηλυκός']

@pytest.mark.parametrize('word,baseline,citation',[
    ('Άδης','Άδης','ᾍδης'),('ΊΝΑ','ίνα','ἵνα'),('ίνα','ίνα','ἵνα'),
])
def test_absent_pos_keeps_scalar_single_citation(word,baseline,citation):
    m=model([candidate(citation)],baseline,None)
    assert m.lemmatize_pos(word,'NOUN',guess=False)==citation
    assert m.lemmatize_batch_pos([word],['NOUN'],guess=False)==[citation]


def test_absent_pos_does_not_repeat_batch_model_inference():
    m=model([], 'model_guess',None)
    def structured_only(word,*,guess):
        assert guess is False
        return []
    m._lemmatize_verbose_impl=structured_only
    assert m.lemmatize_batch_pos(['unknown'],['NOUN'],guess=True)==['model_guess']


def test_absent_pos_does_not_replace_distinct_baseline_with_unmatched_candidate():
    m=model([candidate('different')],'baseline',None)
    assert m.lemmatize_batch_pos(['input'],['NOUN'],guess=False)==['baseline']
