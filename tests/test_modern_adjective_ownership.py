"""Exact Modern adjective definitions survive an unrelated flat homograph."""

import copy
import hashlib
import json

import pytest

import build_data as B


def write_entries(tmp_path, entries):
    path = tmp_path / 'source.jsonl'
    path.write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in entries))
    return path


def adjective_entry():
    return {'word': 'κρατών', 'pos': 'adj',
            'senses': [{'glosses': ['επικρατών, που επικρατεί']}],
            'forms': [
                {'form': 'κρατών', 'tags': ['masculine', 'singular', 'nominative']},
                {'form': 'κρατούσα', 'tags': ['feminine', 'singular', 'nominative']},
                {'form': 'κρατούν', 'tags': ['neuter', 'singular', 'nominative']},
                {'form': 'κρατούσες', 'tags': ['feminine', 'plural', 'nominative']}],
            }


@pytest.mark.parametrize('reverse', [False, True])
def test_defined_adjective_cells_keep_their_owner_beside_noun_and_verb(tmp_path, reverse):
    entries = [adjective_entry(),
        {'word': 'κρατών', 'pos': 'noun', 'tags': ['form-of'],
         'senses': [{'glosses': ['γενική πληθυντικού του κράτος'],
                     'form_of': [{'word': 'κράτος'}], 'tags': ['genitive', 'plural']}],
         'forms': adjective_entry()['forms']},
        {'word': 'κρατώ', 'pos': 'verb', 'senses': [{'glosses': ['hold']}],
         'forms': [{'form': 'κρατούν', 'tags': ['present', 'finite', 'third-person', 'plural']}]}]
    heads, records = set(), []
    pairs, _, _, typed, *_ = B.extract_pairs(
        write_entries(tmp_path, entries[::-1] if reverse else entries), 'el-el',
        adjective_heads=heads, adjective_head_records=records)
    lookup = {'κρατών': ('κράτος', 1), 'κράτος': ('κράτος', 3),
              'κρατώ': ('κρατώ', 3)}
    cleaned = B.clean_training_pairs(pairs, lookup, preserve_cells=True,
                                    pos_map=typed, adjective_heads=heads)
    assert {p['form'] for p in cleaned if p['pos'] == 'adj' and p['lemma'] == 'κρατών'} == {
        'κρατών', 'κρατούσα', 'κρατούν', 'κρατούσες'}
    assert [p for p in cleaned if p['pos'] == 'noun'] == [
        {'form': 'κρατών', 'lemma': 'κράτος', 'pos': 'noun',
         'tags': ['form-of', 'genitive', 'plural']}]
    assert any(p['form'] == 'κρατούν' and p['lemma'] == 'κρατώ' and p['pos'] == 'verb'
               for p in cleaned)
    assert all(set(p) == {'form', 'lemma', 'pos', 'tags'} for p in cleaned)
    assert B.resolve_pair_lemma('κρατών', 'adj', lookup, adjective_heads=heads) == 'κρατών'
    assert B.resolve_pair_lemma('κρατών', 'noun', lookup, adjective_heads=heads) == 'κράτος'
    assert B.resolve_pair_lemma('κρατών', 'verb', lookup, adjective_heads=heads) == 'κράτος'
    assert B.resolve_pair_lemma('Κρατών', 'adj', lookup, adjective_heads=heads) == 'Κρατών'
    inventory = B.adjective_head_inventory(records, lookup, {'el': 'a' * 64})
    assert B.validate_adjective_head_inventory(inventory) == {'κρατών'}
    assert B.training_pair_headword_errors(cleaned, {'κρατών': 'κράτος'})[0] == 4
    assert B.training_pair_headword_errors(cleaned, {'κρατών': 'κράτος'},
                                         adjective_heads=heads)[0] == 0
    assert B.training_pair_headword_errors(
        [{'form': 'κρατών', 'lemma': 'κρατών', 'pos': 'noun', 'tags': []}],
        {'κρατών': 'κράτος'}, adjective_heads=heads)[0] == 1


@pytest.mark.parametrize('kind', ['form_ref', 'alt_ref', 'entry_form', 'sense_form',
                                 'declared_form', 'declared_forms', 'no_gloss',
                                 'empty', 'blank', 'whitespace', 'adjf',
                                 'english_case', 'greek_case', 'gender_case'])
def test_grammatical_or_empty_adjective_pages_do_not_authorize_ownership(tmp_path, kind):
    entry = adjective_entry()
    if kind == 'form_ref': entry['senses'][0]['form_of'] = [{'word': 'κρατώ'}]
    elif kind == 'alt_ref': entry['senses'][0]['alt_of'] = [{'word': 'επικρατών'}]
    elif kind == 'entry_form': entry['tags'] = ['form-of']
    elif kind == 'sense_form': entry['senses'][0]['tags'] = ['form-of']
    elif kind in {'declared_form', 'declared_forms'}:
        entry['head_templates'] = [{'name': 'head', 'args': {'1': 'el', '2':
            'adjective forms' if kind == 'declared_forms' else 'adjective form'}}]
    elif kind == 'no_gloss': entry['senses'][0]['tags'] = ['no-gloss']
    elif kind == 'empty': entry['senses'][0]['glosses'] = []
    elif kind == 'blank': entry['senses'][0]['glosses'] = ['']
    elif kind == 'whitespace': entry['senses'][0]['glosses'] = [' \t ']
    elif kind == 'adjf':
        entry['head_templates'] = [{'name': 'head', 'args': {'1': 'el', '2': 'adjf'}}]
    elif kind == 'english_case': entry['senses'][0]['glosses'] = ['masculine nominative singular of κρατώ']
    elif kind == 'greek_case': entry['senses'][0]['glosses'] = ['γενική πληθυντικού του κρατώ']
    elif kind == 'gender_case': entry['senses'][0]['glosses'] = ['το θηλυκό γενική ενικού του κρατώ']
    heads, records = set(), []
    pairs, *_ = B.extract_pairs(write_entries(tmp_path, [entry]), 'el-en',
                               adjective_heads=heads, adjective_head_records=records)
    assert not heads and not records
    assert not any(p.get('adjective_owner') for p in pairs)


@pytest.mark.parametrize('glosses', [['full of water'], ['γενική πληθυντικού του άλλου', 'prevailing']])
def test_an_ordinary_adjective_gloss_is_positive_evidence(tmp_path, glosses):
    entry = adjective_entry()
    entry['senses'][0]['glosses'] = glosses
    entry['form_of'] = [{'word': 'κρατῶν'}]  # Greek etymology, not an inflected reading.
    assert B.is_independent_adjective_entry(entry, 'el')
    assert not B.is_independent_adjective_entry(entry, 'en')
    heads = set()
    B.extract_pairs(write_entries(tmp_path, [entry]), 'el-el', adjective_heads=heads)
    assert heads == {'κρατών'}


def test_owner_marker_and_exact_reference_readings_remain_separate(tmp_path):
    head = adjective_entry()
    reference = {'word': 'κρατών', 'pos': 'adj',
                 'senses': [{'form_of': [{'word': 'κρατερός'}],
                             'glosses': ['genitive plural of κρατερός'],
                             'tags': ['genitive', 'plural']}],
                 'forms': [{'form': 'κρατών', 'tags': ['masculine']}]}
    heads = set()
    pairs, _, _, typed, *_ = B.extract_pairs(write_entries(tmp_path, [head, reference]),
                                             'el-el', adjective_heads=heads)
    flat = {'κρατών': ('κράτος', 1), 'κράτος': ('κράτος', 3), 'κρατερός': ('κρατερός', 3)}
    cleaned = B.clean_training_pairs(pairs, flat, preserve_cells=True,
                                    pos_map=typed, adjective_heads=heads)
    assert any(p['lemma'] == 'κρατερός' and 'genitive' in p['tags'] for p in cleaned)
    assert any(p['lemma'] == 'κρατών' and 'nominative' in p['tags'] for p in cleaned)
    row = {'form': 'κρατούν', 'lemma': 'κρατών', 'pos': 'adj', 'tags': ['neuter']}
    owned = {**row, 'adjective_owner': True}
    for rows in ([row, owned], [owned, row]):
        assert len(B.deduplicate_pairs(rows, True)) == 2
        assert {p['lemma'] for p in B.clean_training_pairs(rows, flat, True)} == {'κράτος', 'κρατών'}


def test_adjective_ownership_is_modern_and_part_of_speech_specific(tmp_path):
    for lang, pos in [('grc-el', 'adj'), ('el-el', 'noun'), ('el-el', 'name'), ('el-el', 'verb')]:
        entry = {**adjective_entry(), 'pos': pos}
        heads, records = set(), []
        pairs, *_ = B.extract_pairs(write_entries(tmp_path, [entry]), lang,
                                   adjective_heads=heads, adjective_head_records=records)
        assert not heads and not records
        assert not any(p.get('adjective_owner') for p in pairs)


@pytest.mark.parametrize('mutation', ['pos', 'head', 'hash', 'form', 'template', 'grammar',
                                      'empty', 'morphology', 'form_tags', 'no_morphology'])
def test_adjective_proof_rejects_wrong_or_malformed_evidence(mutation):
    entry = adjective_entry()
    record = {'lemma': 'κρατών', **B.adjective_head_evidence(entry, 'el')}
    proof = B.adjective_head_inventory([record], {'κρατών': ('κράτος', 1)}, {'el': 'a' * 64})
    row = proof['heads'][0]['evidence'][0]
    if mutation == 'pos': row['pos'] = 'noun'
    elif mutation == 'head': row['headword'] = 'κρατώ'
    elif mutation == 'hash': proof['source_sha256']['el'] = 'bad'
    elif mutation == 'form': row['senses'][0]['form_of'] = [{'word': 'κρατώ'}]
    elif mutation == 'template': row['head_templates'] = [{'name': 'head', 'args': []}]
    elif mutation == 'grammar': row['senses'][0]['glosses'] = ['nominative singular of κρατώ']
    elif mutation == 'empty': row['senses'][0]['glosses'] = []
    elif mutation == 'morphology': row['forms'] = {}
    elif mutation == 'form_tags': row['forms'][0]['tags'] = ['masculine', 1]
    elif mutation == 'no_morphology': row['forms'] = []
    with pytest.raises(ValueError, match='adj-head evidence'):
        B.validate_adjective_head_inventory(proof)


def test_adjective_companion_is_deterministic_and_omits_flat_self_heads():
    entries = [adjective_entry(), {**adjective_entry(), 'word': 'επικρατών'}]
    records = [{'lemma': e['word'], **B.adjective_head_evidence(e, 'el')} for e in entries]
    lookup = {'κρατών': ('κράτος', 1), 'επικρατών': ('επικρατών', 3)}
    first = B.adjective_head_inventory(records + records, lookup, {'el': 'a' * 64})
    assert first == B.adjective_head_inventory(records[::-1], lookup, {'el': 'a' * 64})
    assert B.validate_adjective_head_inventory(first) == {'κρατών'}


def test_main_generates_adjective_proof_and_publishes_exact_typed_owner(tmp_path, monkeypatch):
    source, output = tmp_path / 'source', tmp_path / 'data'
    source.mkdir()
    en = source / 'en.jsonl'
    el = source / 'el.jsonl'
    en.write_text(json.dumps({'word': 'κρατών', 'pos': 'noun', 'senses': [
        {'form_of': [{'word': 'κράτος'}], 'glosses': ['genitive plural of κράτος'],
         'tags': ['genitive', 'plural']}]}, ensure_ascii=False) + '\n')
    el.write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in [
        adjective_entry(), {'word': 'κράτος', 'pos': 'noun',
                            'senses': [{'glosses': ['state']}]}]))
    monkeypatch.setattr(B, 'DATA_DIR', output)
    monkeypatch.setattr(B, 'DUMPS', {'el': {'en': 'en.jsonl', 'el': 'el.jsonl'}})
    monkeypatch.setattr(B.sys, 'argv', ['build_data.py', '--lang', 'el', '--kaikki', str(source)])
    B.main()
    proof = json.loads((output / 'el_adjective_heads.json').read_text())
    assert proof['source_sha256'] == {
        'en': hashlib.sha256(en.read_bytes()).hexdigest(),
        'el': hashlib.sha256(el.read_bytes()).hexdigest()}
    assert B.validate_adjective_head_inventory(proof) == {'κρατών'}
    flat = json.loads((output / 'mg_lookup.json').read_text())
    typed = json.loads((output / 'mg_pos_lookup.json').read_text())
    assert flat['κρατών'] == 'κράτος'
    assert typed['κρατούν']['ADJ'] == 'κρατών'
    assert typed['κρατούσα']['ADJ'] == 'κρατών'
    assert typed['κρατούσες']['ADJ'] == 'κρατών'
    pairs = json.loads((output / 'mg_pairs.json').read_text())
    assert B.training_pair_headword_errors(pairs, flat, adjective_heads={'κρατών'})[0] == 0


@pytest.mark.parametrize('gloss', [
    'comparative of καλός', 'superlative form of καλός',
    'συγκριτικός βαθμός του καλός, πιο καλός',
    'υπερθετικός βαθμός του καλός',
    'με άρθρο, σχετικός υπερθετικός του καλός',
    'παθητική μετοχή του κρατώ', 'ενεργητική μετοχή του κρατώ',
    'contracted form of κρατών', 'συνηρημένη μορφή του κρατών'])
def test_grammatical_degree_participle_and_contraction_descriptions_are_not_heads(gloss):
    entry = adjective_entry()
    entry['senses'][0]['glosses'] = [gloss, 'more prevalent']
    assert not B.is_independent_adjective_entry(entry, 'el')
    entry['senses'].append({'glosses': ['independently dominant']})
    assert B.is_independent_adjective_entry(entry, 'el')


def test_a_spelling_label_and_circular_restated_head_are_not_positive_evidence():
    entry = {**adjective_entry(), 'word': 'αἰσθητός', 'senses': [
        {'glosses': ['πολυτονική γραφή του αισθητός']}, {'glosses': ['αισθητός']}]}
    assert not B.is_independent_adjective_entry(entry, 'el')


@pytest.mark.parametrize('gloss', [
    'accusative/vocative masculine singular of κανελής',
    'nominative/accusative/vocative neuter plural of ψυκτικός',
    'A rare form of μικτός', 'a more formal variant of ακατάδεχτος',
    'less formal pronunciation of ψυχτική, nominative feminine singular of ψυχτικός',
    'polytonic spelling: nominative feminine singular of πλατύς: wide',
    'etymological speeling of κανελής without simplification of double consonant',
    'βλέπε αέρινος', 'δες καλός'])
def test_unstructured_adjective_form_and_spelling_descriptions_are_not_heads(gloss):
    entry = adjective_entry()
    entry['senses'][0]['glosses'] = [gloss]
    assert not B.is_independent_adjective_entry(entry, 'el')


@pytest.mark.parametrize('gloss,forms', [
    ('με αξονομετρικό τρόπο', [{'form': 'αξονομετρικά', 'source': 'header'}]),
    ('η αμοιβή ενός αχθοφόρου', [
        {'form': "el-κλίση-'βουνό'", 'tags': ['inflection-template'], 'source': 'declension'},
        {'form': 'αχθοφορικά', 'tags': ['neuter', 'plural-only'], 'source': 'header'},
        {'form': 'αχθοφορικών', 'tags': ['genitive', 'plural'], 'source': 'declension'}]),
    ('σκιερό μέρος', [
        {'form': "el-κλίση-'πεύκο'", 'tags': ['inflection-template'], 'source': 'declension'},
        {'form': 'απόσκιο', 'tags': ['neuter'], 'source': 'header'},
        {'form': 'απόσκια', 'tags': ['nominative', 'plural'], 'source': 'declension'}]),
    ('ακέραιος, ολόκληρος', [{'form': 'ακέριος', 'source': 'header'},
                           {'form': '-ια', 'source': 'header'},
                           {'form': 'ιο', 'source': 'header'}]),
])
def test_mislabeled_or_underspecified_morphology_does_not_certify_an_adjective(gloss, forms):
    entry = {**adjective_entry(), 'forms': forms, 'senses': [{'glosses': [gloss]}]}
    assert not B.is_independent_adjective_entry(entry, 'el')


@pytest.mark.parametrize('structure', ['indeclinable', 'template', 'head_template'])
def test_explicit_adjective_structure_does_not_require_two_gendered_table_rows(structure):
    entry = {'word': 'καφέ', 'pos': 'adj', 'senses': [{'glosses': ['brown']}]}
    if structure == 'indeclinable':
        entry['forms'] = [{'form': 'καφέ', 'tags': ['invariable'], 'source': 'header'}]
    elif structure == 'template':
        entry['head_templates'] = [{'name': 'el-adj', 'args': {'1': 'ind'}}]
    else:
        entry['head_templates'] = [{'name': 'head', 'args': {'1': 'el', '2': 'adjective'}}]
    assert B.is_independent_adjective_entry(entry, 'en' if structure != 'indeclinable' else 'el')
    proof = B.adjective_head_inventory(
        [{'lemma': entry['word'], **B.adjective_head_evidence(entry, 'el')}],
        {'καφέ': ('καφές', 1)}, {'el': 'a' * 64})
    assert B.validate_adjective_head_inventory(proof) == {'καφέ'}


def test_uncertified_adjective_keeps_its_previous_processing(tmp_path):
    entry = {'word': 'υψίφωνος', 'pos': 'adj', 'forms': [],
             'senses': [{'glosses': ['που η φωνή του κινείται στις ψηλές περιοχές τονικού ύψους']}]}
    heads, records = set(), []
    path = write_entries(tmp_path, [entry])
    result = B.extract_pairs(path, 'el-el', adjective_heads=heads, adjective_head_records=records)
    assert result == B.extract_pairs(path, 'el-el')
    assert not heads and not records
    lookup = {'υψίφωνος': ('υψίφωνος', 3)}
    assert B.clean_training_pairs(result[0], lookup, True, adjective_heads=heads) == []
    assert result[1]['υψίφωνος'][0] == 'υψίφωνος'
