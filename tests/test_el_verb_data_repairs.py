import copy
import json

import pytest

import build_data as B
from dilemma.el_imperfect_paradigms import complete_imperfect_pairs, unapply_imperfect_pairs


def test_published_modern_snapshot_has_separate_tenses_and_retained_readings():
    path = B.DATA_DIR / 'mg_pairs.json'
    if not path.exists():
        pytest.skip('Published Modern Greek data is not installed')
    rows = json.loads(path.read_text())
    specific = set(B.VERB_TENSE_LABELS.values())
    bad = [r for r in rows if r.get('pos') == 'verb'
           and len(set(r.get('tags', ())) & specific) > 1]
    assert not bad, bad[:5]
    targets = {(r['form'], r['lemma'], r.get('pos')) for r in rows
               if r['form'] in {'τσαι','τσάι','συλλάμβανε','συνελάμβανε'}}
    assert {('τσαι','τσαι','conj'), ('τσάι','τσάι','noun'),
            ('συλλάμβανε','συλλαμβάνω','verb'),
            ('συνελάμβανε','συλλαμβάνω','verb')} <= targets


def test_table_heading_does_not_override_an_explicit_column_tense(tmp_path):
    entry = {"word": "διαρρέω", "pos": "verb", "senses": [{"glosses": ["leak"]}],
             "forms": [{"form": "present", "tags": ["table-tags"]},
                       {"form": "διαρρέει", "tags": ["present", "indicative", "third-person", "singular"]},
                       {"form": "διαρρέαν", "tags": ["imperfect", "indicative", "third-person", "plural"]},
                       {"form": "διάρρεε", "tags": ["imperative", "second-person", "singular"]}]}
    p = tmp_path / "dump.jsonl";p.write_text(json.dumps(entry, ensure_ascii=False) + "\n")
    rows, *_ = B.extract_pairs(p, "el-en")
    by_form = {r["form"]:set(r["tags"]) for r in rows}
    assert "present" in by_form["διαρρέει"]
    assert "imperfect" in by_form["διαρρέαν"] and "present" not in by_form["διαρρέαν"]
    assert "present" in by_form["διάρρεε"]


@pytest.mark.parametrize("tense", ["imperfect", "future", "aorist", "perfect", "pluperfect", "past"])
def test_explicit_cell_tense_wins_over_header(tense):
    assert B.verb_cell_tense_tags([tense, "indicative"], "present") == [tense, "indicative"]


def test_source_aspect_and_head_line_labels_are_not_extra_tenses():
    assert set(B.modern_verb_source_tags(
        ['future','imperfect','first-person','plural'],
        {'raw_tags':['Εξ. Μέλλ.']})) == {'future','imperfective','first-person','plural'}
    assert set(B.modern_verb_source_tags(['present','imperfect','form-of'],
        {'glosses':['third-person singular present imperfect passive of αποκρίνομαι']})) == {
            'present','imperfective','form-of'}
    assert B.modern_verb_source_tags(['aorist','future'], {'source':'header'}) == []


def test_present_future_reference_retains_two_separate_analyses(tmp_path):
    e={'word':'πέρδεστε','pos':'verb','senses':[{'glosses':['second-person plural present/future of πέρδομαι'],
       'tags':['present','future','second-person','plural'], 'form_of':[{'word':'πέρδομαι'}]}]}
    p=tmp_path/'dump.jsonl';p.write_text(json.dumps(e)+'\n')
    rows,*_=B.extract_pairs(p,'el-en')
    cleaned=B.clean_training_pairs(rows,{'πέρδομαι':('πέρδομαι',5)},preserve_cells=True)
    assert {frozenset(r['tags']) & {'present','future'} for r in cleaned} == {
        frozenset({'present'}),frozenset({'future'})}


def test_snapshot_matches_table_after_its_head_was_canonicalized():
    from scripts.repair_el_verb_snapshot import repair_pairs
    rows=[{'form':'ξεκινάω','lemma':'ξεκινώ','pos':'verb',
           'tags':['first-person','singular','present','perfect']},
          {'form':'ξεκινάς','lemma':'ξεκινώ','pos':'verb',
           'tags':['second-person','singular','present','perfect']}]
    proof={'source':'en','source_lemma':'ξεκινάω','form':'ξεκινάς','table_header':'perfect',
           'original_tags':rows[1]['tags'],
           'source_cell':{'tags':['second-person','singular','present']}}
    result,audit=repair_pairs(rows,{}, {'tense_cells':[proof]}, [])
    assert result[1]['tags']==['present','second-person','singular']
    assert audit[0]['evidence'][0]['published_head_reading']==rows[0]


def test_current_cells_resolve_old_tags_without_crossing_owners_or_persons():
    from scripts.repair_el_verb_snapshot import repair_pairs
    row={'form':'εξασθενίζω','lemma':'εξασθενίζω','pos':'verb',
         'tags':['active','indicative','present','perfect','first-person','singular']}
    proof={'source':'el','source_lemma':'εξασθενίζω','form':'εξασθενίζω',
           'corrected_tags':['present','first-person','singular'],
           'source_cell':{'tags':['present','first-person','singular']}}
    result,audit=repair_pairs([row],{}, {'tense_cells':[], 'tense_witnesses':[proof]}, [])
    assert set(result[0]['tags'])==set(row['tags'])-{'perfect'}
    assert audit[0]['evidence']==[proof]
    for invalid in ({**proof,'source_lemma':'another word'},
                    {**proof,'corrected_tags':['present','second-person','singular']}):
        result,audit=repair_pairs([row],{}, {'tense_cells':[], 'tense_witnesses':[invalid]}, [])
        assert result==[row] and audit==[]


def test_reviewed_revision_cells_only_remove_the_documented_extra_heading():
    from scripts.repair_el_verb_snapshot import repair_pairs
    row={'form':'εξασθενούσα','lemma':'εξασθενίζω','pos':'verb',
         'tags':['imperfect','perfect','first-person','singular']}
    fixed,audit=repair_pairs([row],{}, {'tense_cells':[]}, [])
    assert set(fixed[0]['tags'])==set(row['tags'])-{'perfect'}
    assert audit[0]['evidence'][0]['revision_url'].endswith('oldid=93555953')
    for other in ({**row,'lemma':'different owner'},
                  {**row,'tags':['aorist','perfect','first-person','singular']}):
        fixed,audit=repair_pairs([other],{}, {'tense_cells':[]}, [])
        assert fixed==[other] and audit==[]


def test_same_spelling_keeps_separate_present_and_imperfect_readings(tmp_path):
    e = {"word": "λέω", "pos": "verb", "senses": [{"glosses": ["say"]}],
         "forms": [{"form": "present", "tags": ["table-tags"]},
                   {"form": "έλεγε", "tags": ["present", "indicative"]},
                   {"form": "έλεγε", "tags": ["imperfect", "indicative"]}]}
    p=tmp_path/'dump.jsonl';p.write_text(json.dumps(e)+'\n')
    pairs,*_=B.extract_pairs(p,'el-en')
    assert {frozenset(r['tags']) for r in pairs} == {
        frozenset(['present','indicative']),frozenset(['imperfect','indicative'])}


def test_exact_dialect_conjunction_is_not_a_noun_alias(tmp_path):
    entries = [{"word":"τσάι","pos":"noun","senses":[{"glosses":["tea"]}],
                "forms":[{"form":"τσάι","tags":["singular","nominative"]}]},
               {"word":"τσαι","pos":"conj","senses":[{"glosses":["και"],"tags":["idiomatic"]}],
                "forms":[{"form":"τσαι"}]}]
    p=tmp_path/'dump.jsonl';p.write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in entries))
    heads={};pairs,lookup,*_=B.extract_pairs(p,'el-el',unaccented_heads=heads)
    audit=B.recover_unaccented_heads(lookup,heads,{'τσάι':{'noun'}})
    fixed=B.clean_training_pairs(pairs,lookup,preserve_cells=True)
    assert lookup['τσαι'][0]=='τσαι' and lookup['τσάι'][0]=='τσάι'
    assert {(r['form'],r['lemma'],r['pos']) for r in fixed} >= {
        ('τσαι','τσαι','conj'),('τσάι','τσάι','noun')}
    assert audit[0]['original'][0]=='τσάι' and audit[0]['evidence'][0]['source']=='el'
    assert B.recover_unaccented_heads(lookup,heads,{'τσάι':{'noun'}})==[]


def test_a_spelling_reference_or_an_unrelated_target_is_not_recovered():
    for senses,target in [([{'glosses':['spelling'],'alt_of':[{'word':'τσάι'}]}],'τσάι'),
                           ([{'glosses':['και']}],'και')]:
        lookup={'τσαι':(target,3)};before=copy.deepcopy(lookup)
        B.recover_unaccented_heads(lookup,{'τσαι':[{'source':'el','headword':'τσαι','pos':'conj','senses':senses}]},{'τσάι':{'noun'}})
        assert lookup==before


def test_reviewed_imperfect_keeps_both_augments_and_undoes_exactly():
    pairs=[];lookup={'συλλαμβάνω':('συλλαμβάνω',5)}
    heads={'συλλαμβάνω':[{'source':'el','headword':'συλλαμβάνω'}]}
    before=copy.deepcopy((pairs,lookup));audit=complete_imperfect_pairs(pairs,lookup,heads)
    assert lookup['συλλάμβανε'][0]==lookup['συνελάμβανε'][0]=='συλλαμβάνω'
    assert lookup['συλλαμβάναμε'][0]=='συλλαμβάνω'
    assert 'συνελαμβάναμε' not in lookup
    assert all(c['rule_evidence'] and c['original'] is None for c in audit['changes'])
    assert complete_imperfect_pairs(pairs,lookup,heads)['changes']==[]
    unapply_imperfect_pairs(pairs,lookup,audit)
    assert (pairs,lookup)==before


def test_missing_cells_use_recorded_stems_without_guessing_augment_or_stress():
    pairs=[{'form':'υπέγραφα','lemma':'υπογράφω','pos':'verb',
            'tags':['imperfect','first-person','singular']},
           {'form':'υπογράφαμε','lemma':'υπογράφω','pos':'verb',
            'tags':['imperfect','first-person','plural']}]
    lookup={'υπογράφω':('υπογράφω',5),'υπέγραφε':('another valid reading',5)}
    heads={'υπογράφω':[{'source':'el'}],'παραγράφω':[{'source':'el'}]}
    complete_imperfect_pairs(pairs,lookup,heads)
    forms={r['form'] for r in pairs}
    assert {'υπέγραφες','υπέγραφε','υπέγραφαν','υπογράφατε'}<=forms
    assert not {'υποέγραφε','υπογράφε','παρέγραφα','υποεγράφαμε'}&forms
    assert lookup['υπέγραφε'][0]=='another valid reading'


def test_unknown_past_stems_and_impersonal_heads_are_not_guessed():
    lookup={'γράφω':('γράφω',5),'συλλαμβάνω':('συλλαμβάνω',5)};pairs=[]
    audit=complete_imperfect_pairs(pairs,lookup,{'γράφω':[], 'συλλαμβάνω':[{'tags':['impersonal']}]})
    assert audit['changes']==[] and not pairs


@pytest.mark.parametrize('existing_present_reading', [False, True])
def test_pinned_snapshot_repairs_and_round_trip_preserve_other_languages(tmp_path, existing_present_reading):
    import sqlite3
    from scripts.repair_el_verb_snapshot import refresh, sha, unapply
    base=tmp_path/'base';base.mkdir()
    pairs=[{'form':'διαρρέαν','lemma':'διαρρέω','pos':'verb','tags':['present','imperfect','indicative','third-person','plural']},
           {'form':'τσαι','lemma':'τσάι','pos':'conj','tags':[]},
           {'form':'τσάι','lemma':'τσάι','pos':'noun','tags':[]},
           {'form':'πέρδεστε','lemma':'πέρδομαι','pos':'verb',
            'tags':['present','future','form-of','second-person','plural']}]
    if existing_present_reading:
        pairs.append({**pairs[-1], 'tags':['present','form-of','second-person','plural']})
    lookup={'διαρρέω':'διαρρέω','τσαι':'τσάι','τσάι':'τσάι','συλλαμβάνω':'συλλαμβάνω'}
    (base/'mg_pairs.json').write_text(json.dumps(pairs,ensure_ascii=False,indent=2))
    (base/'mg_lookup.json').write_text(json.dumps(lookup,ensure_ascii=False,separators=(',',':')))
    with sqlite3.connect(base/'lookup.db') as db:
        db.executescript('CREATE TABLE lemmas(id INTEGER PRIMARY KEY,text TEXT UNIQUE);CREATE TABLE lookup(form TEXT,lemma_id INTEGER,src TEXT,lang TEXT);')
        for i,word in enumerate(['διαρρέω','τσάι','συλλαμβάνω'],1):db.execute('INSERT INTO lemmas VALUES (?,?)',(i,word))
        db.execute("INSERT INTO lookup VALUES ('τσαι',2,'el','all')")
        db.execute("INSERT INTO lookup VALUES ('separate',1,'grc','all')")
    with sqlite3.connect(base/'spell_index.db') as db:db.execute('CREATE TABLE spell(stripped TEXT PRIMARY KEY,forms TEXT)')
    sources={'input_sha256':{name:sha(base/name) for name in ('mg_pairs.json','mg_lookup.json','lookup.db','spell_index.db')},
             'source_sha256':{'el':'a'*64},'base_revision':'b'*40,'lemma_pos':{'τσάι':['noun']},
             'heads':{'συλλαμβάνω':[{'source':'el'}]},
             'unaccented_heads':{'τσαι':[{'source':'el','headword':'τσαι','pos':'conj','senses':[{'glosses':['και']}]}]},
             'tense_cells':[{'source':'en','source_lemma':'διαρρέω','form':'διαρρέαν','table_header':'present',
                             'original_tags':pairs[0]['tags'],'source_cell':{'form':'διαρρέαν','tags':pairs[0]['tags'][1:]}},
                            {'source':'en','source_lemma':'πέρδομαι','form':'πέρδεστε','table_header':'',
                             'original_tags':pairs[3]['tags'],'source_sense':{'glosses':['present/future of πέρδομαι']}}]}
    out=tmp_path/'out';audit,imperfect=refresh(base,out,sources)
    fixed=json.loads((out/'mg_pairs.json').read_text())
    assert 'present' not in fixed[0]['tags'] and fixed[1]['lemma']=='τσαι'
    assert {frozenset(r['tags']) & {'present','future'} for r in fixed if r['form']=='πέρδεστε'} == {
        frozenset({'present'}),frozenset({'future'})}
    with sqlite3.connect(out/'lookup.db') as db:
        assert db.execute("SELECT l.text FROM lookup k JOIN lemmas l ON l.id=k.lemma_id WHERE form='τσαι'").fetchall()==[('τσαι',)]
        assert db.execute("SELECT lemma_id FROM lookup WHERE src='grc'").fetchall()==[(1,)]
    assert json.loads((out/'mg_lookup.json').read_text())['συνελάμβανε']=='συλλαμβάνω'
    unapply(out,audit,imperfect)
    assert sha(out/'mg_pairs.json')==sha(base/'mg_pairs.json')
    assert sha(out/'mg_lookup.json')==sha(base/'mg_lookup.json')
    for name,tables in [('lookup.db',['lookup','lemmas']),('spell_index.db',['spell'])]:
        with sqlite3.connect(base/name) as before,sqlite3.connect(out/name) as after:
            for table in tables:assert sorted(before.execute('SELECT * FROM '+table))==sorted(after.execute('SELECT * FROM '+table))
    with pytest.raises(ValueError,match='changed after generation'):
        unapply(out,audit,imperfect)


def test_same_pos_orthographic_variants_keep_canonical_preferences():
    lookup={'δυο':('δύο',5)}
    heads={'δυο':[{'source':'en','headword':'δυο','pos':'num','senses':[{'glosses':['two']}]}]}
    assert B.recover_unaccented_heads(lookup,heads,{'δύο':{'num'}})==[]
    assert lookup['δυο'][0]=='δύο'


def test_potentially_mislabeled_form_pages_are_not_past_stem_anchors():
    pairs=[{'form':'άχθηκα','lemma':'άγω','pos':'verb','tags':['first-person','form-of','imperfect','singular']},
           {'form':'παρέσυρα','lemma':'παρασέρνω','pos':'verb','tags':['active','first-person','form-of','imperfect','singular']}]
    heads={'άγω':[{'source':'el'}],'παρασέρνω':[{'source':'en'}]}
    lookup={w:(w,5) for w in heads}
    assert complete_imperfect_pairs(pairs,lookup,heads)['changes']==[]
    assert 'άχθηκαν' not in lookup and 'παρέσυρες' not in lookup


def test_historical_generic_past_cells_still_inherit_the_specific_tense(tmp_path):
    entry={'word':'λύω','pos':'verb','senses':[{'glosses':['release']}],
           'forms':[{'form':'Attic imperfect','tags':['table-tags']},
                    {'form':'ἔλυε','tags':['past','active','indicative','third-person','singular']}]}
    p=tmp_path/'dump.jsonl';p.write_text(json.dumps(entry)+'\n')
    pairs,*_=B.extract_pairs(p,'grc-en')
    assert all({'imperfect','past'}<=set(r['tags']) for r in pairs)


def test_pos_snapshot_keeps_noun_and_name_readings_beside_the_conjunction():
    from scripts.repair_el_verb_pos_snapshot import repair,undo
    old={'τσαι':{'NOUN':'τσάι','PROPN':'Τσάι','CCONJ':'τσάι'}}
    changes={'head_changes':[{'form':'τσαι','original':['τσάι',3],'evidence':[{'source':'el','headword':'τσαι','pos':'conj','senses':[{'glosses':['και']}]}]}]}
    fixed,audit=repair(old,changes,{'changes':[]})
    assert fixed=={'τσαι':{'NOUN':'τσάι','PROPN':'Τσάι','CCONJ':'τσαι'}}
    undo(fixed,audit)
    assert fixed==old


def test_undo_of_an_idempotent_refresh_keeps_earlier_generated_cells():
    pairs=[];lookup={'συλλαμβάνω':('συλλαμβάνω',5)};heads={'συλλαμβάνω':[]}
    original=complete_imperfect_pairs(pairs,lookup,heads)
    before=copy.deepcopy((pairs,lookup))
    repeated=complete_imperfect_pairs(pairs,lookup,heads)
    assert repeated['changes']==[]
    unapply_imperfect_pairs(pairs,lookup,repeated)
    assert (pairs,lookup)==before
    unapply_imperfect_pairs(pairs,lookup,original)
    assert pairs==[] and lookup=={'συλλαμβάνω':('συλλαμβάνω',5)}


def test_full_builder_recovers_pos_readings_without_replacing_other_words():
    lookup={'τσαι':('τσάι',3)};pos={'τσαι':{'NOUN':'τσάι','CCONJ':'τσάι'}}
    heads={'τσαι':[{'source':'el','headword':'τσαι','pos':'conj','senses':[{'glosses':['και']}]}]}
    B.recover_unaccented_heads(lookup,heads,{'τσάι':{'noun'}},pos)
    assert pos=={'τσαι':{'NOUN':'τσάι','CCONJ':'τσαι'}}
