"""Corpus POS recovery and article normalization regressions."""
import json
import sqlite3

import pytest

from dilemma import core
from build import build_treebank_pos_lookup as build

@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(core.Dilemma, '_load_lookups', lambda *a, **k: None)
    monkeypatch.setattr(core.Dilemma, '_check_backend', lambda *a: None)
    monkeypatch.setattr(build, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(build, 'KAIKKI_DIR', tmp_path/'kaikki')
    monkeypatch.setattr(build, 'TREEBANK_DIRS', [])
    monkeypatch.setattr(build, 'AGDT_DIR', tmp_path/'missing_agdt')
    monkeypatch.setattr(build, 'OUTPUT_PATH', tmp_path/'result.json')
    for source in ['GLAUX', 'DIORISIS', 'NT', 'PTNK', 'PEDALION']:
        monkeypatch.setattr(build, source+'_PAIRS_PATH', tmp_path/(source.lower()+'_pairs.json'))
    monkeypatch.setattr(build, 'LOOKUP_DB_PATH', tmp_path/'lookup.db')
    def write(rows, bare, lower, definitions=None):
        definitions = definitions if definitions is not None else [{
            "word": r["lemma"], "lang_code":"grc", "pos":r["pos"],
            "senses":[{"glosses":["A distinct lexical meaning."]}]} for r in rows]
        for directory in ["en-el", "el"]:
            path = build.KAIKKI_DIR/directory/"kaikki.org-dictionary-AncientGreek.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in definitions))
        (tmp_path/'glaux_pairs.json').write_text(json.dumps(rows, ensure_ascii=False))
        for name, data in lower.items():
            (tmp_path/name).write_text(json.dumps(data, ensure_ascii=False))
        with sqlite3.connect(tmp_path/'lookup.db') as db:
            db.execute('CREATE TABLE lemmas (id INTEGER PRIMARY KEY, text TEXT)')
            db.execute('CREATE TABLE lookup (form TEXT, lemma_id INTEGER, lang TEXT)')
            lemmas=sorted({r['lemma'] for r in rows}|set(bare.values()))
            db.executemany('INSERT INTO lemmas VALUES (?,?)',enumerate(lemmas,1))
            ids={v:i for i,v in enumerate(lemmas,1)}
            db.executemany('INSERT INTO lookup VALUES (?,?,?)',[(f,ids[l],'all') for f,l in bare.items()])
    return write

@pytest.mark.parametrize('form', ['ἀκρασία', 'ακρασία', 'ἀκρασίᾳ', 'Ακρασία'])
def test_matching_bare_does_not_hide_conflicting_pos_fallback(inputs, form):
    inputs([{'form':form,'lemma':'ἀκρασία','pos':'noun'}], {form:'ἀκρασία'},
           {'lsj9_pos_lookup.json':{form:{'NOUN':'ἀκράτεια'}}})
    assert build.build_lookup()[form] == {'NOUN':'ἀκρασία'}

@pytest.mark.parametrize('fallback', [None, 'ἀκρασία'])
def test_matching_or_absent_fallback_remains_redundant(inputs, fallback):
    table={} if fallback is None else {'ἀκρασία':{'NOUN':fallback}}
    inputs([{'form':'ἀκρασία','lemma':'ἀκρασία','pos':'noun'}], {'ἀκρασία':'ἀκρασία'},
           {'lsj9_pos_lookup.json':table})
    assert 'ἀκρασία' not in build.build_lookup()


def test_independent_source_priority_is_used(inputs):
    inputs([{'form':'ἀκρασία','lemma':'ἀκρασία','pos':'noun'}], {'ἀκρασία':'ἀκρασία'},
           {'glaux_pos_lookup.json':{'ἀκρασία':{'NOUN':'ἀκρασία'}},
            'lsj9_pos_lookup.json':{'ἀκρασία':{'NOUN':'ἀκράτεια'}}})
    assert 'ἀκρασία' not in build.build_lookup()

@pytest.mark.parametrize(('form','kept'), [('ἀκρασία',False),('ακρασία',True)])
def test_polytonic_ancient_priority_and_combined_monotonic_priority(inputs, form, kept):
    inputs([{'form':form,'lemma':'ἀκρασία','pos':'noun'}], {form:'ἀκρασία'},
           {'ag_pos_lookup.json':{form:{'NOUN':'ἀκρασία'}},
            'mg_pos_lookup.json':{form:{'NOUN':'ἀκράτεια'}}})
    assert (form in build.build_lookup()) is kept


def test_previous_treebank_does_not_mask_lower_fallback(inputs):
    inputs([{'form':'ἀκρασία','lemma':'ἀκρασία','pos':'noun'}], {'ἀκρασία':'ἀκρασία'},
           {'treebank_pos_lookup.json':{'ἀκρασία':{'NOUN':'ἀκρασία'}},
            'lsj9_pos_lookup.json':{'ἀκρασία':{'NOUN':'ἀκράτεια'}}})
    assert build.build_lookup()['ἀκρασία'] == {'NOUN':'ἀκρασία'}


def test_lower_priority_conflict_does_not_promote_weak_or_absent_corpus(inputs):
    inputs([{'form':'ἀκρασία','lemma':'ἀκρασία','pos':'noun'}], {'ἀκρασία':'ἀκρασία'},
           {'lsj9_pos_lookup.json':{'ἀκρασία':{'NOUN':'ἀκράτεια'}}})
    build.DIORISIS_PAIRS_PATH.write_text(build.GLAUX_PAIRS_PATH.read_text())
    build.GLAUX_PAIRS_PATH.unlink()
    assert 'ἀκρασία' not in build.build_lookup()

@pytest.mark.parametrize('form', ['ΑΚΡΑΣΙΑ', 'ἀκρασίὰ'])
def test_runtime_normalized_variants_find_conflict(inputs,form):
    inputs([{'form':form,'lemma':'ἀκρασία','pos':'noun'}], {form:'ἀκρασία'},
           {'lsj9_pos_lookup.json':{'ακρασια':{'NOUN':'ἀκράτεια'}}})
    assert build.build_lookup()[form] == {'NOUN':'ἀκρασία'}


@pytest.mark.parametrize('kind', ['missing', 'wrong_pos', 'reference'])
def test_unsupported_corpus_target_does_not_replace_independent_fallback(inputs, kind):
    definitions = []
    if kind == 'wrong_pos':
        definitions = [{'word':'ἰδικός','lang_code':'grc','pos':'adj',
                        'senses':[{'glosses':['private, particular']}]}]
    elif kind == 'reference':
        definitions = [{'word':'ἰδικός','lang_code':'grc','pos':'adv',
                        'senses':[{'glosses':['alternative form of ἰδικῶς'],
                                   'alt_of':[{'word':'ἰδικῶς'}]}]}]
    inputs([{'form':'ἰδικῶς','lemma':'ἰδικός','pos':'adv'}], {'ἰδικῶς':'ἰδικός'},
           {'lsj9_pos_lookup.json':{'ἰδικῶς':{'ADV':'ἰδικῶς'}}}, definitions)
    assert 'ἰδικῶς' not in build.build_lookup()


def test_missing_dump_does_not_authorize_new_override(inputs, monkeypatch):
    inputs([{'form':'ἀκρασία','lemma':'ἀκρασία','pos':'noun'}], {'ἀκρασία':'ἀκρασία'},
           {'lsj9_pos_lookup.json':{'ἀκρασία':{'NOUN':'ἀκράτεια'}}})
    monkeypatch.setattr(build, 'KAIKKI_DIR', None)
    assert 'ἀκρασία' not in build.build_lookup()


def test_new_projected_corpus_entry_cannot_hide_an_exact_supported_vote(inputs):
    # A case-distinct form gets a necessary bare-lookup correction. Its new
    # lowercase projection must not consume the lower-case form's own vote.
    inputs([{'form':'Ἀκρασία','lemma':'ἀκράτεια','pos':'noun'},
            {'form':'ἀκρασία','lemma':'ἀκρασία','pos':'noun'}],
           {'Ἀκρασία':'ἀκρασία','ἀκρασία':'ἀκρασία'}, {})
    result = build.build_lookup()
    assert result['Ἀκρασία'] == {'NOUN':'ἀκράτεια'}
    assert result['ἀκρασία'] == {'NOUN':'ἀκρασία'}


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('both_defined', [False, True])
def test_new_alias_retains_rejected_and_supported_competing_claims(inputs, reverse, both_defined):
    rows=[{'form':'ἄδος','lemma':'πρῶτος','pos':'noun'},
          {'form':'ᾄδος','lemma':'δεύτερος','pos':'noun'}]
    definitions=[{'word':'πρῶτος','lang_code':'grc','pos':'noun',
                  'senses':[{'glosses':['A first distinct noun.']}]}]
    if both_defined:
        definitions.append({'word':'δεύτερος','lang_code':'grc','pos':'noun',
                            'senses':[{'glosses':['A second distinct noun.']}]})
    inputs(list(reversed(rows)) if reverse else rows,
           {'ἄδος':'πρῶτος','ᾄδος':'δεύτερος'},
           {'lsj9_pos_lookup.json':{'ἄδος':{'NOUN':'τρίτος'},'ᾄδος':{'NOUN':'τέταρτος'}}},
           definitions)
    result=build.build_lookup()
    assert result['ἄδος'] == {'NOUN':'πρῶτος'}
    assert 'άδος' not in result
    if both_defined:
        assert result['ᾄδος'] == {'NOUN':'δεύτερος'}
    else:
        assert 'ᾄδος' not in result


@pytest.mark.parametrize('reverse', [False, True])
def test_projection_merges_parts_of_speech_and_withholds_competing_owners(reverse):
    rows=[('ἄδος',{'NOUN':'πρῶτος','ADJ':'λευκός'}),
          ('ᾄδος',{'NOUN':'δεύτερος','VERB':'ᾄδω'})]
    result=build._project_pos_entries(dict(reversed(rows) if reverse else rows))
    assert result['άδος'] == {'ADJ':'λευκός','VERB':'ᾄδω'}
    assert result['ἄδος'] == rows[0][1]
    assert result['ᾄδος'] == rows[1][1]


@pytest.mark.parametrize('reverse', [False, True])
def test_exact_noun_retains_projected_adjective(reverse):
    rows = [('Κενταύρειον', {'ADJ': 'Κενταύρειος'}),
            ('κενταύρειον', {'NOUN': 'κενταύρειον'})]
    original = dict(reversed(rows) if reverse else rows)
    result = build._project_pos_entries(original,
        defined_targets={("Κενταύρειος", "ADJ")})
    assert result['κενταύρειον'] == {
        'NOUN': 'κενταύρειον', 'ADJ': 'Κενταύρειος'}
    assert original == dict(rows)


def test_exact_owner_wins_only_its_own_part_of_speech():
    result = build._project_pos_entries({
        'Κενταύρειον': {'ADJ': 'Κενταύρειος', 'NOUN': 'ἄλλος'},
        'κενταύρειον': {'NOUN': 'κενταύρειον'},
    }, defined_targets={("Κενταύρειος", "ADJ")})
    assert result['κενταύρειον'] == {
        'NOUN': 'κενταύρειον', 'ADJ': 'Κενταύρειος'}



@pytest.mark.parametrize("kind", ["missing", "wrong_pos", "reference"])
def test_new_pos_beside_exact_form_requires_matching_independent_definition(inputs, kind):
    definitions = []
    if kind == "wrong_pos":
        definitions = [{"word": "ἀπολάω", "lang_code": "grc", "pos": "noun",
                        "senses": [{"glosses": ["A noun meaning."]}]}]
    elif kind == "reference":
        definitions = [{"word": "ἀπολάω", "lang_code": "grc", "pos": "verb",
                        "senses": [{"glosses": ["form of ἀπολαύω"],
                                    "form_of": [{"word": "ἀπολαύω"}]}]}]
    inputs([], {}, {}, definitions)
    result = build._project_pos_entries({
        "Πείραι": {"VERB": "ἀπολάω"}, "πείραι": {"NOUN": "πεῖρα"}})
    assert result["πείραι"] == {"NOUN": "πεῖρα"}
    assert result["Πείραι"] == {"VERB": "ἀπολάω"}


def test_new_pos_beside_exact_form_accepts_fresh_definition(inputs):
    inputs([], {}, {}, [{"word": "Κενταύρειος", "lang_code": "grc", "pos": "adj",
                         "senses": [{"glosses": ["of a centaur"]}]}])
    result = build._project_pos_entries({
        "Κενταύρειον": {"ADJ": "Κενταύρειος"},
        "κενταύρειον": {"NOUN": "κενταύρειον"}})
    assert result["κενταύρειον"] == {"NOUN": "κενταύρειον", "ADJ": "Κενταύρειος"}



@pytest.mark.parametrize("grammar,member,accepted", [
    ("ον", True, True), ("ὁ", True, False), ("ον", False, False)])
def test_projection_accepts_only_matching_merged_lsj_pos(inputs, monkeypatch, grammar, member, accepted):
    import hashlib
    inputs([], {}, {}, [])
    raw = json.dumps([{"headword": "Κενταύρειος", "grammar": grammar}], ensure_ascii=False).encode()
    path = build.DATA_DIR / "lsj9.json"
    path.write_bytes(raw)
    (build.DATA_DIR / "grc_lexical_heads.json").write_text(json.dumps({
        "inputs": {"lsj9_headwords": {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}},
        "heads": {"Κενταύρειος": ["lsj9_headwords"]} if member else {}}))
    monkeypatch.setattr(build, "LSJ9_HEADWORDS", path)
    result = build._project_pos_entries({
        "Κενταύρειον": {"ADJ": "Κενταύρειος"}, "κενταύρειον": {"NOUN": "κενταύρειον"}})
    assert (result["κενταύρειον"].get("ADJ") == "Κενταύρειος") is accepted


def test_projection_rejects_lsj_input_that_does_not_match_merged_manifest(inputs, monkeypatch):
    inputs([], {}, {}, [])
    path = build.DATA_DIR / "lsj9.json";path.write_text('[]')
    (build.DATA_DIR / "grc_lexical_heads.json").write_text(json.dumps({
        "inputs": {"lsj9_headwords": {"size": 2, "sha256": "0"*64}}, "heads": {}}))
    monkeypatch.setattr(build, "LSJ9_HEADWORDS", path)
    with pytest.raises(ValueError, match="merged lexical evidence"):
        build._project_pos_entries({
            "Κενταύρειον": {"ADJ": "Κενταύρειος"}, "κενταύρειον": {"NOUN": "κενταύρειον"}})


def test_missing_dumps_do_not_broaden_existing_exact_pos(monkeypatch):
    monkeypatch.setattr(build, "KAIKKI_DIR", None)
    result = build._project_pos_entries({
        "Πείραι": {"VERB": "ἀπολάω"}, "πείραι": {"NOUN": "πεῖρα"}})
    assert result["πείραι"] == {"NOUN": "πεῖρα"}


def test_conflicting_projection_cannot_remove_another_exact_part_of_speech():
    result = build._project_pos_entries({
        'Κενταύρειον': {'ADJ': 'Κενταύρειος'},
        'κενταύρειον': {'NOUN': 'κενταύρειον'},
    }, {'κενταύρειον': {'NOUN', 'ADJ'}})
    assert result['κενταύρειον'] == {'NOUN': 'κενταύρειον'}


def test_competing_adjective_projections_preserve_exact_noun():
    result = build._project_pos_entries({
        'Κενταύρειον': {'ADJ': 'Κενταύρειος'},
        'ΚΕΝΤΑΎΡΕΙΟΝ': {'ADJ': 'ἄλλος'},
        'κενταύρειον': {'NOUN': 'κενταύρειον'},
    })
    assert result['κενταύρειον'] == {'NOUN': 'κενταύρειον'}


@pytest.mark.parametrize('source', ['en-el', 'el'])
def test_either_dictionary_can_supply_an_independent_definition(inputs, source):
    inputs([{'form':'ἀκρασία','lemma':'ἀκρασία','pos':'noun'}], {'ἀκρασία':'ἀκρασία'},
           {'lsj9_pos_lookup.json':{'ἀκρασία':{'NOUN':'ἀκράτεια'}}})
    other = 'el' if source == 'en-el' else 'en-el'
    (build.KAIKKI_DIR/other/'kaikki.org-dictionary-AncientGreek.jsonl').write_text('')
    assert build.build_lookup()['ἀκρασία'] == {'NOUN':'ἀκρασία'}


@pytest.fixture
def article_model(monkeypatch):
    from types import SimpleNamespace
    model = object.__new__(core.Dilemma)
    model.lang = 'grc'
    model._convention_name = 'wiktionary'
    model._resolve_articles = False
    monkeypatch.setattr(model, '_pos_table_lookup', lambda word, upos: 'τῷν')
    monkeypatch.setattr(model, '_apply_convention', lambda lemma: lemma)
    monkeypatch.setattr(model, '_lemmatize_verbose_impl',
                        lambda word, **kw: [SimpleNamespace(lemma='baseline')])
    monkeypatch.setattr(model, 'lemmatize_batch',
                        lambda words, **kw: ['baseline'] * len(words))
    return model


@pytest.mark.parametrize('lang', ['grc', 'all'])
@pytest.mark.parametrize('resolve', [False, True])
@pytest.mark.parametrize('form', ['τών', 'Τών', 'τῶν'])
def test_explicit_article_pos_rejects_noncanonical_target(article_model, lang, resolve, form):
    model = article_model
    model.lang, model._resolve_articles = lang, resolve
    assert model.lemmatize_pos(form, 'DET', guess=False) == 'ὁ'
    assert model.lemmatize_batch_pos([form], ['DET'], guess=False) == ['ὁ']


@pytest.mark.parametrize('target', ['ὁ', 'ο'])
def test_article_pos_preserves_existing_canonical_target(article_model, monkeypatch, target):
    monkeypatch.setattr(article_model, '_pos_table_lookup', lambda word, upos: target)
    assert article_model.lemmatize_pos('τών', 'DET', guess=False) == target
    assert article_model.lemmatize_batch_pos(['τών'], ['DET'], guess=False) == [target]


@pytest.mark.parametrize('form', ['ἤ', 'ὅ', 'οὗ', 'αὐτῶν', 'τίνων'])
def test_article_pos_does_not_absorb_near_accents_or_pronouns(article_model, form):
    assert article_model.lemmatize_pos(form, 'DET', guess=False) == 'τῷν'
    assert article_model.lemmatize_batch_pos([form], ['DET'], guess=False) == ['τῷν']


@pytest.mark.parametrize('upos', ['PRON', 'NOUN', 'ADJ'])
def test_article_pos_does_not_change_other_pos(article_model, upos):
    assert article_model.lemmatize_pos('τών', upos, guess=False) == 'τῷν'
    assert article_model.lemmatize_batch_pos(['τών'], [upos], guess=False) == ['τῷν']


def test_article_pos_does_not_create_a_missing_pos_entry(article_model, monkeypatch):
    monkeypatch.setattr(article_model, '_pos_table_lookup', lambda word, upos: None)
    assert article_model.lemmatize_pos('τών', 'DET', guess=False) == 'baseline'
    assert article_model.lemmatize_batch_pos(['τών'], ['DET'], guess=False) == ['baseline']


def test_article_pos_keeps_attestation_gate(article_model, monkeypatch):
    monkeypatch.setattr(article_model, '_resolve_attested_form', lambda word: None)
    monkeypatch.setattr(article_model, 'lemmatize_batch', lambda words, **kw: [None] * len(words))
    assert article_model.lemmatize_pos('τών', 'DET', guess=False, attested_only=True) is None
    assert article_model.lemmatize_batch_pos(['τών'], ['DET'], guess=False,
                                            attested_only=True) == [None]


def test_explicit_article_pos_uses_modern_convention(article_model):
    article_model._convention_name = 'triantafyllidis'
    assert article_model.lemmatize_pos('των', 'DET', guess=False) == 'ο'
    assert article_model.lemmatize_batch_pos(['των'], ['DET'], guess=False) == ['ο']


def test_article_pos_does_not_enable_untagged_resolution(article_model):
    assert article_model._resolve_closed_class('τών') is None
    article_model.lemmatize_pos('τών', 'DET', guess=False)
    assert article_model._resolve_articles is False
    assert article_model._resolve_closed_class('τών') is None
    article_model._resolve_articles = True
    assert article_model._resolve_closed_class('τών') == 'ὁ'


def test_tied_corpus_owners_do_not_acquire_a_new_pos_override(inputs):
    inputs([{'form':'κρεῖσσον','lemma':'κρατύς','pos':'adj'},
            {'form':'κρεῖσσον','lemma':'κρείττων','pos':'adj'}],
           {'κρεῖσσον':'κρατύς'},
           {'lsj9_pos_lookup.json':{'κρεῖσσον':{'ADJ':'κρείσσων'}}})
    assert 'κρεῖσσον' not in build.build_lookup()


@pytest.mark.parametrize('reverse', [False, True])
def test_recovery_preserves_tied_claims_without_selecting_an_owner(monkeypatch, reverse):
    votes=[('κρατύς',2),('κρείττων',2)]
    if reverse:
        votes.reverse()
    corpus={'κρεῖσσον':{'ADJ':dict(votes)}}
    monkeypatch.setattr(build, '_defined_pos_targets', lambda wanted: wanted)
    monkeypatch.setattr(build, '_lower_priority_pos_models',
                        lambda: pytest.fail('A tied vote cannot authorize an override'))
    lookup={}
    assert build._restore_shadowed_corpus_votes(
        lookup, {'κρεῖσσον':('ADJ','κρατύς')}, corpus) == (0,{})
    assert lookup == {}
    assert corpus == {'κρεῖσσον':{'ADJ':dict(votes)}}


def test_duplicate_same_owner_rows_do_not_create_a_corpus_tie(inputs):
    row={'form':'ἀκρασία','lemma':'ἀκρασία','pos':'noun'}
    inputs([row,row], {'ἀκρασία':'ἀκρασία'},
           {'lsj9_pos_lookup.json':{'ἀκρασία':{'NOUN':'ἀκράτεια'}}})
    assert build.build_lookup()['ἀκρασία'] == {'NOUN':'ἀκρασία'}


@pytest.mark.parametrize('lang', ['grc', 'all'])
@pytest.mark.parametrize('form', ['ΤΟΥΣ', 'Τα', 'Τους', 'Τοῦ', 'τα', 'τους'])
def test_article_correction_precedes_multiple_candidate_selection(
        article_model, monkeypatch, lang, form):
    from types import SimpleNamespace
    model = article_model
    model.lang = lang
    monkeypatch.setattr(model, '_pos_table_lookup', lambda word, upos: form.lower())
    monkeypatch.setattr(model, '_lemmatize_verbose_impl', lambda word, **kw:
                        [SimpleNamespace(lemma=form), SimpleNamespace(lemma=form.lower())])
    monkeypatch.setattr(model, 'lemmatize_batch', lambda words, **kw: [form] * len(words))
    assert model.lemmatize_pos(form, 'DET', guess=False) == 'ὁ'
    assert model.lemmatize_batch_pos([form], ['DET'], guess=False) == ['ὁ']


@pytest.mark.parametrize("form,lemma,gloss,template", [
    ("δεσμαῖς","δεσμά","neuter heteroclitic nominative/accusative/vocative plural of δεσμός m (desmós)","grc-noun form"),
    ("ΚΣ","ΚΣ","nomen sacrum form of κῡ́ριος (kū́rios)",""),
    ("Κητείων","Κήτειοι","Template:infl of","grc-noun form"),
])
def test_nominal_form_reference_cannot_authorize_new_pos_override(inputs,form,lemma,gloss,template):
    entry={"word":lemma,"lang_code":"grc","pos":"noun","senses":[{"glosses":[gloss]}]}
    if template:
        entry["head_templates"]=[{"name":template}]
    inputs([{"form":form,"lemma":lemma,"pos":"noun"}],{form:lemma},
           {"lsj9_pos_lookup.json":{form:{"NOUN":"θήκη"}}},definitions=[entry])
    # These are English source records; the Greek source has no separate head.
    (build.KAIKKI_DIR/"el/kaikki.org-dictionary-AncientGreek.jsonl").write_text("")
    assert form not in build.build_lookup()


def scalar_models(monkeypatch, bare):
    original = build._lower_priority_pos_models
    def models():
        result = original()
        for model in result:
            model._lookup = bare
        return result
    monkeypatch.setattr(build, '_lower_priority_pos_models', models)


@pytest.mark.parametrize(('form', 'capital'), [('παρούσῃ', 'Παρούση'),
                                             ('παρούσης', 'Παρούσης')])
def test_scalar_case_shadow_keeps_source_verb_vote(inputs, monkeypatch, form, capital):
    bare = {form: 'πάρειμι', capital: capital}
    inputs([{'form': form, 'lemma': 'πάρειμι', 'pos': 'verb'}], bare, {})
    scalar_models(monkeypatch, bare)
    result = build.build_lookup()
    assert result[form] == {'VERB': 'πάρειμι'}
    assert result[build.to_monotonic(form)] == {'VERB': 'πάρειμι'}


@pytest.mark.parametrize('query', ['Παρούση', 'ΠΑΡΟΎΣΗ', 'Παρούσῃ'])
def test_all_supported_case_queries_find_scalar_conflict(inputs, monkeypatch, query):
    bare = {'παρούσῃ': 'πάρειμι', query: query}
    inputs([{'form': 'παρούσῃ', 'lemma': 'πάρειμι', 'pos': 'verb'}], bare, {})
    scalar_models(monkeypatch, bare)
    assert build.build_lookup()['παρούσῃ'] == {'VERB': 'πάρειμι'}


@pytest.mark.parametrize('query', ['Παρούση', 'ΠΑΡΟΎΣΗ', 'Παρούσῃ'])
def test_same_owner_case_query_is_still_redundant(inputs, monkeypatch, query):
    bare = {'παρούσῃ': 'πάρειμι', query: 'πάρειμι'}
    inputs([{'form': 'παρούσῃ', 'lemma': 'πάρειμι', 'pos': 'verb'}], bare, {})
    scalar_models(monkeypatch, bare)
    assert not build.build_lookup()


def test_unaccented_other_word_does_not_supply_scalar_conflict(inputs, monkeypatch):
    bare = {'παρούσῃ': 'πάρειμι', 'Παρουση': 'ἄλλος'}
    inputs([{'form': 'παρούσῃ', 'lemma': 'πάρειμι', 'pos': 'verb'}], bare, {})
    scalar_models(monkeypatch, bare)
    assert not build.build_lookup()


def test_scalar_case_shadow_with_correct_pos_fallback_is_redundant(inputs, monkeypatch):
    bare = {'παρούσῃ': 'πάρειμι', 'Παρούση': 'Παρούση'}
    inputs([{'form': 'παρούσῃ', 'lemma': 'πάρειμι', 'pos': 'verb'}], bare,
           {'lsj9_pos_lookup.json': {'παρούση': {'VERB': 'πάρειμι'}}})
    scalar_models(monkeypatch, bare)
    assert not build.build_lookup()


def test_scalar_case_shadow_does_not_break_tied_vote_guard(inputs, monkeypatch):
    bare = {'παρούσῃ': 'πάρειμι', 'Παρούση': 'Παρούση'}
    inputs([{'form': 'παρούσῃ', 'lemma': 'πάρειμι', 'pos': 'verb'},
            {'form': 'παρούσῃ', 'lemma': 'παρόω', 'pos': 'verb'}], bare, {})
    scalar_models(monkeypatch, bare)
    assert not build.build_lookup()


def test_scalar_case_shadow_requires_independent_matching_pos(inputs, monkeypatch):
    bare = {'παρούσῃ': 'πάρειμι', 'Παρούση': 'Παρούση'}
    inputs([{'form': 'παρούσῃ', 'lemma': 'πάρειμι', 'pos': 'verb'}], bare, {},
           [{'word': 'πάρειμι', 'lang_code': 'grc', 'pos': 'noun',
             'senses': [{'glosses': ['An independent noun.']}]}])
    scalar_models(monkeypatch, bare)
    assert not build.build_lookup()


@pytest.mark.parametrize('batch', [False, True])
def test_retained_vote_repairs_pos_api_without_changing_proper_noun(inputs, monkeypatch, batch):
    from types import SimpleNamespace
    bare = {'παρούσῃ': 'πάρειμι', 'Παρούση': 'Παρούση'}
    inputs([{'form': 'παρούσῃ', 'lemma': 'πάρειμι', 'pos': 'verb'}], bare, {})
    scalar_models(monkeypatch, bare)
    table = build.build_lookup()
    model = object.__new__(core.Dilemma)
    model.lang = 'grc'
    model._convention_name = 'wiktionary'
    model._resolve_articles = False
    model._pos_lookup = table
    model._pos_ag_lookup = table
    monkeypatch.setattr(model, '_apply_convention', lambda lemma: lemma)
    monkeypatch.setattr(model, '_lemmatize_verbose_impl',
                        lambda word, **kw: [SimpleNamespace(lemma='Παρούση')])
    monkeypatch.setattr(model, 'lemmatize_batch', lambda words, **kw: ['Παρούση'] * len(words))
    if batch:
        assert model.lemmatize_batch_pos(['Παρούση'], ['VERB'], guess=False) == ['πάρειμι']
        assert model.lemmatize_batch_pos(['Παρούση'], ['PROPN'], guess=False) == ['Παρούση']
    else:
        assert model.lemmatize_pos('Παρούση', 'VERB', guess=False) == 'πάρειμι'
        assert model.lemmatize_pos('Παρούση', 'PROPN', guess=False) == 'Παρούση'

