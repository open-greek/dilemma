"""Exact textual evidence may preserve a rare spelling beside the usual one."""
import json
import sqlite3

import pytest

from export_hunspell import (
    FormProfileEvidence, add_cited_corpus_forms, cased_form_key,
    exact_form_key, finalize_grc_pairs, mark_skeleton,
)


@pytest.fixture
def cited_evidence(tmp_path):
    profile = tmp_path / 'form_profile.db'
    citations = tmp_path / 'form_citations.db'
    lexical = tmp_path / 'grc_lexical_heads.json'
    with sqlite3.connect(profile) as conn:
        conn.executescript('CREATE TABLE forms(form_id INTEGER,form TEXT);'
                           'CREATE TABLE form_profile(form_id INTEGER,source_counts_json TEXT);')
        conn.execute('INSERT INTO forms VALUES(1,?)', ('ὤοντο',))
        conn.execute('INSERT INTO form_profile VALUES(1,?)', (json.dumps({'diorisis': 2}),))
    with sqlite3.connect(citations) as conn:
        conn.executescript('CREATE TABLE citations(form_id INTEGER,work_id TEXT,source TEXT,'
                           'locus TEXT,locus_scheme TEXT,count INTEGER);')
        conn.executemany('INSERT INTO citations VALUES(?,?,?,?,?,?)', [
            (1, '0540-012', 'diorisis', '93', 'diorisis-sentence', 1),
            (1, '0010-017', 'diorisis', '47', 'diorisis-sentence', 1),
        ])
    metadata = {'schema_version': '1', 'content_hash': 'a' * 64,
                'source_sha': json.dumps({'diorisis_xml': 'b' * 64})}
    for path in (profile, citations):
        with sqlite3.connect(path) as conn:
            conn.execute('CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT)')
            conn.executemany('INSERT INTO meta VALUES(?,?)', metadata.items())
    lexical.write_text(json.dumps({'schema_version': 2, 'heads': {
        'οἴομαι': ['lsj9_headwords', 'wiktionary_el_grc'],
    }}, ensure_ascii=False))
    return dict(profile_path=profile, citations_path=citations, lexical_path=lexical)


def test_exact_cited_spelling_survives_beside_the_usual_reading(cited_evidence):
    original = [('ᾤοντο', 'οἴομαι')]
    pairs, protected, audit = add_cited_corpus_forms(original, **cited_evidence)
    assert original == [('ᾤοντο', 'οἴομαι')]
    assert pairs == [('ᾤοντο', 'οἴομαι'), ('ὤοντο', 'οἴομαι')]
    assert protected == {'ὤοντο'}
    assert {w['work_id'] for w in audit[0]['witnesses']} == {'0540-012', '0010-017'}
    assert all(w['count'] == 1 for w in audit[0]['witnesses'])
    assert audit[0]['lemma_evidence']['sources'] == ['lsj9_headwords', 'wiktionary_el_grc']
    assert 'corpus lemma unknown' in audit[0]['interpretation_source']
    assert 'third-person plural' in audit[0]['interpretation']
    evidence = FormProfileEvidence(
        exact={exact_form_key('ὤοντο'): 2, exact_form_key('ᾤοντο'): 1000},
        dominant={mark_skeleton('ᾤοντο'): 1000},
        treebank={exact_form_key('ὤοντο'): 2}, metadata={},
        cased={cased_form_key('ὤοντο'): 2, cased_form_key('ᾤοντο'): 1000},
    )
    final, report = finalize_grc_pairs(
        pairs, evidence=evidence, compatibility_forms=set(), textbook_forms=set(),
        export_overrides={}, protected_forms=protected,
    )
    assert set(final) == set(pairs)
    assert report['dominated'] == 0


@pytest.mark.parametrize('update', [
    "DELETE FROM citations WHERE work_id='0540-012'",
    "UPDATE citations SET source='pg' WHERE work_id='0540-012'",
    "UPDATE citations SET locus='94' WHERE work_id='0540-012'",
    "UPDATE citations SET locus_scheme='cts' WHERE work_id='0540-012'",
    "UPDATE citations SET count=0 WHERE work_id='0540-012'",
    "UPDATE citations SET form_id=2 WHERE work_id='0540-012'",
])
def test_each_exact_current_corpus_witness_is_required(cited_evidence, update):
    with sqlite3.connect(cited_evidence['citations_path']) as conn:
        conn.execute(update)
    assert add_cited_corpus_forms([], **cited_evidence) == ([], set(), [])


@pytest.mark.parametrize('form', ['ᾤοντο', 'ωοντο', 'Ὤοντο', 'ὤοντό'])
def test_differently_marked_or_cased_profile_does_not_authorize_the_form(cited_evidence, form):
    with sqlite3.connect(cited_evidence['profile_path']) as conn:
        conn.execute('UPDATE forms SET form=?', (form,))
    assert add_cited_corpus_forms([], **cited_evidence) == ([], set(), [])


def test_citation_rows_cannot_replace_current_profile_support(cited_evidence):
    with sqlite3.connect(cited_evidence['profile_path']) as conn:
        conn.execute('UPDATE form_profile SET source_counts_json=?', (json.dumps({'diorisis': 1, 'pg': 100}),))
    assert add_cited_corpus_forms([], **cited_evidence) == ([], set(), [])


@pytest.mark.parametrize('heads', [{'οἰομαι': ['lsj9_headwords']}, {'οἴομαι': []}, {}])
def test_exact_positive_lexical_head_is_required(cited_evidence, heads):
    cited_evidence['lexical_path'].write_text(json.dumps({'schema_version': 2, 'heads': heads}))
    assert add_cited_corpus_forms([], **cited_evidence) == ([], set(), [])


@pytest.mark.parametrize('key', ['profile_path', 'citations_path', 'lexical_path'])
def test_missing_required_source_evidence_fails_closed(cited_evidence, key):
    cited_evidence[key].unlink()
    with pytest.raises(FileNotFoundError, match='cited corpus-form evidence is required'):
        add_cited_corpus_forms([], **cited_evidence)


def test_repeated_admission_preserves_existing_competing_readings(cited_evidence):
    original = [('ὤοντο', 'οἶμαι'), ('ᾤοντο', 'οἴομαι'), ('ὤοντο', 'οἴομαι')]
    pairs, protected, _ = add_cited_corpus_forms(original, **cited_evidence)
    assert pairs == original
    assert protected == {'ὤοντο'}


@pytest.mark.parametrize('key,value', [
    ('schema_version', '2'), ('schema_version', ''),
    ('content_hash', 'c' * 64), ('content_hash', 'not-a-hash'),
    ('source_sha', json.dumps({'diorisis_xml': 'c' * 64})),
    ('source_sha', '{}'), ('source_sha', '[]'), ('source_sha', 'not-json'),
    ('source_sha', json.dumps({'diorisis_xml': 'not-a-hash'})),
])
def test_invalid_or_mismatched_pair_metadata_is_rejected(cited_evidence, key, value):
    with sqlite3.connect(cited_evidence['citations_path']) as conn:
        conn.execute('UPDATE meta SET value=? WHERE key=?', (value, key))
    with pytest.raises(ValueError, match='profile/citation'):
        add_cited_corpus_forms([], **cited_evidence)


@pytest.mark.parametrize('key', ['profile_path', 'citations_path'])
def test_prior_schema_without_pair_metadata_is_rejected(cited_evidence, key):
    with sqlite3.connect(cited_evidence[key]) as conn:
        conn.execute('DROP TABLE meta')
    with pytest.raises(ValueError, match='missing profile/citation pair metadata'):
        add_cited_corpus_forms([], **cited_evidence)


def _build_pair(directory, forms):
    from build import build_form_attestation as builder
    directory.mkdir()
    profiles = {}
    for i, _ in enumerate(forms):
        profiles[i] = builder.FormProfile()
        for _ in range(2):
            profiles[i].observe('diorisis', 'verb')
            profiles[i].add_deduped('oratory', -4, 'Attic')
    citation_tmp = directory / 'citations.db.tmp'
    sink = builder.CiteSink(citation_tmp)
    for i, _ in enumerate(forms):
        sink.add([(i, '0540-012', 'diorisis', '93', 'diorisis-sentence', 1, -4),
                  (i, '0010-017', 'diorisis', '47', 'diorisis-sentence', 1, -4)])
    profile = directory / 'form_profile.db'
    citations = directory / 'form_citations.db'
    content_hash = builder.write_dbs(forms, profiles, {}, sink, citation_tmp, 200,
                                    {'diorisis_xml': 'b' * 64}, profile, citations)
    return profile, citations, content_hash


def test_builder_writes_the_same_pair_identity_to_both_outputs(tmp_path, cited_evidence):
    profile, citations, content_hash = _build_pair(tmp_path / 'pair', ['ὤοντο'])
    metadata = []
    for path in (profile, citations):
        with sqlite3.connect(path) as conn:
            metadata.append(dict(conn.execute('SELECT key,value FROM meta')))
    assert metadata[0] == metadata[1]
    assert metadata[0]['content_hash'] == content_hash
    pairs, _, audit = add_cited_corpus_forms([], profile_path=profile,
        citations_path=citations, lexical_path=cited_evidence['lexical_path'])
    assert pairs == [('ὤοντο', 'οἴομαι')]
    assert audit[0]['profile_content_hash'] == content_hash


def test_mixed_builds_cannot_authorize_reused_form_ids(tmp_path, cited_evidence):
    profile, _, current_hash = _build_pair(tmp_path / 'current', ['λόγος', 'ὤοντο'])
    _, stale_citations, stale_hash = _build_pair(tmp_path / 'stale', ['ὤοντο', 'λόγος'])
    assert current_hash != stale_hash
    with pytest.raises(ValueError, match='mismatched profile/citation databases'):
        add_cited_corpus_forms([], profile_path=profile,
            citations_path=stale_citations, lexical_path=cited_evidence['lexical_path'])
