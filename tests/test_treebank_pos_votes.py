"""Corpus morphology variants cannot multiply one POS-disambiguation vote."""
import json

from build import build_treebank_pos_lookup as build


def test_distinct_morphology_does_not_outvote_an_independent_source(tmp_path, monkeypatch):
    glaux = [
        {'form': 'θεῶ', 'lemma': 'θεός', 'pos': 'noun'},
        {'form': 'γραφῆς', 'lemma': 'γραφεύς', 'pos': 'noun'},
        {'form': 'γραφῆς', 'lemma': 'γραφή', 'pos': 'noun'},
    ]
    diorisis = [
        {'form': 'θεῶ', 'lemma': 'θεά', 'pos': 'noun', 'tags': [case]}
        for case in ('nominative', 'dative', 'genitive')
    ] + [
        {'form': 'θεῶ', 'lemma': 'θεάομαι', 'pos': 'verb'},
        {'form': 'γραφῆς', 'lemma': 'γραφή', 'pos': 'noun'},
        {'form': 'γραφῆς', 'lemma': 'γράφω', 'pos': 'verb'},
    ]
    for name, rows in (('GLAUX_PAIRS_PATH', glaux), ('DIORISIS_PAIRS_PATH', diorisis)):
        path = tmp_path/(name + '.json')
        path.write_text(json.dumps(rows, ensure_ascii=False))
        monkeypatch.setattr(build, name, path)
    for name in ('NT_PAIRS_PATH', 'PTNK_PAIRS_PATH', 'PEDALION_PAIRS_PATH', 'LOOKUP_DB_PATH'):
        monkeypatch.setattr(build, name, tmp_path/name)
    monkeypatch.setattr(build, 'AGDT_DIR', tmp_path/'no_agdt')
    monkeypatch.setattr(build, 'TREEBANK_DIRS', [])
    monkeypatch.setattr(build, 'OUTPUT_PATH', tmp_path/'result.json')
    result = build.build_lookup()
    # Three Diorisis morphology rows are still one weight-1 NOUN edge;
    # they cannot overturn GLAUx's independently annotated weight-2 edge.
    assert result['θεῶ'] == {'NOUN': 'θεός', 'VERB': 'θεάομαι'}
    # The same edge in two independent corpora still receives both votes.
    assert result['γραφῆς'] == {'NOUN': 'γραφή', 'VERB': 'γράφω'}
