import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

import overlay_lsj as overlay
from build.closed_class_elisions import plan_repairs
from scripts import build_source_alias_manifest as publication


def evidence():
    entry = {'word': 'οὗτος', 'pos': 'pron', 'lang_code': 'grc',
             'senses': [{'glosses': ['this; this one']}],
             'forms': [{'form': 'ταῦτα', 'tags': ['plural', 'nominative']}]}
    pair = {'form': 'Ταῦτ᾽', 'lemma': 'οὗτος', 'pos': 'pron',
            'tags': ['plural', 'nominative']}
    return entry, [('current', [pair])]


def dumps(tmp_path):
    entry, _ = evidence()
    for label, (_, relative) in overlay.CLOSED_CLASS_DUMPS.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entry, ensure_ascii=False) + '\n')
    return tmp_path


def empty_alias_stats():
    return {'recovered': [], 'canonicalized': [], 'existing_preserved': [],
            'recovered_count': 0, 'canonicalized_count': 0,
            'existing_preserved_count': 0, 'ambiguous_count': 0,
            'schema_version': 1, 'source_cells': 0, 'scope_aliases': 0}


def block(tmp_path, predecessor=None):
    _, pairs = evidence()
    return overlay.prepare_closed_class_elisions(dumps(tmp_path),
        {'Ταῦτ’': 'Ταύς'}, {}, predecessor or {}, pairs, {'οὗτος'})


def test_both_exact_dumps_and_actual_imported_classifier_are_pinned(tmp_path):
    audit, paths = block(tmp_path)
    assert audit['planned_count'] == 1 and audit['fresh_entry_count'] == 2
    assert set(audit['external_input_sha256']) == set(overlay.CLOSED_CLASS_DUMPS)
    for name, expected in audit['source_sha256'].items():
        assert expected == publication.pin(overlay.ROOT / name)
    overlay.verify_closed_class_inputs(paths, audit['external_input_sha256'])


def test_missing_second_dump_fails_before_planning(tmp_path):
    dumps(tmp_path)
    (tmp_path / overlay.CLOSED_CLASS_DUMPS['wiktionary_el_grc'][1]).unlink()
    with pytest.raises(SystemExit, match='fresh closed-class source missing'):
        overlay.load_closed_class_entries(tmp_path)


def test_invalid_fresh_json_fails_closed(tmp_path):
    dumps(tmp_path)
    (tmp_path / overlay.CLOSED_CLASS_DUMPS['wiktionary_el_grc'][1]).write_text('{')
    with pytest.raises(json.JSONDecodeError):
        overlay.load_closed_class_entries(tmp_path)


def test_changed_fresh_dump_cannot_be_published(tmp_path):
    audit, paths = block(tmp_path)
    next(iter(paths.values())).write_text('{}\n')
    with pytest.raises(SystemExit, match='fresh closed-class source changed'):
        overlay.verify_closed_class_inputs(paths, audit['external_input_sha256'])


def test_disabled_route_has_no_implicit_definitions():
    assert overlay.prepare_closed_class_elisions(None, {}, {}, {}, [], set()) == (
        {'schema_version': 1, 'enabled': False}, {})


def test_imported_classifier_must_belong_to_generator_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(overlay, 'ROOT', tmp_path)
    with pytest.raises(SystemExit, match='classifier imported outside source tree'):
        overlay.prepare_closed_class_elisions(tmp_path, {}, {}, {}, [], set())


def test_historical_counts_stay_separate_and_reversal_uses_same_order(tmp_path):
    audit, _ = block(tmp_path)
    reference = {'before': 'before', 'Ταῦτ’': 'Ταύς', 'after': 'after'}
    target = {'before': 'before'}
    stats = overlay.overlay_expansion(reference, {}, target)
    historical = copy.deepcopy(stats)
    stats_alias = empty_alias_stats()
    overlay.merge_closed_class_elisions(target, stats_alias, audit)
    assert stats == historical and stats['added'] == 1
    assert target == {'before': 'before', 'Ταῦτ’': 'οὗτος'}
    row = stats_alias['canonicalized'][0]
    assert (row['form'], row['current_owner'], row['owner']) == ('Ταῦτ’', 'Ταύς', 'οὗτος')
    assert stats_alias['canonicalized_count'] == 1
    target[row['form']] = row['current_owner']
    assert list(target) == ['before', 'Ταῦτ’']
    assert target == {'before': 'before', 'Ταῦτ’': 'Ταύς'}


def test_existing_predecessor_blocks_proposal(tmp_path):
    audit, _ = block(tmp_path, {'Ταῦτ᾽': 'ἄλλος'})
    assert audit['decisions'] == []


def test_intervening_owner_is_preserved_with_complete_decision(tmp_path):
    audit, _ = block(tmp_path)
    target = {'Ταῦτ’': 'ἄλλος'}
    stats = empty_alias_stats()
    overlay.merge_closed_class_elisions(target, stats, audit)
    assert target['Ταῦτ’'] == 'ἄλλος' and not stats['canonicalized']
    assert stats['existing_preserved_count'] == 1
    assert stats['closed_class_elisions']['decisions'][0]['accepted']


def test_duplicate_reversal_row_fails_closed(tmp_path):
    audit, _ = block(tmp_path)
    stats = empty_alias_stats()
    stats['canonicalized'] = [{'form': 'Ταῦτ’', 'owner': 'Ταύς', 'current_owner': 'other'}]
    with pytest.raises(SystemExit, match='duplicates another alias change'):
        overlay.merge_closed_class_elisions({'Ταῦτ’': 'Ταύς'}, stats, audit)


def publication_fixture(tmp_path):
    audit, _ = block(tmp_path / 'kaikki')
    stats = empty_alias_stats()
    overlay.merge_closed_class_elisions({'Ταῦτ’': 'Ταύς'}, stats, audit)
    sources = {name: publication.pin(overlay.ROOT / name)
               for name in publication.CLOSED_CLASS_SOURCE_FILES}
    return stats, sources


def test_portable_validator_accepts_exact_external_pins(tmp_path):
    stats, sources = publication_fixture(tmp_path)
    external, block = publication.validate_closed_class_evidence(
        stats, overlay.ROOT, tmp_path / 'kaikki', sources)
    publication.require_portable(block)
    assert len(external) == 2


@pytest.mark.parametrize('change', ['missing_kaikki', 'changed_dump', 'relative_path',
                                   'missing_dump_pin', 'wrong_helper', 'wrong_classifier',
                                   'missing_change', 'wrong_count', 'lost_decision',
                                   'wrong_former_owner', 'disabled_hidden_evidence'])
def test_portable_validator_rejects_incomplete_or_stale_proof(tmp_path, change):
    stats, sources = publication_fixture(tmp_path)
    kaikki = tmp_path / 'kaikki'
    b = stats['closed_class_elisions']
    if change == 'missing_kaikki':
        kaikki = None
    elif change == 'changed_dump':
        (kaikki / publication.CLOSED_CLASS_DUMPS['wiktionary_el_grc']).write_text('{}\n')
    elif change == 'relative_path':
        b['external_input_sha256']['wiktionary_el_grc']['path'] = '/private/source'
    elif change == 'missing_dump_pin':
        b['external_input_sha256'].pop('wiktionary_el_grc')
    elif change == 'wrong_helper':
        sources['build/closed_class_elisions.py'] = {'sha256': 'wrong', 'size': 1}
    elif change == 'wrong_classifier':
        sources['build_data.py'] = {'sha256': 'wrong', 'size': 1}
    elif change == 'missing_change':
        stats['canonicalized'] = []
    elif change == 'wrong_count':
        b['planned_count'] += 1
    elif change == 'lost_decision':
        b['decisions'] = []
    elif change == 'wrong_former_owner':
        b['canonicalized'][0]['current_owner'] = 'wrong'
    else:
        b['enabled'] = False
    with pytest.raises(AssertionError):
        publication.validate_closed_class_evidence(stats, overlay.ROOT, kaikki, sources)


def test_legacy_audit_and_explicit_disabled_audit_remain_valid():
    audit = empty_alias_stats()
    assert publication.validate_closed_class_evidence(audit, Path('.'), None, {}) == ({}, None)
    audit['closed_class_elisions'] = {'schema_version': 1, 'enabled': False}
    assert publication.validate_closed_class_evidence(audit, Path('.'), None, {})[0] == {}


def test_tagged_change_cannot_hide_by_removing_evidence_block(tmp_path):
    audit, sources = publication_fixture(tmp_path)
    audit.pop('closed_class_elisions')
    with pytest.raises(AssertionError, match='lack their evidence block'):
        publication.validate_closed_class_evidence(audit, overlay.ROOT, None, sources)


def test_complete_manifest_preserves_strict_byte_reversal(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    stats, _ = publication_fixture(tmp_path)
    for name in publication.SOURCE_FILES + publication.CLOSED_CLASS_SOURCE_FILES:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('source\n')
    base = repo / 'data/base.json'
    base.write_text('{}\n')
    exclusions = repo / 'data/exclusions.json'
    exclusions.write_text('{}\n')
    reference = {'reference': {'sha256': 'reference'},
                 'reference_base': {'path': 'data/base.json', **publication.pin(base)},
                 'exclusions': {'path': 'data/exclusions.json', **publication.pin(exclusions)}}
    (repo / 'data/ag_expansion_reference.json').write_text(json.dumps(reference))
    for name in ('ag_noun_generation_audit.json', 'ag_sophocles_noun_generation_audit.json'):
        (repo / 'data' / name).write_text('{}\n')
    stats.update(input_sha256={}, generator_sha256=publication.pin(repo / 'overlay_lsj.py')['sha256'],
                 cell_pattern_source_sha256=publication.pin(repo / 'build/expand_lsj.py')['sha256'],
                 reference_sha256='reference', reference_base_sha256=publication.pin(base)['sha256'])
    stats['closed_class_elisions']['source_sha256'] = {
        name: publication.pin(repo / name) for name in publication.CLOSED_CLASS_SOURCE_FILES}
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.com',
                    'commit', '-qm', 'Fixture'], check=True)
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps(stats, ensure_ascii=False))
    before = tmp_path / 'before.json'
    before.write_text('{"first":"first","Ταῦτ’":"Ταύς"}\n')
    final = tmp_path / 'lookup.json'
    final.write_text('{"first":"first","Ταῦτ’":"οὗτος"}\n')
    result = tmp_path / 'manifest.json'
    command = [sys.executable, publication.__file__, '--repo', str(repo), '--audit', str(audit),
               '--lookup', str(final), '--output', str(result), '--pre-alias-lookup', str(before),
               '--source-revision', 'HEAD', '--kaikki', str(tmp_path / 'kaikki')]
    subprocess.run(command, check=True, capture_output=True)
    manifest = json.loads(result.read_text())
    assert manifest['reversal']['result'] == 'byte_equal'
    assert manifest['closed_class_elisions']['owner_changes'] == 1
    assert len(manifest['external_input_sha256']) == 2
    # Equal mappings serialized differently cannot pass the unchanged assertion.
    before.write_text('{"first": "first", "Ταῦτ’": "Ταύς"}\n')
    failure = subprocess.run(command, capture_output=True)
    assert failure.returncode != 0 and b'AssertionError' in failure.stderr


def test_cli_plans_before_overlay_and_audits_after_historical_counts(tmp_path, monkeypatch):
    entry, pairs = evidence()
    reference = tmp_path / 'reference.json'
    reference.write_text(json.dumps({'Ταῦτ’': 'Ταύς'}, ensure_ascii=False))
    base = tmp_path / 'base.json'
    base.write_text('{}')
    exclusions = tmp_path / 'exclusions.json'
    exclusions.write_text('{"baseline_conflicts":[],"nonlexical":[]}')
    target = tmp_path / 'target.json'
    target.write_text('{}\n')
    heads = tmp_path / 'grc_lexical_heads.json'
    heads.write_text('{}')
    (tmp_path / 'hunspell_grc_spelling_review.json').write_text('{}')
    manifest = {'reference': publication.pin(reference),
                'reference_base': {'path': str(base), **publication.pin(base)},
                'exclusions': {'path': str(exclusions), **publication.pin(exclusions)},
                'expected_counts': {'reference_entries': 1, 'reference_base_entries': 0,
                                    'expansion_entries': 1, 'unmarked_entries': 0},
                'sentinels': {}}
    monkeypatch.setattr(overlay, 'load_noun_generation_audit', lambda *args: frozenset())
    monkeypatch.setattr(overlay, 'load_manifest', lambda path: manifest)
    monkeypatch.setattr(overlay, 'resolve_reference', lambda *args: reference)
    monkeypatch.setattr(overlay, 'load_alias_source_pairs', lambda path: (pairs, {}))
    monkeypatch.setattr(overlay, 'load_alias_lexical_heads', lambda path: {'οὗτος'})
    monkeypatch.setattr(overlay, 'alias_spelling_exclusions', lambda *args: frozenset())
    original_hash = overlay.file_sha256
    monkeypatch.setattr(overlay, 'file_sha256', lambda path: 'pattern-source'
                        if path == overlay.ROOT / 'build/expand_lsj.py' else original_hash(path))
    audit = tmp_path / 'audit.json'
    monkeypatch.setattr(sys, 'argv', ['overlay_lsj.py', '--target', str(target),
        '--reference', str(reference), '--source-pairs', str(tmp_path),
        '--lexical-heads', str(heads), '--kaikki', str(dumps(tmp_path / 'kaikki')),
        '--source-alias-audit', str(audit)])
    overlay.main()
    result = json.loads(audit.read_text())
    assert result['historical_overlay_counts']['added'] == 1
    assert result['historical_overlay_counts']['already_present'] == 0
    assert result['closed_class_elisions']['planned_count'] == 1
    assert result['canonicalized_count'] == 1
    assert result['canonicalized'][0]['current_owner'] == 'Ταύς'
    assert json.loads(target.read_text()) == {'Ταῦτ’': 'οὗτος'}
