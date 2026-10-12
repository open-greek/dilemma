#!/usr/bin/env python3
"""Build portable publication metadata for a source-alias audit and lookup."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PureWindowsPath

SOURCE_FILES = (
    'overlay_lsj.py', 'build/expand_lsj.py', 'build/lsj_principal_parts.py',
    'build/expand_sophocles.py', 'build/build_diorisis_pairs.py', 'build/diorisis_source.py',
    'build/corpus_freq_key.py', 'dilemma/form_sanitize.py', 'export_hunspell.py',
    'data/hunspell_grc_postclassical_review.json', 'data/ag_expansion_reference.json',
)

CLOSED_CLASS_SOURCE_FILES = ('build/closed_class_elisions.py', 'build_data.py')
CLOSED_CLASS_DUMPS = {
    'wiktionary_en_grc': 'en-el/kaikki.org-dictionary-AncientGreek.jsonl',
    'wiktionary_el_grc': 'el/kaikki.org-dictionary-AncientGreek.jsonl',
}


def pin(path: Path) -> dict:
    with path.open('rb') as handle:
        hasher = hashlib.sha256()
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
        digest = hasher.hexdigest()
    return {'sha256': digest, 'size': path.stat().st_size}


class DigestWriter:
    def __init__(self):
        self.digest = hashlib.sha256()
        self.size = 0
    def write(self, text: str) -> int:
        data = text.encode('utf-8')
        self.digest.update(data)
        self.size += len(data)
        return len(text)


def require_portable(value) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            require_portable(key)
            require_portable(item)
    elif isinstance(value, list):
        for item in value:
            require_portable(item)
    elif isinstance(value, str):
        assert not Path(value).is_absolute() and not PureWindowsPath(value).is_absolute()


def validate_closed_class_evidence(audit, repo, kaikki, sources):
    """Validate explicit external sources without treating them as data files."""
    block = audit.get('closed_class_elisions')
    tagged = [row for row in audit['canonicalized']
              if row.get('repair_kind') == 'historical_closed_class_elision']
    if block is None:
        assert not tagged, 'closed-class changes lack their evidence block'
        return {}, None
    assert block['schema_version'] == 1
    if block.get('enabled') is False:
        assert block == {'schema_version': 1, 'enabled': False}
        assert not tagged
        return {}, block
    assert block.get('enabled') is True
    assert kaikki is not None, '--kaikki is required to verify fresh closed-class sources'
    assert set(block['external_input_sha256']) == set(CLOSED_CLASS_DUMPS)
    external = {}
    for label, relative in CLOSED_CLASS_DUMPS.items():
        expected = block['external_input_sha256'][label]
        assert expected == {'path': relative, **pin(kaikki / relative)}, label
        external[label] = expected
    assert set(block['source_sha256']) == set(CLOSED_CLASS_SOURCE_FILES)
    for name in CLOSED_CLASS_SOURCE_FILES:
        assert block['source_sha256'][name] == sources[name], name
    decisions = block['decisions']
    assert len({row['form'] for row in decisions}) == len(decisions)
    assert block['planned_count'] == sum(row['accepted'] for row in decisions)
    assert block['canonicalized_count'] == len(block['canonicalized'])
    assert block['existing_preserved_count'] == len(block['existing_preserved'])
    assert tagged == block['canonicalized']
    preserved = [row for row in audit['existing_preserved']
                 if row.get('repair_kind') == 'historical_closed_class_elision']
    assert preserved == block['existing_preserved']
    accepted = {row['form']: row for row in decisions if row['accepted']}
    assert len(block['canonicalized']) + len(preserved) == len(accepted)
    for row in block['canonicalized'] + preserved:
        plan = accepted.pop(row['form'])
        assert all(row[key] == value for key, value in plan.items() if key != 'reason')
        if row in block['canonicalized']:
            assert row['reason'] == plan['reason']
            assert row['current_owner'] == row['historical_owner']
        else:
            assert row['reason'] == 'intervening_or_absent_owner_preserved'
            assert row['current_owner'] != row['historical_owner']
    assert not accepted
    return external, block


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'audit', 'lookup', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    predecessor = parser.add_mutually_exclusive_group(required=True)
    predecessor.add_argument('--pre-alias-lookup', type=Path)
    predecessor.add_argument('--pre-alias-manifest', type=Path,
                             help='Use the predecessor hash from an existing portable manifest')
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--kaikki', type=Path,
                        help='Fresh dump directory required when the audit repairs '
                             'historical closed-class elisions')
    args = parser.parse_args()
    resolved_revision = subprocess.check_output(
        ['git', 'rev-parse', args.source_revision + '^{commit}'],
        cwd=args.repo, text=True).strip()
    artifact_paths = (args.audit, args.lookup) + ((args.pre_alias_lookup,) if args.pre_alias_lookup else ())
    before = {path: pin(path) for path in artifact_paths}
    if args.pre_alias_lookup:
        predecessor_pin = before[args.pre_alias_lookup]
    else:
        expected = json.loads(args.pre_alias_manifest.read_text())['artifacts']['pre_alias_target']
        predecessor_pin = {key: expected[key] for key in ('sha256', 'size')}
    audit = json.loads(args.audit.read_text())
    assert audit['schema_version'] == 1
    sources = {}
    closed_class = audit.get('closed_class_elisions', {})
    source_files = SOURCE_FILES + (CLOSED_CLASS_SOURCE_FILES
                                  if closed_class.get('enabled') is True else ())
    for name in source_files:
        sources[name] = pin(args.repo / name)
        recorded = subprocess.check_output(
            ['git', 'show', resolved_revision + ':' + name], cwd=args.repo)
        assert hashlib.sha256(recorded).hexdigest() == sources[name]['sha256'], name
    assert sources['overlay_lsj.py']['sha256'] == audit['generator_sha256']
    assert sources['build/expand_lsj.py']['sha256'] == audit['cell_pattern_source_sha256']
    external_inputs, closed_class = validate_closed_class_evidence(
        audit, args.repo, args.kaikki, sources)
    external_before = {label: pin(args.kaikki / row['path'])
                       for label, row in external_inputs.items()}
    inputs = {}
    for name, expected in audit['input_sha256'].items():
        assert Path(name).name == name
        actual = pin(args.repo / 'data' / name)
        assert actual == expected, name
        inputs['data/' + name] = actual
    reference_manifest = json.loads((args.repo / 'data/ag_expansion_reference.json').read_text())
    assert reference_manifest['reference']['sha256'] == audit['reference_sha256']
    assert reference_manifest['reference_base']['sha256'] == audit['reference_base_sha256']
    for label in ('reference_base', 'exclusions'):
        expected = reference_manifest[label]
        assert pin(args.repo / expected['path']) == {key: expected[key] for key in ('sha256', 'size')}
    noun_audits = {name: pin(args.repo / name) for name in (
        'data/ag_noun_generation_audit.json', 'data/ag_sophocles_noun_generation_audit.json')}
    lookup = json.loads(args.lookup.read_text())
    final_count = len(lookup)
    assert len(audit['recovered']) == audit['recovered_count']
    assert len(audit['canonicalized']) == audit['canonicalized_count']
    recovered_forms = {row['form'] for row in audit['recovered']}
    rewritten_forms = {row['form'] for row in audit['canonicalized']}
    assert len(recovered_forms) == len(audit['recovered'])
    assert len(rewritten_forms) == len(audit['canonicalized'])
    assert recovered_forms.isdisjoint(rewritten_forms)
    for row in audit['recovered'] + audit['canonicalized']:
        assert lookup[row['form']] == row['owner']
    for row in audit['recovered']:
        del lookup[row['form']]
    for row in audit['canonicalized']:
        lookup[row['form']] = row['current_owner']
    writer = DigestWriter()
    json.dump(lookup, writer, ensure_ascii=False, separators=(',', ':'))
    writer.write('\n')
    restored = {'sha256': writer.digest.hexdigest(), 'size': writer.size}
    assert restored == predecessor_pin
    assert before == {path: pin(path) for path in artifact_paths}
    assert external_before == {label: pin(args.kaikki / row['path'])
                               for label, row in external_inputs.items()}
    assert sources == {name: pin(args.repo / name) for name in source_files}
    assert inputs == {name: pin(args.repo / name) for name in inputs}
    manifest = {
        'schema_version': 1,
        'audit_schema_version': audit['schema_version'],
        'source_revision': resolved_revision,
        'artifacts': {
            'lookup': {'path': 'data/ag_lookup.json', **before[args.lookup], 'entries': final_count},
            'source_alias_audit': {'path': 'data/ag_source_alias_audit.json', **before[args.audit]},
            'pre_alias_target': {**predecessor_pin, 'entries': len(lookup)},
        },
        'source_sha256': sources,
        'input_sha256': inputs,
        'external_input_sha256': external_inputs,
        'closed_class_elisions': ({
            'enabled': True,
            'planned': closed_class['planned_count'],
            'owner_changes': closed_class['canonicalized_count'],
            'preserved_intervening_readings': closed_class['existing_preserved_count'],
            'decision_count': len(closed_class['decisions']),
        } if external_inputs else {'enabled': False}),
        'historical_reference': reference_manifest['reference'],
        'historical_reference_base': reference_manifest['reference_base'],
        'historical_exclusions': reference_manifest['exclusions'],
        'noun_generation_audits': noun_audits,
        'scope': {
            'description': 'Recover exact source cells and normalized lookup aliases in historical withheld irregular-compound families; recover independently re-owned cells only within the documented historical-key shape scope. Historical-only marked PRON/DET owners may be canonicalized only from unique current source readings and explicit literal-completion cells in fresh same-POS definitions. No approximate paradigm generation.',
            'source_cells': audit['source_cells'],
            'scope_aliases': audit['scope_aliases'],
            'added_cells_and_aliases': audit['recovered_count'],
            'owner_changes': audit['canonicalized_count'],
            'preserved_existing_readings': audit['existing_preserved_count'],
            'withheld_ambiguous_aliases': audit['ambiguous_count'],
            'reading_fields': ['original_form', 'form', 'lemma', 'source', 'owner'],
            'direct_exception_evidence': ['evidence', 'citation', 'source_url'],
        },
        'reversal': {
            'result': 'byte_equal',
            'procedure': 'Delete recovered rows; restore current_owner for canonicalized rows; serialize unchanged dictionary order as UTF-8 JSON with compact separators and a final newline.',
            'restored_target': restored,
        },
        'purpose': 'Ancillary provenance and reversible change audit; not a runtime or CI download requirement.',
    }
    text = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    require_portable(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(json.dumps({'manifest': args.output.name, **pin(args.output),
                      'lookup_entries': final_count, 'reversal': 'byte_equal'}))


if __name__ == '__main__':
    main()
