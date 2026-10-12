"""Source-proved repairs of historical-only closed-class elisions.

No lookup alias or head is inferred. A current explicit PRON/DET reading must
have a unique positive same-POS definition and a compatible literal complete
cell in that same fresh independently defined entry.
"""
from collections import defaultdict
import unicodedata

from build_data import is_independent_lexical_entry
from dilemma.form_sanitize import (FINAL_ELISION_APOSTROPHES,
                                  canonicalize_final_elision, sanitize_form)

CLOSED_POS = frozenset({'pron', 'det'})
CASES = frozenset({'nominative', 'accusative', 'genitive', 'dative', 'vocative'})
NUMBERS = frozenset({'singular', 'plural', 'dual'})
GENDERS = frozenset({'masculine', 'feminine', 'neuter'})
VOWELS = frozenset('αεηιουωάέήίόύώ')
MARKS = FINAL_ELISION_APOSTROPHES | frozenset({'\u0313', '\u0314'})


def surface(value):
    return canonicalize_final_elision(sanitize_form(value))


def citation(value):
    return unicodedata.normalize('NFC', ''.join(
        char for char in unicodedata.normalize('NFD', sanitize_form(value))
        if char not in '\u0304\u0306-'))


def full_cell(value):
    return surface(unicodedata.normalize('NFC', ''.join(
        char for char in unicodedata.normalize('NFD', value)
        if char not in '\u0304\u0306')))


def initial_lower(value):
    """A sentence-initial capital does not change a closed-class cell."""
    return value[:1].lower() + value[1:]


def nominal_cell(tags):
    tags = frozenset(tags or ())
    return bool(tags & CASES) and bool(tags & NUMBERS)


def compatible_cell(reading, cell):
    """Every positively identified source axis must fit the explicit cell."""
    left, right = frozenset(reading or ()), frozenset(cell or ())
    if not nominal_cell(left) or not nominal_cell(right):
        return False
    return all(not (left & axis) or not (right & axis)
               or (left & axis) <= (right & axis)
               for axis in (CASES, NUMBERS, GENDERS))


def positive_cells(entries):
    """Keep own nominal cells only from independent same-POS source entries."""
    result = defaultdict(list)
    definitions = defaultdict(list)
    for record in entries:
        entry, source = record['entry'], record['source']
        pos = entry.get('pos')
        if (entry.get('lang_code') != 'grc' or pos not in CLOSED_POS
                or not is_independent_lexical_entry(entry, source)):
            continue
        owner = citation(entry['word'])
        witness = {'source': source, 'line': record.get('line'),
                   'word': entry['word'], 'pos': pos, 'senses': entry.get('senses', [])}
        definitions[(owner, pos)].append(witness)
        for row in entry.get('forms') or ():
            form = full_cell(row.get('form', ''))
            if form and not form.endswith('᾽') and nominal_cell(row.get('tags')):
                result[(owner, pos)].append({'form': form, 'tags': row['tags'],
                                             'source': source, 'line': record.get('line'),
                                             'original': row})
    return result, definitions


def plan_repairs(reference, reference_base, predecessor, source_pairs,
                 lexical_heads, fresh_entries):
    """Return eligible historical plans and all competing source evidence."""
    cells, definitions = positive_cells(fresh_entries)
    historical_positive = defaultdict(set)
    for historical, owner in reference.items():
        if historical and historical[-1] in MARKS and citation(owner) in lexical_heads:
            historical_positive[surface(historical)].add(citation(owner))
    source_rows = defaultdict(list)
    for source, pairs in source_pairs:
        for pair in pairs:
            raw = pair.get('form', '')
            if not raw or raw[-1] not in MARKS:
                continue
            form = surface(raw)
            if form.endswith('᾽'):
                source_rows[form].append({'source': source, 'pair': pair,
                                          'owner': citation(pair['lemma']),
                                          'positive': citation(pair['lemma']) in lexical_heads})
    decisions = []
    for historical, old_owner in reference.items():
        if not historical or historical[-1] not in MARKS:
            continue
        form = surface(historical)
        source_spellings = {form[:-1] + mark for mark in MARKS} | {historical, form}
        if (not form.endswith('᾽') or any(key in reference_base for key in source_spellings)
                or any(key in predecessor for key in source_spellings)
                or citation(old_owner) in lexical_heads):
            continue
        readings = source_rows.get(form, [])
        positive = [row for row in readings if row['positive']]
        claims = {(row['owner'], row['pair'].get('pos')) for row in positive}
        detail = {'form': historical, 'canonical_source_form': form,
                  'historical_owner': old_owner, 'source_readings': readings,
                  'positive_claims': [list(claim) for claim in sorted(
                      claims, key=lambda claim: (claim[0], claim[1] or ''))],
                  'accepted': False, 'reason': 'ambiguous_or_non_closed_positive_reading'}
        if len(claims) == 1:
            owner, pos = next(iter(claims))
            if pos in CLOSED_POS:
                proof = definitions.get((owner, pos), [])
                competing_history = historical_positive.get(form, set()) - {owner}
                if competing_history:
                    detail['reason'] = 'positive_historical_marker_conflict'
                    detail['competing_historical_owners'] = sorted(competing_history)
                elif not proof:
                    detail['reason'] = 'missing_positive_same_pos_fresh_definition'
                else:
                    matches = []
                    unmatched = []
                    complete_stem_cells = []
                    stem = initial_lower(form[:-1])
                    for row in positive:
                        row_matches = []
                        for cell in cells.get((owner, pos), []):
                            complete = initial_lower(cell['form'])
                            if (complete == stem
                                    and compatible_cell(row['pair'].get('tags'), cell['tags'])):
                                complete_stem_cells.append({'current_reading': row,
                                                            'complete_cell': cell})
                            if (len(complete) == len(stem) + 1
                                    and complete[:-1] == stem
                                    and complete[-1] in VOWELS
                                    and compatible_cell(row['pair'].get('tags'), cell['tags'])):
                                row_matches.append({'current_reading': row,
                                                    'complete_cell': cell})
                        matches.extend(row_matches)
                        if not row_matches:
                            unmatched.append(row)
                    detail.update(owner=owner, pos=pos, positive_definitions=proof,
                                  matching_cells=matches, unmatched_positive_readings=unmatched,
                                  ambiguous_unelided_cells=complete_stem_cells)
                    detail['accepted'] = bool(matches) and not unmatched and not complete_stem_cells
                    detail['reason'] = ('already_complete_nominal_cell_marker_ambiguous'
                                        if complete_stem_cells else
                                        'positive_same_pos_literal_cell' if detail['accepted']
                                        else 'missing_literal_compatible_nominal_cell')
        decisions.append(detail)
    return sorted(decisions, key=lambda decision: decision['form'])


def apply_repairs(target, plans):
    """Canonicalize only the explicitly planned unsupported historical owner.

    The existing audit reversal schema can restore every former owner. Positive
    predecessor entries and any intervening changes are preserved untouched.
    """
    canonicalized, preserved = [], []
    for plan in plans:
        if not plan['accepted']:
            continue
        form = plan['form']
        current = target.get(form)
        detail = {**plan, 'current_owner': current}
        if current == plan['historical_owner']:
            target[form] = plan['owner']
            canonicalized.append(detail)
        else:
            detail['reason'] = 'intervening_or_absent_owner_preserved'
            preserved.append(detail)
    return {'canonicalized': canonicalized, 'existing_preserved': preserved,
            'decisions': plans}
