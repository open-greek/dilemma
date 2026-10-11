"""Fast benchmark: AG Classical + Katharevousa + Demotic only (skips DBBE).
Runs in ~1 minute vs ~20 minutes for the full benchmark."""
import sys, json, unicodedata
from pathlib import Path
from dilemma import Dilemma

ROOT = Path(__file__).resolve().parent
BENCH_DIR = ROOT / 'data/benchmarks'
EQUIV_PATH = ROOT / 'data/lemma_equivalences.json'

with open(EQUIV_PATH) as f:
    data = json.load(f)
equiv = {}
for group in data['groups']:
    group_set = set(group)
    for lemma in group:
        equiv[lemma] = equiv.get(lemma, set()) | group_set

def strip_accents(s):
    nfkd = unicodedata.normalize('NFD', s)
    return unicodedata.normalize('NFC', ''.join(c for c in nfkd if unicodedata.category(c) != 'Mn'))

def are_equivalent(pred, gold):
    if pred is None:
        return False
    pa, ga = strip_accents(pred).lower(), strip_accents(gold).lower()
    if pa == ga: return True
    for e in equiv.get(gold, set()):
        if strip_accents(e).lower() == pa: return True
    for e in equiv.get(pred, set()):
        if strip_accents(e).lower() == ga: return True
    return False

def safe_lemmatize(d, form):
    try:
        return d.lemmatize(form)
    except Exception:
        return form

def load_tsv(path):
    pairs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[0].strip():
                pairs.append((parts[0], parts[1]))
    return pairs

def main():
    datasets = {
        'AG Classical': load_tsv(BENCH_DIR / 'ag_gold.tsv'),
        'Katharevousa': load_tsv(BENCH_DIR / 'katharevousa_gold.tsv'),
        'Demotic MG': load_tsv(BENCH_DIR / 'demotic_gold.tsv'),
    }

    configs = [
        # (label, kwargs, datasets_to_run)
        ('all/wikt', dict(lang='all', resolve_articles=True), ['AG Classical', 'Katharevousa']),
        ('grc/wikt', dict(lang='grc', resolve_articles=True), ['AG Classical', 'Katharevousa']),
        ('el/triant', dict(lang='el', convention='triantafyllidis'), ['Demotic MG']),
        ('all/triant', dict(lang='all', convention='triantafyllidis'), ['Demotic MG']),
    ]

    for label, kwargs, ds_names in configs:
        d = Dilemma(**kwargs)
        d.preload()
        results = []
        for name in ds_names:
            pairs = datasets[name]
            correct = sum(1 for form, gold in pairs if are_equivalent(safe_lemmatize(d, form), gold))
            pct = 100 * correct / len(pairs)
            results.append(f'{name}: {pct:5.1f}%')
        print(f'{label:<12}  ' + '  '.join(results))


if __name__ == '__main__':
    main()
