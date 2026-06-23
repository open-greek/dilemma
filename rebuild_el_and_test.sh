#!/bin/bash
# Rebuild only EL (MG) data + rebuild lookup.db + benchmark
# Keeps AG data from committed version untouched
# Usage: ./rebuild_el_and_test.sh
set -e
cd "$(dirname "$0")"

echo "=== Step 1: Restore committed AG data ==="
git checkout HEAD -- data/ag_lookup.json data/ag_pairs.json data/ag_pos_lookup.json data/med_lookup.json data/med_pairs.json

echo ""
echo "=== Step 2: build_data.py --lang el ==="
python build_data.py --kaikki kaikki --lang el

echo ""
echo "=== Step 3: Delete raw_lookups.db, rebuild lookup.db from JSON ==="
rm -f data/raw_lookups.db
python build_lookup_db.py

echo ""
echo "=== Step 4: Fast benchmark ==="
python bench_fast.py
