#!/bin/bash
# Rebuild Modern Greek data, the combined lookup, and its benchmark.
# Keep the installed Ancient and Medieval source artifacts unchanged.
# Usage: ./rebuild_el_and_test.sh
set -e
cd "$(dirname "$0")"

echo "=== Step 1: Check installed Ancient and Medieval data ==="
for file in data/ag_lookup.json data/ag_pairs.json data/ag_pos_lookup.json data/med_lookup.json data/med_pairs.json; do
    if [ ! -s "$file" ]; then
        echo "Missing $file; download or build the Ancient and Medieval data first." >&2
        exit 1
    fi
done

echo ""
echo "=== Step 2: build_data.py --lang el ==="
python build_data.py --kaikki kaikki --lang el
python build/build_el_verb_heads.py --kaikki kaikki

echo ""
echo "=== Step 3: Rebuild the combined lookup ==="
python build_lookup_db.py

echo ""
echo "=== Step 4: Fast benchmark ==="
python bench_fast.py
