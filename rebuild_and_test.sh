#!/bin/bash
# Full rebuild + LSJ overlay + benchmark
# Usage: ./rebuild_and_test.sh
set -e
cd "$(dirname "$0")"

echo "=== Step 1: build_data.py ==="
python build_data.py --kaikki kaikki

echo ""
echo "=== Step 2: LSJ overlay (into JSON) ==="
python overlay_lsj.py /tmp/ag_lookup_reference.json data/ag_lookup.json

echo ""
echo "=== Step 3: Delete raw_lookups.db so build_lookup_db.py reads from JSON ==="
rm -f data/raw_lookups.db

echo ""
echo "=== Step 4: build_lookup_db.py ==="
python build_lookup_db.py

echo ""
echo "=== Step 5: Fast benchmark ==="
python bench_fast.py
