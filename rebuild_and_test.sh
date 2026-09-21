#!/bin/bash
# Full rebuild + LSJ overlay + benchmark
# Usage: ./rebuild_and_test.sh
set -e
cd "$(dirname "$0")"

echo "=== Step 1: build_data.py ==="
python build_data.py --kaikki kaikki

echo ""
echo "=== Step 2: current LSJ/Sophocles expansion ==="
python build/expand_lsj.py --expand
python build/expand_lsj.py --expand-verbs
python build/expand_sophocles.py --expand
python build/expand_sophocles.py --expand-verbs

echo ""
echo "=== Step 3: pinned historical expansion recovery ==="
python overlay_lsj.py

echo ""
echo "=== Step 4: build_lookup_db.py (prefers the larger expanded JSON) ==="
python build_lookup_db.py

echo ""
echo "=== Step 5: Fast benchmark ==="
python bench_fast.py
