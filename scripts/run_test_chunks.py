#!/usr/bin/env python3
"""Run the test suite as several pytest processes instead of one.

One process that runs every test file keeps each file's Dilemma instances and
data tables alive, so its memory grows across the run: about 17.5 GB at the
end of the 1.3.25 suite, more than a 16 GB CI runner has. Each chunk here is
a fresh process, so the peak is that of the largest chunk (about 14 GB, for
``tests/test_new_features.py`` alone).

    python scripts/run_test_chunks.py                 # 4 chunks, -x -v
    python scripts/run_test_chunks.py --chunks 6 -- -q --run-slow

Arguments after ``--`` go to every pytest call (default ``-x -v``). The run
stops at the first chunk that fails and exits with its status.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

# Files that run alone because each needs most of a runner's memory.
ALONE = ("tests/test_new_features.py",)


def test_files(tests: Path = TESTS) -> list[str]:
    """Every test file under ``tests/``, as repo-relative paths, sorted."""
    return sorted(p.relative_to(tests.parent).as_posix()
                  for p in tests.rglob("test_*.py")
                  if "__pycache__" not in p.parts)


def chunk(files: list[str], chunks: int, alone=ALONE) -> list[list[str]]:
    """Split ``files`` into ``chunks`` contiguous groups plus one per ALONE file."""
    if chunks < 1:
        raise ValueError("chunks must be at least 1")
    solo = [f for f in files if f in alone]
    rest = [f for f in files if f not in alone]
    size = -(-len(rest) // chunks) if rest else 0
    groups = [rest[i:i + size] for i in range(0, len(rest), size)] if size else []
    return groups + [[f] for f in solo]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("pytest_args", nargs="*")
    args = parser.parse_args(argv)
    extra = args.pytest_args or ["-x", "-v"]
    groups = chunk(test_files(), args.chunks)
    for i, group in enumerate(groups, 1):
        print(f"=== chunk {i}/{len(groups)}: {len(group)} files "
              f"({group[0]} .. {group[-1]})", flush=True)
        status = subprocess.call([sys.executable, "-m", "pytest", *group, *extra],
                                 cwd=ROOT)
        if status != 0:
            print(f"=== chunk {i} failed with exit status {status}", flush=True)
            return status
    return 0


if __name__ == "__main__":
    sys.exit(main())
