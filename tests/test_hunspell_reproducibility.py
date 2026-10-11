"""Hunspell artifacts keep identical bytes across Python hash seeds."""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_exact_frequency_export_is_identical_across_hash_seeds(tmp_path):
    script = """
from pathlib import Path
import sys
from export_hunspell import write_variant

pairs = [("λόγος", "λόγος"), ("λόγου", "λόγος"), ("λόγῳ", "λόγος"),
         ("λόγον", "λόγος"), ("φίλος", "φίλος"), ("φίλου", "φίλος"),
         ("φίλῳ", "φίλος")]
counts = {"λόγος": 1000, "λόγου": 101, "λόγῳ": 3, "λόγον": 0,
          "φίλος": 1000, "φίλου": 101, "φίλῳ": 3}
write_variant("grc", pairs, counts, Path(sys.argv[1]), "sample", "grc",
              "test", "fixed")
"""
    results = []
    for seed in (1, 17, 73):
        directory = tmp_path / str(seed)
        result = subprocess.run(
            [sys.executable, "-c", script, str(directory)], cwd=ROOT,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        results.append({suffix: (directory / f"sample.{suffix}").read_bytes()
                        for suffix in ("aff", "dic", "version")})
    assert results[1:] == results[:-1]
    assert b"fr:C" in results[0]["dic"]
    assert b"fr:M" in results[0]["dic"]
    assert b"fr:R" in results[0]["dic"]
    assert b"fr:X" in results[0]["dic"]
