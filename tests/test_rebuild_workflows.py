"""Exercise rebuild wrappers without running the expensive data pipeline."""
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent.parent
ANCIENT_INPUTS = (
    "ag_lookup.json", "ag_pairs.json", "ag_pos_lookup.json",
    "med_lookup.json", "med_pairs.json",
)


def run_modern_rebuild(tmp_path, missing=None):
    script = tmp_path / "rebuild_el_and_test.sh"
    shutil.copyfile(ROOT / script.name, script)
    data = tmp_path / "data"
    data.mkdir()
    contents = {}
    for name in (*ANCIENT_INPUTS, "raw_lookups.db"):
        if name != missing:
            contents[name] = (name + ": installed input\n").encode()
            (data / name).write_bytes(contents[name])
    commands = tmp_path / "commands.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n')
    python.chmod(0o755)
    result = subprocess.run(
        ["bash", str(script)], capture_output=True, text=True,
        env={**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
             "CALL_LOG": str(commands)},
    )
    return result, contents, data, commands


def test_modern_rebuild_preserves_installed_ancient_inputs(tmp_path):
    result, contents, data, commands = run_modern_rebuild(tmp_path)
    assert result.returncode == 0, result.stderr
    assert {name: (data / name).read_bytes() for name in contents} == contents
    assert commands.read_text().splitlines() == [
        "build_data.py --kaikki kaikki --lang el",
        "build/build_el_verb_heads.py --kaikki kaikki",
        "build_lookup_db.py",
        "bench_fast.py",
    ]


def test_modern_rebuild_stops_before_mutation_when_inputs_are_missing(tmp_path):
    result, contents, data, commands = run_modern_rebuild(tmp_path, "med_lookup.json")
    assert result.returncode != 0
    assert "Missing data/med_lookup.json" in result.stderr
    assert not commands.exists()
    assert {name: (data / name).read_bytes() for name in contents} == contents
