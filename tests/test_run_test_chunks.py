"""The chunked suite runner covers every test file exactly once."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "run_test_chunks", ROOT / "scripts" / "run_test_chunks.py")
rtc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rtc)


def test_every_file_runs_once():
    files = rtc.test_files()
    groups = rtc.chunk(files, 4)
    flat = [f for g in groups for f in g]
    assert sorted(flat) == files
    assert len(flat) == len(set(flat))
    assert "tests/tagger/test_tagger.py" in files
    assert "tests/test_run_test_chunks.py" in files


def test_heavy_files_run_alone():
    files = ["tests/a.py", "tests/test_new_features.py", "tests/b.py", "tests/c.py"]
    groups = rtc.chunk(files, 2)
    assert groups == [["tests/a.py", "tests/b.py"], ["tests/c.py"],
                      ["tests/test_new_features.py"]]


def test_more_chunks_than_files():
    assert rtc.chunk(["tests/a.py"], 4) == [["tests/a.py"]]
    assert rtc.chunk([], 3) == []
