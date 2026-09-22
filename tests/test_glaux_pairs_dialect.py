"""GLAUx work dialects carried onto the pairs the paradigm builders read."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "build" / "build_glaux_pairs.py"

METADATA = "\t".join(["GLAUX_TEXT_ID", "TLG", "DIALECT"]) + "\n" + "\n".join([
    "\t".join(["1", "0012-001", "Ionic/Epic"]),
    "\t".join(["2", "0059-001", "Attic"]),
    "\t".join(["3", "0086-001", "Attic/Koine"]),
    "\t".join(["4", "0031-001", "Koine"]),
    "\t".join(["5", "0010-001", "Doric"]),
    "\t".join(["6", "9999-001", ""]),
]) + "\n"


@pytest.fixture(scope="module")
def b():
    spec = importlib.util.spec_from_file_location(
        "build_glaux_pairs", MODULE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_only_the_marked_dialects_leave_the_attic_default(b, tmp_path):
    path = tmp_path / "metadata.txt"
    path.write_text(METADATA, encoding="utf-8")
    dialects = b.glaux_dialects(path)
    # Homer is Epic; Doric is Doric.
    assert dialects["0012-001"] == "Epic"
    assert dialects["0010-001"] == "Doric"
    # Attic, Attic/Koine and Koine share the default slice, which the
    # paradigm builders read as Attic.
    assert dialects["0059-001"] == "Attic"
    assert dialects["0086-001"] == "Attic"
    assert dialects["0031-001"] == "Attic"
    # A work with no recorded dialect votes for nothing.
    assert dialects["9999-001"] == ""


def test_missing_metadata_is_not_an_error(b, tmp_path):
    assert b.glaux_dialects(tmp_path / "absent.txt") == {}
