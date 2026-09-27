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


def test_each_pair_carries_its_token_counts(b, tmp_path, monkeypatch):
    # φέρε is tagged an imperative twice and an imperfect once; the pair
    # keeps its first token's analysis and gains the counts.
    monkeypatch.syspath_prepend(str(REPO_ROOT / "build"))  # nc_filter
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    words = "".join(
        f'<word id="{i}" form="φέρε" lemma="φέρω" postag="{tag}"/>'
        for i, tag in enumerate(["v3siia---", "v2spma---", "v2spma---"], 1)
    )
    (xml_dir / "0012-001.xml").write_text(
        f'<treebank><sentence id="1">{words}'
        '<word id="4" form="λόγος" lemma="λόγος" postag="n-s---mn-"/>'
        "</sentence></treebank>", encoding="utf-8")
    meta = tmp_path / "metadata.txt"
    meta.write_text(METADATA, encoding="utf-8")
    pairs = b.extract_glaux(xml_dir, metadata_path=meta)
    by_form = {p["form"]: p for p in pairs}
    fere = by_form["φέρε"]
    assert fere["count"] == 3
    assert "imperfect" in fere["tags"]
    assert fere["analyses"] == [
        ["verb", ["second-person", "singular", "present", "imperative", "active"], 2],
        ["verb", ["third-person", "singular", "imperfect", "indicative", "active"], 1],
    ]
    # One analysis, no list.
    assert by_form["λόγος"]["count"] == 1
    assert "analyses" not in by_form["λόγος"]
