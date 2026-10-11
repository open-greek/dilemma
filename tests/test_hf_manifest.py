"""Guards for data/hf_manifest.json and the CI steps that consume it.

The lookup artifacts ship on HuggingFace rather than in git, so the manifest is
what ties a git commit to the exact bytes CI must test against. These tests
keep the three pieces (the tracked list in scripts/hf_data.py, the manifest,
and the workflow's download step) from drifting apart; the byte-level check of
the real artifacts is `hf_data.py verify`, which runs in CI right after the
download.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data" / "hf_manifest.json"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "test.yml"

# scripts/ is not a package, so load the module by path.
_spec = importlib.util.spec_from_file_location(
    "hf_data", ROOT / "scripts" / "hf_data.py")
hf_data = importlib.util.module_from_spec(_spec)
sys.modules["hf_data"] = hf_data
_spec.loader.exec_module(hf_data)


@pytest.fixture(scope="module")
def manifest():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_manifest_covers_exactly_the_tracked_artifacts(manifest):
    assert manifest["repo"] == hf_data.REPO
    assert list(manifest["files"]) == hf_data.TRACKED


def test_manifest_entries_are_well_formed(manifest):
    for path, entry in manifest["files"].items():
        assert path.startswith("data/"), path
        assert entry["size"] > 0
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), path
        assert re.fullmatch(r"[0-9a-f]{40}", entry["git_sha1"]), path


def test_ci_fetches_modern_training_evidence_inputs():
    """A clean CI checkout must run the real-data invariant, not skip it."""
    assert {
        "data/mg_pairs.json", "data/mg_lookup.json", "data/el_participles.json",
        "data/el_noun_heads.json", "data/el_verb_heads.json",
    } <= set(hf_data.TRACKED)


def test_workflow_downloads_the_tracked_list_and_checks_it(manifest):
    """The workflow must take its file list from `hf_data.py files` and run the
    wait/verify steps - a hardcoded list would silently drift from the manifest,
    and skipping the wait brings back the upload/push race."""
    wf = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "scripts/hf_data.py files" in wf
    assert "scripts/hf_data.py wait" in wf
    assert "scripts/hf_data.py verify" in wf
    for path in manifest["files"]:
        assert f"hf download ciscoriordan/dilemma {path}" not in wf


def test_hashes_match_the_hub_encoding(tmp_path):
    """sha256 is the plain file digest and git_sha1 is git's blob id - the two
    forms the Hub reports for LFS/Xet and for small plain-git files."""
    import hashlib
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"dilemma")
    got = hf_data.file_hashes(sample)
    assert got["size"] == 7
    assert got["sha256"] == hashlib.sha256(b"dilemma").hexdigest()
    assert got["git_sha1"] == hashlib.sha1(b"blob 7\0dilemma").hexdigest()
