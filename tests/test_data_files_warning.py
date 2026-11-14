"""A data directory without the lists lemmatization reads warns (issue #118)."""
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from dilemma import DilemmaDataWarning, core

ROOT = Path(__file__).resolve().parent.parent

AG_NAMES = sorted([
    "lsj_headwords.json", "lsj10_headwords.json", "cunliffe_headwords.json",
    "dge_headwords.json", "lbg_headwords.json", "vlg_headwords.json",
    "pd_headwords.json", "wip_headwords.json", "ag_headwords.json",
    "grc_lexical_heads.json", "grc_adverb_heads.json"])
MG_NAMES = ["mg_headwords.json", "triantafyllidis_headwords.json"]


def test_missing_files_follow_language_and_convention(tmp_path):
    assert core.missing_data_files("grc", None, tmp_path) == AG_NAMES
    assert core.missing_data_files("el", "triantafyllidis", tmp_path) == sorted(
        MG_NAMES + ["lemma_equivalences.json", "convention_triantafyllidis.json"])
    assert core.missing_data_files("all", "lsj", tmp_path) == sorted(
        AG_NAMES + MG_NAMES + ["lemma_equivalences.json", "convention_lsj.json"])
    (tmp_path / "lsj_headwords.json").write_text("[]", encoding="utf-8")
    assert "lsj_headwords.json" not in core.missing_data_files("grc", None, tmp_path)


@pytest.fixture
def bare_model(tmp_path, monkeypatch):
    monkeypatch.setattr(core.Dilemma, "_load_lookups", lambda *a, **k: None)
    monkeypatch.setattr(core.Dilemma, "_check_backend", lambda *a: None)
    monkeypatch.setattr(core, "DATA_DIR", tmp_path)
    monkeypatch.setattr(core, "_WARNED_MISSING_DATA", set())
    return tmp_path


def test_warns_once_naming_the_files(bare_model):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        core.Dilemma(lang="el", skip_pos=True)
        core.Dilemma(lang="el", skip_pos=True)
    found = [w for w in caught if issubclass(w.category, DilemmaDataWarning)]
    assert len(found) == 1
    message = str(found[0].message)
    for name in MG_NAMES:
        assert name in message
    assert "lsj_headwords.json" not in message
    assert "python -m dilemma download" in message
    assert found[0].filename == __file__      # points at the caller


def test_a_complete_directory_does_not_warn(bare_model):
    for name in AG_NAMES + MG_NAMES:
        (bare_model / name).write_text("{}", encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DilemmaDataWarning)
        core.Dilemma(lang="all", skip_pos=True)


def test_the_repository_data_directory_does_not_warn():
    if core.missing_data_files("all", "lsj"):
        pytest.skip("the full data download is not installed")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DilemmaDataWarning)
        core.Dilemma(lang="all", convention="lsj", skip_pos=True)


def test_dilemma_data_dir_without_a_list_warns(tmp_path):
    data = core.DATA_DIR
    if not (data / "lookup.db").exists() or core.missing_data_files("grc", None):
        pytest.skip("the full data download is not installed")
    partial = tmp_path / "data"
    partial.mkdir()
    for path in data.iterdir():
        if path.name != "lsj_headwords.json":
            (partial / path.name).symlink_to(path.resolve())
    script = """
import warnings
from dilemma import Dilemma, DilemmaDataWarning
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    Dilemma(lang="grc", skip_pos=True)
    Dilemma(lang="grc", skip_pos=True)
found = [w for w in caught if issubclass(w.category, DilemmaDataWarning)]
assert len(found) == 1, [str(w.message) for w in caught]
message = str(found[0].message)
assert "lsj_headwords.json" in message and "python -m dilemma download" in message
assert "lsj10_headwords.json" not in message
"""
    env = dict(os.environ, DILEMMA_DATA_DIR=str(partial))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
