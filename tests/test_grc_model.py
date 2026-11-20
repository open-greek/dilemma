"""The Ancient-only model `model/grc-s3/` (#149).

`python -m dilemma download` serves `model/grc-s3/` with the rest of
`model/`, and `Dilemma(lang="grc")` then loads it in place of
`combined-s3`. CI downloads and tests only `combined-s3`, so these tests
check the choice of directory without any model, and, where a checkout or
the download cache has `grc-s3` (run with --run-slow), that the copy is the
pinned one and that `lang="grc"` answers with it through both backends.
"""

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

from dilemma import core

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILES = ("vocab.json", "encoder.onnx", "decoder_step.onnx", "model.pt",
               "head_labels.json", "heads.npz")

# Forms no lookup table has, which the Ancient model reads; the combined
# model reads them the same way.
MODEL_CASES = {
    "λελαμπρυσμένος": "λαμπρύνω",
    "προκατανοησάμενος": "προκατανοέω",
    "ἐκκεκαθαρμένοις": "ἐκκαθαίρω",
    "μεταπεπεισμένων": "μεταπείθω",
}


def _stub_model(directory: Path) -> Path:
    directory.mkdir(parents=True)
    (directory / "encoder.onnx").write_bytes(b"")
    return directory


@pytest.fixture
def no_lookups(monkeypatch):
    monkeypatch.setattr(core.Dilemma, "_load_lookups", lambda *a, **k: None)
    monkeypatch.setattr(core, "_warn_missing_data_files", lambda *a, **k: None)


def test_grc_prefers_the_ancient_model_when_it_is_present(tmp_path, monkeypatch,
                                                          no_lookups):
    _stub_model(tmp_path / "combined-s3")
    grc = _stub_model(tmp_path / "grc-s3")
    monkeypatch.setattr(core, "MODEL_DIR", tmp_path)
    assert core.Dilemma(lang="grc")._find_model_dir() == grc
    assert core.Dilemma(lang="all")._find_model_dir() == tmp_path / "combined-s3"
    assert core.Dilemma(lang="el")._find_model_dir() == tmp_path / "combined-s3"


def test_grc_falls_back_to_the_combined_model(tmp_path, monkeypatch, no_lookups):
    combined = _stub_model(tmp_path / "combined-s3")
    monkeypatch.setattr(core, "MODEL_DIR", tmp_path)
    assert core.Dilemma(lang="grc")._find_model_dir() == combined


def test_the_ancient_model_is_pinned_file_for_file():
    spec = importlib.util.spec_from_file_location(
        "hf_data", ROOT / "scripts" / "hf_data.py")
    hf_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hf_data)
    pinned = json.loads((ROOT / "data" / "hf_manifest.json").read_text(
        encoding="utf-8"))["pinned_only"]
    for name in MODEL_FILES:
        assert f"model/grc-s3/{name}" in hf_data.PINNED_ONLY
        assert f"model/grc-s3/{name}" in pinned


def _installed_grc_model() -> Path:
    directory = Path(core.MODEL_DIR) / "grc-s3"
    if not all((directory / name).exists() for name in MODEL_FILES):
        pytest.skip("model/grc-s3 is not installed (python -m dilemma download)")
    return directory


@pytest.mark.slow
def test_installed_ancient_model_is_the_pinned_copy():
    directory = _installed_grc_model()
    pinned = json.loads((ROOT / "data" / "hf_manifest.json").read_text(
        encoding="utf-8"))["pinned_only"]
    for name in MODEL_FILES:
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        assert digest == pinned[f"model/grc-s3/{name}"]["sha256"], name


@pytest.fixture(scope="module")
def grc_engine():
    directory = _installed_grc_model()
    pytest.importorskip("onnxruntime")
    engine = core.Dilemma(lang="grc")
    assert engine._find_model_dir() == directory
    engine._load_model()
    return engine


@pytest.mark.slow
def test_grc_reads_unseen_forms_with_the_ancient_model(grc_engine):
    assert grc_engine._use_onnx
    forms = list(MODEL_CASES)
    assert grc_engine._predict(forms) == [MODEL_CASES[f] for f in forms]
    assert [grc_engine.lemmatize(f) for f in forms] == [MODEL_CASES[f] for f in forms]


@pytest.mark.slow
def test_ancient_model_backends_agree(grc_engine):
    pytest.importorskip("torch")
    torch_engine = object.__new__(core.Dilemma)
    torch_engine.__dict__.update(grc_engine.__dict__)
    torch_engine._prediction_cache = None
    torch_engine._load_pytorch(grc_engine._find_model_dir())
    torch_engine._use_onnx = False
    forms = list(MODEL_CASES)
    assert torch_engine._predict_uncached(forms) == grc_engine._predict_uncached(forms)
