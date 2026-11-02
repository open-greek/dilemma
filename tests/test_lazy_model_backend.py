"""Lookup-only use must never initialize a native inference dependency."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import warnings

import pytest

from dilemma import core


@pytest.mark.parametrize("lang", ["invalid", "med", None])
def test_invalid_language_is_rejected_without_loading_data(monkeypatch, lang):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid language must be rejected before loading data")

    monkeypatch.setattr(core.Dilemma, "_load_lookups", unexpected_load)
    with pytest.raises(ValueError, match="Unknown language"):
        core.Dilemma(lang=lang)


def test_legacy_both_alias_does_not_require_backend_discovery(monkeypatch):
    monkeypatch.setattr(core.Dilemma, "_load_lookups", lambda *a, **k: None)
    engine = core.Dilemma(lang="both")
    assert engine.lang == "all"
    assert engine._model is None


@pytest.mark.parametrize("storage", ["sqlite", "json"])
@pytest.mark.parametrize("artifact", [None, "encoder.onnx", "model.pt"])
@pytest.mark.parametrize("worker_thread", [False, True])
def test_lookup_only_does_not_import_inference_backends(
        tmp_path, storage, artifact, worker_thread):
    data = tmp_path / "data"
    data.mkdir()
    if storage == "sqlite":
        with sqlite3.connect(data / "lookup.db") as conn:
            conn.executescript("""
                CREATE TABLE lemmas (id INTEGER PRIMARY KEY, text TEXT);
                CREATE TABLE lookup (form TEXT, lemma_id INTEGER, lang TEXT,
                                     src TEXT, stripped TEXT);
                INSERT INTO lemmas VALUES (1, 'λόγος');
                INSERT INTO lookup VALUES ('λόγοι', 1, 'all', 'grc', 'λογοι');
                INSERT INTO lookup VALUES ('λόγος', 1, 'all', 'grc', 'λογος');
            """)
    else:
        (data / "ag_lookup.json").write_text(json.dumps(
            {"λόγοι": "λόγος", "λόγος": "λόγος"}), encoding="utf-8")
    heads = data / "grc_lexical_heads.json"
    heads.write_text(json.dumps({"schema_version": 2, "heads": {
        "λόγος": ["fixture"]}, "vernacular_heads": {}}), encoding="utf-8")
    (data / "treebank_pos_lookup.audit.json").write_text(json.dumps({
        "schema_version": 2, "language": "grc", "rejected_corpus_targets": {},
        "inputs": {"grc_lexical_heads": {"size": heads.stat().st_size,
            "sha256": hashlib.sha256(heads.read_bytes()).hexdigest()}}}), encoding="utf-8")
    model = tmp_path / "model"
    (model / "grc").mkdir(parents=True)
    if artifact:
        (model / "grc" / artifact).touch()

    # A fresh interpreter catches eager imports even when the pytest process
    # has already imported ONNX Runtime, PyTorch, or NumPy for other tests.
    script = """
import importlib.abc
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

class NoInferenceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'onnxruntime', 'torch', 'numpy'}:
            raise AssertionError('Lookup-only use imported ' + fullname)

sys.meta_path.insert(0, NoInferenceImports())
from dilemma import Dilemma

def exercise():
    for skip_pos in (False, True):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            engine = Dilemma(lang='grc', device='cpu', skip_pos=skip_pos)
            assert engine.lemmatize('λόγοι', guess=False) == 'λόγος'
            assert engine.lemmatize('μάκαπ', guess=False) is None
            assert engine.lemmatize_verbose('λόγοι', guess=False)[0].lemma == 'λόγος'
            assert engine.lemmatize_verbose('μάκαπ', guess=False) == []
            assert engine.lemmatize_batch(['λόγοι', 'μάκαπ'], guess=False) == ['λόγος', None]
            assert engine.lemmatize_pos('λόγοι', 'NOUN', guess=False) == 'λόγος'
            assert engine.lemmatize_batch_pos(['λόγοι', 'μάκαπ'], ['NOUN', 'NOUN'],
                                             guess=False) == ['λόγος', None]
            assert not caught, [str(item.message) for item in caught]
            assert engine._model is None
            for table in {id(t): t for t in (engine._lookup, engine._ag_lookup,
                                            engine._mg_lookup, engine._med_lookup)}.values():
                if hasattr(table, 'close'):
                    table.close()
    assert not {'onnxruntime', 'torch', 'numpy'} & sys.modules.keys()

if sys.argv[1] == 'worker':
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(exercise).result()
else:
    exercise()
"""
    env = dict(os.environ, DILEMMA_DATA_DIR=str(data),
               DILEMMA_MODEL_DIR=str(model))
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [root, env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", script, "worker" if worker_thread else "main"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("artifact,loader", [
    ("encoder.onnx", "_load_onnx"), ("model.pt", "_load_pytorch"),
])
def test_first_model_request_checks_and_loads_the_backend(
        tmp_path, monkeypatch, artifact, loader):
    monkeypatch.setattr(core.Dilemma, "_load_lookups", lambda *a, **k: None)
    (tmp_path / artifact).touch()
    monkeypatch.setattr(core.Dilemma, "_find_model_dir", lambda self: tmp_path)
    events = []
    monkeypatch.setattr(core.Dilemma, "_check_backend",
                        lambda self: events.append("check"))

    def load(self, directory):
        assert directory == tmp_path
        events.append(loader)
        self._model = object()

    monkeypatch.setattr(core.Dilemma, loader, load)
    engine = core.Dilemma(lang="grc")
    assert events == []
    engine._load_model()
    assert events == ["check", loader]
    engine._load_model()
    assert events == ["check", loader]


def test_missing_backend_warns_only_when_inference_is_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(core.Dilemma, "_load_lookups", lambda *a, **k: None)
    monkeypatch.setattr(core.Dilemma, "_find_model_dir", lambda self: tmp_path)

    def unavailable(self, directory):
        raise ImportError("No model backend installed")

    monkeypatch.setattr(core.Dilemma, "_load_pytorch", unavailable)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine = core.Dilemma(lang="grc")
        assert not caught
        for _ in range(2):
            with pytest.raises(ImportError, match="No model backend"):
                engine._load_model()
        assert len(caught) == 1
        assert "No model backend available" in str(caught[0].message)
