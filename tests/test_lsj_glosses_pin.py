"""The verb pass reads only the pinned LSJ9 glosses export."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
import expand_lsj


def _pin(tmp_path, text):
    glosses = tmp_path / "lsj9_glosses.jsonl"
    glosses.write_text(text, encoding="utf-8")
    pin = tmp_path / "pin.json"
    pin.write_text(json.dumps({
        "path": glosses.name, "size": glosses.stat().st_size,
        "sha256": hashlib.sha256(glosses.read_bytes()).hexdigest()}),
        encoding="utf-8")
    return glosses, pin


def test_repo_pin_names_the_glosses_export():
    pin = json.loads(expand_lsj.LSJ9_GLOSSES_PIN.read_text(encoding="utf-8"))
    assert pin["path"] == "lsj9_glosses.jsonl"
    assert len(pin["sha256"]) == 64 and pin["size"] > 0


def test_matching_export_passes(tmp_path, monkeypatch):
    glosses, pin = _pin(tmp_path, '{"headword": "λύω", "text": "loose"}\n')
    monkeypatch.setattr(expand_lsj, "LSJ9_GLOSSES", glosses)
    monkeypatch.setattr(expand_lsj, "LSJ9_GLOSSES_PIN", pin)
    assert expand_lsj.verify_lsj_glosses()["size"] == glosses.stat().st_size


def test_missing_or_other_export_stops_the_verb_pass(tmp_path, monkeypatch):
    glosses, pin = _pin(tmp_path, '{"headword": "λύω", "text": "loose"}\n')
    monkeypatch.setattr(expand_lsj, "LSJ9_GLOSSES_PIN", pin)
    monkeypatch.setattr(expand_lsj, "LSJ9_GLOSSES", tmp_path / "absent.jsonl")
    with pytest.raises(SystemExit, match="not found"):
        expand_lsj.verify_lsj_glosses()
    glosses.write_text('{"headword": "λύω"}\n', encoding="utf-8")
    monkeypatch.setattr(expand_lsj, "LSJ9_GLOSSES", glosses)
    with pytest.raises(SystemExit, match="does not match"):
        expand_lsj.verify_lsj_glosses()
