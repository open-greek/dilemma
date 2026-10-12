"""Build metadata must remain usable on the supported Python 3.10 runtime."""
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))
import expand_lsj
import expand_sophocles
from build.build_grc_lexical_heads import sha256


@pytest.mark.parametrize("payload", [b"", bytes(range(256)) * 8193],
                         ids=["empty", "multiple_chunks"])
def test_build_hashes_work_without_file_digest(tmp_path, monkeypatch, payload):
    source = tmp_path / "source"
    source.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    assert sha256(source) == expected

    spec = importlib.util.spec_from_file_location(
        "source_alias_manifest", ROOT / "scripts" / "build_source_alias_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.pin(source) == {"sha256": expected, "size": len(payload)}

    source.with_suffix(".db.pickle").write_bytes(payload)
    for name in ("KAIKKI_AG", "KAIKKI_AG_EL", "WTP_DB", "LSJ9_FORMS", "LSJ9_HEADWORDS"):
        monkeypatch.setattr(expand_lsj, name, source)
    hashes = expand_lsj.noun_generation_input_hashes()
    assert len(hashes) == 6 and set(hashes.values()) == {expected}

    chunks = tmp_path / "chunks"
    chunks.mkdir()
    (chunks / "0057.xml").write_bytes(payload)
    monkeypatch.setattr(expand_sophocles, "SOPH_DIR", chunks)
    hashes = expand_sophocles.sophocles_noun_input_hashes()
    assert hashes["sophocles/0057.xml"] == expected
    assert set(hashes.values()) == {expected}
