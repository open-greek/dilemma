"""The kaikki.org snapshot pins (data/kaikki_inputs.json) and their archiver."""
import gzip
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import archive_kaikki_inputs as A  # noqa: E402
from build_data import DUMPS, EL_DIALECT_DUMP  # noqa: E402

PINS = json.loads(A.PINS.read_text(encoding="utf-8"))


def test_pin_file_has_the_documented_shape():
    A.validate(PINS)
    assert PINS["snapshot"] == "20261114"
    assert PINS["repo"] == "ciscoriordan/dilemma"


def test_every_dump_the_builder_reads_is_pinned():
    needed = {name for dumps in DUMPS.values() for name in dumps.values()}
    needed.add(EL_DIALECT_DUMP)
    assert needed <= set(PINS["files"])


def test_kaikki_dates_and_licenses():
    files = PINS["files"]
    for name, rec in files.items():
        assert "GFDL" in rec["license"] and "CC BY-SA 4.0" in rec["license"]
        if "derived" in rec:
            continue
        date = rec["last_modified"][:10]
        if name.startswith("kaikki.org-en-"):
            assert date == "2026-10-30", name
        else:   # Greek edition, served 2026-11-02 US time (2026-11-03 UTC)
            assert date == "2026-11-03", name


def test_dialect_extract_is_pinned_with_its_source_and_meta():
    files = PINS["files"]
    dialects = files[EL_DIALECT_DUMP]
    assert dialects["derived"]["from"] == "kaikki.org-el-raw-wiktextract-data.jsonl.gz"
    assert (ROOT / dialects["derived"]["by"]).exists()
    assert files["kaikki.org-el-dictionary-GreekDialects.meta.json"]["derived"] == dialects["derived"]


@pytest.mark.parametrize("record", ["el_nominal_fragments.json",
                                    "el_generator_refresh.json"])
def test_modern_generator_records_were_made_from_this_snapshot(record):
    path = ROOT / "data" / "changes" / record
    if not path.exists():       # el_generator_refresh.json comes from the Hub
        pytest.skip(f"{record} is not downloaded")
    dumps = json.loads(path.read_text(encoding="utf-8"))["dumps"]
    assert dumps
    for name, digest in dumps.items():
        assert PINS["files"][name]["sha256"] == digest, name


def test_archives_are_staged_and_await_their_hub_revision():
    for name, rec in PINS["files"].items():
        archive = rec["archive"]
        assert archive["sha256"] is not None, f"{name} has not been staged"
        assert archive["revision"] == "pending" or A.REVISION.fullmatch(archive["revision"])


@pytest.mark.parametrize("problem", [
    lambda p: p["files"]["kaikki.org-x.jsonl"].update(sha256="ABC"),
    lambda p: p["files"]["kaikki.org-x.jsonl"]["archive"].update(path="data/x.jsonl.gz"),
    lambda p: p["files"]["kaikki.org-x.jsonl"]["archive"].update(revision="main"),
    lambda p: p["files"]["kaikki.org-x.jsonl"].update(license="unknown"),
    lambda p: p["files"]["kaikki.org-x.jsonl"].pop("last_modified"),
])
def test_validate_refuses_a_malformed_pin(problem):
    pins = _fixture_pins(b"{}\n")
    A.validate(pins)
    problem(pins)
    with pytest.raises(A.PinError):
        A.validate(pins)


def test_validate_refuses_a_nested_or_foreign_name():
    pins = _fixture_pins(b"{}\n")
    pins["files"]["el/x.jsonl"] = pins["files"].pop("kaikki.org-x.jsonl")
    with pytest.raises(A.PinError):
        A.validate(pins)


def _fixture_pins(content: bytes) -> dict:
    import hashlib
    return {"schema_version": 1, "snapshot": "20990101", "repo": "r/r",
            "repo_type": "model", "files": {"kaikki.org-x.jsonl": {
                "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                "license": "CC BY-SA 4.0 / GFDL", "url": "https://kaikki.org/x",
                "last_modified": "2099-01-01T00:00:00Z",
                "archive": {"path": "data/kaikki_inputs/20990101/kaikki.org-x.jsonl.gz",
                            "compression": "gzip", "bytes": None, "sha256": None,
                            "revision": "pending"}}}}


def test_stage_pin_and_download_round_trip(tmp_path):
    content = b'{"word": "x"}\n' * 100
    src, stage, out = tmp_path / "src", tmp_path / "stage", tmp_path / "out"
    src.mkdir()
    (src / "kaikki.org-x.jsonl").write_bytes(content)
    pins_path = tmp_path / "pins.json"
    A.write_pins(_fixture_pins(content), pins_path)
    with pytest.raises(A.PinError):          # nothing to download yet
        A.download(out, pins_path, fetch=lambda *a: None)
    A.stage(src, stage, pins_path)
    archive = stage / "kaikki.org-x.jsonl.gz"
    assert gzip.decompress(archive.read_bytes()) == content
    first = archive.read_bytes()
    A.stage(src, stage, pins_path)           # deterministic bytes
    assert archive.read_bytes() == first
    A.pin_revision("a" * 40, stage, pins_path)
    seen = []

    def fetch(pins, path, revision):
        seen.append((path, revision))
        return archive

    A.download(out, pins_path, fetch=fetch)
    assert seen == [("data/kaikki_inputs/20990101/kaikki.org-x.jsonl.gz", "a" * 40)]
    assert (out / "kaikki.org-x.jsonl").read_bytes() == content
    assert (out / "SHA256SUMS").read_text().endswith("  kaikki.org-x.jsonl\n")
    A.verify(out, pins_path)
    (src / "kaikki.org-x.jsonl").write_bytes(content + b"\n")
    with pytest.raises(A.PinError):
        A.stage(src, stage, pins_path)
