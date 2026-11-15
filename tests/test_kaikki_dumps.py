"""Every kaikki dump reader accepts the flat (archived) and the nested layout."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from dilemma_build import kaikki_dumps as K

ROOT = Path(__file__).resolve().parents[1]


def place(kaikki: Path, filename: str, layout: str, text: str = "") -> Path:
    path = kaikki / (filename if layout == "flat" else K.NESTED_MAP[filename])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_every_builder_dump_has_a_nested_name():
    names = {name for dumps in K.DUMPS.values() for name in dumps.values()}
    assert names | {K.EL_DIALECT_DUMP} == set(K.NESTED_MAP)


def test_build_data_reexports_the_shared_names():
    import build_data
    assert build_data.DUMPS is K.DUMPS
    assert build_data.EL_DIALECT_DUMP == K.EL_DIALECT_DUMP
    assert build_data.resolve_dump is K.resolve_dump


@pytest.mark.parametrize("layout", ["flat", "nested"])
def test_resolve_dump_reads_either_layout(tmp_path, layout):
    for name in K.NESTED_MAP:
        path = place(tmp_path, name, layout)
        assert K.resolve_dump(name, tmp_path) == path
        assert K.relative_dump_path(name, tmp_path) in K.layout_names(name)


def test_flat_wins_and_missing_resolves_to_the_flat_name(tmp_path):
    name = K.DUMPS["grc"]["en"]
    assert K.resolve_dump(name, tmp_path) == tmp_path / name
    place(tmp_path, name, "nested")
    flat = place(tmp_path, name, "flat")
    assert K.resolve_dump(name, tmp_path) == flat


def test_dialect_extract_resolves_at_the_root_of_a_nested_directory(tmp_path):
    """Older nested directories keep the extract beside en-el/ and el/."""
    place(tmp_path, K.DUMPS["grc"]["en"], "nested")
    root = place(tmp_path, K.EL_DIALECT_DUMP, "flat")
    assert K.resolve_dump(K.EL_DIALECT_DUMP, tmp_path) == root


def test_one_environment_variable_with_the_older_name_as_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv(K.ENV_VAR, raising=False)
    monkeypatch.delenv(K.LEGACY_ENV_VAR, raising=False)
    assert K.env_dump_dir() is None
    monkeypatch.setenv(K.LEGACY_ENV_VAR, str(tmp_path / "old"))
    assert K.env_dump_dir() == tmp_path / "old"
    monkeypatch.setenv(K.ENV_VAR, str(tmp_path / "new"))
    assert K.env_dump_dir() == tmp_path / "new"


@pytest.mark.parametrize("layout", ["flat", "nested"])
def test_treebank_definitions_read_either_layout(tmp_path, monkeypatch, layout):
    from dilemma_build import build_treebank_pos_lookup as build
    entry = {"word": "λόγος", "lang_code": "grc", "pos": "noun",
             "senses": [{"glosses": ["a word"]}]}
    for edition in ("en", "el"):
        place(tmp_path, K.DUMPS["grc"][edition], layout, json.dumps(entry) + "\n")
    monkeypatch.setattr(build, "KAIKKI_DIR", tmp_path)
    assert build._defined_pos_targets({("λόγος", "NOUN")}) == {("λόγος", "NOUN")}


@pytest.mark.parametrize("layout", ["flat", "nested"])
def test_overlay_records_the_path_it_read(tmp_path, layout):
    import overlay_lsj as overlay
    entry = {"word": "οὗτος", "pos": "pron", "lang_code": "grc",
             "senses": [{"glosses": ["this"]}]}
    for _, filename in overlay.CLOSED_CLASS_DUMPS.values():
        place(tmp_path, filename, layout, json.dumps(entry, ensure_ascii=False) + "\n")
    entries, pins, paths = overlay.load_closed_class_entries(tmp_path)
    assert len(entries) == 2
    for label, (_, filename) in overlay.CLOSED_CLASS_DUMPS.items():
        expected = filename if layout == "flat" else K.NESTED_MAP[filename]
        # The pin names the flat dump in either layout, so equal dump bytes
        # give equal audit bytes; the path read is the layout's own.
        assert pins[label]["path"] == filename
        assert paths[label] == tmp_path / expected


def test_alias_manifest_accepts_an_audit_recorded_in_the_other_layout(tmp_path):
    """A published audit made from a nested directory validates against the
    flat archive of the same bytes, and the other way round."""
    from scripts import build_source_alias_manifest as publication
    pins = {}
    for label, filename in publication.CLOSED_CLASS_DUMPS.items():
        path = place(tmp_path, filename, "flat", label + "\n")
        pins[label] = {"path": K.NESTED_MAP[filename], **publication.pin(path)}
    sources = {name: publication.pin(ROOT / name)
               for name in publication.CLOSED_CLASS_SOURCE_FILES}
    audit = {"canonicalized": [], "existing_preserved": [],
             "closed_class_elisions": {
                 "schema_version": 1, "enabled": True,
                 "external_input_sha256": pins, "source_sha256": sources,
                 "decisions": [], "planned_count": 0, "canonicalized": [],
                 "canonicalized_count": 0, "existing_preserved": [],
                 "existing_preserved_count": 0}}
    external, _ = publication.validate_closed_class_evidence(audit, ROOT, tmp_path, sources)
    assert external == pins
    pins["wiktionary_el_grc"]["path"] = "elsewhere/" + K.DUMPS["grc"]["el"]
    with pytest.raises(AssertionError):
        publication.validate_closed_class_evidence(audit, ROOT, tmp_path, sources)


@pytest.mark.parametrize("layout", ["flat", "nested"])
def test_expand_lsj_kaikki_and_lsj9_options(tmp_path, layout, monkeypatch):
    sys.path.insert(0, str(ROOT / "build"))
    import expand_lsj
    for name in ("KAIKKI_DIR", "KAIKKI_AG", "KAIKKI_AG_EL", "LSJ9_DIR",
                 "LSJ9_FORMS", "LSJ9_HEADWORDS", "LSJ9_GLOSSES",
                 "LSJ9_GLOSSES_SUPPLEMENT"):
        monkeypatch.setattr(expand_lsj, name, getattr(expand_lsj, name))
    en = place(tmp_path, K.DUMPS["grc"]["en"], layout)
    el = place(tmp_path, K.DUMPS["grc"]["el"], layout)
    expand_lsj.configure_inputs(tmp_path, tmp_path / "lsj9")
    assert (expand_lsj.KAIKKI_AG, expand_lsj.KAIKKI_AG_EL) == (en, el)
    assert expand_lsj.LSJ9_FORMS == tmp_path / "lsj9" / "lsj9_forms.tsv"
    assert expand_lsj.parse_lsj9_entries() == {}     # missing file, no error


def test_expand_scripts_take_the_input_options():
    for script in ("build/expand_lsj.py", "build/expand_sophocles.py"):
        out = subprocess.run([sys.executable, str(ROOT / script), "--help"],
                             capture_output=True, text=True, timeout=60)
        assert out.returncode == 0, out.stderr
        assert "--kaikki" in out.stdout and "--lsj9-dir" in out.stdout
